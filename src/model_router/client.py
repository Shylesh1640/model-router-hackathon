"""OpenRouter API client — calls models with jittered retry and circuit breaker."""

import logging
import random
import time
from typing import Optional, Callable

import requests

from .models import GenerationResult

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Tracks per-model failures and opens the circuit to skip dead models.

    After `max_failures` errors within the `cooldown` window, the model is
    skipped for the remainder of that window. A single success resets it.
    """

    def __init__(self, cooldown: float = 60, max_failures: int = 3):
        self.cooldown = cooldown
        self.max_failures = max_failures
        self._failures: dict[str, list[float]] = {}

    def record_failure(self, model_id: str):
        """Note a failure for the given model."""
        now = time.time()
        if model_id not in self._failures:
            self._failures[model_id] = []
        self._failures[model_id].append(now)
        self._prune(model_id)

    def record_success(self, model_id: str):
        """Reset failure count on success."""
        self._failures.pop(model_id, None)

    def is_open(self, model_id: str) -> bool:
        """Is the circuit open? (model should be skipped)"""
        return self.remaining_cooldown(model_id) > 0

    def remaining_cooldown(self, model_id: str) -> float:
        """Seconds until the circuit closes and the model can be tried again."""
        failures = self._prune(model_id)
        if len(failures) < self.max_failures:
            return 0.0
        # The max_failures'th oldest failure defines the cooldown window
        trigger = failures[-self.max_failures]
        remaining = self.cooldown - (time.time() - trigger)
        return max(0.0, remaining)

    def _prune(self, model_id: str) -> list[float]:
        """Remove failures outside the cooldown window."""
        now = time.time()
        active = [
            t for t in self._failures.get(model_id, [])
            if now - t < self.cooldown
        ]
        if active:
            self._failures[model_id] = active
        else:
            self._failures.pop(model_id, None)
        return active

    @property
    def open_models(self) -> list[str]:
        """List model IDs that are currently on cooldown."""
        result = []
        for mid in list(self._failures.keys()):
            if self.is_open(mid):
                result.append(mid)
        return result


class OpenRouterClient:
    """Client for the OpenRouter API with jittered backoff and circuit breaker.

    Usage:
        client = OpenRouterClient(api_key="sk-or-...")
        result = client.generate("hello", "openai/gpt-oss-20b:free", "thinking")
        print(result.response)
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    MAX_BACKOFF = 30.0
    BASE_DELAY = 1.0

    def __init__(
        self,
        api_key: str,
        timeout: int = 60,
        retry_count: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.retry_count = retry_count
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/San20506/model-router",
            "X-Title": "Model Router",
        })

    # ---- Public API ---------------------------------------------------------

    def generate(
        self,
        query: str,
        model_id: str,
        tier: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        on_attempt: Optional[Callable] = None,
        fallback_models: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate, with jittered backoff and circuit-breaker awareness.

        If the primary ``model_id`` fails with a retryable error after all
        retries, tries each ``fallback_models`` in order (same tier).
        """
        result = self._generate_single(
            query, model_id, tier, max_tokens, temperature,
            system_prompt, on_attempt,
        )
        if result.error and fallback_models:
            for fb_id in fallback_models:
                if fb_id == model_id:
                    continue
                if self.circuit_breaker.is_open(fb_id):
                    continue
                logger.warning(
                    "Primary model %s failed, trying fallback: %s",
                    model_id, fb_id,
                )
                result = self._generate_single(
                    query, fb_id, tier, max_tokens, temperature,
                    system_prompt, on_attempt,
                )
                if not result.error:
                    break
        return result

    def _generate_single(
        self,
        query: str,
        model_id: str,
        tier: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        on_attempt: Optional[Callable] = None,
    ) -> GenerationResult:
        """Generate, with jittered backoff and circuit-breaker awareness.

        ``on_attempt`` — optional callback ``fn(attempt, err)`` for upstream
        cascade / fallback logic to observe each failure.
        """
        # Circuit breaker check
        if self.circuit_breaker.is_open(model_id):
            cooldown = self.circuit_breaker.remaining_cooldown(model_id)
            msg = f"Circuit open for {model_id}, {cooldown:.0f}s remaining"
            logger.warning(msg)
            return GenerationResult(
                query=query, response="", model_id=model_id, tier=tier,
                tokens_in=0, tokens_out=0, latency_ms=0,
                error=msg,
            )

        start = time.perf_counter()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        for attempt in range(self.retry_count + 1):
            try:
                resp = self.session.post(
                    f"{self.BASE_URL}/chat/completions",
                    json={
                        "model": model_id,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    timeout=self.timeout,
                )

                latency_ms = (time.perf_counter() - start) * 1000

                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    usage = data.get("usage", {})
                    self.circuit_breaker.record_success(model_id)
                    return GenerationResult(
                        query=query,
                        response=choice["message"]["content"],
                        model_id=model_id,
                        tier=tier,
                        tokens_in=usage.get("prompt_tokens", 0),
                        tokens_out=usage.get("completion_tokens", 0),
                        latency_ms=round(latency_ms, 1),
                    )

                # --- Transient errors (retryable) ---
                if resp.status_code == 429 and attempt < self.retry_count:
                    wait = self._backoff(attempt, resp)
                    logger.warning(
                        "Rate limited (429), retry %d/%d in %.1fs: %s",
                        attempt + 1, self.retry_count, wait, model_id,
                    )
                    if on_attempt:
                        on_attempt(attempt, f"429")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500 and attempt < self.retry_count:
                    wait = self._backoff(attempt, resp)
                    logger.warning(
                        "Server error %d, retry %d/%d in %.1fs: %s",
                        resp.status_code, attempt + 1, self.retry_count, wait, model_id,
                    )
                    if on_attempt:
                        on_attempt(attempt, f"{resp.status_code}")
                    time.sleep(wait)
                    continue

                # --- Non-retryable — fall through to exhaustion ---
                error_msg = f"API error {resp.status_code}: {resp.text[:200]}"
                logger.error("%s — %s", error_msg, model_id)
                self.circuit_breaker.record_failure(model_id)
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0, latency_ms=round(latency_ms, 1),
                    error=error_msg,
                )

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < self.retry_count:
                    wait = self._backoff(attempt)
                    logger.warning(
                        "%s, retry %d/%d in %.1fs: %s",
                        type(e).__name__, attempt + 1, self.retry_count, wait, model_id,
                    )
                    if on_attempt:
                        on_attempt(attempt, type(e).__name__)
                    time.sleep(wait)
                    continue
                self.circuit_breaker.record_failure(model_id)
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    error=f"{type(e).__name__}: max retries exceeded",
                )

            except Exception as e:
                logger.error("Request failed (%s): %s", model_id, e)
                self.circuit_breaker.record_failure(model_id)
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    error=str(e),
                )

        # All retries exhausted
        self.circuit_breaker.record_failure(model_id)
        return GenerationResult(
            query=query, response="", model_id=model_id, tier=tier,
            tokens_in=0, tokens_out=0,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error="Max retries exceeded",
        )

    # ---- Helpers ------------------------------------------------------------

    def _backoff(self, attempt: int, resp: Optional[requests.Response] = None) -> float:
        """Jittered exponential backoff with Retry-After header support."""
        # Honour Retry-After header first
        if resp and resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), self.max_delay)
                except ValueError:
                    pass
        # Jittered exponential backoff
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.5 * delay)
        return min(delay + jitter, self.max_delay)

    def list_available(self) -> list[dict]:
        """Fetch available models from OpenRouter (for dashboard / refresh)."""
        try:
            resp = self.session.get(f"{self.BASE_URL}/models", timeout=10)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning("Failed to list models: %s", e)
        return []

    @classmethod
    def refresh_model_pool(cls, api_key: str) -> list[dict]:
        """Fetch current free-model list from OpenRouter (static)."""
        client = cls(api_key, timeout=15)
        return client.list_available()
