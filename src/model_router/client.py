"""OpenRouter API client — calls models with retry and rate-limit handling."""

import logging
import time
import requests
from typing import Optional

from .models import GenerationResult

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for the OpenRouter API.

    Handles rate limits, timeouts, and error reporting.
    Returns GenerationResult with response, token usage, and latency.

    Usage:
        client = OpenRouterClient(api_key="sk-or-...")
        result = client.generate("hello", "openai/gpt-oss-20b:free", "thinking")
        print(result.response)
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, timeout: int = 60, retry_count: int = 2):
        self.api_key = api_key
        self.timeout = timeout
        self.retry_count = retry_count
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/San20506/model-router",
            "X-Title": "Model Router",
        })

    def generate(
        self,
        query: str,
        model_id: str,
        tier: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> GenerationResult:
        """Generate a response from a model, with retries."""
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

                    return GenerationResult(
                        query=query,
                        response=choice["message"]["content"],
                        model_id=model_id,
                        tier=tier,
                        tokens_in=usage.get("prompt_tokens", 0),
                        tokens_out=usage.get("completion_tokens", 0),
                        latency_ms=round(latency_ms, 1),
                    )

                elif resp.status_code == 429 and attempt < self.retry_count:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Rate limited (%s), retrying in %ss...",
                        resp.status_code, wait,
                    )
                    time.sleep(wait)
                    continue

                else:
                    error_msg = f"API error {resp.status_code}: {resp.text[:200]}"
                    logger.error(error_msg)
                    return GenerationResult(
                        query=query, response="", model_id=model_id, tier=tier,
                        tokens_in=0, tokens_out=0, latency_ms=round(latency_ms, 1),
                        error=error_msg,
                    )

            except requests.Timeout:
                if attempt < self.retry_count:
                    logger.warning("Timeout on attempt %s, retrying...", attempt + 1)
                    continue
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    error="Request timed out",
                )

            except Exception as e:
                logger.error("Request failed: %s", e)
                return GenerationResult(
                    query=query, response="", model_id=model_id, tier=tier,
                    tokens_in=0, tokens_out=0,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                    error=str(e),
                )

        return GenerationResult(
            query=query, response="", model_id=model_id, tier=tier,
            tokens_in=0, tokens_out=0,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            error="Max retries exceeded",
        )

    def list_available(self) -> list[dict]:
        """Fetch available models from OpenRouter (for dashboard health check)."""
        try:
            resp = self.session.get(f"{self.BASE_URL}/models", timeout=10)
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception as e:
            logger.warning("Failed to list models: %s", e)
        return []
