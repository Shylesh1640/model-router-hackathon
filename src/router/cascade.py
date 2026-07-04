"""Self-verification cascade — checks if fast model's answer is reliable, escalates if not."""

import logging
import json
import time
from typing import Optional

from src.models import GenerationResult, RoutingDecision
from src.config import RouterConfig

logger = logging.getLogger(__name__)


class Cascade:
    """
    AutoMix-style self-verification cascade.

    After a fast-tier generation, asks the same model to verify its own answer.
    If verification confidence is low, escalates to the next tier.
    Max 1 hop by default.
    """

    def __init__(self, config: RouterConfig):
        self.enabled = config.cascade_enabled
        self.max_hops = config.cascade_max_hops
        self.escalation_threshold = config.cascade_escalation_threshold

    def should_escalate(
        self, query: str, response: str, tier: str, confidence: float
    ) -> bool:
        """Check if the response needs escalation."""
        if not self.enabled:
            return False

        # Skip cascade for deep tier — already at max capability
        if tier == "deep":
            return False

        # Skip if response looks confident enough
        if confidence > 0.95:
            logger.debug(f"Skipping cascade: fast-tier confidence {confidence:.2f} > 0.95")
            return False

        # Run self-verification
        verify_result = self._self_verify(query, response)
        if verify_result is None:
            return False  # verification inconclusive, don't escalate

        should = verify_result < self.escalation_threshold
        logger.info(
            f"Self-verify score: {verify_result:.2f} "
            f"(threshold: {self.escalation_threshold}) "
            f"{'→ ESCALATING' if should else '→ OK'}"
        )
        return should

    def _self_verify(self, query: str, response: str) -> Optional[float]:
        """
        Ask the model to verify its own answer.

        Returns confidence score 0-1, or None if verification fails.
        Lightweight — uses a minimal prompt, parses YES/NO.
        """
        verify_prompt = (
            f"Query: {query}\n\n"
            f"Answer: {response}\n\n"
            f"Is this answer correct and complete? "
            f"Reply with exactly one word: YES or NO."
        )

        try:
            import requests

            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._get_api_key()}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "liquid/lfm-2.5-1.2b-instruct:free",  # cheapest verifier
                    "messages": [{"role": "user", "content": verify_prompt}],
                    "max_tokens": 10,
                    "temperature": 0.1,
                },
                timeout=10,
            )

            if resp.status_code != 200:
                logger.warning(f"Verifier returned {resp.status_code}")
                return None

            answer = resp.json()["choices"][0]["message"]["content"].strip().upper()
            if answer.startswith("YES"):
                return 0.9
            elif answer.startswith("NO"):
                return 0.1
            else:
                logger.debug(f"Verifier ambiguous: {answer}")
                return 0.5

        except Exception as e:
            logger.warning(f"Self-verification failed: {e}")
            return None

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        import os
        return os.getenv("OPENROUTER_API_KEY", "")
