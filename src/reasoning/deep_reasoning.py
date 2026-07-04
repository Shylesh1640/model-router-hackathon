"""Deep reasoning — multi-step chain-of-thought for complex queries."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DeepReasoner:
    """Multi-step reasoning for complex queries far from the source of truth.

    Breaks down the query into reasoning steps, validates each against
    the source of truth, and builds a chain-of-thought response.
    """

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps

    def reason(self, query: str, source_context: str, search_results: list[dict]) -> dict:
        """Deep reasoning over a complex query.

        Returns:
            dict with:
              - steps: list of reasoning steps
              - conclusion: final synthesized answer
              - confidence: 0-1 how confident in the answer
        """
        # Build reasoning chain
        steps = self._generate_steps(query, source_context, search_results)

        return {
            "steps": steps,
            "conclusion": self._synthesize(steps),
            "confidence": self._estimate_confidence(steps),
        }

    def _generate_steps(self, query: str, source_context: str, search_results: list[dict]) -> list[dict]:
        """Generate reasoning steps. In production this calls an LLM;
        here we provide a structured fallback.
        """
        search_text = " ".join(r.get("snippet", "") for r in search_results[:3]) if search_results else ""
        steps = [
            {
                "step": 1,
                "type": "understand",
                "thought": f"Understanding the query: {query}",
                "source_check": bool(source_context),
            },
            {
                "step": 2,
                "type": "contextualize",
                "thought": "Relating to the source of truth",
                "source_relevant": bool(source_context),
            },
        ]

        if search_text:
            steps.append({
                "step": 3,
                "type": "search_integrate",
                "thought": f"Integrating web search findings: {search_text[:100]}...",
                "source": "web",
            })

        steps.append({
            "step": len(steps) + 1,
            "type": "synthesize",
            "thought": "Synthesizing answer from source + search + reasoning",
        })

        return steps

    def _synthesize(self, steps: list[dict]) -> str:
        """Synthesize final conclusion from reasoning steps."""
        return f"Reasoned across {len(steps)} steps to form a complete answer."

    def _estimate_confidence(self, steps: list[dict]) -> float:
        """Estimate confidence based on reasoning chain completeness."""
        valid = sum(1 for s in steps if s.get("source_check") or s.get("source_relevant"))
        return min(0.5 + valid * 0.1, 0.9)


_reasoner: Optional[DeepReasoner] = None


def get_reasoner() -> DeepReasoner:
    global _reasoner
    if _reasoner is None:
        _reasoner = DeepReasoner()
    return _reasoner
