"""Routing engine — maps query complexity to cheapest capable model tier.

Considers intent and decomposition flags to optionally bump tier:
- needs_reasoning → minimum thinking tier
- needs_vision → selects a multimodal model from the pool
"""

import logging
from typing import Optional

from .constants import TIER_MODELS, DEFAULT_MODEL_PER_TIER, ALL_MODELS, ModelInfo
from .models import Complexity, ClassificationResult, RoutingDecision

logger = logging.getLogger(__name__)

_TIER_ORDER = ["fast", "thinking", "deep"]
_VISION_CAPABLE_IDS = {
    "nvidia/nemotron-nano-12b-v2-vl:free",
    # Add more vision-capable free models as they appear in the pool
}


class CostRouter:
    """Selects the cheapest model tier that can handle a given query complexity.

    Flags from intent/decomposition stages can bump the minimum tier:
    - needs_reasoning=True → at least 'thinking' tier
    - needs_vision=True → picks vision-capable model if available

    Usage:
        router = CostRouter()
        decision = router.route("hello world", classification)
        print(decision.model_name, decision.tier)
    """

    def route(
        self,
        query: str,
        classification: ClassificationResult,
        force_tier: Optional[str] = None,
        task_type: str = "general",
        needs_reasoning: bool = False,
        needs_vision: bool = False,
    ) -> RoutingDecision:
        if force_tier and force_tier in TIER_MODELS:
            model = self._pick_model(force_tier, needs_vision=needs_vision)
            return RoutingDecision(
                query=query, tier=force_tier,
                model_id=model.openrouter_id, model_name=model.name,
                complexity=classification.complexity,
                confidence=classification.confidence,
                reason=f"Forced tier: {force_tier}",
                needs_vision=needs_vision and model.is_multimodal,
            )

        base_tier = self._complexity_to_tier(classification.complexity)
        tier = self._resolve_tier(base_tier, needs_reasoning)

        model = self._pick_model(tier, needs_vision=needs_vision)

        # Build reason string
        reasons = [f"complexity={classification.complexity} tier={tier}"]
        if needs_reasoning and tier != base_tier:
            reasons.append(f"escalated for reasoning (base={base_tier})")
        if needs_vision:
            reasons.append("vision-capable model selected")

        cost_estimate = {"fast": 500, "thinking": 1500, "deep": 4000}.get(tier, 1000)

        return RoutingDecision(
            query=query, tier=tier,
            model_id=model.openrouter_id, model_name=model.name,
            complexity=classification.complexity,
            confidence=classification.confidence,
            reason="; ".join(reasons),
            cost_estimate_tokens=cost_estimate,
            benchmark_scores=_get_benchmark_scores(model),
            needs_vision=needs_vision and model.is_multimodal,
        )

    @staticmethod
    def _resolve_tier(base_tier: str, needs_reasoning: bool) -> str:
        """Apply flags to determine effective tier.

        If needs_reasoning, floor at 'thinking' tier.
        """
        if not needs_reasoning:
            return base_tier
        base_idx = _TIER_ORDER.index(base_tier) if base_tier in _TIER_ORDER else 0
        thinking_idx = _TIER_ORDER.index("thinking")
        if base_idx < thinking_idx:
            return "thinking"
        return base_tier

    @staticmethod
    def _complexity_to_tier(complexity: str) -> str:
        mapping = {
            "close": "fast", "moderate": "thinking", "distant": "deep",
            Complexity.CLOSE: "fast", Complexity.MODERATE: "thinking",
            Complexity.DISTANT: "deep",
        }
        return mapping.get(complexity, "thinking")

    @staticmethod
    def _pick_model(tier: str, needs_vision: bool = False) -> ModelInfo:
        """Pick best model for tier, preferring vision-capable if needed.

        Sorts by SMALLEST params first — less popular, less rate-limited,
        cheaper for the caller. Cascade handles escalation if the small
        model's output quality is insufficient.
        """
        models = TIER_MODELS.get(tier, [])
        if not models:
            return _fallback_model(tier)

        if needs_vision:
            vision_models = [m for m in models if m.is_multimodal]
            if vision_models:
                # For vision: pick the smallest capable model (less loaded)
                return min(vision_models, key=lambda m: m.total_params_b or 0)

        # Pick the SMALLEST model — cheapest, least rate-limited
        return min(models, key=lambda m: m.total_params_b or 0)


def _fallback_model(tier: str) -> ModelInfo:
    fallback_id = DEFAULT_MODEL_PER_TIER.get(
        tier, "meta-llama/llama-3.3-70b-instruct:free"
    )
    for m in ALL_MODELS:
        if m.openrouter_id == fallback_id:
            return m
    return ALL_MODELS[0]


def _get_benchmark_scores(model: ModelInfo) -> Optional[dict]:
    if not model.benchmarks:
        return None
    b = model.benchmarks
    return {
        "overall": round(b.overall_score, 1) if b.overall_score else None,
        "coding": b.humaneval,
        "reasoning": b.mmlu_pro,
        "swe_bench": b.swe_bench_verified,
        "mmlu_pro": b.mmlu_pro,
        "humaneval": b.humaneval,
        "source": b.source,
    }
