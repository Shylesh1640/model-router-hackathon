"""Routing engine — maps query complexity to model tier."""

import logging
from typing import Optional

from src.constants import TIER_MODELS, DEFAULT_MODEL_PER_TIER, ALL_MODELS, ModelInfo
from src.models import Complexity, ClassificationResult, RoutingDecision

logger = logging.getLogger(__name__)


class CostRouter:
    """Selects model tier from query complexity."""

    def route(
        self,
        query: str,
        classification: ClassificationResult,
        force_tier: Optional[str] = None,
        task_type: str = "general",
    ) -> RoutingDecision:
        if force_tier and force_tier in TIER_MODELS:
            model = self._pick_model(force_tier)
            return RoutingDecision(
                query=query, tier=force_tier,
                model_id=model.openrouter_id, model_name=model.name,
                complexity=classification.complexity,
                confidence=classification.confidence,
                reason=f"Forced tier: {force_tier}",
            )

        tier = self._complexity_to_tier(classification.complexity)
        model = self._pick_model(tier)

        cost_estimate = {"fast": 500, "thinking": 1500, "deep": 4000}.get(tier, 1000)

        return RoutingDecision(
            query=query, tier=tier,
            model_id=model.openrouter_id, model_name=model.name,
            complexity=classification.complexity,
            confidence=classification.confidence,
            reason=f"complexity={classification.complexity} tier={tier}",
            cost_estimate_tokens=cost_estimate,
            benchmark_scores=_get_benchmark_scores(model),
        )

    @staticmethod
    def _complexity_to_tier(complexity: str) -> str:
        mapping = {"close": "fast", "moderate": "thinking", "distant": "deep",
                    Complexity.CLOSE: "fast", Complexity.MODERATE: "thinking", Complexity.DISTANT: "deep"}
        return mapping.get(complexity, "thinking")

    @staticmethod
    def _pick_model(tier: str) -> ModelInfo:
        models = TIER_MODELS.get(tier, [])
        if not models:
            return _fallback(tier)
        scored = sorted(models, key=lambda m: m.total_params_b or 0, reverse=True)
        return scored[0]


def _fallback(tier: str) -> ModelInfo:
    fallback_id = DEFAULT_MODEL_PER_TIER.get(tier, "meta-llama/llama-3.3-70b-instruct:free")
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
        "coding": b.humaneval, "reasoning": b.mmlu_pro,
        "swe_bench": b.swe_bench_verified,
        "mmlu_pro": b.mmlu_pro, "humaneval": b.humaneval,
        "source": b.source,
    }
