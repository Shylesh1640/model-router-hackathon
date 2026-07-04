"""Cost-aware routing engine — picks the cheapest model that fits the complexity.

Enhanced with benchmark-aware model selection:
- Uses SWE-bench, MMLU, HumanEval scores as routing support metrics
- Picks the best model for the *type* of work (code → coding benchmark, reasoning → MMLU)
- Falls back to parameter count when benchmarks are unknown
"""

import logging
import random
from typing import Optional

from src.constants import TIER_MODELS, DEFAULT_MODEL_PER_TIER, ALL_MODELS, ModelInfo
from src.models import Complexity, ClassificationResult, RoutingDecision
from src.scraper.provider import BenchmarkResolver, get_resolver

logger = logging.getLogger(__name__)


class CostRouter:
    """Selects the optimal model tier based on complexity, confidence, and benchmarks."""

    def __init__(
        self,
        confidence_override_threshold: float = 0.4,
        benchmark_resolver: Optional[BenchmarkResolver] = None,
    ):
        self.confidence_override_threshold = confidence_override_threshold
        self.benchmark_resolver = benchmark_resolver

    def route(
        self,
        query: str,
        classification: ClassificationResult,
        force_tier: Optional[str] = None,
        task_type: str = "general",
    ) -> RoutingDecision:
        """Route a query to the optimal model tier.

        Args:
            query: The user's query
            classification: Complexity + task classification
            force_tier: Force a specific tier (bypasses routing logic)
            task_type: Type of work — "code" | "reasoning" | "general"
                       Affects which benchmark score is weighted highest
        """
        # Force tier override (API parameter)
        if force_tier and force_tier in TIER_MODELS:
            model = self._pick_model(force_tier, task_type)
            return RoutingDecision(
                query=query, tier=force_tier,
                model_id=model.openrouter_id, model_name=model.name,
                complexity=classification.complexity,
                confidence=classification.confidence,
                reason=f"Forced tier: {force_tier}",
            )

        # Step 1: Map complexity to base tier
        base_tier = self._complexity_to_tier(classification.complexity)

        # Step 2: Apply confidence override (low confidence → bump up)
        tier = base_tier
        override_applied = False
        if classification.confidence < self.confidence_override_threshold:
            bumped = self._bump_tier(tier)
            if bumped != tier:
                logger.info(
                    f"Confidence {classification.confidence:.2f} < "
                    f"{self.confidence_override_threshold}: bumped {tier} → {bumped}"
                )
                tier = bumped
                override_applied = True

        # Step 3: Pick model — benchmark-aware if resolver available
        bench_scores = None
        if classification.complexity == Complexity.DISTANT or classification.complexity == "distant":
            model = self._pick_best_in_tier(tier, task_type, classification.task_label)
        else:
            model = self._pick_model(tier, task_type)

        # Step 4: Attach benchmark data if available
        bench_scores = self._get_benchmark_scores(model)

        reasons = []
        reasons.append(f"complexity={classification.complexity}")
        if bench_scores:
            reasons.append(f"benchmark={bench_scores.get('overall', 'N/A')}")
        if override_applied:
            reasons.append("confidence-override")
        reasons.append(f"tier={tier}")

        cost_estimate = self._estimate_cost(tier)

        return RoutingDecision(
            query=query, tier=tier,
            model_id=model.openrouter_id, model_name=model.name,
            complexity=classification.complexity,
            confidence=classification.confidence,
            reason=" | ".join(reasons),
            cost_estimate_tokens=cost_estimate,
            benchmark_scores=bench_scores,
        )

    # ------------------------------------------------------------------
    # Tier mapping
    # ------------------------------------------------------------------

    def _complexity_to_tier(self, complexity: str) -> str:
        mapping = {
            Complexity.CLOSE: "fast",
            Complexity.MODERATE: "thinking",
            Complexity.DISTANT: "deep",
            "close": "fast",
            "moderate": "thinking",
            "distant": "deep",
        }
        return mapping.get(complexity, "thinking")

    def _bump_tier(self, current: str) -> str:
        ladder = ["fast", "thinking", "deep"]
        try:
            idx = ladder.index(current)
            return ladder[min(idx + 1, len(ladder) - 1)]
        except ValueError:
            return "thinking"

    # ------------------------------------------------------------------
    # Benchmark-aware model selection
    # ------------------------------------------------------------------

    def _pick_model(self, tier: str, task_type: str = "general") -> ModelInfo:
        """Pick a model from the tier, preferring benchmark-proven ones."""
        models = TIER_MODELS.get(tier, [])
        if not models:
            return self._fallback(tier)

        # If we have benchmark data, prefer higher-scored models
        if self.benchmark_resolver:
            scored = self._score_models(models, task_type)
            if scored:
                return scored[0][1]

        # Random pick — simple load balancing
        return random.choice(models)

    def _pick_best_in_tier(
        self, tier: str, task_type: str = "general", task_label: str = ""
    ) -> ModelInfo:
        """Pick the best model in a tier using benchmarks and task type."""
        models = TIER_MODELS.get(tier, [])
        if not models:
            return self._fallback(tier)

        # If we have benchmark data, use it
        if self.benchmark_resolver:
            scored = self._score_models(models, task_type)
            if scored:
                return scored[0][1]

        # Fallback: parameter count
        scored = [(m.total_params_b or 0, m) for m in models]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def _score_models(
        self, models: list[ModelInfo], task_type: str
    ) -> list[tuple[float, ModelInfo]]:
        """Score models by benchmark relevance to the task type.

        Returns sorted list of (score, model) tuples.
        """
        scored = []
        for m in models:
            if m.benchmarks:
                if task_type == "code":
                    score = m.benchmarks.coding_score or m.benchmarks.overall_score or (m.total_params_b or 0) * 0.5
                elif task_type == "reasoning":
                    score = m.benchmarks.reasoning_score or m.benchmarks.overall_score or (m.total_params_b or 0) * 0.5
                else:
                    score = m.benchmarks.overall_score or (m.total_params_b or 0) * 0.5
            else:
                # No benchmark data — approximate from params
                score = (m.total_params_b or 0) * 0.5

            scored.append((score, m))

        scored.sort(key=lambda x: -x[0])
        return scored

    def _get_benchmark_scores(self, model: ModelInfo) -> Optional[dict]:
        """Get benchmark scores for a model as a dict for the routing decision."""
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

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------

    def _fallback(self, tier: str) -> ModelInfo:
        fallback_id = DEFAULT_MODEL_PER_TIER.get(tier, "meta-llama/llama-3.3-70b-instruct:free")
        for m in ALL_MODELS:
            if m.openrouter_id == fallback_id:
                return m
        return ALL_MODELS[0]

    def _estimate_cost(self, tier: str) -> int:
        estimates = {"fast": 500, "thinking": 1500, "deep": 4000}
        return estimates.get(tier, 1000)
