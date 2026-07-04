"""Cost-aware routing engine — picks the cheapest model that fits the complexity."""

import logging
import random
from typing import Optional

from src.constants import (
    TIER_MODELS, DEFAULT_MODEL_PER_TIER, ALL_MODELS, ModelInfo,
)
from src.models import Complexity, ClassificationResult, RoutingDecision

logger = logging.getLogger(__name__)


class CostRouter:
    """Selects the optimal model tier based on complexity and confidence."""

    def __init__(self, confidence_override_threshold: float = 0.4):
        self.confidence_override_threshold = confidence_override_threshold

    def route(
        self,
        query: str,
        classification: ClassificationResult,
        force_tier: Optional[str] = None,
    ) -> RoutingDecision:
        """Route a query to the optimal model tier."""
        # Force tier override (API parameter)
        if force_tier and force_tier in TIER_MODELS:
            model = self._pick_model(force_tier)
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

        # Step 3: Pick specific model from the tier
        model = self._pick_model(tier)
        if classification.complexity == Complexity.HARD:
            # For hard queries, prefer the most capable model in the tier
            model = self._pick_best_in_tier(tier)

        reasons = []
        reasons.append(f"complexity={classification.complexity}")
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
            override_applied=override_applied,
        )

    def _complexity_to_tier(self, complexity: str) -> str:
        """Map complexity level to routing tier."""
        mapping = {
            Complexity.SIMPLE: "fast",
            Complexity.MEDIUM: "thinking",
            Complexity.HARD: "deep",
        }
        return mapping.get(complexity, "thinking")

    def _bump_tier(self, current: str) -> str:
        """Bump one tier up."""
        ladder = ["fast", "thinking", "deep"]
        try:
            idx = ladder.index(current)
            return ladder[min(idx + 1, len(ladder) - 1)]
        except ValueError:
            return "thinking"

    def _pick_model(self, tier: str) -> ModelInfo:
        """Pick a model from the tier (randomized for load balancing)."""
        models = TIER_MODELS.get(tier, [])
        if not models:
            # Fallback to default
            fallback_id = DEFAULT_MODEL_PER_TIER.get(tier, "meta-llama/llama-3.3-70b-instruct:free")
            for m in ALL_MODELS:
                if m.openrouter_id == fallback_id:
                    return m
            return ALL_MODELS[0]

        # Random pick — simple load balancing across available models
        return random.choice(models)

    def _pick_best_in_tier(self, tier: str) -> ModelInfo:
        """Pick the most capable model in a tier (by total params)."""
        models = TIER_MODELS.get(tier, [])
        if not models:
            return self._pick_model(tier)
        scored = [(m.total_params_b or 0, m) for m in models]
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def _estimate_cost(self, tier: str) -> int:
        """Estimate likely token cost for a query at this tier."""
        estimates = {"fast": 500, "thinking": 1500, "deep": 4000}
        return estimates.get(tier, 1000)
