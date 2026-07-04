"""Pipeline orchestrator — ties classifier → router → generation → cascade."""

import logging
import time
from typing import Optional

from src.config import RouterConfig, get_config
from src.models import (
    RouteRequest, RouteResponse, ClassificationResult,
    RoutingDecision, GenerationResult, Complexity,
)
from src.classifier.complexity import ComplexityClassifier
from src.router.engine import CostRouter
from src.router.cascade import Cascade
from src.router.client import OpenRouterClient
from src.constants import TIER_MODELS, DEFAULT_MODEL_PER_TIER

logger = logging.getLogger(__name__)


class RoutingPipeline:
    """End-to-end routing pipeline."""

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or get_config()
        self.classifier = ComplexityClassifier(method=self.config.complexity_method)
        self.router = CostRouter(
            confidence_override_threshold=self.config.confidence_override_threshold
        )
        self.cascade = Cascade(self.config)
        self.client = OpenRouterClient(
            api_key=self.config.openrouter_api_key,
            timeout=self.config.request_timeout_seconds,
            retry_count=self.config.rate_limit_retry_count,
        )
        self.history: list[RouteResponse] = []
        self._listeners = []

    def on_route(self, callback):
        """Register callback fired on every route (for dashboard WebSocket)."""
        self._listeners.append(callback)

    def route(self, request: RouteRequest) -> RouteResponse:
        """Full pipeline: classify → route → generate → [cascade]."""
        start = time.perf_counter()

        # Step 1: Classify
        classification = self.classifier.estimate(request.query)
        logger.info(
            f"Classified: complexity={classification.complexity} "
            f"task={classification.task_label} "
            f"confidence={classification.confidence:.2f}"
        )

        # Step 2: Route
        routing = self.router.route(
            request.query, classification, force_tier=request.force_tier
        )
        logger.info(
            f"Routed: tier={routing.tier} model={routing.model_id} "
            f"reason={routing.reason}"
        )

        # Step 3: Generate
        generation = self.client.generate(request.query, routing.model_id, routing.tier)

        # Step 4: Cascade (if enabled and fast-tier)
        cascade_override = request.cascade if request.cascade is not None else None
        cascade_enabled = (
            cascade_override if cascade_override is not None
            else self.config.cascade_enabled
        )

        if (
            cascade_enabled
            and generation.error is None
            and self.cascade.should_escalate(
                request.query, generation.response, routing.tier, classification.confidence
            )
        ):
            logger.info(f"Cascade: escalating from {routing.tier} → next tier")

            prev_tier = routing.tier
            next_tier = self._next_tier(routing.tier)
            if next_tier and next_tier != prev_tier:
                # Generate with the next tier up
                next_model = DEFAULT_MODEL_PER_TIER.get(
                    next_tier,
                    "meta-llama/llama-3.3-70b-instruct:free",
                )
                cascade_result = self.client.generate(
                    request.query, next_model, next_tier,
                )

                # Merge cascade info
                cascade_result.cascade_escalated = True
                cascade_result.cascade_from_tier = prev_tier
                cascade_result.cascade_to_tier = next_tier
                generation = cascade_result

                logger.info(
                    f"Cascade complete: {prev_tier} → {next_tier} "
                    f"({generation.tokens_in + generation.tokens_out} tokens)"
                )

        pipeline_ms = round((time.perf_counter() - start) * 1000, 1)
        generation.latency_ms = pipeline_ms

        # Step 5: Build response
        response = RouteResponse(
            query=request.query,
            response=generation.response,
            classification=classification,
            routing=routing,
            generation=generation,
        )

        # Store history + notify listeners
        self.history.append(response)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        for listener in self._listeners:
            try:
                listener(response)
            except Exception as e:
                logger.warning(f"Listener error: {e}")

        return response

    def _next_tier(self, current: str) -> Optional[str]:
        """Get the next tier up."""
        ladder = ["fast", "thinking", "deep"]
        try:
            idx = ladder.index(current)
            if idx < len(ladder) - 1:
                return ladder[idx + 1]
        except ValueError:
            pass
        return None

    def get_history(self, limit: int = 50) -> list[RouteResponse]:
        """Get recent routing history."""
        return self.history[-limit:]

    def get_stats(self) -> dict:
        """Get aggregate statistics."""
        total = len(self.history)
        if total == 0:
            return {"total_routes": 0}

        tier_counts = {}
        escalated = 0
        errors = 0
        total_tokens = 0
        total_latency = 0

        for r in self.history:
            tier_counts[r.routing.tier] = tier_counts.get(r.routing.tier, 0) + 1
            if r.generation.cascade_escalated:
                escalated += 1
            if r.generation.error:
                errors += 1
            total_tokens += r.generation.tokens_in + r.generation.tokens_out
            total_latency += r.generation.latency_ms

        return {
            "total_routes": total,
            "tier_distribution": tier_counts,
            "escalations": escalated,
            "escalation_rate": round(escalated / total * 100, 1) if total else 0,
            "errors": errors,
            "error_rate": round(errors / total * 100, 1) if total else 0,
            "avg_tokens": round(total_tokens / total, 0) if total else 0,
            "avg_latency_ms": round(total_latency / total, 1) if total else 0,
        }
