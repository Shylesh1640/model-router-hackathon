"""Pipeline orchestrator — ties classifier → router → generation → cascade.

Enhanced with:
- Launch hook: scrapes provider for benchmarks on first run
- Benchmark-aware routing: uses SWE-bench/MMLU scores as support metrics
- Task decomposition: breaks complex queries into independently-routed sub-tasks
"""

import logging
import time
from typing import Optional

from src.config import RouterConfig, get_config
from src.models import (
    RouteRequest, RouteResponse, ClassificationResult,
    RoutingDecision, GenerationResult, Complexity,
    DecompositionPlan, SubTask,
)
from src.classifier.complexity import ComplexityClassifier
from src.router.engine import CostRouter
from src.router.cascade import Cascade
from src.router.client import OpenRouterClient
from src.decomposer.decomposer import TaskDecomposer, get_decomposer
from src.scraper.provider import BenchmarkResolver, get_resolver, refresh_model_pool
from src.constants import TIER_MODELS, DEFAULT_MODEL_PER_TIER

logger = logging.getLogger(__name__)


class RoutingPipeline:
    """End-to-end routing pipeline with launch hook and decomposition."""

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or get_config()

        # Launch hook: scrape provider for benchmarks on init
        self.benchmark_resolver = get_resolver(
            api_key=self.config.openrouter_api_key
        )
        self._run_launch_hook()

        # Core modules
        self.classifier = ComplexityClassifier(method=self.config.complexity_method)
        self.router = CostRouter(
            confidence_override_threshold=self.config.confidence_override_threshold,
            benchmark_resolver=self.benchmark_resolver,
        )
        self.cascade = Cascade(self.config)
        self.client = OpenRouterClient(
            api_key=self.config.openrouter_api_key,
            timeout=self.config.request_timeout_seconds,
            retry_count=self.config.rate_limit_retry_count,
        )
        self.decomposer = get_decomposer()

        # State
        self.history: list[RouteResponse] = []
        self._listeners = []

    def _run_launch_hook(self):
        """Initial launch hook: scrape provider for available models + benchmarks.

        Runs once at startup. If an API key is available, pulls the live model list
        and resolves benchmark scores. Otherwise uses the curated pool.
        """
        if self.config.openrouter_api_key:
            logger.info("Launch hook: scraping provider for model benchmarks...")
            try:
                count = refresh_model_pool(self.config.openrouter_api_key)
                logger.info(f"Launch hook complete: {count} models profiled")
            except Exception as e:
                logger.warning(f"Launch hook scrape failed: {e}, using curated pool")
        else:
            logger.info("No API key configured, using curated model pool")

    def on_route(self, callback):
        """Register callback fired on every route (for dashboard WebSocket)."""
        self._listeners.append(callback)

    def route(self, request: RouteRequest) -> RouteResponse:
        """Full pipeline: classify → route → generate → [cascade] → [decompose]."""
        start = time.perf_counter()

        # =====================================================================
        # STEP 0: Check if decomposition is needed
        # =====================================================================
        should_decompose = (
            request.decompose is True
            or (request.decompose is None and self.decomposer.should_decompose(request.query))
        )

        if should_decompose:
            return self._route_decomposed(request, start)

        # =====================================================================
        # SINGLE TASK PATH
        # =====================================================================
        return self._route_single(request, start)

    def _route_single(
        self, request: RouteRequest, start: float
    ) -> RouteResponse:
        """Route a single (non-decomposed) query through the pipeline."""
        # Step 1: Classify
        classification = self.classifier.estimate(request.query)

        # Step 2: Route (benchmark-aware)
        routing = self.router.route(
            request.query, classification,
            force_tier=request.force_tier,
            task_type=classification.task_label,
        )

        # Step 3: Generate
        generation = self.client.generate(
            request.query, routing.model_id, routing.tier
        )

        # Step 4: Cascade
        generation = self._maybe_cascade(
            request, generation, routing, classification
        )

        # Step 5: Build response
        pipeline_ms = round((time.perf_counter() - start) * 1000, 1)
        generation.latency_ms = pipeline_ms

        response = RouteResponse(
            query=request.query,
            response=generation.response,
            classification=classification,
            routing=routing,
            generation=generation,
        )

        self._record_response(response)
        return response

    def _route_decomposed(
        self, request: RouteRequest, start: float
    ) -> RouteResponse:
        """Decompose query, route each sub-task independently, assemble results."""
        logger.info(f"Decomposing query: {request.query[:60]}...")

        # Step 1: Decompose
        plan = self.decomposer.decompose(request.query)

        if len(plan.sub_tasks) <= 1:
            # Decomposition didn't produce useful splits — fall through to single path
            logger.info("Decomposition produced no splits, using single path")
            return self._route_single(request, start)

        # Step 2: Route + generate each sub-task independently
        for i, sub in enumerate(plan.sub_tasks):
            logger.info(f"Sub-task {sub.id}: {sub.description[:40]}...")

            cls = self.classifier.estimate(sub.description)
            routing = self.router.route(
                sub.description, cls,
                task_type=sub.task_type,
            )

            gen = self.client.generate(
                sub.description, routing.model_id, routing.tier
            )

            gen = self._maybe_cascade_simple(sub.description, gen, routing, cls)

            sub.complexity = cls.complexity
            sub.routing = routing
            sub.generation = gen

            # Short delay between sub-tasks to avoid rate limits
            if i < len(plan.sub_tasks) - 1:
                time.sleep(0.3)

        # Step 3: Assemble results
        assembled = self._assemble_results(plan)
        pipeline_ms = round((time.perf_counter() - start) * 1000, 1)

        # Use the most recent generation for the top-level response metadata
        last_gen = plan.sub_tasks[-1].generation if plan.sub_tasks else None
        last_routing = plan.sub_tasks[-1].routing if plan.sub_tasks else None

        # Overall classification from the original query
        overall_cls = self.classifier.estimate(request.query)

        response = RouteResponse(
            query=request.query,
            response=assembled,
            classification=overall_cls,
            routing=RoutingDecision(
                query=request.query,
                tier="decomposed",
                model_id=last_routing.model_id if last_routing else "decomposed",
                model_name=last_routing.model_name if last_routing else "Multi-Model",
                complexity=overall_cls.complexity,
                confidence=overall_cls.confidence,
                reason=f"Decomposed into {len(plan.sub_tasks)} sub-tasks | {plan.strategy}",
            ),
            generation=last_gen or GenerationResult(
                query=request.query, response=assembled,
                model_id="decomposed", tier="decomposed",
                tokens_in=0, tokens_out=0, latency_ms=pipeline_ms,
            ),
            decomposition=plan,
        )

        self._record_response(response)
        return response

    def _assemble_results(self, plan: DecompositionPlan) -> str:
        """Assemble sub-task results into a coherent response."""
        parts = []
        for i, sub in enumerate(plan.sub_tasks):
            result = sub.generation.response if sub.generation and sub.generation.response else "[no result]"
            parts.append(f"**Step {i+1}: {sub.description}**\n{result}\n")

        return "\n---\n".join(parts)

    def _maybe_cascade(
        self, request: RouteRequest, generation: GenerationResult,
        routing: RoutingDecision, classification: ClassificationResult,
    ) -> GenerationResult:
        """Run cascade if conditions are met."""
        if generation.error:
            return generation

        cascade_override = request.cascade if request.cascade is not None else None
        cascade_enabled = (
            cascade_override if cascade_override is not None
            else self.config.cascade_enabled
        )

        if not cascade_enabled:
            return generation

        if not self.cascade.should_escalate(
            request.query, generation.response, routing.tier, classification.confidence
        ):
            return generation

        prev_tier = routing.tier
        next_tier = self._next_tier(routing.tier)
        if next_tier and next_tier != prev_tier:
            next_model = DEFAULT_MODEL_PER_TIER.get(
                next_tier, "meta-llama/llama-3.3-70b-instruct:free"
            )
            cascade_result = self.client.generate(
                request.query, next_model, next_tier,
            )
            cascade_result.cascade_escalated = True
            cascade_result.cascade_from_tier = prev_tier
            cascade_result.cascade_to_tier = next_tier
            return cascade_result

        return generation

    def _maybe_cascade_simple(
        self, query: str, generation: GenerationResult,
        routing: RoutingDecision, classification: ClassificationResult,
    ) -> GenerationResult:
        """Simpler cascade for sub-tasks (no request object)."""
        if generation.error or not self.config.cascade_enabled:
            return generation

        if not self.cascade.should_escalate(
            query, generation.response, routing.tier, classification.confidence
        ):
            return generation

        next_tier = self._next_tier(routing.tier)
        if next_tier:
            next_model = DEFAULT_MODEL_PER_TIER.get(
                next_tier, "meta-llama/llama-3.3-70b-instruct:free"
            )
            cascade_result = self.client.generate(query, next_model, next_tier)
            cascade_result.cascade_escalated = True
            cascade_result.cascade_from_tier = routing.tier
            cascade_result.cascade_to_tier = next_tier
            return cascade_result

        return generation

    def _next_tier(self, current: str) -> Optional[str]:
        ladder = ["fast", "thinking", "deep"]
        try:
            idx = ladder.index(current)
            if idx < len(ladder) - 1:
                return ladder[idx + 1]
        except ValueError:
            pass
        return None

    # =========================================================================
    # HISTORY & STATS
    # =========================================================================

    def _record_response(self, response: RouteResponse):
        """Store response and notify listeners."""
        self.history.append(response)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        for listener in self._listeners:
            try:
                listener(response)
            except Exception as e:
                logger.warning(f"Listener error: {e}")

    def get_history(self, limit: int = 50) -> list[RouteResponse]:
        return self.history[-limit:]

    def get_stats(self) -> dict:
        total = len(self.history)
        if total == 0:
            return {"total_routes": 0}

        tier_counts = {}
        decomposed_count = 0
        escalated = 0
        errors = 0
        total_tokens = 0
        total_latency = 0
        total_sub_tasks = 0

        for r in self.history:
            tier_counts[r.routing.tier] = tier_counts.get(r.routing.tier, 0) + 1
            if r.routing.tier == "decomposed":
                decomposed_count += 1
                if r.decomposition:
                    total_sub_tasks += len(r.decomposition.sub_tasks)
            if r.generation.cascade_escalated:
                escalated += 1
            if r.generation.error:
                errors += 1
            total_tokens += r.generation.tokens_in + r.generation.tokens_out
            total_latency += r.generation.latency_ms

        return {
            "total_routes": total,
            "tier_distribution": tier_counts,
            "decomposed": decomposed_count,
            "total_sub_tasks": total_sub_tasks,
            "escalations": escalated,
            "escalation_rate": round(escalated / total * 100, 1) if total else 0,
            "errors": errors,
            "error_rate": round(errors / total * 100, 1) if total else 0,
            "avg_tokens": round(total_tokens / total, 0) if total else 0,
            "avg_latency_ms": round(total_latency / total, 1) if total else 0,
        }
