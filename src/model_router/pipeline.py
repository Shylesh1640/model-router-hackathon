"""Routing pipeline — intent → decompose → classify → route → generate.

Wires together all analysis stages before routing:
1. Intent detection (what kind of query)
2. Decomposition analysis (sub-tasks? vision content?)
3. Source of Truth lookup
4. Distance classification
5. Cost-aware model selection (flags bump tier)
6. Generation via OpenRouter
7. Optional Langfuse telemetry

Usage:
    from model_router import RoutingPipeline, get_config

    pipe = RoutingPipeline(get_config())
    result = pipe.route("What is the capital of France?")
    print(result.intent.intent, result.routing.tier)
"""

import logging
import time
from typing import Optional

from .config import RouterConfig, get_config
from .models import (
    RouteRequest, RouteResponse, ClassificationResult,
    RoutingDecision, GenerationResult, SourceQueryResult,
    IntentResult, DecompositionResult,
)
from .router import CostRouter
from .client import OpenRouterClient
from .store import SourceOfTruth
from .classify import DistanceClassifier
from .heatmap import HeatmapClassifier
from .search import WebSearcher
from .intent import IntentDetector
from .decompose import DecompositionAnalyzer
from .telemetry import get_telemetry, Telemetry
from .constants import DEFAULT_MODEL_PER_TIER, TIER_MODELS, ALL_MODELS

logger = logging.getLogger(__name__)


class RoutingPipeline:
    """Main routing pipeline — intent → decompose → classify → route → generate.

    Stages:
    0. Intent detection (what kind of query)
    1. Decomposition analysis (sub-tasks, vision content)
    2. Source of Truth lookup (embedding distance)
    3. Complexity classification (close / moderate / distant)
    4. Model selection (cheapest capable model, flags bump tier)
    5. Generation (OpenRouter API call with optional cascade)

    Optional Langfuse telemetry wraps the entire flow.
    """

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        telemetry: Optional[Telemetry] = None,
    ):
        self.config = config or get_config()
        self.sot = SourceOfTruth()
        self.classifier = HeatmapClassifier()
        self.router = CostRouter()
        self.intent_detector = IntentDetector()
        self.decomposer = DecompositionAnalyzer()
        self.client = OpenRouterClient(
            api_key=self.config.openrouter_api_key,
            timeout=self.config.request_timeout_seconds,
            retry_count=self.config.rate_limit_retry_count,
            base_delay=self.config.rate_limit_base_delay,
            max_delay=self.config.rate_limit_max_delay,
        )
        self.searcher = WebSearcher(search_url="http://localhost:8080")
        self.telemetry = telemetry or get_telemetry()

        # History for analytics
        self.history: list[RouteResponse] = []
        self._listeners = []

    def on_route(self, callback):
        """Register a callback invoked on every route completion."""
        self._listeners.append(callback)

    def route(self, request: RouteRequest) -> RouteResponse:
        """Execute the full routing pipeline for a query."""
        start = time.perf_counter()
        trace = self.telemetry.start_trace(
            request.query,
            metadata={"force_tier": request.force_tier},
        )

        # =====================================================================
        # STAGE 0: INTENT DETECTION
        # =====================================================================
        intent_span = trace.span(name="intent")
        intent = self.intent_detector.detect(request.query)
        intent_span.end(output={
            "intent": intent.intent,
            "confidence": intent.confidence,
            "method": intent.method,
        })

        # =====================================================================
        # STAGE 1: DECOMPOSITION ANALYSIS
        # =====================================================================
        decomp_span = trace.span(name="decomposition")
        decomposition = self.decomposer.analyze(request.query)
        decomp_span.end(output={
            "has_sub_tasks": decomposition.has_sub_tasks,
            "needs_reasoning": decomposition.needs_reasoning,
            "needs_vision": decomposition.needs_vision,
            "sub_task_count": len(decomposition.sub_tasks),
        })

        # Combine flags: decomposition flags + intent-level suggestions
        needs_reasoning = self._resolve_reasoning_flag(
            decomposition, intent, request.force_tier
        )
        needs_vision = decomposition.needs_vision

        # =====================================================================
        # STAGE 2: SOURCE OF TRUTH LOOKUP  (skipped for trivial intents)
        # =====================================================================
        _trivial_intents = {"general", "command", "summarization"}
        _skip_sot = (
            intent.intent in _trivial_intents
            and not needs_reasoning
            and not needs_vision
            and not request.force_tier
        )

        if _skip_sot:
            # Trivial intent → skip expensive embedding lookup, treat as empty SOT
            source_result = SourceQueryResult(query=request.query, total_docs=0)
            source_result.min_distance = 1.0
            distance = 1.0
        else:
            sot_span = trace.span(name="sot_lookup")
            source_result = self.sot.query(request.query)
            distance = source_result.min_distance
            sot_span.end(output={
                "distance": distance,
                "matches": len(source_result.matches),
                "total_docs": source_result.total_docs,
            })

        # =====================================================================
        # STAGE 3: CLASSIFY COMPLEXITY
        # =====================================================================
        classify_span = trace.span(name="classification")
        classification = self.classifier.classify(
            request.query, source_result,
            intent=intent,
            decomposition=decomposition,
        )
        classify_span.end(output={
            "complexity": classification.complexity,
            "task_label": classification.task_label,
            "confidence": classification.confidence,
        })

        # =====================================================================
        # STAGE 4: ROUTE — pick cheapest capable model
        # =====================================================================
        route_span = trace.span(name="routing")
        routing = self.router.route(
            request.query, classification,
            force_tier=request.force_tier,
            task_type=classification.task_label,
            needs_reasoning=needs_reasoning,
            needs_vision=needs_vision,
        )
        route_span.end(output={
            "tier": routing.tier,
            "model": routing.model_name,
            "reason": routing.reason,
        })

        # =====================================================================
        # STAGE 5: GENERATE
        # =====================================================================
        gen_span = trace.span(name="generation")
        source_context = self._build_source_context(source_result)
        task_label = classification.task_label
        search_results = []

        if task_label == "web_search":
            logger.info("Query needs web search (distance=%.2f)", distance)
            search_results = self.searcher.search(request.query)
            search_context = self.searcher.format_for_prompt(search_results)
            prompt = self._build_prompt(
                request.query, source_context, search_context=search_context
            )
            generation = self.client.generate(
                prompt, routing.model_id, routing.tier,
                fallback_models=self._fallbacks_for(routing.tier),
            )
            generation.web_search_used = True
            generation.web_search_results = search_results

        elif task_label == "deep_reasoning":
            logger.info("Query needs deep reasoning (distance=%.2f)", distance)
            search_results = self.searcher.search(request.query)
            steps = _deep_reasoning_steps(
                request.query, source_context, search_results
            )
            search_context = self.searcher.format_for_prompt(search_results)
            prompt = self._build_prompt(
                request.query, source_context,
                search_context=search_context,
                reasoning_steps=steps,
            )
            generation = self.client.generate(
                prompt, routing.model_id, routing.tier,
                fallback_models=self._fallbacks_for(routing.tier),
            )
            generation.web_search_used = bool(search_results)
            generation.web_search_results = search_results
            generation.deep_reasoning_used = True
            generation.reasoning_chain = [s["thought"] for s in steps]

        else:
            # Grounded — answer directly from source
            prompt = self._build_prompt(request.query, source_context)
            generation = self.client.generate(
                prompt, routing.model_id, routing.tier,
                fallback_models=self._fallbacks_for(routing.tier),
            )
        generation = self._maybe_cascade(
            request, generation, routing, classification
        )

        pipeline_ms = round((time.perf_counter() - start) * 1000, 1)
        generation.latency_ms = pipeline_ms

        # Source citations
        routing.source_citations = [
            m.content[:100] for m in source_result.matches[:3]
        ] if source_result.matches else []
        generation.source_docs_used = routing.source_citations

        gen_span.end(output={
            "tokens_in": generation.tokens_in,
            "tokens_out": generation.tokens_out,
            "latency_ms": generation.latency_ms,
            "model": generation.model_id,
            "error": generation.error,
        })

        response = RouteResponse(
            query=request.query,
            response=generation.response,
            classification=classification,
            routing=routing,
            generation=generation,
            intent=intent,
            decomposition=decomposition,
            source_query=source_result,
        )

        trace.end(output={
            "tier": routing.tier,
            "model": routing.model_name,
            "latency_ms": pipeline_ms,
            "intent": intent.intent,
            "needs_reasoning": needs_reasoning,
            "needs_vision": needs_vision,
        })

        self._record_response(response)
        return response

    @staticmethod
    def _resolve_reasoning_flag(
        decomposition: DecompositionResult,
        intent: IntentResult,
        force_tier: Optional[str],
    ) -> bool:
        """Combine decomposition and intent signals into reasoning flag.

        If force_tier is already 'thinking' or 'deep', no need to flag.
        """
        if force_tier in ("thinking", "deep"):
            return False
        if decomposition.needs_reasoning:
            return True
        return IntentDetector.needs_reasoning(intent.intent)

    def _build_source_context(
        self, source_result: SourceQueryResult
    ) -> str:
        if not source_result.matches:
            return ""
        parts = []
        for i, doc in enumerate(source_result.matches[:3]):
            parts.append(f"[Source {i+1}] {doc.content[:200]}")
        return "\n".join(parts)

    @staticmethod
    def _build_prompt(
        query: str,
        source_context: str,
        search_context: str = "",
        reasoning_steps: Optional[list] = None,
    ) -> str:
        """Build a prompt. No chatbot persona — just context + query."""
        parts = []

        if source_context:
            parts.append(f"Context:\n{source_context}")

        if search_context:
            parts.append(f"Web Search Results:\n{search_context}")

        if reasoning_steps:
            steps_text = "\n".join(
                f"Step {s.get('step', i+1)}: {s.get('thought', '')}"
                for i, s in enumerate(reasoning_steps)
            )
            parts.append(f"Reasoning Chain:\n{steps_text}")

        parts.append(f"Question: {query}")
        parts.append("Answer:")

        return "\n\n".join(parts)

    def _maybe_cascade(
        self,
        request: RouteRequest,
        generation: GenerationResult,
        routing: RoutingDecision,
        classification: ClassificationResult,
    ) -> GenerationResult:
        """Escalate to next tier on low confidence, with guards.

        Guards:
        - No cascade if the generation already errored or cascade is disabled.
        - No cascade if confidence is high or we're already at the deepest tier.
        - Max cascade hops (`cascade_max_hops`) — prevents infinite loops.
        - Token budget cap (`cascade_max_budget_tokens`) — prevents runaway cost.
        - Emergency fallback to openrouter/free if cascade also fails.
        """
        if generation.error or not self.config.cascade_enabled:
            return generation
        if classification.confidence > 0.7 or routing.tier == "deep":
            return generation

        # Track cascade hops on the request object
        request._cascade_hops = getattr(request, "_cascade_hops", 0) + 1
        if request._cascade_hops > self.config.cascade_max_hops:
            logger.warning(
                "Max cascade hops (%d) reached, returning current result",
                self.config.cascade_max_hops,
            )
            return generation

        # Track estimated token budget
        estimated = {"fast": 500, "thinking": 1500, "deep": 4000}.get(
            routing.tier, 1000
        )
        request._cascade_budget = (
            getattr(request, "_cascade_budget", 0) + estimated
        )
        if request._cascade_budget > self.config.cascade_max_budget_tokens:
            logger.warning(
                "Cascade budget exceeded (%d > %d), returning current result",
                request._cascade_budget,
                self.config.cascade_max_budget_tokens,
            )
            return generation

        next_tier = self._next_tier(routing.tier)
        if not next_tier:
            return generation

        next_model = DEFAULT_MODEL_PER_TIER.get(next_tier)
        if not next_model:
            return generation

        cascade_result = self.client.generate(
            request.query, next_model, next_tier
        )
        cascade_result.cascade_escalated = True
        cascade_result.cascade_from_tier = routing.tier
        cascade_result.cascade_to_tier = next_tier

        # Emergency fallback — if cascade also failed, try openrouter/free
        if cascade_result.error:
            fallback_id = "openrouter/free"
            logger.warning(
                "Cascade to %s (%s) failed: %s. Trying emergency fallback: %s",
                next_tier, next_model, cascade_result.error, fallback_id,
            )
            fallback = self.client.generate(
                request.query, fallback_id, "deep"
            )
            fallback.cascade_escalated = True
            fallback.cascade_from_tier = routing.tier
            fallback.cascade_to_tier = "emergency"
            return fallback

        return cascade_result

    @staticmethod
    def _next_tier(current: str) -> Optional[str]:
        """Return the next tier up, or None if already at max."""
        ladder = ["fast", "thinking", "deep"]
        try:
            idx = ladder.index(current)
            if idx >= len(ladder) - 1:
                return None
            return ladder[idx + 1]
        except ValueError:
            return None

    @staticmethod
    def _fallbacks_for(tier: str) -> list[str]:
        """Return alternate model IDs in the same tier for fallback.

        Appends openrouter/free as the ultimate catch-all so that if every
        specific free model is rate-limited / down, the meta-router picks
        whichever free model has capacity.
        """
        models = TIER_MODELS.get(tier, [])
        if not models:
            return ["openrouter/free"]
        # Sorted smallest-first (same as router picks), skip index 0 (primary)
        sorted_m = sorted(models, key=lambda m: m.total_params_b or 0)
        fallbacks = [m.openrouter_id for m in sorted_m[1:]]
        fallbacks.append("openrouter/free")
        return fallbacks

    def _record_response(self, response: RouteResponse):
        self.history.append(response)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        for listener in self._listeners:
            try:
                listener(response)
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Aggregate stats from routing history."""
        total = len(self.history)
        if total == 0:
            return {"total": 0}
        tiers = {}
        web = 0
        deep = 0
        reasoning_flags = 0
        vision_flags = 0
        for r in self.history:
            tiers[r.routing.tier] = tiers.get(r.routing.tier, 0) + 1
            if r.generation.web_search_used:
                web += 1
            if r.generation.deep_reasoning_used:
                deep += 1
            if r.decomposition and r.decomposition.needs_reasoning:
                reasoning_flags += 1
            if r.routing.needs_vision:
                vision_flags += 1
        return {
            "total_routes": total,
            "tier_distribution": tiers,
            "web_searches": web,
            "deep_reasoning": deep,
            "reasoning_flags": reasoning_flags,
            "vision_flags": vision_flags,
        }

    def get_history(self, limit: int = 50) -> list[RouteResponse]:
        return self.history[-limit:]


def _deep_reasoning_steps(
    query: str,
    source_context: str,
    search_results: list[dict],
) -> list[dict]:
    """Generate reasoning steps for deep tier queries."""
    steps = [
        {
            "step": 1, "type": "understand",
            "thought": f"Understanding the query: {query}",
            "source_check": bool(source_context),
        },
        {
            "step": 2, "type": "contextualize",
            "thought": "Relating to context",
            "source_relevant": bool(source_context),
        },
    ]
    search_text = (
        " ".join(r.get("snippet", "") for r in search_results[:3])
        if search_results
        else ""
    )
    if search_text:
        steps.append({
            "step": 3, "type": "search_integrate",
            "thought": f"Integrating search: {search_text[:100]}...",
            "source": "web",
        })
    steps.append({
        "step": len(steps) + 1, "type": "synthesize",
        "thought": "Synthesizing answer from context + search + reasoning",
    })
    return steps
