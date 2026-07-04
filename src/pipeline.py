"""Pipeline — Safety → Source of Truth → Distance-based routing → Generation."""

import logging
import time
from typing import Optional

from src.config import RouterConfig, get_config
from src.models import (
    RouteRequest, RouteResponse, ClassificationResult,
    RoutingDecision, GenerationResult, SafetyResult, SourceQueryResult,
)
from src.classifier.complexity import ComplexityClassifier
from src.router.engine import CostRouter
from src.router.cascade import Cascade
from src.router.client import OpenRouterClient
from src.sot.source_of_truth import SourceOfTruth, get_sot
from src.sot.safety import SafetyGuard, get_safety
from src.search.web_search import WebSearcher, get_searcher
from src.reasoning.deep_reasoning import DeepReasoner, get_reasoner
from src.constants import DEFAULT_MODEL_PER_TIER

logger = logging.getLogger(__name__)


class ChatbotPipeline:
    """Full chatbot pipeline with Source of Truth at the core."""

    def __init__(self, config: Optional[RouterConfig] = None, domain: str = "this knowledge base"):
        self.config = config or get_config()
        self.domain = domain

        # Core modules
        self.sot: SourceOfTruth = get_sot()
        self.safety: SafetyGuard = get_safety(domain=domain)
        self.classifier = ComplexityClassifier(method="hybrid")
        self.router = CostRouter(confidence_override_threshold=0.4)
        self.cascade = Cascade(self.config)
        self.client = OpenRouterClient(
            api_key=self.config.openrouter_api_key,
            timeout=self.config.request_timeout_seconds,
            retry_count=self.config.rate_limit_retry_count,
        )
        self.searcher: WebSearcher = get_searcher(
            search_url="http://localhost:8080",
        )
        self.reasoner: DeepReasoner = get_reasoner()

        # State
        self.history: list[RouteResponse] = []
        self._listeners = []

    def on_route(self, callback):
        self._listeners.append(callback)

    def route(self, request: RouteRequest) -> RouteResponse:
        """Full pipeline."""
        start = time.perf_counter()

        # =====================================================================
        # STEP 1: EMBED QUERY
        # =====================================================================
        # We need the embedding early for both SOT lookup and safety distance check.

        # =====================================================================
        # STEP 2: SOURCE OF TRUTH LOOKUP
        # =====================================================================
        source_result = self.sot.query(request.query)
        distance = source_result.min_distance

        # =====================================================================
        # STEP 3: SAFETY CHECK (uses distance as proxy for off-topic)
        # =====================================================================
        safety_result = self.safety.check(request.query, source_distance=distance)

        if not safety_result.safe:
            # Blocked — harmful content
            response = RouteResponse(
                query=request.query,
                response=safety_result.rebuke_message,
                classification=ClassificationResult(
                    query=request.query, complexity="close", task_label="blocked",
                    confidence=1.0, method="safety", source_distance=distance,
                ),
                routing=RoutingDecision(
                    query=request.query, tier="blocked",
                    model_id="none", model_name="Safety Guard",
                    complexity="close", confidence=1.0,
                    reason=f"Blocked: {safety_result.category}",
                ),
                generation=GenerationResult(
                    query=request.query, response=safety_result.rebuke_message,
                    model_id="none", tier="blocked",
                    tokens_in=0, tokens_out=0, latency_ms=0,
                ),
                safety=safety_result,
                source_query=source_result,
                rebuked=True,
            )
            self._record_response(response)
            return response

        if safety_result.flagged and safety_result.category == "off_topic":
            # Off-topic — gentle rebuke
            response = RouteResponse(
                query=request.query,
                response=safety_result.rebuke_message,
                classification=ClassificationResult(
                    query=request.query, complexity="distant", task_label="off_topic",
                    confidence=0.9, method="sot_distance", source_distance=distance,
                ),
                routing=RoutingDecision(
                    query=request.query, tier="rebuked",
                    model_id="none", model_name="Domain Guard",
                    complexity="distant", confidence=0.9,
                    reason=f"Off-topic: {safety_result.reason}",
                ),
                generation=GenerationResult(
                    query=request.query, response=safety_result.rebuke_message,
                    model_id="none", tier="rebuked",
                    tokens_in=0, tokens_out=0, latency_ms=0,
                ),
                safety=safety_result,
                source_query=source_result,
                rebuked=True,
            )
            self._record_response(response)
            return response

        # =====================================================================
        # STEP 4: CLASSIFY BY DISTANCE
        # =====================================================================
        complexity = self.sot.complexity_from_distance(distance)
        task_label = self._task_from_distance(distance, source_result)

        classification = ClassificationResult(
            query=request.query,
            complexity=complexity,
            task_label=task_label,
            confidence=max(0.5, 1.0 - distance),
            method="sot_distance",
            source_distance=distance,
        )

        # =====================================================================
        # STEP 5: ROUTE
        # =====================================================================
        routing = self.router.route(
            request.query, classification,
            force_tier=request.force_tier,
            task_type=classification.task_label,
        )

        # =====================================================================
        # STEP 6: GENERATE (context depends on tier)
        # =====================================================================
        source_context = self._build_source_context(source_result)
        search_results = []
        reasoning_chain = []

        if task_label == "web_search":
            logger.info(f"Query needs web search (distance={distance:.2f})")
            search_results = self.searcher.search(request.query)
            search_context = self.searcher.format_for_prompt(search_results)
            prompt = self._build_prompt(request.query, source_context, search_context=search_context)
            generation = self.client.generate(prompt, routing.model_id, routing.tier)
            generation.web_search_used = True
            generation.web_search_results = search_results

        elif task_label == "deep_reasoning":
            logger.info(f"Query needs deep reasoning (distance={distance:.2f})")
            search_results = self.searcher.search(request.query)
            reasoning = self.reasoner.reason(request.query, source_context, search_results)
            reasoning_chain = reasoning.get("steps", [])
            search_context = self.searcher.format_for_prompt(search_results)
            prompt = self._build_prompt(
                request.query, source_context,
                search_context=search_context,
                reasoning_steps=reasoning.get("steps", []),
            )
            generation = self.client.generate(prompt, routing.model_id, routing.tier)
            generation.web_search_used = bool(search_results)
            generation.web_search_results = search_results
            generation.deep_reasoning_used = True
            generation.reasoning_chain = [s.get("thought", "") for s in reasoning_chain]
            generation.tier = "deep_reasoning"

        else:
            # Grounded — answer directly from source
            prompt = self._build_prompt(request.query, source_context)
            generation = self.client.generate(prompt, routing.model_id, routing.tier)
            generation.tier = "grounded"

        # Cascade if needed
        generation = self._maybe_cascade(request, generation, routing, classification)

        pipeline_ms = round((time.perf_counter() - start) * 1000, 1)
        generation.latency_ms = pipeline_ms

        # Source citations
        routing.source_citations = [
            m.content[:100] for m in source_result.matches[:3]
        ] if source_result.matches else []
        generation.source_docs_used = routing.source_citations

        response = RouteResponse(
            query=request.query,
            response=generation.response,
            classification=classification,
            routing=routing,
            generation=generation,
            safety=safety_result,
            source_query=source_result,
        )

        self._record_response(response)
        return response

    def _task_from_distance(self, distance: float, source_result: SourceQueryResult) -> str:
        """Determine the task type based on source distance."""
        if distance < self.sot.CLOSE_THRESHOLD:
            return "grounded"
        elif distance < self.sot.MODERATE_THRESHOLD:
            return "web_search"
        else:
            return "deep_reasoning"

    def _build_source_context(self, source_result: SourceQueryResult) -> str:
        if not source_result.matches:
            return ""
        parts = []
        for i, doc in enumerate(source_result.matches[:3]):
            parts.append(f"[Source {i+1}] {doc.content[:200]}")
        return "\n".join(parts)

    def _build_prompt(
        self,
        query: str,
        source_context: str,
        search_context: str = "",
        reasoning_steps: Optional[list] = None,
    ) -> str:
        domain_prompt = (
            f"You are a helpful assistant for {self.domain}. "
            f"Answer based on the provided sources. "
            f"If the sources don't contain enough information, say so."
        )

        parts = [f"System: {domain_prompt}"]

        if source_context:
            parts.append(f"\nSource of Truth:\n{source_context}")

        if search_context:
            parts.append(f"\nWeb Search Results:\n{search_context}")

        if reasoning_steps:
            steps_text = "\n".join(
                f"Step {s.get('step', i+1)}: {s.get('thought', '')}"
                for i, s in enumerate(reasoning_steps)
            )
            parts.append(f"\nReasoning Chain:\n{steps_text}")

        parts.append(f"\nUser: {query}")
        parts.append("\nAssistant:")

        return "\n\n".join(parts)

    def _maybe_cascade(
        self, request: RouteRequest, generation: GenerationResult,
        routing: RoutingDecision, classification: ClassificationResult,
    ) -> GenerationResult:
        if generation.error or not self.config.cascade_enabled:
            return generation
        if not self.cascade.should_escalate(
            request.query, generation.response, routing.tier, classification.confidence
        ):
            return generation
        next_tier = self._next_tier(routing.tier)
        if next_tier:
            next_model = DEFAULT_MODEL_PER_TIER.get(next_tier, "meta-llama/llama-3.3-70b-instruct:free")
            cascade_result = self.client.generate(request.query, next_model, next_tier)
            cascade_result.cascade_escalated = True
            cascade_result.cascade_from_tier = routing.tier
            cascade_result.cascade_to_tier = next_tier
            return cascade_result
        return generation

    def _next_tier(self, current: str) -> Optional[str]:
        ladder = ["grounded", "web_search", "deep_reasoning"]
        try:
            idx = ladder.index(current)
            return ladder[min(idx + 1, len(ladder) - 1)]
        except ValueError:
            return "deep_reasoning"

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
        total = len(self.history)
        if total == 0:
            return {"total": 0}
        tiers = {}
        rebuked = 0
        web = 0
        deep = 0
        for r in self.history:
            tiers[r.routing.tier] = tiers.get(r.routing.tier, 0) + 1
            if r.rebuked:
                rebuked += 1
            if r.generation.web_search_used:
                web += 1
            if r.generation.deep_reasoning_used:
                deep += 1
        return {
            "total_routes": total,
            "tier_distribution": tiers,
            "rebuked": rebuked,
            "web_searches": web,
            "deep_reasoning": deep,
        }
