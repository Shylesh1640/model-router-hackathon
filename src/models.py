"""Shared data models for the Source-of-Truth chatbot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class Complexity:
    """Query complexity levels — determined by distance from source of truth."""
    CLOSE = "close"        # distance < 0.3 → answer from source, fast model
    MODERATE = "moderate"  # distance 0.3-0.6 → web search, thinking model  
    DISTANT = "distant"    # distance > 0.6 → deep reasoning, deep model

    CHOICES = [CLOSE, MODERATE, DISTANT]


# =============================================================================
# BENCHMARK PROFILES (simplified — primary metric is SOT distance now)
# =============================================================================

@dataclass
class BenchmarkProfile:
    """Benchmark scores used as routing support metrics."""
    swe_bench_verified: Optional[float] = None
    mmlu_pro: Optional[float] = None
    humaneval: Optional[float] = None
    livecodebench: Optional[float] = None
    simpleqa: Optional[float] = None
    source: str = "manual"

    @property
    def overall_score(self) -> Optional[float]:
        scores = [s for s in [self.mmlu_pro, self.humaneval, self.swe_bench_verified] if s is not None]
        return sum(scores) / len(scores) if scores else None


# =============================================================================
# SOURCE OF TRUTH
# =============================================================================

@dataclass
class SourceDocument:
    """A single document in the source of truth."""
    id: str
    content: str
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)
    source: str = ""  # e.g. "manual", "upload", "web"


@dataclass
class SourceQueryResult:
    """Result of querying the source of truth."""
    query: str
    query_embedding: Optional[list[float]] = None
    matches: list[SourceDocument] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    min_distance: float = 1.0  # nearest doc distance (1 = no match)
    is_off_topic: bool = False
    off_topic_reason: str = ""
    total_docs: int = 0


# =============================================================================
# SAFETY
# =============================================================================

@dataclass
class SafetyResult:
    """Result of safety check on a query."""
    safe: bool = True
    flagged: bool = False
    category: str = ""  # "harmful", "off_topic", "safe"
    reason: str = ""
    rebuke_message: str = ""


# =============================================================================
# CLASSIFICATION
# =============================================================================

@dataclass
class ClassificationResult:
    """Output from the classifier stage."""
    query: str
    complexity: str  # close | moderate | distant
    task_label: str  # grounded | web_search | deep_reasoning | off_topic
    confidence: float  # 0-1
    method: str  # "sot_distance" | "heuristic" | "hybrid"
    source_distance: float = 1.0  # distance from source of truth
    metadata: dict = field(default_factory=dict)


# =============================================================================
# ROUTING
# =============================================================================

@dataclass
class RoutingDecision:
    """Output from the router."""
    query: str
    tier: str  # grounded | web_search | deep_reasoning
    model_id: str
    model_name: str
    complexity: str
    confidence: float
    reason: str
    cost_estimate_tokens: int = 0
    source_citations: list[str] = field(default_factory=list)
    benchmark_scores: Optional[dict] = None


# =============================================================================
# GENERATION
# =============================================================================

@dataclass
class GenerationResult:
    """Output from model generation (possibly after cascade)."""
    query: str
    response: str
    model_id: str
    tier: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cascade_escalated: bool = False
    cascade_from_tier: Optional[str] = None
    cascade_to_tier: Optional[str] = None
    web_search_used: bool = False
    web_search_results: list[dict] = field(default_factory=list)
    deep_reasoning_used: bool = False
    reasoning_chain: list[str] = field(default_factory=list)
    source_docs_used: list[str] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def cost(self) -> float:
        return 0.0


# =============================================================================
# TASK DECOMPOSITION
# =============================================================================

@dataclass
class SubTask:
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    complexity: str = Complexity.MODERATE
    task_type: str = "general"
    result: Optional[str] = None
    routing: Optional[RoutingDecision] = None
    generation: Optional[GenerationResult] = None


@dataclass
class DecompositionPlan:
    original_query: str
    sub_tasks: list[SubTask] = field(default_factory=list)
    strategy: str = "sequential"
    reasoning: str = ""


# =============================================================================
# REQUEST / RESPONSE
# =============================================================================

@dataclass
class RouteRequest:
    query: str
    force_tier: Optional[str] = None
    cascade: Optional[bool] = None
    decompose: Optional[bool] = None


@dataclass
class RouteResponse:
    query: str
    response: str
    classification: ClassificationResult
    routing: RoutingDecision
    generation: GenerationResult
    safety: Optional[SafetyResult] = None
    source_query: Optional[SourceQueryResult] = None
    decomposition: Optional[DecompositionPlan] = None
    rebuked: bool = False
    dashboard_url: str = ""
