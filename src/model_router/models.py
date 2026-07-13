"""Data models — shared across the routing pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class Complexity:
    """Query complexity levels derived from distance measurement."""

    CLOSE = "close"  # distance < 0.3 → answer from source, fast model
    MODERATE = "moderate"  # distance 0.3-0.6 → web search, thinking model
    DISTANT = "distant"  # distance > 0.6 → deep reasoning, deep model

    CHOICES = [CLOSE, MODERATE, DISTANT]


class IntentCategory:
    """Recognised query intent categories."""

    QUESTION = "question"
    CODE_GENERATION = "code_generation"
    EXPLANATION = "explanation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    COMMAND = "command"
    GENERAL = "general"

    CHOICES = [
        QUESTION, CODE_GENERATION, EXPLANATION, ANALYSIS,
        CREATIVE, SUMMARIZATION, COMMAND, GENERAL,
    ]


# =============================================================================
# BENCHMARK PROFILES
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
        scores = [
            s
            for s in [self.mmlu_pro, self.humaneval, self.swe_bench_verified]
            if s is not None
        ]
        return sum(scores) / len(scores) if scores else None

    @property
    def coding_score(self) -> Optional[float]:
        scores = [
            s
            for s in [self.humaneval, self.swe_bench_verified, self.livecodebench]
            if s is not None
        ]
        return sum(scores) / len(scores) if scores else None

    @property
    def reasoning_score(self) -> Optional[float]:
        scores = [s for s in [self.mmlu_pro, self.simpleqa] if s is not None]
        return sum(scores) / len(scores) if scores else None


# =============================================================================
# SOURCE OF TRUTH
# =============================================================================


@dataclass
class SourceDocument:
    """A single document in the source of truth."""

    id: str
    content: str
    embedding: Optional[list] = None  # list[str] (Dice words) or list[float] (MiniLM vec)
    metadata: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class SourceQueryResult:
    """Result of querying the source of truth."""

    query: str
    query_embedding: Optional[list[float]] = None
    matches: list[SourceDocument] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    min_distance: float = 1.0
    total_docs: int = 0


# =============================================================================
# INTENT DETECTION
# =============================================================================


@dataclass
class IntentResult:
    """Output from the intent detector."""

    query: str
    intent: str  # one of IntentCategory.*
    confidence: float  # 0-1
    method: str  # "pattern" | "llm" | "unknown"
    entities: dict = field(default_factory=dict)


# =============================================================================
# DECOMPOSITION
# =============================================================================


@dataclass
class SubTask:
    """A single sub-task extracted from a decomposed query."""

    id: str
    description: str
    intent: str = IntentCategory.GENERAL
    needs_reasoning: bool = False
    needs_vision: bool = False


@dataclass
class DecompositionResult:
    """Output from the decomposition analyzer."""

    query: str
    sub_tasks: list[SubTask] = field(default_factory=list)
    has_sub_tasks: bool = False
    has_vision_content: bool = False
    needs_reasoning: bool = False  # sub-tasks found → bump tier
    needs_vision: bool = False  # vision content → flag for multimodal
    reason: str = ""
    method: str = "heuristic"  # "heuristic" | "llm"


# =============================================================================
# CLASSIFICATION
# =============================================================================


@dataclass
class ClassificationResult:
    """Output from the classifier — determines routing tier."""

    query: str
    complexity: str  # close | moderate | distant
    task_label: str  # grounded | web_search | deep_reasoning
    confidence: float  # 0-1
    method: str  # "sot_distance" | "heuristic" | "hybrid"
    source_distance: float = 1.0
    metadata: dict = field(default_factory=dict)


# =============================================================================
# ROUTING
# =============================================================================


@dataclass
class RoutingDecision:
    """Output from the router — which model was selected and why."""

    query: str
    tier: str  # fast | thinking | deep
    model_id: str
    model_name: str
    complexity: str
    confidence: float
    reason: str
    cost_estimate_tokens: int = 0
    source_citations: list[str] = field(default_factory=list)
    benchmark_scores: Optional[dict] = None
    needs_vision: bool = False  # escalated for vision-capable model


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
    cascade_from_tier: Optional[str] = None
    cascade_to_tier: Optional[str] = None
    cascade_escalated: bool = False
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

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


# =============================================================================
# REQUEST / RESPONSE
# =============================================================================


@dataclass
class RouteRequest:
    """Request to route a query through the pipeline."""

    query: str
    force_tier: Optional[str] = None
    cascade: Optional[bool] = None

    # Internal cascade tracking (set by pipeline, not user-facing)
    _cascade_hops: int = field(default=0, repr=False)
    _cascade_budget: int = field(default=0, repr=False)


@dataclass
class RouteResponse:
    """Complete routing result returned by the pipeline."""

    query: str
    response: str
    classification: ClassificationResult
    routing: RoutingDecision
    generation: GenerationResult
    intent: Optional[IntentResult] = None
    decomposition: Optional[DecompositionResult] = None
    source_query: Optional[SourceQueryResult] = None
    dashboard_url: str = ""
