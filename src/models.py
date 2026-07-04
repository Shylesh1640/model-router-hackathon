"""Shared data models for the routing pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union


class Complexity:
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"

    CHOICES = [SIMPLE, MEDIUM, HARD]


# =============================================================================
# BENCHMARK PROFILES
# =============================================================================

@dataclass
class BenchmarkProfile:
    """Benchmark scores used as routing support metrics.

    Scores are 0-100 (percentage) unless noted.
    None = unknown, system will look up on provider scrape.
    """
    swe_bench_verified: Optional[float] = None   # SWE-bench Verified (%)
    mmlu_pro: Optional[float] = None              # MMLU-Pro (%)
    humaneval: Optional[float] = None             # HumanEval pass@1 (%)
    livecodebench: Optional[float] = None         # LiveCodeBench (%)
    simpleqa: Optional[float] = None              # SimpleQA accuracy (%)
    source: str = "manual"                        # "manual" | "scraped" | "inferred"

    @property
    def coding_score(self) -> Optional[float]:
        """Aggregate coding benchmark score."""
        scores = [s for s in [self.humaneval, self.livecodebench, self.swe_bench_verified] if s is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def reasoning_score(self) -> Optional[float]:
        """Aggregate reasoning benchmark score."""
        scores = [s for s in [self.mmlu_pro] if s is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def overall_score(self) -> Optional[float]:
        """Weighted overall capability score."""
        w_coding = 0.4
        w_reasoning = 0.4
        w_factual = 0.2
        components = []
        weights = []
        if self.coding_score is not None:
            components.append(self.coding_score)
            weights.append(w_coding)
        if self.reasoning_score is not None:
            components.append(self.reasoning_score)
            weights.append(w_reasoning)
        if self.simpleqa is not None:
            components.append(self.simpleqa)
            weights.append(w_factual)
        if not components:
            return None
        return sum(c * w for c, w in zip(components, weights)) / sum(weights)


# =============================================================================
# CLASSIFICATION
# =============================================================================

@dataclass
class ClassificationResult:
    """Output from the classifier stage."""
    query: str
    complexity: str  # simple | medium | hard
    task_label: str  # conversation | code | reasoning | etc
    confidence: float  # 0-1
    method: str  # "embedding" | "heuristic" | "hybrid" | "fallback"
    metadata: dict = field(default_factory=dict)


# =============================================================================
# ROUTING
# =============================================================================

@dataclass
class RoutingDecision:
    """Output from the router — which model to use and why."""
    query: str
    tier: str  # fast | thinking | deep
    model_id: str
    model_name: str
    complexity: str
    confidence: float
    reason: str
    cost_estimate_tokens: int = 0
    override_applied: bool = False
    benchmark_scores: Optional[dict] = None  # benchmark data used in decision


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
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def cost(self) -> float:
        """Estimated cost — 0 since all models are free."""
        return 0.0


# =============================================================================
# TASK DECOMPOSITION
# =============================================================================

@dataclass
class SubTask:
    """A single sub-task within a decomposed query."""
    id: str  # e.g. "subtask-1"
    description: str  # what this sub-task does
    depends_on: list[str] = field(default_factory=list)  # sub-task IDs that must complete first
    complexity: str = Complexity.MEDIUM
    task_type: str = "general"  # code | search | reasoning | write | etc
    result: Optional[str] = None
    routing: Optional[RoutingDecision] = None
    generation: Optional[GenerationResult] = None


@dataclass
class DecompositionPlan:
    """Plan for decomposing a complex query into sub-tasks."""
    original_query: str
    sub_tasks: list[SubTask] = field(default_factory=list)
    strategy: str = "sequential"  # sequential | parallel | dependent
    reasoning: str = ""


# =============================================================================
# REQUEST / RESPONSE
# =============================================================================

@dataclass
class RouteRequest:
    """Incoming API request."""
    query: str
    force_tier: Optional[str] = None  # override routing
    cascade: Optional[bool] = None    # override cascade setting
    decompose: Optional[bool] = None  # enable task decomposition


@dataclass
class RouteResponse:
    """Full response from the routing API."""
    query: str
    response: str
    classification: ClassificationResult
    routing: RoutingDecision
    generation: GenerationResult
    decomposition: Optional[DecompositionPlan] = None  # populated if decomposed
    dashboard_url: str = ""
