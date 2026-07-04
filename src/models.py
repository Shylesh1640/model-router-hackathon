"""Shared data models for the routing pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class Complexity:
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"

    CHOICES = [SIMPLE, MEDIUM, HARD]


@dataclass
class ClassificationResult:
    """Output from the classifier stage."""
    query: str
    complexity: str  # simple | medium | hard
    task_label: str  # conversation | code | reasoning | etc
    confidence: float  # 0-1
    method: str  # "embedding" | "heuristic" | "hybrid" | "fallback"
    metadata: dict = field(default_factory=dict)


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


@dataclass
class RouteRequest:
    """Incoming API request."""
    query: str
    force_tier: Optional[str] = None  # override routing
    cascade: Optional[bool] = None    # override cascade setting


@dataclass
class RouteResponse:
    """Full response from the routing API."""
    query: str
    response: str
    classification: ClassificationResult
    routing: RoutingDecision
    generation: GenerationResult
    dashboard_url: str = ""
