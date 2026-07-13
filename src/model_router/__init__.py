"""Model Router — cost-optimized LLM routing library.

Routes queries across model tiers based on query complexity.
Import the classes you need:

    from model_router import RoutingPipeline, CostRouter, OpenRouterClient
    from model_router import SourceOfTruth, WebSearcher
    from model_router.constants import ALL_MODELS, TIER_MODELS
"""

from ._version import __version__

from .config import RouterConfig, get_config
from .models import (
    Complexity,
    RouteRequest,
    RouteResponse,
    ClassificationResult,
    RoutingDecision,
    GenerationResult,
    SourceQueryResult,
    SourceDocument,
    BenchmarkProfile,
    IntentResult,
    IntentCategory,
    DecompositionResult,
    SubTask,
)
from .router import CostRouter
from .client import OpenRouterClient, CircuitBreaker
from .store import SourceOfTruth
from .classify import DistanceClassifier
from .search import WebSearcher
from .intent import IntentDetector
from .decompose import DecompositionAnalyzer
from .telemetry import Telemetry, get_telemetry
from .pipeline import RoutingPipeline

__all__ = [
    "__version__",
    "RouterConfig",
    "get_config",
    "Complexity",
    "RouteRequest",
    "RouteResponse",
    "ClassificationResult",
    "RoutingDecision",
    "GenerationResult",
    "SourceQueryResult",
    "SourceDocument",
    "BenchmarkProfile",
    "IntentResult",
    "IntentCategory",
    "DecompositionResult",
    "SubTask",
    "CostRouter",
    "OpenRouterClient",
    "CircuitBreaker",
    "SourceOfTruth",
    "DistanceClassifier",
    "WebSearcher",
    "IntentDetector",
    "DecompositionAnalyzer",
    "Telemetry",
    "get_telemetry",
    "RoutingPipeline",
]
