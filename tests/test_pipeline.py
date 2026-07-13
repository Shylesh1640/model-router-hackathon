"""Tests for RoutingPipeline (no API key — tests only local logic)."""

from model_router.pipeline import RoutingPipeline
from model_router.models import RouteRequest
from model_router.config import RouterConfig


def _make_config():
    return RouterConfig(
        openrouter_api_key="test-key",
        cascade_enabled=False,
    )


def test_pipeline_classifies_without_api():
    """Pipeline runs classification and intent/decomposition without API calls."""
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("Python is a programming language", source="test")
    pipe.sot.add_document("Paris is the capital of France", source="test")

    req = RouteRequest(query="What is the capital of France?")
    result = pipe.route(req)

    # Classification should work
    assert result.classification.complexity in ("close", "moderate", "distant")
    assert result.classification.task_label in ("grounded", "web_search", "deep_reasoning")
    assert result.classification.confidence > 0

    # Routing decision should be set
    assert result.routing.tier in ("fast", "thinking", "deep")
    assert result.routing.model_name
    assert result.routing.model_id

    # Intent should be detected
    assert result.intent is not None
    assert result.intent.intent == "question"
    assert result.intent.confidence > 0

    # Decomposition should be present (simple query, no sub-tasks)
    assert result.decomposition is not None
    assert result.decomposition.has_sub_tasks is False
    assert result.decomposition.needs_reasoning is False


def test_pipeline_decomposition_flags():
    """Pipeline marks reasoning/vision flags from decomposition."""
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    # Multi-part query that requires reasoning
    req = RouteRequest(
        query="Write a Python function and then explain how it works"
    )
    result = pipe.route(req)
    assert result.decomposition is not None
    assert result.decomposition.needs_reasoning is True
    assert result.routing.tier in ("thinking", "deep")


def test_pipeline_force_tier():
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    req = RouteRequest(query="hello", force_tier="deep")
    result = pipe.route(req)
    assert result.routing.tier == "deep"


def test_pipeline_history():
    config = _make_config()
    pipe = RoutingPipeline(config)
    pipe.sot.add_document("test content", source="test")

    pipe.route(RouteRequest(query="hello"))
    pipe.route(RouteRequest(query="world"))

    stats = pipe.get_stats()
    assert stats["total_routes"] == 2

    history = pipe.get_history(limit=10)
    assert len(history) == 2


def test_pipeline_empty_sot():
    config = _make_config()
    pipe = RoutingPipeline(config)
    # No documents seeded

    req = RouteRequest(query="something completely new")
    result = pipe.route(req)

    # Should classify as distant (no source matches)
    assert result.classification.complexity == "distant"
    assert result.source_query.min_distance >= 0.9
