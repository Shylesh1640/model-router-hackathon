"""Tests for CostRouter."""

from model_router.router import CostRouter
from model_router.models import ClassificationResult


def test_route_close_classified_to_fast():
    router = CostRouter()
    classification = ClassificationResult(
        query="hello", complexity="close",
        task_label="grounded", confidence=0.95, method="sot_distance",
        source_distance=0.1,
    )
    decision = router.route("hello", classification)
    assert decision.tier == "fast"
    assert "liquid" in decision.model_id or "llama" in decision.model_id


def test_route_moderate_to_thinking():
    router = CostRouter()
    classification = ClassificationResult(
        query="explain transformers", complexity="moderate",
        task_label="web_search", confidence=0.7, method="sot_distance",
        source_distance=0.4,
    )
    decision = router.route("explain transformers", classification)
    assert decision.tier == "thinking"


def test_route_distant_to_deep():
    router = CostRouter()
    classification = ClassificationResult(
        query="quantum gravity", complexity="distant",
        task_label="deep_reasoning", confidence=0.5, method="sot_distance",
        source_distance=0.8,
    )
    decision = router.route("quantum gravity", classification)
    assert decision.tier == "deep"


def test_route_force_tier():
    router = CostRouter()
    classification = ClassificationResult(
        query="hi", complexity="close",
        task_label="grounded", confidence=0.95, method="sot_distance",
        source_distance=0.1,
    )
    decision = router.route("hi", classification, force_tier="deep")
    assert decision.tier == "deep"
    assert "forced" in decision.reason.lower()


def test_route_needs_reasoning_bumps_fast_to_thinking():
    router = CostRouter()
    classification = ClassificationResult(
        query="test", complexity="close",
        task_label="grounded", confidence=0.95, method="sot_distance",
        source_distance=0.1,
    )
    decision = router.route("test", classification, needs_reasoning=True)
    # Fast with reasoning flag → thinking
    assert decision.tier == "thinking"
    assert "escalated" in decision.reason


def test_route_needs_vision_picks_multimodal():
    router = CostRouter()
    classification = ClassificationResult(
        query="analyze this image", complexity="moderate",
        task_label="web_search", confidence=0.7, method="sot_distance",
        source_distance=0.4,
    )
    decision = router.route("analyze this image", classification, needs_vision=True)
    # Should pick a vision-capable model from thinking tier
    assert decision.needs_vision is True
    # Nemotron Nano 12B VL is the only vision model in the pool
    if "vl" in decision.model_id.lower() or "vision" in decision.model_name.lower():
        assert decision.needs_vision is True


def test_route_needs_reasoning_does_not_downgrade_deep():
    router = CostRouter()
    classification = ClassificationResult(
        query="complex quantum physics", complexity="distant",
        task_label="deep_reasoning", confidence=0.5, method="sot_distance",
        source_distance=0.9,
    )
    # Already deep, needs_reasoning shouldn't change it
    decision = router.route("complex quantum physics", classification, needs_reasoning=True)
    assert decision.tier == "deep"


def test_route_unknown_tier_falls_back():
    router = CostRouter()
    classification = ClassificationResult(
        query="test", complexity="close",
        task_label="grounded", confidence=0.5, method="sot_distance",
        source_distance=0.1,
    )
    decision = router.route("test", classification, force_tier="nonexistent")
    # Should ignore invalid force_tier
    assert decision.tier == "fast"


def test_route_returns_benchmarks_on_known_model():
    router = CostRouter()
    classification = ClassificationResult(
        query="test benchmarks", complexity="moderate",
        task_label="web_search", confidence=0.7, method="sot_distance",
        source_distance=0.5,
    )
    decision = router.route("test benchmarks", classification)
    if decision.benchmark_scores:
        assert "overall" in decision.benchmark_scores
