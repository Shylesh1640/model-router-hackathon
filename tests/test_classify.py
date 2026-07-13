"""Tests for DistanceClassifier."""

from model_router.classify import DistanceClassifier
from model_router.models import SourceQueryResult


def test_classify_close():
    classifier = DistanceClassifier()
    source = SourceQueryResult(
        query="hello", min_distance=0.1,
        total_docs=5,
    )
    result = classifier.classify("hello", source)
    assert result.complexity == "close"
    assert result.task_label == "grounded"
    assert result.confidence > 0.8


def test_classify_moderate():
    classifier = DistanceClassifier()
    source = SourceQueryResult(
        query="explain AI", min_distance=0.4,
        total_docs=5,
    )
    result = classifier.classify("explain AI", source)
    assert result.complexity == "moderate"
    assert result.task_label == "web_search"


def test_classify_distant():
    classifier = DistanceClassifier()
    source = SourceQueryResult(
        query="quantum gravity", min_distance=0.8,
        total_docs=5,
    )
    result = classifier.classify("quantum gravity", source)
    assert result.complexity == "distant"
    assert result.task_label == "deep_reasoning"


def test_classify_boundary_close():
    classifier = DistanceClassifier()
    # Strict < threshold means 0.30 maps to moderate
    source = SourceQueryResult(
        query="test", min_distance=0.30,  # boundary
        total_docs=5,
    )
    result = classifier.classify("test", source)
    assert result.complexity == "moderate"
    # Just under boundary → close
    close_source = SourceQueryResult(
        query="test", min_distance=0.29,
        total_docs=5,
    )
    close_result = classifier.classify("test", close_source)
    assert close_result.complexity == "close"


def test_classify_boundary_moderate():
    classifier = DistanceClassifier()
    # Strict < threshold means 0.60 maps to distant
    source = SourceQueryResult(
        query="test", min_distance=0.60,  # boundary
        total_docs=5,
    )
    result = classifier.classify("test", source)
    assert result.complexity == "distant"
    # Just under boundary → moderate
    moderate_source = SourceQueryResult(
        query="test", min_distance=0.59,
        total_docs=5,
    )
    moderate_result = classifier.classify("test", moderate_source)
    assert moderate_result.complexity == "moderate"


def test_classify_empty_sot():
    classifier = DistanceClassifier()
    source = SourceQueryResult(
        query="anything", min_distance=1.0,
        total_docs=0,
    )
    result = classifier.classify("anything", source)
    assert result.complexity == "distant"
    assert result.task_label == "deep_reasoning"
    assert result.confidence == 0.5  # max(0.5, 0.0)
