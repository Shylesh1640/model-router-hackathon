"""Tests for IntentDetector."""

from model_router.intent import IntentDetector, IntentCategory
from model_router.models import IntentResult


def test_detect_question():
    detector = IntentDetector()
    result = detector.detect("What is the capital of France?")
    assert result.intent == IntentCategory.QUESTION
    assert result.confidence >= 0.3


def test_detect_code_generation():
    detector = IntentDetector()
    result = detector.detect("Write a Python function to sort a list")
    assert result.intent == IntentCategory.CODE_GENERATION


def test_detect_explanation():
    detector = IntentDetector()
    result = detector.detect("Explain how neural networks work")
    assert result.intent == IntentCategory.EXPLANATION


def test_detect_analysis():
    detector = IntentDetector()
    result = detector.detect("Compare Python and JavaScript for web development")
    assert result.intent == IntentCategory.ANALYSIS


def test_detect_creative():
    detector = IntentDetector()
    result = detector.detect("Write a short story about a robot learning to paint")
    assert result.intent == IntentCategory.CREATIVE


def test_detect_summarization():
    detector = IntentDetector()
    result = detector.detect("Summarize the key points from this article")
    assert result.intent == IntentCategory.SUMMARIZATION


def test_detect_command():
    detector = IntentDetector()
    result = detector.detect("Deploy the application to production")
    assert result.intent == IntentCategory.COMMAND


def test_detect_general_fallback():
    detector = IntentDetector()
    result = detector.detect("Hello world")
    assert result.intent == IntentCategory.GENERAL


def test_detect_empty():
    detector = IntentDetector()
    result = detector.detect("")
    assert result.intent == IntentCategory.GENERAL
    assert result.confidence == 0.0


def test_needs_reasoning():
    assert IntentDetector.needs_reasoning(IntentCategory.ANALYSIS) is True
    assert IntentDetector.needs_reasoning(IntentCategory.CODE_GENERATION) is True
    assert IntentDetector.needs_reasoning(IntentCategory.QUESTION) is False
    assert IntentDetector.needs_reasoning(IntentCategory.GENERAL) is False


def test_suggest_tier():
    assert IntentDetector.suggest_tier(IntentCategory.CODE_GENERATION) == "thinking"
    assert IntentDetector.suggest_tier(IntentCategory.EXPLANATION) == "thinking"
    assert IntentDetector.suggest_tier(IntentCategory.QUESTION) is None
