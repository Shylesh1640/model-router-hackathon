"""Tests for DecompositionAnalyzer."""

from model_router.decompose import DecompositionAnalyzer


def test_simple_query_no_decomp():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("What is the capital of France?")
    assert result.has_sub_tasks is False
    assert result.needs_reasoning is False
    assert result.needs_vision is False
    assert len(result.sub_tasks) == 0


def test_multi_question_detected():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("What is Python? How do I install it?")
    assert result.has_sub_tasks is True
    assert result.needs_reasoning is True


def test_conjunction_detected():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("Write a function to sort data and then test it")
    assert result.has_sub_tasks is True
    assert result.needs_reasoning is True


def test_vision_content_detected():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("What does this diagram show?")
    assert result.has_vision_content is True
    assert result.needs_vision is True
    assert result.needs_reasoning is False  # no sub-tasks, just vision


def test_vision_and_subtasks():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("Analyze this screenshot and explain what it shows")
    assert result.has_vision_content is True
    assert result.needs_vision is True
    assert result.needs_reasoning is True  # has sub-tasks


def test_image_references():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("Extract text from this PNG file")
    assert result.has_vision_content is True
    assert result.needs_vision is True


def test_multi_verb_chain():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("Create a new project, write tests, deploy it, and monitor the logs")
    assert result.has_sub_tasks is True
    assert result.needs_reasoning is True
    assert len(result.sub_tasks) >= 1


def test_numbered_list():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("1. Install the package\n2. Configure it\n3. Run the server")
    assert result.has_sub_tasks is True
    assert len(result.sub_tasks) >= 1


def test_empty_query():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("")
    assert result.has_sub_tasks is False
    assert result.reason == "Empty query"


def test_no_false_positive_short():
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("Hello")
    assert result.has_sub_tasks is False
    assert result.needs_reasoning is False
