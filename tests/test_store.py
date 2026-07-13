"""Tests for SourceOfTruth."""

from model_router.store import SourceOfTruth


def test_empty_sot():
    sot = SourceOfTruth()
    result = sot.query("anything")
    assert result.min_distance == 1.0
    assert result.total_docs == 0
    assert len(result.matches) == 0


def test_add_and_query():
    sot = SourceOfTruth()
    doc_id = sot.add_document("Paris is the capital of France", source="test")
    assert doc_id is not None

    result = sot.query("capital of France")
    assert result.min_distance < 1.0
    assert len(result.matches) >= 1
    assert "Paris" in result.matches[0].content


def test_add_documents_batch():
    sot = SourceOfTruth()
    ids = sot.add_documents([
        {"content": "Python is a language", "source": "test"},
        {"content": "Java is also a language", "source": "test"},
    ])
    assert len(ids) == 2
    assert sot.count() == 2


def test_clear():
    sot = SourceOfTruth()
    sot.add_document("test content")
    assert sot.count() == 1
    sot.clear()
    assert sot.count() == 0


def test_remove():
    sot = SourceOfTruth()
    doc_id = sot.add_document("test content")
    assert sot.count() == 1
    assert sot.remove(doc_id) is True
    assert sot.count() == 0
    assert sot.remove("nonexistent") is False


def test_complexity_from_distance():
    sot = SourceOfTruth()
    assert sot.complexity_from_distance(0.1) == "close"  # MiniLM: < 0.20
    assert sot.complexity_from_distance(0.19) == "close"
    assert sot.complexity_from_distance(0.20) == "moderate"
    assert sot.complexity_from_distance(0.34) == "moderate"
    assert sot.complexity_from_distance(0.35) == "distant"


def test_dice_similarity_identical():
    sot = SourceOfTruth()
    sot.add_document("Python programming language", source="test")
    result = sot.query("Python programming language")
    # Identical content should give very low distance
    assert result.min_distance < 0.3


def test_dice_similarity_unrelated():
    sot = SourceOfTruth()
    sot.add_document("Paris France Eiffel Tower", source="test")
    result = sot.query("quantum physics particles")
    # Unrelated content → high distance
    assert result.min_distance > 0.5


def test_top_k():
    sot = SourceOfTruth()
    for i in range(10):
        sot.add_document(f"Document number {i}")
    result = sot.query("document", top_k=3)
    assert len(result.matches) <= 3
    assert len(result.distances) <= 3
