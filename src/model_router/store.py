"""Source of Truth — in-memory content-word overlap store.

Zero external dependencies. Uses Dice coefficient on stopword-filtered
content words for topic-aware distance measurement.
"""

import logging
import uuid
from typing import Optional

from .models import SourceDocument, SourceQueryResult

logger = logging.getLogger(__name__)


class _TinyEmbed:
    """Minimal embedding — Dice coefficient on content words.

    No external dependencies. Topic separation via stopword-filtered
    bag-of-words intersection.
    """

    _STOP = frozenset(
        "the a an is are was were be been in on at to for of with by and or it its "
        "this that what who how why when where do does did has have had not no i you "
        "he she we they me him her my your our their about into also than then can "
        "just like very from will would could should may might must out up off down "
        "over under again further once here there all each every both few more most "
        "some any no nor own same so such only other new now".split()
    )

    @staticmethod
    def _content_words(text: str) -> frozenset:
        raw = text.lower().strip()
        if "a:" in raw:
            raw = raw[:raw.index("a:")].strip()
        if raw.startswith("q:"):
            raw = raw[2:].strip()
        words = raw.replace("-", " ").replace("'", "").split()
        return frozenset(w for w in words if w not in _TinyEmbed._STOP and len(w) > 1)

    @staticmethod
    def encode(text: str) -> list[str]:
        return list(_TinyEmbed._content_words(text))

    @staticmethod
    def word_overlap(a: list[str], b: list[str]) -> float:
        """Dice coefficient on content word sets.

        Handles asymmetric sizes (short query vs long doc) better than cosine.
        """
        set_a, set_b = frozenset(a), frozenset(b)
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        return 2.0 * inter / (len(set_a) + len(set_b)) if inter else 0.0


class SourceOfTruth:
    """In-memory store of reference documents for distance-based classification.

    Documents are stored with Dice-coefficient content-word embeddings.
    Query returns nearest documents sorted by distance (0 = identical topic).

    Usage:
        sot = SourceOfTruth()
        sot.add_document("Paris is the capital of France")
        result = sot.query("What is the capital of France?")
        print(result.min_distance)  # 0.0 (identical topic)
    """

    CLOSE_THRESHOLD = 0.30
    MODERATE_THRESHOLD = 0.60

    def __init__(self):
        self._docs: list[SourceDocument] = []
        self._embedder = _TinyEmbed()

    # ------------------------------------------------------------------
    # DOCUMENT MANAGEMENT
    # ------------------------------------------------------------------

    def add_document(
        self,
        content: str,
        source: str = "manual",
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        doc_id = doc_id or str(uuid.uuid4())
        words = self._embedder.encode(content)
        doc = SourceDocument(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            source=source,
        )
        doc.embedding = words
        self._docs.append(doc)
        return doc_id

    def add_documents(self, documents: list[dict]) -> list[str]:
        ids = []
        for doc in documents:
            did = self.add_document(
                content=doc["content"],
                source=doc.get("source", "manual"),
                metadata=doc.get("metadata"),
                doc_id=doc.get("id"),
            )
            ids.append(did)
        return ids

    def count(self) -> int:
        return len(self._docs)

    def clear(self):
        self._docs.clear()

    def remove(self, doc_id: str) -> bool:
        """Remove a document by ID. Returns True if found and removed."""
        for i, doc in enumerate(self._docs):
            if doc.id == doc_id:
                self._docs.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def query(self, query: str, top_k: int = 5) -> SourceQueryResult:
        """Query the source of truth. Returns nearest documents and distances."""
        q_words = self._embedder.encode(query)

        result = SourceQueryResult(
            query=query,
            total_docs=len(self._docs),
        )

        if not self._docs:
            result.min_distance = 1.0
            return result

        scored = []
        for doc in self._docs:
            if doc.embedding:
                sim = self._embedder.word_overlap(q_words, doc.embedding)
                dist = 1.0 - sim
                scored.append((dist, doc))

        scored.sort(key=lambda x: x[0])

        for dist, doc in scored[:top_k]:
            result.matches.append(doc)
            result.distances.append(dist)

        result.min_distance = scored[0][0] if scored else 1.0

        return result

    def complexity_from_distance(self, distance: float) -> str:
        """Map distance to complexity level."""
        if distance < self.CLOSE_THRESHOLD:
            return "close"
        elif distance < self.MODERATE_THRESHOLD:
            return "moderate"
        else:
            return "distant"
