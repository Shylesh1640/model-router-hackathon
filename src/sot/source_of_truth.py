"""Source of Truth — zero-dependency in-memory vector store.

Uses numpy cosine similarity when available, falls back to
character-n-gram hashing when numpy isn't installed.

No external dependencies required.
"""

import hashlib
import json
import logging
import math
import uuid
from pathlib import Path
from typing import Optional

from src.models import SourceDocument, SourceQueryResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"


class _TinyEmbed:
    """Minimal embedding — n-gram hash vectors when no sentence-transformers.

    Not as accurate as real embeddings, but good enough for distance-based
    routing decisions in a demo/hackathon context.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._has_numpy = False
        try:
            import numpy as np
            self.np = np
            self._has_numpy = True
        except ImportError:
            pass

    def encode(self, text: str) -> list[float]:
        """Compute embedding vector for text."""
        if self._has_numpy:
            return self._encode_numpy(text)
        return self._encode_pure(text)

    def _encode_numpy(self, text: str) -> list[float]:
        """Character n-gram hashing with numpy."""
        vec = self.np.zeros(self.dim, dtype=self.np.float32)
        text = text.lower().strip()
        for n in range(1, 4):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
        # Normalize
        norm = float(self.np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _encode_pure(self, text: str) -> list[float]:
        """Pure Python fallback — character hashing without numpy."""
        vec = [0.0] * self.dim
        text = text.lower().strip()
        for n in range(1, 4):
            for i in range(len(text) - n + 1):
                gram = text[i:i + n]
                h = int(hashlib.md5(gram.encode()).hexdigest(), 16)
                idx = h % self.dim
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        if self._has_numpy:
            return float(self.np.dot(a, b))
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # both are normalized


class SourceOfTruth:
    """In-memory vector knowledge base — no external deps needed."""

    OFF_TOPIC_THRESHOLD = 0.75
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
        emb = self._embedder.encode(content)
        doc = SourceDocument(
            id=doc_id,
            content=content,
            embedding=emb,
            metadata=metadata or {},
            source=source,
        )
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

    # ------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------

    def query(self, query: str, top_k: int = 5) -> SourceQueryResult:
        """Query the source of truth. Returns nearest documents and distances."""
        q_emb = self._embedder.encode(query)

        result = SourceQueryResult(
            query=query,
            query_embedding=q_emb,
            total_docs=len(self._docs),
        )

        if not self._docs:
            result.min_distance = 1.0
            result.is_off_topic = True
            result.off_topic_reason = "Source of truth is empty"
            return result

        # Compute distances
        scored = []
        for doc in self._docs:
            if doc.embedding:
                sim = self._embedder.cosine_similarity(q_emb, doc.embedding)
                dist = 1.0 - sim  # cosine distance
                scored.append((dist, doc))

        scored.sort(key=lambda x: x[0])

        for i, (dist, doc) in enumerate(scored[:top_k]):
            result.matches.append(doc)
            result.distances.append(dist)

        result.min_distance = scored[0][0] if scored else 1.0

        # Off-topic check
        result.is_off_topic = result.min_distance > self.OFF_TOPIC_THRESHOLD
        if result.is_off_topic:
            result.off_topic_reason = (
                f"Distance {result.min_distance:.2f} from nearest source "
                f"(threshold: {self.OFF_TOPIC_THRESHOLD})"
            )

        return result

    def complexity_from_distance(self, distance: float) -> str:
        """Map distance to complexity level."""
        if distance < self.CLOSE_THRESHOLD:
            return "close"
        elif distance < self.MODERATE_THRESHOLD:
            return "moderate"
        else:
            return "distant"


_sot: Optional[SourceOfTruth] = None


def get_sot() -> SourceOfTruth:
    global _sot
    if _sot is None:
        _sot = SourceOfTruth()
    return _sot
