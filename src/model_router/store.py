"""Source of Truth — in-memory semantic embedding store.

Auto-selects embedder: MiniLM (sentence-transformers) > Dice (zero-dep fallback).
MiniLM gives 384-dim dense vectors with cosine similarity for accurate semantic
matching. Dice provides content-word overlap as a zero-dep fallback.
"""

import logging
import math
import re
import uuid
from typing import Optional, Union

from .models import SourceDocument, SourceQueryResult

logger = logging.getLogger(__name__)

# ─── Stopword list (used by Dice and for heatmap word extraction) ──────────

_STOP = frozenset(
    "the a an is are was were be been in on at to for of with by and or it its "
    "this that what who how why when where do does did has have had not no i you "
    "he she we they me him her my your our their about into also than then can "
    "just like very from will would could should may might must out up off down "
    "over under again further once here there all each every both few more most "
    "some any no nor own same so such only other new now".split()
)


def extract_content_words(text: str) -> list[str]:
    """Extract stopword-filtered content words from text.

    Used for heatmap visualization regardless of which embedder is active.
    """
    raw = text.lower().strip()
    raw = re.sub(r"[^\w\s]", "", raw)
    raw = raw.replace("-", " ").replace("'", "")
    words = raw.split()
    return [w for w in words if w not in _STOP and len(w) > 1]


# ─── Dice Embedder (zero-dep fallback) ────────────────────────────────────

class _TinyEmbed:
    """Minimal embedding — Dice coefficient on content words."""

    name = "dice"

    @staticmethod
    def encode(text: str) -> list[str]:
        return extract_content_words(text)

    @staticmethod
    def similarity(a: Union[list, list[str]], b: Union[list, list[str]]) -> float:
        set_a, set_b = frozenset(a), frozenset(b)
        if not set_a or not set_b:
            return 0.0
        inter = len(set_a & set_b)
        return 2.0 * inter / (len(set_a) + len(set_b)) if inter else 0.0


# ─── MiniLM Embedder ─────────────────────────────────────────────────────

class _MiniLMEmbed:
    """384-dim sentence embeddings via sentence-transformers."""

    name = "minilm"

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, text: str) -> list[float]:
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two normalized 384-dim vectors."""
        dot = sum(av * bv for av, bv in zip(a, b))
        return max(-1.0, min(1.0, dot))


# ─── Embedder Selection ──────────────────────────────────────────────────

def get_embedder(name: Optional[str] = None):
    """Return the best available embedder.

    Priority: MiniLM > Dice. Pass ``name="dice"`` to force Dice.
    """
    if name == "dice":
        return _TinyEmbed()
    try:
        import sentence_transformers  # noqa
        return _MiniLMEmbed()
    except ImportError:
        return _TinyEmbed()


# ═══════════════════════════════════════════════════════════════════════════
# SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════════════

class SourceOfTruth:
    """In-memory store of reference documents for distance-based classification.

    Uses MiniLM sentence embeddings (384-dim, cosine similarity) when available,
    falling back to Dice coefficient on content words.

    Documents store both the embedding vector and content-word list so that
    the heatmap visualizer can show word-level overlap regardless of embedder.

    Usage:
        sot = SourceOfTruth()
        sot.add_document("Paris is the capital of France")
        result = sot.query("What is the capital of France?")
        print(result.min_distance)  # ~0.05 (semantically identical)
    """

    # MiniLM thresholds (cosine distance 0-1)
    CLOSE_THRESHOLD = 0.20
    MODERATE_THRESHOLD = 0.35

    def __init__(self, embedder: Optional[str] = None):
        self._docs: list[SourceDocument] = []
        self._embedder = get_embedder(embedder)

    # ── Document Management ───────────────────────────────────────────

    def add_document(
        self,
        content: str,
        source: str = "manual",
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        doc_id = doc_id or str(uuid.uuid4())
        emb = self._embedder.encode(content)
        words = extract_content_words(content)
        meta = dict(metadata or {})
        meta["content_words"] = words
        doc = SourceDocument(
            id=doc_id,
            content=content,
            embedding=emb if isinstance(emb, list) else list(emb),
            metadata=meta,
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

    def remove(self, doc_id: str) -> bool:
        for i, doc in enumerate(self._docs):
            if doc.id == doc_id:
                self._docs.pop(i)
                return True
        return False

    def upgrade_embeddings(self):
        """Re-embed all documents with the current (best) embedder.

        Call after installing sentence-transformers to upgrade from Dice.
        """
        current = get_embedder()
        if current.name == self._embedder.name:
            return 0  # already using best embedder
        count = 0
        for doc in self._docs:
            new_emb = current.encode(doc.content)
            doc.embedding = new_emb if isinstance(new_emb, list) else list(new_emb)
            count += 1
        self._embedder = current
        logger.info("Upgraded %d docs to %s embedder", count, current.name)
        return count

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    # ── Query ─────────────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 5) -> SourceQueryResult:
        """Query the SOT. Returns nearest documents and distances (0=identical)."""
        q_emb = self._embedder.encode(query)
        q_emb = q_emb if isinstance(q_emb, list) else list(q_emb)

        result = SourceQueryResult(
            query=query,
            total_docs=len(self._docs),
        )

        if not self._docs:
            result.min_distance = 1.0
            return result

        scored = []
        for doc in self._docs:
            d_emb = doc.embedding
            if d_emb:
                sim = self._embedder.similarity(q_emb, d_emb)
                # Clamp similarity to [0, 1] for MiniLM (it can return negative)
                sim = max(0.0, sim)
                dist = 1.0 - sim
                scored.append((dist, doc))

        scored.sort(key=lambda x: x[0])

        for dist, doc in scored[:top_k]:
            result.matches.append(doc)
            result.distances.append(dist)

        result.min_distance = scored[0][0] if scored else 1.0
        return result

    def complexity_from_distance(self, distance: float) -> str:
        if distance < self.CLOSE_THRESHOLD:
            return "close"
        elif distance < self.MODERATE_THRESHOLD:
            return "moderate"
        return "distant"
