"""Multi-dimensional match signature — richer than a single distance number.

Instead of reducing query-doc overlap to one scalar, capture the full
dimensionality: which words matched, how many docs they hit, and whether
the matches are concentrated on one topic or spread thin.
"""

from dataclasses import dataclass
from typing import Optional

from .models import SourceQueryResult, ClassificationResult


@dataclass
class MatchSignature:
    """Multi-dimensional profile of a query against the SOT.

    Flattening to a single distance number (0.0–1.0) loses information.
    A query that hits 3 docs with 2 words each is different from one
    that hits 1 doc with 1 word — even if both have the same min_distance.
    """

    query_words: list[str]
    doc_word_sets: list[list[str]]
    distances: list[float]
    min_distance: float
    total_docs: int

    # ---- Computed metrics ------------------------------------------------

    @property
    def query_word_count(self) -> int:
        return len(self.query_words)

    @property
    def matched_query_words(self) -> int:
        """How many of the query's content words hit at least one doc."""
        if not self.query_words:
            return 0
        hit = 0
        qset = set(self.query_words)
        for qw in self.query_words:
            for dw in self.doc_word_sets:
                if qw in dw:
                    hit += 1
                    break
        return hit

    @property
    def match_density(self) -> float:
        """Fraction of query words that found any match. 0.0–1.0."""
        return self.matched_query_words / max(self.query_word_count, 1)

    @property
    def docs_hit(self) -> int:
        """How many SOT documents got at least one matching word."""
        if not self.query_words:
            return 0
        hit = 0
        qset = set(self.query_words)
        for dw in self.doc_word_sets:
            if qset & set(dw):
                hit += 1
        return hit

    @property
    def coverage(self) -> float:
        """Fraction of SOT docs that got hit. 0.0–1.0."""
        return self.docs_hit / max(self.total_docs, 1)

    @property
    def concentration(self) -> float:
        """Are matches focused on one doc (1.0) or spread across many (0.0)?

        High concentration = the query matches a specific topic doc.
        Low concentration = the query is very broad.
        """
        if self.docs_hit <= 1:
            return 1.0
        # Gini-style: count total matches per doc, see distribution
        qset = set(self.query_words)
        per_doc = [len(qset & set(dw)) for dw in self.doc_word_sets]
        total = sum(per_doc)
        if total == 0:
            return 0.0
        # Normalize: 1.0 = all matches in one doc, near 0 = evenly spread
        max_share = max(per_doc) / total
        return max_share


class HeatmapClassifier:
    """Classifies query complexity using multi-dimensional match signal.

    Instead of thresholding a single distance, consider:
    - match_density: how many query words found matches
    - coverage: how many docs got hit
    - concentration: focused topic or broad query
    - min_distance: classic fallback
    """

    def classify(self, query: str, source: SourceQueryResult) -> ClassificationResult:
        """Build a match signature and classify."""

        # Build the match vectors
        from .store import _TinyEmbed
        embedder = _TinyEmbed()
        q_words = embedder.encode(query)
        doc_word_sets = [
            list(doc.embedding) if doc.embedding else []
            for doc in source.matches
        ]

        sig = MatchSignature(
            query_words=q_words,
            doc_word_sets=doc_word_sets,
            distances=source.distances,
            min_distance=source.min_distance,
            total_docs=source.total_docs,
        )

        complexity, task_label, confidence = self._classify_sig(sig)

        return ClassificationResult(
            query=query,
            complexity=complexity,
            task_label=task_label,
            confidence=confidence,
            method="heatmap",
            source_distance=sig.min_distance,
            metadata={
                "match_density": round(sig.match_density, 3),
                "coverage": round(sig.coverage, 3),
                "concentration": round(sig.concentration, 3),
                "matched_words": sig.matched_query_words,
                "query_words": sig.query_word_count,
                "docs_hit": sig.docs_hit,
            },
        )

    def _classify_sig(self, sig: MatchSignature):
        """Multi-factor decision tree.

        Rules in priority order:
        1. Empty SOT or no matches at all → distant
        2. Most query words found matches in a single doc → close
        3. Some matches, spread across docs → moderate
        4. Only one query word matched anything → moderate
        5. Nothing matched → distant
        """
        # Empty SOT
        if sig.total_docs == 0:
            return "distant", "deep_reasoning", 0.5

        # No matches at all
        if sig.matched_query_words == 0:
            return "distant", "deep_reasoning", 0.5

        # Majority of query words hit matches → close (high confidence)
        if sig.match_density >= 0.5 and sig.min_distance < 0.70:
            conf = min(0.9, 0.5 + sig.match_density * 0.4)
            return "close", "grounded", round(conf, 2)

        # At least some match signal → moderate
        if sig.matched_query_words >= 1 and sig.min_distance < 0.85:
            conf = 0.5 + sig.match_density * 0.3
            return "moderate", "web_search", round(conf, 2)

        # Fallback to classic distance-based
        if sig.min_distance < 0.60:
            return "close", "grounded", 0.6
        elif sig.min_distance < 0.80:
            return "moderate", "web_search", 0.5

        return "distant", "deep_reasoning", 0.5
