"""Query complexity estimation — embedding + heuristic hybrid."""

import logging
import re
from typing import Optional

from src.models import Complexity, ClassificationResult

logger = logging.getLogger(__name__)

# Heuristic complexity signals
_COMPLEXITY_SIGNALS = {
    Complexity.CLOSE: {
        "keywords": {"hello", "hi", "hey", "thanks", "bye", "yes", "no", "good"},
        "max_words": 5,
        "patterns": [
            r"^(what|who|when|where|is|are|can)\s+\w+\s*\??$",
            r"^(hello|hi|hey|thanks|goodbye)\b",
            r"^\w+\s*\??$",
        ],
    },
    Complexity.MODERATE: {
        "keywords": {
            "explain", "describe", "compare", "difference", "why",
            "how does", "how to", "what is", "meaning", "purpose",
            "analyze", "review", "summarize",
        },
        "max_words": 30,
        "patterns": [
            r"(explain|describe|compare|difference)\b.*\w+",
            r"(how|why)\s+(does|is|are|do|can)\b.{10,}",
            r"what\s+is\s+(a|an|the)\s+\w+",
        ],
    },
    Complexity.DISTANT: {
        "keywords": {
            "debug", "implement", "design", "create", "build",
            "optimize", "refactor", "architect", "generate",
            "write code", "fix bug", "error", "exception",
            "stack trace", "traceback", "multi-step", "plan",
            "indexerror", "keyerror", "valueerror", "typeerror",
            "traceback", "attributeerror", "syntaxerror",
        },
        "max_words": 999,
        "patterns": [
            r"(debug|fix|implement|design|create|build|generate)\b",
            r"(Traceback|Error|Exception|traceback):",
            r"```",
            r"(multi-step|complex|distributed|architecture)",
            r"(IndexError|KeyError|ValueError|TypeError|AttributeError)",
            r"out\s+of\s+range",
        ],
    },
}


class ComplexityClassifier:
    """Estimates query complexity using embedding + heuristics."""

    def __init__(self, method: str = "hybrid"):
        self.method = method
        self._embedder = None

    def _lazy_load_embedder(self):
        """Load sentence transformer on first use."""
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not available, using heuristics only")
            self.method = "heuristic"

    def estimate(self, query: str) -> ClassificationResult:
        """Classify query: complexity + task label + confidence."""
        if not query or not query.strip():
            return ClassificationResult(
                query=query, complexity=Complexity.SIMPLE, task_label="empty",
                confidence=1.0, method="fallback",
            )

        clean = query.strip()

        if self.method in ("embedding", "hybrid"):
            self._lazy_load_embedder()

        if self.method == "embedding" and self._embedder:
            return self._estimate_embedding(clean)
        elif self.method == "hybrid" and self._embedder:
            embedding_result = self._estimate_embedding(clean)
            heuristic_result = self._estimate_heuristic(clean)
            return self._merge_results(clean, embedding_result, heuristic_result)
        else:
            return self._estimate_heuristic(clean)

    def _estimate_embedding(self, query: str) -> ClassificationResult:
        """Use embedding norm + distance to classify."""
        try:
            emb = self._embedder.encode(query)
            norm = float(emb @ emb) ** 0.5

            # Short queries cluster near origin in embedding space
            if norm < 5.0:
                complexity = Complexity.CLOSE
                confidence = 0.7 + min(norm / 10.0, 0.2)
            elif norm < 12.0:
                complexity = Complexity.MODERATE
                confidence = 0.6 + min((norm - 5.0) / 20.0, 0.3)
            else:
                complexity = Complexity.DISTANT
                confidence = 0.6 + min(norm / 50.0, 0.3)

            task_label = self._infer_task(query, complexity)

            return ClassificationResult(
                query=query, complexity=complexity, task_label=task_label,
                confidence=min(confidence, 0.95), method="embedding",
                metadata={"embedding_norm": round(norm, 2)},
            )
        except Exception as e:
            logger.warning(f"Embedding classification failed: {e}")
            return self._estimate_heuristic(query)

    def _estimate_heuristic(self, query: str) -> ClassificationResult:
        """Use keyword + regex pattern matching."""
        q_lower = query.lower().strip()
        words = q_lower.split()
        word_count = len(words)

        for complexity in Complexity.CHOICES:
            signals = _COMPLEXITY_SIGNALS[complexity]

            # Check patterns
            for pattern in signals["patterns"]:
                if re.search(pattern, q_lower):
                    confidence = 0.7 if complexity == Complexity.DISTANT else 0.6
                    return ClassificationResult(
                        query=query, complexity=complexity,
                        task_label=self._infer_task(query, complexity),
                        confidence=confidence, method="heuristic",
                        metadata={"pattern": pattern},
                    )

            # Check word count threshold
            if word_count <= signals["max_words"]:
                # Check keyword overlap
                matching = sum(1 for kw in signals["keywords"] if kw in q_lower)
                if matching > 0:
                    confidence = min(0.5 + matching * 0.1, 0.85)
                    return ClassificationResult(
                        query=query, complexity=complexity,
                        task_label=self._infer_task(query, complexity),
                        confidence=confidence, method="heuristic",
                        metadata={"keyword_matches": matching},
                    )

        # Fallback: medium
        return ClassificationResult(
            query=query, complexity=Complexity.MODERATE,
            task_label="general", confidence=0.4,
            method="fallback",
        )

    def _merge_results(
        self, query: str, emb: ClassificationResult, heur: ClassificationResult
    ) -> ClassificationResult:
        """Weighted merge of embedding and heuristic results."""
        w_emb, w_heur = 0.6, 0.4
        complexities = [Complexity.CLOSE, Complexity.MODERATE, Complexity.DISTANT]
        emb_idx = complexities.index(emb.complexity)
        heur_idx = complexities.index(heur.complexity)

        merged_idx = emb_idx if emb.confidence >= heur.confidence else heur_idx
        merged_confidence = w_emb * emb.confidence + w_heur * heur.confidence

        return ClassificationResult(
            query=query,
            complexity=complexities[merged_idx],
            task_label=emb.task_label if emb.confidence > heur.confidence else heur.task_label,
            confidence=round(merged_confidence, 3),
            method="hybrid",
            metadata={"embedding": emb.metadata, "heuristic": heur.metadata},
        )

    def _infer_task(self, query: str, complexity: str) -> str:
        """Infer broad task label from query content."""
        q = query.lower()
        code_patterns = [
            r"```", r"def |class |import |from ", r"function\s+\w+\s*\(",
            r"code|script|function|class", r"debug|fix|error|bug|traceback",
        ]
        if any(re.search(p, q) for p in code_patterns):
            return "code"
        if any(w in q for w in ["explain", "what is", "describe", "meaning"]):
            return "explanation"
        if any(w in q for w in ["hello", "hi", "hey", "thanks"]):
            return "conversation"
        if any(w in q for w in ["plan", "design", "create", "build"]):
            return "planning"
        return "general"
