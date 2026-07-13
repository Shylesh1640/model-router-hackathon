"""Intent detection — classifies query intent from patterns.

Maps queries to intent categories before routing. Used by the pipeline
to inform tier selection and decomposition decisions.

Usage:
    detector = IntentDetector()
    result = detector.detect("Write a Python function to sort a list")
    print(result.intent)  # "code_generation"
"""

import re
from typing import Optional

from .models import IntentResult, IntentCategory


class IntentDetector:
    """Classifies query intent using pattern matching.

    Categories:
      question, code_generation, explanation, analysis,
      creative, summarization, command, general

    Extensible — add patterns to the class-level _PATTERNS dict.
    """

    _PATTERNS: dict[str, list[re.Pattern]] = {
        IntentCategory.QUESTION: [
            re.compile(r"^(what|who|why|how|where|when|which|whose)\b", re.I),
            re.compile(r"\?\s*$"),
            re.compile(r"^(is|are|was|were|do|does|did|can|could|would|should|will)\b.*\?$", re.I),
        ],
        IntentCategory.CODE_GENERATION: [
            re.compile(r"\b(write|create|implement|generate|build|make)\s+(a|an|the|some)\s+(function|class|program|script|code|module|api|endpoint|route|microservice|middleware)\b", re.I),
            re.compile(r"\b(code|script|function|implementation)\s+(for|to|that)\b", re.I),
            re.compile(r"^(write|create|implement|generate)\s+(a|an|the)\s+(function|class|program|script|code)\b", re.I),
            re.compile(r"\b(refactor|optimize|debug|fix)\s", re.I),
        ],
        IntentCategory.EXPLANATION: [
            re.compile(r"^(explain|describe|define|clarify|elaborate)\b", re.I),
            re.compile(r"\bwhat\s+is\s+(a|an|the)\b", re.I),
            re.compile(r"\b(how|why)\s+(does|do|is|are|can|would)\b", re.I),
            re.compile(r"\bmeaning\s+of\b", re.I),
            re.compile(r"\bdifference\s+between\b", re.I),
        ],
        IntentCategory.ANALYSIS: [
            re.compile(r"^(analyze|compare|contrast|evaluate|assess|review)\b", re.I),
            re.compile(r"\b(pros|cons|advantages|disadvantages)\s+(of|and)\b", re.I),
            re.compile(r"\bcompared?\s+to\b", re.I),
            re.compile(r"\btrade[-\s]?offs?\b", re.I),
        ],
        IntentCategory.CREATIVE: [
            re.compile(r"^(write|create|tell)\s+(a|an|the)\s+(\w+\s+)?(story|poem|essay|article|blog|post|tale)\b", re.I),
            re.compile(r"\b(creative|imagine|fantasy|fiction)\b", re.I),
            re.compile(r"\b(storytelling|narrative|plot)\b", re.I),
        ],
        IntentCategory.SUMMARIZATION: [
            re.compile(r"^(summarize|sum up|tl;dr|tl dr|recap)\b", re.I),
            re.compile(r"\bsummary\s+(of|for)\b", re.I),
            re.compile(r"\b(key|main)\s+(points|takeaways|findings)\b", re.I),
            re.compile(r"\bbrief\s+(overview|summary|recap)\b", re.I),
        ],
        IntentCategory.COMMAND: [
            re.compile(r"^(run|execute|start|stop|deploy|install|setup|configure|restart)\b", re.I),
            re.compile(r"\b(do|make|get|set|change|update|delete|remove|add)\s+\w+\s+(for|in|on|to)\b", re.I),
        ],
    }

    # Priority order — first match wins
    _PRIORITY = [
        IntentCategory.QUESTION,
        IntentCategory.CODE_GENERATION,
        IntentCategory.COMMAND,
        IntentCategory.CREATIVE,
        IntentCategory.SUMMARIZATION,
        IntentCategory.ANALYSIS,
        IntentCategory.EXPLANATION,
    ]

    def detect(self, query: str) -> IntentResult:
        """Classify query intent. Returns intent label + confidence."""
        q = query.strip()

        if not q:
            return IntentResult(
                query=query, intent=IntentCategory.GENERAL,
                confidence=0.0, method="unknown",
            )

        scores: dict[str, float] = {}
        for intent, patterns in self._PATTERNS.items():
            matches = sum(1 for p in patterns if p.search(q))
            if matches:
                scores[intent] = min(1.0, matches / len(patterns) + 0.3)

        if not scores:
            return IntentResult(
                query=query, intent=IntentCategory.GENERAL,
                confidence=0.5, method="pattern",
            )

        # Pick highest-confidence intent, tie-break by priority
        best = max(
            scores,
            key=lambda i: (scores[i], -self._PRIORITY.index(i) if i in self._PRIORITY else 0),
        )
        return IntentResult(
            query=query, intent=best,
            confidence=round(scores[best], 2),
            method="pattern",
            entities={"matched_patterns": self._matched_patterns(q, best)},
        )

    def _matched_patterns(self, query: str, intent: str) -> list[str]:
        """Return which patterns matched (for debugging)."""
        matched = []
        for p in self._PATTERNS.get(intent, []):
            if p.search(query):
                matched.append(p.pattern)
        return matched

    @staticmethod
    def needs_reasoning(intent: str) -> bool:
        """Whether this intent typically benefits from a reasoning model."""
        return intent in (
            IntentCategory.ANALYSIS,
            IntentCategory.CODE_GENERATION,
        )

    @staticmethod
    def suggest_tier(intent: str) -> Optional[str]:
        """Suggest a minimum tier for this intent.

        Returns None if intent doesn't force a specific tier.
        """
        suggestions = {
            IntentCategory.QUESTION: None,
            IntentCategory.CODE_GENERATION: "thinking",
            IntentCategory.EXPLANATION: "thinking",
            IntentCategory.ANALYSIS: "thinking",
            IntentCategory.CREATIVE: "thinking",
            IntentCategory.SUMMARIZATION: None,
            IntentCategory.COMMAND: None,
            IntentCategory.GENERAL: None,
        }
        return suggestions.get(intent)
