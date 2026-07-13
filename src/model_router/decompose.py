"""Decomposition analyzer — detects sub-tasks and flags routing requirements.

Analyses a query for:
1. **Sub-tasks** — multiple instructions or questions → needs reasoning model
2. **Vision content** — image/video references → needs vision-capable model

If either is found, the pipeline bumps the tier accordingly.

Usage:
    analyzer = DecompositionAnalyzer()
    result = analyzer.analyze("Write a Python script and explain how it works")
    print(result.has_sub_tasks)      # True
    print(result.needs_reasoning)    # True
"""

import re
from typing import Optional

from .models import DecompositionResult, SubTask


class DecompositionAnalyzer:
    """Heuristic decomposition analysis for query complexity.

    Splits queries on structural markers (conjunctions, bullets,
    numbered lists) to detect sub-tasks. Also detects vision-related
    content (image references, screenshot requests, diagram queries).
    """

    # Conjunctions and connectors that often separate sub-tasks
    _SUB_TASK_SEPARATORS = re.compile(
        r"\b(and\s+(also|then|separately|additionally|furthermore|meanwhile)|"
        r"also\s|"
        r"then\s|"
        r"additionally\s|"
        r"furthermore\s|"
        r"meanwhile\s|"
        r"next[,.]?\s|"
        r"after\s+that\b|"
        r"in\s+addition\b|"
        r",\s+and\s+)", re.I
    )

    # Conjunction "and" between two imperative clauses (no comma required)
    _AND_BETWEEN_VERBS = re.compile(
        r"\b\w+\s+this\s+\w+\s+and\s+(analyze|explain|describe|create|build|"
        r"write|test|deploy|run|check|review|compare|list|generate|convert|"
        r"extract|summarize|implement|debug|fix)\b", re.I
    )

    # Bullet / numbered list indicators
    _LIST_INDICATORS = re.compile(
        r"(?:^|\n)\s*[\-\*•]|\d+[\.\)]\s", re.M
    )

    # Multi-question detection (more than one ?)
    _MULTI_QUESTION = re.compile(r"\?.*\?")

    # Imperative verb chains: "do X and do Y"
    _IMPERATIVE_CHAIN = re.compile(
        r"(?:,\s*(?:then|and|or)\s+)(\w+)", re.I
    )

    # Vision content indicators
    _VISION_INDICATORS = [
        re.compile(r"\b(image|picture|photo|screenshot|diagram|chart|graph|drawing|illustration)\b", re.I),
        re.compile(r"\b(visual|see\s+this|look\s+at|shown?\s+in)\b", re.I),
        re.compile(r"\b(ocr|optical\s+character|extract\s+text\s+from\s+image)\b", re.I),
        re.compile(r"\b(svg|png|jpg|jpeg|gif|webp|bmp)\b", re.I),
    ]

    # Multi-instruction — multiple imperative verbs
    _MULTI_VERB = re.compile(
        r"(write|create|build|make|implement|analyze|explain|describe|compare|"
        r"summarize|debug|test|deploy|fix|refactor|optimize|review|check|"
        r"list|generate|convert|transform|sort|filter|map|reduce)\b", re.I
    )

    def analyze(self, query: str) -> DecompositionResult:
        """Analyse query for sub-tasks, vision content, and reasoning needs."""
        q = query.strip()
        result = DecompositionResult(query=q, method="heuristic")

        if not q:
            result.reason = "Empty query"
            return result

        reasons = []

        # --- Sub-task detection ---

        # 1. Separator-based splitting
        separator_parts = self._SUB_TASK_SEPARATORS.split(q)
        if len(separator_parts) > 1:
            result.has_sub_tasks = True
            reasons.append("conjunctions/connectors found")

        # 1b. "and" between two verb clauses (no comma required)
        if self._AND_BETWEEN_VERBS.search(q):
            result.has_sub_tasks = True
            reasons.append("verb-and-verb clause found")

        # 2. List detection (bullets, numbered items)
        if self._LIST_INDICATORS.search(q):
            result.has_sub_tasks = True
            reasons.append("list markers found")

        # 3. Multi-question
        if self._MULTI_QUESTION.search(q):
            result.has_sub_tasks = True
            reasons.append("multiple questions found")

        # 4. Multiple distinct verbs
        verbs = self._MULTI_VERB.findall(q)
        unique_verbs = set(v.lower() for v in verbs)
        if len(unique_verbs) >= 3:
            result.has_sub_tasks = True
            reasons.append(f"multiple instructions: {', '.join(sorted(unique_verbs))}")

        # --- Vision content detection ---

        for pattern in self._VISION_INDICATORS:
            if pattern.search(q):
                result.has_vision_content = True
                result.needs_vision = True
                reasons.append(f"vision content: {pattern.pattern[:30]}")
                break

        # --- Build sub-tasks if detected ---

        if result.has_sub_tasks:
            result.needs_reasoning = True
            sub_tasks = self._extract_sub_tasks(q)
            result.sub_tasks = sub_tasks
            result.reason = "; ".join(reasons) if reasons else "sub-tasks detected"
        else:
            result.reason = "No decomposition needed"

        return result

    def _extract_sub_tasks(self, query: str) -> list[SubTask]:
        """Extract individual sub-tasks from query text.

        Simple sentence splitting on separators and list markers.
        Returns at most 8 sub-tasks to avoid explosion.
        """
        parts = []

        # Try splitting on numbered list items
        items = re.split(r"(?:^|\n)\s*\d+[\.\)]\s*", query)
        if len(items) > 2:
            parts = [item.strip() for item in items if item.strip()]
        else:
            # Try bullet points
            items = re.split(r"(?:^|\n)\s*[\-\*•]\s*", query)
            if len(items) > 2:
                parts = [item.strip() for item in items if item.strip()]
            else:
                # Split on conjunction boundaries
                parts = re.split(
                    r",\s+(?:and|then|also|additionally|furthermore|meanwhile)\s+|"
                    r"\s+(?:and\s+also|and\s+then|and\s+additionally)\s+",
                    query,
                )

        sub_tasks = []
        for i, part in enumerate(parts[:8]):
            part = part.strip().rstrip(".,;")
            if len(part) > 5:  # skip noise
                sub_tasks.append(SubTask(
                    id=f"st-{i + 1}",
                    description=part[:200],
                    needs_reasoning=len(part) > 50,
                ))

        return sub_tasks
