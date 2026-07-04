"""Task decomposer — breaks complex queries into independently-routable sub-tasks.

For a query like "write a python script that fetches stock data and plots it",
the decomposer identifies sub-tasks:
  1. Research stock API (search/reasoning → fast model)
  2. Write data fetch function (code → thinking model)
  3. Write plot function (code → thinking model)
  4. Combine into script (code → deep model)

Each sub-task gets its own complexity classification and routing decision,
so easy parts use cheap models and hard parts use capable ones.
"""

import logging
import re
import uuid
from typing import Optional

from src.models import (
    Complexity, ClassificationResult, RoutingDecision,
    SubTask, DecompositionPlan, RouteRequest, RouteResponse,
)
from src.classifier.complexity import ComplexityClassifier
from src.router.engine import CostRouter

logger = logging.getLogger(__name__)


# Patterns that suggest a query might benefit from decomposition
_DECOMPOSITION_SIGNALS = [
    r"(and then|then|after that|next\b)",
    r"(first|second|third|finally|lastly)",
    r"(step\s+\d|step\s+by\s+step)",
    r"(write|create|build|implement).+(and|that)\s+(also|should)",
    r"(multi-step|multi.?stage|multi.?part|complex)",
    r"using\s+(multiple|several|different)\s+(api|service|library|tool)",
    r"full\s+(application|system|pipeline|workflow)",
    r"(collect|fetch|scrape|download).+(process|analyze|transform|plot|save)",
    r"(design|architect|plan)\s+(a|an|the)\s+(system|app|service|pipeline)",
]


class TaskDecomposer:
    """Decomposes complex queries into sub-tasks for cost optimization."""

    def __init__(self):
        self.classifier = ComplexityClassifier(method="hybrid")
        self.router = CostRouter()

    def should_decompose(self, query: str) -> bool:
        """Check if a query would benefit from decomposition."""
        q_lower = query.lower().strip()

        # Don't decompose very short queries
        if len(q_lower.split()) < 8:
            return False

        # Check for decomposition signals
        for pattern in _DECOMPOSITION_SIGNALS:
            if re.search(pattern, q_lower):
                return True

        # Decompose if the query is very long (>30 words) — likely complex
        if len(q_lower.split()) > 40:
            return True

        return False

    def decompose(self, query: str) -> DecompositionPlan:
        """Decompose a complex query into sub-tasks."""
        q = query.strip()
        plan = DecompositionPlan(original_query=q)

        # Strategy 1: Sequential steps (looks like a multi-step instruction)
        steps = self._extract_sequential_steps(q)
        if len(steps) >= 2:
            for i, step in enumerate(steps):
                sub = SubTask(
                    id=f"sub-{i+1}",
                    description=step,
                    depends_on=[f"sub-{j+1}" for j in range(i)],  # depends on previous
                    complexity=self._classify_step(step),
                    task_type=self._infer_task_type(step),
                )
                plan.sub_tasks.append(sub)
            plan.strategy = "sequential"
            plan.reasoning = f"Decomposed into {len(steps)} sequential steps"
            return plan

        # Strategy 2: Parallel concerns (multiple independent requirements)
        concerns = self._extract_parallel_concerns(q)
        if len(concerns) >= 2:
            for i, concern in enumerate(concerns):
                sub = SubTask(
                    id=f"concern-{i+1}",
                    description=concern,
                    depends_on=[],
                    complexity=self._classify_step(concern),
                    task_type=self._infer_task_type(concern),
                )
                plan.sub_tasks.append(sub)
            plan.strategy = "parallel"
            plan.reasoning = f"Decomposed into {len(concerns)} parallel concerns"
            return plan

        # Strategy 3: Dependent pipeline (A → B → C)
        pipeline = self._extract_pipeline(q)
        if len(pipeline) >= 2:
            for i, stage in enumerate(pipeline):
                sub = SubTask(
                    id=f"stage-{i+1}",
                    description=stage,
                    depends_on=[] if i == 0 else [f"stage-{j+1}" for j in range(i)],
                    complexity=self._classify_step(stage),
                    task_type=self._infer_task_type(stage),
                )
                plan.sub_tasks.append(sub)
            plan.strategy = "dependent"
            plan.reasoning = f"Decomposed into {len(pipeline)} pipeline stages"
            return plan

        # No decomposition found — return single sub-task
        plan.sub_tasks.append(SubTask(
            id="sub-1",
            description=query,
            complexity=self._classify_step(query),
            task_type=self._infer_task_type(query),
        ))
        plan.strategy = "sequential"
        plan.reasoning = "Single task, no decomposition needed"
        return plan

    def _extract_sequential_steps(self, query: str) -> list[str]:
        """Extract sequentially ordered steps from a query.

        Handles patterns like:
        - "first... then... finally..."
        - "step 1... step 2..."
        - Numbered lists
        """
        # Check for explicit step markers
        step_patterns = [
            r"(?:^(?:first|second|third|fourth|finally|lastly),\s*)(.+?)(?=(?:,\s*(?:second|third|fourth|finally|lastly)|$))",
            r"(?:Step\s+(\d+)[:.)]\s*)(.+?)(?=(?:Step\s+\d+|$))",
        ]

        # Try splitting on "then", "and then", "after that"
        splits = re.split(
            r"(?:,?\s*(?:and\s+)?then\s*,?|,?\s+after\s+that\s*,?|,?\s+next\s*,?)",
            query, flags=re.IGNORECASE
        )
        splits = [s.strip().rstrip(".,") for s in splits if s.strip()]
        if len(splits) >= 2:
            return splits

        return []

    def _extract_parallel_concerns(self, query: str) -> list[str]:
        """Extract parallel/independent concerns from a query.

        Handles patterns like:
        - "both X and Y"
        - "X. Also, Y."
        - "X. Additionally, Y."
        """
        # Split on bullet points or semicolons
        parts = re.split(r"[;]", query)
        parts = [p.strip() for p in parts if len(p.strip()) > 10]

        # Try splitting on "also", "additionally", "moreover"
        if not parts or len(parts) < 2:
            parts = re.split(
                r"(?:also|additionally|moreover|in\s+addition)",
                query, flags=re.IGNORECASE
            )
            parts = [p.strip().rstrip(".,") for p in parts if p.strip()]

        if len(parts) >= 2:
            return parts

        return []

    def _extract_pipeline(self, query: str) -> list[str]:
        """Extract pipeline stages (A, then B, then C)."""
        # Look for "collect X, process Y, output Z" pattern
        pipeline_match = re.search(
            r"(?:collect|fetch|get|read)\s+([^,.]+)"
            r"(?:,|then|and)\s*(?:process|analyze|transform|parse)\s+([^,.]+)"
            r"(?:,|then|and|finally)\s*(?:output|write|save|plot|return|display)\s+([^,.]+)",
            query, flags=re.IGNORECASE
        )
        if pipeline_match:
            return [p.strip() for p in pipeline_match.groups() if p.strip()]

        return []

    def _classify_step(self, text: str) -> str:
        """Classify complexity of a single sub-task."""
        result = self.classifier.estimate(text)
        return result.complexity

    def _infer_task_type(self, text: str) -> str:
        """Infer the type of work a sub-task represents."""
        t = text.lower()
        if any(w in t for w in ["write", "code", "function", "class", "implement", "create", "build"]):
            return "code"
        if any(w in t for w in ["search", "find", "look up", "research", "google"]):
            return "search"
        if any(w in t for w in ["explain", "describe", "analyze", "compare", "reason"]):
            return "reasoning"
        if any(w in t for w in ["plot", "chart", "graph", "visualize", "display"]):
            return "visualization"
        if any(w in t for w in ["test", "debug", "verify", "validate", "check"]):
            return "testing"
        if any(w in t for w in ["write", "document", "comment"]):
            return "writing"
        return "general"

    def route_subtasks(
        self,
        plan: DecompositionPlan,
        pipeline,
    ) -> DecompositionPlan:
        """Route each sub-task through the pipeline independently."""
        for i, sub in enumerate(plan.sub_tasks):
            logger.info(f"Routing sub-task {sub.id}: {sub.description[:50]}...")

            # Create a mini route request for this sub-task
            req = RouteRequest(query=sub.description)

            # Classify
            cls = self.classifier.estimate(sub.description)
            sub.complexity = cls.complexity

            # Route
            routing = self.router.route(sub.description, cls)
            sub.routing = routing

        return plan


# Singleton
_decomposer: Optional[TaskDecomposer] = None


def get_decomposer() -> TaskDecomposer:
    """Get or create the global task decomposer."""
    global _decomposer
    if _decomposer is None:
        _decomposer = TaskDecomposer()
    return _decomposer
