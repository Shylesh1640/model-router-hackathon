"""Distance-based query complexity classification.

Maps a source-of-truth distance to routing tier labels.
Can be used standalone or as the first stage of the pipeline.
"""

from .models import ClassificationResult, SourceQueryResult


class DistanceClassifier:
    """Classifies query complexity based on distance from a source of truth.

    Low distance → query is grounded in known content → use fast cheap model.
    High distance → query is novel → escalate to more capable model.

    Thresholds are tuned for Dice-coefficient distances (0-1 scale):
    - close < 0.60:   grounded, fast tier
    - moderate < 0.85: web_search, thinking tier
    - distant > 0.85:  deep_reasoning, deep tier

    Usage:
        classifier = DistanceClassifier()
        result = classifier.classify(query, source_result)
        print(result.complexity, result.task_label)
    """

    CLOSE_THRESHOLD = 0.60
    MODERATE_THRESHOLD = 0.80

    def classify(
        self,
        query: str,
        source_result: SourceQueryResult,
    ) -> ClassificationResult:
        """Classify query complexity from source-of-truth query result."""
        distance = source_result.min_distance
        complexity = self._complexity_from_distance(distance)
        task_label = self._task_from_distance(distance, source_result)
        confidence = max(0.5, 1.0 - distance)

        return ClassificationResult(
            query=query,
            complexity=complexity,
            task_label=task_label,
            confidence=confidence,
            method="sot_distance",
            source_distance=distance,
        )

    def _complexity_from_distance(self, distance: float) -> str:
        if distance < self.CLOSE_THRESHOLD:
            return "close"
        elif distance < self.MODERATE_THRESHOLD:
            return "moderate"
        return "distant"

    def _task_from_distance(
        self, distance: float, source_result: SourceQueryResult
    ) -> str:
        """Determine the task type based on source distance.

        - Close: grounded — answer from source, fast model
        - Moderate: web_search — source + web augmentation
        - Distant: deep_reasoning — full reasoning chain
        """
        if distance < self.CLOSE_THRESHOLD:
            return "grounded"
        elif distance < self.MODERATE_THRESHOLD:
            return "web_search"
        return "deep_reasoning"
