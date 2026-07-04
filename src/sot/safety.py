"""Safety guard — detects harmful queries and off-topic content.

Two-stage filtering:
1. Content safety: blocks harmful/dangerous/abusive queries
2. Domain relevance: catches off-topic queries for gentle rebuke
"""

import logging
import re
from typing import Optional

from src.models import SafetyResult

logger = logging.getLogger(__name__)

# High-severity patterns — immediate block
_HARMFUL_PATTERNS = [
    r"how\s+to\s+(make|create|build|synthesize|manufacture).*(bomb|explosive|weapon|drug|poison|meth)",
    r"how\s+to\s+(hack|crack|bypass|break\s+into|exploit)\s+",
    r"(sql\s+injection|xss|csrf)\s+(tutorial|example|how\s+to)",
    r"steal\s+(identity|password|credit\s+card|bank|account)",
    r"how\s+to\s+(commit|get\s+away\s+with)\s+(fraud|murder|theft|crime)",
    r"child\s+(abuse|pornography|exploitation)",
    r"self[- ]harm|suicide\s+(method|how\s+to|commit)",
    r"generate\s+(malware|ransomware|virus|trojan|worm)",
    r"ddos\s+(attack|tool|how\s+to)",
    r"phishing\s+(email|page|link|scam|template)",
    r"(bomb|explosive|weapon).*(make|create|build|how)",
]

# Low-severity — off-topic but not dangerous
_OFF_TOPIC_PATTERNS = [
    r"(what\s+is\s+your\s+favorite|do\s+you\s+have\s+a\s+girlfriend|tell\s+me\s+a\s+joke)",
    r"(who\s+is\s+the\s+(president|ceo|best)\b.*)",
    r"(sports|game|movie|celebrity|gossip|tabloid)\s",
    r"(cook\s+|recipe\s+for|baking\s+)",
    r"(horoscope|astrology|palm\s+reading)",
]


class SafetyGuard:
    """Two-stage safety filter for chatbot queries."""

    def __init__(self, domain_name: str = "this knowledge base"):
        self.domain_name = domain_name

    def check(self, query: str, source_distance: float = 1.0) -> SafetyResult:
        """Run safety + off-topic check. Returns result with optional rebuke."""
        q = query.lower().strip()

        # Stage 1: Harmful content check
        for pattern in _HARMFUL_PATTERNS:
            if re.search(pattern, q):
                return SafetyResult(
                    safe=False,
                    flagged=True,
                    category="harmful",
                    reason=f"Query matched harmful pattern: {pattern}",
                    rebuke_message=(
                        "I can't help with that request. "
                        "Please ask questions related to the knowledge base."
                    ),
                )

        # Stage 2: Off-topic check (only if far from source)
        if source_distance > 0.6:
            for pattern in _OFF_TOPIC_PATTERNS:
                if re.search(pattern, q):
                    return SafetyResult(
                        safe=True,
                        flagged=True,
                        category="off_topic",
                        reason="Query appears off-topic",
                        rebuke_message=(
                            f"I'm designed to answer questions about {self.domain_name}. "
                            f"Your question doesn't seem related. "
                            f"Could you rephrase it in the context of {self.domain_name}?"
                        ),
                    )

        # Stage 3: Very far from source with no domain link
        if source_distance > 0.85 and len(q.split()) > 3:
            return SafetyResult(
                safe=True,
                flagged=True,
                category="off_topic",
                reason=f"Query too distant from source (d={source_distance:.2f})",
                rebuke_message=(
                    f"That doesn't appear to be related to {self.domain_name}. "
                    f"Please ask something within the scope of the knowledge base."
                ),
            )

        return SafetyResult(safe=True, category="safe")


_safety: Optional[SafetyGuard] = None


def get_safety(domain: str = "this knowledge base") -> SafetyGuard:
    global _safety
    if _safety is None:
        _safety = SafetyGuard(domain_name=domain)
    return _safety
