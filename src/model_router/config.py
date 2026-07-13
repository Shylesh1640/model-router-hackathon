"""Configuration for Model Router — loaded from environment."""

import os
from dataclasses import dataclass, field


@dataclass
class RouterConfig:
    """Configuration loaded from environment with sensible defaults.

    All fields can be overridden via environment variables.
    No secrets in code — API key from env only.
    """

    # OpenRouter
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    request_timeout_seconds: int = 20

    # Classification
    complexity_method: str = "hybrid"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Routing
    default_tier: str = "fast"
    rate_limit_retry_count: int = 3          # max retries per model call
    rate_limit_base_delay: float = 1.0       # base backoff in seconds
    rate_limit_max_delay: float = 30.0       # cap for exponential backoff

    # Cascade (escalation to next tier on low confidence)
    cascade_enabled: bool = True
    cascade_max_hops: int = 2                # fast → thinking → deep max
    cascade_max_budget_tokens: int = 10000   # hard cap across all hops

    # Circuit breaker (skip failing models temporarily)
    circuit_breaker_cooldown: int = 60       # seconds to skip a model
    circuit_breaker_max_failures: int = 3    # failures before tripping

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080
    dashboard_history_size: int = 1000

    # Logging
    log_level: str = "INFO"


def get_config() -> RouterConfig:
    """Load config from environment, with env var overrides."""
    return RouterConfig(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT", "20")),
        cascade_enabled=os.getenv("CASCADE_ENABLED", "true").lower() == "true",
        cascade_max_hops=int(os.getenv("CASCADE_MAX_HOPS", "2")),
        cascade_max_budget_tokens=int(os.getenv("CASCADE_MAX_BUDGET", "10000")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8080")),
        rate_limit_retry_count=int(os.getenv("RATE_LIMIT_RETRIES", "3")),
        circuit_breaker_cooldown=int(os.getenv("CB_COOLDOWN", "60")),
        circuit_breaker_max_failures=int(os.getenv("CB_MAX_FAILURES", "3")),
    )
