"""Configuration for Model Router."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RouterConfig:
    """Configuration loaded from environment with sensible defaults."""
    
    # OpenRouter
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    request_timeout_seconds: int = 60
    
    # Classification
    complexity_method: str = "hybrid"  # "embedding" | "heuristic" | "hybrid"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Routing
    default_tier: str = "thinking"  # fallback if classification fails
    confidence_override_threshold: float = 0.4  # below this = bump tier
    rate_limit_retry_count: int = 2
    
    # Cascade
    cascade_enabled: bool = True
    cascade_max_hops: int = 1  # max 1 escalation per query
    cascade_escalation_threshold: float = 0.3  # if self-verify confidence below this
    
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
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT", "60")),
        cascade_enabled=os.getenv("CASCADE_ENABLED", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        dashboard_port=int(os.getenv("DASHBOARD_PORT", "8080")),
    )
