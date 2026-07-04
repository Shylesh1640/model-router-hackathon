"""Model pool definitions — all OpenRouter free models categorized into tiers."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelInfo:
    """Information about a model in the routing pool."""
    name: str
    openrouter_id: str
    tier: str  # "fast" | "thinking" | "deep" | "utility" | "specialty"
    total_params_b: Optional[float] = None
    active_params_b: Optional[float] = None
    context_length: int = 131072
    is_moe: bool = False
    is_multimodal: bool = False
    specialties: list[str] = field(default_factory=list)
    description: str = ""


# =============================================================================
# UTILITY — pre/post-processing, not for generation
# =============================================================================

UTILITY_MODELS = [
    ModelInfo(
        name="Nemotron 3.5 Content Safety",
        openrouter_id="nvidia/nemotron-3.5-content-safety:free",
        tier="utility",
        context_length=128000,
        description="Content safety filter — pre-flight routing guard",
    ),
    ModelInfo(
        name="Free Models Router",
        openrouter_id="openrouter/free",
        tier="utility",
        context_length=200000,
        description="OpenRouter meta-router — delegates to any available free model",
    ),
]

# =============================================================================
# FAST — queries under 5B active params, near-instant
# =============================================================================

FAST_MODELS = [
    ModelInfo(
        name="Liquid LFM 2.5 1.2B Thinking",
        openrouter_id="liquid/lfm-2.5-1.2b-thinking:free",
        tier="fast",
        total_params_b=1.2,
        context_length=32768,
        specialties=["thinking", "fast"],
        description="Tiny 1.2B model with thinking tokens. Near-instant.",
    ),
    ModelInfo(
        name="Liquid LFM 2.5 1.2B Instruct",
        openrouter_id="liquid/lfm-2.5-1.2b-instruct:free",
        tier="fast",
        total_params_b=1.2,
        context_length=32768,
        description="Tiny 1.2B instruct model. Fastest option.",
    ),
    ModelInfo(
        name="Llama 3.2 3B Instruct",
        openrouter_id="meta-llama/llama-3.2-3b-instruct:free",
        tier="fast",
        total_params_b=3.0,
        context_length=131072,
        description="Meta's smallest Llama. Solid general-purpose tiny model.",
    ),
]

# =============================================================================
# THINKING — 3B–12B active params or equivalent, reasoning capable
# =============================================================================

THINKING_MODELS = [
    ModelInfo(
        name="GPT-OSS-20b",
        openrouter_id="openai/gpt-oss-20b:free",
        tier="thinking",
        total_params_b=21.0,
        active_params_b=3.6,
        context_length=131072,
        is_moe=True,
        specialties=["reasoning", "tool-use"],
        description="OpenAI's open-weight 20B MoE (3.6B active). Strong reasoning + tool use.",
    ),
    ModelInfo(
        name="Poolside Laguna XS 2.1",
        openrouter_id="poolside/laguna-xs-2.1:free",
        tier="thinking",
        total_params_b=33.0,
        active_params_b=3.0,
        context_length=262144,
        is_moe=True,
        specialties=["coding"],
        description="Agentic coding MoE. Strong on multi-step code tasks.",
    ),
    ModelInfo(
        name="Poolside Laguna XS.2",
        openrouter_id="poolside/laguna-xs.2:free",
        tier="thinking",
        total_params_b=33.0,
        active_params_b=3.0,
        context_length=262144,
        is_moe=True,
        specialties=["coding"],
        description="Updated Laguna XS. Apache 2.0. Agentic coding specialist.",
    ),
    ModelInfo(
        name="Cohere North Mini Code",
        openrouter_id="cohere/north-mini-code:free",
        tier="thinking",
        total_params_b=30.0,
        active_params_b=3.0,
        context_length=256000,
        is_moe=True,
        specialties=["coding"],
        description="Cohere's agentic coding model. 30B total / 3B active. Apache 2.0.",
    ),
    ModelInfo(
        name="Nemotron 3 Nano 30B A3B",
        openrouter_id="nvidia/nemotron-3-nano-30b-a3b:free",
        tier="thinking",
        total_params_b=30.0,
        active_params_b=3.0,
        context_length=256000,
        is_moe=True,
        description="NVIDIA general purpose MoE. 30B total / 3B active.",
    ),
    ModelInfo(
        name="Nemotron 3 Nano Omni Reasoning",
        openrouter_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        tier="thinking",
        total_params_b=30.0,
        active_params_b=3.0,
        context_length=256000,
        is_moe=True,
        specialties=["reasoning"],
        description="Reasoning-tuned variant of Nemotron Nano.",
    ),
    ModelInfo(
        name="Nemotron Nano 9B V2",
        openrouter_id="nvidia/nemotron-nano-9b-v2:free",
        tier="thinking",
        total_params_b=9.0,
        context_length=128000,
        description="Small dense NVIDIA model. No MoE complexity.",
    ),
    ModelInfo(
        name="Nemotron Nano 12B V2 VL",
        openrouter_id="nvidia/nemotron-nano-12b-v2-vl:free",
        tier="thinking",
        total_params_b=12.0,
        context_length=128000,
        is_multimodal=True,
        description="Multimodal model — vision + language.",
    ),
    ModelInfo(
        name="Gemma 4 26B A4B",
        openrouter_id="google/gemma-4-26b-a4b-it:free",
        tier="thinking",
        total_params_b=26.0,
        active_params_b=4.0,
        context_length=262144,
        is_moe=True,
        specialties=["reasoning"],
        description="Google's high-efficiency MoE. Strong per-parameter.",
    ),
    ModelInfo(
        name="Gemma 4 31B",
        openrouter_id="google/gemma-4-31b-it:free",
        tier="thinking",
        total_params_b=31.0,
        context_length=262144,
        description="Google's dense 31B. Predictable quality.",
    ),
    ModelInfo(
        name="Qwen3 Next 80B A3B Instruct",
        openrouter_id="qwen/qwen3-next-80b-a3b-instruct:free",
        tier="thinking",
        total_params_b=80.0,
        active_params_b=3.0,
        context_length=262144,
        is_moe=True,
        description="Qwen's 80B MoE. Strong Chinese + English.",
    ),
    ModelInfo(
        name="Venice Uncensored (Dolphin 24B)",
        openrouter_id="cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
        tier="thinking",
        total_params_b=24.0,
        context_length=32768,
        specialties=["uncensored"],
        description="Uncensored Dolphin Mistral 24B via Venice.",
    ),
]

# =============================================================================
# DEEP — frontier-class, 10B+ active or 70B+ dense
# =============================================================================

DEEP_MODELS = [
    ModelInfo(
        name="Llama 3.3 70B Instruct",
        openrouter_id="meta-llama/llama-3.3-70b-instruct:free",
        tier="deep",
        total_params_b=70.0,
        context_length=131072,
        description="Proven frontier workhorse. Reliable 70B dense.",
    ),
    ModelInfo(
        name="GPT-OSS-120b",
        openrouter_id="openai/gpt-oss-120b:free",
        tier="deep",
        total_params_b=117.0,
        active_params_b=5.1,
        context_length=131072,
        is_moe=True,
        specialties=["reasoning", "tool-use", "agentic"],
        description="OpenAI's open-weight 120B MoE (5.1B active). Strongest GPT-OSS reasoning.",
    ),
    ModelInfo(
        name="Poolside Laguna M.1",
        openrouter_id="poolside/laguna-m.1:free",
        tier="deep",
        total_params_b=225.0,
        active_params_b=23.0,
        context_length=262144,
        is_moe=True,
        specialties=["coding", "agentic"],
        description="SOTA agentic coding. 225B total / 23B active. Long-horizon.",
    ),
    ModelInfo(
        name="Nemotron 3 Super 120B A12B",
        openrouter_id="nvidia/nemotron-3-super-120b-a12b:free",
        tier="deep",
        total_params_b=120.0,
        active_params_b=12.0,
        context_length=1_000_000,
        is_moe=True,
        specialties=["reasoning", "science", "coding"],
        description="NVIDIA's heavy reasoning MoE. 1M context.",
    ),
    ModelInfo(
        name="Nemotron 3 Ultra 550B A55B",
        openrouter_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        tier="deep",
        total_params_b=550.0,
        active_params_b=55.0,
        context_length=1_000_000,
        is_moe=True,
        specialties=["frontier"],
        description="Largest free model on OpenRouter. 550B total / 55B active. 1M context.",
    ),
    ModelInfo(
        name="Qwen3 Coder 480B A35B",
        openrouter_id="qwen/qwen3-coder:free",
        tier="deep",
        total_params_b=480.0,
        active_params_b=35.0,
        context_length=1_048_576,
        is_moe=True,
        specialties=["coding"],
        description="SOTA coding MoE. 480B total / 35B active. 1M context.",
    ),
    ModelInfo(
        name="Hermes 3 405B Instruct",
        openrouter_id="nousresearch/hermes-3-llama-3.1-405b:free",
        tier="deep",
        total_params_b=405.0,
        context_length=131072,
        description="Largest dense free model. Broad knowledge.",
    ),
]

# =============================================================================
# SPECIALTY — not for text generation routing
# =============================================================================

SPECIALTY_MODELS = [
    ModelInfo(
        name="Lyria 3 Pro Preview",
        openrouter_id="google/lyria-3-pro-preview",
        tier="specialty",
        context_length=1_048_576,
        specialties=["music"],
        description="Google's music generation model.",
    ),
    ModelInfo(
        name="Lyria 3 Clip Preview",
        openrouter_id="google/lyria-3-clip-preview",
        tier="specialty",
        context_length=1_048_576,
        specialties=["music"],
        description="Google's music clip generation.",
    ),
]

# =============================================================================
# MASTER POOL
# =============================================================================

ALL_MODELS: list[ModelInfo] = (
    UTILITY_MODELS + FAST_MODELS + THINKING_MODELS + DEEP_MODELS + SPECIALTY_MODELS
)

TIER_MODELS: dict[str, list[ModelInfo]] = {
    "fast": FAST_MODELS,
    "thinking": THINKING_MODELS,
    "deep": DEEP_MODELS,
}

DEFAULT_MODEL_PER_TIER: dict[str, str] = {
    "fast": "liquid/lfm-2.5-1.2b-thinking:free",
    "thinking": "openai/gpt-oss-20b:free",
    "deep": "meta-llama/llama-3.3-70b-instruct:free",
}
