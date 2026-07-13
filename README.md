# Model Router

**Cost-optimized LLM routing library.** Classify query complexity, pick the cheapest capable model from OpenRouter's free pool, generate. No chatbot, no safety guardrails, no domain prompts — just the routing engine.

```
pip install model-router
```

## Pipeline stages

```
Query → Intent → Decompose → SOT Lookup → Classify → Route → Generate
```

| Stage | Module | What it does |
|-------|--------|-------------|
| Intent | `intent.py` | Classifies query type: question, code, analysis, creative, etc. |
| Decompose | `decompose.py` | Detects sub-tasks, vision content → flags for reasoning/vision models |
| SOT | `store.py` | Dice-coefficient content-word store, zero deps |
| Classify | `classify.py` | d<0.30 close, d 0.3-0.6 moderate, d>0.6 distant |
| Route | `router.py` | Maps complexity + flags → cheapest capable model tier |
| Generate | `client.py` | OpenRouter API call with context + retries |

### Intent detection

Seven categories: `question`, `code_generation`, `explanation`, `analysis`, `creative`, `summarization`, `command`, `general`. Pattern-based, zero-dependency.

```python
from model_router import IntentDetector

detector = IntentDetector()
result = detector.detect("Write a Python function to sort a list")
print(result.intent)  # "code_generation"
```

### Decomposition & flagging

The decomposition analyzer checks for:
- **Sub-tasks** — conjunctions ("and then"), multi-question, numbered lists, multiple imperative verbs → sets `needs_reasoning=True`
- **Vision content** — image/screenshot/diagram references → sets `needs_vision=True`

When `needs_reasoning=True`, the router floors the tier at `thinking`. When `needs_vision=True`, it prefers a multimodal model (e.g. Nemotron Nano 12B VL).

```python
result = pipe.route(RouteRequest(query="Analyze this diagram and explain what it shows"))
print(result.decomposition.needs_reasoning)  # True
print(result.decomposition.needs_vision)     # True
print(result.decomposition.sub_tasks)        # [SubTask, SubTask]
```

### Langfuse telemetry

Optional. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`, or pass them directly:

```python
from model_router import Telemetry

tel = Telemetry()
tel.setup(public_key="pk-...", secret_key="sk-...")

pipe = RoutingPipeline(config, telemetry=tel)
```

Every pipeline stage creates a Langfuse span with input/output, timing, and metadata. No-op when Langfuse is not installed or configured.

## Routing tier selection

The router considers three signals:
1. **SOT distance** — base complexity
2. **Intent** — code/analysis tasks suggest thinking tier
3. **Decomposition flags** — sub-tasks → reasoning, vision content → multimodal

| Signal | Effect |
|--------|--------|
| SOT distance < 0.30 | base tier = fast |
| SOT distance 0.30–0.60 | base tier = thinking |
| SOT distance > 0.60 | base tier = deep |
| needs_reasoning=True | floor at thinking (escalates fast→thinking) |
| needs_vision=True | prefer multimodal model within tier |

## Route with flags

```python
from model_router import CostRouter, ClassificationResult

router = CostRouter()
classification = ClassificationResult(
    query="complex query", complexity="close",
    task_label="grounded", confidence=0.8,
    method="sot_distance", source_distance=0.2,
)

# Without flag → fast tier
decision = router.route("complex query", classification)
print(decision.tier)  # "fast"

# With reasoning flag → bumped to thinking
decision = router.route("complex query", classification, needs_reasoning=True)
print(decision.tier)  # "thinking"
```

## Model tiers

Routes queries across 25+ free OpenRouter models based on how well they match your reference data:

| Distance | Complexity | Tier | Model | Strategy |
|----------|-----------|------|-------|----------|
| < 0.30 | close | **fast** | Liquid 1.2B / Llama 3.2 3B | Answer from source |
| 0.30–0.60 | moderate | **thinking** | GPT-OSS-20b / Laguna XS | Source + web search |
| > 0.60 | distant | **deep** | Llama 3.3 70B / Qwen3 Coder | Full reasoning chain |

The **Source of Truth** (SOT) holds your reference documents. Queries close to known content get cheap fast models. Novel queries escalate to frontier models. The cascade feature optionally retries with a bigger model when confidence is low.

## Quick start

```bash
# Install
pip install model-router

# Or from source with dev extras
uv venv && uv pip install -e ".[dev,dashboard]"

# Configure
cp .env.example .env
# Add OPENROUTER_API_KEY
```

### Use as a library

```python
from model_router import RoutingPipeline, SourceOfTruth, get_config
from model_router.models import RouteRequest

# Seed your knowledge base
sot = SourceOfTruth()
sot.add_document("Paris is the capital of France.")

# Build the pipeline
pipe = RoutingPipeline(get_config())
pipe.sot = sot

# Route a query
result = pipe.route(RouteRequest(query="What is the capital of France?"))
print(result.response)
print(f"Model: {result.routing.model_name}  Tier: {result.routing.tier}")
```

### Run the demo

```bash
uv run python -m scripts.demo
```

### Run the dashboard

```bash
uv run python -m dashboard.app
```

Opens a web UI at `http://localhost:8080` — test routes, watch the live feed, explore the model pool.

## Package structure

```
model-router/
├── src/model_router/       # The library
│   ├── __init__.py         # Public API
│   ├── config.py           # Env-based config
│   ├── models.py           # Data models
│   ├── constants.py        # Model pool (25+ OpenRouter free models)
│   ├── router.py           # CostRouter — cheapest capable model selection
│   ├── client.py           # OpenRouterClient — API calls with retries
│   ├── store.py            # SourceOfTruth — Dice-coefficient document store
│   ├── classify.py         # DistanceClassifier — complexity from distance
│   ├── search.py           # WebSearcher — SearXNG integration
│   ├── intent.py           # IntentDetector — query intent classification
│   ├── decompose.py        # DecompositionAnalyzer — sub-tasks, vision flags
│   ├── telemetry.py        # Telemetry — optional Langfuse tracing
│   └── pipeline.py         # RoutingPipeline — intent → decompose → classify → route → generate
├── dashboard/              # Standalone FastAPI + WebSocket dashboard
├── scripts/demo.py         # Demo script
├── tests/                  # 54 tests (pytest)
├── data/                   # Sample datasets
└── pyproject.toml          # PEP 621 build config
```

## Why not a chatbot?

This library is explicitly **not a chatbot**. It doesn't:
- Have a CLI chat interface or interactive mode
- Block or rebuke off-topic queries
- Add "you are a helpful assistant" system prompts
- Manage conversations or personas

It **does** one thing: take a query, measure its distance from your reference data, pick the cheapest model that can handle it, and return the result. What you build on top — chatbot, API endpoint, batch processor — is your call.

## How the routing works

1. **Source of Truth lookup** — Dice coefficient on stopword-filtered content words. Zero external dependencies.
2. **Distance classification** — `close` (< 0.30), `moderate` (0.30–0.60), `distant` (> 0.60).
3. **Cost-aware selection** — picks the cheapest tier that matches the complexity. Within a tier, picks the most capable model (highest total params).
4. **Generation** — calls OpenRouter with context from the SOT. Optional web search for moderate queries, reasoning chain for distant queries.
5. **Cascade** (optional) — if confidence is low, retries with the next tier up.

## Model tiers

| Tier | Models | Params | Best for |
|------|--------|--------|----------|
| Fast | Liquid 1.2B, Llama 3.2 3B | < 3B | Simple Q&A, greetings, lookups |
| Thinking | GPT-OSS-20b, Laguna XS, North Mini Code, Gemma 4 | 3–31B | Code review, explanations, analysis |
| Deep | Llama 3.3 70B, GPT-OSS-120b, Qwen3 Coder, Hermes 405B | 70–550B | Debugging, generation, multi-step reasoning |

## Developing

```bash
# Install with dev deps
uv venv && uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Demo
uv run python -m scripts.demo
```
