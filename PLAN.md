# Model Router — Project Plan

## Concept
A **cost-optimized LLM routing library** — classify query complexity by distance from a source of truth, then select the cheapest capable model from OpenRouter's free pool.

## Core Architecture

```
Query → SOT Lookup → Distance → Classify → Route → Generate → [Cascade]
```

### Pipeline Stages

| Stage | Module | What it does |
|-------|--------|-------------|
| SOT | `store.py` | Dice-coefficient content-word store, zero deps |
| Classify | `classify.py` | d<0.30 close, d 0.3-0.6 moderate, d>0.6 distant |
| Route | `router.py` | Maps complexity to cheapest capable model tier |
| Generate | `client.py` | OpenRouter API call with context + retries |
| Cascade | `pipeline.py` | Low-confidence? Retry with next tier up |
| Web Search | `search.py` | SearXNG for queries beyond SOT coverage |

### Routing Matrix

| SOT Distance | Complexity | Tier | Model | Strategy |
|-------------|-----------|------|-------|----------|
| < 0.30 | close | fast | Liquid 1.2B / Llama 3.2 3B | Answer from source |
| 0.30–0.60 | moderate | thinking | GPT-OSS-20b / Laguna XS | Source + web search |
| > 0.60 | distant | deep | Llama 3.3 70B / Qwen3 Coder | Full reasoning chain |

### NOT included (by design)

This is a library, not a chatbot. No:
- CLI chat interface or interactive mode
- Safety guard (no harmful-content filtering, no off-topic rebukes)
- Domain-specific system prompts
- Conversation management or persona system

Those belong in the calling application.

## Package Structure

```
src/model_router/
├── __init__.py       # Public API — exports all key classes
├── _version.py       # 0.2.0
├── config.py         # RouterConfig — env-based, no hardcoded values
├── models.py         # Data models — RouteRequest/Response, ClassificationResult, etc.
├── constants.py      # Model pool — 25+ OpenRouter free models with benchmarks
├── router.py         # CostRouter — cheapest capable model selection
├── client.py         # OpenRouterClient — API client with retries
├── store.py          # SourceOfTruth — Dice-coefficient document store
├── classify.py       # DistanceClassifier — complexity from SOT distance
├── search.py         # WebSearcher — SearXNG integration
└── pipeline.py       # RoutingPipeline — classify → route → generate
```

## Tests

25 tests across 4 test files, covering:
- Router tier selection and fallback
- Source of Truth CRUD and similarity
- Distance classifier boundaries
- Pipeline integration (without API key)

## Data Sources

1. **Alexa QA Dataset** (HuggingFace: `theblackcat102/alexa-qa`)
   - 136K question-answer pairs — can seed the SOT for domain-specific routing
2. **Alexa Top 1M Domains** (portions) — optional web content for SOT seeding

## Milestones

| # | Milestone | Status |
|---|-----------|--------|
| 1 | Core pipeline: classify → route → generate | ✅ |
| 2 | Web search + deep reasoning tiers | ✅ |
| 3 | Distance-based difficulty classification | ✅ |
| 4 | Installable package (pyproject.toml, PEP 621) | ✅ |
| 5 | Dashboard (FastAPI + WebSocket) | ✅ |
| 6 | 25 passing tests | ✅ |
| 7 | Removed all chatbot/safety/domain cruft | ✅ |
| 8 | Embedding model training pipeline | |
| 9 | Deployable demo | |
