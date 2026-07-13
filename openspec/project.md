# Model Router — Project Context

## Purpose
**Model Router** is a cost-optimized LLM routing library. Classifies query complexity by distance from a source of truth, then selects the cheapest capable model from OpenRouter's free pool.

## Tech Stack
- **Runtime:** Python 3.11+, requests
- **Embeddings:** Dice coefficient on content words (zero deps). Optional: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store:** In-memory Source of Truth
- **LLMs:** OpenRouter API (25+ free models across 3 tiers)
- **Search:** SearXNG (local meta-search)
- **Dashboard:** FastAPI + WebSocket + Tailwind

## Design Principles
1. **Zero hardcoded values** — every threshold in constants/config
2. **Graceful degradation** — every module has a fallback path
3. **No external dependencies for core** — Dice coefficient works without ML libs
4. **Distance as difficulty** — single metric drives routing decisions
5. **Library, not chatbot** — no safety guards, no domain prompts, no conversation management

## Code Conventions
- 100-char line limit, 4-space indent
- Type hints on every public function
- Google-style docstrings
- Error handling at every boundary
- No secrets in code — env vars only

## File Structure
```
src/model_router/
├── __init__.py       # Public API
├── _version.py       # 0.2.0
├── config.py         # Env-based config
├── models.py         # Data models
├── constants.py      # Model pool definitions
├── router.py         # CostRouter
├── client.py         # OpenRouter API client
├── store.py          # Source of Truth (Dice coefficient)
├── classify.py       # Distance-based classification
├── search.py         # SearXNG integration
└── pipeline.py       # Main orchestrator
```
