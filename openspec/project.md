# Model Router — Project Context

## Purpose
**Model Router** is a Source-of-Truth chatbot-as-a-service. Every answer is grounded in a knowledge base. Query difficulty = embedding distance from nearest source document.

## Tech Stack
- **Runtime:** Python 3.14+, FastAPI, uvicorn, websockets
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2), fallback: character n-gram hashing
- **Vector Store:** In-memory (numpy) with ChromaDB option
- **LLMs:** OpenRouter API (27 free models across 3 tiers)
- **Search:** SearXNG (local meta-search)
- **Storage:** Local filesystem (JSON/CSV for datasets, pickle for models)

## Design Principles
1. **Zero hardcoded values** — every threshold in constants/config
2. **Graceful degradation** — every module has a fallback path
3. **No external dependencies for core** — n-gram fallback when sentence-transformers absent
4. **Distance as difficulty** — single metric drives routing decisions
5. **Safety first** — harmful content blocked before any processing

## Code Conventions
- 100-char line limit, 4-space indent
- Type hints on every public function
- Google-style docstrings
- Error handling at every boundary
- No secrets in code — env vars only

## File Structure
```
src/
├── sot/
│   ├── source_of_truth.py   # Vector KB
│   └── safety.py            # Safety guard
├── data/
│   ├── scraper.py           # Dataset scraping + SOT seeding
│   ├── dataset.py           # Dataset management
│   └── clean.py             # Text cleaning
├── train/
│   ├── embedder.py          # Fine-tune embedding model
│   └── evaluate.py          # Distance accuracy eval
├── router/
│   ├── engine.py            # CostRouter
│   ├── cascade.py           # Self-verify cascade
│   └── client.py            # OpenRouter API
├── search/
│   └── web_search.py        # SearXNG integration
├── reasoning/
│   └── deep_reasoning.py    # Multi-step CoT
├── dashboard/
│   └── app.py               # FastAPI + WebSocket
├── pipeline.py              # Main orchestrator
├── cli.py                   # CLI entry
├── config.py                # Env-based config
├── models.py                # Shared data models
└── constants.py             # Model pool definitions
```
