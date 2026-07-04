# Model Router — Project Plan

## Concept
A **Source-of-Truth chatbot** where every answer is grounded in a knowledge base.
Query difficulty = embedding distance from nearest source document.

## Core Architecture

```
Query → Safety → SOT Lookup → Distance → Route → Generate
```

### Pipeline Stages

| Stage | Module | What it does |
|-------|--------|-------------|
| Safety | `src/sot/safety.py` | Blocks harmful, rebukes off-topic |
| SOT | `src/sot/source_of_truth.py` | Vector KB, n-gram embeddings, cosine distance |
| Classify | — | d<0.30 close, d 0.3-0.6 moderate, d>0.6 distant |
| Route | `src/router/engine.py` | Maps complexity to model tier |
| Generate | `src/router/client.py` | OpenRouter API call with context |
| Cascade | `src/router/cascade.py` | Self-verify, escalate if needed |
| Web Search | `src/search/web_search.py` | SearXNG for external info |
| Deep Reason | `src/reasoning/deep_reasoning.py` | Multi-step CoT for complex queries |

### Routing Matrix

| SOT Distance | Complexity | Tier | Model | Action |
|-------------|-----------|------|-------|--------|
| < 0.30 | close | grounded | Liquid 1.2B | Answer from source |
| 0.30–0.60 | moderate | web_search | GPT-OSS-20b | Source + web search |
| > 0.60 | distant | deep_reasoning | Llama 3.3 70B | Full reasoning chain |

## Data Pipeline

### Sources

1. **Alexa QA Dataset** (HuggingFace: `theblackcat102/alexa-qa`)
   - 136K question-answer pairs
   - Split: 70% train, 10% validation, 20% test
   - Used for: training the embedding model + seeding SOT

2. **Alexa Top 1M Domains** (portions)
   - Domain list from S3/Kaggle/Majestic
   - Scrape top N domains for content
   - Used for: wide-coverage source of truth seeding

### Processing Pipeline

```
Raw data → Clean → Embed → Store in SOT → Train embedder
```

### Training

- Fine-tune `all-MiniLM-L6-v2` on Alexa QA pairs
- Contrastive loss: similar Qs close, different Qs far
- Export to ONNX for fast CPU inference
- Fallback: character n-gram hashing (already working)

## Implementation Order

### Phase 1: Data Pipeline (current)
- [ ] Scrape Alexa QA dataset from HuggingFace
- [ ] Scrape Top 1M domain list
- [ ] Scrape content from top domains (first 10K)
- [ ] Clean and deduplicate
- [ ] Store in SOT

### Phase 2: Embedding Training
- [ ] Fine-tune MiniLM on QA pairs
- [ ] Evaluate distance accuracy
- [ ] Export optimized model
- [ ] Wire into Source of Truth

### Phase 3: Production Hardening
- [ ] Dashboard updates
- [ ] Rate limiting
- [ ] Caching
- [ ] Monitoring

## File Structure

```
src/
├── sot/
│   ├── source_of_truth.py   # Vector KB
│   └── safety.py            # Safety guard
├── data/
│   ├── scraper.py           # Scrape domains + HF datasets
│   ├── dataset.py           # Dataset management
│   └── clean.py             # Text cleaning
├── train/
│   ├── embedder.py          # Fine-tune embedding model
│   └── evaluate.py          # Evaluate distance accuracy
├── router/
│   ├── engine.py            # CostRouter
│   ├── cascade.py           # Self-verify cascade
│   └── client.py            # OpenRouter API
├── search/
│   └── web_search.py        # SearXNG
├── reasoning/
│   └── deep_reasoning.py    # CoT chain
├── pipeline.py              # Main orchestrator
├── cli.py                   # CLI entry point
└── config.py                # Env configuration
```

## Milestones

| # | Milestone | Done |
|---|-----------|------|
| 1 | Core pipeline: safety → SOT → route → generate | ✅ |
| 2 | Web search + deep reasoning tiers | ✅ |
| 3 | Distance-based difficulty classification | ✅ |
| 4 | Drawio architecture diagram | ✅ |
| 5 | Alexa QA + domain scraper |  |
| 6 | Embedding model training |  |
| 7 | Accurate distance metrics |  |
| 8 | Dashboard with SOT analytics |  |
| 9 | Deployable demo |  |
