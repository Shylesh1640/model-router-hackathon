# Model Router — Hackathon Project

**Cost-optimized LLM routing with real-time dashboard.**

Route queries across OpenRouter's free models using a smart classification layer. Simple questions hit tiny models. Hard problems escalate to frontier models. The dashboard shows every decision — which model, why, what it cost (in tokens), and whether the cascade fired.

## Architecture

```
Query → Classifier → Router → Model → [Cascade] → Response → Dashboard
```

### Classification Layer
- **Complexity estimation:** embedding-based (simple / medium / hard)
- **Task detection:** semantic similarity to known categories
- **Fallback:** keyword/regex for cold start

### Routing Engine
- **Model pool:** all 27 free OpenRouter models
- **Cost-aware selection:** picks cheapest model that can handle the complexity
- **Fallback chain:** if a model is rate-limited or down, try next in tier

### Cascade (optional)
- If fast model's confidence is low, escalate to next tier
- Self-verification: ask the same model to check its own answer
- Max 1 hop per query

### Dashboard
- Real-time WebSocket updates
- Shows: query → classified tier → model selected → cost → response quality
- Model pool health (rate limits, uptime)
- History with filters by tier, model, date

## Project Structure

```
model-router-hackathon/
├── src/
│   ├── classifier/        # Complexity + task classification
│   │   ├── __init__.py
│   │   ├── complexity.py  # Simple/Medium/Hard estimator
│   │   ├── task_detector.py
│   │   └── classifier.py  # Orchestrator
│   ├── router/            # Routing engine
│   │   ├── __init__.py
│   │   ├── engine.py      # Cost-aware model selection
│   │   ├── pool.py        # Model pool manager (27 models)
│   │   ├── cascade.py     # Self-verification + escalation
│   │   └── client.py      # OpenRouter API client
│   ├── dashboard/         # Web dashboard
│   │   ├── __init__.py
│   │   ├── app.py         # FastAPI server
│   │   ├── websocket.py   # Real-time updates
│   │   └── static/        # Frontend assets
│   ├── models.py          # Shared data models
│   ├── config.py          # Configuration
│   └── constants.py       # Model pool definitions
├── dashboard/
│   └── frontend/          # React/Vue/Svelte frontend
├── tests/
├── docs/
│   └── models.md          # Categorized model list
└── data/                  # Sample queries, benchmark results
```

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your OpenRouter API key

# Run classifier test
python -m src.classifier.complexity "hello"

# Start dashboard
python -m src.dashboard.app

# API
curl http://localhost:8000/route -d '{"query": "explain python decorators"}'
```

## Model Tiers (OpenRouter Free)

| Tier | Active Params | Models | Use Case |
|------|--------------|--------|----------|
| Fast | <3B | Llama 3.2 3B, Liquid 1.2B | Greetings, facts, simple Q&A |
| Thinking | 3B–12B | GPT-OSS-20b, Laguna XS, North Mini Code, Nemotron Nano | Code review, explanations, analysis |
| Deep | 10B+ active | GPT-OSS-120b, Llama 3.3 70B, Qwen3 Coder, Hermes 405B | Debugging, generation, multi-step reasoning |
