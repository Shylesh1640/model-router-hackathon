# Model Pool — Categorized for Cost-Aware Routing

Source: `openrouter_free_models.xlsx` — all free models available on OpenRouter.
Categorized by active parameter count and capability for Stark's three-tier routing.

---

## Utility — Pre/Post Processing

| Model | ID | Context | Role |
|-------|----|---------|------|
| Nemotron 3.5 Content Safety | `nvidia/nemotron-3.5-content-safety:free` | 128K | Safety filter — pre-flight guard |
| Free Models Router | `openrouter/free` | 200K | Meta-router — delegates to any free model |

---

## FAST Tier — <5B params, near-instant

| Model | ID | Params | Context | Notes |
|-------|----|--------|---------|-------|
| Liquid LFM 2.5 1.2B Thinking | `liquid/lfm-2.5-1.2b-thinking:free` | 1.2B | 32K | Has thinking mode, fastest option |
| Liquid LFM 2.5 1.2B Instruct | `liquid/lfm-2.5-1.2b-instruct:free` | 1.2B | 32K | Fastest instruct, no thinking overhead |
| Llama 3.2 3B Instruct | `meta-llama/llama-3.2-3b-instruct:free` | 3B | 128K | Solid general-purpose tiny model |

**Use case:** Greetings, factual lookups, simple Q&A, "what is X", confirmations.

---

## THINKING Tier — 3B–12B active params, reasoning capable

| Model | ID | Total | Active | Context | Type | Notes |
|-------|----|-------|--------|---------|------|-------|
| GPT-OSS-20b | `openai/gpt-oss-20b:free` | 21B | 3.6B | 128K | MoE | Strong reasoning + tool use, Apache 2.0 |
| Poolside Laguna XS 2.1 | `poolside/laguna-xs-2.1:free` | 33B | 3B | 262K | MoE | Agentic coding specialist |
| Poolside Laguna XS.2 | `poolside/laguna-xs.2:free` | 33B | 3B | 262K | MoE | Updated coding model, Apache 2.0 |
| Cohere North Mini Code | `cohere/north-mini-code:free` | 30B | 3B | 256K | MoE | Agentic coding, Apache 2.0 |
| Nemotron 3 Nano 30B A3B | `nvidia/nemotron-3-nano-30b-a3b:free` | 30B | 3B | 256K | MoE | General purpose |
| Nemotron 3 Nano Omni | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` | 30B | 3B | 256K | MoE | Reasoning-tuned Nano |
| Nemotron Nano 9B V2 | `nvidia/nemotron-nano-9b-v2:free` | 9B | — | 128K | Dense | Simple, no MoE overhead |
| Nemotron Nano 12B V2 VL | `nvidia/nemotron-nano-12b-v2-vl:free` | 12B | — | 128K | Dense | Multimodal (vision + language) |
| Gemma 4 26B A4B | `google/gemma-4-26b-a4b-it:free` | 26B | 4B | 262K | MoE | High efficiency, strong per-param |
| Gemma 4 31B | `google/gemma-4-31b-it:free` | 31B | — | 262K | Dense | Dense, predictable quality |
| Qwen3 Next 80B A3B | `qwen/qwen3-next-80b-a3b-instruct:free` | 80B | 3B | 262K | MoE | Strong bilingual (ZH+EN) |
| Venice Uncensored | `cognitivecomputations/dolphin-mistral-24b-venice-edition:free` | 24B | — | 32K | Dense | Uncensored path |

**Use case:** Code review, explanation, analysis, planning, error debugging, medium reasoning.

---

## DEEP Tier — 10B+ active or 70B+ dense, frontier

| Model | ID | Total | Active | Context | Type | Notes |
|-------|----|-------|--------|---------|------|-------|
| Llama 3.3 70B Instruct | `meta-llama/llama-3.3-70b-instruct:free` | 70B | — | 128K | Dense | Proven workhorse |
| GPT-OSS-120b | `openai/gpt-oss-120b:free` | 117B | 5.1B | 128K | MoE | Strongest GPT-OSS, tool use |
| Poolside Laguna M.1 | `poolside/laguna-m.1:free` | 225B | 23B | 262K | MoE | SOTA agentic coding |
| Nemotron 3 Super 120B A12B | `nvidia/nemotron-3-super-120b-a12b:free` | 120B | 12B | 1M | MoE | Heavy reasoning, 1M context |
| Nemotron 3 Ultra 550B A55B | `nvidia/nemotron-3-ultra-550b-a55b:free` | 550B | 55B | 1M | MoE | Largest free model, 1M context |
| Qwen3 Coder 480B A35B | `qwen/qwen3-coder:free` | 480B | 35B | 1M | MoE | SOTA coding, 1M context |
| Hermes 3 405B Instruct | `nousresearch/hermes-3-llama-3.1-405b:free` | 405B | — | 128K | Dense | Largest dense free model |

**Use case:** Complex debugging, code generation, research synthesis, multi-agent orchestration.

---

## Specialty — Audio/Music Generation

| Model | ID | Context | Use |
|-------|----|---------|-----|
| Lyria 3 Pro Preview | `google/lyria-3-pro-preview` | 1M | Text-to-music generation |
| Lyria 3 Clip Preview | `google/lyria-3-clip-preview` | 1M | Music clip generation |

---

## Routing Logic

```
Query → ComplexityClassifier → tier:
  simple  → FAST tier   (cheapest model)
  medium  → THINKING tier
  hard    → DEEP tier   (most capable)

Override: classification confidence < 0.4 → bump one tier up
Cascade:  fast response → self-verify → escalate if unsure (max 1 hop)
```

**Default model per tier (load-balanced):**
- Fast: `liquid/lfm-2.5-1.2b-thinking:free`
- Thinking: `openai/gpt-oss-20b:free`
- Deep: `meta-llama/llama-3.3-70b-instruct:free`
