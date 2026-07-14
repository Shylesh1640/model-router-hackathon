"""Model Router Dashboard — FastAPI + WebSocket real-time UI.

Standalone web app for testing and monitoring the routing pipeline.
"""

import csv
import logging
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse

# Load .env before anything else reads config
load_dotenv()

from model_router.config import get_config
from model_router.models import RouteRequest
from model_router.pipeline import RoutingPipeline
from model_router.constants import ALL_MODELS, FAST_MODELS, THINKING_MODELS, DEEP_MODELS
from model_router.store import SourceOfTruth

logger = logging.getLogger(__name__)

config = get_config()
pipeline = RoutingPipeline(config)

# WebSocket connections for live dashboard
_websockets: list[WebSocket] = []


def seed_sot_from_csv(csv_path: str, max_rows: int = 500):
    """Seed the Source of Truth from a CSV with question/answer columns."""
    path = Path(csv_path)
    if not path.exists():
        logger.warning("Seed CSV not found: %s", csv_path)
        return 0
    sot = pipeline.sot  # direct access for seeding
    if sot.count() > 0:
        logger.info("SOT already has %s docs, skipping seed", sot.count())
        return sot.count()
    count = 0
    encodings = ("utf-8-sig", "cp1252", "latin-1")
    last_error = None
    for encoding in encodings:
      try:
        with open(path, encoding=encoding, newline="") as f:
          reader = csv.DictReader(f)
          for i, row in enumerate(reader):
            if i >= max_rows:
              break
            q = row.get("question", row.get("Question", ""))
            a = row.get("answer", row.get("Answer", ""))
            if q and a:
              sot.add_document(f"Q: {q}\nA: {a}", source="alexa-qa")
              count += 1
        break
      except UnicodeDecodeError as exc:
        last_error = exc
        count = 0
        continue
    else:
      raise last_error
    logger.info("Seeded %s docs into SOT from %s", count, csv_path)
    return count


def broadcast(route_response):
    """Push route decision to all connected dashboard clients."""
    meta = route_response.classification.metadata or {}
    data = {
        "type": "route",
        "query": route_response.query,
        "response_preview": route_response.response[:200],
        "complexity": route_response.classification.complexity,
        "task": route_response.classification.task_label,
        "confidence": route_response.classification.confidence,
        "method": route_response.classification.method,
        "tier": route_response.routing.tier,
        "model_id": route_response.routing.model_id,
        "model_name": route_response.routing.model_name,
        "reason": route_response.routing.reason,
        "tokens_in": route_response.generation.tokens_in,
        "tokens_out": route_response.generation.tokens_out,
        "latency_ms": route_response.generation.latency_ms,
        "escalated": route_response.generation.cascade_escalated,
        "error": route_response.generation.error,
        # Heatmap metadata
        "match_density": meta.get("match_density"),
        "coverage": meta.get("coverage"),
        "concentration": meta.get("concentration"),
        "matched_words": meta.get("matched_words"),
        "query_words": meta.get("query_words"),
        "docs_hit": meta.get("docs_hit"),
    }
    for ws in _websockets[:]:
        try:
            import anyio
            anyio.from_thread.run(ws.send_json, data)
        except Exception:
            _websockets.remove(ws)


pipeline.on_route(broadcast)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model Router dashboard starting...")
    seed_path = os.getenv("SOT_SEED_CSV", "data/alexa_qa/train.csv")
    seed_sot_from_csv(
        seed_path,
        max_rows=int(os.getenv("SOT_SEED_MAX", "500")),
    )
    yield
    logger.info("Model Router dashboard stopping.")


app = FastAPI(title="Model Router Dashboard", lifespan=lifespan)


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    return DASHBOARD_HTML


@app.post("/route")
async def route_query(req: RouteRequest) -> dict:
    result = pipeline.route(req)
    return _response_to_dict(result)


@app.get("/history")
async def get_history(limit: int = Query(50, ge=1, le=200)):
    return [_response_to_dict(r) for r in pipeline.get_history(limit=limit)]


@app.get("/stats")
async def get_stats():
    return pipeline.get_stats()


@app.get("/models")
async def get_models():
    return {
        "fast": [
            {"name": m.name, "id": m.openrouter_id, "params": m.total_params_b}
            for m in FAST_MODELS
        ],
        "thinking": [
            {
                "name": m.name, "id": m.openrouter_id,
                "params": m.total_params_b, "active": m.active_params_b,
            }
            for m in THINKING_MODELS
        ],
        "deep": [
            {
                "name": m.name, "id": m.openrouter_id,
                "params": m.total_params_b, "active": m.active_params_b,
            }
            for m in DEEP_MODELS
        ],
        "count": len(ALL_MODELS),
    }


# =============================================================================
# WebSocket — Real-time route updates
# =============================================================================


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _websockets.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _websockets.remove(ws)
    except Exception:
        if ws in _websockets:
            _websockets.remove(ws)


def _response_to_dict(r) -> dict:
    meta = r.classification.metadata or {}
    return {
        "query": r.query,
        "response": r.response[:500],
        "complexity": r.classification.complexity,
        "task": r.classification.task_label,
        "confidence": r.classification.confidence,
        "method": r.classification.method,
        "tier": r.routing.tier,
        "model_id": r.routing.model_id,
        "model_name": r.routing.model_name,
        "reason": r.routing.reason,
        "tokens_in": r.generation.tokens_in,
        "tokens_out": r.generation.tokens_out,
        "latency_ms": r.generation.latency_ms,
        "escalated": r.generation.cascade_escalated,
        "error": r.generation.error,
        "timestamp": r.generation.timestamp.isoformat(),
        # Heatmap
        "match_density": meta.get("match_density"),
        "coverage": meta.get("coverage"),
        "concentration": meta.get("concentration"),
        "matched_words": meta.get("matched_words"),
        "query_words": meta.get("query_words"),
        "docs_hit": meta.get("docs_hit"),
    }


def run_dashboard():
  import uvicorn

  port = _find_available_port(config.dashboard_host, config.dashboard_port)
  if port != config.dashboard_port:
    logger.warning(
      "Dashboard port %s is in use, falling back to %s",
      config.dashboard_port,
      port,
    )

  uvicorn.run(
    app,
    host=config.dashboard_host,
    port=port,
    log_level=config.log_level.lower(),
  )


def _find_available_port(host: str, start_port: int, max_tries: int = 20) -> int:
  """Return the first available port at or above start_port."""
  for port in range(start_port, start_port + max_tries):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      try:
        sock.bind((host, port))
      except OSError:
        continue
      return port
  raise RuntimeError(
    f"No free port found between {start_port} and {start_port + max_tries - 1}"
  )


# =============================================================================
# Dashboard HTML
# =============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Router — Cost-Optimized LLM Routing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Unbounded:wght@500;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #020202;
  --surface: #1A1515;
  --card: #2A2222;
  --card-hover: #332A2A;
  --border: #443B3B;
  --border-hover: #5C4F4F;
  --border-strong: #6B5C5C;
  --text: #E8E8E8;
  --text-secondary: #988686;
  --text-muted: #857979;
  --accent: #5C4F4F;
  --accent-hover: #6B5C5C;
  --fast-bg: #1A2A1A;
  --fast-text: #8AB88A;
  --thinking-bg: #2A2A1A;
  --thinking-text: #B8A86A;
  --deep-bg: #2A1A2A;
  --deep-text: #A888B8;
  --error: #6B2A2A;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  font-size: 0.9375rem;
  line-height: 1.6;
  min-height: 100vh;
}
/* ─── TYPOGRAPHY ─── */
h1, h2, h3 { font-family: 'Unbounded', sans-serif; font-weight: 700; letter-spacing: -0.01em; line-height: 1.1; }
h1 { font-size: 2.5rem; }
h2 { font-size: 1.25rem; letter-spacing: 0; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 0.8125rem; }
.label { font-family: 'Inter', sans-serif; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
/* ─── LAYOUT ─── */
.wrapper { max-width: 1280px; margin: 0 auto; padding: 48px 32px; }
.section { margin-bottom: 32px; }
/* ─── HERO ─── */
.hero { border-bottom: 1px solid var(--border); padding-bottom: 32px; margin-bottom: 40px; }
.hero h1 { margin-bottom: 8px; }
.hero .tagline { color: var(--text-secondary); font-size: 1.125rem; max-width: 540px; }
.hero-meta { display: flex; gap: 24px; margin-top: 20px; }
.hero-meta > div { display: flex; align-items: center; gap: 8px; }
.status-dot { width: 8px; height: 8px; background: var(--error); display: inline-block; }
.status-dot.connected { background: #7ACC7A; }
/* ─── CARDS ─── */
.card { background: var(--card); padding: 24px; border: 1px solid var(--border); }
.card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
/* ─── STATS ─── */
.stats-grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px; background: var(--border); border: 1px solid var(--border); }
.stat { background: var(--card); padding: 20px 16px; text-align: center; }
.stat-value { font-family: 'Unbounded', sans-serif; font-size: 1.75rem; font-weight: 700; }
.stat-label { color: var(--text-secondary); font-size: 0.75rem; letter-spacing: 0.08em; margin-top: 4px; }
/* ─── MAIN GRID ─── */
.main-grid { display: grid; grid-template-columns: 1fr 320px; gap: 24px; }
/* ─── FEED ─── */
.feed { max-height: 540px; overflow-y: auto; }
.feed-empty { color: var(--text-secondary); text-align: center; padding: 48px 0; font-size: 0.875rem; }
.entry {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 12px 16px; margin-bottom: 4px;
  border-left: 3px solid var(--border);
  transition: background 0.15s; cursor: default;
}
.entry:hover { background: var(--card-hover); }
.entry-fast { border-left-color: var(--fast-text); }
.entry-thinking { border-left-color: var(--thinking-text); }
.entry-deep { border-left-color: var(--deep-text); }
.entry-body { flex: 1; min-width: 0; }
.entry-tags { display: flex; gap: 6px; align-items: center; margin-bottom: 4px; flex-wrap: wrap; }
.entry-query { font-size: 0.875rem; font-weight: 500; margin-bottom: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.entry-response { font-size: 0.8125rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.entry-meta { text-align: right; font-size: 0.75rem; color: var(--text-secondary); white-space: nowrap; }
.entry-heatmap { margin-top: 6px; display: flex; gap: 4px; flex-wrap: wrap; }
.heat-dot { width: 8px; height: 14px; background: var(--border-strong); display: inline-block; }
.heat-hit { background: var(--text-muted); }
.heat-word { font-size: 0.625rem; color: var(--text-secondary); font-family: 'JetBrains Mono', monospace; margin-right: 2px; }
/* ─── BADGES ─── */
.badge { display: inline-block; padding: 2px 10px; font-size: 0.6875rem; font-weight: 600; letter-spacing: 0.04em; }
.badge-fast { background: var(--fast-bg); color: var(--fast-text); }
.badge-thinking { background: var(--thinking-bg); color: var(--thinking-text); }
.badge-deep { background: var(--deep-bg); color: var(--deep-text); }
.badge-close { background: #1A1A1A; color: #5A8A5A; }
.badge-moderate { background: #1A1A1A; color: #8A7A4A; }
.badge-distant { background: #1A1A1A; color: #7A5A8A; }
/* ─── QUERY BAR ─── */
.query-bar { display: flex; gap: 0; border: 1px solid var(--border); }
.query-bar input {
  flex: 1; background: var(--bg); border: none; padding: 16px;
  color: var(--text); font-family: 'Inter', sans-serif; font-size: 0.9375rem;
  outline: none;
}
.query-bar input::placeholder { color: var(--text-muted); }
.query-bar select {
  background: var(--card); border: none; border-left: 1px solid var(--border);
  padding: 0 16px; color: var(--text-secondary); font-size: 0.8125rem;
  font-family: 'Inter', sans-serif; outline: none; cursor: pointer;
}
.query-bar button {
  background: var(--accent); border: none; padding: 0 24px;
  color: var(--text); font-family: 'Inter', sans-serif;
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; cursor: pointer; transition: background 0.15s;
}
.query-bar button:hover { background: var(--accent-hover); }
/* ─── MODEL POOL ─── */
.model-tier { margin-bottom: 16px; }
.model-tier:last-child { margin-bottom: 0; }
.model-tier-header { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 8px; }
.model-item { font-size: 0.75rem; color: var(--text-secondary); padding: 2px 0; }
/* ─── FILTERS ─── */
.filters { display: flex; gap: 8px; }
.filters select {
  background: var(--bg); border: 1px solid var(--border); padding: 4px 10px;
  color: var(--text-secondary); font-size: 0.75rem; font-family: 'Inter', sans-serif;
  outline: none; cursor: pointer;
}
/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; }
/* ─── ANIMATIONS ─── */
@keyframes fadeIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
.entry { animation: fadeIn 0.25s ease-out; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
.loading { animation: pulse 1.5s ease-in-out infinite; }
</style>
</head>
<body>
<div class="wrapper">

  <!-- ═══════════ HERO ═══════════ -->
  <div class="hero">
    <div style="display:flex;align-items:flex-start;justify-content:space-between">
      <div>
        <h1>Model Router</h1>
        <p class="tagline">Cost-optimized LLM routing — classify complexity, pick the cheapest capable model from the OpenRouter free pool.</p>
      </div>
      <div class="hero-meta">
        <div>
          <span class="status-dot" id="status-dot"></span>
          <span class="mono" id="status-text" style="font-size:0.75rem;color:var(--text-secondary)">disconnected</span>
        </div>
        <div class="mono" id="hero-model-count" style="font-size:0.75rem;color:var(--text-secondary)">— models</div>
      </div>
    </div>
  </div>

  <!-- ═══════════ STATS ═══════════ -->
  <div class="stats-grid section" id="stats-grid">
    <div class="stat"><div class="stat-value" id="stat-total">0</div><div class="stat-label">Routes</div></div>
    <div class="stat"><div class="stat-value" style="color:var(--fast-text)" id="stat-fast">0</div><div class="stat-label">Fast</div></div>
    <div class="stat"><div class="stat-value" style="color:var(--thinking-text)" id="stat-thinking">0</div><div class="stat-label">Thinking</div></div>
    <div class="stat"><div class="stat-value" style="color:var(--deep-text)" id="stat-deep">0</div><div class="stat-label">Deep</div></div>
    <div class="stat"><div class="stat-value" style="color:#D4A84B" id="stat-escalated">0</div><div class="stat-label">Escalated</div></div>
    <div class="stat"><div class="stat-value" id="stat-tokens">0</div><div class="stat-label">Tokens Used</div></div>
  </div>

  <!-- ═══════════ MAIN GRID ═══════════ -->
  <div class="main-grid section">

    <!-- LEFT: Feed -->
    <div class="card">
      <div class="card-header">
        <h2>Live Feed</h2>
        <div class="filters">
          <select id="tier-filter" onchange="applyFilter()">
            <option value="all">All Tiers</option>
            <option value="fast">Fast</option>
            <option value="thinking">Thinking</option>
            <option value="deep">Deep</option>
          </select>
          <select id="complexity-filter" onchange="applyFilter()">
            <option value="all">All Complexity</option>
            <option value="close">Close</option>
            <option value="moderate">Moderate</option>
            <option value="distant">Distant</option>
          </select>
        </div>
      </div>
      <div class="feed" id="feed">
        <div class="feed-empty">Awaiting queries — type below to begin</div>
      </div>
    </div>

    <!-- RIGHT: Model Pool -->
    <div class="card">
      <div class="card-header">
        <h2>Model Pool</h2>
      </div>
      <div id="model-pool">
        <div class="mono" style="color:var(--text-secondary)">loading…</div>
      </div>
    </div>

  </div>

  <!-- ═══════════ QUERY BAR ═══════════ -->
  <div class="query-bar" style="margin-bottom:16px">
    <input id="query-input" type="text" placeholder="Type a query to route…" onkeydown="if(event.key==='Enter') testRoute()">
    <select id="force-tier">
      <option value="">Auto (cost-optimized)</option>
      <option value="fast">Fast tier</option>
      <option value="thinking">Thinking tier</option>
      <option value="deep">Deep tier</option>
    </select>
    <button onclick="testRoute()">Route</button>
  </div>

  <div style="color:var(--text-secondary);font-size:0.75rem;text-align:center;padding:16px 0 0;border-top:1px solid var(--border)">
    <span class="mono">Model Router v0.2.0</span> · <span class="mono">OpenRouter free pool</span>
  </div>
</div>

<script>
let ws = null;
let routes = [];

// ─── Connection ──────────────────────────────────
function connect() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + window.location.host + '/ws');
  ws.onopen = () => {
    document.getElementById('status-dot').className = 'status-dot connected';
    document.getElementById('status-text').textContent = 'connected';
  };
  ws.onclose = () => {
    document.getElementById('status-dot').className = 'status-dot';
    document.getElementById('status-text').textContent = 'disconnected';
    setTimeout(connect, 3000);
  };
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.type === 'route') addRouteEntry(d);
  };
}

// ─── Route Entry ──────────────────────────────────
function addRouteEntry(d) {
  routes.unshift(d);
  if (routes.length > 500) routes.pop();
  applyFilter();
  updateStats();
}

// ─── Heatmap Mini ────────────────────────────────
function heatmapHTML(d) {
  if (!d.matched_words && d.matched_words !== 0) return '';
  const qw = d.query_words || 0;
  const mw = d.matched_words || 0;
  const dh = d.docs_hit || 0;
  // Word-match bar: filled dots for matched, empty for unmatched
  let html = '<span class="heat-word">words</span>';
  for (let i = 0; i < qw; i++) {
    html += '<span class="heat-dot' + (i < mw ? ' heat-hit' : '') + '"></span>';
  }
  html += ' <span class="heat-word">docs</span><span class="heat-dot' + (dh > 0 ? ' heat-hit' : '') + '" style="width:' + Math.min(dh * 8, 40) + 'px"></span>';
  if (d.match_density) {
    html += ' <span class="heat-word">' + (d.match_density * 100).toFixed(0) + '%</span>';
  }
  return html;
}

// ─── Filter + Render ────────────────────────────
function applyFilter() {
  const tierF = document.getElementById('tier-filter').value;
  const compF = document.getElementById('complexity-filter').value;
  const filtered = routes.filter(r => {
    if (tierF !== 'all' && r.tier !== tierF) return false;
    if (compF !== 'all' && r.complexity !== compF) return false;
    return true;
  });
  const feed = document.getElementById('feed');
  if (filtered.length === 0) {
    feed.innerHTML = '<div class="feed-empty">No matching routes</div>';
    return;
  }
  feed.innerHTML = filtered.slice(0, 80).map(r => {
    const tC = 'entry-' + (r.tier || 'deep');
    const bC = 'badge-' + (r.tier || 'deep');
    const cC = 'badge-' + (r.complexity || 'distant');
    const tok = (r.tokens_in || 0) + (r.tokens_out || 0);
    const err = r.error ? ' <span style="color:var(--error);font-size:0.75rem">✗</span>' : '';
    const esc = r.escalated ? ' <span style="color:#D4A84B;font-size:0.75rem">↗</span>' : '';
    const hm = heatmapHTML(r);
    return '<div class="entry ' + tC + '">' +
      '<div class="entry-body">' +
        '<div class="entry-tags">' +
          '<span class="badge ' + bC + '">' + (r.tier || '?') + '</span>' +
          '<span class="badge ' + cC + '">' + (r.complexity || '?') + '</span>' +
          '<span class="mono" style="font-size:0.6875rem;color:var(--text-secondary)">' + escapeHtml(r.model_name || '').slice(0, 24) + '</span>' +
          esc + err +
        '</div>' +
        '<div class="entry-query">' + escapeHtml(r.query || '') + '</div>' +
        '<div class="entry-response">' + escapeHtml((r.response_preview || '').slice(0, 120)) + '</div>' +
        (hm ? '<div class="entry-heatmap">' + hm + '</div>' : '') +
      '</div>' +
      '<div class="entry-meta"><div>' + tok + ' tok</div><div>' + (r.latency_ms || 0) + 'ms</div></div>' +
    '</div>';
  }).join('');
}

// ─── Stats ──────────────────────────────────────
function updateStats() {
  const total = routes.length;
  const fast = routes.filter(r => r.tier === 'fast').length;
  const thinking = routes.filter(r => r.tier === 'thinking').length;
  const deep = routes.filter(r => r.tier === 'deep').length;
  const esc = routes.filter(r => r.escalated).length;
  const toks = routes.reduce((s, r) => s + (r.tokens_in || 0) + (r.tokens_out || 0), 0);
  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-fast').textContent = fast;
  document.getElementById('stat-thinking').textContent = thinking;
  document.getElementById('stat-deep').textContent = deep;
  document.getElementById('stat-escalated').textContent = esc;
  document.getElementById('stat-tokens').textContent = toks > 999 ? (toks/1000).toFixed(1)+'K' : toks;
}

// ─── Route Query ────────────────────────────────
async function testRoute() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;
  const force = document.getElementById('force-tier').value;
  try {
    const resp = await fetch('/route', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ query, force_tier: force || null }),
    });
    const d = await resp.json();
    addRouteEntry({
      type: 'route',
      query: d.query,
      response_preview: (d.response || '').substring(0, 200),
      complexity: d.complexity,
      task: d.task,
      confidence: d.confidence,
      method: d.method,
      tier: d.tier,
      model_id: d.model_id,
      model_name: d.model_name,
      reason: d.reason,
      tokens_in: d.tokens_in,
      tokens_out: d.tokens_out,
      latency_ms: d.latency_ms,
      escalated: d.escalated,
      error: d.error,
      match_density: d.match_density,
      coverage: d.coverage,
      concentration: d.concentration,
      matched_words: d.matched_words,
      query_words: d.query_words,
      docs_hit: d.docs_hit,
    });
  } catch (e) {
    console.error('Route failed:', e);
  }
}

// ─── Model Pool ─────────────────────────────────
async function loadModelPool() {
  try {
    const resp = await fetch('/models');
    const data = await resp.json();
    document.getElementById('hero-model-count').textContent = data.count + ' models';
    const pool = document.getElementById('model-pool');
    let html = '';
    const tiers = [
      { key: 'fast', label: 'Fast', color: 'var(--fast-text)' },
      { key: 'thinking', label: 'Thinking', color: 'var(--thinking-text)' },
      { key: 'deep', label: 'Deep', color: 'var(--deep-text)' },
    ];
    for (const t of tiers) {
      const models = data[t.key] || [];
      html += '<div class="model-tier">' +
        '<div class="model-tier-header" style="color:' + t.color + '">' + t.label + ' (' + models.length + ')</div>';
      for (const m of models.slice(0, 6)) {
        html += '<div class="model-item mono">' + escapeHtml(m.name) + '</div>';
      }
      if (models.length > 6) {
        html += '<div class="model-item" style="color:#555">+ ' + (models.length - 6) + ' more</div>';
      }
      html += '</div>';
    }
    pool.innerHTML = html;
  } catch (e) {
    document.getElementById('model-pool').innerHTML = '<div class="mono" style="color:var(--error)">Failed to load</div>';
  }
}

function escapeHtml(t) {
  if (!t) return '';
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── Init ───────────────────────────────────────
connect();
loadModelPool();
</script>
</body>
</html>
"""


if __name__ == "__main__":
  run_dashboard()
