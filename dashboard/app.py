"""Model Router Dashboard — FastAPI + WebSocket real-time UI.

Standalone web app for testing and monitoring the routing pipeline.
"""

import csv
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse

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
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            q = row.get("question", row.get("Question", ""))
            a = row.get("answer", row.get("Answer", ""))
            if q and a:
                sot.add_document(f"Q: {q}\nA: {a}", source="alexa-qa")
                count += 1
    logger.info("Seeded %s docs into SOT from %s", count, csv_path)
    return count


def broadcast(route_response):
    """Push route decision to all connected dashboard clients."""
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
    }


def run_dashboard():
    import uvicorn
    uvicorn.run(
        app,
        host=config.dashboard_host,
        port=config.dashboard_port,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    run_dashboard()


# =============================================================================
# Dashboard HTML
# =============================================================================

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Model Router — Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body { background: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
  .tier-fast { border-left: 4px solid #22c55e; }
  .tier-thinking { border-left: 4px solid #f59e0b; }
  .tier-deep { border-left: 4px solid #a855f7; }
  .card { background: #1e293b; border-radius: 12px; padding: 1rem; }
  .badge { padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
  .badge-fast { background: #166534; color: #86efac; }
  .badge-thinking { background: #92400e; color: #fcd34d; }
  .badge-deep { background: #581c87; color: #d8b4fe; }
  .mono { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; }
  .fade-in { animation: fadeIn 0.3s ease-in; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
  .entry { transition: all 0.2s; }
  .entry:hover { background: #334155; }
</style>
</head>
<body class="p-6">
  <div class="max-w-7xl mx-auto">

    <!-- Header -->
    <div class="flex items-center justify-between mb-8">
      <div>
        <h1 class="text-3xl font-bold text-white">Model Router</h1>
        <p class="text-slate-400 text-sm">Cost-optimized LLM routing · OpenRouter free pool</p>
      </div>
      <div class="flex gap-4 items-center">
        <div id="connection-status" class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-red-500" id="status-dot"></span>
          <span class="text-sm text-slate-400" id="status-text">Disconnected</span>
        </div>
        <button onclick="testRoute()" class="bg-indigo-600 hover:bg-indigo-700 px-4 py-2 rounded-lg text-sm font-medium transition">Test Route</button>
      </div>
    </div>

    <!-- Stats Row -->
    <div class="grid grid-cols-5 gap-4 mb-8" id="stats-row">
      <div class="card text-center">
        <div class="text-2xl font-bold" id="stat-total">0</div>
        <div class="text-xs text-slate-400">Total Routes</div>
      </div>
      <div class="card text-center">
        <div class="text-2xl font-bold text-green-400" id="stat-fast">0</div>
        <div class="text-xs text-slate-400">Fast (cheap)</div>
      </div>
      <div class="card text-center">
        <div class="text-2xl font-bold text-yellow-400" id="stat-thinking">0</div>
        <div class="text-xs text-slate-400">Thinking</div>
      </div>
      <div class="card text-center">
        <div class="text-2xl font-bold text-purple-400" id="stat-deep">0</div>
        <div class="text-xs text-slate-400">Deep</div>
      </div>
      <div class="card text-center">
        <div class="text-2xl font-bold text-orange-400" id="stat-escalated">0</div>
        <div class="text-xs text-slate-400">Escalations</div>
      </div>
    </div>

    <!-- Main: History + Model Pool -->
    <div class="grid grid-cols-3 gap-6">
      <!-- History Feed -->
      <div class="col-span-2 card">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold">Live Feed</h2>
          <div class="flex gap-2">
            <select id="tier-filter" onchange="applyFilter()" class="bg-slate-700 text-sm rounded px-2 py-1 border border-slate-600">
              <option value="all">All Tiers</option>
              <option value="fast">Fast</option>
              <option value="thinking">Thinking</option>
              <option value="deep">Deep</option>
            </select>
            <select id="complexity-filter" onchange="applyFilter()" class="bg-slate-700 text-sm rounded px-2 py-1 border border-slate-600">
              <option value="all">All Complexity</option>
              <option value="close">Close</option>
              <option value="moderate">Moderate</option>
              <option value="distant">Distant</option>
            </select>
          </div>
        </div>
        <div id="feed" class="space-y-2 max-h-[600px] overflow-y-auto">
          <div class="text-slate-500 text-sm text-center py-8">Waiting for routes...</div>
        </div>
      </div>

      <!-- Model Pool -->
      <div class="card">
        <h2 class="text-lg font-semibold mb-4">Model Pool</h2>
        <div id="model-pool" class="space-y-3">
          <div class="text-slate-500 text-sm">Loading...</div>
        </div>
      </div>
    </div>

    <!-- Query Input -->
    <div class="mt-6 card">
      <div class="flex gap-4">
        <input id="query-input" type="text" placeholder="Type a query to route..."
          class="flex-1 bg-slate-700 rounded-lg px-4 py-3 text-white placeholder-slate-400 border border-slate-600 focus:outline-none focus:border-indigo-500"
          onkeydown="if(event.key==='Enter') testRoute()">
        <select id="force-tier" class="bg-slate-700 rounded-lg px-3 border border-slate-600 text-sm">
          <option value="">Auto (cost-optimized)</option>
          <option value="fast">Force Fast</option>
          <option value="thinking">Force Thinking</option>
          <option value="deep">Force Deep</option>
        </select>
        <button onclick="testRoute()"
          class="bg-indigo-600 hover:bg-indigo-700 px-6 py-3 rounded-lg font-medium transition">
          Route
        </button>
      </div>
    </div>

  </div>

<script>
let ws = null;
let routes = [];

function connect() {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${window.location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('status-dot').className = 'w-2 h-2 rounded-full bg-green-500';
    document.getElementById('status-text').textContent = 'Connected';
  };

  ws.onclose = () => {
    document.getElementById('status-dot').className = 'w-2 h-2 rounded-full bg-red-500';
    document.getElementById('status-text').textContent = 'Disconnected';
    setTimeout(connect, 3000);
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'route') {
      addRouteEntry(data);
    }
  };
}

function addRouteEntry(data) {
  routes.unshift(data);
  if (routes.length > 200) routes.pop();
  applyFilter();
  updateStats();
}

function applyFilter() {
  const tierFilter = document.getElementById('tier-filter').value;
  const complexityFilter = document.getElementById('complexity-filter').value;

  const filtered = routes.filter(r => {
    if (tierFilter !== 'all' && r.tier !== tierFilter) return false;
    if (complexityFilter !== 'all' && r.complexity !== complexityFilter) return false;
    return true;
  });

  const feed = document.getElementById('feed');
  if (filtered.length === 0) {
    feed.innerHTML = '<div class="text-slate-500 text-sm text-center py-8">No matching routes</div>';
    return;
  }

  feed.innerHTML = filtered.slice(0, 50).map(r => {
    const tierClass = `tier-${r.tier}`;
    const badgeClass = `badge-${r.tier}`;
    const escaped = r.escalated ? ' <span class="text-orange-400 text-xs">↗ escalated</span>' : '';
    const error = r.error ? ` <span class="text-red-400 text-xs">✗ ${r.error}</span>` : '';

    return `
      <div class="entry card fade-in ${tierClass}">
        <div class="flex items-start justify-between gap-4">
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2 mb-1">
              <span class="badge ${badgeClass}">${r.tier}</span>
              <span class="badge bg-slate-700 text-slate-300">${r.complexity}</span>
              <span class="text-xs text-slate-500">${r.model_name}</span>
              ${escaped}${error}
            </div>
            <div class="text-sm font-medium truncate">${escapeHtml(r.query)}</div>
            <div class="text-xs text-slate-400 mt-1 mono truncate">${escapeHtml(r.response_preview || '')}</div>
          </div>
          <div class="text-right text-xs text-slate-500 whitespace-nowrap">
            <div>${r.tokens_in + r.tokens_out} tok</div>
            <div>${r.latency_ms}ms</div>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function updateStats() {
  const total = routes.length;
  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-fast').textContent = routes.filter(r => r.tier === 'fast').length;
  document.getElementById('stat-thinking').textContent = routes.filter(r => r.tier === 'thinking').length;
  document.getElementById('stat-deep').textContent = routes.filter(r => r.tier === 'deep').length;
  document.getElementById('stat-escalated').textContent = routes.filter(r => r.escalated).length;
}

async function testRoute() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) return;

  const forceTier = document.getElementById('force-tier').value;

  try {
    const resp = await fetch('/route', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        query: query,
        force_tier: forceTier || null,
      }),
    });
    const data = await resp.json();
    addRouteEntry({
      type: 'route',
      query: data.query,
      response_preview: (data.response || '').substring(0, 200),
      complexity: data.complexity,
      task: data.task,
      confidence: data.confidence,
      method: data.method,
      tier: data.tier,
      model_id: data.model_id,
      model_name: data.model_name,
      reason: data.reason,
      tokens_in: data.tokens_in,
      tokens_out: data.tokens_out,
      latency_ms: data.latency_ms,
      escalated: data.escalated,
      error: data.error,
    });
  } catch (e) {
    console.error('Route failed:', e);
  }
}

async function loadModelPool() {
  try {
    const resp = await fetch('/models');
    const data = await resp.json();
    const pool = document.getElementById('model-pool');
    pool.innerHTML = `
      <div class="mb-4">
        <div class="text-sm font-medium text-green-400">Fast (${data.fast.length})</div>
        ${data.fast.map(m => `<div class="text-xs text-slate-400 mono">${m.name}</div>`).join('')}
      </div>
      <div class="mb-4">
        <div class="text-sm font-medium text-yellow-400">Thinking (${data.thinking.length})</div>
        ${data.thinking.map(m => `<div class="text-xs text-slate-400 mono">${m.name}</div>`).join('')}
      </div>
      <div>
        <div class="text-sm font-medium text-purple-400">Deep (${data.deep.length})</div>
        ${data.deep.map(m => `<div class="text-xs text-slate-400 mono">${m.name}</div>`).join('')}
      </div>
      <div class="mt-4 pt-4 border-t border-slate-700">
        <div class="text-xs text-slate-500">Total: ${data.count} models</div>
      </div>
    `;
  } catch (e) {
    document.getElementById('model-pool').innerHTML = '<div class="text-red-400 text-sm">Failed to load models</div>';
  }
}

function escapeHtml(text) {
  if (!text) return '';
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Init
connect();
loadModelPool();
</script>
</body>
</html>
"""
