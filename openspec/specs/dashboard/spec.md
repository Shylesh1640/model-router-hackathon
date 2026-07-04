# Dashboard Specification

## Overview
Real-time web dashboard showing routing decisions, model pool status, and aggregate statistics. FastAPI backend with WebSocket push.

**Implementation:** `src/dashboard/app.py`

---

## Requirements

### Requirement: Live Route Feed
The dashboard SHALL display routing decisions in real-time.

#### Scenario: WebSocket push
- **WHEN** a route completes
- **THEN** route data is pushed to all connected WebSocket clients
- **AND** data includes: query, tier, model, complexity, distance, tokens, latency
- **AND** connection status indicator updates

#### Scenario: History display
- **WHEN** the dashboard loads
- **THEN** show last N routing entries in a scrollable feed
- **AND** each entry shows tier badge, model name, query preview, token count

#### Scenario: Filter by tier
- **WHEN** user selects a tier filter
- **THEN** only entries matching that tier are shown
- **AND** filter applies to both new and existing entries

---

### Requirement: Statistics Display
The dashboard SHALL show aggregate route statistics.

#### Scenario: Stats cards
- **WHEN** dashboard is loaded or updated
- **THEN** display: total routes, fast/thinking/deep counts, escalation count
- **AND** stats update in real-time as new routes arrive

---

### Requirement: Model Pool Display
The dashboard SHALL show the available model pool by tier.

#### Scenario: Model list
- **WHEN** dashboard loads
- **THEN** fetch and display models grouped by tier (fast, thinking, deep)
- **AND** show model names and total count

---

### Requirement: Test Route Input
The dashboard SHALL allow testing queries directly from the UI.

#### Scenario: Test route
- **WHEN** user enters a query and clicks Route
- **THEN** POST to `/route` endpoint
- **AND** result appears in the live feed
- **AND** tier forcing dropdown is available

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML page |
| `/route` | POST | Route a query |
| `/history` | GET | Recent route history |
| `/stats` | GET | Aggregate statistics |
| `/models` | GET | Model pool by tier |
| `/ws` | WS | Real-time WebSocket feed |

---

## Success Criteria

- [ ] Dashboard serves at configured port
- [ ] WebSocket pushes every route
- [ ] Tier filter works correctly
- [ ] Stats update in real-time
- [ ] Model pool displays correctly
- [ ] Test route input works
- [ ] Connection indicator shows status
