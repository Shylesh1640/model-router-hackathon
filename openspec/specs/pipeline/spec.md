# Pipeline Specification

## Overview
Orchestrates the full query flow: Safety → SOT → Distance → Route → Generate → Cascade.

**Implementation:** `src/pipeline.py`

---

## Requirements

### Requirement: Full Query Flow
The system SHALL process queries through all pipeline stages in order.

#### Scenario: Happy path
- **WHEN** a safe, in-domain query is received
- **THEN** execute in order:
  1. Safety check passes
  2. SOT lookup returns distance
  3. Distance determines complexity
  4. Router selects tier and model
  5. Generation produces response
  6. Optional cascade verification

#### Scenario: Blocked by safety
- **WHEN** a query is flagged as harmful
- **THEN** return immediately with rejection message
- **AND** no SOT lookup, routing, or generation occurs

#### Scenario: Rebuked for off-topic
- **WHEN** a query is flagged as off-topic
- **THEN** return immediately with gentle redirect
- **AND** no generation cost incurred

---

### Requirement: Web Search Augmentation
The system SHALL optionally augment moderate-distance queries with web search.

#### Scenario: Web search triggered
- **WHEN** complexity is `"moderate"` and SOT context is insufficient
- **THEN** run web search via SearXNG
- **AND** include search results in the LLM prompt
- **AND** mark `web_search_used=true` on the generation result

#### Scenario: Web search unavailable
- **WHEN** no search backend is configured
- **THEN** skip web search gracefully
- **AND** respond using SOT context only
- **AND** note the missing backend in response

---

### Requirement: Deep Reasoning
The system SHALL use multi-step reasoning for distant queries.

#### Scenario: Deep reasoning triggered
- **WHEN** complexity is `"distant"`
- **THEN** run deep reasoning chain
- **AND** include reasoning steps as context in the prompt
- **AND** mark `deep_reasoning_used=true` on the generation result

---

### Requirement: Response Recording
The system SHALL record every response for dashboard and stats.

#### Scenario: Record response
- **WHEN** any route completes
- **THEN** store the full `RouteResponse` in `self.history`
- **AND** keep last 1000 responses
- **AND** notify WebSocket listeners

#### Scenario: Get stats
- **WHEN** `get_stats()` is called
- **THEN** return:
  - Total routes, tier distribution
  - Number of rebukes, web searches, deep reasoning calls
  - Average tokens and latency

---

## Success Criteria

- [ ] Full pipeline processes queries end-to-end
- [ ] Safety block returns immediately
- [ ] Off-topic rebuke returns without generation
- [ ] Web search results included in prompts
- [ ] Deep reasoning chain included in prompts
- [ ] History limited to 1000 entries
- [ ] Stats calculated correctly
