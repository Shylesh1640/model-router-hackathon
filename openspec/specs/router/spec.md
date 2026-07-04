# Routing Engine Specification

## Overview
Maps query complexity to model tier. Selects the most cost-effective model that can handle the task.

**Implementation:** `src/router/engine.py`

---

## Requirements

### Requirement: Tier Selection
The system SHALL select the appropriate model tier based on complexity.

#### Scenario: Close complexity
- **WHEN** complexity is `"close"`
- **THEN** route to `"grounded"` tier
- **AND** select model from fast-tier pool (e.g. Liquid 1.2B)
- **AND** answer is sourced directly from SOT

#### Scenario: Moderate complexity
- **WHEN** complexity is `"moderate"`
- **THEN** route to `"web_search"` tier
- **AND** select model from thinking-tier pool (e.g. GPT-OSS-20b)
- **AND** answer augments SOT with web search results

#### Scenario: Distant complexity
- **WHEN** complexity is `"distant"`
- **THEN** route to `"deep_reasoning"` tier
- **AND** select model from deep-tier pool (e.g. Llama 3.3 70B)
- **AND** answer uses full reasoning chain

---

### Requirement: Benchmark-Aware Model Selection
The system SHALL prefer models with higher benchmark scores for the task type.

#### Scenario: Code task
- **WHEN** the task type is `"code"`
- **THEN** prefer models with higher HumanEval / SWE-bench scores
- **AND** ranking within tier uses benchmark data

#### Scenario: Reasoning task
- **WHEN** the task type is `"reasoning"`
- **THEN** prefer models with higher MMLU-Pro scores
- **AND** ranking within tier uses benchmark data

#### Scenario: No benchmark data
- **WHEN** no benchmark scores are available for a model
- **THEN** fall back to parameter count for ranking
- **AND** include a note in the routing decision

---

### Requirement: Confidence Override
The system SHALL bump tier when classification confidence is low.

#### Scenario: Low confidence override
- **WHEN** classification confidence < 0.4
- **THEN** bump one tier up (close → moderate, moderate → distant)
- **AND** note the override in the routing reason

#### Scenario: Already at highest tier
- **WHEN** already at deep_reasoning tier
- **THEN** no bump applied

---

### Requirement: Cascade (Self-Verification)
The system SHALL optionally verify its own response and escalate if unsure.

#### Scenario: Fast tier self-verification
- **WHEN** grounded tier generates a response
- **THEN** run self-verification (cheap model checks its answer)
- **AND** if verification confidence < threshold, escalate to web_search tier
- **AND** max 1 hop per query

#### Scenario: Cascade skipped for high confidence
- **WHEN** classification confidence > 0.95
- **THEN** skip self-verification entirely
- **AND** return response from initial tier

---

## Success Criteria

- [ ] Correct tier selected for each complexity level
- [ ] Benchmark scores influence model selection
- [ ] Confidence override works at all tiers
- [ ] Cascade fires when verification fails
- [ ] Cascade does not fire when confidence is high
- [ ] All routing decisions include a clear reason string
