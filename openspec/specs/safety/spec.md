# Safety Guard Specification

## Overview
Two-stage filter that blocks harmful content and rebukes off-topic queries with a gentle message.

**Implementation:** `src/sot/safety.py`

---

## Requirements

### Requirement: Harmful Content Detection
The system SHALL detect and block harmful/dangerous queries before processing.

#### Scenario: Block harmful query
- **WHEN** a query matches a high-severity harmful pattern
- **THEN** `SafetyResult.safe=false`
- **AND** `SafetyResult.flagged=true`
- **AND** `SafetyResult.category="harmful"`
- **AND** a clear rejection message is returned
- **AND** the query does not reach the SOT or generation stages

#### Scenario: Allow safe query
- **WHEN** a query does not match any harmful pattern
- **THEN** `SafetyResult.safe=true`
- **AND** processing continues to next stage

#### Harmful patterns include:
- Weapons/explosives/drugs manufacturing instructions
- Hacking/cracking/exploit tutorials
- Identity theft, fraud, criminal activity
- Child exploitation content
- Self-harm or suicide methods
- Malware/ransomware/virus generation
- Phishing/scam templates

---

### Requirement: Off-Topic Detection with Gentle Rebuke
The system SHALL detect queries unrelated to the source of truth domain.

#### Scenario: Off-topic query with high distance
- **WHEN** source distance > 0.75 AND query matches off-topic patterns
- **THEN** `SafetyResult.flagged=true`
- **AND** `SafetyResult.category="off_topic"`
- **AND** a gentle redirect message is returned:
  - References the domain name
  - Invites rephrasing in context
  - Does not block (safe=true)

#### Scenario: Off-topic query with very high distance
- **WHEN** source distance > 0.85 AND query length > 3 words
- **THEN** flagged as off-topic
- **AND** gentle rebuke returned

#### Scenario: Valid query with high distance
- **WHEN** source distance > 0.75 but query has domain-related keywords
- **THEN** NOT flagged as off-topic
- **AND** allowed through to deep reasoning tier

---

### Requirement: Configurable Domain Name
The system SHALL use the configured domain name in rebuke messages.

#### Scenario: Domain in rebuke
- **WHEN** an off-topic rebuke is generated
- **THEN** the message includes the configured domain name
- **AND** the message is specific to the knowledge base context

---

## Success Criteria

- [ ] Harmful patterns match correctly
- [ ] Safe queries pass through unchanged
- [ ] Off-topic queries get gentle redirect (not block)
- [ ] Domain name appears in rebuke messages
- [ ] No false positives for technically-related queries
