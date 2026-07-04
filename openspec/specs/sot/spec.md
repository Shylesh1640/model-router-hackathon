# Source of Truth Specification

## Overview
The SOT is the core of the system. It stores the knowledge base as embedding vectors and returns distance metrics used for all routing decisions.

**Implementation:** `src/sot/source_of_truth.py`

---

## Requirements

### Requirement: Vector Storage
The system SHALL store documents as embedding vectors for similarity search.

#### Scenario: Add document
- **WHEN** `add_document(content, source)` is called
- **THEN** document is embedded and stored
- **AND** returns a unique document ID
- **AND** document is queryable immediately

#### Scenario: Add multiple documents
- **WHEN** `add_documents([{content, source}])` is called
- **THEN** all documents are embedded and stored
- **AND** returns list of document IDs

#### Scenario: Count documents
- **WHEN** `count()` is called
- **THEN** returns the number of stored documents

#### Scenario: Clear documents
- **WHEN** `clear()` is called
- **THEN** all documents are removed
- **AND** SOT returns to empty state

---

### Requirement: Query with Distance Metric
The system SHALL return cosine distance between query and nearest source document.

#### Scenario: Query with matches
- **WHEN** `query(text)` is called
- **THEN** returns `SourceQueryResult` with:
  - `min_distance`: 0.0-1.0 (0=identical, 1=no match)
  - `matches`: up to N nearest documents
  - `distances`: cosine distances per match
  - `is_off_topic`: true if min_distance > threshold (0.75)

#### Scenario: Empty SOT
- **WHEN** `query(text)` is called on empty SOT
- **THEN** returns `min_distance=1.0`
- **AND** `is_off_topic=true`
- **AND** `off_topic_reason` describes the empty state

#### Scenario: No close matches
- **WHEN** query has no documents within threshold
- **THEN** `is_off_topic=true`
- **AND** `off_topic_reason` includes distance and threshold

---

### Requirement: Distance-to-Complexity Mapping
The system SHALL map cosine distance to a complexity level.

#### Scenario: Close match
- **WHEN** distance < 0.30
- **THEN** `complexity_from_distance()` returns `"close"`
- **AND** answer can be grounded directly in source

#### Scenario: Moderate match
- **WHEN** distance 0.30-0.60
- **THEN** `complexity_from_distance()` returns `"moderate"`
- **AND** answer may need web search augmentation

#### Scenario: Distant match
- **WHEN** distance > 0.60
- **THEN** `complexity_from_distance()` returns `"distant"`
- **AND** answer needs deep reasoning chain

---

### Requirement: Zero-Dependency Operation
The system SHALL work without sentence-transformers or ChromaDB installed.

#### Scenario: No sentence-transformers
- **WHEN** sentence-transformers is not installed
- **THEN** character n-gram hashing is used for embeddings
- **AND** all SOT operations continue to function
- **AND** a warning is logged

#### Scenario: No ChromaDB
- **WHEN** ChromaDB is not installed
- **THEN** in-memory numpy-backed vector store is used
- **AND** all SOT operations continue to function
- **AND** state is not persisted across restarts

---

## Success Criteria

- [ ] Add documents returns unique IDs
- [ ] Query returns correct nearest neighbors
- [ ] Distance correctly maps to complexity levels
- [ ] Off-topic detection works at threshold
- [ ] Zero-dependency fallbacks work without errors
- [ ] Performance: <10ms per query with in-memory store
