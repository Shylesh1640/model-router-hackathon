# Data Pipeline Specification

## Overview
Scrapes, cleans, and seeds datasets into the Source of Truth. Sources: Alexa QA (HuggingFace) and Cisco Umbrella Top 1M domains.

**Implementation:** `src/data/scraper.py`

---

## Requirements

### Requirement: Alexa QA Dataset
The system SHALL download the Alexa QA dataset from HuggingFace.

#### Scenario: Download CSVs
- **WHEN** `download_alexa_qa()` is called
- **THEN** download `train.csv`, `validation.csv`, `test.csv` from HuggingFace
- **AND** merge into a single CSV
- **AND** return path to merged file
- **AND** cache files to avoid re-download

#### Scenario: Download failure
- **WHEN** download fails (network, 404, timeout)
- **THEN** create a sample QA dataset for development
- **AND** log the failure
- **AND** continue without crashing

---

### Requirement: Top 1M Domains
The system SHALL download the Cisco Umbrella Top 1M domain list.

#### Scenario: Download domain list
- **WHEN** `download_top_1m()` is called
- **THEN** download from primary source (Cisco Umbrella S3)
- **AND** fall back to alternative sources on failure
- **AND** return path to CSV with `rank,domain` columns

#### Scenario: Load domains
- **WHEN** `load_domains(path, max_domains)` is called
- **THEN** parse CSV and return up to `max_domains` domain names
- **AND** handle encoding errors gracefully

---

### Requirement: Domain Content Crawling
The system SHALL extract readable text from web domains.

#### Scenario: Successful crawl
- **WHEN** `crawl_domain(domain)` succeeds
- **THEN** return extracted text content (>100 chars)
- **AND** remove script/style tags, normalize whitespace
- **AND** keep first 50 substantial lines
- **AND** cache content to disk

#### Scenario: Crawl failure
- **WHEN** domain is unreachable or returns error
- **THEN** return None gracefully
- **AND** continue to next domain

---

### Requirement: SOT Seeding
The system SHALL seed scraped data into the Source of Truth.

#### Scenario: Seed from Alexa QA
- **WHEN** `seed_sot_from_alexa_qa(max_entries)` is called
- **THEN** read CSV entries as Q:... A:... pairs
- **AND** add up to `max_entries` documents to SOT
- **AND** log count of seeded documents

#### Scenario: Seed from domains
- **WHEN** `seed_sot_from_domains(max_domains)` is called
- **THEN** download domain list, crawl top N domains
- **AND** add content to SOT with `web:{domain}` source tag

---

## Success Criteria

- [ ] Alexa QA CSVs download and merge correctly
- [ ] Sample data created when download fails
- [ ] Top 1M domain list downloads and parses
- [ ] Domain content extraction removes HTML
- [ ] SOT seeding completes without errors
- [ ] All modules handle network failures gracefully
