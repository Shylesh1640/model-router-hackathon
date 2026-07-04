"""Data scraper — fetches Alexa QA dataset and Top 1M domain content."""

import csv
import gzip
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from src.sot.source_of_truth import get_sot

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ALEXA_QA_DIR = DATA_DIR / "alexa_qa"
DOMAINS_DIR = DATA_DIR / "domains"
CRAWL_DIR = DATA_DIR / "crawled"


# =============================================================================
# ALEXA QA DATASET (HuggingFace)
# =============================================================================

ALEXA_QA_URL = "https://huggingface.co/datasets/theblackcat102/alexa-qa/resolve/main/"

ALEXA_QA_FILES = [
    "train.csv",
    "validation.csv",
    "test.csv",
]


def download_alexa_qa(force: bool = False) -> Path:
    """Download the Alexa QA dataset from HuggingFace.

    Downloads train.csv, validation.csv, test.csv.
    Returns path to the merged CSV.
    """
    ALEXA_QA_DIR.mkdir(parents=True, exist_ok=True)

    merged_path = ALEXA_QA_DIR / "alexa_qa.csv"
    if merged_path.exists() and not force:
        logger.info(f"Alexa QA already cached: {merged_path}")
        return merged_path

    import csv as csv_module

    all_rows = []
    headers = None

    for fname in ALEXA_QA_FILES:
        url = ALEXA_QA_URL + fname
        local = ALEXA_QA_DIR / fname

        if local.exists() and not force:
            logger.info(f"Using cached {local}")
        else:
            logger.info(f"Downloading {url}...")
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with open(local, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Downloaded {len(resp.content)} bytes to {local}")
            except Exception as e:
                logger.warning(f"Failed to download {fname}: {e}")
                continue

        # Read CSV
        try:
            with open(local, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv_module.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        headers = row
                        continue
                    all_rows.append(row)
            logger.info(f"  {len(all_rows)} rows from {fname}")
        except Exception as e:
            logger.warning(f"Failed to read {fname}: {e}")

    if not all_rows:
        logger.warning("No data downloaded, creating sample")
        _create_sample_qa(merged_path)
        return merged_path

    # Write merged CSV
    with open(merged_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.writer(f)
        if headers:
            writer.writerow(headers)
        writer.writerows(all_rows)

    logger.info(f"Merged {len(all_rows)} rows into {merged_path}")
    return merged_path


def _create_sample_qa(path: Path):
    """Create a small sample Q&A dataset for development."""
    import random

    sample_qa = [
        ("what is python", "Python is a high-level programming language."),
        ("how to sort a list in python", "Use list.sort() or sorted()."),
        ("what is machine learning", "ML is a subset of AI."),
        ("what is an API", "API stands for Application Programming Interface."),
        ("how does http work", "HTTP is a request-response protocol."),
        ("what is docker", "Docker is a containerization platform."),
        ("what is kubernetes", "K8s is an orchestration platform."),
        ("how to write a file in python", "Use open() with 'w' mode."),
        ("what is git", "Git is a version control system."),
        ("what is a database", "A database stores structured data."),
    ]

    # Try to save as parquet, fallback to JSON
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            "question": [q for q, a in sample_qa],
            "answer": [a for q, a in sample_qa],
        })
        pq.write_table(table, path)
    except ImportError:
        # JSON fallback
        with open(path.with_suffix(".json"), "w") as f:
            json.dump([
                {"question": q, "answer": a} for q, a in sample_qa
            ], f, indent=2)
        logger.info(f"Created sample JSON at {path.with_suffix('.json')}")

    logger.info(f"Created sample dataset with {len(sample_qa)} entries")


# =============================================================================
# ALEXA TOP 1M DOMAINS
# =============================================================================

TOP_1M_SOURCES = [
    # Cisco Umbrella Top 1M (active)
    "https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip",
    # Fallbacks
    "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip",
]

DOMAINS_TO_SCRAPE = 500  # How many domains to actually crawl content from


def download_top_1m(force: bool = False) -> Optional[Path]:
    """Download Alexa Top 1M domains list.

    Returns path to CSV or None if all sources fail.
    """
    DOMAINS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DOMAINS_DIR / "top-1m.csv"

    if local_path.exists() and not force:
        logger.info(f"Top 1M already cached: {local_path} ({local_path.stat().st_size} bytes)")
        return local_path

    for source_url in TOP_1M_SOURCES:
        try:
            logger.info(f"Downloading from {source_url}...")
            resp = requests.get(source_url, timeout=60)
            resp.raise_for_status()

            # Handle gzipped content
            content = resp.content
            if source_url.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
                    with zf.open(csv_name) as f:
                        content = f.read()

            with open(local_path, "wb") as f:
                f.write(content)

            logger.info(f"Downloaded top 1M domains: {len(content)} bytes")
            return local_path

        except Exception as e:
            logger.warning(f"Failed to download from {source_url}: {e}")
            continue

    logger.error("All top 1M download sources failed")
    return None


def load_domains(path: Path, max_domains: int = 10000) -> list[str]:
    """Load domain names from the CSV.

    CSV format: rank,domain
    Returns list of domains (up to max_domains).
    """
    domains = []
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    domain = row[1].strip()
                    if domain:
                        domains.append(domain)
                        if len(domains) >= max_domains:
                            break
    except Exception as e:
        logger.warning(f"Failed to parse domains: {e}")

    logger.info(f"Loaded {len(domains)} domains from {path}")
    return domains


# =============================================================================
# DOMAIN CONTENT CRAWLER
# =============================================================================

def crawl_domain(domain: str, timeout: int = 10) -> Optional[str]:
    """Crawl a single domain and extract text content.

    Tries http:// and https://, returns extracted text or None.
    """
    for scheme in ["https", "http"]:
        url = f"{scheme}://{domain}"
        try:
            resp = requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (compatible; ModelRouter/1.0)",
            })
            if resp.status_code == 200:
                text = _extract_text(resp.text)
                if text and len(text) > 100:
                    return text
        except Exception:
            continue
    return None


def _extract_text(html: str) -> str:
    """Extract readable text from HTML."""
    import re
    # Remove scripts, styles, tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Only keep lines with content
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 40]
    return "\n".join(lines[:50])  # First 50 substantial lines


def crawl_top_domains(domains: list[str], max_crawl: int = 100) -> list[dict]:
    """Crawl content from the top N domains.

    Returns list of {domain, content} dicts.
    """
    CRAWL_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, domain in enumerate(domains[:max_crawl]):
        logger.info(f"Crawling [{i+1}/{max_crawl}] {domain}...")
        content = crawl_domain(domain)
        if content:
            results.append({"domain": domain, "content": content})
            # Cache individual
            safe_name = domain.replace(".", "_").replace("/", "_")
            with open(CRAWL_DIR / f"{safe_name}.txt", "w") as f:
                f.write(content)

        # Be polite
        if i % 10 == 0:
            time.sleep(0.5)

    logger.info(f"Crawled {len(results)}/{max_crawl} domains successfully")
    return results


# =============================================================================
# SEED INTO SOURCE OF TRUTH
# =============================================================================

def seed_sot_from_alexa_qa(max_entries: int = 1000):
    """Load Alexa QA entries into the Source of Truth."""
    import csv

    path = download_alexa_qa()
    if not path:
        logger.error("No Alexa QA data available")
        return

    sot = get_sot()
    count = 0

    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_entries:
                    break
                question = row.get("question", row.get("Question", ""))
                answer = row.get("answer", row.get("Answer", ""))
                if question and answer:
                    content = f"Q: {question}\nA: {answer}"
                    sot.add_document(content, source="alexa-qa")
                    count += 1
    except Exception as e:
        logger.warning(f"Failed to read CSV: {e}")
        # Fallback: try JSON
        json_path = path.with_suffix(".json")
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            for entry in data[:max_entries]:
                content = f"Q: {entry.get('question', '')}\nA: {entry.get('answer', '')}"
                if content.strip():
                    sot.add_document(content, source="alexa-qa")
                    count += 1

    logger.info(f"Seeded {count} Alexa QA entries into Source of Truth")


def seed_sot_from_domains(max_domains: int = 100):
    """Crawl top domains and add content to Source of Truth."""
    path = download_top_1m()
    if not path:
        logger.error("No domain list available")
        return

    domains = load_domains(path, max_domains=5000)
    results = crawl_top_domains(domains, max_crawl=max_domains)

    sot = get_sot()
    for r in results:
        content = f"Domain: {r['domain']}\n{r['content'][:1000]}"
        sot.add_document(content, source=f"web:{r['domain']}")

    logger.info(f"Seeded {len(results)} domains into Source of Truth")


def run_full_seed(alexa_entries: int = 1000, domain_crawls: int = 100):
    """Full data pipeline: download, crawl, seed SOT."""
    logger.info("=== Starting full data seed ===")

    logger.info("Step 1: Seeding Alexa QA...")
    seed_sot_from_alexa_qa(max_entries=alexa_entries)

    logger.info("Step 2: Crawling and seeding domains...")
    seed_sot_from_domains(max_domains=domain_crawls)

    sot = get_sot()
    logger.info(f"=== Seed complete: {sot.count()} documents in SOT ===")
