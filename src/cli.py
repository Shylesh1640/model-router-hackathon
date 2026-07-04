#!/usr/bin/env python3
"""CLI for the Source-of-Truth chatbot."""

import argparse
import json
import logging
import sys

from src.config import get_config
from src.models import RouteRequest
from src.pipeline import ChatbotPipeline
from src.sot.source_of_truth import get_sot


def main():
    parser = argparse.ArgumentParser(description="Source-of-Truth Chatbot")
    parser.add_argument("query", nargs="?", help="Query (omit for interactive mode)")
    parser.add_argument("--domain", default="this knowledge base", help="Domain name for rebukes")
    parser.add_argument("--seed", help="Seed source of truth from a text file")
    parser.add_argument("--add", help="Add a document to source of truth")
    parser.add_argument("--list-sources", action="store_true", help="List source document count")
    parser.add_argument("--clear", action="store_true", help="Clear all source documents")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    config = get_config()
    sot = get_sot()
    pipeline = ChatbotPipeline(config, domain=args.domain)

    # Handle source management commands
    if args.seed:
        with open(args.seed) as f:
            content = f.read()
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        ids = sot.add_documents([{"content": l, "source": args.seed} for l in lines])
        print(f"Seeded {len(ids)} documents from {args.seed}")
        return

    if args.add:
        doc_id = sot.add_document(args.add, source="cli")
        print(f"Added document: {doc_id}")
        print(f"Total sources: {sot.count()}")
        return

    if args.list_sources:
        print(f"Source of Truth: {sot.count()} documents")
        return

    if args.clear:
        sot.clear()
        print("Source of Truth cleared.")
        return

    if args.dashboard:
        print("Dashboard: python -m src.dashboard.app")
        return

    # Interactive or single query
    if args.query:
        _route(pipeline, args.query, args.json)
    else:
        _interactive(pipeline, args.json)


def _route(pipeline, query, json_output=False):
    req = RouteRequest(query=query)
    result = pipeline.route(req)

    if json_output:
        out = {
            "query": result.query,
            "response": result.response,
            "complexity": result.classification.complexity,
            "task": result.classification.task_label,
            "source_distance": round(result.classification.source_distance, 3),
            "confidence": result.classification.confidence,
            "tier": result.routing.tier,
            "model": result.routing.model_name,
            "rebuked": result.rebuked,
            "web_search": result.generation.web_search_used,
            "deep_reasoning": result.generation.deep_reasoning_used,
        }
        print(json.dumps(out, indent=2))
    else:
        dist = result.classification.source_distance
        tier_badge = {
            "grounded": "\033[32mGROUNDED\033[0m",
            "web_search": "\033[33mWEB+SEARCH\033[0m",
            "deep_reasoning": "\033[35mDEEP\033[0m",
            "rebuked": "\033[31mREBUKED\033[0m",
            "blocked": "\033[31mBLOCKED\033[0m",
        }.get(result.routing.tier, result.routing.tier)

        print(f"\n{'='*60}")
        print(f"  {tier_badge}  distance={dist:.3f}  "
              f"confidence={result.classification.confidence:.2f}")
        print(f"  model={result.routing.model_name}  |  tier={result.routing.tier}")
        if result.rebuked:
            print(f"  \033[33m⛔ {result.safety.reason}\033[0m")
        if result.generation.web_search_used:
            print(f"  \033[36m🌐 Web search consulted\033[0m")
        if result.generation.deep_reasoning_used:
            print(f"  \033[35m🧠 Deep reasoning chain\033[0m")
        print(f"{'='*60}")
        print(f"\n{result.response}\n")


def _interactive(pipeline, json_output=False):
    sot = get_sot()
    print(f"\033[36mSource-of-Truth Chatbot\033[0m")
    print(f"  \033[33m{sot.count()} documents in source of truth\033[0m")
    print(f"  Commands: /add <text>  /seed <file>  /count  /quit\n")

    while True:
        try:
            query = input("\033[36mchat>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query == "/quit":
            break
        if query.startswith("/add "):
            text = query[5:]
            sot.add_document(text, source="interactive")
            print(f"  \033[32mAdded. Total: {sot.count()}\033[0m")
            continue
        if query == "/count":
            print(f"  \033[33m{sot.count()} documents\033[0m")
            continue
        if query.startswith("/seed "):
            path = query[6:]
            try:
                with open(path) as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                ids = sot.add_documents([{"content": l, "source": path} for l in lines])
                print(f"  \033[32mSeeded {len(ids)} from {path}\033[0m")
            except Exception as e:
                print(f"  \033[31mError: {e}\033[0m")
            continue
        _route(pipeline, query, json_output)


if __name__ == "__main__":
    main()
