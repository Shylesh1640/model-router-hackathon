#!/usr/bin/env python3
"""CLI entry point for the Model Router."""

import argparse
import json
import logging
import sys

from src.config import get_config
from src.models import RouteRequest, Complexity
from src.pipeline import RoutingPipeline


def main():
    parser = argparse.ArgumentParser(description="Model Router — Cost-optimized LLM routing")
    parser.add_argument("query", nargs="?", help="Query to route (omit for interactive mode)")
    parser.add_argument("--tier", choices=["fast", "thinking", "deep"], help="Force a specific tier")
    parser.add_argument("--no-cascade", action="store_true", help="Disable cascade")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--dashboard", action="store_true", help="Start dashboard server")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if args.dashboard:
        from src.dashboard.app import run_dashboard
        run_dashboard()
        return

    config = get_config()
    pipeline = RoutingPipeline(config)

    if args.query:
        _route_single(pipeline, args.query, args.tier, not args.no_cascade, args.json)
    else:
        _interactive(pipeline, args.json)


def _route_single(pipeline, query, force_tier, cascade, json_output):
    req = RouteRequest(query=query, force_tier=force_tier, cascade=cascade)
    result = pipeline.route(req)

    if json_output:
        print(json.dumps({
            "query": result.query,
            "response": result.response,
            "complexity": result.classification.complexity,
            "task": result.classification.task_label,
            "confidence": result.classification.confidence,
            "tier": result.routing.tier,
            "model": result.routing.model_name,
            "model_id": result.routing.model_id,
            "tokens": result.generation.tokens_in + result.generation.tokens_out,
            "latency_ms": result.generation.latency_ms,
            "escalated": result.generation.cascade_escalated,
            "error": result.generation.error,
        }, indent=2))
    else:
        tier_badge = {
            "fast": "\033[32mFAST\033[0m",
            "thinking": "\033[33mTHINK\033[0m",
            "deep": "\033[35mDEEP\033[0m",
        }.get(result.routing.tier, result.routing.tier)

        print(f"\n{'='*60}")
        print(f"  {tier_badge}  {result.routing.model_name}")
        print(f"  Complexity: {result.classification.complexity}  |  "
              f"Confidence: {result.classification.confidence:.2f}")
        print(f"  Reason: {result.routing.reason}")
        if result.generation.cascade_escalated:
            print(f"  \033[33m↗ Escalated: {result.generation.cascade_from_tier} → {result.generation.cascade_to_tier}\033[0m")
        print(f"  Tokens: {result.generation.tokens_in + result.generation.tokens_out}  |  "
              f"Latency: {result.generation.latency_ms}ms")
        if result.generation.error:
            print(f"  \033[31mError: {result.generation.error}\033[0m")
        print(f"{'='*60}")
        print(f"\n{result.response}\n")


def _interactive(pipeline, json_output):
    print("\033[36mModel Router — interactive mode. Type queries or /stats /models /quit\033[0m")
    while True:
        try:
            query = input("\n\033[36mroute>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query == "/quit":
            break
        if query == "/stats":
            stats = pipeline.get_stats()
            print(json.dumps(stats, indent=2))
            continue
        if query == "/models":
            from src.constants import FAST_MODELS, THINKING_MODELS, DEEP_MODELS
            for tier, models in [("fast", FAST_MODELS), ("thinking", THINKING_MODELS), ("deep", DEEP_MODELS)]:
                print(f"\n\033[3{'2' if tier == 'fast' else '3' if tier == 'thinking' else '5'}m{tier.upper()}\033[0m")
                for m in models:
                    print(f"  {m.name:40s} {m.openrouter_id}")
            print()
            continue
        _route_single(pipeline, query, None, True, json_output)


if __name__ == "__main__":
    main()
