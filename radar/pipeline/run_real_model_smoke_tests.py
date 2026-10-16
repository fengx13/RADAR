#!/usr/bin/env python3
"""API-ready A2/A3/A4 smoke-test harness; dry-run never needs credentials."""
from __future__ import annotations
from radar import paths

import argparse
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

from radar.agents import run_agent2
from radar.agents import run_agent3
from radar.agents import run_agent4
from radar.agents import run_agent1
from radar.common.run_provenance import model_runtime_metadata, new_run_metadata


HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
def safe_print(value):
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding) + "\n")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", choices=("A2", "A3", "A4"), default=["A2", "A3", "A4"])
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=str(paths.DATA / "real_model_smoke_test.json"))
    return parser


def _load_first(path: pathlib.Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"no fixture cases in {path}")
    return cases[0]


def load_smoke_cases() -> dict:
    a2 = _load_first(paths.DATA / "agent2_synthetic_cases.json")
    a2 = dict(a2)
    a2["agent1_context"] = {
        "tier": "possible_sepsis",
        "shock_concern": False,
        "decision_time": "T0",
        "provenance": "synthetic_fixture",
    }
    return {
        "A2": a2,
        "A3": _load_first(paths.DATA / "agent3_synthetic_cases.json"),
        "A4": _load_first(paths.DATA / "agent4_dryrun_cases.json"),
    }


def _client(args):
    if args.dry_run:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set")
    from openai import OpenAI
    return OpenAI(base_url=args.base_url) if args.base_url else OpenAI()


def run_smoke(args) -> dict:
    run_metadata = new_run_metadata("a2-a3-a4-smoke", ROOT)
    client = _client(args)
    cases = load_smoke_cases()
    runner_args = SimpleNamespace(
        dry_run=args.dry_run,
        model="synthetic-stub" if args.dry_run else args.model,
        temperature=args.temperature,
        seed=args.seed,
    )
    runners = {
        "A2": run_agent2.run_case,
        "A3": run_agent3.run_case,
        "A4": run_agent4.run_case,
    }
    results = {}
    for agent in args.agents:
        results[agent] = runners[agent](cases[agent], client, runner_args)
    modules = {"A2": run_agent2, "A3": run_agent3, "A4": run_agent4}
    return {
        "meta": {
            **run_metadata,
            "purpose": "real-model invocation and structured traversal smoke test",
            "dry_run": args.dry_run,
            "model": runner_args.model,
            "model_requested": args.model,
            "model_runtime": {
                agent: model_runtime_metadata(modules[agent])
                for agent in args.agents
            },
            "base_url": args.base_url,
            "agents": args.agents,
            "sampling": {
                "temperature_requested": args.temperature,
                "seed_requested": args.seed,
                "parameter_support": {
                    "A2": dict(run_agent2.CAPS),
                    "A3": dict(run_agent3.CAPS),
                    "A4": {"temperature": True, "seed": False},
                },
            },
            "graph_versions": {
                "A1_repository_baseline": run_agent1.GRAPH["version"],
                "A2": run_agent2.GRAPH["version"],
                "A3": run_agent3.GRAPH["version"],
                "A4_checkpoint": run_agent4.GRAPH["version"],
            },
            "pipeline_mode": "standalone_agent_smoke",
            "replay_mode": None,
            "credentials_source": "OPENAI_API_KEY environment variable",
            "credentials_stored": False,
            "clinical_validation": False,
        },
        "results": results,
    }


def main():
    args = create_parser().parse_args()
    started = time.time()
    output = run_smoke(args)
    output["meta"]["seconds"] = round(time.time() - started, 3)
    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    safe_print(f"smoke agents: {list(output['results'])}")
    safe_print(f"-> {destination}")


if __name__ == "__main__":
    main()
