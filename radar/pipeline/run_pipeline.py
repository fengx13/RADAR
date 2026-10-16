#!/usr/bin/env python3
"""Longitudinal Sepsis ED Agent engineering orchestrator.

The runner accepts the same externally assembled contract in dry-run and real
model modes. Real EHR extraction remains an external dependency.
"""
from __future__ import annotations
from radar import paths

import argparse
import copy
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

from radar.agents import run_agent1
from radar.agents import run_agent2
from radar.agents import run_agent3
from radar.agents import run_agent4
from radar.common.integration_schema import (
    FUTURE_OUTCOME_KEYS,
    integrated_case_skeleton,
    strip_inference_forbidden,
)
from radar.pipeline.longitudinal_integration import (
    assemble_t1_state,
    build_agent4_payload,
    build_treatment_state,
    build_workup_state,
    determine_t1_gate,
    reconcile_treatment,
)


HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
def safe_print(value):
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding) + "\n")


def pipeline_config(
    dry_run: bool = True,
    model: str = "synthetic-stub",
    base_url: str | None = None,
    temperature: float = 0.0,
    seed: int = 7,
):
    return SimpleNamespace(
        dry_run=dry_run,
        model=model,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
    )


def _args(config, decision_time: str | None = None):
    values = {
        "dry_run": config.dry_run,
        "model": config.model,
        "temperature": config.temperature,
        "seed": config.seed,
    }
    if decision_time is not None:
        values["decision_time"] = decision_time
    return SimpleNamespace(**values)


def _evaluation_only(case: dict) -> dict | None:
    hidden: dict = {}
    for key in FUTURE_OUTCOME_KEYS:
        if key in case and key != "evaluation_only":
            hidden[key] = case[key]
    if isinstance(case.get("evaluation_only"), dict):
        hidden.update(case["evaluation_only"])
    return hidden or None


def _stop(envelope: dict, status: str, stage: str, reason: str) -> dict:
    envelope["status"] = status
    envelope["stopped_at"] = stage
    envelope["stop_reason"] = reason
    return envelope


def run_longitudinal_case(case: dict, client=None, config=None) -> dict:
    """Run one externally assembled case and preserve each component trace."""
    config = config or pipeline_config()
    case_id = case.get("case_id")
    if not case_id:
        return {
            "case_id": None,
            "status": "invalid",
            "stopped_at": "input",
            "stop_reason": "missing case_id",
        }

    envelope = integrated_case_skeleton(case_id)
    envelope["stay_id"] = case.get("stay_id")
    envelope["status"] = "running"
    envelope["evaluation_only"] = _evaluation_only(case)
    envelope["t0_state"] = strip_inference_forbidden(case.get("t0_state") or {})
    mocks = (case.get("engineering_mocks") or {}) if config.dry_run else {}

    a1_t0_case = {
        "case_id": case_id,
        "input": envelope["t0_state"],
        "mock_agent1_answers": mocks.get("agent1_t0") or {},
    }
    envelope["agent1_t0"] = run_agent1.run_case(a1_t0_case, client, _args(config, "T0"))
    if envelope["agent1_t0"].get("result", {}).get("tier") == "abstained":
        return _stop(envelope, "upstream_abstain", "agent1_t0", "Agent 1 T0 abstained")

    a2_case = {"case_id": case_id, "input": envelope["t0_state"]}
    envelope["agent2"] = run_agent2.run_case(a2_case, client, _args(config))
    if envelope["agent2"].get("status") != "complete":
        return _stop(envelope, "upstream_abstain", "agent2", "Agent 2 did not complete")

    envelope["workup_state"] = build_workup_state(
        envelope["agent2"], case.get("observed_workup") or []
    )
    envelope["t1_gate"] = determine_t1_gate(envelope["workup_state"])
    if envelope["t1_gate"]["status"] == "not_ready":
        return _stop(envelope, "t1_not_ready", "t1_gate", envelope["t1_gate"]["reason"])
    if envelope["t1_gate"]["status"] != "ready":
        return _stop(envelope, "invalid", "t1_gate", envelope["t1_gate"]["reason"])

    t1_time = envelope["t1_gate"]["t1_time"]
    envelope["t1_state"] = assemble_t1_state(
        case.get("longitudinal_events") or {}, envelope["workup_state"], t1_time
    )

    a1_t1_case = {
        "case_id": case_id,
        "input": {
            "T1_state": envelope["t1_state"],
            "agent1_t0_context": envelope["agent1_t0"].get("result"),
        },
        "mock_agent1_answers": mocks.get("agent1_t1") or {},
    }
    envelope["agent1_t1"] = run_agent1.run_case(a1_t1_case, client, _args(config, "T1"))
    if envelope["agent1_t1"].get("result", {}).get("tier") == "abstained":
        return _stop(envelope, "upstream_abstain", "agent1_t1", "Agent 1 T1 abstained")

    a3_case = {
        "case_id": case_id,
        "input": {
            "age": envelope["t0_state"].get("age"),
            "agent1_result": envelope["agent1_t1"].get("result"),
            "T1_state": envelope["t1_state"],
        },
        "dry_run_answers": mocks.get("agent3") or {},
    }
    envelope["agent3"] = run_agent3.run_case(a3_case, client, _args(config))
    if envelope["agent3"].get("status") != "complete":
        return _stop(envelope, "upstream_abstain", "agent3", "Agent 3 did not complete")

    envelope["treatment_state"] = build_treatment_state(
        case.get("observed_treatment") or {}, t1_time
    )
    if envelope["treatment_state"].get("status") != "valid":
        return _stop(envelope, "invalid", "treatment_state", "invalid observed treatment state")
    envelope["reconciliation"] = reconcile_treatment(
        envelope["agent3"], envelope["treatment_state"]
    )

    agent4_payload = build_agent4_payload(
        case_id=case_id,
        t1_time=t1_time,
        t1_state=envelope["t1_state"],
        agent1_t1_run=envelope["agent1_t1"],
        agent2_run=envelope["agent2"],
        workup_state=envelope["workup_state"],
        agent3_run=envelope["agent3"],
        treatment_state=envelope["treatment_state"],
        reconciliation=envelope["reconciliation"],
    )
    if config.dry_run:
        agent4_payload["mock_llm_answers"] = mocks.get("agent4") or {}
    envelope["agent4_input"] = strip_inference_forbidden(agent4_payload)
    envelope["agent4"] = run_agent4.run_case(agent4_payload, client, _args(config))
    envelope["status"] = "complete"
    envelope["stopped_at"] = None
    envelope["stop_reason"] = None
    return envelope


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_cases(path: pathlib.Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        defaults = data.get("defaults") or {}
        return [_deep_merge(defaults, case) for case in data["cases"]]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("pipeline input must be a case object, list, or {'cases': [...]}")


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infile",
        default=str(paths.DATA / "pipeline_synthetic_cases.json"),
        help="externally assembled longitudinal case JSON",
    )
    parser.add_argument("--out", default=str(paths.DATA / "pipeline_run.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible local or hosted endpoint")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pilot-mode", action="store_true", help="mark output for qualitative trajectory review")
    return parser


def load_env():
    env_path = paths.ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main(argv=None):
    parser = legacy_create_parser()
    args = parser.parse_args(argv)
    config = legacy_pipeline_config(
        dry_run=args.dry_run,
        model="synthetic-stub" if args.dry_run else args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        seed=args.seed,
    )
    client = None
    if not args.dry_run:
        load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set (put it in .env in the repository root or the environment)")
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    started = time.time()
    cases = legacy_load_cases(pathlib.Path(args.infile))
    results = [run_legacy_single_t1_case(case, client=client, config=config) for case in cases]
    output = {
        "meta": {
            "pipeline": "sepsis_ed_longitudinal",
            "version": "0.2",
            "pipeline_mode": "legacy_single_t1",
            "dry_run": args.dry_run,
            "model": config.model,
            "base_url": args.base_url,
            "pilot_mode": args.pilot_mode,
            "information_cutoff": "single T1 cutoff for Agent 1 reassessment, Agent 3, and Agent 4",
            "clinical_validation": False,
            "n": len(results),
            "seconds": round(time.time() - started, 3),
        },
        "results": results,
    }
    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    safe_print(f"pipeline statuses: {counts}")
    safe_print(f"-> {destination}")


# Preserve the previous implementation under explicit legacy names, then make
# the sequential RADAR vNext engine the default public/CLI path.
legacy_pipeline_config = pipeline_config
run_legacy_single_t1_case = run_longitudinal_case
legacy_run_longitudinal_case = run_longitudinal_case
legacy_load_cases = load_cases
legacy_create_parser = create_parser
legacy_main = main

from radar.pipeline import run_pipeline_vnext as _vnext  # noqa: E402

pipeline_config = _vnext.pipeline_config
run_longitudinal_case = _vnext.run_longitudinal_case
load_cases = _vnext.load_cases
create_parser = _vnext.create_parser


def _requested_pipeline_mode(argv):
    for index, value in enumerate(argv):
        if value == "--pipeline-mode" and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith("--pipeline-mode="):
            return value.split("=", 1)[1]
    return "sequential"


def _without_pipeline_mode(argv):
    cleaned = []
    skip_next = False
    for value in argv:
        if skip_next:
            skip_next = False
            continue
        if value == "--pipeline-mode":
            skip_next = True
            continue
        if value.startswith("--pipeline-mode="):
            continue
        cleaned.append(value)
    return cleaned


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = _requested_pipeline_mode(argv)
    if mode == "legacy_single_t1":
        return legacy_main(_without_pipeline_mode(argv))
    if mode == "legacy_cohort_two_time":
        from radar.pipeline.run_full_pipeline import run_legacy_cohort_main
        return run_legacy_cohort_main(_without_pipeline_mode(argv))
    return _vnext.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
