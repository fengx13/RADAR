#!/usr/bin/env python3
"""Generic insufficient/indeterminate response evaluator.

The specific prompt intervention reported by Yana was not present in the
available repository/local project artifacts when this harness was added.
"""
from __future__ import annotations
from radar import paths

import argparse
import json
import pathlib
import sys
from collections import defaultdict


HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
INSUFFICIENT_ANSWERS = {"insufficient_evidence", "indeterminate", "insufficient_information"}
SPECIFIC_INTERVENTION_FOUND = False


def safe_print(value):
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding) + "\n")


def _load_records(path: pathlib.Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return data["results"]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("input must be an object, list, or results/cases envelope")


def _agent_name(run: dict, fallback: str) -> str:
    return str((run.get("result") or {}).get("agent") or fallback)


def _agent_runs(record: dict):
    if isinstance(record.get("trace"), list):
        yield _agent_name(record, "standalone"), record
        return
    for key in ("agent1_t0", "agent2", "agent4_t0"):
        run = record.get(key)
        if isinstance(run, dict):
            yield _agent_name(run, key), run
    for checkpoint in record.get("checkpoints") or []:
        if checkpoint.get("checkpoint_status") != "processed":
            continue
        for key in ("agent1_reassessment", "agent3", "agent4"):
            run = checkpoint.get(key)
            if isinstance(run, dict):
                yield _agent_name(run, key), run
    # Legacy single-T1 envelope fallback.
    if not record.get("checkpoints"):
        for key in ("agent1_t1", "agent3", "agent4"):
            run = record.get(key)
            if isinstance(run, dict):
                yield _agent_name(run, key), run


def _new_counter():
    return {
        "model_node_responses": 0,
        "model_attempts": 0,
        "insufficient_or_indeterminate": 0,
        "retry_nodes": 0,
        "successful_repairs": 0,
        "abstentions": 0,
    }


def _finalize(counter: dict) -> dict:
    result = dict(counter)
    denominator = counter["model_node_responses"]
    result["insufficient_or_indeterminate_rate"] = (
        None if denominator == 0 else counter["insufficient_or_indeterminate"] / denominator
    )
    result["retry_frequency"] = None if denominator == 0 else counter["retry_nodes"] / denominator
    result["successful_repair_rate_among_retries"] = (
        None if counter["retry_nodes"] == 0 else counter["successful_repairs"] / counter["retry_nodes"]
    )
    return result


def calculate_insufficiency_metrics(records: list[dict]) -> dict:
    overall = _new_counter()
    by_agent = defaultdict(_new_counter)
    by_node = defaultdict(_new_counter)
    agent_runs = 0

    for record in records:
        for agent, run in _agent_runs(record):
            agent_runs += 1
            result = run.get("result") or {}
            if run.get("status") == "abstain" or result.get("status") == "abstain" or result.get("tier") == "abstained":
                overall["abstentions"] += 1
                by_agent[agent]["abstentions"] += 1
            for step in run.get("trace") or []:
                attempts = step.get("attempts")
                if not isinstance(attempts, list):
                    continue
                node = str(step.get("node") or "unknown")
                overall["model_node_responses"] += 1
                by_agent[agent]["model_node_responses"] += 1
                by_node[f"{agent}:{node}"]["model_node_responses"] += 1
                attempt_count = len(attempts)
                for counter in (overall, by_agent[agent], by_node[f"{agent}:{node}"]):
                    counter["model_attempts"] += attempt_count
                answer = step.get("answer")
                if answer in INSUFFICIENT_ANSWERS:
                    for counter in (overall, by_agent[agent], by_node[f"{agent}:{node}"]):
                        counter["insufficient_or_indeterminate"] += 1
                if attempt_count > 1:
                    for counter in (overall, by_agent[agent], by_node[f"{agent}:{node}"]):
                        counter["retry_nodes"] += 1
                    if any(attempt.get("ok") for attempt in attempts[1:]):
                        for counter in (overall, by_agent[agent], by_node[f"{agent}:{node}"]):
                            counter["successful_repairs"] += 1

    return {
        "artifact_status": {
            "specific_intervention_found": SPECIFIC_INTERVENTION_FOUND,
            "specific_intervention_status": "blocked_pending_yana_artifact",
            "performance_claim": False,
        },
        "records": len(records),
        "agent_runs": agent_runs,
        "overall": _finalize(overall),
        "by_agent": {key: _finalize(value) for key, value in sorted(by_agent.items())},
        "by_node": {key: _finalize(value) for key, value in sorted(by_node.items())},
    }


def paired_comparison(baseline: list[dict], intervention: list[dict]) -> dict:
    base = calculate_insufficiency_metrics(baseline)
    after = calculate_insufficiency_metrics(intervention)
    base_rate = base["overall"]["insufficient_or_indeterminate_rate"]
    after_rate = after["overall"]["insufficient_or_indeterminate_rate"]
    return {
        "label": "MATCHED ENGINEERING COMPARISON; NO PERFORMANCE CLAIM",
        "baseline": base,
        "intervention": after,
        "delta_insufficient_or_indeterminate_rate": (
            None if base_rate is None or after_rate is None else after_rate - base_rate
        ),
        "case_ids_baseline": [record.get("case_id") for record in baseline],
        "case_ids_intervention": [record.get("case_id") for record in intervention],
        "case_ids_matched": [record.get("case_id") for record in baseline]
        == [record.get("case_id") for record in intervention],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--intervention")
    parser.add_argument("--out", default=str(paths.DATA / "insufficiency_eval.json"))
    args = parser.parse_args()

    baseline = _load_records(pathlib.Path(args.baseline))
    output = (
        paired_comparison(baseline, _load_records(pathlib.Path(args.intervention)))
        if args.intervention
        else calculate_insufficiency_metrics(baseline)
    )
    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    safe_print(f"-> {destination}")


if __name__ == "__main__":
    main()
