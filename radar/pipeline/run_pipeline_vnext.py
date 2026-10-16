#!/usr/bin/env python3
"""RADAR vNext T0 plus result-triggered longitudinal replay orchestrator."""
from __future__ import annotations
from radar import paths

import argparse
import copy
import hashlib
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
from radar.agents import run_agent4_t0
from radar.agents.agent3_observed_treatment import build_observed_antimicrobials
from radar.pipeline.checkpoint_replay import build_checkpoint_plan
from radar.evaluation.evaluation_export import build_case_evaluation_export
from radar.common.integration_schema import FUTURE_OUTCOME_KEYS, integrated_case_skeleton, strip_inference_forbidden
from radar.pipeline.longitudinal_integration import (
    assemble_checkpoint_state,
    assemble_t0_agent4_state,
    build_agent1_context,
    build_agent4_payload,
    build_agent4_t0_payload,
    build_treatment_state,
    build_workup_state,
    determine_t1_gate,
    reconcile_treatment,
)
from radar.common.run_provenance import model_runtime_metadata, new_run_metadata
from radar.pipeline.pipeline_batch import (
    add_batch_arguments,
    output_destination,
    run_batches,
    validate_longitudinal_cases,
    write_output,
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
    risk_signal: str = "none",
    risk_signal_map: dict | None = None,
    risk_signal_file: str | None = None,
    replay_mode: str = "prospective",
    continue_on_failure: bool = False,
    deployment_unavailable: list | None = None,
):
    return SimpleNamespace(
        dry_run=dry_run,
        model=model,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        risk_signal=risk_signal,
        risk_signal_map=risk_signal_map or {},
        risk_signal_file=risk_signal_file,
        replay_mode=replay_mode,
        continue_on_failure=continue_on_failure,
        deployment_unavailable=list(deployment_unavailable or []),
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
        values["risk_signal"] = config.risk_signal if decision_time == "T0" else "none"
        values["risk_signal_map"] = config.risk_signal_map if decision_time == "T0" else {}
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
    envelope["evaluation_export"] = build_case_evaluation_export(envelope)
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


def _mocks_for_checkpoint(mocks: dict, checkpoint_id: str) -> dict:
    checkpoint = ((mocks.get("checkpoints") or {}).get(checkpoint_id) or {})
    return {
        "agent1": checkpoint.get("agent1") or mocks.get("agent1_reassessment") or mocks.get("agent1_t1") or {},
        "agent3": checkpoint.get("agent3") or mocks.get("agent3") or {},
        "agent4": checkpoint.get("agent4") or mocks.get("agent4") or {},
    }


def _set_legacy_aliases(envelope: dict, checkpoint: dict, agent4_payload: dict):
    """Retain read compatibility while checkpoints[] is the canonical trace."""
    if envelope.get("agent1_t1") is not None:
        return
    envelope["t1_state"] = checkpoint["patient_state"]
    envelope["agent1_t1"] = checkpoint["agent1_reassessment"]
    envelope["agent3"] = checkpoint["agent3"]
    envelope["treatment_state"] = checkpoint["observed_treatment"]
    envelope["reconciliation"] = checkpoint["reconciliation"]
    envelope["agent4_input"] = strip_inference_forbidden(agent4_payload)
    envelope["agent4"] = checkpoint["agent4"]


def run_longitudinal_case(case: dict, client=None, config=None, *, agent_runners=None) -> dict:
    """Run T0 and only those reassessment checkpoints triggered by new results."""
    config = config or pipeline_config()
    runners = agent_runners or {}
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
    t0_time = case.get("t0_time", 0.0)

    a1_t0_case = {
        "case_id": case_id,
        "stay_id": case.get("stay_id"),
        "input": envelope["t0_state"],
        "mock_agent1_answers": mocks.get("agent1_t0") or {},
    }
    envelope["agent1_t0"] = runners.get("A1", run_agent1.run_case)(a1_t0_case, client, _args(config, "T0"))
    if envelope["agent1_t0"].get("result", {}).get("tier") == "abstained":
        return _stop(envelope, "upstream_abstain", "agent1_t0", "Agent 1 T0 abstained")

    a1_context = build_agent1_context(envelope["agent1_t0"], "T0")
    a2_case = {
        "case_id": case_id,
        "input": envelope["t0_state"],
        "agent1_context": a1_context,
    }
    envelope["agent2"] = runners.get("A2", run_agent2.run_case)(a2_case, client, _args(config))
    if envelope["agent2"].get("status") != "complete":
        return _stop(envelope, "upstream_abstain", "agent2", "Agent 2 did not complete")

    t0_agent4_state = assemble_t0_agent4_state(
        envelope["t0_state"], case.get("longitudinal_events") or {}, t0_time
    )
    t0_payload = build_agent4_t0_payload(
        case_id, t0_agent4_state, envelope["agent1_t0"], envelope["agent2"], t0_time
    )
    if config.dry_run:
        t0_payload["mock_llm_answers"] = mocks.get("agent4_t0") or {}
    envelope["agent4_t0_input"] = run_agent4_t0.sanitize_t0_payload(t0_payload)
    envelope["agent4_t0"] = runners.get("A4_T0", run_agent4_t0.run_case)(t0_payload, client, _args(config))
    t0_result = envelope["agent4_t0"].get("result") or {}
    if t0_result.get("status") == "decision":
        envelope["final_simulated_disposition"] = t0_result.get("disposition")
        envelope["disposition_checkpoint"] = "T0"
        return _stop(envelope, "complete", "agent4_t0", "first valid T0 disposition finalized")
    if t0_result.get("status") == "abstain":
        return _stop(envelope, "upstream_abstain", "agent4_t0", "T0 Agent 4 abstained")

    envelope["workup_state"] = build_workup_state(
        envelope["agent2"], case.get("observed_workup") or []
    )
    envelope["t1_gate"] = determine_t1_gate(envelope["workup_state"])
    checkpoint_plan = build_checkpoint_plan(
        envelope["workup_state"],
        case.get("checkpoint_boundaries"),
        case.get("longitudinal_events") or {},
    )
    envelope["checkpoint_plan"] = checkpoint_plan
    if checkpoint_plan["status"] == "not_ready":
        return _stop(envelope, "t1_not_ready", "checkpoint_plan", checkpoint_plan["reason"])
    if checkpoint_plan["status"] != "ready":
        return _stop(envelope, "invalid", "t1_gate", checkpoint_plan["reason"])

    previous_agent1 = envelope["agent1_t0"]
    previous_decision_time = "T0"
    latest_replay_disposition = None

    for planned in checkpoint_plan["checkpoints"]:
        checkpoint = copy.deepcopy(planned)
        if checkpoint["checkpoint_id"] == "Cfinal" and envelope["final_simulated_disposition"] is not None:
            checkpoint["checkpoint_status"] = "skipped"
            checkpoint["skip_reason"] = "simulated_disposition_already_resolved"
        if checkpoint["checkpoint_status"] == "skipped":
            envelope["checkpoints"].append(checkpoint)
            continue

        checkpoint_id = checkpoint["checkpoint_id"]
        cutoff = checkpoint["cutoff_time"]
        state = assemble_checkpoint_state(
            case.get("longitudinal_events") or {},
            envelope["workup_state"],
            cutoff,
            checkpoint_id,
        )
        checkpoint["patient_state"] = state
        checkpoint_mocks = _mocks_for_checkpoint(mocks, checkpoint_id)

        a1_case = {
            "case_id": case_id,
            "input": {
                "checkpoint_state": state,
                "presentation_context": {
                    "as_of_time": t0_time,
                    "source": "historical_T0_presentation_not_current_reassessment",
                    **{key: copy.deepcopy(envelope["t0_state"][key])
                       for key in ("age", "sex", "chief_complaint", "comorbidities", "prior_history")
                       if key in envelope["t0_state"]},
                },
                "checkpoint": {
                    "checkpoint_id": checkpoint_id,
                    "cutoff_time": cutoff,
                    "newly_available_results": checkpoint["newly_available_results"],
                },
                "agent1_prior_context": build_agent1_context(previous_agent1, previous_decision_time),
            },
            "mock_agent1_answers": checkpoint_mocks["agent1"],
        }
        checkpoint["agent1_reassessment"] = runners.get("A1", run_agent1.run_case)(
            a1_case, client, _args(config, "checkpoint")
        )
        if checkpoint["agent1_reassessment"].get("result", {}).get("tier") == "abstained":
            checkpoint["stop_reason"] = "Agent 1 reassessment abstained"
            envelope["checkpoints"].append(checkpoint)
            if getattr(config, "continue_on_failure", False):
                checkpoint["checkpoint_status"] = "processed_with_failure"
                continue
            return _stop(envelope, "upstream_abstain", checkpoint_id, checkpoint["stop_reason"])

        a3_case = {
            "case_id": case_id,
            "input": {
                "age": envelope["t0_state"].get("age"),
                "agent1_result": checkpoint["agent1_reassessment"].get("result"),
                "checkpoint_state": state,
                "observed_antimicrobials": build_observed_antimicrobials(
                    case.get("observed_treatment"), cutoff
                ),
                "checkpoint": {"checkpoint_id": checkpoint_id, "cutoff_time": cutoff},
            },
            "dry_run_answers": checkpoint_mocks["agent3"],
        }
        checkpoint["agent3"] = runners.get("A3", run_agent3.run_case)(a3_case, client, _args(config))
        if checkpoint["agent3"].get("status") != "complete":
            checkpoint["stop_reason"] = "Agent 3 did not complete"
            envelope["checkpoints"].append(checkpoint)
            if getattr(config, "continue_on_failure", False):
                checkpoint["checkpoint_status"] = "processed_with_failure"
                previous_agent1 = checkpoint["agent1_reassessment"]
                previous_decision_time = checkpoint_id
                continue
            return _stop(envelope, "upstream_abstain", checkpoint_id, checkpoint["stop_reason"])

        checkpoint["observed_treatment"] = build_treatment_state(
            case.get("observed_treatment") or {}, cutoff
        )
        if checkpoint["observed_treatment"].get("status") != "valid":
            checkpoint["stop_reason"] = "invalid observed treatment state"
            envelope["checkpoints"].append(checkpoint)
            return _stop(envelope, "invalid", checkpoint_id, checkpoint["stop_reason"])
        checkpoint["reconciliation"] = reconcile_treatment(
            checkpoint["agent3"], checkpoint["observed_treatment"]
        )

        agent4_payload = build_agent4_payload(
            case_id=case_id,
            t1_time=cutoff,
            t1_state=state,
            agent1_t1_run=checkpoint["agent1_reassessment"],
            agent2_run=envelope["agent2"],
            workup_state=envelope["workup_state"],
            agent3_run=checkpoint["agent3"],
            treatment_state=checkpoint["observed_treatment"],
            reconciliation=checkpoint["reconciliation"],
            checkpoint_id=checkpoint_id,
        )
        if getattr(config, "deployment_unavailable", None):
            agent4_payload["deployment_unavailable_fields"] = list(config.deployment_unavailable)
        if config.dry_run:
            agent4_payload["mock_llm_answers"] = checkpoint_mocks["agent4"]
        checkpoint["agent4_input"] = run_agent4.normalize_payload(agent4_payload)
        checkpoint["agent4"] = runners.get("A4", run_agent4.run_case)(agent4_payload, client, _args(config))
        result = checkpoint["agent4"].get("result") or {}
        checkpoint["stop_reason"] = None
        envelope["checkpoints"].append(checkpoint)
        _set_legacy_aliases(envelope, checkpoint, agent4_payload)

        previous_agent1 = checkpoint["agent1_reassessment"]
        previous_decision_time = checkpoint_id
        if result.get("status") == "decision":
            latest_replay_disposition = result.get("disposition")
            if envelope["final_simulated_disposition"] is None:
                envelope["final_simulated_disposition"] = result.get("disposition")
                envelope["disposition_checkpoint"] = checkpoint_id
            if config.replay_mode == "prospective":
                checkpoint["stop_reason"] = "first valid final disposition"
                break
        elif result.get("status") == "abstain" and config.replay_mode == "prospective":
            checkpoint["stop_reason"] = "Agent 4 abstained"
            break

    envelope["latest_replay_disposition"] = latest_replay_disposition
    if envelope["final_simulated_disposition"] is not None:
        return _stop(
            envelope,
            "complete",
            envelope["disposition_checkpoint"],
            "first valid final disposition ended prospective pathway"
            if config.replay_mode == "prospective"
            else "full replay completed; first disposition preserved",
        )
    last = next(
        (checkpoint for checkpoint in reversed(envelope["checkpoints"]) if checkpoint.get("checkpoint_status") == "processed"),
        None,
    )
    if last and _result_status(last.get("agent4")) == "abstain":
        return _stop(envelope, "upstream_abstain", last["checkpoint_id"], "Agent 4 abstained")
    return _stop(envelope, "checkpoint_not_ready", "checkpoints", "no final disposition was produced")


def _result_status(run: dict | None):
    return ((run or {}).get("result") or {}).get("status")


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
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pilot-mode", action="store_true")
    parser.add_argument(
        "--pipeline-mode",
        choices=("sequential", "legacy_single_t1", "legacy_cohort_two_time"),
        default="sequential",
        help="canonical entrypoints dispatch explicit legacy modes; sequential is the active framework",
    )
    parser.add_argument("--replay-mode", choices=("prospective", "full"), default="prospective")
    parser.add_argument("--deployment-unavailable", action="append", default=[],
                        metavar="FIELD",
                        help="checkpoint Agent 4 field this data source never records, e.g. "
                             "patient_state.oxygen_requirement for MIMIC-IV-ED; it cannot block a "
                             "disposition and the uncertainty ceiling prevents discharge")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run: keep the cases already in the output file "
                             "(except runner_error) and process only the rest; implies --force")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="research replay: record an A1 abstention or A3 failure at a checkpoint "
                             "and still process later checkpoints instead of stopping the case")
    parser.add_argument("--risk-signal", choices=("none", "esrp"), default="none")
    parser.add_argument("--risk-signal-file", default=str(paths.DATA / "phase0_esrp_mapping.json"))
    return add_batch_arguments(parser)


def load_env():
    env_path = paths.ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.pipeline_mode != "sequential":
        parser.error("select legacy modes through run_pipeline.py or run_full_pipeline.py")
    if args.max_batches is not None and args.batch_size is None:
        parser.error("--max-batches requires --batch-size")
    try:
        cases = load_cases(pathlib.Path(args.infile))
        validate_longitudinal_cases(cases)
        destination = output_destination(args.out, args.infile, args.run_tag, args.force or args.resume)
        input_sha256 = hashlib.sha256(pathlib.Path(args.infile).read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    run_metadata = new_run_metadata("radar-vnext", ROOT)
    risk_map = (
        run_agent1.load_risk_signals(args.risk_signal_file)
        if args.risk_signal == "esrp"
        else {}
    )
    config = pipeline_config(
        dry_run=args.dry_run,
        model="synthetic-stub" if args.dry_run else args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        seed=args.seed,
        risk_signal=args.risk_signal,
        risk_signal_map=risk_map,
        risk_signal_file=args.risk_signal_file if args.risk_signal == "esrp" else None,
        replay_mode=args.replay_mode,
        continue_on_failure=args.continue_on_failure,
        deployment_unavailable=args.deployment_unavailable,
    )
    client = None
    if not args.dry_run:
        load_env()
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set (put it in .env in the repository root or the environment)")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    started = time.time()
    output = {
        "meta": {
            **run_metadata,
            "pipeline": "sepsis_ed_radar_longitudinal",
            "version": "0.3",
            "dry_run": args.dry_run,
            "model": config.model,
            "model_requested": args.model,
            "model_runtime": {
                "A1": model_runtime_metadata(run_agent1),
                "A2": model_runtime_metadata(run_agent2),
                "A3": model_runtime_metadata(run_agent3),
                "A4_T0": model_runtime_metadata(run_agent4_t0),
                "A4_checkpoint": model_runtime_metadata(run_agent4),
            },
            "base_url": args.base_url,
            "sampling": {
                "temperature_requested": args.temperature,
                "seed_requested": args.seed,
                "parameter_support": {
                    "A1": dict(run_agent1.CAPS),
                    "A2": dict(run_agent2.CAPS),
                    "A3": dict(run_agent3.CAPS),
                    "A4_T0": dict(run_agent4_t0.CAPS),
                    "A4_checkpoint": dict(run_agent4.CAPS),
                },
                "reasoning_effort_requested": os.environ.get("RADAR_REASONING_EFFORT"),
            },
            "pilot_mode": args.pilot_mode,
            "replay_mode": args.replay_mode,
            "continue_on_failure": args.continue_on_failure,
            "deployment_unavailable_fields": args.deployment_unavailable,
            "pipeline_mode": "sequential",
            "risk_signal": args.risk_signal,
            "agent1_graph_version": run_agent1.GRAPH["version"],
            "agent_graph_versions": {
                "A1": run_agent1.GRAPH["version"],
                "A2": run_agent2.GRAPH["version"],
                "A3": run_agent3.GRAPH["version"],
                "A4_T0": run_agent4_t0.GRAPH["version"],
                "A4_checkpoint": run_agent4.GRAPH["version"],
            },
            "checkpoint_policy": "C0_5/C1/C3/C6 plus Cfinal only for new checkpoint-eligible evidence after 6h while disposition is unresolved",
            "clinical_validation": False,
            "n_input": len(cases),
            "n": 0,
            "input_sha256": input_sha256,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "run_tag": args.run_tag,
            "batch_summaries": [],
            "cohort_execution_complete": False,
            "runner_error_count": 0,
            "seconds": 0,
        },
        "results": [],
    }

    prior = []
    if args.resume and destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        prior = [r for r in previous.get("results", []) if r.get("status") != "runner_error"]
        done = {r["case_id"] for r in prior}
        all_cases = cases
        cases = [c for c in cases if c["case_id"] not in done]
        output["meta"]["resumed_from"] = {"kept": len(prior), "remaining": len(cases),
                                          "previous_run_id": previous.get("meta", {}).get("run_id")}
        safe_print(f"resume: {len(prior)} cases kept, {len(cases)} remaining")
        if not cases:
            safe_print("nothing left to run")
            return 0
    else:
        all_cases = cases

    def publish_batch(results, batch_index, batch_n):
        results = prior + results
        output["results"] = results
        meta = output["meta"]
        meta["n"] = len(results)
        meta["cohort_execution_complete"] = len(results) == len(all_cases)
        meta["runner_error_count"] = sum(r.get("status") == "runner_error" for r in results)
        meta["seconds"] = round(time.time() - started, 3)
        meta["batch_summaries"].append({"batch": batch_index, "n": batch_n})
        for name, module in (("A1", run_agent1), ("A2", run_agent2), ("A3", run_agent3),
                             ("A4_T0", run_agent4_t0), ("A4_checkpoint", run_agent4)):
            meta["model_runtime"][name] = model_runtime_metadata(module)
        write_output(destination, output)

    results = run_batches(
        cases, lambda case: run_longitudinal_case(case, client=client, config=config),
        workers=args.workers, batch_size=args.batch_size, max_batches=args.max_batches,
        on_batch=publish_batch,
    )
    counts: dict[str, int] = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    safe_print(f"pipeline statuses: {counts}")
    if not output["meta"]["cohort_execution_complete"]:
        safe_print(f"partial pilot: {len(output['results'])}/{len(all_cases)} cases processed")
    safe_print(f"-> {destination}")
    return 1 if output["meta"]["runner_error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
