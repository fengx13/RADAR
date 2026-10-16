#!/usr/bin/env python3
"""Machine-readable per-case export for future Hema evaluation."""
from __future__ import annotations

from radar.evaluation.evaluate_agent4 import derive_decision_icu_transfer_12h
from radar.evaluation.evaluate_insufficiency import calculate_insufficiency_metrics
from radar.common.observation_semantics import classify_observed_disposition_detail


TIER_ORDER = {
    "unlikely_sepsis": 0,
    "possible_sepsis": 1,
    "probable_sepsis": 2,
    "definite_sepsis": 3,
}


def tier_transition(previous: str | None, current: str | None) -> str | None:
    if previous not in TIER_ORDER or current not in TIER_ORDER:
        return None
    if TIER_ORDER[current] > TIER_ORDER[previous]:
        return "upgrade"
    if TIER_ORDER[current] < TIER_ORDER[previous]:
        return "downgrade"
    return "unchanged"


def _result(run: dict | None) -> dict:
    return (run or {}).get("result") or {}


def _agent1_trace_fields(run: dict | None) -> dict:
    trace = (run or {}).get("trace") or []
    alternatives = [
        step.get("leading_alternative", (step.get("reply") or {}).get("leading_alternative"))
        for step in trace if step.get("node") in {"R3a", "R3b"}
    ]
    versions = {step.get("graph_version") for step in trace if step.get("graph_version")}
    return {
        "agent1_graph_version": next(iter(versions), _result(run).get("graph_version")),
        "leading_alternative": alternatives[-1] if alternatives else None,
    }


def _agent4_governance(run: dict | None) -> dict | None:
    step = next((item for item in (run or {}).get("trace") or [] if item.get("node") in {"D1", "D1_T0"}), None)
    if not step:
        return None
    return {
        "icu_gate_governance": step.get("icu_gate_governance"),
        "tier_b_candidates": step.get("tier_b_candidates", []),
    }


def _pipeline_terminated_in_abstention(envelope: dict) -> bool:
    """Identify explicit terminal abstention without counting unresolved routing states."""
    status = envelope.get("status")
    return isinstance(status, str) and (
        status == "abstain" or status.endswith("_abstain")
    )


def build_case_evaluation_export(envelope: dict) -> dict:
    t0_a1 = _result(envelope.get("agent1_t0"))
    t0_a4 = _result(envelope.get("agent4_t0"))
    previous_tier = t0_a1.get("tier")
    checkpoints = []
    for checkpoint in envelope.get("checkpoints") or []:
        row = {
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "cutoff_time": checkpoint.get("cutoff_time"),
            "checkpoint_status": checkpoint.get("checkpoint_status"),
            "skip_reason": checkpoint.get("skip_reason"),
            "newly_available_results": checkpoint.get("newly_available_results", []),
            "agent1_tier": None,
            "tier_transition": None,
            "agent3_recommendation": None,
            "agent4_status": None,
            "agent4_disposition": None,
            "abstention": False,
            "pipeline_alarm": False,
        }
        if checkpoint.get("checkpoint_status") == "processed":
            a1 = _result(checkpoint.get("agent1_reassessment"))
            a3 = _result(checkpoint.get("agent3"))
            a4 = _result(checkpoint.get("agent4"))
            row.update({
                "agent1_tier": a1.get("tier"),
                "tier_transition": tier_transition(previous_tier, a1.get("tier")),
                "agent3_recommendation": a3.get("treatment_plan"),
                "agent4_status": a4.get("status"),
                "agent4_disposition": a4.get("disposition"),
                "abstention": a1.get("tier") == "abstained" or a3.get("status") == "abstain" or a4.get("status") == "abstain",
                "pipeline_alarm": bool(a4.get("pipeline_alarm")),
                **_agent1_trace_fields(checkpoint.get("agent1_reassessment")),
                "agent4_governance": _agent4_governance(checkpoint.get("agent4")),
            })
            if a1.get("tier") in TIER_ORDER:
                previous_tier = a1.get("tier")
        checkpoints.append(row)

    insufficiency = calculate_insufficiency_metrics([envelope])["overall"]
    final_disposition = envelope.get("final_simulated_disposition")
    disposition_checkpoint = envelope.get("disposition_checkpoint")
    cutoff = envelope.get("t0_time", 0.0) if disposition_checkpoint == "T0" else next(
        (
            checkpoint.get("cutoff_time")
            for checkpoint in envelope.get("checkpoints") or []
            if checkpoint.get("checkpoint_id") == disposition_checkpoint
        ),
        None,
    )
    outcomes = envelope.get("evaluation_only")
    observed_disposition = None
    if isinstance(outcomes, dict):
        icu_detail = derive_decision_icu_transfer_12h(cutoff, outcomes)
        observed_disposition = classify_observed_disposition_detail(
            outcomes,
            icu_transfer_after_decision=icu_detail.get("event") is True,
        )
        observed_disposition["evaluation_only"] = True
    return {
        "case_id": envelope.get("case_id"),
        "stay_id": envelope.get("stay_id"),
        "tier_T0": t0_a1.get("tier"),
        **_agent1_trace_fields(envelope.get("agent1_t0")),
        "agent4_T0_status": t0_a4.get("status"),
        "agent4_T0_disposition": t0_a4.get("disposition"),
        "agent4_T0_governance": _agent4_governance(envelope.get("agent4_t0")),
        "workup_continued": bool(t0_a4.get("continue_workup")),
        "checkpoint_sequence": checkpoints,
        "final_simulated_disposition": final_disposition,
        "disposition_checkpoint": disposition_checkpoint,
        "time_to_disposition": cutoff,
        "insufficient_or_indeterminate_count": insufficiency["insufficient_or_indeterminate"],
        "model_node_response_count": insufficiency["model_node_responses"],
        "abstention": _pipeline_terminated_in_abstention(envelope),
        "pipeline_alarm": bool(t0_a4.get("pipeline_alarm")) or any(row["pipeline_alarm"] for row in checkpoints),
        "esrp_used": bool(((envelope.get("agent1_t0") or {}).get("risk_signal") or {}).get("enabled")),
        "evaluation_only_available": isinstance(envelope.get("evaluation_only"), dict),
        "observed_disposition_evaluation": observed_disposition,
        "clinical_correctness_adjudicated": False,
        "run_provenance": "current_radar_vnext",
    }
