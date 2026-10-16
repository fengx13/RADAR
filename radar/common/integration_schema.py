#!/usr/bin/env python3
"""Shared longitudinal contracts for the synthetic Sepsis ED Agent pipeline.

This module contains deterministic data plumbing only. It does not make clinical
recommendations and never converts an agent recommendation into observed care.
"""
from __future__ import annotations

import copy
from datetime import datetime
from typing import Any


PROVENANCE_TYPES = {
    "model_recommendation",
    "observed_ehr",
    "deterministic_derived",
    "pending",
    "unavailable",
    "synthetic_fixture",
}

WORKUP_LIFECYCLE = {
    "not_ordered",
    "ordered",
    "collected",
    "pending",
    "resulted",
    "unavailable",
}

TREATMENT_RESPONSE_STATES = {
    "stable",
    "improving",
    "worsening",
    "pending",
    "not_yet_assessable",
    "unavailable",
}

CHECKPOINT_IDS = ("C0_5", "C1", "C3", "C6", "Cfinal")

FUTURE_OUTCOME_KEYS = {
    "truth",
    "outcomes",
    "evaluation_only",
    "observed_disposition",
    "ed_disposition",
    "admission_type",
    "hadm_id",
    "icu_transfer_time",
    "icu_transfer_status",
    "critical_deterioration",
    "mortality_inhospital",
    "mortality_30d",
    "ed_revisit_time",
    "ed_revisit_3d",
    "dsp_hospitalization",
    "dsp_icu_transfer_12h",
    "dsp_critical",
    "dsp_ed_revisit_3d",
    "dsp_mortality_inhospital",
    "dsp_mortality_30d",
}

ENGINEERING_MOCK_KEYS = {
    "engineering_mocks",
    "mock_answers",
    "mock_agent1_answers",
    "mock_llm_answers",
    "dry_run_answers",
}


def strip_inference_forbidden(obj: Any, *, exclude_temporal_audit: bool = False) -> Any:
    """Recursively remove evaluation-only outcomes and engineering mock answers."""
    if isinstance(obj, dict):
        return {
            key: strip_inference_forbidden(value, exclude_temporal_audit=exclude_temporal_audit)
            for key, value in obj.items()
            if key not in FUTURE_OUTCOME_KEYS
            and not key.startswith("dsp_")
            and key not in ENGINEERING_MOCK_KEYS
            and not (exclude_temporal_audit and key.startswith("excluded_post_"))
        }
    if isinstance(obj, list):
        return [strip_inference_forbidden(value, exclude_temporal_audit=exclude_temporal_audit) for value in obj]
    return copy.deepcopy(obj)


def provenance(
    source_type: str,
    source_field: str | None = None,
    note: str | None = None,
    event_time: Any = None,
    available_time: Any = None,
    included_at_t1: bool | None = None,
) -> dict:
    if source_type not in PROVENANCE_TYPES:
        raise ValueError(f"unsupported provenance source_type: {source_type!r}")
    result = {"source_type": source_type}
    if source_field is not None:
        result["source_field"] = source_field
    if note is not None:
        result["note"] = note
    if event_time is not None:
        result["event_time"] = event_time
    if available_time is not None:
        result["available_time"] = available_time
    if included_at_t1 is not None:
        result["included_at_t1"] = included_at_t1
    return result


def _time_key(value: Any) -> tuple[str, float] | None:
    """Return a comparable key for numeric or ISO-8601 timestamps."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return "numeric", float(value)
    if isinstance(value, str):
        try:
            return "numeric", float(value)
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return "datetime", parsed.timestamp()
        except (ValueError, OverflowError):
            return None
    return None


def time_at_or_before(value: Any, boundary: Any) -> bool:
    left = _time_key(value)
    right = _time_key(boundary)
    return left is not None and right is not None and left[0] == right[0] and left[1] <= right[1]


def validate_workup_state(workup_state: dict) -> list[str]:
    failures: list[str] = []
    if not isinstance(workup_state, dict):
        return ["workup_state_not_object"]
    items = workup_state.get("items")
    if not isinstance(items, list):
        return ["workup_state_items_not_list"]
    seen: set[str] = set()
    for index, item in enumerate(items):
        prefix = f"workup_state.items[{index}]"
        if not isinstance(item, dict):
            failures.append(f"{prefix}:not_object")
            continue
        name = item.get("item")
        if not name:
            failures.append(f"{prefix}:missing_item")
        elif name in seen:
            failures.append(f"{prefix}:duplicate_item:{name}")
        else:
            seen.add(name)
        order_status = item.get("actual_order_status")
        result_status = item.get("result_status")
        if order_status not in WORKUP_LIFECYCLE:
            failures.append(f"{prefix}:invalid_actual_order_status:{order_status!r}")
        if result_status not in WORKUP_LIFECYCLE:
            failures.append(f"{prefix}:invalid_result_status:{result_status!r}")
        if result_status == "resulted" and item.get("result_available_time") in (None, ""):
            failures.append(f"{prefix}:resulted_without_result_available_time")
        checkpoint_eligible = item.get("checkpoint_eligible")
        if checkpoint_eligible is not None and not isinstance(checkpoint_eligible, bool):
            failures.append(f"{prefix}:invalid_checkpoint_eligible:{checkpoint_eligible!r}")
        disposition_relevant = item.get("disposition_relevant")
        if disposition_relevant is not None and not isinstance(disposition_relevant, bool):
            failures.append(f"{prefix}:invalid_disposition_relevant:{disposition_relevant!r}")
        prov = item.get("provenance")
        if not isinstance(prov, dict) or prov.get("source_type") not in PROVENANCE_TYPES:
            failures.append(f"{prefix}:invalid_provenance")
    return failures


def validate_treatment_state(treatment_state: dict) -> list[str]:
    failures: list[str] = []
    if not isinstance(treatment_state, dict):
        return ["treatment_state_not_object"]
    ongoing = treatment_state.get("ongoing_treatment")
    if not isinstance(ongoing, list):
        failures.append("treatment_state.ongoing_treatment_not_list")
    response = treatment_state.get("treatment_response_flag")
    if response not in TREATMENT_RESPONSE_STATES:
        failures.append(f"treatment_state.invalid_response:{response!r}")
    prov = treatment_state.get("provenance")
    if not isinstance(prov, dict) or prov.get("source_type") not in PROVENANCE_TYPES:
        failures.append("treatment_state.invalid_provenance")
    return failures


def integrated_case_skeleton(case_id: str) -> dict:
    """Return the shared longitudinal envelope without inventing patient values."""
    return {
        "case_id": case_id,
        "stay_id": None,
        "t0_state": None,
        "agent1_t0": None,
        "agent2": None,
        "agent4_t0": None,
        "workup_state": None,
        "checkpoints": [],
        "final_simulated_disposition": None,
        "disposition_checkpoint": None,
        "evaluation_export": None,
        # Backward-compatible aliases populated from the first processed
        # reassessment checkpoint. New consumers should use checkpoints[].
        "t1_gate": None,
        "t1_state": None,
        "agent1_t1": None,
        "agent3": None,
        "treatment_state": None,
        "reconciliation": None,
        "agent4": None,
        "evaluation_only": None,
    }
