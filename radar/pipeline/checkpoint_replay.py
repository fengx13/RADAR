#!/usr/bin/env python3
"""Deterministic checkpoint planning for offline longitudinal replay."""
from __future__ import annotations

from typing import Any

from radar.common.integration_schema import _time_key, validate_workup_state


DEFAULT_CHECKPOINTS = (
    ("C0_5", 0.5),
    ("C1", 1.0),
    ("C3", 3.0),
    ("C6", 6.0),
)


def _result_records(workup_state: dict) -> tuple[list[dict], list[str], list[dict], list[dict]]:
    records: list[dict] = []
    failures: list[str] = []
    non_checkpoint: list[dict] = []
    unresolved: list[dict] = []
    for item in workup_state.get("items") or []:
        if item.get("result_status") != "resulted":
            continue
        if item.get("actual_order_status") == "not_ordered":
            failures.append(f"{item.get('item')}:resulted_but_not_ordered")
            continue
        key = _time_key(item.get("result_available_time"))
        if key is None:
            failures.append(f"{item.get('item')}:invalid_result_available_time")
            continue
        record = {
            "item": item.get("item"),
            "result_available_time": item.get("result_available_time"),
            "result_kind": item.get("result_kind"),
            "checkpoint_eligible": item.get("checkpoint_eligible"),
            "checkpoint_eligibility_reason": item.get("checkpoint_eligibility_reason"),
            "_time_key": key,
        }
        if item.get("checkpoint_eligible") is True:
            records.append(record)
        elif item.get("checkpoint_eligible") is False:
            non_checkpoint.append(record)
        else:
            unresolved.append(record)
    return records, failures, non_checkpoint, unresolved


def _boundaries(custom: list[dict] | None, result_kind: str) -> tuple[list[dict], list[str]]:
    if custom is None:
        if result_kind != "numeric":
            return [], ["datetime_results_require_explicit_checkpoint_boundaries"]
        return [
            {"checkpoint_id": checkpoint_id, "cutoff_time": cutoff}
            for checkpoint_id, cutoff in DEFAULT_CHECKPOINTS
        ], []

    failures: list[str] = []
    result: list[dict] = []
    seen: set[str] = set()
    previous = None
    for index, entry in enumerate(custom):
        checkpoint_id = entry.get("checkpoint_id") if isinstance(entry, dict) else None
        cutoff = entry.get("cutoff_time") if isinstance(entry, dict) else None
        key = _time_key(cutoff)
        if checkpoint_id not in {"C0_5", "C1", "C3", "C6"}:
            failures.append(f"checkpoint_boundaries[{index}]:invalid_checkpoint_id")
        elif checkpoint_id in seen:
            failures.append(f"checkpoint_boundaries[{index}]:duplicate_checkpoint_id")
        else:
            seen.add(checkpoint_id)
        if key is None or key[0] != result_kind:
            failures.append(f"checkpoint_boundaries[{index}]:invalid_or_incomparable_cutoff")
        elif previous is not None and key[1] <= previous:
            failures.append(f"checkpoint_boundaries[{index}]:cutoffs_not_strictly_increasing")
        else:
            previous = key[1]
        result.append({"checkpoint_id": checkpoint_id, "cutoff_time": cutoff})
    return result, failures


def _public_result(record: dict) -> dict:
    return {
        "item": record["item"],
        "result_available_time": record["result_available_time"],
        "result_kind": record.get("result_kind"),
        "checkpoint_eligible": record.get("checkpoint_eligible"),
        "checkpoint_eligibility_reason": record.get("checkpoint_eligibility_reason"),
        "trigger_source": record.get("trigger_source", "workup_state"),
    }


def _longitudinal_trigger_records(longitudinal_events: dict | None) -> tuple[list[dict], list[dict]]:
    eligible: list[dict] = []
    unresolved: list[dict] = []
    events = longitudinal_events or {}
    specifications = (
        ("treatment_response", "assessment_time", "treatment_response"),
        ("vitals", "measurement_time", "vitals_trend"),
    )
    for group, time_field, result_kind in specifications:
        for index, event in enumerate(events.get(group) or []):
            if not isinstance(event, dict):
                continue
            if group == "vitals" and event.get("trend") in (None, ""):
                continue
            key = _time_key(event.get(time_field))
            if key is None:
                continue
            record = {
                "item": f"{result_kind}:{index}",
                "result_available_time": event.get(time_field),
                "result_kind": result_kind,
                "checkpoint_eligible": event.get("checkpoint_eligible"),
                "checkpoint_eligibility_reason": event.get("checkpoint_eligibility_reason", "explicit longitudinal-event metadata"),
                "trigger_source": f"longitudinal_events.{group}",
                "_time_key": key,
            }
            if event.get("checkpoint_eligible") is True:
                eligible.append(record)
            elif event.get("checkpoint_eligible") is None:
                unresolved.append(record)
    return eligible, unresolved


def build_checkpoint_plan(
    workup_state: dict,
    checkpoint_boundaries: list[dict] | None = None,
    longitudinal_events: dict | None = None,
) -> dict:
    """Plan predefined checkpoints using result availability, never sample/order time.

    The planner may inspect future result timestamps to decide which checkpoints are
    necessary. It returns no future result values, and callers must pass only the
    current checkpoint snapshot to inference.
    """
    failures = validate_workup_state(workup_state)
    if failures or workup_state.get("status") == "invalid":
        return {
            "status": "invalid",
            "reason": "invalid observed workup state",
            "validation_failures": failures or workup_state.get("validation_failures", []),
            "checkpoints": [],
        }

    records, result_failures, non_checkpoint, unresolved = _result_records(workup_state)
    if result_failures:
        return {
            "status": "invalid",
            "reason": "invalid result-availability timestamp",
            "validation_failures": result_failures,
            "checkpoints": [],
        }
    all_workup_results = [*records, *non_checkpoint, *unresolved]
    if not all_workup_results:
        return {
            "status": "not_ready",
            "reason": "no ordered investigation result is available to trigger T1",
            "checkpoints": [],
            "non_checkpoint_results": [_public_result(record) for record in non_checkpoint],
            "unresolved_checkpoint_eligibility": [_public_result(record) for record in unresolved],
        }

    workup_kinds = {record["_time_key"][0] for record in all_workup_results}
    if len(workup_kinds) != 1:
        return {
            "status": "invalid",
            "reason": "incomparable result-availability timestamp formats",
            "validation_failures": ["mixed_numeric_and_datetime_result_times"],
            "checkpoints": [],
        }

    t1_record = min(all_workup_results, key=lambda record: record["_time_key"][1])
    extra_eligible, extra_unresolved = _longitudinal_trigger_records(longitudinal_events)
    t1_key = t1_record["_time_key"]
    for record in extra_eligible:
        if record["_time_key"][0] == t1_key[0] and record["_time_key"][1] >= t1_key[1]:
            records.append(record)
    unresolved.extend(
        record for record in extra_unresolved
        if record["_time_key"][0] == t1_key[0] and record["_time_key"][1] >= t1_key[1]
    )
    if not records:
        return {
            "status": "not_ready",
            "reason": "T1 exists but no checkpoint-eligible information is available for replay",
            "t1_trigger": _public_result(t1_record),
            "checkpoints": [],
            "non_checkpoint_results": [_public_result(record) for record in non_checkpoint],
            "unresolved_checkpoint_eligibility": [_public_result(record) for record in unresolved],
        }

    kinds = {record["_time_key"][0] for record in records}
    if len(kinds) != 1:
        return {
            "status": "invalid",
            "reason": "incomparable result-availability timestamp formats",
            "validation_failures": ["mixed_numeric_and_datetime_result_times"],
            "checkpoints": [],
        }
    kind = next(iter(kinds))
    boundaries, boundary_failures = _boundaries(checkpoint_boundaries, kind)
    if boundary_failures:
        return {
            "status": "invalid",
            "reason": "invalid checkpoint boundaries",
            "validation_failures": boundary_failures,
            "checkpoints": [],
        }

    records.sort(key=lambda record: record["_time_key"][1])
    keyed_boundaries = [
        (entry, _time_key(entry["cutoff_time"])[1])
        for entry in boundaries
    ]
    last_fixed = keyed_boundaries[-1][1]
    within_fixed = [record for record in records if record["_time_key"][1] <= last_fixed]
    after_fixed = [record for record in records if record["_time_key"][1] > last_fixed]

    if after_fixed:
        fixed_to_include = len(keyed_boundaries)
    elif within_fixed:
        latest = within_fixed[-1]["_time_key"][1]
        fixed_to_include = next(
            index + 1
            for index, (_, cutoff) in enumerate(keyed_boundaries)
            if latest <= cutoff
        )
    else:
        fixed_to_include = 0

    plan: list[dict] = []
    previous = float("-inf")
    for entry, cutoff in keyed_boundaries[:fixed_to_include]:
        newly_available = [
            record for record in records
            if previous < record["_time_key"][1] <= cutoff
        ]
        plan.append({
            "checkpoint_id": entry["checkpoint_id"],
            "cutoff_time": entry["cutoff_time"],
            "checkpoint_status": "processed" if newly_available else "skipped",
            "skip_reason": None if newly_available else "no_new_result",
            "newly_available_results": [_public_result(record) for record in newly_available],
        })
        previous = cutoff

    if after_fixed:
        final_record = after_fixed[-1]
        final_cutoff = final_record["result_available_time"]
        plan.append({
            "checkpoint_id": "Cfinal",
            "cutoff_time": final_cutoff,
            "checkpoint_status": "processed",
            "skip_reason": None,
            "newly_available_results": [_public_result(record) for record in after_fixed],
        })

    return {
        "status": "ready",
        "reason": "result-availability-driven checkpoint plan",
        "timestamp_kind": kind,
        "t1_trigger": _public_result(t1_record),
        "result_count": len(records),
        "non_checkpoint_results": [_public_result(record) for record in non_checkpoint],
        "unresolved_checkpoint_eligibility": [_public_result(record) for record in unresolved],
        "checkpoints": plan,
    }
