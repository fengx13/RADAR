#!/usr/bin/env python3
"""Deterministic longitudinal state builders and agent interface adapters."""
from __future__ import annotations

import copy
from typing import Any

from radar.pipeline.event_visibility import event_after_cutoff, event_available_time, event_visible_at
from radar.common.integration_schema import (
    TREATMENT_RESPONSE_STATES,
    _time_key,
    provenance,
    strip_inference_forbidden,
    time_at_or_before,
    validate_treatment_state,
    validate_workup_state,
)


AGENT2_ITEMS = (
    "blood_cultures",
    "serum_lactate",
    "source_imaging_or_sampling",
    "alternative_diagnosis_testing",
)

CHECKPOINT_ELIGIBILITY_BY_RESULT_KIND = {
    "repeat_lactate": True,
    "imaging": True,
    "diagnostic_laboratory": True,
    "treatment_response": True,
    "vitals_trend": True,
    "final_culture": False,
}


def _checkpoint_policy(observed: dict) -> tuple[bool | None, str]:
    explicit = observed.get("checkpoint_eligible")
    if isinstance(explicit, bool):
        return explicit, "explicit integration metadata"
    result_kind = observed.get("result_kind")
    if result_kind in CHECKPOINT_ELIGIBILITY_BY_RESULT_KIND:
        return CHECKPOINT_ELIGIBILITY_BY_RESULT_KIND[result_kind], f"frozen policy for result_kind={result_kind}"
    return None, "unresolved: integration metadata do not safely establish checkpoint eligibility"


def build_agent1_context(agent1_run: dict | None, decision_time: str = "T0") -> dict:
    """Expose only the approved recognition result to downstream agents."""
    result = _agent_result(agent1_run)
    return {
        "tier": result.get("tier"),
        "shock_concern": result.get("shock_concern"),
        "decision_time": decision_time,
        "provenance": "agent1_model_output",
    }


def assemble_t0_agent4_state(
    t0_state: dict | None,
    event_data: dict | None,
    t0_time: Any = 0.0,
) -> dict:
    """Add only genuinely T0-visible/pre-arrival support to the T0 state."""
    state = strip_inference_forbidden(t0_state or {})
    events = strip_inference_forbidden(event_data or {})
    active_support: dict[str, Any] = {}

    respiratory = _latest_event(events.get("respiratory_support") or [], "start_time", t0_time)
    if respiratory is not None:
        active_support["ventilation"] = {
            "status": respiratory.get("ventilation_status"),
            "start_time": respiratory.get("start_time"),
            "current_encounter": respiratory.get("current_encounter"),
            "oxygen_therapy_documented": respiratory.get("oxygen_therapy_documented"),
            "mode": respiratory.get("mode"),
            "provenance": _temporal_provenance(respiratory, "start_time", t0_time),
        }

    vasopressor = _latest_event(events.get("vasopressors") or [], "start_time", t0_time)
    if vasopressor is not None:
        active_support["vasopressor"] = {
            "active": vasopressor.get("active"),
            "setting": vasopressor.get("setting"),
            "agent": vasopressor.get("agent"),
            "start_time": vasopressor.get("start_time"),
            "provenance": _temporal_provenance(vasopressor, "start_time", t0_time),
        }

    if active_support:
        state["active_support"] = active_support
    state["information_cutoff"] = t0_time
    return strip_inference_forbidden(state)


def build_agent4_t0_payload(
    case_id: str,
    t0_state: dict,
    agent1_t0_run: dict,
    agent2_run: dict,
    t0_time: Any = 0.0,
) -> dict:
    return strip_inference_forbidden({
        "case_id": case_id,
        "decision_time": "T0",
        "t0_time": t0_time,
        "information_cutoff": t0_time,
        "t0_state": t0_state,
        "agent1_context": build_agent1_context(agent1_t0_run, "T0"),
        "agent2": {
            "workup_recommendation": _agent_result(agent2_run).get("workup_plan"),
            "provenance": provenance("model_recommendation", source_field="agent2.result.workup_plan"),
        },
    })


def _agent_result(run: dict | None) -> dict:
    return (run or {}).get("result") or {}


def _recommendations_by_item(agent2_run: dict | None) -> dict[str, str]:
    plan = _agent_result(agent2_run).get("workup_plan") or {}
    result: dict[str, str] = {}
    for group, decision in (
        ("recommendations", "recommend"),
        ("already_addressed", "already_addressed"),
        ("not_indicated", "not_indicated"),
        ("indeterminate", "indeterminate"),
    ):
        for entry in plan.get(group) or []:
            if entry.get("item"):
                result[entry["item"]] = decision
    return result


def build_workup_state(agent2_run: dict | None, observed_items: list[dict] | None) -> dict:
    """Join A2 recommendations to observed events without conflating them."""
    recommendations = _recommendations_by_item(agent2_run)
    observed_by_item = {
        item.get("item"): copy.deepcopy(item)
        for item in (observed_items or [])
        if isinstance(item, dict) and item.get("item")
    }
    names = list(dict.fromkeys([*AGENT2_ITEMS, *recommendations, *observed_by_item]))
    items: list[dict] = []
    for name in names:
        observed = observed_by_item.get(name)
        if observed is None:
            observed = {
                "actual_order_status": "unavailable",
                "order_time": None,
                "sample_time": None,
                "result_status": "unavailable",
                "result_available_time": None,
                "value": None,
                "result_summary": None,
                "provenance": provenance(
                    "unavailable",
                    note="no observed workup event supplied",
                ),
            }
        entry = {
            "item": name,
            "agent2_recommendation": recommendations.get(name),
            "actual_order_status": observed.get("actual_order_status", "unavailable"),
            "order_time": observed.get("order_time"),
            "sample_time": observed.get("sample_time"),
            "result_status": observed.get("result_status", "unavailable"),
            "result_available_time": observed.get("result_available_time"),
            "value": observed.get("value"),
            "unit": observed.get("unit"),
            "result_summary": observed.get("result_summary"),
            "result_kind": observed.get("result_kind"),
            "disposition_relevant": observed.get("disposition_relevant"),
            "provenance": observed.get("provenance") or provenance(
                "observed_ehr", note="observed workup event supplied by integration input"
            ),
        }
        entry["checkpoint_eligible"], entry["checkpoint_eligibility_reason"] = _checkpoint_policy(observed)
        items.append(entry)

    pending = [
        item["item"]
        for item in items
        if item["actual_order_status"] in {"ordered", "collected", "pending", "resulted"}
        and item["result_status"] != "resulted"
    ]
    state = {
        "status": "valid",
        "items": items,
        "pending_workup": pending,
        "provenance": provenance(
            "deterministic_derived",
            note="A2 recommendation joined to separately supplied observed workup events",
        ),
    }
    failures = validate_workup_state(state)
    if failures:
        state["status"] = "invalid"
        state["validation_failures"] = failures
    return state


def determine_t1_gate(workup_state: dict) -> dict:
    """Return T1 at the earliest reliable result-availability time."""
    failures = validate_workup_state(workup_state)
    if failures or workup_state.get("status") == "invalid":
        return {
            "status": "invalid",
            "t1_time": None,
            "triggering_item": None,
            "reason": "invalid observed workup state",
            "validation_failures": failures or workup_state.get("validation_failures", []),
        }

    candidates: list[tuple[tuple[str, float], dict]] = []
    invalid_times: list[str] = []
    for item in workup_state.get("items", []):
        if item.get("result_status") != "resulted":
            continue
        if item.get("actual_order_status") == "not_ordered":
            invalid_times.append(f"{item.get('item')}:resulted_but_not_ordered")
            continue
        result_time = item.get("result_available_time")
        key = _time_key(result_time)
        if key is None:
            invalid_times.append(f"{item.get('item')}:invalid_result_available_time")
            continue
        order_key = _time_key(item.get("order_time"))
        if order_key is not None and (order_key[0] != key[0] or order_key[1] > key[1]):
            invalid_times.append(f"{item.get('item')}:result_time_precedes_or_mismatches_order_time")
            continue
        sample_key = _time_key(item.get("sample_time"))
        if sample_key is not None and (sample_key[0] != key[0] or sample_key[1] > key[1]):
            invalid_times.append(f"{item.get('item')}:result_time_precedes_or_mismatches_sample_time")
            continue
        candidates.append((key, item))

    if invalid_times:
        return {
            "status": "invalid",
            "t1_time": None,
            "triggering_item": None,
            "reason": "invalid result-availability timestamp",
            "validation_failures": invalid_times,
        }
    if not candidates:
        return {
            "status": "not_ready",
            "t1_time": None,
            "triggering_item": None,
            "reason": "no ordered investigation result is yet available",
        }
    kinds = {key[0] for key, _ in candidates}
    if len(kinds) != 1:
        return {
            "status": "invalid",
            "t1_time": None,
            "triggering_item": None,
            "reason": "incomparable result-availability timestamp formats",
            "validation_failures": ["mixed_numeric_and_datetime_result_times"],
        }
    _, trigger = min(candidates, key=lambda pair: pair[0][1])
    return {
        "status": "ready",
        "t1_time": trigger["result_available_time"],
        "information_cutoff": trigger["result_available_time"],
        "triggering_item": trigger["item"],
        "reason": "first ordered investigation result available",
        "provenance": provenance(
            "deterministic_derived",
            source_field="workup_state.items[].result_available_time",
        ),
    }


def workup_state_as_of(workup_state: dict, cutoff_time: Any, checkpoint_id: str | None = None) -> dict:
    """Create an inference-safe observed-workup snapshot at one cutoff."""
    result = copy.deepcopy(workup_state)
    pending: list[str] = []
    excluded = 0
    for item in result.get("items", []):
        order_time = item.get("order_time")
        sample_time = item.get("sample_time")
        result_time = item.get("result_available_time")
        original_provenance = copy.deepcopy(item.get("provenance") or provenance("unavailable"))

        if order_time is not None and not time_at_or_before(order_time, cutoff_time):
            item.update({
                "actual_order_status": "unavailable",
                "order_time": None,
                "sample_time": None,
                "result_status": "unavailable",
                "result_available_time": None,
                "value": None,
                "result_summary": None,
                "provenance": provenance("unavailable", note="workup order was not visible at this checkpoint"),
            })
            excluded += 1
            continue

        if sample_time is not None and not time_at_or_before(sample_time, cutoff_time):
            item["sample_time"] = None
            if item.get("actual_order_status") in {"collected", "pending", "resulted"}:
                item["actual_order_status"] = "ordered"

        if item.get("result_status") == "resulted" and not time_at_or_before(result_time, cutoff_time):
            item["result_status"] = "pending"
            item["result_available_time"] = None
            item["value"] = None
            item["result_summary"] = None
            item["provenance"] = provenance(
                "pending", note="result was not available at this checkpoint"
            )
            excluded += 1
        item_provenance = copy.deepcopy(item.get("provenance") or original_provenance)
        item_provenance.update({
            "event_time": item.get("sample_time"),
            "available_time": item.get("result_available_time"),
            "included_at_checkpoint": item.get("result_status") == "resulted",
            "included_at_t1": item.get("result_status") == "resulted",
        })
        item["provenance"] = item_provenance
        if item.get("actual_order_status") in {"ordered", "collected", "pending", "resulted"} and item.get("result_status") != "resulted":
            pending.append(item["item"])
    result["pending_workup"] = pending
    result["as_of_time"] = cutoff_time
    result["information_cutoff"] = cutoff_time
    result["checkpoint_id"] = checkpoint_id
    result["excluded_post_checkpoint_results"] = excluded
    result["excluded_post_t1_results"] = excluded
    return strip_inference_forbidden(result)


def _latest_event(events: list[dict] | None, time_field: str, boundary: Any) -> dict | None:
    eligible: list[tuple[tuple[str, float], dict]] = []
    for event in events or []:
        key = _time_key(event.get(time_field))
        boundary_key = _time_key(boundary)
        if key is None or boundary_key is None or key[0] != boundary_key[0]:
            continue
        if event_visible_at(event, time_field, boundary):
            eligible.append((key, event))
    if not eligible:
        return None
    return copy.deepcopy(max(eligible, key=lambda pair: pair[0][1])[1])


def _excluded_count(events: list[dict] | None, time_field: str, boundary: Any) -> int:
    boundary_key = _time_key(boundary)
    if boundary_key is None:
        return 0
    count = 0
    for event in events or []:
        if event_after_cutoff(event, time_field, boundary):
            count += 1
    return count


def _temporal_provenance(event: dict, time_field: str, cutoff: Any) -> dict:
    result = copy.deepcopy(event.get("provenance") or provenance("observed_ehr"))
    event_time = event.get(time_field)
    result.update({
        "event_time": event_time,
        "available_time": event_available_time(event, time_field),
        "included_at_checkpoint": event_visible_at(event, time_field, cutoff),
        "included_at_t1": event_visible_at(event, time_field, cutoff),
    })
    return result



SOFA_CORE = ("platelets", "bilirubin_total", "creatinine")


def sofa_from_checkpoint(investigations: dict, vitals: dict, on_vasopressor: bool) -> dict:
    """SOFA subscores computed in code from the results visible at the checkpoint.

    Same bands as make_t1_states.sofa_scores (the workshop T1 states). A component
    with no result is reported missing and never scored 0. A total >= 2 from the
    components that have resulted establishes SOFA >= 2, because a component that
    has not resulted can only add to the score; a total < 2 with components
    missing establishes nothing.
    """
    def value(name):
        item = investigations.get(name) if isinstance(investigations, dict) else None
        v = item.get("value") if isinstance(item, dict) else None
        return v if isinstance(v, (int, float)) else None

    def band(v, cuts):
        if v is None:
            return None
        return sum(1 for c in cuts if v >= c) if cuts[0] < cuts[-1] else sum(1 for c in cuts if v <= c)

    plt, bil, cre, pao2 = value("platelets"), value("bilirubin_total"), value("creatinine"), value("pao2")
    map_value = vitals.get("map") if isinstance(vitals, dict) else None
    scores = {
        "coag": band(plt, [150, 100, 50, 20]),
        "liver": band(bil, [1.2, 2.0, 6.0, 12.0]),
        "renal": band(cre, [1.2, 2.0, 3.5, 5.0]),
        "resp": None,
        "cardio": 1 if on_vasopressor else (None if map_value is None else (1 if map_value < 70 else 0)),
    }
    missing = [k for k, v in scores.items() if v is None]
    total = sum(v for v in scores.values() if v is not None)
    available = [n for n in ("platelets", "bilirubin_total", "creatinine", "pao2") if value(n) is not None]
    return {
        "components_available": available,
        "components_missing": [n for n in ("platelets", "bilirubin_total", "creatinine", "pao2") if value(n) is None],
        "core_labs_complete": all(value(n) is not None for n in SOFA_CORE),
        "subscores": scores,
        "not_scored": missing,
        "total_from_available_components": total,
        "sofa_ge_2_established": total >= 2,
        "inputs": {"platelets": plt, "bilirubin_total": bil, "creatinine": cre, "pao2": pao2,
                   "map": map_value, "vasopressor_active": on_vasopressor},
        "note": ("Computed in code from results available at this cutoff; the model never scores SOFA. "
                 "A component with no result is reported missing, never scored 0. total >= 2 from the "
                 "resulted components establishes SOFA >= 2 (unresulted components can only raise it); "
                 "total < 2 with components missing is inconclusive. Respiratory needs an arterial gas "
                 "and FiO2 most ED patients never have; mental status is not available in this deployment."),
        "provenance": provenance("deterministic_derived", note="code-computed SOFA at checkpoint cutoff"),
    }


def assemble_checkpoint_state(
    event_data: dict | None,
    workup_state: dict,
    cutoff_time: Any,
    checkpoint_id: str,
) -> dict:
    """Assemble only observations available at or before one checkpoint."""
    events = strip_inference_forbidden(event_data or {})
    snapshot = workup_state_as_of(workup_state, cutoff_time, checkpoint_id)
    state: dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "as_of_time": cutoff_time,
        "information_cutoff": cutoff_time,
        "provenance": provenance(
            "deterministic_derived", note="latest supplied observations available at or before checkpoint cutoff"
        ),
    }
    excluded: dict[str, int] = {}

    vital_events = events.get("vitals") or []
    vitals: dict[str, Any] = {}
    vital_meta: dict[str, Any] = {}
    for field in ("sbp", "map", "resp_rate", "spo2", "heart_rate"):
        candidates = [event for event in vital_events if event.get(field) is not None]
        latest = _latest_event(candidates, "measurement_time", cutoff_time)
        if latest is not None:
            vitals[field] = latest[field]
            vital_meta[field] = {
                "measurement_time": latest.get("measurement_time"),
                "provenance": _temporal_provenance(latest, "measurement_time", cutoff_time),
            }
    if vitals:
        state["vitals"] = vitals
        state["vitals_metadata"] = vital_meta
    latest_vital = _latest_event(vital_events, "measurement_time", cutoff_time)
    if latest_vital is not None and "trend" in latest_vital:
        state["vitals_trend"] = latest_vital.get("trend")
    excluded["vitals"] = _excluded_count(vital_events, "measurement_time", cutoff_time)

    respiratory_events = events.get("respiratory_support") or []
    respiratory = _latest_event(respiratory_events, "start_time", cutoff_time)
    if respiratory is not None:
        state["oxygen_requirement"] = {
            "oxygen_therapy_documented": respiratory.get("oxygen_therapy_documented"),
            "mode": respiratory.get("mode"),
            "start_time": respiratory.get("start_time"),
            "provenance": _temporal_provenance(respiratory, "start_time", cutoff_time),
        }
        state["ventilation"] = {
            "status": respiratory.get("ventilation_status"),
            "start_time": respiratory.get("start_time"),
            "current_encounter": respiratory.get("current_encounter"),
            "provenance": _temporal_provenance(respiratory, "start_time", cutoff_time),
        }
    else:
        # Unknown support is a structural value, never evidence of room air.
        state["oxygen_requirement"] = {
            "oxygen_therapy_documented": None,
            "mode": None,
            "start_time": None,
            "provenance": provenance(
                "unavailable", note="no respiratory support observation supplied by checkpoint cutoff"
            ),
        }
    excluded["respiratory_support"] = _excluded_count(respiratory_events, "start_time", cutoff_time)

    vasopressor_events = events.get("vasopressors") or []
    vasopressor = _latest_event(vasopressor_events, "start_time", cutoff_time)
    if vasopressor is not None:
        state["vasopressor"] = {
            "active": vasopressor.get("active"),
            "setting": vasopressor.get("setting"),
            "agent": vasopressor.get("agent"),
            "start_time": vasopressor.get("start_time"),
            "provenance": _temporal_provenance(vasopressor, "start_time", cutoff_time),
        }
    excluded["vasopressors"] = _excluded_count(vasopressor_events, "start_time", cutoff_time)

    fluid_events = events.get("fluids") or []
    fluid = _latest_event(fluid_events, "event_time", cutoff_time)
    if fluid is not None:
        fluid_completion_time = fluid.get("completion_time")
        completion_visible = time_at_or_before(fluid_completion_time, cutoff_time)
        completion_claim = fluid.get("initial_fluid_30ml_per_kg_completed")
        if completion_claim is True and not completion_visible:
            completion_at_t1 = None
        else:
            completion_at_t1 = completion_claim
        state["resuscitation"] = {
            "amount_ml": fluid.get("amount_ml"),
            "provenance": _temporal_provenance(fluid, "event_time", cutoff_time),
            "initial_fluid_30ml_per_kg": {
                "completed": completion_at_t1,
                "completion_time": fluid_completion_time if completion_visible else None,
            },
        }
    excluded["fluids"] = _excluded_count(fluid_events, "event_time", cutoff_time)
    excluded["fluid_completion"] = sum(
        1 for event in fluid_events
        if event.get("completion_time") is not None
        and not time_at_or_before(event.get("completion_time"), cutoff_time)
    )

    post_events = events.get("post_resuscitation") or []
    post = _latest_event(post_events, "measurement_time", cutoff_time)
    if post is not None:
        state["post_resuscitation"] = {}
        if time_at_or_before(post.get("map_time"), cutoff_time):
            state["post_resuscitation"].update({
                "vital_map_min": post.get("vital_map_min"),
                "map_time": post.get("map_time"),
            })
        if time_at_or_before(post.get("o2sat_time"), cutoff_time):
            state["post_resuscitation"].update({
                "vital_o2sat_min": post.get("vital_o2sat_min"),
                "o2sat_time": post.get("o2sat_time"),
            })
        state["post_resuscitation"]["provenance"] = _temporal_provenance(post, "measurement_time", cutoff_time)
    excluded["post_resuscitation"] = _excluded_count(post_events, "measurement_time", cutoff_time)
    excluded["post_resuscitation_fields"] = sum(
        1 for event in post_events
        for field in ("map_time", "o2sat_time")
        if event.get(field) is not None and not time_at_or_before(event.get(field), cutoff_time)
    )

    labs: dict[str, Any] = {}
    by_name = {item["item"]: item for item in snapshot.get("items", [])}
    lactate = by_name.get("serum_lactate")
    if lactate is not None:
        labs["lactate_initial"] = {
            "status": lactate.get("result_status"),
            "value": lactate.get("value") if lactate.get("result_status") == "resulted" else None,
            "sample_time": lactate.get("sample_time"),
            "result_time": lactate.get("result_available_time"),
            "provenance": lactate.get("provenance"),
        }
    repeat = by_name.get("lactate_repeat")
    labs["lactate_repeat"] = None if repeat is None or repeat.get("result_status") != "resulted" else {
        "status": "resulted",
        "value": repeat.get("value"),
        "sample_time": repeat.get("sample_time"),
        "result_time": repeat.get("result_available_time"),
        "provenance": repeat.get("provenance"),
    }
    cultures = by_name.get("blood_cultures")
    labs["culture_result"] = None if cultures is None or cultures.get("result_status") != "resulted" else {
        "status": "resulted",
        "summary": cultures.get("result_summary"),
        "result_time": cultures.get("result_available_time"),
        "provenance": cultures.get("provenance"),
    }
    imaging = by_name.get("source_imaging_or_sampling")
    labs["imaging_result"] = None if imaging is None or imaging.get("result_status") != "resulted" else {
        "status": "resulted",
        "summary": imaging.get("result_summary"),
        "result_time": imaging.get("result_available_time"),
        "provenance": imaging.get("provenance"),
    }
    if labs.get("lactate_initial") is not None:
        state["labs"] = labs

    # Keep other supplied investigations visible to reassessment without
    # deriving scores or treating an order/sample timestamp as result availability.
    available = {
        item["item"]: {
            key: copy.deepcopy(item.get(key))
            for key in ("value", "unit", "abnormal_flag", "result_summary", "sample_time",
                        "result_available_time", "provenance")
            if key != "abnormal_flag" or item.get(key) is not None
        }
        for item in snapshot.get("items", [])
        if item.get("result_status") == "resulted"
        and time_at_or_before(item.get("result_available_time"), cutoff_time)
    }
    if available:
        state["investigation_results"] = available
    vasopressor_active = bool(vasopressor is not None and vasopressor.get("active") is True)
    state["sofa"] = sofa_from_checkpoint(available, vitals, vasopressor_active)

    response_events = events.get("treatment_response") or []
    response = _latest_event(response_events, "assessment_time", cutoff_time)
    if response is not None:
        state["treatment_response"] = response.get("state")
        state["treatment_response_detail"] = {
            "assessment_time": response.get("assessment_time"),
            "provenance": _temporal_provenance(response, "assessment_time", cutoff_time),
        }
    excluded["treatment_response"] = _excluded_count(response_events, "assessment_time", cutoff_time)

    state["excluded_post_checkpoint_event_counts"] = {key: count for key, count in excluded.items() if count}
    state["excluded_post_t1_event_counts"] = {key: count for key, count in excluded.items() if count}
    return strip_inference_forbidden(state)


def assemble_t1_state(event_data: dict | None, workup_state: dict, t1_time: Any) -> dict:
    """Backward-compatible wrapper for the former single-T1 interface."""
    return assemble_checkpoint_state(event_data, workup_state, t1_time, "T1")


def build_treatment_state(observed: dict | None, as_of_time: Any) -> dict:
    """Build observed-care state independently of Agent 3 recommendations."""
    source = strip_inference_forbidden(observed or {})
    observed_any = False
    components: dict[str, Any] = {}
    ongoing: list[str] = []
    excluded: dict[str, int] = {}

    specs = {
        "antimicrobials": ("administration_time", "antimicrobials"),
        "fluids": ("event_time", "fluids"),
        "vasopressors": ("start_time", "vasopressors"),
        "respiratory_support": ("start_time", "respiratory_support"),
    }
    for key, (time_field, label) in specs.items():
        entries = source.get(key) or []
        latest = _latest_event(entries, time_field, as_of_time)
        excluded[key] = _excluded_count(entries, time_field, as_of_time)
        if latest is None:
            components[key] = {
                "status": "unavailable",
                "provenance": provenance("unavailable", note=f"no {key} event supplied by decision time"),
            }
            continue
        observed_any = True
        latest["provenance"] = _temporal_provenance(latest, time_field, as_of_time)
        components[key] = latest
        if latest.get("active") is True or latest.get("status") in {"active", "ongoing"}:
            ongoing.append(label)

    response_events = source.get("treatment_response") or []
    response = _latest_event(response_events, "assessment_time", as_of_time)
    excluded["treatment_response"] = _excluded_count(response_events, "assessment_time", as_of_time)
    if response is None:
        response_state = "not_yet_assessable" if observed_any else "unavailable"
        response_detail = {
            "state": response_state,
            "assessment_time": None,
            "provenance": provenance(
                "deterministic_derived" if observed_any else "unavailable",
                note="no explicit treatment-response assessment available by decision time",
            ),
        }
    else:
        response_state = response.get("state")
        if response_state not in TREATMENT_RESPONSE_STATES:
            response_state = "unavailable"
        response_detail = {
            "state": response_state,
            "assessment_time": response.get("assessment_time"),
            "provenance": _temporal_provenance(response, "assessment_time", as_of_time),
        }

    state = {
        **components,
        "ongoing_treatment": ongoing,
        "treatment_response_flag": response_state,
        "treatment_response": response_detail,
        "as_of_time": as_of_time,
        "information_cutoff": as_of_time,
        "excluded_post_checkpoint_event_counts": {key: count for key, count in excluded.items() if count},
        "excluded_post_t1_event_counts": {key: count for key, count in excluded.items() if count},
        "provenance": provenance(
            "deterministic_derived", note="observed treatment events summarized independently of Agent 3"
        ),
    }
    failures = validate_treatment_state(state)
    state["status"] = "valid" if not failures else "invalid"
    if failures:
        state["validation_failures"] = failures
    return state


def _observed_positive(component: dict, kind: str) -> bool | None:
    if not isinstance(component, dict) or component.get("status") == "unavailable":
        return None
    if kind == "vasopressors":
        if component.get("active") is not None:
            return bool(component.get("active"))
        if component.get("administered") is not None:
            return bool(component.get("administered"))
    if component.get("administered") is not None:
        return bool(component.get("administered"))
    return component.get("status") in {"active", "ongoing", "completed", "administered"}


def reconcile_treatment(agent3_run: dict | None, treatment_state: dict) -> dict:
    """Describe recommendation/observation alignment without clinical adjudication."""
    plan = _agent_result(agent3_run).get("treatment_plan") or {}
    definitions = {
        "antimicrobials": (
            ((plan.get("antimicrobial_timing") or {}).get("recommend_antimicrobials")),
            treatment_state.get("antimicrobials") or {},
        ),
        "fluids": (
            ((plan.get("fluids") or {}).get("give_fluids")),
            treatment_state.get("fluids") or {},
        ),
        "vasopressors": (
            ((plan.get("vasopressors") or {}).get("start_vasopressor")),
            treatment_state.get("vasopressors") or {},
        ),
    }
    items: list[dict] = []
    for name, (recommended, observed_component) in definitions.items():
        observed = _observed_positive(observed_component, name)
        active = observed_component.get("active") is True or observed_component.get("status") in {"active", "ongoing"}
        if recommended is True and observed is True:
            alignment = "recommended_and_ongoing" if active else "recommended_and_given"
        elif recommended is True and observed is False:
            alignment = "recommended_not_yet_given"
        elif recommended is False and observed is True:
            alignment = "not_recommended_but_given"
        elif recommended is False and observed is False:
            alignment = "not_recommended_and_not_given"
        else:
            alignment = "observed_state_unavailable"
        items.append({
            "treatment": name,
            "recommended": recommended,
            "observed": observed,
            "alignment": alignment,
            "recommendation_provenance": provenance("model_recommendation", source_field=f"agent3.treatment_plan.{name}"),
            "observed_provenance": observed_component.get("provenance") or provenance("unavailable"),
        })

    response = treatment_state.get("treatment_response_flag")
    response_alignment = {
        "pending": "response_pending",
        "not_yet_assessable": "response_not_yet_assessable",
        "unavailable": "response_unavailable",
    }.get(response, "response_assessed")
    return {
        "status": "complete",
        "items": items,
        "response_state": response_alignment,
        "provenance": provenance(
            "deterministic_derived", note="descriptive comparison only; no clinical judgment added"
        ),
    }


def build_agent4_payload(
    case_id: str,
    t1_time: Any,
    t1_state: dict,
    agent1_t1_run: dict,
    agent2_run: dict,
    workup_state: dict,
    agent3_run: dict,
    treatment_state: dict,
    reconciliation: dict,
    checkpoint_id: str | None = None,
) -> dict:
    """Adapt one reassessment checkpoint to Agent 4's inference-safe contract."""
    safe_t1 = copy.deepcopy(strip_inference_forbidden(t1_state))
    decision_label = checkpoint_id or "T1"
    workup_at_t1 = workup_state_as_of(workup_state, t1_time, decision_label)
    a1_result = _agent_result(agent1_t1_run)
    a2_result = _agent_result(agent2_run)
    a3_result = _agent_result(agent3_run)

    if "vasopressor" not in safe_t1 and treatment_state.get("vasopressors", {}).get("status") != "unavailable":
        observed = treatment_state["vasopressors"]
        safe_t1["vasopressor"] = {
            "active": observed.get("active"),
            "setting": observed.get("setting"),
            "agent": observed.get("agent"),
            "start_time": observed.get("start_time"),
            "provenance": observed.get("provenance"),
        }
    if "treatment_response" not in safe_t1:
        safe_t1["treatment_response"] = treatment_state.get("treatment_response_flag")

    payload = {
        "case_id": case_id,
        "decision_time": decision_label,
        "decision_time_value": t1_time,
        "checkpoint_id": checkpoint_id,
        "information_cutoff": t1_time,
        "patient_state": safe_t1,
        "agent1": {
            "tier": a1_result.get("tier"),
            "shock_concern": a1_result.get("shock_concern"),
            "decision_time": decision_label,
            "provenance": provenance("model_recommendation", source_field="agent1_reassessment.result"),
        },
        "agent2": {
            "workup_recommendation": a2_result.get("workup_plan"),
            "provenance": provenance("model_recommendation", source_field="agent2.result.workup_plan"),
        },
        "workup_state": workup_at_t1,
        "agent3": {
            "treatment_recommendation": a3_result.get("treatment_plan"),
            "provenance": provenance("model_recommendation", source_field="agent3.result.treatment_plan"),
        },
        "treatment_state": strip_inference_forbidden(treatment_state),
        "reconciliation": strip_inference_forbidden(reconciliation),
    }
    return strip_inference_forbidden(payload)
