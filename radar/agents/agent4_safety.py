#!/usr/bin/env python3
"""Governed deterministic ICU safety triggers for Agent 4."""
from __future__ import annotations

from radar.agents.agent4_insufficiency import _MISSING, resolve
from radar.common.integration_schema import _time_key, time_at_or_before


TIER_A_STATUS = "active"
TIER_B_STATUS = "pending_clinical_signoff"
EXPLICIT_NON_CRITERIA = (
    "supplemental_oxygen_alone",
    "chronic_tracheostomy_predating_encounter",
    "sep3",
    "sofa_ge_2",
    "high_news",
    "sep3_shock_derived_label",
)


def _num(value):
    if value is _MISSING or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gte(a, b) -> bool:
    left = _time_key(a)
    right = _time_key(b)
    return left is not None and right is not None and left[0] == right[0] and left[1] >= right[1]


def _bool(value) -> bool | None:
    if value is _MISSING or value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "documented"}
    return bool(value)


def _after(value, boundary) -> bool:
    left = _time_key(value)
    right = _time_key(boundary)
    return left is not None and right is not None and left[0] == right[0] and left[1] > right[1]


def evaluate_icu_triggers(payload: dict) -> dict:
    """Evaluate active Tier A triggers and audit inactive Tier B candidates.

    Tier B candidates are deliberately inactive until explicit clinical sign-off.
    Their matches are returned for governance review but never set ``icu=True``.
    """
    legacy_input = "t1_state" in payload and "patient_state" not in payload
    cutoff = resolve(payload, "information_cutoff")
    if cutoff is _MISSING:
        cutoff = resolve(payload, "t1_time")
    tier_a: list[dict] = []
    tier_b: list[dict] = []
    tier_a_missing: set[str] = set()
    tier_b_missing: set[str] = set()
    excluded_post_checkpoint: set[str] = set()
    governance_exclusions: list[dict] = []

    setting = resolve(payload, "patient_state.vasopressor.setting")
    active_raw = resolve(payload, "patient_state.vasopressor.active")
    active = _bool(active_raw)
    start_time = resolve(payload, "patient_state.vasopressor.start_time")
    if start_time is _MISSING:
        start_time = resolve(payload, "patient_state.vasopressor.hours")
    if setting in {"ed", "pre_arrival"} and active is True and time_at_or_before(start_time, cutoff):
        tier_a.append({"id": "ICU1_vasopressor_active", "field": "patient_state.vasopressor", "tier": "A"})
    elif setting in {"ed", "pre_arrival"} and active is True and (start_time in {_MISSING, None, ""} or cutoff is _MISSING):
        tier_a_missing.update({"patient_state.vasopressor.start_time", "information_cutoff"})
    elif setting in {"ed", "pre_arrival"} and active_raw is _MISSING:
        tier_a_missing.add("patient_state.vasopressor.active")
    if setting == "icu":
        governance_exclusions.append({
            "criterion": "vasopressor",
            "reason": "vaso_setting_icu_is_future_information_for_prospective_ed_disposition",
        })
    if setting in {"ed", "pre_arrival"} and _after(start_time, cutoff):
        excluded_post_checkpoint.add("patient_state.vasopressor.start_time")

    vent_status = resolve(payload, "patient_state.ventilation.status")
    vent_start = resolve(payload, "patient_state.ventilation.start_time")
    current_encounter_raw = resolve(payload, "patient_state.ventilation.current_encounter")
    current_encounter = _bool(current_encounter_raw)
    if vent_status in {"InvasiveVent", "NonInvasiveVent", "HFNC"} and current_encounter_raw is _MISSING:
        tier_a_missing.add("patient_state.ventilation.current_encounter")
    if vent_status == "InvasiveVent" and current_encounter is True and time_at_or_before(vent_start, cutoff):
        tier_a.append({"id": "ICU2_invasive_ventilation", "field": "patient_state.ventilation", "tier": "A"})
    elif vent_status == "InvasiveVent" and (vent_start is _MISSING or cutoff is _MISSING):
        tier_a_missing.update({"patient_state.ventilation.start_time", "information_cutoff"})
    if vent_status in {"NonInvasiveVent", "HFNC"} and current_encounter is True and time_at_or_before(vent_start, cutoff):
        tier_a.append({"id": "ICU3_niv_or_hfnc", "field": "patient_state.ventilation", "tier": "A"})
    elif vent_status in {"NonInvasiveVent", "HFNC"} and (vent_start is _MISSING or cutoff is _MISSING):
        tier_a_missing.update({"patient_state.ventilation.start_time", "information_cutoff"})
    if vent_status in {"InvasiveVent", "NonInvasiveVent", "HFNC"} and _after(vent_start, cutoff):
        excluded_post_checkpoint.add("patient_state.ventilation.start_time")
    if vent_status == "Tracheostomy" and current_encounter is not True:
        governance_exclusions.append({
            "criterion": "tracheostomy",
            "reason": "chronic_tracheostomy_predating_encounter_is_not_an_icu_trigger",
        })

    fluid_raw = resolve(payload, "patient_state.resuscitation.initial_fluid_30ml_per_kg.completed")
    fluid_completed = _bool(fluid_raw)
    if fluid_raw is _MISSING:
        tier_b_missing.add("patient_state.resuscitation.initial_fluid_30ml_per_kg.completed")
    fluid_time = resolve(payload, "patient_state.resuscitation.initial_fluid_30ml_per_kg.completion_time")
    map_min = _num(resolve(payload, "patient_state.post_resuscitation.vital_map_min"))
    map_time = resolve(payload, "patient_state.post_resuscitation.map_time")
    map_valid = time_at_or_before(fluid_time, cutoff) and time_at_or_before(map_time, cutoff) and _gte(map_time, fluid_time)
    if fluid_completed is True and map_min is not None and map_min < 65 and map_valid:
        tier_b.append({"id": "ICU4_post_resuscitation_hypotension", "field": "patient_state.post_resuscitation.vital_map_min", "tier": "B", "governance_status": TIER_B_STATUS})
    elif fluid_completed is True and map_min is not None and map_min < 65:
        tier_b_missing.update({"patient_state.resuscitation.initial_fluid_30ml_per_kg.completion_time", "patient_state.post_resuscitation.map_time"})

    lactate = _num(resolve(payload, "patient_state.labs.lactate_initial.value"))
    lactate_sample = resolve(payload, "patient_state.labs.lactate_initial.sample_time")
    lactate_result = resolve(payload, "patient_state.labs.lactate_initial.result_time")
    lactate_valid = time_at_or_before(fluid_time, cutoff) and time_at_or_before(lactate_sample, cutoff) and time_at_or_before(lactate_result, cutoff) and _gte(lactate_sample, fluid_time) and _gte(lactate_result, lactate_sample)
    if fluid_completed is True and lactate is not None and lactate >= 4 and lactate_valid:
        tier_b.append({"id": "ICU5_post_resuscitation_lactate", "field": "patient_state.labs.lactate_initial.value", "tier": "B", "governance_status": TIER_B_STATUS})
    elif fluid_completed is True and lactate is not None and lactate >= 4:
        tier_b_missing.update({"patient_state.resuscitation.initial_fluid_30ml_per_kg.completion_time", "patient_state.labs.lactate_initial.sample_time", "patient_state.labs.lactate_initial.result_time"})

    spo2_min = _num(resolve(payload, "patient_state.post_resuscitation.vital_o2sat_min"))
    spo2_time = resolve(payload, "patient_state.post_resuscitation.o2sat_time")
    oxygen_raw = resolve(payload, "patient_state.oxygen_requirement.oxygen_therapy_documented")
    oxygen_documented = _bool(oxygen_raw)
    oxygen_start = resolve(payload, "patient_state.oxygen_requirement.start_time")
    if oxygen_raw is _MISSING:
        tier_b_missing.add("patient_state.oxygen_requirement.oxygen_therapy_documented")
    oxygen_valid = time_at_or_before(oxygen_start, cutoff) and time_at_or_before(spo2_time, cutoff)
    if oxygen_documented is True and spo2_min is not None and spo2_min < 90 and oxygen_valid:
        tier_b.append({"id": "ICU6_hypoxemia_despite_oxygen", "field": "patient_state.post_resuscitation.vital_o2sat_min", "tier": "B", "governance_status": TIER_B_STATUS})
    elif oxygen_documented is True and spo2_min is not None and spo2_min < 90:
        tier_b_missing.update({"patient_state.oxygen_requirement.start_time", "patient_state.post_resuscitation.o2sat_time"})
    elif oxygen_documented is True:
        governance_exclusions.append({"criterion": "supplemental_oxygen", "reason": "supplemental_oxygen_alone_is_not_an_icu_trigger"})

    for field, value in (
        ("patient_state.resuscitation.initial_fluid_30ml_per_kg.completion_time", fluid_time),
        ("patient_state.post_resuscitation.map_time", map_time),
        ("patient_state.labs.lactate_initial.sample_time", lactate_sample),
        ("patient_state.labs.lactate_initial.result_time", lactate_result),
        ("patient_state.oxygen_requirement.start_time", oxygen_start),
        ("patient_state.post_resuscitation.o2sat_time", spo2_time),
    ):
        if _after(value, cutoff):
            excluded_post_checkpoint.add(field)

    def aliases(fields: set[str]) -> list[str]:
        ordered = sorted(fields)
        return [field.replace("patient_state.", "t1_state.") for field in ordered] if legacy_input else ordered

    return {
        "icu": bool(tier_a),
        "triggers": tier_a,
        "tier_a_triggers": tier_a,
        "tier_b_candidates": tier_b,
        "governance": {
            "tier_a": TIER_A_STATUS,
            "tier_b": TIER_B_STATUS,
            "tier_b_active": False,
            "explicit_non_criteria": list(EXPLICIT_NON_CRITERIA),
        },
        "governance_exclusions": governance_exclusions,
        "tier_a_missing_fields": aliases(tier_a_missing),
        "tier_b_missing_fields": aliases(tier_b_missing),
        "missing_checkpoint_fields": sorted(tier_a_missing),
        "missing_fields": aliases(tier_a_missing),
        "excluded_post_checkpoint_fields": sorted(excluded_post_checkpoint),
        "excluded_post_t1_fields": aliases(excluded_post_checkpoint),
    }
