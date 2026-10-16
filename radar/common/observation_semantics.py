#!/usr/bin/env python3
"""Canonical retrospective Agent 4 disposition labels and exclusions."""
from __future__ import annotations


OBSERVATION_ADMISSION_TYPES = frozenset({
    "EU OBSERVATION",
    "OBSERVATION ADMIT",
    "DIRECT OBSERVATION",
    "AMBULATORY OBSERVATION",
})

TRUE_INPATIENT_ADMISSION_TYPES = frozenset({
    "EW EMER.",
    "URGENT",
    "DIRECT EMER.",
    "ELECTIVE",
    "SURGICAL SAME DAY",
})

NO_MODELED_DISPOSITION_ED_VALUES = frozenset({
    "LEFT WITHOUT BEING SEEN",
    "LWBS",
    "ELOPED",
    "LEFT AGAINST MEDICAL ADVICE",
    "AMA",
})

ED_DEATH_VALUES = frozenset({"EXPIRED", "DIED", "DEAD"})


def _normalize(value) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).strip().upper().split())
    return normalized or None


def normalize_admission_type(value) -> str | None:
    return _normalize(value)


def normalize_ed_disposition(value) -> str | None:
    return _normalize(value)


def is_observation_admission_type(value) -> bool:
    return normalize_admission_type(value) in OBSERVATION_ADMISSION_TYPES


def denominator_exclusion_reason(outcomes: dict, decision_status: str | None = "decision", icu_delta_hours=None) -> str | None:
    if decision_status != "decision":
        return "no_valid_modeled_disposition_decision"
    ed_disposition = normalize_ed_disposition(outcomes.get("ed_disposition", outcomes.get("disposition")))
    if ed_disposition in NO_MODELED_DISPOSITION_ED_VALUES:
        return "no_clinician_disposition_lwbs_eloped_or_ama"
    if outcomes.get("died_in_ed_before_decision") is True or ed_disposition in ED_DEATH_VALUES:
        return "died_in_ed_before_modeled_decision"
    if icu_delta_hours is not None and icu_delta_hours <= 0:
        return "icu_transfer_at_or_before_modeled_decision"
    return None


def classify_observed_disposition_detail(outcomes: dict, icu_transfer_after_decision: bool = False) -> dict:
    """Classify evaluation-only disposition without using dsp_hospitalization."""
    admission_type = normalize_admission_type(outcomes.get("admission_type"))
    ed_disposition = normalize_ed_disposition(outcomes.get("ed_disposition", outcomes.get("disposition")))
    hadm_id = outcomes.get("hadm_id")
    if icu_transfer_after_decision:
        return {"label": "ICU", "provenance": "checkpoint_anchored_icu_transfer"}
    if admission_type in OBSERVATION_ADMISSION_TYPES:
        return {"label": "observation", "provenance": "hosp.admissions.admission_type"}
    if hadm_id not in (None, "") and admission_type in TRUE_INPATIENT_ADMISSION_TYPES:
        return {"label": "ward", "provenance": "hadm_id_plus_true_inpatient_admission_type"}
    if ed_disposition == "HOME" and hadm_id in (None, ""):
        return {"label": "discharge", "provenance": "ed.disposition_HOME_plus_no_hadm_id"}
    return {"label": None, "provenance": "insufficient_or_noncanonical_observed_fields"}


def classify_observed_disposition(outcomes: dict, icu_transfer_after_decision: bool = False):
    return classify_observed_disposition_detail(outcomes, icu_transfer_after_decision)["label"]


def observation_semantics_metadata() -> dict:
    return {
        "source_fields": ["ed.disposition", "hadm_id", "hosp.admissions.admission_type"],
        "observation_admission_types": sorted(OBSERVATION_ADMISSION_TYPES),
        "true_inpatient_admission_types": sorted(TRUE_INPATIENT_ADMISSION_TYPES),
        "discharge_rule": "ed.disposition=HOME and no hadm_id",
        "hospitalization_flag_role": "ignored_for_disposition_boundaries",
        "hospitalization_trap": {
            "field": "dsp_hospitalization",
            "home_with_flag_reported": 36492,
            "of_these_observation_reported": 35953,
            "provenance": "Yana agent4_disposition_graph v0.3-draft",
        },
    }
