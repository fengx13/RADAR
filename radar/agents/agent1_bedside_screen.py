"""Deterministic bedside organ-dysfunction screen for Agent 1 node R2 at T0.

The R2 node carries a `t0_criterion` in the decision graph (altered mental status /
RR >= 22 or new hypoxia / SBP <= 100 or shock signs) that was never placed in the
prompt, so the model re-derived its own bar (SOFA components, vasopressors) and
misread boundary values (RR 22 read as below 22). This module compares the triage
vitals against the same thresholds `make_cohort100.py` uses for its
`bedside_organ_dysfunction_at_triage` proxy and hands the result to the model as
structured context. The model still answers R2: the screen states which criteria
are met, not whether the dysfunction is attributable to infection.

Thresholds are fixed here and must match make_cohort100.py.
"""
from __future__ import annotations

CRITERIA = (
    # field in triage_vitals, label, operator, threshold
    ("resp_rate", "resp_rate", ">=", 22),
    ("sbp", "sbp", "<=", 100),
    ("spo2", "spo2", "<", 92),
)
MENTAL_STATUS_FIELDS = ("mental_status", "gcs", "avpu", "altered_mental_status")


def _met(value, op, thr) -> bool:
    return {">=": value >= thr, "<=": value <= thr, "<": value < thr}[op]


def bedside_screen(patient_input: dict) -> dict:
    """Return the per-criterion screen for one patient input dict (already privacy-stripped)."""
    vitals = (patient_input or {}).get("triage_vitals") or {}
    items = []
    for field, label, op, thr in CRITERIA:
        value = vitals.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            status = "not_recorded"
        else:
            status = "met" if _met(value, op, thr) else "not_met"
        items.append({"field": label, "value": value, "criterion": f"{op} {thr}", "status": status})
    mental = None
    for f in MENTAL_STATUS_FIELDS:
        if f in vitals or f in (patient_input or {}):
            mental = vitals.get(f, (patient_input or {}).get(f))
            break
    items.append({"field": "mental_status", "value": mental,
                  "criterion": "altered", "status": "not_recorded" if mental is None else "recorded"})
    return {
        "source": "triage_vitals",
        "items": items,
        "any_met": any(i["status"] == "met" for i in items),
        "all_recorded": all(i["status"] != "not_recorded" for i in items),
    }


def prompt_block(screen: dict) -> str:
    rows = []
    for i in screen["items"]:
        val = "" if i["value"] is None else str(i["value"])
        status = {"met": "MET", "not_met": "not met", "not_recorded": "not recorded",
                  "recorded": "recorded — judge from the finding"}[i["status"]]
        rows.append(f"{i['field']:<14}{val:<8}(criterion {i['criterion']:<10}) {status}")
    return (
        "BEDSIDE ORGAN-DYSFUNCTION SCREEN (computed deterministically from the triage vitals above)\n"
        + "=" * 60 + "\n"
        + "\n".join(rows) + "\n"
        "These are the T0 criteria for this node. A criterion marked MET is bedside evidence of "
        "acute organ dysfunction; your task is to judge whether it is attributable to the suspected "
        "infection. Do not re-derive the thresholds, and do not require SOFA components, vasopressors "
        "or respiratory support at T0. A criterion marked 'not recorded' is missing information, "
        "not a normal finding."
    )
