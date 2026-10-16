"""Narrow A1 literal checks against explicit unknown checkpoint support state.

This is not a respiratory assessment or a general semantic validator. Only
patient_finding assertions in the two evidence lists are inspected. Unsupported
grammar, other fields and clinical applicability remain unchecked.
"""
from __future__ import annotations

import re

CONTRACT = "a1_explicit_unknown_support_v1"
SUPPORT = r"(?:respiratory support|oxygen therapy|supplemental oxygen)"
NEGATIVE = re.compile(
    rf"(?:(?:(?:the patient has|there is)\s+)?no\s+{SUPPORT}"
    rf"|{SUPPORT}\s*(?::\s*none|is\s+(?:absent|none))"
    r"|(?:the patient is\s+)?(?:on|breathing)\s+room air)[.!]?", re.I
)
NONASSERTION = re.compile(
    r"\b(?:if|unless|whether|could|may|might|would|should|assume|assuming"
    r"|hypothetical|previous|previously|historical|historically|earlier|baseline"
    r"|initially|before|at triage|at T0|recommended|planned|ordered|consider)\b", re.I
)
PARENTHETICAL_UNKNOWN = re.compile(
    rf"(?:{SUPPORT}|room air)\s*[\[(]\s*(?:unknown|uncertain|unconfirmed"
    r"|unavailable|not documented|not recorded|not known|not confirmed|not verified)\b", re.I
)


def _unknown_support(patient, decision_time):
    if decision_time == "T0":
        return None, "outside_checkpoint_scope"
    if not isinstance(patient, dict):
        return None, "no_structured_patient_input"
    states = [key for key in ("checkpoint_state", "T1_state") if key in patient]
    if len(states) != 1:
        return None, "no_unique_current_state"
    key = states[0]
    state = patient[key]
    if not isinstance(state, dict):
        return None, "no_structured_current_state"
    oxygen = state.get("oxygen_requirement")
    if not isinstance(oxygen, dict):
        return None, "no_explicit_unknown_support"
    provenance = oxygen.get("provenance")
    if (not isinstance(provenance, dict) or provenance.get("source_type") != "unavailable"
            or "oxygen_therapy_documented" not in oxygen or "mode" not in oxygen
            or oxygen["oxygen_therapy_documented"] is not None or oxygen["mode"] is not None):
        return None, "no_explicit_unknown_support"
    return f"{key}.oxygen_requirement", "explicit_unknown_support"


def _negative_assertions(text):
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        # Prefer leaving ambiguous prose unchecked to interpreting its semantics.
        if (NONASSERTION.search(sentence) or PARENTHETICAL_UNKNOWN.search(sentence)
                or any(mark in sentence for mark in ('"', "'", "\x60", "\u201c", "\u201d"))):
            continue
        for clause in re.split(r"[,;()]|\s+(?:and|but)\s+", sentence, flags=re.I):
            clause = re.sub(r"\s+", " ", clause).strip()
            if NEGATIVE.fullmatch(clause):
                found.append(clause)
    return found


def fact_consistency_checks(reply, patient, decision_time="T0"):
    """Return auditable checks; not_checked must never mean factually verified."""
    if not isinstance(reply, dict):
        return []
    source_field, reason = _unknown_support(patient, decision_time)
    checks = []
    for group in ("supporting_evidence", "opposing_evidence"):
        items = reply.get(group)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not isinstance(item.get("patient_finding"), str):
                continue
            literals = _negative_assertions(item["patient_finding"]) if source_field else []
            checks.append({
                "contract": CONTRACT, "group": group, "index": index,
                "reply_field": f"{group}.{index}.patient_finding",
                "source_field": source_field, "status": "unsupported_negative" if literals else "not_checked",
                "reason": "explicit_unknown_as_negative" if literals else (
                    "no_supported_literal_assertion" if source_field else reason),
                "matched_assertions": literals,
            })
    return checks
