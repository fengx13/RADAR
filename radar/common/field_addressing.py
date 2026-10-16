#!/usr/bin/env python3
"""Schema-grounded field addressing for Agent 4.

This module is deliberately non-clinical: it exposes only paths already present
in the inference-safe payload and repairs only explicitly registered,
unambiguous aliases.
"""
from __future__ import annotations
from radar import paths

import copy


FORBIDDEN_ROOTS = {"truth", "outcomes", "evaluation_only"}
FUTURE_OUTCOME_FIELDS = {
    "dsp_hospitalization",
    "dsp_icu_transfer_12h",
    "dsp_critical",
    "dsp_ed_revisit_3d",
    "dsp_mortality_inhospital",
    "dsp_mortality_30d",
    "observed_disposition",
    "icu_transfer_time",
}

LATER_ALLOWED_ROOTS = {
    "patient_state",
    "agent1",
    "agent2",
    "workup_state",
    "agent3",
    "treatment_state",
    "reconciliation",
}
T0_ALLOWED_ROOTS = {"t0_state", "agent1_context", "agent2"}

# Only aliases with one defensible destination belong here. A bare
# `investigation_status` is intentionally absent because one encounter may have
# many workup items and statuses.
EXACT_ALIASES = {
    "patient_state.treatment_response_flag": "treatment_state.treatment_response_flag",
}


def canonical_field_path(path: str | None) -> str | None:
    """Return the canonical path for an unambiguous legacy/model alias."""
    if not isinstance(path, str):
        return path
    if path in EXACT_ALIASES:
        return EXACT_ALIASES[path]
    if path.startswith("t1_state."):
        return "patient_state." + path[len("t1_state."):]
    return path


def _walk_paths(value, prefix: str, paths: set[str]) -> None:
    if prefix:
        paths.add(prefix)
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_ROOTS or key in FUTURE_OUTCOME_FIELDS or key.startswith("dsp_"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else key
            _walk_paths(child, child_prefix, paths)
    elif isinstance(value, list):
        # Lists are cited at their stable container path; numeric indices would
        # make field addresses depend on item ordering.
        return


def available_field_paths(payload: dict, *, decision_point: str = "checkpoint") -> list[str]:
    """List stable, inference-safe paths that Agent 4 may cite as evidence."""
    roots = T0_ALLOWED_ROOTS if decision_point == "T0" else LATER_ALLOWED_ROOTS
    paths: set[str] = set()
    for root in sorted(roots):
        if root in payload:
            _walk_paths(payload[root], root, paths)
    return sorted({canonical_field_path(path) for path in paths})


def canonicalize_reply_fields(reply: dict | None) -> tuple[dict | None, list[dict]]:
    """Rewrite only registered aliases and return an auditable repair list."""
    if not isinstance(reply, dict):
        return reply, []
    result = copy.deepcopy(reply)
    repairs: list[dict] = []
    items = []
    for group in ("key_reasons", "unresolved_risks", "blocking"):
        for item in result.get(group) or []:
            if isinstance(item, dict):
                items.append((group, item))
    basis = result.get("observation_basis")
    if isinstance(basis, dict):
        items.append(("observation_basis", basis))
    for group, item in items:
        original = item.get("field")
        canonical = canonical_field_path(original)
        if canonical != original:
            item["field"] = canonical
            repairs.append({"group": group, "original": original, "canonical": canonical})
    return result, repairs


def field_registry_metadata() -> list[dict]:
    """Small human/machine-readable registry of cross-layer A4 concepts."""
    return [
        {
            "clinical_concept": "observed treatment response state",
            "canonical_field_path": "treatment_state.treatment_response_flag",
            "aliases": ["patient_state.treatment_response_flag"],
            "source_layer": "observed treatment state",
            "t0_available": False,
            "checkpoint_available": True,
            "allowed_for_blocking": True,
            "allowed_for_evidence_citation": True,
        },
        {
            "clinical_concept": "checkpoint treatment-response convenience value",
            "canonical_field_path": "patient_state.treatment_response",
            "aliases": ["t1_state.treatment_response"],
            "source_layer": "checkpoint state mirror",
            "t0_available": False,
            "checkpoint_available": True,
            "allowed_for_blocking": True,
            "allowed_for_evidence_citation": True,
        },
        {
            "clinical_concept": "pending observed workup",
            "canonical_field_path": "workup_state.pending_workup",
            "aliases": [],
            "source_layer": "observed workup state",
            "t0_available": False,
            "checkpoint_available": True,
            "allowed_for_blocking": True,
            "allowed_for_evidence_citation": True,
        },
        {
            "clinical_concept": "individual investigation lifecycle",
            "canonical_field_path": "workup_state.items",
            "aliases": [],
            "rejected_ambiguous_aliases": ["investigation_status"],
            "source_layer": "observed workup state",
            "t0_available": False,
            "checkpoint_available": True,
            "allowed_for_blocking": False,
            "allowed_for_evidence_citation": True,
        },
    ]
