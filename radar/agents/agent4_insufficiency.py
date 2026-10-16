#!/usr/bin/env python3
"""Agent 4 information-insufficiency validation and routing.

This module implements Yana's V4 rules for insufficient_evidence:

V4a: the blocking field must be allowed for the node.
V4b: the claimed missing field must actually be empty or absent.
V4c: PENDING information must have a decision-changing counterfactual.

Routing:
- MUST_EXIST missing -> ABSTAIN + pipeline_alarm
- PENDING + valid decision-changing counterfactual -> OBSERVATION
- NEVER_AVAILABLE cannot be used as a blocking reason

The uncertainty ceiling prevents discharge after any valid insufficient-evidence
state has been carried through the graph.
"""
from __future__ import annotations

from radar.common.field_addressing import canonical_field_path

PENDING = {
    "patient_state.labs.culture_result": "blood culture result, 24-48h",
    "patient_state.labs.lactate_repeat": "repeat lactate, approximately 2h",
    "patient_state.labs.imaging_result": "imaging result",
    "patient_state.treatment_response": "response to initial treatment, time-dependent",
    "treatment_state.treatment_response_flag": "observed response state, time-dependent",
    "patient_state.vitals_trend": "vital-sign trend, requires repeated measurement",
    "workup_state.pending_workup": "observed workup ordered but not resulted",
}

PENDING_POLICY = {
    "patient_state.labs.culture_result": {
        "disposition_relevant": True,
        "checkpoint_eligible": False,
        "reason": "final culture may remain an unresolved risk but must not schedule a 24-48h ED checkpoint",
        "provenance": "Yana authoritative Agent 4 pending policy",
    },
    "patient_state.labs.lactate_repeat": {
        "disposition_relevant": True,
        "checkpoint_eligible": True,
        "reason": "a newly available repeat lactate may change disposition during the ED pathway",
        "provenance": "Yana authoritative Agent 4 pending policy",
    },
    "patient_state.labs.imaging_result": {
        "disposition_relevant": True,
        "checkpoint_eligible": True,
        "reason": "a newly available imaging result may change disposition during the ED pathway",
        "provenance": "Yana authoritative Agent 4 pending policy",
    },
    "patient_state.treatment_response": {
        "disposition_relevant": True,
        "checkpoint_eligible": True,
        "reason": "observed treatment response may change disposition",
        "provenance": "Yana authoritative Agent 4 pending policy",
    },
    "treatment_state.treatment_response_flag": {
        "disposition_relevant": True,
        "checkpoint_eligible": True,
        "reason": "a pending observed treatment-response state may change disposition",
        "provenance": "RADAR vNext observed treatment-state contract",
    },
    "patient_state.vitals_trend": {
        "disposition_relevant": True,
        "checkpoint_eligible": True,
        "reason": "a clinically relevant new vital-sign trend may change disposition",
        "provenance": "Yana authoritative Agent 4 pending policy",
    },
    "workup_state.pending_workup": {
        "disposition_relevant": True,
        "checkpoint_eligible": None,
        "reason": "eligibility depends on the underlying ordered item and must remain unresolved when metadata are insufficient",
        "provenance": "RADAR vNext workup-state ownership adapted from Yana policy",
    },
}

VALID_DISPOSITIONS = {"discharge", "observation", "ward", "ICU"}

MUST_EXIST = {
    "patient_state.vitals.sbp": "checkpoint assembly should provide this",
    "patient_state.vitals.map": "checkpoint assembly should provide this",
    "patient_state.vitals.resp_rate": "checkpoint assembly should provide this",
    "patient_state.vitals.spo2": "checkpoint assembly should provide this",
    "patient_state.oxygen_requirement": "checkpoint assembly should provide this",
    "patient_state.labs.lactate_initial": "field/status must exist structurally at the checkpoint",
    "agent1.tier": "upstream Agent 1 must provide this",
    "agent1.shock_concern": "upstream Agent 1 must provide this",
    "treatment_state.ongoing_treatment": "observed treatment-state layer must provide this",
    "treatment_state.treatment_response_flag": "observed treatment-state layer must provide this",
}

NEVER_AVAILABLE = {
    "patient_state.mental_status": "not available in this MIMIC-IV-ED deployment",
    "patient_state.gcs": "not available in this deployment; retrospective proxy must not enter inference",
}

CANONICAL_PENDING = set(PENDING)
CANONICAL_MUST_EXIST = set(MUST_EXIST)
CANONICAL_NEVER_AVAILABLE = set(NEVER_AVAILABLE)

# Legacy standalone Agent 4 fixtures used t1_state. Keep path aliases at the
# compatibility boundary while new checkpoint payloads use patient_state.
PENDING.update({key.replace("patient_state.", "t1_state."): value for key, value in list(PENDING.items()) if key.startswith("patient_state.")})
MUST_EXIST.update({key.replace("patient_state.", "t1_state."): value for key, value in list(MUST_EXIST.items()) if key.startswith("patient_state.")})
NEVER_AVAILABLE.update({key.replace("patient_state.", "t1_state."): value for key, value in list(NEVER_AVAILABLE.items()) if key.startswith("patient_state.")})

BLOCKABLE = set(PENDING) | set(MUST_EXIST)

_MISSING = object()


def deployment_unavailable_fields(payload: dict | None) -> set:
    """Fields the run declares the data source cannot provide (e.g. oxygen therapy
    in MIMIC-IV-ED). Declared per run, never inferred. Like NEVER_AVAILABLE they
    cannot be a blocking reason; unlike it they are listed explicitly in the
    payload so the trace records that the deployment lacked them."""
    fields = (payload or {}).get("deployment_unavailable_fields") or []
    return {canonical_field_path(f) for f in fields if isinstance(f, str)}


def resolve(payload: dict, path: str):
    """Resolve a dotted path. Missing path returns _MISSING; None is still empty."""
    cur = payload
    parts = canonical_field_path(path).split(".")
    if parts and parts[0] not in cur:
        if parts[0] == "patient_state" and "t1_state" in cur:
            parts[0] = "t1_state"
        elif parts[0] == "t1_state" and "patient_state" in cur:
            parts[0] = "patient_state"
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def is_empty(value) -> bool:
    return value is _MISSING or value is None or value == "" or value == [] or value == {}


def is_pending_or_empty(value) -> bool:
    if is_empty(value):
        return True
    if isinstance(value, str):
        return value in {"pending", "not_yet_assessable", "unavailable"}
    if not isinstance(value, dict):
        return False
    status = value.get("status", value.get("result_status"))
    return status in {"pending", "not_yet_assessable", "unavailable"} and value.get("value") in (None, "")


def pending_evidence(field: str, value, payload: dict) -> bool | None:
    """Check observed availability, not whether waiting is clinically appropriate.

    None preserves legacy unspecified-field semantics. A structured workup
    record or explicit unavailable status must not be turned into a pending order.
    """
    workup = payload.get("workup_state") or {}
    items = workup.get("items") if isinstance(workup, dict) else None
    observed_contract = isinstance(items, list) and (
        workup.get("status") in {"valid", "invalid"}
        or any(isinstance(item, dict) and ("actual_order_status" in item or "result_status" in item)
               for item in items)
    )

    def outstanding(item):
        return (
            isinstance(item, dict)
            and item.get("actual_order_status") in {"ordered", "collected", "pending", "resulted"}
            and item.get("result_status") in {"pending", "unavailable", "not_yet_assessable"}
        )

    if field == "workup_state.pending_workup" and isinstance(value, list):
        if not value:
            return False
        if observed_contract:
            observed = {item["item"] for item in items
                        if outstanding(item) and isinstance(item.get("item"), str)}
            return all(isinstance(name, str) and name in observed for name in value)
        return True

    # These names are the same field-to-item mappings used by checkpoint assembly.
    lab_items = {
        "patient_state.labs.culture_result": "blood_cultures",
        "patient_state.labs.imaging_result": "source_imaging_or_sampling",
        "patient_state.labs.lactate_repeat": "lactate_repeat",
    }
    if field in lab_items and observed_contract:
        return any(outstanding(item) and item.get("item") == lab_items[field] for item in items)

    status = value.get("status", value.get("result_status")) if isinstance(value, dict) else value
    if isinstance(status, str) and status == "unavailable":
        return False
    if field == "patient_state.treatment_response":
        response = resolve(payload, "treatment_state.treatment_response_flag")
        if response == "unavailable":
            return False
    return None


def is_unresolved_field(field: str, value, payload: dict | None = None) -> bool:
    if field in CANONICAL_PENDING:
        evidence = pending_evidence(field, value, payload or {})
        if field == "workup_state.pending_workup" and isinstance(value, list):
            return evidence is True
        if evidence is False:
            return False
        return is_pending_or_empty(value)
    if is_empty(value):
        return True
    if field == "patient_state.oxygen_requirement" and isinstance(value, dict):
        # Source metadata cannot turn an explicitly unknown oxygen state into
        # a measurement. False, zero, and partially known states remain present.
        metadata = {"source", "provenance"}
        clinical = {key: item for key, item in value.items() if key not in metadata}
        return all(is_empty(item) for item in clinical.values())
    return False


def field_availability(payload: dict, fields) -> list[dict]:
    """Expose the same V4b facts to the model and validator; not a disposition."""
    rows = []
    unavailable = deployment_unavailable_fields(payload)
    for field in sorted(fields):
        value = resolve(payload, field)
        row = {
            "field": field,
            "field_exists": value is not _MISSING,
            "v4b_unresolved": is_unresolved_field(field, value, payload),
            "observed_pending": pending_evidence(field, value, payload) if field in CANONICAL_PENDING else None,
        }
        if field in unavailable:
            row["v4b_unresolved"] = False
            row["deployment_unavailable"] = True
        rows.append(row)
    return rows


def canonical_field(path: str | None) -> str | None:
    return canonical_field_path(path)


def pending_policy_for(path: str | None) -> dict | None:
    policy = PENDING_POLICY.get(canonical_field(path))
    if policy is None:
        return None
    return {"field": canonical_field(path), **policy}


def validate_insufficiency(reply: dict, node: dict, payload: dict):
    """Validate insufficient_evidence. Return (ok, failures)."""
    failures = []
    blocking = reply.get("blocking")
    if not blocking or not isinstance(blocking, list):
        return False, ["V4_no_blocking_field"]

    allowed = {canonical_field(field) for field in node.get("blockable_fields", BLOCKABLE)}
    for item in blocking:
        if not isinstance(item, dict):
            failures.append("V4_invalid_blocking_item")
            continue
        field = (item or {}).get("field")
        if not isinstance(field, str):
            failures.append("V4a_blocking_field_not_string")
            continue
        canonical = canonical_field(field)

        if canonical in CANONICAL_NEVER_AVAILABLE:
            failures.append(f"V4a_never_available:{field}")
            continue
        if canonical in deployment_unavailable_fields(payload):
            failures.append(f"V4a_deployment_unavailable:{field}")
            continue
        if canonical not in allowed:
            failures.append(f"V4a_not_blockable:{field}")
            continue

        value = resolve(payload, field)
        unresolved = is_unresolved_field(canonical, value, payload)
        if not unresolved:
            if canonical in CANONICAL_PENDING and pending_evidence(canonical, value, payload) is False:
                failures.append(f"V4b_pending_not_observed:{field}")
            else:
                failures.append(f"V4b_field_is_present:{field}={value!r}")

        if canonical in CANONICAL_PENDING:
            policy = pending_policy_for(canonical)
            if not policy or policy.get("disposition_relevant") is not True:
                failures.append(f"V4c_pending_not_disposition_relevant:{field}")
            counterfactual = (item or {}).get("counterfactual")
            counterfactual = counterfactual if isinstance(counterfactual, dict) else {}
            branches = []
            for branch_name in ("if_a", "if_b"):
                branch = counterfactual.get(branch_name)
                branch = branch if isinstance(branch, dict) else {}
                condition = branch.get("condition")
                if not isinstance(condition, str) or not condition.strip():
                    failures.append(f"V4c_empty_counterfactual_condition:{field}:{branch_name}")
                branches.append(branch)
            if_a, if_b = branches
            disp_a = if_a.get("disposition")
            disp_b = if_b.get("disposition")
            if not disp_a or not disp_b:
                failures.append(f"V4c_no_counterfactual:{field}")
            elif (not isinstance(disp_a, str) or not isinstance(disp_b, str)
                  or disp_a not in VALID_DISPOSITIONS or disp_b not in VALID_DISPOSITIONS):
                failures.append(f"V4c_counterfactual_disposition_not_allowed:{field}")
            elif disp_a == disp_b:
                failures.append(f"V4c_counterfactual_not_decisive:{field}")

    return not failures, failures


def route_insufficiency(reply: dict):
    """Route after V4 validation passes. Return (outcome, detail)."""
    fields = [item["field"] for item in reply["blocking"]]
    must_missing = [
        field for field in fields
        if canonical_field(field) in CANONICAL_MUST_EXIST
        and canonical_field(field) not in CANONICAL_PENDING
    ]
    if must_missing:
        kind = (
            "upstream_missing"
            if any(
                field.startswith(("agent1.", "agent2.", "agent3.", "workup_state.", "treatment_state."))
                for field in must_missing
            )
            else "record_missing"
        )
        return "ABSTAIN", {
            "abstain_reason": kind,
            "missing_information": must_missing,
            "pipeline_alarm": True,
        }
    return "OBSERVATION", {
        "observation_basis": "pending_resolvable",
        "awaiting": fields,
        "pending_items": [pending_policy_for(field) for field in fields],
        "counterfactuals": [item.get("counterfactual") for item in reply["blocking"]],
    }


def apply_uncertainty_ceiling(terminal: str, carried: int) -> tuple[str, str | None]:
    """Any carried valid insufficient-evidence state prevents final discharge."""
    if carried > 0 and terminal == "T_DISCHARGE":
        return "T_OBSERVATION", (
            f"uncertainty_ceiling: {carried} valid insufficient_evidence state(s) "
            "carried through the graph; discharge downgraded to observation"
        )
    return terminal, None
