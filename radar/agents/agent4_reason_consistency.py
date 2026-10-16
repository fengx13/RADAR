"""Narrow checks of literal observed facts, never clinical interpretation.

Only complete, simple English assertions are checked. Unsupported wording,
comparisons, historical values and unit ambiguity remain explicitly unchecked.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from radar.agents.agent4_insufficiency import _MISSING, resolve


NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
LABELS = {
    "lactate_initial": ("lactate", "initial lactate", "lactate value"),
    "lactate_repeat": ("lactate", "repeat lactate", "lactate value"),
    "sbp": ("sbp", "systolic blood pressure"),
    "map": ("map", "mean arterial pressure"),
    "resp_rate": ("rr", "respiratory rate"),
    "spo2": ("spo2", "oxygen saturation"),
}
PENDING_STATES = {"pending", "not_yet_assessable", "unavailable"}
RESULT_STATES = {"normal", "abnormal", "positive", "negative", "resulted", "available"}


def _decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def _literal(subject, content):
    # Full clauses only: no inference about thresholds, trends, negation or
    # nearby numbers belonging to a different measurement.
    return re.compile(
        rf"(?:the\s+)?(?:current\s+)?(?:{subject})\s*"
        rf"(?:(?:is|was|equals)\s+|[:=]\s*)?{content}[.!]?", re.I
    )


def check_reason(field, detail, payload):
    result = {"field": field, "status": "not_checked", "reason": "unsupported_literal"}
    if not isinstance(field, str) or not isinstance(detail, str) or not detail.strip():
        return result
    value = resolve(payload, field)
    if value is _MISSING:
        return {**result, "reason": "unresolved_reference"}
    parts = field.split(".")
    parent = resolve(payload, ".".join(parts[:-1]))
    parent = parent if isinstance(parent, dict) else {}
    leaf = parts[-1]
    concept = parts[-2] if leaf in {"value", "status", "result_status"} and len(parts) > 1 else leaf
    labels = LABELS.get(concept, ())
    subjects = [re.escape(label) for label in labels]
    subjects.append(re.escape(field))
    clauses = re.split(r";|\n|(?<=[.!?])\s+(?=[A-Z])", detail.strip())
    checks = []
    for clause in clauses:
        clause = clause.strip()
        numeric = _literal("|".join(subjects), rf"(?P<number>{NUMBER})(?:\s+(?P<unit>[A-Za-z%][A-Za-z0-9/%^ .-]*?))?")
        match = numeric.fullmatch(clause)
        actual = _decimal(value)
        if match and actual is not None:
            unit = match.group("unit")
            source_unit = parent.get("unit")
            if unit and (not isinstance(source_unit, str) or unit.casefold().rstrip(". ") != source_unit.casefold().strip()):
                checks.append({"status": "not_checked", "reason": "unit_not_comparable"})
            else:
                equal = _decimal(match.group("number")) == actual
                checks.append({"status": "consistent" if equal else "contradiction", "reason": "numeric_literal"})
            continue

        status_value = value.get("status", value.get("result_status")) if isinstance(value, dict) else value
        if isinstance(status_value, str) and status_value in PENDING_STATES and (
            isinstance(value, dict) or leaf in {"status", "result_status"}
        ):
            status_subjects = subjects + [r"(?:test\s+)?result", r"status"]
            for label in labels:
                status_subjects.append(re.escape(label) + r"(?:\s+result|\s+status)?")
            status_match = _literal("|".join(status_subjects), r"(?P<status>normal|abnormal|positive|negative|resulted|available|pending|not_yet_assessable|unavailable)").fullmatch(clause)
            if status_match:
                claimed = status_match.group("status").lower()
                if claimed in RESULT_STATES:
                    checks.append({"status": "contradiction", "reason": "pending_result_literal"})
                elif claimed == status_value:
                    checks.append({"status": "consistent", "reason": "pending_result_literal"})
                else:
                    checks.append({"status": "not_checked", "reason": "unresolved_status_mapping"})

        # An exact quoted scalar assertion has no clinical vocabulary mapping.
        quoted = _literal(re.escape(field), r'["\'](?P<value>[^"\']+)["\']').fullmatch(clause)
        if quoted and isinstance(value, str):
            equal = quoted.group("value") == value
            checks.append({"status": "consistent" if equal else "contradiction", "reason": "quoted_scalar_literal"})

    for status in ("contradiction", "consistent", "not_checked"):
        found = next((check for check in checks if check["status"] == status), None)
        if found:
            return {**result, **found}
    return result


def reason_consistency_checks(reply, payload):
    if not isinstance(reply, dict):
        return []
    checks = []
    for group in ("key_reasons", "unresolved_risks"):
        for index, item in enumerate(reply.get(group) or []):
            if isinstance(item, dict):
                checks.append({"group": group, "index": index, **check_reason(item.get("field"), item.get("detail"), payload)})
    return checks
