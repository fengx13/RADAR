"""A narrow, as-of projection of supplied antimicrobial records for Agent 3."""
from __future__ import annotations

import copy
import math
from typing import Any

from radar.pipeline.event_visibility import event_available_time, event_visible_at
from radar.common.integration_schema import strip_inference_forbidden


CONTRACT_VERSION = "agent3_observed_antimicrobials_v1"
RECORD_FIELDS = (
    "administration_time", "available_time", "documentation_time",
    "agent", "medication", "dose", "dose_unit", "route", "administered", "status",
)
PROVENANCE_FIELDS = (
    "source_type", "source_field", "source_file", "source_table", "source_record_id",
    "event_time", "available_time", "documentation_time",
)
RELATED_TIMES = ("event_time", "completion_time", "stop_time", "end_time", "assessment_time", "status_time")


def _scalar_fields(source: dict, fields: tuple[str, ...]) -> dict:
    return {
        key: copy.deepcopy(source[key])
        for key in fields if key in source
        and (source[key] is None or isinstance(source[key], (str, bool, int, float)))
        and (not isinstance(source[key], float) or math.isfinite(source[key]))
    }


def _visible_record(event: Any, cutoff: Any) -> dict | None:
    if not isinstance(event, dict):
        return None
    provenance = event.get("provenance")
    if not isinstance(provenance, dict):
        return None
    source_type = provenance.get("source_type")
    if not isinstance(source_type, str) or source_type not in {"observed_ehr", "synthetic_fixture"}:
        return None
    # The new projection must not inherit legacy event-time-only visibility.
    if not any(key in source for source in (event, provenance)
               for key in ("available_time", "documentation_time")):
        return None
    if not event_visible_at(event, "administration_time", cutoff):
        return None
    # Do not backdate a later completion/stop state carried on an earlier event.
    for source in (event, provenance):
        for key in RELATED_TIMES:
            if source.get(key) not in (None, "") and not event_visible_at(source, key, cutoff):
                return None
    record = _scalar_fields(event, RECORD_FIELDS)
    record["provenance"] = _scalar_fields(provenance, PROVENANCE_FIELDS)
    record["effective_available_time"] = event_available_time(event, "administration_time")
    return record


def build_observed_antimicrobials(observed: dict | None, cutoff: Any) -> dict:
    """Preserve eligible source records, not an inferred current regimen or plan."""
    events = observed.get("antimicrobials") if isinstance(observed, dict) else None
    records = []
    for event in events if isinstance(events, list) else []:
        record = _visible_record(event, cutoff)
        if record is not None:
            records.append(record)
    return strip_inference_forbidden({
        "schema_version": CONTRACT_VERSION,
        "information_cutoff": cutoff,
        "record_availability": "available" if records else "unavailable",
        "coverage": "supplied_antimicrobial_records_only",
        "records": records,
        "interpretation": (
            "Timestamped source records, not a current treatment plan. "
            "Preserve each reported status: recommended/ordered is not administered, "
            "and administered is not necessarily ongoing. Multiple records may differ; "
            "do not silently reconcile them. No eligible records means unknown, "
            "not no antimicrobial treatment. A negative record is limited to that "
            "record, not all treatment. No treatment response is inferred."
        ),
    }, exclude_temporal_audit=True)
