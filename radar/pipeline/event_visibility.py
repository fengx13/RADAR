"""As-of checks for normalized observation and documentation timestamps.

Omitted availability preserves the legacy event-time contract; it is not proof
of documentation availability. New source adapters must supply available_time
explicitly, including null when unknown. Explicit unknowns never fall back.
"""
from __future__ import annotations

import math

from radar.common.integration_schema import _time_key


def _visibility_times(event, time_field):
    times = [event.get(time_field)]
    for source in (event, event.get("provenance") or {}):
        for field in ("available_time", "documentation_time"):
            if field in source:
                times.append(source[field])
    return times


def _finite_key(value):
    key = _time_key(value)
    return key if key is not None and math.isfinite(key[1]) else None


def event_visible_at(event, time_field, cutoff):
    boundary = _finite_key(cutoff)
    keys = [_finite_key(value) for value in _visibility_times(event, time_field)]
    return boundary is not None and all(
        key is not None and key[0] == boundary[0] and key[1] <= boundary[1] for key in keys
    )


def event_after_cutoff(event, time_field, cutoff):
    boundary = _finite_key(cutoff)
    keys = [_finite_key(value) for value in _visibility_times(event, time_field)]
    return boundary is not None and any(
        key is not None and key[0] == boundary[0] and key[1] > boundary[1] for key in keys
    )


def event_available_time(event, time_field):
    times = _visibility_times(event, time_field)
    keys = [_finite_key(value) for value in times]
    if any(key is None for key in keys) or len({key[0] for key in keys}) != 1:
        return None
    return max(zip(keys, times), key=lambda pair: pair[0][1])[1]
