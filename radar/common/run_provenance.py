#!/usr/bin/env python3
"""Small, non-clinical provenance helpers for reproducible model runs."""
from __future__ import annotations

import datetime as dt
import os
import pathlib
import subprocess
import uuid


def _git_state(root: pathlib.Path) -> dict:
    override = os.environ.get("RADAR_CODE_COMMIT")
    if override:
        return {
            "code_commit": override,
            "code_commit_source": "RADAR_CODE_COMMIT",
            "code_dirty": None,
        }
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip())
        return {
            "code_commit": commit or None,
            "code_commit_source": "git",
            "code_dirty": dirty,
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "code_commit": None,
            "code_commit_source": "unavailable",
            "code_dirty": None,
        }


def new_run_metadata(prefix: str, root: pathlib.Path) -> dict:
    """Return a unique run id, UTC timestamp, and honest code state."""
    now = dt.datetime.now(dt.timezone.utc)
    timestamp = now.isoformat().replace("+00:00", "Z")
    compact = now.strftime("%Y%m%dT%H%M%SZ")
    return {
        "run_id": f"{prefix}-{compact}-{uuid.uuid4().hex[:8]}",
        "timestamp_utc": timestamp,
        **_git_state(root),
    }


def model_runtime_metadata(module) -> dict:
    """Read response metadata collected by an agent module, if available."""
    return {
        "model_snapshots": sorted(getattr(module, "MODEL_SNAPSHOTS", set())),
        "system_fingerprints": sorted(getattr(module, "FINGERPRINTS", set())),
        "usage": getattr(module, "USAGE", None),
    }
