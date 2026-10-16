"""Non-clinical cohort execution helpers shared by the dynamic CLI."""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import tempfile

from radar.common.parallel import run_cases


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def run_tag(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError("run tag must contain only letters, digits, '-' or '_'")
    return value


def add_batch_arguments(parser):
    parser.add_argument("--workers", type=positive_int, default=1,
                        help="concurrent patients; checkpoints within each patient remain sequential")
    parser.add_argument("--batch-size", type=positive_int, default=None,
                        help="persist the cumulative output after each group of patients")
    parser.add_argument("--max-batches", type=positive_int, default=None,
                        help="limit groups for an explicitly partial pilot; requires --batch-size")
    parser.add_argument("--run-tag", type=run_tag, default=None,
                        help="append a label to the output filename without replacing earlier runs")
    parser.add_argument("--force", action="store_true", help="explicitly allow replacing the selected output")
    return parser


def validate_longitudinal_cases(cases):
    if not cases:
        raise ValueError("no longitudinal cases supplied")
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"case {index} needs a unique nonempty case_id")
        seen.add(case_id)
        if not isinstance(case.get("t0_state"), dict):
            raise ValueError(
                f"{case_id}: sequential mode requires t0_state and longitudinal observations; "
                "a cohort input or T1/T_SOFA snapshot is not a longitudinal case. "
                "Supply the longitudinal_checkpoint_contract or explicitly select legacy_cohort_two_time."
            )
        if "checkpoint_state" in case["t0_state"] or "available_checkpoints" in case["t0_state"]:
            raise ValueError(f"{case_id}: a later checkpoint snapshot must not be used as t0_state")
        for field, expected in (("observed_workup", list), ("longitudinal_events", dict),
                                ("observed_treatment", dict)):
            if field in case and not isinstance(case[field], expected):
                raise ValueError(f"{case_id}: {field} must be a {expected.__name__}")


def output_destination(out, infile, tag=None, force=False):
    destination = pathlib.Path(out)
    if tag:
        destination = destination.with_name(f"{destination.stem}_{run_tag(tag)}{destination.suffix}")
    if destination.resolve() == pathlib.Path(infile).resolve():
        raise ValueError("output must not replace the input, even with --force")
    if destination.exists() and not force:
        raise ValueError(f"output already exists: {destination}; use a new --run-tag or explicit --force")
    return destination


def write_output(destination, output):
    """Publish a complete JSON snapshot without truncating the last completed batch."""
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=destination.parent, delete=False) as handle:
            temporary = pathlib.Path(handle.name)
            json.dump(output, handle, ensure_ascii=False, indent=1)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_batches(cases, run_one, *, workers=1, batch_size=None, max_batches=None,
                on_batch=None, progress=True):
    size = batch_size or len(cases)
    if workers < 1 or size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("workers, batch size and batch limit must be positive")
    if max_batches is not None and batch_size is None:
        raise ValueError("--max-batches requires --batch-size")
    results = []
    for batch_index, start in enumerate(range(0, len(cases), size), 1):
        if max_batches is not None and batch_index > max_batches:
            break
        batch = run_cases(cases[start:start + size], run_one, workers=workers, progress=progress)
        results.extend(batch)
        if on_batch is not None:
            on_batch(results, batch_index, len(batch))
    return results
