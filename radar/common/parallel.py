"""Run cases concurrently without changing what any runner decides.

The pipeline is network-bound: a node call is roughly six seconds of waiting on
the API and almost no local work. Running cases in parallel therefore turns
hours into minutes, and the number of workers is limited by the API rate limit
rather than by cores.

Guarantees this makes, because a clinical result set is worthless without them:

  * Output order is the input order. Results are placed by index, not appended
    as they finish, so two runs of the same cohort line up row for row.
  * Each case is independent. Runners take (case, client, args) and return a
    fresh dict; nothing is shared but the OpenAI client, which is thread-safe.
  * A case that raises returns a runner_error record instead of killing the run,
    so one bad case cannot cost you the other ninety-nine.
  * workers=1 runs the plain sequential loop, so the old behaviour is still
    reachable exactly.
"""
from __future__ import annotations

import concurrent.futures
import sys
import threading

_print_lock = threading.Lock()


def add_workers_argument(parser, default=1):
    parser.add_argument(
        "--workers", type=int, default=default,
        help="cases to run concurrently (network-bound; 8 is a good starting point)")
    return parser


def say(message):
    """Print from a worker thread without interleaving with another."""
    with _print_lock:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.write(
            str(message).encode(encoding, errors="backslashreplace").decode(encoding) + "\n")
        sys.stdout.flush()


def run_cases(cases, run_one, workers=1, label=lambda c: c.get("case_id"), progress=True):
    """Apply run_one to every case, in parallel, preserving input order.

    run_one takes one case and returns its result dict.
    """
    total = len(cases)
    results = [None] * total

    def guarded(index, case):
        try:
            return index, run_one(case)
        except Exception as exc:                                  # noqa: BLE001
            return index, {"case_id": label(case),
                           "status": "runner_error",
                           "error": f"{type(exc).__name__}: {exc}"}

    if workers <= 1:
        for index, case in enumerate(cases):
            _, results[index] = guarded(index, case)
            if progress:
                say(f"  [{index + 1}/{total}] {label(case)}")
        return results

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guarded, i, c) for i, c in enumerate(cases)]
        for future in concurrent.futures.as_completed(futures):
            index, result = future.result()
            results[index] = result
            done += 1
            if progress:
                say(f"  [{done}/{total}] {label(cases[index])}")
    return results


def error_count(results):
    return sum(1 for r in results if isinstance(r, dict) and r.get("status") == "runner_error")
