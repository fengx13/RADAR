#!/usr/bin/env python3
"""Prove all four Table 3 tiers actually come out of the runner, at T0 and at a checkpoint.

Structural reachability in the graph JSON is not the same claim as the runner
emitting the tier. This drives the real run_agent1.run_case with forced node
answers and checks what comes back, at both decision times, on real cohort cases.

    python test_four_tiers.py
    python test_four_tiers.py --cases cohort100_cases.json
"""
from __future__ import annotations
from radar import paths

import argparse
import json
import pathlib
from types import SimpleNamespace

from radar.agents import run_agent1

HERE = pathlib.Path(__file__).resolve().parent
D = paths.DATA

# One answer path per tier. R1 yes = infection suspected; R2 yes routes to R3a,
# which is the only node that can exceed possible; R4 decides the shock flag.
PATHS = [
    ("unlikely_sepsis", False, {"R1": "yes", "R2": "yes", "R3a": "alternative_more_likely",   "R4": "no"}),
    ("possible_sepsis", False, {"R1": "yes", "R2": "yes", "R3a": "alternative_also_likely",   "R4": "no"}),
    ("probable_sepsis", False, {"R1": "yes", "R2": "yes", "R3a": "alternative_less_likely",   "R4": "no"}),
    ("definite_sepsis", False, {"R1": "yes", "R2": "yes", "R3a": "alternative_very_unlikely", "R4": "no"}),
    ("possible_sepsis", True,  {"R1": "yes", "R2": "yes", "R3a": "alternative_also_likely",   "R4": "yes"}),
    ("probable_sepsis", True,  {"R1": "yes", "R2": "yes", "R3a": "alternative_less_likely",   "R4": "yes"}),
    ("definite_sepsis", True,  {"R1": "yes", "R2": "yes", "R3a": "alternative_very_unlikely", "R4": "yes"}),
    # No organ dysfunction: R3b, whose ceiling is possible by design.
    ("possible_sepsis", False, {"R1": "yes", "R2": "no",  "R3b": "alternative_very_unlikely", "R4": "no"}),
]


def run(case, answers, decision_time):
    payload = dict(case)
    payload["mock_agent1_answers"] = {
        node: ({"answer": a, "leading_alternative": "none_identified"}
               if node.startswith("R3") else a)
        for node, a in answers.items()
    }
    args = SimpleNamespace(dry_run=True, model="test-stub", temperature=0.0, seed=7,
                           decision_time=decision_time, risk_signal="none", risk_signal_map={})
    return run_agent1.run_case(payload, None, args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="cohort100_cases.json")
    args = p.parse_args()

    cases = json.loads((D / args.cases).read_text(encoding="utf-8"))
    case = next(c for c in cases if c["truth"]["sep3"] == 1)
    print(f"graph v{run_agent1.GRAPH['version']}   driving {case['case_id']} "
          f"through {len(PATHS)} answer paths x 2 decision times\n")

    failures = []
    seen = {"T0": set(), "checkpoint": set()}

    for decision_time in ("T0", "checkpoint"):
        basis = "pre_test" if decision_time == "T0" else "post_test"
        print(f"--- {decision_time}  (evidence_basis should be {basis})")
        for want_tier, want_shock, answers in PATHS:
            got = run(case, answers, decision_time)
            res = got.get("result") or {}
            tier, shock = res.get("tier"), res.get("shock_concern")
            ok = tier == want_tier and bool(shock) == want_shock and res.get("evidence_basis") == basis
            seen[decision_time].add(tier)
            route = " ".join(f"{k}={v}" for k, v in answers.items())
            print(f"   {'ok ' if ok else 'FAIL'}  {tier:<16} shock={str(bool(shock)):<5} "
                  f"[{got.get('terminal')}]   {route}")
            if not ok:
                failures.append((decision_time, want_tier, want_shock, tier, shock,
                                 res.get("evidence_basis")))
        print()

    want = {"unlikely_sepsis", "possible_sepsis", "probable_sepsis", "definite_sepsis"}
    print("four tiers produced by the runner:")
    for decision_time in ("T0", "checkpoint"):
        missing = want - seen[decision_time]
        print(f"   {decision_time:<12} {len(seen[decision_time] & want)}/4"
              + (f"   MISSING {sorted(missing)}" if missing else "   all four"))
        if missing:
            failures.append((decision_time, "coverage", None, sorted(missing), None, None))

    print()
    if failures:
        print(f"{len(failures)} FAILURES")
        for f in failures:
            print("  ", f)
        raise SystemExit(1)
    print("PASS — every tier is produced at both decision times, "
          "and each run is stamped with the right evidence_basis")


if __name__ == "__main__":
    main()
