#!/usr/bin/env python3
"""Split direct identifiers out of a cohort file, and emit a shareable summary.

Why
---
MIMIC stay_ids are direct identifiers and the PhysioNet data use agreement
forbids redistributing them. This repository is currently public, so no file
that reaches git may contain one.

What this does
--------------
1. Rewrites the cohort file in place with `stay_id` removed and a stable
   `surrogate_id` (RADAR-001 ...) added. Every downstream run output is keyed
   on case_id / surrogate_id, so nothing further along the pipeline can leak an
   identifier even by accident.

2. Writes the crosswalk to a separate file that .gitignore excludes. That file
   is the only place the mapping exists, so keep it — without it the cohort
   cannot be traced back to MIMIC or rebuilt for a reviewer.

3. Writes an aggregate cohort description containing counts only, no patient
   rows. That one is safe to put in the paper or show outside the team.

The de-identified cohort file still holds chief complaints, vitals and
comorbidities, because the agent cannot run without them. It stays local. This
script removes the direct identifier; it does not make the file publishable.

Usage
-----
    python -m radar.cohort.deidentify_cohort
    python -m radar.cohort.deidentify_cohort --cases cohort100_cases.json --check
"""
from __future__ import annotations
from radar import paths

import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
D = paths.DATA


def load(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


def save(obj, name):
    (D / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {paths.DATA / name}")


def band(age):
    if age is None:
        return None
    if age >= 89:                       # HIPAA safe-harbour ceiling
        return "89+"
    return f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"


def summarise(cases):
    """Counts only. No patient rows, no free text, no dates."""
    by = lambda f: dict(collections.Counter(f(c) for c in cases))
    return {
        "n": len(cases),
        "by_stratum": by(lambda c: c["stratum"]),
        "by_sub_stratum": by(lambda c: c["sub_stratum"]),
        "by_age_band": by(lambda c: band(c["input"]["age"])),
        "by_sex": by(lambda c: c["input"]["sex"]),
        "sepsis3_positive": sum(c["truth"]["sep3"] for c in cases),
        "septic_shock": sum(c["truth"]["septic_shock"] for c in cases),
        "bedside_organ_dysfunction_at_triage":
            sum(c["truth"]["bedside_organ_dysfunction_at_triage"] for c in cases),
        "outcomes": {
            k: sum(1 for c in cases if c["truth"]["outcomes"].get(k))
            for k in ("dsp_hospitalization", "dsp_icu_transfer_12h",
                      "dsp_critical", "dsp_ed_revisit_3d", "dsp_mortality_inhospital")
        },
        "note": ("Aggregate counts derived from MIMIC-IV-ED. No patient-level rows, "
                 "identifiers, free text or dates. Safe to share."),
    }


def check(cases, name):
    bad = [c.get("case_id") for c in cases if "stay_id" in c]
    if bad:
        raise SystemExit(
            f"!! {name} still carries stay_id on {len(bad)} cases: {bad[:5]}\n"
            "   run without --check to strip them")
    missing = [c.get("case_id") for c in cases if not c.get("surrogate_id")]
    if missing:
        raise SystemExit(f"!! {name}: {len(missing)} cases have no surrogate_id")
    print(f"{name}: {len(cases)} cases, no stay_id present, every case has a surrogate id")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="cohort100_cases.json")
    p.add_argument("--crosswalk", default="cohort100_identifiers.json")
    p.add_argument("--summary", default="cohort100_summary.json")
    p.add_argument("--check", action="store_true",
                   help="verify the cohort is already de-identified; change nothing")
    args = p.parse_args()

    cases = load(args.cases)

    if args.check:
        check(cases, args.cases)
        return

    existing = {}
    cw = D / args.crosswalk
    if cw.exists():
        existing = {r["case_id"]: r for r in json.loads(cw.read_text(encoding="utf-8"))["mapping"]}

    crosswalk, moved = [], 0
    for i, c in enumerate(sorted(cases, key=lambda x: x["case_id"]), start=1):
        prior = existing.get(c["case_id"])
        c["surrogate_id"] = prior["surrogate_id"] if prior else f"RADAR-{i:03d}"
        stay = c.pop("stay_id", None) or (prior or {}).get("stay_id")
        if stay is None:
            raise SystemExit(f"!! {c['case_id']} has no stay_id here and none in the crosswalk")
        moved += "stay_id" not in (prior or {})
        crosswalk.append({"surrogate_id": c["surrogate_id"],
                          "case_id": c["case_id"], "stay_id": int(stay)})

    save(cases, args.cases)
    save({"warning": "DIRECT IDENTIFIERS — never commit, never share outside the DUA",
          "source": "MIMIC-IV-ED", "n": len(crosswalk), "mapping": crosswalk},
         args.crosswalk)
    save(summarise(cases), args.summary)

    print(f"\n{len(cases)} cases: stay_id moved out of the cohort file")
    print(f"  the mapping now lives only in {paths.DATA / args.crosswalk}")
    print(f"  {paths.DATA / args.summary} holds counts only and is safe to share")


if __name__ == "__main__":
    main()
