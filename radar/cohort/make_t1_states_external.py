#!/usr/bin/env python3
"""Build the reassessment states for the external cohort, from the raw ED record.

The curated laboratory panel for this source covers the sepsis stratum alone,
which would leave the external reassessment with no negative cases and so no
specificity. The raw emergency-department record covers every visit and carries
an explicit result time per component, so the same two checkpoints can be built
for all four strata.

Same definitions as make_t1_states.py, so the internal and external numbers mean
the same thing:

    T1      the moment the first ordered investigation actually returns a result
    T_SOFA  the first moment platelets, bilirubin and creatinine have all resulted

Result time is the recorded result time, not the order time. A result a
clinician cannot yet see is not available.

Usage
-----
    python -m radar.cohort.make_t1_states_external
    python -m radar.cohort.make_t1_states_external --limit 10
"""
from __future__ import annotations
from radar import paths

import argparse
import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
D = paths.DATA

DATA_ROOTS = [paths.RAW_ROOT]
SUBDIR = "raw/mc-med-1.0.0"

# Component names as this source spells them. Several analytes appear under both
# a laboratory and a point-of-care name; both are taken and the earliest result
# is kept, because a point-of-care lactate is a result the clinician can act on.
COMPONENTS = {
    "PLATELET COUNT (PLT)": "platelets",
    "BILIRUBIN, TOTAL": "bilirubin_total",
    "CREATININE": "creatinine",
    "POC:CREATININE,ISTAT": "creatinine",
    "POC:LACTATE, ISTAT": "lactate",
    "LACTATE.WHOLE BLD": "lactate",
    "POC:PO2 (A), ISTAT": "pao2",
    "PO2 (A)": "pao2",
    "POC:PO2, ISTAT": "pao2",
    "WHITE BLOOD CELLS (WBC)": "wbc",
    "HEMOGLOBIN": "hemoglobin",
    "SODIUM": "sodium",
    "POTASSIUM": "potassium",
    "BICARBONATE": "bicarbonate",
    "ANION GAP": "anion_gap",
    "GLUCOSE": "glucose",
    "BUN": "bun",
    "UREA NITROGEN": "bun",
    "INR": "inr",
}

SOFA_CORE = ("platelets", "bilirubin_total", "creatinine")
SOFA_COMPONENTS = SOFA_CORE + ("pao2",)



def iso_dt(series):
    """Parse this source's ISO timestamps at second resolution.

    The dates are shifted into the 23rd century, beyond the pandas nanosecond
    range (year 2262), so pd.to_datetime(errors="coerce") silently returns NaT
    for about a third of the rows. numpy datetime64[s] has no such limit.
    """
    import numpy as np
    text = series.astype("string").str.replace("Z", "", regex=False).str.replace("T", " ", regex=False)
    values = np.array([x if isinstance(x, str) and x else "NaT" for x in text], dtype="datetime64[s]")
    return pd.Series(values, index=series.index)


def find_data(explicit=None):
    for root in ([pathlib.Path(explicit)] if explicit else DATA_ROOTS):
        for candidate in (root / SUBDIR, root):
            if (candidate / "labs.csv").exists():
                return candidate
    raise SystemExit("labs.csv not found; pass --data-root pointing at 00_Data")


def load(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


def save(obj, name):
    (D / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {paths.DATA / name}")


def sofa_scores(labs, vitals):
    """Same subscores as the development extractor. A component with no result is
    reported missing, never scored 0, which would read as a healthy organ."""
    def band(value, cuts):
        if value is None:
            return None
        return sum(1 for c in cuts if value >= c) if cuts[0] < cuts[-1] else \
               sum(1 for c in cuts if value <= c)

    plt = labs.get("platelets", {}).get("value")
    bil = labs.get("bilirubin_total", {}).get("value")
    cre = labs.get("creatinine", {}).get("value")
    scores = {
        "coag": band(plt, [150, 100, 50, 20]),
        "liver": band(bil, [1.2, 2.0, 6.0, 12.0]),
        "renal": band(cre, [1.2, 2.0, 3.5, 5.0]),
        "resp": None,                       # needs FiO2 to form PaO2/FiO2
        "cardio": (1 if (vitals.get("map") is not None and vitals["map"] < 70)
                   else (0 if vitals.get("map") is not None else None)),
    }
    missing = [k for k, v in scores.items() if v is None]
    return scores, sum(v for v in scores.values() if v is not None), missing


def stream_labs(data_root, stays, chunksize):
    keep, seen = [], 0
    reader = pd.read_csv(data_root / "labs.csv",
                         usecols=["CSN", "Order_time", "Result_time", "Component_name",
                                  "Component_value"],
                         chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        seen += len(chunk)
        hit = chunk[chunk.CSN.isin(stays) & chunk.Component_name.isin(COMPONENTS)]
        if len(hit):
            keep.append(hit)
        if i % 5 == 0:
            print(f"    labs: {seen:,} rows scanned, {sum(len(k) for k in keep):,} kept",
                  flush=True)
    out = pd.concat(keep) if keep else pd.DataFrame()
    print(f"    labs: {seen:,} rows scanned, {len(out):,} kept")
    if len(out):
        out["Result_time"] = iso_dt(out.Result_time)
        out = out[out.Result_time.notna()]
        out["analyte"] = out.Component_name.map(COMPONENTS)
        out["value"] = pd.to_numeric(out.Component_value, errors="coerce")
    return out


def build(args):
    data_root = find_data(args.data_root)
    print(f"reading the raw external record from {data_root}")

    cases = load(args.cases)
    cw = D / args.crosswalk
    if not cw.exists():
        raise SystemExit(f"{cw} not found; it holds the stay_id mapping and is gitignored")
    stay_of = {r["case_id"]: int(r["stay_id"])
               for r in json.loads(cw.read_text(encoding="utf-8"))["mapping"]}
    if args.limit:
        cases = cases[:args.limit]
    stays = {stay_of[c["case_id"]] for c in cases if c["case_id"] in stay_of}

    visits = pd.read_csv(data_root / "visits.csv", usecols=["CSN", "Arrival_time"],
                         low_memory=False)
    visits = visits[visits.CSN.isin(stays)].copy()
    visits["Arrival_time"] = iso_dt(visits.Arrival_time)
    arrival = dict(zip(visits.CSN, visits.Arrival_time))
    print(f"  {len(cases)} cases, {len(stays)} visits")

    print("  scanning the laboratory record ...")
    labs = stream_labs(data_root, stays, args.chunksize)

    t1_out, sofa_out, no_result = [], [], []
    for case in cases:
        cid = case["case_id"]
        sid = stay_of.get(cid)
        if sid is None or pd.isna(arrival.get(sid)):
            no_result.append(cid)
            continue
        intime = arrival[sid]
        mine = labs[(labs.CSN == sid) & (labs.Result_time >= intime)] if len(labs) else labs
        if not len(mine):
            no_result.append(cid)
            continue

        vitals = dict(case["input"]["triage_vitals"])
        if vitals.get("sbp") is not None and vitals.get("dbp") is not None:
            vitals["map"] = round((vitals["sbp"] + 2 * vitals["dbp"]) / 3.0, 1)

        def state_at(cut, checkpoint_id, definition):
            got = mine[mine.Result_time <= cut].sort_values("Result_time")
            lab_state = {}
            for _, r in got.iterrows():
                if r.analyte in lab_state:
                    continue
                lab_state[r.analyte] = {
                    "value": None if pd.isna(r.value) else float(r.value),
                    "result_available_hours_after_arrival":
                        round((r.Result_time - intime).total_seconds() / 3600.0, 3),
                }
            scores, total, missing = sofa_scores(lab_state, vitals)
            hours = round((cut - intime).total_seconds() / 3600.0, 3)
            pending = sorted(set(mine[mine.Result_time > cut].analyte) - set(lab_state))
            return {
                "checkpoint_id": checkpoint_id,
                "definition": definition,
                "information_cutoff": f"{checkpoint_id} = {hours} h after ED arrival",
                "hours_after_arrival": hours,
                "labs_available_at_cutoff": lab_state,
                "cultures_at_cutoff": {
                    "blood_culture_resulted": False,
                    "any_culture_resulted": False,
                    "organisms_reported": [],
                    "source": "microbiology not extracted for the external cohort",
                },
                "still_pending_at_cutoff": {"labs_ordered_not_yet_resulted": pending,
                                            "cultures_pending": None},
                "vitals_at_cutoff": vitals,
                "sofa": {
                    "components_available": sorted(set(SOFA_COMPONENTS) & set(lab_state)),
                    "components_missing": sorted(set(SOFA_COMPONENTS) - set(lab_state)),
                    "usable_for_sofa_ge_2": not (set(SOFA_CORE) - set(lab_state)),
                    "respiratory_gas_available": "pao2" in lab_state,
                    "subscores": scores,
                    "total_from_available_components": total,
                    "not_scored": missing,
                    "note": ("Computed in code from results present at this cutoff. A component "
                             "with no result is reported missing, never scored 0."),
                },
            }

        t1 = mine.Result_time.min()
        rec = dict(case)
        rec["input"] = dict(case["input"])
        rec["input"]["checkpoint_state"] = state_at(
            t1, "T1", "first ordered investigation to actually return a result")
        rec["input"].pop("labs", None)
        t1_out.append(rec)

        core = [mine[mine.analyte == a].Result_time.min() for a in SOFA_CORE]
        if all(pd.notna(t) for t in core):
            t_sofa = max(core)
            rec2 = dict(case)
            rec2["input"] = dict(case["input"])
            rec2["input"]["checkpoint_state"] = state_at(
                t_sofa, "T_SOFA",
                "first instant at which platelets, bilirubin and creatinine have all resulted")
            rec2["input"].pop("labs", None)
            sofa_out.append(rec2)

    save(t1_out, args.out)
    save(sofa_out, args.out.replace("_t1_", "_tsofa_"))

    print(f"\n{len(t1_out)} T1 states"
          + (f"   ({len(no_result)} cases had no result at all)" if no_result else ""))
    if t1_out:
        s = pd.Series([r["input"]["checkpoint_state"]["hours_after_arrival"] for r in t1_out])
        print(f"  T1 hours after arrival: median {s.median():.2f}  "
              f"IQR {s.quantile(.25):.2f}-{s.quantile(.75):.2f}")
    if sofa_out:
        s = pd.Series([r["input"]["checkpoint_state"]["hours_after_arrival"] for r in sofa_out])
        n = sum(r["truth"]["sep3"] for r in sofa_out)
        print(f"  T_SOFA reached by {len(sofa_out)}/{len(t1_out)}   "
              f"median {s.median():.2f} h   of which sepsis {n}, non-sepsis {len(sofa_out)-n}")
    leak_check(t1_out); leak_check(sofa_out)


def leak_check(records):
    bad = [(r["case_id"], k) for r in records
           for k, v in r["input"]["checkpoint_state"]["labs_available_at_cutoff"].items()
           if v["result_available_hours_after_arrival"]
           > r["input"]["checkpoint_state"]["hours_after_arrival"] + 1e-6]
    if bad:
        raise SystemExit(f"!! LEAK: {len(bad)} results postdate their cutoff, e.g. {bad[:3]}")
    print("  leakage check: no result postdates its cutoff")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="stanford100_cases.json")
    p.add_argument("--crosswalk", default="stanford100_identifiers.json")
    p.add_argument("--out", default="stanford100_t1_states.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=1_000_000)
    build(p.parse_args())


if __name__ == "__main__":
    main()
