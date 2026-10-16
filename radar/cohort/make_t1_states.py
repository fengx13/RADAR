#!/usr/bin/env python3
"""Build the T1 reassessment state for each cohort case, from raw MIMIC-IV.

What T1 is
----------
T1 is the moment the first ordered investigation actually returns a result. Not
when it was ordered, not when the specimen was drawn, not when someone charted
it later. In MIMIC that is `storetime` — the time the result became available in
the system. `charttime` is when the specimen was taken, which is earlier and is
not what a clinician can act on.

So per stay:

    T1 = min(storetime) over every lab and every microbiology result
         belonging to that ED encounter

and the state at T1 contains only what had resulted or been charted at or before
that instant. Everything later is withheld — that is the whole point of the
checkpoint, and the leakage check at the end enforces it.

What this produces
------------------
`cohort100_t1_states.json`, one record per case, shaped for

    python -m radar.agents.run_agent1 --decision-time checkpoint --infile <that file>

Each record keeps the T0 triage snapshot, adds `checkpoint_state` with the labs
and vitals available by the cutoff, and reports which SOFA components exist. The
SOFA subscores are computed in code, never by the model — R2 may use SOFA >= 2 at
a checkpoint only when its components actually resulted, and this file is what
tells it whether they did.

Direct identifiers stay out: records are keyed on case_id and surrogate_id. The
stay_id needed to query MIMIC is read from the local crosswalk, never written.

Usage
-----
    python -m radar.cohort.make_t1_states
    python -m radar.cohort.make_t1_states --data-root /path/to/00_Data --limit 10
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

DATA_CANDIDATES = [
    paths.RAW_ROOT,
]

# Blood only. Where MIMIC carries the same analyte under several itemids we take
# them all and keep the earliest result.
LAB_ITEMS = {
    50813: "lactate",       52442: "lactate",       53154: "lactate",
    50912: "creatinine",    52546: "creatinine",
    51265: "platelets",     53189: "platelets",
    50885: "bilirubin_total", 53089: "bilirubin_total",
    50821: "pao2",
    51301: "wbc",           51755: "wbc",           51756: "wbc",
    50882: "bicarbonate",
    50820: "ph",
    51237: "inr",           51675: "inr",
    51006: "bun",           52647: "bun",
    50983: "sodium",        52623: "sodium",
    50971: "potassium",     52610: "potassium",
    51222: "hemoglobin",    50811: "hemoglobin",
    50868: "anion_gap",     52500: "anion_gap",
    50931: "glucose",       50809: "glucose",
    50889: "crp",
}

# The three SOFA organ systems that rest on labs an ED patient reliably gets.
# Respiratory needs PaO2/FiO2 and so needs an arterial gas, which most ED
# patients never receive; it is reported separately and never assumed normal.
SOFA_CORE = ("platelets", "bilirubin_total", "creatinine")
SOFA_COMPONENTS = SOFA_CORE + ("pao2",)


def find_data(explicit=None):
    for d in ([pathlib.Path(explicit)] if explicit else DATA_CANDIDATES):
        if (d / "raw/mimic-iv-3.1/hosp/labevents.csv.gz").exists():
            return d
    raise SystemExit(
        "raw MIMIC-IV not found. Looked in:\n  "
        + "\n  ".join(str(d) for d in DATA_CANDIDATES)
        + "\nPass --data-root to point at the 00_Data directory.")


# ------------------------------------------------------------------ SOFA
def sofa_scores(labs, vitals, on_vasopressor=False):
    """The four SOFA subscores computable from blood results, plus cardio.

    Returns (per-component scores, total, list of components that had no result).
    A component with no result contributes nothing and is reported as missing —
    it is never silently scored 0, which would read as 'organ is fine'.
    """
    def band(value, cuts):
        """cuts ascending by severity; returns how many thresholds are crossed."""
        if value is None:
            return None
        return sum(1 for c in cuts if value >= c) if cuts[0] < cuts[-1] else \
               sum(1 for c in cuts if value <= c)

    plt = labs.get("platelets", {}).get("value")
    bil = labs.get("bilirubin_total", {}).get("value")
    cre = labs.get("creatinine", {}).get("value")
    pao2 = labs.get("pao2", {}).get("value")
    spo2 = vitals.get("spo2")

    scores, missing = {}, []
    scores["coag"] = band(plt, [150, 100, 50, 20])          # descending: <150 =1 ... <20 =4
    scores["liver"] = band(bil, [1.2, 2.0, 6.0, 12.0])
    scores["renal"] = band(cre, [1.2, 2.0, 3.5, 5.0])
    # Respiratory needs FiO2 to form PaO2/FiO2. Without a recorded FiO2 the ratio
    # cannot be formed, so this component is reported missing rather than guessed.
    scores["resp"] = None
    scores["cardio"] = 1 if on_vasopressor else (
        1 if (vitals.get("map") is not None and vitals["map"] < 70) else
        (0 if vitals.get("map") is not None else None))

    for name, source in (("coag", plt), ("liver", bil), ("renal", cre),
                         ("resp", pao2), ("cardio", vitals.get("map"))):
        if scores[name] is None:
            missing.append(name)
    total = sum(v for v in scores.values() if v is not None)
    return scores, total, missing


# ------------------------------------------------------------------ extraction
def load_cohort(args):
    cases = json.loads((D / args.cases).read_text(encoding="utf-8"))
    cw = D / args.crosswalk
    if not cw.exists():
        raise SystemExit(
            f"{cw} not found. It holds the stay_id mapping and is gitignored, so a "
            "fresh clone will not have it. Restore it from your local copy, or "
            "rerun make_cohort100.py followed by deidentify_cohort.py.")
    stay = {r["case_id"]: int(r["stay_id"])
            for r in json.loads(cw.read_text(encoding="utf-8"))["mapping"]}
    if args.limit:
        cases = cases[:args.limit]
    return cases, stay


def ed_windows(data_root, stay_ids):
    ed = pd.read_csv(data_root / "raw/mimic-iv-ed-2.2/ed/edstays.csv.gz",
                     usecols=["subject_id", "stay_id", "intime", "outtime"],
                     parse_dates=["intime", "outtime"], low_memory=False)
    ed = ed[ed.stay_id.isin(stay_ids)]
    return ed.set_index("stay_id")


def stream_labs(data_root, subjects, chunksize):
    """Pull only our subjects' rows out of the 2.6 GB table."""
    keep, seen = [], 0
    reader = pd.read_csv(
        data_root / "raw/mimic-iv-3.1/hosp/labevents.csv.gz",
        usecols=["subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom", "flag"],
        parse_dates=["charttime", "storetime"], chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        seen += len(chunk)
        hit = chunk[chunk.subject_id.isin(subjects) & chunk.itemid.isin(LAB_ITEMS)]
        if len(hit):
            keep.append(hit)
        if i % 20 == 0:
            print(f"    labevents: {seen:,} rows scanned, "
                  f"{sum(len(k) for k in keep):,} kept", flush=True)
    out = pd.concat(keep) if keep else pd.DataFrame(columns=["subject_id"])
    print(f"    labevents: {seen:,} rows scanned, {len(out):,} kept")
    return out


def stream_micro(data_root, subjects):
    m = pd.read_csv(data_root / "raw/mimic-iv-3.1/hosp/microbiologyevents.csv.gz",
                    usecols=["subject_id", "charttime", "storetime", "spec_type_desc",
                             "test_name", "org_name"],
                    parse_dates=["charttime", "storetime"], low_memory=False)
    return m[m.subject_id.isin(subjects)]


def build(args):
    data_root = find_data(args.data_root)
    print(f"reading raw MIMIC from {data_root}")
    cases, stay_of = load_cohort(args)
    stay_ids = {stay_of[c["case_id"]] for c in cases if c["case_id"] in stay_of}
    windows = ed_windows(data_root, stay_ids)
    subjects = set(windows.subject_id)
    print(f"  {len(cases)} cases, {len(subjects)} subjects")

    print("  scanning labevents (this is the slow part) ...")
    labs = stream_labs(data_root, subjects, args.chunksize)
    print("  scanning microbiologyevents ...")
    micro = stream_micro(data_root, subjects)
    print(f"    microbiology: {len(micro):,} rows kept")

    out, no_result = [], []
    for case in cases:
        cid = case["case_id"]
        sid = stay_of.get(cid)
        if sid is None or sid not in windows.index:
            no_result.append(cid)
            continue
        w = windows.loc[sid]
        intime, outtime = w.intime, w.outtime
        # A result belongs to this encounter if the specimen was taken from
        # arrival onward and the result landed before the stay is long over.
        horizon = outtime + pd.Timedelta(hours=args.horizon_hours)

        L = labs[(labs.subject_id == w.subject_id)
                 & (labs.charttime >= intime) & (labs.storetime <= horizon)
                 & labs.storetime.notna()]
        M = micro[(micro.subject_id == w.subject_id)
                  & (micro.charttime >= intime) & (micro.storetime <= horizon)
                  & micro.storetime.notna()]

        times = list(L.storetime) + list(M.storetime)
        if not times:
            no_result.append(cid)
            continue

        v0 = case["input"]["triage_vitals"]
        vitals = dict(v0)
        if vitals.get("sbp") is not None and vitals.get("dbp") is not None:
            vitals["map"] = round((vitals["sbp"] + 2 * vitals["dbp"]) / 3.0, 1)

        def labs_by(cut):
            state = {}
            for _, r in L[L.storetime <= cut].sort_values("storetime").iterrows():
                name = LAB_ITEMS[r.itemid]
                if name in state:                      # keep the earliest result
                    continue
                state[name] = {
                    "value": None if pd.isna(r.valuenum) else float(r.valuenum),
                    "unit": None if pd.isna(r.valueuom) else str(r.valueuom),
                    "abnormal_flag": None if pd.isna(r.flag) else str(r.flag),
                    "result_available_hours_after_arrival":
                        round((r.storetime - intime).total_seconds() / 3600.0, 3),
                }
            return state

        def state_at(cut, checkpoint_id, definition):
            lab_state = labs_by(cut)
            micro_avail = M[M.storetime <= cut]
            scores, total, missing = sofa_scores(lab_state, vitals)
            hours = round((cut - intime).total_seconds() / 3600.0, 3)
            return {
                "checkpoint_id": checkpoint_id,
                "definition": definition,
                "information_cutoff": f"{checkpoint_id} = {hours} h after ED arrival",
                "hours_after_arrival": hours,
                "labs_available_at_cutoff": lab_state,
                "cultures_at_cutoff": {
                    "blood_culture_resulted": bool(
                        micro_avail.spec_type_desc.str.contains("BLOOD", case=False, na=False).any()),
                    "any_culture_resulted": bool(len(micro_avail)),
                    "organisms_reported": sorted({str(o) for o in micro_avail.org_name.dropna()}),
                },
                "still_pending_at_cutoff": {
                    "labs_ordered_not_yet_resulted": sorted(
                        {LAB_ITEMS[i] for i in L[L.storetime > cut].itemid} - set(lab_state)),
                    "cultures_pending": bool(len(M[M.storetime > cut])),
                },
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
                             "with no result is reported missing, never scored 0. R2 may use "
                             "SOFA >= 2 only when usable_for_sofa_ge_2 is true, which needs "
                             "platelets, bilirubin and creatinine. Respiratory needs an arterial "
                             "gas most ED patients never receive."),
                },
            }

        t1 = min(times)
        checkpoints = [state_at(t1, "T1",
                                "first ordered investigation to actually return a result")]

        # The chemistry panel lands later than the blood count. Find the first
        # instant at which SOFA's core components have all resulted, if ever.
        core_times = []
        for name in SOFA_CORE:
            got = L[[LAB_ITEMS[i] == name for i in L.itemid]]
            if len(got):
                core_times.append(got.storetime.min())
        if len(core_times) == len(SOFA_CORE):
            t_sofa = max(core_times)
            if t_sofa > t1:
                checkpoints.append(state_at(
                    t_sofa, "T_SOFA",
                    "first instant at which platelets, bilirubin and creatinine have all resulted"))

        record = dict(case)
        record["input"] = dict(case["input"])
        record["input"]["checkpoint_state"] = checkpoints[0]     # run_agent1 reads this one
        record["input"].pop("labs", None)
        record["available_checkpoints"] = checkpoints
        out.append(record)

    save(out, args.out)

    # A second file, cut at the point SOFA becomes computable, for the arm of the
    # T1 group that is allowed to use the SOFA >= 2 criterion.
    sofa_file = []
    for r in out:
        later = [c for c in r["available_checkpoints"] if c["checkpoint_id"] == "T_SOFA"]
        if not later:
            continue
        rec = dict(r)
        rec["input"] = dict(r["input"])
        rec["input"]["checkpoint_state"] = later[0]
        rec.pop("available_checkpoints", None)
        sofa_file.append(rec)
    save(sofa_file, args.out.replace("_t1_", "_tsofa_"))

    t1s = [r["input"]["checkpoint_state"]["hours_after_arrival"] for r in out]
    ready_t1 = sum(r["input"]["checkpoint_state"]["sofa"]["usable_for_sofa_ge_2"] for r in out)
    print(f"\n{len(out)} T1 states built"
          + (f"   ({len(no_result)} cases had no result at all: {no_result[:6]})" if no_result else ""))
    if t1s:
        s = pd.Series(t1s)
        print(f"  T1 after arrival, hours:  median {s.median():.2f}   "
              f"IQR {s.quantile(.25):.2f}-{s.quantile(.75):.2f}   max {s.max():.2f}")
    print(f"  SOFA computable at T1:      {ready_t1}/{len(out)}")
    if sofa_file:
        t = pd.Series([r["input"]["checkpoint_state"]["hours_after_arrival"] for r in sofa_file])
        print(f"  SOFA computable at T_SOFA:  {len(sofa_file)}/{len(out)}   "
              f"median {t.median():.2f} h   IQR {t.quantile(.25):.2f}-{t.quantile(.75):.2f}")
    leak_check(out)
    leak_check(sofa_file)


def leak_check(records):
    """Nothing in the checkpoint state may postdate the cutoff."""
    bad = []
    for r in records:
        st = r["input"]["checkpoint_state"]
        cut = st["hours_after_arrival"]
        for name, lab in st["labs_available_at_cutoff"].items():
            if lab["result_available_hours_after_arrival"] > cut + 1e-6:
                bad.append((r["case_id"], name,
                            lab["result_available_hours_after_arrival"], cut))
    if bad:
        print(f"\n!! LEAK: {len(bad)} results postdate their own cutoff")
        for b in bad[:5]:
            print("   ", b)
        raise SystemExit(1)
    print("  leakage check: no result in any state postdates its cutoff")


def save(obj, name):
    (D / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {paths.DATA / name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="cohort100_cases.json")
    p.add_argument("--crosswalk", default="cohort100_identifiers.json")
    p.add_argument("--out", default="cohort100_t1_states.json")
    p.add_argument("--data-root", default=None)
    p.add_argument("--limit", type=int, default=None, help="first N cases, for a quick test")
    p.add_argument("--chunksize", type=int, default=2_000_000)
    p.add_argument("--horizon-hours", type=float, default=72.0,
                   help="how long after ED departure a result may still land")
    build(p.parse_args())


if __name__ == "__main__":
    main()
