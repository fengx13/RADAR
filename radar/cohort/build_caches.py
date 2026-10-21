#!/usr/bin/env python3
"""Build the raw-row caches that the treatment and infection-result steps read.

Until 2026-10-20 these caches existed only as files made by hand on 2026-10-12;
no script in the repository wrote them, so a fresh extraction could not run.
Each cache is a plain subset of one raw table, restricted to the cohort's own
visits (MIMIC: stay_id or subject_id from the crosswalk; MC-MED: CSN). No value
is derived here; the consuming step does all interpretation.

  site   cache file                              raw table                     restricted by     read by
  mimic  _cache_cohort100_pyxis.csv              ed/pyxis.csv.gz               stay_id           make_observed_treatment
  mimic  _cache_cohort100_emar.csv               hosp/emar.csv.gz              subject_id        make_observed_treatment
  mimic  _cache_cohort100_labevents_extra.csv    hosp/labevents.csv.gz         subject_id+itemid make_infection_results
  mimic  _cache_cohort100_radiology.csv          note/radiology.csv            subject_id        make_infection_results
  mcmed  _cache_stanford100_orders_all.csv       orders.csv                    CSN               make_observed_treatment
  mcmed  _cache_stanford100_labs_extra.csv       labs.csv                      CSN+Component     make_infection_results
  mcmed  _cache_stanford100_rads.csv             rads.csv                      CSN               make_infection_results

Item lists follow PY-131 (UA components, troponin T, NT-proBNP, CRP for MIMIC;
UA, influenza A/B, RSV, CRP, procalcitonin, troponin I / high-sensitivity,
NT-proBNP for MC-MED). Point-of-care troponin and high-sensitivity CRP are not
included. MIMIC urine items exist under two lab categories (Hematology, and the
later Chemistry ids); both are kept.

Usage
    python -m radar.cohort.build_caches --site mimic            # builds the missing caches
    python -m radar.cohort.build_caches --site mcmed --force    # rebuild all
"""
import argparse
import json

import pandas as pd

from radar import paths

D = paths.DATA
RAW = paths.RAW_ROOT / "raw"

MIMIC_EXTRA_ITEMS = {
    51487: "ua_nitrite", 51987: "ua_nitrite",
    51486: "ua_leukocytes", 51985: "ua_leukocytes",
    51466: "ua_blood", 51967: "ua_blood",
    51492: "ua_protein", 51992: "ua_protein",
    51484: "ua_ketone", 51984: "ua_ketone",
    51463: "ua_bacteria",
    51516: "ua_wbc",
    51003: "troponin_t",
    50963: "ntprobnp",
    50889: "crp",
}
MCMED_EXTRA_COMPONENTS = {
    "NITRITE: URINE (UA)": "ua_nitrite",
    "LEUKOCYTE ESTERASE: URINE (UA)": "ua_leukocytes",
    "BLOOD: URINE (UA)": "ua_blood",
    "PROTEIN: URINE (UA)": "ua_protein",
    "KETONE: URINE (UA)": "ua_ketone",
    "INFLUENZA A": "influenza_a",
    "INFLUENZA B": "influenza_b",
    "RSV": "rsv",
    "C-REACTIVE PROTEIN (CRP)": "crp",
    "PROCALCITONIN": "procalcitonin",
    "TROPONIN I": "troponin_i",
    "HIGH SENSITIVITY TROPONIN": "troponin_hs",
    "BNP, NT-PRO": "ntprobnp",
}


def crosswalk_ids(site):
    stem = "cohort100" if site == "mimic" else "stanford100"
    cw = json.loads((D / f"{stem}_identifiers.json").read_text(encoding="utf-8"))
    key = "stay_id" if site == "mimic" else "CSN"
    return {int(r.get(key) or r.get("stay_id") or r.get("csn")) for r in cw["mapping"]}


def scan(path, keep_rows, usecols=None, chunksize=2_000_000, label=None, **kw):
    keep, seen = [], 0
    for i, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False, **kw), 1):
        seen += len(chunk)
        hit = keep_rows(chunk)
        if len(hit):
            keep.append(hit)
        if i % 10 == 0:
            print(f"    {label or path.name}: {seen:,} scanned, {sum(len(k) for k in keep):,} kept", flush=True)
    return pd.concat(keep) if keep else pd.DataFrame(columns=usecols or [])


def mimic_builders(stays):
    ed = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/edstays.csv.gz", usecols=["subject_id", "stay_id"])
    subjects = set(ed[ed.stay_id.isin(stays)].subject_id)

    def pyxis():
        return scan(RAW / "mimic-iv-ed-2.2/ed/pyxis.csv.gz", lambda c: c[c.stay_id.isin(stays)])

    def emar():
        return scan(RAW / "mimic-iv-3.1/hosp/emar.csv.gz", lambda c: c[c.subject_id.isin(subjects)],
                    usecols=["subject_id", "hadm_id", "charttime", "medication", "event_txt", "storetime"])

    def labevents_extra():
        out = scan(RAW / "mimic-iv-3.1/hosp/labevents.csv.gz",
                   lambda c: c[c.subject_id.isin(subjects) & c.itemid.isin(MIMIC_EXTRA_ITEMS)],
                   usecols=["subject_id", "itemid", "charttime", "storetime", "value", "valuenum", "valueuom", "flag"],
                   label="labevents (extra items)")
        out["analyte"] = out.itemid.map(MIMIC_EXTRA_ITEMS)
        return out

    def radiology():
        return scan(RAW / "mimic-iv-note-2.2/note/radiology.csv", lambda c: c[c.subject_id.isin(subjects)],
                    usecols=["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"], chunksize=200_000)

    return {"_cache_cohort100_pyxis.csv": pyxis, "_cache_cohort100_emar.csv": emar,
            "_cache_cohort100_labevents_extra.csv": labevents_extra,
            "_cache_cohort100_radiology.csv": radiology}


def mcmed_builders(csns):
    root = RAW / "mc-med-1.0.0"

    def orders_all():
        return scan(root / "orders.csv", lambda c: c[c.CSN.isin(csns)])

    def labs_extra():
        out = scan(root / "labs.csv", lambda c: c[c.CSN.isin(csns) & c.Component_name.isin(MCMED_EXTRA_COMPONENTS)])
        out["analyte"] = out.Component_name.map(MCMED_EXTRA_COMPONENTS)
        return out

    def rads():
        return scan(root / "rads.csv", lambda c: c[c.CSN.isin(csns)], chunksize=200_000)

    return {"_cache_stanford100_orders_all.csv": orders_all,
            "_cache_stanford100_labs_extra.csv": labs_extra,
            "_cache_stanford100_rads.csv": rads}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=("mimic", "mcmed"), required=True)
    p.add_argument("--only", nargs="*", default=None, help="cache file names to build (default: all for the site)")
    p.add_argument("--force", action="store_true", help="rebuild caches that already exist")
    a = p.parse_args()
    ids = crosswalk_ids(a.site)
    builders = (mimic_builders if a.site == "mimic" else mcmed_builders)(ids)
    for name, build in builders.items():
        if a.only and name not in a.only:
            continue
        path = D / name
        if path.exists() and not a.force:
            print(f"  {name}: exists, kept (use --force to rebuild)")
            continue
        print(f"  building {name} for {len(ids)} visits ...", flush=True)
        df = build()
        df.to_csv(path, index=False)
        print(f"  -> {path} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
