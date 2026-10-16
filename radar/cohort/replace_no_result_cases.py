#!/usr/bin/env python3
"""Replace cohort visits that have no ED laboratory result within the replay horizon.

Inclusion criterion added 2026-10-11 (team decision): a visit enters the cohort
only if at least one ED laboratory result becomes available within 6 h of
arrival. RADAR's reassessment is result-triggered, so a visit with no result
cannot be evaluated beyond T0. The criterion is applied uniformly: a case that
fails it is retired and replaced by a draw from the same stratum and
sub-stratum, with the same sampler rules (make_cohort100.sub_strata, same
completeness requirement), using a fresh seed stream so the original pins are
untouched. Retired cases and their crosswalk rows are kept in *_retired.json.

Usage
    python -m radar.cohort.replace_no_result_cases --site mimic
    python -m radar.cohort.replace_no_result_cases --site mcmed
"""
from radar import paths
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from radar.cohort import make_cohort100 as mc
from radar.cohort.make_t1_states import LAB_ITEMS
from radar.cohort.make_t1_states_external import COMPONENTS as MCMED_COMPONENTS, iso_dt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
D = paths.DATA
RAW = paths.RAW_ROOT / "raw"
SITE = {
    "mimic": dict(source="mimic", stem="cohort100", longitudinal="cohort100_longitudinal_cases.json"),
    "mcmed": dict(source="stanford", stem="stanford100", longitudinal="stanford100_longitudinal_cases.json"),
}
CANDIDATES = 40     # candidates drawn per retired case before checking results


def load_json(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


def save_json(obj, name):
    (D / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------- result checks
def mimic_first_result(stay_ids, horizon):
    """stay_id -> hours to first lab result (storetime) within horizon, else None."""
    ed = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/edstays.csv.gz", usecols=["subject_id", "stay_id", "intime"],
                     parse_dates=["intime"])
    ed = ed[ed.stay_id.isin(stay_ids)]
    subjects = set(ed.subject_id)
    keep = []
    for chunk in pd.read_csv(RAW / "mimic-iv-3.1/hosp/labevents.csv.gz",
                             usecols=["subject_id", "itemid", "charttime", "storetime", "valuenum", "valueuom", "flag"],
                             parse_dates=["charttime", "storetime"], chunksize=2_000_000, low_memory=False):
        hit = chunk[chunk.subject_id.isin(subjects) & chunk.itemid.isin(LAB_ITEMS)]
        if len(hit):
            keep.append(hit)
    labs = pd.concat(keep) if keep else pd.DataFrame(columns=["subject_id"])
    first = {}
    for _, w in ed.iterrows():
        L = labs[(labs.subject_id == w.subject_id) & (labs.charttime >= w.intime) & labs.storetime.notna()]
        if len(L):
            h = (L.storetime.min() - w.intime).total_seconds() / 3600
            first[int(w.stay_id)] = round(h, 3) if h <= horizon else None
        else:
            first[int(w.stay_id)] = None
    return first, labs


def mcmed_first_result(csns, horizon):
    root = RAW / "mc-med-1.0.0"
    visits = pd.read_csv(root / "visits.csv", usecols=["CSN", "Arrival_time"])
    visits = visits[visits.CSN.isin(csns)]
    arrival = dict(zip(visits.CSN, iso_dt(visits.Arrival_time)))
    keep = []
    for chunk in pd.read_csv(root / "labs.csv", usecols=["CSN", "Order_time", "Result_time", "Component_name",
                                                         "Component_value", "Component_units", "Component_abnormal"],
                             chunksize=2_000_000, low_memory=False):
        hit = chunk[chunk.CSN.isin(csns) & chunk.Component_name.isin(MCMED_COMPONENTS)]
        if len(hit):
            keep.append(hit)
    labs = pd.concat(keep) if keep else pd.DataFrame(columns=["CSN"])
    rt = iso_dt(labs.Result_time) if len(labs) else pd.Series(dtype="datetime64[s]")
    first = {}
    for csn in csns:
        a = arrival.get(csn)
        L = rt[(labs.CSN == csn) & rt.notna()] if len(labs) else rt
        if a is None or pd.isna(a) or not len(L):
            first[int(csn)] = None
            continue
        h = (L.min() - a).total_seconds() / 3600
        first[int(csn)] = round(h, 3) if 0 <= h <= horizon else None
    return first, labs


def extend_emar_cache():
    """Add emar rows for every cohort subject missing from the cache (separate call: slow scan)."""
    cw = load_json("cohort100_identifiers.json")
    ed = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/edstays.csv.gz", usecols=["subject_id", "stay_id"])
    subjects = set(ed[ed.stay_id.isin({r["stay_id"] for r in cw["mapping"]})].subject_id)
    ec = D / "_cache_cohort100_emar.csv"
    old = pd.read_csv(ec)
    need = subjects - set(old.subject_id)
    print(f"emar: {len(need)} subjects missing from cache")
    if not need:
        return
    keep = []
    for chunk in pd.read_csv(RAW / "mimic-iv-3.1/hosp/emar.csv.gz",
                             usecols=["subject_id", "hadm_id", "charttime", "medication", "event_txt", "storetime"],
                             chunksize=1_000_000, low_memory=False):
        h = chunk[chunk.subject_id.isin(need)]
        if len(h):
            keep.append(h)
    pd.concat([old] + keep).to_csv(ec, index=False)
    print(f"  emar cache extended by {sum(len(k) for k in keep)} rows")


# ----------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=SITE, required=True)
    p.add_argument("--horizon-hours", type=float, default=6.0)
    p.add_argument("--retire", nargs="*", default=None, help="case ids; default: longitudinal meta list")
    p.add_argument("--emar-only", action="store_true",
                   help="MIMIC: only extend the emar cache for already-chosen replacement subjects (slow scan)")
    a = p.parse_args()
    cfg = SITE[a.site]
    stem = cfg["stem"]
    if a.emar_only:
        return extend_emar_cache()
    retire = a.retire or load_json(cfg["longitudinal"])["meta"]["cases_with_no_visible_result"]
    cases = load_json(f"{stem}_cases.json")
    pins = load_json(f"{stem}_case_ids.json")
    cw = load_json(f"{stem}_identifiers.json")
    by_id = {c["case_id"]: c for c in cases}
    missing = [cid for cid in retire if cid not in by_id]
    if missing:
        raise SystemExit(f"not in cohort: {missing}")

    data_root = mc.find_data(cfg["source"], None)
    d = mc.load(data_root, cfg["source"])
    S, news, bedside_od, shock, fever, complete = mc.derive(d)
    subs = {(s, k): (desc, mask) for s, k, desc, mask, _ in mc.sub_strata(d, S, news, bedside_od, shock, fever)}
    taken = set(pins.values()) | {r["stay_id"] for r in cw["mapping"]}

    # Draw candidates per retired case, deterministically, from the same sub-stratum.
    plan = {}
    for cid in retire:
        c = by_id[cid]
        key = (c["stratum"], c["sub_stratum"])
        desc, mask = subs[key]
        pool = d.index[(S == key[0]) & mask & complete & ~d.stay_id.isin(taken)]
        rng = np.random.default_rng([mc.SEED, 20261011, abs(hash(f"{key[0]}_{key[1]}_{cid}")) % 10 ** 6])
        picks = list(rng.choice(pool, size=min(CANDIDATES, len(pool)), replace=False))
        plan[cid] = (key, desc, picks)
        taken |= {int(d.loc[i, "stay_id"]) for i in picks}      # candidates must not collide across slots
    cand_ids = {int(d.loc[i, "stay_id"]) for v in plan.values() for i in v[2]}
    print(f"{len(retire)} cases to replace; checking {len(cand_ids)} candidates for a result within {a.horizon_hours} h ...")
    first, labs = (mimic_first_result if a.site == "mimic" else mcmed_first_result)(cand_ids, a.horizon_hours)

    # Assign new ids/surrogates and build records with the sampler's own constructor.
    def next_index(prefix):
        n = [int(k.rsplit("_", 1)[1]) for k in list(pins) + list(by_id) if k.startswith(prefix + "_") and k.rsplit("_", 1)[1].isdigit()]
        return max(n, default=0) + 1
    used_surr = {r["surrogate_id"] for r in cw["mapping"]}
    surr_n = max(int(s.split("-")[1]) for s in used_surr) + 1
    chosen_stays = set()
    replacements, retired = [], []
    for cid in retire:
        key, desc, picks = plan[cid]
        pick = next((i for i in picks if first.get(int(d.loc[i, "stay_id"])) is not None
                     and int(d.loc[i, "stay_id"]) not in chosen_stays), None)
        if pick is None:
            raise SystemExit(f"no candidate with a result within horizon for {cid} ({key}); raise CANDIDATES")
        prefix = f"{key[0]}_{key[1]}"
        new_id = f"{prefix}_{next_index(prefix):02d}"
        rec = mc.case_record(d, mc.SOURCES[cfg["source"]], pick, new_id, key[0], key[1], desc, bedside_od[pick])
        stay = rec.pop("stay_id")
        chosen_stays.add(stay)
        rec["surrogate_id"] = f"RADAR-{surr_n:03d}"; surr_n += 1
        rec["replaces"] = cid
        rec["inclusion"] = f"first ED laboratory result at {first[stay]} h (criterion: <= {a.horizon_hours} h)"
        pins[new_id] = int(stay)
        cw["mapping"].append({"surrogate_id": rec["surrogate_id"], "case_id": new_id, "stay_id": int(stay)})
        replacements.append(rec)
        old = by_id[cid]
        old["retired_reason"] = f"no ED laboratory result within {a.horizon_hours} h of arrival"
        old["replaced_by"] = new_id
        retired.append(old)
        print(f"  {cid:>16} -> {new_id:<16} first result {first[stay]:.2f} h  {rec['surrogate_id']}")

    # Write: retired copies, then the live files with the slot replaced in place (keeps interleaving).
    retired_name = f"{stem}_cases_retired.json"
    prev = load_json(retired_name) if (D / retired_name).exists() else []
    save_json(prev + retired, retired_name)
    cw_ret_name = f"{stem}_identifiers_retired.json"
    prev_cw = load_json(cw_ret_name) if (D / cw_ret_name).exists() else []
    save_json(prev_cw + [r for r in cw["mapping"] if r["case_id"] in retire], cw_ret_name)
    cw["mapping"] = [r for r in cw["mapping"] if r["case_id"] not in retire]
    cw["n"] = len(cw["mapping"])
    for cid in retire:
        pins.pop(cid, None)
    pos = {c["case_id"]: n for n, c in enumerate(cases)}
    for rec in replacements:
        cases[pos[rec["replaces"]]] = rec
    save_json(cases, f"{stem}_cases.json")
    save_json(pins, f"{stem}_case_ids.json")
    save_json(cw, f"{stem}_identifiers.json")
    meta_name = f"{stem}_cases_meta.json"
    meta = load_json(meta_name)
    meta.setdefault("inclusion_criteria", []).append(
        f"at least one ED laboratory result available within {a.horizon_hours} h of arrival (added 2026-10-11)")
    meta.setdefault("replacements", []).extend({"retired": r["replaces"], "replacement": r["case_id"]} for r in replacements)
    save_json(meta, meta_name)

    # Extend the raw-row caches so the longitudinal builder does not need a rescan.
    if a.site == "mimic":
        cache = D / "_cache_cohort100_labevents_subset.csv"
        ed = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/edstays.csv.gz", usecols=["subject_id", "stay_id"])
        new_subj = set(ed[ed.stay_id.isin(chosen_stays)].subject_id)
        old = pd.read_csv(cache)
        add = labs[labs.subject_id.isin(new_subj)]
        pd.concat([old, add]).to_csv(cache, index=False)
        px = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/pyxis.csv.gz")
        pc = D / "_cache_cohort100_pyxis.csv"
        pd.concat([pd.read_csv(pc), px[px.stay_id.isin(chosen_stays)]]).to_csv(pc, index=False)
        print("  caches extended: labevents, pyxis. Now run: python -m radar.cohort.replace_no_result_cases --site mimic --emar-only")
    else:
        lc = D / "_cache_stanford100_labs.csv"
        pd.concat([pd.read_csv(lc), labs[labs.CSN.isin(chosen_stays)]]).to_csv(lc, index=False)
        root = RAW / "mc-med-1.0.0"
        for name, cols, pred, cache in (
            ("numerics.csv", ["CSN", "Source", "Measure", "Value", "Time"],
             lambda h: (h.Source == "Chart") & h.Measure.isin({"HR", "RR", "SpO2", "SBP", "DBP", "MAP", "Temp"}),
             "_cache_stanford100_numerics.csv"),
            ("orders.csv", None, None, "_cache_stanford100_orders_all.csv"),
        ):
            keep = []
            for chunk in pd.read_csv(root / name, usecols=cols, chunksize=2_000_000, low_memory=False):
                h = chunk[chunk.CSN.isin(chosen_stays)]
                h = h[pred(h)] if pred else h
                if len(h):
                    keep.append(h)
            pd.concat([pd.read_csv(D / cache, low_memory=False)] + keep).to_csv(D / cache, index=False)
        bc = D / "_cache_stanford100_orders.csv"      # blood-culture subset used by the builder
        o = pd.read_csv(D / "_cache_stanford100_orders_all.csv", low_memory=False)
        o = o[o.Procedure_name.str.contains("CULTURE", case=False, na=False) & o.Procedure_name.str.contains("BLOOD", case=False, na=False)]
        o[["CSN", "Order_time", "Procedure_name"]].to_csv(bc, index=False)
        print("  caches extended: labs, numerics, orders")
    print(f"-> {stem}_cases.json / _case_ids.json / _identifiers.json updated; retired copies in {retired_name}")


if __name__ == "__main__":
    main()
