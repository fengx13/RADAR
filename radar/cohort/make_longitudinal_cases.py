#!/usr/bin/env python3
"""Build RADAR vNext longitudinal cases for the two 100-visit cohorts.

Turns the T0 cohort snapshots (cohort100_cases.json, stanford100_cases.json)
plus the raw record into the externally assembled longitudinal case contract
that run_pipeline_vnext.py consumes, so the full 200-visit evaluation can run
through the result-triggered checkpoints C0_5 / C1 / C3 / C6.

Time basis: hours since ED arrival (edstays.intime / visits.Arrival_time).

Per case:
  t0_state            the T0 snapshot already in the cohort file, minus "labs"
  observed_workup     one entry per analyte: earliest result within the horizon.
                      sample_time   = specimen time (labevents.charttime / labs.Order_time)
                      result_available_time = labevents.storetime / labs.Result_time
                      A second lactate within the horizon becomes `lactate_repeat`.
                      Blood cultures collected within the horizon are `pending`
                      (result_kind final_culture, never a checkpoint trigger).
                      An analyte that only results after the horizon is recorded
                      as `pending` with no value, so it is never visible.
  longitudinal_events.vitals   time-stamped ED vitals (ed/vitalsign, numerics Chart rows)
  evaluation_only     the `truth` block, which inference never sees

Repeated non-lactate results are dropped (earliest kept) because the workup
contract is keyed by item name. Order times are unknown in MIMIC and recorded
as null. Observed treatment is not extracted: empty means unknown, not none.

Usage
-----
    python -m radar.cohort.make_longitudinal_cases --site mimic
    python -m radar.cohort.make_longitudinal_cases --site mcmed
    python -m radar.cohort.make_longitudinal_cases --site mimic --limit 5
"""
from radar import paths
import argparse
import json
import pathlib

import numpy as np
import pandas as pd

from radar.cohort.make_t1_states import LAB_ITEMS, find_data as find_mimic
from radar.cohort.make_t1_states_external import COMPONENTS as MCMED_COMPONENTS, find_data as find_mcmed, iso_dt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
D = paths.DATA

ITEM_NAME = {"lactate": "serum_lactate"}      # the contract's own spelling
VITAL_MIMIC = {"heartrate": "heart_rate", "resprate": "resp_rate", "o2sat": "spo2",
               "sbp": "sbp", "dbp": "dbp", "temperature": "temperature_F"}
VITAL_MCMED = {"HR": "heart_rate", "RR": "resp_rate", "SpO2": "spo2", "SBP": "sbp",
               "DBP": "dbp", "MAP": "map", "Temp": "temperature_F"}

SITES = {
    "mimic": dict(cases="cohort100_cases.json", crosswalk="cohort100_identifiers.json",
                  out="cohort100_longitudinal_cases.json", source="MIMIC-IV-ED"),
    "mcmed": dict(cases="stanford100_cases.json", crosswalk="stanford100_identifiers.json",
                  out="stanford100_longitudinal_cases.json", source="MC-MED"),
}




def hours(t, origin):
    return round((t - origin).total_seconds() / 3600.0, 3)


def prov(**extra):
    return {"source_type": "observed_ehr", **extra}


def load_cohort(site, limit):
    cfg = SITES[site]
    cases = json.loads((D / cfg["cases"]).read_text(encoding="utf-8"))
    cw = json.loads((D / cfg["crosswalk"]).read_text(encoding="utf-8"))
    key = "stay_id" if site == "mimic" else "CSN"
    stay = {}
    for r in cw["mapping"]:
        stay[r["case_id"]] = int(r.get(key) or r.get("stay_id") or r.get("csn"))
    return cases[:limit] if limit else cases, stay


# ---------------------------------------------------------------- raw readers
def mimic_raw(data_root, stay_ids, chunksize):
    ed = pd.read_csv(data_root / "raw/mimic-iv-ed-2.2/ed/edstays.csv.gz",
                     usecols=["subject_id", "stay_id", "intime", "outtime"],
                     parse_dates=["intime", "outtime"], low_memory=False)
    ed = ed[ed.stay_id.isin(stay_ids)].set_index("stay_id")
    subjects = set(ed.subject_id)
    vs = pd.read_csv(data_root / "raw/mimic-iv-ed-2.2/ed/vitalsign.csv.gz",
                     parse_dates=["charttime"], low_memory=False)
    vs = vs[vs.stay_id.isin(stay_ids)]
    cache = D / "_cache_cohort100_labevents_subset.csv"
    if cache.exists():
        labs = pd.read_csv(cache, parse_dates=["charttime", "storetime"])
        labs = labs[labs.subject_id.isin(subjects)]
        print(f"    labevents: {len(labs):,} rows from cache {cache.name}")
    else:
        labs = _scan_labevents(data_root, subjects, chunksize)
        labs.to_csv(cache, index=False)
    labs = labs[labs.storetime.notna()].copy()
    labs["analyte"] = labs.itemid.map(LAB_ITEMS)
    micro = pd.read_csv(data_root / "raw/mimic-iv-3.1/hosp/microbiologyevents.csv.gz",
                        usecols=["subject_id", "charttime", "storetime", "spec_type_desc"],
                        parse_dates=["charttime", "storetime"], low_memory=False)
    micro = micro[micro.subject_id.isin(subjects)]
    return ed, vs, labs, micro


def _scan_labevents(data_root, subjects, chunksize):
    keep, seen = [], 0
    reader = pd.read_csv(data_root / "raw/mimic-iv-3.1/hosp/labevents.csv.gz",
                         usecols=["subject_id", "itemid", "charttime", "storetime", "valuenum",
                                  "valueuom", "flag"],
                         parse_dates=["charttime", "storetime"], chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        seen += len(chunk)
        hit = chunk[chunk.subject_id.isin(subjects) & chunk.itemid.isin(LAB_ITEMS)]
        if len(hit):
            keep.append(hit)
        if i % 20 == 0:
            print(f"    labevents: {seen:,} scanned, {sum(len(k) for k in keep):,} kept", flush=True)
    return pd.concat(keep) if keep else pd.DataFrame(columns=["subject_id"])


def mcmed_raw(data_root, csns, chunksize):
    visits = pd.read_csv(data_root / "visits.csv", usecols=["CSN", "Arrival_time", "Departure_time"],
                         low_memory=False)
    visits = visits[visits.CSN.isin(csns)].copy()
    for c in ("Arrival_time", "Departure_time"):
        visits[c] = iso_dt(visits[c])
    visits = visits.set_index("CSN")

    def scan(name, usecols, pred, every):
        cache = D / f"_cache_stanford100_{name}"
        if cache.exists():
            out = pd.read_csv(cache, low_memory=False)
            print(f"    {name}: {len(out):,} rows from cache")
            return out
        out = _scan(name, usecols, pred, every)
        out.to_csv(cache, index=False)
        return out

    def _scan(name, usecols, pred, every):
        keep, seen = [], 0
        for i, chunk in enumerate(pd.read_csv(data_root / name, usecols=usecols, chunksize=chunksize,
                                              low_memory=False), 1):
            seen += len(chunk)
            hit = chunk[chunk.CSN.isin(csns)]
            hit = hit[pred(hit)] if pred else hit
            if len(hit):
                keep.append(hit)
            if i % every == 0:
                print(f"    {name}: {seen:,} scanned, {sum(len(k) for k in keep):,} kept", flush=True)
        return pd.concat(keep) if keep else pd.DataFrame()

    labs = scan("labs.csv", ["CSN", "Order_time", "Result_time", "Component_name", "Component_value",
                             "Component_units", "Component_abnormal"],
                lambda h: h.Component_name.isin(MCMED_COMPONENTS), 5)
    for c in ("Order_time", "Result_time"):
        labs[c] = iso_dt(labs[c])
    labs = labs[labs.Result_time.notna()]
    labs["analyte"] = labs.Component_name.map(MCMED_COMPONENTS)
    labs["value"] = pd.to_numeric(labs.Component_value, errors="coerce")

    vs = scan("numerics.csv", ["CSN", "Source", "Measure", "Value", "Time"],
              lambda h: (h.Source == "Chart") & h.Measure.isin(VITAL_MCMED), 10)
    vs["Time"] = iso_dt(vs.Time)
    vs["Value"] = pd.to_numeric(vs.Value, errors="coerce")

    orders = scan("orders.csv", ["CSN", "Order_time", "Procedure_name"],
                  lambda h: h.Procedure_name.str.contains("CULTURE", case=False, na=False)
                  & h.Procedure_name.str.contains("BLOOD", case=False, na=False), 5)
    if len(orders):
        orders["Order_time"] = iso_dt(orders.Order_time)
    return visits, vs, labs, orders


# ---------------------------------------------------------------- assembly
def workup_from_rows(rows, horizon, site):
    """rows: list of dict(analyte, sample, available, value, unit, flag, ref)."""
    rows = sorted(rows, key=lambda r: r["available"])
    seen, workup, lactates = {}, [], []
    for r in rows:
        if r["analyte"] == "lactate":
            lactates.append(r)
            continue
        if r["analyte"] in seen:
            continue
        seen[r["analyte"]] = r
    for name, r in seen.items():
        workup.append(_entry(ITEM_NAME.get(name, name), r, horizon, "diagnostic_laboratory", site))
    if lactates:
        workup.append(_entry("serum_lactate", lactates[0], horizon, "diagnostic_laboratory", site))
        later = [r for r in lactates[1:] if r["available"] <= horizon]
        if later:
            workup.append(_entry("lactate_repeat", later[0], horizon, "repeat_lactate", site))
    return workup


def _entry(item, r, horizon, kind, site):
    visible = r["available"] <= horizon
    base = {
        "item": item,
        "actual_order_status": "collected",
        "order_time": r.get("order"),
        "sample_time": r["sample"],
        "result_kind": kind,
        "disposition_relevant": True,
        "checkpoint_eligible": True,
        "provenance": prov(source_table=("hosp.labevents" if site == "mimic" else "labs.csv"),
                           source_ref=r["ref"],
                           time_basis=("sample=charttime, available=storetime" if site == "mimic"
                                       else "order=Order_time, sample=Order_time (collection not recorded), available=Result_time")),
    }
    if visible:
        base.update({"result_status": "resulted", "result_available_time": r["available"],
                     "value": r["value"], "unit": r["unit"], "abnormal_flag": r["flag"],
                     "result_summary": None})
    else:
        base.update({"result_status": "pending", "result_available_time": None, "value": None,
                     "unit": None, "result_summary": None,
                     "provenance": {**base["provenance"], "note": "resulted after the replay horizon; withheld"}})
    return base


def culture_entry(sample, site, ref):
    return {
        "item": "blood_cultures", "actual_order_status": "collected", "order_time": None,
        "sample_time": sample, "result_status": "pending", "result_available_time": None,
        "value": None, "result_summary": None, "result_kind": "final_culture",
        "disposition_relevant": True, "checkpoint_eligible": False,
        "provenance": prov(source_table=("hosp.microbiologyevents" if site == "mimic" else "orders.csv"),
                           source_ref=ref, note="final culture results after the ED horizon; pending"),
    }


def vital_events(frame, origin, horizon, site):
    out = []
    if site == "mimic":
        for _, r in frame.sort_values("charttime").iterrows():
            t = hours(r.charttime, origin)
            if t < 0 or t > horizon:
                continue
            ev = {"measurement_time": t}
            for src, dst in VITAL_MIMIC.items():
                v = r.get(src)
                if v is not None and not pd.isna(v):
                    ev[dst] = float(v)
            if "sbp" in ev and "dbp" in ev:
                ev["map"] = round((ev["sbp"] + 2 * ev["dbp"]) / 3.0, 1)
                map_basis = "derived from same-row SBP/DBP"
            else:
                map_basis = None
            ev["provenance"] = prov(source_table="ed.vitalsign", time_column="charttime",
                                    availability_basis="charttime proxy", map_basis=map_basis)
            out.append(ev)
    else:
        frame = frame[frame.Time.notna() & frame.Value.notna()].copy()
        frame["t"] = [hours(t, origin) for t in frame.Time]
        frame = frame[(frame.t >= 0) & (frame.t <= horizon)]
        for t, grp in frame.groupby("t"):
            ev = {"measurement_time": float(t)}
            for _, r in grp.iterrows():
                ev[VITAL_MCMED[r.Measure]] = float(r.Value)
            if "map" not in ev and "sbp" in ev and "dbp" in ev:
                ev["map"] = round((ev["sbp"] + 2 * ev["dbp"]) / 3.0, 1)
            ev["provenance"] = prov(source_table="numerics.csv", source_filter="Source=Chart",
                                    time_column="Time", availability_basis="chart time proxy")
            out.append(ev)
    return out


def build(args):
    site, cfg = args.site, SITES[args.site]
    cases, stay_of = load_cohort(site, args.limit)
    ids = {stay_of[c["case_id"]] for c in cases if c["case_id"] in stay_of}
    if site == "mimic":
        root = find_mimic(args.data_root)
        print(f"reading raw MIMIC from {root}")
        ed, vs, labs, micro = mimic_raw(root, ids, args.chunksize)
    else:
        root = find_mcmed(args.data_root)
        print(f"reading raw MC-MED from {root}")
        visits, vs, labs, orders = mcmed_raw(root, ids, args.chunksize)

    H = args.horizon_hours
    out, stats = [], {"no_result_in_horizon": [], "no_vitals": []}
    for case in cases:
        cid = case["case_id"]
        sid = stay_of.get(cid)
        rows, cultures, vitals = [], [], []
        if site == "mimic":
            if sid not in ed.index:
                stats["no_result_in_horizon"].append(cid); continue
            w = ed.loc[sid]
            origin = w.intime
            L = labs[(labs.subject_id == w.subject_id) & (labs.charttime >= origin)
                     & (labs.storetime <= origin + pd.Timedelta(hours=72))]
            for _, r in L.iterrows():
                rows.append(dict(analyte=r.analyte, sample=hours(r.charttime, origin),
                                 available=hours(r.storetime, origin),
                                 value=None if pd.isna(r.valuenum) else float(r.valuenum),
                                 unit=None if pd.isna(r.valueuom) else str(r.valueuom),
                                 flag=None if pd.isna(r.flag) else str(r.flag),
                                 ref=f"itemid={int(r.itemid)}"))
            M = micro[(micro.subject_id == w.subject_id) & (micro.charttime >= origin)
                      & micro.spec_type_desc.str.contains("BLOOD", case=False, na=False)]
            for _, r in M.sort_values("charttime").head(1).iterrows():
                t = hours(r.charttime, origin)
                if 0 <= t <= H:
                    cultures.append(culture_entry(t, site, "spec_type_desc=BLOOD CULTURE"))
            vitals = vital_events(vs[vs.stay_id == sid], origin, H, site)
        else:
            if sid not in visits.index:
                stats["no_result_in_horizon"].append(cid); continue
            origin = visits.loc[sid].Arrival_time
            L = labs[labs.CSN == sid]
            for _, r in L.iterrows():
                if pd.isna(r.value):
                    continue
                order = None if pd.isna(r.Order_time) else hours(r.Order_time, origin)
                available = hours(r.Result_time, origin)
                # Point-of-care results are often entered before the order is filed
                # (Result_time precedes Order_time by minutes). The result is what the
                # clinician can act on; an order time later than it is not a usable
                # sample time and is dropped rather than invalidating the case.
                if order is not None and order > available:
                    order = None
                sample = order if order is not None else available
                rows.append(dict(analyte=r.analyte, order=order, sample=max(sample, 0.0),
                                 available=available, value=float(r.value),
                                 unit=None if pd.isna(r.Component_units) else str(r.Component_units),
                                 flag=None if pd.isna(r.Component_abnormal) else str(r.Component_abnormal),
                                 ref=f"Component_name={r.Component_name}"))
            if len(orders):
                O = orders[orders.CSN == sid].sort_values("Order_time").head(1)
                for _, r in O.iterrows():
                    t = hours(r.Order_time, origin)
                    if 0 <= t <= H:
                        cultures.append(culture_entry(t, site, f"Procedure_name={r.Procedure_name}"))
            vitals = vital_events(vs[vs.CSN == sid], origin, H, site)

        rows = [r for r in rows if r["sample"] >= 0 and r["available"] >= r["sample"]]
        workup = workup_from_rows(rows, H, site) + cultures
        if not any(w["result_status"] == "resulted" for w in workup):
            stats["no_result_in_horizon"].append(cid)
        if not vitals:
            stats["no_vitals"].append(cid)

        t0_state = {k: v for k, v in case["input"].items() if k != "labs"}
        # The triage vital signs are a charted observation at arrival, so they are
        # the latest known vitals at any later cutoff until a re-measurement exists.
        tv = t0_state.get("triage_vitals") or {}
        t0_vitals = {"measurement_time": 0.0}
        for k in ("sbp", "dbp", "resp_rate", "spo2", "heart_rate"):
            if tv.get(k) is not None:
                t0_vitals[k] = tv[k]
        if tv.get("temperature_C") is not None:
            t0_vitals["temperature_C"] = tv["temperature_C"]
        if t0_vitals.get("sbp") is not None and t0_vitals.get("dbp") is not None:
            t0_vitals["map"] = round((t0_vitals["sbp"] + 2 * t0_vitals["dbp"]) / 3.0, 1)
        if len(t0_vitals) > 1:
            t0_vitals["provenance"] = prov(source_table=("ed.triage" if site == "mimic" else "visits.csv Triage_*"),
                                           time_column="arrival", availability_basis="triage snapshot at t0",
                                           map_basis="derived from SBP/DBP" if "map" in t0_vitals else None)
            vitals = [t0_vitals] + [v for v in vitals if v["measurement_time"] > 0]
        out.append({
            "case_id": cid,
            "surrogate_id": case.get("surrogate_id"),
            "stratum": case.get("stratum"), "sub_stratum": case.get("sub_stratum"),
            "label": case.get("label"),
            "t0_time": 0.0,
            "t0_state": t0_state,
            "observed_workup": workup,
            "longitudinal_events": {"vitals": vitals},
            "observed_treatment": {},
            "evaluation_only": {"truth": case.get("truth", {})},
        })

    meta = {
        "data_classification": "real_patient_derived; keep inside the DUA",
        "source": cfg["source"], "n_cases": len(out), "replay_horizon_hours": H,
        "time_basis": "hours since ED arrival",
        "checkpoints": "C0_5 / C1 / C3 / C6 from result_available_time; Cfinal never scheduled because results after the horizon are withheld",
        "limitations": [
            "Earliest result per analyte kept; a second lactate is lactate_repeat; other repeats dropped.",
            "MIMIC order times unknown (null); collection = labevents.charttime; availability = storetime.",
            "MC-MED collection time not recorded; sample_time set to Order_time, or to Result_time when the order was filed after the result (point-of-care).",
            "ED vital availability uses chart time as a proxy.",
            "Blood cultures are pending throughout; final culture results are withheld.",
            "Observed treatment not extracted: empty means unknown, not absent.",
        ],
        "cases_with_no_visible_result": stats["no_result_in_horizon"],
        "cases_with_no_vitals": stats["no_vitals"],
    }
    path = D / (args.out or cfg["out"])
    path.write_text(json.dumps({"meta": meta, "cases": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    n_vis = sum(1 for c in out if any(w["result_status"] == "resulted" for w in c["observed_workup"]))
    first = [min(w["result_available_time"] for w in c["observed_workup"] if w["result_status"] == "resulted")
             for c in out if any(w["result_status"] == "resulted" for w in c["observed_workup"])]
    print(f"-> {path.name}: {len(out)} cases; {n_vis} with a result within {H} h; "
          f"{len(stats['no_vitals'])} without ED vitals")
    if first:
        s = pd.Series(first)
        print(f"   first result at median {s.median():.2f} h (IQR {s.quantile(.25):.2f}-{s.quantile(.75):.2f})")
        for cut in (0.5, 1, 3, 6):
            print(f"   results by {cut} h: {(s <= cut).sum()} cases")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=SITES, required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--data-root", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunksize", type=int, default=2_000_000)
    p.add_argument("--horizon-hours", type=float, default=6.0)
    build(p.parse_args())


if __name__ == "__main__":
    main()
