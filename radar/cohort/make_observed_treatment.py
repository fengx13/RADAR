#!/usr/bin/env python3
"""Add observed ED treatment to the longitudinal case files.

Reads the cached raw rows produced alongside make_longitudinal_cases.py and
writes, per case, `observed_treatment` (antimicrobials, fluids, vasopressors,
respiratory_support) and the matching `longitudinal_events` groups that the
checkpoint state reads (respiratory_support, vasopressors, fluids).

Sources
  MIMIC-IV-ED   ed/pyxis (ED dispensing, charttime)  +  hosp/emar rows with
                event_txt Administered/Started inside the ED window.
                No ED fluid-bolus or oxygen record exists: both stay unknown.
  MC-MED        orders.csv Medications (First_admin_time = administered,
                otherwise ordered only) and Nursing/Respiratory Care oxygen,
                NIV and ventilation orders. --no-oxygen-order-means-none adds a
                t=0 "no supplemental oxygen ordered" baseline for MC-MED
                visits, as a deterministic_derived inference, so that A4 is not
                blind to oxygen on every visit. It is an inference, not a
                charted observation, and is labelled as such.

Usage
    python -m radar.cohort.make_observed_treatment --site mimic
    python -m radar.cohort.make_observed_treatment --site mcmed --no-oxygen-order-means-none
"""
from radar import paths
import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd

from radar.cohort.make_longitudinal_cases import SITES, iso_dt, hours, D, load_cohort

ABX = re.compile(r"cef|cillin|mycin|cycline|penem|floxacin|vancomycin|metronidazole|azithro|"
                 r"tazo|sulfameth|nitrofuran|clinda|linezolid|doxy|amox|acyclovir|fluconazole|"
                 r"meropenem|ertapenem|aztreonam|gentamicin|tobramycin|bactrim|zosyn", re.I)
ABX_EXCLUDE = re.compile(r"trough|peak|level|ophth|otic|topical|cream|oint", re.I)
FLUID = re.compile(r"ns iv bolus|lr iv bolus|lactated ringers iv|sodium chloride 0\.9 ?% +iv|plasma-?lyte|normal saline iv", re.I)
PRESSOR = re.compile(r"norepinephrine|phenylephrine|vasopressin|dopamine|^epinephrine", re.I)
PRESSOR_EXCLUDE = re.compile(r"lidocaine|racepinephrine|inh nebu|epipen|auto-?inj", re.I)
OXY = {
    "OXYGEN: NASAL CANNULA(NURSING ONLY)": ("nasal_cannula", None),
    "RESP - OXYGEN ADMINISTRATION": ("supplemental_oxygen", None),
    "RESP - NON INVASIVE POS. PRESS VENT (BIPAP/CPAP)": ("niv", "noninvasive"),
    "INTUBATION": ("invasive", "invasive"),
    "RESP - VENTILATOR SETTINGS": ("invasive", "invasive"),
    "RESP - LUNG PROTECTIVE VENTILATION PROTOCOL": ("invasive", "invasive"),
}


def prov(**kw):
    return {"source_type": "observed_ehr", **kw}


def ml_from_name(name):
    m = re.search(r"(\d+)\s*ML", name, re.I)
    return int(m.group(1)) if m else None


def route_from_name(name):
    n = name.lower()
    if "po " in n or " tabs" in n or " caps" in n or "oral" in n:
        return "PO"
    if "iv" in n or "bag" in n or "pgbk" in n or "ivpb" in n or "inj" in n or "vial" in n:
        return "IV"
    return None


# ------------------------------------------------------------------ MIMIC
def mimic_events(stay_of, cases, horizon):
    R = paths.RAW_ROOT / "raw"
    ed = pd.read_csv(R / "mimic-iv-ed-2.2/ed/edstays.csv.gz", parse_dates=["intime", "outtime"])
    ed = ed[ed.stay_id.isin(set(stay_of.values()))].set_index("stay_id")
    px = pd.read_csv(D / "_cache_cohort100_pyxis.csv", parse_dates=["charttime"])
    em = pd.read_csv(D / "_cache_cohort100_emar.csv", parse_dates=["charttime", "storetime"])
    em = em[em.event_txt.isin(["Administered", "Started"])]
    out = {}
    for case in cases:
        sid = stay_of.get(case["case_id"])
        if sid not in ed.index:
            continue
        w = ed.loc[sid]
        origin = w.intime
        abx, pressors = [], []
        P = px[px.stay_id == sid]
        for _, r in P.iterrows():
            t = hours(r.charttime, origin)
            if not 0 <= t <= horizon:
                continue
            name = str(r["name"])
            if ABX.search(name) and not ABX_EXCLUDE.search(name):
                abx.append({"administration_time": t, "available_time": t, "documentation_time": t,
                            "medication": name, "agent": name, "route": route_from_name(name),
                            "administered": True, "status": "dispensed_in_ed",
                            "provenance": prov(source_table="ed.pyxis", event_time=t, available_time=t,
                                               note="ED dispensing time; administration time not separately recorded")})
            elif PRESSOR.search(name) and not PRESSOR_EXCLUDE.search(name):
                pressors.append({"start_time": t, "active": True, "setting": "ed", "agent": name,
                                 "provenance": prov(source_table="ed.pyxis", event_time=t)})
        E = em[(em.subject_id == w.subject_id) & (em.charttime >= origin)]
        for _, r in E.iterrows():
            t = hours(r.charttime, origin)
            if not 0 <= t <= horizon:
                continue
            avail = hours(r.storetime, origin) if pd.notna(r.storetime) else t
            name = str(r.medication)
            if ABX.search(name) and not ABX_EXCLUDE.search(name):
                abx.append({"administration_time": t, "available_time": max(avail, t), "documentation_time": max(avail, t),
                            "medication": name, "agent": name, "route": None, "administered": True,
                            "status": "administered" if r.event_txt == "Administered" else "started",
                            "provenance": prov(source_table="hosp.emar", event_time=t, available_time=max(avail, t),
                                               source_record_id=str(r.emar_id) if "emar_id" in r else None)})
            elif PRESSOR.search(name) and not PRESSOR_EXCLUDE.search(name):
                pressors.append({"start_time": t, "active": True, "setting": "ed", "agent": name,
                                 "provenance": prov(source_table="hosp.emar", event_time=t)})
        abx.sort(key=lambda e: e["administration_time"]); pressors.sort(key=lambda e: e["start_time"])
        out[case["case_id"]] = {"antimicrobials": abx, "fluids": [], "vasopressors": pressors,
                                "respiratory_support": []}
    return out


# ------------------------------------------------------------------ MC-MED
def mcmed_events(stay_of, cases, horizon, baseline_no_oxygen):
    R = paths.RAW_ROOT / "raw/mc-med-1.0.0"
    visits = pd.read_csv(R / "visits.csv", usecols=["CSN", "Arrival_time"])
    visits = visits[visits.CSN.isin(set(stay_of.values()))]
    visits["Arrival_time"] = iso_dt(visits.Arrival_time)
    arrival = dict(zip(visits.CSN, visits.Arrival_time))
    o = pd.read_csv(D / "_cache_stanford100_orders_all.csv", low_memory=False)
    o["Order_time"] = iso_dt(o.Order_time)
    o["First_admin_time"] = iso_dt(o.First_admin_time)
    out = {}
    for case in cases:
        csn = stay_of.get(case["case_id"])
        if csn not in arrival or pd.isna(arrival[csn]):
            continue
        origin = arrival[csn]
        abx, fluids, pressors, resp = [], [], [], []
        if baseline_no_oxygen:
            resp.append({"start_time": 0.0, "oxygen_therapy_documented": False, "mode": "none_ordered",
                         "ventilation_status": None, "current_encounter": True,
                         "provenance": {"source_type": "deterministic_derived",
                                        "note": "no oxygen/NIV/ventilation order in orders.csv by this time; "
                                                "order-based inference, not a charted observation"}})
        O = o[o.CSN == csn]
        for _, r in O.iterrows():
            name = str(r.Procedure_name)
            t_order = hours(r.Order_time, origin) if pd.notna(r.Order_time) else None
            t_admin = hours(r.First_admin_time, origin) if pd.notna(r.First_admin_time) else None
            if r.Order_type == "Medications":
                t = t_admin if t_admin is not None else t_order
                if t is None or not 0 <= t <= horizon:
                    continue
                administered = t_admin is not None
                if ABX.search(name) and not ABX_EXCLUDE.search(name):
                    abx.append({"administration_time": t, "available_time": t, "documentation_time": t,
                                "order_time": t_order, "medication": name, "agent": name,
                                "route": route_from_name(name), "administered": administered,
                                "status": "administered" if administered else "ordered_not_yet_administered",
                                "provenance": prov(source_table="orders.csv", event_time=t, available_time=t,
                                                   time_basis="First_admin_time" if administered else "Order_time")})
                elif FLUID.search(name):
                    if not administered:
                        continue
                    fluids.append({"event_time": t, "administered": True, "amount_ml": ml_from_name(name),
                                   "product": name, "initial_fluid_30ml_per_kg_completed": None,
                                   "completion_time": None,
                                   "provenance": prov(source_table="orders.csv", event_time=t,
                                                      note="30 mL/kg completion not assessable: weight unknown")})
                elif PRESSOR.search(name) and not PRESSOR_EXCLUDE.search(name):
                    if not administered:
                        continue
                    pressors.append({"start_time": t, "active": True, "setting": "ed", "agent": name,
                                     "provenance": prov(source_table="orders.csv", event_time=t)})
            elif name in OXY:
                if t_order is None or not 0 <= t_order <= horizon:
                    continue
                mode, vent = OXY[name]
                resp.append({"start_time": t_order, "oxygen_therapy_documented": True, "mode": mode,
                             "ventilation_status": vent, "current_encounter": True,
                             "provenance": prov(source_table="orders.csv", event_time=t_order,
                                                note="order time; start of therapy not separately recorded")})
        for lst, key in ((abx, "administration_time"), (fluids, "event_time"), (pressors, "start_time"), (resp, "start_time")):
            lst.sort(key=lambda e: e[key])
        out[case["case_id"]] = {"antimicrobials": abx, "fluids": fluids, "vasopressors": pressors,
                                "respiratory_support": resp}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=SITES, required=True)
    p.add_argument("--horizon-hours", type=float, default=6.0)
    p.add_argument("--no-oxygen-order-means-none", action="store_true")
    a = p.parse_args()
    cases_t0, stay_of = load_cohort(a.site, None)
    path = D / SITES[a.site]["out"]
    doc = json.loads(path.read_text(encoding="utf-8"))
    ev = mimic_events(stay_of, cases_t0, a.horizon_hours) if a.site == "mimic" else \
        mcmed_events(stay_of, cases_t0, a.horizon_hours, a.no_oxygen_order_means_none)
    n = {"antimicrobials": 0, "fluids": 0, "vasopressors": 0, "respiratory_support": 0}
    for c in doc["cases"]:
        e = ev.get(c["case_id"], {"antimicrobials": [], "fluids": [], "vasopressors": [], "respiratory_support": []})
        c["observed_treatment"] = e
        c["longitudinal_events"].update({k: e[k] for k in ("respiratory_support", "vasopressors", "fluids")})
        for k in n:
            n[k] += bool([x for x in e[k] if x.get("provenance", {}).get("source_type") == "observed_ehr"])
    doc["meta"]["observed_treatment"] = {
        "horizon_hours": a.horizon_hours, "cases_with_observed": n,
        "sources": ("ed.pyxis + hosp.emar (Administered/Started); fluids and oxygen not recorded in MIMIC ED (unknown)"
                    if a.site == "mimic" else
                    "orders.csv Medications (First_admin_time) and oxygen/NIV/ventilation orders"),
        "no_oxygen_order_means_none": a.no_oxygen_order_means_none if a.site == "mcmed" else None,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {path.name}: cases with observed {n}")
    abx_t = [x["administration_time"] for c in doc["cases"] for x in c["observed_treatment"]["antimicrobials"] if x["administered"]]
    if abx_t:
        s = pd.Series(abx_t)
        print(f"   first antimicrobial per record: median {s.median():.2f} h; within 1 h: {(s<=1).sum()} records, within 3 h: {(s<=3).sum()}")


if __name__ == "__main__":
    main()
