#!/usr/bin/env python3
"""Add infection-relevant and alternative-diagnosis results to the longitudinal cases.

Recognize at a checkpoint sees `investigation_results`; until now that held
chemistry and blood counts only, so R1 (is infection suspected?) kept answering
`indeterminate` for want of any evidence about a source. This adds, each with
its own result-availability time and withheld after the 6 h horizon:

  imaging          radiology IMPRESSION text (MIMIC: hosp radiology notes within the
                   ED stay, storetime; MC-MED: rads.csv Impression, Result_time).
                   The first report also fills the contract item
                   `source_imaging_or_sampling`; later ones are imaging_2, _3 ...
  urinalysis       nitrite, leukocyte esterase, blood, protein, ketone (+bacteria,
                   WBC in MIMIC) summarised at the panel's latest result time
  respiratory_viral_panel   influenza A/B, RSV (MC-MED only)
  gram_stain       microbiology Gram stain comments (MIMIC, rare within 6 h)
  numeric          crp, procalcitonin, troponin (T / I / hs), ntprobnp

Nothing is derived or interpreted: free text is passed verbatim (identifier
blanks "___" kept as in the source), values are copied with units.

Reads <site>_longitudinal_cases.json and writes <site>_longitudinal_cases_v4.json.

Usage
    python -m radar.cohort.make_infection_results --site mimic
    python -m radar.cohort.make_infection_results --site mcmed
"""
from radar import paths
import argparse
import json
import pathlib
import re

import pandas as pd

from radar.cohort.make_longitudinal_cases import SITES, iso_dt, hours, D, load_cohort

RAW = paths.RAW_ROOT / "raw"
UA = ("ua_nitrite", "ua_leukocytes", "ua_blood", "ua_protein", "ua_ketone", "ua_bacteria", "ua_wbc")
VIRAL = ("influenza_a", "influenza_b", "rsv")
NUMERIC = ("crp", "procalcitonin", "troponin_t", "troponin_i", "troponin_hs", "ntprobnp")


def prov(**kw):
    return {"source_type": "observed_ehr", **kw}


def entry(item, sample, available, horizon, kind, summary=None, value=None, unit=None, flag=None, **p):
    visible = available is not None and available <= horizon
    e = {"item": item, "actual_order_status": "collected", "order_time": None,
         "sample_time": sample, "result_kind": kind, "disposition_relevant": True,
         "checkpoint_eligible": True, "provenance": prov(**p)}
    if visible:
        e.update({"result_status": "resulted", "result_available_time": available, "value": value,
                  "unit": unit, "abnormal_flag": flag, "result_summary": summary})
    else:
        e.update({"result_status": "pending", "result_available_time": None, "value": None, "unit": None,
                  "result_summary": None})
        e["provenance"]["note"] = "resulted after the replay horizon; withheld"
    return e


def impression(text):
    t = str(text)
    m = re.search(r"IMPRESSION:?\s*(.*)", t, re.S | re.I)
    body = m.group(1) if m else t
    body = re.sub(r"\s+", " ", body).strip()
    return body[:600] + ("…" if len(body) > 600 else "")


def exam_name(text):
    m = re.search(r"EXAMINATION:\s*(.+)", str(text))
    if m:
        return m.group(1).strip()[:80]
    return str(text).strip().split("\n")[0][:80]


def mimic_events(stay_of, cases, H):
    ed = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/edstays.csv.gz", usecols=["subject_id", "stay_id", "intime", "outtime"],
                     parse_dates=["intime", "outtime"])
    ed = ed[ed.stay_id.isin(set(stay_of.values()))].set_index("stay_id")
    lab = pd.read_csv(D / "_cache_cohort100_labevents_extra.csv", parse_dates=["charttime", "storetime"], low_memory=False)
    rad = pd.read_csv(D / "_cache_cohort100_radiology.csv", parse_dates=["charttime", "storetime"])
    mic = pd.read_csv(RAW / "mimic-iv-3.1/hosp/microbiologyevents.csv.gz",
                      usecols=["subject_id", "charttime", "storetime", "spec_type_desc", "test_name", "comments"],
                      parse_dates=["charttime", "storetime"], low_memory=False)
    mic = mic[mic.subject_id.isin(set(ed.subject_id)) & (mic.test_name == "GRAM STAIN")]
    out = {}
    for case in cases:
        sid = stay_of.get(case["case_id"])
        if sid not in ed.index:
            continue
        w = ed.loc[sid]; o = w.intime
        items = []
        # imaging
        R = rad[(rad.subject_id == w.subject_id) & (rad.charttime >= o) & rad.storetime.notna()].sort_values("storetime")
        for n, (_, r) in enumerate(R.iterrows(), 1):
            sample, avail = hours(r.charttime, o), hours(r.storetime, o)
            if sample > H:
                break
            name = "source_imaging_or_sampling" if n == 1 else f"imaging_{n}"
            items.append(entry(name, sample, avail, H, "imaging",
                               summary=f"{exam_name(r.text)} — IMPRESSION: {impression(r.text)}",
                               source_table="note.radiology", source_record_id=str(r.note_id),
                               time_basis="sample=charttime (exam), available=storetime (report)"))
        # UA panel
        L = lab[(lab.subject_id == w.subject_id) & (lab.charttime >= o) & lab.storetime.notna()]
        ua = L[L.analyte.isin(UA)]
        if len(ua):
            first = ua.charttime.min()
            panel = ua[ua.charttime == first]
            avail = hours(panel.storetime.max(), o); sample = hours(first, o)
            parts = [f"{r.analyte[3:]}: {r.value}" for _, r in panel.sort_values("analyte").iterrows() if pd.notna(r.value)]
            items.append(entry("urinalysis", sample, avail, H, "diagnostic_laboratory", summary="; ".join(parts),
                               source_table="hosp.labevents", time_basis="sample=charttime, available=max storetime of the panel"))
        # numeric
        for name in NUMERIC:
            N = L[L.analyte == name].sort_values("storetime")
            if len(N):
                r = N.iloc[0]
                items.append(entry(name, hours(r.charttime, o), hours(r.storetime, o), H, "diagnostic_laboratory",
                                   value=None if pd.isna(r.valuenum) else float(r.valuenum),
                                   unit=None if pd.isna(r.valueuom) else str(r.valueuom),
                                   flag=None if pd.isna(r.flag) else str(r.flag),
                                   source_table="hosp.labevents", source_ref=f"itemid={int(r.itemid)}"))
        # gram stain
        M = mic[(mic.subject_id == w.subject_id) & (mic.charttime >= o) & mic.storetime.notna()].sort_values("storetime")
        if len(M):
            r = M.iloc[0]
            items.append(entry("gram_stain", hours(r.charttime, o), hours(r.storetime, o), H, "diagnostic_laboratory",
                               summary=f"{r.spec_type_desc}: {str(r.comments)[:300]}", source_table="hosp.microbiologyevents"))
        out[case["case_id"]] = items
    return out


def mcmed_events(stay_of, cases, H):
    root = RAW / "mc-med-1.0.0"
    visits = pd.read_csv(root / "visits.csv", usecols=["CSN", "Arrival_time"])
    visits = visits[visits.CSN.isin(set(stay_of.values()))]
    arrival = dict(zip(visits.CSN, iso_dt(visits.Arrival_time)))
    lab = pd.read_csv(D / "_cache_stanford100_labs_extra.csv", low_memory=False)
    lab["o"] = iso_dt(lab.Order_time); lab["r"] = iso_dt(lab.Result_time)
    rad = pd.read_csv(D / "_cache_stanford100_rads.csv")
    rad["o"] = iso_dt(rad.Order_time); rad["r"] = iso_dt(rad.Result_time)
    out = {}
    for case in cases:
        csn = stay_of.get(case["case_id"])
        if csn not in arrival or pd.isna(arrival[csn]):
            continue
        a = arrival[csn]; items = []
        R = rad[(rad.CSN == csn) & rad.r.notna()].sort_values("r")
        n = 0
        for _, r in R.iterrows():
            avail = hours(r.r, a)
            if avail < 0:
                continue
            sample = min(hours(r.o, a), avail) if pd.notna(r.o) else avail
            if sample > H:
                continue
            n += 1
            name = "source_imaging_or_sampling" if n == 1 else f"imaging_{n}"
            items.append(entry(name, max(sample, 0.0), avail, H, "imaging",
                               summary=f"{r.Study} — IMPRESSION: {impression(r.Impression)}",
                               source_table="rads.csv", time_basis="sample=Order_time, available=Result_time"))
        L = lab[(lab.CSN == csn) & lab.r.notna()].copy()
        ua = L[L.analyte.isin(UA)]
        if len(ua):
            first = ua.o.min() if ua.o.notna().any() else ua.r.min()
            avail = hours(ua.r.max(), a); sample = max(min(hours(first, a), avail), 0.0)
            parts = [f"{r.analyte[3:]}: {r.Component_result if pd.notna(r.Component_result) else r.Component_value}" for _, r in ua.sort_values("analyte").iterrows()]
            items.append(entry("urinalysis", sample, avail, H, "diagnostic_laboratory", summary="; ".join(parts),
                               source_table="labs.csv", time_basis="available=max Result_time of the panel"))
        vp = L[L.analyte.isin(VIRAL)]
        if len(vp):
            avail = hours(vp.r.max(), a); sample = max(min(hours(vp.o.min(), a) if vp.o.notna().any() else avail, avail), 0.0)
            parts = [f"{r.analyte}: {r.Component_result if pd.notna(r.Component_result) else r.Component_value}" for _, r in vp.sort_values("analyte").iterrows()]
            items.append(entry("respiratory_viral_panel", sample, avail, H, "diagnostic_laboratory", summary="; ".join(parts),
                               source_table="labs.csv"))
        for name in NUMERIC:
            N = L[L.analyte == name].sort_values("r")
            if len(N):
                r = N.iloc[0]; v = pd.to_numeric(r.Component_value, errors="coerce")
                avail = hours(r.r, a); sample = max(min(hours(r.o, a) if pd.notna(r.o) else avail, avail), 0.0)
                items.append(entry(name, sample, avail, H, "diagnostic_laboratory",
                                   value=None if pd.isna(v) else float(v), summary=None if not pd.isna(v) else str(r.Component_value),
                                   unit=None if pd.isna(r.Component_units) else str(r.Component_units),
                                   flag=None if pd.isna(r.Component_abnormal) else str(r.Component_abnormal),
                                   source_table="labs.csv", source_ref=f"Component_name={r.Component_name}"))
        out[case["case_id"]] = items
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=SITES, required=True)
    p.add_argument("--horizon-hours", type=float, default=6.0)
    a = p.parse_args()
    cases_t0, stay_of = load_cohort(a.site, None)
    base = D / SITES[a.site]["out"]
    path = base.with_name(base.stem + "_v4.json")     # never overwrite the input a run may be reading
    doc = json.loads(base.read_text(encoding="utf-8"))
    ev = (mimic_events if a.site == "mimic" else mcmed_events)(stay_of, cases_t0, a.horizon_hours)
    counts = {}
    for c in doc["cases"]:
        existing = {w["item"] for w in c["observed_workup"]}
        new = [e for e in ev.get(c["case_id"], []) if e["item"] not in existing]
        # keep the earlier builder's entries; drop a stale copy of ours from a previous run
        c["observed_workup"] = [w for w in c["observed_workup"] if not (w["item"].startswith("imaging_") or w["item"] in
                               {"source_imaging_or_sampling", "urinalysis", "respiratory_viral_panel", "gram_stain", *NUMERIC})] + ev.get(c["case_id"], [])
        for e in ev.get(c["case_id"], []):
            if e["result_status"] == "resulted":
                k = "imaging" if e["result_kind"] == "imaging" else e["item"]
                counts[k] = counts.get(k, 0) + 1
    doc["meta"]["infection_results"] = {"horizon_hours": a.horizon_hours, "resulted_items_by_kind": counts,
                                        "note": "verbatim report impressions / panel summaries; nothing derived"}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    vis = sum(1 for c in doc["cases"] if any(w["result_kind"] == "imaging" and w["result_status"] == "resulted" for w in c["observed_workup"]))
    print(f"-> {path.name}: resulted items {counts}; visits with an imaging report within {a.horizon_hours} h: {vis}")


if __name__ == "__main__":
    main()
