#!/usr/bin/env python3
"""v5: the full record a clinician could open — pre-arrival history and every
laboratory result of the ED stay — under strict time gating.

Adds to each case of <site>_longitudinal_cases_v4.json and writes ..._v5.json:

  t0_state.prior_history           everything dated BEFORE ED arrival
    problem_list                   MIMIC: ICD titles of previous admissions (index
                                   admission excluded); MC-MED: pmh.csv rows with
                                   Noted_date <= arrival
    previous_encounters            MIMIC: previous admissions (days before, LOS,
                                   diagnoses); MC-MED: previous ED visits of the
                                   same patient (days before, chief complaint,
                                   disposition, diagnosis of THAT visit)
    last_discharge_summary         MIMIC only: excerpt (Allergies, Past Medical
                                   History, Discharge Diagnosis) of the most
                                   recent discharge summary written before arrival
    home_medications               MIMIC: ED medication reconciliation (medrecon);
                                   MC-MED: meds.csv entries active at arrival
    baseline_labs                  most recent pre-arrival value per analyte
                                   (creatinine, platelets, bilirubin, ...), with
                                   days before arrival — serves sepsis3::baseline_sofa
  observed_workup += lab_<name>    every analyte resulted in the ED stay that the
                                   earlier builders did not carry (MIMIC labevents
                                   all itemids; MC-MED labs.csv all components),
                                   first result per analyte, result_available_time
                                   gated exactly like the other items

Never included: the index visit's discharge summary, its diagnosis codes, and
anything time-stamped after the replay horizon. Free text is copied verbatim
(MIMIC de-identification blanks "___" kept).

Caches (_cache_<site>_fullrecord_*.csv) are built once from the raw
tables; MIMIC's discharge.csv (3.3 GB) and labevents (2.4 GB gz) take minutes:
    python -m radar.cohort.make_full_record --site mimic --build-cache
    python -m radar.cohort.make_full_record --site mimic
    python -m radar.cohort.make_full_record --site mcmed --build-cache && python -m radar.cohort.make_full_record --site mcmed
"""
import argparse
import json
import pathlib
import re

import pandas as pd

from radar.cohort.make_longitudinal_cases import SITES, iso_dt, hours, D, load_cohort
from radar.cohort.make_infection_results import entry, RAW
from radar.cohort.make_t1_states import LAB_ITEMS

H_DEFAULT = 6.0
BASELINE = ["creatinine", "platelets", "bilirubin_total", "wbc", "hemoglobin", "lactate", "inr", "bun",
            "sodium", "potassium", "albumin", "hba1c", "bnp", "ntprobnp"]
BASELINE_ITEMS = {**{k: v for k, v in LAB_ITEMS.items()}, 50862: "albumin", 50852: "hba1c", 50963: "ntprobnp",
                  51000: "bnp"}
MC_BASELINE = {"CREATININE": "creatinine", "PLATELET COUNT (PLT)": "platelets", "BILIRUBIN, TOTAL": "bilirubin_total",
               "WHITE BLOOD CELLS (WBC)": "wbc", "HEMOGLOBIN (HGB)": "hemoglobin", "LACTATE.WHOLE BLD": "lactate",
               "INR": "inr", "BUN": "bun", "SODIUM": "sodium", "POTASSIUM": "potassium", "ALBUMIN": "albumin",
               "HEMOGLOBIN A1C": "hba1c", "NT-PROBNP": "ntprobnp"}
PRE_WINDOW_DAYS = 730
MAX_SECTION = 1200
MAX_LIST = 40


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")[:40]


def section(text, *titles):
    """Verbatim excerpt of one discharge-summary section (first title that matches)."""
    for t in titles:
        m = re.search(rf"\n{re.escape(t)}:?\s*\n(.*?)(?=\n[A-Z][A-Za-z /]{{2,40}}:|\Z)", text, re.S)
        if m:
            body = re.sub(r"[ \t]+", " ", m.group(1)).strip()
            body = re.sub(r"\n{2,}", "\n", body)
            return body[:MAX_SECTION] + ("…" if len(body) > MAX_SECTION else "")
    return None


def days(t, origin):
    return round((origin - t).total_seconds() / 86400.0, 1)


# ------------------------------------------------------------------ MIMIC
def mimic_cache(stay_ids, parts=("small", "discharge", "labevents")):
    ed = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/edstays.csv.gz", parse_dates=["intime", "outtime"])
    ed = ed[ed.stay_id.isin(stay_ids)]
    subjects = set(ed.subject_id)
    lo = ed.groupby("subject_id").intime.min() - pd.Timedelta(days=PRE_WINDOW_DAYS)
    hi = ed.groupby("subject_id").outtime.max() + pd.Timedelta(hours=1)
    if "small" in parts:
        _mimic_cache_small(ed, subjects, stay_ids)
    if "discharge" in parts:
        _mimic_cache_discharge(ed, subjects)
    if "labevents" in parts:
        _mimic_cache_labevents(subjects, lo, hi)


def _mimic_cache_small(ed, subjects, stay_ids):
    ed.to_csv(D / "_cache_cohort100_fullrecord_edstays.csv", index=False)
    adm = pd.read_csv(RAW / "mimic-iv-3.1/hosp/admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    adm = adm[adm.subject_id.isin(subjects)]
    adm.to_csv(D / "_cache_cohort100_fullrecord_admissions.csv", index=False)
    dx = pd.read_csv(RAW / "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz")
    dx = dx[dx.subject_id.isin(subjects)]
    titles = pd.read_csv(RAW / "mimic-iv-3.1/hosp/d_icd_diagnoses.csv.gz")
    dx = dx.merge(titles, on=["icd_code", "icd_version"], how="left")
    dx.to_csv(D / "_cache_cohort100_fullrecord_diagnoses.csv", index=False)
    mr = pd.read_csv(RAW / "mimic-iv-ed-2.2/ed/medrecon.csv.gz")
    mr = mr[mr.stay_id.isin(stay_ids)]
    mr.to_csv(D / "_cache_cohort100_fullrecord_medrecon.csv", index=False)
    print("edstays/admissions/diagnoses/medrecon cached", flush=True)


def _mimic_cache_discharge(ed, subjects):
    # discharge summaries written before the index arrival only
    first_arrival = ed.groupby("subject_id").intime.min()
    keep = []
    for i, ch in enumerate(pd.read_csv(RAW / "mimic-iv-note-2.2/note/discharge.csv", chunksize=20000,
                                       parse_dates=["charttime", "storetime"]), 1):
        ch = ch[ch.subject_id.isin(subjects)]
        if len(ch):
            ch = ch[ch.storetime < ch.subject_id.map(first_arrival)]
            keep.append(ch)
        if i % 10 == 0:
            print(f"  discharge.csv chunk {i}, kept {sum(len(k) for k in keep)}", flush=True)
    (pd.concat(keep) if keep else pd.DataFrame()).to_csv(D / "_cache_cohort100_fullrecord_discharge.csv", index=False)
    print("discharge cached", flush=True)


def _mimic_cache_labevents(subjects, lo, hi):
    items = pd.read_csv(RAW / "mimic-iv-3.1/hosp/d_labitems.csv.gz")
    keep, seen = [], 0
    for i, ch in enumerate(pd.read_csv(RAW / "mimic-iv-3.1/hosp/labevents.csv.gz", chunksize=2_000_000,
                                       usecols=["subject_id", "hadm_id", "itemid", "charttime", "storetime", "value",
                                                "valuenum", "valueuom", "flag"], low_memory=False), 1):
        seen += len(ch)
        ch = ch[ch.subject_id.isin(subjects)]
        if len(ch):
            ch = ch.copy(); ch["charttime"] = pd.to_datetime(ch.charttime); ch["storetime"] = pd.to_datetime(ch.storetime)
            ch = ch[(ch.charttime >= ch.subject_id.map(lo)) & (ch.charttime <= ch.subject_id.map(hi))]
            keep.append(ch)
        if i % 10 == 0:
            print(f"  labevents {seen:,} scanned, kept {sum(len(k) for k in keep):,}", flush=True)
    lab = pd.concat(keep) if keep else pd.DataFrame()
    lab = lab.merge(items[["itemid", "label", "fluid", "category"]], on="itemid", how="left")
    lab.to_csv(D / "_cache_cohort100_fullrecord_labevents.csv", index=False)
    print("labevents cached", flush=True)


def mimic_build(stay_of, cases, H):
    ed = pd.read_csv(D / "_cache_cohort100_fullrecord_edstays.csv", parse_dates=["intime", "outtime"]).set_index("stay_id")
    adm = pd.read_csv(D / "_cache_cohort100_fullrecord_admissions.csv", parse_dates=["admittime", "dischtime"])
    dx = pd.read_csv(D / "_cache_cohort100_fullrecord_diagnoses.csv")
    mr = pd.read_csv(D / "_cache_cohort100_fullrecord_medrecon.csv")
    ds = pd.read_csv(D / "_cache_cohort100_fullrecord_discharge.csv", parse_dates=["charttime", "storetime"])
    lab = pd.read_csv(D / "_cache_cohort100_fullrecord_labevents.csv", parse_dates=["charttime", "storetime"], low_memory=False)
    known_items = set(LAB_ITEMS) | set(BASELINE_ITEMS)
    out = {}
    for case in cases:
        sid = stay_of.get(case["case_id"])
        if sid not in ed.index:
            continue
        w = ed.loc[sid]; o = w.intime; subj = w.subject_id; index_hadm = w.hadm_id
        hist = {"as_of": "ED arrival (T0); nothing dated after arrival is included",
                "source": "MIMIC-IV hosp (previous admissions, diagnoses, labs), MIMIC-IV-Note (previous discharge summary), MIMIC-IV-ED medrecon"}
        # previous admissions (index admission excluded by hadm_id and by time)
        A = adm[(adm.subject_id == subj) & (adm.dischtime < o)]
        A = A[A.hadm_id != index_hadm].sort_values("dischtime", ascending=False)
        enc = []
        for _, a in A.head(5).iterrows():
            d_ = dx[dx.hadm_id == a.hadm_id].sort_values("seq_num")
            enc.append({"days_before_arrival": days(a.dischtime, o), "length_of_stay_days": round((a.dischtime - a.admittime).days, 1),
                        "admission_type": a.admission_type, "diagnoses": [str(t) for t in d_.long_title.head(12)]})
        hist["previous_encounters"] = enc
        hist["previous_encounters_count"] = int(len(A))
        allp = dx[dx.hadm_id.isin(set(A.hadm_id))].long_title.dropna()
        hist["problem_list"] = list(dict.fromkeys(allp))[:MAX_LIST]
        # last discharge summary before arrival
        N = ds[(ds.subject_id == subj) & (ds.storetime < o) & (ds.hadm_id != index_hadm)].sort_values("storetime", ascending=False)
        if len(N):
            n = N.iloc[0]; t = str(n.text)
            hist["last_discharge_summary"] = {
                "days_before_arrival": days(n.storetime, o), "note_type": "discharge summary (verbatim excerpt)",
                "allergies": section(t, "Allergies"),
                "past_medical_history": section(t, "Past Medical History"),
                "discharge_diagnosis": section(t, "Discharge Diagnosis", "Discharge Diagnoses")}
        # home medications
        M = mr[mr.stay_id == sid]
        hist["home_medications"] = [{"name": str(r["name"]), "class": None if pd.isna(r.etcdescription) else str(r.etcdescription)}
                                    for _, r in M.drop_duplicates("name").head(MAX_LIST).iterrows()]
        hist["home_medications_source"] = "ED medication reconciliation at this visit (medrecon)"
        # baseline labs
        L = lab[lab.subject_id == subj]
        pre = L[(L.charttime < o)].copy()
        pre["analyte"] = pre.itemid.map(BASELINE_ITEMS)
        base = {}
        for an in BASELINE:
            P = pre[(pre.analyte == an) & pre.valuenum.notna()].sort_values("charttime", ascending=False)
            if len(P):
                r = P.iloc[0]
                base[an] = {"value": float(r.valuenum), "unit": None if pd.isna(r.valueuom) else str(r.valueuom),
                            "days_before_arrival": days(r.charttime, o)}
        hist["baseline_labs"] = base
        hist["baseline_labs_note"] = "most recent value before this ED arrival; absent = no prior value in MIMIC-IV"
        # all other ED-stay labs
        items = []
        W = L[(L.charttime >= o) & (L.charttime <= o + pd.Timedelta(hours=H)) & ~L.itemid.isin(known_items)]
        W = W[W.label.notna()].sort_values("storetime")
        for label, grp in W.groupby("label", sort=False):
            r = grp.iloc[0]
            if pd.isna(r.storetime):
                continue
            val = None if pd.isna(r.valuenum) else float(r.valuenum)
            items.append(entry(f"lab_{slug(label)}", hours(r.charttime, o), hours(r.storetime, o), H, "diagnostic_laboratory",
                               value=val, summary=None if val is not None else (None if pd.isna(r.value) else str(r.value)[:200]),
                               unit=None if pd.isna(r.valueuom) else str(r.valueuom),
                               flag=None if pd.isna(r.flag) else str(r.flag),
                               source_table="hosp.labevents", source_ref=f"itemid={int(r.itemid)} ({r.fluid}; {r.category})"))
        out[case["case_id"]] = (hist, items)
    return out


# ------------------------------------------------------------------ MC-MED
def mcmed_cache(csns):
    root = RAW / "mc-med-1.0.0"
    v = pd.read_csv(root / "visits.csv", low_memory=False)
    mrns = set(v[v.CSN.isin(csns)].MRN)
    v[v.MRN.isin(mrns)].to_csv(D / "_cache_stanford100_fullrecord_visits.csv", index=False)
    p = pd.read_csv(root / "pmh.csv"); p[p.MRN.isin(mrns)].to_csv(D / "_cache_stanford100_fullrecord_pmh.csv", index=False)
    m = pd.read_csv(root / "meds.csv", low_memory=False); m[m.MRN.isin(mrns)].to_csv(D / "_cache_stanford100_fullrecord_meds.csv", index=False)
    print("visits/pmh/meds cached", flush=True)
    keep = []
    for i, ch in enumerate(pd.read_csv(root / "labs.csv", chunksize=500000, low_memory=False), 1):
        ch = ch[ch.CSN.isin(set(v[v.MRN.isin(mrns)].CSN))]
        if len(ch):
            keep.append(ch)
        if i % 5 == 0:
            print(f"  labs.csv chunk {i}, kept {sum(len(k) for k in keep):,}", flush=True)
    pd.concat(keep).to_csv(D / "_cache_stanford100_fullrecord_labs.csv", index=False)
    print("labs cached", flush=True)


def mcmed_build(stay_of, cases, H):
    v = pd.read_csv(D / "_cache_stanford100_fullrecord_visits.csv", low_memory=False)
    v["arr"] = iso_dt(v.Arrival_time); v["dep"] = iso_dt(v.Departure_time)
    p = pd.read_csv(D / "_cache_stanford100_fullrecord_pmh.csv"); p["t"] = iso_dt(p.Noted_date)
    m = pd.read_csv(D / "_cache_stanford100_fullrecord_meds.csv", low_memory=False)
    m["s"] = iso_dt(m.Start_date); m["e"] = iso_dt(m.End_date); m["entry"] = iso_dt(m.Entry_date)
    lab = pd.read_csv(D / "_cache_stanford100_fullrecord_labs.csv", low_memory=False)
    lab["o"] = iso_dt(lab.Order_time); lab["r"] = iso_dt(lab.Result_time)
    v = v.set_index("CSN")
    known = set(MC_BASELINE) | {"GLUCOSE", "ANION GAP", "POC:LACTATE, ISTAT", "POC:PO2 (A), ISTAT", "POC:CREATININE,ISTAT", "PO2 (A)"}
    out = {}
    for case in cases:
        csn = stay_of.get(case["case_id"])
        if csn not in v.index or pd.isna(v.loc[csn, "arr"]):
            continue
        a = v.loc[csn, "arr"]; mrn = v.loc[csn, "MRN"]
        day0 = a.normalize()      # pmh/meds carry dates only: entries dated the arrival day may stem from this visit
        hist = {"as_of": "ED arrival (T0); problem-list and medication entries dated on or after the arrival day are excluded",
                "source": "MC-MED pmh.csv (problem list), visits.csv (previous ED visits of this patient), meds.csv (medication list), labs.csv"}
        P = p[(p.MRN == mrn) & (p.t < day0)].sort_values("t", ascending=False)
        hist["problem_list"] = [{"diagnosis": str(r.Desc10), "noted_days_before_arrival": days(r.t, a)}
                                for _, r in P.drop_duplicates("Desc10").head(MAX_LIST).iterrows()]
        V = v[(v.MRN == mrn) & (v.dep < a)].sort_values("arr", ascending=False)
        hist["previous_encounters"] = [{"days_before_arrival": days(r.arr, a), "chief_complaint": str(r.CC),
                                        "ed_disposition": None if pd.isna(r.ED_dispo) else str(r.ED_dispo),
                                        "diagnosis_of_that_visit": None if pd.isna(r.Dx_name) else str(r.Dx_name)}
                                       for _, r in V.head(5).iterrows()]
        hist["previous_encounters_count"] = int(len(V))
        M = m[(m.MRN == mrn) & (m.s.notna()) & (m.s < day0) & ((m.e.isna()) | (m.e >= day0))]
        hist["home_medications"] = [{"name": str(r.Generic_name if pd.notna(r.Generic_name) else r.Name),
                                     "class": None if pd.isna(r.Med_class) else str(r.Med_class)}
                                    for _, r in M.drop_duplicates("Generic_name").head(MAX_LIST).iterrows()]
        hist["home_medications_source"] = "medication list entries active on the arrival date (meds.csv)"
        # baseline labs from previous visits of the same patient
        prev_csns = set(V.index)
        PL = lab[lab.CSN.isin(prev_csns) & (lab.r < a)].copy()
        base = {}
        for comp, an in MC_BASELINE.items():
            Q = PL[PL.Component_name == comp].sort_values("r", ascending=False)
            Q = Q[pd.to_numeric(Q.Component_value, errors="coerce").notna()]
            if len(Q):
                r = Q.iloc[0]
                base[an] = {"value": float(r.Component_value), "unit": None if pd.isna(r.Component_units) else str(r.Component_units),
                            "days_before_arrival": days(r.r, a)}
        hist["baseline_labs"] = base
        hist["baseline_labs_note"] = "most recent value from a previous MC-MED visit; absent = none in the data set"
        # all other labs of this visit
        items = []
        L = lab[(lab.CSN == csn) & lab.r.notna() & ~lab.Component_name.isin(known)].sort_values("r")
        for comp, grp in L.groupby("Component_name", sort=False):
            r = grp.iloc[0]
            avail = hours(r.r, a)
            if avail < 0:
                continue
            sample = max(min(hours(r.o, a), avail) if pd.notna(r.o) else avail, 0.0)
            if sample > H:
                continue
            val = pd.to_numeric(r.Component_value, errors="coerce")
            items.append(entry(f"lab_{slug(comp)}", sample, avail, H, "diagnostic_laboratory",
                               value=None if pd.isna(val) else float(val),
                               summary=None if not pd.isna(val) else (None if pd.isna(r.Component_result) else str(r.Component_result)[:200]),
                               unit=None if pd.isna(r.Component_units) else str(r.Component_units),
                               flag=None if pd.isna(r.Component_abnormal) else str(r.Component_abnormal),
                               source_table="labs.csv", source_ref=f"Component_name={comp}; panel={r.Display_name}",
                               time_basis="sample=Order_time, available=Result_time"))
        out[case["case_id"]] = (hist, items)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", choices=SITES, required=True)
    ap.add_argument("--build-cache", action="store_true")
    ap.add_argument("--cache-parts", default="small,discharge,labevents", help="MIMIC only: subset of small,discharge,labevents")
    ap.add_argument("--horizon-hours", type=float, default=H_DEFAULT)
    ap.add_argument("--infile-suffix", default="_v4")
    a = ap.parse_args()
    cases_t0, stay_of = load_cohort(a.site, None)
    if a.build_cache:
        if a.site == "mimic":
            mimic_cache(set(stay_of.values()), tuple(a.cache_parts.split(",")))
        else:
            mcmed_cache(set(stay_of.values()))
        return
    base = D / SITES[a.site]["out"]
    src = base.with_name(base.stem + a.infile_suffix + ".json")
    dst = base.with_name(base.stem + "_v5.json")
    doc = json.loads(src.read_text(encoding="utf-8"))
    built = (mimic_build if a.site == "mimic" else mcmed_build)(stay_of, cases_t0, a.horizon_hours)
    n_hist = n_items = n_ds = 0
    for c in doc["cases"]:
        hist, items = built.get(c["case_id"], ({}, []))
        if hist:
            c["t0_state"]["prior_history"] = hist; n_hist += 1
            n_ds += "last_discharge_summary" in hist
        c["observed_workup"] = [w for w in c["observed_workup"] if not w["item"].startswith("lab_")] + items
        n_items += sum(1 for w in items if w["result_status"] == "resulted")
    doc["meta"]["full_record"] = {"input": src.name, "horizon_hours": a.horizon_hours, "cases_with_prior_history": n_hist,
                                  "cases_with_previous_discharge_summary": n_ds, "additional_resulted_labs": n_items,
                                  "gating": "prior_history dated before arrival only; index-visit discharge summary and diagnosis codes excluded; "
                                            "labs gated by result_available_time like all other workup items"}
    dst.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {dst.name}: prior_history for {n_hist} cases ({n_ds} with a previous discharge summary); "
          f"{n_items} additional resulted lab items within {a.horizon_hours} h")


if __name__ == "__main__":
    main()
