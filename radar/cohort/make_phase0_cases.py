#!/usr/bin/env python3
"""Phase 0 test set: 8 real MIMIC-IV-ED encounters, two per stratum, as T0 snapshots.

T0 = triage complete.  Only what a clinician could see at that moment goes into `input`.
Labs, treatments and outcomes stay in `truth`, which is never shown to the model.

Reproducibility: once a case is chosen its stay_id is pinned in
phase0_case_ids.json.  Re-running never reshuffles an already-pinned case,
so changing one case's criteria cannot cascade into the others.
"""
from radar import paths
from radar.common.seeding import stable_seed
import pandas as pd, numpy as np, json, pathlib

ROOT = paths.RAW_ROOT / "master/mimic"
OUT  = paths.DATA
PINS = OUT / "phase0_case_ids.json"
SEED = 20260807

MIMICS = ['dx_hf_acute','dx_pe','dx_aki','dx_ards','dx_copd_exac','dx_asthma_exac',
          'dx_ams','dx_delirium','dx_ade','dx_stroke','dx_acs_mi','dx_resp_failure','dx_tbi']
TRIAGE = ['triage_temperature','triage_heartrate','triage_resprate','triage_o2sat',
          'triage_sbp','triage_dbp','triage_pain','triage_acuity']
DSP    = ['dsp_hospitalization','dsp_icu_transfer_12h','dsp_critical',
          'dsp_ed_revisit_3d','dsp_mortality_inhospital','dsp_mortality_30d']

use = (['stay_id','age','gender','arrival_transport','chiefcomplaint','sep3',
        'sep3_suspected_infection','sep3_shock','score_NEWS','score_qSOFA']
       + TRIAGE + MIMICS + DSP)
d = pd.read_csv(ROOT / "master_data.csv.gz",
                usecols=lambda c: c in use or c.startswith(('cci_','eci_')), low_memory=False)

inf = d.sep3_suspected_infection.fillna(0).astype(int) == 1
sep = d.sep3.fillna(0).astype(int) == 1
mim = d[MIMICS].fillna(0).astype(float).sum(axis=1) > 0
S = pd.Series('S4', index=d.index)
S[~inf & mim] = 'S3'; S[inf & ~sep] = 'S2'; S[sep] = 'S1'

complete = d[TRIAGE].notna().all(axis=1) & d.chiefcomplaint.notna()
news  = d.score_NEWS.fillna(0)
fever = d.chiefcomplaint.str.contains('fever', case=False, na=False)
sob   = d.chiefcomplaint.str.contains(r'dyspnea|short.{0,3}of breath|\bsob\b|edema|swelling',
                                      case=False, na=False, regex=True)
chf_hx = (d.get('cci_CHF', 0).fillna(0).astype(float) +
          d.get('eci_CHF', 0).fillna(0).astype(float)) > 0

SPEC = [
 ("S1a", "典型 sepsis：分诊生命体征已恶化",       (S=='S1') & (news>=5)),
 ("S1b", "非典型 sepsis：分诊生命体征基本正常",   (S=='S1') & (news<=1) & (d.triage_acuity>=3)),
 ("S2a", "有感染、未达 sepsis：发热主诉",         (S=='S2') & fever),
 ("S2b", "有感染、未达 sepsis：非发热主诉",       (S=='S2') & ~fever & (news>=3)),
 # S3a 重抽：老年 + 呼吸困难/水肿主诉 + 有心衰病史，才是典型的心衰拟态呈现
 ("S3a", "拟态：急性心衰（老年·呼吸困难·有心衰史）",
                                                  (S=='S3') & (d.dx_hf_acute.fillna(0)==1)
                                                  & (d.age>=60) & sob & chf_hx),
 ("S3b", "拟态：肺栓塞",                          (S=='S3') & (d.dx_pe.fillna(0)==1)),
 ("S4a", "普通就诊者：分诊指标正常",              (S=='S4') & (news<=1)),
 ("S4b", "普通就诊者：指标略异常但无感染无拟态",  (S=='S4') & (news>=3)),
]

pins = json.load(open(PINS)) if PINS.exists() else {}
by_stay = {int(v): k for k, v in pins.items()}
cases, used = [], set()

for cid, label, mask in SPEC:
    if cid in pins:                                   # 已锁定 -> 按 stay_id 取回，不重抽
        idx = d.index[d.stay_id == pins[cid]]
        if len(idx) == 0:
            print(f"  !! {cid} 锁定的 stay_id 不存在"); continue
        i = int(idx[0])
    else:
        pool = [j for j in d.index[mask & complete] if j not in used]
        if not pool:
            print(f"  !! {cid} 无匹配病例"); continue
        i = int(np.random.default_rng([SEED, stable_seed(cid)]).choice(pool))
        pins[cid] = int(d.loc[i, 'stay_id'])
    used.add(i)
    r = d.loc[i]
    comorb = sorted(c.split('_', 1)[1] for c in d.columns
                    if c.startswith(('cci_', 'eci_')) and str(r.get(c)) in ('1', '1.0'))
    cases.append({
      "case_id": cid, "label": label, "stratum": S[i], "stay_id": int(r.stay_id),
      "input": {
        "chief_complaint": str(r.chiefcomplaint),
        "age": None if pd.isna(r.age) else int(r.age),
        "sex": r.gender,
        "arrival_transport": r.arrival_transport,
        "triage_vitals": {
          "temperature_C": None if pd.isna(r.triage_temperature) else round(float(r.triage_temperature), 1),
          "heart_rate":    None if pd.isna(r.triage_heartrate) else int(r.triage_heartrate),
          "resp_rate":     None if pd.isna(r.triage_resprate) else int(r.triage_resprate),
          "spo2":          None if pd.isna(r.triage_o2sat) else int(r.triage_o2sat),
          "sbp":           None if pd.isna(r.triage_sbp) else int(r.triage_sbp),
          "dbp":           None if pd.isna(r.triage_dbp) else int(r.triage_dbp),
          "pain_0_10":     None if pd.isna(r.triage_pain) else int(r.triage_pain),
        },
        "nurse_esi": None if pd.isna(r.triage_acuity) else int(r.triage_acuity),
        "comorbidities": comorb,
        "labs": "no result available before this decision time",
      },
      "truth": {
        "sep3": int(r.sep3), "suspected_infection": int(r.sep3_suspected_infection),
        "septic_shock": int(r.sep3_shock),
        "mimic_dx": [c[3:] for c in MIMICS if str(r.get(c)) in ('1', '1.0')],
        "score_NEWS": None if pd.isna(r.score_NEWS) else float(r.score_NEWS),
        "score_qSOFA": None if pd.isna(r.score_qSOFA) else float(r.score_qSOFA),
        "outcomes": {k: (None if pd.isna(r.get(k)) else int(r.get(k))) for k in DSP if k in d.columns},
      }})

(OUT / "phase0_cases.json").parent.mkdir(parents=True, exist_ok=True)
json.dump(cases, open(OUT / "phase0_cases.json", "w"), ensure_ascii=False, indent=1)
json.dump(pins,  open(PINS, "w"), indent=1)
print(f"\n{len(cases)} 例 -> phase0_cases.json   stay_id 已锁定于 phase0_case_ids.json\n")
for c in cases:
    v, t = c['input']['triage_vitals'], c['truth']
    print(f"[{c['case_id']}] {c['label']}")
    print(f"    主诉: {c['input']['chief_complaint'][:58]} | {c['input']['age']}岁 {c['input']['sex']} | ESI {c['input']['nurse_esi']}")
    print(f"    T{v['temperature_C']} HR{v['heart_rate']} RR{v['resp_rate']} SpO2{v['spo2']} "
          f"BP{v['sbp']}/{v['dbp']} 疼痛{v['pain_0_10']}  合并症{len(c['input']['comorbidities'])}项")
    print(f"    [真值] sep3={t['sep3']} 感染={t['suspected_infection']} 拟态={t['mimic_dx'] or '—'} "
          f"NEWS={t['score_NEWS']:.0f} qSOFA={t['score_qSOFA']:.0f} 住院={t['outcomes'].get('dsp_hospitalization')}")
