#!/usr/bin/env python3
"""Stratified 100-case cohort from MIMIC-IV-ED, as T0 snapshots.

Same design as make_phase0_cases.py, scaled from 8 to 100 and sampled by rule
instead of by hand-written case specs.

Design
------
Four strata over the full 425,011 ED visits:

    S1  sepsis (Sepsis-3)                24,744
    S2  infection, does not meet sepsis  51,170
    S3  mimic, no suspected infection    55,851
    S4  routine                         293,246

25 cases per stratum. Within each stratum the sample is spread over clinically
distinct sub-strata so the cohort covers the presentations the agent has to
separate, not just the common ones. Sampling weights are N_stratum / n_stratum,
so weighted estimates still describe the source population.

The eight Phase-0 cases are carried in unchanged, so anything measured on them
before stays comparable.

T0 = triage complete. Only what a clinician could see at that moment goes into
`input`. Labs, treatments and outcomes stay in `truth`, never shown to a model.

Reproducibility: every chosen stay_id is pinned to cohort100_case_ids.json.
Re-running never reshuffles a pinned case.

Usage
-----
    python -m radar.cohort.make_cohort100
    python -m radar.cohort.make_cohort100 --n-per-stratum 25 --data-root /path/to/mimic
"""
from __future__ import annotations
from radar import paths
from radar.common.seeding import stable_seed

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
OUT = paths.DATA
SEED = 20260908

# The master table lives outside the repo. These are tried in order.
# Both source databases carry the same column schema, so one sampler serves
# development and external validation. Only the file, the disposition fields and
# two optional triage columns differ.
SOURCES = {
    "mimic": {
        "subdir": "master/mimic",
        "file": "master_data.csv.gz",
        "dsp": ['dsp_hospitalization', 'dsp_icu_transfer_12h', 'dsp_critical',
                'dsp_ed_revisit_3d', 'dsp_mortality_inhospital', 'dsp_mortality_30d'],
        "icu_field": "dsp_icu_transfer_12h",
    },
    "stanford": {
        "subdir": "master/stanford",
        "file": "stanford_data.csv.gz",
        "dsp": ['dsp_hospitalization', 'dsp_icu', 'dsp_critical',
                'dsp_ed_revisit_3d', 'dsp_mortality_inhospital', 'dsp_observation'],
        "icu_field": "dsp_icu",
    },
}
DATA_ROOTS = [paths.RAW_ROOT]

MIMICS = ['dx_hf_acute', 'dx_pe', 'dx_aki', 'dx_ards', 'dx_copd_exac', 'dx_asthma_exac',
          'dx_ams', 'dx_delirium', 'dx_ade', 'dx_stroke', 'dx_acs_mi', 'dx_resp_failure', 'dx_tbi']
TRIAGE = ['triage_temperature', 'triage_heartrate', 'triage_resprate', 'triage_o2sat',
          'triage_sbp', 'triage_dbp', 'triage_pain', 'triage_acuity']
DSP = ['dsp_hospitalization', 'dsp_icu_transfer_12h', 'dsp_critical',
       'dsp_ed_revisit_3d', 'dsp_mortality_inhospital', 'dsp_mortality_30d']


def find_data(source, explicit=None):
    cfg = SOURCES[source]
    roots = [pathlib.Path(explicit)] if explicit else DATA_ROOTS
    for root in roots:
        for candidate in (root / cfg["subdir"], root):
            if (candidate / cfg["file"]).exists():
                return candidate
    raise SystemExit(
        f"{cfg['file']} not found. Looked under:\n  "
        + "\n  ".join(str(r) for r in roots)
        + "\nPass --data-root to point at the 00_Data directory.")


def load(data_root, source):
    cfg = SOURCES[source]
    use = (['stay_id', 'age', 'gender', 'arrival_transport', 'chiefcomplaint', 'sep3',
            'sep3_suspected_infection', 'sep3_shock', 'score_NEWS', 'score_qSOFA', 'score_ESRP']
           + TRIAGE + MIMICS + cfg["dsp"])
    d = pd.read_csv(data_root / cfg["file"],
                    usecols=lambda c: c in use or c.startswith(('cci_', 'eci_')),
                    low_memory=False)
    # Columns absent from one source are created empty rather than special-cased
    # downstream, so a missing pain score reads as missing and not as zero.
    for col in TRIAGE + MIMICS + ['arrival_transport']:
        if col not in d.columns:
            d[col] = None
    return d


def derive(d):
    """Stratum label plus the T0-observable flags the sub-strata are built from."""
    inf = d.sep3_suspected_infection.fillna(0).astype(int) == 1
    sep = d.sep3.fillna(0).astype(int) == 1
    mim = d[MIMICS].fillna(0).astype(float).sum(axis=1) > 0

    S = pd.Series('S4', index=d.index)
    S[~inf & mim] = 'S3'
    S[inf & ~sep] = 'S2'
    S[sep] = 'S1'

    news = d.score_NEWS.fillna(0)
    # Bedside organ-dysfunction proxy: exactly the R2 T0 criterion, minus mental
    # status, which the triage record does not carry. This is what decides
    # whether probable/definite are reachable for a case at all.
    bedside_od = ((d.triage_resprate.fillna(0) >= 22)
                  | (d.triage_sbp.fillna(999) <= 100)
                  | (d.triage_o2sat.fillna(100) < 92))
    shock = d.sep3_shock.fillna(0).astype(int) == 1
    fever = d.chiefcomplaint.str.contains('fever|febrile|chills', case=False, na=False, regex=True)
    # Require only the vitals this source actually records. Stanford has no
    # pain score, and demanding one would reject all 118,385 of its visits.
    recorded = [c for c in TRIAGE if d[c].notna().any()]
    complete = d[recorded].notna().all(axis=1) & d.chiefcomplaint.notna()
    return S, news, bedside_od, shock, fever, complete


def sub_strata(d, S, news, bedside_od, shock, fever):
    """(stratum, key, description, mask, share of that stratum's quota)."""
    dx = lambda c: d[c].fillna(0) == 1
    return [
        # S1 sepsis — deliberately over-sample the presentations that make the
        # recognition problem hard, and the ones that can reach probable at all.
        ('S1', 'shock',      'septic shock at presentation',              shock, 0.20),
        ('S1', 'od',         'bedside organ dysfunction, no shock',       bedside_od & ~shock, 0.32),
        ('S1', 'illlooking', 'NEWS >= 5 without bedside organ dysfunction', (news >= 5) & ~bedside_od & ~shock, 0.16),
        ('S1', 'atypical',   'near-normal triage vitals',                 (news <= 1) & ~bedside_od & ~shock, 0.32),

        # S2 infection that does not meet sepsis
        ('S2', 'fever',      'febrile complaint',                          fever, 0.52),
        ('S2', 'nonfever',   'non-febrile complaint',                      ~fever, 0.48),

        # S3 mimics — one bucket per disease so the mimic knowledge-base gap is
        # measurable per disease rather than in aggregate.
        ('S3', 'hf',         'acute heart failure',                        dx('dx_hf_acute'), 0.20),
        ('S3', 'pe',         'pulmonary embolism',                         dx('dx_pe'), 0.20),
        ('S3', 'acs',        'acute coronary syndrome',                    dx('dx_acs_mi'), 0.16),
        ('S3', 'aki',        'acute kidney injury',                        dx('dx_aki'), 0.16),
        ('S3', 'copd',       'COPD exacerbation',                          dx('dx_copd_exac'), 0.16),
        ('S3', 'ards',       'ARDS or respiratory failure',                dx('dx_ards') | dx('dx_resp_failure'), 0.12),

        # S4 routine
        ('S4', 'normal',     'normal triage indices',                      news <= 1, 0.52),
        ('S4', 'abnormal',   'mildly abnormal, no infection, no mimic',    news >= 3, 0.48),
    ]



def case_record(d, cfg, i, cid, stratum, key, desc, bedside_od_i):
    """One cohort record from master-table row i. Shared by the sampler and the
    replacement step, so a replacement case is built exactly like the original."""
    r = d.loc[i]
    comorb = sorted(c.split('_', 1)[1] for c in d.columns
                    if c.startswith(('cci_', 'eci_')) and str(r.get(c)) in ('1', '1.0'))
    return {
        "case_id": cid,
        "label": desc,
        "stratum": stratum,
        "sub_stratum": key,
        "stay_id": int(r.stay_id),
        "input": {
            "chief_complaint": str(r.chiefcomplaint),
            "age": None if pd.isna(r.age) else int(r.age),
            "sex": r.gender,
            "arrival_transport": r.arrival_transport,
            "triage_vitals": {
                "temperature_C": None if pd.isna(r.triage_temperature) else round(float(r.triage_temperature), 1),
                "heart_rate": None if pd.isna(r.triage_heartrate) else int(r.triage_heartrate),
                "resp_rate": None if pd.isna(r.triage_resprate) else int(r.triage_resprate),
                "spo2": None if pd.isna(r.triage_o2sat) else int(r.triage_o2sat),
                "sbp": None if pd.isna(r.triage_sbp) else int(r.triage_sbp),
                "dbp": None if pd.isna(r.triage_dbp) else int(r.triage_dbp),
                "pain_0_10": None if pd.isna(r.triage_pain) else int(r.triage_pain),
            },
            "nurse_esi": None if pd.isna(r.triage_acuity) else int(r.triage_acuity),
            "comorbidities": comorb,
            "labs": "no result available before this decision time",
        },
        "truth": {
            "sep3": int(r.sep3),
            "suspected_infection": int(r.sep3_suspected_infection),
            "septic_shock": int(r.sep3_shock),
            "mimic_dx": [c[3:] for c in MIMICS if str(r.get(c)) in ('1', '1.0')],
            "score_NEWS": None if pd.isna(r.score_NEWS) else float(r.score_NEWS),
            "score_qSOFA": None if pd.isna(r.score_qSOFA) else float(r.score_qSOFA),
            "score_ESRP": None if pd.isna(r.get('score_ESRP')) else float(r.get('score_ESRP')),
            "bedside_organ_dysfunction_at_triage": bool(bedside_od_i),
            "outcomes": {k: (None if pd.isna(r.get(k)) else int(r.get(k)))
                         for k in cfg["dsp"] if k in d.columns},
        },
    }


def quota(share, n):
    return max(1, int(round(share * n)))


def build(args):
    cfg = SOURCES[args.source]
    data_root = find_data(args.source, args.data_root)
    print(f"source {args.source}: {data_root / cfg['file']}")
    d = load(data_root, args.source)
    S, news, bedside_od, shock, fever, complete = derive(d)
    print(f"  {len(d):,} visits   " + "  ".join(f"{s} {int((S==s).sum()):,}" for s in "S1 S2 S3 S4".split()))

    pins_path = OUT / args.pins
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
    # A pinned cohort is rebuilt exactly as pinned: no new draws. Sampling only happens
    # on a first build (no pins file). Replacements are made by replace_no_result_cases.
    pinned_cohort = bool(pins)

    # Carry the original eight in, so earlier measurements stay comparable.
    seeded = OUT / "phase0_case_ids.json"
    if args.source == "mimic" and seeded.exists() and not pins:
        pins = {f"P0_{k}": v for k, v in json.loads(seeded.read_text()).items()}

    taken = set(pins.values())
    by_index = {int(v): k for k, v in pins.items()}
    chosen = []

    for stratum, key, desc, mask, share in sub_strata(d, S, news, bedside_od, shock, fever):
        want = quota(share, args.n_per_stratum)
        prefix = f"{stratum}_{key}"
        already = [cid for cid in pins if cid.startswith(prefix + "_")]
        pool = d.index[(S == stratum) & mask & complete & ~d.stay_id.isin(taken)]
        rng = np.random.default_rng([SEED, stable_seed(prefix)])
        need = 0 if pinned_cohort else want - len(already)
        picks = list(rng.choice(pool, size=min(need, len(pool)), replace=False)) if need > 0 and len(pool) else []
        if need > len(pool):
            print(f"  !! {prefix}: wanted {want}, only {len(pool) + len(already)} available")
        for cid in already:
            idx = d.index[d.stay_id == pins[cid]]
            if len(idx):
                chosen.append((cid, stratum, key, desc, int(idx[0])))
        for j, i in enumerate(picks, start=len(already) + 1):
            cid = f"{prefix}_{j:02d}"
            pins[cid] = int(d.loc[i, 'stay_id'])
            taken.add(pins[cid])
            chosen.append((cid, stratum, key, desc, int(i)))

    # The eight carried-over cases keep their own ids and are placed by stratum.
    for cid, stay in list(pins.items()):
        if cid.startswith("P0_") and not any(c[0] == cid for c in chosen):
            idx = d.index[d.stay_id == stay]
            if len(idx):
                i = int(idx[0])
                chosen.append((cid, S[i], "phase0", "carried over from the 8-case pilot", i))

    # A carried-over case occupies a slot rather than adding one, so trim each
    # stratum back to its quota. Trim from the largest sub-stratum first, and
    # never drop a carried-over case.
    trimmed = []
    for stratum in ("S1", "S2", "S3", "S4"):
        rows = [c for c in chosen if c[1] == stratum]
        keep = [c for c in rows if c[0].startswith("P0_")]
        rest = [c for c in rows if not c[0].startswith("P0_")]
        while len(keep) + len(rest) > args.n_per_stratum and rest:
            sizes = {}
            for c in rest:
                sizes.setdefault(c[2], []).append(c)
            biggest = max(sizes.values(), key=len)
            rest.remove(biggest[-1])
        trimmed.extend(keep + rest)
    dropped = {c[0] for c in chosen} - {c[0] for c in trimmed}
    for cid in dropped:
        pins.pop(cid, None)
    chosen = trimmed

    cases = [case_record(d, cfg, i, cid, stratum, key, desc, bedside_od[i])
             for cid, stratum, key, desc, i in chosen]

    # Interleave the strata round-robin. Batched runs take the cohort in file
    # order, so a stratum-ordered file makes the first batch entirely sepsis and
    # tells you nothing about specificity. This keeps any prefix balanced, and is
    # deterministic — a rotation, not a shuffle.
    buckets = {s: [c for c in cases if c["stratum"] == s] for s in ("S1", "S2", "S3", "S4")}
    interleaved = []
    while any(buckets.values()):
        for s in ("S1", "S2", "S3", "S4"):
            if buckets[s]:
                interleaved.append(buckets[s].pop(0))
    cases = interleaved

    counts = pd.Series([c["stratum"] for c in cases]).value_counts().to_dict()
    N = {s: int((S == s).sum()) for s in ("S1", "S2", "S3", "S4")}
    weights = {s: round(N[s] / counts[s], 1) for s in counts}
    w = np.array([weights[c["stratum"]] for c in cases])
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())

    meta = {
        "design": "stratified sample of MIMIC-IV-ED at T0 (triage complete)",
        "source_visits": int(len(d)),
        "stratum_sizes": N,
        "sampled": counts,
        "sampling_weights": weights,
        "effective_sample_size": round(n_eff, 1),
        "source": args.source,
        "seed": SEED,
        "carried_over_from_phase0": sorted(c["case_id"] for c in cases if c["case_id"].startswith("P0_")),
    }

    (OUT / args.out).write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / args.out.replace(".json", "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    pins_path.write_text(json.dumps(pins, indent=1), encoding="utf-8")

    print(f"\n{len(cases)} cases")
    for s in ("S1", "S2", "S3", "S4"):
        subs = [c for c in cases if c["stratum"] == s]
        pos = sum(c["truth"]["bedside_organ_dysfunction_at_triage"] for c in subs)
        print(f"  {s}  n={len(subs):>3}  weight {weights.get(s)}  bedside organ dysfunction {pos}")
    print(f"  effective sample size {n_eff:.1f}")
    print(f"\n-> {paths.DATA / args.out}")
    print(f"-> {paths.DATA / args.out.replace('.json', '_meta.json')}")
    print(f"-> {paths.DATA / args.pins}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=tuple(SOURCES), default="mimic",
                   help="mimic for development, stanford for external validation")
    p.add_argument("--n-per-stratum", type=int, default=25)
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None,
                   help="default: cohort100_cases.json for mimic, "
                        "stanford100_cases.json for stanford")
    p.add_argument("--pins", default=None)
    args = p.parse_args()
    stem = "cohort100" if args.source == "mimic" else "stanford100"
    args.out = args.out or f"{stem}_cases.json"
    args.pins = args.pins or f"{stem}_case_ids.json"
    build(args)


if __name__ == "__main__":
    main()
