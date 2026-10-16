#!/usr/bin/env python3
"""Score RADAR vNext cohort runs (result-triggered checkpoints C0_5/C1/C3/C6).

Reads one or more run files for a site plus the longitudinal case file (truth,
observed treatment, checkpoint inputs) and writes a JSON of every number the
revised paper/poster quotes, plus a short markdown summary.

What is scored
  Recognition   tier available at t in {0, 0.5, 1, 3, 6} h (latest processed
                checkpoint at or before t, else the T0 tier) against Sepsis-3,
                at probable-or-definite and possible-or-above, Wilson 95% CIs;
                comparators from identical inputs: NEWS>=5 and qSOFA>=2 at
                triage, a deterministic SOFA>=2 rule computed in code from the
                labs/MAP visible at the same cutoff, and SOFA>=2 AND an
                antimicrobial already given by the cutoff (Sepsis-3 operational
                proxy for suspected infection). Exact McNemar vs NEWS.
  Revision      T0 -> final tier transitions, up/down/unchanged.
  Act           time of first treatment recommendation, TX1 timing answer by
                truth stratum, agreement with observed first antimicrobial.
  Route         T0 and checkpoint Agent 4 outcomes, abstention reasons, final
                disposition against observed hospitalization / ICU within 12 h.
  Grounding     first-pass validation, retries, abstentions, uncited evidence.
  Reproduc.     across runs: exact tier agreement per decision point, node-level
                flip rates for R1/R2/R3a/R3b/R4, disposition agreement.

Usage
  python -m radar.evaluation.score_vnext --site mimic --runs cohort100_vnext_run_luna_r1.json [more runs]
  python -m radar.evaluation.score_vnext --site mcmed --runs stanford100_vnext_run_luna_r1.json
"""
from radar import paths
import argparse
import collections
import json
import math
import pathlib

from radar.cohort.make_t1_states import sofa_scores

HERE = pathlib.Path(__file__).resolve().parent
D = paths.DATA
TIMES = [0.0, 0.5, 1.0, 3.0, 6.0]
TIER_RANK = {"unlikely_sepsis": 0, "possible_sepsis": 1, "probable_sepsis": 2, "definite_sepsis": 3}
CASE_FILES = {"mimic": "cohort100_longitudinal_cases.json", "mcmed": "stanford100_longitudinal_cases.json"}


# ----------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(p, 3), round(centre - half, 3), round(centre + half, 3)


def rate(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "ci": [lo, hi]}


def binary_metrics(pred, truth):
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    tn = sum(1 for p, t in zip(pred, truth) if not p and not t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    out = {"n": len(pred), "positives": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "sensitivity": rate(tp, tp + fn), "specificity": rate(tn, tn + fp)}
    if tp + fp:
        out["ppv"] = rate(tp, tp + fp)
    return out


def mcnemar_exact(a, b, truth):
    """Paired comparison of two binary predictors on the positive class (sensitivity)."""
    n01 = sum(1 for x, y, t in zip(a, b, truth) if t and x and not y)
    n10 = sum(1 for x, y, t in zip(a, b, truth) if t and not x and y)
    n = n01 + n10
    if n == 0:
        return {"a_only": n01, "b_only": n10, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)
    return {"a_only": n01, "b_only": n10, "p": round(p, 4)}


# ----------------------------------------------------------------- extraction
def processed(r):
    return [c for c in r.get("checkpoints", []) if c.get("checkpoint_status", "").startswith("processed")]


def tier_at(r, t):
    """Latest recognition tier a clinician would hold at time t (carry-forward)."""
    tier = ((r.get("agent1_t0") or {}).get("result") or {}).get("tier")
    when, source = 0.0, "T0"
    for c in processed(r):
        a1 = ((c.get("agent1_reassessment") or {}).get("result") or {}).get("tier")
        if c["cutoff_time"] <= t + 1e-9 and a1 and a1 != "abstained":
            tier, when, source = a1, c["cutoff_time"], c["checkpoint_id"]
    return tier, when, source


def final_tier(r):
    return tier_at(r, 1e9)


def sofa_rule_at(r, t, case):
    """SOFA>=2 from the labs and MAP visible at the latest processed checkpoint <= t."""
    state = None
    for c in processed(r):
        if c["cutoff_time"] <= t + 1e-9:
            state = c.get("patient_state")
    if state is None:
        return None, None
    inv = state.get("investigation_results") or {}
    labs = {k: {"value": v.get("value")} for k, v in inv.items() if isinstance(v, dict)}
    vit = dict(state.get("vitals") or {})
    pressor = any(e.get("start_time", 1e9) <= t for e in (case["observed_treatment"].get("vasopressors") or []))
    scores, total, missing = sofa_scores(labs, vit, on_vasopressor=pressor)
    return total, missing


def abx_by(case, t):
    return any(e.get("administered") and e.get("administration_time", 1e9) <= t
               for e in case["observed_treatment"].get("antimicrobials") or [])


def first_abx_time(case):
    ts = [e["administration_time"] for e in case["observed_treatment"].get("antimicrobials") or [] if e.get("administered")]
    return min(ts) if ts else None


def node_answers(agent_run):
    return {step["node"]: step.get("answer") for step in (agent_run or {}).get("trace", []) if step.get("node")}


def grounding(agent_run):
    calls = first_pass = retries = 0
    evidence = uncited = 0
    for step in (agent_run or {}).get("trace", []):
        attempts = step.get("attempts") or []
        if not attempts:
            continue
        calls += len(attempts)
        first_pass += 1 if attempts[0].get("ok") else 0
        retries += len(attempts) - 1
        for item in ((step.get("reply") or {}).get("supporting_evidence") or []):
            evidence += 1
            uncited += 1 if not item.get("statement_id") else 0
    return calls, first_pass, retries, evidence, uncited


def median_iqr(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    def q(p):
        i = (len(xs) - 1) * p
        lo, hi = math.floor(i), math.ceil(i)
        return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 2)
    return {"n": len(xs), "median": q(0.5), "iqr": [q(0.25), q(0.75)]}


# ----------------------------------------------------------------- scoring one run
def score_run(results, cases):
    by_case = {c["case_id"]: c for c in cases}
    rs = [r for r in results if r["case_id"] in by_case]
    truth = {r["case_id"]: by_case[r["case_id"]]["evaluation_only"]["truth"] for r in rs}
    sep3 = [bool(truth[r["case_id"]]["sep3"]) for r in rs]
    out = {"n_cases": len(rs),
           "pipeline_status": dict(collections.Counter(r["status"] for r in rs)),
           "stopped_at": dict(collections.Counter(f"{r.get('stopped_at')}:{r.get('stop_reason')}" for r in rs if r["status"] != "complete")),
           "checkpoints_processed": dict(collections.Counter(c["checkpoint_id"] for r in rs for c in processed(r)))}

    # recognition over time
    rec = {}
    news = [(truth[r["case_id"]].get("score_NEWS") or 0) >= 5 for r in rs]
    qsofa = [(truth[r["case_id"]].get("score_qSOFA") or 0) >= 2 for r in rs]
    for t in TIMES:
        tiers = [tier_at(r, t) for r in rs]
        reassessed = [tr[2] != "T0" for tr in tiers]
        pod = [TIER_RANK.get(tr[0], -1) >= 2 for tr in tiers]
        poa = [TIER_RANK.get(tr[0], -1) >= 1 for tr in tiers]
        sofa = [sofa_rule_at(r, t, by_case[r["case_id"]]) for r in rs]
        sofa_pos = [s[0] is not None and s[0] >= 2 for s in sofa]
        sofa_abx = [sp and abx_by(by_case[r["case_id"]], t) for sp, r in zip(sofa_pos, rs)]
        block = {"reassessed_by_t": sum(reassessed),
                 "tier_distribution": dict(collections.Counter(tr[0] for tr in tiers)),
                 "radar_probable_or_definite": binary_metrics(pod, sep3),
                 "radar_possible_or_above": binary_metrics(poa, sep3),
                 "news_ge5": binary_metrics(news, sep3),
                 "qsofa_ge2": binary_metrics(qsofa, sep3),
                 "sofa_rule_ge2_same_inputs": binary_metrics(sofa_pos, sep3),
                 "sofa_rule_ge2_and_antimicrobial_given": binary_metrics(sofa_abx, sep3),
                 "sofa_rule_computable": sum(1 for s in sofa if s[0] is not None),
                 "mcnemar_radar_pod_vs_news": mcnemar_exact(pod, news, sep3),
                 "mcnemar_radar_pod_vs_sofa_rule": mcnemar_exact(pod, sofa_pos, sep3)}
        if t > 0:
            idx = [i for i, re_ in enumerate(reassessed) if re_]
            block["reassessed_only"] = {
                "n": len(idx),
                "radar_probable_or_definite": binary_metrics([pod[i] for i in idx], [sep3[i] for i in idx]),
                "news_ge5": binary_metrics([news[i] for i in idx], [sep3[i] for i in idx]),
                "sofa_rule_ge2_same_inputs": binary_metrics([sofa_pos[i] for i in idx], [sep3[i] for i in idx])}
        rec[f"t{t:g}h"] = block
    out["recognition"] = rec

    # revision
    trans = collections.Counter()
    moves = []
    for r in rs:
        t0 = tier_at(r, 0)[0]; fin = final_tier(r)[0]
        if fin is None or t0 is None or final_tier(r)[2] == "T0":
            trans["not_reassessed"] += 1; continue
        d = TIER_RANK.get(fin, 0) - TIER_RANK.get(t0, 0)
        trans["up" if d > 0 else "down" if d < 0 else "unchanged"] += 1
        moves.append({"case_id": r["case_id"], "from": t0, "to": fin, "sep3": truth[r["case_id"]]["sep3"],
                      "stratum": by_case[r["case_id"]].get("stratum")})
    out["revision"] = {"t0_to_final": dict(trans),
                       "up_by_truth": dict(collections.Counter(m["sep3"] for m in moves if TIER_RANK[m["to"]] > TIER_RANK[m["from"]])),
                       "down_by_truth": dict(collections.Counter(m["sep3"] for m in moves if TIER_RANK[m["to"]] < TIER_RANK[m["from"]])),
                       "moves": moves}

    # act
    first_act, tx1_by_truth, tx1_vs_obs, map_without_pressor = [], collections.defaultdict(collections.Counter), collections.Counter(), 0
    tx1_within1_sep = collections.Counter()
    for r in rs:
        case = by_case[r["case_id"]]; tr = truth[r["case_id"]]
        cs = [c for c in processed(r) if (c.get("agent3") or {}).get("status") == "complete"]
        if not cs:
            continue
        c0 = cs[0]
        first_act.append(c0["cutoff_time"])
        plan = c0["agent3"]["result"]["treatment_plan"]
        timing = plan.get("antimicrobial_timing", {}).get("timing")
        key = "shock" if tr.get("septic_shock") else "sepsis" if tr.get("sep3") else "infection_no_sepsis" if tr.get("suspected_infection") else "no_infection"
        tx1_by_truth[key][timing] += 1
        if tr.get("sep3"):
            tx1_within1_sep[timing] += 1
        obs = first_abx_time(case)
        if timing in ("within_1h", "within_3h"):
            limit = 1 if timing == "within_1h" else 3
            tx1_vs_obs["observed_within_window" if obs is not None and obs <= c0["cutoff_time"] + limit else
                       "observed_later" if obs is not None else "no_antimicrobial_recorded_6h"] += 1
        for c in cs:
            p = c["agent3"]["result"]["treatment_plan"]
            if p.get("map_target", {}).get("map_target_mmHg") and not p.get("vasopressors", {}).get("start_vasopressor") \
                    and not any(e.get("start_time", 1e9) <= c["cutoff_time"] for e in case["observed_treatment"].get("vasopressors") or []):
                map_without_pressor += 1
    out["act"] = {"first_recommendation_time_h": median_iqr(first_act),
                  "first_recommendation_by": {f"<= {t:g} h": sum(1 for x in first_act if x <= t) for t in (0.5, 1, 3, 6)},
                  "tx1_timing_by_truth": {k: dict(v) for k, v in tx1_by_truth.items()},
                  "tx1_vs_observed_antimicrobial": dict(tx1_vs_obs),
                  "observed_first_antimicrobial_h": median_iqr([first_abx_time(by_case[r["case_id"]]) for r in rs]),
                  "map_target_without_vasopressor_checkpoints": map_without_pressor}

    # route
    t0_status = collections.Counter(((r.get("agent4_t0") or {}).get("result") or {}).get("status") for r in rs)
    ck_status, ck_reason = collections.Counter(), collections.Counter()
    final_vs_outcome = collections.Counter()
    for r in rs:
        tr = truth[r["case_id"]]; oc = tr.get("outcomes") or {}
        for c in processed(r):
            res = (c.get("agent4") or {}).get("result") or {}
            ck_status[res.get("status") or "not_run"] += 1
            if res.get("status") == "abstain":
                ck_reason[f"{res.get('abstain_reason')}:{','.join(res.get('missing_information') or [])[:60]}"] += 1
        disp = r.get("final_simulated_disposition")
        if disp:
            hosp = oc.get("dsp_hospitalization"); icu = oc.get("dsp_icu_transfer_12h", oc.get("dsp_icu"))
            if disp == "discharge" and hosp == 1:
                final_vs_outcome["discharge_but_admitted"] += 1
            elif disp in ("ward", "observation") and icu == 1:
                final_vs_outcome["ward_or_obs_but_icu_12h"] += 1
            elif disp == "ICU" and hosp == 0:
                final_vs_outcome["icu_but_not_admitted"] += 1
            else:
                final_vs_outcome["not_contradicted"] += 1
    out["route"] = {"t0_status": dict(t0_status), "checkpoint_status": dict(ck_status),
                    "checkpoint_abstain_reasons": dict(ck_reason),
                    "final_disposition": dict(collections.Counter(r.get("final_simulated_disposition") for r in rs)),
                    "final_vs_observed_course": dict(final_vs_outcome)}

    # grounding
    g = {}
    for label, getter in (("A1", lambda r: [r.get("agent1_t0")] + [c.get("agent1_reassessment") for c in processed(r)]),
                          ("A2", lambda r: [r.get("agent2")]),
                          ("A3", lambda r: [c.get("agent3") for c in processed(r)]),
                          ("A4", lambda r: [r.get("agent4_t0")] + [c.get("agent4") for c in processed(r)])):
        calls = fp = rt = ev = un = abst = 0
        for r in rs:
            for run in getter(r):
                if not run:
                    continue
                c, f, t, e, u = grounding(run)
                calls += c; fp += f; rt += t; ev += e; un += u
                res = run.get("result") or {}
                abst += 1 if (res.get("tier") == "abstained" or res.get("status") == "abstain" or run.get("status") == "abstained") else 0
        g[label] = {"node_calls": calls, "first_pass_valid": rate(fp, max(1, calls - rt)), "re_asks": rt,
                    "abstentions": abst, "uncited_evidence": rate(un, ev) if ev else None}
    out["grounding"] = g
    return out


# ----------------------------------------------------------------- reproducibility
def reproducibility(runs, case_ids=None):
    """runs: list of results lists (same cohort). Agreement across all runs."""
    if len(runs) < 2:
        return None
    by = [{r["case_id"]: r for r in run if case_ids is None or r["case_id"] in case_ids} for run in runs]
    ids = set.intersection(*(set(b) for b in by))
    points, agree = collections.Counter(), collections.Counter()
    node_total, node_flip = collections.Counter(), collections.Counter()
    disp_total = disp_agree = 0
    final_total = final_agree = 0
    for cid in ids:
        rs = [b[cid] for b in by]
        # T0
        t0 = [tier_at(r, 0)[0] for r in rs]
        points["T0"] += 1; agree["T0"] += len(set(t0)) == 1
        ans = [node_answers(r.get("agent1_t0")) for r in rs]
        for node in ("R1", "R2", "R3a", "R3b", "R4"):
            vals = [a.get(node) for a in ans if node in a]
            if len(vals) == len(rs):
                node_total[f"T0:{node}"] += 1; node_flip[f"T0:{node}"] += len(set(vals)) > 1
        # checkpoints by id
        cps = [{c["checkpoint_id"]: c for c in processed(r)} for r in rs]
        for ck in set.intersection(*(set(c) for c in cps)):
            tiers = [((c[ck].get("agent1_reassessment") or {}).get("result") or {}).get("tier") for c in cps]
            points[ck] += 1; agree[ck] += len(set(tiers)) == 1
            ans = [node_answers(c[ck].get("agent1_reassessment")) for c in cps]
            for node in ("R1", "R2", "R3a", "R3b", "R4"):
                vals = [a.get(node) for a in ans if node in a]
                if len(vals) == len(rs):
                    node_total[f"{ck}:{node}"] += 1; node_flip[f"{ck}:{node}"] += len(set(vals)) > 1
            d = [((c[ck].get("agent4") or {}).get("result") or {}).get("disposition") for c in cps]
            disp_total += 1; disp_agree += len(set(d)) == 1
        f = [final_tier(r)[0] for r in rs]
        final_total += 1; final_agree += len(set(f)) == 1
    return {"runs": len(runs), "cases": len(ids),
            "exact_tier_agreement": {k: rate(agree[k], points[k]) for k in points},
            "final_tier_agreement": rate(final_agree, final_total),
            "node_flip_rate": {k: rate(node_flip[k], node_total[k]) for k in sorted(node_total)},
            "checkpoint_disposition_agreement": rate(disp_agree, disp_total)}


# ----------------------------------------------------------------- main
def load_run(path):
    doc = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return doc.get("meta", {}), doc.get("results") if isinstance(doc, dict) else doc


def summary_md(site, scored, repro):
    s = scored
    lines = [f"# RADAR vNext scores — {site}", ""]
    lines.append(f"Cases {s['n_cases']}; pipeline status {s['pipeline_status']}; checkpoints processed {s['checkpoints_processed']}")
    lines.append("")
    lines.append("| time | reassessed | RADAR prob/def sens | spec | NEWS>=5 sens | spec | SOFA>=2 rule sens | spec | RADAR poss+ sens | spec |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, b in s["recognition"].items():
        f = lambda m: f"{m['sensitivity']['p']} ({m['sensitivity']['ci'][0]}-{m['sensitivity']['ci'][1]})"
        g = lambda m: f"{m['specificity']['p']}"
        lines.append(f"| {k} | {b['reassessed_by_t']} | {f(b['radar_probable_or_definite'])} | {g(b['radar_probable_or_definite'])} | "
                     f"{f(b['news_ge5'])} | {g(b['news_ge5'])} | {f(b['sofa_rule_ge2_same_inputs'])} | {g(b['sofa_rule_ge2_same_inputs'])} | "
                     f"{f(b['radar_possible_or_above'])} | {g(b['radar_possible_or_above'])} |")
    lines.append("")
    lines.append(f"Revision T0->final: {s['revision']['t0_to_final']} (up by sep3 {s['revision']['up_by_truth']}, down by sep3 {s['revision']['down_by_truth']})")
    a = s["act"]
    lines.append(f"Act: first recommendation {a['first_recommendation_time_h']}, by {a['first_recommendation_by']}; TX1 by truth {a['tx1_timing_by_truth']}; vs observed antimicrobial {a['tx1_vs_observed_antimicrobial']}; observed first antimicrobial {a['observed_first_antimicrobial_h']}; MAP target without vasopressor {a['map_target_without_vasopressor_checkpoints']}")
    r = s["route"]
    lines.append(f"Route: T0 {r['t0_status']}; checkpoints {r['checkpoint_status']}; final {r['final_disposition']}; vs course {r['final_vs_observed_course']}")
    lines.append(f"Route abstain reasons: {r['checkpoint_abstain_reasons']}")
    lines.append(f"Grounding: {json.dumps(s['grounding'])}")
    if repro:
        lines.append(f"Reproducibility ({repro['runs']} runs, {repro['cases']} cases): exact tier agreement {repro['exact_tier_agreement']}; final {repro['final_tier_agreement']}; disposition {repro['checkpoint_disposition_agreement']}")
        lines.append(f"Node flips: {repro['node_flip_rate']}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", choices=CASE_FILES, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    cases = json.loads((D / CASE_FILES[a.site]).read_text(encoding="utf-8"))["cases"]
    metas, runs = zip(*(load_run(r) for r in a.runs))
    per_run = {pathlib.Path(r).name: score_run(res, cases) for r, res in zip(a.runs, runs)}
    repro = reproducibility(list(runs), {c["case_id"] for c in cases})
    primary = per_run[pathlib.Path(a.runs[0]).name]
    doc = {"site": a.site, "runs": [{"file": pathlib.Path(r).name, "model": m.get("model"), "run_id": m.get("run_id"),
                                     "usage": {k: v.get("usage") for k, v in (m.get("model_runtime") or {}).items()}}
                                    for r, m in zip(a.runs, metas)],
           "primary": primary, "per_run": per_run, "reproducibility": repro}
    out = pathlib.Path(a.out) if a.out else D / f"scores_vnext_{a.site}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(summary_md(a.site, primary, repro), encoding="utf-8")
    print(md.read_text())
    print(f"\n-> {out.name}, {md.name}")


if __name__ == "__main__":
    main()
