#!/usr/bin/env python3
"""Compute Hema's evaluation metrics E1-E8 from whatever run files exist.

Every metric prints either a number or the exact command that would produce it.
No model is called here.

    python -m radar.evaluation.score_hema_metrics
    python -m radar.evaluation.score_hema_metrics --out hema_metrics.json
"""
from __future__ import annotations
from radar import paths

import argparse
import collections
import json
import pathlib

from radar.evaluation.evaluate_agent4 import calculate_metrics
from radar.evaluation.evaluation_export import tier_transition

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
D = paths.DATA

POS_TIERS = {"probable_sepsis", "definite_sepsis"}
ALL_POS = POS_TIERS | {"possible_sepsis"}


def load(name):
    p = D / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(n, d):
    return None if not d else round(100.0 * n / d, 1)


def fmt(v, suffix="%"):
    return "n/a" if v is None else f"{v}{suffix}"


def rates(tp, fn, tn, fp):
    se = tp / (tp + fn) if tp + fn else None
    sp = tn / (tn + fp) if tn + fp else None
    ppv = tp / (tp + fp) if tp + fp else None
    npv = tn / (tn + fn) if tn + fn else None
    lrp = se / (1 - sp) if (se is not None and sp is not None and sp < 1) else None
    lrn = (1 - se) / sp if (se is not None and sp) else None
    return dict(TP=tp, FN=fn, TN=tn, FP=fp, sensitivity=se, specificity=sp,
                ppv=ppv, npv=npv, lr_pos=lrp, lr_neg=lrn)


def show(d):
    g = lambda k: ("n/a" if d[k] is None else f"{d[k]:.2f}")
    return (f"TP {d['TP']}  FN {d['FN']}  TN {d['TN']}  FP {d['FP']}   "
            f"Se {g('sensitivity')}  Sp {g('specificity')}  "
            f"PPV {g('ppv')}  NPV {g('npv')}  LR+ {g('lr_pos')}  LR- {g('lr_neg')}")


# ───────────────────────────── E1 ─────────────────────────────
def e1(res):
    cases = load("phase0_cases.json")
    a1 = load("phase0_agent1_v06.json")
    if not cases or not a1:
        return res_missing(res, "E1", "phase0_agent1_v06.json missing")
    cases = {c["case_id"]: c for c in cases}
    run = {r["case_id"]: r for r in a1["results"]}

    def confusion(pred):
        tp = fn = tn = fp = 0
        for k in cases:
            y = cases[k]["truth"]["sep3"]
            p = pred(k)
            if y and p:
                tp += 1
            elif y:
                fn += 1
            elif p:
                fp += 1
            else:
                tn += 1
        return rates(tp, fn, tn, fp)

    out = {
        "n": len(cases),
        "operating_points": {
            "probable_or_above": confusion(lambda k: run[k]["result"]["tier"] in POS_TIERS),
            "possible_or_above": confusion(lambda k: run[k]["result"]["tier"] in ALL_POS),
        },
        "baselines": {
            "NEWS>=5": confusion(lambda k: (cases[k]["truth"].get("score_NEWS") or 0) >= 5),
            "qSOFA>=2": confusion(lambda k: (cases[k]["truth"].get("score_qSOFA") or 0) >= 2),
        },
        "table_A_per_stratum": {},
        "tier_distribution": dict(collections.Counter(
            run[k]["result"]["tier"] for k in cases)),
        "note": ("Weighted Table B needs the 1,500-case stratified sample; "
                 "with 8 cases only Table A is meaningful."),
    }
    for s in ("S1", "S2", "S3", "S4"):
        ks = [k for k in cases if cases[k]["stratum"] == s]
        pos = sum(1 for k in ks if run[k]["result"]["tier"] in POS_TIERS)
        out["table_A_per_stratum"][s] = (
            {"sensitivity": pct(pos, len(ks)), "n": len(ks)} if s == "S1"
            else {"specificity": pct(len(ks) - pos, len(ks)), "n": len(ks)})
    res["E1"] = out

    print("E1  Agent 1 recognition versus Sepsis-3      n = 8")
    for k, v in out["operating_points"].items():
        print(f"   agent, {k:<20} {show(v)}")
    for k, v in out["baselines"].items():
        print(f"   baseline {k:<24} {show(v)}")
    print(f"   tiers observed: {out['tier_distribution']}")
    print()


# ───────────────────────────── E2 ─────────────────────────────
def e2(res):
    run = load("phase0_agent2_run.json") or load("synthetic_real_model_run.json")
    if not run:
        return res_missing(res, "E2",
                           "run: python -m radar.agents.run_agent2 --infile phase0_agent2_input.json "
                           "--out phase0_agent2_run.json --model gpt-4.1-mini")
    src = "phase0" if (D / "phase0_agent2_run.json").exists() else "synthetic"
    recs = run["results"] if src == "phase0" else run["results"]["A2"]
    inp = load("phase0_agent2_input.json") if src == "phase0" else \
        (load("agent2_synthetic_cases.json") or [])
    inp = inp.get("cases") if isinstance(inp, dict) else inp
    status = {c["case_id"]: c["input"]["investigation_status"] for c in inp}
    tiers = {c["case_id"]: (c.get("agent1_context") or {}).get("tier", "possible_sepsis")
             for c in inp}

    def ref(s, tier):
        """SSC stmt 7 and 8 apply to possible, probable or definite sepsis.
        Below that threshold, not ordering cultures or lactate is guideline-consistent."""
        if s in ("collected", "ordered", "resulted"):
            return "already_addressed"
        return "recommend" if tier in ALL_POS else "not_indicated"

    NODE = {"W1": "blood_cultures", "W2": "serum_lactate"}
    om = den = agree = tot = indet = 0
    rows = []
    for r in recs:
        got = {t["node"]: t.get("answer") for t in r.get("trace", []) if "answer" in t}
        for node, item in NODE.items():
            rf = ref(status[r["case_id"]][item], tiers.get(r["case_id"], "possible_sepsis"))
            a = got.get(node)
            tot += 1
            agree += a == rf
            indet += a == "indeterminate"
            if rf == "recommend":
                den += 1
                om += a != "recommend"
            rows.append([r["case_id"], item, rf, a])
    res["E2"] = {"source": src, "items_scored": "blood_cultures, serum_lactate",
                 "reference_rule": "SSC stmt 7 / 8, conditioned on the Agent 1 tier",
                 "clinically_indicated_omission_rate_pct": pct(om, den),
                 "agreement_with_reference_pct": pct(agree, tot),
                 "indeterminate_rate_pct": pct(indet, tot),
                 "decisions": tot, "rows": rows,
                 "note": "Source imaging and alternative-diagnosis testing need a clinician-adjudicated reference."}
    print(f"E2  Agent 2 workup appropriateness           source = {src}")
    print(f"   clinically indicated omission rate  {fmt(pct(om, den))}   ({om}/{den})")
    print(f"   agreement with rule reference       {fmt(pct(agree, tot))}   ({agree}/{tot})")
    print(f"   indeterminate rate                  {fmt(pct(indet, tot))}")
    print("   scored on blood cultures and lactate only; the other two items need clinician review")
    print()


# ───────────────────────────── E3 ─────────────────────────────
def e3(res):
    run = load("synthetic_real_model_run.json")
    src = load("agent3_synthetic_cases.json")
    if not run or "A3" not in run.get("results", {}):
        return res_missing(res, "E3",
                           "run: python run_synthetic_all.py --agents A3 --model gpt-4.1-mini")
    src = src.get("cases") if isinstance(src, dict) else src
    idx = {c["case_id"]: c for c in src}
    recs = {r["case_id"]: r for r in run["results"]["A3"]}

    def ref_tx1(tier, shock):
        if shock:
            return "within_1h"
        if tier in POS_TIERS:
            return "within_1h"
        if tier == "possible_sepsis":
            return "within_3h"
        return "defer_and_monitor"

    ok1 = ok4 = n = 0
    rows = []
    for cid, c in idx.items():
        inp = c.get("input", c)
        tier = (inp.get("agent1_result") or {}).get("tier")
        shock = bool((inp.get("agent1_result") or {}).get("shock_concern"))
        age = inp.get("age")
        got = {t["node"]: t.get("answer") for t in recs[cid]["trace"] if "answer" in t}
        r1 = ref_tx1(tier, shock)
        vaso = got.get("TX3") == "start"
        r4 = "not_applicable" if not vaso else ("target_60_65" if age >= 65 else "target_65")
        n += 1
        ok1 += got.get("TX1") == r1
        ok4 += got.get("TX4") == r4
        rows.append([cid, age, tier, r1, got.get("TX1"), r4, got.get("TX4")])
    res["E3"] = {"n": n,
                 "TX1_antimicrobial_concordance_pct": pct(ok1, n),
                 "TX4_map_target_concordance_pct": pct(ok4, n),
                 "TX2_TX3": "no independent reference available",
                 "rows": rows,
                 "reference": "SSC 2026 stmt 13, 14, 16, 17, 18, 20"}
    print("E3  Agent 3 treatment appropriateness        rule-derived reference")
    print(f"   TX1 antimicrobial timing concordance {fmt(pct(ok1, n))}   ({ok1}/{n})")
    print(f"   TX4 blood-pressure target concordance {fmt(pct(ok4, n))}   ({ok4}/{n})")
    print("   TX2 fluids and TX3 vasopressor need a clinician-adjudicated reference")
    print()


# ───────────────────────────── E4 ─────────────────────────────
def historical_pre_vnext_e4(recs, cases):
    """Preserve the original arrival-anchored summary without calling it vNext."""
    case_lookup = {case["case_id"]: case for case in cases}
    discharge = discharge_bad = low = low_icu = over = 0
    for record in recs:
        result = record.get("result") or {}
        disposition = result.get("disposition")
        outcomes = (case_lookup.get(record.get("case_id"), {}).get("truth") or {}).get("outcomes", {})
        bad = bool(outcomes.get("dsp_icu_transfer_12h")) or bool(outcomes.get("dsp_mortality_inhospital"))
        if disposition == "discharge":
            discharge += 1
            discharge_bad += bad
        if disposition in {"observation", "ward"}:
            low += 1
            low_icu += bool(outcomes.get("dsp_icu_transfer_12h"))
        if disposition == "ICU" and not bad:
            over += 1
    return {
        "status": "historical_pre_vnext_only",
        "anchor": "legacy dsp_icu_transfer_12h field; not checkpoint-relative",
        "discharge_boundary_under_triage_pct": pct(discharge_bad, discharge),
        "discharge_denominator": discharge,
        "obs_ward_to_icu_pct": pct(low_icu, low),
        "over_triage_pct": pct(over, len(recs)),
        "current_performance_claim": False,
    }


def e4(res):
    source = next((name for name in (
        "pipeline_run_vnext.json",
        "pipeline_run.json",
        "phase0_agent4_t0_run.json",
    ) if (D / name).exists()), None)
    run = load(source) if source else None
    if not run:
        return res_missing(res, "E4",
                           "run current RADAR vNext and write pipeline_run_vnext.json")
    cases = load("phase0_cases.json") or []
    recs = run.get("results", []) if isinstance(run, dict) else run
    metrics = calculate_metrics(recs, cases)
    primary = metrics["primary"]
    secondary = metrics["secondary"]
    obs_ward = secondary["observation_or_ward_to_icu_12h"]
    over = secondary["possible_overtriage_icu_without_recorded_icu_need"]
    out = {
        "source": source,
        "n": len(recs),
        "decision_anchor": "actual modeled disposition decision cutoff",
        "icu_transfer_boundary": "0 < ICU_transfer_time - modeled_decision_cutoff <= 12 hours",
        "disposition_distribution": secondary["disposition_distribution"],
        "discharge_boundary_under_triage_pct": None if primary["rate"] is None else round(primary["rate"] * 100, 1),
        "discharge_denominator": primary["denominator"],
        "obs_ward_to_icu_pct": None if obs_ward["rate"] is None else round(obs_ward["rate"] * 100, 1),
        "over_triage_pct": None if over["rate"] is None else round(over["rate"] * 100, 1),
        "abstention_rate_pct": None if secondary["abstention_rate"] is None else round(secondary["abstention_rate"] * 100, 1),
        "pipeline_alarm_rate_pct": None if secondary["pipeline_alarm_rate"] is None else round(secondary["pipeline_alarm_rate"] * 100, 1),
        "current_vnext_metrics": metrics,
    }
    if source == "phase0_agent4_t0_run.json":
        out["historical_pre_vnext_summary"] = historical_pre_vnext_e4(recs, cases)
    res["E4"] = out
    print("E4  Agent 4 disposition safety               decision-cutoff anchored")
    print(f"   source: {source}   dispositions: {out['disposition_distribution']}")
    print(f"   discharge-boundary under-triage  {fmt(out['discharge_boundary_under_triage_pct'])}   "
          f"(evaluable denominator {out['discharge_denominator']})")
    print(f"   observation or ward -> ICU 12 h  {fmt(out['obs_ward_to_icu_pct'])}")
    print(f"   abstention {fmt(out['abstention_rate_pct'])}   "
          f"pipeline alarm {fmt(out['pipeline_alarm_rate_pct'])}")
    if "historical_pre_vnext_summary" in out:
        print("   legacy arrival-anchored values are retained separately as historical only")
    print()


# ───────────────────── E5 / E6 / E7 ─────────────────────
def e5(res):
    out = D / "clinician_review_packet.json"
    a1 = load("phase0_agent1_v06.json")
    if not a1:
        return res_missing(res, "E5", "needs the Agent 1 run")
    cases = {c["case_id"]: c for c in load("phase0_cases.json")}
    a2 = load("phase0_agent2_run.json")
    a4 = load("phase0_agent4_t0_run.json")
    packet = []
    for r in a1["results"]:
        cid = r["case_id"]
        packet.append({
            "case_id": cid,
            "information_available_at_decision": cases[cid]["input"],
            "agent1_tier": r["result"]["tier"],
            "agent1_graph_version": a1.get("meta", {}).get("graph_version"),
            "agent1_trace": [{"node": t["node"], "answer": t.get("answer"),
                              "leading_alternative": t.get("leading_alternative", (t.get("reply") or {}).get("leading_alternative")),
                              "evidence": (t.get("reply") or {}).get("supporting_evidence"),
                              "against": (t.get("reply") or {}).get("opposing_evidence")}
                             for t in r["trace"]],
            "agent2_plan": next((x.get("result") for x in (a2 or {}).get("results", [])
                                 if x.get("case_id") == cid), None),
            "agent4_disposition": next((x.get("result") for x in (a4 or {}).get("results", [])
                                        if x.get("case_id") == cid), None),
            "reviewer_verdict": "",     # no major violation | major safety violation | uncertain
            "reviewer_category": "",
            "reviewer_comment": "",
        })
    out.write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
    res["E5"] = {"status": "packet written, awaiting clinician",
                 "file": "clinician_review_packet.json", "cases": len(packet)}
    print("E5  Patient-level safety review")
    print(f"   review packet written -> clinician_review_packet.json  ({len(packet)} cases)")
    print("   fill reviewer_verdict for each case, then rerun this script")
    print()


def longitudinal_transitions(records):
    direction = collections.Counter()
    rows = []
    for record in records:
        previous = ((record.get("agent1_t0") or {}).get("result") or {}).get("tier")
        for checkpoint in record.get("checkpoints") or []:
            if checkpoint.get("checkpoint_status") != "processed":
                continue
            current = ((checkpoint.get("agent1_reassessment") or {}).get("result") or {}).get("tier")
            transition = tier_transition(previous, current)
            if transition:
                direction[transition] += 1
                rows.append({
                    "case_id": record.get("case_id"),
                    "checkpoint_id": checkpoint.get("checkpoint_id"),
                    "from_tier": previous,
                    "to_tier": current,
                    "direction": transition,
                })
            if current in ALL_POS | {"unlikely_sepsis"}:
                previous = current
    return {"transitions_observed": len(rows), "direction_counts": dict(direction), "rows": rows}


def e6(res):
    for name in ("pipeline_run_vnext.json", "pipeline_run.json"):
        run = load(name)
        if run:
            break
    else:
        return res_missing(res, "E6",
                           "run: python -m radar.pipeline.run_pipeline_vnext --infile "
                           "checkpoint_replay_fixture.json "
                           "--out pipeline_run_vnext.json --model gpt-4.1-mini")
    recs = run.get("results") or run
    summary = longitudinal_transitions(recs if isinstance(recs, list) else [])
    summary["source"] = name
    summary["architecture"] = "RADAR vNext T0 plus processed sequential checkpoints"
    summary["note"] = "Correctness needs a clinician reference for the expected direction."
    res["E6"] = summary
    print("E6  Longitudinal tier transitions")
    print(f"   transitions observed {summary['transitions_observed']}   {summary['direction_counts']}")
    print("   correctness requires the clinician direction reference")
    print()


def e7(res):
    have = [k for k in ("E1", "E2", "E3", "E4") if k in res and "missing" not in res[k]]
    res["E7"] = {"status": f"components available: {have}",
                 "note": "Composite pathway success needs all four component criteria and a "
                         "single cohort run end to end."}
    print("E7  End-to-end composite")
    print(f"   components available: {have}")
    print()


# ───────────────────────────── E8 ─────────────────────────────
def e8(res):
    out = {}
    a1 = load("phase0_agent1_v06.json")
    if a1:
        calls = req = retry = ev = nullc = ind = 0
        allowed = 0
        for r in a1["results"]:
            for t in r["trace"]:
                calls += 1
                att = t.get("attempts") or []
                req += len(att)
                retry += len(att) > 1
                if t.get("node") != "R4":
                    allowed += 1
                    ind += t.get("answer") == "indeterminate"
                rep = t.get("reply") or {}
                for kk in ("supporting_evidence", "opposing_evidence"):
                    for e in rep.get(kk) or []:
                        ev += 1
                        nullc += not e.get("statement_id")
        out["Agent 1 (8 MIMIC)"] = {
            "node_calls": calls, "first_pass_pct": pct(calls - retry, calls),
            "retry_pct": pct(retry, calls), "abstention_pct": 0.0,
            "unsupported_evidence_pct": pct(nullc, ev),
            "indeterminate_pct": pct(ind, allowed)}
    syn = load("synthetic_real_model_run.json")
    if syn:
        for a, m in syn["metrics"].items():
            out[f"Agent {a[-1]} (synthetic)"] = {
                "node_calls": m["node_calls"],
                "first_pass_pct": m["first_pass_validation_rate_pct"],
                "retry_pct": m["retry_rate_pct"],
                "abstention_pct": m["abstention_rate_pct"],
                "unsupported_evidence_pct": m["unsupported_evidence_rate_pct"],
                "indeterminate_pct": m["indeterminate_rate_pct"]}
    r3 = load("synthetic_real_model_run_r3.json")
    if r3 and r3.get("consistency"):
        out["repeat_consistency"] = {
            a: {"decisions_pct": c["decision_consistency_pct"],
                "verbatim_pct": c["verbatim_consistency_pct"], "cases": c["total_cases"]}
            for a, c in r3["consistency"].items()}
    res["E8"] = out
    print("E8  Grounding and system reliability")
    hdr = ("", "calls", "1st pass", "retry", "abstain", "uncited", "not sure")
    print("   {:<22}{:>7}{:>10}{:>8}{:>9}{:>9}{:>10}".format(*hdr))
    for k, v in out.items():
        if k == "repeat_consistency":
            continue
        print("   {:<22}{:>7}{:>10}{:>8}{:>9}{:>9}{:>10}".format(
            k, v["node_calls"], fmt(v["first_pass_pct"]), fmt(v["retry_pct"]),
            fmt(v["abstention_pct"]), fmt(v["unsupported_evidence_pct"]),
            fmt(v["indeterminate_pct"])))
    if "repeat_consistency" in out:
        for a, c in out["repeat_consistency"].items():
            print(f"   repeat {a}: decisions {c['decisions_pct']}%  "
                  f"verbatim {c['verbatim_pct']}%  (n={c['cases']})")
    print()


def res_missing(res, key, how):
    res[key] = {"missing": how}
    print(f"{key}  NOT AVAILABLE")
    print(f"   {how}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(D / "hema_metrics.json"))
    args = p.parse_args()
    res = {}
    print()
    for fn in (e1, e2, e3, e4, e5, e6, e7, e8):
        try:
            fn(res)
        except Exception as exc:                                  # noqa: BLE001
            print(f"{fn.__name__.upper()}  error: {type(exc).__name__}: {exc}\n")
    pathlib.Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
