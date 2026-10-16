"""Build the interactive RADAR demo page from a run over demo/scenarios/demo_cases.json.

    python demo/build_demo.py --run ../RADAR_output/demo_runs/demo_run_v1.json \
        [--run ../RADAR_output/demo_runs/demo_run_luna_v1.json] --out ../RADAR_output/demo_runs/radar_demo.html

Only synthetic cases (made by demo/make_scenarios.py) are accepted, so the
page contains no patient-level data and may be shared.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from radar.paths import PARSED  # noqa: E402

SCEN = HERE / "scenarios" / "demo_cases.json"
TITLES = {  # node: (zh, en)
    "R1": ("感染证据", "Evidence of infection"), "R2": ("器官功能障碍", "Organ dysfunction"),
    "R3a": ("替代诊断排序（有器官功能障碍）", "Alternative diagnoses (with organ dysfunction)"),
    "R3b": ("替代诊断排序（无明确器官功能障碍）", "Alternative diagnoses (no clear organ dysfunction)"),
    "R4": ("休克 / 低灌注", "Shock / hypoperfusion"),
    "W1": ("血培养", "Blood cultures"), "W2": ("乳酸", "Lactate"),
    "W3": ("感染源影像或取样", "Source imaging or sampling"), "W4": ("替代诊断检查", "Alternative-diagnosis testing"),
    "TX1": ("抗菌药时机", "Antimicrobial timing"), "TX2": ("初始补液", "Initial fluids"),
    "TX3": ("升压药", "Vasopressors"), "TX4": ("MAP 目标", "MAP target"),
    "D0": ("必需信息检查", "Required-field check"), "D1": ("ICU 安全闸门", "ICU safety gate"),
    "D2": ("是否需要住院级别照护", "Inpatient-level need"), "D3": ("是否需要留观", "Observation need"),
    "D4": ("能否安全出院", "Safe for discharge"),
}


def statements() -> dict:
    out = {}
    for s in json.loads((PARSED / "ssc2026_statements.json").read_text()):
        out[s["id"]] = {"t": s["statement"], "r": s.get("remarks", []), "s": s.get("strength"),
                        "c": s.get("certainty"), "src": f"SSC 2026 · statement {s.get('number')}"}
    for s in json.loads((PARSED / "sepsis3_propositions.json").read_text())["propositions"]:
        out[s["id"]] = {"t": s["statement"], "r": s.get("remarks", []), "s": s.get("strength"),
                        "c": s.get("certainty"), "src": f"Sepsis-3 · {s.get('located_in')}"}
    t = json.loads((PARSED / "table3_terminology.json").read_text())
    out[t["id"]] = {"t": t["verbatim_intro"], "r": [f"{r['tier'].replace('_', ' ')}: {r['verbatim']}" for r in t["tiers"]],
                    "s": None, "c": None, "src": "SSC 2026 · Table 3"}
    return out


def ev(e):
    return {"f": e.get("patient_finding"), "id": e.get("statement_id"), "q": e.get("quote"),
            "im": e.get("implication") or e.get("alternative_explanation")}


def node(n):
    zh, en = TITLES.get(n.get("node"), (n.get("title"), n.get("title")))
    out = {"n": n.get("node"), "zh": zh, "en": en, "a": n.get("answer")}
    if n.get("attempts") is not None:
        out["v"] = [{"ok": a.get("ok"), "f": [f if isinstance(f, str) else json.dumps(f) for f in a.get("failures") or []]}
                    for a in n["attempts"]]
    r = n.get("reply")
    if r:
        out.update(sup=[ev(e) for e in r.get("supporting_evidence") or []],
                   opp=[ev(e) for e in r.get("opposing_evidence") or []],
                   miss=r.get("missing_information") or [], conf=r.get("confidence"),
                   alt=r.get("leading_alternative"))
    else:
        out["det"] = True
        if "icu_triggers" in n:
            out["icu"] = n["icu_triggers"]
    return out


def agent(b):
    if not b:
        return None
    r = b.get("result") or {}
    keep = {k: r.get(k) for k in ("tier", "shock_concern", "table3_verbatim", "status", "disposition",
                                  "key_reasons", "unresolved_risks", "missing_information", "abstain_reason",
                                  "escalation_trigger", "workup_plan", "treatment_plan") if r.get(k) is not None}
    return {"r": keep, "status": b.get("status") or r.get("status"), "nodes": [node(n) for n in b.get("trace") or []]}


def cited(o, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "id" and isinstance(v, str):
                acc.add(v)
            elif k == "basis" and isinstance(v, list):
                acc.update(x for x in v if isinstance(x, str))
            else:
                cited(v, acc)
    elif isinstance(o, list):
        for v in o:
            cited(v, acc)
    return acc


def build_one(run_path: Path, scen: dict) -> dict:
    run = json.loads(run_path.read_text())
    by_id = {c["case_id"]: c for c in run["results"]}
    t0s, cps, missing = {}, {}, []
    for pid in scen["patients"]:
        t0 = by_id.get(f"{pid}_C0_5_L2_M2") or next((c for k, c in by_id.items() if k.startswith(pid)), None)
        t0s[pid] = {"a1": agent(t0.get("agent1_t0")), "a2": agent(t0.get("agent2")),
                    "a4": agent(t0.get("agent4_t0"))} if t0 and t0.get("agent1_t0") else None
    for case in scen["cases"]:
        d, c = case["demo"], by_id.get(case["case_id"])
        key = f"{d['patient']}|{d['checkpoint']}|{d['lactate']}|{d['map']}"
        if not c:
            missing.append(case["case_id"])
            continue
        cp = next((x for x in c.get("checkpoints") or [] if x["checkpoint_id"] == d["checkpoint"]), None)
        if not cp or cp.get("checkpoint_status") != "processed":
            cps[key] = {"err": c.get("stop_reason") or c.get("error") or "not processed"}
            continue
        ps = cp.get("patient_state") or {}
        cps[key] = {
            "a1": agent(cp.get("agent1_reassessment")), "a3": agent(cp.get("agent3")), "a4": agent(cp.get("agent4")),
            "rec": [{"t": i["treatment"], "r": i.get("recommended"), "o": i.get("observed"), "al": i.get("alignment")}
                    for i in (cp.get("reconciliation") or {}).get("items") or []],
            "sofa": {k: (ps.get("sofa") or {}).get(k) for k in ("total_from_available_components", "subscores", "not_scored")}}
    meta = run.get("meta", {})
    return {"meta": {"run_id": meta.get("run_id"), "model": meta.get("model"), "dry_run": meta.get("dry_run"),
                     "code_commit": (meta.get("code_commit") or "")[:7], "missing": missing},
            "t0": t0s, "cp": cps}


LABELS = {"claude-sonnet-5-5": "Sonnet 5.5", "gpt-6-luna": "GPT-6 Luna"}


def build(run_paths: list[Path]) -> dict:
    scen = json.loads(SCEN.read_text())
    models = []
    for rp in run_paths:
        m = build_one(rp, scen)
        name = m["meta"]["model"] or rp.stem
        models.append(dict(m, id=rp.stem, label=LABELS.get(name, name)))
    allst = statements()
    ids = cited([[m["t0"], m["cp"]] for m in models], set())
    patients = {k: {kk: v[kk] for kk in ("label", "t0", "oxygen", "labs")} for k, v in scen["patients"].items()}
    return {"levels": scen["levels"], "patients": patients, "models": models,
            "st": {i: allst[i] for i in sorted(ids) if i in allst}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, action="append",
                    help="run file; repeat to add a model switch to the page (first one is the default)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    data = build(a.run)
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    page = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page, encoding="utf-8")
    for m in data["models"]:
        print(f"  {m['label']}: checkpoints {len(m['cp'])}, missing {len(m['meta']['missing'])}"
              + (" [DRY RUN]" if m["meta"]["dry_run"] else ""))
    print(f"wrote {a.out} ({a.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
