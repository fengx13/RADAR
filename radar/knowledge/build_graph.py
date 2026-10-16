#!/usr/bin/env python3
"""Build the SSC 2026 knowledge graph from the extracted triples.

The graph is used OFFLINE only — to generate each decision node's frozen scope
table. Nothing traverses it at inference time, so the same patient always gets
the same statements (TRIPOD-LLM asks for run-to-run consistency).
"""
from radar import paths
import json, pathlib, re
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).parent
TRIPLES = json.loads((paths.GRAPH / "triples_all.json").read_text())
STMTS = {s["id"]: s for s in json.loads((paths.PARSED / "ssc2026_statements.json").read_text())}
T1 = json.loads((paths.PARSED / "table1_findings.json").read_text())

# ---------------------------------------------------------------- 1. normalise
# Approach A: keep the compound concept as a real node and give it an
# IS_DISJUNCTION_OF edge, rather than nesting booleans inside applies_when.
DISJUNCTIONS = {
    "sepsis_or_septic_shock":                    ["sepsis", "septic_shock"],
    "probable_or_definite_sepsis":               ["probable_sepsis", "definite_sepsis"],
    "definite_or_probable_sepsis":               ["probable_sepsis", "definite_sepsis"],
    "sepsis_with_elevated_lactate":              ["sepsis", "elevated_lactate"],
    "sepsis_associated_hypoxemic_respiratory_failure": ["sepsis", "acute_hypoxemic_respiratory_failure"],
    "severe_ards_due_to_sepsis":                 ["sepsis", "severe_ards"],
    "hospitalization_for_sepsis_or_septic_shock":["sepsis", "septic_shock"],
    "possible_septic_shock":                     ["possible_sepsis", "septic_shock"],
    "probable_septic_shock":                     ["probable_sepsis", "septic_shock"],
    "definite_septic_shock":                     ["definite_sepsis", "septic_shock"],
}
# exact synonyms — same clinical concept, different wording across statements
SYNONYMS = {
    "definite_or_probable_sepsis": "probable_or_definite_sepsis",
    "adequate_volume_status":      "adequate_fluid_status",
    "adequate_fluid_resuscitation":"adequate_fluid_status",
}
norm = lambda c: SYNONYMS.get(c, c)

# a more specific statement wins over a general one covering the same target
OVERRIDES = [("stmt_014", "stmt_013", "adds age_ge_65 to the same MAP target"),
             ("stmt_096", "stmt_095", "adds pH<=7.2 and AKIN 2-3 to the same drug, opposite direction")]

# ---------------------------------------------------------------- 2. nodes
nodes, edges = {}, []
def node(nid, ntype, **kw):
    nodes.setdefault(nid, dict(id=nid, type=ntype, **kw))
    return nid
def edge(src, rel, dst, **kw):
    edges.append(dict(src=src, rel=rel, dst=dst, **kw))

for r in TRIPLES["statements"]:
    sid, s = r["id"], STMTS[r["id"]]
    node(sid, "Statement", number=r["number"], domain=s["domain"],
         strength=s["strength"], certainty=s["certainty"], stmt_type=s["type"],
         depth=r["depth"], extraction_confidence=r["extraction_confidence"],
         journal_page=s["journal_page"])

    for c in r["applies_when"]["conditions"]:
        c = norm(c); node(c, "Condition")
        edge(sid, "APPLIES_WHEN", c, mode=r["applies_when"]["mode"])
    for a in r["recommends"]:
        node(a["action"], "Action")
        edge(sid, "RECOMMENDS", a["action"], polarity=a["polarity"],
             **{k: v for k, v in a.items() if k not in ("action", "polarity")})
    for p in r["preferred_over"]:
        node(p, "Action")
        edge(sid, "PREFERRED_OVER", p)
    if r["equivalent_options"]:
        for o in r["equivalent_options"]["options"]:
            node(o, "Action"); edge(sid, "EQUIVALENT_OPTIONS", o)
    for tg in r["targets"]:
        tid = f"target::{tg['parameter']}"
        node(tid, "Target", parameter=tg["parameter"])
        edge(sid, "TARGETS", tid, **{k: v for k, v in tg.items() if k != "parameter"})
    for pr in r["must_precede"]:
        node(pr["before"], "Action"); node(pr["after"], "Action")
        edge(pr["before"], "MUST_PRECEDE", pr["after"], via=sid, strength=pr.get("strength"))
    for cn in r["constrains"]:
        cid = "constraint::" + (cn.get("forbidden") or cn.get("insufficient_evidence_for"))
        node(cid, "Constraint", kind="forbidden" if "forbidden" in cn else "insufficient_evidence")
        edge(sid, "CONSTRAINS", cid)
    for rf in r["refined_by"]:
        if rf.get("action"):
            node(rf["action"], "Action")
            edge(sid, "REFINED_BY", rf["action"], remark=rf["remark"],
                 polarity=rf.get("polarity", "for"), kind=rf["kind"],
                 modifier_condition=rf.get("modifier_condition"), carries_action=True)
        if rf.get("modifier_condition"):
            mc = norm(rf["modifier_condition"]); node(mc, "Condition")
            edge(sid, "MODIFIED_BY", mc, remark=rf["remark"])

for compound, parts in DISJUNCTIONS.items():
    c = norm(compound)
    if c in nodes:
        for p in parts:
            node(p, "Condition"); edge(c, "IS_DISJUNCTION_OF", p)

for sid in T1["has_2021_statement"]:
    s = f"stmt_{sid:03d}"
    if s in nodes:
        edge(s, "SUPERSEDES", f"ssc2021::{s}",
             change=("upgraded" if sid in T1["strength_upgraded_from_2021"]
                     else "downgraded" if sid in T1["strength_downgraded_from_2021"] else "unchanged"))
for spec, gen, why in OVERRIDES:
    edge(gen, "OVERRIDDEN_BY", spec, reason=why)

# ---------------------------------------------------------------- 3. checks
warn = []
for a, b, _ in OVERRIDES:
    if a not in nodes or b not in nodes: warn.append(f"override refers to missing node {a}/{b}")
orphan = [n for n, d in nodes.items() if d["type"] == "Condition"
          and not any(e["dst"] == n for e in edges)]
# a Condition used by exactly one statement is fine, but flag Actions that appear
# both as RECOMMENDS-for and RECOMMENDS-against — that is where 22 vs 40 lives
pol = defaultdict(set)
for e in edges:
    if e["rel"] == "RECOMMENDS": pol[e["dst"]].add(e["polarity"])
both = {k: sorted(v) for k, v in pol.items() if len(v) > 1}

graph = {"meta": {"built_from": "SSC 2026 (Prescott et al. Crit Care Med 2026;54:725-812)",
                  "statements": len(TRIPLES["statements"]),
                  "traversal": "offline only — scope tables are frozen before inference"},
         "nodes": list(nodes.values()), "edges": edges}
json.dump(graph, open(paths.GRAPH / "ssc2026_graph.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    nt, et = Counter(d["type"] for d in nodes.values()), Counter(e["rel"] for e in edges)
    print(f"节点 {len(nodes)}   边 {len(edges)}\n")
    print("节点类型："); [print(f"   {k:<12}{v:>5}") for k, v in nt.most_common()]
    print("\n边类型：");  [print(f"   {k:<22}{v:>5}") for k, v in et.most_common()]
    print(f"\n同一动作既被推荐又被反对（分支条件不同）：{len(both)}")
    for k, v in both.items(): print(f"   {k}  {v}")
    print(f"\n自检告警：{warn or '无'}")
    deg = Counter(e["src"] for e in edges if e["src"].startswith("stmt_"))
    print("\n出边最多的条款：")
    for s, n in deg.most_common(5):
        print(f"   {s}  {n} 条边   {STMTS[s]['statement'][:78]}…")
