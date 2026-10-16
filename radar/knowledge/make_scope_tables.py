#!/usr/bin/env python3
"""Generate each decision node's scope table by traversing the graph OFFLINE.

Give a node a handful of seed concepts; the traversal returns the statements it
may read. The result is frozen to disk and is what the runtime injects — nothing
walks the graph while a patient is being processed.

Expansion rules, in order:
  1. seeds are widened through IS_DISJUNCTION_OF (a compound concept matches its parts,
     and a part matches any compound built from it)
  2. reverse APPLIES_WHEN  -> statements that fire on any seed condition
  3. OVERRIDDEN_BY         -> if a statement is in scope, the statement that overrides it
                              must be too, or the agent sees a superseded target
  4. MUST_PRECEDE          -> pull in the statements on the other side of a temporal
                              constraint, even when they sit in a different domain
  5. opposite polarity     -> if a statement recommends action X, also pull statements
                              that recommend AGAINST X. This is the branch partner, and
                              it is the rule a similarity search cannot express.
"""
from radar import paths
import json, pathlib, sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

HERE = pathlib.Path(__file__).parent
ROOT = paths.ROOT
def find(name):
    """Resolve a data file wherever it lives in the repo layout.

    Parsed statements sit in knowledge/parsed, graph artifacts in knowledge/graph.
    """
    for d in (paths.GRAPH, paths.PARSED, paths.FIXTURES):
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found under {ROOT}")


G = json.loads(find("ssc2026_graph.json").read_text(encoding="utf-8"))
STMTS = {s["id"]: s for s in json.loads(find("ssc2026_statements.json").read_text(encoding="utf-8"))}
NODES = {n["id"]: n for n in G["nodes"]}
E = G["edges"]

by_rel = defaultdict(list)
for e in E:
    by_rel[e["rel"]].append(e)

disj_down = defaultdict(set)   # compound -> parts
disj_up   = defaultdict(set)   # part -> compounds
for e in by_rel["IS_DISJUNCTION_OF"]:
    disj_down[e["src"]].add(e["dst"]); disj_up[e["dst"]].add(e["src"])

applies = defaultdict(set)     # condition -> statements
for e in by_rel["APPLIES_WHEN"]:
    applies[e["dst"]].add(e["src"])

recs = defaultdict(list)       # action -> [(statement, polarity)]
for e in by_rel["RECOMMENDS"]:
    recs[e["dst"]].append((e["src"], e["polarity"]))
stmt_actions = defaultdict(set)
for e in by_rel["RECOMMENDS"]:
    stmt_actions[e["src"]].add(e["dst"])

overridden = {e["src"]: e["dst"] for e in by_rel["OVERRIDDEN_BY"]}
precede = by_rel["MUST_PRECEDE"]


# every edge that ties a statement to a topic (an Action / Target / Constraint)
touches = defaultdict(set)     # statement -> topic ids
for e in E:
    if e["rel"] in ("RECOMMENDS", "PREFERRED_OVER", "EQUIVALENT_OPTIONS",
                    "TARGETS", "CONSTRAINS", "REFINED_BY"):
        touches[e["src"]].add(e["dst"])


def scope(cond_seeds, topic_seeds):
    """A scope is an INTERSECTION, not a traversal closure.

    Seeding only on patient state returns everything that applies to a septic
    shock patient — 62 statements, most of them about therapies this node is
    not deciding. The node's topic (which Actions / Targets / Constraints it is
    reasoning about) is the second, indispensable filter.
    """
    why = defaultdict(list)

    wide = set(cond_seeds)
    for c in list(cond_seeds):
        wide |= disj_down.get(c, set()) | disj_up.get(c, set())

    state_ok = set()
    for c in wide:
        for sid in applies.get(c, ()):
            state_ok.add(sid); why[sid].append(f"state:{c}")

    topic = set(topic_seeds)
    topic_ok = {sid for sid in state_ok if touches[sid] & topic}
    for sid in topic_ok:
        for t in sorted(touches[sid] & topic):
            why[sid].append(f"topic:{t}")

    out = set(topic_ok)

    # overrides: a superseded target must never be shown without its override
    for sid in list(out):
        if sid in overridden:
            o = overridden[sid]; out.add(o); why[o].append(f"overrides:{sid}")

    # opposite-polarity partners: seeing only one side of a for/against pair is
    # the failure mode we are trying to design out, so these always come along
    for sid in list(out):
        for a in stmt_actions.get(sid, set()):
            if len({p for _, p in recs[a]}) > 1:
                for s2, _ in recs[a]:
                    if s2 not in out:
                        out.add(s2); why[s2].append(f"opposite_polarity_on:{a}")

    # temporal partners, but only when the partner action is itself on topic
    for pr in precede:
        for sid in list(out):
            if pr["src"] in stmt_actions.get(sid, set()) and pr["dst"] in topic:
                for s2, _ in recs.get(pr["dst"], []):
                    if s2 not in out:
                        out.add(s2); why[s2].append(f"must_precede_partner_of:{sid}")

    for sid in list(out):
        why.setdefault(sid, []).append("pulled_by_expansion")
    ordered = sorted(out, key=lambda s: NODES[s]["number"])
    provenance = {sid: sorted(set(why[sid])) for sid in ordered}
    return ordered, provenance


SCREEN   = ["screen_for_sepsis", "screen_for_sepsis_with_tool", "use_standard_sepsis_screening_tool",
            "use_code_sepsis_or_sepsis_huddle_protocol",
            "constraint::rule_in_or_out_sepsis_on_single_test",
            "constraint::novel_rapid_host_response_diagnostics"]
MIMIC    = ["assess_likelihood_infectious_vs_noninfectious", "search_for_alternative_diagnoses",
            "reevaluate_continuously", "discontinue_empiric_antimicrobial_therapy",
            "constraint::rule_in_or_out_sepsis_on_single_test"]
SEVERITY = ["begin_treatment_and_resuscitation", "measure_blood_lactate",
            "use_serial_lactate_measurements_to_guide_resuscitation",
            "constraint::rule_in_or_out_sepsis_on_single_test"]
SHOCK    = ["administer_iv_crystalloid", "administer_iv_crystalloid_bolus", "start_vasopressor",
            "start_vasopressor_peripherally", "target::MAP", "begin_treatment_and_resuscitation",
            "titrate_vasopressor_to_maintain_MAP"]
CULTURE  = ["collect_blood_cultures", "administer_antimicrobial_therapy"]
LACTATE  = ["measure_blood_lactate", "use_serial_lactate_measurements_to_guide_resuscitation",
            "use_capillary_refill_time_to_guide_resuscitation"]
PCT      = ["use_clinical_evaluation_alone_to_start_antimicrobials",
            "procalcitonin_plus_clinical_evaluation"]
ABX      = ["administer_antimicrobial_therapy", "perform_time_limited_rapid_investigation",
            "monitor_patient_closely"]
FLUID    = ["administer_iv_crystalloid", "administer_iv_crystalloid_bolus",
            "use_crystalloids_as_first_line_fluid", "use_balanced_crystalloids",
            "use_crystalloids_alone", "use_starches_for_resuscitation",
            "use_gelatin_for_resuscitation", "liberal_fluid_resuscitation_strategy",
            "restrictive_fluid_resuscitation_strategy", "reassess_fluid_status_frequently",
            "calculate_dose_on_actual_body_weight",
            "use_dynamic_measures_to_guide_fluid_resuscitation"]
VASO     = ["start_vasopressor", "start_vasopressor_peripherally", "target::MAP",
            "use_norepinephrine_as_first_line_vasopressor", "add_vasopressin", "add_epinephrine",
            "use_terlipressin", "use_inotropes", "use_levosimendan",
            "use_beta_blockers_for_septic_shock", "titrate_vasopressor_to_maintain_MAP",
            "norepinephrine_first_line", "epinephrine_first_line",
            "add_dobutamine_to_norepinephrine", "epinephrine_alone"]

SEEDS = {
  "agent1_recognition": {
    "R1":  (["suspected_sepsis", "acutely_ill", "in_hospital", "positive_sepsis_screen",
             "prehospital_ambulance_or_flight", "acutely_ill_high_risk_patients"], SCREEN),
    "R2":  (["suspected_sepsis", "sepsis", "septic_shock", "possible_sepsis",
             "probable_sepsis", "definite_sepsis"], SEVERITY),
    "R3a": (["suspected_sepsis", "possible_sepsis", "low_likelihood_of_infection",
             "suspected_sepsis_or_septic_shock", "no_shock"], MIMIC),
    "R3b": (["suspected_sepsis", "possible_sepsis", "low_likelihood_of_infection",
             "suspected_sepsis_or_septic_shock", "no_shock"], MIMIC),
    "R4":  (["septic_shock", "sepsis_induced_hypoperfusion", "sepsis_induced_hypotension"], SHOCK),
  },
  "agent2_workup": {
    "W1_cultures":     (["possible_sepsis", "probable_sepsis", "definite_sepsis", "septic_shock"], CULTURE),
    "W2_lactate":      (["possible_sepsis", "probable_sepsis", "definite_sepsis", "septic_shock",
                         "sepsis_with_elevated_lactate", "sepsis"], LACTATE),
    "W3_mimic_workup": (["possible_sepsis", "no_shock", "suspected_sepsis"], MIMIC),
    "W4_procalcitonin":(["possible_sepsis", "probable_sepsis", "septic_shock"], PCT),
  },
  "agent3_treatment": {
    "TX1_antibiotic_timing": (["possible_sepsis", "probable_sepsis", "definite_sepsis", "septic_shock",
                              "no_shock", "low_likelihood_of_infection", "possible_septic_shock"], ABX),
    "TX2_fluids":            (["sepsis_induced_hypoperfusion", "sepsis_induced_hypotension", "sepsis",
                              "septic_shock", "persistent_hypoperfusion"], FLUID),
    "TX3_vasopressors":      (["septic_shock", "escalating_norepinephrine_dose", "cardiac_dysfunction",
                              "sepsis_induced_hypotension"], VASO),
  },
  "agent4_disposition": {
    "D1_disposition": ([], []),   # SSC has no ED-disposition statement — empty by design
  },
}

HANDWRITTEN_AGENT1 = {   # what we authored before the graph existed
  "R1":  ["stmt_001","stmt_002","stmt_003","stmt_004","stmt_005","stmt_006"],
  "R2":  ["stmt_004","stmt_005","stmt_008","stmt_009"],
  "R3a": ["stmt_005","stmt_006"],
  "R3b": ["stmt_005","stmt_006"],
  "R4":  ["stmt_009","stmt_010","stmt_011","stmt_013"],
}

if __name__ == "__main__":
    tables, report = {}, []
    for agent, nodes in SEEDS.items():
        tables[agent] = {}
        for nid, (cseeds, tseeds) in nodes.items():
            ids, why = scope(cseeds, tseeds)
            tables[agent][nid] = {"condition_seeds": cseeds, "topic_seeds": tseeds,
                                  "statements": ids, "provenance": why}
            report.append((agent, nid, len(ids)))

    json.dump({"meta": {"generated_from": "ssc2026_graph.json",
                        "frozen": True,
                        "note": "runtime injects these ids verbatim; no traversal at inference"},
               "scope": tables}, open(paths.GRAPH / "scope_tables.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"{'agent':<22}{'node':<24}{'条数':>5}")
    for a, n, k in report: print(f"{a:<22}{n:<24}{k:>5}")

    print("\n" + "="*78)
    print("Agent 1：手写书签表  vs  图生成书签表")
    print("="*78)
    for nid, hand in HANDWRITTEN_AGENT1.items():
        gen = set(tables["agent1_recognition"][nid]["statements"])
        h = set(hand)
        why = tables["agent1_recognition"][nid]["provenance"]
        print(f"\n[{nid}]  手写 {len(h)} 条 → 图生成 {len(gen)} 条")
        for s in sorted(gen - h, key=lambda x: NODES[x]["number"]):
            print(f"   ＋ {s}  {STMTS[s]['statement'][:88]}")
            print(f"       ↳ {'; '.join(why[s])}")
        for s in sorted(h - gen, key=lambda x: NODES[x]["number"]):
            print(f"   － {s}  {STMTS[s]['statement'][:88]}")

    allh = set().union(*HANDWRITTEN_AGENT1.values())
    allg = set().union(*(set(v["statements"]) for v in tables["agent1_recognition"].values()))
    print("\n" + "-"*78)
    print(f"Agent 1 合计   手写 {len(allh)} 条独立条款 → 图生成 {len(allg)} 条")
    print(f"  图找到而手写漏掉：{len(allg-allh)} 条 —— {sorted(allg-allh, key=lambda x: NODES[x]['number'])}")
    print(f"  手写有而图没给：{len(allh-allg)} 条 —— {sorted(allh-allg, key=lambda x: NODES[x]['number']) or '无'}")
