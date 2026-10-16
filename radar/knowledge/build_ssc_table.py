#!/usr/bin/env python3
"""Assemble the final SSC 2026 structured table: parser output + 2 hand-filled + QC."""
from radar import paths
import json, re, csv, importlib.util, sys, pathlib
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
OUT = paths.PARSED
OUT.mkdir(exist_ok=True)

# 01_Parsed/ssc2026_statements.json is a FROZEN artifact: 129 statements, verified
# statement by statement against the printed guideline in SSC2026_129条_校验表.html.
# Re-running the parser on a different copy of the PDF does not reproduce it — text
# extraction differs between renderings, and 2 statements are hand-filled. Everything
# downstream (the graph, the scope tables, the ablation) cites these ids, so the file
# is not regenerated casually.
FROZEN = OUT / "ssc2026_statements.json"
if FROZEN.exists() and "--force" not in sys.argv:
    have = len(json.load(open(FROZEN, encoding="utf-8")))
    raise SystemExit(
        f"{FROZEN.relative_to(ROOT)} already exists with {have} statements and is frozen.\n"
        "  Nothing downstream needs this script rerun. If you really mean to rebuild\n"
        "  the table from the PDF, back the file up first and pass --force.")

spec = importlib.util.spec_from_file_location("p", str(HERE / "parse_ssc2026.py"))
p = importlib.util.module_from_spec(spec); sys.argv = ['x']; spec.loader.exec_module(p)
recs = dict(p.records)

# ---- 2 statements the parser could not anchor (column-flow artefacts); read off the page ----
MANUAL = {
 29: dict(id="stmt_029", number=29, pdf_page=27, journal_page=751,
   statement='For adults with sepsis or septic shock with specific risk factors for anaerobic '
             'infection, we “suggest” using an empiric antibiotic regimen that includes anaerobic coverage.',
   strength="conditional_for", certainty="very_low", type="new",
   remarks=["Risk factors for anaerobic infection include intra-abdominal or deep seated "
            "gynecological/obstetric source of infection, necrotizing soft-tissue infection, "
            "head and neck infection, and CNS abscesses or empyema."]),
 121: dict(id="stmt_121", number=121, pdf_page=61, journal_page=785,
   statement='There is “insufficient evidence” to issue a recommendation regarding providing '
             'sepsis-focused educational material to primary care providers as compared with usual care.',
   strength="insufficient_evidence", certainty=None, type="new", remarks=[]),
}
for n, r in MANUAL.items():
    r['source'] = 'manual_from_page_image'
    recs[n] = r
for n, r in recs.items():
    r.setdefault('source', 'auto_parsed')
    r.setdefault('review_flags', [])

# ---- decisions confirmed by the team ----
# A1: certainty is stated per comparator, not in the parenthetical -> keep it, don't collapse it
BY_COMPARATOR = {
    53: {"dopamine": "high", "epinephrine": "low", "selepressin": "low"},
    55: {"vasopressin": "low", "angiotensin_II": "very_low"},
}
for n, cb in BY_COMPARATOR.items():
    recs[n]['certainty_by_comparator'] = cb
    recs[n]['certainty'] = None
    recs[n]['review_flags'] = [f for f in recs[n]['review_flags'] if f != 'no_certainty']
    recs[n]['source'] = 'auto_parsed+verified'
# "Revisit" in statements 60/61 is a typo for the "Revisited" label used everywhere else
for n in (60, 61):
    if recs[n]['type'] is None:
        recs[n]['type'] = 'revisited'
        recs[n]['review_flags'] = [f for f in recs[n]['review_flags'] if f != 'no_type']
        recs[n]['source'] = 'auto_parsed+verified'
for n in (55,):
    if recs[n]['type'] is None:
        recs[n]['type'] = 'revisited'
        recs[n]['review_flags'] = [f for f in recs[n]['review_flags'] if f != 'no_type']
# 29 / 121 transcription confirmed against the page image by the team
for n in (29, 121):
    recs[n]['source'] = 'manual_from_page_image+verified'
for r in recs.values():
    r.setdefault('certainty_by_comparator', None)

# Table 1 shows no strength symbol on these — the guideline declares two options equivalent
# rather than recommending one. Modelling them as RECOMMENDS would assert a preference the
# guideline does not make, so they carry an EQUIVALENT_OPTIONS edge in the graph instead.
EQUIVALENT_OPTIONS = {"42": ["invasive_blood_pressure_monitoring", "noninvasive_blood_pressure_monitoring"], "48": ["liberal_fluid_resuscitation_strategy", "restrictive_fluid_resuscitation_strategy"], "58": ["norepinephrine_first_line", "epinephrine_first_line"], "61": ["add_dobutamine_to_norepinephrine", "epinephrine_alone"], "65": ["pulse_oximetry_SpO2", "arterial_blood_gas_SaO2"], "66": ["liberal_oxygen_target", "conservative_oxygen_target"], "94": ["continuous_renal_replacement_therapy", "intermittent_renal_replacement_therapy"]}
for r in recs.values():
    r['equivalent_options'] = EQUIVALENT_OPTIONS.get(str(r['number']))

# ---- domains: read off the section headers inside TABLE 1 (journal p727-734) ----
# NOT derived from the Executive Summary counts — those disagree at the screening/infection
# boundary, and Table 1 is what a reader of the guideline actually sees.
T1 = json.load(open(OUT / "table1_findings.json", encoding="utf-8"))
RANGES = [(v[0], v[1], k) for k, v in T1["domain_boundaries"].items()
          if not k.startswith("_") and isinstance(v, list)]
SUB = {"additional_and_adjunctive_therapies": [(79, 86, "adjunctive_therapies"),
                                               (87, 99, "additional_supportive")]}
for n, r in recs.items():
    dom = next((k for a, b, k in RANGES if a <= n <= b), None)
    if dom in SUB:                        # 细分边界 Table 1 未给，标记为未核实
        r['domain'] = next((k for a, b, k in SUB[dom] if a <= n <= b), dom)
        r['domain_boundary_verified'] = False
    else:
        r['domain'] = dom
        r['domain_boundary_verified'] = True
    r['supersedes_2021']       = n in T1["has_2021_statement"]
    r['change_from_2021']      = ("upgraded"   if n in T1["strength_upgraded_from_2021"]
                             else "downgraded" if n in T1["strength_downgraded_from_2021"] else None)
    r['multi_grading']         = T1["multi_grading_statements"].get(str(n))

rows = [recs[n] for n in sorted(recs)]

# ---- QC vs the numbers the guideline reports about itself ----
st = Counter(r['strength'] for r in rows)
strong = st['strong_for'] + st['strong_against']
cond = st['conditional_for'] + st['conditional_against']
OFFICIAL = {"total": 129, "strong": 17, "conditional": 81, "good_practice": 19, "insufficient": 11}
qc = [("总条数", len(rows), OFFICIAL['total']),
      ("强推荐", strong, OFFICIAL['strong']),
      ("条件推荐", cond, OFFICIAL['conditional']),
      ("good practice", st['good_practice'], OFFICIAL['good_practice']),
      ("证据不足", st['insufficient_evidence'], OFFICIAL['insufficient'])]

# ---- remark analysis: does the remark carry a directive the statement does not? ----
DIRECT = re.compile(r'\b(should|must|avoid|use|using|administer|titrate|calculat|consider|'
                    r'give|start|stop|monitor|reassess|perform|obtain|target)\w*\b', re.I)
with_remark = [r for r in rows if r['remarks']]
directive = []
for r in with_remark:
    rm = " ".join(r['remarks'])
    if DIRECT.search(rm):
        # crude novelty test: a directive verb in the remark absent from the statement
        rv = {m.group(0).lower()[:6] for m in DIRECT.finditer(rm)}
        sv = {m.group(0).lower()[:6] for m in DIRECT.finditer(r['statement'])}
        if rv - sv:
            directive.append((r, sorted(rv - sv)))
for r in rows:
    r['remark_adds_directive'] = any(x[0]['number'] == r['number'] for x in directive)

# ---- outputs ----
json.dump(rows, open(OUT / "ssc2026_statements.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(OUT / "ssc2026_statements.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "number", "domain", "strength", "certainty", "certainty_by_comparator",
                "type", "equivalent_options", "supersedes_2021", "change_from_2021", "journal_page", "statement",
                "remarks", "remark_adds_directive", "source", "review_flags"])
    for r in rows:
        w.writerow([r['id'], r['number'], r['domain'], r['strength'], r['certainty'],
                    json.dumps(r['certainty_by_comparator'], ensure_ascii=False) if r['certainty_by_comparator'] else "",
                    r['type'], "|".join(r['equivalent_options']) if r['equivalent_options'] else "",
                    r['supersedes_2021'], r['change_from_2021'] or "",
                    r['journal_page'], r['statement'], " || ".join(r['remarks']),
                    r['remark_adds_directive'], r['source'], ",".join(r['review_flags'])])

if __name__ == "__main__":
    print(f"总计 {len(rows)} 条  (自动 {sum(1 for r in rows if r['source']=='auto_parsed')} "
          f"+ 手工 {sum(1 for r in rows if r['source']!='auto_parsed')})\n")
    print(f"{'核对项':<16}{'解析':>6}{'官方':>6}   {'':<4}")
    for name, mine, off in qc:
        print(f"{name:<16}{mine:>6}{off:>6}   {'✓' if mine == off else '差 %+d' % (mine - off)}")
    print("\n按 domain（边界取自 Table 1 的分区标题）：")
    from collections import Counter as C
    for k, v in C(r['domain'] for r in rows).items():
        vf = all(r['domain_boundary_verified'] for r in rows if r['domain'] == k)
        print(f"  {k:<38}{v:>4}   {'Table 1 已核实' if vf else '内部边界未核实'}")
    print(f"\n等价选项条款      {sum(1 for r in rows if r['equivalent_options'])}   ← 无方向，走 EQUIVALENT_OPTIONS 边")
    print(f"带 2021 对照条款  {sum(1 for r in rows if r['supersedes_2021'])}"
          f"   强度升级 {sum(1 for r in rows if r['change_from_2021']=='upgraded')}"
          f"   降级 {sum(1 for r in rows if r['change_from_2021']=='downgraded')}")
    print(f"\n带 remark 的条数           : {len(with_remark)}  ({len(with_remark)/len(rows)*100:.0f}%)")
    print(f"remark 里含正文没有的祈使动作: {len(directive)}  ({len(directive)/len(rows)*100:.0f}%)")
    print(f"需人工复核（自动标记）      : {sum(1 for r in rows if r['review_flags'])}")
    print("\n示例 —— remark 携带独立动作：")
    for r, extra in directive[:4]:
        print(f"  [{r['id']}] {r['statement'][:88]}…")
        print(f"      remark 独有动作 {extra}: {r['remarks'][0][:110]}…")
