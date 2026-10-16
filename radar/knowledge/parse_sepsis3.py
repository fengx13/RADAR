#!/usr/bin/env python3
"""Sepsis-3 as citable propositions.

Why this file exists — from the 2026-08-07 gpt-5-mini run:

    R2 asks "is there new organ dysfunction attributable to the infection?"
    Its scope was stmt_005 / 008 / 009 / 051 + table_03. NONE of those say what
    organ dysfunction IS. The only citable material was about ORDERING a lactate,
    so the model cited stmt_008 ("we suggest measuring lactate") as supporting
    evidence for "no organ dysfunction" — because no lactate had been measured.
    It answered `no` for both true-sepsis cases.

    The model did not break a rule. We handed it a question with no yardstick.
    `SOFA >= 2` was not in the pool. This file puts it there.

Source of every quote below:
    Singer M, Deutschman CS, Seymour CW, et al. The Third International Consensus
    Definitions for Sepsis and Septic Shock (Sepsis-3). JAMA. 2016;315(8):801-810.
    PMID 26903338.  (Boxes 3 and 4, Table 1, and the narrative on pp. 805-807.)
    Seymour CW, Liu VX, Iwashyna TJ, et al. Assessment of Clinical Criteria for
    Sepsis. JAMA. 2016;315(8):762-774.  PMID 26903335.

Text fidelity: quotes are copied from the PDF text layer of the PMC author
manuscript. Two OCR artefacts were repaired and are listed in `normalisations`
so the change is auditable — nothing else was touched. Every unit is stored
exactly as it will be injected, so the V3 verbatim check compares like with like.
"""
from radar import paths
import json, pathlib

HERE = pathlib.Path(__file__).parent
OUT = paths.PARSED / "sepsis3_propositions.json"

SINGER = {"citation": "Singer M, et al. JAMA. 2016;315(8):801-810", "pmid": "26903338"}
SEYMOUR = {"citation": "Seymour CW, et al. JAMA. 2016;315(8):762-774", "pmid": "26903335"}

NORMALISATIONS = [
    {"where": "sepsis3::sofa_threshold_mortality",
     "from": "ASOFA score", "to": "A SOFA score",
     "why": "PDF text layer lost the space after the leading article"},
    {"where": "sepsis3::septic_shock_criteria",
     "from": "(18mg/dL)", "to": "(18 mg/dL)",
     "why": "PDF text layer lost the space before the unit"},
]

# The SOFA table is rendered as flat text because that is what gets injected and
# therefore what the verbatim check must match. Column order is the published one.
SOFA_TABLE = (
    "Sequential [Sepsis-Related] Organ Failure Assessment Score. "
    "Respiration, PaO2/FIO2, mm Hg (kPa): 0 = >=400 (53.3); 1 = <400 (53.3); "
    "2 = <300 (40); 3 = <200 (26.7) with respiratory support; "
    "4 = <100 (13.3) with respiratory support. "
    "Coagulation, Platelets, x10^3/uL: 0 = >=150; 1 = <150; 2 = <100; 3 = <50; 4 = <20. "
    "Liver, Bilirubin, mg/dL (umol/L): 0 = <1.2 (20); 1 = 1.2-1.9 (20-32); "
    "2 = 2.0-5.9 (33-101); 3 = 6.0-11.9 (102-204); 4 = >12.0 (204). "
    "Cardiovascular: 0 = MAP >=70 mm Hg; 1 = MAP <70 mm Hg; "
    "2 = Dopamine <5 or dobutamine (any dose); "
    "3 = Dopamine 5.1-15 or epinephrine <=0.1 or norepinephrine <=0.1; "
    "4 = Dopamine >15 or epinephrine >0.1 or norepinephrine >0.1. "
    "Central nervous system, Glasgow Coma Scale score: 0 = 15; 1 = 13-14; "
    "2 = 10-12; 3 = 6-9; 4 = <6. "
    "Renal, Creatinine, mg/dL (umol/L): 0 = <1.2 (110); 1 = 1.2-1.9 (110-170); "
    "2 = 2.0-3.4 (171-299); 3 = 3.5-4.9 (300-440); 4 = >5.0 (440); "
    "Urine output, mL/d: 3 = <500; 4 = <200. "
    "Catecholamine doses are given as ug/kg/min for at least 1 hour. "
    "Glasgow Coma Scale scores range from 3-15; higher score indicates better "
    "neurological function."
)

P = []
def p(pid, verbatim, *, src, where, answers, note=None, remarks=(), derived=None):
    """answers = which decision-node questions this unit can actually adjudicate."""
    P.append({"id": f"sepsis3::{pid}", "statement": verbatim,
              "remarks": list(remarks), "strength": "consensus definition",
              "certainty": None, "source": src["citation"], "pmid": src["pmid"],
              "located_in": where, "adjudicates": list(answers),
              **({"note": note} if note else {}),
              **({"machine_readable": derived} if derived else {})})


p("definition_sepsis",
  "Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated "
  "host response to infection.",
  src=SINGER, where="Box 3", answers=["R1", "R2"],
  note="the operative clause for us is 'organ dysfunction' — R2's whole question")

p("organ_dysfunction_criterion",
  "Organ dysfunction can be identified as an acute change in total SOFA score >=2 "
  "points consequent to the infection.",
  src=SINGER, where="Box 3", answers=["R2"],
  remarks=["The baseline SOFA score can be assumed to be zero in patients not known "
           "to have preexisting organ dysfunction."],
  derived={"criterion": "delta_total_SOFA >= 2", "baseline_default": 0,
           "requires": "attribution to the infection"},
  note="THIS is the yardstick R2 was missing on 2026-08-07")

p("baseline_sofa",
  "The baseline SOFA score should be assumed to be zero unless the patient is known "
  "to have preexisting (acute or chronic) organ dysfunction before the onset of infection.",
  src=SINGER, where="narrative, p.806", answers=["R2"],
  note="matters for our S1b-type patients: chronic CHF/COPD does NOT get counted as "
       "the acute change")

p("sofa_threshold_mortality",
  "A SOFA score >=2 reflects an overall mortality risk of approximately 10% in a "
  "general hospital population with suspected infection. Even patients presenting "
  "with modest dysfunction can deteriorate further, emphasizing the seriousness of "
  "this condition and the need for prompt and appropriate intervention, if not "
  "already being instituted.",
  src=SINGER, where="Box 3", answers=["R2"],
  note="the second sentence is the one that argues against reading a low score as reassurance")

p("sofa_table",
  SOFA_TABLE, src=SINGER, where="Table 1", answers=["R2"],
  derived={"systems": ["respiration", "coagulation", "liver", "cardiovascular",
                       "cns", "renal"], "points_each": [0, 1, 2, 3, 4]},
  note="six systems; R2 can now check the patient against each one instead of "
       "reasoning from whether a lactate was ordered")

p("qsofa_criteria",
  "Patients with suspected infection who are likely to have a prolonged ICU stay or "
  "to die in the hospital can be promptly identified at the bedside with qSOFA, ie, "
  "alteration in mental status, systolic blood pressure <=100 mm Hg, or respiratory "
  "rate >=22/min.",
  src=SINGER, where="Box 3 / Box 4", answers=["R1"],
  derived={"criteria": ["altered_mentation", "sbp_le_100", "resp_rate_ge_22"],
           "positive_at": 2},
  note="all three are available at triage — unlike SOFA, which needs labs")

p("neither_is_standalone",
  "Neither qSOFA nor SOFA is intended to be a stand-alone definition of sepsis. It is "
  "crucial, however, that failure to meet 2 or more qSOFA or SOFA criteria should not "
  "lead to a deferral of investigation or treatment of infection or to a delay in any "
  "other aspect of care deemed necessary by the practitioners.",
  src=SINGER, where="narrative, p.807", answers=["R1", "R2", "R3a", "R3b"],
  note="the direct antidote to the failure mode we observed — a negative or "
       "unmeasurable score is explicitly NOT grounds for ruling sepsis out")

p("septic_shock_concept",
  "Septic shock is a subset of sepsis in which underlying circulatory and "
  "cellular/metabolic abnormalities are profound enough to substantially increase "
  "mortality.",
  src=SINGER, where="Box 3", answers=["R4"])

p("septic_shock_criteria",
  "Patients with septic shock can be identified with a clinical construct of sepsis "
  "with persisting hypotension requiring vasopressors to maintain MAP >=65 mm Hg and "
  "having a serum lactate level >2 mmol/L (18 mg/dL) despite adequate volume "
  "resuscitation. With these criteria, hospital mortality is in excess of 40%.",
  src=SINGER, where="Box 3", answers=["R4"],
  derived={"all_of": ["sepsis", "persisting_hypotension_on_vasopressors_for_MAP_ge_65",
                      "lactate_gt_2_mmol_L", "despite_adequate_volume_resuscitation"]},
  note="note 'despite adequate volume resuscitation' — at triage this is usually "
       "not yet establishable, which is itself the honest answer for R4")

p("qsofa_validity_outside_icu",
  "Among non-ICU encounters in the validation cohort (n = 66 522 with suspected "
  "infection, of whom 1886 [3%] died), qSOFA had predictive validity (AUROC = 0.81; "
  "95% CI, 0.80-0.82) that was greater than SOFA (AUROC = 0.79; 95% CI, 0.78-0.80; "
  "P < .001) and SIRS (AUROC = 0.76; 95% CI, 0.75-0.77; P < .001).",
  src=SEYMOUR, where="Abstract / Results", answers=["R1"],
  note="our setting is the ED — this is the number that applies, not the ICU one")

p("qsofa_validity_in_icu",
  "In the ICU, the predictive validity for hospital mortality of qSOFA above baseline "
  "risk (AUROC = 0.66; 95% CI, 0.64-0.68) was statistically greater than SIRS criteria "
  "(P = .01) but significantly less than SOFA (P < .001).",
  src=SEYMOUR, where="Results", answers=["R1"],
  note="kept alongside the ED number so the contrast is citable rather than asserted")


if __name__ == "__main__":
    doc = {"meta": {
        "built": "2026-08-07",
        "why": "R2 had no definition of organ dysfunction in its citation pool; the "
               "gpt-5-mini run answered 'no organ dysfunction' for both true-sepsis "
               "cases by citing a statement about ordering lactate",
        "sources": [SINGER, SEYMOUR],
        "normalisations": NORMALISATIONS,
        "n": len(P)}, "propositions": P}
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1))

    print(f"Sepsis-3 → {len(P)} 条可引用单元  -> {OUT.name}\n")
    by_node = {}
    for u in P:
        for n in u["adjudicates"]: by_node.setdefault(n, []).append(u["id"])
    for n in ("R1", "R2", "R3a", "R3b", "R4"):
        ids = by_node.get(n, [])
        print(f"  {n:<4} {len(ids)} 条")
        for i in ids: print(f"        {i}")
    print(f"\n  文本修正 {len(NORMALISATIONS)} 处（已记录在 meta.normalisations）：")
    for x in NORMALISATIONS:
        print(f"     {x['where']}:  {x['from']!r} → {x['to']!r}   {x['why']}")
    longest = max(P, key=lambda u: len(u["statement"]))
    print(f"\n  最长单元 {longest['id']}  {len(longest['statement'])} 字符")
