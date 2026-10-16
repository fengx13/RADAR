# Knowledge layer

This folder holds what the agents cite: `parsed/` — the 129 SSC 2026 recommendation statements with their strength and certainty, the Sepsis-3 definitions, and SSC 2026 Table 3 (sepsis terminology) as structured JSON; `graph/` — the four decision graphs (A1–A4), the SSC 2026 triples and the frozen citable scope of every node. The Python modules here rebuild those files from the source PDFs (`2_build_knowledge.sh --rebuild`); without the PDFs the script only verifies the committed files.

The source PDFs are not distributed with this repository (publisher copyright). They are listed below so the knowledge layer can be rebuilt and every quote can be checked against its page.

## Sources

| Role | Reference | Where to get it |
|---|---|---|
| Primary guideline; all statement numbers and page citations follow this edition | Prescott HC et al. Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2026. *Crit Care Med* 2026;54:725–812. PMID 41869847 | journals.lww.com/ccmjournal; executive summary and the two infographics (antimicrobial timing, resuscitation quick guide) from sccm.org |
| Previous edition, used for version comparison and the knowledge-layer ablation | Evans L et al. Surviving Sepsis Campaign 2021. *Crit Care Med* 2021. PMID 34599691 | journals.lww.com/ccmjournal |
| Sepsis and septic shock definitions (A1 nodes R2, R4) | Singer M et al. The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). *JAMA* 2016;315:801–810. PMID 26903338 | jamanetwork.com |
| Operational criteria for suspected infection and SOFA/qSOFA (A1 nodes R1, R2) | Seymour CW et al. Assessment of Clinical Criteria for Sepsis. *JAMA* 2016;315:762–774. PMID 26903335 | jamanetwork.com |

### Mimic guidelines (cited by A1 as opposing evidence)

| Condition | Reference |
|---|---|
| Pulmonary embolism | 2026 AHA/ACC guideline. *Circulation* 2026;153:e977–e1051 |
| Acute heart failure | 2022 AHA/ACC/HFSA heart failure guideline. *Circulation* 2022;145:e895–e1032 |
| Acute coronary syndrome | 2025 ACC/AHA guideline. *Circulation* 2025;151:e771–e862 |
| Heart failure versus sepsis | Jones AE et al. 2021, review of sepsis–heart failure confounding |
| Acute kidney injury | KDIGO 2012 AKI guideline (kdigo.org/guidelines/acute-kidney-injury) — not yet parsed |

### Reporting standards (for the manuscript, not cited by the agents)

STARD 2015 (*BMJ*), STARD-AI 2025 (*Nat Med*), TRIPOD+AI 2024 (*BMJ*), TRIPOD-LLM 2025 (*Nat Med*).

## Files

| File | Content |
|---|---|
| `parsed/ssc2026_statements.json` | 129 statements: id, number, verbatim text, strength, certainty, remarks, PDF page |
| `parsed/sepsis3_propositions.json` | Sepsis-3 definitions and criteria as citable propositions |
| `parsed/table3_terminology.json` | SSC 2026 Table 3: definite / probable / possible / unlikely sepsis |
| `parsed/table1_findings.json`, `parsed/fig_abx_timing.json` | Table 1 and the antimicrobial-timing infographic |
| `graph/agent{1,2,3,4}_*_graph.json` | Decision graphs: nodes, questions, allowed answers, citable scope per node |
| `graph/ssc2026_graph.json`, `graph/triples_*.json` | Guideline knowledge graph and the extracted triples |
| `graph/scope_tables.json` | Frozen citable scope of every node |
| `graph/ablation_report.txt` | Knowledge-layer retrieval ablation |
| `*_校验表.html` | Review tables used when the statements and the graph were checked by hand |
| `policy_snapshot.json` | Byte hashes of the knowledge files and clinical-policy code; `policy_integrity.check_snapshot` compares against it |
