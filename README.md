# RADAR

Guideline-grounded multi-agent decision support for the emergency-department sepsis pathway.

▶ **Interactive demo: https://fengx13.github.io/RADAR/** — three synthetic patients, four agents, replayed over six hours; switch models in the top-right corner.

**What it is.** Four agents (Recognize → Assay → Act → Route) answer fixed questions from frozen decision graphs built from SSC 2026 and Sepsis-3; code walks the graphs, binds verbatim guideline quotes and validates every reply. The model never sees future data: each checkpoint (triage, 0.5 / 1 / 3 / 6 h) is replayed with only what was available then.

**Install** `pip install -e .` (Python ≥ 3.10) · **Run the tests** `pytest` (synthetic fixtures only, no API calls) · **Run on your own cases** `radar/scripts/1_build_input.sh` … `4_score.sh`; model key in `.env` as `OPENAI_API_KEY`; inputs and outputs stay outside the repository (`radar/paths.py`).

**Data.** No patient data are included. The evaluation used MIMIC-IV-ED and MC-MED under PhysioNet credentialed access; apply there to reproduce.

**Demo results.** Claude Sonnet 5.5 and GPT-6 Luna on 108 synthetic scenarios written for the demo (not real patients, not a clinical validation).

**Team** Yanwei (Yana) Jin, Xiaodong (Sheldon) Huang, Hema Chandu · Dr. Feng Xie's Lab · **Not for clinical use.**
