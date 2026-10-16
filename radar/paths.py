"""Where RADAR finds its knowledge files, patient inputs and outputs.

Patient-level data never live in this repository. Three locations are
configurable through environment variables:

    RADAR_DATA    patient inputs (default: ../RADAR_input next to the repository),
                  organised by database: MIMIC/ and Stanford/
    RADAR_OUTPUT  run outputs (default: ../RADAR_output next to the repo)
    RADAR_RAW     the lab data root holding raw/ and master/
                  (default: ../../00_Data, i.e. XLab/00_Data)

``DATA / name`` resolves a file name the way the original flat 04_Data
folder did, so the code can keep writing ``D / "cohort100_cases.json"``:

* cohort inputs (cases, case ids, identifiers, meta, longitudinal inputs,
  states, caches) go to RADAR_DATA/MIMIC or RADAR_DATA/Stanford;
* synthetic test fixtures shipped in tests/fixtures are found there;
* everything else (runs, scores, evaluations) goes to RADAR_OUTPUT.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent          # the installed radar package
ROOT = PKG.parent                               # repository root (or site-packages)
KNOWLEDGE = PKG / "knowledge"
PARSED = KNOWLEDGE / "parsed"
GRAPH = KNOWLEDGE / "graph"
SOURCES = KNOWLEDGE / "sources"          # guideline PDFs, local only (git-ignored)
FIXTURES = PKG / "tests" / "fixtures"

INPUT_DIR = Path(os.environ.get("RADAR_DATA", ROOT.parent / "RADAR_input")).expanduser()
OUTPUT_DIR = Path(os.environ.get("RADAR_OUTPUT", ROOT.parent / "RADAR_output")).expanduser()
RAW_ROOT = Path(os.environ.get("RADAR_RAW", ROOT.parent.parent / "00_Data")).expanduser()

_SITE = (("cohort100_", "MIMIC"), ("_cache_cohort100_", "MIMIC"), ("phase0_", "MIMIC"),
         ("stanford100_", "Stanford"), ("_cache_stanford100_", "Stanford"))
_INPUT = re.compile(r"(^_cache_|_cases(_meta|_retired)?\.json$|_case_ids\.json$|_identifiers(_retired)?\.json$"
                    r"|_longitudinal_cases(_v\d+)?\.json$|_t1_states\.json$|_tsofa_states\.json$)")


def _site(name: str) -> str | None:
    for prefix, site in _SITE:
        if name.startswith(prefix):
            return site
    return None


class _DataDir:
    """``DATA / "file.json"`` -> the Path that file has in the split layout."""

    def __truediv__(self, name) -> Path:
        name = str(name)
        site = _site(name)
        if site and _INPUT.search(name):
            return INPUT_DIR / site / name
        fixture = FIXTURES / name
        if fixture.exists():
            return fixture
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR / name

    def __fspath__(self) -> str:
        return str(OUTPUT_DIR)

    __str__ = __fspath__

    def __repr__(self) -> str:
        return f"DataDir(input={INPUT_DIR}, output={OUTPUT_DIR})"


DATA = _DataDir()
