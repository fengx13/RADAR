#!/usr/bin/env python3
"""
Candidate Agent 3 nodes:
  TX1_antibiotic_timing
  TX2_fluids
  TX3_vasopressors
  TX4_map_target
"""

from __future__ import annotations
from radar import paths

import json
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------

ROOT = paths.ROOT
GRAPH_PATH = paths.GRAPH / "ssc2026_graph.json"
STATEMENTS_PATH = paths.PARSED / "ssc2026_statements.json"


G = json.loads(
    GRAPH_PATH.read_text(encoding="utf-8")
)

STATEMENTS = {
    s["id"]: s
    for s in json.loads(
        STATEMENTS_PATH.read_text(encoding="utf-8")
    )
}


# ---------------------------------------------------------------------
# graph indexes
# ---------------------------------------------------------------------

edges = G["edges"]

by_rel = defaultdict(list)

for e in edges:
    by_rel[e["rel"]].append(e)


# IS_DISJUNCTION_OF
disj_down = defaultdict(set)
disj_up = defaultdict(set)

for e in by_rel["IS_DISJUNCTION_OF"]:
    disj_down[e["src"]].add(e["dst"])
    disj_up[e["dst"]].add(e["src"])


# APPLIES_WHEN
applies = defaultdict(set)

for e in by_rel["APPLIES_WHEN"]:
    applies[e["dst"]].add(e["src"])


# statement -> touched topics
touches = defaultdict(set)

for e in edges:
    if e["rel"] in (
        "RECOMMENDS",
        "PREFERRED_OVER",
        "EQUIVALENT_OPTIONS",
        "TARGETS",
        "CONSTRAINS",
        "REFINED_BY",
    ):
        touches[e["src"]].add(e["dst"])


# ---------------------------------------------------------------------
# simple exploratory scope
# ---------------------------------------------------------------------

def scope(condition_seeds, topic_seeds):
    """
    Exploratory scope:
      1. widen conditions through IS_DISJUNCTION_OF
      2. find statements via APPLIES_WHEN
      3. intersect with topic matches

    Intentionally does NOT apply production expansion rules such as:
      - OVERRIDDEN_BY
      - opposite polarity partners
      - MUST_PRECEDE temporal partners

    We first want to see the direct candidate scope.
    """

    wide_conditions = set(condition_seeds)

    for c in list(condition_seeds):
        wide_conditions |= disj_down.get(c, set())
        wide_conditions |= disj_up.get(c, set())

    state_matches = set()

    for c in wide_conditions:
        state_matches |= applies.get(c, set())

    topic_set = set(topic_seeds)

    topic_matches = {
        sid
        for sid in state_matches
        if touches[sid] & topic_set
    }

    return sorted(topic_matches)


# ---------------------------------------------------------------------
# Agent 3 candidate seeds
# ---------------------------------------------------------------------

T1_CONDITIONS = [
    "possible_sepsis",
    "probable_sepsis",
    "definite_sepsis",
    "septic_shock",
    "possible_septic_shock",
    "no_shock",
    "low_likelihood_of_infection",
]

T1_TOPICS = [
    "administer_antimicrobial_therapy",
    "perform_time_limited_rapid_investigation",
    "monitor_patient_closely",
]


T2_CONDITIONS = [
    "sepsis_induced_hypoperfusion",
    "sepsis_induced_hypotension",
    "sepsis",
    "septic_shock",
    "persistent_hypoperfusion",
]

T2_TOPICS = [
    "administer_iv_crystalloid",
    "administer_iv_crystalloid_bolus",
    "use_crystalloids_as_first_line_fluid",
    "use_balanced_crystalloids",
    "use_crystalloids_alone",
    "use_starches_for_resuscitation",
    "use_gelatin_for_resuscitation",
    "liberal_fluid_resuscitation_strategy",
    "restrictive_fluid_resuscitation_strategy",
    "reassess_fluid_status_frequently",
    "calculate_dose_on_actual_body_weight",
    "use_dynamic_measures_to_guide_fluid_resuscitation",
]


T3_CONDITIONS = [
    "septic_shock",
    "sepsis_induced_hypotension",
    "escalating_norepinephrine_dose",
    "cardiac_dysfunction",
    "persistent_hypoperfusion",
]

T3_TOPICS = [
    "start_vasopressor",
    "start_vasopressor_peripherally",
    "use_norepinephrine_as_first_line_vasopressor",
    "add_vasopressin",
    "add_epinephrine",
    "use_terlipressin",
    "use_inotropes",
    "use_levosimendan",
    "add_dobutamine_to_norepinephrine",
    "epinephrine_alone",
]


T4_CONDITIONS = [
    "septic_shock",
    "sepsis_induced_hypotension",
]

T4_TOPICS = [
    "target::MAP",
    "titrate_vasopressor_to_maintain_MAP",
]


# ---------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------

def print_scope(name, conditions, topics):
    ids = scope(
        conditions,
        topics,
    )

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print("\nCondition seeds:")
    for x in conditions:
        print(f"  - {x}")

    print("\nTopic seeds:")
    for x in topics:
        print(f"  - {x}")

    print(f"\nMatched statements: {len(ids)}\n")

    if not ids:
        print("  NO DIRECT MATCHES")
        return

    for sid in ids:
        s = STATEMENTS.get(sid, {})

        print(f"{sid}")
        print(f"  {s.get('statement', '<missing statement text>')}")

        if s.get("strength"):
            print(
                f"  strength: {s.get('strength')}"
            )

        if s.get("certainty"):
            print(
                f"  certainty: {s.get('certainty')}"
            )

        print()


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "Agent 3 direct scope exploration\n"
        "READ-ONLY: no project files will be modified."
    )

    print_scope(
        "TX1_antibiotic_timing",
        T1_CONDITIONS,
        T1_TOPICS,
    )

    print_scope(
        "TX2_fluids",
        T2_CONDITIONS,
        T2_TOPICS,
    )

    print_scope(
        "TX3_vasopressors",
        T3_CONDITIONS,
        T3_TOPICS,
    )

    print_scope(
        "TX4_map_target",
        T4_CONDITIONS,
        T4_TOPICS,
    )
