from radar import paths
import json
from pathlib import Path
from collections import defaultdict

ROOT = paths.ROOT
GRAPH_PATH = paths.GRAPH / "ssc2026_graph.json"
STATEMENTS_PATH = paths.PARSED / "ssc2026_statements.json"

# Read only — nothing is written anywhere
G = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
STATEMENTS = json.loads(STATEMENTS_PATH.read_text(encoding="utf-8"))

STMTS = {s["id"]: s for s in STATEMENTS}
NODES = {n["id"]: n for n in G["nodes"]}
EDGES = G["edges"]

by_rel = defaultdict(list)

for edge in EDGES:
    by_rel[edge["rel"]].append(edge)


# ---------------------------------------------------------
# Build graph indexes
# ---------------------------------------------------------

# compound condition <-> component conditions
disj_down = defaultdict(set)
disj_up = defaultdict(set)

for edge in by_rel["IS_DISJUNCTION_OF"]:
    disj_down[edge["src"]].add(edge["dst"])
    disj_up[edge["dst"]].add(edge["src"])


# condition -> statements
applies = defaultdict(set)

for edge in by_rel["APPLIES_WHEN"]:
    applies[edge["dst"]].add(edge["src"])


# statement -> topics/actions/constraints
touches = defaultdict(set)

TOPIC_RELATIONS = {
    "RECOMMENDS",
    "PREFERRED_OVER",
    "EQUIVALENT_OPTIONS",
    "TARGETS",
    "CONSTRAINS",
    "REFINED_BY",
}

for edge in EDGES:
    if edge["rel"] in TOPIC_RELATIONS:
        touches[edge["src"]].add(edge["dst"])


# ---------------------------------------------------------
# Simple scope function
# ---------------------------------------------------------

def get_scope(condition_seeds, topic_seeds):

    # Expand compound/disjunction conditions
    wide_conditions = set(condition_seeds)

    for condition in list(condition_seeds):
        wide_conditions |= disj_down.get(condition, set())
        wide_conditions |= disj_up.get(condition, set())

    # Statements applicable to patient state
    state_matches = set()

    for condition in wide_conditions:
        state_matches |= applies.get(condition, set())

    # Keep only statements touching one of our desired topics
    topic_seeds = set(topic_seeds)

    matches = {
        sid
        for sid in state_matches
        if touches[sid] & topic_seeds
    }

    return sorted(
        matches,
        key=lambda sid: NODES[sid].get("number", 9999)
    )


def print_scope(name, conditions, topics):

    ids = get_scope(conditions, topics)

    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    print("\nCondition seeds:")
    for x in conditions:
        print("  -", x)

    print("\nTopic seeds:")
    for x in topics:
        print("  -", x)

    print("\nRetrieved statements:")

    if not ids:
        print("  NONE")

    for sid in ids:
        stmt = STMTS[sid]
        print(f"\n{sid} | Statement {stmt['number']}")
        print(stmt["statement"])


# ---------------------------------------------------------
# Candidate Agent 2 nodes
# ---------------------------------------------------------

W3_CONDITIONS = [
    "sepsis",
    "septic_shock",
]

W3_TOPICS = [
    "evaluate_for_source_requiring_emergent_source_control",
]


W4_CONDITIONS = [
    "possible_sepsis",
    "no_shock",
    "suspected_sepsis",
]

W4_TOPICS = [
    "assess_likelihood_infectious_vs_noninfectious",
    "search_for_alternative_diagnoses",
    "constraint::rule_in_or_out_sepsis_on_single_test",
]


print_scope(
    "W3_source_workup",
    W3_CONDITIONS,
    W3_TOPICS,
)

print_scope(
    "W4_alternative_diagnosis_workup",
    W4_CONDITIONS,
    W4_TOPICS,
)