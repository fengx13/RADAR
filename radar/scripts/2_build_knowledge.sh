#!/usr/bin/env bash
# Step 2 - guideline knowledge layer: knowledge/parsed and knowledge/graph.
# The parsed statements and graphs in this repository are frozen and protected by
# knowledge/policy_snapshot.json. By default this script only CHECKS them.
# Pass --rebuild to regenerate from the guideline PDFs in knowledge/sources/
# (then review the diff and refresh the snapshot deliberately).
set -euo pipefail
cd "$(dirname "$0")/../.."
run() { echo "+ python -m $*"; python -m "$@"; }

if [[ "${1:-}" == "--rebuild" ]]; then
  run radar.knowledge.parse_ssc2026          # SSC 2026 -> 129 citable statements
  run radar.knowledge.build_ssc_table --force
  run radar.knowledge.parse_sepsis3          # Sepsis-3 definitions
  run radar.knowledge.extract_triples        # statements -> triples
  run radar.knowledge.build_graph            # triples -> guideline graph
  run radar.knowledge.make_scope_tables      # which statements each node may cite
fi
run radar.knowledge.check_agent2_scopes
run radar.knowledge.check_agent3_scopes
run radar.knowledge.audit_agent1_graph
