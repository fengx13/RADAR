#!/usr/bin/env bash
# Step 4 - score one or more runs per site against Sepsis-3 truth and observed care.
# Usage:  bash radar/scripts/4_score.sh <run_tag> [more run tags]   e.g. bash radar/scripts/4_score.sh v3_r1
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${RADAR_OUTPUT:-../RADAR_output}
TAGS=("$@"); [ ${#TAGS[@]} -gt 0 ] || { echo "usage: bash radar/scripts/4_score.sh <run_tag> ..."; exit 1; }
LABEL=$(IFS=_; echo "${TAGS[*]}")
for SITE in mimic mcmed; do
  P=$([ "$SITE" = mimic ] && echo cohort100 || echo stanford100)
  RUNS=(); for t in "${TAGS[@]}"; do RUNS+=("$OUT/${P}_vnext_run_${t}.json"); done
  python -m radar.evaluation.score_vnext --site "$SITE" --runs "${RUNS[@]}" \
     --out "results/scores_vnext_${SITE}_${LABEL}.json"
done
