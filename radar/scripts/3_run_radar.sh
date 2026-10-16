#!/usr/bin/env bash
# Step 3 - replay both sites through T0 and checkpoints C0.5 / C1 / C3 / C6.
# Usage:  bash radar/scripts/3_run_radar.sh <model> <run_tag> [extra args]
#   CASES_SUFFIX=_v5 bash radar/scripts/3_run_radar.sh gpt-6-luna v5_r1
#   bash radar/scripts/3_run_radar.sh gpt-6-luna v3_r1            # base inputs (the v3 run)
#   ... --resume                                           # continue an interrupted run
# Needs OPENAI_API_KEY in the environment or in .env at the repository root.
set -euo pipefail
MODEL=${1:?model}; TAG=${2:?run_tag}; shift 2
cd "$(dirname "$0")/../.."
[ -f .env ] && export $(grep -v '^#' .env | xargs)
OUT=${RADAR_OUTPUT:-../RADAR_output}; mkdir -p "$OUT"
for SITE in cohort100 stanford100; do
  EXTRA=""
  # MIMIC-IV-ED records no oxygen therapy; declare it so Route is not blocked on it (PY-128).
  [ "$SITE" = cohort100 ] && EXTRA="--deployment-unavailable patient_state.oxygen_requirement"
  python -u -m radar.pipeline.run_pipeline_vnext \
     --infile "$(python -c "from radar import paths; print(paths.DATA / '${SITE}_longitudinal_cases${CASES_SUFFIX:-}.json')")" \
     $EXTRA --out "$OUT/${SITE}_vnext_run_${TAG}.json" --run-tag "$TAG" \
     --model "$MODEL" --temperature 0 --seed 7 --replay-mode full --continue-on-failure \
     --workers "${WORKERS:-6}" --batch-size 10 "$@" 2>&1 | tee -a "$OUT/_log_${SITE}_${TAG}.txt"
done
