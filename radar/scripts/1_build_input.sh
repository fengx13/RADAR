#!/usr/bin/env bash
# Step 1 - raw MIMIC-IV-ED / MC-MED tables -> RADAR_input/{MIMIC,Stanford}
# Needs the lab data root (RADAR_RAW, default XLab/00_Data) with raw/ and master/.
# Order follows each script's documented inputs; the MC-MED oxygen flag is the
# one recorded in stanford100_longitudinal_cases_v5.json (no_oxygen_order_means_none: true).
set -euo pipefail
cd "$(dirname "$0")/../.."
run() { echo "+ python -m $*"; python -m "$@"; }

# MIMIC-IV-ED (BIDMC)
run radar.cohort.make_phase0_cases                    # 8-case pilot, carried into the 100
run radar.cohort.make_cohort100 --source mimic        # stratified 4 x 25 at T0
run radar.cohort.deidentify_cohort --cases cohort100_cases.json --crosswalk cohort100_identifiers.json
run radar.cohort.make_longitudinal_cases --site mimic # results and vitals over 6 h
run radar.cohort.replace_no_result_cases --site mimic # inclusion: >=1 ED lab result within 6 h
run radar.cohort.make_observed_treatment --site mimic
run radar.cohort.make_infection_results --site mimic  # -> _v4
run radar.cohort.make_full_record --site mimic --build-cache   # -> _v5

# MC-MED (Stanford), external validation
run radar.cohort.make_cohort100 --source stanford
run radar.cohort.deidentify_cohort --cases stanford100_cases.json --crosswalk stanford100_identifiers.json --summary stanford100_summary.json
run radar.cohort.make_longitudinal_cases --site mcmed
run radar.cohort.replace_no_result_cases --site mcmed
run radar.cohort.make_observed_treatment --site mcmed --no-oxygen-order-means-none
run radar.cohort.make_infection_results --site mcmed
run radar.cohort.make_full_record --site mcmed --build-cache
