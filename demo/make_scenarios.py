"""Generate the synthetic cases behind the interactive RADAR demo.

Every value here was written by Bunny (Yana's AI assistant) for the demo.
None of it comes from a real patient, so the output may be committed and
published. 3 patients x 4 checkpoints x 3 lactate x 3 MAP = 108 cases.

Each case carries one batch of results that comes back just before its
target checkpoint, so earlier checkpoints have nothing new and are skipped:
the page shows "the state as of that checkpoint", not a path through the
earlier ones.

    python demo/make_scenarios.py            # writes demo/scenarios/demo_cases.json
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "tests" / "fixtures" / "pipeline_synthetic_cases.json"
OUT = HERE / "scenarios" / "demo_cases.json"

CHECKPOINTS = {"C0_5": 0.5, "C1": 1.0, "C3": 3.0, "C6": 6.0}
LACTATE = {"L1": 1.2, "L2": 2.8, "L3": 4.8}          # mmol/L; SSC thresholds 2 and 4
MAP = {"M1": 78, "M2": 66, "M3": 58}                 # mmHg; hypotension threshold 65
LAB_UNITS = {"wbc": "K/uL", "hemoglobin": "g/dL", "platelets": "K/uL", "creatinine": "mg/dL",
             "bun": "mg/dL", "bilirubin_total": "mg/dL", "glucose": "mg/dL", "sodium": "mEq/L",
             "potassium": "mEq/L", "bicarbonate": "mEq/L", "anion_gap": "mEq/L"}

PATIENTS = {
    "P1": {
        "label": {"en": "Typical infection", "zh": "典型感染"},
        "t0": {"chief_complaint": "Fever and productive cough for 2 days", "age": 67, "sex": "F",
               "arrival_transport": "WALK IN", "nurse_esi": 2,
               "comorbidities": ["type 2 diabetes", "COPD"],
               "triage_vitals": {"temperature_C": 38.6, "heart_rate": 112, "resp_rate": 24, "spo2": 93,
                                 "sbp": 104, "dbp": 62, "pain_0_10": 3}},
        "oxygen": True,
        "labs": {"wbc": 16.2, "hemoglobin": 12.8, "platelets": 210, "creatinine": 1.3, "bun": 24,
                 "bilirubin_total": 0.8, "glucose": 212, "sodium": 134, "potassium": 4.2,
                 "bicarbonate": 21, "anion_gap": 15},
    },
    "P2": {
        "label": {"en": "Atypical presentation", "zh": "不典型表现"},
        "t0": {"chief_complaint": "Confusion, found by family", "age": 84, "sex": "M",
               "arrival_transport": "AMBULANCE", "nurse_esi": 2,
               "comorbidities": ["dementia", "chronic kidney disease", "indwelling urinary catheter"],
               "triage_vitals": {"temperature_C": 36.4, "heart_rate": 98, "resp_rate": 20, "spo2": 95,
                                 "sbp": 128, "dbp": 70, "pain_0_10": 0}},
        "oxygen": False,
        "labs": {"wbc": 13.5, "hemoglobin": 11.2, "platelets": 180, "creatinine": 2.1, "bun": 38,
                 "bilirubin_total": 0.6, "glucose": 128, "sodium": 146, "potassium": 4.8,
                 "bicarbonate": 22, "anion_gap": 14},
    },
    "P3": {
        "label": {"en": "Non-infectious mimic", "zh": "非感染（像脓毒症）"},
        "t0": {"chief_complaint": "Shortness of breath and leg swelling", "age": 58, "sex": "M",
               "arrival_transport": "WALK IN", "nurse_esi": 2,
               "comorbidities": ["heart failure with reduced ejection fraction", "hypertension"],
               "triage_vitals": {"temperature_C": 36.9, "heart_rate": 104, "resp_rate": 24, "spo2": 91,
                                 "sbp": 158, "dbp": 94, "pain_0_10": 2}},
        "oxygen": True,
        "labs": {"wbc": 8.9, "hemoglobin": 13.9, "platelets": 250, "creatinine": 1.4, "bun": 28,
                 "bilirubin_total": 1.1, "glucose": 118, "sodium": 136, "potassium": 4.4,
                 "bicarbonate": 24, "anion_gap": 12},
    },
}

PROV = {"source_type": "synthetic_fixture", "dataset": "synthetic_demo", "author": "Bunny (Yana's AI assistant)"}


def _result(item, value, unit, sample, avail):
    return {"item": item, "actual_order_status": "collected", "order_time": max(0.05, sample - 0.1),
            "sample_time": sample, "result_status": "resulted", "result_kind": "diagnostic_laboratory",
            "disposition_relevant": True, "checkpoint_eligible": True, "result_available_time": avail,
            "value": value, "unit": unit, "abnormal_flag": None, "result_summary": None,
            "provenance": dict(PROV, source_field=f"observed_workup.{item}")}


def make_case(pid, cp, lk, mk, mocks):
    p, t = PATIENTS[pid], CHECKPOINTS[cp]
    avail = round(t - 0.1, 2)
    sample = round(max(0.1, t - 0.4), 2)
    tv = p["t0"]["triage_vitals"]
    m = MAP[mk]
    labs = [_result("serum_lactate", LACTATE[lk], "mmol/L", sample, avail)]
    labs += [_result(k, v, LAB_UNITS[k], sample, avail) for k, v in p["labs"].items()]
    labs.append({"item": "blood_cultures", "actual_order_status": "collected", "order_time": 0.1,
                 "sample_time": 0.2, "result_status": "pending", "result_kind": "final_culture",
                 # cultures take 24-48 h; the ED disposition does not wait for them
                 "disposition_relevant": False, "checkpoint_eligible": False, "result_available_time": None,
                 "value": None, "unit": None, "result_summary": None,
                 "provenance": dict(PROV, source_field="observed_workup.blood_cultures")})
    o2 = p["oxygen"]
    resp = {"start_time": 0.1, "oxygen_therapy_documented": o2, "mode": "nasal_cannula" if o2 else "room_air",
            "ventilation_status": "SupplementalOxygen" if o2 else "None", "current_encounter": True,
            "provenance": dict(PROV, source_field="respiratory_support")}
    vitals = [
        {"measurement_time": 0.0, **{k: tv[k] for k in ("sbp", "dbp", "resp_rate", "spo2", "heart_rate", "temperature_C")},
         "map": round((tv["sbp"] + 2 * tv["dbp"]) / 3, 1), "provenance": dict(PROV, source_field="vitals.triage")},
        {"measurement_time": avail, "sbp": m + 27, "dbp": m - 13, "map": m, "resp_rate": tv["resp_rate"],
         "spo2": tv["spo2"] + (2 if o2 else 0), "heart_rate": tv["heart_rate"], "temperature_C": tv["temperature_C"],
         "provenance": dict(PROV, source_field="vitals.checkpoint")},
    ]
    # Bedside reassessment at the checkpoint (written for the demo): worsening when MAP
    # is below 65 or lactate is >= 4, otherwise stable. A4 treats both as present fields.
    trend = "worsening" if (m < 65 or LACTATE[lk] >= 4) else "stable"
    vitals[-1]["trend"] = trend
    response = [{"assessment_time": avail, "state": trend,
                 "provenance": dict(PROV, source_field="treatment_response")}]
    none = lambda k, extra: [dict({"administered": False, "status": "not_administered",
                                   "provenance": dict(PROV, source_field=f"observed_treatment.{k}")}, **extra)]
    return {
        "case_id": f"{pid}_{cp}_{lk}_{mk}",
        "demo": {"patient": pid, "checkpoint": cp, "lactate": lk, "map": mk},
        "t0_time": 0.0,
        "t0_state": copy.deepcopy(p["t0"]) | {"investigation_status": {
            "blood_cultures": "not_ordered", "serum_lactate": "not_ordered",
            "source_imaging_or_sampling": "not_indicated", "alternative_diagnosis_testing": "not_indicated"},
            "labs": "no result available before T0"},
        "observed_workup": labs,
        "longitudinal_events": {
            "vitals": vitals,
            "respiratory_support": [resp],
            "vasopressors": [{"start_time": 0.0, "active": False, "setting": "none", "agent": None,
                              "provenance": dict(PROV, source_field="vasopressors")}],
            "fluids": [{"event_time": 0.1, "administered": False, "amount_ml": 0,
                        "initial_fluid_30ml_per_kg_completed": False, "completion_time": None,
                        "provenance": dict(PROV, source_field="fluids")}],
            "treatment_response": response,
        },
        "observed_treatment": {
            "antimicrobials": none("antimicrobials", {"administration_time": 0.1}),
            "fluids": none("fluids", {"event_time": 0.1, "amount_ml": 0}),
            "vasopressors": none("vasopressors", {"start_time": 0.0, "active": False, "setting": "none", "agent": None}),
            "respiratory_support": [{"start_time": 0.1, "active": o2, "status": resp["mode"], "mode": resp["mode"],
                                     "provenance": dict(PROV, source_field="observed_treatment.respiratory_support")}],
            "treatment_response": response,
        },
        "engineering_mocks": mocks,
        "evaluation_only": {"truth": None, "note": "synthetic demo case, no outcome"},
    }


def main():
    mocks = json.loads(FIXTURE.read_text())["defaults"]["engineering_mocks"]
    cases = [make_case(p, c, l, m, mocks) for p in PATIENTS for c in CHECKPOINTS for l in LACTATE for m in MAP]
    out = {"fixture_type": "synthetic_demo", "clinical_validation": False,
           "author": "Bunny (Yana's AI assistant); illustrative, not real patients",
           "patients": {k: {"label": v["label"], "t0": v["t0"], "oxygen": v["oxygen"], "labs": v["labs"]}
                        for k, v in PATIENTS.items()},
           "levels": {"lactate": LACTATE, "map": MAP, "checkpoints": CHECKPOINTS, "lab_units": LAB_UNITS},
           "defaults": {}, "cases": cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
