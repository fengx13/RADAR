#!/usr/bin/env python3
"""Lightweight Agent 4 tests. Run with: python test_agent4.py"""
from __future__ import annotations
from radar import paths

import argparse
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
sys.path.insert(0, str(HERE))

from radar.agents.agent4_insufficiency import (  # noqa: E402
    apply_uncertainty_ceiling,
    validate_insufficiency,
)
from radar.agents.agent4_safety import evaluate_icu_triggers  # noqa: E402
from radar.evaluation.evaluate_agent4 import calculate_metrics  # noqa: E402
from radar.agents.run_agent4 import run_case, strip_sensitive, user_prompt, validate_reply, GRAPH  # noqa: E402


class Args(argparse.Namespace):
    dry_run = True
    model = "dry"
    temperature = 0.0


def base_payload():
    return {
        "case_id": "T",
        "t1_time": 4,
        "t1_state": {
            "vitals": {"sbp": 122, "map": 82, "resp_rate": 18, "spo2": 97},
            "oxygen_requirement": {"oxygen_therapy_documented": False, "mode": "room_air", "start_time": 0},
            "vasopressor": {"active": False, "setting": "none", "start_time": None},
            "ventilation": {"status": "None", "start_time": None, "current_encounter": True},
            "resuscitation": {"initial_fluid_30ml_per_kg": {"completed": False, "completion_time": None}},
            "post_resuscitation": {"vital_map_min": 82, "map_time": 4, "vital_o2sat_min": 97, "o2sat_time": 4},
            "labs": {
                "lactate_initial": {"status": "resulted", "value": 1.5, "sample_time": 2, "result_time": 3},
                "lactate_repeat": None,
                "culture_result": None,
                "imaging_result": None,
            },
            "treatment_response": "assessed_stable",
            "vitals_trend": "stable",
        },
        "agent1": {"tier": "possible_sepsis", "shock_concern": False},
        "agent2": {"workup_recommendation": {}},
        "workup_state": {"pending_workup": []},
        "agent3": {"treatment_recommendation": {}},
        "treatment_state": {"ongoing_treatment": [], "treatment_response_flag": "stable"},
        "mock_llm_answers": {
            "D2": {"answer": "no", "key_reasons": [{"field": "agent1.tier", "detail": "ok"}], "unresolved_risks": [], "missing_information": [], "escalation_trigger": None, "blocking": []},
            "D3": {"answer": "no", "key_reasons": [{"field": "t1_state.vitals_trend", "detail": "ok"}], "unresolved_risks": [], "missing_information": [], "escalation_trigger": None, "blocking": []},
            "D4": {"answer": "yes", "key_reasons": [{"field": "t1_state.vitals.sbp", "detail": "ok"}], "unresolved_risks": [], "missing_information": [], "escalation_trigger": "Return for worsening symptoms.", "blocking": []},
        },
    }


class TestAgent4Safety(unittest.TestCase):
    def test_vasopressor_active_icu(self):
        p = base_payload()
        p["t1_state"]["vasopressor"] = {"active": True, "setting": "ed", "start_time": 1}
        self.assertTrue(evaluate_icu_triggers(p)["icu"])
        self.assertEqual(run_case(p, None, Args()).get("result", {}).get("disposition"), "ICU")

    def test_invasive_ventilation_icu(self):
        p = base_payload()
        p["t1_state"]["ventilation"] = {"status": "InvasiveVent", "start_time": 1, "current_encounter": True}
        ids = [t["id"] for t in evaluate_icu_triggers(p)["triggers"]]
        self.assertIn("ICU2_invasive_ventilation", ids)

    def test_niv_hfnc_icu(self):
        p = base_payload()
        p["t1_state"]["ventilation"] = {"status": "HFNC", "start_time": 1, "current_encounter": True}
        ids = [t["id"] for t in evaluate_icu_triggers(p)["triggers"]]
        self.assertIn("ICU3_niv_or_hfnc", ids)

    def test_post_resuscitation_map_is_inactive_tier_b_candidate(self):
        p = base_payload()
        p["t1_state"]["resuscitation"]["initial_fluid_30ml_per_kg"] = {"completed": True, "completion_time": 2}
        p["t1_state"]["post_resuscitation"]["vital_map_min"] = 60
        p["t1_state"]["post_resuscitation"]["map_time"] = 3
        result = evaluate_icu_triggers(p)
        ids = [t["id"] for t in result["tier_b_candidates"]]
        self.assertIn("ICU4_post_resuscitation_hypotension", ids)
        self.assertFalse(result["icu"])

    def test_post_resuscitation_lactate_is_inactive_tier_b_candidate(self):
        p = base_payload()
        p["t1_state"]["resuscitation"]["initial_fluid_30ml_per_kg"] = {"completed": True, "completion_time": 2}
        p["t1_state"]["labs"]["lactate_initial"] = {"status": "resulted", "value": 4.0, "sample_time": 3, "result_time": 3.5}
        result = evaluate_icu_triggers(p)
        ids = [t["id"] for t in result["tier_b_candidates"]]
        self.assertIn("ICU5_post_resuscitation_lactate", ids)
        self.assertFalse(result["icu"])

    def test_hypoxemia_despite_oxygen_is_inactive_tier_b_candidate(self):
        p = base_payload()
        p["t1_state"]["oxygen_requirement"]["oxygen_therapy_documented"] = True
        p["t1_state"]["post_resuscitation"]["vital_o2sat_min"] = 89
        result = evaluate_icu_triggers(p)
        ids = [t["id"] for t in result["tier_b_candidates"]]
        self.assertIn("ICU6_hypoxemia_despite_oxygen", ids)
        self.assertFalse(result["icu"])

    def test_missing_current_encounter_does_not_trigger_ventilation_icu(self):
        p = base_payload()
        p["t1_state"]["ventilation"] = {"status": "InvasiveVent", "start_time": 1}
        result = evaluate_icu_triggers(p)
        self.assertFalse(result["icu"])
        self.assertIn("t1_state.ventilation.current_encounter", result["missing_fields"])

    def test_missing_fluid_completed_does_not_trigger_map_or_lactate_icu(self):
        p = base_payload()
        p["t1_state"]["resuscitation"]["initial_fluid_30ml_per_kg"] = {"completion_time": 2}
        p["t1_state"]["post_resuscitation"]["vital_map_min"] = 60
        p["t1_state"]["post_resuscitation"]["map_time"] = 3
        p["t1_state"]["labs"]["lactate_initial"] = {"status": "resulted", "value": 4.5, "sample_time": 3, "result_time": 3.5}
        result = evaluate_icu_triggers(p)
        self.assertFalse(result["icu"])
        self.assertIn("t1_state.resuscitation.initial_fluid_30ml_per_kg.completed", result["tier_b_missing_fields"])

    def test_missing_oxygen_documented_does_not_trigger_hypoxemia_icu(self):
        p = base_payload()
        p["t1_state"]["oxygen_requirement"] = {"mode": "nasal_cannula"}
        p["t1_state"]["post_resuscitation"]["vital_o2sat_min"] = 89
        result = evaluate_icu_triggers(p)
        self.assertFalse(result["icu"])
        self.assertIn("t1_state.oxygen_requirement.oxygen_therapy_documented", result["tier_b_missing_fields"])


class TestAgent4Insufficiency(unittest.TestCase):
    def test_pending_decision_changing_observation(self):
        p = base_payload()
        reply = {"answer": "insufficient_evidence", "blocking": [{"field": "t1_state.labs.lactate_repeat", "counterfactual": {"if_a": {"condition": "normal", "disposition": "discharge"}, "if_b": {"condition": "high", "disposition": "ward"}}}]}
        node = GRAPH["nodes"]["D3"]
        ok, failures = validate_insufficiency(reply, node, strip_sensitive(p))
        self.assertTrue(ok, failures)
        p["mock_llm_answers"]["D3"] = {**reply, "key_reasons": [], "unresolved_risks": [], "missing_information": ["repeat lactate"], "escalation_trigger": None}
        self.assertEqual(run_case(p, None, Args())["result"]["disposition"], "observation")

    def test_pending_counterfactual_not_decisive_fails(self):
        p = base_payload()
        reply = {"answer": "insufficient_evidence", "blocking": [{"field": "t1_state.labs.culture_result", "counterfactual": {"if_a": {"condition": "positive", "disposition": "discharge"}, "if_b": {"condition": "negative", "disposition": "discharge"}}}]}
        ok, failures = validate_insufficiency(reply, GRAPH["nodes"]["D3"], strip_sensitive(p))
        self.assertFalse(ok)
        self.assertTrue(any("V4c_counterfactual_not_decisive" in f for f in failures))

    def test_model_claims_present_field_missing_fails(self):
        p = base_payload()
        reply = {"answer": "insufficient_evidence", "blocking": [{"field": "t1_state.vitals.sbp"}]}
        ok, failures = validate_insufficiency(reply, GRAPH["nodes"]["D2"], strip_sensitive(p))
        self.assertFalse(ok)
        self.assertTrue(any("V4b_field_is_present" in f for f in failures))

    def test_must_exist_missing_abstain_pipeline_alarm(self):
        p = base_payload()
        del p["treatment_state"]["treatment_response_flag"]
        p["mock_llm_answers"]["D2"] = {"answer": "insufficient_evidence", "key_reasons": [], "unresolved_risks": [], "missing_information": ["observed treatment response"], "escalation_trigger": None, "blocking": [{"field": "treatment_state.treatment_response_flag"}]}
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "abstain")
        self.assertTrue(r["pipeline_alarm"])

    def test_never_available_blocking_fails(self):
        p = base_payload()
        reply = {"answer": "insufficient_evidence", "blocking": [{"field": "t1_state.mental_status"}]}
        ok, failures = validate_insufficiency(reply, {"blockable_fields": ["t1_state.mental_status"]}, strip_sensitive(p))
        self.assertFalse(ok)
        self.assertTrue(any("V4a_never_available" in f for f in failures))

    def test_uncertainty_ceiling_prevents_discharge(self):
        terminal, note = apply_uncertainty_ceiling("T_DISCHARGE", 1)
        self.assertEqual(terminal, "T_OBSERVATION")
        self.assertIsNotNone(note)

    def test_v4_counterfactual_disposition_must_be_closed_set(self):
        p = base_payload()
        reply = {"answer": "insufficient_evidence", "blocking": [{"field": "t1_state.labs.lactate_repeat", "counterfactual": {"if_a": {"condition": "normal", "disposition": "home"}, "if_b": {"condition": "high", "disposition": "ward"}}}]}
        ok, failures = validate_insufficiency(reply, GRAPH["nodes"]["D3"], strip_sensitive(p))
        self.assertFalse(ok)
        self.assertTrue(any("V4c_counterfactual_disposition_not_allowed" in f for f in failures))


class TestAgent4OutputAndEvaluation(unittest.TestCase):
    def test_discharge_without_escalation_trigger_validation_failure(self):
        p = base_payload()
        p["mock_llm_answers"]["D4"]["escalation_trigger"] = None
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "abstain")

    def test_d4_prompt_and_validator_expose_conditional_trigger_contract(self):
        p = base_payload()
        node = dict(GRAPH["nodes"]["D4"], id="D4")
        prompt = user_prompt(node, p)
        self.assertIn("D4 CONDITIONAL OUTPUT CONTRACT", prompt)
        self.assertIn("escalation_trigger=null", prompt)

        reply = dict(p["mock_llm_answers"]["D4"])
        reply["escalation_trigger"] = None
        ok, failures = validate_reply(reply, node, p)
        self.assertFalse(ok)
        self.assertIn("V3_discharge_requires_escalation_trigger", failures)

        reply["escalation_trigger"] = "Return for worsening symptoms."
        ok, failures = validate_reply(reply, node, p)
        self.assertTrue(ok, failures)

    def test_valid_direct_d3_observation_requires_decision_changing_basis(self):
        p = base_payload()
        p["mock_llm_answers"]["D3"] = {
            "answer": "yes",
            "key_reasons": [{"field": "t1_state.labs.lactate_repeat", "detail": "pending repeat lactate may determine level of care"}],
            "unresolved_risks": [{"field": "t1_state.labs.lactate_repeat", "detail": "pending"}],
            "missing_information": ["repeat lactate"],
            "escalation_trigger": None,
            "observation_basis": {
                "field": "t1_state.labs.lactate_repeat",
                "counterfactual": {
                    "if_a": {"condition": "normal", "disposition": "discharge"},
                    "if_b": {"condition": "high", "disposition": "ward"}
                }
            },
            "blocking": []
        }
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "decision")
        self.assertEqual(r["disposition"], "observation")

    def test_direct_d3_observation_non_decision_changing_basis_rejected(self):
        p = base_payload()
        p["mock_llm_answers"]["D3"] = {
            "answer": "yes",
            "key_reasons": [{"field": "t1_state.labs.culture_result", "detail": "pending culture"}],
            "unresolved_risks": [],
            "missing_information": ["culture"],
            "escalation_trigger": None,
            "observation_basis": {
                "field": "t1_state.labs.culture_result",
                "counterfactual": {
                    "if_a": {"condition": "positive", "disposition": "discharge"},
                    "if_b": {"condition": "negative", "disposition": "discharge"}
                }
            },
            "blocking": []
        }
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "abstain")

    def test_direct_d3_generic_observation_without_basis_rejected(self):
        p = base_payload()
        p["mock_llm_answers"]["D3"] = {
            "answer": "yes",
            "key_reasons": [{"field": "t1_state.vitals_trend", "detail": "uncertain"}],
            "unresolved_risks": [],
            "missing_information": [],
            "escalation_trigger": None,
            "blocking": []
        }
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "abstain")

    def test_valid_discharge_output(self):
        r = run_case(base_payload(), None, Args())["result"]
        self.assertEqual(r["status"], "decision")
        self.assertEqual(r["disposition"], "discharge")
        self.assertTrue(r["escalation_trigger"])

    def test_primary_endpoint_calculation(self):
        results = [
            {"case_id": "A", "result": {"status": "decision", "disposition": "discharge"}},
            {"case_id": "B", "result": {"status": "decision", "disposition": "ward"}},
        ]
        cases = [
            {"case_id": "A", "t1_time": 2, "evaluation_only": {"icu_transfer_time": 10, "dsp_mortality_inhospital": 0}},
            {"case_id": "B", "t1_time": 2, "evaluation_only": {"icu_transfer_time": 10, "dsp_mortality_inhospital": 0}},
        ]
        metrics = calculate_metrics(results, cases)
        self.assertEqual(metrics["primary"]["numerator"], 1)
        self.assertEqual(metrics["primary"]["denominator"], 1)
        self.assertEqual(metrics["primary"]["rate"], 1.0)

    def test_primary_endpoint_requires_decision_discharge_for_numerator(self):
        results = [
            {"case_id": "A", "result": {"status": "abstain", "disposition": "discharge"}},
            {"case_id": "B", "result": {"status": "decision", "disposition": "discharge"}},
        ]
        cases = [
            {"case_id": "A", "t1_time": 2, "evaluation_only": {"icu_transfer_time": 10, "dsp_mortality_inhospital": 0}},
            {"case_id": "B", "t1_time": 2, "evaluation_only": {"icu_transfer_status": "no_transfer", "dsp_mortality_inhospital": 0}},
        ]
        metrics = calculate_metrics(results, cases)
        self.assertEqual(metrics["primary"]["numerator"], 0)
        self.assertEqual(metrics["primary"]["denominator"], 1)

    def test_zero_discharge_denominator(self):
        metrics = calculate_metrics([
            {"case_id": "A", "result": {"status": "decision", "disposition": "ward"}}
        ], [])
        self.assertEqual(metrics["primary"]["denominator"], 0)
        self.assertIsNone(metrics["primary"]["rate"])

    def test_future_truth_not_exposed_to_inference_input(self):
        p = base_payload()
        p["truth"] = {"outcomes": {"dsp_mortality_inhospital": 1}}
        safe = strip_sensitive(p)
        self.assertNotIn("truth", safe)
        prompt = user_prompt(dict(GRAPH["nodes"]["D2"], id="D2"), p)
        self.assertNotIn("dsp_mortality_inhospital", prompt)

    def test_future_dsp_field_outside_truth_not_exposed(self):
        p = base_payload()
        p["t1_state"]["dsp_mortality_inhospital"] = 1
        p["agent2"]["dsp_icu_transfer_12h"] = 1
        safe = strip_sensitive(p)
        self.assertNotIn("dsp_mortality_inhospital", safe["t1_state"])
        self.assertNotIn("dsp_icu_transfer_12h", safe["agent2"])
        prompt = user_prompt(dict(GRAPH["nodes"]["D2"], id="D2"), p)
        self.assertNotIn("dsp_icu_transfer_12h", prompt)

    def test_d0_missing_must_exist_structural_field_abstains(self):
        p = base_payload()
        del p["agent1"]["tier"]
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "abstain")
        self.assertTrue(r["pipeline_alarm"])

    def test_d0_lactate_structure_present_value_pending_does_not_abstain(self):
        p = base_payload()
        p["t1_state"]["labs"]["lactate_initial"] = {"status": "pending", "value": None, "sample_time": None, "result_time": None}
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "decision")

    def test_d0_treatment_response_flag_pending_value_does_not_abstain(self):
        p = base_payload()
        p["treatment_state"]["treatment_response_flag"] = "pending"
        r = run_case(p, None, Args())["result"]
        self.assertEqual(r["status"], "decision")


if __name__ == "__main__":
    unittest.main(verbosity=2)
