#!/usr/bin/env python3
"""Offline tests for the RADAR vNext orchestration and evaluation interfaces."""
from __future__ import annotations
from radar import paths

import argparse
import copy
import json
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
sys.path.insert(0, str(HERE))

from radar.agents import run_agent1  # noqa: E402
from radar.agents import run_agent2  # noqa: E402
from radar.agents import run_agent3  # noqa: E402
from radar.agents import run_agent4_t0  # noqa: E402
from radar.pipeline import run_pipeline  # noqa: E402
from radar.pipeline import run_real_model_smoke_tests  # noqa: E402
from radar.pipeline.checkpoint_replay import build_checkpoint_plan  # noqa: E402
from radar.evaluation.evaluate_insufficiency import (  # noqa: E402
    calculate_insufficiency_metrics,
    paired_comparison,
)
from radar.evaluation.evaluation_export import build_case_evaluation_export, tier_transition  # noqa: E402
from radar.common.integration_schema import provenance  # noqa: E402
from radar.pipeline.longitudinal_integration import (  # noqa: E402
    assemble_checkpoint_state,
    build_treatment_state,
    build_workup_state,
)
from radar.common.observation_semantics import (  # noqa: E402
    OBSERVATION_ADMISSION_TYPES,
    classify_observed_disposition,
    is_observation_admission_type,
    observation_semantics_metadata,
)


DRY_ARGS = SimpleNamespace(dry_run=True, model="synthetic-stub", temperature=0.0, seed=7)


def observed_result(item: str, result_time: float, value=None, sample_time=0.1, checkpoint_eligible=True, result_kind="diagnostic_laboratory") -> dict:
    return {
        "item": item,
        "actual_order_status": "ordered",
        "order_time": 0.05,
        "sample_time": sample_time,
        "result_status": "resulted",
        "result_available_time": result_time,
        "value": value,
        "result_summary": f"synthetic {item}",
        "result_kind": result_kind,
        "disposition_relevant": True,
        "checkpoint_eligible": checkpoint_eligible,
        "provenance": provenance("synthetic_fixture", source_field=f"test.{item}"),
    }


def t0_payload(unresolved=True) -> dict:
    recommendations = [{"item": "serum_lactate"}] if unresolved else []
    return {
        "case_id": "T0_TEST",
        "t0_time": 0.0,
        "information_cutoff": 0.0,
        "t0_state": {
            "chief_complaint": "Synthetic fever fixture",
            "age": 60,
            "triage_vitals": {"sbp": 118, "resp_rate": 20, "spo2": 97},
            "active_support": {
                "vasopressor": {"active": False, "setting": "none", "start_time": None},
                "ventilation": {"status": "None", "start_time": None, "current_encounter": True},
            },
        },
        "agent1_context": {
            "tier": "possible_sepsis",
            "shock_concern": False,
            "decision_time": "T0",
            "provenance": "agent1_model_output",
        },
        "agent2": {
            "workup_recommendation": {
                "recommendations": recommendations,
                "already_addressed": [],
                "not_indicated": [],
                "indeterminate": [],
            }
        },
        "mock_llm_answers": {
            "D2_T0": {
                "answer": "no",
                "key_reasons": [{"field": "t0_state.triage_vitals.sbp", "detail": "synthetic"}],
                "unresolved_risks": [],
                "missing_information": [],
                "escalation_trigger": None,
            },
            "D4_T0": {
                "answer": "discharge",
                "key_reasons": [{"field": "agent1_context.tier", "detail": "synthetic"}],
                "unresolved_risks": [],
                "missing_information": [],
                "escalation_trigger": "Return for worsening symptoms.",
            },
        },
    }


class TestAgent1ToAgent2(unittest.TestCase):
    def setUp(self):
        self.case = json.loads((paths.DATA / "agent2_synthetic_cases.json").read_text(encoding="utf-8"))[0]
        self.case["agent1_context"] = {
            "tier": "probable_sepsis",
            "shock_concern": True,
            "decision_time": "T0",
            "provenance": "agent1_model_output",
            "ESRP": 91,
            "risk_signal": {"value": 91},
        }

    def test_a2_receives_whitelisted_a1_tier(self):
        context = run_agent2.upstream_recognition_context(self.case)
        self.assertEqual(context["tier"], "probable_sepsis")
        self.assertTrue(context["shock_concern"])

    def test_a2_trace_records_upstream_recognition(self):
        result = run_agent2.run_case(self.case, None, DRY_ARGS)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["trace"])
        self.assertTrue(all(step["upstream_recognition"]["tier"] == "probable_sepsis" for step in result["trace"]))

    def test_esrp_is_not_direct_a2_context(self):
        context = run_agent2.upstream_recognition_context(self.case)
        self.assertNotIn("ESRP", context)
        self.assertNotIn("risk_signal", context)
        node_id = run_agent2.GRAPH["entry"]
        node = dict(run_agent2.GRAPH["nodes"][node_id], id=node_id)
        prompt = run_agent2.user_prompt(node, self.case, run_agent2.materials(node))
        self.assertNotIn("ESRP", prompt)
        self.assertNotIn('"value": 91', prompt)

    def test_truth_and_outcomes_do_not_enter_a2(self):
        self.case["input"]["truth"] = {"sepsis": True}
        self.case["input"]["dsp_mortality_30d"] = 1
        safe = run_agent2.patient_input(self.case)
        self.assertNotIn("truth", safe)
        self.assertNotIn("dsp_mortality_30d", json.dumps(safe))


class TestAgent4AtT0(unittest.TestCase):
    def test_t0_safe_state_is_accepted(self):
        result = run_agent4_t0.run_case(t0_payload(), None, DRY_ARGS)
        self.assertEqual(result["result"]["status"], "continue_workup")
        self.assertTrue(all("question" in step and "allowed_answers" in step for step in result["trace"]))

    def test_post_t0_lab_result_is_hidden(self):
        payload = t0_payload()
        payload["t0_state"]["labs"] = {
            "lactate": {
                "result_status": "resulted",
                "sample_time": 0.3,
                "result_available_time": 0.8,
                "value": 5.0,
                "result_summary": "future high result",
            }
        }
        safe = run_agent4_t0.sanitize_t0_payload(payload)
        serialized = json.dumps(safe)
        self.assertNotIn("5.0", serialized)
        self.assertNotIn("future high result", serialized)
        self.assertEqual(safe["t0_state"]["labs"]["lactate"]["result_status"], "pending")
        self.assertIsNone(safe["t0_state"]["labs"]["lactate"]["sample_time"])

    def test_post_t0_treatment_is_hidden(self):
        payload = t0_payload()
        payload["t0_state"]["treatments"] = [
            {"administration_time": 1.0, "agent": "future_antimicrobial"}
        ]
        serialized = json.dumps(run_agent4_t0.sanitize_t0_payload(payload))
        self.assertNotIn("future_antimicrobial", serialized)

    def test_future_outcomes_are_hidden(self):
        payload = t0_payload()
        payload["evaluation_only"] = {"dsp_mortality_inhospital": 1}
        payload["observed_disposition"] = "ICU"
        payload["icu_transfer_time"] = 2.0
        payload["dsp_future_new_endpoint"] = 1
        serialized = json.dumps(run_agent4_t0.sanitize_t0_payload(payload))
        self.assertNotIn("dsp_mortality_inhospital", serialized)
        self.assertNotIn("observed_disposition", serialized)
        self.assertNotIn("icu_transfer_time", serialized)
        self.assertNotIn("dsp_future_new_endpoint", serialized)

    def test_unresolved_workup_routes_to_continue(self):
        result = run_agent4_t0.run_case(t0_payload(unresolved=True), None, DRY_ARGS)
        self.assertEqual(result["terminal"], "T_CONTINUE_WORKUP")
        self.assertTrue(result["result"]["continue_workup"])

    def test_continue_workup_is_not_a_disposition(self):
        result = run_agent4_t0.run_case(t0_payload(unresolved=True), None, DRY_ARGS)["result"]
        self.assertEqual(result["status"], "continue_workup")
        self.assertIsNone(result["disposition"])

    def test_discharge_requires_return_trigger(self):
        payload = t0_payload(unresolved=False)
        payload["mock_llm_answers"]["D4_T0"]["escalation_trigger"] = None
        result = run_agent4_t0.run_case(payload, None, DRY_ARGS)
        self.assertEqual(result["result"]["status"], "abstain")
        self.assertEqual(len(result["trace"][-1]["attempts"]), 2)

    def test_t1_only_hard_evidence_does_not_fire_at_t0(self):
        payload = t0_payload()
        payload["t0_state"]["post_resuscitation"] = {"vital_map_min": 50, "map_time": 0.0}
        payload["t0_state"]["labs"] = {
            "lactate": {"result_status": "resulted", "result_available_time": 0.0, "value": 8.0}
        }
        safety = run_agent4_t0.evaluate_t0_icu_triggers(payload)
        self.assertFalse(safety["icu"])
        self.assertEqual(len(safety["tier_b_inactive_at_t0"]), 3)

    def test_visible_prearrival_support_can_trigger_icu(self):
        payload = t0_payload()
        payload["t0_state"]["active_support"]["vasopressor"] = {
            "active": True, "setting": "pre_arrival", "start_time": -0.1
        }
        result = run_agent4_t0.run_case(payload, None, DRY_ARGS)
        self.assertEqual(result["result"]["disposition"], "ICU")


class TestCheckpointReplay(unittest.TestCase):
    def plan(self, times):
        workup = build_workup_state(None, [
            observed_result(f"result_{index}", result_time, index)
            for index, result_time in enumerate(times)
        ])
        return build_checkpoint_plan(workup)

    def test_result_at_point_two_is_processed_at_c0_5(self):
        plan = self.plan([0.2])
        self.assertEqual(plan["checkpoints"][0]["checkpoint_id"], "C0_5")
        self.assertEqual(plan["checkpoints"][0]["checkpoint_status"], "processed")

    def test_point_eight_result_not_visible_at_c0_5(self):
        workup = build_workup_state(None, [observed_result("serum_lactate", 0.8, 2.2, 0.3)])
        state = assemble_checkpoint_state({}, workup, 0.5, "C0_5")
        self.assertEqual(state["labs"]["lactate_initial"]["status"], "pending")
        self.assertIsNone(state["labs"]["lactate_initial"]["value"])

    def test_point_eight_result_visible_at_c1(self):
        workup = build_workup_state(None, [observed_result("serum_lactate", 0.8, 2.2, 0.3)])
        state = assemble_checkpoint_state({}, workup, 1.0, "C1")
        self.assertEqual(state["labs"]["lactate_initial"]["value"], 2.2)

    def test_two_hour_result_is_processed_at_c3(self):
        plan = self.plan([2.0])
        self.assertEqual(plan["checkpoints"][-1]["checkpoint_id"], "C3")
        self.assertEqual(plan["checkpoints"][-1]["checkpoint_status"], "processed")

    def test_window_without_new_result_is_skipped(self):
        statuses = {row["checkpoint_id"]: row["checkpoint_status"] for row in self.plan([0.2, 2.0])["checkpoints"]}
        self.assertEqual(statuses["C1"], "skipped")

    def test_c6_is_not_planned_without_later_result(self):
        ids = [row["checkpoint_id"] for row in self.plan([0.2, 0.8, 2.0])["checkpoints"]]
        self.assertEqual(ids, ["C0_5", "C1", "C3"])

    def test_post_six_hour_result_adds_cfinal(self):
        plan = self.plan([0.2, 7.4])
        self.assertEqual(plan["checkpoints"][-1]["checkpoint_id"], "Cfinal")
        self.assertEqual(plan["checkpoints"][-1]["cutoff_time"], 7.4)

    def test_future_result_existence_not_in_earlier_prompt(self):
        workup = build_workup_state(None, [observed_result("serum_lactate", 0.8, 2.2, 0.3)])
        state = assemble_checkpoint_state({}, workup, 0.5, "C0_5")
        case = {"input": {"checkpoint_state": state, "checkpoint": {"checkpoint_id": "C0_5", "cutoff_time": 0.5}}}
        node_id = run_agent1.GRAPH["entry"]
        node = dict(run_agent1.GRAPH["nodes"][node_id], id=node_id)
        prompt = run_agent1.user_prompt(node, case, run_agent1.materials(node), "checkpoint")
        self.assertNotIn('"result_available_time": 0.8', prompt)

    def test_future_result_value_not_in_earlier_prompt(self):
        workup = build_workup_state(None, [observed_result("serum_lactate", 0.8, 9.9, 0.3)])
        state = assemble_checkpoint_state({}, workup, 0.5, "C0_5")
        case = {"input": {"checkpoint_state": state}}
        node_id = run_agent1.GRAPH["entry"]
        node = dict(run_agent1.GRAPH["nodes"][node_id], id=node_id)
        prompt = run_agent1.user_prompt(node, case, run_agent1.materials(node), "checkpoint")
        self.assertNotIn("9.9", prompt)

    def test_latest_eligible_vital_is_selected(self):
        workup = build_workup_state(None, [observed_result("serum_lactate", 0.2, 1.5)])
        state = assemble_checkpoint_state({"vitals": [
            {"measurement_time": 0.2, "sbp": 110},
            {"measurement_time": 0.5, "sbp": 115},
            {"measurement_time": 0.8, "sbp": 70},
        ]}, workup, 0.5, "C0_5")
        self.assertEqual(state["vitals"]["sbp"], 115)

    def test_post_cutoff_treatment_is_excluded(self):
        state = build_treatment_state({"antimicrobials": [
            {"administration_time": 0.8, "administered": True, "status": "completed"}
        ]}, 0.5)
        self.assertEqual(state["antimicrobials"]["status"], "unavailable")
        self.assertEqual(state["excluded_post_checkpoint_event_counts"]["antimicrobials"], 1)


class TestAgent3Rename(unittest.TestCase):
    def test_graph_uses_treatment_node_ids(self):
        self.assertEqual(list(run_agent3.GRAPH["nodes"]), ["TX1", "TX2", "TX3", "TX4"])

    def test_four_clinical_decisions_are_preserved(self):
        case = json.loads((paths.DATA / "agent3_synthetic_cases.json").read_text(encoding="utf-8"))[0]
        result = run_agent3.run_case(case, None, DRY_ARGS)
        self.assertEqual([step["node"] for step in result["trace"]], ["TX1", "TX2", "TX3", "TX4"])
        plan = result["result"]["treatment_plan"]
        self.assertEqual(set(plan), {"antimicrobial_timing", "fluids", "vasopressors", "map_target"})


class TestObservationSemantics(unittest.TestCase):
    def test_all_canonical_admission_types_are_observation(self):
        for admission_type in OBSERVATION_ADMISSION_TYPES:
            with self.subTest(admission_type=admission_type):
                self.assertTrue(is_observation_admission_type(admission_type.lower()))

    def test_hospitalization_alone_does_not_exclude_observation(self):
        outcomes = {"dsp_hospitalization": 1, "admission_type": "EU OBSERVATION"}
        self.assertEqual(classify_observed_disposition(outcomes), "observation")
        self.assertIsNone(classify_observed_disposition({
            "observed_disposition": "observation", "admission_type": "URGENT"
        }))
        self.assertEqual(
            observation_semantics_metadata()["hospitalization_flag_role"],
            "ignored_for_disposition_boundaries",
        )


class TestEvaluationTrace(unittest.TestCase):
    @staticmethod
    def envelope():
        def a1(tier):
            return {"result": {"tier": tier}, "trace": []}

        return {
            "case_id": "EVAL",
            "stay_id": 99,
            "agent1_t0": {"result": {"tier": "possible_sepsis"}, "risk_signal": {"enabled": True}, "trace": []},
            "agent4_t0": {"result": {"status": "continue_workup", "disposition": None, "continue_workup": True}, "trace": []},
            "checkpoints": [
                {"checkpoint_id": "C0_5", "cutoff_time": 0.5, "checkpoint_status": "processed", "newly_available_results": [], "agent1_reassessment": a1("probable_sepsis"), "agent3": {"result": {"status": "complete", "treatment_plan": {}}}, "agent4": {"result": {"status": "continue_workup", "disposition": None}}},
                {"checkpoint_id": "C1", "cutoff_time": 1.0, "checkpoint_status": "processed", "newly_available_results": [], "agent1_reassessment": a1("probable_sepsis"), "agent3": {"result": {"status": "complete", "treatment_plan": {}}}, "agent4": {"result": {"status": "decision", "disposition": "ward"}}},
                {"checkpoint_id": "C3", "cutoff_time": 3.0, "checkpoint_status": "processed", "newly_available_results": [], "agent1_reassessment": a1("unlikely_sepsis"), "agent3": {"result": {"status": "complete", "treatment_plan": {}}}, "agent4": {"result": {"status": "decision", "disposition": "ward"}}},
            ],
            "final_simulated_disposition": "ward",
            "disposition_checkpoint": "C1",
            "evaluation_only": {"dsp_mortality_30d": 0},
        }

    def test_t0_tier_is_retained(self):
        self.assertEqual(build_case_evaluation_export(self.envelope())["tier_T0"], "possible_sepsis")

    def test_all_processed_checkpoint_tiers_are_retained(self):
        rows = build_case_evaluation_export(self.envelope())["checkpoint_sequence"]
        self.assertEqual([row["agent1_tier"] for row in rows], ["probable_sepsis", "probable_sepsis", "unlikely_sepsis"])

    def test_transition_labels_are_deterministic(self):
        self.assertEqual(tier_transition("possible_sepsis", "probable_sepsis"), "upgrade")
        self.assertEqual(tier_transition("probable_sepsis", "unlikely_sepsis"), "downgrade")
        self.assertEqual(tier_transition("possible_sepsis", "possible_sepsis"), "unchanged")

    def test_evaluation_only_values_are_not_exported(self):
        export = build_case_evaluation_export(self.envelope())
        serialized = json.dumps(export)
        self.assertTrue(export["evaluation_only_available"])
        self.assertNotIn("dsp_mortality_30d", serialized)

    def test_observed_disposition_label_provenance_is_derived_without_raw_fields(self):
        envelope = self.envelope()
        envelope["evaluation_only"].update({
            "admission_type": "DIRECT OBSERVATION",
            "hadm_id": 42,
        })
        export = build_case_evaluation_export(envelope)
        observed = export["observed_disposition_evaluation"]
        self.assertEqual(observed["label"], "observation")
        self.assertEqual(observed["provenance"], "hosp.admissions.admission_type")
        self.assertTrue(observed["evaluation_only"])
        self.assertNotIn("DIRECT OBSERVATION", json.dumps(export))
        self.assertNotIn('"hadm_id"', json.dumps(export))


class TestApiReadiness(unittest.TestCase):
    def test_smoke_parser_initializes_without_external_call(self):
        args = run_real_model_smoke_tests.create_parser().parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_missing_key_does_not_affect_dry_run(self):
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            args = run_real_model_smoke_tests.create_parser().parse_args(["--dry-run"])
            result = run_real_model_smoke_tests.run_smoke(args)
            self.assertEqual(set(result["results"]), {"A2", "A3", "A4"})
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old

    def test_secret_is_never_stored_in_smoke_output(self):
        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-secret-must-not-appear"
        try:
            args = run_real_model_smoke_tests.create_parser().parse_args(["--dry-run"])
            output = run_real_model_smoke_tests.run_smoke(args)
            self.assertNotIn("test-secret-must-not-appear", json.dumps(output))
            self.assertFalse(output["meta"]["credentials_stored"])
        finally:
            if old is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = old


class TestInsufficiencyEvaluation(unittest.TestCase):
    @staticmethod
    def record(answer="indeterminate", repaired=False):
        attempts = [{"attempt": 0, "ok": not repaired}]
        if repaired:
            attempts.append({"attempt": 1, "ok": True})
        return {
            "case_id": "I1",
            "status": "complete",
            "result": {"agent": "fixture_agent", "status": "complete"},
            "trace": [{"node": "N1", "answer": answer, "attempts": attempts}],
        }

    def test_generic_rates_are_computed(self):
        metrics = calculate_insufficiency_metrics([self.record(), self.record("yes", repaired=True)])
        self.assertEqual(metrics["overall"]["model_node_responses"], 2)
        self.assertEqual(metrics["overall"]["insufficient_or_indeterminate_rate"], 0.5)
        self.assertEqual(metrics["overall"]["successful_repairs"], 1)
        self.assertFalse(metrics["artifact_status"]["specific_intervention_found"])

    def test_paired_comparison_is_supported(self):
        comparison = paired_comparison([self.record()], [self.record("yes")])
        self.assertTrue(comparison["case_ids_matched"])
        self.assertEqual(comparison["delta_insufficient_or_indeterminate_rate"], -1.0)
        self.assertIn("NO PERFORMANCE CLAIM", comparison["label"])


class TestPipelineModes(unittest.TestCase):
    def test_sequential_is_default_mode(self):
        self.assertEqual(run_pipeline.create_parser().parse_args([]).pipeline_mode, "sequential")

    def test_legacy_single_t1_entrypoint_is_retained(self):
        self.assertTrue(callable(run_pipeline.legacy_run_longitudinal_case))

    def test_full_replay_records_all_triggered_checkpoints(self):
        case = run_pipeline.load_cases(paths.DATA / "checkpoint_replay_fixture.json")[0]
        result = run_pipeline.run_longitudinal_case(
            case,
            None,
            run_pipeline.pipeline_config(dry_run=True, replay_mode="full"),
        )
        statuses = [(row["checkpoint_id"], row["checkpoint_status"]) for row in result["checkpoints"]]
        self.assertEqual(statuses, [
            ("C0_5", "processed"),
            ("C1", "processed"),
            ("C3", "processed"),
            ("C6", "skipped"),
            ("Cfinal", "skipped"),
        ])
        self.assertEqual(result["disposition_checkpoint"], "C0_5")


if __name__ == "__main__":
    unittest.main(verbosity=2)
