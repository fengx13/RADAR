#!/usr/bin/env python3
"""Focused regressions for A1 quote hydration and case-level abstention export."""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from radar.agents import run_agent1  # noqa: E402
from radar.evaluation.evaluation_export import build_case_evaluation_export  # noqa: E402


def node(node_id="R2"):
    return dict(run_agent1.GRAPH["nodes"][node_id], id=node_id)


def reply(statement_id, quote, *, field="supporting_evidence"):
    value = {
        "answer": "no",
        "supporting_evidence": [],
        "opposing_evidence": [],
        "missing_information": [],
        "confidence": "moderate",
    }
    value[field] = [{
        "patient_finding": "synthetic finding",
        "statement_id": statement_id,
        "quote": quote,
        "implication": "synthetic implication",
    }]
    return value


class TestAgent1CanonicalQuoteHydration(unittest.TestCase):
    def hydrate_and_validate(self, value, current_node=None):
        current_node = current_node or node()
        hydrated, audit = run_agent1.hydrate_evidence_quotes(value, current_node)
        injectable_node = dict(current_node)
        injectable_node["scope"] = [
            statement_id
            for statement_id in current_node["scope"]
            if statement_id in run_agent1.STMTS
        ]
        valid, failures, stats = run_agent1.validate(
            hydrated, current_node, run_agent1.materials(injectable_node)
        )
        return hydrated, audit, valid, failures, stats

    def test_paraphrased_quote_is_hydrated_and_passes(self):
        sid = "sepsis3::sofa_table"
        hydrated, _, valid, failures, _ = self.hydrate_and_validate(
            reply(sid, "slightly paraphrased SOFA table")
        )
        self.assertTrue(valid, failures)
        self.assertEqual(hydrated["supporting_evidence"][0]["quote"], run_agent1.STMTS[sid]["statement"])

    def test_missing_quote_is_hydrated_and_passes(self):
        sid = node()["scope"][0]
        hydrated, _, valid, failures, _ = self.hydrate_and_validate(reply(sid, None))
        self.assertTrue(valid, failures)
        self.assertEqual(hydrated["supporting_evidence"][0]["quote"], run_agent1.STMTS[sid]["statement"])

    def test_hydration_resolves_ssc_sepsis3_and_table_ids(self):
        for sid in ("stmt_005", "sepsis3::sofa_table", "table_03"):
            with self.subTest(statement_id=sid):
                hydrated, _, valid, failures, _ = self.hydrate_and_validate(
                    reply(sid, "model wording")
                )
                self.assertTrue(valid, failures)
                self.assertEqual(
                    hydrated["supporting_evidence"][0]["quote"],
                    run_agent1.STMTS[sid]["statement"],
                )

    def test_invalid_statement_id_still_fails(self):
        _, audit, valid, failures, _ = self.hydrate_and_validate(reply("invalid::statement", "invented"))
        self.assertFalse(valid)
        self.assertEqual(audit, [])
        self.assertTrue(any(item.startswith("V2_out_of_scope") for item in failures))

    def test_valid_but_out_of_scope_statement_id_still_fails(self):
        current_node = node()
        sid = next(item for item in run_agent1.STMTS if item not in current_node["scope"])
        _, audit, valid, failures, _ = self.hydrate_and_validate(
            reply(sid, run_agent1.STMTS[sid]["statement"]), current_node
        )
        self.assertFalse(valid)
        self.assertEqual(audit, [])
        self.assertTrue(any(item.startswith("V2_out_of_scope") for item in failures))

    def test_scoped_but_unresolved_statement_id_still_fails(self):
        current_node = node()
        current_node["scope"] = [*current_node["scope"], "test::unresolved"]
        _, audit, valid, failures, _ = self.hydrate_and_validate(
            reply("test::unresolved", "invented"), current_node
        )
        self.assertFalse(valid)
        self.assertEqual(audit, [])
        self.assertIn(
            "V2_statement_unresolved:supporting_evidence:test::unresolved",
            failures,
        )

    def test_null_statement_id_does_not_create_citation(self):
        hydrated, audit, valid, failures, stats = self.hydrate_and_validate(reply(None, None))
        evidence = hydrated["supporting_evidence"][0]
        self.assertTrue(valid, failures)
        self.assertEqual(audit, [])
        self.assertIsNone(evidence["statement_id"])
        self.assertIsNone(evidence["quote"])
        self.assertNotIn("quote_source", evidence)
        self.assertEqual(stats["uncited"], 1)

    def test_supporting_and_opposing_evidence_are_both_hydrated(self):
        current_node = node()
        first, second = current_node["scope"][:2]
        value = reply(first, "support paraphrase")
        value["opposing_evidence"] = [{
            "patient_finding": "synthetic opposing finding",
            "alternative_explanation": "synthetic alternative",
            "statement_id": second,
            "quote": "opposing paraphrase",
        }]
        hydrated, audit, valid, failures, _ = self.hydrate_and_validate(value, current_node)
        self.assertTrue(valid, failures)
        self.assertEqual(len(audit), 2)
        self.assertEqual(hydrated["supporting_evidence"][0]["quote"], run_agent1.STMTS[first]["statement"])
        self.assertEqual(hydrated["opposing_evidence"][0]["quote"], run_agent1.STMTS[second]["statement"])

    def test_hydration_provenance_is_recorded_in_trace(self):
        if not run_agent1.CASES:
            self.skipTest("needs the local phase0 pilot cases (patient data, RADAR_input/MIMIC)")
        current_node = node("R1")
        sid = current_node["scope"][0]
        model_reply = reply(sid, "model wording")
        case = copy.deepcopy(run_agent1.CASES[0])
        args = SimpleNamespace(
            dry_run=False,
            model="fixture",
            temperature=0.0,
            seed=7,
            decision_time="T0",
            risk_signal="none",
        )
        with mock.patch.object(
            run_agent1,
            "call_model",
            return_value=(json.dumps(model_reply), 0),
        ):
            result = run_agent1.run_case(case, None, args)
        first_step = result["trace"][0]
        hydration = first_step["attempts"][0]["quote_hydrations"][0]
        self.assertEqual(hydration["model_quote"], "model wording")
        self.assertEqual(hydration["quote_source"], "deterministic_statement_lookup")
        self.assertTrue(first_step["reply"]["supporting_evidence"][0]["quote_hydrated"])

    def test_historical_sofa_table_nonverbatim_quote_no_longer_fails_for_wording(self):
        value = reply(
            "sepsis3::sofa_table",
            "Sequential Organ Failure Assessment table, shortened by the model.",
        )
        _, _, valid, failures, _ = self.hydrate_and_validate(value)
        self.assertTrue(valid, failures)
        self.assertFalse(any(item.startswith("V3_quote_not_verbatim") for item in failures))


class TestAgent1LeadingAlternativeContract(unittest.TestCase):
    @staticmethod
    def value(current_node, *, include=True, leading=None):
        value = {
            "answer": "alternative_very_unlikely",
            "supporting_evidence": [],
            "opposing_evidence": [],
            "missing_information": [],
            "confidence": "moderate",
        }
        if include:
            value["leading_alternative"] = (
                run_agent1.alternative_candidate_ids(current_node)[0]
                if leading is None
                else leading
            )
        return value

    @staticmethod
    def validate(value, current_node):
        return run_agent1.validate(value, current_node, run_agent1.materials(current_node))

    def test_required_valid_leading_alternative_passes(self):
        for node_id in ("R3a", "R3b"):
            with self.subTest(node_id=node_id):
                current_node = node(node_id)
                valid, failures, _ = self.validate(self.value(current_node), current_node)
                self.assertTrue(valid, failures)

    def test_omitted_leading_alternative_fails(self):
        current_node = node("R3b")
        valid, failures, _ = self.validate(
            self.value(current_node, include=False), current_node
        )
        self.assertFalse(valid)
        self.assertIn("V1_leading_alternative_not_allowed:None", failures)

    def test_null_leading_alternative_fails(self):
        current_node = node("R3b")
        value = self.value(current_node)
        value["leading_alternative"] = None
        valid, failures, _ = self.validate(value, current_node)
        self.assertFalse(valid)
        self.assertIn("V1_leading_alternative_not_allowed:None", failures)

    def test_invalid_leading_alternative_category_fails(self):
        current_node = node("R3a")
        valid, failures, _ = self.validate(
            self.value(current_node, leading="not_a_graph_category"), current_node
        )
        self.assertFalse(valid)
        self.assertIn(
            "V1_leading_alternative_not_allowed:'not_a_graph_category'",
            failures,
        )

    def test_r3_prompt_and_schema_explicitly_require_field(self):
        if not run_agent1.CASES:
            self.skipTest("needs the local phase0 pilot cases (patient data, RADAR_input/MIMIC)")
        self.assertIn('"leading_alternative"', run_agent1.SYSTEM)
        for node_id in ("R3a", "R3b"):
            with self.subTest(node_id=node_id):
                current_node = node(node_id)
                prompt = run_agent1.user_prompt(
                    current_node,
                    run_agent1.CASES[0],
                    run_agent1.materials(current_node),
                )
                self.assertIn("MUST include BOTH `answer` and `leading_alternative`", prompt)
                self.assertIn("cannot be null and cannot be omitted", prompt)
                self.assertIn("will fail V1 validation", prompt)
                self.assertIn('"leading_alternative": "other_noninfectious"', prompt)


class TestEvaluationAbstentionExport(unittest.TestCase):
    @staticmethod
    def envelope(status):
        return {
            "case_id": "ABSTENTION_EXPORT",
            "status": status,
            "agent1_t0": {"result": {"tier": "possible_sepsis"}, "trace": []},
            "agent4_t0": {"result": {"status": "continue_workup", "continue_workup": True}},
            "checkpoints": [],
            "evaluation_only": None,
        }

    def test_agent1_t0_terminal_abstain_exports_true(self):
        envelope = self.envelope("upstream_abstain")
        envelope["stopped_at"] = "agent1_t0"
        envelope["agent1_t0"]["result"]["tier"] = "abstained"
        self.assertTrue(build_case_evaluation_export(envelope)["abstention"])

    def test_later_agent_terminal_abstain_exports_true(self):
        envelope = self.envelope("upstream_abstain")
        envelope["stopped_at"] = "C1"
        envelope["checkpoints"] = [{
            "checkpoint_id": "C1",
            "cutoff_time": 1.0,
            "checkpoint_status": "processed",
            "agent1_reassessment": {"result": {"tier": "possible_sepsis"}, "trace": []},
            "agent3": {"result": {"status": "abstain"}},
            "agent4": None,
        }]
        self.assertTrue(build_case_evaluation_export(envelope)["abstention"])

    def test_continue_workup_is_not_case_level_abstention(self):
        envelope = self.envelope("checkpoint_not_ready")
        self.assertFalse(build_case_evaluation_export(envelope)["abstention"])

    def test_skipped_checkpoint_is_not_case_level_abstention(self):
        envelope = self.envelope("checkpoint_not_ready")
        envelope["checkpoints"] = [{
            "checkpoint_id": "C1",
            "cutoff_time": 1.0,
            "checkpoint_status": "skipped",
            "skip_reason": "no_new_result",
        }]
        self.assertFalse(build_case_evaluation_export(envelope)["abstention"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
