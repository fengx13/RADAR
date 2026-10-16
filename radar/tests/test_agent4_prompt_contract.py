"""Offline regressions for the two captured synthetic A4_BRANCH_02 replies."""
from __future__ import annotations
from radar import paths

import copy
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from radar.agents import run_agent4 as a4
from radar.agents.agent4_insufficiency import CANONICAL_PENDING, VALID_DISPOSITIONS, canonical_field


def section_json(prompt, title, *, skip_lines=0):
    body = prompt.split(title + "\n" + "=" * 60 + "\n", 1)[1]
    body = body.split("\n", skip_lines)[skip_lines]
    return json.JSONDecoder().raw_decode(body)[0]


class TestA4PromptContract(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((a4.paths.DATA / "agent4_contract_regression_replies.json").read_text(encoding="utf-8"))
        cases = json.loads((a4.ROOT / self.fixture["source_case_fixture"]).read_text(encoding="utf-8"))["cases"]
        self.case = next(case for case in cases if case["case_id"] == self.fixture["case_id"])
        self.d2 = dict(a4.GRAPH["nodes"]["D2"], id="D2")
        self.args = SimpleNamespace(dry_run=True, model="dry", temperature=0)

    def syntax_only_reply(self, attempt=0):
        # An artificial positive control, NOT a correction of the saved API run
        # or an adjudication of which clinical counterfactual is appropriate.
        reply = copy.deepcopy(self.fixture["attempts"][attempt]["reply"])
        reply["blocking"][0]["field"] = "patient_state.labs.lactate_repeat"
        for item in reply["blocking"]:
            item["counterfactual"]["if_a"]["disposition"] = "discharge"
            item["counterfactual"]["if_b"]["disposition"] = "ward"
        return reply

    def test_blocking_lists_match_current_node_not_all_evidence_paths(self):
        for node_id in ("D2", "D3", "D4"):
            with self.subTest(node=node_id):
                node = dict(a4.GRAPH["nodes"][node_id], id=node_id)
                fields = section_json(a4.user_prompt(node, self.case), "NODE-SPECIFIC V4 FIELD SETS")
                expected = {canonical_field(path) for path in node["blockable_fields"]}
                self.assertEqual(expected, set(fields["blocking.field"]))
                self.assertEqual(expected & CANONICAL_PENDING if node_id == "D3" else set(),
                                 set(fields["observation_basis.field"]))

    def test_status_leaf_remains_evidence_but_is_not_a_blocker(self):
        prompt = a4.user_prompt(self.d2, self.case)
        evidence = section_json(prompt, "SCHEMA-GROUNDED EVIDENCE FIELD PATHS", skip_lines=1)
        fields = section_json(prompt, "NODE-SPECIFIC V4 FIELD SETS")
        self.assertIn("patient_state.labs.lactate_repeat.status", evidence)
        self.assertNotIn("patient_state.labs.lactate_repeat.status", fields["blocking.field"])
        self.assertIn("patient_state.labs.lactate_repeat", fields["blocking.field"])
        self.assertIn("This is NOT the blocking-field allowlist", prompt)

    def test_contract_uses_selected_node_and_canonicalizes_legacy_alias(self):
        node = {**self.d2, "blockable_fields": ["t1_state.labs.lactate_repeat"]}
        fields = section_json(a4.user_prompt(node, self.case), "NODE-SPECIFIC V4 FIELD SETS")
        self.assertEqual(["patient_state.labs.lactate_repeat"], fields["blocking.field"])
        node["blockable_fields"] = []
        fields = section_json(a4.user_prompt(node, self.case), "NODE-SPECIFIC V4 FIELD SETS")
        self.assertEqual([], fields["blocking.field"])

    def test_counterfactual_enum_matches_validator_and_rejects_ambiguous_examples(self):
        prompt = a4.user_prompt(self.d2, self.case)
        enum_text = prompt.split("COUNTERFACTUAL DISPOSITION ENUM (one string per branch): ", 1)[1]
        values = json.JSONDecoder().raw_decode(enum_text)[0]
        self.assertEqual(VALID_DISPOSITIONS, set(values))
        self.assertIn("ONE exact, case-sensitive value", a4.SYSTEM)
        self.assertIn("if_a.disposition must differ from if_b.disposition", prompt)
        self.assertIn('Invalid: "discharge or observation", "ward or ICU", "ward admission"', prompt)
        self.assertNotIn('"disposition": "discharge|observation|ward|ICU"', a4.SYSTEM)

    def test_hidden_outcomes_and_mock_targets_do_not_affect_prompt(self):
        before = a4.user_prompt(self.d2, self.case)
        self.case["evaluation_only"]["secret"] = "HIDDEN_TARGET_SENTINEL"
        self.case["mock_llm_answers"] = {"D2": {"answer": "HIDDEN_MOCK_SENTINEL"}}
        self.case["outcomes"] = {"secret": "HIDDEN_OUTCOME_SENTINEL"}
        self.case["truth"] = {"secret": "HIDDEN_TRUTH_SENTINEL"}
        self.case["patient_state"]["dsp_icu_transfer_12h"] = 1
        self.assertEqual(before, a4.user_prompt(self.d2, self.case))

    def test_both_captured_replies_still_fail_with_exact_original_errors(self):
        for attempt in self.fixture["attempts"]:
            with self.subTest(attempt=attempt["attempt"]):
                original = copy.deepcopy(attempt["reply"])
                ok, failures = a4.validate_reply(attempt["reply"], self.d2, self.case)
                self.assertFalse(ok)
                self.assertEqual(attempt["expected_failures"], failures)
                self.assertEqual(original, attempt["reply"])

    def test_captured_attempts_replay_to_abstention_with_no_third_attempt(self):
        replies = [(json.dumps(item["reply"]), 0) for item in self.fixture["attempts"]]
        args = SimpleNamespace(dry_run=False, model="fake-offline", temperature=0)
        with mock.patch.object(a4, "call_model", side_effect=replies) as call:
            result = a4.run_case(self.case, None, args)
        self.assertEqual(2, call.call_count)
        self.assertEqual("T_ABSTAIN", result["terminal"])
        self.assertIsNone(result["result"]["disposition"])
        self.assertFalse(result["result"]["pipeline_alarm"])
        self.assertEqual("invalid_model_output", result["result"]["abstain_reason"])
        self.assertEqual(["D1", "D2"], [step["node"] for step in result["trace"]])
        for actual, captured in zip(result["trace"][-1]["attempts"], self.fixture["attempts"]):
            self.assertEqual(captured["expected_failures"], actual["failures"])
        repair_prompt = call.call_args_list[1].args[2]
        for failure in self.fixture["attempts"][0]["expected_failures"]:
            self.assertIn(failure, repair_prompt)

    def test_syntax_only_positive_controls_validate_without_changing_capture(self):
        before = copy.deepcopy(self.fixture)
        for index in range(2):
            ok, failures = a4.validate_reply(self.syntax_only_reply(index), self.d2, self.case)
            self.assertTrue(ok, failures)
        self.assertEqual(before, self.fixture)

    def test_fixing_only_enum_still_rejects_status_as_blocker(self):
        reply = self.syntax_only_reply()
        reply["blocking"][0]["field"] = "patient_state.labs.lactate_repeat.status"
        ok, failures = a4.validate_reply(reply, self.d2, self.case)
        self.assertFalse(ok)
        self.assertEqual(["V4a_not_blockable:patient_state.labs.lactate_repeat.status"], failures)

    def test_fixing_only_field_still_rejects_compound_or_alias_dispositions(self):
        for invalid in ("discharge or observation", "ward or ICU", "ward admission", "WARD"):
            with self.subTest(value=invalid):
                reply = self.syntax_only_reply()
                reply["blocking"][0]["counterfactual"]["if_b"]["disposition"] = invalid
                ok, failures = a4.validate_reply(reply, self.d2, self.case)
                self.assertFalse(ok)
                self.assertIn("V4c_counterfactual_disposition_not_allowed:patient_state.labs.lactate_repeat", failures)

    def test_same_disposition_counterfactual_is_still_rejected(self):
        reply = self.syntax_only_reply()
        reply["blocking"][0]["counterfactual"]["if_b"]["disposition"] = "discharge"
        ok, failures = a4.validate_reply(reply, self.d2, self.case)
        self.assertFalse(ok)
        self.assertIn("V4c_counterfactual_not_decisive:patient_state.labs.lactate_repeat", failures)

    def test_present_result_cannot_be_claimed_as_missing(self):
        self.case["patient_state"]["labs"]["lactate_repeat"] = {"status": "resulted", "value": 1.7}
        ok, failures = a4.validate_reply(self.syntax_only_reply(), self.d2, self.case)
        self.assertFalse(ok)
        self.assertTrue(any(failure.startswith("V4b_field_is_present:") for failure in failures))

    def test_valid_synthetic_d2_pending_route_is_observation_not_forced_d3(self):
        self.case["mock_llm_answers"]["D2"] = self.syntax_only_reply()
        result = a4.run_case(self.case, None, self.args)
        self.assertEqual("T_OBSERVATION", result["terminal"])
        self.assertEqual(["D1", "D2"], [step["node"] for step in result["trace"]])

    def test_d3_basis_remains_pending_only_and_requires_valid_enum(self):
        d3 = dict(a4.GRAPH["nodes"]["D3"], id="D3")
        reply = copy.deepcopy(self.case["mock_llm_answers"]["D3"])
        ok, failures = a4.validate_reply(reply, d3, self.case)
        self.assertTrue(ok, failures)
        reply["observation_basis"]["counterfactual"]["if_b"]["disposition"] = "ward admission"
        ok, failures = a4.validate_reply(reply, d3, self.case)
        self.assertFalse(ok)
        self.assertIn("observation_basis_V4c_counterfactual_disposition_not_allowed:patient_state.labs.lactate_repeat", failures)


if __name__ == "__main__":
    unittest.main()
