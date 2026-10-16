"""Offline citation/availability regressions; synthetic except optional local replay."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from radar.agents import run_agent3 as a3
from radar.agents import run_agent4 as a4
from radar.agents.agent4_insufficiency import field_availability, route_insufficiency, validate_insufficiency
from radar.pipeline.longitudinal_integration import build_workup_state, workup_state_as_of
from test_agent4 import base_payload


def node(module, name):
    return dict(module.GRAPH["nodes"][name], id=name)


def counterfactual():
    return {
        "if_a": {"condition": "reassuring assessment", "disposition": "discharge"},
        "if_b": {"condition": "concerning assessment", "disposition": "ward"},
    }


def insufficient(field):
    return {"answer": "insufficient_evidence", "blocking": [
        {"field": field, "counterfactual": counterfactual()}]}


def cited_reply(spec, sid, quote):
    return {
        "answer": spec["allowed_answers"][0],
        "supporting_evidence": [{"patient_finding": "synthetic fixture", "statement_id": sid,
                                 "quote": quote, "implication": "synthetic validator check"}],
        "opposing_evidence": [], "missing_information": [], "confidence": "low",
    }


class CitationCopyContractTests(unittest.TestCase):
    def test_copy_fragments_are_exact_and_scoped(self):
        for name in a3.GRAPH["nodes"]:
            spec = node(a3, name)
            contract = a3.citation_copy_contract(spec)
            fragments, _ = json.JSONDecoder().raw_decode(contract[contract.index("["):])
            expected = [{"statement_id": sid, "quote_ref": ref, "quote": quote} for sid in spec["scope"]
                        for ref, quote in [("statement", a3.STMTS[sid]["statement"]),
                                           *[(f"remark_{i}", text) for i, text in
                                             enumerate(a3.STMTS[sid].get("remarks") or [])]]]
            self.assertEqual(expected, fragments)
            for fragment in fragments:
                ok, errors, _ = a3.validate(cited_reply(spec, fragment["statement_id"], fragment["quote"]),
                                           spec, a3.materials(spec))
                self.assertTrue(ok, errors)

    def test_display_label_still_fails_v3(self):
        spec = node(a3, "TX3")
        source = a3.STMTS["stmt_011"]
        bad = source["statement"] + " Remark: " + source["remarks"][0]
        ok, errors, _ = a3.validate(cited_reply(spec, "stmt_011", bad), spec, a3.materials(spec))
        self.assertFalse(ok)
        self.assertIn("V3_quote_not_verbatim:supporting_evidence:stmt_011", errors)

    def test_inserted_word_still_fails_v3(self):
        spec = node(a3, "TX3")
        source = a3.STMTS["stmt_011"]["statement"]
        bad = source.replace("bolus resuscitation", "bolus fluid resuscitation")
        self.assertNotEqual(source, bad)
        self.assertFalse(a3.validate(cited_reply(spec, "stmt_011", bad), spec, a3.materials(spec))[0])

    def test_out_of_scope_copy_still_fails_v2(self):
        spec = node(a3, "TX3")
        sid = next(sid for sid in a3.STMTS if sid not in spec["scope"])
        ok, errors, _ = a3.validate(cited_reply(spec, sid, a3.STMTS[sid]["statement"]), spec, a3.materials(spec))
        self.assertFalse(ok)
        self.assertTrue(any(error.startswith("V2_out_of_scope") for error in errors))

    def test_prompt_separates_copying_from_applicability(self):
        spec = node(a3, "TX3")
        prompt = a3.user_prompt(spec, {"input": {}}, a3.materials(spec))
        self.assertIn(a3.citation_copy_contract(spec), prompt)
        self.assertIn("These fragments do not establish a patient condition", prompt)
        self.assertIn("non-verbatim quotes still fail V3", prompt)
        self.assertIn("Return quote=null", prompt)
        self.assertIn(spec["question"], prompt)
        self.assertIn(str(spec["allowed_answers"]), prompt)

    def test_prompt_has_one_quote_source_without_legacy_display_labels(self):
        for name in a3.GRAPH["nodes"]:
            spec = node(a3, name)
            prompt = a3.user_prompt(spec, {"input": {}}, "LEGACY_RENDERER_MUST_NOT_BE_INJECTED")
            self.assertNotIn("LEGACY_RENDERER_MUST_NOT_BE_INJECTED", prompt)
            self.assertNotIn("Remark:", prompt)
            self.assertEqual(1, prompt.count("CITATION LOOKUP OPTIONS"))
            self.assertEqual(1, prompt.count("STATEMENT METADATA"))

    def test_single_source_preserves_statement_strength_and_certainty(self):
        for name in a3.GRAPH["nodes"]:
            spec = node(a3, name)
            contract = a3.citation_copy_contract(spec)
            start = contract.index("STATEMENT METADATA (not quote text)\n")
            metadata, _ = json.JSONDecoder().raw_decode(contract[contract.index("{", start):])
            self.assertEqual({sid: {"strength": a3.STMTS[sid].get("strength"),
                                    "certainty": a3.STMTS[sid].get("certainty")}
                              for sid in spec["scope"]}, metadata)

    def test_single_source_does_not_mutate_knowledge_or_graph(self):
        statements, graph = copy.deepcopy(a3.STMTS), copy.deepcopy(a3.GRAPH)
        for name in a3.GRAPH["nodes"]:
            spec = node(a3, name)
            a3.user_prompt(spec, {"input": {}}, a3.materials(spec))
        self.assertEqual(statements, a3.STMTS)
        self.assertEqual(graph, a3.GRAPH)

    def test_legacy_renderer_and_uncited_reasoning_remain_supported(self):
        spec = node(a3, "TX3")
        self.assertIn("Remark:", a3.materials(spec))
        ok, errors, _ = a3.validate(cited_reply(spec, None, None), spec, a3.materials(spec))
        self.assertTrue(ok, errors)


class ObservedPendingContractTests(unittest.TestCase):
    def setUp(self):
        self.payload = a4.normalize_payload(base_payload())
        self.payload["workup_state"] = {"items": [{
            "item": "blood_cultures", "agent2_recommendation": "recommend",
            "actual_order_status": "unavailable", "result_status": "unavailable",
        }], "pending_workup": []}
        self.spec = node(a4, "D3")

    def validate(self, field):
        return validate_insufficiency(insufficient(field), self.spec, self.payload)

    def mark_order_pending(self):
        self.payload["workup_state"]["items"][0].update(
            actual_order_status="collected", result_status="pending")
        self.payload["workup_state"]["pending_workup"] = ["blood_cultures"]

    def test_empty_pending_list_is_not_missing(self):
        ok, errors = self.validate("workup_state.pending_workup")
        self.assertFalse(ok)
        self.assertIn("V4b_pending_not_observed:workup_state.pending_workup", errors)

    def test_nonempty_observed_pending_list_can_block(self):
        self.mark_order_pending()
        self.assertTrue(self.validate("workup_state.pending_workup")[0])
        self.assertEqual("OBSERVATION", route_insufficiency(insufficient("workup_state.pending_workup"))[0])

    def test_recommendation_is_not_a_pending_order(self):
        self.assertFalse(self.validate("patient_state.labs.culture_result")[0])

    def test_observed_pending_culture_can_block(self):
        self.mark_order_pending()
        self.assertTrue(self.validate("patient_state.labs.culture_result")[0])

    def test_pending_list_must_match_observed_items_when_supplied(self):
        self.payload["workup_state"]["pending_workup"] = ["blood_cultures"]
        self.assertFalse(self.validate("workup_state.pending_workup")[0])

    def test_documented_order_without_result_remains_pending(self):
        self.mark_order_pending()
        self.payload["workup_state"]["items"][0].update(actual_order_status="ordered", result_status="unavailable")
        self.assertTrue(self.validate("patient_state.labs.culture_result")[0])

    def test_not_ordered_or_cancelled_is_not_pending(self):
        for status in ("not_ordered", "cancelled", "unavailable"):
            with self.subTest(status=status):
                self.mark_order_pending()
                self.payload["workup_state"]["items"][0]["actual_order_status"] = status
                self.assertFalse(self.validate("patient_state.labs.culture_result")[0])

    def test_resulted_item_is_not_pending(self):
        self.mark_order_pending()
        self.payload["workup_state"]["items"][0]["result_status"] = "resulted"
        self.assertFalse(self.validate("patient_state.labs.culture_result")[0])

    def test_all_three_lab_mappings_require_observed_pending(self):
        for leaf, item in (("culture_result", "blood_cultures"), ("lactate_repeat", "lactate_repeat"),
                           ("imaging_result", "source_imaging_or_sampling")):
            with self.subTest(leaf=leaf):
                self.payload["workup_state"]["items"][0]["item"] = item
                self.payload["workup_state"]["items"][0].update(actual_order_status="unavailable", result_status="unavailable")
                self.assertFalse(self.validate("patient_state.labs." + leaf)[0])
                self.payload["workup_state"]["items"][0].update(actual_order_status="ordered", result_status="pending")
                self.assertTrue(self.validate("patient_state.labs." + leaf)[0])

    def test_malformed_observed_items_do_not_crash(self):
        self.payload["workup_state"]["pending_workup"] = ["blood_cultures"]
        self.payload["workup_state"]["items"] = [None, "bad", {
            "item": [], "actual_order_status": "ordered", "result_status": "pending"}]
        self.assertFalse(self.validate("workup_state.pending_workup")[0])

    def test_unavailable_response_is_not_pending(self):
        self.payload["treatment_state"]["treatment_response_flag"] = "unavailable"
        self.payload["patient_state"]["treatment_response"] = "unavailable"
        for field in ("patient_state.treatment_response", "treatment_state.treatment_response_flag"):
            self.assertFalse(self.validate(field)[0])

    def test_explicit_pending_response_keeps_existing_route(self):
        self.payload["treatment_state"]["treatment_response_flag"] = "pending"
        reply = insufficient("treatment_state.treatment_response_flag")
        self.assertTrue(validate_insufficiency(reply, self.spec, self.payload)[0])
        self.assertEqual("OBSERVATION", route_insufficiency(reply)[0])

    def test_initial_lactate_status_contract_remains_structural(self):
        self.payload["patient_state"]["labs"]["lactate_initial"] = {
            "status": "unavailable", "value": None, "sample_time": None, "result_time": None}
        reply = insufficient("patient_state.labs.lactate_initial")
        ok, errors = validate_insufficiency(reply, node(a4, "D2"), self.payload)
        self.assertFalse(ok)
        self.assertTrue(any(error.startswith("V4b_field_is_present:") for error in errors))
        self.assertIsNone(a4.validate_d0(self.payload))

    def test_truly_missing_initial_lactate_keeps_alarm_route(self):
        self.payload["patient_state"]["labs"]["lactate_initial"] = None
        reply = insufficient("patient_state.labs.lactate_initial")
        self.assertTrue(validate_insufficiency(reply, node(a4, "D2"), self.payload)[0])
        outcome, detail = route_insufficiency(reply)
        self.assertEqual("ABSTAIN", outcome)
        self.assertTrue(detail["pipeline_alarm"])

    def test_legacy_unspecified_null_is_not_silently_reclassified(self):
        self.payload["workup_state"].pop("items")
        self.assertTrue(self.validate("patient_state.labs.culture_result")[0])

    def test_yana_legacy_result_rows_are_not_full_observed_order_records(self):
        self.payload["workup_state"] = {
            "items": [{"item": "serum_lactate", "status": "resulted", "result_time": 1}],
            "pending_workup": ["blood_cultures"],
        }
        self.assertTrue(self.validate("patient_state.labs.culture_result")[0])
        self.assertTrue(self.validate("workup_state.pending_workup")[0])
        row = field_availability(self.payload, {"patient_state.labs.culture_result"})[0]
        self.assertIsNone(row["observed_pending"])

    def test_future_order_cannot_establish_pending_at_current_checkpoint(self):
        workup = build_workup_state(None, [{"item": "blood_cultures", "actual_order_status": "ordered",
                                          "order_time": 2, "result_status": "pending"}])
        self.payload["workup_state"] = workup_state_as_of(workup, 1, "C1")
        self.assertFalse(self.validate("patient_state.labs.culture_result")[0])
        self.assertFalse(self.validate("workup_state.pending_workup")[0])

    def test_availability_view_does_not_mutate_inputs(self):
        original = copy.deepcopy(self.payload)
        field_availability(self.payload, {"patient_state.labs.culture_result", "workup_state.pending_workup"})
        self.assertEqual(original, self.payload)

    def test_counterfactual_requirements_still_apply(self):
        self.mark_order_pending()
        reply = insufficient("workup_state.pending_workup")
        reply["blocking"][0]["counterfactual"]["if_a"]["condition"] = ""
        self.assertFalse(validate_insufficiency(reply, self.spec, self.payload)[0])
        reply = insufficient("workup_state.pending_workup")
        reply["blocking"][0]["counterfactual"]["if_b"]["disposition"] = "discharge"
        self.assertFalse(validate_insufficiency(reply, self.spec, self.payload)[0])

    def test_d3_yes_basis_cannot_bypass_observed_pending_check(self):
        reply = {"answer": "yes", "key_reasons": [], "unresolved_risks": [],
                 "missing_information": [], "escalation_trigger": None,
                 "observation_basis": insufficient("patient_state.labs.culture_result")["blocking"][0],
                 "blocking": []}
        ok, errors = a4.validate_reply(reply, self.spec, self.payload)
        self.assertFalse(ok)
        self.assertTrue(any("observation_basis_V4b_pending_not_observed" in error for error in errors))

    def test_prompt_explains_same_availability_without_future_fields(self):
        fields = {"workup_state.pending_workup", "patient_state.labs.culture_result"}
        rows = field_availability(self.payload, fields)
        self.assertTrue(all(row["v4b_unresolved"] is False for row in rows))
        self.payload["truth"] = {"dsp_mortality_inhospital": "HIDDEN_OUTCOME_SENTINEL"}
        self.payload["workup_state"]["excluded_post_checkpoint_results"] = 123
        prompt = a4.user_prompt(self.spec, self.payload)
        self.assertIn("V4b FIELD AVAILABILITY", prompt)
        self.assertIn("An A2 recommendation is not an observed order", prompt)
        self.assertNotIn("HIDDEN_OUTCOME_SENTINEL", prompt)
        self.assertNotIn("excluded_post_checkpoint", prompt)


LOCAL_RUN = Path(__file__).resolve().parent.parent / (
    "local_pilot_vnext/runs/"
    "2112fe65274e8c8b82347da13d2187d6e902b2a02cf04d51bcaa7ebaef3bcb96.json")


@unittest.skipUnless(LOCAL_RUN.is_file(), "Optional private recorded run is not distributed")
class LocalRecordedReplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorded_run = json.loads(LOCAL_RUN.read_text(encoding="utf-8"))

    def checkpoint(self, case_id):
        row = next(row for row in self.recorded_run["results"] if row["case_id"] == case_id)
        return next(cp for cp in row["result"]["checkpoints"] if cp["checkpoint_status"] == "processed")

    def test_recorded_s2a_observation_is_now_rejected_not_relabelled(self):
        cp = self.checkpoint("S2a")
        trace = next(t for t in cp["agent4"]["trace"] if t["node"] == "D3")
        ok, errors = a4.validate_reply(copy.deepcopy(trace["reply"]), node(a4, "D3"), cp["agent4_input"])
        self.assertFalse(ok)
        self.assertTrue(any("V4b_pending_not_observed" in error for error in errors))
        self.assertEqual("observation", cp["agent4"]["result"]["disposition"])

    def test_recorded_s1a_pending_list_is_recognized(self):
        cp = self.checkpoint("S1a")
        reply = insufficient("workup_state.pending_workup")
        self.assertTrue(validate_insufficiency(reply, node(a4, "D2"), cp["agent4_input"])[0])

    def test_six_failed_recorded_tx3_quotes_are_not_auto_repaired(self):
        count = 0
        for record in self.recorded_run["budget"]["requests"]:
            scope = record["node"].split(":")
            if scope[0] not in {"S2b", "S3a", "S4b"} or scope[2] != "A3" or scope[-1] != "TX3":
                continue
            spec = node(a3, "TX3")
            self.assertFalse(a3.validate(json.loads(record["response_text"]), spec, a3.materials(spec))[0])
            count += 1
        self.assertEqual(6, count)


RETEST_RUN = LOCAL_RUN.with_name("a8a44f4a9f8c9eead1e7456492edc61228c276ceb55d0fe697805bafa59e157f.json")


@unittest.skipUnless(RETEST_RUN.is_file(), "Optional private retest is not distributed")
class LocalRetestQuoteTests(unittest.TestCase):
    def test_retest_s4b_invalid_quotes_remain_invalid(self):
        saved = json.loads(RETEST_RUN.read_text(encoding="utf-8"))
        quotes = [row for row in saved["budget"]["requests"] if row["node"] == "S4b:baseline:A3:C3:TX3"]
        self.assertEqual(2, len(quotes))
        spec = node(a3, "TX3")
        for row in quotes:
            ok, failures, _ = a3.validate(json.loads(row["response_text"]), spec, a3.materials(spec))
            self.assertFalse(ok)
            self.assertIn("V3_quote_not_verbatim:supporting_evidence:stmt_011", failures)


if __name__ == "__main__":
    unittest.main()
