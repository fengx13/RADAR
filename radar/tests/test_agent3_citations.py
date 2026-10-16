"""Synthetic citation-contract regressions; no patient data or model calls."""
from __future__ import annotations
from radar import paths

import copy
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from radar.agents import run_agent3 as a3
from radar.agents.agent3_citations import bind_evidence_quotes, citation_fragments
from radar.common.policy_integrity import ROOT, check_snapshot, restore_interface_changes


def node(name="TX3"):
    return {**a3.GRAPH["nodes"][name], "id": name}


def reply(sid="stmt_011", ref="statement", quote=None, answer="hold"):
    return {"answer": answer, "supporting_evidence": [{
        "patient_finding": "Synthetic current observation, not a real patient",
        "statement_id": sid, "quote_ref": ref, "quote": quote, "implication": "Synthetic reasoning"}],
        "opposing_evidence": [], "missing_information": ["synthetic unknown"], "confidence": "low"}


class CitationBindingTests(unittest.TestCase):
    def test_all_scoped_statements_and_remarks_bind_exactly_and_validate(self):
        for name in a3.GRAPH["nodes"]:
            spec = node(name)
            for field in ("supporting_evidence", "opposing_evidence"):
                for option in citation_fragments(spec, a3.STMTS):
                    raw = reply(option["statement_id"], option["quote_ref"], answer=spec["allowed_answers"][0])
                    if field == "opposing_evidence":
                        raw[field], raw["supporting_evidence"] = raw["supporting_evidence"], []
                    bound, bindings, errors = bind_evidence_quotes(raw, spec, a3.STMTS)
                    with self.subTest(node=name, field=field, option=option["quote_ref"]):
                        self.assertFalse(errors)
                        self.assertEqual(option["quote"], bound[field][0]["quote"])
                        self.assertEqual(option["quote_ref"], bindings[0]["quote_ref"])
                        self.assertTrue(a3.validate(bound, spec, a3.materials(spec))[0])

    def test_binding_preserves_model_decision_reasoning_and_original_input(self):
        raw = reply()
        before = copy.deepcopy(raw)
        statements, graph = copy.deepcopy(a3.STMTS), copy.deepcopy(a3.GRAPH)
        bound, bindings, errors = bind_evidence_quotes(raw, node(), a3.STMTS)
        self.assertEqual(before, raw)
        self.assertEqual(statements, a3.STMTS)
        self.assertEqual(graph, a3.GRAPH)
        for field in ("answer", "missing_information", "confidence"):
            self.assertEqual(raw[field], bound[field])
        for field in ("patient_finding", "implication", "statement_id", "quote_ref"):
            self.assertEqual(raw["supporting_evidence"][0][field], bound["supporting_evidence"][0][field])
        self.assertEqual("deterministic_scoped_fragment_lookup", bindings[0]["quote_source"])
        self.assertIsNone(bindings[0]["model_quote"])
        self.assertFalse(errors)

    def test_legacy_inserted_word_is_not_silently_repaired(self):
        bad = a3.STMTS["stmt_011"]["statement"].replace("bolus resuscitation", "bolus fluid resuscitation")
        raw = reply(ref=None, quote=bad)
        bound, bindings, errors = bind_evidence_quotes(raw, node(), a3.STMTS)
        self.assertEqual(raw, bound)
        self.assertEqual([], bindings)
        self.assertEqual([], errors)
        ok, failures, _ = a3.validate(bound, node(), a3.materials(node()))
        self.assertFalse(ok)
        self.assertIn("V3_quote_not_verbatim:supporting_evidence:stmt_011", failures)

    def test_explicit_ref_cannot_launder_conflicting_quote(self):
        bad = a3.STMTS["stmt_011"]["statement"].replace("bolus resuscitation", "bolus fluid resuscitation")
        bound, _, errors = bind_evidence_quotes(reply(quote=bad), node(), a3.STMTS)
        self.assertIsNone(bound)
        self.assertIn("V3_quote_ref_text_mismatch:supporting_evidence:stmt_011", errors)

    def test_scope_is_enforced_before_lookup(self):
        bound, bindings, errors = bind_evidence_quotes(reply(sid="stmt_007"), node(), a3.STMTS)
        self.assertIsNone(bound)
        self.assertFalse(bindings)
        self.assertIn("V2_out_of_scope:supporting_evidence:stmt_007", errors)

    def test_unknown_or_null_statement_cannot_select_fragment(self):
        for sid in ("unknown", None):
            with self.subTest(sid=sid):
                bound, _, errors = bind_evidence_quotes(reply(sid=sid), node(), a3.STMTS)
                self.assertIsNone(bound)
                self.assertTrue(errors)

    def test_unknown_remark_and_cross_statement_reference_are_rejected(self):
        for ref in ("remark_999", "stmt_007.statement", "statement+remark_0", "", "remark_-1"):
            with self.subTest(ref=ref):
                bound, _, errors = bind_evidence_quotes(reply(ref=ref), node(), a3.STMTS)
                self.assertIsNone(bound)
                self.assertTrue(any(e.startswith("V3_unknown_quote_ref") for e in errors))

    def test_invalid_reference_types_fail_closed_instead_of_crashing(self):
        for field in ("statement_id", "quote_ref", "quote"):
            for value in ([], {}, 7, True):
                raw = reply()
                raw["supporting_evidence"][0][field] = value
                with self.subTest(field=field, value=value):
                    bound, _, errors = bind_evidence_quotes(raw, node(), a3.STMTS)
                    self.assertIsNone(bound)
                    self.assertTrue(errors)

    def test_matching_explicit_text_and_legacy_excerpt_remain_valid(self):
        text = a3.STMTS["stmt_011"]["statement"]
        for raw in (reply(quote=text), reply(ref=None, quote=text), reply(ref=None, quote=text[:30])):
            bound, _, errors = bind_evidence_quotes(raw, node(), a3.STMTS)
            self.assertFalse(errors)
            self.assertTrue(a3.validate(bound, node(), a3.materials(node()))[0])

    def test_uncited_reasoning_does_not_acquire_a_citation(self):
        raw = reply(sid=None, ref=None)
        bound, bindings, errors = bind_evidence_quotes(raw, node(), a3.STMTS)
        self.assertEqual(raw, bound)
        self.assertFalse(bindings or errors)
        self.assertTrue(a3.validate(bound, node(), a3.materials(node()))[0])

    def test_binding_does_not_bypass_closed_answer_set(self):
        bound, _, errors = bind_evidence_quotes(reply(answer="made_up"), node(), a3.STMTS)
        self.assertFalse(errors)
        ok, failures, _ = a3.validate(bound, node(), a3.materials(node()))
        self.assertFalse(ok)
        self.assertTrue(any(e.startswith("V1_answer_not_allowed") for e in failures))

    def test_no_partial_candidate_when_one_citation_fails(self):
        raw = reply()
        raw["opposing_evidence"] = reply(sid="stmt_007")["supporting_evidence"]
        bound, bindings, errors = bind_evidence_quotes(raw, node(), a3.STMTS)
        self.assertIsNone(bound)
        self.assertEqual(1, len(bindings))
        self.assertTrue(errors)

    def test_missing_reference_does_not_backfill_a_missing_legacy_quote(self):
        raw = reply(ref=None)
        bound, bindings, errors = bind_evidence_quotes(raw, node(), a3.STMTS)
        self.assertFalse(bindings or errors)
        self.assertFalse(a3.validate(bound, node(), a3.materials(node()))[0])


class RunnerBindingTests(unittest.TestCase):
    def run_replies(self, replies):
        graph = copy.deepcopy(a3.GRAPH)
        graph["entry"] = "TX3"
        args = SimpleNamespace(dry_run=False, model="offline-fake", temperature=0, seed=7)
        case = {"case_id": "SYN_CITATION", "input": {"checkpoint": {"checkpoint_id": "C1", "cutoff_time": 1}}}
        with patch.object(a3, "GRAPH", graph), patch.object(a3, "call_model",
                side_effect=[(json.dumps(r), 0) for r in replies]) as transport:
            result = a3.run_case(case, None, args)
        return result, transport

    def map_reply(self):
        return reply(sid=None, ref=None, answer="not_applicable")

    def test_runner_binds_then_validates_and_records_raw_response(self):
        raw = reply()
        result, transport = self.run_replies([raw, self.map_reply()])
        self.assertEqual("complete", result["status"])
        self.assertEqual(2, transport.call_count)
        step = result["trace"][0]
        attempt = step["attempts"][0]
        self.assertEqual(a3.CITATION_CONTRACT, attempt["citation_contract"])
        self.assertEqual(raw, json.loads(attempt["raw"]))
        self.assertEqual(1, len(attempt["quote_bindings"]))
        self.assertEqual(a3.STMTS["stmt_011"]["statement"], step["reply"]["supporting_evidence"][0]["quote"])
        self.assertEqual("hold", result["result"]["treatment_plan"]["vasopressors"]["decision"])

    def test_invalid_lookup_gets_only_one_retry_then_abstains(self):
        bad = reply(ref="remark_999")
        result, transport = self.run_replies([bad, bad])
        self.assertEqual("abstain", result["status"])
        self.assertEqual(2, transport.call_count)
        self.assertEqual(1, len(result["trace"]))
        self.assertEqual(2, len(result["trace"][0]["attempts"]))

    def test_reference_can_be_repaired_within_existing_retry_budget(self):
        result, transport = self.run_replies([reply(ref="unknown"), reply(), self.map_reply()])
        self.assertEqual("complete", result["status"])
        self.assertEqual(3, transport.call_count)
        self.assertEqual([False, True], [a["ok"] for a in result["trace"][0]["attempts"]])

    def test_free_text_error_still_abstains_in_active_runner(self):
        bad = reply(ref=None, quote=a3.STMTS["stmt_011"]["statement"] + " Invented words.")
        result, transport = self.run_replies([bad, bad])
        self.assertEqual("abstain", result["status"])
        self.assertEqual(2, transport.call_count)
        self.assertTrue(all(not a["quote_bindings"] for a in result["trace"][0]["attempts"]))


class FrozenPolicyTests(unittest.TestCase):
    def test_clinical_graph_and_guidelines_pass_original_snapshot(self):
        self.assertEqual([], check_snapshot()["mismatches"])

    def test_lookup_does_not_allow_validator_tampering_in_policy_snapshot(self):
        path = "radar/agents/run_agent3.py"
        source = (ROOT / path).read_text(encoding="utf-8")
        changed = source.replace("if answer not in node[\"allowed_answers\"]:", "if False:", 1)
        self.assertNotEqual(source, changed)
        restored = restore_interface_changes(path, changed)
        expected = json.loads((paths.KNOWLEDGE / "policy_snapshot.json").read_text())["files"][path]
        candidates = [] if restored is None else [restored.encode(), restored.replace("\n", "\r\n").encode()]
        self.assertFalse(any(hashlib.sha256(c).hexdigest() == expected for c in candidates))


if __name__ == "__main__":
    unittest.main()
