#!/usr/bin/env python3
"""Regression tests for the targeted failure-mode audit."""
from __future__ import annotations
from radar import paths

import json
import pathlib
import sys
import unittest


HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
sys.path.insert(0, str(HERE))

from radar.agents import run_agent4  # noqa: E402
from radar.evaluation.evaluation_export import build_case_evaluation_export  # noqa: E402
from radar.common.field_addressing import (  # noqa: E402
    available_field_paths,
    canonical_field_path,
    canonicalize_reply_fields,
    field_registry_metadata,
)
from radar.evaluation.reliability_audit import (  # noqa: E402
    classify_evidence_item,
    first_node_divergence,
    reliability_rows,
    stability_by_case,
)


def a4_payload() -> dict:
    data = json.loads((paths.DATA / "agent4_dryrun_cases.json").read_text(encoding="utf-8"))
    return (data.get("cases") or data.get("payloads"))[0]


def a4_reply(field: str) -> dict:
    return {
        "answer": "no",
        "key_reasons": [{"field": field, "detail": "fixture"}],
        "unresolved_risks": [],
        "missing_information": [],
        "escalation_trigger": None,
        "observation_basis": None,
        "blocking": [],
    }


def envelope(checkpoint_id="C1", cutoff=1.0) -> dict:
    checkpoints = [
        {
            "checkpoint_id": checkpoint_id,
            "cutoff_time": cutoff,
            "checkpoint_status": "processed",
            "agent1_reassessment": {"result": {"tier": "possible_sepsis"}, "trace": []},
            "agent3": {"result": {"status": "complete"}},
            "agent4": {"result": {"status": "decision", "disposition": "ward"}},
        }
    ]
    return {
        "case_id": "TIME",
        "t0_time": 5.0,
        "agent1_t0": {"result": {"tier": "possible_sepsis"}, "trace": []},
        "agent4_t0": {"result": {"status": "continue_workup"}, "trace": []},
        "checkpoints": checkpoints,
        "disposition_checkpoint": checkpoint_id,
        "final_simulated_disposition": "ward",
    }


def repeated_run(decisions, wordings=None, fingerprints=None) -> dict:
    wordings = wordings or ["same"] * len(decisions)
    fingerprints = fingerprints or ["fp_a"] * len(decisions)
    results = []
    for repeat, (decision, wording, fingerprint) in enumerate(zip(decisions, wordings, fingerprints)):
        results.append({
            "case_id": "X",
            "repeat": repeat,
            "result": {"tier": "possible_sepsis"},
            "trace": [{
                "node": "R1",
                "answer": decision,
                "reply": {"answer": decision, "missing_information": [wording]},
                "attempts": [{"attempt": 0, "ok": True, "system_fingerprint": fingerprint}],
            }],
        })
    return {"meta": {"agent": "agent1", "model": "fixture", "repeats": len(results)}, "results": results}


class TestFieldAddressing(unittest.TestCase):
    def test_01_canonical_field_registry_loads(self):
        self.assertTrue(field_registry_metadata())

    def test_02_unambiguous_alias_resolves(self):
        self.assertEqual(
            canonical_field_path("patient_state.treatment_response_flag"),
            "treatment_state.treatment_response_flag",
        )

    def test_03_nonexistent_field_is_rejected(self):
        node = dict(run_agent4.GRAPH["nodes"]["D2"], id="D2")
        ok, failures = run_agent4.validate_reply(a4_reply("investigation_status"), node, a4_payload())
        self.assertFalse(ok)
        self.assertTrue(any("field_not_allowed" in failure for failure in failures))
        metadata_ok, metadata_failures = run_agent4.validate_reply(a4_reply("case_id"), node, a4_payload())
        self.assertFalse(metadata_ok)
        self.assertTrue(any("field_not_allowed" in failure for failure in metadata_failures))

    def test_04_valid_field_path_passes(self):
        node = dict(run_agent4.GRAPH["nodes"]["D2"], id="D2")
        ok, failures = run_agent4.validate_reply(a4_reply("agent1.tier"), node, a4_payload())
        self.assertTrue(ok, failures)

    def test_05_prompt_whitelist_contains_canonical_path(self):
        node = dict(run_agent4.GRAPH["nodes"]["D2"], id="D2")
        prompt = run_agent4.user_prompt(node, a4_payload())
        self.assertIn("SCHEMA-GROUNDED EVIDENCE FIELD PATHS", prompt)
        self.assertIn("treatment_state.treatment_response_flag", prompt)

    def test_06_truth_outcome_paths_never_allowed(self):
        payload = a4_payload()
        payload["evaluation_only"] = {"dsp_mortality_inhospital": 1}
        paths = available_field_paths(run_agent4.normalize_payload(payload))
        self.assertFalse(any("mortality" in path or "evaluation_only" in path for path in paths))

    def test_07_alias_is_rewritten_and_audited(self):
        reply, repairs = canonicalize_reply_fields(a4_reply("patient_state.treatment_response_flag"))
        self.assertEqual(reply["key_reasons"][0]["field"], "treatment_state.treatment_response_flag")
        self.assertEqual(len(repairs), 1)
        node = dict(run_agent4.GRAPH["nodes"]["D3"], id="D3")
        pending_reply = {
            "answer": "insufficient_evidence",
            "key_reasons": [],
            "unresolved_risks": [],
            "missing_information": ["response pending"],
            "escalation_trigger": None,
            "observation_basis": None,
            "blocking": [{
                "field": "patient_state.treatment_response_flag",
                "counterfactual": {
                    "if_a": {"condition": "improves", "disposition": "discharge"},
                    "if_b": {"condition": "worsens", "disposition": "ward"},
                },
            }],
        }
        payload = a4_payload()
        payload["treatment_state"]["treatment_response_flag"] = "pending"
        ok, failures = run_agent4.validate_reply(pending_reply, node, payload)
        self.assertTrue(ok, failures)


class TestTimeAnchoring(unittest.TestCase):
    def test_08_t0_disposition_uses_t0_cutoff(self):
        item = envelope()
        item["disposition_checkpoint"] = "T0"
        item["final_simulated_disposition"] = "discharge"
        item["agent4_t0"] = {"result": {"status": "decision", "disposition": "discharge"}, "trace": []}
        self.assertEqual(build_case_evaluation_export(item)["time_to_disposition"], 5.0)

    def test_09_c1_disposition_uses_c1_cutoff(self):
        self.assertEqual(build_case_evaluation_export(envelope("C1", 1.0))["time_to_disposition"], 1.0)

    def test_10_c3_disposition_uses_c3_cutoff(self):
        self.assertEqual(build_case_evaluation_export(envelope("C3", 3.0))["time_to_disposition"], 3.0)

    def test_11_legacy_t1_is_not_silently_used_for_vnext(self):
        item = envelope("C3", 3.0)
        item["t1_time"] = 1.0
        self.assertEqual(build_case_evaluation_export(item)["time_to_disposition"], 3.0)

    def test_12_skipped_checkpoint_is_not_a_decision_point(self):
        item = envelope("C3", 3.0)
        item["checkpoints"].insert(0, {
            "checkpoint_id": "C1",
            "cutoff_time": 1.0,
            "checkpoint_status": "skipped",
            "skip_reason": "no_new_result",
            "agent4": {"result": {"status": "decision", "disposition": "ICU"}},
        })
        export = build_case_evaluation_export(item)
        self.assertEqual(export["time_to_disposition"], 3.0)
        self.assertIsNone(export["checkpoint_sequence"][0]["agent4_disposition"])


class TestGroundingClassification(unittest.TestCase):
    def test_13_patient_fact_is_distinguishable(self):
        result = classify_evidence_item({"patient_finding": "temperature measured"}, node="R1")
        self.assertEqual(result["category"], "patient_fact_no_guideline_citation_required")

    def test_14_kb_gap_differs_from_scope_miss(self):
        gap = classify_evidence_item(
            {"patient_finding": "dyspnea", "alternative_explanation": "heart failure"}, node="R3b"
        )
        miss = classify_evidence_item(
            {"statement_id": "stmt_x"}, node="R1", scope_ids=set(), knowledge_base_ids={"stmt_x"}
        )
        self.assertEqual(gap["category"], "known_mimic_kb_gap")
        self.assertEqual(miss["category"], "citation_scope_miss")

    def test_15_r3_mimic_domain_is_categorized(self):
        result = classify_evidence_item(
            {"patient_finding": "chest pain", "alternative_explanation": "cardiac ischemia"}, node="R3a"
        )
        self.assertEqual(result["mimic_domain"], "acute_coronary_syndrome")


class TestStability(unittest.TestCase):
    def test_16_tier_consistency_computed(self):
        report = stability_by_case(repeated_run(["yes", "no"]))
        self.assertEqual(report["stable_tier_cases"], 1)

    def test_17_first_node_divergence_computed(self):
        records = repeated_run(["yes", "no"])["results"]
        self.assertEqual(first_node_divergence(records), "R1")

    def test_18_wording_change_is_not_clinical_instability(self):
        report = stability_by_case(repeated_run(["yes", "yes"], ["alpha", "beta"]))
        self.assertEqual(report["stable_tier_cases"], 1)
        self.assertEqual(report["stable_node_answer_cases"], 1)
        self.assertEqual(report["stable_wording_cases"], 0)

    def test_19_fingerprint_metadata_is_preserved(self):
        rows, _ = reliability_rows(repeated_run(["yes"], fingerprints=["fp_fixture"]), "fixture.json")
        self.assertEqual(rows[0]["system_fingerprint"], "fp_fixture")


if __name__ == "__main__":
    unittest.main(verbosity=2)
