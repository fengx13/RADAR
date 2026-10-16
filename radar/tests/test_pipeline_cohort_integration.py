"""Offline coverage for the shared dynamic entrypoints and cohort execution."""
from __future__ import annotations
from radar import paths

import contextlib
import copy
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
sys.path.insert(0, str(HERE))

from radar.pipeline import pipeline_batch
from radar.pipeline import run_full_pipeline
from radar.pipeline import run_pipeline
from radar.pipeline import run_pipeline_vnext
from radar.pipeline.longitudinal_integration import assemble_checkpoint_state, build_workup_state, workup_state_as_of


def fixture_cases(n=3):
    base = run_pipeline.load_cases(paths.DATA / "checkpoint_replay_fixture.json")[0]
    cases = []
    for index in range(n):
        case = copy.deepcopy(base)
        case["case_id"] = f"BATCH_{index}"
        cases.append(case)
    return cases


class TestInvestigationHandoff(unittest.TestCase):
    def result(self, available=0.2, unit="mg/dL"):
        return {"item": "creatinine", "actual_order_status": "ordered", "order_time": 0.05,
                "sample_time": 0.1, "result_status": "resulted", "result_available_time": available,
                "value": 432.123, "unit": unit, "result_kind": "diagnostic_laboratory",
                "checkpoint_eligible": True, "disposition_relevant": True,
                "provenance": {"source_type": "synthetic_fixture", "source_field": "test.value"}}

    def test_other_investigations_reach_checkpoint_with_original_unit(self):
        workup = build_workup_state(None, [self.result()])
        state = assemble_checkpoint_state({}, workup, 0.5, "C0_5")
        result = state["investigation_results"]["creatinine"]
        self.assertEqual(result["value"], 432.123)
        self.assertEqual(result["unit"], "mg/dL")
        self.assertEqual(result["result_available_time"], 0.2)
        # PY-129: SOFA is computed in code at the checkpoint (never by the model).
        self.assertEqual(state["sofa"]["provenance"]["source_type"], "deterministic_derived")
        self.assertEqual(state["sofa"]["components_available"], ["creatinine"])
        self.assertTrue(state["sofa"]["sofa_ge_2_established"])

    def test_later_values_and_availability_do_not_leak_via_provenance(self):
        workup = build_workup_state(None, [self.result(available=0.8)])
        state = assemble_checkpoint_state({}, workup, 0.5, "C0_5")
        snapshot = workup_state_as_of(workup, 0.5)
        self.assertNotIn("investigation_results", state)
        self.assertNotIn("432.123", json.dumps(snapshot))
        item = next(item for item in snapshot["items"] if item["item"] == "creatinine")
        self.assertIsNone(item["provenance"]["available_time"])
        self.assertEqual(item["result_status"], "pending")

    def test_sample_time_does_not_release_result(self):
        workup = build_workup_state(None, [self.result(available=0.8)])
        self.assertNotIn("investigation_results", assemble_checkpoint_state({}, workup, 0.1, "C0_5"))
        later = assemble_checkpoint_state({}, workup, 1.0, "C1")
        self.assertEqual(later["investigation_results"]["creatinine"]["value"], 432.123)

    def test_missing_unit_is_not_invented(self):
        workup = build_workup_state(None, [self.result(unit=None)])
        state = assemble_checkpoint_state({}, workup, 0.5, "C0_5")
        self.assertIsNone(state["investigation_results"]["creatinine"]["unit"])

    def test_future_sample_time_is_not_exposed_in_pending_provenance(self):
        event = self.result(available=2.0)
        event["sample_time"] = 1.5
        snapshot = workup_state_as_of(build_workup_state(None, [event]), 0.5)
        item = next(item for item in snapshot["items"] if item["item"] == "creatinine")
        self.assertIsNone(item["provenance"]["event_time"])
        self.assertIsNone(item["provenance"]["available_time"])

    def test_dynamic_pipeline_passes_result_to_downstream_state(self):
        case = fixture_cases(1)[0]
        case["observed_workup"].append(self.result())
        run = run_pipeline.run_longitudinal_case(case, config=run_pipeline.pipeline_config(replay_mode="full"))
        processed = [c for c in run["checkpoints"] if c["checkpoint_status"] == "processed"]
        self.assertTrue(processed)
        self.assertEqual(processed[0]["patient_state"]["investigation_results"]["creatinine"]["value"], 432.123)
        self.assertEqual(processed[0]["agent4_input"]["patient_state"]["investigation_results"]["creatinine"]["unit"], "mg/dL")


class TestBatchExecution(unittest.TestCase):
    def test_parallel_preserves_input_order(self):
        cases = fixture_cases()
        results = pipeline_batch.run_batches(cases, lambda c: {"case_id": c["case_id"]},
                                             workers=3, batch_size=2, progress=False)
        self.assertEqual([r["case_id"] for r in results], [c["case_id"] for c in cases])

    def test_parallel_preserves_full_dynamic_trajectories(self):
        cases = fixture_cases()
        original = copy.deepcopy(cases)
        config = run_pipeline.pipeline_config(replay_mode="full")
        run = lambda case: run_pipeline.run_longitudinal_case(case, config=config)
        serial = pipeline_batch.run_batches(cases, run, progress=False)
        concurrent = pipeline_batch.run_batches(cases, run, workers=3, batch_size=2, progress=False)
        self.assertEqual(concurrent, serial)
        self.assertEqual(cases, original)
        processed = [c for c in concurrent[0]["checkpoints"] if c["checkpoint_status"] == "processed"]
        self.assertGreater(len(processed), 1)
        self.assertTrue(all(c["agent3"] and c["agent4"] for c in processed))

    def test_case_failure_is_not_silently_dropped(self):
        def run(case):
            if case["case_id"] == "BATCH_1":
                raise ValueError("synthetic failure")
            return {"case_id": case["case_id"], "status": "complete"}
        results = pipeline_batch.run_batches(fixture_cases(), run, workers=3, progress=False)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[1]["status"], "runner_error")
        self.assertEqual(results[2]["status"], "complete")

    def test_batch_limit_processes_only_requested_patients(self):
        results = pipeline_batch.run_batches(fixture_cases(5), lambda c: c, batch_size=2,
                                             max_batches=1, progress=False)
        self.assertEqual([r["case_id"] for r in results], ["BATCH_0", "BATCH_1"])

    def test_completed_batch_is_published_before_next_batch(self):
        published = []
        pipeline_batch.run_batches(fixture_cases(5), lambda c: c, batch_size=2, progress=False,
                                   on_batch=lambda rows, index, n: published.append((len(rows), index, n)))
        self.assertEqual(published, [(2, 1, 2), (4, 2, 2), (5, 3, 1)])

    def test_malformed_or_static_inputs_are_rejected(self):
        for cases in ([], [None], [{"case_id": "T1", "input": {"checkpoint_state": {}}}],
                      [{"case_id": "T0", "t0_state": {"checkpoint_state": {}}}],
                      [{"case_id": "T0", "t0_state": {}, "observed_workup": {}}]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                pipeline_batch.validate_longitudinal_cases(cases)

    def test_duplicate_case_ids_are_rejected(self):
        cases = fixture_cases(2)
        cases[1]["case_id"] = cases[0]["case_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            pipeline_batch.validate_longitudinal_cases(cases)

    def test_t0_only_case_is_not_forced_to_have_future_results(self):
        pipeline_batch.validate_longitudinal_cases([{"case_id": "EARLY", "t0_state": {}}])


class TestUnifiedEntrypoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="radar-cohort-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.infile = self.root / "cases.json"
        self.infile.write_text(json.dumps(fixture_cases()), encoding="utf-8")
        self.outfile = self.root / "run.json"

    def args(self, *extra):
        return ["--infile", str(self.infile), "--out", str(self.outfile), "--dry-run", *extra]

    def invoke(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_pipeline.main(self.args(*extra))

    def cli(self, script, *extra):
        env = dict(os.environ, OPENAI_API_KEY="", PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run([sys.executable, "-B", "-m", "radar.pipeline." + script.removesuffix(".py"), *self.args(*extra)],
                              cwd=ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)

    def test_both_entrypoints_default_to_sequential(self):
        with patch.object(run_pipeline_vnext, "main", return_value=0) as dynamic:
            self.assertEqual(run_pipeline.main(["--dry-run"]), 0)
            self.assertEqual(run_full_pipeline.main(["--dry-run"]), 0)
        self.assertEqual(dynamic.call_count, 2)

    def test_full_entrypoint_executes_multiple_checkpoints(self):
        result = self.cli("run_full_pipeline.py", "--workers", "2", "--batch-size", "2",
                          "--replay-mode", "full")
        self.assertEqual(result.returncode, 0, result.stderr)
        doc = json.loads(self.outfile.read_text(encoding="utf-8"))
        self.assertEqual(doc["meta"]["pipeline_mode"], "sequential")
        self.assertEqual(doc["meta"]["n"], 3)
        self.assertTrue(doc["meta"]["cohort_execution_complete"])
        self.assertEqual(len(doc["meta"]["batch_summaries"]), 2)
        processed = [c for c in doc["results"][0]["checkpoints"] if c["checkpoint_status"] == "processed"]
        self.assertGreater(len(processed), 1)

    def test_run_tag_isolates_repeated_outputs(self):
        self.outfile.write_text("previous run", encoding="utf-8")
        self.assertEqual(self.invoke("--run-tag", "r2"), 0)
        self.assertEqual(self.outfile.read_text(encoding="utf-8"), "previous run")
        doc = json.loads((self.root / "run_r2.json").read_text(encoding="utf-8"))
        self.assertEqual(doc["meta"]["run_tag"], "r2")
        self.assertEqual(len(doc["meta"]["input_sha256"]), 64)

    def test_existing_output_is_not_overwritten_without_force(self):
        self.outfile.write_text("preserve me", encoding="utf-8")
        with self.assertRaises(SystemExit) as raised:
            self.invoke()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.outfile.read_text(encoding="utf-8"), "preserve me")

    def test_force_is_explicit(self):
        self.outfile.write_text("previous run", encoding="utf-8")
        self.assertEqual(self.invoke("--force"), 0)
        self.assertEqual(json.loads(self.outfile.read_text(encoding="utf-8"))["meta"]["n"], 3)

    def test_output_cannot_replace_input_even_with_force(self):
        with self.assertRaisesRegex(ValueError, "input"):
            pipeline_batch.output_destination(self.infile, self.infile, force=True)

    def test_partial_pilot_is_labelled_and_denominator_preserved(self):
        self.assertEqual(self.invoke("--batch-size", "2", "--max-batches", "1"), 0)
        meta = json.loads(self.outfile.read_text(encoding="utf-8"))["meta"]
        self.assertEqual((meta["n"], meta["n_input"]), (2, 3))
        self.assertFalse(meta["cohort_execution_complete"])

    def test_invalid_batch_flags_are_rejected(self):
        for flags in (("--workers", "0"), ("--batch-size", "0"), ("--max-batches", "1"),
                      ("--run-tag", "../unsafe"), ("--max-batches", "-1")):
            with self.subTest(flags=flags), self.assertRaises(SystemExit) as raised:
                self.invoke(*flags)
            self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.outfile.exists())

    def test_static_snapshot_rejected_before_client_initialization(self):
        self.infile.write_text(json.dumps([{"case_id": "STATIC", "input": {
            "checkpoint_state": {"checkpoint_id": "T_SOFA"}}}]), encoding="utf-8")
        with patch.object(run_pipeline_vnext, "load_env") as load_env:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                run_pipeline.main(["--infile", str(self.infile), "--out", str(self.outfile)])
        load_env.assert_not_called()
        self.assertFalse(self.outfile.exists())

    def test_runner_error_is_saved_and_returns_nonzero(self):
        with patch.object(run_pipeline_vnext, "run_longitudinal_case", side_effect=ValueError("synthetic")):
            self.assertEqual(self.invoke("--workers", "2", "--batch-size", "2"), 1)
        doc = json.loads(self.outfile.read_text(encoding="utf-8"))
        self.assertEqual(doc["meta"]["runner_error_count"], 3)
        self.assertEqual(len(doc["results"]), 3)

    def test_legacy_cohort_mode_is_explicit_and_preserved(self):
        with patch.object(run_full_pipeline, "run_legacy_cohort_main", return_value=0) as legacy:
            run_full_pipeline.main(["--pipeline-mode", "legacy_cohort_two_time", "--prefix", "cohort100"])
        legacy.assert_called_once_with(["--prefix", "cohort100"])

    def test_single_t1_cli_really_uses_legacy_not_rebound_dynamic_globals(self):
        legacy_cases = run_pipeline.legacy_load_cases(paths.DATA / "pipeline_synthetic_cases.json")
        self.infile.write_text(json.dumps(legacy_cases[:1]), encoding="utf-8")
        result = self.cli("run_pipeline.py", "--pipeline-mode=legacy_single_t1")
        self.assertEqual(result.returncode, 0, result.stderr)
        doc = json.loads(self.outfile.read_text(encoding="utf-8"))
        self.assertEqual(doc["meta"]["pipeline_mode"], "legacy_single_t1")
        self.assertEqual(doc["meta"]["version"], "0.2")
        self.assertIsNotNone(doc["results"][0]["agent1_t1"])
        self.assertTrue(all(not r.get("checkpoints") for r in doc["results"]))
        self.assertTrue(all(r.get("agent4_t0") is None for r in doc["results"]))

    def test_atomic_writer_preserves_prior_snapshot_on_serialization_failure(self):
        self.outfile.write_text("previous completed batch", encoding="utf-8")
        with self.assertRaises(TypeError):
            pipeline_batch.write_output(self.outfile, {"invalid": object()})
        self.assertEqual(self.outfile.read_text(encoding="utf-8"), "previous completed batch")
        self.assertEqual(list(self.root.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
