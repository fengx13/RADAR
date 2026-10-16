"""PY-124: deterministic bedside organ-dysfunction screen for Agent 1 node R2 at T0."""
from __future__ import annotations

import copy
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from radar.agents import agent1_bedside_screen as B  # noqa: E402
from radar.agents import run_agent1  # noqa: E402


def case_with(vitals: dict) -> dict:
    return {"case_id": "T", "input": {"chief_complaint": "x", "age": 60, "sex": "F",
                                       "triage_vitals": vitals, "labs": "none"}}


class ThresholdTests(unittest.TestCase):
    def status(self, vitals, field):
        return {i["field"]: i["status"] for i in B.bedside_screen({"triage_vitals": vitals})["items"]}[field]

    def test_boundaries_match_make_cohort100(self):
        self.assertEqual(self.status({"resp_rate": 22}, "resp_rate"), "met")
        self.assertEqual(self.status({"resp_rate": 21}, "resp_rate"), "not_met")
        self.assertEqual(self.status({"sbp": 100}, "sbp"), "met")
        self.assertEqual(self.status({"sbp": 101}, "sbp"), "not_met")
        self.assertEqual(self.status({"spo2": 91}, "spo2"), "met")
        self.assertEqual(self.status({"spo2": 92}, "spo2"), "not_met")

    def test_missing_values_are_not_recorded_not_normal(self):
        s = B.bedside_screen({"triage_vitals": {"sbp": 130}})
        st = {i["field"]: i["status"] for i in s["items"]}
        self.assertEqual(st["resp_rate"], "not_recorded")
        self.assertEqual(st["spo2"], "not_recorded")
        self.assertEqual(st["mental_status"], "not_recorded")
        self.assertFalse(s["any_met"])
        self.assertFalse(s["all_recorded"])
        self.assertFalse(B.bedside_screen({})["any_met"])

    def test_any_met(self):
        self.assertTrue(B.bedside_screen({"triage_vitals": {"resp_rate": 18, "sbp": 96, "spo2": 99}})["any_met"])
        self.assertFalse(B.bedside_screen({"triage_vitals": {"resp_rate": 18, "sbp": 120, "spo2": 99}})["any_met"])

    def test_prompt_block_states_criteria_and_status(self):
        text = B.prompt_block(B.bedside_screen({"triage_vitals": {"resp_rate": 22, "sbp": 128, "spo2": 100}}))
        self.assertIn(">= 22", text)
        self.assertIn("MET", text)
        self.assertIn("not recorded", text)
        self.assertIn("not require SOFA components", text)


class InjectionTests(unittest.TestCase):
    def node(self, nid):
        return dict(run_agent1.GRAPH["nodes"][nid], id=nid)

    def test_only_r2_at_t0_gets_the_screen(self):
        case = case_with({"resp_rate": 22, "sbp": 128, "spo2": 100})
        self.assertIsNotNone(run_agent1.bedside_screen_for(self.node("R2"), case, "T0", True))
        self.assertIsNone(run_agent1.bedside_screen_for(self.node("R1"), case, "T0", True))
        self.assertIsNone(run_agent1.bedside_screen_for(self.node("R3b"), case, "T0", True))
        self.assertIsNone(run_agent1.bedside_screen_for(self.node("R2"), case, "checkpoint", True))
        self.assertIsNone(run_agent1.bedside_screen_for(self.node("R2"), case, "T0", False))

    def test_prompt_contains_block_only_when_screen_supplied(self):
        case = case_with({"resp_rate": 22, "sbp": 128, "spo2": 100})
        node = self.node("R2")
        screen = run_agent1.bedside_screen_for(node, case, "T0", True)
        with_block = run_agent1.user_prompt(node, case, "", "T0", None, screen)
        without = run_agent1.user_prompt(node, case, "", "T0", None, None)
        self.assertIn("BEDSIDE ORGAN-DYSFUNCTION SCREEN", with_block)
        self.assertNotIn("BEDSIDE ORGAN-DYSFUNCTION SCREEN", without)
        # the screen sits after the patient block and before the question
        self.assertLess(with_block.find("PATIENT AT TRIAGE"), with_block.find("BEDSIDE ORGAN"))
        self.assertLess(with_block.find("BEDSIDE ORGAN"), with_block.find("QUESTION\n"))

    def test_dry_run_trace_records_screen_on_r2_and_flag_off_removes_it(self):
        if not run_agent1.CASES:
            self.skipTest("needs the local phase0 pilot cases (patient data, RADAR_input/MIMIC)")
        class Args:
            dry_run = True; decision_time = "T0"; risk_signal = "none"; risk_signal_map = {}
            model = "x"; temperature = 0.0; seed = 1; bedside_screen = "on"
        case = copy.deepcopy(run_agent1.CASES[0])
        # force a path through R2 regardless of the stub's R1 answer
        for c in [case] + list(run_agent1.CASES):
            r = run_agent1.run_case(c, None, Args())
            r2 = [t for t in r["trace"] if t["node"] == "R2"]
            if r2:
                self.assertIn("bedside_screen", r2[0])
                self.assertIn("items", r2[0]["bedside_screen"])
                Args.bedside_screen = "off"
                r_off = run_agent1.run_case(c, None, Args())
                self.assertTrue(all("bedside_screen" not in t for t in r_off["trace"]))
                return
        self.skipTest("dry-run stub never reached R2 on the frozen cases")


if __name__ == "__main__":
    unittest.main()
