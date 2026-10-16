"""Event/documentation cutoff regressions; synthetic records, no model calls."""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from radar.common.integration_schema import strip_inference_forbidden
from radar.pipeline.longitudinal_integration import (
    _latest_event,
    _temporal_provenance,
    assemble_checkpoint_state,
    assemble_t0_agent4_state,
    build_treatment_state,
    build_workup_state,
)


class EventVisibilityTests(unittest.TestCase):
    def test_delayed_documentation_does_not_replace_visible_vital(self):
        events = [
            {"measurement_time": 0.1, "available_time": 0.2, "map": 80},
            {"measurement_time": 0.4, "available_time": 2.0, "map": 50},
        ]
        before = copy.deepcopy(events)
        latest = _latest_event(events, "measurement_time", 1.0)
        self.assertEqual(latest["map"], 80)
        self.assertEqual(events, before)

    def test_event_and_availability_boundary_are_inclusive(self):
        event = {"start_time": 0.5, "available_time": 1.0}
        self.assertIsNone(_latest_event([event], "start_time", 0.9))
        self.assertEqual(_latest_event([event], "start_time", 1.0), event)

    def test_explicit_unknown_invalid_or_mixed_availability_is_excluded(self):
        for value in (None, "", "not-a-time", True, "2026-09-22T00:00:00Z", float("nan"), float("inf")):
            with self.subTest(available_time=value):
                event = {"start_time": 0.1, "available_time": value}
                self.assertIsNone(_latest_event([event], "start_time", 1.0))

    def test_nested_and_top_level_documentation_constraints_both_apply(self):
        for source in ("event", "provenance"):
            for field in ("available_time", "documentation_time"):
                with self.subTest(source=source, field=field):
                    event = {"start_time": 0.1, "available_time": 0.2, "provenance": {}}
                    target = event if source == "event" else event["provenance"]
                    target[field] = 2.0
                    self.assertIsNone(_latest_event([event], "start_time", 1.0))

    def test_iso_timestamps_keep_documentation_cutoff(self):
        event = {"start_time": "2026-09-22T10:00:00+08:00",
                 "available_time": "2026-09-22T03:00:00Z"}
        self.assertIsNone(_latest_event([event], "start_time", "2026-09-22T10:30:00+08:00"))
        self.assertEqual(_latest_event([event], "start_time", "2026-09-22T11:00:00+08:00"), event)

    def test_future_event_is_hidden_even_when_documented_earlier(self):
        self.assertIsNone(_latest_event([{"start_time": 2.0, "available_time": 0.5}], "start_time", 1.0))

    def test_omitted_availability_preserves_existing_contract(self):
        event = {"measurement_time": 0.5, "map": 80}
        self.assertEqual(_latest_event([event], "measurement_time", 1.0), event)
        self.assertEqual(_temporal_provenance(event, "measurement_time", 1.0)["available_time"], 0.5)

    def test_t0_support_waits_for_documentation(self):
        events = {
            "respiratory_support": [{"start_time": -0.1, "available_time": 0.5,
                                     "ventilation_status": "HFNC", "current_encounter": True}],
            "vasopressors": [{"start_time": -0.1, "available_time": 0.5, "active": True}],
        }
        state = assemble_t0_agent4_state({}, events, 0.0)
        self.assertNotIn("active_support", state)

    def test_checkpoint_support_fluids_and_response_wait_for_documentation(self):
        events = {
            "respiratory_support": [{"start_time": 0.1, "available_time": 2.0,
                                     "oxygen_therapy_documented": True, "mode": "HFNC"}],
            "vasopressors": [{"start_time": 0.1, "available_time": 2.0, "active": True}],
            "fluids": [{"event_time": 0.1, "available_time": 2.0, "amount_ml": 500}],
            "post_resuscitation": [{"measurement_time": 0.1, "available_time": 2.0,
                                    "map_time": 0.1, "vital_map_min": 55}],
            "treatment_response": [{"assessment_time": 0.1, "available_time": 2.0, "state": "improving"}],
        }
        state = assemble_checkpoint_state(events, build_workup_state(None, []), 1.0, "C1")
        self.assertIsNone(state["oxygen_requirement"]["oxygen_therapy_documented"])
        for field in ("vasopressor", "ventilation", "resuscitation", "post_resuscitation", "treatment_response"):
            self.assertNotIn(field, state)
        self.assertEqual(state["excluded_post_checkpoint_event_counts"], {key: 1 for key in events})

    def test_actual_administration_waits_for_documentation(self):
        event = {"administration_time": 0.1, "available_time": 2.0,
                 "administered": True, "status": "ongoing"}
        source = {"antimicrobials": [event]}
        early = build_treatment_state(source, 1.0)
        self.assertEqual(early["antimicrobials"]["status"], "unavailable")
        self.assertEqual(early["ongoing_treatment"], [])
        self.assertEqual(early["treatment_response_flag"], "unavailable")
        late = build_treatment_state(source, 2.0)
        self.assertTrue(late["antimicrobials"]["administered"])
        self.assertEqual(late["ongoing_treatment"], ["antimicrobials"])

    def test_delayed_response_does_not_become_observed_response(self):
        state = build_treatment_state({
            "antimicrobials": [{"administration_time": 0.1, "available_time": 0.2,
                                "administered": True, "status": "completed"}],
            "treatment_response": [{"assessment_time": 0.3, "available_time": 2.0, "state": "improving"}],
        }, 1.0)
        self.assertEqual(state["treatment_response_flag"], "not_yet_assessable")

    def test_provenance_uses_latest_required_time_and_same_visibility_check(self):
        event = {"start_time": 0.1, "available_time": 0.2,
                 "provenance": {"source_type": "observed_ehr", "documentation_time": 0.8}}
        early = _temporal_provenance(event, "start_time", 0.5)
        self.assertFalse(early["included_at_checkpoint"])
        self.assertFalse(early["included_at_t1"])
        late = _temporal_provenance(event, "start_time", 1.0)
        self.assertEqual(late["available_time"], 0.8)
        self.assertTrue(late["included_at_checkpoint"])

    def test_delayed_value_and_hidden_outcome_perturbations_do_not_change_inference(self):
        events = {"vitals": [{"measurement_time": 0.1, "available_time": 2.0, "map": 45}]}
        changed = copy.deepcopy(events)
        changed["vitals"][0]["map"] = 999
        changed["truth"] = {"dsp_mortality_inhospital": 1}
        workup = build_workup_state(None, [])
        states = [strip_inference_forbidden(assemble_checkpoint_state(source, workup, 1.0, "C1"),
                                            exclude_temporal_audit=True) for source in (events, changed)]
        self.assertEqual(states[0], states[1])
        self.assertNotIn("999", json.dumps(states[1]))
        self.assertNotIn("dsp_", json.dumps(states[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
