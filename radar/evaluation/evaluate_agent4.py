#!/usr/bin/env python3
"""Retrospective Agent 4 evaluation with decision-time anchored exclusions."""
from __future__ import annotations
from radar import paths

import argparse
import json
import pathlib
import sys
from collections import Counter

from radar.common.integration_schema import _time_key
from radar.common.observation_semantics import (
    classify_observed_disposition,
    classify_observed_disposition_detail,
    denominator_exclusion_reason,
    observation_semantics_metadata,
)


ROOT = paths.ROOT
def safe_print(text: str):
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding) + "\n")


def _outcomes(record: dict, case_lookup: dict) -> dict:
    if isinstance(record.get("evaluation_only"), dict):
        return record["evaluation_only"]
    if "truth" in record:
        return record["truth"].get("outcomes", {})
    case = case_lookup.get(record.get("case_id"), {})
    if isinstance(case.get("evaluation_only"), dict):
        return case["evaluation_only"]
    return case.get("truth", {}).get("outcomes", {})


def _decision_view(record: dict) -> tuple[dict, object, str | None]:
    if isinstance(record.get("result"), dict):
        result = record["result"]
        cutoff = result.get("information_cutoff", record.get("information_cutoff", record.get("t1_time")))
        return result, cutoff, result.get("decision_time")

    checkpoint_id = record.get("disposition_checkpoint")
    if checkpoint_id == "T0":
        result = ((record.get("agent4_t0") or {}).get("result") or {})
        return result, record.get("t0_time", 0.0), "T0"
    if checkpoint_id:
        checkpoint = next(
            (item for item in record.get("checkpoints") or [] if item.get("checkpoint_id") == checkpoint_id),
            None,
        )
        if checkpoint:
            return ((checkpoint.get("agent4") or {}).get("result") or {}), checkpoint.get("cutoff_time"), checkpoint_id
    result = ((record.get("agent4") or {}).get("result") or {})
    cutoff = record.get("t1_time", (record.get("t1_gate") or {}).get("t1_time"))
    return result, cutoff, "T1" if cutoff is not None else None


def derive_decision_icu_transfer_12h(decision_time, outcomes: dict) -> dict:
    """Derive 0 < ICU transfer time - modeled decision cutoff <= 12 hours."""
    if outcomes.get("icu_transfer_status") == "no_transfer":
        return {"status": "available", "event": False, "delta_hours": None, "boundary": "0 < delta <= 12h"}
    transfer_time = outcomes.get("icu_transfer_time")
    if transfer_time is None:
        reason = "arrival_anchored_flag_not_accepted" if "dsp_icu_transfer_12h" in outcomes else "icu_transfer_time_not_supplied"
        return {"status": "unavailable", "event": None, "delta_hours": None, "reason": reason, "boundary": "0 < delta <= 12h"}
    decision_key = _time_key(decision_time)
    transfer_key = _time_key(transfer_time)
    if decision_key is None or transfer_key is None or decision_key[0] != transfer_key[0]:
        return {"status": "unavailable", "event": None, "delta_hours": None, "reason": "invalid_or_incomparable_timestamps", "boundary": "0 < delta <= 12h"}
    delta = transfer_key[1] - decision_key[1]
    delta_hours = delta if decision_key[0] == "numeric" else delta / 3600.0
    return {
        "status": "available",
        "event": 0 < delta_hours <= 12,
        "delta_hours": delta_hours,
        "pre_or_at_decision": delta_hours <= 0,
        "boundary": "0 < delta <= 12h",
    }


def derive_t1_icu_transfer_12h(t1_time, outcomes: dict) -> dict:
    """Backward-compatible alias for legacy single-T1 evaluation callers."""
    return derive_decision_icu_transfer_12h(t1_time, outcomes)


def calculate_metrics(agent4_results: list[dict], cases: list[dict] | None = None) -> dict:
    case_lookup = {case["case_id"]: case for case in cases or []}
    primary_numer: list[str] = []
    primary_evaluable: list[str] = []
    primary_unavailable: list[dict] = []
    exclusions: list[dict] = []
    revisit: list[str] = []
    critical: list[str] = []
    missed_escalation: list[str] = []
    observation_or_ward_count = 0
    possible_overtriage: list[str] = []
    modeled_icu_count = 0
    observed_dispositions = Counter()
    modeled_dispositions = Counter()
    recommended_discharge_count = 0

    for record in agent4_results:
        case_id = record.get("case_id")
        result, cutoff, decision_label = _decision_view(record)
        if cutoff is None:
            case = case_lookup.get(case_id, {})
            cutoff = case.get("t1_time", (case.get("t1_gate") or {}).get("t1_time"))
        disposition = result.get("disposition")
        outcomes = _outcomes(record, case_lookup)
        icu_detail = derive_decision_icu_transfer_12h(cutoff, outcomes)
        exclusion = denominator_exclusion_reason(
            outcomes,
            result.get("status"),
            icu_detail.get("delta_hours") if icu_detail.get("status") == "available" else None,
        )
        if exclusion:
            exclusions.append({"case_id": case_id, "decision_time": decision_label, "reason": exclusion})
            continue

        modeled_dispositions[disposition] += 1
        icu12 = icu_detail.get("event") is True
        observed = classify_observed_disposition_detail(outcomes, icu_transfer_after_decision=icu12)
        if observed["label"] is not None:
            observed_dispositions[observed["label"]] += 1

        death_value = outcomes.get("dsp_mortality_inhospital", outcomes.get("mortality_inhospital"))
        death_available = death_value is not None
        death_inhospital = death_value == 1
        critical_event = outcomes.get("dsp_critical", outcomes.get("critical_deterioration")) == 1
        revisit3 = outcomes.get("dsp_ed_revisit_3d", outcomes.get("ed_revisit_3d")) == 1

        if disposition == "discharge":
            recommended_discharge_count += 1
            if icu_detail["status"] == "available" and death_available:
                primary_evaluable.append(case_id)
                if icu12 or death_inhospital:
                    primary_numer.append(case_id)
            else:
                primary_unavailable.append({
                    "case_id": case_id,
                    "decision_time": decision_label,
                    "icu_transfer_endpoint": icu_detail,
                    "in_hospital_mortality_available": death_available,
                })
            if revisit3:
                revisit.append(case_id)
            if critical_event:
                critical.append(case_id)
        if disposition in {"observation", "ward"}:
            observation_or_ward_count += 1
            if icu12:
                missed_escalation.append(case_id)
        if disposition == "ICU":
            modeled_icu_count += 1
            if icu_detail["status"] == "available" and not (icu12 or critical_event or death_inhospital):
                possible_overtriage.append(case_id)

    denominator = len(primary_evaluable)
    evaluable_n = sum(modeled_dispositions.values())
    collapsed = None
    if evaluable_n:
        top, count = modeled_dispositions.most_common(1)[0]
        collapsed = {"disposition": top, "count": count, "fraction": count / evaluable_n}
    return {
        "observation_semantics": observation_semantics_metadata(),
        "denominator": {
            "modeled_decision_count": evaluable_n,
            "excluded_count": len(exclusions),
            "exclusions": exclusions,
            "icu_boundary": "ICU transfer at or before each modeled decision cutoff is excluded from prospective prediction at that cutoff",
        },
        "primary": {
            "name": "discharge_boundary_undertriage_rate",
            "definition": "P(12h post-decision ICU transfer OR in-hospital death | valid Agent 4 discharge decision)",
            "numerator": len(primary_numer),
            "denominator": denominator,
            "rate": None if denominator == 0 else len(primary_numer) / denominator,
            "case_ids": primary_numer,
            "recommended_discharge_count": recommended_discharge_count,
            "evaluable_case_ids": primary_evaluable,
            "unavailable": primary_unavailable,
            "icu_transfer_boundary": "0 < ICU_transfer_time - modeled_decision_cutoff <= 12 hours",
            "legacy_dsp_icu_transfer_12h_accepted": False,
        },
        "secondary": {
            "discharge_3d_revisit": {"n": len(revisit), "case_ids": revisit},
            "discharge_critical_deterioration": {"n": len(critical), "case_ids": critical},
            "observation_or_ward_to_icu_12h": {
                "n": len(missed_escalation),
                "denominator": observation_or_ward_count,
                "rate": None if observation_or_ward_count == 0 else len(missed_escalation) / observation_or_ward_count,
                "case_ids": missed_escalation,
            },
            "possible_overtriage_icu_without_recorded_icu_need": {
                "n": len(possible_overtriage),
                "denominator": modeled_icu_count,
                "rate": None if modeled_icu_count == 0 else len(possible_overtriage) / modeled_icu_count,
                "case_ids": possible_overtriage,
                "note": "Requires clinician adjudication; recorded outcomes may not capture all ICU-level need.",
            },
            "disposition_distribution": dict(modeled_dispositions),
            "observed_disposition_distribution": dict(observed_dispositions),
            "collapse_check": collapsed,
            "abstention_rate": sum(_decision_view(record)[0].get("status") == "abstain" for record in agent4_results) / len(agent4_results) if agent4_results else None,
            "pipeline_alarm_rate": sum(bool(_decision_view(record)[0].get("pipeline_alarm")) for record in agent4_results) / len(agent4_results) if agent4_results else None,
        },
    }


def load_case_records(path: pathlib.Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]
    if isinstance(data, dict):
        return [data]
    raise ValueError("case input must be an object, list, or {'cases': [...]} envelope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(paths.DATA / "agent4_run.json"))
    parser.add_argument("--cases", default=str(paths.DATA / "phase0_cases.json"))
    parser.add_argument("--out", default=str(paths.DATA / "agent4_eval.json"))
    args = parser.parse_args()
    results_data = json.loads(pathlib.Path(args.results).read_text(encoding="utf-8"))
    cases = load_case_records(pathlib.Path(args.cases)) if pathlib.Path(args.cases).exists() else []
    metrics = calculate_metrics(results_data["results"], cases)
    pathlib.Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    safe_print(f"-> {args.out}")


if __name__ == "__main__":
    main()
