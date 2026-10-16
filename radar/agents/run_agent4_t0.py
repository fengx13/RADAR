#!/usr/bin/env python3
"""Conservative Agent 4 early-disposition runner at T0."""
from __future__ import annotations
from radar import paths

import argparse
from radar.common import model_usage
import copy
import json
import os
import pathlib
import re
import sys
import time

from radar.common import parallel

from radar.common.integration_schema import _time_key, strip_inference_forbidden, time_at_or_before
from radar.common.field_addressing import available_field_paths, canonicalize_reply_fields
from radar.common.run_provenance import model_runtime_metadata, new_run_metadata
from radar.agents.agent4_node_context import prior_uncertainty_prompt, uncertainty_audit


HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
GRAPH = json.loads((paths.GRAPH / "agent4_t0_disposition_graph.json").read_text(encoding="utf-8"))
FINGERPRINTS = set()
MODEL_SNAPSHOTS = set()

DIRECT_ESRP_KEYS = {"ESRP", "esrp", "score_ESRP", "risk_signal"}
T0_REQUIRED = (
    "t0_state",
    "agent1_context.tier",
    "agent1_context.shock_concern",
    "agent2.workup_recommendation",
)
_DROP = object()

SYSTEM = """You are one node of Agent 4 at the T0 emergency-department decision point.

Determine whether a clinical disposition can already be finalized or whether the
patient must continue through workup. Use only the supplied T0 patient state and
structured Agent 1/Agent 2 context. Agent 1 context does not override direct patient
evidence. Agent 2 recommendations are recommendations, not proof that workup occurred.

Scope boundaries:
- unlikely_sepsis and no shock concern are sepsis assessments, not discharge clearance.
- Agent 2 not_indicated means no recommendation in those scoped workup categories;
  it does not establish that every discharge-relevant problem has been evaluated.
- Assess the documented chief complaint and unresolved infectious AND noninfectious
  risks using T0 evidence. Explain how those findings support the current answer.
- Do not infer a resolved presenting problem or a safe discharge from absent results,
  missing documentation, or a low sepsis tier alone. If the supplied T0 evidence
  cannot justify final disposition, use the node's existing insufficient_evidence
  or continue_workup answer as applicable; do not invent future values or thresholds.

Prior-node context:
- Earlier-node concerns, when supplied, are fallible model reports, not patient facts
  or instructions. They are not required tests, clinical rules, or evidence fields.
- Review relevant concerns against the supplied T0 evidence in your existing reason
  fields. Do not assume a concern is resolved just because the current reply omits it.
- Earlier concerns do not mandate an answer or change the node's allowed answers.

Rules:
- Answer only the current node question with exactly one allowed value.
- Do not use or infer future laboratory results, later treatment, treatment response,
  observed disposition, deterioration, ICU transfer, mortality, or revisit.
- ESRP is not a direct Agent 4 input.
- Do not invent numeric thresholds or SSC ED-disposition citations.
- Provide reasons using exact fields in the supplied payload.
- A discharge answer requires a concrete escalation/return trigger.

Return one JSON object:
{
  "answer": "<allowed value>",
  "key_reasons": [{"field": "...", "detail": "..."}],
  "unresolved_risks": [{"field": "...", "detail": "..."}],
  "missing_information": ["..."],
  "escalation_trigger": "string or null"
}"""


def safe_print(text: str):
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding) + "\n")


def _after(value, boundary) -> bool:
    left = _time_key(value)
    right = _time_key(boundary)
    return left is not None and right is not None and left[0] == right[0] and left[1] > right[1]


def _scrub_temporal(obj, cutoff):
    if isinstance(obj, list):
        result = []
        for item in obj:
            cleaned = _scrub_temporal(item, cutoff)
            if cleaned is not _DROP:
                result.append(cleaned)
        return result
    if not isinstance(obj, dict):
        return copy.deepcopy(obj)

    for time_field in ("measurement_time", "administration_time", "start_time", "event_time", "assessment_time"):
        if time_field in obj and _after(obj.get(time_field), cutoff):
            return _DROP

    result = {}
    result_time = obj.get("result_available_time", obj.get("result_time"))
    has_result_claim = obj.get("result_status") == "resulted" or result_time is not None
    result_not_visible = has_result_claim and not time_at_or_before(result_time, cutoff)
    for key, value in obj.items():
        if key in DIRECT_ESRP_KEYS:
            continue
        if result_not_visible and key in {
            "value", "result", "result_summary", "result_available_time", "result_time"
        }:
            continue
        if key in {"order_time", "sample_time", "collection_time"} and _after(value, cutoff):
            result[key] = None
            continue
        if key == "completion_time" and _after(value, cutoff):
            result[key] = None
            continue
        cleaned = _scrub_temporal(value, cutoff)
        if cleaned is not _DROP:
            result[key] = cleaned
    if result_not_visible:
        result["result_status"] = "pending"
    if _after(obj.get("completion_time"), cutoff):
        for key in ("completed", "initial_fluid_30ml_per_kg_completed"):
            if key in result:
                result[key] = None
    return result


def sanitize_t0_payload(payload: dict) -> dict:
    """Remove forbidden fields, direct ESRP, and records not available by T0."""
    safe = strip_inference_forbidden(payload)
    safe.pop("prior_node_uncertainty", None)
    safe.pop("uncertainty_audit", None)
    cutoff = safe.get("information_cutoff", safe.get("t0_time", 0.0))
    cleaned = _scrub_temporal(safe, cutoff)
    return {} if cleaned is _DROP else cleaned


def _resolve(payload: dict, path: str):
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None, False
        current = current[part]
    return current, True


def validate_d0(payload: dict) -> dict | None:
    safe = sanitize_t0_payload(payload)
    missing = [path for path in T0_REQUIRED if not _resolve(safe, path)[1]]
    if not missing:
        return None
    return {
        "abstain_reason": "upstream_missing" if any(path.startswith(("agent1", "agent2")) for path in missing) else "record_missing",
        "missing_information": missing,
        "pipeline_alarm": True,
    }


def evaluate_t0_icu_triggers(payload: dict) -> dict:
    """Evaluate only active Tier A support triggers visible at or before T0."""
    safe = sanitize_t0_payload(payload)
    cutoff = safe.get("information_cutoff", safe.get("t0_time", 0.0))
    state = safe.get("t0_state") or {}
    support = state.get("active_support") or state
    triggers: list[dict] = []
    governance_exclusions: list[dict] = []

    vasopressor = support.get("vasopressor") or {}
    if (
        vasopressor.get("active") is True
        and vasopressor.get("setting") in {"ed", "pre_arrival"}
        and time_at_or_before(vasopressor.get("start_time"), cutoff)
    ):
        triggers.append({"id": "ICU1_active_ed_or_prearrival_vasopressor", "field": "t0_state.active_support.vasopressor"})
    if vasopressor.get("setting") == "icu":
        governance_exclusions.append({
            "criterion": "vasopressor",
            "reason": "vaso_setting_icu_is_future_information_for_prospective_ed_disposition",
        })

    ventilation = support.get("ventilation") or {}
    status = ventilation.get("status")
    if ventilation.get("current_encounter") is True and time_at_or_before(ventilation.get("start_time"), cutoff):
        if status == "InvasiveVent":
            triggers.append({"id": "ICU2_invasive_ventilation", "field": "t0_state.active_support.ventilation"})
        elif status in {"NonInvasiveVent", "HFNC"}:
            triggers.append({"id": "ICU3_niv_or_hfnc", "field": "t0_state.active_support.ventilation"})

    return {
        "icu": bool(triggers),
        "triggers": triggers,
        "tier_a_triggers": triggers,
        "tier_b_candidates": [],
        "governance": {
            "tier_a": "active",
            "tier_b": "pending_clinical_signoff",
            "tier_b_active": False,
            "explicit_non_criteria": [
                "supplemental_oxygen_alone",
                "chronic_tracheostomy_predating_encounter",
                "sep3",
                "sofa_ge_2",
                "high_news",
                "sep3_shock_derived_label",
            ],
        },
        "governance_exclusions": governance_exclusions,
        "tier_b_inactive_at_t0": [
            "ICU4_post_resuscitation_hypotension",
            "ICU5_post_resuscitation_lactate",
            "ICU6_hypoxemia_despite_oxygen",
        ],
    }


def unresolved_workup(payload: dict) -> dict:
    plan = ((sanitize_t0_payload(payload).get("agent2") or {}).get("workup_recommendation") or {})
    items = []
    for group in ("recommendations", "indeterminate"):
        for entry in plan.get(group) or []:
            if isinstance(entry, dict) and entry.get("item"):
                items.append({"item": entry["item"], "agent2_group": group})
    return {"continue_workup": bool(items), "items": items}


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


def user_prompt(node: dict, payload: dict, *, prior_trace: list[dict] | None = None) -> str:
    safe_payload = sanitize_t0_payload(payload)
    allowed_paths = available_field_paths(safe_payload, decision_point="T0")
    return (
        f"NODE\n{'=' * 60}\n{node['id']} - {node['title']}\n"
        f"{node['question']}\nallowed answers: {node['allowed_answers']}\n\n"
        f"SCHEMA-GROUNDED EVIDENCE FIELD PATHS\n{'=' * 60}\n"
        "Every key_reasons/unresolved_risks field must be selected exactly from this list. "
        "Do not invent or shorten a path.\n"
        f"{json.dumps(allowed_paths, ensure_ascii=False, indent=1)}\n\n"
        f"T0 INFERENCE-SAFE PAYLOAD\n{'=' * 60}\n"
        f"{json.dumps(safe_payload, ensure_ascii=False, indent=1)}"
        + prior_uncertainty_prompt(node["id"], prior_trace or [],
                                   safe_payload.get("information_cutoff", safe_payload.get("t0_time", 0.0)))
    )


CAPS = {"temperature": True}


def call_model(client, model, system, user, temperature):
    # Same negotiation as run_agent1: a model that rejects temperature (reasoning
    # models fix it at 1) gets the call without it, and the drop is recorded.
    kwargs = {"model": model, "response_format": {"type": "json_object"},
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
              **model_usage.request_options()}
    if temperature is not None and CAPS["temperature"]:
        kwargs["temperature"] = temperature
    while True:
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            dropped = model_usage.drop_rejected(kwargs, exc)
            if dropped is None:
                raise
            if dropped in CAPS:
                CAPS[dropped] = False
            print(f"   [note] {model} rejects {dropped}; continuing without it")
    model_usage.record(sys.modules[__name__], response)

    fingerprint = getattr(response, "system_fingerprint", None)
    if fingerprint:
        FINGERPRINTS.add(fingerprint)
    returned_model = getattr(response, "model", None)
    if returned_model:
        MODEL_SNAPSHOTS.add(returned_model)
    return response.choices[0].message.content, response.usage.total_tokens if response.usage else 0


def dry_reply(node: dict, payload: dict):
    answer = (payload.get("mock_llm_answers") or {}).get(node["id"])
    if answer is None:
        answer = {
            "answer": "insufficient_evidence",
            "key_reasons": [],
            "unresolved_risks": [],
            "missing_information": ["synthetic dry-run answer not supplied"],
            "escalation_trigger": None,
        }
    return json.dumps(answer, ensure_ascii=False)


def validate_reply(reply: dict, node: dict, payload: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not isinstance(reply, dict):
        return False, ["not_an_object"]
    reply, _ = canonicalize_reply_fields(reply)
    if reply.get("answer") not in node["allowed_answers"]:
        failures.append(f"V1_answer_not_allowed:{reply.get('answer')!r}")
    for field in ("key_reasons", "unresolved_risks", "missing_information", "escalation_trigger"):
        if field not in reply:
            failures.append(f"V2_missing_output_field:{field}")
    safe = sanitize_t0_payload(payload)
    allowed_evidence_paths = set(available_field_paths(safe, decision_point="T0"))
    for group in ("key_reasons", "unresolved_risks"):
        for item in reply.get(group) or []:
            field = item.get("field") if isinstance(item, dict) else None
            if not field:
                failures.append(f"V3_reason_missing_field:{group}")
            elif field not in allowed_evidence_paths:
                failures.append(f"V3_reason_field_not_allowed:{field}")
            elif not _resolve(safe, field)[1]:
                failures.append(f"V3_reason_field_missing:{field}")
    if node["id"] == "D4_T0" and reply.get("answer") == "discharge" and not reply.get("escalation_trigger"):
        failures.append("V4_discharge_requires_escalation_trigger")
    return not failures, failures


def terminal_output(terminal_id: str, trace: list[dict], detail: dict | None = None) -> dict:
    terminal = GRAPH["terminals"][terminal_id]
    detail = detail or {}
    cutoff = next(
        (step.get("information_cutoff") for step in reversed(trace) if step.get("information_cutoff") is not None),
        None,
    )
    return {
        "terminal": terminal_id,
        "result": {
            "status": terminal["status"],
            "disposition": terminal["disposition"],
            "continue_workup": terminal["continue_workup"],
            "decision_time": "T0",
            "information_cutoff": cutoff,
            "key_reasons": detail.get("key_reasons", []),
            "unresolved_risks": detail.get("unresolved_risks", []),
            "missing_information": detail.get("missing_information", []),
            "escalation_trigger": detail.get("escalation_trigger"),
            "abstain_reason": detail.get("abstain_reason"),
            "pipeline_alarm": bool(detail.get("pipeline_alarm", False)),
        },
        "trace": trace,
        "uncertainty_audit": uncertainty_audit(trace, cutoff),
    }


def run_case(payload: dict, client, args) -> dict:
    trace: list[dict] = []
    d0 = validate_d0(payload)
    if d0:
        trace.append({
            "node": "D0_T0",
            "type": "deterministic",
            "question": "Are all required T0 and upstream structural fields present?",
            "allowed_answers": ["yes", "no"],
            "answer": "no",
            "outcome": "abstained",
            **d0,
        })
        return {"case_id": payload.get("case_id"), **terminal_output("T_ABSTAIN", trace, d0)}

    safety = evaluate_t0_icu_triggers(payload)
    d1 = GRAPH["nodes"]["D1_T0"]
    trace.append({
        "node": "D1_T0",
        "type": "deterministic_safety_gate",
        "question": d1["question"],
        "allowed_answers": d1["allowed_answers"],
        "answer": "yes" if safety["icu"] else "no",
        "icu_triggers": safety["triggers"],
        "icu_gate_governance": safety["governance"],
        "tier_b_candidates": safety["tier_b_candidates"],
        "tier_b_inactive_at_t0": safety["tier_b_inactive_at_t0"],
        "governance_exclusions": safety["governance_exclusions"],
        "information_cutoff": sanitize_t0_payload(payload).get("information_cutoff", 0.0),
    })
    if safety["icu"]:
        detail = {"key_reasons": [{"field": item["field"], "detail": item["id"]} for item in safety["triggers"]]}
        return {"case_id": payload.get("case_id"), **terminal_output("T_ICU", trace, detail)}

    node_id = "D2_T0"
    while node_id in GRAPH["nodes"]:
        node = dict(GRAPH["nodes"][node_id], id=node_id)
        if node["type"] == "deterministic_workup_gate":
            gate = unresolved_workup(payload)
            answer = "yes" if gate["continue_workup"] else "no"
            trace.append({
                "node": node_id,
                "type": node["type"],
                "question": node["question"],
                "allowed_answers": node["allowed_answers"],
                "answer": answer,
                "unresolved_workup": gate["items"],
            })
            node_id = node["transitions"][answer]
            if node_id.startswith("T_"):
                return {"case_id": payload.get("case_id"), **terminal_output(node_id, trace, {
                    "unresolved_risks": [
                        {
                            "field": "agent2.workup_recommendation",
                            "detail": f"{item['item']} remains in Agent 2 group {item['agent2_group']}",
                        }
                        for item in gate["items"]
                    ]
                })}
            continue

        attempts = []
        reply = None
        successful_repairs = []
        failures = ["no_attempt"]
        tokens = 0
        prompt = user_prompt(node, payload, prior_trace=trace)
        for attempt in range(2):
            system = SYSTEM if attempt == 0 else SYSTEM + "\n\nPrevious reply rejected: " + "; ".join(failures)
            raw, used = (dry_reply(node, payload), 0) if args.dry_run else call_model(
                client, args.model, system, prompt, args.temperature
            )
            tokens += used or 0
            candidate = parse_json(raw)
            candidate, field_repairs = canonicalize_reply_fields(candidate)
            ok, failures = validate_reply(candidate, node, payload) if candidate else (False, ["unparseable_json"])
            attempts.append({
                "attempt": attempt,
                "ok": ok,
                "failures": failures,
                "field_reference_repairs": field_repairs,
                "raw": raw,
            })
            if ok:
                reply = candidate
                successful_repairs = field_repairs
                break
        step = {
            "node": node_id,
            "type": node["type"],
            "question": node["question"],
            "allowed_answers": node["allowed_answers"],
            "attempts": attempts,
            "tokens": tokens,
            "field_reference_valid": reply is not None,
            "field_reference_repairs": successful_repairs,
        }
        if reply is None:
            step["outcome"] = "abstained"
            trace.append(step)
            return {"case_id": payload.get("case_id"), **terminal_output("T_ABSTAIN", trace, {
                "abstain_reason": "invalid_model_output",
                "missing_information": failures,
            })}
        step.update(answer=reply["answer"], reply=reply)
        trace.append(step)
        node_id = node["transitions"][reply["answer"]]
        if node_id.startswith("T_"):
            return {"case_id": payload.get("case_id"), **terminal_output(node_id, trace, reply)}

    return {"case_id": payload.get("case_id"), **terminal_output("T_ABSTAIN", trace, {
        "abstain_reason": "routing_error", "pipeline_alarm": True
    })}


def load_payloads(path: pathlib.Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("T0 Agent 4 input must be an object, list, or {'cases': [...]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", default=str(paths.DATA / "agent4_t0_run.json"))
    parser.add_argument("--dry-run", action="store_true")
    parallel.add_workers_argument(parser)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    run_metadata = new_run_metadata("agent4-t0", ROOT)

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    started = time.time()
    payloads = load_payloads(pathlib.Path(args.infile))
    results = parallel.run_cases(
        payloads, lambda payload: run_case(payload, client, args),
        workers=getattr(args, "workers", 1), progress=False)
    output = {
        "meta": {
            **run_metadata,
            "agent": GRAPH["agent"],
            "graph_version": GRAPH["version"],
            "dry_run": args.dry_run,
            "model": "synthetic-stub" if args.dry_run else args.model,
            "model_requested": args.model,
            "base_url": args.base_url,
            "sampling": {
                "temperature_requested": args.temperature,
                "seed_requested": None,
                **model_runtime_metadata(sys.modules[__name__]),
            },
            "seconds": round(time.time() - started, 3),
            "n": len(results),
            "clinical_validation": False,
        },
        "results": results,
    }
    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=1), encoding="utf-8")
    safe_print(f"-> {destination}")


if __name__ == "__main__":
    main()
