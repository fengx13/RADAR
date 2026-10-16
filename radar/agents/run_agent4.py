#!/usr/bin/env python3
"""Agent 4 (ED disposition) runner.

Agent 4 consumes an externally assembled reassessment-checkpoint payload and upstream Agent 1/2/3
outputs. It never uses future truth/outcome fields during inference.
"""
from __future__ import annotations
from radar import paths

import argparse
from radar.common import model_usage
import json
import os
import pathlib
import re
import sys
import time

from radar.common import parallel

from radar.agents.agent4_insufficiency import (
    deployment_unavailable_fields,
    _MISSING,
    BLOCKABLE,
    CANONICAL_MUST_EXIST,
    CANONICAL_PENDING,
    MUST_EXIST,
    PENDING,
    VALID_DISPOSITIONS,
    apply_uncertainty_ceiling,
    canonical_field,
    field_availability,
    resolve,
    route_insufficiency,
    validate_insufficiency,
)
from radar.agents.agent4_reason_consistency import reason_consistency_checks
from radar.agents.agent4_safety import evaluate_icu_triggers
from radar.common.field_addressing import available_field_paths, canonicalize_reply_fields
from radar.common.integration_schema import strip_inference_forbidden
from radar.common.run_provenance import model_runtime_metadata, new_run_metadata

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
GRAPH = json.loads((paths.GRAPH / "agent4_disposition_graph.json").read_text(encoding="utf-8"))
FINGERPRINTS = set()
MODEL_SNAPSHOTS = set()

SENSITIVE_KEYS = {"truth", "outcomes", "evaluation_only"}
FUTURE_OUTCOME_KEYS = {
    "dsp_hospitalization",
    "dsp_icu_transfer_12h",
    "dsp_critical",
    "dsp_ed_revisit_3d",
    "dsp_mortality_inhospital",
    "dsp_mortality_30d",
}


def safe_print(text: str):
    encoding = sys.stdout.encoding or "utf-8"
    sys.stdout.write(text.encode(encoding, errors="backslashreplace").decode(encoding) + "\n")

SYSTEM = """You are one node of Agent 4, an ED disposition system.

Agent 4 answers: what is the minimum safe level of care at the current reassessment checkpoint?
Do not re-diagnose sepsis, repeat workup decisions, or re-decide treatment.

Rules:
- Answer only the current node question.
- `answer` must be exactly one allowed value.
- Reasons must be grounded in exact fields from the supplied checkpoint/upstream payload.
- Do not invent patient facts.
- Do not cite SSC disposition recommendations; Agent 4 currently has no SSC ED-disposition scope.
- If answering insufficient_evidence, provide `blocking` fields and, for pending fields, a counterfactual where different possible results lead to different dispositions.
- If answering yes at D3, provide `observation_basis` using the same field/counterfactual structure as an insufficient_evidence blocking item.
- Evidence field paths and V4 field allowlists are different. Select blocking[].field and observation_basis.field only from their node-specific V4 lists, not the evidence-path list.
- Each counterfactual disposition must be ONE exact, case-sensitive value from the supplied disposition enum. Do not combine values or add words. if_a and if_b must have different dispositions justified by the current case; pending alone is not an observation basis.
- Both counterfactual branches require a non-empty condition string. These are hypothetical conditions, not observed results.
- When quoting an observed value or result status, use a short literal assertion about the referenced field. Keep interpretation, historical comparisons and hypothetical conditions separate. Never claim a pending result has already returned.
- At D4, if `answer` is `yes`, `escalation_trigger` is REQUIRED and must be a non-empty string containing a concrete escalation/return instruction. A D4 discharge response with a null or empty trigger fails validation.

Return one JSON object. Replace placeholders; set observation_basis to null unless answering yes at D3:
{
  "answer": "<allowed value>",
  "key_reasons": [{"field": "...", "detail": "..."}],
  "unresolved_risks": [{"field": "...", "detail": "..."}],
  "missing_information": ["..."],
  "escalation_trigger": "REQUIRED non-empty string when D4 answer=yes; otherwise string or null",
  "observation_basis": {
    "field": "<one exact observation_basis.field from the node V4 list>",
    "counterfactual": {
      "if_a": {"condition": "...", "disposition": "<one exact disposition enum value>"},
      "if_b": {"condition": "...", "disposition": "<one different exact disposition enum value>"}
    }
  },
  "blocking": [
    {
      "field": "<one exact blocking.field from the node V4 list>",
      "counterfactual": {
        "if_a": {"condition": "...", "disposition": "<one exact disposition enum value>"},
        "if_b": {"condition": "...", "disposition": "<one different exact disposition enum value>"}
      }
    }
  ]
}"""


def strip_sensitive(obj):
    """Remove future truth/outcome keys before inference prompting."""
    if isinstance(obj, dict):
        return {
            k: strip_sensitive(v)
            for k, v in obj.items()
            if k not in SENSITIVE_KEYS and k not in FUTURE_OUTCOME_KEYS and k != "mock_llm_answers"
        }
    if isinstance(obj, list):
        return [strip_sensitive(v) for v in obj]
    return obj


def normalize_payload(payload: dict) -> dict:
    """Normalize the legacy single-T1 payload to the checkpoint contract."""
    safe = strip_inference_forbidden(strip_sensitive(payload), exclude_temporal_audit=True)
    if "patient_state" not in safe and "t1_state" in safe:
        safe["patient_state"] = safe.pop("t1_state")
    if "information_cutoff" not in safe and "t1_time" in safe:
        safe["information_cutoff"] = safe["t1_time"]
    safe.setdefault("decision_time", safe.get("checkpoint_id") or "T1")
    return safe


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


def user_prompt(node: dict, payload: dict) -> str:
    safe_payload = normalize_payload(payload)
    allowed_paths = available_field_paths(safe_payload)
    blockable = {canonical_field(field) for field in node.get("blockable_fields", BLOCKABLE)}
    v4_fields = {
        "blocking.field": sorted(blockable),
        "observation_basis.field": sorted(blockable & CANONICAL_PENDING) if node["id"] == "D3" else [],
    }
    v4_contract = (
        f"NODE-SPECIFIC V4 FIELD SETS\n{'=' * 60}\n"
        f"{json.dumps(v4_fields, ensure_ascii=False)}\n"
        "Use exact paths from these lists. A listed field may block only when actually missing/empty "
        "or pending under V4; do not claim a present result is missing. "
        "Do not append .status or .value unless that full path is explicitly listed.\n"
        "COUNTERFACTUAL DISPOSITION ENUM (one string per branch): "
        f"{json.dumps(sorted(VALID_DISPOSITIONS))}\n"
        'Schema only: "disposition": "<one exact value from the enum>". '
        'Invalid: "discharge or observation", "ward or ICU", "ward admission", or a list. '
        "if_a.disposition must differ from if_b.disposition. Explain case-grounded conditions; "
        "do not invent a result or force observation. Invalid paths fail V4a; invalid disposition "
        "values or non-decisive alternatives fail V4c.\n\n"
        "V4b FIELD AVAILABILITY (data facts, not disposition recommendations)\n"
        f"{json.dumps(field_availability(safe_payload, blockable), ensure_ascii=False)}\n"
        "v4b_unresolved=false means that field cannot be claimed as missing/pending in blocking "
        "or observation_basis. True only passes the availability check; V4a/V4c and the "
        "clinical node question still apply. null observed_pending means unspecified legacy "
        "metadata, not confirmation of an observed order.\n"
        "An unavailable result is not a normal result and does not establish an outstanding order. "
        "An A2 recommendation is not an observed order; an A3 recommendation is not administered "
        "treatment. Do not infer pending treatment response from unavailable treatment records. "
        "An empty pending_workup list means no observed pending items in that snapshot, not a "
        "missing list and not proof that all possible investigations were completed.\n"
        "lactate_initial is a structural field/status contract: an object with status=unavailable "
        "and value=null is present, but its result is unknown. Report that uncertainty in "
        "missing_information/unresolved_risks; do not invent a value, mark it normal, or claim "
        "the container is absent. Do not force a disposition to avoid validation.\n\n"
    )
    unavailable = sorted(deployment_unavailable_fields(safe_payload))
    if unavailable:
        v4_contract += (
            "DEPLOYMENT-UNAVAILABLE FIELDS (this data source does not record them)\n"
            f"{json.dumps(unavailable)}\n"
            "Their value is unknown for every patient here, not missing for this one. They cannot be "
            "used as a blocking field (V4a). Decide from the information that is recorded; state the "
            "unknown in unresolved_risks. Discharge is already prevented by the uncertainty ceiling.\n\n"
        )
    conditional_contract = ""
    if node["id"] == "D4":
        conditional_contract = (
            "\nD4 CONDITIONAL OUTPUT CONTRACT\n"
            + "=" * 60
            + "\nIf answer == \"yes\", escalation_trigger is REQUIRED and must be a "
            "non-null, non-empty string with a concrete escalation/return instruction.\n"
            "A discharge response with escalation_trigger=null or an empty string will fail "
            "validation with V3_discharge_requires_escalation_trigger.\n"
            "Valid dependency example: "
            '{"answer":"yes","escalation_trigger":"Return for worsening symptoms or clinician-specified concern."}\n'
            "Invalid dependency example: "
            '{"answer":"yes","escalation_trigger":null}\n'
        )
    return (
        f"NODE\n{'=' * 60}\n"
        f"{node['id']} — {node['title']}\n"
        f"{node['question']}\n"
        f"allowed answers: {node['allowed_answers']}\n"
        f"{conditional_contract}\n"
        f"SCHEMA-GROUNDED EVIDENCE FIELD PATHS\n{'=' * 60}\n"
        "Every key_reasons/unresolved_risks field must be selected exactly from this list. "
        "Do not invent or shorten a path. This is NOT the blocking-field allowlist.\n"
        f"{json.dumps(allowed_paths, ensure_ascii=False, indent=1)}\n\n"
        f"{v4_contract}"
        f"CURRENT CHECKPOINT AND UPSTREAM PAYLOAD\n{'=' * 60}\n"
        f"{json.dumps(safe_payload, ensure_ascii=False, indent=1)}"
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
    answers = payload.get("mock_llm_answers") or {}
    item = answers.get(node["id"])
    if item is None:
        return json.dumps({
            "answer": "insufficient_evidence",
            "key_reasons": [],
            "unresolved_risks": [],
            "missing_information": [f"missing mock_llm_answers.{node['id']}"],
            "escalation_trigger": None,
            "observation_basis": None,
            "blocking": [{"field": "treatment_state.treatment_response_flag"}],
        })
    return json.dumps(item, ensure_ascii=False)


def _iter_reason_items(reply: dict):
    for field in ("key_reasons", "unresolved_risks"):
        for item in reply.get(field) or []:
            yield field, item


def validate_reply(reply: dict, node: dict, payload: dict):
    failures = []
    if not isinstance(reply, dict):
        return False, ["not_an_object"]

    reply, _ = canonicalize_reply_fields(reply)
    answer = reply.get("answer")
    if answer not in node["allowed_answers"]:
        failures.append(f"V1_answer_not_allowed:{answer!r}")

    for field in ("key_reasons", "unresolved_risks", "missing_information"):
        if field not in reply:
            failures.append(f"V2_missing_output_field:{field}")
    if "escalation_trigger" not in reply:
        failures.append("V2_missing_output_field:escalation_trigger")

    safe_payload = normalize_payload(payload)
    allowed_evidence_paths = set(available_field_paths(safe_payload))
    for field_name, item in _iter_reason_items(reply):
        if not isinstance(item, dict) or not item.get("field"):
            failures.append(f"V4_reason_missing_field:{field_name}")
            continue
        ref = item["field"]
        if ref not in allowed_evidence_paths:
            failures.append(f"V4_reason_field_not_allowed:{ref}")
        elif resolve(safe_payload, ref) is _MISSING:
            failures.append(f"V4_reason_field_missing:{ref}")

    if answer == "insufficient_evidence":
        ok, v4_failures = validate_insufficiency(reply, node, safe_payload)
        if not ok:
            failures.extend(v4_failures)

    if node["id"] == "D3" and answer == "yes":
        basis = reply.get("observation_basis")
        if not isinstance(basis, dict):
            failures.append("V3_observation_requires_structured_basis")
        else:
            field = basis.get("field")
            if field not in PENDING:
                failures.append(f"V3_observation_basis_not_pending:{field}")
            proxy = {"answer": "insufficient_evidence", "blocking": [basis]}
            ok, v4_failures = validate_insufficiency(proxy, node, safe_payload)
            if not ok:
                failures.extend(f"observation_basis_{failure}" for failure in v4_failures)

    if node["id"] == "D4" and answer == "yes" and not reply.get("escalation_trigger"):
        failures.append("V3_discharge_requires_escalation_trigger")

    for check in reason_consistency_checks(reply, safe_payload):
        if check["status"] == "contradiction":
            failures.append(f"V4_reason_value_contradiction:{check['field']}:{check['reason']}")

    return not failures, failures


def terminal_output(terminal_id: str, trace: list[dict], detail: dict | None = None):
    terminal = GRAPH["terminals"][terminal_id]
    detail = detail or {}
    status = terminal["status"]
    disposition = terminal["disposition"]
    result = {
        "status": status,
        "disposition": disposition,
        "graph_version": GRAPH["version"],
        "key_reasons": detail.get("key_reasons", []),
        "unresolved_risks": detail.get("unresolved_risks", []),
        "missing_information": detail.get("missing_information", []),
        "escalation_trigger": detail.get("escalation_trigger"),
        "abstain_reason": detail.get("abstain_reason"),
        "pipeline_alarm": bool(detail.get("pipeline_alarm", False)),
        "insufficiency": detail.get("v4_detail"),
    }
    if status == "abstain":
        result["disposition"] = None
    return {"terminal": terminal_id, "result": result, "trace": trace}


def route_v4(reply: dict, trace: list[dict]):
    outcome, detail = route_insufficiency(reply)
    if outcome == "ABSTAIN":
        return "T_ABSTAIN", {
            "abstain_reason": detail.get("abstain_reason"),
            "missing_information": detail.get("missing_information", []),
            "pipeline_alarm": detail.get("pipeline_alarm", False),
            "key_reasons": [],
            "unresolved_risks": [],
        }, 0
    return "T_OBSERVATION", {
        "key_reasons": [{"field": "blocking", "detail": detail.get("observation_basis")}],
        "unresolved_risks": [
            {
                "field": item.get("field"),
                "detail": "pending disposition-relevant information",
                "checkpoint_eligible": item.get("checkpoint_eligible"),
                "provenance": item.get("provenance"),
            }
            for item in detail.get("pending_items", []) if item
        ],
        "missing_information": detail.get("awaiting", []),
        "escalation_trigger": None,
        "abstain_reason": None,
        "v4_detail": detail,
    }, 1


def validate_d0(payload: dict):
    """Minimal structural validation before Agent 4 disposition reasoning."""
    safe_payload = normalize_payload(payload)
    missing = [field for field in sorted(CANONICAL_MUST_EXIST) if resolve(safe_payload, field) is _MISSING]
    if not missing:
        return None
    upstream = any(
        field.startswith(("agent1.", "agent2.", "agent3.", "workup_state.", "treatment_state."))
        for field in missing
    )
    return {
        "abstain_reason": "upstream_missing" if upstream else "record_missing",
        "missing_information": missing,
        "pipeline_alarm": True,
        "key_reasons": [],
        "unresolved_risks": [],
    }


def run_case(payload: dict, client, args):
    trace = []
    carried_insufficient = 0

    d0 = validate_d0(payload)
    if d0 is not None:
        trace.append({
            "node": "D0",
            "title": "Minimal input schema validation",
            "type": "deterministic",
            "question": "Are all required checkpoint and upstream structural fields present?",
            "allowed_answers": ["yes", "no"],
            "answer": "no",
            "graph_version": GRAPH["version"],
            "outcome": "abstained",
            "missing_information": d0["missing_information"],
            "pipeline_alarm": True,
        })
        return {"case_id": payload.get("case_id"), **terminal_output("T_ABSTAIN", trace, d0)}

    d1 = dict(GRAPH["nodes"]["D1"], id="D1")
    safety = evaluate_icu_triggers(normalize_payload(payload))
    d1_answer = "yes" if safety["icu"] else "no"
    trace.append({
        "node": "D1",
        "title": d1["title"],
        "type": d1["type"],
        "question": d1["question"],
        "allowed_answers": d1["allowed_answers"],
        "answer": d1_answer,
        "graph_version": GRAPH["version"],
        "icu_triggers": safety["tier_a_triggers"],
        "icu_gate_governance": safety["governance"],
        "tier_b_candidates": safety["tier_b_candidates"],
        "tier_a_missing_fields": safety["tier_a_missing_fields"],
        "tier_b_missing_fields": safety["tier_b_missing_fields"],
        "governance_exclusions": safety["governance_exclusions"],
        "excluded_post_checkpoint_fields": safety.get("excluded_post_checkpoint_fields", []),
        "information_cutoff": normalize_payload(payload).get("information_cutoff"),
    })
    if safety["icu"]:
        return {"case_id": payload.get("case_id"), **terminal_output("T_ICU", trace, {
            "key_reasons": [{"field": t["field"], "detail": t["id"]} for t in safety["triggers"]],
        })}

    node_id = "D2"
    last_reply = {}
    while node_id in GRAPH["nodes"]:
        node = dict(GRAPH["nodes"][node_id], id=node_id)
        if node["type"] != "llm_closed_set":
            return {"case_id": payload.get("case_id"), **terminal_output("T_ABSTAIN", trace, {
                "abstain_reason": f"unsupported_node_type:{node['type']}",
                "pipeline_alarm": True,
            })}

        attempts = []
        reply = None
        successful_repairs = []
        failures = ["no_attempt"]
        tokens = 0
        prompt = user_prompt(node, payload)
        for attempt in range(2):
            sys_prompt = SYSTEM if attempt == 0 else SYSTEM + (
                "\n\nPrevious reply rejected: " + "; ".join(failures)
            )
            raw, tok = (dry_reply(node, payload), 0) if args.dry_run else call_model(
                client, args.model, sys_prompt, prompt, args.temperature
            )
            tokens += tok or 0
            candidate = parse_json(raw)
            candidate, field_repairs = canonicalize_reply_fields(candidate)
            ok, failures = validate_reply(candidate, node, payload) if candidate else (False, ["unparseable_json"])
            attempts.append({
                "attempt": attempt,
                "ok": ok,
                "failures": failures,
                "field_reference_repairs": field_repairs,
                "reason_consistency_checks": reason_consistency_checks(candidate, normalize_payload(payload)),
                "raw": raw,
            })
            if ok:
                reply = candidate
                successful_repairs = field_repairs
                break

        step = {
            "node": node_id,
            "title": node["title"],
            "type": node["type"],
            "question": node["question"],
            "allowed_answers": node["allowed_answers"],
            "graph_version": GRAPH["version"],
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
                "pipeline_alarm": False,
                "missing_information": failures,
            })}

        step.update(answer=reply["answer"], reply=reply)
        trace.append(step)
        last_reply = reply

        if reply["answer"] == "insufficient_evidence":
            terminal_id, detail, carried = route_v4(reply, trace)
            carried_insufficient += carried
            return {"case_id": payload.get("case_id"), **terminal_output(terminal_id, trace, detail)}

        nxt = node["transitions"][reply["answer"]]
        if nxt.startswith("T_"):
            unavailable = sorted(deployment_unavailable_fields(normalize_payload(payload)))
            ceiling_count = carried_insufficient + (1 if unavailable else 0)
            terminal_id, ceiling_note = apply_uncertainty_ceiling(nxt, ceiling_count)
            if ceiling_note and unavailable and not carried_insufficient:
                ceiling_note = (f"{ceiling_note} (deployment-unavailable fields: {', '.join(unavailable)})")
            detail = {
                "key_reasons": last_reply.get("key_reasons", []),
                "unresolved_risks": last_reply.get("unresolved_risks", []),
                "missing_information": last_reply.get("missing_information", []),
                "escalation_trigger": last_reply.get("escalation_trigger"),
                "abstain_reason": None,
            }
            if ceiling_note:
                detail["unresolved_risks"] = detail["unresolved_risks"] + [
                    {"field": "insufficient_evidence", "detail": ceiling_note}
                ]
            if terminal_id == "T_ABSTAIN":
                detail["abstain_reason"] = "discharge_not_supported_without_alternative_disposition"
            return {"case_id": payload.get("case_id"), **terminal_output(terminal_id, trace, detail)}
        node_id = nxt

    return {"case_id": payload.get("case_id"), **terminal_output("T_ABSTAIN", trace, {
        "abstain_reason": "routing_error",
        "pipeline_alarm": True,
    })}


def load_payloads(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("payload file must be a case object, list, or {'cases': [...]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="JSON file containing Agent 4 checkpoint payload(s)")
    parser.add_argument("--out", default=str(paths.DATA / "agent4_run.json"))
    parser.add_argument("--dry-run", action="store_true")
    parallel.add_workers_argument(parser)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    run_metadata = new_run_metadata("agent4", ROOT)

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set")
        from openai import OpenAI

        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    payloads = load_payloads(pathlib.Path(args.infile))
    started = time.time()
    results = parallel.run_cases(
        payloads, lambda payload: run_case(payload, client, args),
        workers=getattr(args, "workers", 1), progress=False)
    meta = {
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
    }
    out = {"meta": meta, "results": results}
    pathlib.Path(args.out).parent.mkdir(exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    safe_print(f"-> {args.out}")


if __name__ == "__main__":
    main()
