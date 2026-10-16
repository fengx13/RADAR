#!/usr/bin/env python3
"""Agent 1 (recognition) — walk the SSC 2026 decision graph for each Phase 0 case.

Design (mirrors EASY-Triage, with one addition):
  1. control flow      : code + decision graph        — the model has no say
  2. evidence supply   : node-scoped verbatim inject  — no similarity search
  3. clinical judgement: model picks from a closed answer set
  4. expression        : left to a later coordinator; not this script

Three validations run on every model reply:
  V1  answer must be in the node's allowed_answers
  V2  every citation id must be inside the node's scope        (EASY-Triage has this)
  V3  every deterministically hydrated quote must exactly match its frozen source
      and appear verbatim in the injected text                         (EASY-Triage does NOT)

V3 is the addition: in the P4 audit 21.8% of page-cited quotes could not be found in the
text the model had actually been given, while V2 passed 100%.

Usage:
    export OPENAI_API_KEY=...            # or put it in .env next to this file
    python -m radar.agents.run_agent1 --model gpt-5-mini
    python -m radar.agents.run_agent1 --model qwen2.5-32b-instruct --base-url http://localhost:8000/v1
    python -m radar.agents.run_agent1 --dry-run       # no API calls; exercises graph + validators
"""
from __future__ import annotations
from radar import paths
import argparse, copy, json, os, pathlib, re, sys, time

from radar.common import parallel
from radar.common import model_usage
from radar.agents.agent1_fact_consistency import fact_consistency_checks
from radar.common.integration_schema import strip_inference_forbidden
from radar.agents import agent1_bedside_screen

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
GRAPH = json.loads((paths.GRAPH / "agent1_decision_graph.json").read_text(encoding="utf-8"))
STMTS = {s["id"]: s for s in json.loads((paths.PARSED / "ssc2026_statements.json").read_text(encoding="utf-8"))}

# Table 3 is not one of the 129 numbered statements but is quotable, and is the only place
# the possible/probable/definite ladder is operationalised. Expose it as an injectable unit.
_t3 = json.loads((paths.PARSED / "table3_terminology.json").read_text(encoding="utf-8"))
STMTS[_t3["id"]] = {
    "id": _t3["id"], "strength": "definitional table", "certainty": None,
    "statement": _t3["verbatim_intro"] + "  " + "  ".join(
        f'{t["tier"].replace("_", " ").title()}: {t["verbatim"]}' for t in _t3["tiers"]),
    "remarks": [_t3["axis"]],
}
# Sepsis-3 supplies the definitions SSC 2026 assumes but never restates — above all
# what organ dysfunction IS. Without these R2 has no yardstick; see parse_sepsis3.py.
for _u in json.loads((paths.PARSED / "sepsis3_propositions.json").read_text(encoding="utf-8"))["propositions"]:
    STMTS[_u["id"]] = {"id": _u["id"], "strength": _u["strength"], "certainty": None,
                       "statement": _u["statement"], "remarks": _u["remarks"],
                       "source": _u["source"]}

_PHASE0 = paths.DATA / "phase0_cases.json"   # default pilot cases; patient data, outside the repo
CASES = json.loads(_PHASE0.read_text(encoding="utf-8")) if _PHASE0.exists() else []

SYSTEM = """You are one node of a sepsis recognition system in an emergency department.

You are shown (a) a patient as they appear at the moment triage is completed, and
(b) the complete verbatim text of the guideline statements that apply to THIS node.

Rules you must follow exactly:
- Answer ONLY the question for this node. Do not assign an overall sepsis level;
  that is decided elsewhere by code.
- `answer` must be exactly one of the allowed values. Nothing else is accepted.
- When the node requires `leading_alternative`, return it together with `answer`.
  It must be a non-null member of that node's allowed closed set and cannot be omitted.
- For grounded evidence, select the exact `statement_id` shown in the supplied text.
  The system attaches its canonical source quote deterministically; a model-provided
  `quote` is optional and is not trusted as the canonical grounding text.
- If a piece of reasoning has no supporting statement in the supplied text, you may
  still report it, but set `statement_id` and `quote` to null. Do not invent a citation.
- `opposing_evidence` is not optional. List what argues against the answer you gave.
  If nothing does, return an empty list.

CRITICAL — do not confuse "absent" with "not measured":
- A negative answer means the supplied information POSITIVELY SHOWS the thing is not
  there. It does not mean the information is missing.
- If the data needed to decide has not been obtained, the answer is `indeterminate`
  (when that is one of the allowed values), NOT `no`. Put what you would need in
  `missing_information`.
- A guideline statement telling you to ORDER a test is never evidence about that
  test's result. "We suggest measuring lactate" does not support "lactate is normal"
  or "there is no organ dysfunction". Do not cite an ordering statement as evidence
  about a finding.
- Do not treat a normal or unobtainable severity score as ruling the condition out.

Return one JSON object and nothing else:
{
  "answer": "<one of the allowed values>",
  "leading_alternative": "<required allowed value when requested for this node; otherwise omit>",
  "supporting_evidence": [
    {"patient_finding": "...", "statement_id": "stmt_00X", "quote": "optional", "implication": "..."}
  ],
  "opposing_evidence": [
    {"patient_finding": "...", "alternative_explanation": "...", "statement_id": null, "quote": null}
  ],
  "missing_information": ["..."],
  "confidence": "low" | "moderate" | "high"
}"""


def materials(node) -> str:
    out = []
    for sid in node["scope"]:
        s = STMTS[sid]
        head = f"[{sid}]  ({s['strength']}"
        head += f", {s['certainty']} certainty" if s["certainty"] else ""
        head += f" — {s['source']}" if s.get("source") else ""
        block = [head + ")", s["statement"]]
        block += [f"Remark: {r}" for r in s["remarks"]]
        out.append("\n".join(block))
    return "\n\n".join(out)


def question_for_time(node, decision_time: str) -> str:
    question = node["question"]
    if decision_time != "T0":
        question = question.replace(
            "Based only on the information available at triage",
            "Based only on the information available by the current reassessment checkpoint",
        ).replace(
            "what is known at triage",
            "what is known by the current reassessment checkpoint",
        )
        if node.get("id") == "R2":
            question = (
                "Based only on information available by the current reassessment checkpoint, "
                "is there acute organ dysfunction attributable to the suspected infection? "
                "Use only bedside findings and diagnostic results available by the checkpoint "
                "cutoff. `checkpoint_state.sofa` is computed in code from the components that have "
                "resulted: when `sofa_ge_2_established` is true, SOFA >= 2 is met (components not yet "
                "resulted can only raise the score). Apply sepsis3::baseline_sofa: the baseline SOFA is "
                "zero unless the supplied history or comorbidities show preexisting organ dysfunction "
                "(for example chronic kidney disease, cirrhosis, chronic thrombocytopenia), so a "
                "code-established SOFA >= 2 without such documented preexisting dysfunction is an acute "
                "change; do not require baseline laboratory values that this setting never has. "
                "Attribute that change to the suspected infection unless a documented non-infectious "
                "cause explains it; the competing diagnosis is ranked at R3, not here. Do not infer "
                "unresulted components yourself. Answer `indeterminate` only when the resulted "
                "components and bedside findings cannot settle the question, not because cultures, "
                "imaging or prior values are absent."
            )
    return question


def alternative_candidate_ids(node: dict) -> list[str]:
    return [item["id"] for item in node.get("alternative_candidates") or []]


def node_output_contract(node: dict) -> str:
    if not node.get("requires_leading_alternative"):
        return ""
    candidates = alternative_candidate_ids(node)
    example_answer = (
        "alternative_very_unlikely"
        if "alternative_very_unlikely" in node["allowed_answers"]
        else node["allowed_answers"][0]
    )
    example_alternative = (
        "other_noninfectious"
        if "other_noninfectious" in candidates
        else candidates[0]
    )
    example = {
        "answer": example_answer,
        "leading_alternative": example_alternative,
        "supporting_evidence": [],
        "opposing_evidence": [],
        "missing_information": [],
        "confidence": "moderate",
    }
    return (
        "\n\nREQUIRED CLOSED-SET FIELD FOR THIS NODE\n"
        f"{'=' * 60}\n"
        "Your JSON response MUST include BOTH `answer` and `leading_alternative`, plus "
        "the existing evidence/reasoning fields. `leading_alternative` is REQUIRED: it "
        "cannot be null and cannot be omitted. A null or omitted value will fail "
        "V1 validation. Return it as exactly one candidate id from the list below. "
        "Select a named diagnosis when supported. `other_noninfectious` and "
        "`none_identified` are explicit auditable fallback values, not guideline citations.\n"
        f"allowed leading_alternative values: {candidates}\n"
        "Valid response-shape example (values remain case-dependent):\n"
        f"{json.dumps(example, ensure_ascii=False, indent=2)}"
    )


def patient_input(case) -> dict:
    return strip_inference_forbidden(case.get("input", {}), exclude_temporal_audit=True)


def information_cutoff_for(case, decision_time: str):
    safe = patient_input(case)
    if decision_time != "T0":
        state = safe.get("checkpoint_state") or safe.get("T1_state") or safe
        return state.get("information_cutoff", state.get("as_of_time", decision_time))
    return safe.get("information_cutoff", "T0")


def decision_label_for(case, decision_time: str) -> str:
    if decision_time != "checkpoint":
        return decision_time
    safe = patient_input(case)
    state = safe.get("checkpoint_state") or {}
    checkpoint = safe.get("checkpoint") or {}
    return state.get("checkpoint_id", checkpoint.get("checkpoint_id", "checkpoint"))


def disabled_risk_signal() -> dict:
    return {"enabled": False, "status": "disabled"}


def load_risk_signals(path) -> dict:
    """Load an ESRP mapping artifact and index one usable score per stay_id."""
    if path is None:
        return {}
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("risk-signal file must contain a list or {'records': [...]}")
    by_stay = {}
    for rec in records:
        sid = rec.get("stay_id")
        if sid is None:
            raise ValueError("risk-signal record missing stay_id")
        sid = int(sid)
        if sid in by_stay:
            raise ValueError(f"duplicate risk-signal stay_id: {sid}")
        value = rec.get("esrp_value")
        if value is None:
            raise ValueError(f"missing ESRP value for stay_id {sid}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric ESRP value for stay_id {sid}: {value!r}") from exc
        if not 0 <= value <= 100:
            raise ValueError(f"ESRP value out of 0-100 range for stay_id {sid}: {value}")
        if rec.get("t0_safe") is not True:
            raise ValueError(f"ESRP record is not marked T0-safe for stay_id {sid}")
        if rec.get("usable") is False:
            raise ValueError(f"ESRP record is marked unusable for stay_id {sid}")
        provenance_source = rec.get("source") or data.get("source_esrp_table")
        source_label = rec.get("source_label") or (
            f"{provenance_source}::{data.get('esrp_column', 'score_ESRP')}"
            if provenance_source
            else "external_mapping::score_ESRP"
        )
        by_stay[sid] = {
            "enabled": True,
            "status": "available",
            "name": "ESRP",
            "value": int(value) if value.is_integer() else value,
            "output_type": "score",
            "score_range": [0, 100],
            "decision_time": "T0",
            "source_type": "deterministic_score",
            "source_field": "score_ESRP",
            "provenance": provenance_source,
            "source": source_label,
            "t0_safe": True,
            "case_id": rec.get("case_id"),
            "stay_id": sid,
            "computed_or_existing": rec.get("computed_or_existing", "existing"),
        }
    return by_stay


def risk_signal_for_case(case, args) -> dict:
    mode = getattr(args, "risk_signal", "none")
    if mode == "none" or getattr(args, "decision_time", "T0") != "T0":
        return disabled_risk_signal()
    if mode != "esrp":
        raise ValueError(f"unsupported risk signal: {mode}")
    mapping = getattr(args, "risk_signal_map", None) or {}
    sid = case.get("stay_id")
    if sid is None:
        return {"enabled": False, "status": "unavailable", "reason": "case_missing_stay_id"}
    signal = mapping.get(int(sid))
    if signal is None:
        return {"enabled": False, "status": "unavailable", "reason": "missing_esrp_for_stay_id", "stay_id": int(sid)}
    if signal.get("case_id") and signal["case_id"] != case.get("case_id"):
        return {"enabled": False, "status": "unavailable", "reason": "case_id_mismatch", "stay_id": int(sid)}
    return copy.deepcopy(signal)


def risk_signal_prompt_block(risk_signal: dict, node_id: str) -> str:
    if not risk_signal or not risk_signal.get("enabled"):
        return "STRUCTURED T0 RISK SIGNAL\n" + "=" * 60 + "\nNo optional structured risk signal is enabled for this run."
    return (
        "STRUCTURED T0 RISK SIGNAL\n"
        f"{'=' * 60}\n"
        "ESRP is a triage-time machine-learning risk score derived from pre-test patient information. "
        "Treat it as an initial risk signal only. It does not establish infection, organ dysfunction, "
        "shock, or the final sepsis tier. Do not apply any ESRP threshold unless an approved artifact "
        "is explicitly supplied.\n"
        "Node safeguards: for R1 it is only general prior context and does not establish suspected infection; "
        "for R2 it is not evidence of organ dysfunction; for R3 it may inform general prior risk context but "
        "must not override actual case evidence; for R4 it is not proof of shock or hypoperfusion.\n"
        f"risk_signal: {json.dumps(risk_signal, ensure_ascii=False, indent=1)}"
    )


def risk_signal_trace_fields(risk_signal: dict) -> dict:
    if not risk_signal or not risk_signal.get("enabled"):
        return {"risk_signal_used": False}
    return {
        "risk_signal_used": True,
        "risk_signal_name": risk_signal.get("name"),
        "risk_signal_value": risk_signal.get("value"),
        "risk_signal_source": risk_signal.get("source"),
        "risk_signal_decision_time": risk_signal.get("decision_time"),
        "risk_signal_t0_safe": risk_signal.get("t0_safe"),
    }


def system_for_time(decision_time: str) -> str:
    if decision_time == "T0":
        return SYSTEM
    return SYSTEM.replace(
        "as they appear at the moment triage is completed",
        "as reassessed at the current longitudinal checkpoint after new result information becomes available",
    ).replace(
        "at triage",
        "by the current checkpoint cutoff",
    ) + (
        "\n\nREASSESSMENT MODE: Use only information available by the current checkpoint "
        "cutoff. The existing graph topology and frozen scopes are unchanged; do not "
        "invent a new transition or clinical rule."
    )


def bedside_screen_for(node, case, decision_time: str, enabled: bool = True):
    """R2 at T0 only: deterministic comparison of triage vitals against the node's T0 criteria."""
    if not enabled or node.get("id") != "R2" or decision_time != "T0":
        return None
    return agent1_bedside_screen.bedside_screen(patient_input(case))


def user_prompt(node, case, materials_text, decision_time="T0", risk_signal=None,
                bedside_screen=None) -> str:
    label = "PATIENT AT TRIAGE" if decision_time == "T0" else "PATIENT AT CURRENT REASSESSMENT CHECKPOINT"
    question = question_for_time(node, decision_time)
    screen_block = (agent1_bedside_screen.prompt_block(bedside_screen) + "\n\n") if bedside_screen else ""
    return (f"GUIDELINE STATEMENTS AVAILABLE AT THIS NODE\n"
            f"{'=' * 60}\n{materials_text}\n\n"
            f"{label}\n{'=' * 60}\n"
            f"{json.dumps(patient_input(case), ensure_ascii=False, indent=1)}\n\n"
            f"{screen_block}"
            f"{risk_signal_prompt_block(risk_signal or disabled_risk_signal(), node['id'])}\n\n"
            f"QUESTION\n{'=' * 60}\n{question}\n"
            f"allowed answers: {node['allowed_answers']}"
            f"{node_output_contract(node)}")


# ----------------------------------------------------------------- validation
def normalise(s: str) -> str:
    """Quote matching must survive smart quotes and whitespace, nothing more."""
    s = (s or "").replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip().lower()


def hydrate_evidence_quotes(reply: dict, node) -> tuple[dict, list[dict]]:
    """Attach exact frozen quotes for explicitly selected, scoped statement IDs."""
    hydrated = copy.deepcopy(reply)
    records = []
    scope = set(node["scope"])
    for field in ("supporting_evidence", "opposing_evidence"):
        for index, evidence in enumerate(hydrated.get(field) or []):
            if not isinstance(evidence, dict):
                continue
            statement_id = evidence.get("statement_id")
            if statement_id is None or statement_id not in scope:
                continue
            source = STMTS.get(statement_id)
            canonical_quote = source.get("statement") if isinstance(source, dict) else None
            if not isinstance(canonical_quote, str) or not canonical_quote:
                continue
            model_quote = evidence.get("quote")
            evidence["quote"] = canonical_quote
            evidence["quote_source"] = "deterministic_statement_lookup"
            evidence["quote_hydrated"] = True
            records.append({
                "evidence_field": field,
                "evidence_index": index,
                "statement_id": statement_id,
                "model_quote": model_quote,
                "quote_source": "deterministic_statement_lookup",
            })
    return hydrated, records


def validate(reply: dict, node, materials_text, *, case=None, decision_time="T0"):
    """Return (ok, [failure codes], stats)."""
    fails, stats = [], {"cited": 0, "uncited": 0}
    if not isinstance(reply, dict):
        return False, ["not_an_object"], stats
    if reply.get("answer") not in node["allowed_answers"]:
        fails.append(f"V1_answer_not_allowed:{reply.get('answer')!r}")
    if node.get("requires_leading_alternative"):
        leading = reply.get("leading_alternative")
        allowed_alternatives = alternative_candidate_ids(node)
        if leading not in allowed_alternatives:
            fails.append(f"V1_leading_alternative_not_allowed:{leading!r}")

    scope = set(node["scope"])
    for field in ("supporting_evidence", "opposing_evidence"):
        for ev in reply.get(field) or []:
            sid, quote = ev.get("statement_id"), ev.get("quote")
            if sid is None and quote is None:
                stats["uncited"] += 1
                continue
            stats["cited"] += 1
            if sid not in scope:
                fails.append(f"V2_out_of_scope:{field}:{sid}")
                continue
            source = STMTS.get(sid)
            if not isinstance(source, dict):
                fails.append(f"V2_statement_unresolved:{field}:{sid}")
                continue
            canonical_quote = source.get("statement")
            if not isinstance(canonical_quote, str) or not canonical_quote:
                fails.append(f"V3_canonical_source_missing:{field}:{sid}")
                continue
            if quote != canonical_quote or canonical_quote not in materials_text:
                fails.append(f"V3_quote_not_verbatim:{field}:{sid}")
    if case is not None:
        checks = fact_consistency_checks(reply, patient_input(case), decision_time)
        stats["fact_consistency_checks"] = checks
        fails.extend(f"V4_unknown_as_negative:{check['reply_field']}:{check['source_field']}"
                     for check in checks if check["status"] == "unsupported_negative")
    return not fails, fails, stats


# ----------------------------------------------------------------- model call
# Reasoning models (gpt-5*, o*) reject temperature and fix it at 1. We still want the
# most reproducible setting each model will accept, and we want the paper to state
# honestly which knobs were actually honoured — so negotiate once and record the result.
CAPS = {"temperature": True, "seed": True}
FINGERPRINTS = set()
MODEL_SNAPSHOTS = set()


def call_model(client, model, system, user, temperature, seed):
    kw = {"model": model, "response_format": {"type": "json_object"},
          "messages": [{"role": "system", "content": system},
                       {"role": "user", "content": user}], **model_usage.request_options()}
    if temperature is not None and CAPS["temperature"]: kw["temperature"] = temperature
    if seed is not None and CAPS["seed"]:               kw["seed"] = seed
    while True:
        try:
            r = client.chat.completions.create(**kw)
            break
        except Exception as e:
            dropped = model_usage.drop_rejected(kw, e)
            if dropped is None:
                raise
            if dropped in CAPS:
                CAPS[dropped] = False
            print(f"   [note] {model} 不接受 {dropped}，已去掉该参数继续"
                  f"{'（本次运行不再是确定性的，meta 会记录）' if dropped == 'temperature' else ''}")
    fp = getattr(r, "system_fingerprint", None)
    if fp: FINGERPRINTS.add(fp)
    returned_model = getattr(r, "model", None)
    if returned_model: MODEL_SNAPSHOTS.add(returned_model)
    model_usage.record(sys.modules[__name__], r)
    return r.choices[0].message.content, r.usage.total_tokens if r.usage else None


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def stub(node, case):
    """Offline stand-in so the graph and validators can be exercised without an API."""
    mock_answers = case.get("mock_agent1_answers") or {}
    if node["id"] in mock_answers:
        mock = mock_answers[node["id"]]
        ans = mock.get("answer") if isinstance(mock, dict) else mock
        ans = ans if ans in node["allowed_answers"] else node["allowed_answers"][0]
        sid = node["scope"][0]
        reply = {
            "answer": ans,
            "supporting_evidence": [{
                "patient_finding": "synthetic reassessment fixture",
                "statement_id": sid,
                "quote": STMTS[sid]["statement"],
                "implication": "engineering dry-run answer",
            }],
            "opposing_evidence": [],
            "missing_information": [],
            "confidence": "low",
        }
        if node.get("requires_leading_alternative"):
            requested = mock.get("leading_alternative") if isinstance(mock, dict) else None
            reply["leading_alternative"] = (
                requested if requested in alternative_candidate_ids(node) else "none_identified"
            )
        return json.dumps(reply)
    # Dry-run stub only. At a checkpoint the input carries checkpoint_state and
    # presentation_context instead of the triage snapshot; tolerate both.
    v = (case["input"].get("triage_vitals")
         or (case["input"].get("checkpoint_state") or {}).get("vitals") or {})
    v = {k: v.get(k) for k in ("resp_rate", "sbp")}
    cc = (case["input"].get("chief_complaint")
          or (case["input"].get("presentation_context") or {}).get("chief_complaint") or "").lower()
    infected = any(k in cc for k in ("fever", "ili", "wound", "infect", "cough", "uti"))
    ans = {"R1":  "yes" if infected else "no",
           "R2":  "yes" if (v["resp_rate"] or 0) >= 22 or (v["sbp"] or 999) <= 100 else "no",
           "R3a": "alternative_less_likely" if infected else "alternative_also_likely",
           "R3b": "alternative_also_likely",
           "R4":  "yes" if (v["sbp"] or 999) <= 100 else "no"}[node["id"]]
    ans = ans if ans in node["allowed_answers"] else node["allowed_answers"][0]
    sid = node["scope"][0]
    reply = {
        "answer": ans,
        "supporting_evidence": [{"patient_finding": "stub", "statement_id": sid,
                                 "quote": STMTS[sid]["statement"], "implication": "stub"}],
        "opposing_evidence": [], "missing_information": ["laboratory results"],
        "confidence": "low"}
    if node.get("requires_leading_alternative"):
        reply["leading_alternative"] = "none_identified" if infected else "other_noninfectious"
    return json.dumps(reply)


# ----------------------------------------------------------------- graph walk
def run_case(case, client, args):
    decision_time = getattr(args, "decision_time", "T0")
    decision_label = decision_label_for(case, decision_time)
    information_cutoff = information_cutoff_for(case, decision_time)
    risk_signal = risk_signal_for_case(case, args)
    trace, node_id, guard, answers_so_far = [], GRAPH["entry"], 0, {}
    while node_id in GRAPH["nodes"] and guard < 12:
        guard += 1
        node = dict(GRAPH["nodes"][node_id], id=node_id)
        mats = materials(node)
        screen = bedside_screen_for(node, case, decision_time,
                                    getattr(args, "bedside_screen", "on") == "on")
        up = user_prompt(node, case, mats, decision_time, risk_signal, screen)

        reply, fails, attempts, tokens = None, ["no_attempt"], [], 0
        for attempt in range(2):                      # one repair retry, then abstain
            base_system = system_for_time(decision_time)
            sys_prompt = base_system if attempt == 0 else base_system + (
                "\n\nYour previous reply was rejected: " + "; ".join(fails) +
                "\nSelect only statement ids that appear in the supplied statements; "
                "the system will attach their canonical quotes deterministically. Satisfy every required "
                "closed-set field for this node.")
            raw, tok = (stub(node, case), 0) if args.dry_run else \
                       call_model(client, args.model, sys_prompt, up,
                                  args.temperature, args.seed)
            tokens += tok or 0
            cand = parse_json(raw)
            hydrated, quote_hydrations = hydrate_evidence_quotes(cand, node) if cand else (None, [])
            ok, fails, stats = validate(hydrated, node, mats, case=case, decision_time=decision_time) if hydrated else (False, ["unparseable_json"], {})
            attempts.append({
                "attempt": attempt,
                "ok": ok,
                "failures": fails,
                "raw": raw,
                "quote_hydrations": quote_hydrations,
                "fact_consistency_checks": stats.get("fact_consistency_checks", []),
            })
            if ok:
                reply = hydrated
                break

        step = {"node": node_id, "title": node["title"],
                "question": question_for_time(node, decision_time),
                "allowed_answers": node["allowed_answers"],
                "decision_time": decision_label,
                "information_cutoff": information_cutoff,
                "graph_version": GRAPH["version"],
                "scope": node["scope"], "attempts": attempts, "tokens": tokens}
        step.update(risk_signal_trace_fields(risk_signal))
        if screen is not None:
            step["bedside_screen"] = screen
        if reply is None:
            step["outcome"] = "abstained"
            trace.append(step)
            abstain = dict(GRAPH["abstain"])
            abstain["graph_version"] = GRAPH["version"]
            return {"case_id": case["case_id"], "result": abstain,
                    "risk_signal": risk_signal, "trace": trace}
        step.update(answer=reply["answer"], reply=reply, ungrounded=stats.get("uncited", 0))
        if node.get("requires_leading_alternative"):
            leading = reply["leading_alternative"]
            step["leading_alternative"] = leading
            step["leading_alternative_specific"] = leading not in {
                "other_noninfectious", "none_identified"
            }
        trace.append(step)
        answers_so_far[node_id] = reply["answer"]
        nxt = node["transitions"][reply["answer"]]
        if isinstance(nxt, dict):                       # terminal = f(this answer, an earlier one)
            key = next((f"{n}:{answers_so_far[n]}" for n in nxt["depends_on"]
                        if n in answers_so_far), None)
            nxt = nxt["map"].get(key)
            if nxt is None:
                step["outcome"] = "routing_error"
                return {"case_id": case["case_id"], "terminal": "ROUTING_ERROR",
                        "result": {"tier": "error", "detail": f"no transition for {key!r}",
                                   "graph_version": GRAPH["version"]},
                        "trace": trace}
        node_id = nxt

    terminal = dict(GRAPH["terminals"].get(node_id, {"tier": "error"}))
    terminal["decision_time"] = decision_label
    terminal["information_cutoff"] = information_cutoff
    terminal["mode"] = "recognition" if decision_time == "T0" else "reassessment"
    # Which of Table 3's three information sources this tier actually rests on.
    # Table 3 words every tier as "history, clinical examination, and diagnostic
    # testing". At T0 the third does not exist, so a T0 tier — definite most of
    # all — is a pre-test judgement and must be reported separately from a
    # checkpoint one. E1 is scored as two groups on exactly this split.
    terminal["evidence_basis"] = "pre_test" if decision_time == "T0" else "post_test"
    if decision_time != "T0":
        terminal["graph_limitations"] = [
            "The v0.7 topology is shared by T0 and checkpoint reassessment; the two "
            "differ in what R2 and R3 may be answered on, not in their transitions.",
        ]
    terminal["graph_version"] = GRAPH["version"]
    return {"case_id": case["case_id"], "terminal": node_id,
            "result": terminal, "risk_signal": risk_signal, "trace": trace}


def load_cases(path):
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("Agent 1 input must be a case object, list, or {'cases': [...]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--base-url", default=None, help="set for vLLM / local servers")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="dropped automatically if the model refuses it (gpt-5*, o*)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=1,
                    help="rerun every case N times; reports how often the tier changes")
    ap.add_argument("--dry-run", action="store_true")
    parallel.add_workers_argument(ap)
    ap.add_argument("--decision-time", choices=("T0", "T1", "checkpoint"), default="T0",
                    help="T1 is a legacy reassessment label; checkpoint uses the supplied checkpoint id")
    ap.add_argument("--infile", default=None,
                    help="optional case JSON; defaults to the frozen Phase-0 cases")
    ap.add_argument("--risk-signal", choices=("none", "esrp"), default="none",
                    help="optional structured T0 risk prior; baseline none is the default")
    ap.add_argument("--risk-signal-file", default=str(paths.DATA / "phase0_esrp_mapping.json"),
                    help="ESRP mapping artifact used only when --risk-signal esrp")
    ap.add_argument("--bedside-screen", choices=("on", "off"), default="on",
                    help="R2 at T0: hand the model a deterministic RR/SBP/SpO2 criterion screen (PY-124); off for ablation")
    ap.add_argument("--preview-prompts", action="store_true",
                    help="print prompt previews without calling a model or writing results")
    ap.add_argument("--preview-case-id", default=None,
                    help="limit --preview-prompts to one case_id")
    ap.add_argument("--out", default=str(paths.DATA / "phase0_agent1_run.json"))
    args = ap.parse_args()
    args.risk_signal_map = load_risk_signals(args.risk_signal_file) if args.risk_signal == "esrp" else {}

    cases = load_cases(args.infile) if args.infile else CASES
    if args.preview_prompts:
        shown = 0
        for case in cases:
            if args.preview_case_id and case.get("case_id") != args.preview_case_id:
                continue
            risk_signal = risk_signal_for_case(case, args)
            node_id = GRAPH["entry"]
            node = dict(GRAPH["nodes"][node_id], id=node_id)
            print(f"--- PROMPT PREVIEW {case.get('case_id')} {node_id} risk_signal={args.risk_signal} ---")
            print(user_prompt(node, case, materials(node), args.decision_time, risk_signal,
                              bedside_screen_for(node, case, args.decision_time,
                                                 args.bedside_screen == "on")))
            shown += 1
        if shown == 0:
            raise SystemExit("no cases matched --preview-case-id")
        return

    client = None
    if not args.dry_run:
        env = paths.ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not set (put it in .env in the repository root)")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()

    t0, results, repeat_tiers = time.time(), [], {}

    def all_repeats(case):
        """Every repeat of one case, kept together so ordering stays stable."""
        out = []
        for rep in range(max(1, args.repeats)):
            r = run_case(case, client, args)
            if "truth" in case:
                r["truth"] = case["truth"]
            if "label" in case:
                r["label"] = case["label"]
            r["repeat"] = rep                 # repeat 0 is the run everything else scores
            out.append(r)
        return out

    per_case = parallel.run_cases(
        cases, all_repeats, workers=getattr(args, "workers", 1),
        label=lambda c: c.get("case_id"), progress=False)

    for case, group in zip(cases, per_case):
        if isinstance(group, dict):           # a runner_error came back, not a list
            results.append(group)
            print(f"[{case['case_id']}] {group.get('error', 'runner error')}")
            continue
        results.extend(group)
        tiers = [r["result"].get("tier", "?") for r in group]
        repeat_tiers[case["case_id"]] = tiers
        shown = tiers[0] + ("" if len(set(tiers)) == 1 else f"  ⚠ 重复间不一致 {tiers}")
        label = case.get("label", "")[:26]
        truth_text = f" (truth sep3={case['truth'].get('sep3')})" if "truth" in case else ""
        print(f"[{case['case_id']}] {label:<28} -> {shown:<14}{truth_text}")

    if args.repeats > 1:
        unstable = [c for c, t in repeat_tiers.items() if len(set(t)) > 1]
        print(f"\n  稳定性：{args.repeats} 次重复中档位发生变化的病例 "
              f"{len(unstable)}/{len(cases)}   {unstable or '无'}")

    meta = {"model": args.model, "dry_run": args.dry_run,
            "sampling": {
                "temperature_requested": args.temperature,
                "temperature_accepted": CAPS["temperature"],
                "seed_requested": args.seed,
                "seed_accepted": CAPS["seed"],
                "effective": ("deterministic" if args.dry_run or CAPS["temperature"]
                              else "model default (temperature fixed at 1) — "
                                   "run-to-run variation expected"),
                "model_snapshots": sorted(MODEL_SNAPSHOTS),
                "system_fingerprints": sorted(FINGERPRINTS)},
            "graph_version": GRAPH["version"], "guideline": GRAPH["guideline"],
            "decision_time": args.decision_time,
            "mode": "recognition" if args.decision_time == "T0" else "reassessment",
            "risk_signal": {
                "mode": args.risk_signal,
                "enabled": args.risk_signal == "esrp",
                "file": args.risk_signal_file if args.risk_signal == "esrp" else None,
                "source_field": "score_ESRP" if args.risk_signal == "esrp" else None,
                "source_type": "deterministic_score" if args.risk_signal == "esrp" else None,
            },
            "n_statements_in_library": len(STMTS), "repeats": args.repeats,
            "repeat_tiers": repeat_tiers, "seconds": round(time.time() - t0, 1)}
    if not args.dry_run and not CAPS["temperature"]:
        print("\n  ⚠ 该模型不允许 temperature=0，本次结果不可逐字复现。"
              "\n    要报告稳定性，跑 --repeats 3 看档位有没有跳。")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}   ({meta['seconds']}s)")


if __name__ == "__main__":
    main()
