#!/usr/bin/env python3
"""
Agent 2 (diagnostic workup) — sequential accumulation graph.

Workflow:
    W1 blood cultures
    -> W2 lactate
    -> W3 source-directed workup
    -> W4 alternative-diagnosis workup
    -> WORKUP_COMPLETE

Engineering pattern follows Agent 1:
  1. control flow       : code + decision graph
  2. evidence supply    : node-scoped verbatim guideline injection
  3. clinical judgement : model selects one closed-set answer per node
  4. validation         : V1 / V2 / V3
  5. one repair retry   : then abstain
  6. final output       : accumulated workup decisions

Usage:
    python radar/agents/run_agent2.py --infile agent2_synthetic_cases.json --dry-run

Real model:
    python radar/agents/run_agent2.py \
        --infile agent2_cases.json \
        --model gpt-5-mini
"""

from __future__ import annotations
from radar import paths

import argparse
import json
import os
import pathlib
import re
import sys
import time

from radar.common import parallel
from radar.common import model_usage

from radar.common.integration_schema import strip_inference_forbidden


# ---------------------------------------------------------------------
# paths / static resources
# ---------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
GRAPH_PATH = paths.GRAPH / "agent2_decision_graph.json"
STATEMENTS_PATH = paths.PARSED / "ssc2026_statements.json"

GRAPH = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))

STMTS = {
    s["id"]: s
    for s in json.loads(
        STATEMENTS_PATH.read_text(encoding="utf-8")
    )
}


# ---------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------

SYSTEM = """
You are one node of a guideline-grounded diagnostic workup system
for suspected sepsis in the emergency department.

Your output is a workup recommendation. It is not evidence that a test was
ordered, collected, resulted, or available at any particular time.

You are shown:

(a) the patient information available at T0,
(b) the current investigation status available at T0, and
(c) the complete verbatim guideline statements available to THIS node.

You must answer ONLY the current node question.

The Agent 2 workflow evaluates four workup categories independently:
1. blood cultures
2. serum lactate
3. source-directed imaging or sampling
4. evaluation of important noninfectious alternative diagnoses

Rules:

- `answer` must be exactly one of the allowed values.
- Do not recommend workup outside the category being evaluated by this node.
- Use only information that was available at T0.
- Do not use laboratory, microbiology, imaging, or other investigation results
  that became available after T0.
- Do not infer that a test was not performed simply because no record is present.
- Use `already_addressed` only when the supplied T0 data document that the
  investigation was already ordered, collected, or otherwise addressed.
- Use `not_indicated` only when the available clinical information supports
  that additional workup in this category is not indicated.
- When the available information is insufficient to make the decision,
  return `indeterminate`.
- Do not invent a specific imaging modality or source-specific diagnostic test
  unless directly supported by an approved source supplied to this node.
- A guideline recommendation to order a test is not evidence about the result
  of that test.

Citation rules:

- Every cited statement id must appear in this node's supplied guideline scope.
- Every `quote` must be copied verbatim from the supplied guideline text.
- If a patient-specific reasoning point has no directly supporting supplied
  guideline statement, use statement_id=null and quote=null.
- Never invent a citation.

Return ONE JSON object and nothing else:

{
  "answer": "<one of the allowed values>",
  "supporting_evidence": [
    {
      "patient_finding": "...",
      "statement_id": "stmt_00X",
      "quote": "...",
      "implication": "..."
    }
  ],
  "opposing_evidence": [
    {
      "patient_finding": "...",
      "statement_id": null,
      "quote": null,
      "implication": "..."
    }
  ],
  "missing_information": ["..."],
  "confidence": "low" | "moderate" | "high"
}
""".strip()


# ---------------------------------------------------------------------
# material injection
# ---------------------------------------------------------------------

def materials(node: dict) -> str:
    """Materialize the frozen statement scope for one Agent 2 node."""
    blocks = []

    for sid in node["scope"]:
        if sid not in STMTS:
            raise KeyError(
                f"{sid} is in node scope but missing from statement library"
            )

        s = STMTS[sid]

        head = f"[{sid}] ({s.get('strength')}"

        if s.get("certainty"):
            head += f", {s['certainty']} certainty"

        head += ")"

        block = [
            head,
            s["statement"],
        ]

        for remark in s.get("remarks") or []:
            block.append(f"Remark: {remark}")

        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


def patient_input(case: dict) -> dict:
    """
    Return inference-safe T0 input.

    Evaluation-only fields should live outside case['input'].
    """
    return strip_inference_forbidden(case.get("input", {}))


def upstream_recognition_context(case: dict) -> dict:
    """Whitelist Agent 1 outputs; ESRP and other upstream internals stay excluded."""
    source = strip_inference_forbidden(case.get("agent1_context") or {})
    return {
        "tier": source.get("tier"),
        "shock_concern": source.get("shock_concern"),
        "decision_time": source.get("decision_time", "T0"),
        "provenance": source.get("provenance", "unavailable"),
    }


def user_prompt(node: dict, case: dict, materials_text: str) -> str:
    return (
        "GUIDELINE STATEMENTS AVAILABLE AT THIS NODE\n"
        + "=" * 60
        + "\n"
        + materials_text
        + "\n\n"
        + "PATIENT AT T0\n"
        + "=" * 60
        + "\n"
        + json.dumps(
            patient_input(case),
            ensure_ascii=False,
            indent=1
        )
        + "\n\n"
        + "UPSTREAM AGENT 1 RECOGNITION CONTEXT\n"
        + "=" * 60
        + "\n"
        + (
            "The upstream recognition tier is contextual information. Use it together "
            "with the current T0 patient state and node-specific guideline scope when "
            "deciding whether this workup item is recommended, already addressed, not "
            "indicated, or indeterminate. It does not override direct patient evidence "
            "or guideline scope.\n"
        )
        + json.dumps(upstream_recognition_context(case), ensure_ascii=False, indent=1)
        + "\n\n"
        + "WORKUP CATEGORY\n"
        + "=" * 60
        + "\n"
        + node["item"]
        + "\n\n"
        + "QUESTION\n"
        + "=" * 60
        + "\n"
        + node["question"]
        + "\n"
        + f"allowed answers: {node['allowed_answers']}"
    )


# ---------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------

def normalise(text: str) -> str:
    """
    Verbatim matching tolerant only to quote style and whitespace.
    Same philosophy as Agent 1 V3.
    """
    text = (text or "")
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )

    return re.sub(r"\s+", " ", text).strip().lower()


def validate(reply: dict, node: dict, materials_text: str):
    """
    V1: answer must be allowed.
    V2: citation id must be inside current node scope.
    V3: quote must appear verbatim in injected materials.
    """

    failures = []

    stats = {
        "cited": 0,
        "uncited": 0,
    }

    if not isinstance(reply, dict):
        return False, ["not_an_object"], stats

    # V1
    answer = reply.get("answer")

    if answer not in node["allowed_answers"]:
        failures.append(
            f"V1_answer_not_allowed:{answer!r}"
        )

    # Basic schema checks
    for field in (
        "supporting_evidence",
        "opposing_evidence",
        "missing_information",
    ):
        if field not in reply:
            failures.append(
                f"schema_missing_field:{field}"
            )

    confidence = reply.get("confidence")

    if confidence not in {
        "low",
        "moderate",
        "high",
    }:
        failures.append(
            f"schema_invalid_confidence:{confidence!r}"
        )

    scope = set(node["scope"])
    # haystack = normalise(materials_text)

    # V2 / V3
    for field in (
        "supporting_evidence",
        "opposing_evidence",
    ):

        evidence = reply.get(field) or []

        if not isinstance(evidence, list):
            failures.append(
                f"schema_not_list:{field}"
            )
            continue

        for ev in evidence:

            if not isinstance(ev, dict):
                failures.append(
                    f"schema_invalid_evidence:{field}"
                )
                continue

            sid = ev.get("statement_id")
            quote = ev.get("quote")

            # explicitly uncited evidence is permitted
            if sid is None and quote is None:
                stats["uncited"] += 1
                continue

            stats["cited"] += 1

            # citation/quote must appear together
            if sid is None or quote is None:
                failures.append(
                    f"V2V3_incomplete_citation:{field}:{sid}"
                )
                continue

            # V2
            if sid not in scope:
                failures.append(
                    f"V2_out_of_scope:{field}:{sid}"
                )

            # V3
            # V3
            if sid in STMTS:
                specific_text = STMTS[sid]["statement"]

                for remark in STMTS[sid].get("remarks") or []:
                    specific_text += "\n" + remark

                if normalise(quote) not in normalise(specific_text):
                    failures.append(
                        f"V3_quote_not_verbatim:{field}:{sid}"
                    )

    return not failures, failures, stats


# ---------------------------------------------------------------------
# model call
# ---------------------------------------------------------------------

CAPS = {
    "temperature": True,
    "seed": True,
}

FINGERPRINTS = set()
MODEL_SNAPSHOTS = set()


def call_model(
    client,
    model,
    system,
    user,
    temperature,
    seed,
):
    kwargs = {
        "model": model,
        "response_format": {
            "type": "json_object"
        },
        "messages": [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": user,
            },
        ],
    }

    if temperature is not None and CAPS["temperature"]:
        kwargs["temperature"] = temperature

    if seed is not None and CAPS["seed"]:
        kwargs["seed"] = seed

    kwargs.update(model_usage.request_options())

    while True:
        try:
            response = client.chat.completions.create(
                **kwargs
            )
            break

        except Exception as exc:

            msg = str(exc)

            dropped = model_usage.drop_rejected(kwargs, exc)

            if dropped is None:
                raise

            if dropped in CAPS:
                CAPS[dropped] = False

            print(
                f"[note] {model} does not accept "
                f"{dropped}; continuing without it."
            )

    model_usage.record(sys.modules[__name__], response)
    fingerprint = getattr(
        response,
        "system_fingerprint",
        None,
    )

    if fingerprint:
        FINGERPRINTS.add(fingerprint)

    returned_model = getattr(response, "model", None)
    if returned_model:
        MODEL_SNAPSHOTS.add(returned_model)

    tokens = (
        response.usage.total_tokens
        if response.usage
        else None
    )

    return (
        response.choices[0].message.content,
        tokens,
    )


def parse_json(text):
    try:
        return json.loads(text)

    except Exception:
        match = re.search(
            r"\{.*\}",
            text or "",
            re.S,
        )

        if match:
            try:
                return json.loads(
                    match.group(0)
                )
            except Exception:
                pass

    return None


# ---------------------------------------------------------------------
# dry-run stub
# ---------------------------------------------------------------------

def get_investigation_status(case: dict, item: str):
    """
    Optional structured field for future Agent 2 cases:

    input.investigation_status = {
        "blood_cultures": "...",
        "serum_lactate": "...",
        "source_imaging_or_sampling": "...",
        "alternative_diagnosis_testing": "..."
    }
    """

    status = (
        case.get("input", {})
        .get("investigation_status", {})
        .get(item)
    )

    return status


def stub(node: dict, case: dict):
    """
    Offline stand-in.

    It does NOT attempt clinical reasoning.
    It only exercises graph traversal, accumulation,
    validation, and output construction.
    """

    status = get_investigation_status(
        case,
        node["item"],
    )

    if status in {
        "ordered",
        "collected",
        "performed",
        "pending",
        "result_available_before_T0",
        "already_addressed",
    }:
        answer = "already_addressed"

    elif status in {
        "not_indicated",
    }:
        answer = "not_indicated"

    elif status == "not_ordered":
        answer = "recommend"
#
    elif status == "no_record_before_T0":
        answer = "indeterminate"
    else:
        answer = "indeterminate"

    # Guideline citation is useful when the dry-run answer is "recommend".
    # For status-based answers such as already_addressed/not_indicated,
    # the status itself comes from the patient input rather than the guideline.

    if answer == "recommend":
        sid = node["scope"][0]

        supporting_evidence = [
            {
                "patient_finding": (
                    f"dry-run investigation status: {status!r}"
                ),
                "statement_id": sid,
                "quote": STMTS[sid]["statement"],
                "implication": (
                    "The investigation has not yet been addressed, "
                    "and the supplied guideline supports evaluating "
                    "this workup category."
                ),
            }
        ]

    else:
        supporting_evidence = [
            {
                "patient_finding": (
                    f"dry-run investigation status: {status!r}"
                ),
                "statement_id": None,
                "quote": None,
                "implication": (
                    "This dry-run decision is based on the structured "
                    "T0 investigation status."
                ),
            }
        ]

    response = {
        "answer": answer,
        "supporting_evidence": supporting_evidence,
        "opposing_evidence": [],
        "missing_information": (
            [
                f"Whether {node['item']} was addressed before or at T0"
            ]
            if answer == "indeterminate"
            else []
        ),
        "confidence": "low",
    }

    return json.dumps(
        response,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------
# accumulation
# ---------------------------------------------------------------------

def empty_plan():
    return {
        "recommendations": [],
        "already_addressed": [],
        "not_indicated": [],
        "indeterminate": [],
    }


def accumulate(
    plan: dict,
    node_id: str,
    node: dict,
    reply: dict,
):
    """
    Turn one valid node answer into one final-plan entry.
    """

    answer = reply["answer"]
    mapping = node["decision_mapping"][answer]

    item = node["item"]

    cited_basis = []

    for field in (
        "supporting_evidence",
        "opposing_evidence",
    ):
        for ev in reply.get(field) or []:
            sid = ev.get("statement_id")

            if (
                sid is not None
                and sid not in cited_basis
            ):
                cited_basis.append(sid)

    if answer == "recommend":

        plan["recommendations"].append(
            {
                "item": item,
                "priority": mapping["priority"],
                "node": node_id,
                "basis": cited_basis,
                "reason": (
                    reply.get(
                        "supporting_evidence"
                    )
                    or [{}]
                )[0].get(
                    "implication",
                    "",
                ),
            }
        )

    elif answer == "already_addressed":

        plan["already_addressed"].append(
            {
                "item": item,
                "node": node_id,
                "status": "already_addressed",
            }
        )

    elif answer == "not_indicated":

        plan["not_indicated"].append(
            {
                "item": item,
                "node": node_id,
            }
        )

    elif answer == "indeterminate":

        plan["indeterminate"].append(
            {
                "item": item,
                "node": node_id,
                "missing_information":
                    reply.get(
                        "missing_information",
                        [],
                    ),
            }
        )


# ---------------------------------------------------------------------
# graph walk
# ---------------------------------------------------------------------

def run_case(
    case: dict,
    client,
    args,
):
    trace = []

    node_id = GRAPH["entry"]

    guard = 0

    plan = empty_plan()

    while (
        node_id in GRAPH["nodes"]
        and guard < 12
    ):
        guard += 1

        node = dict(
            GRAPH["nodes"][node_id],
            id=node_id,
        )

        materials_text = materials(node)

        prompt = user_prompt(
            node,
            case,
            materials_text,
        )

        reply = None

        failures = [
            "no_attempt"
        ]

        attempts = []

        tokens = 0

        stats = {
            "cited": 0,
            "uncited": 0,
        }

        max_attempts = GRAPH.get(
            "abstention",
            {},
        ).get(
            "max_model_attempts_per_node",
            2,
        )

        for attempt in range(
            max_attempts
        ):

            if attempt == 0:
                system_prompt = SYSTEM

            else:
                system_prompt = (
                    SYSTEM
                    + "\n\n"
                    + "Your previous reply was rejected: "
                    + "; ".join(failures)
                    + "\n"
                    + (
                        "Correct only those validation "
                        "failures. Copy quotes "
                        "character-for-character from "
                        "the supplied statements and "
                        "use only statement ids that "
                        "appear in this node's scope."
                    )
                )

            if args.dry_run:
                raw = stub(
                    node,
                    case,
                )
                tok = 0

            else:
                raw, tok = call_model(
                    client,
                    args.model,
                    system_prompt,
                    prompt,
                    args.temperature,
                    args.seed,
                )

            tokens += tok or 0

            candidate = parse_json(raw)

            if candidate is None:

                ok = False

                failures = [
                    "unparseable_json"
                ]

                stats = {}

            else:

                (
                    ok,
                    failures,
                    stats,
                ) = validate(
                    candidate,
                    node,
                    materials_text,
                )

            attempts.append(
                {
                    "attempt": attempt,
                    "ok": ok,
                    "failures": failures,
                    "raw": raw,
                }
            )

            if ok:
                reply = candidate
                break

        step = {
            "node": node_id,
            "title": node["title"],
            "item": node["item"],
            "question": node["question"],
            "scope": node["scope"],
            "attempts": attempts,
            "tokens": tokens,
            "upstream_recognition": upstream_recognition_context(case),
        }

        # second failure => whole Agent 2 abstains
        if reply is None:

            step["outcome"] = "abstained"

            trace.append(step)

            return {
                "case_id": case.get(
                    "case_id"
                ),
                "terminal": "ABSTAIN",
                "status": "abstain",
                "result": {
                    "agent": GRAPH["agent"],
                    "status": "abstain",
                    "workup_plan": None,
                },
                "trace": trace,
            }

        step.update(
            {
                "answer": reply["answer"],
                "reply": reply,
                "ungrounded": stats.get(
                    "uncited",
                    0,
                ),
            }
        )

        trace.append(step)

        # accumulate before moving on
        accumulate(
            plan,
            node_id,
            node,
            reply,
        )

        next_node = node[
            "transitions"
        ][
            reply["answer"]
        ]

        node_id = next_node

    # ------------------------------------------------------------
    # terminal
    # ------------------------------------------------------------

    if node_id not in GRAPH["terminals"]:

        return {
            "case_id": case.get(
                "case_id"
            ),
            "terminal": "ROUTING_ERROR",
            "status": "error",
            "result": {
                "agent": GRAPH["agent"],
                "status": "error",
                "detail": (
                    f"Unknown terminal: "
                    f"{node_id!r}"
                ),
            },
            "trace": trace,
        }

    terminal = GRAPH["terminals"][
        node_id
    ]

    return {
        "case_id": case.get(
            "case_id"
        ),
        "terminal": node_id,
        "status": terminal["status"],
        "result": {
            "agent": GRAPH["agent"],
            "status": terminal[
                "status"
            ],
            "decision_time": "T0",
            "output_kind": "workup_recommendation",
            "workup_plan": plan,
            "upstream_recognition": upstream_recognition_context(case),
        },
        "trace": trace,
    }


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------

def load_cases(path: pathlib.Path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, list):
        raise ValueError(
            "Agent 2 infile must contain "
            "a JSON list of cases."
        )

    return data


def load_env():
    env_path = paths.ROOT / ".env"

    if not env_path.exists():
        return

    for line in env_path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = line.split(
            "=",
            1,
        )

        os.environ.setdefault(
            key.strip(),
            value.strip(),
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--infile",
        required=True,
        help=(
            "Agent 2 T0 input JSON file."
        ),
    )

    parser.add_argument(
        "--out",
        default=str(
            paths.DATA
            / "agent2_run.json"
        ),
    )

    parser.add_argument(
        "--model",
        default="gpt-5-mini",
    )

    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Set for vLLM / local "
            "OpenAI-compatible servers."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--dry-run",

        action="store_true",
    )

    parallel.add_workers_argument(parser)

    args = parser.parse_args()

    infile = pathlib.Path(
        args.infile
    )

    if not infile.exists():
        sys.exit(
            f"Input file not found: "
            f"{infile}"
        )

    cases = load_cases(
        infile
    )

    client = None

    if not args.dry_run:

        load_env()

        if not os.environ.get(
            "OPENAI_API_KEY"
        ):
            sys.exit(
                "OPENAI_API_KEY not set "
                "(put it in .env in the repository root)"
            )

        from openai import OpenAI

        client = (
            OpenAI(
                base_url=args.base_url
            )
            if args.base_url
            else OpenAI()
        )

    start = time.time()

    results = parallel.run_cases(
        cases,
        lambda case: run_case(case, client, args),
        workers=getattr(args, "workers", 1),
        progress=False,
    )

    for case, result in zip(cases, results):

        plan = (
            result.get(
                "result",
                {},
            ).get(
                "workup_plan"
            )
        )

        if plan is None:

            print(
                f"[{case.get('case_id', '?')}] -> ABSTAIN"
            )

        else:

            def items(group):
                return [
                    x["item"]
                    for x in plan[group]
                ]

            print(
                f"[{case.get('case_id', '?')}]"
                f"\n  recommend:       {items('recommendations')}"
                f"\n  addressed:       {items('already_addressed')}"
                f"\n  not_indicated:   {items('not_indicated')}"
                f"\n  indeterminate:   {items('indeterminate')}"
            )

    meta = {
        "agent": GRAPH["agent"],
        "graph_version": GRAPH[
            "version"
        ],
        "guideline": GRAPH[
            "guideline"
        ],
        "model": args.model,
        "dry_run": args.dry_run,
        "infile": str(infile),
        "n_cases": len(cases),
        "sampling": {
            "temperature_requested":
                args.temperature,
            "temperature_accepted":
                CAPS["temperature"],
            "seed_requested":
                args.seed,
            "seed_accepted":
                CAPS["seed"],
            "model_snapshots":
                sorted(
                    MODEL_SNAPSHOTS
                ),
            "system_fingerprints":
                sorted(
                    FINGERPRINTS
                ),
        },
        "seconds": round(
            time.time() - start,
            1,
        ),
    }

    output_path = pathlib.Path(
        args.out
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "results": results,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    print(
        f"\n-> {output_path} "
        f"({meta['seconds']}s)"
    )


if __name__ == "__main__":
    main()
