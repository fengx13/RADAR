#!/usr/bin/env python3
"""
Agent 3 - Initial treatment at a longitudinal reassessment checkpoint.

Workflow:
    TX1 antimicrobial timing
    -> TX2 fluids
    -> TX3 vasopressors
    -> TX4 MAP target
    -> TREATMENT_COMPLETE

Engineering pattern:
  1. code controls traversal
  2. node-scoped guideline injection
  3. closed-set model answers
  4. V1 / V2 / V3 validation
  5. one repair retry, then abstain
  6. accumulate node decisions into one treatment plan

Dry run:
    python radar/agents/run_agent3.py \
        --infile agent3_synthetic_cases.json \
        --dry-run
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
from radar.agents.agent3_citations import CITATION_CONTRACT, bind_evidence_quotes, citation_reference_contract


# ---------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
ROOT = paths.ROOT
GRAPH_PATH = paths.GRAPH / "agent3_decision_graph.json"
STATEMENTS_PATH = paths.PARSED / "ssc2026_statements.json"

GRAPH = json.loads(
    GRAPH_PATH.read_text(encoding="utf-8")
)

STMTS = {
    s["id"]: s
    for s in json.loads(
        STATEMENTS_PATH.read_text(encoding="utf-8")
    )
}


# ---------------------------------------------------------------------
# system prompt
# ---------------------------------------------------------------------

SYSTEM = """
You are one node of a guideline-grounded initial treatment system
for sepsis in the emergency department.

Your output is a treatment recommendation. It is not evidence that treatment
was administered, remains active, was completed, or produced a response.

You are shown:

(a) the patient state available by the current checkpoint cutoff,
(b) relevant treatment already provided by that cutoff, and
(c) the complete verbatim guideline statements available to THIS node.

You must answer ONLY the current node question.

Rules:

- `answer` must be exactly one of the allowed values.
- Use only patient information available by the current checkpoint cutoff.
- Do not use future information or outcomes.
- Do not invent treatments outside the current node decision.
- Do not infer septic shock or hypoperfusion solely from one abnormal value.
- Consider both evidence supporting treatment and evidence arguing against it.
- Avoid unnecessary treatment when the available evidence does not support it.
- If a treatment has already been given, use that information only as part of
  the current checkpoint patient state; do not automatically recommend repeating it.
- Do not use a guideline recommendation as evidence that the patient's clinical
  condition is actually present.

Citation rules:

- Every cited statement id must be inside the current node scope.
- Select its exact quote_ref from the supplied lookup options and return quote=null.
- Code inserts the selected verbatim text before the unchanged V1/V2/V3 checks.
- Citation lookup establishes text provenance, not clinical applicability.
- Legacy free-text quotes must still be verbatim; conflicting text is rejected,
  never silently corrected by selecting a quote_ref.
- If reasoning has no directly supporting statement in the supplied scope,
  use statement_id=null, quote_ref=null and quote=null.
- Never invent a citation.

Return ONE JSON object and nothing else:

{
  "answer": "<one of the allowed values>",
  "supporting_evidence": [
    {
      "patient_finding": "...",
      "statement_id": "stmt_00X",
      "quote_ref": "statement",
      "quote": null,
      "implication": "..."
    }
  ],
  "opposing_evidence": [
    {
      "patient_finding": "...",
      "statement_id": null,
      "quote_ref": null,
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

def materials(node):
    blocks = []

    for sid in node["scope"]:

        if sid not in STMTS:
            raise KeyError(
                f"{sid} missing from statement library"
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
            block.append(
                f"Remark: {remark}"
            )

        blocks.append(
            "\n".join(block)
        )

    return "\n\n".join(blocks)


def patient_input(case):
    return strip_inference_forbidden(case.get("input", {}), exclude_temporal_audit=True)


def checkpoint_context(case):
    safe = patient_input(case)
    state = safe.get("checkpoint_state") or safe.get("T1_state") or {}
    checkpoint = safe.get("checkpoint") or {}
    return {
        "checkpoint_id": state.get("checkpoint_id", checkpoint.get("checkpoint_id", "T1")),
        "information_cutoff": state.get(
            "information_cutoff",
            checkpoint.get("cutoff_time", state.get("as_of_time", "T1")),
        ),
    }


def citation_copy_contract(node):
    return citation_reference_contract(node, STMTS)


def user_prompt(node, case, materials_text):

    return (
        "GUIDELINE STATEMENTS AVAILABLE AT THIS NODE\n"
        + "=" * 60
        + "\n"
        + citation_copy_contract(node)
        + "\n\n"

        + "PATIENT AT CURRENT REASSESSMENT CHECKPOINT\n"
        + "=" * 60
        + "\n"
        + json.dumps(
            patient_input(case),
            ensure_ascii=False,
            indent=1,
        )
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

def normalise(text):

    text = (text or "")

    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip().lower()


def validate(reply, node, materials_text):

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

    # schema
    for field in (
        "supporting_evidence",
        "opposing_evidence",
        "missing_information",
    ):

        if field not in reply:
            failures.append(
                f"schema_missing_field:{field}"
            )

    if reply.get("confidence") not in {
        "low",
        "moderate",
        "high",
    }:

        failures.append(
            f"schema_invalid_confidence:"
            f"{reply.get('confidence')!r}"
        )

    scope = set(
        node["scope"]
    )

    # haystack = normalise(
    #     materials_text
    # )

    # V2 + V3
    for field in (
        "supporting_evidence",
        "opposing_evidence",
    ):

        evidence = (
            reply.get(field)
            or []
        )

        if not isinstance(
            evidence,
            list,
        ):
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

            sid = ev.get(
                "statement_id"
            )

            quote = ev.get(
                "quote"
            )

            # uncited reasoning is allowed
            if sid is None and quote is None:
                stats["uncited"] += 1
                continue

            stats["cited"] += 1

            if sid is None or quote is None:
                failures.append(
                    f"V2V3_incomplete_citation:"
                    f"{field}:{sid}"
                )
                continue

            # V2
            if sid not in scope:
                failures.append(
                    f"V2_out_of_scope:"
                    f"{field}:{sid}"
                )

            # V3
            # V3
            if sid in STMTS:

                specific_text = STMTS[sid]["statement"]

                for remark in STMTS[sid].get("remarks") or []:
                    specific_text += "\n" + remark

                if normalise(quote) not in normalise(specific_text):

                    failures.append(
                        f"V3_quote_not_verbatim:"
                        f"{field}:{sid}"
                    )

    return (
        not failures,
        failures,
        stats,
    )


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

    if (
        temperature is not None
        and CAPS["temperature"]
    ):
        kwargs[
            "temperature"
        ] = temperature

    if (
        seed is not None
        and CAPS["seed"]
    ):
        kwargs["seed"] = seed

    kwargs.update(model_usage.request_options())

    while True:

        try:

            response = (
                client
                .chat
                .completions
                .create(**kwargs)
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
                f"[note] {model} does not "
                f"accept {dropped}; "
                f"continuing without it."
            )

    model_usage.record(sys.modules[__name__], response)
    fingerprint = getattr(
        response,
        "system_fingerprint",
        None,
    )

    if fingerprint:
        FINGERPRINTS.add(
            fingerprint
        )

    returned_model = getattr(response, "model", None)
    if returned_model:
        MODEL_SNAPSHOTS.add(returned_model)

    tokens = (
        response.usage.total_tokens
        if response.usage
        else None
    )

    return (
        response
        .choices[0]
        .message
        .content,
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
DRY_RUN_BASIS = {
    "TX1": {
        "within_1h": "stmt_016",
        "within_3h": "stmt_018",
        "defer_and_monitor": "stmt_020",
        "not_indicated": None,
    },

    "TX2": {
        "give": "stmt_010",
        "withhold": None,
    },

    "TX3": {
        "start": ["stmt_011", "stmt_053"],
        "hold": [],
    },

    "TX4": {
        "target_65": "stmt_013",
        "target_60_65": "stmt_014",
        "not_applicable": None,
    },
}
def stub(node, case):
    """
    Structural dry-run only.

    Uses explicit synthetic expected states to exercise:
      - graph traversal
      - validation
      - accumulation

    Does NOT represent clinical model reasoning.
    """

    expected = (
        case.get("dry_run_answers", {})
        .get(node["id"])
    )

    if expected not in node["allowed_answers"]:
        expected = node[
            "allowed_answers"
        ][0]

    # sid = node["scope"][0]
    basis = DRY_RUN_BASIS[node["id"]].get(expected)

    if basis is None:
        basis = []

    elif isinstance(basis, str):
        basis = [basis]

    if basis:

        supporting_evidence = []

        for sid in basis:

            supporting_evidence.append(
                {
                    "patient_finding": "synthetic dry-run state",
                    "statement_id": sid,
                    "quote": STMTS[sid]["statement"],
                    "implication": "dry-run guideline mapping test"
                }
            )

    else:

        supporting_evidence = [
            {
                "patient_finding": "synthetic dry-run state",
                "statement_id": None,
                "quote": None,
                "implication": (
                    "Dry-run negative or not-applicable decision; "
                    "no guideline statement is claimed as direct support."
                )
            }
        ]
    # response = {
    #     "answer": expected,

    #     "supporting_evidence": [
    #         {
    #             "patient_finding":
    #                 "synthetic dry-run state",

    #             "statement_id":
    #                 sid,

    #             "quote":
    #                 STMTS[sid]["statement"],

    #             "implication":
    #                 "dry-run placeholder used "
    #                 "to test Agent 3 pipeline"
    #         }
    #     ],

    #     "opposing_evidence": [],

    #     "missing_information": [],

    #     "confidence": "low",
    # }
    response = {
        "answer": expected,

        "supporting_evidence": supporting_evidence,

        "opposing_evidence": [],

        "missing_information": [],

        "confidence": "low",
    }

    return json.dumps(
        response,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------
# treatment plan accumulation
# ---------------------------------------------------------------------

def empty_plan():

    return {
        "antimicrobial_timing": None,
        "fluids": None,
        "vasopressors": None,
        "map_target": None,
    }


def cited_basis(reply):

    ids = []

    for field in (
        "supporting_evidence",
        "opposing_evidence",
    ):

        for ev in (
            reply.get(field)
            or []
        ):

            sid = ev.get(
                "statement_id"
            )

            if (
                sid is not None
                and sid not in ids
            ):
                ids.append(
                    sid
                )

    return ids


def accumulate(
    plan,
    node_id,
    node,
    reply,
):

    answer = reply["answer"]

    mapping = (
        node["decision_mapping"][
            answer
        ]
    )

    basis = cited_basis(
        reply
    )

    if node_id == "TX1":

        plan[
            "antimicrobial_timing"
        ] = {
            "decision": answer,
            "node": node_id,
            "basis": basis,
            **mapping,
        }

    elif node_id == "TX2":

        plan["fluids"] = {
            "decision": answer,
            "node": node_id,
            "basis": basis,
            **mapping,
        }

    elif node_id == "TX3":

        plan["vasopressors"] = {
            "decision": answer,
            "node": node_id,
            "basis": basis,
            **mapping,
        }

    elif node_id == "TX4":

        plan["map_target"] = {
            "decision": answer,
            "node": node_id,
            "basis": basis,
            **mapping,
        }


# ---------------------------------------------------------------------
# graph walk
# ---------------------------------------------------------------------

def run_case(
    case,
    client,
    args,
):

    trace = []

    node_id = GRAPH[
        "entry"
    ]

    guard = 0

    plan = empty_plan()

    while (
        node_id in GRAPH["nodes"]
        and guard < 12
    ):

        guard += 1

        node = dict(
            GRAPH["nodes"][
                node_id
            ],
            id=node_id,
        )

        materials_text = (
            materials(node)
        )

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

        max_attempts = (
            GRAPH
            .get("abstention", {})
            .get(
                "max_model_attempts_per_node",
                2,
            )
        )

        for attempt in range(
            max_attempts
        ):

            if attempt == 0:

                system_prompt = (
                    SYSTEM
                )

            else:

                system_prompt = (
                    SYSTEM
                    + "\n\n"
                    + "Your previous reply "
                    + "was rejected: "
                    + "; ".join(
                        failures
                    )
                    + "\n"
                    + (
                        "Correct only the "
                        "validation failures. "
                        "Use only allowed "
                        "answers and select a scoped "
                        "statement_id/quote_ref with quote=null. "
                        "Do not rewrite guideline text."
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

            tokens += (
                tok or 0
            )

            candidate = (
                parse_json(raw)
            )
            candidate, quote_bindings, binding_failures = bind_evidence_quotes(candidate, node, STMTS)

            if candidate is None:

                ok = False

                failures = binding_failures or [
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
                    "attempt":
                        attempt,

                    "ok":
                        ok,

                    "failures":
                        failures,

                    "raw":
                        raw,
                    "citation_contract": CITATION_CONTRACT,
                    "quote_bindings": quote_bindings,
                }
            )

            if ok:
                reply = candidate
                break

        step = {
            "node": node_id,
            "title":
                node["title"],
            "question":
                node["question"],
            "scope":
                node["scope"],
            "attempts":
                attempts,
            "tokens":
                tokens,
        }

        if reply is None:

            step[
                "outcome"
            ] = "abstained"

            trace.append(
                step
            )

            return {
                "case_id":
                    case.get(
                        "case_id"
                    ),

                "terminal":
                    "ABSTAIN",

                "status":
                    "abstain",

                "result": {
                    "agent":
                        GRAPH["agent"],

                    "status":
                        "abstain",

                    "treatment_plan":
                        None,
                },

                "trace":
                    trace,
            }

        step.update(
            {
                "answer":
                    reply["answer"],

                "reply":
                    reply,

                "ungrounded":
                    stats.get(
                        "uncited",
                        0,
                    ),
            }
        )

        trace.append(
            step
        )

        # save node decision
        accumulate(
            plan,
            node_id,
            node,
            reply,
        )

        node_id = (
            node["transitions"][
                reply["answer"]
            ]
        )

    # -----------------------------------------------------------------
    # terminal
    # -----------------------------------------------------------------

    if (
        node_id
        not in GRAPH["terminals"]
    ):

        return {
            "case_id":
                case.get(
                    "case_id"
                ),

            "terminal":
                "ROUTING_ERROR",

            "status":
                "error",

            "result": {
                "agent":
                    GRAPH["agent"],

                "status":
                    "error",

                "detail":
                    f"Unknown terminal "
                    f"{node_id!r}",
            },

            "trace":
                trace,
        }

    terminal = (
        GRAPH["terminals"][
            node_id
        ]
    )

    return {
        "case_id":
            case.get(
                "case_id"
            ),

        "terminal":
            node_id,

        "status":
            terminal["status"],

        "result": {
            "agent":
                GRAPH["agent"],

            "status":
                terminal["status"],

            "decision_time":
                checkpoint_context(case)["checkpoint_id"],

            "information_cutoff":
                checkpoint_context(case)["information_cutoff"],

            "output_kind":
                "treatment_recommendation",

            "treatment_plan":
                plan,
        },

        "trace":
            trace,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def load_cases(path):

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Agent 3 input file must "
            "contain a JSON list."
        )

    return data


def load_env():

    env_path = (
        paths.ROOT / ".env"
    )

    if not env_path.exists():
        return

    for line in (
        env_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ):

        line = (
            line.strip()
        )

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, value = (
            line.split(
                "=",
                1,
            )
        )

        os.environ.setdefault(
            key.strip(),
            value.strip(),
        )


def main():

    parser = (
        argparse
        .ArgumentParser()
    )

    parser.add_argument(
        "--infile",
        required=True,
    )

    parser.add_argument(
        "--out",
        default=str(
            paths.DATA
            / "agent3_run.json"
        ),
    )

    parser.add_argument(
        "--model",
        default="gpt-5-mini",
    )

    parser.add_argument(
        "--base-url",
        default=None,
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

    args = (
        parser.parse_args()
    )

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
                "OPENAI_API_KEY not set"
            )

        from openai import (
            OpenAI
        )

        client = (
            OpenAI(
                base_url=
                    args.base_url
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
            result
            .get(
                "result",
                {},
            )
            .get(
                "treatment_plan"
            )
        )

        if plan is None:

            print(
                f"[{case.get('case_id', '?')}]"
                f" -> ABSTAIN"
            )

        else:

            print(
                f"[{case.get('case_id', '?')}]"
            )

            print(
                "  antibiotics:  "
                f"{plan['antimicrobial_timing']['decision']}"
            )

            print(
                "  fluids:       "
                f"{plan['fluids']['decision']}"
            )

            print(
                "  vasopressors: "
                f"{plan['vasopressors']['decision']}"
            )

            print(
                "  MAP target:   "
                f"{plan['map_target']['decision']}"
            )

    meta = {
        "agent":
            GRAPH["agent"],

        "graph_version":
            GRAPH["version"],

        "guideline":
            GRAPH["guideline"],

        "model":
            args.model,

        "dry_run":
            args.dry_run,

        "infile":
            str(infile),

        "n_cases":
            len(cases),

        "sampling": {
            "temperature_requested":
                args.temperature,

            "temperature_accepted":
                CAPS[
                    "temperature"
                ],

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

        "seconds":
            round(
                time.time()
                - start,
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
                "meta":
                    meta,

                "results":
                    results,
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
