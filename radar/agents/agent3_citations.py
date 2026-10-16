"""Bind explicitly selected A3 citation fragments, without editing model reasoning."""
from __future__ import annotations

import copy
import json

CITATION_CONTRACT = "agent3_scoped_fragment_lookup_v1"


def citation_fragments(node, statements):
    return [
        {"statement_id": sid, "quote_ref": ref, "quote": quote}
        for sid in node["scope"]
        for ref, quote in [("statement", statements[sid]["statement"]),
                           *[(f"remark_{index}", text)
                             for index, text in enumerate(statements[sid].get("remarks") or [])]]
    ]


def citation_reference_contract(node, statements):
    metadata = {sid: {"strength": statements[sid].get("strength"),
                      "certainty": statements[sid].get("certainty")} for sid in node["scope"]}
    return (
        "\n\nCITATION LOOKUP OPTIONS (current node scope only)\n"
        + json.dumps(citation_fragments(node, statements), ensure_ascii=False, indent=1)
        + "\n\nSTATEMENT METADATA (not quote text)\n"
        + json.dumps(metadata, ensure_ascii=False, indent=1)
        + "\nSelect a statement_id and its exact quote_ref from one option. Return quote=null; "
        "the runner will insert that option's complete verbatim text before V1/V2/V3 validation. "
        "Use statement for the recommendation itself, or the displayed remark_N for that remark. "
        "Do not generate, combine or edit quote text. Explain applicability in implication. "
        "These fragments do not establish a patient condition. If no scoped statement supports "
        "the reasoning, use statement_id=null, quote_ref=null and quote=null. "
        "Legacy free-text citations remain supported, but non-verbatim quotes still fail V3; "
        "a quote_ref never repairs conflicting quote text.\n"
    )


def bind_evidence_quotes(reply, node, statements):
    """Resolve explicit references only. Legacy free-text quotes are never repaired."""
    if not isinstance(reply, dict):
        return reply, [], []
    candidate, bindings, failures = copy.deepcopy(reply), [], []
    options = {(item["statement_id"], item["quote_ref"]): item["quote"]
               for item in citation_fragments(node, statements)}
    for field in ("supporting_evidence", "opposing_evidence"):
        evidence = candidate.get(field)
        if not isinstance(evidence, list):
            continue
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                continue
            sid, ref, quote = item.get("statement_id"), item.get("quote_ref"), item.get("quote")
            if sid is not None and not isinstance(sid, str):
                failures.append(f"V2_invalid_statement_id:{field}:{index}")
                continue
            if quote is not None and not isinstance(quote, str):
                failures.append(f"V3_invalid_quote_type:{field}:{index}")
                continue
            if ref is None:
                continue
            if not isinstance(ref, str):
                failures.append(f"V3_invalid_quote_ref:{field}:{index}")
                continue
            if sid not in node["scope"]:
                failures.append(f"V2_out_of_scope:{field}:{sid}")
                continue
            canonical = options.get((sid, ref))
            if not isinstance(canonical, str) or not canonical.strip():
                failures.append(f"V3_unknown_quote_ref:{field}:{sid}:{ref}")
                continue
            if quote is not None and quote != canonical:
                failures.append(f"V3_quote_ref_text_mismatch:{field}:{sid}")
                continue
            item["quote"] = canonical
            item["quote_source"] = "deterministic_scoped_fragment_lookup"
            bindings.append({"evidence_field": field, "evidence_index": index,
                             "statement_id": sid, "quote_ref": ref, "model_quote": quote,
                             "quote_source": "deterministic_scoped_fragment_lookup"})
    # A partially bound reply must never be routed when any reference is invalid.
    return (None if failures else candidate), bindings, failures
