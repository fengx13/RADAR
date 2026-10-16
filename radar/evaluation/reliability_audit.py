#!/usr/bin/env python3
"""Offline reliability export and failure-mode diagnostics.

The exporter reads already-produced run artifacts. It never calls a model and
does not copy truth/outcome payloads into its output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
from collections import Counter, defaultdict


MIMIC_PATTERNS = {
    "heart_failure": ("heart failure", "chf"),
    "pulmonary_embolism": ("pulmonary embol", " pe "),
    "acute_coronary_syndrome": ("acute coronary", "cardiac ischem", "myocardial infar", " mi "),
    "acute_kidney_injury": ("acute kidney", " aki ", "renal"),
    "pulmonary_other": ("copd", "pulmonary disease", "pneumonia", "ards"),
}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_wording(value) -> str:
    """Normalize free text without treating it as a clinical decision."""
    if isinstance(value, dict):
        parts = [normalized_wording(value[key]) for key in sorted(value)]
    elif isinstance(value, list):
        parts = [normalized_wording(item) for item in value]
    elif isinstance(value, str):
        parts = [value]
    else:
        parts = []
    return " ".join(part for part in parts if part).lower().strip()


def wording_hash(reply: dict | None) -> str | None:
    text = re.sub(r"\s+", " ", normalized_wording(reply or {}))
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None


def evidence_hash(reply: dict | None) -> str:
    reply = reply or {}
    evidence = {
        key: reply.get(key) or []
        for key in ("supporting_evidence", "opposing_evidence")
    }
    raw = json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def mimic_domain(item: dict) -> str | None:
    text = " " + " ".join(
        str(item.get(key) or "").lower()
        for key in ("patient_finding", "alternative_explanation", "implication")
    ) + " "
    for domain, patterns in MIMIC_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return domain
    return None


def classify_evidence_item(
    item: dict,
    *,
    node: str | None,
    scope_ids: set[str] | None = None,
    knowledge_base_ids: set[str] | None = None,
) -> dict:
    """Conservative structural classification; semantic matching stays manual."""
    scope_ids = scope_ids or set()
    knowledge_base_ids = knowledge_base_ids or set()
    statement_id = item.get("statement_id")
    domain = mimic_domain(item)
    if statement_id:
        if node in {"R3a", "R3b"} and domain:
            # The current R3 scope cites the duty to assess alternatives, not
            # disease-specific mimic guidance (a documented graph v0.6 gap).
            category = "general_duty_citation_not_mimic_specific_support"
        elif statement_id in scope_ids:
            category = "cited_in_injected_scope"
        elif statement_id in knowledge_base_ids:
            category = "citation_scope_miss"
        else:
            category = "citation_id_not_in_knowledge_base"
        return {"category": category, "mimic_domain": domain}

    has_interpretation = bool(item.get("alternative_explanation") or item.get("implication"))
    if node in {"R3a", "R3b"} and has_interpretation:
        category = "known_mimic_kb_gap" if domain else "alternative_diagnosis_kb_gap_or_uncertain"
    elif has_interpretation:
        category = "mixed_patient_fact_and_uncited_medical_interpretation"
    elif item.get("patient_finding"):
        category = "patient_fact_no_guideline_citation_required"
    else:
        category = "unsupported_or_unclassifiable"
    return {"category": category, "mimic_domain": domain}


def _fingerprint(step: dict) -> str | None:
    attempts = step.get("attempts") or []
    successful = next((attempt for attempt in reversed(attempts) if attempt.get("ok")), None)
    selected = successful or (attempts[-1] if attempts else {})
    return selected.get("system_fingerprint")


def _field_reference_valid(agent: str, step: dict) -> bool | None:
    if not agent.startswith("agent4"):
        return None
    if "field_reference_valid" in step:
        return bool(step["field_reference_valid"])
    failures = [
        failure
        for attempt in step.get("attempts") or []
        for failure in attempt.get("failures") or []
    ]
    return not any("field_missing" in failure or "not_blockable" in failure for failure in failures)


def _guideline_claim_supported(step: dict, classifications: list[dict]) -> bool | None:
    evidence = [
        item
        for key in ("supporting_evidence", "opposing_evidence")
        for item in (step.get("reply") or {}).get(key) or []
    ]
    if not evidence:
        return None
    categories = {item["category"] for item in classifications}
    if categories & {"citation_scope_miss", "citation_id_not_in_knowledge_base", "unsupported_or_unclassifiable"}:
        return False
    if categories & {
        "known_mimic_kb_gap",
        "alternative_diagnosis_kb_gap_or_uncertain",
        "mixed_patient_fact_and_uncited_medical_interpretation",
        "general_duty_citation_not_mimic_specific_support",
    }:
        return None
    return True


def reliability_rows(run: dict, artifact_name: str) -> tuple[list[dict], list[dict]]:
    meta = run.get("meta") or {}
    model = meta.get("model")
    sampling = meta.get("sampling") or {}
    temperature = sampling.get("temperature_requested", meta.get("temperature"))
    seed = sampling.get("seed_requested", meta.get("seed"))
    agent = meta.get("agent") or "agent1"
    rows: list[dict] = []
    evidence_rows: list[dict] = []
    for record_index, record in enumerate(run.get("results") or []):
        repeat = record.get("repeat", 0)
        run_id = f"{artifact_name}:repeat_{repeat}"
        tier = (record.get("result") or {}).get("tier")
        abstain = tier == "abstained" or (record.get("result") or {}).get("status") == "abstain"
        for step_index, step in enumerate(record.get("trace") or []):
            reply = step.get("reply") or {}
            scope_ids = set(step.get("scope") or [])
            classifications = []
            for evidence_type in ("supporting_evidence", "opposing_evidence"):
                for item_index, item in enumerate(reply.get(evidence_type) or []):
                    classification = classify_evidence_item(item, node=step.get("node"), scope_ids=scope_ids)
                    classifications.append(classification)
                    evidence_rows.append({
                        "artifact": artifact_name,
                        "case_id": record.get("case_id"),
                        "run_id": run_id,
                        "node": step.get("node"),
                        "evidence_type": evidence_type,
                        "item_index": item_index,
                        "statement_id_null": item.get("statement_id") is None,
                        **classification,
                    })
            attempts = step.get("attempts") or []
            rows.append({
                "case_id": record.get("case_id"),
                "agent": agent,
                "node": step.get("node"),
                "run_id": run_id,
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "system_fingerprint": _fingerprint(step),
                "decision": step.get("answer"),
                "tier": tier,
                "leading_alternative": step.get("leading_alternative", reply.get("leading_alternative")),
                "validation_first_pass": attempts[0].get("ok") if attempts else None,
                "retry": len(attempts) > 1,
                "abstain": abstain,
                "field_reference_valid": _field_reference_valid(agent, step),
                "guideline_claim_supported": _guideline_claim_supported(step, classifications),
                "wording_hash": wording_hash(reply),
                "evidence_hash": evidence_hash(reply),
                "artifact": artifact_name,
                "record_index": record_index,
                "step_index": step_index,
            })
    return rows, evidence_rows


def first_node_divergence(records: list[dict]) -> str | None:
    sequences = [[(step.get("node"), step.get("answer")) for step in record.get("trace") or []] for record in records]
    longest = max((len(sequence) for sequence in sequences), default=0)
    for index in range(longest):
        values = {sequence[index] if index < len(sequence) else None for sequence in sequences}
        if len(values) > 1:
            return next((value[0] for value in values if value is not None), None)
    return None


def stability_by_case(run: dict) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in run.get("results") or []:
        grouped[record.get("case_id")].append(record)
    cases = {}
    for case_id, records in sorted(grouped.items()):
        records.sort(key=lambda item: item.get("repeat", 0))
        tiers = [(record.get("result") or {}).get("tier") for record in records]
        node_signatures = [tuple((step.get("node"), step.get("answer")) for step in record.get("trace") or []) for record in records]
        alternatives = [tuple(
            (step.get("node"), step.get("leading_alternative", (step.get("reply") or {}).get("leading_alternative")))
            for step in record.get("trace") or [] if step.get("node") in {"R3a", "R3b"}
        ) for record in records]
        evidence_signatures = [tuple(evidence_hash(step.get("reply")) for step in record.get("trace") or []) for record in records]
        wording_signatures = [tuple(wording_hash(step.get("reply")) for step in record.get("trace") or []) for record in records]
        validation_signatures = [tuple(
            (len(step.get("attempts") or []), tuple(attempt.get("ok") for attempt in step.get("attempts") or []))
            for step in record.get("trace") or []
        ) for record in records]
        abstentions = [tier == "abstained" for tier in tiers]
        fingerprints = [sorted({
            attempt.get("system_fingerprint")
            for step in record.get("trace") or []
            for attempt in step.get("attempts") or []
            if attempt.get("system_fingerprint")
        }) for record in records]
        cases[case_id] = {
            "repeat_count": len(records),
            "tiers": tiers,
            "tier_consistent": len(set(tiers)) <= 1,
            "node_answers_consistent": len(set(node_signatures)) <= 1,
            "leading_alternative_consistent": len(set(alternatives)) <= 1,
            "evidence_set_consistent": len(set(evidence_signatures)) <= 1,
            "wording_consistent": len(set(wording_signatures)) <= 1,
            "validation_retry_consistent": len(set(validation_signatures)) <= 1,
            "abstention_consistent": len(set(abstentions)) <= 1,
            "first_node_divergence": first_node_divergence(records),
            "fingerprints_by_repeat": fingerprints,
        }
    return {
        "case_count": len(cases),
        "stable_tier_cases": sum(item["tier_consistent"] for item in cases.values()),
        "stable_node_answer_cases": sum(item["node_answers_consistent"] for item in cases.values()),
        "stable_leading_alternative_cases": sum(item["leading_alternative_consistent"] for item in cases.values()),
        "stable_evidence_set_cases": sum(item["evidence_set_consistent"] for item in cases.values()),
        "stable_wording_cases": sum(item["wording_consistent"] for item in cases.values()),
        "stable_validation_retry_cases": sum(item["validation_retry_consistent"] for item in cases.values()),
        "stable_abstention_cases": sum(item["abstention_consistent"] for item in cases.values()),
        "cases": cases,
    }


def build_export(paths: list[pathlib.Path]) -> dict:
    all_rows: list[dict] = []
    all_evidence: list[dict] = []
    inputs = []
    stability = {}
    grounding_by_artifact = {}
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        rows, evidence = reliability_rows(run, path.name)
        all_rows.extend(rows)
        all_evidence.extend(evidence)
        artifact_counts = Counter(item["category"] for item in evidence)
        grounding_by_artifact[path.name] = {
            "evidence_item_count": len(evidence),
            "null_statement_id_count": sum(item["statement_id_null"] for item in evidence),
            "refined_category_counts": dict(sorted(artifact_counts.items())),
        }
        inputs.append({"path": str(path), "sha256": sha256_file(path), "result_count": len(run.get("results") or [])})
        if (run.get("meta") or {}).get("repeats", 1) > 1:
            stability[path.name] = stability_by_case(run)
    grounding_counts = Counter(item["category"] for item in all_evidence)
    return {
        "artifact_type": "offline_failure_mode_reliability_export",
        "external_api_calls": 0,
        "contains_raw_future_outcomes": False,
        "inputs": inputs,
        "rows": all_rows,
        "grounding": {
            "historical_null_statement_id_count": sum(item["statement_id_null"] for item in all_evidence),
            "evidence_item_count": len(all_evidence),
            "refined_category_counts": dict(sorted(grounding_counts.items())),
            "by_artifact": grounding_by_artifact,
            "note": "Structural/conservative classification; semantic claim matching requires adjudication.",
        },
        "stability": stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out")
    args = parser.parse_args()
    report = build_export([pathlib.Path(value) for value in args.runs])
    json_path = pathlib.Path(args.json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.csv_out:
        csv_path = pathlib.Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(report["rows"][0]) if report["rows"] else [])
            if report["rows"]:
                writer.writeheader()
                writer.writerows(report["rows"])
    print(json.dumps({"json_out": str(json_path), "csv_out": args.csv_out, "rows": len(report["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
