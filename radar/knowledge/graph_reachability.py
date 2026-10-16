#!/usr/bin/env python3
"""Enumerate Agent 1 graph paths without calling a model."""
from __future__ import annotations
from radar import paths

import argparse
import json
import pathlib
from collections import Counter


ROOT = paths.ROOT
DEFAULT_GRAPH = paths.GRAPH / "agent1_decision_graph.json"


def display_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_graph(path: pathlib.Path = DEFAULT_GRAPH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _next_node(node: dict, answer: str, answers: dict[str, str]) -> str:
    target = node["transitions"][answer]
    if not isinstance(target, dict):
        return target
    for dependency in target.get("depends_on") or []:
        key = f"{dependency}:{answers.get(dependency)}"
        if key in target.get("map", {}):
            return target["map"][key]
    return "ROUTING_ERROR"


def enumerate_paths(graph: dict, decision_time: str) -> dict:
    restrictions = (
        graph.get("reachability_constraints", {}).get(decision_time, {}).get("forbidden_answers", {})
    )
    paths: list[dict] = []
    reachable_nodes: set[str] = set()

    def visit(node_id: str, answers: dict[str, str], path: list[dict]):
        if node_id not in graph["nodes"]:
            paths.append({"terminal": node_id, "path": path})
            return
        reachable_nodes.add(node_id)
        node = graph["nodes"][node_id]
        forbidden = set(restrictions.get(node_id) or [])
        for answer in node["allowed_answers"]:
            if answer in forbidden:
                continue
            next_answers = {**answers, node_id: answer}
            next_path = [*path, {"node": node_id, "answer": answer}]
            visit(_next_node(node, answer, next_answers), next_answers, next_path)

    visit(graph["entry"], {}, [])
    counts = Counter(row["terminal"] for row in paths)
    declared = set(graph.get("terminals") or {})
    reachable_terminals = sorted(declared & set(counts))
    return {
        "decision_time": decision_time,
        "restrictions_applied": restrictions,
        "reachable_nodes": sorted(reachable_nodes),
        "unreachable_nodes": sorted(set(graph["nodes"]) - reachable_nodes),
        "reachable_terminals": reachable_terminals,
        "unreachable_terminals": sorted(declared - set(counts)),
        "path_count": len(paths),
        "path_counts_by_terminal": {key: counts[key] for key in sorted(counts)},
    }


def analyze_graph(graph: dict) -> dict:
    contexts = {
        decision_time: enumerate_paths(graph, decision_time)
        for decision_time in ("T0", "reassessment")
    }
    limitations = []
    if "T_DEFINITE" in graph.get("terminals", {}) and all(
        "T_DEFINITE" in result["unreachable_terminals"] for result in contexts.values()
    ):
        limitations.append(
            "T_DEFINITE is described as reassessment-only but is unreachable in the supplied v0.6 transitions."
        )
    limitations.append(
        "Reachability is structural only. The real v0.6 baseline answered R2=no in 8/8; clinical correctness is not inferred."
    )
    return {
        "agent": graph.get("agent"),
        "graph_version": graph.get("version"),
        "graph_path": display_path(DEFAULT_GRAPH),
        "contexts": contexts,
        "known_limitations": limitations,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", nargs="?", default=str(DEFAULT_GRAPH))
    parser.add_argument("--out")
    args = parser.parse_args()
    graph_path = pathlib.Path(args.graph)
    report = analyze_graph(load_graph(graph_path))
    report["graph_path"] = display_path(graph_path)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        destination = pathlib.Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
