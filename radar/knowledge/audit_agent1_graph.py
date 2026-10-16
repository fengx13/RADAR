#!/usr/bin/env python3
"""Backward-compatible entry point for the current Agent 1 reachability audit."""
from __future__ import annotations
from radar import paths

import argparse
import json
import pathlib

from radar.knowledge.graph_reachability import DEFAULT_GRAPH, analyze_graph, display_path, load_graph


ROOT = paths.ROOT
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument("--out", default=str(paths.DATA / "agent1_graph_audit.json"))
    args = parser.parse_args()
    graph_path = pathlib.Path(args.graph)
    report = analyze_graph(load_graph(graph_path))
    report["graph_path"] = display_path(graph_path)
    report["artifact_status"] = "current_v0.6_structural_audit"
    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
