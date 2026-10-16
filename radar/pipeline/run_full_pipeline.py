#!/usr/bin/env python3
"""Compatibility entrypoint for the active sequential RADAR pipeline.

Default execution delegates to run_pipeline.py. Supply externally assembled
longitudinal cases with --infile; --workers, --batch-size and --run-tag apply to
whole patient trajectories without parallelizing checkpoints within a patient.

Yana's historical cohort workflow is preserved behind the explicit option
--pipeline-mode legacy_cohort_two_time. The remaining documentation describes
that two-time comparison mode, not the active longitudinal implementation.

Six stages:

    T0   Agent 1 recognition      -> tier
         Agent 2 workup           -> what to order
         Agent 4 disposition@T0   -> stop, or continue the workup
    T1   Agent 1 reassessment     -> tier, now with results
         Agent 3 treatment        -> TX1-TX4
         Agent 4 disposition@T1   -> where the patient goes

Batches
-------
With --batch-size the cohort is cut into groups that each go through all six
stages before the next one starts. You see a real per-case time after the first
group instead of after the whole run, and a crash costs you one group. Each
group writes its own files; they are merged into the canonical
<prefix>_* names at the end, so scoring does not know batching happened.

Every stage writes its own file. Re-running skips stages whose output already
exists unless you pass --force.

    python -m radar.pipeline.run_full_pipeline --dry-run                        # dynamic fixture
    python -m radar.pipeline.run_full_pipeline --pipeline-mode legacy_cohort_two_time --dry-run
    python -m radar.pipeline.run_full_pipeline --pipeline-mode legacy_cohort_two_time --workers 8
"""
from __future__ import annotations
from radar import paths

import argparse
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
D = paths.DATA


def sh(cmd, label, quiet=False):
    if not quiet:
        print(f"  $ {' '.join(cmd[1:])}", flush=True)
    started = time.time()
    r = subprocess.run(cmd, cwd=paths.ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:])
        print(r.stderr[-2500:], file=sys.stderr)
        raise SystemExit(f"!! {label} failed")
    return time.time() - started


def read(name):
    p = D / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write(obj, name):
    (D / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def rows_of(doc):
    """Both shapes are in use: a bare list, or a dict with results and meta."""
    if isinstance(doc, dict):
        return doc.get("results", [])
    return doc or []


def count(name):
    doc = read(name)
    return len(rows_of(doc)) if doc is not None else "?"


def stage_plan(P, OP, mode, par, droot, py, states):
    return [
        (1, "Agent 1 @T0", f"{OP}_agent1_t0.json", [
            py + ["-m", "radar.agents.run_agent1", *mode, *par, "--infile", str(paths.DATA / f"{P}_cases.json"),
                  "--out", str(paths.DATA / f"{OP}_agent1_t0.json")]]),
        (2, "Agent 2 @T0", f"{OP}_agent2_run.json", [
            py + ["-m", "radar.legacy_unavailable.build_phase0_inputs", "--stage", "a2", "--cases", f"{P}_cases.json",
                  "--out-prefix", OP, "--agent1-run", f"{OP}_agent1_t0.json"],
            py + ["-m", "radar.agents.run_agent2", *mode, *par, "--infile", str(paths.DATA / f"{OP}_agent2_input.json"),
                  "--out", str(paths.DATA / f"{OP}_agent2_run.json")]]),
        (3, "Agent 4 @T0", f"{OP}_agent4_t0_run.json", [
            py + ["-m", "radar.legacy_unavailable.build_phase0_inputs", "--stage", "a4", "--cases", f"{P}_cases.json",
                  "--out-prefix", OP, "--agent1-run", f"{OP}_agent1_t0.json",
                  "--agent2-run", f"{OP}_agent2_run.json"],
            py + ["-m", "radar.agents.run_agent4_t0", *mode, *par, "--infile", str(paths.DATA / f"{OP}_agent4_t0_input.json"),
                  "--out", str(paths.DATA / f"{OP}_agent4_t0_run.json")]]),
        (4, "Agent 1 @T1", f"{OP}_agent1_t1.json", [
            py + ["-m", "radar.agents.run_agent1", *mode, *par, "--decision-time", "checkpoint",
                  "--infile", str(paths.DATA / f"{P}_{states}.json"),
                  "--out", str(paths.DATA / f"{OP}_agent1_t1.json")]]),
        (5, "Agent 3 @T1", f"{OP}_agent3_run.json", [
            py + ["-m", "radar.legacy_unavailable.build_checkpoint_payload", "--states", f"{P}_{states}.json",
                  "--crosswalk", f"{P.split('_b')[0]}_identifiers.json",
                  "--agent1-run", f"{OP}_agent1_t1.json",
                  "--out-a3", f"{OP}_agent3_input.json",
                  "--out-a4", f"{OP}_agent4_ckpt_input.json", *droot],
            py + ["-m", "radar.agents.run_agent3", *mode, *par, "--infile", str(paths.DATA / f"{OP}_agent3_input.json"),
                  "--out", str(paths.DATA / f"{OP}_agent3_run.json")]]),
        (6, "Agent 4 @T1", f"{OP}_agent4_ckpt_run.json", [
            py + ["-m", "radar.agents.run_agent4", *mode, *par, "--infile", str(paths.DATA / f"{OP}_agent4_ckpt_input.json"),
                  "--out", str(paths.DATA / f"{OP}_agent4_ckpt_run.json")]]),
    ]


STAGE_OUTPUTS = ["agent1_t0.json", "agent2_run.json", "agent4_t0_run.json",
                 "agent1_t1.json", "agent3_run.json", "agent4_ckpt_run.json"]


def run_group(P, OP, args, py, mode, par, droot, indent="", quiet=False):
    """One prefix through the requested stages. Returns seconds spent on models."""
    spent = 0.0
    for number, label, output, cmds in stage_plan(P, OP, mode, par, droot, py, args.states):
        if number not in args.stages:
            continue
        if (D / output).exists() and not args.force:
            print(f"{indent}[{number}/6] {label:<16} done already, {count(output)} cases")
            continue
        took = sum(sh(c, label, quiet) for c in cmds)
        spent += took
        print(f"{indent}[{number}/6] {label:<16} {count(output):>4} cases  {took/60:5.1f} min")
    return spent


def make_batches(P, size, stages, states):
    """Slice the cohort and its T1 states into groups, and return their prefixes."""
    cases = read(f"{P}_cases.json")
    if cases is None:
        raise SystemExit(f"{paths.DATA / P}_cases.json not found")
    t1 = read(f"{P}_{states}.json") or []
    t1_by_id = {r["case_id"]: r for r in t1}

    prefixes = []
    for i in range(0, len(cases), size):
        group = cases[i:i + size]
        prefix = f"{P}_b{i // size + 1:02d}"
        write(group, f"{prefix}_cases.json")
        if any(s in stages for s in (4, 5, 6)):
            write([t1_by_id[c["case_id"]] for c in group if c["case_id"] in t1_by_id],
                  f"{prefix}_{states}.json")
        prefixes.append((prefix, len(group)))
    return prefixes


def merge(OP, prefixes, stages, tag=None):
    """Fold the group outputs back into the canonical names, in cohort order."""
    wanted = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    for number, index in wanted.items():
        if number not in stages:
            continue
        name = STAGE_OUTPUTS[index]
        merged, meta, missing = [], None, 0
        for prefix, _ in prefixes:
            doc = read(f"{prefix}_{tag}_{name}" if tag else f"{prefix}_{name}")
            if doc is None:
                missing += 1
                continue
            if isinstance(doc, dict) and meta is None:
                meta = {k: v for k, v in doc.items() if k != "results"}
            merged.extend(rows_of(doc))
        if not merged:
            continue
        out = ({**meta, "results": merged} if meta is not None else merged)
        write(out, f"{OP}_{name}")
        note = f"   ({missing} groups had no output)" if missing else ""
        print(f"  merged {len(merged):>4} cases -> {paths.DATA / OP}_{name}{note}")


def run_legacy_cohort_main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--prefix", default="cohort100")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="rerun stages that already have output")
    p.add_argument("--stages", nargs="+", type=int, choices=range(1, 7), default=list(range(1, 7)))
    p.add_argument("--data-root", default=None)
    p.add_argument("--workers", type=int, default=8,
                   help="cases run concurrently; the work is API latency, not CPU")
    p.add_argument("--states", default="tsofa_states",
                   help="which checkpoint the T1 stages use: tsofa_states, where SOFA is "
                        "computable, or t1_states, the first returning result")
    p.add_argument("--run-tag", default=None,
                   help="label this run so it writes its own file set, e.g. r2. "
                        "Inputs are shared; nothing from an earlier run is overwritten.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="run this many cases through all six stages before starting the next group")
    p.add_argument("--max-batches", type=int, default=None,
                   help="stop after this many groups, to time a run before committing to it")
    args = p.parse_args(argv)

    P = args.prefix
    OP = f"{P}_{args.run_tag}" if args.run_tag else P
    mode = ["--dry-run"] if args.dry_run else ["--model", args.model]
    py = [sys.executable]
    droot = ["--data-root", args.data_root] if args.data_root else []
    par = ["--workers", str(args.workers)]

    if not (D / f"{P}_{args.states}.json").exists() and any(s in args.stages for s in (4, 5, 6)):
        raise SystemExit(
            f"{paths.DATA / P}_{args.states}.json is missing; the reassessment stages need it.\n"
            "  python -m radar.cohort.make_t1_states     (about 4 minutes, it scans labevents)")

    header = (f"model {'dry-run stub' if args.dry_run else args.model}   cohort {P}   "
              f"{args.workers} workers   reassessment at {args.states}"
              + (f"   run tag {args.run_tag}" if args.run_tag else ""))
    if not args.batch_size:
        print(header + "\n")
        total = run_group(P, OP, args, py, mode, par, droot)
        print(f"\ntotal {total/60:.1f} min")
    else:
        groups = make_batches(P, args.batch_size, args.stages, args.states)
        if args.max_batches:
            groups = groups[:args.max_batches]
        planned = sum(n for _, n in groups)
        print(header + f"   {len(groups)} groups of up to {args.batch_size}\n")

        total, done = 0.0, 0
        for order, (prefix, n) in enumerate(groups, 1):
            print(f"--- group {order}/{len(groups)}  ({n} cases)")
            took = run_group(prefix, f"{prefix}_{args.run_tag}" if args.run_tag else prefix,
                             args, py, mode, par, droot, indent="    ", quiet=True)
            total += took
            done += n
            per_case = total / done if done else 0
            left = planned - done
            print(f"    group done in {took/60:.1f} min"
                  f"   |   {per_case:.1f} s per case"
                  f"   |   {done}/{planned} cases"
                  + (f"   |   about {left * per_case / 60:.0f} min left" if left else "")
                  + "\n", flush=True)

        print("merging groups")
        merge(OP, groups, args.stages, args.run_tag)
        print(f"\ntotal {total/60:.1f} min for {done} cases "
              f"({total/done:.1f} s per case)" if done else "")

    if not args.dry_run:
        print("\nnow score it:\n    python -m radar.evaluation.score_hema_metrics --out hema_metrics.json")


def main(argv=None):
    from radar.pipeline.run_pipeline import main as run_pipeline_main
    return run_pipeline_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
