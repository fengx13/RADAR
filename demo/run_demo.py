"""Run the demo scenarios through the unchanged RADAR pipeline.

    python demo/run_demo.py --model claude-sonnet-5-5 --run-tag v1 [--smoke] [--resume]
    python demo/run_demo.py --base-url openai --model gpt-6-luna --run-tag v1

Keys are read from ../.env (outside the repository): OPENAI_API_KEY and ANTHROPIC_API_KEY.

A thin wrapper around radar.pipeline.run_pipeline_vnext (a frozen policy file,
so it is not edited). When the endpoint is Anthropic's OpenAI-compatible API,
the OpenAI client is wrapped so requests go without response_format=json_object,
which that endpoint rejects (see radar/common/provider_compat.py). The output
meta records json_mode afterwards.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ANTHROPIC = "https://api.anthropic.com/v1/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--base-url", default=ANTHROPIC, help="'openai' for the default OpenAI endpoint")
    ap.add_argument("--smoke", action="store_true", help="only the first case")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", default="6")
    a = ap.parse_args()
    base_url = None if a.base_url == "openai" else a.base_url

    # Keys live one level above the repository (P4_RADAR/.env), never inside it:
    #   OPENAI_API_KEY=...      used when --base-url openai
    #   ANTHROPIC_API_KEY=...   used for Anthropic's OpenAI-compatible endpoint
    # The pipeline itself only reads OPENAI_API_KEY, so the right key is copied into it.
    env_file = HERE.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    if base_url and "anthropic.com" in base_url and os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

    import openai
    from radar.common.provider_compat import needs_prompt_only_json, prompt_only_json
    if needs_prompt_only_json(base_url):
        Original = openai.OpenAI

        class Patched(Original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                prompt_only_json(self)

        openai.OpenAI = Patched
        print("[note] endpoint rejects json_object; sending without response_format "
              "(prompts require JSON, replies still validated)")

    out = HERE.parent.parent / "RADAR_output" / "demo_runs" / "demo_run.json"
    argv = ["--infile", str(HERE / "scenarios" / "demo_cases.json"), "--out", str(out),
            "--run-tag", a.run_tag, "--model", a.model, "--temperature", "0", "--seed", "7",
            "--replay-mode", "full", "--continue-on-failure"]
    if base_url:
        argv += ["--base-url", base_url]
    argv += ["--batch-size", "1", "--max-batches", "1", "--force"] if a.smoke else \
            ["--workers", a.workers, "--batch-size", "12"] + (["--resume"] if a.resume else ["--force"])
    from radar.pipeline import run_pipeline_vnext
    rc = run_pipeline_vnext.main(argv)
    dest = out.with_name(f"{out.stem}_{a.run_tag}{out.suffix}")
    if dest.exists():
        data = json.loads(dest.read_text(encoding="utf-8"))
        from radar.agents import run_agent1, run_agent2, run_agent3, run_agent4, run_agent4_t0
        # the pipeline writes parameter_support before any call; record what was honoured in the end
        data["meta"]["parameter_support_after_run"] = {
            name: dict(getattr(mod, "CAPS", {})) for name, mod in
            (("A1", run_agent1), ("A2", run_agent2), ("A3", run_agent3), ("A4_T0", run_agent4_t0), ("A4_checkpoint", run_agent4))}
        data["meta"]["json_mode"] = "prompt_only" if needs_prompt_only_json(base_url) else "response_format_json_object"
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
