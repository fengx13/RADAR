"""Adapt the OpenAI client to OpenAI-compatible endpoints with narrower support.

Anthropic's OpenAI-compatible endpoint rejects response_format={"type": "json_object"}
(it accepts only json_schema). Every agent already asks for JSON in its prompt and
parses tolerantly (parse_json extracts the outermost {...}), and every reply still goes
through the same V1-V4 validators, so on such endpoints the request is sent without
response_format. Agent code is unchanged; the run metadata records the adaptation.
"""
from __future__ import annotations


def needs_prompt_only_json(base_url: str | None) -> bool:
    return bool(base_url) and "anthropic.com" in base_url


def prompt_only_json(client):
    completions = client.chat.completions
    original = completions.create

    def create(*args, **kwargs):
        rf = kwargs.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            kwargs.pop("response_format")
        return original(*args, **kwargs)

    completions.create = create
    return client
