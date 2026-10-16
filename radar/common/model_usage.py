"""Shared per-process accounting of model usage, and optional request options.

record(module, response) accumulates prompt/completion/total tokens on the
calling agent module (USAGE dict) so run metadata can report cost per agent.

request_options() returns extra chat.completions kwargs taken from the
environment, so a run can be configured without changing agent signatures:
  RADAR_REASONING_EFFORT = minimal | low | medium | high   (reasoning models)
A model that rejects an option gets the call without it (see each call_model).
"""
import os


def record(module, response):
    usage = getattr(response, "usage", None)
    acc = module.__dict__.setdefault("USAGE", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                               "reasoning_tokens": 0, "total_tokens": 0})
    acc["calls"] += 1
    if usage is None:
        return
    acc["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
    acc["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    acc["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    acc["reasoning_tokens"] += (getattr(details, "reasoning_tokens", 0) or 0) if details is not None else 0


def request_options():
    opts = {}
    effort = os.environ.get("RADAR_REASONING_EFFORT")
    if effort:
        opts["reasoning_effort"] = effort
    return opts


def drop_rejected(kwargs, exc):
    """If the error names one of our optional parameters, remove it and return its name."""
    msg = str(exc)
    for key in ("reasoning_effort", "temperature", "seed"):
        # OpenAI quotes the name as 'key'; Anthropic's compatible endpoint uses `key`
        if key in kwargs and any(f"{q}{key}{q}" in msg for q in "'`\"") and "400" in msg:
            kwargs.pop(key)
            return key
    return None
