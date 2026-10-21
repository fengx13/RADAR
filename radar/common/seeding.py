"""Stable seeds for cohort sampling.

Python's built-in hash() of a string changes with every process (PYTHONHASHSEED),
so a seed built from it gives a different draw on each run. stable_seed() uses
CRC-32 of the UTF-8 text instead, which is the same on every machine and run.
"""
import zlib


def stable_seed(text: str) -> int:
    """A non-negative integer below 10**6 that depends only on `text`."""
    return zlib.crc32(str(text).encode("utf-8")) % 10 ** 6
