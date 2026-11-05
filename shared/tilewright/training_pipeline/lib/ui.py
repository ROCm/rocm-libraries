# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Terminal output helpers shared by the stages: coloured `[tag]` lines,
phase banners, and human-readable integer / duration formatting.

`warn` and `err` write to stderr, everything else to stdout. Colours are off
when stdout is not a TTY, so stage logs stay plain text.
"""
from __future__ import annotations

import sys


class _C:
    _tty = sys.stdout.isatty()
    HEAD = "\033[95m" if _tty else ""
    BLUE = "\033[94m" if _tty else ""
    GREEN = "\033[92m" if _tty else ""
    YELLOW = "\033[93m" if _tty else ""
    RED = "\033[91m" if _tty else ""
    GREY = "\033[90m" if _tty else ""
    ENDC = "\033[0m" if _tty else ""
    BOLD = "\033[1m" if _tty else ""


C = _C


def banner(title: str, color: str = _C.HEAD) -> None:
    """80-column divider with the title on the middle row."""
    print(f"\n{color}{'='*80}{_C.ENDC}", flush=True)
    print(f"{color}{_C.BOLD}  {title}{_C.ENDC}", flush=True)
    print(f"{color}{'='*80}{_C.ENDC}", flush=True)


def ok(tag: str, msg: str) -> None:
    print(f"{_C.GREEN}[{tag}]{_C.ENDC} {msg}", flush=True)


def info(tag: str, msg: str) -> None:
    print(f"{_C.BLUE}[{tag}]{_C.ENDC} {msg}", flush=True)


def warn(tag: str, msg: str) -> None:
    sys.stdout.flush()
    print(f"{_C.YELLOW}[{tag}]{_C.ENDC} {msg}", file=sys.stderr, flush=True)


def err(tag: str, msg: str) -> None:
    sys.stdout.flush()
    print(f"{_C.RED}[{tag}]{_C.ENDC} {msg}", file=sys.stderr, flush=True)


def grey(tag: str, msg: str) -> None:
    print(f"{_C.GREY}[{tag}]{_C.ENDC} {msg}", flush=True)


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_dur(s: float) -> str:
    """Seconds as '45s', '23m45s' or '1h23m' depending on magnitude."""
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"
