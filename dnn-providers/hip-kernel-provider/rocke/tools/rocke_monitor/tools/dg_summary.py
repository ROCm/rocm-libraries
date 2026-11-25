#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Summarize a run_diff.py --check-golden log on one line: each check's counts (every flavor), then FAIL if one failed.
The log's last lines are only the last flavor's: every flavor's counts are summed.

Usage: dg_summary.py LOG
"""

import re
import sys

WS = re.compile(rb"[ \t\n]+")


def fields(line: bytes) -> list[bytes]:
    """awk's fields: the line split on runs of blanks."""
    return [f for f in WS.split(line) if f]


def summary(data: bytes) -> bytes:
    out, sep, bad = b"", b"", False
    w = c = g = b""
    for line in data.split(b"\n"):
        f = fields(line)

        def nth(k: int) -> bytes:
            return f[k - 1] if len(f) >= k else b""

        last = f[-1] if f else line
        if line.startswith(b"=== GOLDEN CHECK ("):
            w = nth(4).replace(b"(", b"").replace(b")", b"")
        if line.startswith(b"  UNBLESSED CHANGE:"):
            c = last
        if line.startswith(b"  MISSING (gone)"):
            g = last
        if line.startswith(b"  NEW (unblessed)"):
            out += sep + w + b": changed " + c + b", gone " + g + b", new " + last
            sep = b"; "
        if line.startswith(b"NO GOLDEN for "):
            w = nth(4)
            if w.endswith(b"."):
                w = w[:-1]
            out += sep + w + b": no golden"
            sep = b"; "
            bad = True
        if line.startswith(b"GOLDEN MISMATCH"):
            bad = True
    if out == b"":
        out, bad = b"no golden check in the log", True
    return out + (" · FAIL".encode() if bad else b"") + b"\n"


def main(argv: list[str]) -> int:
    path = argv[1] if len(argv) > 1 else "-"
    try:
        if path == "-":
            data = sys.stdin.buffer.read()
        else:
            with open(path, "rb") as fh:
                data = fh.read()
    except OSError as exc:  # as awk: nothing printed, exit 2
        sys.stderr.write(f"dg_summary: cannot open {path} ({exc.strerror})\n")
        return 2
    sys.stdout.buffer.write(summary(data))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
