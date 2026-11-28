#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The code quality pass's job on the quality role's binding (config roles.quality; rkmon_lib/review.py starts it once
the PR is staged, beside the draft and the checks).

Usage: quality_run.py N [--prepare-only]
  makes the pass's folder <workers>/prN/quality afresh: pr/ (a link to the draft's staged data), the staging's
  mkwt.log, and the checks' overlap leads (`rk overlaps N --section`, read-only); then runs worker/quality_lane.py
  there, which writes quality.md. --prepare-only: the folder only.
From the review driver: RKMON_REVIEW_ROOT (config paths.root), RKMON_WORKERS (config paths.workers), and the job's
RKMON_* environment. Exit: the pass's (0 with quality.md); 1 when the PR is not staged.
"""

import os
import subprocess
import sys

M = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(M, "worker"))

from steps import PROG, echo, make_dirs, make_writable, remove, run  # noqa: E402


def main(argv: list) -> int:
    if len(argv) < 2:
        print(f"{PROG}: $1: unbound variable", file=sys.stderr, flush=True)
        return 1
    N, MODE = argv[1], (argv[2] if len(argv) > 2 else "")
    D = os.environ.get("RKMON_REVIEW_ROOT") or os.path.dirname(M)
    WK = os.environ.get("RKMON_WORKERS") or f"{D}/workers"
    WS = f"{WK}/pr{N}"
    Q = f"{WS}/quality"
    meta = f"{WS}/pr/meta.md"
    if not (os.path.exists(meta) and os.path.getsize(meta) > 0):
        echo(f"PR {N} is not staged ({meta}): no code quality pass")
        return 1
    make_writable(Q)
    remove(Q)
    make_dirs(Q)
    try:
        os.symlink("../pr", f"{Q}/pr")
    except OSError as exc:
        print(
            f"ln: failed to create symbolic link '{Q}/pr': {exc.strerror}",
            file=sys.stderr,
            flush=True,
        )
    try:
        with open(f"{WS}/mkwt.log", "rb") as src, open(f"{Q}/mkwt.log", "wb") as dst:
            dst.write(src.read())
    except OSError:
        pass
    with open(f"{Q}/overlap_section.md", "wb") as out:
        if run(
            [f"{M}/rk", "overlaps", N, "--section"],
            stdout=out,
            stderr=subprocess.DEVNULL,
        ):
            out.seek(0)
            out.truncate()
    if MODE == "--prepare-only":
        return 0
    echo("== code quality pass ==")
    return run(["python3", "-B", f"{M}/worker/quality_lane.py", Q, N])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
