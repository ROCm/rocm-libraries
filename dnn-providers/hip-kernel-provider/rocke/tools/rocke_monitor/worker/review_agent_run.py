#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Usage: review_agent_run.py <PR number>  (env NUM_BATCH default 256; REVIEW_ROUND and REVIEW_LINE1, set by the
monitor, reach fetch_pr.py, which stages them in pr/meta.md)

Constrained agentic first-round review: the local model runs vetted tools on the worktree.
The review offload (RKMON_OFFLOAD_MODE, config offload.mode): on, the offload passes (offload_agent.py: draft,
quality, regrade) write the review in place of review_agent.py's draft; shadow, they run after it, into offload/,
and a failure there leaves the draft alone.
Paths and settings come from the job's RKMON_* environment (rkenv.py). A failing step ends it with that step's exit
code.
"""

import os
import shutil
import stat
import subprocess
import sys

T = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, T)

import rkenv  # noqa: E402
from steps import echo, grep, make_dirs, make_writable, must, run, tail  # noqa: E402


def copy(src: str, dst: str) -> None:
    """`cp SRC DST` (set -e): a new file gets the source's mode less the umask."""
    new = not os.path.exists(dst)
    try:
        shutil.copyfile(src, dst)
        if new:
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(dst, stat.S_IMODE(os.stat(src).st_mode) & ~umask)
    except OSError as exc:
        print(
            f"cp: cannot copy '{src}' to '{dst}': {exc.strerror}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


def main(argv: list) -> int:
    rkenv.load()
    N = argv[1] if len(argv) > 1 else ""
    D = os.environ["RKMON_WORKING_DIR"]
    WS = f"{os.environ.get('RKMON_WORKERS', '')}/pr{N}"
    if not N:
        echo("usage: review_agent_run.py <PR number>")
        return 2
    book = os.environ.get("RKMON_REVIEW_BOOK", "")
    if not (book and os.path.exists(book) and os.path.getsize(book) > 0):
        echo(
            f"no review book at '{book}' (paths.review_book): the container's first start puts it there (rkmon start)"
        )
        return 2
    make_writable(WS)
    if make_dirs(WS):
        return 1
    must(["python3", "-B", f"{T}/fetch_pr.py", N, f"{WS}/pr"])
    copy(book, f"{WS}/review_book.md")
    if make_dirs(f"{WS}/.gh-none"):
        return 1
    # Always refresh both worktrees to the current PR head and develop (abort any earlier conflicted merge).
    for W in (f"wt{N}", f"wt{N}m"):
        if os.path.isdir(f"{D}/{W}"):
            run(
                ["git", "-C", f"{D}/{W}", "merge", "--abort"], stderr=subprocess.DEVNULL
            )
            run(
                ["git", "-C", f"{D}/{W}", "reset", "-q", "--hard"],
                stderr=subprocess.DEVNULL,
            )
    try:
        with open(f"{WS}/mkwt.log", "wb") as log:
            rc = run(["python3", "-B", f"{T}/mkwt.py", N], stdout=log, stderr=log)
    except OSError as exc:
        print(f"{WS}/mkwt.log: {exc.strerror}", file=sys.stderr, flush=True)
        rc = 1
    if rc:
        echo("mkwt failed:")
        tail(f"{WS}/mkwt.log", 20)
        return 1
    rc = grep("head=|behind/ahead", f"{WS}/mkwt.log")
    if rc:
        return rc
    mode = os.environ.get("RKMON_OFFLOAD_MODE") or "off"
    if mode == "on":
        echo("== offload passes (draft, quality, regrade) ==")
        must(
            [
                "python3",
                "-B",
                f"{T}/offload_agent.py",
                WS,
                N,
                "draft",
                "quality",
                "regrade",
            ]
        )
        echo(f"== review: {WS}/draft.md, quality.md, regrade.md ==")
    else:
        batch = os.environ.get("NUM_BATCH") or "256"
        echo(f"== agent run, NUM_BATCH={batch} ==")
        must(
            ["python3", "-B", f"{T}/review_agent.py", WS, N],
            env=dict(os.environ, NUM_BATCH=batch),
        )
        echo(f"== review: {WS}/review_agent.md ==")
        if mode == "shadow":
            echo("== shadow offload passes (they post nothing) ==")
            if run(
                [
                    "python3",
                    "-B",
                    f"{T}/offload_agent.py",
                    f"{WS}/offload",
                    N,
                    "draft",
                    "quality",
                    "regrade",
                    "--pr-dir",
                    f"{WS}/pr",
                ]
            ):
                echo("(the shadow offload passes failed; the draft above stands)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
