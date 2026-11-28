#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tool-loop and shadow draft stages' job (rkmon_lib/draft_stage.py).

Usage: draft_run.py N [--retry | --stage-only | --shadow BINDING HEAD]
  (none)        stage PR N's data (fetch_pr.py, and our last review: rk stage-ours) and trees (mkwt.py) as the
                local worker's launcher does, mark them staged (the checks may start), draft with the drafter role
                (worker/drafter.py), then check the draft's evidence (rk evidence-check), so the check is done before
                REVIEW_READY
  --retry       draft again on the staging already there
  --stage-only  stage only (the stage "none")
  --shadow BINDING HEAD [--stage-only]   a shadow draft: in its own folder <workers>/shadow_BINDING/prN, with its own
                staged data (never the live draft), on the live review's trees once they hold HEAD: it waits for the
                live job to make them (up to RKMON_SHADOW_WAIT_MIN minutes, default 20; RKMON_SHADOW_POLL_S seconds
                between looks, default 20) and never makes or resets them; drafted with the shadow_drafter role, then
                its evidence checked in its folder
From the review driver: RKMON_REVIEW_ROOT (config paths.root), RKMON_WORKERS (config paths.workers), REVIEW_ROUND and
REVIEW_LINE1 (fetch_pr.py stages them in pr/meta.md), SPARSE_EXTRA (mkwt.py), RKMON_DRAFT_PRIO (the drafter's rank at
the limiter), and the job's RKMON_* environment (the worker tools' folder, RKMON_WORKER_TOOLS). Staging holds a lock
(<workers>/.stage.lock), so drafts that run at once fetch and make their trees one at a time. Exit: the drafter's (0
with a draft).
"""

import fcntl
import os
import subprocess
import sys
import time

M = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(M, "worker"))

from steps import (
    PROG,
    echo,
    grep,
    found,
    make_dirs,
    make_writable,
    remove,
    run,
    tail,
    utc_stamp,
)  # noqa: E402


def need(argv: list, i: int, usage: str) -> str:
    """A required argument (the shell's ${i:?usage}): missing, the usage on stderr and exit 1."""
    if len(argv) > i and argv[i]:
        return argv[i]
    print(f"{PROG}: {i}: {usage}", file=sys.stderr, flush=True)
    sys.exit(1)


def head_of(tree: str, *args: str) -> str:
    p = subprocess.run(
        ["git", "-C", tree, "rev-parse", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return p.stdout.decode("utf-8", "surrogateescape").rstrip("\n")


def stamp(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(utc_stamp() + "\n")


def evidence_check(N: str, *extra: str) -> None:
    echo("== evidence check ==")
    run([f"{M}/rk", "evidence-check", N, *extra])


def shadow(argv: list, N: str, D: str, T: str, WK: str) -> int:
    usage = "usage: draft_run.py N --shadow BINDING HEAD"
    B, H = need(argv, 3, usage), need(argv, 4, usage)
    WS = f"{WK}/shadow_{B}/pr{N}"
    make_writable(WS)
    remove(WS)
    make_dirs(WS)
    if run(["python3", f"{T}/fetch_pr.py", N, f"{WS}/pr"]):
        return 1
    run([f"{M}/rk", "stage-ours", N, f"{WS}/pr"])
    # The live trees hold HEAD once the head tree is at it, the merged tree is a merge of it (or a conflicted merge on
    # it), and the live job's mkwt.log names it and has ended its conflict list.
    end = int(time.time()) + int(os.environ.get("RKMON_SHADOW_WAIT_MIN") or 20) * 60
    log = f"{WK}/pr{N}/mkwt.log"

    def held() -> bool:
        if head_of(f"{D}/wt{N}", "HEAD") != H:
            return False
        merged = f"{D}/wt{N}m"
        if head_of(merged, "-q", "--verify", "HEAD^1") != H and not (
            head_of(merged, "HEAD") == H
            and head_of(merged, "-q", "--verify", "MERGE_HEAD")
        ):
            return False
        return found(f"head={H[:10]}", log) and found(
            "^conflicting files vs develop:", log
        )

    while not held():
        if not int(time.time()) < end:
            echo(f"the live trees never held {H[:10]}: no shadow draft")
            return 1
        time.sleep(float(os.environ.get("RKMON_SHADOW_POLL_S") or 20))
    copy_log(log, f"{WS}/mkwt.log")
    grep("head=|behind/ahead", f"{WS}/mkwt.log")
    stamp(f"{WS}/staged")
    if len(argv) > 5 and argv[5] == "--stage-only":
        return 0
    echo(f"== shadow draft ({B}) ==")
    rc = run(
        ["python3", "-B", f"{M}/worker/drafter.py", WS, N, "--role", "shadow_drafter"]
    )
    if (
        os.path.isfile(f"{WS}/review_agent.md")
        and os.path.getsize(f"{WS}/review_agent.md") > 0
    ):
        evidence_check(N, "--dir", WS)
    return rc


def copy_log(src: str, dst: str) -> None:
    """`cp SRC DST`: an error is said and the job goes on."""
    try:
        with open(src, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(data)
    except OSError as exc:
        print(f"cp: cannot copy '{src}': {exc.strerror}", file=sys.stderr, flush=True)


def stage(N: str, D: str, T: str, WK: str, WS: str) -> int:
    """The PR's data and trees, under the staging lock: 0, or 1 when a step failed."""
    lock_path = f"{WK}/.stage.lock"
    try:
        lock = open(lock_path, "w")
    except OSError as exc:
        print(f"{PROG}: {lock_path}: {exc.strerror}", file=sys.stderr, flush=True)
        return 1
    with lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)  # blocks while another draft stages
        if run(["python3", f"{T}/fetch_pr.py", N, f"{WS}/pr"]):
            return 1
        run([f"{M}/rk", "stage-ours", N, f"{WS}/pr"])
        for W in (f"wt{N}", f"wt{N}m"):
            if os.path.isdir(f"{D}/{W}"):
                run(
                    ["git", "-C", f"{D}/{W}", "merge", "--abort"],
                    stderr=subprocess.DEVNULL,
                )
                run(
                    ["git", "-C", f"{D}/{W}", "reset", "-q", "--hard"],
                    stderr=subprocess.DEVNULL,
                )
        # The installed mkwt.py, or the package's own while the install lacks it (rkmon_lib/paths.py worker_tool).
        mkwt = (
            f"{T}/mkwt.py" if os.path.isfile(f"{T}/mkwt.py") else f"{M}/worker/mkwt.py"
        )
        try:
            with open(f"{WS}/mkwt.log", "wb") as log:
                rc = run(["python3", "-B", mkwt, N], stdout=log, stderr=log)
        except OSError as exc:
            print(f"{PROG}: {WS}/mkwt.log: {exc.strerror}", file=sys.stderr, flush=True)
            rc = 1
        if rc:
            echo("mkwt failed:")
            tail(f"{WS}/mkwt.log", 20)
            return 1
    return 0


def main(argv: list) -> int:
    if len(argv) < 2:
        print(f"{PROG}: $1: unbound variable", file=sys.stderr, flush=True)
        return 1
    N, MODE = argv[1], (argv[2] if len(argv) > 2 else "")
    D = os.environ.get("RKMON_REVIEW_ROOT") or os.path.dirname(M)
    T = os.environ.get("RKMON_WORKER_TOOLS") or f"{D}/tools"
    WK = os.environ.get("RKMON_WORKERS") or f"{D}/workers"
    WS = f"{WK}/pr{N}"
    if MODE == "--shadow":
        return shadow(argv, N, D, T, WK)
    if MODE != "--retry":
        make_dirs(WS)
        if stage(N, D, T, WK, WS):
            return 1
        grep("head=|behind/ahead", f"{WS}/mkwt.log")
        stamp(f"{WS}/staged")
    if MODE == "--stage-only":
        return 0
    echo("== draft ==")
    rc = run(["python3", "-B", f"{M}/worker/drafter.py", WS, N])
    if (
        os.path.isfile(f"{WS}/review_agent.md")
        and os.path.getsize(f"{WS}/review_agent.md") > 0
    ):
        evidence_check(N)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
