#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Re-run the saved probes of our open findings on a PR's tree, for the verifier (lessons L-525).

Usage: run_probes.py <probe dir, e.g. $D/probes/pr<N>> <tree root> [<probe name>]

A probe is `bash <probe>.sh <tree root>`: exit 1 while the finding reproduces, 0 once it is fixed, anything else is an
error; its last output line is the summary. One line per probe: "<name>: REPRODUCES|FIXED|ERROR rc=N: <summary>".
A Python traceback ending the output of an exit 1 is a crash: "ERROR rc=1 (crashed)". With a probe name, only that
probe runs (the drafter's settling step runs each on its own, as worker/review_tools.py's `probe` tool).
A finding can sit in a file a push does not touch: its saved probe re-checks it on every round, by script.
PROBE_DETAIL=1 (the review checks): each line also gives the exit code, the tree's commit and the seconds the probe
took: "<name>: REPRODUCES|FIXED|ERROR rc=N (tree <commit>, <s> s): <summary>", so a reader trusts it without a re-run.
A probe runs off the GPU as every check runs (rkmon_lib/nogpu.py): without HIP_VISIBLE_DEVICES, whose -1 this host's
HIP runtime refuses with an abort (rc 134) as soon as a probe imports torch, and with an empty CUDA_VISIBLE_DEVICES,
which the HIP runtime honours by showing no device; torch's CPU paths need none. A caller's setting is dropped too.
A probe runs PR code, with only what worker/pr_env.py lists of its caller's environment, on which the contract's names
are set: in the container in a PR runner's slot (the launcher, docker/rk_prcode.py), on a host as the session user.
It runs clean (CLEAN): no HIP_VISIBLE_DEVICES and no ROCKE_LLVM_FLAVOR, whatever the caller set, so it reads the tree
as CI does; a probe that needs a flavor sets it itself. pytest's rc 5 (it collected no test) reads "ERROR rc=5 (no test
ran)": a probe that ran nothing never reads as a verdict.
The probes are listed in the locale's collation order, as `ls` lists them.

The probe environment contract: a probe names no machine's folder. Its tree root is `$1`, and every probe gets
  PROBE_PYTHON          the checks venv's python ($RKMON_VENV/bin/python; python3 without one): rocKE's tests and
                        imports run with it, as the checks do
  RKMON_ROCM_LLVM_BIN   ROCm's LLVM tools (clang, llc, ...): $RKMON_LLVM_BIN, paths.rocm_llvm_bin
  RKMON_BUILD_DIR       the tree's native build (build_native.py's <build>/<tree's folder name>; core/ and pybind/)
  RKMON_REPO            the repository (owner/name), and RKMON_REPO_DIR the monitor's clone of it
A value the job does not know is empty; a probe that needs it says so and exits 2 (an ERROR, never a verdict).
"""

import locale
import os
import re
import subprocess
import sys
import time

START = time.monotonic()
TRACEBACK = re.compile(rb"(?m)^Traceback \(most recent call last\):")
EXCEPTION = re.compile(rb"^[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)\b")
ECHO_OPTION = re.compile(rb"-[neE]+")


def seconds() -> int:
    """bash's $SECONDS: whole seconds since the start."""
    return int(time.monotonic() - START)


def out(line: bytes) -> None:
    sys.stdout.buffer.write(line + b"\n")
    sys.stdout.buffer.flush()


def echoed(text: bytes) -> bytes:
    """What `echo "$text"` passes on: nothing when the whole text is an echo option such as -n."""
    return b"" if ECHO_OPTION.fullmatch(text) else text


def probe_list(folder: str, only: str) -> list[str]:
    if only:
        p = f"{folder}/{only}.sh"
        return [p] if os.path.lexists(p) else []
    try:
        names = [
            n for n in os.listdir(folder) if n.endswith(".sh") and not n.startswith(".")
        ]
    except OSError:
        return []
    prefix = folder + "/"
    return sorted((prefix + n for n in names), key=locale.strxfrm)


# Dropped from every probe's environment: a caller's GPU mask or forced LLVM flavor changes what a probe reads.
CLEAN = ("HIP_VISIBLE_DEVICES", "ROCKE_LLVM_FLAVOR")


def tree_sha(tree: str) -> bytes:
    """`git -C TREE rev-parse --short=10 HEAD 2>/dev/null || echo '?'`, as a command substitution keeps it."""
    try:
        p = subprocess.run(
            ["git", "-C", tree, "rev-parse", "--short=10", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        text, rc = p.stdout, p.returncode
    except OSError:
        text, rc = b"", 127
    if rc:
        text += b"?\n"
    return text.rstrip(b"\n")


def probe_env(tree: str, environ=os.environ) -> dict:
    """The probe environment contract (the module's docstring): the job's RKMON_* settings as the names a probe uses."""
    venv = environ.get("RKMON_VENV", "")
    py = f"{venv}/bin/python" if venv and os.path.exists(f"{venv}/bin/python") else ""
    build = environ.get("RKMON_BUILD", "")
    name = os.path.basename(os.path.normpath(tree)) if tree else ""
    return {
        "PROBE_PYTHON": py or "python3",
        "RKMON_ROCM_LLVM_BIN": environ.get("RKMON_LLVM_BIN", ""),
        "RKMON_BUILD_DIR": f"{build}/{name}" if build and name else "",
        "RKMON_REPO": environ.get("RKMON_REPO", ""),
        "RKMON_REPO_DIR": environ.get("RKMON_REPO_DIR", ""),
    }


def main(argv: list[str]) -> int:
    locale.setlocale(locale.LC_COLLATE, "")
    folder = argv[1] if len(argv) > 1 else ""
    tree = argv[2] if len(argv) > 2 else ""
    only = argv[3] if len(argv) > 3 else ""
    probes = probe_list(folder, only)
    if not probes:
        out(b"no saved probe")
        return 0
    if not os.path.isdir(tree):
        out(b"no tree to probe: " + os.fsencode(tree))
        return 0
    sha = tree_sha(tree)
    detail = bool(os.environ.get("PROBE_DETAIL"))
    limit = os.environ.get("PROBE_TIMEOUT") or "1200"
    # The slot's environment (worker/pr_env.py's list), the contract's names on top (probe_env).
    env = dict(os.environ, **probe_env(tree))
    for k in CLEAN:
        env.pop(k, None)
    for p in probes:
        t0 = seconds()
        argv_probe = [
            "env",
            "-u",
            "HIP_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES=",
            "timeout",
            limit,
            "nice",
            "-n",
            "10",
            "bash",
            p,
            tree,
        ]
        r = subprocess.run(
            argv_probe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        rc = r.returncode if r.returncode >= 0 else 128 - r.returncode
        text = r.stdout.replace(b"\0", b"").rstrip(b"\n")
        v = {0: b"FIXED", 1: b"REPRODUCES", 5: b"ERROR rc=5 (no test ran)"}.get(
            rc, b"ERROR rc=%d" % rc
        )
        shown = echoed(text)  # last=$(echo "$out" | tail -1)
        last = shown.split(b"\n")[-1] if shown else b""
        # An uncaught Python exception exits 1 too: a probe that crashed (a missing module, a moved file) never reached
        # its summary, and must not read as REPRODUCES. A deliberate assert still reads as REPRODUCES.
        if (
            rc == 1
            and TRACEBACK.search(text)
            and EXCEPTION.search(last)
            and not last.startswith(b"AssertionError")
        ):
            v = b"ERROR rc=1 (crashed)"
        if detail:
            took = seconds() - t0
            if v.startswith(b"ERROR"):
                v = v + b" (tree " + sha + b", %d s)" % took
            else:
                v = v + b" rc=%d (tree " % rc + sha + b", %d s)" % took
        name = os.path.basename(p)
        if name.endswith(".sh") and name != ".sh":
            name = name[:-3]
        # The summary: the last line's first 200 bytes (a character of several bytes can be cut through), so a probe
        # whose last line is long (a printed array, a long assertion message) still reads as one short line where its
        # result is read: the review checks and the verifier.
        out(os.fsencode(name) + b": " + v + b": " + echoed(last)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
