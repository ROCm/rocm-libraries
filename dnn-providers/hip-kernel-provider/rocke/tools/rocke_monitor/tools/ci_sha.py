#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GET-only: for a commit SHA, list the runs of the watched workflows and grep the logs of their jobs that test the
watched folder for a pattern.

Usage: ci_sha.py SHA PATTERN
The repository, the scratch folder, the watched workflows (config develop_watched_workflows) and the jobs that test the
watched folder (config develop_rocke_job_pattern, matched without regard to case; blank: every job of those workflows)
come from the monitor home's configuration, as rk reads it: REFUSED, exit 4, when it cannot be read. PATTERN is a
regular expression (Python syntax, which covers the extended POSIX patterns the checks pass). Each run's job logs are
kept in <scratch>/ci_sha/job_<id>.log and read again from there. Exit 1 when a GitHub read failed (its line says "not
read" or "not available", gh's error is on stderr), so the checks run it once more when it failed on the network
(rkmon_lib/review.py, NET_FN); else 0.
"""

import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
from rkmon_lib import applied, config, paths, registry, store, whole_file  # noqa: E402

# gh under the read login, reads only (tools/ghr.py).
GHR = [
    sys.executable,
    "-I",
    "-B",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "ghr.py"),
]

LOWERER = b"fell back to the Python lowerer"
STAMP = re.compile(rb"^[0-9T:.Z-]+ ")
CTEST = re.compile(rb"ctest .*-L [^ ]+")


def gh() -> list[str]:
    return GHR


def gh_api(*args: str) -> bytes | None:
    """One `gh api ARGS`: its output, or None when gh fails, gh's error then on stderr (where the checks' `net` reads a
    network failure). On an HTTP error gh prints the error body (e.g. a 5xx "Server Error") on stdout and exits 1.
    """
    try:
        p = subprocess.run(
            [*gh(), "api", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except OSError as exc:
        err = str(exc)
    else:
        if p.returncode == 0:
            return p.stdout
        err = p.stderr.decode("utf-8", "replace").strip()
        err = err or f"gh exited {p.returncode} with no message"
    sys.stderr.write(err + "\n")
    return None


def ghget(*args: str) -> bytes | None:
    """`gh api ARGS` (gh_api), retried once after 5 s; None when both fail. An error body is never parsed as ids (jobs
    named `{` and `"message":`).
    """
    for _ in (1, 2):
        out = gh_api(*args)
        if out is not None:
            return out.replace(b"\0", b"").rstrip(b"\n")
        time.sleep(5)
    return None


def keep_log(url: str, log: str) -> bool:
    """The job log `gh api URL` gives (gh_api), kept as the file LOG, written whole (rkmon_lib/whole_file.py) at the
    mode of the files the monitor and the writer share (worker/perms.py). False when gh fails: nothing is kept then,
    nor when the file cannot be written."""
    data = gh_api(url)
    if data is None:
        return False
    try:
        whole_file.write(log, data, paths.worker_module("perms").SHARED_FILE_MODE)
    except OSError:
        pass
    return True


def say(line: bytes) -> None:
    sys.stdout.buffer.write(line + b"\n")
    sys.stdout.buffer.flush()


def nonempty(path: str) -> bool:
    """test -s: the path exists and is not empty."""
    try:
        return os.stat(path).st_size > 0
    except OSError:
        return False


def lines_of(data: bytes) -> list[bytes]:
    parts = data.split(b"\n")
    if parts and parts[-1] == b"":
        parts.pop()
    return parts


def main(argv: list[str]) -> int:
    try:  # the monitor home's configuration (RKMON_HOME), as rk reads it
        cfg = store.load_config()
    except config.ConfigError as exc:
        return applied.refuse(exc)
    head = argv[1] if len(argv) > 1 else ""
    pat = re.compile(os.fsencode(argv[2] if len(argv) > 2 else ""))
    watched = {os.fsencode(n) for n in registry.value(cfg, "develop_watched_workflows")}
    tests = re.compile(registry.value(cfg, "develop_rocke_job_pattern") or "", re.I)
    folder = f"{paths.get(cfg, 'scratch')}/ci_sha"
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f"mkdir: cannot create directory '{folder}': {exc.strerror}\n")
    r = f"repos/{cfg['repo']}"
    runs = ghget(
        f"{r}/actions/runs?head_sha={head}&per_page=30",
        "--jq",
        '.workflow_runs[] | [.id, .name, .status, (.conclusion // "-"), .created_at] | @tsv',
    )
    if runs is None:
        say(f"-- CI runs for {head}: not read (GitHub API error)".encode())
        return 1
    failed = False  # a GitHub read that failed: exit 1
    picked = []
    for line in lines_of(runs + b"\n"):
        f = line.split(b"\t")  # the id, the name, the status, the conclusion, the time
        if len(f) > 1 and f[1] in watched:
            say(line)
            if len(f) < 4 or f[3] != b"cancelled":
                picked.append(f[0])
    for run in picked:
        run_s = run.decode("utf-8", "surrogateescape")
        # per_page=50: GitHub answers 502 for the first 100 jobs of some large runs (147 jobs). filter=all: every
        # attempt, since a re-run hides the earlier attempt's failed shards (L-108).
        jobs = ghget(
            f"{r}/actions/runs/{run_s}/jobs?filter=all&per_page=50",
            "--paginate",
            "--jq",
            ".jobs[] | [.id, .name] | @tsv",
        )
        if jobs is None:
            say(f"-- run {run_s}: job list not read (GitHub API error)".encode())
            failed = True
            continue
        for row in lines_of(jobs + b"\n"):
            j, _, name = row.partition(b"\t")
            if not re.fullmatch(rb"[0-9]+", j) or not tests.search(
                name.decode("utf-8", "replace")
            ):
                continue
            js = j.decode()
            n = ghget(
                f"{r}/actions/jobs/{js}",
                "--jq",
                '.name + " [" + (.conclusion // .status) + ", attempt " + (.run_attempt | tostring) + "]"',
            )
            if n is None:
                n = b"(job details not read)"
                failed = True
            log = f"{folder}/job_{js}.log"
            if not nonempty(log) and not keep_log(f"{r}/actions/jobs/{js}/logs", log):
                failed = True
            try:
                with open(log, "rb") as fh:
                    data = fh.read()
            except OSError:
                data = b""
            if not data:
                say(b"-- job " + j + b": " + n + b" (log not available)")
                continue
            m = None
            for line in lines_of(data):
                m = CTEST.search(line)
                if m:
                    break
            label = m.group(0)[:80] if m else b""
            say(
                b"-- job "
                + j
                + b": "
                + n
                + b" (%d lines); ctest label: " % data.count(b"\n")
                + label
            )
            # the cpp->Python lowerer fallback warnings carry "BackendError:" and would crowd out the real failures
            shown = 0
            for line in lines_of(data):
                if shown == 8:
                    break
                if pat.search(line) and LOWERER not in line:
                    say(b"     " + STAMP.sub(b"", line, count=1)[:200])
                    shown += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
