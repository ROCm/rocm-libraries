# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI at a PR head as the orchestrator reads it around a post: `rk ci-fails N` and `rk job-log ID`, and the
latest-attempt picture `rk post --premises` checks (premises.py).

The latest attempt of a check run is the one of its name that started last; a re-run that has not started yet does not
replace a started attempt. The post scripts counted that way, and the verifier writes the spec's FAILOK and RUNOK from
`rk ci-fails`, so both read one picture (lessons L-224). GETs only.
"""

from __future__ import annotations

import os
import pathlib
import re
import tempfile
import time

from . import gh

FAILED = ("failure", "timed_out", "action_required")
LOG_HITS = re.compile(
    r"\*\*\*Failed|FAILED|Error|error:|tests failed|Timeout|timed out"
)
LOG_HITS_SHOWN, LOG_LINE_CHARS = 25, 260


def check_runs(cfg: dict, sha: str) -> list[dict]:
    """Every check run at `sha`, every attempt, in GitHub's order: name, status, conclusion ("-" while it has none),
    started ("" before it starts), completed ("" before it ends) and url."""
    return gh.rest(
        f"repos/{cfg['repo']}/commits/{sha}/check-runs?per_page=100",
        paginate=True,
        jq='[.check_runs[] | {name, status, conclusion: (.conclusion // "-"), started: (.started_at // ""), '
        'completed: (.completed_at // ""), url: (.html_url // "")}]',
    )


def latest(runs: list[dict]) -> dict[str, dict]:
    """The latest attempt of each check run, by name, in the order the names first appear."""
    out: dict[str, dict] = {}
    for r in runs:
        if r["name"] not in out or r["started"] > out[r["name"]]["started"]:
            out[r["name"]] = r
    return out


def failed(run: dict) -> bool:
    return run["status"] == "completed" and run["conclusion"] in FAILED


def running(run: dict) -> bool:
    """Queued or in progress."""
    return run["status"] != "completed"


def fails(cfg: dict, n: int) -> tuple[list[str], bool]:
    """`rk ci-fails N`: PR n's live head, then by name a RUN line (name, status, url) for each latest attempt queued or
    running and a FAIL line (name, conclusion, url) for each that failed, tab-separated, then the count of every attempt
    and of the names. The FAIL names are what a spec's FAILOK accounts for, the RUN names its RUNOK.
    """
    try:
        sha = gh.rest(f"repos/{cfg['repo']}/pulls/{n}", jq=".head.sha")
    except gh.GhError as exc:
        return [f"(head not read: {exc})"], False
    lines = [f"head {sha}"]
    try:
        runs = check_runs(cfg, sha)
    except gh.GhError as exc:
        return lines + [f"(check runs not read: {exc})"], False
    last = latest(runs)
    for name in sorted(last):
        r = last[name]
        if running(r):
            lines.append(f"RUN\t{name}\t{r['status']}\t{r['url']}")
        if failed(r):
            lines.append(f"FAIL\t{name}\t{r['conclusion']}\t{r['url']}")
    return lines + [f"runs {len(runs)} names {len(last)}"], True


# A job's whole log as `rk job-log` keeps it in the temporary folder, by the job's ID.
JOB_LOG = "joblog_{}.txt"
# The hours a log kept that way stays: its readers (the agents' commands, first_pass.py) read it within one review.
JOB_LOG_KEEP_H = 24


def job_log_dir() -> pathlib.Path:
    """The folder `rk job-log` keeps logs in: TMPDIR when it is set, even while that folder is missing (job_log makes
    it; tempfile.gettempdir would pass it over for the system's temporary folder), else the system's temporary
    folder.
    """
    return pathlib.Path(os.environ.get("TMPDIR") or tempfile.gettempdir())


def job_log_path(job: int) -> pathlib.Path:
    """Where `rk job-log` keeps a job's whole log, to read around a failing line: joblog_<ID>.txt in job_log_dir().
    Each time it keeps one, it removes the ones kept there more than JOB_LOG_KEEP_H hours ago (prune_job_logs).
    """
    return job_log_dir() / JOB_LOG.format(job)


def prune_job_logs(keep_h: float = JOB_LOG_KEEP_H) -> None:
    """Remove the logs `rk job-log` kept (job_log_path) more than KEEP_H hours ago: this user's only, as the temporary
    folder may be shared."""
    cut = time.time() - keep_h * 3600
    for p in job_log_dir().glob(JOB_LOG.format("*")):
        try:
            st = p.lstat()
            if st.st_mtime < cut and st.st_uid == os.getuid():
                p.unlink()
        except OSError:
            pass


ERROR_LINE = re.compile(r"\berror:")
LOG_TIMESTAMP = re.compile(r"^\ufeff?\d{4}-\d\d-\d\dT[\d:.]+Z\s?")


def log_errors(text: str, files: list[str] | None = None) -> list[str]:
    """`rk job-log ID --errors`: every distinct `error:` line of a log, in log order, as `<N>:<the line>` (timestamps
    cut; "error: 0" left out), then, for each changed C/C++ source in `files`, `never compiled: <file>` when no
    `/<name>.o` object of it appears in the log (the build stopped before it, or it is in no target).
    """
    out, seen = [], set()
    lines = text.split("\n")
    for i, line in enumerate(lines, 1):
        s = LOG_TIMESTAMP.sub("", line).rstrip("\r")
        if ERROR_LINE.search(s) and "error: 0" not in s and s not in seen:
            seen.add(s)
            out.append(f"{i}:{s}"[:LOG_LINE_CHARS])
    head = [f"{len(out)} distinct error lines"]
    for f in files or []:
        name = f.rsplit("/", 1)[-1]
        if name.endswith((".cpp", ".cc", ".cxx", ".c")) and not re.search(
            rf"/{re.escape(name)}\.o\b", text
        ):
            out.append(f"never compiled: {f}")
    return head + out


def job_log(
    cfg: dict, job: int, errors: bool = False, files=None
) -> tuple[list[str], bool]:
    """`rk job-log ID`: a CI job's log line count, then its failing lines as `<N>:<the line>` (at most 25, each cut to
    260 characters; "error: 0" lines left out). The whole log is kept at job_log_path(ID); a log it cannot keep
    there is said, as one it cannot read. `errors`: log_errors instead (every distinct error line, and the changed
    sources in `files` the log never compiled).
    """
    try:
        text = gh.raw(f"repos/{cfg['repo']}/actions/jobs/{job}/logs")
    except gh.GhError as exc:
        return [f"(log not read: {exc})"], False
    kept = job_log_path(job)
    try:
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(text, encoding="utf-8")
    except OSError as exc:
        return [f"(log not kept: {exc})"], False
    prune_job_logs()
    if errors:
        return log_errors(text, files), True
    hits = [
        f"{i}:{line}"[:LOG_LINE_CHARS]
        for i, line in enumerate(text.split("\n"), 1)
        if LOG_HITS.search(line) and "error: 0" not in line
    ]
    return [str(text.count("\n"))] + hits[:LOG_HITS_SHOWN], True
