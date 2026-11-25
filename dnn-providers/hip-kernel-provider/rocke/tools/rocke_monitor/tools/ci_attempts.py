# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Jobs that failed in an earlier attempt of a workflow run at a PR head.

A re-run of failed jobs (by hand, or TheRock's auto re-run) replaces them in the check-run view and in a run's default
job list (`filter=latest`), so "CI at head" sees only the re-run, and a PR's own failures in attempt 1 can hide behind
a hand re-run that shows an infra failure (lessons L-108). GET-only.

Usage: python -I ci_attempts.py SHA OWNER/REPO. One line per earlier-attempt failure, with the job's state in the
latest attempt; a job that failed earlier and passed only in a later attempt is tagged `FLAKE` (it passed on a
re-run: record it, never read it as the PR's pass or failure). Nothing when no run at this head was re-run. Exit 1
when a GitHub read failed (its line says "not read", gh's error is on stderr), so the checks run it once more when it
failed on the network (rkmon_lib/review.py, NET_FN); else 0.
"""
import json
import os
import subprocess
import sys
import time

# gh under the read login, reads only (tools/ghr.py).
GHR = [
    sys.executable,
    "-I",
    "-B",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "ghr.py"),
]

PAGE = 50  # GitHub answers 502 for the first 100 jobs of some TheRock runs (ci_sha.py)


def fetch(path: str) -> dict | None:
    """One GET through gh, retried once after 5 s. None when both fail, each one's error then on stderr (gh's whole
    error, where the checks' `net` reads a network failure). On an HTTP error gh prints the error body on stdout and
    exits 1."""
    for attempt in range(2):
        try:
            done = subprocess.run(
                [*GHR, "api", path],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            err = str(exc)
        else:
            if done.returncode == 0:
                try:
                    return json.loads(done.stdout)
                except ValueError:
                    err = f"gh api {path}: the answer is not JSON"
            else:
                err = (
                    done.stderr.strip()
                    or f"gh exited {done.returncode} with no message"
                )
        sys.stderr.write(err + "\n")
        if attempt == 0:
            time.sleep(5)
    return None


def all_jobs(repo: str, run_id: int, get=fetch) -> list[dict] | None:
    """Every job of every attempt of one run, or None when a page could not be read."""
    jobs, page = [], 1
    while True:
        d = get(
            f"repos/{repo}/actions/runs/{run_id}/jobs?filter=all&per_page={PAGE}&page={page}"
        )
        if d is None:
            return None
        batch = d.get("jobs") or []
        jobs += batch
        if len(batch) < PAGE or len(jobs) >= (d.get("total_count") or 0):
            return jobs
        page += 1


def earlier_failures(run: dict, jobs: list[dict]) -> list[str]:
    """The jobs that failed or timed out before the run's latest attempt, each with its state in the latest one."""
    latest = run.get("run_attempt") or 1
    now = {
        j.get("name"): j.get("conclusion") or j.get("status") or "?"
        for j in jobs
        if (j.get("run_attempt") or 1) == latest
    }
    out = []
    for j in sorted(
        jobs, key=lambda j: (j.get("run_attempt") or 1, j.get("name") or "")
    ):
        a = j.get("run_attempt") or 1
        if a < latest and j.get("conclusion") in ("failure", "timed_out"):
            state = now.get(j.get("name"), "absent (not re-run, or not started yet)")
            out.append(
                f"FAILED in attempt {a} of {latest} ({run.get('name') or '?'}): {j.get('name')}; "
                f"attempt {latest}: {state}"
                + ("; FLAKE: passed only on a re-run" if state == "success" else "")
            )
    return out


def main(argv: list[str], get=fetch) -> int:
    if len(argv) < 3:
        print("usage: python -I ci_attempts.py SHA OWNER/REPO")
        return 2
    sha, repo = argv[1], argv[2]
    runs = get(f"repos/{repo}/actions/runs?head_sha={sha}&per_page=100")
    if runs is None:
        print("-- earlier attempts: not read (GitHub API error)")
        return 1
    failed = False  # a GitHub read that failed: exit 1
    for run in runs.get("workflow_runs") or []:
        if (run.get("run_attempt") or 1) < 2:
            continue
        jobs = all_jobs(repo, run["id"], get)
        if jobs is None:
            print(
                f"-- run {run['id']} ({run.get('name')}): earlier attempts not read (GitHub API error)"
            )
            failed = True
            continue
        for line in earlier_failures(run, jobs):
            print(line)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
