# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI at a PR head, per failing job: what the verifier and the lane read job logs by hand for (34 of 39 reviews;
agent_script_overlap.md, fix 1).

For each check run whose latest attempt failed at the head (ci_runs.latest / failed, as `rk ci-fails` counts them):
- its attempt ("attempt 2 of 2: a re-run" when the head has more than one run of that name);
- its log's first error line (known_infra.first_error);
- the failing tests its log names: pytest's `FAILED` lines, CTest's "The following tests FAILED" list and its
  `***Failed` / `***Timeout` lines (at most TESTS_SHOWN);
- develop's latest result for the same job name: the newest of the given develop commits with a completed run of it,
  its conclusion and, when it failed, whether its first error is the same;
- a known-infrastructure label, as known_infra.classify reads one: a known signature (known_infra.signatures: config
  `ci_infra_signatures` and the state's `site.ci_infra_signatures`, each with its job scope, `unless` text and ending
  develop commit) in the log, with its note, or develop's latest run of the job failing with the same first
  error. Anything else is unlabelled: the PR's, or not known.
Then the failing tests by name: each test the failing jobs' logs name, with the jobs (shards) that fail it, the most jobs
first (at most TESTS_BY_NAME). A job re-run at the head and not failing at it is listed too ("re-run: JOB (attempt k
of k): STATE"). GETs only; a read that fails says so on its line.

Usage: python -I ci_head.py SHA [--develop SHA ...]
Prints the lines, then one last line: "CIHEAD F failing, K known infrastructure, R re-run", F, K and R the counts of
the failing jobs, of those labelled known infrastructure, and of the jobs re-run. When the check runs at SHA are not
read, the first line says why (the start of gh's error) and the last is "CIHEAD not read" (NOT_READ). Exit 1 when a
GitHub read failed (the check runs at SHA or at a develop commit, or a job's log: its line says "not read", and gh's
whole error is on stderr), so the checks run it once more when it failed on the network (rkmon_lib/review.py,
NET_FN); else 0.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
from rkmon_lib import applied, ci_runs, config, gh, known_infra, store  # noqa: E402

JOBS_SHOWN, TESTS_SHOWN, TESTS_BY_NAME = 12, 6, 20
# The last line when the check runs at the head were not read.
NOT_READ = "CIHEAD not read"
PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+::\S+|\S+\.py)")
CTEST_LIST = re.compile(
    r"^\s*\d+ - (\S+) \((Failed|Timeout|Not Run|Exception|SEGFAULT|Subprocess aborted)\)"
)
ANSI = re.compile(
    r"\x1b\[[0-9;]*[A-Za-z]|\[[0-9;]*m"
)  # a log's colour codes, with or without their ESC
CTEST_STAR = re.compile(
    r"Test\s+#\d+: (\S+) \.+\*\*\*(Failed|Timeout|Exception|Not Run)"
)


def first_error(text: str) -> str:
    """known_infra.first_error without the log's colour codes."""
    return ANSI.sub("", known_infra.first_error(text)).strip()


def failing_tests(text: str) -> list[str]:
    """The failing tests a job log names, in order, once each: pytest ids, then CTest names."""
    out = []
    for raw in text.split("\n"):
        ln = known_infra.TIMESTAMP.sub("", raw).rstrip("\r").strip()
        for pat in (PYTEST_FAILED, CTEST_LIST, CTEST_STAR):
            m = pat.search(ln) if pat is CTEST_STAR else pat.match(ln)
            if m and m.group(1) not in out:
                out.append(m.group(1))
    return out


class Reader:
    """The GitHub reads, once each (tests replace it). A read that fails is not tried again: gh's whole error goes to
    stderr, where the checks' `net` reads a network failure, and `failed` counts it."""

    def __init__(self, cfg: dict):
        self.cfg, self._runs, self._logs, self.failed = cfg, {}, {}, 0

    def _not_read(self, exc: gh.GhError) -> None:
        sys.stderr.write(f"{exc}\n")
        self.failed += 1

    def runs(self, sha: str) -> list[dict]:
        """The check runs at SHA; gh.GhError, each time they are asked for, when they were not read."""
        if sha not in self._runs:
            try:
                self._runs[sha] = ci_runs.check_runs(self.cfg, sha)
            except gh.GhError as exc:
                self._not_read(exc)
                self._runs[sha] = exc
        if isinstance(self._runs[sha], gh.GhError):
            raise self._runs[sha]
        return self._runs[sha]

    def log(self, run: dict) -> str | None:
        """The log of the run's job; None when the run names no job, or its log was not read."""
        job = known_infra.job_id(run)
        if job is None:
            return None
        if job not in self._logs:
            try:
                self._logs[job] = gh.raw(
                    f"repos/{self.cfg['repo']}/actions/jobs/{job}/logs"
                )
            except gh.GhError as exc:
                self._not_read(exc)
                self._logs[job] = None
        return self._logs[job]


def develop_result(
    reader: Reader, name: str, develop: list[str], first: str
) -> tuple[str, bool]:
    """Develop's latest completed run of job `name` among `develop` (newest first): its line, and whether it fails
    with the same first error."""
    for sha in develop:
        try:
            last = ci_runs.latest(reader.runs(sha)).get(name)
        except gh.GhError as exc:
            return f"develop {sha[:10]}: not read ({str(exc)[:80]})", False
        if not last or ci_runs.running(last):
            continue
        if not ci_runs.failed(last):
            return f"develop {sha[:10]}: {last['conclusion']}", False
        dfirst = first_error(reader.log(last) or "")
        same = bool(first) and dfirst == first
        return (
            f"develop {sha[:10]}: {last['conclusion']}, "
            + (
                "the same first error"
                if same
                else f"another first error: {dfirst[:120] or '(none read)'}"
            ),
            same,
        )
    return (
        f"develop: no completed run of this job in its last {len(develop)} commits",
        False,
    )


def short_job(name: str) -> str:
    """A job's name without its workflow and stage path: its last ` / ` part (`Test hipkernelprovider (shard 2/2)
    (gfx94X-dcgpu)`)."""
    return name.rsplit(" / ", 1)[-1]


def by_test_lines(by_test: dict[str, list[str]]) -> list[str]:
    """The failing tests by name, each with the failing jobs (shards) whose logs name it, the most jobs first: a test
    the PR does not change shows here as plainly as its own, never only in the shard logs.
    """
    if not by_test:
        return []
    order = sorted(by_test, key=lambda t: (-len(by_test[t]), t))
    out = [f"failing tests by name ({len(order)}, from the failing jobs' logs):"]
    for t in order[:TESTS_BY_NAME]:
        jobs = by_test[t]
        out.append(
            f"   {t}: {len(jobs)} job{'s' if len(jobs) != 1 else ''}: {'; '.join(map(short_job, jobs[:4]))}"
            + (f" (+{len(jobs) - 4} more)" if len(jobs) > 4 else "")
        )
    if len(order) > TESTS_BY_NAME:
        out.append(f"   (+{len(order) - TESTS_BY_NAME} more failing tests)")
    return out


def report(
    cfg: dict, sha: str, develop: list[str], reader: Reader | None = None
) -> list[str]:
    reader = reader or Reader(cfg)
    try:
        runs = reader.runs(sha)
    except gh.GhError as exc:
        return [
            f"NOT RUN: the check runs at {sha[:10]} were not read ({str(exc)[:120]})",
            NOT_READ,
        ]
    last = ci_runs.latest(runs)
    attempts = {name: sum(1 for r in runs if r["name"] == name) for name in last}
    sigs = known_infra.signatures(cfg)
    ended = known_infra.Ancestry(cfg, sha)  # read only for a signature with ends_at
    out, known, failing = [], 0, sorted(n for n, r in last.items() if ci_runs.failed(r))
    by_test: dict[str, list[str]] = (
        {}
    )  # each failing test, and the failing jobs whose logs name it
    for name in failing[:JOBS_SHOWN]:
        r, k = last[name], attempts[name]
        text = reader.log(r)
        tag = f" (attempt {k} of {k}: a re-run)" if k > 1 else ""
        if text is None:
            out += [f"FAILED {name}{tag}: log not read ({r['url']})"]
            continue
        first = first_error(text)
        out.append(f"FAILED {name}{tag}: {first or '(no error line read)'}")
        tests = failing_tests(text)
        for x in tests:
            by_test.setdefault(x, []).append(name)
        if tests:
            out.append(
                f"   failing tests ({len(tests)}): {', '.join(tests[:TESTS_SHOWN])}"
                + (
                    f" (+{len(tests) - TESTS_SHOWN} more)"
                    if len(tests) > TESTS_SHOWN
                    else ""
                )
            )
        dev, same = (
            develop_result(reader, name, develop, first)
            if develop
            else ("develop: not given", False)
        )
        out.append(f"   {dev}")
        sig = known_infra.match(sigs, name, text, ended)
        label = (
            f"known infrastructure (signature `{sig['pattern']}`"
            + (f": {sig['note']}" if sig["note"] else "")
            + ")"
            if sig
            else (
                "known infrastructure (develop fails the same)"
                if same
                else "not known infrastructure"
            )
        )
        known += label.startswith("known")
        out.append(f"   {label}; job {r['url']}")
    if len(failing) > JOBS_SHOWN:
        out.append(f"(+{len(failing) - JOBS_SHOWN} more failing jobs not read)")
    out += by_test_lines(by_test)
    reruns = sorted(
        n for n, r in last.items() if attempts[n] > 1 and not ci_runs.failed(r)
    )
    for name in reruns[:JOBS_SHOWN]:
        r = last[name]
        out.append(
            f"re-run: {name} (attempt {attempts[name]} of {attempts[name]}): "
            f"{r['status'] if ci_runs.running(r) else r['conclusion']}"
        )
    rerun = len(reruns) + sum(1 for n in failing if attempts[n] > 1)
    return out + [
        f"CIHEAD {len(failing)} failing, {known} known infrastructure, {rerun} re-run"
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sha")
    ap.add_argument(
        "--develop",
        nargs="*",
        default=[],
        help="develop's newest commits, newest first",
    )
    a = ap.parse_args(argv)
    try:  # the monitor home's configuration (RKMON_HOME), as rk reads it
        cfg = store.load_config()
    except config.ConfigError as exc:
        return applied.refuse(exc)
    gh.configure(cfg)  # the gh binary and its login's folder, as rk sets them
    reader = Reader(cfg)
    print("\n".join(report(cfg, a.sha, a.develop, reader)))
    return 1 if reader.failed else 0


if __name__ == "__main__":
    sys.exit(main())
