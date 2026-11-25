# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk shard-times N`: whether a PR's CTest shards take longer because of the PR or because of the runner pool.

A shard stopped at its Test step limit with no failed test is the pool's unless the PR's own entries add the time, and
pools drift week to week, so entry times are compared, not pool names. For each shard job at PR N's live head (its
name holds `shard k/m`; those that failed or timed out, or every one when none did, or with `every`):
- its runner pool: the job's runs-on labels (else its runner group), and its time;
- its CTest entries' times, from the job log's `k/m Test #i: <name> .... <state> <s> sec` lines, and the entries it
  started and never finished (the step cut short);
- develop's runs of the same job (the same name) on the same pool, completed within WINDOW_H hours of the PR's run:
  the newest DEVELOP_RUNS of them, from the base branch's last DEVELOP_COMMITS commits;
- the time the PR adds to the shard: the entries develop's runs never ran (whole), plus each shared entry's time over
  develop's slowest run of it when that is more than NOISE_S seconds and NOISE_SHARE of it.
The verdict: `PR-added +a..b s per shard` when a compared shard gains ADDED_MIN_S seconds or more (a..b over the
compared shards), else `pool (develop same pool m–n min)` (develop's job times), else `not compared` (no develop run of
a shard on its pool in the window). GETs only, under the read login.
"""

import datetime as dt
import re

from . import ci_runs, gh, known_infra

WINDOW_H = 24  # develop's runs this close to the PR's run, before or after
DEVELOP_COMMITS = 15  # the base branch's newest commits read for runs
DEVELOP_RUNS = 3  # develop runs compared per shard, newest first
NOISE_S, NOISE_SHARE = 5.0, 0.2  # a shared entry's slower time counts above both
ADDED_MIN_S = 30.0  # a shard this much slower is the PR's
SHARD = re.compile(r"\bshard\s+\d+\s*/\s*\d+", re.I)
TIMESTAMP = known_infra.TIMESTAMP
CTEST_DONE = re.compile(
    r"\d+/\d+\s+Test\s+#\s*\d+:\s+(\S+)\s+\.*\s*(?:\*\*\*)?([A-Za-z][\w :]*?)\s+([\d.]+)\s+sec\b"
)
CTEST_START = re.compile(r"^\s*Start\s+\d+:\s+(\S+)\s*$")


def entry_times(text: str) -> tuple[dict[str, float], list[str]]:
    """A job log's CTest entries: {name: seconds} for each finished one, and the names started and never finished."""
    done: dict[str, float] = {}
    started: list[str] = []
    for raw in text.split("\n"):
        ln = TIMESTAMP.sub("", raw).rstrip("\r")
        m = CTEST_DONE.search(ln)
        if m:
            done[m.group(1)] = max(done.get(m.group(1), 0.0), float(m.group(3)))
            continue
        m = CTEST_START.match(ln)
        if m and m.group(1) not in started:
            started.append(m.group(1))
    return done, [s for s in started if s not in done]


def _when(stamp: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat((stamp or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def minutes(job: dict) -> float | None:
    a, b = _when(job.get("started_at") or ""), _when(job.get("completed_at") or "")
    return (b - a).total_seconds() / 60 if a and b else None


def pool(job: dict) -> str:
    """A job's runner pool: its runs-on labels, else its runner group; "" when GitHub names neither."""
    labels = sorted(str(x) for x in (job.get("labels") or []))
    return ",".join(labels) or str(job.get("runner_group_name") or "")


def added_time(
    pr: dict[str, float], develop: list[dict[str, float]]
) -> tuple[float, list[str]]:
    """The seconds the PR adds to a shard against develop's runs of it, and the entries that add the most."""
    parts: list[tuple[float, str]] = []
    for name, secs in pr.items():
        seen = [d[name] for d in develop if name in d]
        if not seen:
            parts.append((secs, f"{name} new {secs:.0f} s"))
            continue
        over = secs - max(seen)
        if over > NOISE_S and over > NOISE_SHARE * max(seen):
            parts.append((over, f"{name} +{over:.0f} s"))
    parts.sort(reverse=True)
    return sum(p[0] for p in parts), [p[1] for p in parts[:4]]


class Reader:
    """The GitHub reads, once each (tests replace gh)."""

    def __init__(self, cfg: dict):
        self.cfg, self.repo = cfg, cfg["repo"]
        self._runs: dict[str, list[dict]] = {}
        self._jobs: dict[int, dict] = {}
        self._logs: dict[int, str | None] = {}

    def pr(self, n: int) -> tuple[str, str]:
        d = gh.rest(
            f"repos/{self.repo}/pulls/{n}", jq="{head: .head.sha, base: .base.ref}"
        )
        return d["head"], d["base"]

    def runs(self, sha: str) -> list[dict]:
        if sha not in self._runs:
            self._runs[sha] = ci_runs.check_runs(self.cfg, sha)
        return self._runs[sha]

    def job(self, job: int) -> dict:
        if job not in self._jobs:
            self._jobs[job] = gh.rest(
                f"repos/{self.repo}/actions/jobs/{job}",
                jq="{labels, runner_name, runner_group_name, started_at, completed_at}",
            )
        return self._jobs[job]

    def log(self, job: int) -> str | None:
        if job not in self._logs:
            try:
                self._logs[job] = gh.raw(f"repos/{self.repo}/actions/jobs/{job}/logs")
            except gh.GhError:
                self._logs[job] = None
        return self._logs[job]

    def develop(self, base: str) -> list[str]:
        """The base branch's newest commits, newest first."""
        return gh.rest(
            f"repos/{self.repo}/commits?sha={base}&per_page={DEVELOP_COMMITS}",
            jq="[.[].sha]",
        )


def _job_id(run: dict) -> int | None:
    m = re.search(r"/job/(\d+)", run.get("url") or "")
    return int(m.group(1)) if m else None


def shard(reader: Reader, name: str, run: dict, base_shas: list[str]) -> dict:
    """One shard at the head against develop's runs of it on the same pool."""
    out = {"name": name, "conclusion": run.get("conclusion") or run.get("status")}
    jid = _job_id(run)
    if jid is None:
        return dict(out, error="no job id")
    try:
        job = reader.job(jid)
    except gh.GhError as exc:
        return dict(out, error=f"job not read ({str(exc)[:80]})")
    text = reader.log(jid)
    if text is None:
        return dict(out, error="log not read")
    out["pool"], out["min"] = pool(job), minutes(job)
    out["entries"], out["unfinished"] = entry_times(text)
    at = _when(job.get("completed_at") or "") or _when(job.get("started_at") or "")
    dev: list[tuple[dict, dict[str, float]]] = []
    for sha in base_shas:
        if len(dev) >= DEVELOP_RUNS:
            break
        try:
            r = ci_runs.latest(reader.runs(sha)).get(name)
        except gh.GhError:
            continue
        if not r or ci_runs.running(r) or _job_id(r) is None:
            continue
        try:
            dj = reader.job(_job_id(r))
        except gh.GhError:
            continue
        when = _when(dj.get("completed_at") or "")
        if out["pool"] and pool(dj) != out["pool"]:
            continue
        if at and when and abs((when - at).total_seconds()) > WINDOW_H * 3600:
            continue
        dtext = reader.log(_job_id(r))
        if dtext is None:
            continue
        times, _ = entry_times(dtext)
        if times:
            dev.append((dj, times))
    out["develop_min"] = [m for m in (minutes(dj) for dj, _ in dev) if m is not None]
    out["develop_runs"] = len(dev)
    if dev:
        out["added"], out["top"] = added_time(out["entries"], [t for _, t in dev])
    return out


def report(
    cfg: dict, n: int, every: bool = False, reader: Reader | None = None
) -> tuple[list[str], bool]:
    """The lines `rk shard-times N` prints, and whether the reads went through."""
    reader = reader or Reader(cfg)
    try:
        head, base = reader.pr(n)
        last = ci_runs.latest(reader.runs(head))
    except gh.GhError as exc:
        return [f"(not read: {str(exc)[:160]})"], False
    shards = {
        k: r for k, r in last.items() if SHARD.search(k) and not ci_runs.running(r)
    }
    if not shards:
        return [
            f"head {head[:10]}: no finished shard job (`shard k/m` in its name)"
        ], True
    picked = {k: r for k, r in shards.items() if ci_runs.failed(r)}
    which = "failed or timed out"
    if every or not picked:
        picked, which = shards, "every shard"
    try:
        base_shas = reader.develop(base)
    except gh.GhError as exc:
        base_shas = []
        lines_err = f"{base}'s commits not read ({str(exc)[:80]})"
    else:
        lines_err = ""
    lines = [f"head {head[:10]} base {base}: {len(picked)} shard job(s), {which}"]
    if lines_err:
        lines.append(lines_err)
    compared, dev_min = [], []
    for name in sorted(picked):
        s = shard(reader, name, picked[name], base_shas)
        if "error" in s:
            lines.append(f"{name}: {s['conclusion']}; {s['error']}")
            continue
        total = sum(s["entries"].values())
        mins = f"{s['min']:.0f} min" if s["min"] is not None else "time unknown"
        lines.append(
            f"{name}: {s['conclusion']}, {mins}, pool {s['pool'] or '(unknown)'}; "
            f"{len(s['entries'])} entries {total:.0f} s"
            + (
                f"; unfinished at the cut: {', '.join(s['unfinished'][:4])}"
                if s["unfinished"]
                else ""
            )
        )
        if not s["develop_runs"]:
            lines.append(
                f"   develop: no run of this shard on its pool within {WINDOW_H} h: not compared"
            )
            continue
        dev_min += s["develop_min"]
        compared.append(s["added"])
        rng = _range(s["develop_min"])
        lines.append(
            f"   develop: {s['develop_runs']} run(s) on the same pool{f', {rng} min' if rng else ''}; "
            f"PR-added +{s['added']:.0f} s"
            + (f" ({'; '.join(s['top'])})" if s["top"] else "")
        )
    if not compared:
        lines.append(
            f"VERDICT not compared: no develop run of a shard on its pool within {WINDOW_H} h"
        )
    elif max(compared) >= ADDED_MIN_S:
        lines.append(
            f"VERDICT PR-added +{min(compared):.0f}..{max(compared):.0f} s per shard"
        )
    else:
        rng = _range(dev_min)
        lines.append(f"VERDICT pool (develop same pool {rng or '?'} min)")
    return lines, True


def _range(values: list[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    return f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}"
