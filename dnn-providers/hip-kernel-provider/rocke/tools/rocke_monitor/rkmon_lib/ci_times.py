# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI time per PR: how long CI takes on average per PR, since CI time blocks progress.

`refresh` (the tick, at most hourly) and `collect` record into `state/ci_runs.jsonl`, with GitHub reads only, one row
per PR head whose GitHub Actions runs have all completed, or that a newer push replaced: the tracked PRs' current heads,
and the heads PUSH events named in the last `days` days (10-character SHAs, expanded in the mirror or through GitHub's
commits API). A recorded head is never read again. A pass is capped in API calls and seconds, stops at the first
GitHub error, and the next pass carries on; a failure never fails the tick.

A head's CI is its pull_request and push workflow runs created within an hour of the first one. The bots on
pull_request_target (labeler, PR bot, base freshness) and Copilot reviews (dynamic) are not CI; a fork's runs waiting
for a maintainer's approval (action_required) never ran; a run created later (a label added days on) is a new request.
Per head:
- wall_s, the PR's "push → CI result" time: first run created → last run completed. It includes re-run attempts (the
  auto-retry), as the author waits for them. queue_s: first run created → first run started;
- outcome (success / failure / cancelled / mixed) of each workflow's newest run: a label added minutes after the push
  starts a new run and cancels the one before, so that cancel is not the head's result; and superseded: a newer push
  came before its CI finished.
The jobs (latest attempt) of the longest workflows of each head whose CI finished before a newer push name the slowest
jobs and the runner queue wait (job created → started) by runner label. A job a re-run copied from an earlier attempt
started before it was created: it counts for its duration only.
"""

from __future__ import annotations

import datetime as dt
import math
import statistics
import subprocess
import time
from collections import Counter

from . import gh, gitops, idle, registry
from .store import minutes_since, now

FILE = "ci_runs.jsonl"
UTC = dt.timezone.utc
CI_EVENTS = ("pull_request", "push", "merge_group")
WAITING = ("action_required",)
OK = {"success", "skipped", "neutral"}
BURST_S = 3600
EMPTY_RECHECK_MIN = (
    12 * 60
)  # a current head without CI (conflicting, or a fork waiting for approval): ask again later
GIVE_UP_MIN = (
    24 * 60
)  # a replaced head whose runs never finish is recorded as it stands
DAYS = 7  # the dashboard's window
DEFAULTS = registry.package_section("ci_times")
RUN_KEYS = (
    "id",
    "name",
    "path",
    "event",
    "status",
    "conclusion",
    "created_at",
    "run_started_at",
    "updated_at",
    "run_attempt",
)
RUNS_JQ = "[.workflow_runs[] | {" + ", ".join(RUN_KEYS) + "}]"
JOBS_JQ = "[.jobs[] | {name, status, conclusion, created_at, started_at, completed_at, labels}]"


def settings(cfg: dict) -> dict:
    return dict(DEFAULTS, **(cfg.get("ci_times") or {}))


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _secs(a: str, b: str) -> float:
    return (_ts(b) - _ts(a)).total_seconds()


def _z(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def outcome(concl: set, complete: bool) -> str:
    if not complete:
        return "incomplete"
    if concl <= OK:
        return "success"
    if "cancelled" in concl:
        return "mixed" if concl - OK - {"cancelled"} else "cancelled"
    return "failure"


def head_row(
    pr: int, sha: str, runs: list[dict], replaced_at: str | None = None
) -> dict | None:
    """One head's CI from its workflow runs (see the module docstring); None when no CI run exists."""
    ci = sorted(
        (
            r
            for r in runs
            if r.get("event") in CI_EVENTS and r.get("conclusion") not in WAITING
        ),
        key=lambda r: r["created_at"],
    )
    if not ci:
        return None
    first = ci[0]["created_at"]
    burst = [r for r in ci if _secs(first, r["created_at"]) <= BURST_S]
    complete = all(r["status"] == "completed" for r in burst)
    newest = (
        {}
    )  # a label added after the push starts a new run of a workflow and cancels the one before
    for r in burst:
        newest[r.get("path") or r["name"]] = r
    concl = {r["conclusion"] for r in newest.values() if r["status"] == "completed"}
    started = min(
        (r["run_started_at"] for r in burst if r.get("run_started_at")), default=None
    )
    last = max(r["updated_at"] for r in burst) if complete else None
    return {
        "ts": now(),
        "pr": pr,
        "head": sha,
        "first_created": first,
        "first_started": started,
        "last_completed": last,
        "wall_s": round(_secs(first, last)) if last else None,
        "queue_s": max(0, round(_secs(first, started))) if started else None,
        "outcome": outcome(concl, complete),
        "complete": complete,
        "superseded": bool(replaced_at)
        and (not complete or "cancelled" in concl or (last or "") > replaced_at),
        "reruns": sum(1 for r in burst if (r.get("run_attempt") or 1) > 1),
        "later": len(ci) - len(burst),
        "runs": [{k: r.get(k) for k in RUN_KEYS} for r in burst],
    }


def job_summary(jobs: list[dict], keep: int = 6) -> dict:
    """One run's latest attempt: its slowest jobs [name, runner label, duration s, queue wait s or None, conclusion] and
    the queue waits by runner label. Skipped jobs never ran."""
    slow, queue, n = [], {}, 0
    for j in jobs:
        if (
            j.get("conclusion") == "skipped"
            or not j.get("started_at")
            or not j.get("completed_at")
        ):
            continue
        n += 1
        label = (j.get("labels") or ["?"])[0]
        wait = (
            round(_secs(j["created_at"], j["started_at"]))
            if j.get("created_at")
            else None
        )
        if wait is not None and wait < 0:
            wait = None  # copied from an earlier attempt: its own wait is not in this attempt
        if wait is not None:
            queue.setdefault(label, []).append(wait)
        slow.append(
            [
                j["name"],
                label,
                round(_secs(j["started_at"], j["completed_at"])),
                wait,
                j.get("conclusion"),
            ]
        )
    slow.sort(key=lambda s: -s[2])
    return {"n": n, "slow": slow[:keep], "queue": queue}


def _run_wall(r: dict) -> float:
    return _secs(r["created_at"], r["updated_at"]) if r.get("updated_at") else 0.0


def add_jobs(cfg: dict, row: dict, budget: int) -> int:
    """Jobs of the head's longest workflows (`jobs_runs` of them, each at least `job_min_s` long), within `budget` API
    calls; returns the calls used. A failed read leaves that run without jobs."""
    c, used = settings(cfg), 0
    runs = sorted(
        (
            r
            for r in row["runs"]
            if r["status"] == "completed"
            and r["conclusion"] not in ("cancelled", "skipped")
        ),
        key=lambda r: -_run_wall(r),
    )
    for r in runs[: c["jobs_runs"]]:
        if _run_wall(r) < c["job_min_s"] or used >= budget:
            break
        try:  # per_page=50: GitHub answers 502 for the first 100 jobs of some TheRock runs (snapshot.develop_ci)
            jobs = gh.rest(
                f"repos/{cfg['repo']}/actions/runs/{r['id']}/jobs?per_page=50",
                paginate=True,
                jq=JOBS_JQ,
                timeout=60,
            )
        except gh.GhError:
            used += 1
            continue
        used += max(1, math.ceil(len(jobs) / 50))
        r["jobs"] = job_summary(jobs)
    return used


def candidates(
    st: dict, events: list[dict], start: dt.datetime, end: dt.datetime
) -> list[dict]:
    """The heads to read, the open PRs' current heads first, then newest first: {"pr", "sha" (full, or the events' 10
    characters), "current", "rollup", "replaced_at" (when a tick saw the PR move on from it), "at"}.
    """
    pushes = sorted(
        (
            e
            for e in events
            if e.get("kind") == "PUSH" and e.get("ts") and e.get("old") and e.get("new")
        ),
        key=lambda e: e["ts"],
    )
    replaced: dict[tuple, str] = {}
    for e in pushes:
        replaced.setdefault((e["pr"], e["old"][:10]), e["ts"])
    out: dict[tuple, dict] = {}
    for rec in st["prs"].values():
        head, live = rec.get("head"), rec.get("status") == "open"
        if (
            not head
            or (live and (rec.get("head_seen_at") or "") >= _z(end))
            or (
                not live
                and not (rec.get("gone_at") and start <= _ts(rec["gone_at"]) < end)
            )
        ):
            continue
        out[(rec["n"], head[:10])] = {
            "pr": rec["n"],
            "sha": head,
            "current": live,
            "rollup": rec.get("ci_rollup"),
            "replaced_at": None,
            "at": rec.get("head_seen_at") or rec.get("first_seen") or "",
        }
    for e in pushes:
        if start <= _ts(e["ts"]) < end:
            for sha in (e["new"], e["old"]):
                key = (e["pr"], sha[:10])
                out.setdefault(
                    key,
                    {
                        "pr": e["pr"],
                        "sha": sha,
                        "current": False,
                        "rollup": None,
                        "replaced_at": replaced.get(key),
                        "at": e["ts"],
                    },
                )
    return sorted(out.values(), key=lambda x: (x["current"], x["at"]), reverse=True)


def collect(
    cfg: dict,
    st: dict,
    store,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    max_calls: int | None = None,
    budget_s: float | None = None,
) -> dict:
    """Record the heads of [since, until) not recorded yet (default: the last `days` days), within `max_calls` API calls
    and `budget_s` seconds. Returns the counts; `left`: heads the caps left for the next pass.
    """
    c = settings(cfg)
    max_calls = c["max_calls"] if max_calls is None else max_calls
    stop = time.monotonic() + (c["budget_s"] if budget_s is None else budget_s)
    end = until or dt.datetime.now(UTC)
    start = since or end - dt.timedelta(days=c["days"])
    mem = st.setdefault("ci_times", {})
    empty = mem.setdefault("empty", {})
    for h in [
        h for h, at in empty.items() if minutes_since(at) > (c["days"] + 1) * 1440
    ]:
        del empty[h]
    have = {
        r["head"][:10] for r in store.read_jsonl(FILE, lenient=True) if r.get("head")
    }
    todo = [
        x
        for x in candidates(
            st, store.read_jsonl("events.jsonl", lenient=True), start, end
        )
        if x["sha"][:10] not in have
    ]
    repo = (cfg.get("paths") or {}).get("repo")
    shorts = [x["sha"] for x in todo if len(x["sha"]) < 40 and x["sha"] not in empty]
    try:
        full = gitops.expand(repo, shorts) if repo and shorts else {}
    except (OSError, subprocess.SubprocessError):
        full = {}
    out = {
        "recorded": 0,
        "pending": 0,
        "no_ci": 0,
        "calls": 0,
        "left": 0,
        "error": None,
    }
    for x in todo:
        if out["error"] or out["calls"] >= max_calls or time.monotonic() > stop:
            out["left"] += 1
            continue
        if x["current"] and (
            x["rollup"] in ("PENDING", "EXPECTED")
            or minutes_since(empty.get(x["sha"]) or "2000-01-01T00:00:00Z")
            < EMPTY_RECHECK_MIN
        ):
            out["pending"] += 1  # CI still running, or none to read yet
            continue
        sha = x["sha"] if len(x["sha"]) == 40 else full.get(x["sha"])
        try:
            if not sha:
                if x["sha"] in empty:
                    continue  # GitHub did not know it either
                out["calls"] += 1
                try:
                    sha = gh.rest(
                        f"repos/{cfg['repo']}/commits/{x['sha']}", jq=".sha", timeout=60
                    )
                except gh.GhError as exc:
                    if gh.transient(str(exc)):
                        raise
                    empty[x["sha"]] = now()
                    continue
            out["calls"] += 1
            runs = gh.rest(
                f"repos/{cfg['repo']}/actions/runs?head_sha={sha}&per_page=100",
                jq=RUNS_JQ,
                timeout=60,
            )
        except gh.GhError as exc:  # GitHub trouble: stop here, the next pass carries on
            out["error"] = str(exc)[:200]
            out["left"] += 1
            continue
        row = head_row(
            x["pr"], sha, runs if isinstance(runs, list) else [], x["replaced_at"]
        )
        if (
            row is None
        ):  # no CI run: a conflicting PR, a fork waiting for approval, a push replaced at once
            out["no_ci"] += 1
            if x["current"]:
                empty[x["sha"]] = now()
            else:  # recorded, so never read again
                store.append_jsonl(
                    FILE,
                    [
                        {
                            "ts": now(),
                            "pr": x["pr"],
                            "head": sha,
                            "outcome": "none",
                            "complete": False,
                            "superseded": True,
                            "runs": [],
                        }
                    ],
                )
            continue
        if not row["complete"] and (
            x["current"]
            or not x["replaced_at"]
            or minutes_since(x["replaced_at"]) < GIVE_UP_MIN
        ):
            out["pending"] += 1
            continue
        if row["complete"] and not row["superseded"]:
            out["calls"] += add_jobs(cfg, row, max_calls - out["calls"])
        store.append_jsonl(FILE, [row])
        out["recorded"] += 1
    return out


def refresh(cfg: dict, st: dict, store) -> dict | None:
    """The tick's pass, at most every `every_min` minutes (as merges.refresh); a failed pass waits for the next one."""
    mem = st.setdefault("ci_times", {})
    if mem.get("at") and minutes_since(mem["at"]) < settings(cfg)["every_min"]:
        return None
    mem["at"] = now()
    res = collect(cfg, st, store)
    mem["last"] = dict(res, at=mem["at"])
    return res


def weekly(cfg: dict, st: dict, store, start: dt.datetime, end: dt.datetime) -> dict:
    """The weekly report's view. The week's heads not recorded yet are read first (at most `weekly_calls` calls), so a
    week the hourly passes did not cover, such as the first, is complete as far as the cap allows.
    """
    note = None
    try:
        res = collect(
            cfg,
            st,
            store,
            since=start,
            until=end,
            max_calls=settings(cfg)["weekly_calls"],
            budget_s=300,
        )
        if res["left"]:
            note = f"{res['left']} heads of the week not read yet (the cap, or {res['error'] or 'time'})"
    except (gh.GhError, OSError) as exc:
        note = f"heads of the week not read: {str(exc)[:160]}"
    s = summary(store, start, end)
    s["note"] = note
    return s


def p90(xs: list[float]) -> float:
    """Nearest rank: always one of the values, never extrapolated from a few."""
    s = sorted(xs)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p90": p90(xs),
    }


def summary(store, start: dt.datetime, end: dt.datetime) -> dict:
    """CI time per PR head pushed in [start, end) (its first CI run created in the window). The times count every head
    whose CI ran to the end, nothing cancelled, also when a newer push came meanwhile: the slowest CI is the likeliest
    to be overtaken, so leaving those out would understate it. Cut short: heads a newer push replaced before their CI
    finished, or whose CI was cancelled. `runner_prs`: each PR's jobs that waited for a runner, longest in all first
    (the per-PR insights, insights.pr_statistics)."""
    rows, raw = {}, store.read_jsonl(FILE, lenient=True)
    for r in raw:
        if r.get("head"):
            rows[r["head"]] = r  # the last row of a head wins
    heads = [
        r
        for r in rows.values()
        if r.get("first_created") and start <= _ts(r["first_created"]) < end
    ]
    done = [
        r
        for r in heads
        if r.get("wall_s") is not None and r.get("outcome") in ("success", "failure")
    ]
    cut = [
        r
        for r in heads
        if r.get("superseded") or r.get("outcome") in ("cancelled", "mixed")
    ]
    per_pr: dict[int, dict] = {}
    for r in heads:
        p = per_pr.setdefault(
            r["pr"], {"pr": r["pr"], "walls": [], "heads": 0, "failed": 0}
        )
        p["heads"] += 1
        if r in done:
            p["walls"].append(r["wall_s"])
            p["failed"] += r["outcome"] == "failure"
    prs = sorted(
        (
            {
                "pr": p["pr"],
                "mean_s": statistics.mean(p["walls"]),
                "max_s": max(p["walls"]),
                "n": len(p["walls"]),
                "heads": p["heads"],
                "failed": p["failed"],
            }
            for p in per_pr.values()
            if p["walls"]
        ),
        key=lambda p: (-p["mean_s"], p["pr"]),
    )
    wf: dict[str, dict] = {}
    jobs: dict[str, list] = {}
    waits: list[float] = []
    labels: dict[str, list] = {}
    runner: dict[int, dict] = {}
    for r in heads:
        for run in r.get("runs") or []:
            if run.get("status") == "completed" and run.get("conclusion") in (
                "success",
                "failure",
                "timed_out",
            ):
                e = wf.setdefault(
                    run.get("path") or run["name"], {"names": Counter(), "walls": []}
                )
                e["names"][run["name"]] += 1
                e["walls"].append(_run_wall(run))
            j = run.get("jobs") or {}
            for name, label, dur, _, _ in j.get("slow") or []:
                jobs.setdefault(name, []).append(dur)
            for label, ws in (j.get("queue") or {}).items():
                waits += ws
                labels.setdefault(label, []).extend(ws)
                p = runner.setdefault(r["pr"], {"heads": set(), "waits": []})
                p["heads"].add(r["head"])
                p["waits"] += ws

    def wf_name(key: str, e: dict) -> str:
        name, n = e["names"].most_common(1)[0]
        return (
            name
            if n == sum(e["names"].values())
            else key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        )

    return {
        "heads": len(heads),
        "done": len(done),
        "cut": len(cut),
        "cut_share": len(cut) / len(heads) if heads else None,
        "overall": _stats([r["wall_s"] for r in done]),
        "queue": _stats([r["queue_s"] for r in done if r.get("queue_s") is not None]),
        "job_wait": _stats(waits),
        "reruns": sum(1 for r in done if r.get("reruns")),
        "outcomes": dict(Counter(r["outcome"] for r in done)),
        "prs": prs[:10],
        "workflows": sorted(
            (
                {
                    "name": wf_name(k, e),
                    "median_s": statistics.median(e["walls"]),
                    "p90_s": p90(e["walls"]),
                    "n": len(e["walls"]),
                }
                for k, e in wf.items()
            ),
            key=lambda w: (-w["p90_s"], w["name"]),
        )[:10],
        "jobs": sorted(
            (
                {"name": k, "median_s": statistics.median(v), "n": len(v)}
                for k, v in jobs.items()
            ),
            key=lambda j: (-j["median_s"], j["name"]),
        )[:5],
        "labels": sorted(
            (
                {
                    "label": k,
                    "median_s": statistics.median(v),
                    "p90_s": p90(v),
                    "n": len(v),
                    "total_s": sum(v),
                }
                for k, v in labels.items()
            ),
            key=lambda x: (-x["p90_s"], x["label"]),
        )[:5],
        "runner_prs": sorted(
            (
                {
                    "pr": pr,
                    "heads": len(p["heads"]),
                    "jobs": len(p["waits"]),
                    "total_s": sum(p["waits"]),
                    "max_s": max(p["waits"]),
                }
                for pr, p in runner.items()
                if p["waits"]
            ),
            key=lambda p: (-p["total_s"], p["pr"]),
        ),
        "since": raw[0].get("ts") if raw else None,
    }


def _cell(text: str) -> str:
    """Text for a markdown table cell (names come from GitHub)."""
    return " ".join(str(text).split()).replace("|", "\\|")


def hours(s: float) -> str:
    return f"{s / 3600:.1f} h"


def minutes(s: float) -> str:
    return f"{s / 60:.0f} min"


def dur(s: float) -> str:
    """Minutes under an hour, hours with one decimal from there."""
    return minutes(s) if s < 3600 else hours(s)


def headline(s: dict) -> str:
    o = s.get("overall")
    if not o:
        return "— (no finished CI recorded in the window)"
    return f"median {hours(o['median'])} · mean {hours(o['mean'])} · p90 {hours(o['p90'])} · n={o['n']} heads"


def render_md(s: dict, who=None, title: str = "## CI time per PR") -> list[str]:
    """The markdown section: push → result, runner queue wait, then per workflow, slowest PRs, slowest jobs and runner
    labels. `who(n)` labels a PR (default "#n")."""
    who = who or (lambda n: f"#{n}")
    lines = [
        title,
        "",
        "Push → CI result per PR head: from the first workflow run created to the last one completed "
        "(pull_request runs; bots, Copilot reviews and later re-triggers left out; re-run attempts included). "
        "The times count every head whose CI ran to the end, also when a newer push came meanwhile; a cancelled "
        "CI counts only in the cut-short share."
        + (f" Recorded since {s['since'][:10]}." if s.get("since") else ""),
        "",
    ]
    if s.get("note"):
        lines += [f"Note: {s['note']}.", ""]
    if not s.get("overall"):
        return lines + ["No finished CI recorded for these heads yet."]
    q, w = s.get("queue"), s.get("job_wait")
    lines += [
        "| | Median | Mean | p90 | n |",
        "|---|---|---|---|---|",
        f"| Push → CI result | {hours(s['overall']['median'])} | {hours(s['overall']['mean'])} | "
        f"{hours(s['overall']['p90'])} | {s['overall']['n']} heads |",
    ]
    if (
        q and q["p90"] >= 60
    ):  # whole runs wait only for a fork's approval or a concurrency group
        lines.append(
            f"| Run queued (first run created → started) | {minutes(q['median'])} | {minutes(q['mean'])} | "
            f"{minutes(q['p90'])} | {q['n']} heads |"
        )
    if w:
        lines.append(
            f"| Runner queue wait (jobs of the longest workflows) | {minutes(w['median'])} | "
            f"{minutes(w['mean'])} | {minutes(w['p90'])} | {w['n']} jobs |"
        )
    out = ", ".join(f"{k} {v}" for k, v in sorted(s["outcomes"].items()))
    lines += [
        "",
        f"- Cut short (a newer push, or a cancel, before CI finished): {s['cut']} of {s['heads']} heads "
        f"({100 * s['cut_share']:.0f}%). Finished: {out}; {s['reruns']} of them needed a re-run attempt.",
        "",
    ]
    lines += [
        "### By workflow",
        "",
        "| Workflow | Median | p90 | Runs |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {_cell(x['name'])} | {dur(x['median_s'])} | {dur(x['p90_s'])} | {x['n']} |"
        for x in s["workflows"]
    ]
    lines += [
        "",
        "### Slowest PRs (mean push → result)",
        "",
        "| PR | Mean | Longest | Heads (finished / all) | Failed |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {who(p['pr'])} | {dur(p['mean_s'])} | {dur(p['max_s'])} | {p['n']} / {p['heads']} | {p['failed']} |"
        for p in s["prs"]
    ]
    if s["jobs"]:
        lines += [
            "",
            "### Slowest jobs (median duration)",
            "",
            "| Job | Median | Runs |",
            "|---|---|---|",
        ]
        lines += [
            f"| {_cell(j['name'][:110])} | {minutes(j['median_s'])} | {j['n']} |"
            for j in s["jobs"]
        ]
    if s["labels"]:
        lines += [
            "",
            "### Runner queue wait by runner label",
            "",
            "| Runner label | Median | p90 | Jobs |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {_cell(x['label'])} | {minutes(x['median_s'])} | {minutes(x['p90_s'])} | {x['n']} |"
            for x in s["labels"]
        ]
    return lines


@idle.timer("CI times")
def _refresh_due(cfg: dict, st: dict, store):
    """When the tick's CI-time pass runs again (`every_min`)."""
    at = (st.get("ci_times") or {}).get("at")
    if at:
        yield "CI times", idle.plus(at, settings(cfg)["every_min"])
