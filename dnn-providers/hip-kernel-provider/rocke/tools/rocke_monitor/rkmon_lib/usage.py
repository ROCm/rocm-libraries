# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk usage`: what the monitor's models spent, in tokens only (no prices and no cost here; the token records stay;
`rk spend` prices the Claude transcripts).

Three records feed it:
- state/claude_usage.jsonl (rkmon_lib/orchestrator.py): the Claude session's turns (input, output, cache read, cache
  written; what started each, and the PRs it audited), its subagents (the PR and round each brief names, from the Agent
  tool's PreToolUse hook; each task's total from its notification; the input / output / cache split from the subagent's
  transcript when it stops), and the commands the session's guard refused;
- <RUN_DIR>/tokens.jsonl (worker/reviewer_client.py): every request to the reviewer model, by who asked (a parallel
  review, pr<N>, or the tick triage), its prompt and answer tokens;
- the ledger: the reviews we posted.

`daily` is the table per UTC day (rk usage, the dashboard's tile). `by_review` gives each review we posted its tokens:
a subagent, a turn or a reviewer-model request that names PR N counts toward our next post on N after it started (one
not posted yet shows as pending), and a turn that audited several PRs is split evenly among them.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib

from . import paths

KINDS = ("input", "output", "cache_read", "cache_write")
NOT_TURNS = ("task", "task_start", "agent", "subagent", "refused")
TRIAGE = "tick"  # who the tick triage's requests are from (rkmon_lib/triage.py)


def _jsonl(path: pathlib.Path, since: str = "") -> list[dict]:
    """The records at or after `since`, the rotated part (`<name>.1`, retention.py) first."""
    from .retention import with_older

    out, text = [], ""
    for p in with_older(pathlib.Path(path)):
        try:
            text += p.read_text(encoding="utf-8") + "\n"
        except OSError:
            continue
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and isinstance(r.get("ts"), str) and r["ts"] >= since:
            out.append(r)
    return out


def _since(days: int) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).strftime(
        "%Y-%m-%d"
    )


def claude_rows(cfg: dict, since: str = "") -> list[dict]:
    from .config import toolkit_state_dir
    from .orchestrator import USAGE

    return _jsonl(toolkit_state_dir(cfg) / USAGE, since)


def model_rows(cfg: dict, since: str = "") -> list[dict]:
    from . import models

    try:
        run = paths.get(cfg, "run")
    except paths.PathsError:
        return []
    return _jsonl(pathlib.Path(run) / models.client().TOKENS_LOG, since) if run else []


def turn_tokens(r: dict) -> int:
    return sum(int(r.get(k) or 0) for k in KINDS)


def is_turn(r: dict) -> bool:
    return r.get("kind") not in NOT_TURNS


def subagents(rows: list[dict]) -> list[dict]:
    """Each subagent run: {ts (dispatched, else finished), done (finished), pr, round, type, tokens, split}. Its tokens
    are the transcript's split when the SubagentStop hook read it, else the total its task notification reported.
    """
    runs: dict[str, dict] = {}
    by_task: dict[str, str] = {}
    for r in rows:
        k = r.get("kind")
        if k == "agent" and r.get("tool_use_id"):
            runs.setdefault(r["tool_use_id"], {}).update(
                ts=r.get("ts"),
                pr=r.get("pr"),
                round=r.get("round"),
                type=r.get("subagent_type") or "",
            )
        elif k == "task_start" and r.get("task_id"):
            by_task[r["task_id"]] = r.get("tool_use_id") or r["task_id"]
    for r in rows:
        k = r.get("kind")
        if k == "task":
            key = (
                r.get("tool_use_id")
                or by_task.get(r.get("task_id"))
                or r.get("task_id")
                or ""
            )
            run = runs.setdefault(key, {})
            run.update(total=int(r.get("total_tokens") or 0), done=r.get("ts"))
        elif k == "subagent":
            key = (
                (r.get("tool_use_id") if r.get("tool_use_id") in runs else None)
                or by_task.get(r.get("agent_id"))
                or r.get("agent_id")
                or ""
            )
            run = runs.setdefault(key, {})
            run.update(
                split={x: int(r.get(x) or 0) for x in KINDS},
                done=run.get("done") or r.get("ts"),
            )
            run.setdefault("type", r.get("agent_type") or "")
    out = []
    for run in runs.values():
        if "total" not in run and "split" not in run:
            continue  # dispatched, not finished yet
        tokens = sum(run["split"].values()) if run.get("split") else run.get("total", 0)
        out.append(
            {
                "ts": run.get("ts") or run.get("done") or "",
                "done": run.get("done") or run.get("ts") or "",
                "pr": run.get("pr"),
                "round": run.get("round"),
                "type": run.get("type") or "subagent",
                "tokens": tokens,
                "split": run.get("split"),
            }
        )
    return out


def posts(store) -> list[dict]:
    """Our review posts, oldest first: {pr, ts, round} (round from their order on the PR: first round, then
    follow-up 1, 2, ...)."""
    seen: dict[int, int] = {}
    out = []
    for r in store.ledger():
        if r.get("kind") != "post":
            continue
        k = seen[r["pr"]] = seen.get(r["pr"], 0) + 1
        out.append(
            {
                "pr": r["pr"],
                "ts": r["ts"],
                "round": "first round" if k == 1 else f"follow-up {k - 1}",
            }
        )
    return out


def daily(cfg: dict, store, days: int = 7) -> list[dict]:
    """Per UTC day with anything recorded, oldest first: the session's turns (by what started them) and their tokens by
    kind, the subagents' tokens, Claude's total, the reviews posted, the reviewer model's prompt and answer tokens and
    requests (`model_roles`: those that are not a parallel review's, by who asked: the tick triage, the planner, ...),
    and the commands the session's guard refused."""
    since = _since(days)
    every = claude_rows(cfg)
    rows, model = [r for r in every if r["ts"] >= since], model_rows(cfg, since)
    sub = subagents(every)  # a run dispatched before the window can finish in it
    posted = [p for p in posts(store) if p["ts"][:10] >= since]
    days_seen = sorted(
        {r["ts"][:10] for r in rows + model}
        | {s["done"][:10] for s in sub if s["done"][:10] >= since}
        | {p["ts"][:10] for p in posted}
    )
    out = []
    for day in days_seen:
        turns = [r for r in rows if r["ts"][:10] == day and is_turn(r)]
        kinds: dict[str, int] = {}
        for r in turns:
            kinds[r.get("kind") or "turn"] = kinds.get(r.get("kind") or "turn", 0) + 1
        mine = [m for m in model if m["ts"][:10] == day]
        roles: dict[str, list[int]] = (
            {}
        )  # the requests that are not a parallel review's, by who asked
        for m in mine:
            who = str(m.get("who") or "-")
            if _pr_of(who) is None and who != "-":
                r = roles.setdefault(who, [0, 0])
                r[0] += int(m.get("prompt") or 0)
                r[1] += int(m.get("answer") or 0)
        d = {
            "day": day,
            "turns": len(turns),
            "turn_kinds": kinds,
            **{k: sum(int(r.get(k) or 0) for r in turns) for k in KINDS},
            "subagents": sum(s["tokens"] for s in sub if s["done"][:10] == day),
            "posted": sum(1 for p in posted if p["ts"][:10] == day),
            "model_prompt": sum(int(m.get("prompt") or 0) for m in mine),
            "model_answer": sum(int(m.get("answer") or 0) for m in mine),
            "model_requests": len(mine),
            "model_roles": roles,
            "refused": sum(
                1 for r in rows if r["ts"][:10] == day and r.get("kind") == "refused"
            ),
        }
        d["claude"] = sum(d[k] for k in KINDS) + d["subagents"]
        out.append(d)
    return out


def report(cfg: dict, store, days: int = 7) -> list[str]:
    """`rk usage`: one line per UTC day."""
    out = [
        f"Tokens per UTC day, the last {days} day(s): Claude's session (turns; input / output / cache read / cache "
        "written) and its subagents, the reviews posted, the reviewer model (prompt / answer)"
    ]
    for d in daily(cfg, store, days):
        kinds = ", ".join(f"{n} {k}" for k, n in sorted(d["turn_kinds"].items()))
        line = (
            f"{d['day']}: Claude {d['turns']} turns"
            + (f" ({kinds})" if kinds else "")
            + f"; {d['input']} / {d['output']} / {d['cache_read']} / {d['cache_write']}; subagents {d['subagents']}"
            + f"; {d['posted']} review(s) posted"
        )
        if d["posted"]:
            line += f", about {d['claude'] // d['posted']} Claude tokens a review"
        if d["model_requests"]:
            line += (
                f"; reviewer model {d['model_prompt']} / {d['model_answer']} ({d['model_requests']} requests"
                + "".join(
                    f"; {'the tick triage' if who == TRIAGE else who} {p} / {a}"
                    for who, (p, a) in sorted(d["model_roles"].items())
                )
                + ")"
            )
        if d["refused"]:
            line += f"; {d['refused']} command(s) refused by the session's guard"
        out.append(line)
    if len(out) == 1:
        out.append(
            "none recorded yet (rk orchestrate records the session's turns, the reviewer model its requests)"
        )
    return out


def _pr_of(who) -> int | None:
    w = str(who or "")
    return int(w[2:]) if w.startswith("pr") and w[2:].isdigit() else None


def _start(r: dict) -> str:
    """When a turn started: its finish less its duration."""
    try:
        t = dt.datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ") - dt.timedelta(
            milliseconds=int(r.get("duration_ms") or 0)
        )
    except (KeyError, ValueError):
        return r.get("ts") or ""
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def by_review(cfg: dict, store, days: int = 7) -> list[dict]:
    """Per review: {pr, round, posted (None: not posted yet), session, subagents ({type: tokens}), claude,
    model_prompt, model_answer, model_requests}, for the reviews posted in the window and those still pending.
    """
    since = _since(days)
    every = claude_rows(cfg)
    per_pr: dict[int, list[dict]] = {}
    for p in posts(store):
        per_pr.setdefault(p["pr"], []).append(p)
    rows: dict[tuple, dict] = {}

    def slot(pr, ts: str) -> dict | None:
        if pr is None:
            return None
        p = next((p for p in per_pr.get(pr, []) if p["ts"] >= ts), None)
        key = (pr, p["ts"] if p else None)
        if key not in rows:
            rows[key] = {
                "pr": pr,
                "round": p["round"] if p else "",
                "posted": p["ts"] if p else None,
                "session": 0,
                "subagents": {},
                "model_prompt": 0,
                "model_answer": 0,
                "model_requests": 0,
            }
        return rows[key]

    for r in every:
        prs = [int(n) for n in r.get("prs") or [] if str(n).isdigit()]
        if is_turn(r) and prs and r["ts"] >= since:
            share = turn_tokens(r) // len(prs)
            for n in prs:
                slot(n, _start(r))["session"] += share
    for s in subagents(every):
        if s["done"][:10] >= since and str(s.get("pr") or "").isdigit():
            row = slot(int(s["pr"]), s["ts"])
            row["subagents"][s["type"]] = (
                row["subagents"].get(s["type"], 0) + s["tokens"]
            )
    for m in model_rows(cfg, since):
        row = slot(_pr_of(m.get("who")), m["ts"])
        if row:
            row["model_prompt"] += int(m.get("prompt") or 0)
            row["model_answer"] += int(m.get("answer") or 0)
            row["model_requests"] += 1
    out = [r for r in rows.values() if r["posted"] is None or r["posted"][:10] >= since]
    for r in out:
        r["claude"] = r["session"] + sum(r["subagents"].values())
    return sorted(out, key=lambda r: (r["posted"] or "9999", r["pr"]))


def by_review_report(cfg: dict, store, days: int = 7) -> list[str]:
    """`rk usage --by-review`: one line per review."""
    out = [
        f"Tokens per review, the last {days} day(s) (UTC): Claude's (the session's turns that audited it, split "
        "evenly when a turn audited several, and the subagents its briefs named) and the reviewer model's (prompt / "
        "answer)"
    ]
    for r in by_review(cfg, store, days):
        when = (
            f"{r['round']}, posted {r['posted'][:16].replace('T', ' ')}Z"
            if r["posted"]
            else "not posted yet"
        )
        subs = ", ".join(f"{k} {v}" for k, v in sorted(r["subagents"].items()))
        out.append(
            f"#{r['pr']} {when}: Claude {r['claude']} (session {r['session']}; subagents "
            f"{sum(r['subagents'].values())}"
            + (f": {subs}" if subs else "")
            + ")"
            + (
                f"; reviewer model {r['model_prompt']} / {r['model_answer']} ({r['model_requests']} requests)"
                if r["model_requests"]
                else ""
            )
        )
    if len(out) == 1:
        out.append("none recorded yet")
    return out


def tile(cfg: dict, store, days: int = 7) -> dict:
    """The dashboard's tile: per UTC day of the window (oldest first) Claude's tokens, the reviews posted and the
    reviewer model's tokens; their totals, and Claude's tokens per review."""
    by_day = {d["day"]: d for d in daily(cfg, store, days)}
    today = dt.datetime.now(dt.timezone.utc).date()
    series = []
    for i in range(days - 1, -1, -1):
        day = (today - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        d = by_day.get(day) or {}
        series.append(
            (
                day,
                d.get("claude", 0),
                d.get("posted", 0),
                d.get("model_prompt", 0) + d.get("model_answer", 0),
            )
        )
    claude, posted, model = (sum(x[k] for x in series) for k in (1, 2, 3))
    return {
        "days": series,
        "claude": claude,
        "posted": posted,
        "model": model,
        "per_review": claude // posted if posted else None,
    }
