# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Weekly statistics for the user, presented every week.

Sources:
- GitHub, read-only: PRs opened, merged and closed unmerged in the week, and our reviews among our account's
  (snapshot.account_reviews).
- `state/events.jsonl`: every event the monitor saw.
- `state/metrics.jsonl`: one snapshot per tick.
- Incidents (`state/incidents.json`).
- `state/conflicts.jsonl` and `state/ci_runs.jsonl`: the files PRs conflict on most
  (conflicts.py) and CI time per PR (ci_times.py; the week's heads not read yet are read first, capped).
- Review spend (spend.py): the week's section, from the cached transcripts and the post ledger.

Anything the monitor only tracks since it started (conflicts, friction, develop time red) says so in the report.
Weeks are ISO weeks in UTC, Monday 00:00 to Monday 00:00.
"""

from __future__ import annotations

import datetime as dt
import re
import statistics

from . import gh
from . import config, registry, reviewer_name, snapshot
from .snapshot import parse_verdict

UTC = dt.timezone.utc
_SEARCH = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 50, after: $cursor) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number title createdAt mergedAt closedAt additions deletions isDraft headRefOid author { login } } }
  }
}"""


def week_bounds(
    label: str | None = None, now: dt.datetime | None = None
) -> tuple[dt.datetime, dt.datetime, str]:
    """(start, end, 'YYYY-Www'); without a label, the last complete ISO week."""
    if label:
        y, w = re.fullmatch(r"(\d{4})-W(\d{2})", label).groups()
        start = dt.datetime.fromisocalendar(int(y), int(w), 1).replace(tzinfo=UTC)
    else:
        now = now or dt.datetime.now(UTC)
        this_monday = (now - dt.timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start = this_monday - dt.timedelta(days=7)
    end = start + dt.timedelta(days=7)
    y, w, _ = start.isocalendar()
    return start, end, f"{y}-W{w:02d}"


CATCH_UP_WEEKS = (
    8  # the most weeks a late tick writes at once after the monitor was down
)


def owed_weeks(
    done: str | None, now: dt.datetime, due: bool, limit: int = CATCH_UP_WEEKS
) -> list[str]:
    """The weekly reports to write now, oldest first: every complete ISO week after `done` (the last one written), up
    to the last complete week when this week's report time has passed (`due`), else up to the week before it. A week
    missed while the monitor was down is written late, never skipped. Never written before (`done` None): the last
    complete week only, once due."""
    last_start = week_bounds(None, now)[0]
    newest = last_start if due else last_start - dt.timedelta(days=7)
    if done is None:
        return [week_bounds(None, now)[2]] if due else []
    out = []
    start = newest
    while len(out) < limit:
        lab = week_bounds(None, start + dt.timedelta(days=7))[2]
        if lab <= done:
            break
        out.append(lab)
        start -= dt.timedelta(days=7)
    return sorted(out)


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _search(q: str) -> list[dict]:
    out, cursor = [], None
    while True:
        data = gh.graphql(_SEARCH, q=q, cursor=cursor)["search"]
        out += [n for n in data["nodes"] if n]
        if not data["pageInfo"]["hasNextPage"]:
            return out
        cursor = data["pageInfo"]["endCursor"]


def _scoped(cfg: dict, st: dict, extra: str) -> list[dict]:
    """The search, run for every label in scope (config.Scope.searched) and merged: a PR counts when the label it was
    found by puts it in scope (snapshot.in_scope: a shared label, when the PR changes files in that label's folder).
    """
    from .snapshot import in_scope

    seen = {}
    for label in config.scope(cfg).searched():
        for p in _search(f'repo:{cfg["repo"]} is:pr label:"{label}" {extra}'):
            if p["number"] in seen:
                continue
            pr = {
                "n": p["number"],
                "head": p.get("headRefOid") or "?",
                "labels": [label],
            }
            if in_scope(cfg, pr, st.setdefault("scope_cache", {})):
                seen[p["number"]] = p
    return list(seen.values())


def _days(a: dt.datetime, b: dt.datetime) -> float:
    return (b - a).total_seconds() / 86400


def _summ(xs: list[float]) -> str:
    if not xs:
        return "—"
    return f"median {statistics.median(xs):.1f} d · mean {statistics.mean(xs):.1f} d · n={len(xs)}"


def our_reviews(cfg: dict, prs: list[int], end: dt.datetime) -> dict[int, list[dict]]:
    """Every review we posted on each PR, up to `end`: of our account's reviews (snapshot.account_reviews, comments and
    formal reviews), this deployment's own, by its line 1 and the time it was posted (reviewer_name.own_review), never
    another deployment's under the same account.
    """
    out = {}
    for n in prs:
        posts = [
            r
            for r in snapshot.account_reviews(cfg, n)
            if reviewer_name.own_review(cfg, r.get("body"), r.get("at"))
            and _ts(r["at"]) < end
        ]
        out[n] = [
            {
                "at": r["at"],
                "verdict": parse_verdict(r["body"]),
                "follow_up": i > 0 or "follow-up" in r["body"][:400].lower(),
            }
            for i, r in enumerate(posts)
        ]
    return out


def compute(cfg: dict, st: dict, store, start: dt.datetime, end: dt.datetime) -> dict:
    repo = cfg["repo"]
    rng = f"{start:%Y-%m-%dT%H:%M:%SZ}..{end - dt.timedelta(seconds=1):%Y-%m-%dT%H:%M:%SZ}"
    merged = _scoped(cfg, st, "is:merged merged:" + rng)
    opened = _scoped(cfg, st, "created:" + rng)
    closed = _scoped(cfg, st, "is:unmerged is:closed closed:" + rng)
    # Any PR we commented on that was still active at the week's start; our comments are filtered by date below.
    touched = _scoped(
        cfg,
        st,
        f"commenter:{config.operator_login(cfg)} updated:>={start:%Y-%m-%dT%H:%M:%SZ}",
    )
    reviews = our_reviews(cfg, sorted({p["number"] for p in touched}), end)
    in_week = {
        n: [r for r in rs if start <= _ts(r["at"]) < end] for n, rs in reviews.items()
    }
    posted = [r for rs in in_week.values() for r in rs]
    first_latency = []
    created = {
        p["number"]: _ts(p["createdAt"]) for p in merged + opened + closed + touched
    }
    for n, rs in reviews.items():
        if rs and start <= _ts(rs[0]["at"]) < end and n in created:
            first_latency.append(_days(created[n], _ts(rs[0]["at"])))

    events = [
        e for e in store.read_jsonl("events.jsonl") if start <= _ts(e["ts"]) < end
    ]
    metrics = [
        m for m in store.read_jsonl("metrics.jsonl") if start <= _ts(m["ts"]) < end
    ]
    tracked_since = (store.read_jsonl("events.jsonl") or [{"ts": st.get("created")}])[
        0
    ]["ts"]
    conflicted = sorted({e["pr"] for e in events if e["kind"] == "CONFLICT"})

    # Friction, as signals that cost other people time:
    # (a) a merged PR followed within the window by other PRs turning CONFLICTING (likely forced rebases);
    # (b) ABI renumbers it forced on others;
    # (c) its own review rounds (3 or more);
    # (d) repeated CI failures (2 or more in the week).
    win = dt.timedelta(hours=registry.value(cfg, "weekly_report.friction_window_h"))
    merges = sorted((e for e in events if e["kind"] == "MERGED"), key=lambda e: e["ts"])
    friction: dict[int, list[str]] = {}
    for e in events:
        if e["kind"] != "CONFLICT":
            continue
        before = [
            m
            for m in merges
            if _ts(m["ts"]) <= _ts(e["ts"]) <= _ts(m["ts"]) + win and m["pr"] != e["pr"]
        ]
        if before:
            friction.setdefault(before[-1]["pr"], []).append(
                f"likely forced a rebase of #{e['pr']}"
            )
    for e in events:
        if e["kind"] == "ABI_RENUMBER_NEEDED":
            friction.setdefault(e["pr"], []).append("ABI renumber needed")
    for n, rs in reviews.items():
        if len(rs) >= 4 and in_week.get(n):
            friction.setdefault(n, []).append(f"{len(rs)} review rounds")
    ci_fail: dict[int, int] = {}
    for e in events:
        if e["kind"] == "CI" and e.get("state") == "FAILURE":
            ci_fail[e["pr"]] = ci_fail.get(e["pr"], 0) + 1
    for n, k in ci_fail.items():
        if k >= 2:
            friction.setdefault(n, []).append(f"CI failed {k}×")

    incidents = [
        i
        for i in store_incidents(store)
        if not i.get("rehearsal")
        and not i.get("void")
        and _ts(i["start"]) < end
        and _ts(i.get("end") or f"{end:%Y-%m-%dT%H:%M:%SZ}") >= start
    ]
    for i in incidents:
        if i.get("culprit"):
            friction.setdefault(i["culprit"], []).insert(0, "broke develop")
        for n in i.get("contributing", []):
            friction.setdefault(n, []).insert(0, "contributed to a develop break")
    red_hours = 0.0
    for i in incidents:
        s0, e0 = max(_ts(i["start"]), start), min(
            _ts(i.get("end") or f"{end:%Y-%m-%dT%H:%M:%SZ}"), end
        )
        red_hours += max(0.0, (e0 - s0).total_seconds() / 3600)

    now = min(end, dt.datetime.now(UTC))
    open_ages = [
        _days(_ts(r["created_at"]), now)
        for r in st["prs"].values()
        if r.get("status") == "open" and not r["draft"] and _ts(r["created_at"]) < end
    ]
    ttm = [_days(_ts(p["createdAt"]), _ts(p["mergedAt"])) for p in merged]
    authors = {}
    for p in st["prs"].values():
        authors[p["n"]] = p["author"]
    for p in merged + opened + closed + touched:
        authors.setdefault(p["number"], (p.get("author") or {}).get("login", "?"))
    for (
        i
    ) in (
        incidents
    ):  # PRs outside the label scope (fixes, contributors) still get an author
        for n in [i.get("culprit"), i.get("fixed_by"), *i.get("contributing", [])]:
            if n and n not in authors:
                try:
                    authors[n] = gh.rest(f"repos/{repo}/pulls/{n}", jq=".user.login")
                except gh.GhError:
                    authors[n] = "?"
    verdicts = {}
    for r in posted:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    tick_states = {}
    for m in metrics:
        tick_states[m.get("develop_ci", "?")] = (
            tick_states.get(m.get("develop_ci", "?"), 0) + 1
        )
    # The files PRs conflict on most, and how long CI takes per PR.
    from . import ci_times, conflicts
    from . import spend

    spend_md = spend.weekly_section(
        cfg, store, spend.week_of(start.strftime("%Y-%m-%d"))
    )
    hot = conflicts.weekly(cfg, st, store, start, end)
    ci = ci_times.weekly(cfg, st, store, start, end)
    # Where the time goes, per PR: the week's rebases, conflicts, review and fix
    # rounds and runner waits, as the dashboard's insights show them for the last 14 days.
    from . import insights

    try:
        per_pr = insights.pr_statistics(
            cfg,
            st,
            store,
            days=max(1, round((end - start).total_seconds() / 86400)),
            end=end,
        )
    except Exception as exc:  # noqa: BLE001 - the report never breaks on the statistics
        per_pr = {"error": str(exc)[:200]}
    for p in ci["prs"]:
        authors.setdefault(
            p["pr"], (st["prs"].get(str(p["pr"])) or {}).get("author", "?")
        )
    return {
        "conflict_hot": hot,
        "ci": ci,
        "per_pr": per_pr,
        "conflict_hot": hot,
        "ci": ci,
        "spend": spend_md,
        "week": None,
        "start": start,
        "end": end,
        "tracked_since": tracked_since,
        "merged": merged,
        "opened": opened,
        "closed": closed,
        "reviews": {
            "posts": len(posted),
            "prs": sorted(n for n, rs in in_week.items() if rs),
            "first_rounds": sum(1 for r in posted if not r["follow_up"]),
            "follow_ups": sum(1 for r in posted if r["follow_up"]),
            "verdicts": verdicts,
            "first_latency": first_latency,
        },
        "conflicted": conflicted,
        "friction": friction,
        "incidents": incidents,
        "red_hours": red_hours,
        "open_ages": open_ages,
        "ttm": ttm,
        "authors": authors,
        "ticks": len(metrics),
        "tick_states": tick_states,
        "max_conflicting": max(
            (m.get("conflicting", 0) for m in metrics), default=None
        ),
    }


def store_incidents(store) -> list[dict]:
    from .guard import incidents  # one reader: it recovers an empty or unreadable file

    return incidents(store)


def render(s: dict, label: str) -> str:
    from . import ci_times, conflicts

    a = s["authors"]
    who = lambda n: f"#{n} ({a.get(n, '?')})"
    start, end = s["start"], s["end"]
    hot, ci = s.get("conflict_hot") or {}, s.get("ci") or {}
    lines = [
        f"# rocKE weekly report — {label} ({start:%Y-%m-%d} → {end - dt.timedelta(days=1):%Y-%m-%d}, UTC)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| PRs opened | {len(s['opened'])} |",
        f"| PRs merged | {len(s['merged'])} |",
        f"| PRs closed unmerged | {len(s['closed'])} |",
        f"| Reviews posted (first rounds / follow-ups) | {s['reviews']['posts']} "
        f"({s['reviews']['first_rounds']} / {s['reviews']['follow_ups']}) on {len(s['reviews']['prs'])} PRs |",
        f"| Verdict mix | "
        + (
            ", ".join(f"{k} {v}" for k, v in sorted(s["reviews"]["verdicts"].items()))
            or "—"
        )
        + " |",
        f"| PRs that conflicted with develop | {len(s['conflicted'])} |",
        f"| Top merge-conflict file | {conflicts.top_line(hot)} |",
        f"| CI push → result | {ci_times.headline(ci)} |",
        f"| Develop breaks (incidents) | {len(s['incidents'])} · develop red ≈ {s['red_hours']:.1f} h |",
        f"| PRs with friction | {len(s['friction'])} |",
        f"| Open PR age (non-draft) | {_summ(s['open_ages'])} |",
        f"| Time to merge (merged this week) | {_summ(s['ttm'])} |",
        f"| Time to our first review | {_summ(s['reviews']['first_latency'])} |",
        "",
    ]
    lines += ["## PRs that broke develop", ""]
    if s["incidents"]:
        lines += [
            "| When | Culprit | Contributing | Fixed by | Red for | What |",
            "|---|---|---|---|---|---|",
        ]
        for i in s["incidents"]:
            dur = (
                f"{(_ts(i['end']) - _ts(i['start'])).total_seconds() / 3600:.1f} h"
                if i.get("end")
                else "ongoing"
            )
            lines.append(
                f"| {i['start'][:16]}Z | {who(i['culprit']) if i.get('culprit') else 'not established'} | "
                + (", ".join(who(n) for n in i.get("contributing", [])) or "—")
                + f" | {who(i['fixed_by']) if i.get('fixed_by') else '—'} | {dur} | {i.get('summary', '')} |"
            )
    else:
        lines.append("None.")
    lines += ["", "## Friction (signals that cost other people time)", ""]
    if s["friction"]:
        lines += ["| PR | Signals |", "|---|---|"]
        for n, why in sorted(s["friction"].items(), key=lambda kv: -len(kv[1])):
            lines.append(f"| {who(n)} | {'; '.join(why)} |")
    else:
        lines.append("None recorded.")
    lines += ["", "## Merged", ""] + (
        [
            f"- {who(p['number'])} {p['title'][:90]} — "
            f"{_days(_ts(p['createdAt']), _ts(p['mergedAt'])):.1f} d open"
            for p in sorted(s["merged"], key=lambda p: p["mergedAt"])
        ]
        or ["None."]
    )
    lines += [
        "",
        "## Conflicted with develop during the week",
        "",
        ", ".join(who(n) for n in s["conflicted"]) or "None recorded.",
    ]
    if hot:
        lines += [""] + conflicts.render_md(hot)
    if ci:
        lines += [""] + ci_times.render_md(ci, who=who)
    per_pr = s.get("per_pr") or {}
    if per_pr.get("categories"):
        from . import insights

        lines += [""] + insights.stats_md(
            per_pr, labels=False
        )  # the CI section above has the runner labels
    elif per_pr.get("error"):
        lines += ["", f"(Per-PR statistics not computed: {per_pr['error']})"]
    if s.get("spend"):
        lines += [""] + s["spend"]
    lines += [
        "",
        "## Monitor coverage",
        "",
        f"- Events tracked since {s['tracked_since']}. Conflicts, friction and develop time-red come from the "
        "monitor's own history and incident records, so earlier weeks are incomplete.",
        f"- Merge-conflict files: {conflicts.tracked(hot)}. CI times: "
        + (
            f"heads recorded since {ci['since'][:10]} (hourly)."
            if ci.get("since")
            else "none recorded yet."
        ),
        f"- Ticks this week: {s['ticks']} (develop CI as seen: "
        + (", ".join(f"{k} {v}" for k, v in sorted(s["tick_states"].items())) or "—")
        + ")",
        "- Friction heuristics: a merge followed within 6 h by another PR turning CONFLICTING counts as a likely "
        "forced rebase; breaking develop or contributing to a break; 4+ review rounds; 2+ CI failures in the "
        "week; ABI renumbers.",
        "",
    ]
    return "\n".join(lines)
