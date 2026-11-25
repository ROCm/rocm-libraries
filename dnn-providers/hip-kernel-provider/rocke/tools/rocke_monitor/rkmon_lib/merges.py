# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocKE PRs merged into develop per UTC day, for
the dashboard's tile and chart and the board's merge-queue row.

Two sources, each PR counted once:
- GitHub, read-only: the weekly report's scoped search (`stats._scoped`: the monitor's rocKE scope rules), for the last
  `DAYS` days, at most hourly, into st["merges_daily"]. A PR's day is its mergedAt.
- The daily log's `merged` entries: the monitor's own record of each tracked PR merging, so a merge counts at once
  between two GitHub reads. A PR the GitHub read already has keeps GitHub's day.
Merges into another PR's branch (stacked PRs) are not merges into develop and are not counted: the daily log records
them as `stacked`, and older `merged` entries are checked against the PR's base. The dashboard's tile, chart and
"Merged today" stage all read `series`.
"""

import datetime as dt

from . import daylog, idle
from .store import minutes_since, now

DAYS = 14
MAX_AGE_MIN = 60  # GitHub is read again at most this often


def refresh(
    cfg: dict, st: dict, days: int = DAYS, max_age_min: int = MAX_AGE_MIN
) -> bool:
    """Re-read the last `days` days of rocKE merges into develop from GitHub, at most every `max_age_min` minutes."""
    cache = st.get("merges_daily") or {}
    if (
        cache.get("at")
        and minutes_since(cache["at"]) < max_age_min
        and cache.get("days", 0) >= days
    ):
        return False
    from . import stats

    since = (
        dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=days - 1)
    ).isoformat()
    found = stats._scoped(cfg, st, f"is:merged base:develop merged:>={since}")
    st["merges_daily"] = {
        "at": now(),
        "days": days,
        "prs": {
            str(p["number"]): p["mergedAt"][:10] for p in found if p.get("mergedAt")
        },
    }
    return True


def series(st: dict, store, days: int = DAYS) -> list[tuple[str, int, list[int]]]:
    """(UTC day, merged, PR numbers) for the last `days` days, oldest first."""
    today = dt.datetime.now(dt.timezone.utc).date()
    keys = [(today - dt.timedelta(days=i)).isoformat() for i in range(days - 1, -1, -1)]
    day_of = dict((st.get("merges_daily") or {}).get("prs") or {})
    for e in daylog.entries(store):
        base = (st["prs"].get(str(e.get("pr"))) or {}).get("base", "develop")
        if e["kind"] == "merged" and e.get("pr") and base == "develop":
            day_of.setdefault(str(e["pr"]), e["ts"][:10])
    by = {k: [] for k in keys}
    for n, day in day_of.items():
        if day in by:
            by[day].append(int(n))
    return [(k, len(by[k]), sorted(by[k])) for k in keys]


def board_note(st: dict, store) -> str:
    week = series(st, store, 7)
    return f"merged today {week[-1][1]} (last 7 days {sum(c for _, c, _ in week)})"


@idle.timer("merges per day")
def _refresh_due(cfg: dict, st: dict, store):
    """When the merges per day are read from GitHub again (MAX_AGE_MIN)."""
    at = (st.get("merges_daily") or {}).get("at")
    if at:
        yield "merges per day", idle.plus(at, MAX_AGE_MIN)
