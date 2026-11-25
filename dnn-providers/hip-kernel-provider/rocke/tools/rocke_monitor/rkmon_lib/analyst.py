# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The analyst, an optional on-prem model role (rkmon_lib/roles.py): it ranks and
annotates the per-PR insights (insights.pr_statistics) and never invents a number.

At most every `every_h` hours (its settings', or the analytics model's; default 24), and only when the statistics
changed, the tick asks it for a reading of them: which categories and PRs matter most, and a short note on each. Before
anything shows, `check` holds the reading to the data it was given: every number its summary and notes cite must be
one the statistics hold (a count, hours or minutes, a PR number), each PR it names must be one they name, and each
category one of theirs. A reading that fails is refused, the journal says which number or PR, and nothing of it is
shown. The analyst never acts.
"""

from __future__ import annotations

import json
import re

from . import insights, roles

ROLE = "analyst"
EVERY_H = 24
SYSTEM = (
    "You read statistics about a repository's pull requests and say what matters most. Answer with one JSON "
    "object only, no prose around it. Cite only numbers that appear in the data."
)


def every_h(cfg: dict) -> int:
    return int((roles.wanted(cfg, ROLE) or {}).get("every_h") or EVERY_H)


def messages(stats: dict) -> list[dict]:
    """The analyst's request: the window's statistics, the rule that every number cited is one of them, and the
    answer's form."""
    data = {k: stats[k] for k in ("days", "categories")}
    text = "\n".join(
        [
            f"These are the last {stats['days']} days of a repository's pull requests, per category: the overall figures "
            f"and the {insights.TOP} PRs where each was largest.",
            json.dumps(data, ensure_ascii=False),
            "Rank what deserves attention first, and say why in a few words. Every number you write must appear in the "
            "data above, exactly; name only PRs and categories it names.",
            'Answer: {"summary": "two sentences at most", "ranking": [{"category": "<key>", "pr": <PR or null>, '
            '"note": "a few words"}]} (at most 8 ranking entries).',
        ]
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]


def check(stats: dict, ans: dict) -> tuple[dict | None, str]:
    """(the reading, "") when every figure, PR and category it cites is in `stats`; else (None, why it is refused)."""
    summary, ranking = ans.get("summary"), ans.get("ranking") or []
    if (
        not isinstance(summary, str)
        or not summary.strip()
        or not isinstance(ranking, list)
    ):
        return None, "it is not the reading asked for (a summary and a ranking)"
    keys = {c["key"] for c in stats["categories"]}
    prs = {r["pr"] for c in stats["categories"] for r in c["top"]}
    allowed = insights.numbers(stats) | {str(p) for p in prs}
    rows = []
    for r in ranking[:8]:
        if not isinstance(r, dict) or r.get("category") not in keys:
            return (
                None,
                f"it ranks a category the statistics do not have ({str((r or {}).get('category'))[:40]})",
            )
        pr = r.get("pr")
        if pr is not None and (not isinstance(pr, int) or pr not in prs):
            return None, f"it names PR {pr}, which the statistics do not"
        note = str(r.get("note") or "").strip()
        rows.append({"category": r["category"], "pr": pr, "note": note[:200]})
    texts = [summary] + [r["note"] for r in rows]
    for t in texts:
        for m in insights.NUMBER.finditer(t):
            if insights.norm(m.group(0)) not in allowed:
                return None, f"it cites {m.group(0)}, which the statistics do not hold"
    return {"summary": re.sub(r"\s+", " ", summary).strip()[:400], "ranking": rows}, ""


def due(cfg: dict, st: dict, stats: dict) -> bool:
    """A reading is due: the analyst is on, none runs, the statistics changed since the last reading, and the last is
    older than every_h hours (or there is none)."""
    if not roles.on(cfg, ROLE):
        return False
    r = roles.state(st, ROLE)
    if r.get("job"):
        return False
    if r.get("data_fp") == stats["fp"]:
        return False
    at = r.get("started")
    if not at:
        return True
    from .store import minutes_since

    return minutes_since(at) >= every_h(cfg) * 60


def start(cfg: dict, st: dict, store, stats: dict) -> str:
    """Ask for a reading of `stats` (kept with the role's state: the reading is checked against them)."""
    line = roles.start(cfg, st, store, ROLE, stats["fp"], messages(stats))
    if line:
        r = roles.state(st, ROLE)
        r["data_fp"], r["data"] = stats["fp"], {
            k: stats[k] for k in ("days", "start", "end", "categories", "fp")
        }
    return line


def reading(st: dict, store=None) -> dict | None:
    """The last reading that passed `check`, with its model and time; None when there is none. A reading that fails
    is refused once, in the journal."""
    r = roles.state(st, ROLE)
    last, data = r.get("last") or {}, r.get("data")
    if not last or not data or last.get("fp") != data.get("fp") or not last.get("ok"):
        return None
    if r.get("checked") != last.get("at"):
        ok, why = check(data, last.get("answer") or {})
        r["checked"], r["reading"], r["refused"] = last.get("at"), ok, why
        if not ok and store is not None:
            store.journal(
                "role", f"analyst: its reading of {last.get('at')} was refused: {why}"
            )
    if not r.get("reading"):
        return None
    return dict(
        r["reading"],
        model=last.get("model"),
        at=last.get("at"),
        data_at=data.get("end"),
    )
