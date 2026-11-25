# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Claude's tokens from Claude Code's transcripts, counted one way for both sides of a mirror test (rkmon_lib/mirror.py,
`rk mirror-report`): the live monitor's interactive session and the container's orchestrator write the same files.

A transcript is one JSON line per message part; a model call is one assistant message (its id), counted once, with its
last part's usage and the effort it ran at (the line's perTurnEffort). Per call: uncached input, cache reads, cache
writes (5-minute and 1-hour, from usage.cache_creation; a write it does not split counts at the 5-minute rate) and
output. A tally sums them over the calls: `processed` is every input token (uncached, read and written), `output` the
output as the transcript records it (it is recorded from the start of a stream, so a lane's output and thinking are
undercounted on both sides alike), `usd` the kinds at the Opus prices of config `prices` (`prices`: the table
`rk spend` reads, so the mirror's cost and rk spend agree; DEFAULT_PRICES when it has none), `ieq` the same in uncached
input tokens (the prices relative to an uncached input token), and `efforts` the calls at each effort.

- `runs(root, ...)`: each subagent run under a Claude config's projects folder (agent-<id>.jsonl beside its
  .meta.json): its lane (verifier, quality, notice, other), the PR and round its brief names, its first and last
  call's time, and its tally.
- `session_windows(root, windows, marker)`: the main sessions' calls in each window, in one pass, those of the
  sessions that hold `marker` (MONITOR: the live monitor's slash command; "" for every session, as in the container,
  whose sessions are all the monitor's), never a sidechain's.
"""

from __future__ import annotations

import json
import os
import pathlib

# US dollars per million tokens, by kind, when config `prices` has no Claude Opus entry (Claude Opus 5.5's).
DEFAULT_PRICES = {
    "uncached": 4.0,
    "cache_read": 0.20,
    "cache_write_5m": 5.0,
    "cache_write_1h": 8.0,
    "output": 20.0,
}
KINDS = tuple(DEFAULT_PRICES)
# config `prices`' names for the kinds (spend.usd reads the same table).
PRICE_KEYS = {
    "uncached": "in",
    "cache_read": "cache_read",
    "cache_write_5m": "cache_write_5m",
    "cache_write_1h": "cache_write_1h",
    "output": "out",
}


def prices(cfg: dict | None) -> dict:
    """The kinds' prices: config `prices`' newest claude-opus-* entry (spend.rates), else DEFAULT_PRICES."""
    from . import spend

    try:
        r = spend.rates(spend.price_table(cfg or {}), "opus")
        return {k: float(r[PRICE_KEYS[k]]) for k in KINDS}
    except (ValueError, KeyError, TypeError):
        return dict(DEFAULT_PRICES)


MONITOR = "<command-name>/rocke-monitor</command-name>"  # the monitor's slash command in a session's prompts
BRIEF_CHARS = 4000  # the start of a subagent's first prompt read for its PR and round


def claude_projects() -> pathlib.Path:
    """This machine's Claude Code projects folder: $CLAUDE_CONFIG_DIR/projects, else ~/.claude/projects."""
    base = os.environ.get("CLAUDE_CONFIG_DIR") or os.path.expanduser("~/.claude")
    return pathlib.Path(base) / "projects"


def calls(
    path, since: str = "", until: str = "", sidechain: bool | None = None
) -> list[tuple[str, dict, str]]:
    """(timestamp, usage, effort) of each model call in a transcript, in order, within [since, until) when given;
    `sidechain` False keeps the main thread's only. Times compare to the second (the first 19 characters).
    """
    since, until = since[:19], until[:19]
    msgs, order = {}, []
    try:
        f = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return []
    with f:
        for line in f:
            if '"usage"' not in line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict) or e.get("type") != "assistant":
                continue
            if sidechain is not None and bool(e.get("isSidechain")) != sidechain:
                continue
            t = str(e.get("timestamp") or "")[:19]
            if (since and t < since) or (until and t >= until):
                continue
            msg = e.get("message") or {}
            u = msg.get("usage")
            if not isinstance(u, dict):
                continue
            mid = msg.get("id") or e.get("uuid") or f"#{len(order)}"
            if mid not in msgs:
                order.append(mid)
            msgs[mid] = (t, u, str(e.get("perTurnEffort") or e.get("effort") or ""))
    return [msgs[m] for m in order]


def tally(cs, price: dict | None = None) -> dict:
    """The calls' tokens by kind, `processed` (every input token), `output`, `usd` (at `price`, `prices`' form;
    DEFAULT_PRICES when None), `ieq`, `calls` and `efforts`."""
    price = price or DEFAULT_PRICES
    weights = {k: v / price["uncached"] for k, v in price.items()}
    out: dict = {k: 0 for k in KINDS}
    efforts: dict[str, int] = {}
    for c in cs:
        u = c[1]
        if len(c) > 2 and c[2]:
            efforts[c[2]] = efforts.get(c[2], 0) + 1
        written = int(u.get("cache_creation_input_tokens") or 0)
        split = (
            u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
        )
        w1h = int(split.get("ephemeral_1h_input_tokens") or 0)
        w5 = (
            max(written - w1h, int(split.get("ephemeral_5m_input_tokens") or 0))
            if split
            else written
        )
        out["uncached"] += int(u.get("input_tokens") or 0)
        out["cache_read"] += int(u.get("cache_read_input_tokens") or 0)
        out["cache_write_5m"] += w5
        out["cache_write_1h"] += w1h
        out["output"] += int(u.get("output_tokens") or 0)
    out["calls"] = len(cs)
    out["processed"] = (
        out["uncached"]
        + out["cache_read"]
        + out["cache_write_5m"]
        + out["cache_write_1h"]
    )
    out["usd"] = round(sum(out[k] * price[k] for k in KINDS) / 1e6, 4)
    out["ieq"] = round(sum(out[k] * weights[k] for k in KINDS))
    out["efforts"] = efforts
    return out


def add(*tallies: dict) -> dict:
    """The sum of tallies."""
    keys = (*KINDS, "calls", "processed", "ieq")
    out: dict = {k: sum(int(t.get(k) or 0) for t in tallies if t) for k in keys}
    out["usd"] = round(sum(float(t.get("usd") or 0) for t in tallies if t), 4)
    out["efforts"] = {}
    for t in tallies:
        for e, n in ((t or {}).get("efforts") or {}).items():
            out["efforts"][e] = out["efforts"].get(e, 0) + n
    return out


def effort_of(t: dict) -> str:
    """A tally's effort in words: the one effort its calls ran at, else each with its calls, most first ("" when the
    transcripts record none)."""
    eff = sorted(
        ((t or {}).get("efforts") or {}).items(), key=lambda kv: (-kv[1], kv[0])
    )
    return eff[0][0] if len(eff) == 1 else " · ".join(f"{e} {n}" for e, n in eff)


def lane_of(meta: dict) -> str:
    """A subagent run's lane, by its agent type (and a code quality run's description: a notice's draft)."""
    t, d = meta.get("agentType") or "", (meta.get("description") or "").lower()
    if t == "rocke-review-verifier" or (
        not t.startswith("rocke-") and ("verify" in d or "re-review" in d)
    ):
        return "verifier"
    if t == "rocke-code-quality-reviewer" or "quality lane" in d or "code quality" in d:
        return "notice" if "notice" in d else "quality"
    return "other"


def _first_prompt(path) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict) and e.get("type") == "user":
                    c = (e.get("message") or {}).get("content")
                    text = (
                        c
                        if isinstance(c, str)
                        else " ".join(
                            p.get("text", "") for p in c or [] if isinstance(p, dict)
                        )
                    )
                    return text[:BRIEF_CHARS]
    except OSError:
        pass
    return ""


def runs(
    root, since: str = "", until: str = "", price: dict | None = None
) -> list[dict]:
    """Each subagent run under `root` whose first call is in [since, until): {id, type, lane, description, pr, round,
    start, end, tally} (tally at `price`)."""
    from .orchestrator import brief_pr

    since, until = since[:19], until[:19]
    out = []
    for meta_p in sorted(pathlib.Path(root).rglob("agent-*.meta.json")):
        jl = meta_p.with_name(meta_p.name.replace(".meta.json", ".jsonl"))
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cs = calls(jl)
        if not cs or (since and cs[0][0] < since) or (until and cs[0][0] >= until):
            continue
        pr, rnd = brief_pr(f"{meta.get('description') or ''}\n{_first_prompt(jl)}")
        out.append(
            {
                "id": jl.stem.replace("agent-", ""),
                "type": meta.get("agentType") or "",
                "lane": lane_of(meta),
                "description": (meta.get("description") or "")[:120],
                "pr": pr,
                "round": rnd,
                "start": cs[0][0],
                "end": cs[-1][0],
                "tally": tally(cs, price),
            }
        )
    return out


def lanes_for(all_runs: list[dict], n: int, since: str, until: str = "") -> dict:
    """PR n's lanes in [since, until): each lane's runs and summed tally."""
    since, until = since[:19], until[:19]
    out: dict[str, dict] = {}
    for r in all_runs:
        if r["pr"] != n or r["start"] < since or (until and r["start"] >= until):
            continue
        lane = out.setdefault(r["lane"], {"runs": 0, "tally": {}})
        lane["runs"] += 1
        lane["tally"] = add(lane["tally"], r["tally"])
    return out


def session_windows(
    root,
    windows: list[tuple[str, str]],
    marker: str = MONITOR,
    price: dict | None = None,
) -> list[dict]:
    """The main sessions' calls in each window [since, until), in one pass over the main transcripts under `root` that
    hold `marker` (never a subagent's, never a sidechain's): a tally per window ({} for a window with none).
    """
    if not windows:
        return []
    windows = [(a[:19], b[:19]) for a, b in windows]
    lo, hi = min(w[0] for w in windows), max(w[1] for w in windows)
    per: list[list] = [[] for _ in windows]
    for p in sorted(pathlib.Path(root).rglob("*.jsonl")):
        if p.name.startswith("agent-") or "subagents" in p.parts:
            continue
        if marker:
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    if not any(marker in line for line in f):
                        continue
            except OSError:
                continue
        for c in calls(p, lo, hi, sidechain=False):
            for i, (a, b) in enumerate(windows):
                if a <= c[0] < b:
                    per[i].append(c)
    return [tally(cs, price) if cs else {} for cs in per]


def session(
    root, since: str, until: str, marker: str = MONITOR, price: dict | None = None
) -> dict:
    """The main sessions' calls in [since, until) (session_windows for one window)."""
    return session_windows(root, [(since, until)], marker, price)[0]
