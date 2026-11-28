# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The daily log: one tractable record per UTC day of what the monitor did, plus the
"issues avoided" record that evaluates the system.

Every entry goes to `state/daylog.jsonl` ({ts, kind, pr, text, ...}). Each append re-renders that day's file,
`<daily_dir>/rocke_<YYYY-MM-DD>.md`: one summary line, then one table row per event.

Kinds: review (posted) · unposted (a review kept unposted: review posting off) · rebase (verdict kept) · skipped
(review failed or dropped) · gate · proposed · hold · release · merged (into develop) · stacked (merged into another
PR's branch) · develop (sanity) · incident · notice · arm · disarm · ticket · refused (a tool refused to the session or
its agents, which the session records) · cleanup · wait (gh calls that waited for the GitHub request cap:
rkmon.gh_cap_log) · save.

A `save` is an issue the system kept out of develop, by type (a review kept unposted makes the saves its post would
make, so a prototype's log reads as a live monitor's):
- finding: a 🔴/🟠 finding of our review before merge;
- fixed: one of our earlier findings confirmed fixed by a follow-up, before merge;
- gate: a merge blocked by a failing gate (a confirmed hold);
- develop: a develop break caught by the sanity run after a merge;
- verify: the reviewer model's claims that verification corrected before the review was posted or kept;
- other: anything recorded by hand.
"""

from __future__ import annotations

import pathlib
import re
from collections import Counter

from . import idle, paths
from .store import minutes_since, now

SAVE_TYPES = ("finding", "fixed", "gate", "develop", "verify", "other")
FILE = "daylog.jsonl"  # in state/
UNPOSTED = "unposted"  # the kind of a review kept unposted (from_review)

# An entry added by hand (`rk log`, `rk save`) changes the board: the next tick is full (idle.py).
idle.file_part("day log", lambda store: store.dir / FILE)


def add(store, cfg: dict, kind: str, pr: int | None, text: str, **extra) -> dict:
    rec = {
        "ts": now(),
        "kind": kind,
        "pr": pr,
        "text": " ".join(text.split())[:300],
        **extra,
    }
    store.append_jsonl(FILE, [rec])
    try:
        render(store, cfg, rec["ts"][:10])
    except OSError:
        pass
    return rec


def save(store, cfg: dict, pr: int | None, typ: str, text: str) -> dict:
    return add(store, cfg, "save", pr, text, save=typ if typ in SAVE_TYPES else "other")


def logged_line(rec: dict, pr: str) -> str:
    """What `rk log` and `rk save` print for the entry they added (`pr` as given: a number or -)."""
    return (
        f"logged {rec['kind']}"
        + (f" ({rec['save']})" if rec.get("save") else "")
        + f" for {pr}"
    )


def entries(store, day: str | None = None) -> list[dict]:
    return [r for r in store.read_jsonl(FILE) if day is None or r["ts"][:10] == day]


def saves_since(store, days: int = 7) -> Counter:
    return Counter(
        r.get("save", "other")
        for r in store.read_jsonl(FILE)
        if r["kind"] == "save" and minutes_since(r["ts"]) <= days * 1440
    )


def summary(rows: list[dict]) -> str:
    n = Counter(r["kind"] for r in rows)
    sv = Counter(r.get("save", "other") for r in rows if r["kind"] == "save")

    def verdicts(kind: str) -> str:
        """The verdicts of the `kind` entries, " (1 🟠, 2 🟢)", or "" when there are none."""
        v = Counter(r.get("verdict", "?") for r in rows if r["kind"] == kind)
        return f" ({', '.join(f'{c} {k}' for k, c in sorted(v.items()))})" if v else ""

    parts = [
        f"reviews posted {n['review']}" + verdicts("review"),
        *(
            [f"reviews kept unposted {n[UNPOSTED]}" + verdicts(UNPOSTED)]
            if n[UNPOSTED]
            else []
        ),
        f"rebase-only kept {n['rebase']}",
        f"merged {n['merged']}"
        + (f" (+{n['stacked']} stacked)" if n["stacked"] else ""),
        f"gate runs {n['gate']}",
        f"holds {n['hold']}",
        f"develop checks {n['develop']}",
        f"incidents {n['incident']}",
        f"issues avoided {sum(sv.values())}"
        + (f" ({', '.join(f'{k} {v}' for k, v in sorted(sv.items()))})" if sv else ""),
    ]
    return " · ".join(parts)


def render(store, cfg: dict, day: str) -> pathlib.Path:
    rows = entries(store, day)
    d = pathlib.Path(paths.get(cfg, "daily_dir"))
    d.mkdir(parents=True, exist_ok=True)
    out = [
        f"# rocKE monitor log — {day} (UTC)",
        "",
        f"**Summary:** {summary(rows)}",
        "",
        "| Time | What | PR | Detail |",
        "|---|---|---|---|",
    ]
    for r in rows:
        what = r["kind"] + (f" ({r['save']})" if r["kind"] == "save" else "")
        detail = r["text"].replace("|", "\\|") + (
            f" ([link]({r['url']}))" if r.get("url") else ""
        )
        out.append(
            f"| {r['ts'][11:16]}Z | {what} | {'#' + str(r['pr']) if r.get('pr') else '—'} | {detail} |"
        )
    path = d / f"rocke_{day}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return path


# Tick and advance output lines worth an entry: (pattern, kind). Group 1 is the PR number when there is one.
LINES = [
    (r"^REVIEW_SKIP #(\d+): (.*)", "rebase"),
    (r"^REVIEW_(?:FAIL|DROP) #(\d+): (.*)", "skipped"),
    (r"^GATE_DONE #(\d+): (.*)", "gate"),
    (r"^TRIAGE #(\d+): (.*)", "hold"),
    (r"^HOLD_CLEARED #(\d+): (.*)", "release"),
    (r"^HOLD_STALE #(\d+): (.*)", "hold"),
    (r"^BATCH_(?:VALIDATED|FAILED) ()(.*)", "gate"),
    (r"^SANITY_GREEN ()(.*)", "develop"),
    (r"^(?:!! CRITICAL )?INCIDENT_(?:OPEN|CLOSED) ()(.*)", "incident"),
    (r"^(?:!! CRITICAL )?CULPRIT ()(.*)", "incident"),
    (r"^CLEANUP #(\d+)(.*)", "cleanup"),
    (r"^CLEANUP ()(old runs.*)", "cleanup"),
    (r"^ANALYSIS_START #(\d+)(.*)", "gate"),
    (r"^AUTOMERGE_RISK #(\d+)(.*)", "hold"),
    (r"^BREAKAGE_(?:OPEN|CLOSED) ()(.*)", "develop"),
    (r"^HOLD_FIXED #(\d+): (.*)", "hold"),
    (r"^HOLD_REAPPROVED #(\d+): (.*)", "release"),
]


def from_events(store, cfg: dict, st: dict, events: list[dict]) -> None:
    """Merges and closures of tracked PRs (merges are counted whoever merged them; LANDED lines would double-count). A
    merge into another PR's branch is `stacked`, not `merged`: only develop merges count (a collapsed stack would
    otherwise inflate the day's merges)."""
    for e in events:
        if e["kind"] in ("MERGED", "CLOSED") and e.get("pr"):
            rec = st["prs"].get(str(e["pr"])) or {}
            title, base = rec.get("title", ""), rec.get("base") or "develop"
            if e["kind"] == "CLOSED":
                add(store, cfg, "closed", e["pr"], title)
            elif base == "develop":
                add(store, cfg, "merged", e["pr"], title)
            else:
                add(store, cfg, "stacked", e["pr"], f"into {base}: {title}")


def from_lines(store, cfg: dict, st: dict, lines: list[str]) -> None:
    """Log the meaningful lines a tick or an advance printed. PROPOSE is logged only when the proposed order changes."""
    for line in lines:
        if line.startswith("PROPOSE "):
            order = line.split(":", 1)[0][len("PROPOSE ") :]
            if st.get("daylog_proposed") != order:
                st["daylog_proposed"] = order
                add(store, cfg, "proposed", None, line)
            continue
        for pat, kind in LINES:
            m = re.match(pat, line)
            if m:
                pr = int(m.group(1)) if m.group(1) else None
                add(store, cfg, kind, pr, line)
                if kind == "incident" and "INCIDENT_OPEN" in line:
                    save(
                        store,
                        cfg,
                        None,
                        "develop",
                        f"develop break caught by the sanity check after a merge: {line}",
                    )
                break


VERDICT = re.compile(r"^## PR #\d+ .*?— ([🟢🟡🟠🔴])\s*(.*)$", re.M)
FINDING = re.compile(r"^\| (N\d+) \| ([🔴🟠🟡]) \| ([^|]*)\| (.*?)\s*\|\s*$", re.M)
FIXED = re.compile(r"^\| Our (\d\d-\d\d) (N\d+)[^|]*\| 🟢 FIXED", re.M)
# A follow-up's status table carries each prior item forward, ours and other reviewers': "| Our 11-02 N3: what |
# 🟠 OPEN<br>• why |", "| reviewer (11-02) ×8 threads | 🟠 OPEN<br>…". Its summary rows ("Bots / CI", "Our checks")
# say no OPEN.
STILL_OPEN = re.compile(r"^\| ([^|]+?) \| ([🔴🟠])[^|<]*?\bOPEN\b([^|]*)\|\s*$", re.M)


def from_review(
    store, cfg: dict, n: int, kind: str, body: str, url: str | None
) -> None:
    """Log one of our reviews (`kind`, its round as said: first round, follow-up, CI update), and its saves: each 🔴/🟠
    finding, and each earlier finding of ours now confirmed FIXED. A posted review is a `review` entry with its
    comment's url; one kept unposted (url None: review posting off, post.save_draft) is an `unposted` entry.
    """
    v = VERDICT.search(body)
    light, label = (v.group(1), v.group(2)) if v else ("?", "")
    found = FINDING.findall(body)
    fl = ", ".join(f"{k} {sev}" for k, sev, _f, _t in found) or "no net-new findings"
    add(
        store,
        cfg,
        UNPOSTED if url is None else "review",
        n,
        f"{kind} {light} {label[:120]} · {fl}",
        verdict=light,
        **({} if url is None else {"url": url}),
    )
    for k, sev, where, text in found:
        if sev in "🔴🟠":
            save(store, cfg, n, "finding", f"{k} {sev} {where.strip()}: {text[:160]}")
    for day, k in FIXED.findall(body):
        save(store, cfg, n, "fixed", f"our {day} {k} confirmed fixed before merge")
