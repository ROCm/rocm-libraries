# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Insights from recent PRs, an expandable dashboard card: what leads to common issues, what we do that causes issues,
and suggestions for better practices.

Two live parts, recomputed every tick from the monitor's own records, and one static part:
- findings: every 🔴/🟠 finding of our reviews in the last `DAYS` days, posted or kept unposted while posting is off
  (the daily log's `finding` saves), grouped by cause with keyword rules read from the finding's title and impact,
  not its file list (first match wins). Each cause shows how many we later confirmed fixed before merge (`fixed`
  saves, matched by PR, post day and N-number), the last 7 days against the 7 before, and the practice that prevents
  it;
- develop: the open rocKE develop issues, grouped by cause;
- report: the tables of the newest `rocke_insights_*.md`, the analysts' report;
- per PR (`pr_statistics`): where the time goes, the TOP PRs of each category and
  the whole window: how often a PR was rebased (force-pushes, and develop merges into it), conflicted with develop and
  for how long, our review rounds and the fix rounds after them, and the CI runner waits (per PR, and per runner
  label). Computed from what the monitor already records: events.jsonl, the ledger, conflicts.jsonl, and ci_runs.jsonl
  through ci_times.summary; the force-pushes from the clone (a push whose old head is no ancestor of the new one),
  cached. The analyst (rkmon_lib/analyst.py), when it is on, ranks and annotates them, and never cites a number they
  do not hold.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess

from . import daylog, gitops, paths
from .breakage import entry_kind
from .store import minutes_since

DAYS = 14

# (key, cause, pattern, what prevents it). Order matters: the first match wins, so the narrow causes come first.
CAUSES = (
    (
        "public",
        "Perf numbers or internal details in public files",
        r"absolute (?:\w+ ){0,2}(?:timings?|values?|numbers?|figures?|times?)|TFLOP|GFLOP|\b[MG]Hz\b"
        r"|internal (?:build )?host|pack times?|clock values?|§Compliance",
        "Ratios only in docs, PR bodies and commits; rule_scan.py on the diff, the body and the messages",
    ),
    (
        "windows",
        "Fails on Windows",
        r"Windows|cp1252|WinError|\bfork\b|command line|\.exe\b",
        "encoding= on every open and read_text; spawn, not fork; short command lines; a Windows lint in pre-commit",
    ),
    (
        "scope",
        "Copied or unrelated work in the PR",
        r"scratch files?|unrelated|\bbundles?\b|ships #\d+|copies #\d+|stale copy",
        "One topic per PR; no scratch files; never copy another PR's commits",
    ),
    (
        "golden",
        "Goldens not re-blessed",
        r"golden|re-?bless|sha256|re-?record|re-?measured",
        "Re-bless the touched families on the merged tree right before merge; never hand-merge goldens",
    ),
    (
        "abi",
        "ABI numbering",
        r"\bABI\b|abi\.(?:h|py)",
        "Declare a bump; the number is develop + 1 at landing (`rk abi` gives the order)",
    ),
    (
        "skew",
        "Develop moved underneath the PR",
        r"conflict|re-?merge|\brebase|no longer merges|hand-merge|develop's #\d+|develop already|since #\d+|after the base"
        r"|behind develop|pre-#\d+|#\d+(?:'s| also| already| renamed| added| appended| moved| split| took)",
        "Merge develop the day you land; port develop's changes instead of picking a side",
    ),
    (
        "parity",
        "C++ and Python engines disagree",
        r"both engines|C\+\+ engine|Python only|byte[- ]identi|C\+\+/Python",
        "Change both engines in one commit; byte identity at all three flavors",
    ),
    (
        "coverage",
        "Tests no CI lane runs",
        r"CTest|CI lane|outside CI|untested|not collected|\bcollects?\b|llvm-objdump|importorskip|no torch|never runs?",
        "Register every new test in a CI lane, and check that CTest collects it",
    ),
    (
        "ci",
        "Build or CI red",
        r"fails at Build|clang-tidy|CI is red|CI reds?|red at head|builds? (?:break|fail)|shards? fail|fails? \w+_pytest",
        "Build and lint C++ outside rocke (clang-tidy, the superbuild) before asking for review",
    ),
    (
        "correctness",
        "Correctness and robustness",
        r"overflow|\bwrong\b|incorrect|mismatch|\bNaN\b|out of (?:range|bounds)|(?:does|do) not check|unchecked|\bcrash"
        r"|\bhangs?\b|\bleaks?\b|\brace\b|still picks|\bexceeds?\b|builds every|\bdense\b.{0,30}\bgrids?\b",
        "A test with the failing input lands with the fix",
    ),
)
OTHER = ("other", "Other", "", "")
_RULES = [(k, c, re.compile(p, re.I), w) for k, c, p, w in CAUSES]

# Open develop issues by cause: (key, cause, pattern), first match wins.
DEV_CAUSES = (
    ("timeout", "Intermittent timeouts", r"time[sd]? out|timeout|intermittent|flak"),
    ("windows", "Fails on Windows only", r"Windows"),
    ("broke", "A merged PR broke a test", r"since #\d+"),
)
_DEV = [(k, c, re.compile(p, re.I)) for k, c, p in DEV_CAUSES]

FIND = re.compile(r"^(N\d+)\s+(\U0001F534|\U0001F7E0)")
FIXED = re.compile(r"our (\d\d-\d\d) (N\d+)")


def _body(text: str) -> str:
    """A finding's title and impact, without its `N1 🔴 <files>:` prefix: the text after the first "**", or, in the
    older unbolded posts, after the location (which ends at the first `: ` that follows a backtick, ")" or quote).
    """
    if "**" in text:
        return text.split("**", 1)[1]
    t = re.sub(r"^N\d+\s+\S+\s+", "", text)
    m = re.search(r"[`)\"]:\s", t)
    return t[m.end() :] if m and re.search(r"[`\"]", t[: m.start()]) else t


def classify(text: str) -> tuple[str, str, str, str]:
    """The cause of one finding, from its title and impact; only when those match nothing, from its file list too."""
    body = _body(text)
    for s in (body, text):
        hit = next(((k, c, p.pattern, w) for k, c, p, w in _RULES if p.search(s)), None)
        if hit:
            return hit
    return OTHER


def _title(text: str) -> str:
    m = re.search(r"\*\*(?:Problem:\*\*\s*)?(.+?)(?:\*\*|<br>|$)", text)
    t = (m.group(1) if m else _body(text)).strip()
    return t if len(t) <= 110 else t[:109] + "…"


def findings(store, days: int = DAYS) -> list[dict]:
    """Our reviews' 🔴/🟠 findings of the last `days` days (posted, or kept unposted while posting is off), each with
    its cause and whether we confirmed it fixed."""
    rows = daylog.entries(store)
    fixed = {
        (r["pr"], m.group(1), m.group(2))
        for r in rows
        if r["kind"] == "save"
        and r.get("save") == "fixed"
        and (m := FIXED.search(r["text"]))
    }
    out = []
    for r in rows:
        if (
            r["kind"] != "save"
            or r.get("save") != "finding"
            or not r.get("pr")
            or minutes_since(r["ts"]) > days * 1440
        ):
            continue
        m = FIND.match(r["text"])
        sev = (
            m.group(2)
            if m
            else next(
                (c for c in ("\U0001f534", "\U0001f7e0") if c in r["text"][:16]), ""
            )
        )
        key, cause, _, practice = classify(r["text"])
        out.append(
            {
                "ts": r["ts"],
                "pr": r["pr"],
                "nid": m.group(1) if m else None,
                "sev": sev,
                "key": key,
                "cause": cause,
                "practice": practice,
                "title": _title(r["text"]),
                "fixed": bool(m) and (r["pr"], r["ts"][5:10], m.group(1)) in fixed,
            }
        )
    return out


def by_cause(fs: list[dict]) -> list[dict]:
    """Findings per cause, most frequent first."""
    rows = {}
    for f in fs:
        r = rows.setdefault(
            f["key"],
            {
                "key": f["key"],
                "cause": f["cause"],
                "practice": f["practice"],
                "n": 0,
                "red": 0,
                "orange": 0,
                "fixed": 0,
                "last7": 0,
                "prior7": 0,
                "prs": {},
            },
        )
        r["n"] += 1
        r["red"] += f["sev"] == "\U0001f534"
        r["orange"] += f["sev"] == "\U0001f7e0"
        r["fixed"] += f["fixed"]
        age = minutes_since(f["ts"]) / 1440
        r["last7" if age <= 7 else "prior7"] += 1
        r["prs"].setdefault(f["pr"], []).append(f["title"])
    return sorted(
        rows.values(), key=lambda r: (r["key"] == "other", -r["n"], r["cause"])
    )


def develop(cfg: dict, st: dict) -> list[dict]:
    """Open rocKE develop issues per cause: {"cause", "issues": [{"n", "title", "kind"}]}, most frequent first."""
    rows = {}
    for i in (st.get("post_merge") or {}).get("open") or []:
        if not i.get("rocke"):
            continue
        key, cause = next(
            ((k, c) for k, c, p in _DEV if p.search(i.get("title", ""))),
            ("other", "Other"),
        )
        rows.setdefault(key, {"key": key, "cause": cause, "issues": []})[
            "issues"
        ].append(
            {
                "n": i["n"],
                "title": i.get("title", ""),
                "kind": entry_kind(cfg, i),
            }
        )
    return sorted(
        rows.values(),
        key=lambda r: (r["key"] == "other", -len(r["issues"]), r["cause"]),
    )


def latest_report(cfg: dict) -> pathlib.Path | None:
    """The newest insights report in `paths.insights_dir`; None when there is none (or no folder is configured)."""
    where = paths.resolve(cfg).get("insights_dir")
    d = pathlib.Path(where) if where else None
    files = sorted(d.glob("rocke_insights_*.md")) if d and d.is_dir() else []
    return files[-1] if files else None


def report_tables(path: pathlib.Path) -> list[tuple[str, list[list[str]]]]:
    """The report's tables, each under its `## ` heading: [(heading, [header, row, ...])], in order. Only the first table
    under a heading is read."""
    out, head, rows, done = [], None, None, False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            head, rows, done = line[3:].strip(), None, False
        elif head and not done and line.startswith("|"):
            cells = [
                c.strip().replace("\\|", "|")
                for c in re.split(r"(?<!\\)\|", line.strip()[1:-1])
            ]
            if all(c and set(c) <= set("-: ") for c in cells):
                continue  # the separator row
            if rows is None:
                rows = []
                out.append((head, rows))
            rows.append(cells)
        elif rows is not None:
            done = True
    return out


# --- per PR: where the time goes ------------------------------------------------------------------------------------

TOP = 5  # PRs shown per category
CACHE = "insights_cache.json"  # the force-push verdicts, by old..new head: the clone is asked once per push
CACHE_KEEP = 5000  # verdicts kept (the oldest go first)
UTC = dt.timezone.utc
ENDS = (
    "PUSH",
    "MERGED",
    "CLOSED",
    "OUT_OF_SCOPE",
)  # what ends a conflict, unless it is seen conflicting after
CATEGORIES = (
    (
        "rebased",
        "Rebased",
        "force-pushes (a rebase or an amend), and develop merges into the PR",
    ),
    (
        "conflicted",
        "Conflicted with develop",
        "each conflict with develop GitHub reported, and the hours it lasted",
    ),
    ("review_rounds", "Review rounds", "our review posts"),
    (
        "fix_rounds",
        "Fix rounds",
        "our reviews a push to the PR followed (a round per review), and those pushes",
    ),
    (
        "runner_waits",
        "CI runner waits",
        "jobs of the longest workflows waiting for a runner (created to started)",
    ),
)


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _iso(t: dt.datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _in(ts: str | None, start: dt.datetime, end: dt.datetime) -> bool:
    try:
        return bool(ts) and start <= _ts(ts) < end
    except ValueError:
        return False


def _hours(secs: float) -> float:
    return round(secs / 3600, 1)


def _minutes(secs: float) -> float:
    return round(secs / 60, 1)


def _top(per_pr: dict, value, detail, n: int = TOP) -> list[dict]:
    rows = [
        {"pr": pr, "value": value(v), "detail": detail(v)} for pr, v in per_pr.items()
    ]
    return sorted(
        (r for r in rows if r["value"]), key=lambda r: (-r["value"], r["pr"])
    )[:n]


def pr_statistics(
    cfg: dict, st: dict, store, days: int = DAYS, end: dt.datetime | None = None
) -> dict:
    """The last `days` days, per PR (the TOP of each category) and overall (module docstring): {"days", "start",
    "end", "categories": [{"key", "title", "what", "overall": {...}, "top": [{"pr", "value", "detail"}], and for the
    runner waits "labels": [...]}], "fp"}. Every figure is a count, hours or minutes, rounded to one decimal.
    """
    from . import ci_times, conflicts

    end = end or dt.datetime.now(UTC)
    start = end - dt.timedelta(days=days)
    events = store.read_jsonl("events.jsonl", lenient=True)
    rows = store.read_jsonl(conflicts.FILE, lenient=True)
    pushes = [
        e
        for e in events
        if e.get("kind") == "PUSH" and e.get("pr") and _in(e.get("ts"), start, end)
    ]
    found = [
        _rebased(cfg, store, pushes, rows, start, end),
        _conflicted(st, events, rows, start, end),
        *_rounds(store.ledger(), pushes, start, end),
        _runner_waits(ci_times.summary(store, start, end)),
    ]
    cats = [
        {"key": key, "title": title, "what": what, **c}
        for (key, title, what), c in zip(CATEGORIES, found)
    ]
    out = {"days": days, "start": _iso(start), "end": _iso(end), "categories": cats}
    out["fp"] = hashlib.sha1(json.dumps(cats, sort_keys=True).encode()).hexdigest()[:12]
    return out


def _forced(repo: str, old: str, new: str, cache: dict) -> bool | None:
    """Was the push old -> new a force-push: old no ancestor of new (a rebase, an amend)? None when the clone lacks
    either commit. Cached by "old..new"."""
    key = f"{old}..{new}"
    if key not in cache:
        try:
            have = (
                bool(repo and old and new)
                and gitops.has_commit(repo, old)
                and gitops.has_commit(repo, new)
            )
            cache[key] = not gitops.is_ancestor(repo, old, new) if have else None
        except (OSError, subprocess.SubprocessError):
            return None  # asked again next time
    return cache[key]


def _rebased(
    cfg: dict,
    store,
    pushes: list[dict],
    rows: list[dict],
    start: dt.datetime,
    end: dt.datetime,
) -> dict:
    """Force-pushes (the clone's verdict) and develop merges into the PR (conflicts.jsonl's merge rows, the clean ones
    included). A push the clone cannot judge counts in `pushes_not_checked`."""
    repo = paths.resolve(cfg).get("repo") or ""
    path = store.dir / CACHE
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    known = cache.setdefault("ancestry", {})
    asked = len(known)
    per: dict[int, dict] = {}
    unknown = 0
    for e in pushes:
        forced = _forced(repo, e.get("old") or "", e.get("new") or "", known)
        if forced is None:
            unknown += 1
        elif forced:
            per.setdefault(e["pr"], {"forced": 0, "merges": 0})["forced"] += 1
    for r in rows:
        if (
            r.get("source") == "resolved"
            and r.get("pr")
            and _in(r.get("ts"), start, end)
        ):
            per.setdefault(r["pr"], {"forced": 0, "merges": 0})["merges"] += 1
    if len(known) != asked:
        for k in list(known)[: max(0, len(known) - CACHE_KEEP)]:
            del known[k]
        try:
            tmp = path.with_suffix(".json.new")
            tmp.write_text(json.dumps(cache), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass  # asked again next time
    forced, merges = sum(v["forced"] for v in per.values()), sum(
        v["merges"] for v in per.values()
    )
    return {
        "overall": {
            "rebases": forced + merges,
            "force_pushes": forced,
            "develop_merges": merges,
            "prs": len(per),
            "pushes_not_checked": unknown,
        },
        "top": _top(
            per,
            lambda v: v["forced"] + v["merges"],
            lambda v: f"{v['forced']} force-push(es), {v['merges']} develop merge(s)",
        ),
    }


def _conflicted(
    st: dict, events: list[dict], rows: list[dict], start: dt.datetime, end: dt.datetime
) -> dict:
    """Each CONFLICT event opens a conflict with develop. It ends at the PR's next push, merge, close or exit from
    scope, or later when conflicts.jsonl still saw it conflicting after that; one nothing ended lasts while the PR
    still conflicts, else until its last sighting. The conflicts that overlap the window count, with their hours in
    it."""
    from . import conflicts

    by_pr: dict[int, list[dict]] = {}
    for e in events:
        if e.get("pr") and e.get("ts"):
            by_pr.setdefault(e["pr"], []).append(e)
    live: dict[int, list[str]] = {}
    for r in rows:
        if r.get("source") == "live" and r.get("pr") and r.get("ts"):
            live.setdefault(r["pr"], []).append(r["ts"])
    per: dict[int, dict] = {}
    for pr, evs in by_pr.items():
        evs.sort(key=lambda x: x["ts"])
        for i, e in enumerate(evs):
            if e.get("kind") != "CONFLICT" or _ts(e["ts"]) >= end:
                continue
            later = evs[i + 1 :]
            nxt = next((x["ts"] for x in later if x.get("kind") == "CONFLICT"), None)
            stop = next(
                (
                    x["ts"]
                    for x in later
                    if x.get("kind") in ENDS and (not nxt or x["ts"] < nxt)
                ),
                None,
            )
            seen = [
                t for t in live.get(pr, []) if e["ts"] <= t and (not nxt or t < nxt)
            ]
            if (
                not stop
                and not nxt
                and conflicts.conflicting(st["prs"].get(str(pr)) or {})
            ):
                finish = end
            else:
                finish = _ts(max(seen + [stop] if stop else seen, default=e["ts"]))
            began = _ts(e["ts"])
            if finish < start or (finish == began < start):
                continue
            d = per.setdefault(pr, {"times": 0, "secs": 0.0})
            d["times"] += 1
            d["secs"] += max(
                0.0, (min(finish, end) - max(began, start)).total_seconds()
            )
    return {
        "overall": {
            "conflicts": sum(v["times"] for v in per.values()),
            "prs": len(per),
            "hours": _hours(sum(v["secs"] for v in per.values())),
        },
        "top": _top(
            per,
            lambda v: _hours(v["secs"]),
            lambda v: f"{v['times']} time(s), {_hours(v['secs'])} h in all",
        ),
    }


def _rounds(
    ledger: list[dict], pushes: list[dict], start: dt.datetime, end: dt.datetime
) -> tuple[dict, dict]:
    """Review rounds: our review posts in the window. Fix rounds: our reviews a push to the PR followed before our
    next review (a push counts for the latest review before it, one from before the window too), and those pushes.
    """
    posted: dict[int, list[str]] = {}
    for r in sorted(
        (r for r in ledger if r.get("kind") == "post" and r.get("pr") and r.get("ts")),
        key=lambda r: r["ts"],
    ):
        posted.setdefault(r["pr"], []).append(r["ts"])
    rr = {
        pr: n
        for pr, ts in posted.items()
        if (n := sum(1 for t in ts if _in(t, start, end)))
    }
    fr: dict[int, dict] = {}
    for e in pushes:
        before = [t for t in posted.get(e["pr"], []) if t < e["ts"]]
        if before:
            d = fr.setdefault(e["pr"], {"rounds": set(), "pushes": 0})
            d["rounds"].add(before[-1])
            d["pushes"] += 1
    review = {
        "overall": {"posts": sum(rr.values()), "prs": len(rr)},
        "top": _top(rr, lambda v: v, lambda v: f"{v} review post(s)"),
    }
    fix = {
        "overall": {
            "rounds": sum(len(v["rounds"]) for v in fr.values()),
            "pushes": sum(v["pushes"] for v in fr.values()),
            "prs": len(fr),
        },
        "top": _top(
            fr,
            lambda v: len(v["rounds"]),
            lambda v: f"{len(v['rounds'])} round(s), {v['pushes']} push(es) after our reviews",
        ),
    }
    return review, fix


def _runner_waits(ci: dict) -> dict:
    """The CI runner waits of ci_times.summary: the jobs of each head's longest workflows, job created to started, per
    PR and per runner label."""
    w = ci.get("job_wait") or {}
    per = {p["pr"]: p for p in ci.get("runner_prs") or []}
    return {
        "overall": {
            "jobs": w.get("n", 0),
            "prs": len(per),
            "median_min": _minutes(w.get("median", 0)),
            "p90_min": _minutes(w.get("p90", 0)),
            "total_h": _hours(sum(p["total_s"] for p in per.values())),
        },
        "top": _top(
            per,
            lambda p: _minutes(p["total_s"]),
            lambda p: f"{p['jobs']} job(s) on {p['heads']} head(s), the longest "
            f"{_minutes(p['max_s'])} min",
        ),
        "labels": [
            {
                "label": x["label"],
                "jobs": x["n"],
                "median_min": _minutes(x["median_s"]),
                "p90_min": _minutes(x["p90_s"]),
                "total_min": _minutes(x["total_s"]),
            }
            for x in ci.get("labels") or []
        ],
    }


def numbers(stats: dict) -> set[str]:
    """Every number the statistics hold, as text ("12", "3.5"): what a reading may cite (rkmon_lib/analyst.py)."""
    out: set[str] = set()

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            out.add(norm(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            for m in NUMBER.finditer(v):
                out.add(norm(m.group(0)))

    walk({k: v for k, v in stats.items() if k not in ("start", "end", "fp")})
    return out


# A number in a text: "3", "3.5", "3." at a sentence's end, "#123", "50%"; not the parts of "1.5.2" or "x86".
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?!\.?\d)(?!\w)")


def norm(v) -> str:
    """A number as text, the way `numbers` keeps it: 3.0 -> "3", "3.50" -> "3.5"."""
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.1f}".rstrip("0").rstrip(".")


def stats_md(
    stats: dict, reading: dict | None = None, labels: bool = True
) -> list[str]:
    """The statistics as Markdown, for `rk insights` and the weekly report, with the analyst's checked reading.
    `labels`: the runner-label table (the weekly report's CI section has its own)."""
    out = [f"## Where the time goes, per PR (last {stats['days']} days)", ""]
    titles = {c["key"]: c["title"] for c in stats["categories"]}
    if reading:
        out += [
            f"The analyst's reading ({reading.get('model')}, {reading.get('at')}): {reading['summary']}",
            "",
        ]
        out += [
            f"- {titles.get(r['category'], r['category'])}"
            + (f" #{r['pr']}" if r.get("pr") else "")
            + f": {r['note']}"
            for r in reading.get("ranking") or []
        ] + [""]
    for c in stats["categories"]:
        o = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in c["overall"].items())
        out += [f"### {c['title']}: {c['what']}", "", f"Overall: {o}.", ""]
        if c["top"]:
            out += ["| PR | Value | Detail |", "|---|---|---|"]
            out += [
                f"| #{r['pr']} | {r['value']} | {r['detail']} |" for r in c["top"]
            ] + [""]
        else:
            out += ["None in the window.", ""]
        if labels and c.get("labels"):
            from .ci_times import _cell  # runner names come from GitHub

            out += [
                "| Runner label | Jobs | Median wait (min) | p90 (min) | In all (min) |",
                "|---|---|---|---|---|",
            ]
            out += [
                f"| {_cell(x['label'])} | {x['jobs']} | {x['median_min']} | {x['p90_min']} | {x['total_min']} |"
                for x in c["labels"]
            ] + [""]
    return out
