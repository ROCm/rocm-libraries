# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Scoreboard: per-author productivity, for the admin dashboard (the local page only) and `rk scoreboard`: what
each PR author lands, and how much extra work their merged PRs cause other people.

One durable, append-only record, `state/scoreboard_events.jsonl`: one small event per fact, {kind, ts, author, pr,
other_pr, ref}, where `author` is the PR author the fact counts for and `pr` their PR. The card and `rk scoreboard` read
only this file. Each full tick derives the events from the monitor's own records of the last `scoreboard.lookback_days`
days (`tick`) and appends the new ones; `rk scoreboard-backfill` derives them over a longer span, with GitHub GETs for
what the records lack. An event's identity (IDENT) decides what is new, so a tick run twice, or a backfill over a span
the ticks covered, adds nothing. The limits are the `scoreboard` settings (DEFAULTS).

Attribution is conservative: when the records do not single out one PR, nothing is counted. The rules:
- merged: an in-scope PR merged into develop (a MERGED event, the state's merged PRs; backfill: GitHub's search);
- rebase_forced: an open PR of another author turned CONFLICTING within `conflict_window_h` hours after a landing, with
  no push of its own in between, and the landing is the only candidate: the only develop commit in that window that
  changed a file the PR conflicts on, or (with no file record) the only landing whose merge-plan forced-rebase list
  named the PR; or a develop merge on the PR's branch resolved conflicts in files that exactly one develop PR changed
  since the branch's base. Re-blesses count when the golden files conflict. One per (landing, PR);
- rereview_caused: a follow-up round we posted on another author's PR that exists for porting one landing: its title
  says that PR landed or names it as what to port or rebase over; or its head is a develop merge whose whole delta
  since our previous post resolves a conflict attributed to that landing; or it is the PR's first follow-up after
  the owner of an overlap group it got our notice for landed, and its title speaks of a port, rebase or land order.
  A round with two candidates counts for neither;
- develop_break: a develop break filed on GitHub (of either kind, rocKE) whose title says "since #N",
  or whose incident names the breaking PR, with exactly one such PR, an in-scope landing;
- landed_ahead: a PR that merged while the merge plan had an AUTOMERGE_JUMP or a PRIORITY_READY_EARLY on it, or from a
  wave after the first while its landing made a PR of an earlier wave rebase (the plan the tick saw at the landing;
  backfill: an AUTOMERGE_JUMP or PRIORITY_READY_EARLY line in the tick log at most `ahead_h` hours before the merge);
- merged_open_items: a MERGED_OVER_RED or HELD_MERGED for the PR (the state's merged_over; backfill: the journal);
- followup: a follow-up ticket filed for a merged PR: a "- KEY-N:" line of state/followups/tickets_<N>.md (an existing
  ticket only commented on is not counted), or a daily-log ticket entry "follow-up from merged N".
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from urllib.parse import quote

from . import config, conflicts, daylog, gh, gitops, idle, registry
from .store import now, parse_ts

FILE = "scoreboard_events.jsonl"
KINDS = {
    "merged": "Merged",
    "rebase_forced": "Rebases forced",
    "rereview_caused": "Re-reviews caused",
    "develop_break": "Develop breaks",
    "landed_ahead": "Landed ahead of plan",
    "merged_open_items": "Merged with open review items",
    "followup": "Follow-ups",
}
# What makes an event the same fact: a re-run, or a second source of the same fact, adds nothing.
IDENT = {
    "merged": ("pr",),
    "rebase_forced": ("pr", "other_pr"),
    "rereview_caused": ("other_pr", "ref"),
    "develop_break": ("ref",),
    "landed_ahead": ("pr",),
    "merged_open_items": ("pr",),
    "followup": ("pr", "ref"),
}
WINDOWS = (7, 30, 90)
DEFAULT_DAYS = 30
DEFAULTS = registry.package_section("scoreboard")
GIT_TIMEOUT_S = 60
MERGED_PR = conflicts.MERGED_PR


def settings(cfg: dict | None) -> dict:
    """The `scoreboard` settings over DEFAULTS."""
    return dict(DEFAULTS, **((cfg or {}).get("scoreboard") or {}))


idle.file_part("scoreboard events", lambda store: store.dir / FILE)


def ident(e: dict) -> str:
    return e["kind"] + ":" + ":".join(str(e.get(k) or "") for k in IDENT[e["kind"]])


def event(
    kind: str, ts: str, author: str, pr: int, other_pr: int | None = None, ref: str = ""
) -> dict:
    return {
        "kind": kind,
        "ts": ts,
        "author": author,
        "pr": pr,
        "other_pr": other_pr,
        "ref": ref,
    }


def load(store) -> list[dict]:
    return [
        e
        for e in store.read_jsonl(FILE, lenient=True)
        if e.get("kind") in IDENT and e.get("author")
    ]


def new_events(store, events: list[dict]) -> list[dict]:
    """The events not in the file yet, each once."""
    seen = {ident(e) for e in load(store)}
    out = []
    for e in sorted(events, key=lambda e: (e["ts"], e["kind"], e["pr"])):
        k = ident(e)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def _z(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _t(s: str) -> dt.datetime:
    return parse_ts(s[:19] + "Z") if len(s) >= 19 else parse_ts(s[:10] + "T00:00:00Z")


# --- the records ----------------------------------------------------------------------------------------------------

PORT_RX = (
    re.compile(r"#?(\d{4,6})(?:'s)?\s+(?:landed|merged)\b", re.I),
    re.compile(r"\bports?\s+(?:for|of)\s+#?(\d{4,6})(?:\s+and\s+#?(\d{4,6}))?", re.I),
    re.compile(
        r"\b(?:rebase|re-?merge|port)\w*\s+(?:over|onto|on|after)\s+#?(\d{4,6})", re.I
    ),
)
PORT_WORDS = re.compile(r"\b(?:port\w*|rebase\w*|re-?merge\w*|land[- ]order)\b", re.I)
SINCE = re.compile(r"\bsince #(\d{4,6})\b")
TICKET_LINE = re.compile(r"^\s*-\s*([A-Z][A-Z0-9]+-\d+)\s*:", re.M)
TICKET_KEY = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
FOLLOWUP_FROM = re.compile(r"follow-up from merged #?(\d{4,6})", re.I)
TICKETS_FILE = re.compile(r"^tickets_(\d+)\.md$")
JOURNAL_OVER = re.compile(r"^(\S+Z) queue\s+#(\d+) merged by \S+ over ")
JOURNAL_HELD = re.compile(r"^(\S+Z) queue\s+#(\d+) merged while (?:held|in triage)")
LOOP_TICK = re.compile(r"^TICK (\S+Z) ")
LOOP_AHEAD = re.compile(r"\b(AUTOMERGE_JUMP|PRIORITY_READY_EARLY) #(\d+)")


class Records:
    """The monitor's records the rules read, each read once. `cache`: the git answers (small lists, keyed by a digest of
    the question; they name commits, so an answer never changes), kept between ticks; `budget_s`: the git time one pass
    may spend on questions not cached yet (None: no limit). A question past the budget, or one git cannot answer (a
    pruned commit), gets None: its rule counts nothing this pass, and a later pass asks again.
    """

    def __init__(
        self,
        cfg: dict,
        st: dict,
        store,
        cache: dict | None = None,
        budget_s: float | None = None,
        git=None,
    ):
        self.cfg, self.st, self.store = cfg, st, store
        self.repo = (cfg.get("paths") or {}).get("repo")
        self.cache, self.used = (cache if cache is not None else {}), set()
        self.budget_s, self.spent, self.deferred = budget_s, 0.0, 0
        self._git = git or self._run_git
        self._windows: dict[tuple, list] = {}
        self.events = store.read_jsonl("events.jsonl", lenient=True)
        self.ledger = store.read_jsonl("actions.jsonl", lenient=True)
        self.daylog = store.read_jsonl(daylog.FILE, lenient=True)
        self.conflicts = store.read_jsonl(conflicts.FILE, lenient=True)
        self.prs = st.get("prs") or {}
        self.extra_landings: dict[int, dict] = (
            {}
        )  # backfill: GitHub's merges the records lack
        self.extra_breaks: list[dict] = (
            []
        )  # backfill: the develop breaks filed on GitHub
        self.journal: list[tuple[str, int, str]] = (
            []
        )  # backfill: (ts, pr, MERGED_OVER_RED | HELD_MERGED)
        self.loop_alerts: list[tuple[str, int, str]] = (
            []
        )  # backfill: (ts, pr, AUTOMERGE_JUMP | PRIORITY_READY_EARLY)
        self._landings = None

    def _run_git(self, *args: str) -> str:
        if not self.repo:
            raise gitops.GitError("no paths.repo")
        return gitops.git(self.repo, *args, timeout=GIT_TIMEOUT_S, lazy=False)

    def answer(self, question: str, compute):
        """`compute()`'s answer to `question`, from the cache or computed within the budget; None when not known."""
        key = hashlib.sha1(question.encode()).hexdigest()[:20]
        if key in self.cache:
            self.used.add(key)
            return self.cache[key]
        if self.budget_s is not None and self.spent >= self.budget_s:
            self.deferred += 1
            return None
        t0 = time.monotonic()
        try:
            value = compute()
        except (gitops.GitError, OSError, subprocess.SubprocessError, ValueError):
            return None
        finally:
            self.spent += time.monotonic() - t0
        self.cache[key] = value
        self.used.add(key)
        return value

    def author(self, n: int) -> str | None:
        x = self.landings().get(n)
        return (x or {}).get("author") or (self.prs.get(str(n)) or {}).get("author")

    def landings(self) -> dict[int, dict]:
        """In-scope PRs merged into develop: {n: {ts, author, sha, exact}}; `exact`: ts is the merge time."""
        if self._landings is not None:
            return self._landings
        out: dict[int, dict] = {}
        for e in self.events:
            if e.get("kind") != "MERGED" or not e.get("pr"):
                continue
            rec = self.prs.get(str(e["pr"])) or {}
            if rec.get("base", "develop") != "develop":
                continue
            out[int(e["pr"])] = {
                "ts": e.get("merged_at") or e["ts"],
                "author": rec.get("author"),
                "sha": e.get("merge_sha") or rec.get("merge_sha"),
                "exact": True,
            }
        for n, x in self.extra_landings.items():
            out.setdefault(n, x)
        logged = {}
        for r in self.daylog:
            if r.get("kind") == "merged" and r.get("pr"):
                logged.setdefault(int(r["pr"]), r["ts"])
        for k, rec in self.prs.items():
            n = int(k)
            if (
                rec.get("status") != "merged"
                or n in out
                or rec.get("base", "develop") != "develop"
            ):
                continue
            ts = logged.get(n) or rec.get("gone_at")
            if ts:
                out[n] = {
                    "ts": ts,
                    "author": rec.get("author"),
                    "sha": rec.get("merge_sha"),
                    "exact": False,
                }
        self._landings = {n: x for n, x in out.items() if x.get("author")}
        return self._landings

    @staticmethod
    def _commits(out: str, names: bool) -> list[tuple]:
        """`git log --format=%x01%ct %s [--name-only]` as (PR or None, commit time[, files])."""
        rows = []
        for line in out.splitlines():
            if line.startswith("\x01"):
                ct, _, subj = line[1:].partition(" ")
                m = MERGED_PR.search(subj)
                rows.append(
                    [int(m.group(1)) if m else None, int(ct)]
                    + ([set()] if names else [])
                )
            elif names and line.strip() and rows:
                rows[-1][2].add(line.strip())
        return rows

    def window_hits(
        self, develop: str, start: dt.datetime, end: dt.datetime, files: list[str]
    ) -> list | None:
        """[PR or None, commit time] of each first-parent develop commit (up to `develop`) committed in (start, end] that
        changed one of `files`. The window's commits are read once a pass, with their files (one `git log`).
        """
        s, e = int(start.timestamp()), int(end.timestamp())

        def compute():
            w = self._windows.get((develop, s, e))
            if w is None:
                w = self._windows[(develop, s, e)] = self._commits(
                    self._git(
                        "-c",
                        "core.quotepath=false",
                        "log",
                        "--first-parent",
                        "--name-only",
                        "--format=%x01%ct %s",
                        f"--since=@{s}",
                        f"--until=@{e}",
                        develop,
                    ),
                    names=True,
                )
            want = set(files)
            return [[pr, ct] for pr, ct, changed in w if s < ct <= e and changed & want]

        return self.answer(
            f"window {develop} {s} {e} " + "\n".join(sorted(files)), compute
        )

    def since_base_hits(
        self, parent: str, develop: str, files: list[str]
    ) -> list | None:
        """[PR or None, commit time] of each first-parent develop commit since the merge base of `parent` and `develop`
        that changed one of `files`."""

        def compute():
            base = self._git("merge-base", parent, develop).strip()
            return [
                [pr, ct]
                for pr, ct in self._commits(
                    self._git(
                        "log",
                        "--first-parent",
                        "--format=%x01%ct %s",
                        f"{base}..{develop}",
                        "--",
                        *files,
                    ),
                    names=False,
                )
            ]

        return self.answer(
            f"since-base {parent} {develop} " + "\n".join(sorted(files)), compute
        )


# --- the rules ------------------------------------------------------------------------------------------------------


def _in(ts: str, since: dt.datetime, until: dt.datetime) -> bool:
    try:
        return since <= _t(ts) <= until
    except ValueError:
        return False


def merged(R: Records, since, until) -> list[dict]:
    return [
        event("merged", x["ts"], x["author"], n, None, (x.get("sha") or "")[:10])
        for n, x in R.landings().items()
        if _in(x["ts"], since, until)
    ]


def _other(R: Records, landing: int, pr: int) -> bool:
    """`pr` is another author's PR than `landing`'s (both authors known)."""
    a, b = R.author(landing), R.author(pr)
    return bool(a and b and a != b and landing != pr)


def rebases(R: Records, since, until, snaps: dict) -> tuple[list[dict], dict, dict]:
    """rebase_forced events, and what they resolve: {(pr, head12): landing} for a conflict seen live, {merge12: landing}
    for a develop merge on a PR's branch that resolved one."""
    out, episodes, merges = [], {}, {}
    lands = R.landings()
    window = dt.timedelta(hours=settings(R.cfg)["conflict_window_h"])
    live = defaultdict(list)
    for r in R.conflicts:
        if r.get("source") == "live" and r.get("pr") and r.get("ts"):
            live[int(r["pr"])].append(r)
    pushes = defaultdict(list)
    for e in R.events:
        if e.get("kind") == "PUSH" and e.get("pr") and e.get("ts"):
            pushes[int(e["pr"])].append(_t(e["ts"]))
    for e in R.events:
        if (
            e.get("kind") != "CONFLICT"
            or not e.get("pr")
            or not _in(e.get("ts") or "", since, until)
        ):
            continue
        p, t = int(e["pr"]), _t(e["ts"])
        recent = {
            n: _t(x["ts"])
            for n, x in lands.items()
            if x["exact"] and t - window < _t(x["ts"]) <= t and n != p
        }
        if not recent:
            continue
        if any(min(recent.values()) < u <= t for u in pushes.get(p, [])):
            continue  # its own push may have made the conflict: not counted
        cands = {
            n for n in recent if p in ((snaps.get(str(n)) or {}).get("forces") or [])
        }
        row = next(
            (
                r
                for r in sorted(live.get(p, []), key=lambda r: r["ts"])
                if t - dt.timedelta(minutes=10)
                <= _t(r["ts"])
                <= t + dt.timedelta(hours=6)
            ),
            None,
        )
        if row and row.get("files"):
            hits = R.window_hits(row["develop"], t - window, t, row["files"])
            if hits is None:
                cands.add(None)  # not known (yet): unsure
            else:
                cands |= {n for n, _ in hits}
        if len(cands) != 1:
            continue
        n = next(iter(cands))
        if n in lands and _other(R, n, p):
            out.append(
                event("rebase_forced", e["ts"], lands[n]["author"], n, p, "conflict")
            )
            if row:
                episodes[(p, row["head"][:12])] = n
    for r in R.conflicts:
        if (
            r.get("source") != "resolved"
            or not r.get("files")
            or not _in(r.get("ts") or "", since, until)
        ):
            continue
        p = int(r["pr"])
        hits = R.since_base_hits(r["parent"], r["develop"], r["files"])
        if not hits:
            continue
        cands = {n for n, _ in hits if n != p}
        if len(cands) != 1:
            continue
        n = next(iter(cands))
        if n in lands and _other(R, n, p):
            merges[r["merge"][:12]] = n
            out.append(
                event(
                    "rebase_forced",
                    r["ts"],
                    lands[n]["author"],
                    n,
                    p,
                    f"merge {r['merge'][:10]}",
                )
            )
    return out, episodes, merges


def _titles(R: Records) -> dict[str, str]:
    return {
        r["url"]: r.get("text") or ""
        for r in R.daylog
        if r.get("kind") == "review" and r.get("url")
    }


def rereviews(R: Records, since, until, episodes: dict, merges: dict) -> list[dict]:
    lands, titles = R.landings(), _titles(R)
    posts = defaultdict(list)
    for r in R.ledger:
        if r.get("kind") == "post" and r.get("pr"):
            posts[int(r["pr"])].append(r)
    resolved = {
        (int(r["pr"]), r["merge"][:12]): r
        for r in R.conflicts
        if r.get("source") == "resolved" and r.get("pr")
    }
    decided = _decided(R.store)
    owners = defaultdict(
        set
    )  # PR -> the owners of the overlap groups we sent it a notice for
    for r in R.ledger:
        if r.get("kind") == "notice-overlap" and r.get("pr"):
            for key in str(r.get("head") or "").split(","):
                o = (decided.get(key) or {}).get("owner")
                if o and int(o) != int(r["pr"]):
                    owners[int(r["pr"])].add((int(o), r["ts"]))
    out = []
    for p, rows in posts.items():
        rows.sort(key=lambda r: r["ts"])
        for i, post in enumerate(rows):
            if not str(post.get("note") or "").startswith("followup") or not _in(
                post["ts"], since, until
            ):
                continue
            t, title = _t(post["ts"]), titles.get(post.get("ref") or "", "")
            cands = set()
            for rx in PORT_RX:
                for m in rx.finditer(title):
                    for g in m.groups():
                        if (
                            g
                            and int(g) != p
                            and int(g) in lands
                            and _t(lands[int(g)]["ts"]) < t
                        ):
                            cands.add(int(g))
            if i > 0:
                r = resolved.get((p, (post.get("head") or "")[:12]))
                if r and r["parent"][:12] == (rows[i - 1].get("head") or "")[:12]:
                    n = merges.get(r["merge"][:12]) or episodes.get(
                        (p, r["parent"][:12])
                    )
                    if n:
                        cands.add(n)
            for o, noticed in owners.get(p, ()):
                if (
                    o in lands
                    and noticed < post["ts"]
                    and _t(lands[o]["ts"]) < t
                    and PORT_WORDS.search(title)
                ):
                    first = next(
                        (
                            x
                            for x in rows
                            if str(x.get("note") or "").startswith("followup")
                            and _t(x["ts"]) > _t(lands[o]["ts"])
                        ),
                        None,
                    )
                    if first is post:
                        cands.add(o)
            if len(cands) == 1:
                n = next(iter(cands))
                if _other(R, n, p):
                    out.append(
                        event(
                            "rereview_caused",
                            post["ts"],
                            lands[n]["author"],
                            n,
                            p,
                            post.get("ref") or "",
                        )
                    )
    return out


def _decided(store) -> dict:
    try:
        return json.loads(
            (store.dir / "overlaps_decided.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}


def breaks(R: Records, since, until) -> list[dict]:
    lands = R.landings()
    filed: dict[int, dict] = {}
    for r in R.ledger:  # the develop breaks we filed
        if r.get("kind") == "issue" and r.get("pr"):
            filed.setdefault(
                int(r["pr"]), {"title": r.get("note") or "", "ts": r["ts"]}
            )
    for i in (R.st.get("post_merge") or {}).get("open") or []:
        if i.get("rocke"):
            filed.setdefault(
                int(i["n"]),
                {"title": i.get("title") or "", "ts": i.get("created") or ""},
            )
    for i in R.extra_breaks:
        filed.setdefault(
            int(i["n"]), {"title": i.get("title") or "", "ts": i.get("created") or ""}
        )
    named = {}
    for x in _incidents(R.store):
        if x.get("issue") and x.get("culprit"):
            named.setdefault(int(x["issue"]), set()).add(int(x["culprit"]))
    out = []
    for n, i in filed.items():
        if not i["ts"] or not _in(i["ts"], since, until):
            continue
        cands = {int(x) for x in SINCE.findall(i["title"])} | named.get(n, set())
        if len(cands) == 1:
            c = next(iter(cands))
            if c in lands:
                out.append(
                    event(
                        "develop_break",
                        i["ts"][:19] + "Z",
                        lands[c]["author"],
                        c,
                        None,
                        f"#{n}",
                    )
                )
    return out


def _incidents(store) -> list[dict]:
    try:
        return json.loads((store.dir / "incidents.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def ahead(R: Records, since, until, snaps: dict) -> list[dict]:
    lands, out = R.landings(), []
    for k, s in snaps.items():
        n = int(k)
        if s.get("ahead") and n in lands and _in(lands[n]["ts"], since, until):
            out.append(
                event(
                    "landed_ahead",
                    lands[n]["ts"],
                    lands[n]["author"],
                    n,
                    None,
                    s["ahead"],
                )
            )
    for ts, n, kind in R.loop_alerts:
        x = lands.get(n)
        if (
            x
            and x["exact"]
            and _in(x["ts"], since, until)
            and dt.timedelta(0)
            <= _t(x["ts"]) - _t(ts)
            <= dt.timedelta(hours=settings(R.cfg)["ahead_h"])
        ):
            out.append(event("landed_ahead", x["ts"], x["author"], n, None, kind))
    return out


def open_items(R: Records, since, until) -> list[dict]:
    lands, out = R.landings(), []
    rows = [
        (m.get("at") or "", int(k), m.get("kind") or "MERGED_OVER_RED")
        for k, m in (R.st.get("merged_over") or {}).items()
    ] + list(R.journal)
    for ts, n, kind in rows:
        if n in lands and ts and _in(ts, since, until):
            out.append(
                event("merged_open_items", ts, lands[n]["author"], n, None, kind)
            )
    return out


def followups(R: Records, since, until) -> list[dict]:
    lands, out = R.landings(), []
    d = R.store.dir / config.FOLLOWUPS
    for p in sorted(d.glob("tickets_*.md")) if d.is_dir() else []:
        m = TICKETS_FILE.match(p.name)
        if not m or int(m.group(1)) not in lands:
            continue
        n = int(m.group(1))
        ts = _z(dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc))
        if not _in(ts, since, until):
            continue
        for key in dict.fromkeys(
            TICKET_LINE.findall(p.read_text(encoding="utf-8", errors="replace"))
        ):
            out.append(event("followup", ts, lands[n]["author"], n, None, key))
    for r in R.daylog:
        if r.get("kind") != "ticket":
            continue
        m, k = FOLLOWUP_FROM.search(r.get("text") or ""), TICKET_KEY.match(
            r.get("text") or ""
        )
        if m and k and int(m.group(1)) in lands and _in(r["ts"], since, until):
            n = int(m.group(1))
            out.append(
                event("followup", r["ts"], lands[n]["author"], n, None, k.group(1))
            )
    return out


def derive(
    R: Records, since: dt.datetime, until: dt.datetime, snaps: dict | None = None
) -> list[dict]:
    """Every event the records show in [since, until]."""
    snaps = snaps or {}
    reb, episodes, merges = rebases(R, since, until, snaps)
    found = (
        merged(R, since, until)
        + reb
        + rereviews(R, since, until, episodes, merges)
        + breaks(R, since, until)
        + ahead(R, since, until, snaps)
        + open_items(R, since, until)
        + followups(R, since, until)
    )
    out = {}
    for e in sorted(found, key=lambda e: e["ts"]):  # the first of the same fact stands
        out.setdefault(ident(e), e)
    return list(out.values())


# --- the tick ---------------------------------------------------------------------------------------------------------


def note_landings(st: dict, events: list[dict], cfg: dict | None = None) -> None:
    """Right after the poll, before the queue plans again: what the merge plan the tick last computed said about each
    PR that just merged (its forced-rebase list, and whether it lands ahead of the plan). Kept `lookback_days` days.
    """
    lookback = settings(cfg)["lookback_days"]
    sb = st.setdefault("scoreboard", {})
    snaps = sb.setdefault("landings", {})
    mp = st.get("merge_plan") or {}
    for e in events:
        if e.get("kind") != "MERGED" or not e.get("pr"):
            continue
        k = str(e["pr"])
        node = (mp.get("nodes") or {}).get(k) or {}
        snaps[k] = {
            "at": now(),
            "forces": list(node.get("forces") or []),
            "ahead": _ahead_kind(mp, k, node),
        }
    for k in [
        k
        for k, s in snaps.items()
        if (dt.datetime.now(dt.timezone.utc) - _t(s["at"])).days > lookback
    ]:
        del snaps[k]


def _ahead_kind(mp: dict, k: str, node: dict) -> str:
    if ((mp.get("automerge") or {}).get(k) or {}).get("alert"):
        return "AUTOMERGE_JUMP"
    if k in (mp.get("ready_early") or {}):
        return "PRIORITY_READY_EARLY"
    w = node.get("wave")
    if not w or w < 2:
        return ""
    waves = {int(n): x.get("wave") for n, x in (mp.get("nodes") or {}).items()}
    if any(
        (waves.get(m) or 99) < w
        for m in (node.get("conflicts") or []) + (node.get("rebase_ahead") or [])
    ):
        return "against the plan's order"
    return ""


def tick(cfg: dict, st: dict, store, git=None) -> list[str]:
    """The full tick's step: the events of the last `lookback_days` days not recorded yet, appended, within the
    `git_budget_s` of new git questions. Its git answers are kept in st["scoreboard"]["git"] while a record of that
    span uses them. Advisory: a failure is one line."""
    sb = st.setdefault("scoreboard", {})
    t0 = time.monotonic()
    try:
        s = settings(cfg)
        R = Records(
            cfg,
            st,
            store,
            cache=dict(sb.get("git") or {}),
            budget_s=float(s["git_budget_s"]),
            git=git,
        )
        until = dt.datetime.now(dt.timezone.utc)
        new = new_events(
            store,
            derive(
                R,
                until - dt.timedelta(days=s["lookback_days"]),
                until,
                sb.get("landings"),
            ),
        )
        store.append_jsonl(FILE, new)
        sb["git"] = {k: v for k, v in R.cache.items() if k in R.used}
    except Exception as exc:  # noqa: BLE001 - advisory: never fail a tick on it
        return [f"(scoreboard not recorded: {str(exc)[:200]})"]
    sb["last"] = {
        "at": now(),
        "added": len(new),
        "secs": round(time.monotonic() - t0, 2),
        "git_s": round(R.spent, 2),
        "deferred": R.deferred,
    }
    return []


# --- the backfill -----------------------------------------------------------------------------------------------------


def read_journal(store, since: dt.datetime) -> list[tuple[str, int, str]]:
    out = []
    path = store.journal_path
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if " queue " not in line or " merged " not in line:
                continue
            for rx, kind in (
                (JOURNAL_OVER, "MERGED_OVER_RED"),
                (JOURNAL_HELD, "HELD_MERGED"),
            ):
                m = rx.match(line)
                if m and _t(m.group(1)) >= since:
                    out.append((m.group(1), int(m.group(2)), kind))
    return out


def read_loop_log(store, since: dt.datetime) -> list[tuple[str, int, str]]:
    path = store.dir / idle.LOOP_FILES["LOOP_LOG"]
    out, at = [], None
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LOOP_TICK.match(line)
            if m:
                at = m.group(1)
                continue
            m = LOOP_AHEAD.search(line)
            if m and at and _t(at) >= since:
                out.append((at, int(m.group(2)), m.group(1)))
    return out


def _merge_changes(R: Records, sha: str | None, folder: str) -> bool:
    """Whether merge commit SHA changes files in FOLDER (git, read-only); False when that is not known."""
    if not sha:
        return False
    try:
        return bool(
            R._git("diff", "--name-only", f"{sha}^1", sha, "--", folder).strip()
        )
    except (gitops.GitError, OSError, subprocess.SubprocessError):
        return False


def github_merges(
    cfg: dict, R: Records, since: dt.datetime
) -> tuple[dict[int, dict], list[str]]:
    """In-scope PRs merged into develop since `since` that the records lack, from GitHub's search (GETs), by the label
    each was found by (config.Scope.takes): a scope label's PR always; a shared label's only when its merge commit
    changes files in that label's folder (git, read-only).
    """
    sc, out, notes = config.scope(cfg), {}, []
    have = R.landings()
    for label in sc.searched():
        q = f'repo:{cfg["repo"]} is:pr is:merged base:develop label:"{label}" merged:>={since.date().isoformat()}'
        try:
            items = gh.rest(
                f"search/issues?q={quote(q)}&per_page=100",
                paginate=True,
                jq="[.items[] | {n: .number, author: .user.login, merged_at: .pull_request.merged_at}]",
            )
        except gh.GhError as exc:
            notes.append(f"search for {label!r} failed: {str(exc)[:160]}")
            continue
        for i in items:
            n = int(i["n"])
            if n in have or n in out or not i.get("merged_at"):
                continue
            try:
                sha = gh.rest(
                    f"repos/{cfg['repo']}/pulls/{n}", jq="{s: .merge_commit_sha}"
                )["s"]
            except gh.GhError:
                sha = None
            if not sc.takes([label], lambda folder: _merge_changes(R, sha, folder)):
                continue  # not known to change files in its shared label's folder: left out
            out[n] = {
                "ts": i["merged_at"],
                "author": i["author"],
                "sha": (sha or "")[:10] or None,
                "exact": True,
            }
    return out, notes


def github_breaks(cfg: dict, since: dt.datetime) -> tuple[list[dict], list[str]]:
    """The develop breaks of both kinds filed on GitHub since `since`, open or closed, rocKE ones (GETs)."""
    from .breakage import issue_repository, labels, relevant

    out, notes = {}, []
    for lab in labels(cfg).values():
        try:
            rows = gh.rest(
                f"repos/{issue_repository(cfg)}/issues?labels={quote(lab)}&state=all&since={_z(since)}"
                "&per_page=100",
                paginate=True,
                jq="[.[] | select(.pull_request == null) | {n: .number, title, "
                "labels: [.labels[].name], created: .created_at}]",
            )
        except gh.GhError as exc:
            notes.append(f"the {lab!r} label's reports not read: {str(exc)[:160]}")
            continue
        for i in rows:
            if i.get("created") and _t(i["created"]) >= since and relevant(cfg, i):
                out[int(i["n"])] = i
    return list(out.values()), notes


def backfill(cfg: dict, st: dict, store, days: int, github: bool = True) -> dict:
    """`rk scoreboard-backfill`'s reading: the events of the last `days` days from every record (the journal and the
    tick log too), GitHub's merges and develop breaks filling what the records lack (GETs). Writes nothing; `record`
    appends the new ones."""
    until = dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(days=days)
    R = Records(
        cfg, st, store, cache=dict((st.get("scoreboard") or {}).get("git") or {})
    )
    R.journal, R.loop_alerts = read_journal(store, since), read_loop_log(store, since)
    notes = []
    if github:
        R.extra_landings, n1 = github_merges(cfg, R, since)
        R.extra_breaks, n2 = github_breaks(cfg, since)
        R._landings = None
        notes += n1 + n2
    found = derive(R, since, until, (st.get("scoreboard") or {}).get("landings"))
    return {
        "days": days,
        "since": since,
        "until": until,
        "found": found,
        "cache": R.cache,
        "notes": notes,
        "github": github,
        "merges": len(R.extra_landings),
        "breaks": len(R.extra_breaks),
    }


def record(store, st: dict, res: dict) -> list[dict]:
    """Append the backfill's events not recorded yet (under the caller's lock), and give the tick the git answers it
    found, so the tick's first passes ask git nothing the backfill answered."""
    new = new_events(store, res["found"])
    store.append_jsonl(FILE, new)
    st.setdefault("scoreboard", {}).setdefault("git", {}).update(res["cache"])
    return new


def backfill_lines(store, res: dict, new: list[dict], dry: bool) -> list[str]:
    have = Counter(e["kind"] for e in res["found"])
    add = Counter(e["kind"] for e in new)
    first = min((e["ts"] for e in res["found"]), default=None)
    out = [
        f"scoreboard backfill, last {res['days']} days ({_z(res['since'])[:10]} to {_z(res['until'])[:10]})"
        + (
            f", dry run: nothing written ({len(new)} events would be appended)"
            if dry
            else f": {len(new)} events appended to {store.dir / FILE}"
        ),
        (
            f"first event {first[:10] if first else '-'}; GitHub: {res['merges']} merges and {res['breaks']} develop "
            "breaks the records lack"
            if res["github"]
            else f"first event {first[:10] if first else '-'}; GitHub not " "read"
        ),
    ]
    out += [f"  {KINDS[k]:<30} {have[k]:>4} found, {add[k]:>4} new" for k in KINDS]
    return out + [f"  ({x})" for x in res["notes"]]


# --- the table --------------------------------------------------------------------------------------------------------


def window_start(days: int, end: dt.datetime) -> dt.datetime:
    """A window is `days` UTC days ending with today's: it starts at midnight `days - 1` days ago."""
    return dt.datetime.combine(
        end.date() - dt.timedelta(days=days - 1), dt.time(), dt.timezone.utc
    )


def window(events: list[dict], days: int, end: dt.datetime | None = None) -> list[dict]:
    end = end or dt.datetime.now(dt.timezone.utc)
    return [e for e in events if _in(e["ts"], window_start(days, end), end)]


def summary(events: list[dict], days: int, end: dt.datetime | None = None) -> dict:
    """One row per PR author with an event in the window (Rework for others first), the team's totals, and the trend:
    per bucket (a day for a 7-day window, else an ISO week, the first cut at the window's start) the PRs merged and the
    rework for others, from the first day the record holds (`from`: no bucket claims a zero the record cannot know).
    """
    end = end or dt.datetime.now(dt.timezone.utc)
    start = window_start(days, end)
    evs = window(events, days, end)
    by = defaultdict(Counter)
    for e in evs:
        by[e["author"]][e["kind"]] += 1
    rows = []
    for who, c in by.items():
        r = {
            "author": who,
            "rebase_forced": c["rebase_forced"],
            "rereview_caused": c["rereview_caused"],
            "rework": c["rebase_forced"] + c["rereview_caused"],
        }
        r.update(
            {
                k: c[k]
                for k in (
                    "merged",
                    "develop_break",
                    "landed_ahead",
                    "merged_open_items",
                    "followup",
                )
            }
        )
        rows.append(r)
    rows.sort(key=lambda r: (-r["rework"], -r["merged"], r["author"].lower()))
    tot = Counter()
    for r in rows:
        tot.update({k: v for k, v in r.items() if k != "author"})
    first = min((_t(e["ts"]) for e in events), default=end)
    since = max(start, dt.datetime.combine(first.date(), dt.time(), dt.timezone.utc))
    return {
        "days": days,
        "start": _z(start),
        "end": _z(end),
        "from": _z(since),
        "rows": rows,
        "totals": dict(tot),
        "trend": trend(evs, days, since, end),
    }


def trend(
    evs: list[dict], days: int, start: dt.datetime, end: dt.datetime
) -> list[dict]:
    """The buckets from `start`'s to `end`'s: days for a 7-day window, else ISO weeks (keyed by their Monday)."""
    if days <= 7:
        firsts = [
            start.date() + dt.timedelta(days=i)
            for i in range((end.date() - start.date()).days + 1)
        ]
        key = lambda t: t.date().isoformat()  # noqa: E731
    else:
        firsts, monday = [], start.date() - dt.timedelta(days=start.weekday())
        while monday <= end.date():
            firsts.append(monday)
            monday += dt.timedelta(days=7)
        key = lambda t: (
            t.date() - dt.timedelta(days=t.weekday())
        ).isoformat()  # noqa: E731
    out = {
        d.isoformat(): {
            "key": d.isoformat(),
            "label": d.strftime("%m-%d"),
            "merged": 0,
            "rework": 0,
        }
        for d in firsts
    }
    for e in evs:
        b = out.get(key(_t(e["ts"])))
        if not b:
            continue
        if e["kind"] == "merged":
            b["merged"] += 1
        elif e["kind"] in ("rebase_forced", "rereview_caused"):
            b["rework"] += 1
    return list(out.values())


def lines(events: list[dict], days: int, end: dt.datetime | None = None) -> list[str]:
    """`rk scoreboard`: the table in the terminal."""
    s = summary(events, days, end)
    head = (
        "Author",
        "Merged",
        "Rework (rebases + re-reviews)",
        "Develop breaks",
        "Landed ahead",
        "Open items merged",
        "Follow-ups",
    )
    body = [
        (
            r["author"],
            str(r["merged"]),
            f"{r['rework']} ({r['rebase_forced']} + {r['rereview_caused']})",
            str(r["develop_break"]),
            str(r["landed_ahead"]),
            str(r["merged_open_items"]),
            str(r["followup"]),
        )
        for r in s["rows"]
    ]
    t = s["totals"]
    total = (
        "Team",
        str(t.get("merged", 0)),
        f"{t.get('rework', 0)} ({t.get('rebase_forced', 0)} + "
        f"{t.get('rereview_caused', 0)})",
        str(t.get("develop_break", 0)),
        str(t.get("landed_ahead", 0)),
        str(t.get("merged_open_items", 0)),
        str(t.get("followup", 0)),
    )
    width = [max(len(x[i]) for x in [head, total, *body]) for i in range(len(head))]

    def fmt(row) -> str:
        return "  ".join(
            c.ljust(width[i]) if i == 0 else c.rjust(width[i])
            for i, c in enumerate(row)
        )

    out = [
        f"Scoreboard, last {days} days ({s['start'][:10]} to {s['end'][:10]} UTC"
        + (f"; recorded from {s['from'][:10]}" if s["from"] > s["start"] else "")
        + "); sorted by Rework for others",
        fmt(head),
        "  ".join("-" * w for w in width),
    ] + [fmt(r) for r in body]
    out += ["  ".join("-" * w for w in width), fmt(total)]
    out.append(
        "Trend ("
        + ("per day" if days <= 7 else "per ISO week, by its Monday")
        + "): "
        + ", ".join(
            f"{b['label']} {b['merged']} merged / {b['rework']} rework"
            for b in s["trend"]
        )
    )
    return out if s["rows"] else out[:1] + ["No activity recorded in the window."]
