# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Phase 0 of a draft: the prior items, and the rules a script settles some of them by before any model (drafter.py runs
them; `rk evidence-check` re-reads their evidence with the same helpers).

The prior items, in the order the draft's table shows them:
- approvals, from the staged reviews (data rows);
- the rows and findings of our last review, posted or kept unposted (pr/ours.md, staged by `rk stage-ours`), each with
  the status that review gave it and the head it reviewed; its data rows (approvals, Bots / CI, Our checks, Code
  quality) are not items;
- the other reviewers' items with activity after that review (every one of them on a first round): reviews, inline
  threads and conversation comments (drafter.prior_items).
An item that holds no review point of its own is skipped (no row): the PR author's replies and status notes, and the
texts the repository profile's skip_rx names (a bot's overview of its own threads, a nudge about open threads).

What a script settles, with its evidence (the settler is "script"):
- data: an approval (its line in reviews.md); the Threads row (the thread states in threads.md);
- probe: an item whose labels (our review's N1, C2 ...) all have saved probes, by their verdicts when each is FIXED or
  REPRODUCES. A probe checks its own controls: its exit 2 (it cannot show its finding on this tree) and its errors read
  as ERROR, which settles nothing;
- unchanged: an item with a status from our last review whose cited files are as they were at that review's head, in
  the PR's own patch (each file's blob at the head then and now) and in develop's (its blob at the develop that review
  merged, else the merge base then, and at today's develop): its status carries. Not when a comment since that review
  names one of the item's labels, and not when an open item cites a file that cannot be found;
- closed: a row our last review closed that cites no file carries closed (a regression is the hunt's to find).
What is left goes to the model, one short task an item (drafter.py, phase "prior").

A status flips a 🔴/🟠 (`flips`) when a 🔴/🟠 opens (a new one, or one reopened), closes, or moves between 🔴 and 🟠
against our last review: such a row is one prior_settled.md asks the verifier to look at.
"""

from __future__ import annotations

import re

import review_tools as rt

LIGHTS = "🔴🟠🟡🟢⚪"
LIGHT_WORDS = {
    "red": "🔴",
    "orange": "🟠",
    "yellow": "🟡",
    "green": "🟢",
    "white": "⚪",
}
SERIOUS = ("🔴", "🟠")
OPEN_WORDS = ("OPEN", "DEFERRED", "PARTIAL", "NOT CHECKED", "NOT SETTLED", "PENDING")
DATA_ROWS = re.compile(
    r"^(Bots / CI|Our checks|Code quality|Reviewers|Threads)\b|^Approvals?\b", re.I
)
ROW = re.compile(r"^\|\s*(.+?)\s*\|\s*(.*\S)\s*\|\s*$")
FINDING_ROW = re.compile(
    r"^\|\s*([A-Z]{1,3}\d+[a-z]?)\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(.*\S)\s*\|\s*$"
)
LABEL = re.compile(
    r"\b([A-Z]{1,3})(\d{1,3})([a-z]?)\b(?:\s*[–-]\s*(?:\1)?(\d{1,3})\b)?"
)
EXTS = (
    "py",
    "pyi",
    "cpp",
    "cc",
    "cxx",
    "c",
    "h",
    "hpp",
    "hxx",
    "hh",
    "inc",
    "inl",
    "cmake",
    "txt",
    "md",
    "rst",
    "json",
    "yaml",
    "yml",
    "toml",
    "cfg",
    "ini",
    "sh",
    "ll",
    "mlir",
    "td",
    "kdp",
    "csv",
    "in",
    "hip",
    "cu",
    "cuh",
    "s",
    "jinja",
    "j2",
    "tmpl",
)
_EXT = "(?:" + "|".join(EXTS) + ")"
FILE_TOKEN = re.compile(
    r"^((?:[\w.+-]+/)*[\w.+-]*[A-Za-z][\w.+-]*\."
    + _EXT
    + r")(?::(\d+)(?:\s*[-–/]\s*\d+)*)?$"
)
BARE_PATH = re.compile(
    r"(?<![\w/.@`-])((?:[\w.+-]+/)+[\w.+-]*[A-Za-z][\w.+-]*\."
    + _EXT
    + r")(?::(\d+))?(?![\w/])"
)
TICKS = re.compile(r"`([^`\n]{1,300})`")
MAX_MATCHES = 5  # a file name that names more files than this cites none of them
THREAD = re.compile(r"(?ms)^## (\S+?):(\S+) \[([^\]]+)\]\n(.*?)(?=^## |\Z)")


def light_of(text) -> str:
    """The first severity light in a text, or the light a word names (red, orange, ...); "" for none."""
    t = str(text or "")
    m = re.search(f"[{LIGHTS}]", t)
    if m:
        return m.group(0)
    return LIGHT_WORDS.get(t.strip().lower(), "")


def is_open(light: str, status: str) -> bool:
    """Whether a status leaves its item open: OPEN, DEFERRED, PARTIAL, not checked; a light that is not green when the
    status says nothing."""
    s = (status or "").upper()
    if s:
        return s in OPEN_WORDS
    return light not in ("🟢", "⚪", "")


def is_closed(light: str, status: str) -> bool:
    return not is_open(light, status)


def flips(prev_light: str, prev_status: str, light: str, status: str) -> bool:
    """Whether a row's status flips a 🔴/🟠 against our last review's: one opens, closes, or moves between 🔴 and 🟠.
    With no earlier status (a new item), a 🔴/🟠 that is open flips."""
    was = prev_light in SERIOUS and is_open(prev_light, prev_status)
    now = light in SERIOUS and is_open(light, status)
    return was != now or (was and now and prev_light != light)


def status_of(cell: str) -> tuple[str, str, str, list[str]]:
    """A status cell of our review: its light, status word, sha and points."""
    parts = [p.strip() for p in re.split(r"<br>\s*•?\s*", cell or "")]
    first, points = parts[0], [p for p in parts[1:] if p]
    m = re.match(rf"([{LIGHTS}])\s*(.*)$", first)
    light, rest = (m.group(1), m.group(2)) if m else ("", first)
    w = re.match(r"(N/A|NOT CHECKED|NOT SETTLED|[A-Z]{3,})\b", rest)
    sha = re.search(r"`([0-9a-f]{7,40})`", rest)
    return light, (w.group(1) if w else ""), (sha.group(1) if sha else ""), points


def _bold(text: str) -> str:
    m = re.search(r"\*\*(.+?)\*\*", text or "")
    return m.group(1) if m else re.split(r"<br>", text or "")[0]


def parse_ours(text: str) -> dict | None:
    """pr/ours.md (`rk stage-ours`): our last review's head, `at` the time it was posted, or kept unposted while posting
    is off (then `kept`), the develop its merge was checked against, its prior rows and its findings, each with its line
    in the file; None without a review.
    """
    if "\n## Review\n" not in (text or ""):
        return None
    head = re.search(r"^- Head: ([0-9a-f]{7,40})\s*$", text, re.M)
    if not head:
        return None
    when = re.search(r"^- (Posted|Kept unposted): (\S+)", text, re.M)
    lines = text.split("\n")
    start = lines.index("## Review") + 1
    rows, findings, table = [], [], ""
    for i in range(start, len(lines)):
        ln = lines[i]
        if ln.startswith("| Prior item |"):
            table = "prior"
            continue
        if re.match(r"^\|\s*#\s*\|\s*Sev\s*\|", ln):
            table = "find"
            continue
        if not ln.startswith("|"):
            table = ""
            continue
        if re.match(r"^\|\s*:?-{3}", ln):
            continue
        if table == "prior" and (m := ROW.match(ln)):
            light, status, sha, points = status_of(m.group(2))
            rows.append(
                {
                    "line": i + 1,
                    "key": m.group(1),
                    "cell": m.group(2),
                    "light": light,
                    "status": status,
                    "sha": sha,
                    "points": points,
                    "raw": ln,
                }
            )
        elif table == "find" and (m := FINDING_ROW.match(ln)):
            parts = [p.strip() for p in re.split(r"<br>\s*•?\s*", m.group(4))]
            findings.append(
                {
                    "line": i + 1,
                    "id": m.group(1),
                    "light": light_of(m.group(2)),
                    "where": m.group(3).strip(),
                    "problem": _bold(parts[0]),
                    "points": [p for p in parts[1:] if p],
                    "text": m.group(4),
                    "raw": ln,
                }
            )
    body = "\n".join(lines[start:])
    dev = re.search(r"merge vs `([0-9a-f]{7,40})`", body)
    return {
        "head": head.group(1),
        "at": when.group(2) if when else "",
        "kept": bool(when) and when.group(1) == "Kept unposted",
        "rows": rows,
        "findings": findings,
        "develop": dev.group(1) if dev else "",
    }


def labels(text: str) -> list[str]:
    """The item labels a text names outside code spans (N12, C2, KH3; a range N1–N3 names each), in order."""
    t = TICKS.sub(" ", text or "")
    out = []
    for m in LABEL.finditer(t):
        pre, a, suf, b = m.groups()
        if b and not suf and 0 < int(b) - int(a) <= 40:
            out += [f"{pre}{k}" for k in range(int(a), int(b) + 1)]
        else:
            out.append(f"{pre}{a}{suf}")
    return list(dict.fromkeys(out))


def probe_label(name: str) -> str:
    """The item label a saved probe's name starts with: N12_abi_mirror is N12's, N2b_1102_x N2b's; "" for none."""
    m = re.match(r"([A-Z]{1,3})(\d{1,3})([a-z](?=_|$))?", name or "")
    return f"{m.group(1)}{m.group(2)}{m.group(3) or ''}" if m else ""


def citations(text: str) -> list[tuple[str, int]]:
    """The files a text cites, in code spans or as bare paths with a folder: (the name as written, its line or 0)."""
    t = re.sub(r"https?://\S+", " ", text or "")
    out = []
    for m in TICKS.finditer(t):
        span = m.group(1).strip()
        for cand in (span, span.split()[0].rstrip("),.;:") if span.split() else ""):
            fm = FILE_TOKEN.match(cand)
            if fm:
                out.append((fm.group(1), int(fm.group(2) or 0)))
                break
    for m in BARE_PATH.finditer(TICKS.sub(" ", t)):
        out.append((m.group(1), int(m.group(2) or 0)))
    seen, uniq = set(), []
    for tok, ln in out:
        if (tok, ln) not in seen:
            seen.add((tok, ln))
            uniq.append((tok, ln))
    return uniq


class Resolver:
    """A cited name to the repository paths it names: the PR's changed files first, then the head's files under the
    project path; a name with a folder matches a path that ends with it. More than MAX_MATCHES matches, or none:
    unresolved."""

    def __init__(self, changed: list[str], tree_files, project: str):
        self.changed = list(dict.fromkeys(changed))
        self._changed = set(self.changed)
        self._tree_files = tree_files
        self._tree: list[str] | None = None
        self._tree_set: set[str] = set()
        self.project = (project or "").strip("/")

    def tree(self) -> list[str]:
        if self._tree is None:
            self._tree = list(self._tree_files() or [])
            self._tree_set = set(self._tree)
        return self._tree

    def resolve(self, token: str) -> list[str]:
        tok = (token or "").strip().strip("/")
        while tok.startswith("./"):
            tok = tok[2:]
        if not tok:
            return []
        if "/" in tok:
            for cand in (tok, f"{self.project}/{tok}" if self.project else tok):
                if cand in self._changed:
                    return [cand]
                self.tree()
                if cand in self._tree_set:
                    return [cand]
            hits = {p for p in self.changed if p.endswith("/" + tok)} or {
                p for p in self.tree() if p.endswith("/" + tok)
            }
        else:
            hits = {p for p in self.changed if p.rsplit("/", 1)[-1] == tok} or {
                p for p in self.tree() if p.rsplit("/", 1)[-1] == tok
            }
        return sorted(hits) if 0 < len(hits) <= MAX_MATCHES else []


def unchanged(
    repo, then_head: str, head: str, dev_then: str, dev_now: str, paths: list[str]
) -> tuple[bool | None, dict]:
    """Whether every path is as it was at our last review's head: the same blob at the head then and now (the PR's own
    patch) and at the develop then and today (develop's). None when a commit is not in the clone. The record: the four
    commits and each path's four blobs, for the evidence check."""
    commits = (then_head, head, dev_then, dev_now)
    maps: dict[str, dict] = {}
    for c in commits:
        if c not in maps:
            m = rt.blobs(repo, c, list(paths))
            if m is None:
                return None, {"missing": c}
            maps[c] = m
    files = {p: [maps[c][p] for c in commits] for p in paths}
    same = bool(files) and all(v[0] == v[1] and v[2] == v[3] for v in files.values())
    return same, {
        "since": then_head,
        "head": head,
        "develop_then": dev_then,
        "develop": dev_now,
        "files": files,
    }


def changed_paths(rec: dict) -> list[str]:
    """The paths an `unchanged` record shows changed."""
    return [
        p
        for p, v in (rec.get("files") or {}).items()
        if not (v[0] == v[1] and v[2] == v[3])
    ]


def skip_reason(item: dict, author: str, skip_rx: dict) -> str:
    """Why an item holds no review point of its own ("" when it holds one): the PR author's own reply or status note,
    or a text the profile's skip_rx names ({why: regular expression})."""
    if (
        item.get("kind") in ("review", "thread", "comment")
        and author
        and str(item.get("who") or "").lower() == author.lower()
    ):
        return "the PR author's reply or status note"
    for why, rx in (skip_rx or {}).items():
        try:
            if re.search(rx, item.get("text") or "", re.I | re.M):
                return str(why)
        except re.error:
            continue
    return ""


def threads_state(text: str) -> dict:
    """threads.md's threads: how many, how many open, resolved and outdated, and the open ones (path, line, who)."""
    rows = []
    for m in THREAD.finditer(text or ""):
        who = re.search(r"^### @(\S+)", m.group(4), re.M)
        rows.append(
            {
                "path": m.group(1),
                "line": m.group(2),
                "resolved": m.group(3).startswith("RESOLVED"),
                "outdated": "outdated" in m.group(3),
                "who": who.group(1) if who else "",
            }
        )
    return {
        "total": len(rows),
        "open": sum(1 for r in rows if not r["resolved"]),
        "resolved": sum(1 for r in rows if r["resolved"]),
        "outdated": sum(1 for r in rows if r["outdated"]),
        "open_list": [r for r in rows if not r["resolved"]],
    }


def threads_value(state: dict) -> dict:
    """The part of threads_state a Threads row's evidence records and the evidence check compares."""
    return {k: state[k] for k in ("total", "open", "resolved", "outdated")}


def gist(text: str, cap: int = 60) -> str:
    """A short gist of an item's text: its first sentence, without markup."""
    t = re.sub(r"<!--.*?-->", " ", text or "", flags=re.S)
    t = re.sub(r"^#+\s*", "", t, flags=re.M)
    t = re.sub(r"\*\*|__|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", t)
    t = " ".join(t.split())
    first = re.split(r"(?<=[.!?])\s", t, maxsplit=1)[0]
    return first if len(first) <= cap else first[: cap - 1].rstrip() + "…"


def row_items(ours: dict) -> list[dict]:
    """Our last review's prior rows (not its data rows) and findings, as prior items: each with its key (kept as that
    review wrote it), its labels, the files it cites, and that review's status of it (prev: light, status, sha, points,
    the line in ours.md and the line itself)."""
    at = ours.get("at") or ""
    mmdd = f"{at[5:7]}-{at[8:10]}" if len(at) >= 10 else ""
    items = []
    for r in ours["rows"]:
        if DATA_ROWS.match(r["key"]) or r["status"] == "APPROVED":
            continue
        items.append(
            {
                "kind": "row",
                "who": r["key"].split()[0].rstrip(":,"),
                "date": mmdd,
                "at": at,
                "last": at,
                "state": "",
                "where": "",
                "key": r["key"],
                "text": f"{r['key']} — {r['light']} {r['status']}".strip()
                + "".join(f"; {p}" for p in r["points"]),
                "replies": [],
                "labels": labels(r["key"]),
                "cites": citations(r["key"] + " " + r["cell"]),
                "prev": {
                    "light": r["light"],
                    "status": r["status"],
                    "sha": r["sha"],
                    "points": r["points"],
                    "line": r["line"],
                    "quote": r["raw"],
                },
            }
        )
    for f in ours["findings"]:
        key = (
            f"Ours ({mmdd}) {f['id']}: {f['problem']}"
            if mmdd
            else f"Ours {f['id']}: {f['problem']}"
        )
        cell = (
            f["where"] if "`" in f["where"] else f"`{f['where']}`"
        )  # its file:line cell may name several files
        cites = citations(cell)
        items.append(
            {
                "kind": "finding",
                "who": "Ours",
                "date": mmdd,
                "at": at,
                "last": at,
                "state": "",
                "where": (
                    f"{cites[0][0]}:{cites[0][1]}" if cites and cites[0][1] else ""
                ),
                "key": key[:150],
                "text": f"{f['id']} {f['light']} {cell} {f['problem']}"
                + "".join(f"; {p}" for p in f["points"]),
                "replies": [],
                "labels": [f["id"]],
                "cites": cites + citations(f["text"]),
                "prev": {
                    "light": f["light"],
                    "status": "OPEN",
                    "sha": "",
                    "points": f["points"],
                    "line": f["line"],
                    "quote": f["raw"],
                },
            }
        )
    return items
