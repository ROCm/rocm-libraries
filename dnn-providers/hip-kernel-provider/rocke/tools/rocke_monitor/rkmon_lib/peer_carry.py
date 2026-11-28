# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A merge-only head's item flags against what our last review saw.

Two kinds of flag name items a develop merge leaves as they were:
- the `threads` and `title` peer checks read PR-level state (the live review threads no owner answered; the PR title);
- the `tests` and `installed` checks name the failing test node ids, and the `prnumbers` peer check names each
  pull-request reference by file:line (CARRIED).
The checks write their items beside checks.md (ITEMS; the tests' tests/tests_pr_failed.txt and
tests/tests_develop_unrun.txt; the installed tests' installed/results.json), and each finished review's checks record
them for their head (`record`, rec["peer_seen"]). A merge-only head whose only flags are these (beside CI failures that
are known infrastructure) is judged item by item against the items at a head our review covers (`baseline`):
- every flagged thread (by its first comment's url) and the flagged title were flagged there too, so our review statused
  them: the head is carried like a clean merge-only head; a thread or a title problem that was not: a micro round, the
  verifier alone on the new items (REVIEW_READY's `micro:`), with no code quality pass;
- every failing test (by node id) and every pull-request reference (by file and text, so a moved line still matches)
  was flagged there too, or our last review's open rows name it (`review_rows`): the head is carried; one that was
  not leaves its flag a full flag, and the head goes to verification.
An item list that cannot be read (checks from before the lists were written, a check that did not run) leaves its flag
a full flag: the head goes to verification as before.
"""

from __future__ import annotations

import json
import pathlib
import re

from . import daylog
from .snapshot import covers

ITEMS = {
    "threads": "peer_threads.json",
    "title": "peer_title.json",
    "prnumbers": "peer_prnumbers.json",
}  # check -> its items file in the checks folder
MICRO = ("threads", "title")  # new items make a micro round
CARRIED = ("tests", "installed", "prnumbers")  # new items leave the flag a full flag
KEYS = MICRO + CARRIED
TESTS_FILES = (
    "tests/tests_pr_failed.txt",
    "tests/tests_develop_unrun.txt",
)  # tools/pytest_vs_base.py, in the folder the checks copy its results to: the ids its fail names
INSTALLED_RESULTS = (
    "installed/results.json"  # tools/installed_tests.py: each failing id's label
)
INSTALLED_PASS = (
    "develop"  # the label of a failure develop has too, which flags nothing
)


def _peer(out_dir, key: str):
    """A peer check's result in its items file; None when it is missing, unreadable or the check did not run."""
    try:
        res = json.loads(
            (pathlib.Path(out_dir) / ITEMS[key]).read_text(encoding="utf-8")
        )[key]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return res if isinstance(res, dict) and not res.get("error") else None


def _tests(out_dir) -> list[str] | None:
    ids: set[str] = set()
    for name in TESTS_FILES:
        try:
            ids |= {
                ln.strip()
                for ln in (pathlib.Path(out_dir) / name)
                .read_text(encoding="utf-8")
                .splitlines()
                if ln.strip()
            }
        except OSError:
            if name == TESTS_FILES[0]:
                return None
    return sorted(ids)


def _installed(out_dir) -> list[str] | None:
    try:
        labels = json.loads(
            (pathlib.Path(out_dir) / INSTALLED_RESULTS).read_text(encoding="utf-8")
        )["labels"]
        return sorted(
            {
                key.partition(" ")[2] or key
                for key, lab in labels.items()
                if lab[0] != INSTALLED_PASS
            }
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None


def read(out_dir) -> dict:
    """The checks folder's items: {"threads": [{"url", "reviewer", "where"}, ...], "title": {"title", "why"} when
    flagged, {} when not, "tests" and "installed": [node id, ...], "prnumbers": [{"at", "what"}, ...]}; None for a
    check whose list is missing or that did not run."""
    out: dict = {}
    for key in ("threads", "title", "prnumbers"):
        res = _peer(out_dir, key)
        items = (res or {}).get("items") or []
        if res is None:
            out[key] = None
        elif key == "threads":
            out[key] = [
                {
                    "url": t.get("url") or "",
                    "reviewer": t.get("reviewer") or "?",
                    "where": t.get("where") or "-",
                }
                for t in items
            ]
        elif key == "title":
            out[key] = (
                {"title": res.get("title") or "", "why": items[0].get("why") or ""}
                if items
                else {}
            )
        else:
            out[key] = [
                {"at": x.get("at") or "", "what": x.get("what") or ""} for x in items
            ]
    out["tests"], out["installed"] = _tests(out_dir), _installed(out_dir)
    return out


def _ref(item: dict) -> list[str]:
    """A pull-request reference as state keeps it: [file, text], the line number left out."""
    return [item["at"].rpartition(":")[0] or item["at"], item["what"]]


def flagged(items: dict) -> dict:
    """What `read` gives, as state keeps it: the flagged threads' urls, the title when flagged ("" when not), the
    failing test ids, and the pull-request references as [file, text]; None where unknown.
    """
    threads, title, refs = (
        items.get("threads"),
        items.get("title"),
        items.get("prnumbers"),
    )
    return {
        "threads": None if threads is None else sorted(t["url"] for t in threads),
        "title": None if title is None else title.get("title", ""),
        "tests": items.get("tests"),
        "installed": items.get("installed"),
        "prnumbers": None if refs is None else sorted(_ref(x) for x in refs),
    }


def _head(out_dir) -> str:
    try:
        return (pathlib.Path(out_dir) / "head_sha").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def record(rec: dict, out_dir, head: str | None) -> None:
    """A review's finished checks at `head`: their items (rec["peer_seen"])."""
    if head:
        rec["peer_seen"] = {"head": head, **flagged(read(out_dir))}


def baseline(rec: dict, out_dir) -> dict | None:
    """The items at a head our review covers: the last recorded, else the checks folder's when they ran at such a head
    (read before a new review's staging wipes it). None when neither is known; a key None when that head's record
    lacks it."""
    seen = rec.get("peer_seen") or {}
    if seen.get("head") and covers(rec.get("ours"), seen["head"]):
        return {k: seen.get(k) for k in KEYS}
    head = _head(out_dir)
    if head and covers(rec.get("ours"), head):
        return flagged(read(out_dir))
    return None


def review_rows(body: str) -> list[str]:
    """A review's open finding rows (🔴, 🟠 and 🟡), each as its where and text, and the prior items its status table
    still calls OPEN."""
    rows = [
        f"{where} {text}"
        for _k, _sev, where, text in daylog.FINDING.findall(body or "")
    ]
    return rows + [
        f"{item} {rest}" for item, _sev, rest in daylog.STILL_OPEN.findall(body or "")
    ]


def _same_file(cited: str, path: str) -> bool:
    """A path a row cites (with or without its :line) and an item's path name one file, one path relative to a
    subtree of the other (a review names rocKE files from the repo root)."""
    cited = re.sub(r":\d+$", "", cited)
    return cited == path or cited.endswith("/" + path) or path.endswith("/" + cited)


def _test_in(tid: str, rows: list[str]) -> bool:
    """A failing test id a row names: its file's name and its test (the id's last part, parameters and all)."""
    path, _, rest = tid.partition("::")
    name = rest.rpartition("::")[2]
    return bool(name) and any(path.rpartition("/")[2] in r and name in r for r in rows)


def _ref_in(ref: list[str], rows: list[str]) -> bool:
    """A pull-request reference a row names: its file (where the row cites it) and the reference itself."""
    m = re.match(r"`([^`]+)`", ref[1])
    what = m.group(1) if m else ref[1]
    for r in rows:
        if what in r and any(
            _same_file(p, ref[0]) for p in re.findall(r"[\w./-]+\.\w+(?::\d+)?", r)
        ):
            return True
    return False


def _carried(key: str, now: list, was: list | None, rows) -> bool:
    """Every item of a CARRIED check flagged at the baseline (`was`) or named by our last review's open rows (`rows`:
    () -> them)."""
    if key == "prnumbers":
        known = {tuple(x) for x in was or []}
        return all(_ref_in(_ref(x), rows()) for x in now if tuple(_ref(x)) not in known)
    known = set(was or [])
    return all(_test_in(t, rows()) for t in now if t not in known)


def judge(
    status: dict, now: dict, base: dict | None, ours=None
) -> tuple[list[str], list[str]]:
    """The failing checks (`status`: section -> (verdict, detail)) these items account for, and the threads and title
    items not flagged at the baseline, as text. A failing check whose items are unknown, or empty, is not accounted
    for: it stays a full flag; so does a CARRIED check with an item neither the baseline nor our last review's open
    rows (`ours`: () -> its body, read only when needed) name."""
    settled: list[str] = []
    new: list[str] = []
    cache: dict = {}

    def rows() -> list[str]:
        if "rows" not in cache:
            cache["rows"] = review_rows(ours() if ours else "")
        return cache["rows"]

    for key in KEYS:
        if (status.get(key) or ("",))[0] != "fail" or not now.get(key):
            continue
        was = (base or {}).get(key)
        if key in CARRIED:
            if not _carried(key, now[key], was, rows):
                continue
        elif key == "threads":
            known = set(was or [])
            new += [
                f"thread {t['reviewer']} at {t['where']} {t['url']}".rstrip()
                for t in now[key]
                if not t["url"] or t["url"] not in known
            ]
        elif not was or was != now[key]["title"]:
            new.append(f"title {now[key]['title']!r}: {now[key]['why']}")
        settled.append(key)
    return settled, new
