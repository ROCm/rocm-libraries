# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""References to moved or deleted files, in every review's checks. A PR that
moves or deletes a file leaves references to its old path behind. A build or a test catches a broken include or
import; docs, CMake lists and YAML keep naming the old path unseen (a README that still links a doc the PR moved to
another directory).

**The files**: each file the PR deletes or moves against merge-base(develop, head) (`git diff -M`).

**The references**, at the head, in the provider's tree and .github: the files that hold a moved file's name or a
moved module's dotted name, and the Python files that import a moved module's stem (`git grep`), read as:
- docs (.md, .rst, .txt, .html): links ([text](target), [id]: target, href/src, an RST link's target in angle
  brackets), which name a path from the doc's directory, and path-like tokens (a `/` and an extension, as in a
  backticked path), which name a path from the doc's directory or the tail of any path; a moved module's dotted name
  (`python -m rocke.x.y`);
- code, CMake, YAML and scripts: C/C++ includes, Python imports and path strings, as the clash scan reads them
  (rkmon_lib/clashes.refs; outside Python, a `${VAR}/` before a path is read past, and a CMake file's own directory,
  `${CMAKE_CURRENT_SOURCE_DIR}/` in a CMakeLists.txt or `${CMAKE_CURRENT_LIST_DIR}/`, names the path from there
  alone), and a moved module's dotted name in a Python string (a `mock.patch` target).
A reference that names a moved or deleted file and resolves to no file at the head is reported: file:line, the
reference, and where the file went. One that still resolves (an include's tail under another include root, a path
that exists) is not: the build and the clash scan judge those. Changelogs and history notes (CHANGELOG, CHANGES,
HISTORY, NEWS, RELEASE_NOTES) and backup files (*.orig, *.rej, *.bak) are left out.

**Changed docs**: the links in each doc the PR adds, changes or moves (Markdown, RST, HTML) whose target is missing at
the head, outside fenced code; URLs, anchors and templated targets aside.

Usage: python3 -I moved_refs.py --repo R --head SHA [--base REV] [--rocke-path K] [--json FILE] [--budget 120]
Prints the section, then one last line: "MOVEDREFS pass|fail|skip <detail>".
"""
import argparse
import bisect
import json
import os
import pathlib
import posixpath
import re
import subprocess
import sys
import time
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
from rkmon_lib import clashes, gitops, overlaps, prtree  # noqa: E402

BUDGET_S = 120
SHOWN = 40  # references listed, then broken links
CANDIDATES = 3000  # files read at most (a generic file name can be in thousands)
DOCS = (".md", ".rst", ".txt", ".html", ".htm")
LINK_DOCS = (".md", ".rst", ".html", ".htm")
HISTORY = re.compile(
    r"(?i)(^|/)(changelog|changes|history|news|release[-_]?notes)(\.[^/]*)?$"
)
BACKUP = re.compile(r"\.(orig|rej|bak|swp)$|~$")
GENERIC = {
    "__init__.py",
    "__main__.py",
}  # names in nearly every file's neighbourhood: the dotted name finds them
FENCE = re.compile(r"^[ \t]{0,3}(```|~~~)[^\n]*\n.*?^[ \t]{0,3}\1[ \t]*$", re.M | re.S)
LINKS = (
    re.compile(
        r"\]\(\s*<?([^()<>\s]+)>?(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\s*\)"
    ),  # [text](target "title")
    re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*<?([^\s>]+)>?", re.M),  # [id]: target
    re.compile(
        r"\b(?:href|src)[ \t]*=[ \t]*[\"']([^\"'\n]+)[\"']", re.I
    ),  # <a href="target">
    re.compile(r"`[^`<\n]*<([^`>\n]+)>`__?"),
)  # RST `text <target>`_
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:|^//")
TEMPLATED = re.compile(r"[{}$<>*]|\.\.\.")
DOTTED = re.compile(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
# A variable before a path (`${_ROCKE_LIBRARY_DIR}/tests/x.py`, `${ROOT}/x`): blanked, so the path after it reads as a
# token (from the file's directory, or a tail anywhere). The file's own directory (`${CMAKE_CURRENT_SOURCE_DIR}/` in a
# CMakeLists.txt, `${CMAKE_CURRENT_LIST_DIR}/` in any CMake file) reads as `./`: such a path resolves from there alone.
VARIABLE = re.compile(r"\$\{([^}\n]*)\}/?")


def is_doc(path: str) -> bool:
    return (
        path.endswith(DOCS) and overlaps.language(path) != "cmake"
    )  # CMakeLists.txt is code


def unvar(path: str, text: str) -> str:
    """The text with each `${VAR}/` before a path blanked (line and column kept), the file's own directory as `./`."""
    own = (
        ("CMAKE_CURRENT_SOURCE_DIR", "CMAKE_CURRENT_LIST_DIR")
        if path.endswith("CMakeLists.txt")
        else ("CMAKE_CURRENT_LIST_DIR",) if overlaps.language(path) == "cmake" else ()
    )

    def blank(m: re.Match) -> str:
        s = m.group()
        return (
            " " * (len(s) - 2) + "./"
            if m.group(1) in own and s.endswith("/")
            else " " * len(s)
        )

    return VARIABLE.sub(blank, text)


# --- what a file names ------------------------------------------------------------------------------------------------


def link_targets(text: str) -> list[tuple[int, str]]:
    """(line, target) of each link in a doc; fenced code blocks are not read (their lines keep their numbers)."""
    masked = FENCE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), text)
    starts = clashes._starts(masked)
    return sorted(
        {
            (clashes._line_of(starts, m.start(1)), m.group(1))
            for rx in LINKS
            for m in rx.finditer(masked)
        }
    )


def link_path(doc: str, target: str) -> str | None:
    """The repository path a link names: from the doc's directory, or from the repository root for `/x`; None for a
    URL, an anchor, a templated target, or one that leaves the repository."""
    t = target.strip()
    if not t or t.startswith("#") or SCHEME.match(t) or TEMPLATED.search(t):
        return None
    t = urllib.parse.unquote(t.split("#", 1)[0].split("?", 1)[0])
    if not t:
        return None
    p = posixpath.normpath(
        t.lstrip("/")
        if t.startswith("/")
        else posixpath.join(posixpath.dirname(doc), t)
    )
    return None if p.startswith("..") or p == "." else p


def doc_refs(path: str, text: str, lines: set[int]) -> list[dict]:
    """A doc's references on the given lines: links (the exact path they name) and path-like tokens (the path from
    the doc's directory, or a tail anywhere), as clashes.refs spells references."""
    out, here = [], posixpath.dirname(path)
    linked = set()
    if path.endswith(LINK_DOCS):
        for ln, target in link_targets(text):
            p = link_path(path, target) if ln in lines else None
            if p:
                linked.add((ln, target))
                out.append(
                    {
                        "kind": "link",
                        "ref": target,
                        "line": ln,
                        "exact": [p],
                        "tails": [],
                    }
                )
    rows = text.split("\n")
    for ln in sorted(lines):
        line = rows[ln - 1] if ln <= len(rows) else ""
        for m in clashes.PATH_TOKEN.finditer(line):
            tok = m.group()
            if (
                (ln, tok) in linked
                or not clashes.HAS_EXT.search(tok)
                or "://" in line[max(0, m.start() - 3) : m.start()]
            ):
                continue
            tail = re.sub(r"^(?:\.{1,2}/)+", "", tok)
            exact = clashes._norm(posixpath.join(here, tok))
            out.append(
                {
                    "kind": "path",
                    "ref": tok,
                    "line": ln,
                    "exact": [exact] if exact else [],
                    "tails": [tail] if "/" in tail else [],
                }
            )
    return out


def module_refs(
    path: str, text: str, lines: set[int], modules: dict[str, dict]
) -> list[dict]:
    """Dotted names of moved modules on the given lines (a doc's `python -m rocke.x`, a Python string's mock.patch
    target): each with the module it names."""
    out, rows = [], text.split("\n")
    for ln in sorted(lines):
        line = rows[ln - 1] if ln <= len(rows) else ""
        for m in DOTTED.finditer(line):
            name = m.group()
            for mod in modules:
                if name == mod or name.startswith(mod + "."):
                    out.append(
                        {
                            "kind": "module",
                            "ref": name,
                            "line": ln,
                            "module": mod,
                            "exact": [],
                            "tails": clashes._module_paths(mod),
                        }
                    )
    return out


# --- the head ---------------------------------------------------------------------------------------------------------


def grep_files(
    repo: str,
    head: str,
    areas: list[str],
    fixed: list[str],
    imports: re.Pattern | None,
    timeout: float,
) -> list[str]:
    """The files at `head` under `areas` that hold one of the `fixed` strings, or a Python import line that `imports`
    matches (its stems), in one `git grep` each (a blob the clone lacks is fetched)."""
    env = {k: v for k, v in os.environ.items() if k != "GIT_NO_LAZY_FETCH"}
    found: set[str] = set()
    runs = []
    if fixed:
        runs.append(["-F", *[x for w in fixed for x in ("-e", w)], head, "--", *areas])
    if imports is not None:
        runs.append(
            [
                "-E",
                "-e",
                imports.pattern_posix,
                head,
                "--",
                *[f"{a}/*.py" for a in areas],
            ]
        )
    for args in runs:
        p = subprocess.run(
            [
                "git",
                "-C",
                repo,
                "-c",
                "core.quotepath=false",
                "grep",
                "-l",
                "-I",
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if p.returncode not in (0, 1):
            raise prtree.NotRun(
                f"git grep at the head failed: {p.stderr.strip()[:160]}"
            )
        found.update(
            line.split(":", 1)[1] for line in p.stdout.splitlines() if ":" in line
        )
    return sorted(found)


class ImportRx:
    """A Python import line naming one of the stems: the Python pattern, and its POSIX spelling for git grep."""

    def __init__(self, stems: list[str]):
        alt = "|".join(re.escape(s) for s in stems)
        self.rx = re.compile(rf"^\s*(?:from|import)\s.*\b(?:{alt})\b")
        self.pattern_posix = (
            rf"^[[:space:]]*(from|import)[[:space:]].*({'|'.join(stems)})"
        )

    def search(self, line: str):
        return self.rx.search(line)


def lines_with(
    text: str, fixed: re.Pattern | None, imports: ImportRx | None
) -> set[int]:
    out = set()
    for i, line in enumerate(text.split("\n"), 1):
        if (fixed is not None and fixed.search(line)) or (
            imports is not None and imports.search(line)
        ):
            out.add(i)
    return out


class Paths:
    """The head's paths (clashes.Tree), and its directories for a link that names one."""

    def __init__(self, repo: str, head: str):
        self.tree = clashes.Tree(repo, head, set())
        self.sorted = sorted(self.tree.paths())

    def exists(self, p: str) -> bool:
        if p in self.tree.paths():
            return True
        i = bisect.bisect_left(self.sorted, p.rstrip("/") + "/")
        return i < len(self.sorted) and self.sorted[i].startswith(p.rstrip("/") + "/")


# --- the check --------------------------------------------------------------------------------------------------------


def _show(path: str, K: str) -> str:
    return path[len(K) + 1 :] if path.startswith(K + "/") else path


def check(
    repo: str,
    head: str,
    K: str,
    base: str = prtree.DEVELOP,
    budget: float = BUDGET_S,
) -> dict:
    t0 = time.monotonic()

    def left() -> float:
        return max(10.0, budget - (time.monotonic() - t0))

    res: dict = {
        "head": head,
        "error": None,
        "moved": [],
        "stale": [],
        "broken": [],
        "docs": 0,
        "links": 0,
    }
    try:
        res["mb"] = mb = prtree.merge_base(repo, base, head)
        cs = prtree.changes(repo, mb, head, timeout=int(left()))
        moved = [c for c in cs if c["status"] in ("D", "R")]
        docs = [
            c["new"]
            for c in cs
            if c["status"] in ("A", "M", "R")
            and c["new"].endswith(LINK_DOCS)
            and not HISTORY.search(c["new"])
            and not BACKUP.search(c["new"])
        ]
        res["moved"] = [[c["old"], c["new"]] for c in moved]
        res["docs"] = len(docs)
        if not moved and not docs:
            res["secs"] = round(time.monotonic() - t0, 1)
            return res
        provider = K.rsplit("/", 1)[0]
        areas = prtree.present(repo, head, [provider, ".github"])
        modules = {
            m: c for c in moved if (m := prtree.module_name(c["old"], K)) and "." in m
        }
        fixed = sorted(
            {posixpath.basename(c["old"]) for c in moved} - GENERIC | set(modules)
        )
        stems = sorted(
            {m.rsplit(".", 1)[1] for m in modules} - {"__init__", "__main__"}
        )
        imports = ImportRx(stems) if stems else None
        files = (
            grep_files(repo, head, areas, fixed, imports, left())
            if moved and areas
            else []
        )
        files = [f for f in files if not HISTORY.search(f) and not BACKUP.search(f)]
        res["candidates"] = len(files)
        files = files[:CANDIDATES]
        texts = gitops.cat_blobs(
            repo,
            [f"{head}:{f}" for f in sorted(set(files) | set(docs))],
            timeout=int(left()),
        )
        paths = Paths(repo, head)
        by_base: dict[str, list[dict]] = {}
        for c in moved:
            by_base.setdefault(posixpath.basename(c["old"]), []).append(c)
        fixed_rx = re.compile("|".join(re.escape(w) for w in fixed)) if fixed else None
        seen = set()
        for f in files:
            text = texts.get(f"{head}:{f}")
            if text is None:
                continue
            py = overlaps.language(f) == "py"
            hit = lines_with(text, fixed_rx, imports if py else None)
            if not hit:
                continue
            if is_doc(f):
                refs = doc_refs(f, text, hit)
            elif py:
                refs = clashes.refs(f, text, [(1, 10**9)])
            else:
                refs = [
                    (
                        dict(r, tails=[])
                        if overlaps.language(f) == "cmake" and r["ref"].startswith("./")
                        else r
                    )
                    for r in clashes.refs(
                        f, unvar(f, text), [(ln, ln) for ln in sorted(hit)]
                    )
                ]
            refs += module_refs(f, text, hit, modules) if (py or is_doc(f)) else []
            for r in refs:
                named = (
                    [modules[r["module"]]]
                    if r["kind"] == "module"
                    else [
                        c
                        for cand in r["exact"] + r["tails"]
                        for c in by_base.get(posixpath.basename(cand), ())
                        if c["old"] == cand
                        or (cand in r["tails"] and c["old"].endswith("/" + cand))
                    ]
                )
                if not named or paths.tree.resolve(r["exact"], r["tails"]):
                    continue
                c = named[0]
                key = (f, r["line"], c["old"])
                if key not in seen:
                    seen.add(key)
                    res["stale"].append(
                        {
                            "path": f,
                            "line": r["line"],
                            "kind": r.get("kind") or "path",
                            "ref": r["ref"],
                            "old": c["old"],
                            "new": c["new"],
                        }
                    )
        for d in docs:
            text = texts.get(f"{head}:{d}")
            for ln, target in link_targets(text) if text else []:
                p = link_path(d, target)
                if (
                    p
                    and not paths.exists(p)
                    and not any(
                        s["path"] == d and s["line"] == ln for s in res["stale"]
                    )
                ):
                    res["broken"].append(
                        {"path": d, "line": ln, "ref": target, "target": p}
                    )
            res["links"] += len(link_targets(text)) if text else 0
    except (
        prtree.NotRun,
        gitops.GitError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as exc:
        res["error"] = str(exc)[:300]
    order = {"include": 0, "import": 0, "module": 1, "path": 2, "link": 3}
    res["stale"].sort(
        key=lambda s: (is_doc(s["path"]), order.get(s["kind"], 2), s["path"], s["line"])
    )
    res["secs"] = round(time.monotonic() - t0, 1)
    return res


def _where(s: dict, K: str) -> str:
    return f"{_show(s['path'], K)}:{s['line']}"


def verdict(res: dict, K: str) -> tuple[str, str]:
    if res.get("error"):
        return "skip", f"NOT RUN: {res['error'][:180]}"
    if not res["moved"] and not res["docs"]:
        return (
            "skip",
            "nothing to check: the PR moves or deletes no file and changes no doc",
        )
    stale, broken = res["stale"], res["broken"]
    capped = res.get("candidates", 0) > CANDIDATES
    if stale or broken:
        parts = []
        if stale:
            parts.append(
                f"{len(stale)} reference{'s' if len(stale) > 1 else ''} name moved or deleted files ("
                + ", ".join(_where(s, K).rsplit("/", 1)[-1] for s in stale[:3])
                + (f" +{len(stale) - 3} more" if len(stale) > 3 else "")
                + ")"
            )
        if broken:
            parts.append(
                f"{len(broken)} broken link{'s' if len(broken) > 1 else ''} in changed docs ("
                + ", ".join(_where(b, K).rsplit("/", 1)[-1] for b in broken[:2])
                + (f" +{len(broken) - 2} more" if len(broken) > 2 else "")
                + ")"
            )
        return "fail", "; ".join(parts) + (
            "; part NOT RUN (the file cap)" if capped else ""
        )
    if capped:  # files the cap left unread never read clean
        return (
            "skip",
            f"NOT RUN in part: {res['candidates'] - CANDIDATES} of the files that name one were not read",
        )
    n_del = sum(1 for _, new in res["moved"] if new is None)
    return "pass", (
        f"no reference names a moved or deleted file ({len(res['moved']) - n_del} moved, {n_del} "
        f"deleted); {res['links']} links in {res['docs']} changed docs resolve"
    )


def render(res: dict, K: str) -> list[str]:
    if res.get("error"):
        return [f"NOT RUN: {res['error']}", " ".join(("MOVEDREFS",) + verdict(res, K))]
    n_del = sum(1 for _, new in res["moved"] if new is None)
    out = [
        f"the PR moves {len(res['moved']) - n_del} files and deletes {n_del} (merge base "
        f"{res.get('mb', '?')[:10]}); {res.get('candidates', 0)} files at the head name one; {res['docs']} docs "
        f"changed, {res['links']} links ({res.get('secs', 0)} s)"
    ]
    if res.get("candidates", 0) > CANDIDATES:
        out.append(
            f"read the first {CANDIDATES} of the {res['candidates']} files that name one (cap); the rest are "
            "NOT RUN"
        )
    if res["stale"]:
        out.append(
            "references at the head that name a moved or deleted file and resolve to no file (file:line, the "
            "reference, where the file went):"
        )
        for s in res["stale"][:SHOWN]:
            went = f"moved to {_show(s['new'], K)}" if s["new"] else "deleted"
            out.append(f"  {_where(s, K)} `{s['ref']}` ({s['kind']}): {went}")
        if len(res["stale"]) > SHOWN:
            out.append(f"  +{len(res['stale']) - SHOWN} more (moved_refs.json)")
    if res["broken"]:
        out.append(
            "links in the docs the PR adds, changes or moves whose target is missing at the head:"
        )
        for b in res["broken"][:SHOWN]:
            out.append(
                f"  {_where(b, K)} `{b['ref']}` -> {_show(b['target'], K)} (missing)"
            )
        if len(res["broken"]) > SHOWN:
            out.append(f"  +{len(res['broken']) - SHOWN} more (moved_refs.json)")
    out.append(" ".join(("MOVEDREFS",) + verdict(res, K)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--base", default=prtree.DEVELOP)
    ap.add_argument("--rocke-path", default="", type=prtree.folder_argument)
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    ap.add_argument("--json", default=None, help="also write the result to this file")
    a = ap.parse_args(argv)
    res = check(a.repo, a.head, a.rocke_path, a.base, a.budget)
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    print("\n".join(render(res, a.rocke_path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
