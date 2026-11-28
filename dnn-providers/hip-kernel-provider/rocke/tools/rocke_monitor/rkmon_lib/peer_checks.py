# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Zero-inference checks from what peer reviewers catch and our reviews did not raise first (peer_reviewer_gap.md), each
its own section of a review's checks. Git, the PR's staged metadata and GitHub reads only; read-only.

Flags (a fail counts like the other check flags: a merge-only head that has one goes to verification):
- threads: the PR's live review threads no owner answered: unresolved, not outdated, started by someone other than the
  PR's owners (its author and its commits' authors) and not by a bot (Copilot counts), with no comment from an owner;
- trackers: tracker IDs of any prefix in added lines (`<KEY>-<n>`, of any project): the repo keeps
  internal tracker references out of committed artifacts. Names of a standard or a register that look like one
  (`IEEE-754`, rocKE's semantic register IDs `SEM-<n>`) are allowed (config `tracker_allow`); data files (.json,
  .csv, …) are not read;
- envdocs: a `ROCKE_*` environment variable that non-test code newly reads (getenv, environ.get, environ[...], an
  env-reading helper) and that `platform/dsl_docs/reference/env_flags.md` at the head does not name (the PR's own edit
  of it counts; a documented `ROCKE_X_*` family covers its members); a name the merge base's code has is not new;
- testlists: a new `library/tests/**/test_*.py` that no CMakeLists.txt, .cmake or YAML file under the provider names,
  by file name or by a directory above it (the library's tests run only from those lists, so no CI lane runs it);
- cppenv: a `getenv` call in added C/C++ lines outside tests (rule_scan reads Python only);
- title: a PR title that is not a Conventional Commits title, `type(scope): subject` (platform/AGENTS.md: TheRock PR
  bot policy);
- splittaut: a string literal that holds the separator tested against split()'s pieces (`"_dg" not in s.split("_")`:
  always true, so the assertion cannot fail);
- sibimports: a test's bare import of a sibling test module (`from test_x import …`) when neither the file nor a
  conftest.py beside it puts the directory on sys.path (pytest's importlib mode does not);
- workload: M<m>N<n>K<k> sizes in an added kernel name (name lines, golden files): name by operation, not workload;
- prnumbers: pull-request references in added lines of any non-data file, code and docs (`#1234`, `PR #567`,
  `rocm-libraries#123`, `/pull/N`): the operator's repo rule, the repo states generalized facts, never a ledger;
- perfnotes: a measured timing or speedup in an added source comment or docstring (`~18 s`, `3.5% faster`,
  `<n> GB/s`, `well under a second`; platform/AGENTS.md: source comments stay qualitative); a parameter (a timeout, a
  default, a gap, a bound) is not one; tests are not read;
- backedges: a platform file newly reaching into library/ (an import of a library package, a sys.path edit, a
  ../library/ path) at a target it did not reach at the merge base;
- abiversion: a dispatch family's kernarg or signature code lines changed in a file whose ABI string (`abi_version`,
  `*_ABI*`) the PR leaves untouched;
- arraywrites: a fixed C array written through a running index at more sites than it has slots, more so than at the
  merge base (a moved file is compared with its old path), with no bound check of the index;
- nanblind: an error-vs-tolerance failure test NaN passes (`if err >= tol:`, `(err > tol).sum()`), no isnan nearby;
- optimports: a test module importing an optional package (ml_dtypes, scipy, …) at module level with no importorskip;
- mdheadings: a numbered Markdown heading the PR adds whose number repeats at its level.
Leads (never flags):
- peerleads: a request field forced to bool (`bool(req.x)`, `bool(kwargs.get(...))`, `bool(params[...])`: Rule 4,
  refuse a bad value, never normalize it; `int(...)`/`float(...)` of typed fields are everywhere and are not leads) in
  non-test library code; torch device queries in non-test library code (`torch.cuda.get_device_properties`,
  `torch.version.hip`, …); a CMake `file(GLOB …)` without CONFIGURE_DEPENDS anywhere in the PR (a glob at install
  time may mean it);
- paritypair: a shared lowerer (`core/lower_*.py`, `cpp/core/lower_*/`) changed with no parity `*_emit.{py,c}` pair in
  the PR (whether the change needs one is the verifier's call);
- unpack: a tuple-unpack target in an added line its function never reads.

tools/peer_checks.py runs one check (or all) and prints its section, then one last line "<TAG> pass|fail|skip
<detail>"; the review's checks run each in its own section (review.checks_script)."""
from __future__ import annotations

import ast
import collections
import json
import pathlib
import re
import subprocess
import time
import types

from . import gitops, prtree, registry

SHOWN = 25
# check -> (status tag, status key, section title, lead)
CHECKS = {
    "threads": (
        "THREADS",
        "threads",
        "Unanswered review threads (live: unresolved, not outdated, no owner reply)",
        False,
    ),
    "trackers": (
        "TRACKERS",
        "trackers",
        "Tracker IDs in added lines (any prefix)",
        False,
    ),
    "envdocs": ("ENVDOCS", "envdocs", "ROCKE_* env reads env_flags.md lacks", False),
    "testlists": (
        "TESTLISTS",
        "testlists",
        "New library tests in no CMake or CTest list",
        False,
    ),
    "cppenv": ("CPPENV", "cppenv", "C/C++ getenv reads in added lines", False),
    "title": ("TITLE", "title", "PR title (Conventional Commits)", False),
    "splittaut": (
        "SPLITTAUT",
        "splittaut",
        "Split tautologies (a literal that contains the separator, tested against the "
        "pieces of split())",
        False,
    ),
    "sibimports": (
        "SIBIMPORTS",
        "sibimports",
        "Bare sibling test imports (pytest --import-mode=importlib, no sys.path insert)",
        False,
    ),
    "workload": (
        "WORKLOAD",
        "workload",
        "Workload sizes in kernel names (such as M256N128K64)",
        False,
    ),
    "prnumbers": (
        "PRNUMBERS",
        "prnumbers",
        "Pull-request references in added lines, code and docs (a repo rule of the "
        "operator, not of platform/AGENTS.md)",
        False,
    ),
    "perfnotes": (
        "PERFNOTES",
        "perfnotes",
        "Measured timings and speedups in source comments",
        False,
    ),
    "backedges": (
        "BACKEDGES",
        "backedges",
        "New platform -> library back-edges",
        False,
    ),
    "abiversion": (
        "ABIVERSION",
        "abiversion",
        "Kernarg or signature changes with the kernel ABI string kept",
        False,
    ),
    "arraywrites": (
        "ARRAYWRITES",
        "arraywrites",
        "Fixed C arrays written at more sites than slots (vs the merge base)",
        False,
    ),
    "nanblind": ("NANBLIND", "nanblind", "NaN-blind failure tests", False),
    "optimports": (
        "OPTIMPORTS",
        "optimports",
        "Optional packages imported at test module level, unguarded",
        False,
    ),
    "mdheadings": (
        "MDHEADINGS",
        "mdheadings",
        "Repeated numbered Markdown headings",
        False,
    ),
    "peerleads": (
        "PEERLEADS",
        "peerleads",
        "Peer-review leads, never a flag (bool() of request fields, torch device "
        "queries, file(GLOB) without CONFIGURE_DEPENDS)",
        True,
    ),
    "paritypair": (
        "PARITYPAIR",
        "paritypair",
        "Lead, never a flag: lowerer changed with no parity *_emit pair",
        True,
    ),
    "unpack": (
        "UNPACK",
        "unpack",
        "Lead, never a flag: unused tuple-unpack targets",
        True,
    ),
}

TEST = re.compile(
    r"(^|/)(tests?|testing|benchmarks?)/|(^|/)test_[^/]*$|_test\.[a-z]+$|(^|/)conftest\.py$"
)
DOC = re.compile(r"\.(md|rst|txt)$|(^|/)(docs|dsl_docs)/")
DATA = re.compile(
    r"\.(json|jsonl|csv|tsv|manifest|lock|ll|s|svg|ipynb|patch|diff)$", re.I
)
CXX = re.compile(r"\.(c|cc|cpp|cxx|h|hh|hpp|hxx|inl|cu|hip)$")

# --- the PR's diff ----------------------------------------------------------------------------------------------------


def pr_diff(
    repo: str, mb: str, head: str, timeout: int = 600, removed: bool = False
) -> dict[str, dict]:
    """path -> {"new": the PR adds the file, "lines": [(line number at head, added line)]}: merge base to head, git's
    rename detection on (a moved file's lines are not added lines). `removed`: each file also has "old" (its path at the
    merge base, None when added) and "removed": [(line number at the merge base, removed line)].
    """
    text = gitops.git(
        repo,
        "-c",
        "core.quotepath=false",
        "-c",
        "diff.renameLimit=20000",
        "diff",
        "-U0",
        "--no-color",
        "-M",
        "--no-ext-diff",
        mb,
        head,
        timeout=timeout,
    )
    return parse_diff(text, removed)


def parse_diff(text: str, removed: bool = False) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cur, line, oline, new, old, hunk = None, 0, 0, False, None, False
    for ln in text.split("\n"):
        if ln.startswith("diff --git "):
            cur, new, old, hunk = None, False, None, False
            continue
        if not hunk and ln.startswith("new file mode"):
            new = True
            continue
        if not hunk and ln.startswith("--- "):
            new = new or ln == "--- /dev/null"
            old = ln[6:] if ln.startswith("--- a/") else None
            continue
        if not hunk and ln.startswith("+++ "):
            cur = ln[6:] if ln.startswith("+++ b/") else None
            if cur is not None:
                rec = out.setdefault(cur, {"new": new, "lines": []})
                if removed:
                    rec.setdefault("old", old)
                    rec.setdefault("removed", [])
            continue
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", ln)
        if m:
            oline, line, hunk = int(m.group(1)), int(m.group(2)), True
            continue
        if cur is not None and ln.startswith("+"):
            out[cur]["lines"].append((line, ln[1:]))
            line += 1
        elif removed and cur is not None and ln.startswith("-"):
            out[cur]["removed"].append((oline, ln[1:]))
            oline += 1
    return out


def _rel(path: str, K: str) -> str:
    return path[len(K) + 1 :] if path.startswith(K + "/") else path


def _comment(path: str, text: str) -> bool:
    s = text.strip()
    if path.endswith(".py") or path.endswith(
        (".cmake", "CMakeLists.txt", ".sh", ".yaml", ".yml")
    ):
        return s.startswith("#")
    if CXX.search(path):
        return s.startswith(("//", "/*", "*"))
    return False


# --- trackers ---------------------------------------------------------------------------------------------------------

TRACKER = re.compile(r"(?<![\w-])([A-Z][A-Z0-9]{1,11})-(\d{2,7})(?![\w-])")


def tracker_allow(cfg: dict) -> tuple:
    """The tracker-like prefixes that never flag, each the name of a standard, an encoding, a register or a
    product: config `tracker_allow`, the package's (config.defaults.json) while it is empty.
    """
    return tuple(registry.value(cfg, "tracker_allow", registry.FALSY))


def trackers(diff: dict, K: str, allow=None) -> list[dict]:
    """The tracker IDs in DIFF's added lines; ALLOW, the prefixes that never flag (the package's,
    tracker_allow, when it is not given)."""
    allow = set(tracker_allow({}) if allow is None else allow)
    out = []
    for path, rec in sorted(diff.items()):
        if DATA.search(path):
            continue
        for num, text in rec["lines"]:
            for m in TRACKER.finditer(text):
                if m.group(1) not in allow:
                    out.append({"at": f"{_rel(path, K)}:{num}", "id": m.group(0)})
    return out


# --- env reads and env_flags.md ---------------------------------------------------------------------------------------

ENV_FLAGS = "platform/dsl_docs/reference/env_flags.md"
# A read: getenv / environ.get / environ[...] (not an assignment) / environ.setdefault, or a helper whose name says it
# reads the environment (`_env_flag("ROCKE_X")`, `env_int(...)`); setenv, putenv, unsetenv, delenv and pop write.
ENV_READ = re.compile(
    r"(?:\b(?:std::)?getenv|\benviron\.get|\benviron\.setdefault|\benviron\s*\[|\benv\.get"
    r"|\b(?!set|put|unset|del|pop)[A-Za-z_]*env[A-Za-z_]*)\s*\(?\s*[\"'](ROCKE_[A-Z0-9_]*[A-Z0-9])[\"']"
    r"(\s*\]\s*=(?!=))?"
)
DOC_NAME = re.compile(r"\bROCKE_[A-Z0-9_]*[A-Z0-9](?:_\*|\*)?")


def env_reads(diff: dict, K: str) -> list[dict]:
    """The `ROCKE_*` names non-test, non-doc code reads on the PR's added lines: {"name", "at"}, first place each."""
    seen, out = set(), []
    for path, rec in sorted(diff.items()):
        rel = _rel(path, K)
        if (
            not path.startswith(K + "/")
            or TEST.search(rel)
            or DOC.search(rel)
            or DATA.search(rel)
        ):
            continue
        if not (path.endswith(".py") or CXX.search(path)):
            continue
        for num, text in rec["lines"]:
            if _comment(path, text):
                continue
            for m in ENV_READ.finditer(text):
                name = m.group(1)
                if m.group(2) or name in seen:
                    continue
                seen.add(name)
                out.append({"name": name, "at": f"{rel}:{num}"})
    return out


def documented(doc: str) -> tuple[set, list]:
    """The names env_flags.md gives, and the families it documents with a trailing `*` (`ROCKE_EXP_*`)."""
    names, fams = set(), []
    for x in DOC_NAME.findall(doc or ""):
        if x.endswith("*"):
            fams.append(x.rstrip("*").rstrip("_") + "_")
        else:
            names.add(x)
    return names, fams


def envdocs(repo: str, head: str, mb: str, K: str, diff: dict) -> list[dict]:
    reads = env_reads(diff, K)
    if not reads:
        return []
    try:
        doc = gitops.git(repo, "show", f"{head}:{K}/{ENV_FLAGS}")
    except gitops.GitError:
        raise prtree.NotRun(f"no {ENV_FLAGS} at the head")
    names, fams = documented(doc)
    left = [
        r
        for r in reads
        if r["name"] not in names and not any(r["name"].startswith(f) for f in fams)
    ]
    if not left:
        return []
    old = _grep_words(repo, mb, K, [r["name"] for r in left])
    return [r for r in left if r["name"] not in old]


def _grep_words(repo: str, rev: str, K: str, words: list[str]) -> set[str]:
    """The `words` that the code (docs aside) at `rev` has, as whole words."""
    args = ["grep", "-h", "-o", "-I", "-w", "-F"] + [
        x for w in words for x in ("-e", w)
    ]
    p = subprocess.run(
        [
            "git",
            "-C",
            repo,
            *args,
            rev,
            "--",
            K + "/",
            f":(exclude){K}/*.md",
            f":(exclude){K}/*.rst",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if p.returncode not in (0, 1):
        raise prtree.NotRun(f"git grep at {rev[:10]}: {p.stderr.strip()[:160]}")
    return set(p.stdout.split())


# --- new library tests in no list -------------------------------------------------------------------------------------

LIB_TEST = re.compile(r"^library/tests/(?:.+/)?test_[^/]+\.py$")
LISTS = re.compile(r"(^|/)(CMakeLists\.txt|[^/]+\.cmake|[^/]+\.ya?ml)$")


def new_library_tests(diff: dict, K: str) -> list[str]:
    return sorted(
        _rel(p, K)
        for p, rec in diff.items()
        if rec["new"] and p.startswith(K + "/") and LIB_TEST.match(_rel(p, K))
    )


def list_text(repo: str, head: str, root: str) -> str:
    """Every CMakeLists.txt, .cmake and YAML file under `root` at the head, as one text."""
    files = [
        f
        for f in gitops.git(
            repo,
            "-c",
            "core.quotepath=false",
            "ls-tree",
            "-r",
            "--name-only",
            head,
            "--",
            root + "/",
        ).splitlines()
        if LISTS.search(f)
    ]
    if not files:
        return ""
    p = subprocess.run(
        ["git", "-C", repo, "cat-file", "--batch"],
        capture_output=True,
        timeout=300,
        input="".join(f"{head}:{f}\n" for f in files).encode(),
    )
    return p.stdout.decode("utf-8", "replace")


def unlisted(tests: list[str], text: str) -> list[dict]:
    """The new tests `text` names neither by file name nor by a directory above them (below library/tests)."""
    out = []
    for t in tests:
        parts = t.split("/")
        if re.search(rf"(?<![\w.-]){re.escape(parts[-1])}(?![\w.-])", text):
            continue
        # A directory names its tests when it ends a path (a quote, a space, a parenthesis, the line's end, or a glob
        # below it): "tests/library/tests/dispatch/gdn", not ".../dispatch/test_other.py".
        dirs = [
            "/".join(parts[:i]) for i in range(3, len(parts))
        ]  # library/tests/<sub>[/...]
        if any(
            re.search(
                rf"(?<![\w.-]){re.escape(d)}(?=/?[\\\"'\s)}};]|/?$|/[^\s\"')]*\*)",
                text,
                re.M,
            )
            for d in dirs
        ):
            continue
        out.append({"test": t})
    return out


def testlists(repo: str, head: str, K: str, diff: dict) -> list[dict]:
    tests = new_library_tests(diff, K)
    if not tests:
        return []
    return unlisted(
        tests, list_text(repo, head, K.rsplit("/", 1)[0] if "/" in K else K)
    )


# --- C/C++ getenv -----------------------------------------------------------------------------------------------------

GETENV = re.compile(r"(?<![\w.>])(?:std::|::)?(?:secure_)?getenv\s*\(\s*([^)]*)\)")


def cppenv(diff: dict, K: str) -> list[dict]:
    out = []
    for path, rec in sorted(diff.items()):
        rel = _rel(path, K)
        if not path.startswith(K + "/") or not CXX.search(path) or TEST.search(rel):
            continue
        for num, text in rec["lines"]:
            if _comment(path, text):
                continue
            for m in GETENV.finditer(text):
                out.append({"at": f"{rel}:{num}", "what": m.group(1).strip()[:60]})
    return out


# --- the PR title -----------------------------------------------------------------------------------------------------

TITLE_TYPES = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "test",
    "docs",
    "build",
    "ci",
    "chore",
    "style",
    "revert",
)
TITLE = re.compile(
    rf"^(?:{'|'.join(TITLE_TYPES)})(?:\([a-z0-9][a-z0-9/_.,-]*\))?!?: \S"
)


def title_problem(title: str) -> str | None:
    """Why `title` is not a Conventional Commits title, None when it is."""
    if TITLE.match(title):
        return None
    if title[:1].isspace():
        return "it starts with a space"
    m = re.match(r"^([A-Za-z]+)(\([^)]*\))?(!?):", title)
    if not m:
        return f"no `type(scope): ` prefix (types: {', '.join(TITLE_TYPES)})"
    if m.group(1) not in TITLE_TYPES:
        return f"`{m.group(1)}` is not a type ({', '.join(TITLE_TYPES)})"
    if m.group(2) and not re.fullmatch(r"\([a-z0-9][a-z0-9/_.,-]*\)", m.group(2)):
        return f"the scope `{m.group(2)}` is not lower-case words"
    return "no single space and subject after the colon"


def meta_title(path: str) -> str:
    """The title in a staged meta.md (`# PR #N: <title>`)."""
    first = (
        pathlib.Path(path)
        .read_text(encoding="utf-8", errors="replace")
        .split("\n", 1)[0]
    )
    m = re.match(r"^# PR #\d+: (.*)$", first)
    if not m:
        raise prtree.NotRun(f"no `# PR #N: title` line in {path}")
    return m.group(1)


# --- live unanswered threads ------------------------------------------------------------------------------------------


def _login(c: dict) -> str:
    return (c.get("author") or {}).get("login") or "ghost"


def _bot(c: dict) -> bool:
    who = _login(c)
    return (
        (c.get("author") or {}).get("__typename") == "Bot" or who.endswith("[bot]")
    ) and "copilot" not in who.lower()


def unanswered(data: dict, K: str = "") -> list[dict]:
    """The live threads no owner answered, from PR data as followups.read returns it ("author", "commits",
    "threads"): {"reviewer", "where", "url", "at", "replies"}."""
    owners = {data.get("author")} | {
        c.get("login") for c in data.get("commits") or [] if c.get("login")
    }
    out = []
    for t in data.get("threads") or []:
        cs = sorted(t["comments"]["nodes"], key=lambda c: c.get("createdAt") or "")
        if (
            not cs
            or t.get("isResolved")
            or t.get("isOutdated")
            or _bot(cs[0])
            or _login(cs[0]) in owners
        ):
            continue
        if any(_login(c) in owners for c in cs[1:]):
            continue
        path = t.get("path") or "-"
        line = t.get("line") or t.get("originalLine")
        where = _rel(path, K) + (f":{line}" if line else "")
        out.append(
            {
                "reviewer": _login(cs[0]),
                "where": where,
                "url": cs[0].get("url") or "",
                "at": (cs[0].get("createdAt") or "")[:16],
                "replies": len(cs) - 1,
            }
        )
    return out


def read_threads(slug: str, n: int) -> dict:
    from . import followups, gh

    try:
        return followups.read({"repo": slug}, n)
    except (gh.GhError, KeyError, TypeError) as exc:
        raise prtree.NotRun(f"GitHub not read: {str(exc)[:160]}")


# --- leads ------------------------------------------------------------------------------------------------------------

COERCE = re.compile(
    r"\bbool\(\s*(?:req|request|problem|prob|params|kwargs|kw|opts|options)"
    r"\s*(?:\.get\(|\.[A-Za-z_]\w*\s*\)|\[)"
)
TORCH_DEVICE = re.compile(
    r"\btorch\.(?:cuda\.(?:get_device_properties|current_device|get_device_name|device_count|"
    r"get_device_capability|is_available)|version\.hip)\b"
)
GLOB = re.compile(r"\bfile\s*\(\s*GLOB(?:_RECURSE)?\b", re.I)
LIB_CODE = ("library/", "platform/python/")


def _statement(lines: list[tuple[int, str]], i: int) -> str:
    """The CMake command that starts on added line i, to its closing parenthesis (or the end of the hunk's lines)."""
    text, depth, j = "", 0, i
    while j < len(lines) and (j == i or lines[j][0] == lines[j - 1][0] + 1):
        seg = re.sub(r"#.*", "", lines[j][1])
        text += " " + seg
        depth += seg.count("(") - seg.count(")")
        if depth <= 0 and "(" in text:
            break
        j += 1
    return text


def peerleads(diff: dict, K: str) -> list[dict]:
    out = []
    for path, rec in sorted(diff.items()):
        rel = _rel(path, K)
        lines = rec["lines"]
        if (
            path.endswith(".py")
            and path.startswith(K + "/")
            and rel.startswith(LIB_CODE)
            and not TEST.search(rel)
            and "/examples/" not in rel
        ):
            for num, text in lines:
                if _comment(path, text):
                    continue
                if COERCE.search(text):
                    out.append(
                        {
                            "kind": "coercion",
                            "at": f"{rel}:{num}",
                            "text": text.strip()[:110],
                        }
                    )
                if TORCH_DEVICE.search(text):
                    out.append(
                        {
                            "kind": "torch device query",
                            "at": f"{rel}:{num}",
                            "text": text.strip()[:110],
                        }
                    )
        if path.endswith(("CMakeLists.txt", ".cmake")):
            for i, (num, text) in enumerate(lines):
                if GLOB.search(re.sub(r"#.*", "", text)):
                    stmt = _statement(lines, i)
                    if "CONFIGURE_DEPENDS" not in stmt.upper() and not re.search(
                        r"\$\{[^}]*configure_depends[^}]*\}", stmt, re.I
                    ):
                        out.append(
                            {
                                "kind": "file(GLOB) without CONFIGURE_DEPENDS",
                                "at": f"{rel}:{num}",
                                "text": text.strip()[:110],
                            }
                        )
    return out


# --- line rules: each item {"at": "<path>:<line>", "what": ...} ------------------------------------------------------

PY_TEST = re.compile(r"(^|/)tests?/|(^|/)test_[^/]*\.py$|(^|/)conftest\.py$")
CODE = re.compile(
    r"\.(py|c|cc|cpp|cxx|h|hh|hpp|hxx|inl|cu|hip|cmake|sh)$|(^|/)CMakeLists\.txt$"
)
_STR = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')


def _doc(path: str) -> bool:
    """A document (Markdown, reST, text, or under a docs tree), CMakeLists.txt aside."""
    return bool(DOC.search(path)) and not path.endswith("CMakeLists.txt")


def _texts(repo: str, rev: str, paths) -> dict[str, str | None]:
    """path -> its text at `rev` (None when absent), in one cat-file."""
    paths = list(dict.fromkeys(p for p in paths if p))
    got = gitops.cat_blobs(repo, [f"{rev}:{p}" for p in paths])
    return {p: got.get(f"{rev}:{p}") for p in paths}


def _old(path: str, rec: dict) -> str | None:
    """A changed file's path at the merge base (its old path when moved), None when the PR adds it."""
    return rec["old"] if "old" in rec else (None if rec["new"] else path)


def _join(d: str, name: str) -> str:
    return f"{d}/{name}" if d else name


def _parse(src: str | None) -> ast.Module | None:
    try:
        return ast.parse(src or "")
    except (SyntaxError, ValueError):
        return None


def docstring_lines(src: str | None) -> set[int]:
    """The lines of a Python source's docstrings (module, class and function), 1-based; none when it does not parse."""
    out: set[int] = set()
    tree = _parse(src)
    for node in ast.walk(tree) if tree else ():
        if (
            isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                out.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return out


def comment_part(path: str, text: str) -> str:
    """A code line's comment: the whole line when it is one, else its trailing comment (strings blanked); '' if none."""
    if _comment(path, text):
        return text
    bare = _STR.sub('""', text)
    if path.endswith((".py", ".cmake", "CMakeLists.txt", ".sh")):
        i = bare.find("#")
    elif CXX.search(path):
        i = bare.find("//")
    else:
        return ""
    return bare[i:] if i >= 0 else ""


def _code_lines(
    path: str, lines: list[tuple[int, str]], src: str | None
) -> list[tuple[int, str]]:
    """The lines that are code: not comments, not docstrings, not blank."""
    ds = docstring_lines(src) if path.endswith(".py") else set()
    return [
        (n, t) for n, t in lines if t.strip() and not _comment(path, t) and n not in ds
    ]


# A literal that holds the separator is never one of split()'s pieces: `"_dg" not in name.split("_")` is always true.
SPLIT_TAUT = re.compile(
    r"""(["'])([^"'\\\n]*)\1\s+(not\s+)?in\s+[\w.\[\]()]+\.split\("""
    r"""\s*(?:(["'])([^"'\\\n]+)\4)?\s*\)"""
)


def splittaut(diff: dict, K: str) -> list[dict]:
    out = []
    for path, rec in sorted(diff.items()):
        if not path.endswith(".py"):
            continue
        for num, text in rec["lines"]:
            if _comment(path, text):
                continue
            for m in SPLIT_TAUT.finditer(text):
                lit, neg, sep = m.group(2), m.group(3), m.group(5)
                if sep is not None and sep not in lit:
                    continue
                if sep is None and lit and not any(c.isspace() for c in lit):
                    continue
                what = (
                    f"no piece of .split({sep!r})"
                    if sep is not None
                    else "no piece of .split()"
                )
                out.append(
                    {
                        "at": f"{_rel(path, K)}:{num}",
                        "what": f"`{m.group(0)}` is always "
                        f"{'true' if neg else 'false'}: {what} can hold {lit!r}",
                    }
                )
    return out


# A test that imports a sibling test module by its bare name: pytest's importlib mode (library/ and platform/ tests)
# does not put the test's directory on sys.path, so it resolves only when the file or a conftest.py beside it inserts it
# (a bare sibling import that collects here can fail in another lane).
SIBLING = re.compile(
    r"^\s*(?:from\s+(test_\w+|conftest)\s+import\b|import\s+(test_\w+|conftest)\b)"
)
OWN_DIR = re.compile(
    r"Path\(\s*__file__\s*\)(?:\.resolve\(\))?\.parent(?![\w.\[])"
    r"|os\.path\.dirname\(\s*(?:os\.path\.(?:abspath|realpath)\(\s*)?__file__\s*\)?\s*\)"
)
PATH_EDIT = re.compile(r"\bsys\.path\.(?:insert|append)\(")


def sibimports(repo: str, head: str, diff: dict, K: str) -> list[dict]:
    cands = []
    for path, rec in sorted(diff.items()):
        if not path.endswith(".py") or not PY_TEST.search(path):
            continue
        d = path.rsplit("/", 1)[0] if "/" in path else ""
        for num, text in rec["lines"]:
            m = SIBLING.match(text)
            if m:
                cands.append((path, d, num, m.group(1) or m.group(2), text.strip()))
    if not cands:
        return []
    t = _texts(
        repo,
        head,
        [p for p, *_ in cands]
        + [_join(d, "conftest.py") for _, d, *_ in cands]
        + [_join(d, f"{mod}.py") for _, d, _, mod, _ in cands],
    )
    out = []
    for path, d, num, mod, text in cands:
        conf = t.get(_join(d, "conftest.py")) or ""
        if (
            t.get(_join(d, f"{mod}.py")) is None
            or PATH_EDIT.search(t.get(path) or "")
            or (PATH_EDIT.search(conf) and OWN_DIR.search(conf))
        ):
            continue
        out.append(
            {
                "at": f"{_rel(path, K)}:{num}",
                "what": f"`{text[:90]}`: {mod}.py is a sibling, and neither the "
                "file nor a conftest.py beside it puts the directory on sys.path (pytest --import-mode=importlib)",
            }
        )
    return out


# Workload sizes in a kernel's name (the peer-review rules §C.3, name by operation): `..._M32N48K128`, and the f-string that
# makes it.
WORKLOAD = re.compile(
    r"(?<![A-Za-z0-9])M\d{2,}_?N\d{2,}_?K\d{2,}(?!\d)"
    r"|(?<![A-Za-z0-9])[mM]\d{2,}_[nN]\d{2,}_[kK]\d{2,}(?!\d)|M\{[^{}]+\}_?N\{[^{}]+\}_?K\{[^{}]+\}"
)


def workload(diff: dict, K: str) -> list[dict]:
    out = []
    for path, rec in sorted(diff.items()):
        rel = _rel(path, K)
        if not path.startswith(K + "/") or _doc(rel):
            continue
        golden = "golden" in rel
        for num, text in rec["lines"]:
            if not (golden or "name" in text.lower()) or _comment(path, text):
                continue
            m = WORKLOAD.search(text)
            if m:
                out.append(
                    {
                        "at": f"{rel}:{num}",
                        "what": f"`{m.group(0)}` in a kernel name: {text.strip()[:100]}",
                    }
                )
    return out


# Pull-request references in any committed file the PR adds lines to (the operator's repo rule: the repo states
# generalized facts, never a ledger of history or of particular cases): `#1234`, `#567`, `PR #89`,
# `rocm-libraries#123`, `/pull/1234`, in code, comments, docstrings and docs alike. Data files are not
# read; tracker IDs are the trackers check's.
PRREF = re.compile(
    r"rocm-libraries#\d+|\bPR\s?#\d{3,5}\b|\bPR\s?\d{4,5}\b|(?<![\w/&#])#[1-9]\d{3,4}\b|/pull/\d{3,5}\b"
)


def prnumbers(diff: dict, K: str) -> list[dict]:
    out = []
    for path, rec in sorted(diff.items()):
        if DATA.search(path):
            continue
        for num, text in rec["lines"]:
            m = PRREF.search(text)
            if m:
                out.append(
                    {
                        "at": f"{_rel(path, K)}:{num}",
                        "what": f"`{m.group(0)}`: {text.strip()[:100]}",
                    }
                )
    return out


# Measured timings and speedups in source comments and docstrings (platform/AGENTS.md: source comments stay
# qualitative). A parameter (a timeout, a default, a gap, a bound) is not a measurement.
PERF_REL = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:%|x|×)\s+(?:faster|slower|speed-?ups?|quicker|better|worse|regressions?|"
    r"improvements?|gains?|wins?)\b"
    r"|\b(?:faster|slower|speed-?up|regression|improvement|gain)\s+(?:of|by)\s+(?:~|about\s+)?\d+"
    r"(?:\.\d+)?\s?(?:%|x|×)",
    re.I,
)
PERF_RATE = re.compile(
    r"(?<![\w.])~?\d+(?:\.\d+)?\s?(?:[TGM]FLOP/?s|[TGM]FLOPS|[KMGT]i?B/s|tok(?:ens)?/s)(?![\w/])"
)
PERF_TIME = re.compile(
    r"(?<![\w.:{%$/#-])(?:~|≈)?\d+(?:\.\d+)?\s?"
    r"([uµn]s|usecs?|ms|msecs?|seconds?|secs?|minutes?|mins?|hours?|s|h)(?![\w/]|\.\.|\s*[+*^(])"
)
PERF_PHRASE = re.compile(
    r"\b(?:well\s+)?under\s+(?:a|one)\s+(?:second|millisecond)\b"
    r"|\b(?:a few\s+|several\s+)?(?:tens|hundreds?|thousands?)(?:\s+of)?\s+"
    r"(?:micro|milli|nano)seconds\b",
    re.I,
)
PERF_CUE = re.compile(
    r"~|≈|\b(?:about|around|roughly|approx\w*|takes?|took|costs?|spends?|spent|measured|"
    r"per (?:call|card|run|launch|iteration|step|shard))\b",
    re.I,
)
PERF_PARAM = re.compile(
    r"time-?out|\blimit|budget|deadline|\bcap(?:s|ped)?\b|\bgaps?\b|\bidle\b|sleep|\bwait|interval|"
    r"\bpoll|retr(?:y|ies)|backoff|default|threshold|at most|up to|no more than|"
    r"\bmax(?:imum)?\b|\bbelow\b|\bbound|\bcron\b|\bTTL\b|expir|\bevery\b|warm-?up",
    re.I,
)
PERF_TESTS = re.compile(
    r"(^|/)tests?/|(^|/)test_[^/]*$|_test\.[a-z]+$|(^|/)conftest\.py$"
)


def perf_claim(text: str) -> str | None:
    """The measured timing or speedup a comment states, None when it states none."""
    m = PERF_REL.search(text) or PERF_RATE.search(text)
    if m:
        return m.group(0)
    if PERF_PARAM.search(text):
        return None
    m = PERF_PHRASE.search(text)
    if m:
        return m.group(0)
    for m in PERF_TIME.finditer(text):
        if m.group(1) in ("s", "h", "min", "mins") and not PERF_CUE.search(text):
            continue
        return m.group(0).strip()
    return None


def perfnotes(repo: str, head: str, diff: dict, K: str) -> list[dict]:
    files = [
        p
        for p, rec in sorted(diff.items())
        if rec["lines"]
        and p.startswith(K + "/")
        and CODE.search(p)
        and not PERF_TESTS.search(_rel(p, K))
        and not _doc(_rel(p, K))
    ]
    t = _texts(repo, head, [p for p in files if p.endswith(".py")])
    out = []
    for path in files:
        ds = docstring_lines(t.get(path)) if path.endswith(".py") else set()
        prev = (0, "")
        for num, text in diff[path]["lines"]:
            prose = text if num in ds else comment_part(path, text)
            claim = perf_claim(prose) if prose else None
            if (
                not claim
                and prose
                and prev[0] == num - 1
                and prev[1]
                and not perf_claim(prev[1])
            ):
                # A claim wrapped over two lines ("At a few hundred" / "microseconds each").
                claim = perf_claim(
                    prev[1].strip()[-60:] + " " + prose.strip().lstrip("#/* ")
                )
            if claim:
                out.append(
                    {
                        "at": f"{_rel(path, K)}:{num}",
                        "what": f"`{claim}`: {text.strip()[:100]}",
                    }
                )
            prev = (num, prose)
    return out


# platform/ must not reach into library/ (the peer-review rules §E.1: library -> platform, one way): an import of a library
# package, a sys.path edit naming library, or a path into ../library/ (`os.path.join(here, "..", "library")` too). Each
# names a target (the package, or library/<dir>); a line flags when its file did not reach that target at the merge
# base (develop's edges are develop's). CMake's install-layout lists name tests/library/tests/... and are not edges.
BACKEDGE = re.compile(
    r"\bfrom\s+(?:rocke_library|library|kernels|dispatch|builders|benchmarks)(?:\.[\w.]+)?\s+import\b"
    r"|\bimport\s+(?:rocke_library\w*|kernels\.|dispatch\.|builders\.)"
    r"|sys\.path\.(?:insert|append)\([^)]*library|[\"'/]\.\./(?:\.\./)*library/"
    r"|[\"']\.\.[\"']\s*,\s*[\"']library[\"']|\$\{[^}]*\}/library/"
    r"|[\"']library/(?:tests|kernels|dispatch|builders|benchmarks)/"
)


def edge_targets(line: str) -> set[str]:
    """What a back-edge line reaches: library packages it imports and library/<dir> paths it names."""
    t = set(
        re.findall(
            r"\bfrom\s+(rocke_library|library|kernels|dispatch|builders|benchmarks)\b",
            line,
        )
    )
    t |= set(re.findall(r"\bimport\s+(rocke_library\w*)", line))
    t |= set(re.findall(r"\bimport\s+(kernels|dispatch|builders)\.", line))
    t |= {f"library/{x}" for x in re.findall(r"(?<![\w-])library/(\w+)", line)}
    t |= {
        f"library/{x}"
        for x in re.findall(r"[\"']library[\"']\s*,\s*[\"'](\w+)[\"']", line)
    }
    return t or {"library"}


def _edges(path: str, text: str | None) -> set[str]:
    return {
        x
        for ln in (text or "").split("\n")
        if not _comment(path, ln)
        and "tests/library/tests/" not in ln
        and BACKEDGE.search(ln)
        for x in edge_targets(ln)
    }


def backedges(repo: str, mb: str, diff: dict, K: str) -> list[dict]:
    cands = []
    for path, rec in sorted(diff.items()):
        rel = _rel(path, K)
        if (
            not rel.startswith("platform/")
            or not path.startswith(K + "/")
            or _doc(rel)
            or DATA.search(rel)
        ):
            continue
        hits = [
            (n, t)
            for n, t in rec["lines"]
            if not _comment(path, t)
            and "tests/library/tests/" not in t
            and BACKEDGE.search(t)
        ]
        if hits:
            cands.append((path, _old(path, rec), hits))
    if not cands:
        return []
    base = _texts(repo, mb, [old for _, old, _ in cands])
    out = []
    for path, old, hits in cands:
        had = _edges(old, base.get(old)) if old else set()
        for n, t in hits:
            new = sorted(edge_targets(t) - had)
            if new:
                out.append(
                    {
                        "at": f"{_rel(path, K)}:{n}",
                        "what": f"reaches {', '.join(new)} (`"
                        f"{BACKEDGE.search(t).group(0).strip()}`): {t.strip()[:100]}",
                    }
                )
    return out


# A dispatch family's kernel ABI string (`abi_version="…/v1"`, `<FAMILY>_ABI_VERSION`) names its kernarg list: a code
# line that changes the signature or the kernargs with the string untouched keeps a stale ABI.
ABI_CONST = re.compile(r"^\s*(\w*ABI\w*|abi_version)\s*[:=](?!=)\s*(\w*)", re.M)
KERNARG = re.compile(
    r"(?<!\w)(?:signature|kernargs?|kernel_args|arg_types)\s*=(?!=)|\bKernelArg\s*\("
    r"|\bkernargs?\.(?:append|extend|insert)\("
)


def abiversion(repo: str, mb: str, head: str, diff: dict, K: str) -> list[dict]:
    cands = [
        p
        for p, rec in sorted(diff.items())
        if p.startswith(K + "/")
        and not rec["new"]
        and (p.endswith(".py") or CXX.search(p))
        and not TEST.search(_rel(p, K))
        and any(KERNARG.search(t) for _, t in rec["lines"] + rec.get("removed", []))
    ]
    if not cands:
        return []
    now = _texts(repo, head, cands)
    was = _texts(repo, mb, [diff[p].get("old") or p for p in cands])
    added = "\n".join(t for rec in diff.values() for _, t in rec["lines"])
    out = []
    for path in cands:
        consts = ABI_CONST.findall(now.get(path) or "")
        if not consts:
            continue
        names = {c for c, _ in consts} | {
            v for c, v in consts if c == "abi_version" and v and not v[0].isdigit()
        }
        if any(re.search(rf"\b{re.escape(c)}\s*[:=](?!=)", added) for c in names):
            continue  # bumped (here, or where the string is defined)
        rec = diff[path]
        code = _code_lines(path, rec["lines"], now.get(path)) + _code_lines(
            path, rec.get("removed", []), was.get(rec.get("old") or path)
        )
        kern = [(n, t) for n, t in code if KERNARG.search(t)]
        if kern:
            n, t = min(kern)
            out.append(
                {
                    "at": f"{_rel(path, K)}:{n}",
                    "what": f"{len(kern)} kernarg or signature code line"
                    f"{'s' * (len(kern) != 1)} changed (`{t.strip()[:70]}`) and {', '.join(sorted(names))} kept",
                }
            )
    return out


# A fixed C array filled through a running index at more write sites than it has slots, more so than at the merge base
# (13 flags written into a 12-slot array, say). A moved file is compared with its old path.
ARRAY_DECL = re.compile(
    r"(?<![\w.])(?!(?:return|delete|case|goto|else|sizeof|throw)\b)(?:const\s+)?[\w:]+"
    r"(?:\s*\*\s*|\s+)(\w+)\s*\[\s*(\d+)\s*\]\s*(?:=\s*\{[^;]*\})?\s*;"
)


def _blocks(src: str) -> list[tuple[int, str]]:
    """The top-level brace blocks of a C/C++ source (function bodies, roughly): (offset, text)."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(src):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                out.append((start, src[start : i + 1]))
                start = None
    return out


def over_written(src: str | None) -> dict[tuple[str, str], tuple[int, int, int]]:
    """(array, index) -> (slots, write sites, line) for each fixed array written through a running index at more sites
    than it has slots, with no bound check of the index in its block."""
    res = {}
    for off, body in _blocks(src or ""):
        for m in ARRAY_DECL.finditer(body):
            name, n = m.group(1), int(m.group(2))
            if not 0 < n <= 64:
                continue
            writes = re.findall(
                rf"\b{name}\s*\[\s*(\w+)\s*(?:\+\+)?\s*\]\s*=(?!=)", body
            )
            for v, c in collections.Counter(
                w for w in writes if not w.isdigit()
            ).items():
                if c <= n or not re.search(
                    rf"\b{v}\s*\+\+|\+\+\s*{v}\b|\b{v}\s*\+=\s*1\b", body
                ):
                    continue
                bound = rf"(?:{n}\b|sizeof|std::size|\w*(?:MAX|CAP|COUNT)\w*)"
                if re.search(rf"\b{v}\s*(?:<|>=)\s*(?:\(\s*\w+\s*\)\s*)?{bound}", body):
                    continue
                res[(name, v)] = (n, c, (src or "")[: off + m.start()].count("\n") + 1)
    return res


def arraywrites(repo: str, mb: str, head: str, diff: dict, K: str) -> list[dict]:
    cands = [
        p
        for p, rec in sorted(diff.items())
        if rec["lines"]
        and p.startswith(K + "/")
        and CXX.search(p)
        and not TEST.search(_rel(p, K))
    ]
    if not cands:
        return []
    now = _texts(repo, head, cands)
    hot = {p: over_written(now.get(p)) for p in cands}
    hot = {p: h for p, h in hot.items() if h}
    if not hot:
        return []
    olds = {p: _old(p, diff[p]) for p in hot}
    missing = [p for p, o in olds.items() if not o]
    if (
        missing
    ):  # an added file git did not pair with a move: the merge base's one file of that name, if any
        tree = gitops.git(
            repo,
            "-c",
            "core.quotepath=false",
            "ls-tree",
            "-r",
            "--name-only",
            mb,
            "--",
            K + "/",
        ).split("\n")
        by: dict[str, list[str]] = {}
        for f in tree:
            by.setdefault(f.rsplit("/", 1)[-1], []).append(f)
        for p in missing:
            same = by.get(p.rsplit("/", 1)[-1], [])
            olds[p] = same[0] if len(same) == 1 else None
    was = _texts(repo, mb, [o for o in olds.values() if o])
    out = []
    for path, h in hot.items():
        base = over_written(was.get(olds[path])) if olds[path] else {}
        for (name, v), (n, c, line) in sorted(h.items()):
            b = base.get((name, v))
            if b and c - n <= b[1] - b[0]:
                continue
            out.append(
                {
                    "at": f"{_rel(path, K)}:{line}",
                    "what": f"{name}[{n}] is written at {c} sites through {v}"
                    + (f" ({b[1]} sites into [{b[0]}] at the merge base)" if b else "")
                    + ", with no bound check",
                }
            )
    return out


# A failure test NaN passes: `if err >= tol: fail` and `(err > tol).sum()` are false for NaN.
NAN_IF = re.compile(
    r"\bif\b[^#\n]*\b\w*(?:err|diff|delta|error|dev|mismatch)\w*(?:\.max\(\)|\))?\s*>=?\s*"
    r"\w*(?:tol|atol|rtol|eps|bound|limit|thresh)\w*"
)
NAN_COUNT = re.compile(
    r"\((?:[^()]*\b(?:err|diff|delta|error)\w*)\s*>=?\s*\w*(?:tol|atol|rtol|eps)\w*\s*\)\s*\."
    r"(?:sum|any|nonzero|count_nonzero)\("
)
NAN_SAFE = re.compile(
    r"isnan|isfinite|nan_to_num|allclose|assert_close|equal_nan|\bnan\b", re.I
)


def nanblind(repo: str, head: str, diff: dict, K: str) -> list[dict]:
    cands = [
        (p, n, t)
        for p, rec in sorted(diff.items())
        if p.endswith(".py") and not _doc(p)
        for n, t in rec["lines"]
        if not _comment(p, t)
        and (NAN_IF.search(t) or NAN_COUNT.search(t))
        and not NAN_SAFE.search(t)
    ]
    if not cands:
        return []
    t = _texts(repo, head, [p for p, _, _ in cands])
    out = []
    for path, num, text in cands:
        lines = (t.get(path) or "").split("\n")
        if NAN_SAFE.search("\n".join(lines[max(0, num - 7) : num + 3])):
            continue
        out.append(
            {
                "at": f"{_rel(path, K)}:{num}",
                "what": f"`{text.strip()[:90]}`: a NaN error compares false, so "
                "it passes (no isnan/isfinite nearby)",
            }
        )
    return out


# A test module that imports an optional package at module level with no importorskip: the shared CTest lanes do not
# all install it, so collection fails there (ml_dtypes, for example).
OPTIONAL = (
    "ml_dtypes",
    "scipy",
    "triton",
    "aiter",
    "flash_attn",
    "matplotlib",
    "transformers",
    "numba",
    "hypothesis",
)


def optimports(repo: str, head: str, diff: dict, K: str) -> list[dict]:
    files = [
        p
        for p, rec in sorted(diff.items())
        if p.endswith(".py")
        and PY_TEST.search(p)
        and any(
            re.match(rf"\s*(?:import|from)\s+(?:{'|'.join(OPTIONAL)})\b", t)
            for _, t in rec["lines"]
        )
    ]
    if not files:
        return []
    t = _texts(repo, head, files)
    out = []
    for path in files:
        src = t.get(path) or ""
        tree = _parse(src)
        added = {n for n, _ in diff[path]["lines"]}
        for node in tree.body if tree else []:
            if (
                not isinstance(node, (ast.Import, ast.ImportFrom))
                or node.lineno not in added
            ):
                continue
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for mod in mods:
                root = mod.split(".")[0]
                if root in OPTIONAL and not re.search(
                    rf"importorskip\(\s*[\"']{root}\b", src
                ):
                    out.append(
                        {
                            "at": f"{_rel(path, K)}:{node.lineno}",
                            "what": f"`import {mod}` at module level with "
                            "no pytest.importorskip: collection fails where it is not installed",
                        }
                    )
    return out


# A Markdown file whose numbered headings repeat a number at one level, the PR adding one of them.
MD_HEADING = re.compile(r"^(#{1,6})\s+§?(\d+(?:\.\d+)*)[.)]?\s")


def md_numbers(src: str | None) -> dict[tuple[str, str], list[int]]:
    """(level, number) -> the lines of the numbered headings, outside fenced code."""
    out: dict[tuple[str, str], list[int]] = {}
    fence = None
    for i, ln in enumerate((src or "").split("\n"), 1):
        f = re.match(r"^\s*(`{3,}|~{3,})", ln)
        if f:
            fence = (
                None if fence and f.group(1)[0] == fence else (fence or f.group(1)[0])
            )
            continue
        m = None if fence else MD_HEADING.match(ln)
        if m:
            out.setdefault((m.group(1), m.group(2)), []).append(i)
    return out


def mdheadings(repo: str, mb: str, head: str, diff: dict, K: str) -> list[dict]:
    files = [
        p
        for p, rec in sorted(diff.items())
        if p.endswith(".md") and any(MD_HEADING.match(t) for _, t in rec["lines"])
    ]
    if not files:
        return []
    now = _texts(repo, head, files)
    olds = {p: _old(p, diff[p]) for p in files}
    was = _texts(repo, mb, [o for o in olds.values() if o])
    out = []
    for path in files:
        added = {n for n, _ in diff[path]["lines"]}
        before = md_numbers(was.get(olds[path])) if olds[path] else {}
        for (lvl, num), at in sorted(
            md_numbers(now.get(path)).items(), key=lambda kv: kv[1][0]
        ):
            if (
                len(at) > 1
                and added & set(at)
                and len(at) > len(before.get((lvl, num), []))
            ):
                out.append(
                    {
                        "at": f"{_rel(path, K)}:{at[-1]}",
                        "what": f"heading `{lvl} {num}` appears {len(at)} times "
                        f"(lines {', '.join(map(str, at))})",
                    }
                )
    return out


# Lead: a shared lowerer changed with no `<op>_emit.{py,c}` parity pair in the PR (the peer-review rules §A.9). Peers asked for one
# on a few of the PRs this fires on: whether the change needs a pair is the verifier's call.
LOWERER = re.compile(
    r"^platform/(?:python/rocke/core/lower_\w+\.py|cpp/core/lower_\w+/)"
)
PARITY_PAIR = re.compile(r"/parity/\w+_emit\.(?:py|c)$")


def paritypair(diff: dict, K: str) -> list[dict]:
    low = [
        _rel(p, K)
        for p in sorted(diff)
        if p.startswith(K + "/") and LOWERER.match(_rel(p, K))
    ]
    if not low or any(PARITY_PAIR.search(p) for p in diff):
        return []
    return [
        {
            "at": low[0],
            "what": f"{len(low)} lowerer file{'s' * (len(low) != 1)} changed ("
            + ", ".join(p.rsplit("/", 1)[-1] for p in low[:4])
            + (f" +{len(low) - 4}" if len(low) > 4 else "")
            + ") and no parity *_emit.{py,c} pair in the PR",
        }
    ]


# Lead: a tuple-unpack target in new code that its function never reads (the peer-review rules §E.6).
def unpack(repo: str, head: str, diff: dict, K: str) -> list[dict]:
    files = [
        p
        for p, rec in sorted(diff.items())
        if rec["lines"]
        and p.endswith(".py")
        and p.startswith(K + "/")
        and not TEST.search(_rel(p, K))
    ]
    t = _texts(repo, head, files)
    out, seen = [], set()
    for path in files:
        tree = _parse(t.get(path))
        added = {n for n, _ in diff[path]["lines"]}
        for f in ast.walk(tree) if tree else ():
            if not isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            loads = {
                n.id
                for n in ast.walk(f)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            for n in ast.walk(f):
                if (
                    isinstance(n, ast.Assign)
                    and n.lineno in added
                    and len(n.targets) == 1
                    and isinstance(n.targets[0], ast.Tuple)
                    and (path, n.lineno) not in seen
                ):
                    dead = [
                        e.id
                        for e in n.targets[0].elts
                        if isinstance(e, ast.Name)
                        and not e.id.startswith("_")
                        and e.id not in loads
                    ]
                    if dead:
                        seen.add((path, n.lineno))
                        out.append(
                            {
                                "at": f"{_rel(path, K)}:{n.lineno}",
                                "what": f"{', '.join(dead)} never read "
                                f"in {f.name}()",
                            }
                        )
    return out


# check -> (function of the run's context, one item, many items, the pass line)
LINE_CHECKS = {
    "splittaut": (
        lambda c: splittaut(c.diff, c.K),
        "split tautology",
        "split tautologies",
        "no literal tested against split() pieces it cannot be",
    ),
    "sibimports": (
        lambda c: sibimports(c.repo, c.head, c.diff, c.K),
        "bare sibling test import",
        "bare sibling test imports",
        "no bare sibling test import without a sys.path insert",
    ),
    "workload": (
        lambda c: workload(c.diff, c.K),
        "kernel name with workload sizes",
        "kernel names with workload sizes",
        "no workload sizes (such as M256N128K64) in added kernel names",
    ),
    "prnumbers": (
        lambda c: prnumbers(c.diff, c.K),
        "pull-request reference",
        "pull-request references",
        "no pull-request reference in the added lines",
    ),
    "perfnotes": (
        lambda c: perfnotes(c.repo, c.head, c.diff, c.K),
        "timing or speedup in a source comment",
        "timings or speedups in source comments",
        "no measured timing or speedup in added source comments",
    ),
    "backedges": (
        lambda c: backedges(c.repo, c.mb, c.diff, c.K),
        "new platform -> library back-edge",
        "new platform -> library back-edges",
        "no platform file newly reaches into library/",
    ),
    "abiversion": (
        lambda c: abiversion(c.repo, c.mb, c.head, c.diff, c.K),
        "kernarg change with the ABI string kept",
        "kernarg changes with the ABI string kept",
        "no kernarg or signature change with its ABI string kept",
    ),
    "arraywrites": (
        lambda c: arraywrites(c.repo, c.mb, c.head, c.diff, c.K),
        "fixed array over-written",
        "fixed arrays over-written",
        "no fixed C array gains write sites beyond its slots",
    ),
    "nanblind": (
        lambda c: nanblind(c.repo, c.head, c.diff, c.K),
        "NaN-blind failure test",
        "NaN-blind failure tests",
        "no NaN-blind error comparison in added lines",
    ),
    "optimports": (
        lambda c: optimports(c.repo, c.head, c.diff, c.K),
        "unguarded optional test import",
        "unguarded optional test imports",
        "no optional package imported unguarded at test module level",
    ),
    "mdheadings": (
        lambda c: mdheadings(c.repo, c.mb, c.head, c.diff, c.K),
        "duplicate numbered heading",
        "duplicate numbered headings",
        "no numbered Markdown heading repeated",
    ),
    "paritypair": (
        lambda c: paritypair(c.diff, c.K),
        "lowerer change with no parity pair",
        "lowerer changes with no parity pair",
        "no lowerer change without a parity *_emit pair",
    ),
    "unpack": (
        lambda c: unpack(c.repo, c.head, c.diff, c.K),
        "unused unpack target",
        "unused unpack targets",
        "no unused tuple-unpack target in added lines",
    ),
}


# --- running and reporting --------------------------------------------------------------------------------------------


def run(check: str, a) -> dict:
    """{"items": [...], "error": str|None, "secs", plus what the report says}."""
    t0 = time.monotonic()
    res: dict = {"check": check, "items": [], "error": None}
    try:
        if check == "title":
            res["title"] = (
                a.title
                if a.title is not None
                else meta_title(a.meta) if a.meta else None
            )
            if res["title"] is None:
                raise prtree.NotRun("no title given (--meta or --title)")
            why = title_problem(res["title"])
            res["items"] = [{"why": why}] if why else []
        elif check == "threads":
            if a.threads_json:
                data = json.loads(
                    pathlib.Path(a.threads_json).read_text(encoding="utf-8")
                )
            elif a.slug and a.pr:
                data = read_threads(a.slug, a.pr)
            else:
                raise prtree.NotRun(
                    "no thread data (--slug and --pr, or --threads-json)"
                )
            res["items"] = unanswered(data, a.rocke_path)
            res["live"] = sum(
                1
                for t in data.get("threads") or []
                if not t.get("isResolved") and not t.get("isOutdated")
            )
        else:
            mb = prtree.merge_base(a.repo, a.base, a.head)
            diff = pr_diff(a.repo, mb, a.head, removed=check in LINE_CHECKS)
            res["files"] = len(diff)
            K = a.rocke_path
            if check in LINE_CHECKS:
                res["items"] = LINE_CHECKS[check][0](
                    types.SimpleNamespace(
                        repo=a.repo, head=a.head, mb=mb, K=K, diff=diff
                    )
                )
            elif check == "trackers":
                res["items"] = trackers(diff, K, a.allow)
            elif check == "envdocs":
                res["items"] = envdocs(a.repo, a.head, mb, K, diff)
                res["reads"] = len(env_reads(diff, K))
            elif check == "testlists":
                res["new_tests"] = len(new_library_tests(diff, K))
                res["items"] = testlists(a.repo, a.head, K, diff)
            elif check == "cppenv":
                res["items"] = cppenv(diff, K)
            elif check == "peerleads":
                res["items"] = peerleads(diff, K)
    except (
        prtree.NotRun,
        gitops.GitError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        res["error"] = str(exc)[:300]
    res["secs"] = round(time.monotonic() - t0, 1)
    return res


def verdict(res: dict) -> tuple[str, str]:
    c, items = res["check"], res["items"]
    if res.get("error"):
        return "skip", f"NOT RUN: {res['error'][:180]}"
    if c in LINE_CHECKS:
        _fn, one, many, clean = LINE_CHECKS[c]
        if not items:
            return "pass", clean
        n = len(items)
        return "fail", f"{n} {one if n == 1 else many}: " + ", ".join(
            x["at"] for x in items[:3]
        ) + (f" +{n - 3}" if n > 3 else "")
    if not items:
        return (
            "pass",
            {
                "threads": f"no live thread waits for an owner's reply ({res.get('live', 0)} live)",
                "trackers": "no tracker ID in the added lines",
                "envdocs": f"env_flags.md names every ROCKE_* variable the code newly reads ({res.get('reads', 0)} read)",
                "testlists": (
                    f"every new library test is in a CMake or CTest list ({res.get('new_tests', 0)} new)"
                    if res.get("new_tests")
                    else "no new library test file"
                ),
                "cppenv": "no getenv in added C/C++ lines",
                "title": "a Conventional Commits title",
                "peerleads": "no lead",
            }[c],
        )
    n = len(items)
    if c == "threads":
        by: dict[str, int] = {}
        for t in items:
            by[t["reviewer"]] = by.get(t["reviewer"], 0) + 1
        return (
            "fail",
            f"{n} live thread{'s' * (n != 1)} with no owner reply: "
            + ", ".join(
                f"{k} {v}"
                for k, v in sorted(by.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        )
    if c == "trackers":
        ids = list(dict.fromkeys(x["id"] for x in items))
        return (
            "fail",
            f"{n} tracker ID{'s' * (n != 1)} in added lines: {', '.join(ids[:4])}"
            + (f" +{len(ids) - 4}" if len(ids) > 4 else ""),
        )
    if c == "envdocs":
        return (
            "fail",
            f"{n} ROCKE_* variable{'s' * (n != 1)} read and not in {ENV_FLAGS}: "
            + ", ".join(x["name"] for x in items[:4])
            + (f" +{n - 4}" if n > 4 else ""),
        )
    if c == "testlists":
        return "fail", f"{n} new library test{'s' * (n != 1)} in no CMake or CTest list (no CI lane runs " f"{'them' if n != 1 else 'it'}): " + ", ".join(
            x["test"].rsplit("/", 1)[-1] for x in items[:3]
        ) + (
            f" +{n - 3}" if n > 3 else ""
        )
    if c == "cppenv":
        return (
            "fail",
            f"{n} getenv call{'s' * (n != 1)} in added C/C++ lines: "
            + ", ".join(x["at"] for x in items[:3])
            + (f" +{n - 3}" if n > 3 else ""),
        )
    if c == "title":
        return "fail", f"not a Conventional Commits title: {items[0]['why']}"
    kinds: dict[str, int] = {}
    for x in items:
        kinds[x["kind"]] = kinds.get(x["kind"], 0) + 1
    return "fail", f"{n} lead{'s' * (n != 1)}: " + ", ".join(
        f"{k} {v}" for k, v in kinds.items()
    )


def render(res: dict) -> list[str]:
    c = res["check"]
    tag, _key, _title, lead = CHECKS[c]
    word = "LEAD" if lead else "FLAG"
    out = []
    if res.get("error"):
        out.append(f"NOT RUN: {res['error']}")
    elif c == "title":
        out.append(f"title: {res['title']}")
        out += [
            f"{word} {x['why']} (platform/AGENTS.md: a Conventional Commits PR title, e.g. "
            "`feat(hip-kernel-provider): …`)"
            for x in res["items"]
        ]
    else:
        for x in res["items"][:SHOWN]:
            if c == "threads":
                out.append(
                    f"{word} {x['reviewer']} at {x['where']} ({x['at']}Z, {x['replies']} other repl"
                    f"{'ies' if x['replies'] != 1 else 'y'}): {x['url']}"
                )
            elif c == "trackers":
                out.append(f"{word} {x['at']}: {x['id']}")
            elif c == "envdocs":
                out.append(
                    f"{word} {x['at']}: reads {x['name']}, which {ENV_FLAGS} does not name"
                )
            elif c == "testlists":
                out.append(
                    f"{word} {x['test']}: no CMakeLists.txt, .cmake or YAML file names it or a directory "
                    "above it"
                )
            elif c == "cppenv":
                out.append(f"{word} {x['at']}: getenv({x['what']})")
            elif c in LINE_CHECKS:
                out.append(f"{word} {x['at']}: {x['what']}")
            else:
                out.append(f"{word} {x['kind']}: {x['at']}: {x['text']}")
        if len(res["items"]) > SHOWN:
            out.append(f"{word} +{len(res['items']) - SHOWN} more")
    out.append(" ".join((tag,) + verdict(res)))
    return out
