# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Impact score of a review round: how much a change can break, however small it is. Scripts and git only.

A round's reviewed hunks (first round: the PR's own change; follow-up: the lines new since our last reviewed head, as
`size.py` measures them) get signals, each with a weight:

  path signals (every reviewed hunk of the file)
    shared      public headers, engine core, portable IR, bindings, runtime, dispatch, backend, helpers, arch data
    abi         an ABI header or source (abi.h, *_abi.*, rocke_abi.cpp)
    registry    CMake, CTest lists and tiers, install rules
    ci          CI config (.github/, CI YAML, the host check runner)
    golden      goldens and recorded hashes (gated files count here even though nobody reads them line by line)
    hot         a file develop broke on before (config hot_spots, develop breakage issues filed before the round)
    twin        a file with a twin in the other engine that this PR does not change
    clash       a file another open PR changes too
  content signals (per hunk)
    abi_line    ABI version counters, struct layouts in headers
    ctest_line  add_test / tier labels / install() in CMake lines
    validator   a refusal, guard or check changed (raise, reject, is_valid, assert, throw, ROCKE_CHECK)
    removed     a definition removed or renamed; `removed_used` when other files still name it at the head
    fanin       callers and importers of the hunk's symbols and of its module or header (both engines)
  round signal
    develop     (follow-up) develop changed this PR's own files since our last reviewed head

hunk score = sum of the weights of its distinct signals; round score = the highest hunk score;
tier: low < 3 <= medium < 6 <= high. Weights and bounds are a priori, not fitted (the prototype's validate.py and
calib.py, in scratch/hybrid/impact/, report them against the eval keys and the ledger).

Impact raises a round's depth; it never removes an agent, an effort or a budget.

The path signals and the engines read a path from below the folder the monitor reviews (`folder`, the configuration's
repository.folder, which the caller passes in; blank: the repository's root), and the tree a round's twins, names and
importers are looked for in is that folder's.
"""

import collections
import os
import re
import subprocess

from . import size

W = {
    "shared": 3,
    "abi": 4,
    "registry": 3,
    "ci": 3,
    "golden": 2,
    "hot": 2,
    "twin": 3,
    "clash": 2,
    "abi_line": 4,
    "ctest_line": 3,
    "validator": 2,
    "removed": 2,
    "removed_used": 4,
    "fanin1": 1,
    "fanin3": 2,
    "fanin10": 3,
    "develop": 3,
}
TIERS = ((3, "low"), (6, "medium"), (None, "high"))
SHARED = re.compile(
    r"(^|/)(platform/cpp/include/|platform/cpp/core/|platform/python/rocke/core/|"
    r"platform/cpp/portable_ir/|platform/python/rocke/portable_ir/|platform/cpp/bindings/|"
    r"platform/python/rocke/runtime/|platform/python/rocke/dispatch/|library/dispatch/|"
    r"platform/cpp/helpers/|platform/python/rocke/helpers/|platform/cpp/support/)|"
    r"(^|/)backend\.py$|/arch/data\.cpp$"
)
ABI_PATH = re.compile(r"(^|/)([\w.]*_abi|abi|rocke_abi)\.(h|hpp|cpp|py|c)$", re.I)
REGISTRY = re.compile(
    r"(^|/)(CMakeLists\.txt|[\w.-]+\.cmake|ctest[\w.-]*|[\w.-]*tiers?[\w.-]*\.(ya?ml|json|txt))$",
    re.I,
)
CI_PATH = re.compile(
    r"(^|/)(\.github/|\.gitlab-ci|ci/[\w./-]*\.ya?ml$|[\w.-]*\.ya?ml$(?<=workflow\.yml)|"
    r"tools/run_checks\.py$|external-builds/)"
)
GOLDEN = re.compile(
    r"(^|/)(golden|goldens)(/|$)|_sha256\.json$|llvm_gfx_all\.json$", re.I
)
ABI_LINE = re.compile(
    r"ABI_VERSION|BINARY_ABI|ABI_MAJOR|ABI_MINOR|\bstruct\s+\w+\s*\{|\boffsetof\b|static_assert\s*\(\s*sizeof"
)
CTEST_LINE = re.compile(
    r"\badd_test\s*\(|set_tests_properties|\bLABELS\b|\binstall\s*\(|rocke_add_\w*test|"
    r"\b(quick|standard|comprehensive|full)\b.*(tier|label)",
    re.I,
)
VALIDATOR = re.compile(
    r"\braise\b|\breject|\brefus|\bis_valid|\bvalidate|\bassert\b|\bthrow\b|ROCKE_CHECK|"
    r"\bsupports?\(|return\s+(False|false)\b|not supported|unsupported"
)
PY_DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
C_DEF = re.compile(r"^(?:[\w:<>,*&\s]+?\s+)?\*?&?([A-Za-z_]\w*)\s*\([^;]*$")
C_DEFINE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)")
CTX_PY = re.compile(r"@@[^@]*@@\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
CTX_C = re.compile(r"@@[^@]*@@.*?\b([A-Za-z_]\w*)\s*\(")
NOT_SYMBOLS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "static_cast",
    "reinterpret_cast",
    "main",
    "test",
    "setUp",
    "tearDown",
    "__init__",
    "__call__",
    "__repr__",
    "run",
    "get",
    "set",
    "name",
    "self",
    "cls",
}


def tier(score: int) -> str:
    for lim, name in TIERS:
        if lim is None or score < lim:
            return name
    return TIERS[-1][1]


def rel(path: str, folder: str) -> str:
    """PATH from below FOLDER (no slash at either end; blank: the repository's root); a path outside it in full."""
    return path[len(folder) + 1 :] if folder and path.startswith(folder + "/") else path


def _within(folder: str) -> list[str]:
    """The pathspec of a git command that reads FOLDER only: none for the repository's root."""
    return ["--", folder] if folder else []


def engine(path: str, folder: str) -> str | None:
    p = rel(path, folder)
    if p.endswith(".py") and (
        p.startswith("library/") or p.startswith("platform/python/")
    ):
        return "py"
    if re.search(r"\.(cpp|cc|hpp|h)$", p) and p.startswith("platform/cpp/"):
        return "cpp"
    return None


def stem(path: str) -> str:
    return re.sub(r"\.(py|cpp|cc|hpp|h)$", "", path.rsplit("/", 1)[-1])


# --- pure signal functions (unit-tested) -------------------------------------------------------------------------


def path_signals(path: str, folder: str, hot: set[str] = frozenset()) -> set[str]:
    out = set()
    p = rel(path, folder)
    if SHARED.search(p):
        out.add("shared")
    if ABI_PATH.search(p):
        out.add("abi")
    if REGISTRY.search(p):
        out.add("registry")
    if CI_PATH.search(path):
        out.add("ci")
    if GOLDEN.search(p):
        out.add("golden")
    base = path.rsplit("/", 1)[-1]
    if (
        path in hot
        or base in hot
        or any(h and path.startswith(h) and h.endswith("/") for h in hot)
    ):
        out.add("hot")
    return out


def changed_lines(h: dict) -> list[str]:
    idx = h.get("dl_idx")
    lines = h["lines"]
    if idx:
        return [lines[k] for k in idx if lines[k][:1] in "+-"]
    return [x for x in lines if x[:1] in "+-"]


def content_signals(path: str, h: dict, folder: str) -> set[str]:
    out = set()
    ch = changed_lines(h)
    body = "\n".join(x[1:] for x in ch)
    if ABI_LINE.search(body) and (re.search(r"\.(h|hpp)$", path) or "ABI" in body):
        out.add("abi_line")
    if REGISTRY.search(rel(path, folder)) and CTEST_LINE.search(body):
        out.add("ctest_line")
    if re.search(r"\.(py|cpp|cc|hpp|h|c)$", path) and VALIDATOR.search(body):
        out.add("validator")
    if removed_defs(path, h):
        out.add("removed")
    return out


def defs_in(path: str, lines: list[str]) -> set[str]:
    names = set()
    for x in lines:
        if path.endswith(".py"):
            m = PY_DEF.match(x)
        else:
            m = C_DEFINE.match(x) or (
                C_DEF.match(x) if re.search(r"\.(h|hpp|cpp|cc|c)$", path) else None
            )
        if m and m.group(1) not in NOT_SYMBOLS and len(m.group(1)) > 3:
            names.add(m.group(1))
    return names


def removed_defs(path: str, h: dict) -> set[str]:
    ch = changed_lines(h)
    gone = defs_in(path, [x[1:] for x in ch if x[:1] == "-"])
    back = defs_in(path, [x[1:] for x in ch if x[:1] == "+"])
    return gone - back


def symbols(path: str, h: dict) -> set[str]:
    """The names a hunk defines or changes: definitions on its changed lines, plus its enclosing function."""
    ch = changed_lines(h)
    names = defs_in(path, [x[1:] for x in ch])
    m = (CTX_PY if path.endswith(".py") else CTX_C).search(h.get("h", ""))
    if m and m.group(1) not in NOT_SYMBOLS and len(m.group(1)) > 3:
        names.add(m.group(1))
    return names


def fanin_signal(n: int) -> str | None:
    return (
        "fanin10" if n >= 10 else "fanin3" if n >= 3 else "fanin1" if n >= 1 else None
    )


def score(sig: set[str]) -> int:
    fan = [s for s in sig if s.startswith("fanin")]
    best = max((W[s] for s in fan), default=0)
    return sum(W[s] for s in sig if not s.startswith("fanin")) + best


# --- git-backed parts ---------------------------------------------------------------------------------------------


def git(repo: str, *args, timeout: int = 300) -> tuple[int, str]:
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    try:
        r = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            errors="replace",
        )
        return r.returncode, r.stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


class Tree:
    """File list and grep at one head, in the folder FOLDER (blank: the whole tree), cached."""

    def __init__(self, repo: str, head: str, folder: str):
        self.repo, self.head, self.folder = repo, head, folder
        rc, out = git(repo, "ls-tree", "-r", "--name-only", head, *_within(folder))
        self.files = out.splitlines() if rc == 0 else []
        self.by_stem = collections.defaultdict(list)
        for f in self.files:
            if engine(f, folder):
                self.by_stem[(engine(f, folder), stem(f))].append(f)

    def twin(self, path: str) -> list[str]:
        e = engine(path, self.folder)
        if not e:
            return []
        other = "cpp" if e == "py" else "py"
        return self.by_stem.get((other, stem(path)), [])

    def files_naming(self, names: list[str]) -> dict[str, set[str]]:
        """name -> files at head that contain it as a word (one git grep for all names)."""
        out = collections.defaultdict(set)
        names = [n for n in names if re.fullmatch(r"[A-Za-z_]\w*", n)]
        for i in range(0, len(names), 40):
            chunk = names[i : i + 40]
            rc, txt = git(
                self.repo,
                "grep",
                "-o",
                "-w",
                "-I",
                "-E",
                "|".join(chunk),
                self.head,
                *_within(self.folder),
            )
            for line in txt.splitlines():
                parts = line.split(":")
                if len(parts) >= 3:
                    out[parts[-1]].add(parts[1])
        return out

    def importers(self, path: str) -> int:
        """Files that import this Python module or include this header (both engines), at head."""
        p = rel(path, self.folder)
        st = stem(path)
        if p.endswith(".py"):
            if st == "__init__":
                st = p.rsplit("/", 2)[-2]
            pat = (
                r"^\s*(from|import)\s+[\w.]*\b"
                + re.escape(st)
                + r"\b|^\s*from\s+[\w.]+\s+import\s+.*\b"
                + re.escape(st)
                + r"\b"
            )
        elif re.search(r"\.(h|hpp)$", p):
            pat = r"#\s*include\s*[<\"][^>\"]*\b" + re.escape(path.rsplit("/", 1)[-1])
        else:
            return 0
        rc, txt = git(
            self.repo, "grep", "-l", "-I", "-E", pat, self.head, *_within(self.folder)
        )
        return len([x for x in txt.splitlines() if not x.endswith(path)])


def pr_files(repo: str, head: str, dev: str, cache: dict) -> set[str]:
    if head in cache:
        return cache[head]
    rc, mb = git(repo, "merge-base", head, dev)
    out = set()
    if rc == 0 and mb.strip():
        rc, txt = git(repo, "diff", "--name-only", "--no-renames", mb.strip(), head)
        out = set(txt.splitlines()) if rc == 0 else set()
    cache[head] = out
    return out


def score_round(
    repo: str,
    head: str,
    dev: str,
    old: str = "",
    hot: set[str] = frozenset(),
    others: dict | None = None,
    dev_prev: str = "",
    file_cache: dict | None = None,
    measured: tuple | None = None,
    *,
    folder: str,
) -> dict:
    """Signals and scores for one round. others: {pr: head} of other open PRs; dev_prev: develop at our last
    reviewed head (follow-ups), for the develop signal; measured: size.measure's result when the caller has it;
    folder: the folder the monitor reviews (module docstring).
    """
    file_cache = {} if file_cache is None else file_cache
    rec, files, reviewed = measured or size.measure(repo, head, dev, old)
    if rec.get("src") == "unknown":
        return {"ok": False, "why": "; ".join(rec.get("notes") or [])}
    tree = Tree(repo, head, folder)
    mine = {f["p"] for f in files} | {f.get("op") for f in files if f.get("op")}
    other_files = collections.defaultdict(set)
    other_twins = collections.defaultdict(set)  # (engine, stem) another open PR changes
    for pr, oh in (others or {}).items():
        for p in pr_files(repo, oh, dev, file_cache):
            other_files[p].add(pr)
            if engine(p, folder):
                other_twins[(engine(p, folder), stem(p))].add(pr)
    # the hunks this round reviews: gated files (goldens) count too, so impact sees a golden edit
    hunks = []
    for f in files:
        if f["cls"] == "stray":
            continue
        for i, h in enumerate(f["hunks"]):
            if h.get("skip") or not changed_lines(h):
                continue
            hunks.append((f, i, h))
    syms = collections.defaultdict(set)
    gone = collections.defaultdict(set)
    for f, i, h in hunks:
        for s in symbols(f["p"], h):
            syms[s].add(f["p"])
        for s in removed_defs(f["p"], h):
            gone[s].add(f["p"])
    names = sorted(syms, key=lambda s: -len(s))[:80] + sorted(gone)[:40]
    naming = tree.files_naming(sorted(set(names))) if names else {}
    imp_cache = {}
    out_hunks = []
    for f, i, h in hunks:
        p = f["p"]
        sig = path_signals(p, folder, hot) | content_signals(p, h, folder)
        if f["cls"] in size.GATED and "golden" not in sig and f["cls"] == "golden":
            sig.add("golden")
        tw = tree.twin(p)
        if tw and not any(t in mine for t in tw) and f["cls"] in ("code",):
            sig.add("twin")
        e = engine(p, folder)
        twin_prs = (
            other_twins.get(("cpp" if e == "py" else "py", stem(p)), set())
            if e
            else set()
        )
        if p in other_files or twin_prs:
            sig.add("clash")
        other_files_p = other_files.get(p, set()) | twin_prs
        fan = 0
        for s in symbols(p, h):
            fan = max(fan, len(naming.get(s, set()) - syms[s]))
        if p not in imp_cache:
            imp_cache[p] = tree.importers(p) if f["cls"] in ("code",) else 0
        fan = max(fan, imp_cache[p])
        fs = fanin_signal(fan)
        if fs:
            sig.add(fs)
        for s in removed_defs(p, h):
            if naming.get(s, set()) - {p}:
                sig.add("removed_used")
        out_hunks.append(
            {
                "p": p,
                "i": i,
                "ns": h["ns"],
                "nl": h["nl"],
                "n": len(changed_lines(h)),
                "cls": f["cls"],
                "sig": sorted(sig),
                "score": score(sig),
                "fan": fan,
                "clash_prs": sorted(other_files_p),
            }
        )
    round_sig = set()
    if old and dev_prev:
        dev_files = set(
            git(repo, "diff", "--name-only", dev_prev, dev, "--", *sorted(mine)[:400])[
                1
            ].splitlines()
        )
        if dev_files:
            round_sig.add("develop")
    top = max((h["score"] for h in out_hunks), default=0) + sum(W[s] for s in round_sig)
    kinds = collections.Counter(s for h in out_hunks for s in h["sig"])
    return {
        "ok": True,
        "head": head,
        "old": old,
        "rnd": rec["rnd"],
        "eff": rec["eff"],
        "pr_eff": rec["pr_eff"],
        "size_tier": rec["tier"],
        "score": top,
        "tier": tier(top),
        "round_sig": sorted(round_sig),
        "kinds": dict(kinds),
        "n_hunks": len(out_hunks),
        "n_high": sum(1 for h in out_hunks if h["score"] >= 6),
        "n_med": sum(1 for h in out_hunks if 3 <= h["score"] < 6),
        "hunks": out_hunks,
    }
