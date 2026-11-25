# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk review-split N`: a review too large for one verifier, cut into parts that one verifier each reads whole, and a
cross-cutting pass that writes the single post.

The PR's own change (its merge base with its base branch, the PR's `base.ref`, to its head) is read from the mirror and
classed by `hybrid.size` (code, test and doc are read, small hand-edited JSON and YAML configs among them; generated,
golden and data files, large JSON, lockfiles and binaries are summarized, not read; pure renames count no lines). Each read file gets an area (AREAS: library kernels and builders
per family, the library's other packages, platform C++ by directory, the bindings, platform Python by package, the
Rust crates, the tests by tree, the docs, CMake/CI and test lists; another project by its top directories). Files
that change together stay in one part (LINKS: a C++ mirror and its Python twin, a header and its source, a test and
the module it is named after or imports, a CMake or YAML list and the files it names), as long as the group stays
within the part limit. A file over the limit is cut into slices. Each part is filled up to `max_lines` changed lines
before the next one opens, in path order area by area (pack): a part may hold several areas, and a slice may share a
part with other files.

The review directory gets `parts/partK.md` (the part's files with their lines, the command that prints its diff, the
checklist lines of its areas quoted from the verifier team file, the code quality checks quoted from the code quality
reviewer's team file, the files it shares with other parts, and the summarized data near it), `parts/partK.files` (its
paths) and `parts/cross.md` (the cross-cutting pass's inputs: the ABI files and public headers, the CMake, CTest and
tier lists, the hot spots, every link between parts, the checks' clash and overlap sections, the summarized data,
renames and stray files), and `parts/plan.json`. The output names the parts, the dispatch plan and its cost: one
verifier per part, which runs its part's code quality checks too, in waves under the subagent cap, then the
cross-cutting verifier.

Reads only (the mirror, the state, the spend card, the review directory's classify.txt and the checks, the verifier's
and the code quality reviewer's team files) and writes only `parts/`. The PR's head is fetched into the mirror under
its own ref, as `rk review-prep` does.
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import shlex

from . import config, gitops, paths, prep, registry, verifier_file
from .hybrid import size
from .queue import DEVELOP, base_of

# The part limit in changed lines. A part's diff is about 11 tokens a changed line, and its verifier reads around it
# (callers, the other engine, the tests), about three times that, on top of the verifier's fixed start (the lessons,
# the team file, facts, checks: 60-80K); a part holds up to `part_lines` changed lines, so a giant PR takes a few parts
# (review_procedure.md, Split review).


def part_lines(cfg: dict) -> int:
    """The most changed lines in one part (config split_part_lines: the deploy settings'
    features.review_automation.large_prs.split_part_lines; its default when blank, or not a whole number above 0).
    """
    v = registry.number(cfg or {}, "split_part_lines", int, registry.FALSY)
    return v if v > 0 else registry.default("split_part_lines")


def parts_for(lines: int, max_lines: int) -> int:
    """The fewest parts of at most `max_lines` changed lines that hold `lines`, at least one: the slices of a file over
    the limit (chunks), and the parts an estimate counts (estimate)."""
    return max(1, math.ceil(lines / max_lines))


def agents(parts: int) -> int:
    """The agents a split of `parts` parts runs: a verifier per part, then the cross pass."""
    return parts + 1


def agent_cost(cfg: dict, store) -> float | None:
    """About what one agent of a split costs, in US$: the spend card's mean cost of a review (spend.review_mean), as a
    part is about one review's reading; None while spend is off or the card has no review.
    """
    from . import spend  # a call-time import: spend loads the report

    if store is None or not spend.on(cfg):
        return None
    return spend.review_mean(spend.load_card(store))


def cost_words(n_agents: int, each: float | None) -> str:
    """'about 11 agents', with their cost when one agent's is known (agent_cost): 'about 11 agents, about $90'."""
    if not each:
        return f"about {n_agents} agents"
    usd = max(1, round(float(f"{n_agents * each:.2g}")))  # two significant digits
    return f"about {n_agents} agents, about ${usd:,}"


def estimate(cfg: dict, store, lines: int, cost: bool = True) -> str:
    """The agents, and their cost when known, of an in-depth review of `lines` changed lines, by the split's rule:
    'about 11 agents, about $90' for 80,000 lines in parts of 8,000. `cost` False: the agents alone, as the team sees
    them (the cost, as all spend, stays on the admin side)."""
    n = agents(parts_for(lines, part_lines(cfg)))
    return cost_words(n, agent_cost(cfg, store) if cost else None)


MAX_AGENTS = 20  # subagents at once (Claude Code's cap)
DIR = "parts"
PLAN = "plan.json"  # in DIR: the parts, the waves of part verifiers and the links between parts (write)
TOKENS_PER_CHAR = 0.25

READ_CLASSES = size.REVIEWED  # code, test, doc
BUILD_NAME = re.compile(
    r"(^|/)(CMakeLists\.txt|[^/]*\.cmake|[^/]*test_categories[^/]*\.ya?ml|pytest\.ini|"
    r"pyproject\.toml|requirements[^/]*\.txt|setup\.(py|cfg)|Makefile|[^/]*\.mk)$"
)
BUILD_DIR = re.compile(r"(^|/)\.github/")
DOC_EXT = (".md", ".rst", ".adoc", ".txt")
CXX_EXT = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
    ".cu",
    ".hip",
)
PUBLIC_HEADER = re.compile(r"(^|/)include/.+\.(h|hh|hpp|hxx|inc)$")
ABI_FILE = re.compile(
    r"(^|/)abi\.(h|py)$|(^|/)include/rocke/(instance|helper)_[^/]*\.h$"
)
GENERIC_STEMS = {
    "__init__",
    "__main__",
    "conftest",
    "utils",
    "util",
    "common",
    "types",
    "base",
    "main",
    "config",
    "helpers",
    "helper",
    "test",
    "tests",
    "setup",
    "version",
    "index",
    "readme",
    "cmakelists",
    "mod",
    "lib",
    "defs",
    "constants",
    "api",
    "core",
    "impl",
    "internal",
}
GENERIC_DIRS = {
    "golden",
    "goldens",
    "data",
    "tests",
    "test",
    "expected",
    "baseline",
    "baselines",
    "reference",
    "generated",
    "tables",
    "datasets",
    "testdata",
    "snapshots",
    "full",
    "standard",
    "quick",
    "comprehensive",
}

# The area of a path under the rocKE tree: (pattern on the path inside the tree, area key, kind). The first match wins;
# `{1}` in a key is the pattern's first group. Kinds pick the checklist lines (CHECKS) and the area's title.
AREAS = (
    (r"^library/(?:kernels|builders)/([^/]+)/", "library/{1}", "kernels"),
    (r"^library/tests/", "library/tests", "tests"),
    (r"^library/([^/]+)/", "library/{1}", "library"),
    (r"^platform/cpp/bindings/", "platform/cpp/bindings", "bindings"),
    (r"^platform/cpp/([^/]+)/", "platform/cpp/{1}", "platform_cpp"),
    (r"^platform/python/rocke/runtime/", "platform/python/runtime", "runtime"),
    (r"^platform/python/rocke/([^/]+)/", "platform/python/{1}", "platform_python"),
    (r"^platform/python/", "platform/python", "platform_python"),
    (r"^platform/tests/([^/]+)/", "platform/tests/{1}", "tests"),
    (r"^platform/tests/", "platform/tests", "tests"),
    (r"^platform/rust/([^/]+)/", "platform/rust/{1}", "rust"),
    (r"^platform/tools/", "platform/tools", "tools"),
    (r"^platform/dsl_docs/", "docs", "docs"),
    (r"^platform/([^/]+)/", "platform/{1}", "platform_python"),
    (r"^([^/]+)/", "rocke/{1}", "code"),
)
KIND_TITLE = {
    "kernels": "library kernels and builders",
    "library": "library",
    "bindings": "platform C++ bindings",
    "platform_cpp": "platform C++",
    "runtime": "platform runtime",
    "platform_python": "platform Python",
    "rust": "platform Rust",
    "tools": "platform tools",
    "tests": "tests",
    "docs": "docs",
    "build": "CMake, CI and test lists",
    "code": "code",
}

PY_IMPORT = re.compile(r"^\s*(?:from\s+(\.*[\w.]*)\s+import\b|import\s+([\w.]+))")
CXX_INCLUDE = re.compile(r"^\s*#\s*include\s*[\"<]([^\">]+)[\">]")
GEN_BY = re.compile(r"generated\s+(?:by|with|from)\s+[`'\"]?([\w./-]+)", re.I)
ARCH = re.compile(r"gfx\d{3,4}[a-z]?")
RECORDS_EXT = (".jsonl",)  # machine-written records: always summarized
CONFIG_EXT = (
    ".json",
    ".yaml",
    ".yml",
)  # a config when small and hand-edited; a large JSON is generated data
CONFIG_MAX_LINES = (
    200  # changed lines of a JSON or YAML file read as a hand-edited config
)
LONG_LINE = 300  # average characters of a file's changed lines over which it is a table or a blob
TABLE_MIN_LINES = (
    50  # a file with this many added lines that are nearly all literals is a table
)
DATA_GROUPS = 40  # directories of summarized files, at most: deeper ones are counted under their parent
GENERATOR_FILE = re.compile(r"\.(py|sh|cmake|ya?ml|toml)$|(^|/)CMakeLists\.txt$")


# --- areas ------------------------------------------------------------------------------------------------------------


def reclass(f: dict) -> str:
    """The split's class of a file `hybrid.size` reads (code, test, doc; goldens and files in data folders are already
    size's golden and data): data when it is JSON lines, when its changed lines are long (a blob, a selector model),
    when it is JSON or YAML that is large or marked generated near its top, or when its added lines are nearly all
    literals (a table); else size's. A small hand-edited JSON or YAML config (CONFIG_MAX_LINES changed lines at most)
    is read."""
    if f["cls"] not in READ_CLASSES:
        return f["cls"]
    p = f["p"].lower()
    if p.endswith(RECORDS_EXT):
        return "data"
    ch = [x for h in f["hunks"] for x in h["lines"] if x[:1] in "+-"]
    if ch and sum(len(x) for x in ch) / len(ch) > LONG_LINE:
        return "data"
    added = _added(f)
    if p.endswith(CONFIG_EXT):
        if len(ch) <= CONFIG_MAX_LINES and not any(
            GEN_BY.search(x) or size.GEN_MARK.search(x) for x in added[:12]
        ):
            return f["cls"]
        if p.endswith(".json"):
            return "data"
    if (
        len(added) >= TABLE_MIN_LINES
        and (size.literal_share(added) or 0) >= size.DATA_SHARE
    ):
        return "data"
    return f["cls"]


def tree_of(path: str, rocke: str) -> str:
    """The tree a file belongs to for links: the rocKE tree, else its project (two directories)."""
    return "rocke" if path.startswith(rocke + "/") else "/".join(path.split("/")[:2])


def is_build(path: str) -> bool:
    return bool(BUILD_NAME.search(path) or BUILD_DIR.search(path))


def area_of(path: str, rocke: str) -> tuple[str, str]:
    """(area key, kind) of a read file. Under the rocKE tree by AREAS; elsewhere by its first directories (a project's
    tree and the two below it), with tests and docs apart."""
    if is_build(path):
        return "build", "build"
    if path.startswith(rocke + "/"):
        rel = path[len(rocke) + 1 :]
        if rel.lower().endswith(DOC_EXT) and not rel.startswith(
            ("library/kernels/", "library/builders/")
        ):
            return "docs", "docs"
        for rx, key, kind in AREAS:
            m = re.match(rx, rel)
            if m:
                return key.replace("{1}", m.group(1) if m.groups() else ""), kind
        return "rocke", "code"
    dirs = path.split("/")[:-1]
    root = 2 if dirs[:1] in (["projects"], ["dnn-providers"]) and len(dirs) >= 2 else 1
    if path.lower().endswith(DOC_EXT) or "docs" in dirs:
        return "/".join(dirs[:root] + ["docs"]), "docs"
    key = "/".join(dirs[: root + 2]) or "."
    return key, "tests" if size.is_test_path(path) else "code"


def group_of(key: str) -> str:
    """The tree an area belongs to, which orders the areas for packing: library, platform/<lang>, a project's top."""
    parts = key.split("/")
    if parts[0] == "platform" and len(parts) > 1:
        return "/".join(parts[:2])
    if parts[0] in ("projects", "dnn-providers") and len(parts) > 1:
        return "/".join(parts[:2])
    return parts[0]


# --- links: files that change together --------------------------------------------------------------------------------


def stem(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[0].lower() if "." in name else name.lower()


def parent(path: str) -> str:
    """The name of a path's directory."""
    dirs = path.split("/")[:-1]
    return dirs[-1] if dirs else ""


def _added(f: dict) -> list[str]:
    return [x[1:] for h in f["hunks"] for x in h["lines"] if x[:1] == "+"]


def _suffixes(paths: list[str]) -> dict[str, list[str]]:
    """Every path suffix (whole components) of each path -> the paths that end with it."""
    out: dict[str, list[str]] = {}
    for p in paths:
        comps = p.split("/")
        for i in range(len(comps)):
            out.setdefault("/".join(comps[i:]), []).append(p)
    return out


def _py_targets(path: str, line: str) -> list[str]:
    """The file suffixes a Python import line may name (a module path of at least two parts, or a relative one)."""
    m = PY_IMPORT.match(line)
    if not m:
        return []
    mod = m.group(1) if m.group(1) is not None else m.group(2)
    if mod.startswith("."):
        dots = len(mod) - len(mod.lstrip("."))
        base = path.split("/")[:-1]
        base = base[: len(base) - (dots - 1)] if dots > 1 else base
        rest = [x for x in mod.lstrip(".").split(".") if x]
        comps = base + rest
        if not rest:
            return []
    else:
        comps = mod.split(".")
        if len(comps) < 2:
            return []
    tail = "/".join(comps)
    return [tail + ".py", tail + "/__init__.py"]


def links(files: list[dict], rocke: str) -> list[tuple[str, str, str]]:
    """(a, b, why) for read files that change together, strongest first: twins (a C/C++ file and a Python file of one
    stem), a header and its source, a test and the module it is named after, a test and what it imports, and a CMake
    or YAML list and the files it names."""
    read = [f for f in files if f["cls"] in READ_CLASSES]
    paths = [f["p"] for f in read]
    by_stem: dict[str, list[str]] = {}
    for p in paths:
        by_stem.setdefault(stem(p).removeprefix("rocke_"), []).append(p)
    out: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str, why: str) -> None:
        if (
            a != b
            and (a, b) not in seen
            and (b, a) not in seen
            and tree_of(a, rocke) == tree_of(b, rocke)
        ):
            seen.add((a, b))
            out.append((a, b, why))

    def pairs(xs: list[str], ys: list[str]) -> list[tuple[str, str]]:
        """Every pair when one side has one file; else only the pairs in a directory of one name (gfx950 and gfx950)."""
        if len(xs) == 1 or len(ys) == 1:
            return [(a, b) for a in xs for b in ys]
        return [(a, b) for a in xs for b in ys if parent(a) == parent(b)]

    for s, ps in sorted(by_stem.items()):
        if s in GENERIC_STEMS or len(s) < 4:
            continue
        cxx = [p for p in ps if p.endswith(CXX_EXT) and not size.is_test_path(p)]
        py = [p for p in ps if p.endswith(".py") and not size.is_test_path(p)]
        for a, b in pairs(cxx, py):
            add(a, b, "C++ mirror and Python twin")
        heads = [p for p in cxx if p.endswith((".h", ".hh", ".hpp", ".hxx"))]
        srcs = [p for p in cxx if p not in heads]
        for a, b in pairs(heads, srcs):
            add(a, b, "header and source")
    code = [p for p in paths if not size.is_test_path(p)]
    code_by_stem: dict[str, list[str]] = {}
    for p in code:
        code_by_stem.setdefault(stem(p), []).append(p)
    for p in paths:
        if not size.is_test_path(p):
            continue
        s = stem(p)
        name = s.removeprefix("test_").removesuffix("_test")
        if name == s:
            continue
        hits = []
        for cs in code_by_stem:
            if cs in GENERIC_STEMS or len(cs) < 5:
                continue
            if name == cs or name.startswith(cs + "_"):
                hits.append(cs)
        if hits:
            best = max(hits, key=len)
            cands = [
                c for c in code_by_stem[best] if tree_of(c, rocke) == tree_of(p, rocke)
            ]
            arch = set(
                ARCH.findall(p)
            )  # a test of one family goes with that family's file
            cands = [c for c in cands if arch & set(ARCH.findall(c))] or cands
            for c in cands:
                add(p, c, f"test named after {best}")
    suffix = _suffixes(code)
    for f in read:
        p = f["p"]
        if not size.is_test_path(p):
            continue
        for line in _added(f):
            if p.endswith(".py"):
                for t in _py_targets(p, line):
                    for c in suffix.get(t, []):
                        add(p, c, "test imports it")
            elif p.endswith(CXX_EXT):
                m = CXX_INCLUDE.match(line)
                if m and "/" in m.group(1):
                    for c in suffix.get(m.group(1), []):
                        add(p, c, "test includes it")
    names: dict[str, list[str]] = {}
    for p in paths:
        names.setdefault(p.rsplit("/", 1)[-1], []).append(p)
    for f in read:
        if not is_build(f["p"]):
            continue
        for line in _added(f):
            for word in set(
                re.findall(r"[\w.+-]+\.(?:py|cpp|cc|c|h|hpp|cmake|yaml|yml|rs)\b", line)
            ):
                for c in names.get(word.rsplit("/", 1)[-1], []):
                    add(f["p"], c, "the list names it")
    return out


class Groups:
    """Union-find over files with each group's changed lines, joined only while the group stays within `cap`."""

    def __init__(self, lines: dict[str, int], cap: int):
        self.up = {p: p for p in lines}
        self.lines = dict(lines)
        self.cap = cap

    def find(self, p: str) -> str:
        while self.up[p] != p:
            self.up[p] = self.up[self.up[p]]
            p = self.up[p]
        return p

    def join(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.lines[ra] + self.lines[rb] > self.cap:
            return False
        self.up[rb] = ra
        self.lines[ra] += self.lines.pop(rb)
        return True


# --- the partition ----------------------------------------------------------------------------------------------------


def file_lines(f: dict) -> int:
    return f["a"] + f["d"]


def diff_tokens(f: dict) -> int:
    return int(
        sum(len(x) + 1 for h in f["hunks"] for x in h["lines"]) * TOKENS_PER_CHAR
    )


def pack(items: list[dict], cap: int) -> list[list[dict]]:
    """Items (in order, each {"lines"} within `cap`) in parts of at most `cap` lines, each filled before another opens:
    an item goes into the last part when it fits there, else into the first part it fits, else into a new part. Parts
    come close to the fewest that hold the lines (parts_for), and a part may hold items of several areas.
    """
    parts: list[list[dict]] = []
    room: list[int] = []
    for it in items:
        k = len(parts) - 1
        if not parts or it["lines"] > room[k]:
            k = next((i for i, r in enumerate(room) if it["lines"] <= r), len(parts))
            if k == len(parts):
                parts.append([])
                room.append(cap)
        parts[k].append(it)
        room[k] -= it["lines"]
    return parts


def chunks(f: dict, cap: int) -> list[dict]:
    """A file over `cap` changed lines cut into about even runs of at most `cap`, at a hunk boundary when one is near:
    [{"lines", "new": (first, last) | None, "old": (first, last) | None}], line numbers of the new and the old file.
    """
    seq = []  # (hunk, old line or 0, new line or 0) per changed line
    for i, h in enumerate(f["hunks"]):
        o, n = h["os"], h["ns"]
        for x in h["lines"]:
            t = x[:1]
            if t == "+":
                seq.append((i, 0, n))
                n += 1
            elif t == "-":
                seq.append((i, o, 0))
                o += 1
            else:
                o, n = o + 1, n + 1
    total = len(seq)
    bounds, start = [], 0
    while total - start > cap:
        left = parts_for(total - start, cap)  # slices still needed
        end = start + math.ceil((total - start) / left)
        # A hunk boundary a little before the even cut, when the rest still fits in the slices left.
        near = next(
            (
                j
                for j in range(end, int(end - 0.15 * (end - start)), -1)
                if seq[j][0] != seq[j - 1][0] and total - j <= (left - 1) * cap
            ),
            None,
        )
        end = near or end
        bounds.append((start, end))
        start = end
    bounds.append((start, total))
    out = []
    for a, b in bounds:
        news = [x[2] for x in seq[a:b] if x[2]]
        olds = [x[1] for x in seq[a:b] if x[1]]
        out.append(
            {
                "lines": b - a,
                "new": (min(news), max(news)) if news else None,
                "old": (min(olds), max(olds)) if olds else None,
            }
        )
    return out


def partition(files: list[dict], rocke: str, max_lines: int) -> dict:
    """{"parts": [{"areas", "kinds", "files", "ranges", "lines"}], "links": [(a, b, why)], "cut": [(a, b, why)]}: the
    read files with changed lines in parts of at most `max_lines`, each filled before the next (pack). Files that change
    together go in as one item while it stays within the limit; a file over the limit is cut into slices (`ranges`:
    path -> its slices in this part); the items go in path order, area by area. A part's areas are its files' (most
    lines first) and its kinds theirs. `cut` holds the links whose files did not end in one part.
    """
    read = [f for f in files if f["cls"] in READ_CLASSES and file_lines(f) > 0]
    by_path = {f["p"]: f for f in read}
    area = {p: area_of(p, rocke) for p in by_path}
    big = {p for p, f in by_path.items() if file_lines(f) > max_lines}
    g = Groups(
        {p: file_lines(f) for p, f in by_path.items() if p not in big}, max_lines
    )
    found = links(read, rocke)
    for a, b, _ in found:
        if a not in big and b not in big:
            g.join(a, b)
    members: dict[str, list[str]] = {}
    for p in sorted(set(by_path) - big):
        members.setdefault(g.find(p), []).append(p)
    items = []
    for ps in members.values():
        weight: dict[str, int] = {}
        for p in ps:
            weight[area[p][0]] = weight.get(area[p][0], 0) + file_lines(by_path[p])
        items.append(
            {
                "area": max(sorted(weight), key=lambda k: weight[k]),
                "paths": ps,
                "lines": sum(file_lines(by_path[p]) for p in ps),
            }
        )
    for p in sorted(big):
        cs = chunks(by_path[p], max_lines)
        items += [
            {
                "area": area[p][0],
                "paths": [p],
                "lines": c["lines"],
                "slice": dict(c, of=(i, len(cs))),
            }
            for i, c in enumerate(cs, 1)
        ]
    items.sort(
        key=lambda i: (
            group_of(i["area"]),
            i["area"],
            i["paths"][0],
            (i.get("slice") or {}).get("of", (0,))[0],
        )
    )
    parts = []
    for run in pack(items, max_lines):
        ranges: dict[str, list[dict]] = {}
        weight = {}
        for it in run:
            if it.get("slice"):
                ranges.setdefault(it["paths"][0], []).append(it["slice"])
            for p in it["paths"]:
                lines = it["lines"] if it.get("slice") else file_lines(by_path[p])
                weight[area[p][0]] = weight.get(area[p][0], 0) + lines
        part_files = sorted({p for it in run for p in it["paths"]})
        parts.append(
            {
                "areas": sorted(weight, key=lambda k: (-weight[k], k)),
                "kinds": sorted(
                    {area[p][1] for p in part_files},
                    key=list(verifier_file.CHECKS).index,
                ),
                "files": part_files,
                "ranges": ranges,
                "lines": sum(i["lines"] for i in run),
            }
        )
    where = holders(parts)
    cut = [
        (a, b, why)
        for a, b, why in found
        if where.get(a) != where.get(b) or len(where.get(a, [])) > 1
    ]
    return {"parts": parts, "links": found, "cut": cut}


def holders(parts: list[dict]) -> dict[str, list[int]]:
    """Each file's parts (1-based): one, or several for a file cut into ranges."""
    out: dict[str, list[int]] = {}
    for k, part in enumerate(parts, 1):
        for p in part["files"]:
            out.setdefault(p, []).append(k)
    return out


# --- generated, golden and data files ---------------------------------------------------------------------------------


def data_groups(files: list[dict], read: list[dict], rocke: str) -> list[dict]:
    """The files summarized, not read, by directory and class: {"dir", "cls", "files", "lines", "area", "generator"}.
    The generator: a "generated by X" line at the top of an added file, else the read files whose added lines name
    the directory or one of its files, else ""."""
    summarized = [f for f in files if f["cls"] not in READ_CLASSES]
    dirs = {
        f["p"]: f["p"].rsplit("/", 1)[0] if "/" in f["p"] else "." for f in summarized
    }
    while len({(dirs[f["p"]], f["cls"]) for f in summarized}) > DATA_GROUPS:
        deep = max(d.count("/") for d in dirs.values())
        if deep == 0:
            break
        dirs = {
            p: d.rsplit("/", 1)[0] if d.count("/") == deep else d
            for p, d in dirs.items()
        }
    groups: dict[tuple[str, str], dict] = {}
    for f in summarized:
        d = dirs[f["p"]]
        e = groups.setdefault(
            (d, f["cls"]),
            {
                "dir": d,
                "cls": f["cls"],
                "files": 0,
                "lines": 0,
                "names": [],
                "marks": set(),
            },
        )
        e["files"] += 1
        e["lines"] += file_lines(f)
        e["names"].append(f["p"].rsplit("/", 1)[-1])
        for x in _added(f)[:12]:
            m = GEN_BY.search(x)
            if m:
                e["marks"].add(m.group(1).rstrip(".,"))
    # Generators are scripts, tests and lists: only those files' added lines are searched.
    text = {f["p"]: "\n".join(_added(f)) for f in read if GENERATOR_FILE.search(f["p"])}
    out = []
    for (d, cls), e in sorted(groups.items()):
        gen = sorted(m for m in e["marks"] if len(m) >= 3)[:3]
        if not gen:
            # A file that names one of the data files comes first; the directory's name counts only when it is
            # distinctive (SdpaFwdPlan, selector_models), not a word like "arch".
            leaf = d.rsplit("/", 1)[-1]
            names = [x for x in e["names"][:8] if len(x) >= 8]
            dir_key = (
                leaf
                if leaf.lower() not in GENERIC_DIRS
                and (len(leaf) >= 8 or re.search(r"[A-Z_]", leaf))
                else ""
            )
            scored = []
            for p, t in text.items():
                score = (
                    2
                    if any(x in t for x in names)
                    else 1 if dir_key and dir_key in t else 0
                )
                if score:
                    scored.append((-score, p))
            gen = [p for _, p in sorted(scored)[:3]]
        out.append(
            {
                "dir": d,
                "cls": cls,
                "files": e["files"],
                "lines": e["lines"],
                "area": area_of(d + "/x", rocke)[0],
                "generator": ", ".join(gen),
            }
        )
    return out


# --- the verifier team file's checklist -------------------------------------------------------------------------------


def checklist(team_text: str | None, kinds: list[str]) -> list[str]:
    """The checklist lines for these kinds (verifier_file.leads), each bullet quoted whole (its continuation and
    sub-bullets) from the team file; without the file, the leads alone; a lead the file lacks says so.
    """
    leads = verifier_file.leads(kinds)
    if not team_text:
        return [
            f"- **{lead}** (the verifier team file, What to Check)" for lead in leads
        ]
    lines = team_text.splitlines()
    out = []
    for lead in leads:
        i = verifier_file.bullet(lines, lead)
        if i < 0:
            out.append(f"- !! not in the verifier team file: **{lead}**")
            continue
        line = lines[i]
        ind = len(verifier_file.BULLET.match(line).group(1))
        block = [line[ind:]]
        for nxt in lines[i + 1 :]:
            if not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= ind:
                break
            block.append(nxt[ind:])
        out += block
    return out


def team_text(cfg: dict, path: str | None) -> tuple[str | None, str]:
    """The verifier team file's text and its path (`--team-file`, else config paths.verifier_team_file)."""
    p = path or ((cfg.get("paths") or {}).get("verifier_team_file") or "")
    if not p:
        return None, ""
    p = str(pathlib.Path(p).expanduser())
    try:
        return pathlib.Path(p).read_text(encoding="utf-8"), p
    except OSError:
        return None, p


# --- the code quality reviewer's team file's checks -------------------------------------------------------------------

# The code quality checks a part's verifier runs on its part: the bold leads of the numbered items of What to Check in
# the code quality reviewer's team file, each quoted whole. The file's last item, overlapping work in other open PRs, is
# the cross pass's, as is one home per fact across parts.
QUALITY_LEADS = (
    "Single responsibility.",
    "DRY.",
    "Primitives and layers.",
    "Interfaces from",
    "Names.",
    "Claims and tests.",
    "Both engines and root cause.",
)
NUMBERED = re.compile(r"^(\s*)\d+\. \*\*(.+)$")  # a numbered item with a bold lead


def quality_checklist(text: str | None) -> list[str]:
    """The code quality checks a part quotes (QUALITY_LEADS), each item quoted whole (its continuation lines,
    sub-bullets and paragraphs) from the What to Check section of the code quality reviewer's team file TEXT; without
    the file, the leads alone; a lead the file lacks says so."""
    if not text:
        return [
            f"- **{lead}** (the code quality reviewer's team file, What to Check)"
            for lead in QUALITY_LEADS
        ]
    lines = text.splitlines()
    start = next(
        (i for i, x in enumerate(lines) if x.startswith("### What to Check")),
        len(lines),
    )
    found = {}
    for i in range(start, len(lines)):
        m = NUMBERED.match(lines[i])
        lead = m and next((x for x in QUALITY_LEADS if m.group(2).startswith(x)), None)
        if lead and lead not in found:
            found[lead] = (i, len(m.group(1)))
    out = []
    for lead in QUALITY_LEADS:
        if lead not in found:
            out.append(f"- !! not in the code quality reviewer's team file: **{lead}**")
            continue
        i, ind = found[lead]
        block = [lines[i][ind:]]
        for nxt in lines[i + 1 :]:
            if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= ind:
                break
            block.append(nxt[ind:])
        while not block[-1].strip():
            block.pop()
        out += block
    return out


def quality_text(team_path: str) -> tuple[str | None, str]:
    """The code quality reviewer's team file's text and its path: the file beside the verifier team file `team_path`,
    where the agentic install puts both (paths.QUALITY_FILE)."""
    if not team_path:
        return None, ""
    p = str(
        pathlib.Path(team_path).with_name(pathlib.PurePath(paths.QUALITY_FILE).name)
    )
    try:
        return pathlib.Path(p).read_text(encoding="utf-8"), p
    except OSError:
        return None, p


# --- waves ------------------------------------------------------------------------------------------------------------


def waves(parts: list[dict], cap: int) -> list[list[dict]]:
    """The parts to dispatch, a verifier each, in waves of at most `cap`, in order."""
    return [parts[i : i + cap] for i in range(0, len(parts), cap)]


def _span(ks: list[int]) -> str:
    """1,2,3,5 -> part1-part3, part5."""
    out, i = [], 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1:
            j += 1
        out.append(f"part{ks[i]}" + (f"-part{ks[j]}" if j > i else ""))
        i = j + 1
    return ", ".join(out)


# --- the files --------------------------------------------------------------------------------------------------------


def _section(text: str, title: str, most: int = 150) -> list[str]:
    """A checks.md section by its title's start, at most `most` lines."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("## " + title):
            body = [line]
            for nxt in lines[i + 1 :]:
                if nxt.startswith("## "):
                    break
                body.append(nxt)
            return body[:most] + (
                [f"… {len(body) - most} more lines in checks.md"]
                if len(body) > most
                else []
            )
    return []


def _label(part: dict) -> str:
    kinds = [KIND_TITLE.get(k, k) for k in part["kinds"]]
    return f"{', '.join(part['areas'][:4])}{' …' if len(part['areas']) > 4 else ''} ({', '.join(kinds)})"


def _slice(s: dict) -> str:
    """A range of a file cut across parts, in words."""
    span = [f"new-file lines {s['new'][0]}-{s['new'][1]}"] if s.get("new") else []
    span += [f"old-file lines {s['old'][0]}-{s['old'][1]}"] if s.get("old") else []
    return f"slice {s['of'][0]} of {s['of'][1]} ({s['lines']} changed lines: {', '.join(span)})"


def _parts_of(ks: list[int] | None) -> str:
    return "+".join(f"part{k}" for k in ks) if ks else "?"


def write(
    cfg: dict,
    n: int,
    out: pathlib.Path,
    head: str,
    mb: str,
    base: str,
    rnd: str,
    files: list[dict],
    plan: dict,
    data: list[dict],
    team: tuple[str | None, str],
    quality: tuple[str | None, str],
    changed: list[str] | None,
    max_lines: int,
    cap: int,
) -> dict:
    """Write parts/ in the review directory `out`; the plan (also parts/plan.json). TEAM and QUALITY: the verifier's
    and the code quality reviewer's team files, each (its text or None, its path)."""
    rocke = paths.repository_folder(cfg)
    repo = cfg["paths"]["repo"]
    d = out / DIR
    d.mkdir(parents=True, exist_ok=True)
    for old in d.iterdir():
        if re.fullmatch(r"part\d+\.(md|files)", old.name):
            old.unlink()
    by_path = {f["p"]: f for f in files}
    parts = plan["parts"]
    where = holders(parts)

    def is_changed(p: str) -> bool:
        return changed is not None and (
            p in changed
            or (p.startswith(rocke + "/") and p[len(rocke) + 1 :] in changed)
        )

    text, team_path = team
    qtext, quality_path = quality
    for k, part in enumerate(parts, 1):
        part["k"] = k
        part["tokens"] = sum(
            (
                diff_tokens(by_path[p])
                * sum(s["lines"] for s in part["ranges"][p])
                // max(file_lines(by_path[p]), 1)
                if p in part["ranges"]
                else diff_tokens(by_path[p])
            )
            for p in part["files"]
        )
        part["changed"] = (
            sum(1 for p in part["files"] if is_changed(p))
            if changed is not None
            else None
        )
        (d / f"part{k}.files").write_text(
            "".join(p + "\n" for p in part["files"]), encoding="utf-8"
        )
        body = [
            f"# PR #{n} part {k} of {len(parts)}: {_label(part)}",
            "",
            f"Head `{head[:10]}`, merge base `{mb[:10]}` ({base}); {part['lines']} changed lines in "
            f"{len(part['files'])} files, about {part['tokens'] // 1000}K tokens of diff.",
            "",
            "Its diff: `xargs -a "
            + shlex.quote(str(d / f"part{k}.files"))
            + f" git -C {repo} diff {mb[:12]} "
            f"{head[:12]} --`. Read every file's hunks whole (a sliced file: only its slice's hunks); open the rest "
            "of the tree only to follow a caller, a twin or a test.",
            "",
            "## Files",
            "",
            "| File | Class | + | − | Note |",
            "|---|---|---|---|---|",
        ]
        for p in part["files"]:
            f = by_path[p]
            note = []
            if f["st"] in ("A", "D", "R"):
                note.append(
                    {"A": "new", "D": "deleted", "R": f"a rename of `{f['op']}`"}[
                        f["st"]
                    ]
                )
            for s in part["ranges"].get(p, []):
                note.append(_slice(s) + f"; the file is in {_parts_of(where[p])}")
            if is_changed(p):
                note.append("changed since our last review")
            body.append(
                f"| `{p}` | {f['cls']} | {f['a']} | {f['d']} | {'; '.join(note)} |"
            )
        body += [
            "",
            "## What to check",
            "",
            f"From the verifier team file ({team_path or 'not configured: paths.verifier_team_file'}), for "
            f"{', '.join(KIND_TITLE.get(x, x) for x in part['kinds'])}; the rest of What to Check applies as "
            "always.",
            "",
        ]
        body += checklist(text, part["kinds"])
        body += [
            "",
            "## Code quality",
            "",
            "From the code quality reviewer's team file "
            f"({quality_path or 'not read: it lies beside paths.verifier_team_file'}), What to Check: run these on "
            f"this part's files and write their items into `part{k}_findings.md`. The cross pass checks one home per "
            "fact across parts and overlapping work in other open PRs.",
            "",
        ]
        body += quality_checklist(qtext)
        mine = set(part["files"])
        touch = [(a, b, why) for a, b, why in plan["cut"] if a in mine or b in mine]
        body += ["", f"## Touches other parts ({len(touch)})", ""]
        body += [
            f"- `{a if a in mine else b}` ↔ `{b if a in mine else a}` ({_parts_of(where.get(b if a in mine else a))}"
            f"): {why}"
            for a, b, why in touch[:100]
        ] or ["- none"]
        if len(touch) > 100:
            body.append(f"- … {len(touch) - 100} more (cross.md)")
        near = [x for x in data if x["area"] in part["areas"]]
        body += [
            "",
            "## Summarized, not read (generated, golden and data files in its areas)",
            "",
        ]
        body += [
            f"- `{x['dir']}/`: {x['files']} {x['cls']} files, {x['lines']} lines"
            f"{'; generator: ' + x['generator'] if x['generator'] else ''}"
            for x in near
        ] or ["- none"]
        if changed is not None:
            body += [
                "",
                f"Changed since our last review: {part['changed']} of its {len(part['files'])} files.",
            ]
        body += [
            "",
            "## Output",
            "",
            f"`part{k}_notes.md` and `part{k}_findings.md` in {out} (the verifier team file, Part review).",
            "",
        ]
        (d / f"part{k}.md").write_text("\n".join(body), encoding="utf-8")

    cross = [
        f"# PR #{n} cross-cutting pass",
        "",
        f"Head `{head[:10]}`, merge base `{mb[:10]}` ({base}); {len(parts)} parts. The verifier team file's Cross "
        "pass section says what it does; it writes the single post.",
        "",
        "## Inputs",
        "",
        f"- every part's `partK_findings.md` (its code quality items among them) and `partK_notes.md` in {out}",
        "- `facts.txt`, `classify.txt`, `files.txt`, `overlap_section.md` in the same directory",
        f"- the checks: {pathlib.Path(cfg['paths']['workers']) / f'pr{n}' / 'checks' / 'checks.md'}",
        "",
    ]

    def listing(title: str, paths: list[str]) -> list[str]:
        rows = [
            f"- `{p}`: {file_lines(by_path[p])} lines, "
            + (
                _parts_of(where[p])
                if p in where
                else (
                    "summarized"
                    if by_path[p]["cls"] not in READ_CLASSES
                    else "no changed lines"
                )
            )
            for p in paths
        ]
        return [f"## {title}", ""] + (rows or ["- none"]) + [""]

    every = sorted(by_path)
    cross += listing(
        "ABI files and public headers",
        [p for p in every if ABI_FILE.search(p) or PUBLIC_HEADER.search(p)],
    )
    cross += listing("CMake, CTest and tier lists", [p for p in every if is_build(p)])
    cross += ["## Hot spots (config hot_spots)", ""]
    hits = 0
    for name, val in sorted((cfg.get("hot_spots") or {}).items()):
        for h in (val if isinstance(val, list) else [val]):
            hit = [p for p in every if p == h or (h.endswith("/") and p.startswith(h))]
            if hit:
                hits += 1
                ks = sorted({k for p in hit for k in where.get(p, [])})
                cross.append(
                    f"- {name} `{h}`: {len(hit)} files, {sum(file_lines(by_path[p]) for p in hit)} lines; "
                    f"{_parts_of(ks) if ks else 'summarized'}"
                )
    cross += ([] if hits else ["- none"]) + [
        "",
        f"## Links between parts ({len(plan['cut'])})",
        "",
    ]
    cross += [
        f"- `{a}` ({_parts_of(where.get(a))}) ↔ `{b}` ({_parts_of(where.get(b))}): {why}"
        for a, b, why in plan["cut"][:300]
    ] or ["- none"]
    if len(plan["cut"]) > 300:
        cross.append(f"- … {len(plan['cut']) - 300} more ({DIR}/{PLAN})")
    sliced = sorted(p for p in where if len(where[p]) > 1)
    cross += ["", f"## Files cut across parts ({len(sliced)})", ""]
    cross += [
        f"- `{p}`: {file_lines(by_path[p])} lines in {_parts_of(where[p])}"
        for p in sliced
    ] or ["- none"]
    checks = pathlib.Path(cfg["paths"]["workers"]) / f"pr{n}" / "checks" / "checks.md"
    try:
        ctext = checks.read_text(encoding="utf-8", errors="replace")
    except OSError:
        ctext = ""
    for title, cmd in (
        ("Clashes with other open PRs", f"rk clash {n}"),
        ("Overlapping work in other open PRs", f"rk overlaps {n} --section"),
    ):
        sec = _section(ctext, title)
        cross += [""] + (
            sec or [f"## {title}", "", f"- not in the checks yet: run `{cmd}`"]
        )
    cross += [
        "",
        f"## Generated, golden and data files ({sum(x['files'] for x in data)} files, "
        f"{sum(x['lines'] for x in data)} lines: summarized, not read)",
        "",
    ]
    cross += [
        f"- `{x['dir']}/`: {x['files']} {x['cls']}, {x['lines']} lines; generator: "
        f"{x['generator'] or 'not named in the PR'}"
        for x in data
    ] or ["- none"]
    renames = [f for f in files if f["st"] == "R" and file_lines(f) == 0]
    cross += ["", f"## Pure renames ({len(renames)})", ""]
    cross += [f"- `{f['op']}` → `{f['p']}`" for f in renames[:100]] or ["- none"]
    if len(renames) > 100:
        cross.append(f"- … {len(renames) - 100} more")
    stray = [f["p"] for f in files if f["cls"] == "stray"]
    cross += (
        ["", "## Stray files", ""] + ([f"- `{p}`" for p in stray] or ["- none"]) + [""]
    )
    (d / "cross.md").write_text("\n".join(cross), encoding="utf-8")

    todo = [p for p in parts if changed is None or p["changed"]]
    result = {
        "pr": n,
        "round": rnd,
        "head": head,
        "merge_base": mb,
        "base": base,
        "max_lines": max_lines,
        "agent_cap": cap,
        "parts": [
            {
                k: p[k]
                for k in (
                    "k",
                    "areas",
                    "kinds",
                    "lines",
                    "tokens",
                    "changed",
                    "files",
                    "ranges",
                )
            }
            for p in parts
        ],
        "waves": [[p["k"] for p in w] for w in waves(todo, cap)],
        "data": data,
        "links_between_parts": plan["cut"],
    }
    (d / PLAN).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


# --- the command ------------------------------------------------------------------------------------------------------


def changed_since(out: pathlib.Path) -> list[str] | None:
    """A follow-up's files changed since our reviewed head (classify.txt's changed_files), else None."""
    try:
        c = json.loads((out / "classify.txt").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return c.get("changed_files") if isinstance(c, dict) and c.get("old") else None


def agent_call(
    n: int, rnd: str, out: pathlib.Path, k: int = 0, of: int = 0
) -> tuple[str, pathlib.Path]:
    """The Agent description and the brief a split review's agent is dispatched with: part k of `of`, or the cross pass
    without k. Each names the PR and the round (the brief lies in the review directory, v<N><round>), which is how the
    orchestrator knows which review the agent verifies (orchestrator.brief_pr), so it does not hand the review's
    REVIEW_READY line back to the session while the agents run."""
    if k:
        return f"PR {n} {rnd} part {k} of {of}", out / DIR / f"part{k}.md"
    return f"PR {n} {rnd} cross pass", out / DIR / "cross.md"


def summary(
    n: int,
    rnd: str,
    out: pathlib.Path,
    files: list[dict],
    res: dict,
    cap: int,
    each: float | None = None,
    *,
    effort: str,
) -> list[str]:
    """`rk review-split N`'s output: the parts; then the dispatch plan, with each agent's description and brief
    (agent_call) and the verifiers' EFFORT (the verifier's: config.role_effort); and the cost: the agents, with about
    what they cost when one agent's cost is known (EACH: agent_cost)."""
    parts = res["parts"]
    read = sum(file_lines(f) for f in files if f["cls"] in READ_CLASSES)
    data = sum(x["lines"] for x in res["data"])
    lines = [
        f"SPLIT #{n} ({rnd}): {len(parts)} part{'s' if len(parts) != 1 else ''} of at most {res['max_lines']} "
        f"changed lines; {read} lines to read in {len({p for x in parts for p in x['files']})} files; {data} "
        f"lines in {sum(x['files'] for x in res['data'])} generated, golden or data files summarized; "
        f"{out / DIR}"
    ]
    for p in parts:
        ch = (
            "" if p["changed"] is None else f"  {p['changed']} changed since our review"
        )
        lines.append(
            f"  part{p['k']:<3} {p['lines']:6d} lines {len(p['files']):4d} files ~{p['tokens'] // 1000:4d}K "
            f"tokens  {_label(p)}{ch}"
        )
    lines.append(
        f"  cross: {out / DIR / 'cross.md'} ({len(res['links_between_parts'])} links between parts)"
    )
    if len(parts) <= 1:
        return lines + [
            f"dispatch: one part, so no split: review #{n} as usual (one verifier and its lane, "
            "review_procedure.md)"
        ]

    def call(*k: int) -> str:
        desc, brief = agent_call(n, rnd, out, *k)
        return f'description "{desc}", brief {brief}'

    lines.append(
        f"dispatch (at most {cap} agents at once; each wave starts as the one before hands back), each agent with the "
        "description and the brief given here, which name the PR and the round:"
    )
    for i, w in enumerate(res["waves"], 1):
        lines.append(
            f"  wave {i} ({len(w)} agent{'s' if len(w) != 1 else ''}): the verifiers of {_span(w)}"
        )
        lines += [f"    part{k}: {call(k, len(parts))}" for k in w]
    skip = [p["k"] for p in parts if p["changed"] == 0]
    if skip:
        lines.append(
            f"  no verifier for {_span(skip)}: nothing there changed since our last review (the cross pass "
            "carries their prior rows)"
        )
    lines += [
        f"  part verifiers: rocke-review-verifier, effort {effort}; each runs its part's checks and code quality "
        "checks and writes partK_notes.md and partK_findings.md",
        f"  then the cross pass, once every part has handed back: rocke-review-verifier, effort {effort}, "
        f"{call()}; it reads every part's findings and writes pr{n}_{rnd}.md and notes.md: the single post",
    ]
    n_ver, n_waves = sum(len(w) for w in res["waves"]), len(res["waves"])
    lines.append(
        f"cost: {cost_words(agents(n_ver), each)}: {n_ver} part verifiers in {n_waves} "
        f"wave{'s' if n_waves != 1 else ''}, then the cross pass"
        + (
            "; each agent about one review's cost (the spend card's mean review)"
            if each
            else ""
        )
    )
    return lines


def run(
    cfg: dict,
    st: dict,
    n: int,
    out: pathlib.Path,
    rnd: str,
    max_lines: int,
    cap: int = MAX_AGENTS,
    team_file: str | None = None,
    head: str | None = None,
    store=None,
) -> tuple[list[str], int]:
    """`rk review-split N`: the parts and the dispatch plan; (lines, exit code). STORE: the state's, for the cost of
    an agent (agent_cost)."""
    repo = cfg["paths"]["repo"]
    rec = st["prs"].get(str(n))
    try:
        head = head or gitops.fetch_pr(repo, n)
        ref = base_of(cfg, rec, fetch=True)
        mb = gitops.git(repo, "merge-base", head, ref).strip()
        text = gitops.git(
            repo,
            "-c",
            "diff.renameLimit=20000",
            "diff",
            "-M",
            "--full-index",
            "--no-color",
            "--no-ext-diff",
            mb,
            head,
            timeout=1800,
        )
    except gitops.GitError as exc:
        return [f"SPLIT #{n}: the diff was not read: {exc}"], 1
    files = size.parse_patch(text)
    for f in files:
        f["cls"] = reclass(f)
    base = "develop" if ref == DEVELOP else ref.removeprefix("refs/remotes/origin/")
    note = []
    if rec and rec.get("head") and rec["head"] != head:
        note.append(
            f"!! the live head {head[:10]} is not the state's {rec['head'][:10]}: the tick has not seen it"
        )
    read = [f for f in files if f["cls"] in READ_CLASSES]
    plan = partition(files, paths.repository_folder(cfg), max_lines)
    data = data_groups(files, read, paths.repository_folder(cfg))
    team = team_text(cfg, team_file)
    quality = quality_text(team[1])
    res = write(
        cfg,
        n,
        out,
        head,
        mb,
        base,
        rnd,
        files,
        plan,
        data,
        team,
        quality,
        changed_since(out),
        max_lines,
        cap,
    )
    effort = config.role_effort(cfg, "verify")[0] or config.MODEL_DEFAULT
    lines = (
        summary(n, rnd, out, files, res, cap, agent_cost(cfg, store), effort=effort)
        + note
    )
    if not team[0]:
        lines.append(
            "!! the verifier team file was not read (paths.verifier_team_file or --team-file): the parts "
            "name its checklist leads only, and the code quality checks' leads"
        )
    elif not quality[0]:
        lines.append(
            f"!! the code quality reviewer's team file was not read ({quality[1]}, beside the verifier team "
            "file): the parts name its checks' leads only"
        )
    return lines, 0


def default_out(
    cfg: dict, st: dict, n: int, rnd: str | None
) -> tuple[pathlib.Path, str]:
    rnd = rnd or prep.default_round(st, n, cfg)
    return prep.review_dir(cfg, n, rnd), rnd
