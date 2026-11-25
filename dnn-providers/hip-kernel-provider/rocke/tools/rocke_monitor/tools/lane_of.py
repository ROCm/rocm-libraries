#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Where a test runs in CI, answered from the tree: `rk lane-of <test file | node id | CTest entry>`.

CI runs rocKE's tests as CTest entries of the installed tree, selected by tier label (`ctest -L '^<tier>$'`, the lanes
of config `ci_test_tiers`, --ci-tiers; quick ⊂ standard ⊂ comprehensive ⊂ full), in an environment without the
modules of config `ci_absent_modules`, --absent (such as torch and the built engines). So a test runs in CI only when
all of these hold, and this tool reads each from the tree at one commit:
1. a CTest entry collects it: the tree configured as hip-kernel-provider configures rocKE (tools/ctest_labels.py's
   shim; the build and the installed tree, the in-tree runner's suites too);
2. a lane selects that entry: one of its tier labels is a lane's tier, and none of its labels is one the lane leaves
   out (`-LE`). An entry whose only tiers are comprehensive or full runs in no CI job when no lane selects them;
3. the entry's own `-m` marker expression keeps it: a module marked `pytestmark = pytest.mark.integration` is
   deselected by `-m "not integration"`;
4. nothing skips it first: a module-level `pytest.importorskip` of a module CI lacks skips the whole file there, a
   module-level import of one is a collection error, a `skip` mark skips it.
With --collect, the file is also collected as CI would: pytest --collect-only with the absent modules made unimportable
(pytest_plugins/rkmon_ci.py), each test's skip marks evaluated; none left to run reads as a skip.

The verdict, one line per target: `<target>: CI: runs (<entry>, <tier>)`, `CI: skips(<reason>)`, `CI: no lane(<why>)`
or `CI: errors(<why>)`; then the entries and the reasons, indented; with --log FILE (a CI shard log), the log's lines
that name the entry. NOT RUN when the tree cannot be read or configured, and for a target a CTest entry collects
while no lane is given (which lane runs it is then unknown). The entries of a commit are cached in
--cache (one JSON a commit, <sha>.json); with --cache-out, the listing made here goes there instead and --cache is
only read (`rk lane-of` runs it in a PR runner's slot: the monitor keeps the listing, rkmon_lib/lanes.py).

Usage: python3 -I lane_of.py --repo R [--rev REV] [--rocke-path K] [--python PY] [--ci-tiers JSON]
       [--absent M,M] [--cache DIR] [--cache-out DIR] [--collect] [--log FILE] [--json] TARGET...
Exit 0 when every target runs in CI, 1 when one does not, 3 NOT RUN.
       python3 -I lane_of.py --repo R [--rev REV] [--rocke-path K] --fetch TARGET...
Fetches the blobs of the clone the tool reads, and runs nothing else (rkmon_lib/prtree.py, fetch_main).
"""

import argparse
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True  # the toolkit's folder is read-only at run time
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from rkmon_lib import gitops, paths, prtree, whole_file  # noqa: E402
import ctest_labels as cl  # noqa: E402

PLUGINS = ROOT / "tools" / "pytest_plugins"
NO_LANE_KNOWN = (
    "config ci_test_tiers names no PR-CI lane: which lane runs it is unknown"
)
BUDGET_S = 300
COLLECT_S = 240
CACHE_VERSION = 1
LOG_SHOWN = 4


class Target:
    """What the caller named: a test file (relative to the rocKE tree), a node id (file::rest), or a CTest entry."""

    def __init__(self, text: str, K: str):
        self.text = text
        path, sep, rest = text.partition("::")
        self.test = rest if sep else ""
        p = path.strip().lstrip("./")
        if p.startswith(K.rstrip("/") + "/"):
            p = p[len(K.rstrip("/")) + 1 :]
        if sep or p.endswith(".py") or "/" in p:
            self.kind, self.file, self.entry = ("node" if sep else "file"), p, ""
        else:
            self.kind, self.file, self.entry = "entry", "", text.strip()


# --- the tree's CTest entries -----------------------------------------------------------------------------------------


def configure_tree(
    repo: str, rev: str, K: str, python: str, work: pathlib.Path, budget: float
) -> dict:
    """The commit's tree written under work/tree and configured as the provider configures rocKE: its entries in the
    build and the installed tree, the in-tree runner's suites, and whether the YAML tiers applied. NotRun when it
    cannot."""
    t0 = time.monotonic()
    shim = work / "shim"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "CMakeLists.txt").write_text(cl.SHIM)
    provider = K.rsplit("/", 1)[0]
    root = work / "tree"
    want = cl.tree_paths(repo, rev, K)
    if K not in want:
        raise prtree.NotRun(f"{rev[:10]} has no {K}")
    prtree.extract(repo, rev, want, root, timeout=max(30.0, budget))
    log = work / "configure.log"
    left = max(10.0, budget - (time.monotonic() - t0))
    err = cl.configure(shim, work / "build", root, provider, python, left, log)
    text = log.read_text(errors="replace")
    if err:
        raise prtree.NotRun(f"the tree does not configure as the provider does: {err}")
    failed = cl.YAML_FAILED.search(text)
    if failed:
        raise prtree.NotRun(
            f"the YAML tiers could not be applied ({failed.group().strip()}): the tiers are unknown"
        )
    got = {"yaml": cl.YAML_ON in text, "version": CACHE_VERSION}
    for tree, where in (
        ("build", work / "build"),
        ("installed", work / "build" / cl.INSTALLED_DIR),
    ):
        left = max(10.0, budget - (time.monotonic() - t0))
        got[tree] = cl.listing(where, root, K, min(left, 120.0))
    suites = cl.read_suites(root, K)
    got["suites"] = (
        {k: sorted(v) for k, v in suites.items()} if suites is not None else None
    )
    return got


def cached_listing(cache: pathlib.Path | None, sha: str) -> dict | None:
    if not cache:
        return None
    try:
        got = json.loads((cache / f"{sha}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if got.get("version") == CACHE_VERSION else None


def keep_listing(cache: pathlib.Path | None, sha: str, got: dict) -> None:
    """The listing GOT of the commit SHA kept in the folder CACHE as the file SHA.json, written whole (whole_file), a
    tree's file (worker/perms.py TREE_FILE_MODE: the monitor's, which a runner reads; in a runner's slot, the runner's,
    which the monitor reads)."""
    if not cache:
        return
    try:
        cache.mkdir(parents=True, exist_ok=True)
        whole_file.write(
            cache / f"{sha}.json",
            json.dumps(got),
            paths.worker_module("perms").TREE_FILE_MODE,
        )
    except OSError:
        pass  # a cache that cannot be written only costs the next call a configure


# --- what the test file says about itself -----------------------------------------------------------------------------


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner else ""
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _mark(node) -> tuple[str, str] | None:
    """A `pytest.mark.<name>` (called or not) as (name, its first argument's source); None for anything else."""
    name = _dotted(node)
    if not name.startswith("pytest.mark.") and not name.startswith("mark."):
        return None
    mark = name.rsplit(".", 1)[1]
    arg = ""
    if isinstance(node, ast.Call):
        if node.args:
            arg = ast.unparse(node.args[0])
        for kw in node.keywords:
            if kw.arg == "reason" and not arg:
                arg = ast.unparse(kw.value)
    return mark, arg


def _marks(value) -> list[tuple[str, str]]:
    items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
    return [m for m in (_mark(x) for x in items) if m]


def _importorskip(node) -> str | None:
    """The module a `pytest.importorskip("m")` call names; None for anything else."""
    if (
        isinstance(node, ast.Call)
        and _dotted(node.func) in ("pytest.importorskip", "importorskip")
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return node.args[0].value
    return None


def _calls(stmt) -> list:
    """The calls a statement makes at its own level (an expression or an assignment's value)."""
    value = getattr(stmt, "value", None)
    return [value] if isinstance(value, ast.Call) else []


def _module_skip(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and _dotted(node.func) in ("pytest.skip", "skip")
        and any(
            kw.arg == "allow_module_level"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value
            for kw in node.keywords
        )
    )


def _imports(stmt) -> list[str]:
    if isinstance(stmt, ast.Import):
        return [a.name for a in stmt.names]
    if isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
        return [stmt.module]
    return []


def file_facts(source: str, test: str = "") -> dict:
    """What a test file's own text says about where it runs: the module's marks, its module-level importorskips,
    unconditional module-level imports and skips; for a node id (`Class::test` or `test`, parameters cut), the marks
    and importorskips of its class and function. Unparsable text gives {"error": ...}.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {"error": f"the file does not parse ({e.msg}, line {e.lineno})"}
    out: dict = {
        "marks": [],
        "importorskip": [],
        "imports": [],
        "skip": False,
        "test_marks": [],
        "test_importorskip": [],
        "test_found": None,
    }
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets
        ):
            out["marks"] += _marks(stmt.value)
        for call in _calls(stmt):
            mod = _importorskip(call)
            if mod:
                out["importorskip"].append(mod)
            if _module_skip(call):
                out["skip"] = True
        out["imports"] += _imports(stmt)
    if test:
        parts = [re.sub(r"\[.*$", "", p) for p in test.split("::") if p]
        scope, found, marks, skips = tree.body, True, [], []
        for i, name in enumerate(parts):
            node = next(
                (
                    n
                    for n in scope
                    if isinstance(
                        n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    and n.name == name
                ),
                None,
            )
            if node is None:
                found = False
                break
            for d in node.decorator_list:
                m = _mark(d)
                if m:
                    marks.append(m)
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and any(
                        isinstance(t, ast.Name) and t.id == "pytestmark"
                        for t in stmt.targets
                    ):
                        marks += _marks(stmt.value)
                scope = node.body
            else:
                for stmt in node.body:
                    for call in _calls(stmt):
                        mod = _importorskip(call)
                        if mod:
                            skips.append(mod)
        out["test_found"] = found
        out["test_marks"], out["test_importorskip"] = marks, skips
    return out


# --- markers ----------------------------------------------------------------------------------------------------------


def marker_expr(cmd: list[str]) -> str:
    """The `-m` expression of an entry's pytest command ("" when it has none)."""
    args = cmd[3:] if len(cmd) > 2 and cmd[1:3] == ["-m", "pytest"] else cmd
    for i, a in enumerate(args):
        if a == "-m" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("-m") and len(a) > 2 and not a.startswith("-m="):
            return a[2:].strip()
        if a.startswith("-m="):
            return a[3:]
    return ""


MARK_TOKEN = re.compile(r"\s*(\(|\)|and\b|or\b|not\b|[A-Za-z_][\w]*)")


def selected(expr: str, marks: set[str]) -> bool | None:
    """Whether pytest's `-m EXPR` keeps a test with these marks; None when the expression is beyond `and`, `or`,
    `not`, parentheses and mark names."""
    if not expr.strip():
        return True
    out, pos = [], 0
    while pos < len(expr):
        m = MARK_TOKEN.match(expr, pos)
        if not m:
            if expr[pos:].strip():
                return None
            break
        tok = m.group(1)
        if tok in ("(", ")", "and", "or", "not"):
            out.append(tok)
        else:
            out.append("True" if tok in marks else "False")
        pos = m.end()
    try:
        return bool(
            eval(" ".join(out), {"__builtins__": {}}, {})
        )  # noqa: S307 (only True/False and operators)
    except SyntaxError:
        return None


# --- the verdict ------------------------------------------------------------------------------------------------------


def lanes_for(entry: dict, tree: str, lanes: list[dict]) -> list[dict]:
    return [
        ln
        for ln in lanes
        if ln.get("tree", "installed") == tree
        and ln["tier"] in entry["labels"]
        and not set(ln.get("exclude") or ()) & set(entry["labels"])
    ]


def entries_of(target: Target, listing: dict) -> list[dict]:
    """The entries that run the target: {tree, name, entry, how}."""
    suites = (
        {k: set(v) for k, v in listing["suites"].items()}
        if listing.get("suites")
        else None
    )
    out = []
    for tree in cl.TREES:
        for name, e in sorted((listing.get(tree) or {}).items()):
            if target.kind == "entry":
                how = "named" if name == target.entry else None
            else:
                how = cl.runs_file(e, tree, target.file, suites)
            if how:
                out.append({"tree": tree, "name": name, "entry": e, "how": how})
    return out


def _why_no_lane(x: dict, lanes: list[dict]) -> str:
    e, tree = x["entry"], x["tree"]
    tiers = cl.tiers(e["labels"])
    lane_trees = {ln.get("tree", "installed") for ln in lanes}
    if tree not in lane_trees:
        return f"{x['name']} is a {tree}-tree entry; CI runs the {' and '.join(sorted(lane_trees))} tree"
    out = [ln for ln in lanes if set(ln.get("exclude") or ()) & set(e["labels"])]
    if out and any(ln["tier"] in e["labels"] for ln in out):
        lab = sorted(set(out[0].get("exclude") or ()) & set(e["labels"]))
        return f"{x['name']} has the label {' '.join(lab)}, which the lanes leave out (-LE)"
    if not tiers:
        return f"{x['name']} has no tier label"
    return (
        f"{x['name']} is tier {' '.join(tiers)} only, and no CI job selects "
        f"{'that tier' if len(tiers) == 1 else 'those tiers'} (lanes: "
        + ", ".join(sorted({ln["tier"] for ln in lanes}))
        + ")"
    )


def verdict(
    target: Target,
    listing: dict,
    lanes: list[dict],
    absent: set[str],
    source: str | None,
) -> dict:
    """{"verdict": runs|skips|no lane|errors|not run, "why": text, "entry", "tier", "detail": [lines]} for one target:
    not run when a CTest entry collects it and LANES names none."""
    found = entries_of(target, listing)
    detail = []
    for x in found:
        e = x["entry"]
        run_by = lanes_for(e, x["tree"], lanes)
        expr = marker_expr(e.get("cmd") or [])
        detail.append(
            f"entry {x['name']} ({x['tree']} tree"
            + (", through the in-tree runner" if x["how"] == "in-tree" else "")
            + f"): tiers {' '.join(cl.tiers(e['labels'])) or 'none'}; labels {' '.join(e['labels']) or 'none'}; "
            + (f'-m "{expr}"; ' if expr else "")
            + "lanes "
            + (", ".join(f"{ln['jobs']} (-L {ln['tier']})" for ln in run_by) or "none")
        )
    res = {"target": target.text, "detail": detail, "entry": "", "tier": ""}
    if not found:
        what = (
            "no CTest entry has that name"
            if target.kind == "entry"
            else "in no CTest entry: no CTest list collects it, or the install leaves it out"
        )
        return dict(res, verdict="no lane", why=what)
    if not lanes:
        return dict(res, verdict="not run", why=NO_LANE_KNOWN)
    laned = [(x, ln) for x in found for ln in lanes_for(x["entry"], x["tree"], lanes)]
    if not laned:
        return dict(
            res,
            verdict="no lane",
            why="; ".join(_why_no_lane(x, lanes) for x in found[:3]),
        )
    facts = file_facts(source, target.test) if source is not None else {}
    if facts.get("error"):
        res["detail"].append(facts["error"])
        facts = {}
    if target.kind == "node" and facts.get("test_found") is False:
        res["detail"].append(
            f"{target.test} is not defined in the file at this commit (a parametrized, generated or inherited test "
            "is read as its file)"
        )
    marks = {m for m, _ in facts.get("marks") or []} | {
        m for m, _ in facts.get("test_marks") or []
    }
    kept = []
    for x, ln in laned:
        expr = marker_expr(x["entry"].get("cmd") or [])
        keep = selected(expr, marks)
        if keep is None:
            res["detail"].append(
                f'-m "{expr}" of {x["name"]} is not read here; taken as selecting the test'
            )
            keep = True
        if keep:
            kept.append((x, ln))
    if not kept:
        x, _ln = laned[0]
        expr = marker_expr(x["entry"].get("cmd") or [])
        return dict(
            res,
            verdict="no lane",
            why=f'deselected by {x["name"]}\'s -m "{expr}" (marked {" ".join(sorted(marks))})',
        )
    x, ln = kept[0]
    res.update(entry=x["name"], tier=ln["tier"])
    mods = [m for m in facts.get("importorskip") or [] if m.split(".")[0] in absent]
    if mods:
        return dict(
            res,
            verdict="skips",
            why=f'module-level importorskip("{mods[0]}"); CI has no {mods[0].split(".")[0]}',
        )
    hard = [m for m in facts.get("imports") or [] if m.split(".")[0] in absent]
    if hard:
        return dict(
            res,
            verdict="errors",
            why=f"imports {hard[0]} at module level; CI has no {hard[0].split('.')[0]}: a collection error",
        )
    if facts.get("skip"):
        return dict(res, verdict="skips", why="an unconditional module-level skip")
    for m, arg in (facts.get("marks") or []) + (facts.get("test_marks") or []):
        if m == "skip":
            return dict(
                res, verdict="skips", why=f"a skip mark{(': ' + arg) if arg else ''}"
            )
    mods = [
        m for m in facts.get("test_importorskip") or [] if m.split(".")[0] in absent
    ]
    if mods:
        return dict(
            res,
            verdict="skips",
            why=f'the test calls importorskip("{mods[0]}"); CI has no {mods[0].split(".")[0]}',
        )
    for m, arg in (facts.get("marks") or []) + (facts.get("test_marks") or []):
        if m == "skipif":
            res["detail"].append(
                f"skipif({arg}): decided at run time (--collect evaluates it)"
            )
    return dict(res, verdict="runs", why="")


def line(res: dict) -> str:
    if res["verdict"] == "runs":
        return f"{res['target']}: CI: runs ({res['entry']}, {res['tier']})"
    if res["verdict"] == "not run":
        return f"{res['target']}: NOT RUN: {res['why']}"
    return f"{res['target']}: CI: {res['verdict']}({res['why']})"


# --- the CI-condition collect -----------------------------------------------------------------------------------------


def collect(
    tree_root: pathlib.Path, target: Target, python: str, absent: set[str], timeout
) -> dict:
    """pytest --collect-only of the target in the tree as CI's environment has it: the absent modules unimportable,
    each test's skip marks evaluated (rkmon_ci.py). {"rc", "run": [ids], "skipped": [(id, reason)], "errors": [..]}.
    """
    path = tree_root / target.file
    if not path.is_file():
        return {"error": f"{target.file} is not in the tree"}
    ini = next(
        (d for d in path.parents if (d / "pytest.ini").is_file()),
        tree_root,
    )
    pp = [str(PLUGINS)] + [
        str(tree_root / p)
        for p in ("platform/python", "library/python", "library")
        if (tree_root / p).is_dir()
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
        and not k.startswith("ROCKE_")
    }
    env.update(
        PYTHONPATH=os.pathsep.join(pp),
        RKMON_CI_ABSENT=",".join(sorted(absent)),
        PYTHONDONTWRITEBYTECODE="1",
    )
    arg = os.path.relpath(path, ini) + (f"::{target.test}" if target.test else "")
    cmd = [
        python,
        "-B",
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "rkmon_ci",
        "-p",
        "no:cacheprovider",
        arg,
    ]
    try:
        p = subprocess.run(
            cmd,
            cwd=ini,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"the collect took over {timeout:.0f} s"}
    except OSError as e:
        return {"error": f"pytest did not start ({e})"}
    out = {"rc": p.returncode, "run": [], "skipped": [], "errors": []}
    for ln in (p.stdout + p.stderr).splitlines():
        parts = ln.split("\t")
        if parts[0] == "rkmon-ci:" and len(parts) >= 3:
            if parts[1] == "RUN":
                out["run"].append(parts[2])
            elif parts[1] == "SKIP":
                out["skipped"].append((parts[2], parts[3] if len(parts) > 3 else ""))
        elif ln.startswith("ERROR ") or "ERROR collecting" in ln:
            out["errors"].append(ln.strip()[:200])
        elif re.match(r"^E\s+(ModuleNotFoundError|ImportError)", ln):
            out["errors"].append(ln.strip()[:200])
    return out


def apply_collect(res: dict, got: dict) -> dict:
    """The collect's say on a verdict that read as runs: errors and all-skipped override it."""
    if got.get("error"):
        res["detail"].append(f"collect NOT RUN: {got['error']}")
        return res
    n_run, n_skip = len(got["run"]), len(got["skipped"])
    res["detail"].append(
        f"collect as CI (absent modules unimportable): {n_run} would run, {n_skip} skip"
        + (f", {len(got['errors'])} errors" if got["errors"] else "")
        + f" (pytest rc {got['rc']})"
    )
    for tid, why in got["skipped"][:3]:
        res["detail"].append(f"   skip {tid}: {why}")
    if res["verdict"] != "runs":
        return res
    if got["errors"] or (got["rc"] not in (0, 5) and not n_run):
        return dict(
            res,
            verdict="errors",
            why=f"collecting it as CI does fails: {(got['errors'] or ['pytest rc ' + str(got['rc'])])[0]}",
        )
    if not n_run:
        why = got["skipped"][0][1] if got["skipped"] else "collects no test"
        return dict(res, verdict="skips", why=f"no test left to run as CI: {why}")
    return res


def needs(repo: str, rev: str, K: str) -> list[str]:
    """The blobs of the clone the tool reads (`--fetch`): the tree it configures at `rev`, which holds the targets'
    files."""
    sha = gitops.git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
    return prtree.blobs(repo, sha, cl.tree_paths(repo, sha, K))


def resolve_file(repo: str, sha: str, K: str, f: str, names: list[str]) -> str | None:
    """A file named without its folder (`test_x.py`) as the tree's only file of that name; `f` itself when the tree has
    it or no single file matches. `names` caches the tree's file list across targets."""
    if "/" in f:
        return f
    if not names:
        names += gitops.git(
            repo, "ls-tree", "-r", "--name-only", sha, "--", K + "/", lazy=False
        ).splitlines()
    hits = [n[len(K) + 1 :] for n in names if n.rsplit("/", 1)[-1] == f]
    return hits[0] if len(hits) == 1 else f


# --- the shard log ----------------------------------------------------------------------------------------------------


def log_lines(text: str, entry: str) -> list[str]:
    """The lines of a CI log that name a CTest entry (its Start, its Test #N result)."""
    pat = re.compile(rf"(?<![\w-]){re.escape(entry)}(?![\w-])")
    return [
        cl_line.strip()[:200]
        for cl_line in text.splitlines()
        if pat.search(cl_line)
        and re.search(r"Test\s+#|Start\s+\d|Passed|Failed|Not Run|\*\*\*", cl_line)
    ][-LOG_SHOWN:]


# --- the tool ---------------------------------------------------------------------------------------------------------


def run(
    repo: str,
    targets: list[str],
    rev: str,
    K: str,
    python: str | None = None,
    lanes: list[dict] | None = None,
    absent: set[str] | None = None,
    cache: pathlib.Path | None = None,
    do_collect: bool = False,
    collect_python: str | None = None,
    log_text: str | None = None,
    budget: float = BUDGET_S,
    cache_out: pathlib.Path | None = None,
) -> dict:
    """The verdicts for `targets` at `rev`: the commit's listing from `cache`, else made here and kept in `cache_out`
    (default `cache`)."""
    lanes = lanes or []
    absent = set(absent or ())
    out: dict = {"rev": rev, "results": [], "error": None}
    work = None
    try:
        sha = gitops.git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()
        out["sha"] = sha
        listing = cached_listing(cache, sha)
        if listing is None or do_collect:
            if not (shutil.which("cmake") and shutil.which("ctest")):
                raise prtree.NotRun("no cmake or ctest on PATH")
            work = pathlib.Path(
                tempfile.mkdtemp(
                    prefix="lane_of_", dir=os.environ.get("TMPDIR") or None
                )
            ).resolve()
            if listing is None:
                listing = configure_tree(
                    repo, sha, K, cl.yaml_python(python), work, budget
                )
                keep_listing(cache_out or cache, sha, listing)
            else:
                prtree.extract(repo, sha, [K], work / "tree", timeout=max(30.0, budget))
        out["yaml"] = listing.get("yaml")
        names: list[str] = (
            []
        )  # the tree's files, read once when a bare file name needs them
        for text in targets:
            t = Target(text, K)
            source, said = None, []
            if t.file:
                found = resolve_file(repo, sha, K, t.file, names)
                if found and found != t.file:
                    said.append(
                        f"{t.file} read as {found}, the tree's only file of that name"
                    )
                    t.file = found
                try:
                    source = gitops.git(repo, "show", f"{sha}:{K}/{t.file}")
                except gitops.GitError:
                    source = None
            if t.file and source is None:
                out["results"].append(
                    {
                        "target": text,
                        "verdict": "not run",
                        "why": f"{t.file} is not in {K} at {sha[:10]}",
                        "detail": said,
                        "entry": "",
                        "tier": "",
                    }
                )
                continue
            res = verdict(t, listing, lanes, absent, source)
            res["detail"][:0] = said
            if do_collect and t.file and source is not None and work:
                res = apply_collect(
                    res,
                    collect(
                        work / "tree" / K,
                        t,
                        collect_python or sys.executable,
                        absent,
                        COLLECT_S,
                    ),
                )
            if log_text is not None and res.get("entry"):
                res["detail"] += [
                    f"log: {x}" for x in log_lines(log_text, res["entry"])
                ] or [f"log: no line names {res['entry']}"]
            out["results"].append(res)
    except (
        prtree.NotRun,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        gitops.GitError,
    ) as exc:
        out["error"] = str(exc)[:300]
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)
    return out


def render(out: dict) -> list[str]:
    if out["error"]:
        return [f"NOT RUN: {out['error']}"]
    lines = [
        f"tree {out['sha'][:10]} ({out['rev']}), configured as hip-kernel-provider configures rocKE"
        + (
            ""
            if out.get("yaml")
            else "; the YAML tiers did not apply: no entry has a tier"
        )
    ]
    for res in out["results"]:
        lines.append(line(res))
        lines += [f"   {d}" for d in res["detail"]]
    return lines


def code(out: dict) -> int:
    if out["error"] or any(r["verdict"] == "not run" for r in out["results"]):
        return 3
    return 0 if all(r["verdict"] == "runs" for r in out["results"]) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="+", metavar="TARGET")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", default=prtree.DEVELOP)
    ap.add_argument("--rocke-path", default="", type=prtree.folder_argument)
    ap.add_argument("--python", default=None, help="an interpreter with PyYAML")
    ap.add_argument(
        "--collect-python",
        default=None,
        help="the interpreter whose pytest collects (default: this one)",
    )
    ap.add_argument("--ci-tiers", default="", help="config ci_test_tiers as JSON")
    ap.add_argument(
        "--absent", default=None, help="config ci_absent_modules, comma-separated"
    )
    ap.add_argument("--cache", default=None)
    ap.add_argument(
        "--cache-out",
        default=None,
        help="where a listing made here goes (default --cache)",
    )
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--log", default=None, help="a CI shard log to read")
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fetch", action="store_true", help=prtree.FETCH_HELP)
    a = ap.parse_args(argv)
    if a.fetch:
        return prtree.fetch_main(a.repo, lambda: needs(a.repo, a.rev, a.rocke_path))
    lanes = json.loads(a.ci_tiers) if a.ci_tiers.strip() else None
    absent = (
        {m.strip() for m in a.absent.split(",") if m.strip()}
        if a.absent is not None
        else None
    )
    log_text = (
        pathlib.Path(a.log).read_text(encoding="utf-8", errors="replace")
        if a.log
        else None
    )
    out = run(
        a.repo,
        a.targets,
        a.rev,
        a.rocke_path,
        a.python,
        lanes,
        absent,
        pathlib.Path(a.cache) if a.cache else None,
        a.collect,
        a.collect_python,
        log_text,
        a.budget,
        pathlib.Path(a.cache_out) if a.cache_out else None,
    )
    if a.json:
        print(json.dumps(out, indent=1))
    else:
        print("\n".join(render(out)))
    return code(out)


if __name__ == "__main__":
    sys.exit(main())
