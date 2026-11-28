# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Changed tests in the installed tree, as CI's rocKE shards run them (lessons L-526, L-142).

CI runs the INSTALLED tree: `cmake --install`, then each installed CTest entry from its CTest directory: `rocke_pytest`
collects the installed platform tests, the `rocke_library_*_pytest` entries the library test lists of
rocke/platform/CMakeLists.txt (`_ROCKE_LIBRARY_ATTENTION_CPU_TEST_RELPATHS` and the like). A test that reaches a
source-only path passes in the source layout and fails there (a test that reads provider headers through
`Path(__file__).parents[3]`, for example), so checks that run tests in the source layout only pass it.

Each step is bounded, and none passes silently (lessons L-107, L-506):
1. `cmake --install` the checks' own native build (ROCKE_INSTALL_TESTS and ROCKE_INSTALL_PYTHON_TESTS default on)
   into a scratch prefix. An install that fails is NOT RUN and fails the section: CI's install fails the same way.
2. `ctest --show-only=json-v1` in the installed CTest directory: the merged tree's test lists exactly as its CMake
   wrote them, with each entry's command, working directory, ENVIRONMENT, labels and timeout.
3. Each test file the PR adds or changes is mapped to its installed copy (the install manifest: same name and bytes)
   and to the entries that collect it (a listed file, or a file under a listed directory that no --ignore drops), or
   that run it through the in-tree lane: an entry that collects the in-tree runner (tests/in_tree/
   test_in_tree_suites.py; `rocke_pytest` collects it) runs each test_*.py file of the packages the runner's `_SUITES`
   names, but the `_EXCLUDED_FILES` names, in a child pytest per file. The suites are read from the installed runner,
   so the lane follows develop's list (`rocke_pytest` runs `rocke.benchmark.perf.tests` that way).
4. The mapped files run under each entry's own pytest options, from its working directory, with its ENVIRONMENT. The
   install's site-packages joins PYTHONPATH: CI gets `rocke` from the platform wheel, and a plain `ctest` here fails
   every file with "No module named 'rocke'", because CTest's ENVIRONMENT replaces PYTHONPATH (L-142).
   The checks' own ROCKE_* exports (the forced LLVM flavor, the source build's native-comparison binaries) are dropped:
   CI sets none of them. A file of the in-tree lane runs as CI runs it: the runner itself, selecting the runner's case
   for each file (-k "[PACKAGE:PATH]", the package and the file's path in it), so its child pytest runs with the
   runner's own options; a file whose case the run does not report is NOT RUN, never passed. Over CAP changed test
   files, the entries that collect or run them run whole instead; GPU entries are named, not run whole (no
   gfx942/gfx950 here).
5. Every skipped test of a changed file is listed with its reason (pytest_plugins/rkmon_skips.py): a pass with skips
   reads "pass (k skipped: ...)", and a skip for a tool CI's lanes have and this run lacks (cmake, PyYAML) reads
   "skip NOT RUN: TOOL missing", never a pass. The report gives the exact command that re-runs each changed file
   under its entry, with the entry's environment in a file beside the logs: through the package's `rk pr-run`
   (RKMON_RERUN_VIA, which the job scripts set), in a slot of its own, that file named where the job copies the
   results (RKMON_RESULTS).
6. Every failure is compared with develop's installed run at the merge's develop SHA: its outcome in the cache of
   develop's outcomes at that SHA (`--cache`). The run reads the cache and never writes it: it runs PR code, and the
   develop-side caches are the monitor's. The failures the cache lacks are not compared, and
   are listed in OUT/develop_todo.tsv: a develop job of their own runs them (`develop`, with develop built and
   installed in that job's own build folder), the monitor adds that job's finished outcomes to the cache
   (`cache-add`), and the run is made again with that job's outcomes (`--outcomes`). A test file the PR adds, or a
   test develop lacks, is the PR's; a test develop fails too is labelled so; a test develop's installed lanes do not
   run is the PR's (its enrolment exposes it). Develop is built and installed only when a failure needs it.

The layout is CMake's standalone one (CTest directory bin/rocke, `rocke` in lib/pythonX.Y/site-packages); CI's provider
layout puts both under bin/hip_kernel_provider. The test files sit under <test root>/tests in both, so a test that
reaches the source tree fails in both.

Usage: python -I installed_tests.py --out DIR --tree ROCKE_TREE --build CORE_BUILD --prefix PREFIX [options] FILE...
FILEs are relative to the rocKE tree (library/tests/test_x.py). Prints the report, then one last line:
"INSTALLED pass|fail|skip DETAIL". RKMON_PYEXTRA (optional) joins the end of each run's PYTHONPATH (the checks'
PyYAML: rkmon_lib/review.py, tool_env_lines).
Usage: python -I installed_tests.py develop --out DIR --todo TODO --dev-tree TREE --dev-sha SHA --dev-build CMD
       --dev-core DIR --dev-prefix DIR [options]
A develop job: develop's tree built (CMD) and installed, and the ids of TODO (a run's develop_todo.tsv: an entry and
an id a line, tab-separated) run under develop's entries; their outcomes into DIR/develop_outcomes.tsv (an entry, an
id and an outcome a line, tab-separated), then one last line "INSTALLED_DEVELOP N of M outcomes ...".
Usage: python -I installed_tests.py cache-add OUTCOMES CACHE
The monitor's step: the well-formed lines of OUTCOMES (a develop job's develop_outcomes.tsv, read as data) for ids
CACHE lacks, added to it when their run finished. Exit 0 when one was added, 1 when none was.
"""
import argparse
import ast
import filecmp
import fnmatch
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

PLUGINS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytest_plugins")
sys.path.insert(0, PLUGINS)
import rkmon_skips  # noqa: E402  (pytest_plugins/: the skip lines rkmon_env prints, and their reports)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rkmon_lib import nogpu, whole_file  # noqa: E402

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "worker"),
)
import perms  # noqa: E402  (the cache's file mode: a tree's file, as the monitor's)
import pr_env  # noqa: E402  (what `rk pr-run --env` gives PR code: the re-run commands' environment files)

CAP = 40  # review.TESTS_CAP: over it, the entries that collect the changed files run whole
BATCH = 10  # files per pytest run, as pytest_vs_base.py: a crash costs one batch
# pytest options that take the next argument as their value: pytest's own and the rocKE conftest's.
VALUE_OPTS = {
    "-m",
    "-k",
    "-p",
    "-c",
    "-o",
    "-W",
    "--ignore",
    "--ignore-glob",
    "--deselect",
    "--rootdir",
    "--confcutdir",
    "--basetemp",
    "--junitxml",
    "--junit-xml",
    "--import-mode",
    "--maxfail",
    "--timeout",
    "--timeout-method",
    "--override-ini",
    "--log-level",
    "--log-file",
    "--durations",
    "--rocke-reference-operation",
    "--rocke-reference-arch",
    "--rocke-reference-bundle",
    "--rocke-reference-lock",
    "--gdn-batch",
    "--gdn-spec-id",
    "--rocke-seam-arches",
}
NORECURSE = (
    "*.egg",
    ".*",
    "_darcs",
    "build",
    "CVS",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
)  # pytest's default
# Ours, after the entry's options: no cache in the install, every outcome listed (-rA), a file that fails to import
# does not stop the others, and this host's hipcc limits read as skips (pytest_plugins/rkmon_env.py).
EXTRA = [
    "-p",
    "no:cacheprovider",
    "-rA",
    "--continue-on-collection-errors",
    "-p",
    "rkmon_env",
]
SUMMARY = re.compile(r"^=+ .+ in [0-9.]+s( \([0-9:]+\))? =+$")
COUNT = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b")
PROGRESS = re.compile(r"^(\S.*?::.*?) (PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
PR, DEVELOP, UNRUN, NOCOMPARE = "pr", "develop", "unrun", "nocompare"
# Develop's outcomes for one id (Develop.label): those of finished runs, which the cache keeps, and a whole file whose
# run did not finish. The ids a run could not compare (in its OUT) and a develop job's outcomes for them (in its OUT);
# the most ids a develop job takes and a cache keeps.
CACHED = ("failed", "passed", "unrun", "absent", "nolane")
OUTCOMES = CACHED + ("unfinished",)
TODO_FILE, OUTCOMES_FILE = "develop_todo.tsv", "develop_outcomes.tsv"
TODO_MAX, CACHE_MAX = 2000, 50000
NOT_KNOWN = "develop's outcome is not in its cache"
ENTRY_NAME = re.compile(r"^[A-Za-z0-9_.+-]{1,200}$")
# The in-tree runner (develop's platform/tests/in_tree/test_in_tree_suites.py) and the files it runs in each suite
# package (its rglob pattern); which packages, and which names it leaves out, are read from the installed runner.
IN_TREE_RUNNER = "test_in_tree_suites.py"
IN_TREE_FILES = "test_*.py"


class NotRun(Exception):
    """A step that could not run; the section reads NOT RUN with the reason. `fail`: the merge itself is at fault (its
    install fails, as CI's would)."""

    def __init__(self, reason: str, fail: bool = False):
        super().__init__(reason)
        self.fail = fail


# --- reading the installed CTest entries -----------------------------------------------------------------------------


def split_args(
    args: list[str], cwd: str
) -> tuple[list[str], list[str], list[str], list[str]]:
    """An entry's pytest arguments as (test paths, --ignore paths, --ignore-glob patterns, every other option in order).
    A bare argument is a test path when it names one (an existing path, a .py file, or a path with a slash);
    otherwise it is the value of a custom option (`--rocke-reference-operation conv`).
    """
    paths, ignores, globs, options = [], [], [], []
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("-"):
            opt, eq, val = a.partition("=")
            if not eq and a in VALUE_OPTS and i + 1 < len(args):
                val, eq = args[i + 1], "next"
            if opt == "--ignore" and eq:
                ignores.append(os.path.normpath(os.path.join(cwd, val)))
            elif opt == "--ignore-glob" and eq:
                globs.append(val if os.path.isabs(val) else os.path.join(cwd, val))
            options += [a, val] if eq == "next" else [a]
            i += 2 if eq == "next" else 1
            continue
        base = a.split("::")[0]
        if (
            os.path.exists(os.path.join(cwd, base))
            or base.endswith(".py")
            or "/" in base
        ):
            paths.append(a)
        else:
            options.append(a)
        i += 1
    return paths, ignores, globs, options


def pytest_entries(ctest_json: str, which=shutil.which) -> list[dict]:
    """The installed CTest entries that run pytest, from `ctest --show-only=json-v1`. NotRun when the JSON cannot be
    read, or when ctest could not resolve the pytest entries' `python` (it then leaves their command out, and the
    mapping would wrongly call every file laneless). A C++ entry whose binary is missing has no command either; it is
    not a pytest entry."""
    try:
        tests = json.loads(ctest_json).get("tests") or []
    except (ValueError, AttributeError) as e:
        raise NotRun(f"ctest --show-only=json-v1 gave no JSON ({e})") from e
    blind = [t.get("name", "?") for t in tests if not t.get("command")]
    if blind and not which("python"):
        raise NotRun(
            f"ctest resolved no command for {len(blind)} entries (no `python` on PATH): "
            + " ".join(blind[:4])
        )
    tests = [t for t in tests if t.get("command")]
    out = []
    for t in tests:
        cmd = t["command"]
        if len(cmd) < 3 or cmd[1:3] != ["-m", "pytest"]:
            continue
        props = {p.get("name"): p.get("value") for p in t.get("properties") or []}
        env = {}
        for kv in props.get("ENVIRONMENT") or []:
            k, _, v = kv.partition("=")
            env[k] = v
        cwd = props.get("WORKING_DIRECTORY") or ""
        paths, ignores, globs, options = split_args(cmd[3:], cwd)
        missing = [
            p for p in paths if not os.path.exists(os.path.join(cwd, p.split("::")[0]))
        ]
        out.append(
            {
                "name": t["name"],
                "python": cmd[0],
                "args": cmd[3:],
                "cwd": cwd,
                "env": env,
                "labels": {
                    x for lab in props.get("LABELS") or [] for x in lab.split(";") if x
                },
                "timeout": float(props.get("TIMEOUT") or 0) or None,
                "paths": paths,
                "ignores": ignores,
                "globs": globs,
                "options": options,
                "missing": missing,
            }
        )
    return out


def collects(entry: dict, path: str) -> bool:
    """Whether the entry's pytest run collects the installed file `path`: a listed file, or a file under a listed
    directory, through no directory pytest skips by default, and under no --ignore path or --ignore-glob.
    """
    path = os.path.normpath(path)

    def under(p: str) -> bool:
        return path == p or path.startswith(p.rstrip(os.sep) + os.sep)

    if any(under(i) for i in entry["ignores"]) or any(
        fnmatch.fnmatch(path, g) for g in entry["globs"]
    ):
        return False
    for a in entry["paths"]:
        p = os.path.normpath(os.path.join(entry["cwd"], a.split("::")[0]))
        if path == p:
            return True
        if under(p):
            rel = os.path.relpath(os.path.dirname(path), p)
            parts = [] if rel == "." else rel.split(os.sep)
            if not any(fnmatch.fnmatch(x, pat) for x in parts for pat in NORECURSE):
                return True
    return False


def in_tree_suites(runner: str) -> tuple[list[str], set[str]]:
    """The packages an installed in-tree runner runs (the keys of its `_SUITES`) and the file names it leaves out (its
    `_EXCLUDED_FILES`, if any), read from its source without running it. ValueError when it names them another way.
    """
    with open(runner, encoding="utf-8", errors="replace") as fh:
        tree = ast.parse(fh.read(), runner)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(t, ast.Name):
                    found[t.id] = node.value
    suites = found.get("_SUITES")
    if not isinstance(suites, ast.Dict) or not all(
        isinstance(k, ast.Constant) and isinstance(k.value, str) for k in suites.keys
    ):
        raise ValueError("no _SUITES dict of package names")
    excluded = (
        ast.literal_eval(found["_EXCLUDED_FILES"])
        if found.get("_EXCLUDED_FILES") is not None
        else ()
    )
    excluded = (excluded,) if isinstance(excluded, str) else excluded
    if not isinstance(excluded, (tuple, list, set, frozenset)) or not all(
        isinstance(x, str) for x in excluded
    ):
        raise ValueError("_EXCLUDED_FILES is not a list of file names")
    return [k.value for k in suites.keys], set(excluded)


def with_in_tree(entries: list[dict], manifest: list[str], site: str) -> list[dict]:
    """Each entry with the installed in-tree runners it collects (`in_tree`): the runner, its suites (package ->
    installed directory, in the site-packages where its import system finds `rocke`), the names it leaves out, and
    why its suites could not be read, if so."""
    runners = []
    for path in sorted(
        os.path.normpath(p) for p in manifest if os.path.basename(p) == IN_TREE_RUNNER
    ):
        packages, excluded, why = [], set(), ""
        try:
            packages, excluded = in_tree_suites(path)
        except (OSError, SyntaxError, ValueError) as e:
            why = str(e)
        runners.append(
            {
                "runner": path,
                "suites": {p: os.path.join(site, *p.split(".")) for p in packages},
                "excluded": excluded,
                "why": why,
            }
        )
    for e in entries:
        e["in_tree"] = [r for r in runners if collects(e, r["runner"])]
    return entries


def in_tree_case(entry: dict, path: str) -> tuple[str, str] | None:
    """(runner, case) when one of the entry's in-tree runners runs the installed file `path`: a test_*.py file under a
    suite package's directory, by a name the runner does not leave out. The case is the runner's parameter id for the
    file, "<package>:<path in the package>"."""
    path = os.path.normpath(path)
    if not fnmatch.fnmatchcase(os.path.basename(path), IN_TREE_FILES):
        return None
    for r in entry.get("in_tree", ()):
        if os.path.basename(path) in r["excluded"]:
            continue
        for package, d in r["suites"].items():
            if path.startswith(d.rstrip(os.sep) + os.sep):
                return (
                    r["runner"],
                    f"{package}:{os.path.relpath(path, d).replace(os.sep, '/')}",
                )
    return None


def lane(entry: dict, path: str) -> str | None:
    """How the entry runs the installed file `path`: "direct" when its pytest collects it, "in-tree" when only its
    in-tree runner runs it, None when it does not run it."""
    if collects(entry, path):
        return "direct"
    return "in-tree" if in_tree_case(entry, path) else None


def case_file(entry: dict, tid: str) -> str:
    """An absolute id of a case of one of the entry's in-tree runners ("<runner>::<test>[<package>:<path>]"), as the
    installed file the case runs; any other id as it is."""
    f, _, rest = tid.partition("::")
    m = re.search(r"\[([^\[\]]+)\]$", rest)
    package, colon, rel = m.group(1).partition(":") if m else ("", "", "")
    for r in entry.get("in_tree", ()):
        if f == r["runner"] and colon and package in r["suites"]:
            return os.path.normpath(os.path.join(r["suites"][package], rel))
    return tid


def installed_copies(
    tree: str, files: list[str], manifest: list[str]
) -> dict[str, str | None]:
    """file (relative to the rocKE tree) -> its installed copy: a manifest path with the same name and bytes, the one
    sharing the longest path suffix with the source; None when nothing installs it."""
    by_name: dict[str, list[str]] = {}
    for p in manifest:
        by_name.setdefault(os.path.basename(p), []).append(p)

    def suffix(p: str, f: str) -> int:
        a, b = p.split(os.sep)[::-1], f.split("/")[::-1]
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n

    out = {}
    for f in files:
        src = os.path.join(tree, f)
        same = [
            p
            for p in by_name.get(os.path.basename(f), [])
            if os.path.isfile(p) and filecmp.cmp(src, p, shallow=False)
        ]
        out[f] = max(same, key=lambda p: suffix(p, f)) if same else None
    return out


def source_of(installed: str, test_root: str, known: dict[str, str]) -> str | None:
    """The source path (relative to the rocKE tree) of an installed test file: from the mapping when it is a changed
    file, else by the two test-tree install rules (tests/library/<x> <- library/<x>, tests/<x> <-
    platform/tests/<x>)."""
    if installed in known:
        return known[installed]
    rel = os.path.relpath(installed, test_root).replace(os.sep, "/")
    if rel.startswith("tests/library/"):
        return "library/" + rel[len("tests/library/") :]
    if rel.startswith("tests/"):
        return "platform/tests/" + rel[len("tests/") :]
    return None


# --- runs ------------------------------------------------------------------------------------------------------------


def plan_runs(
    entries: list[dict], copies: dict[str, str | None], cap: int = CAP
) -> dict:
    """What runs. Up to `cap` present changed files: each entry with the changed files it runs, BATCH per run, the
    files it runs through its in-tree runner in runs of their own. Over it: each entry that runs one of them, whole; a
    GPU entry is named instead (no gfx942/gfx950 here), and the files only GPU entries run are listed as not run.
    `lanes`: file -> {entry name: "direct" or "in-tree"}."""
    lanes = {}
    for f, p in copies.items():
        if p:
            kinds = {e["name"]: lane(e, p) for e in entries}
            lanes[f] = {n: k for n, k in kinds.items() if k}
    mapped = [f for f in copies if lanes.get(f)]
    runs, gpu = [], []
    whole = len(copies) > cap
    if whole:
        for e in entries:
            if any(e["name"] in lanes[f] for f in mapped):
                if "gpu" in e["labels"]:
                    gpu.append(e)
                else:
                    runs.append({"entry": e, "files": None})
    else:
        for e in entries:
            for kind in ("direct", "in-tree"):
                fs = [copies[f] for f in mapped if lanes[f].get(e["name"]) == kind]
                runs += [
                    {
                        "entry": e,
                        "files": fs[i : i + BATCH],
                        "in_tree": kind == "in-tree",
                    }
                    for i in range(0, len(fs), BATCH)
                ]
    whole_names = {r["entry"]["name"] for r in runs}
    return {
        "whole": whole,
        "lanes": lanes,
        "mapped": mapped,
        "runs": runs,
        "gpu_not_run": gpu,
        "gpu_only": [f for f in mapped if whole and not set(lanes[f]) & whole_names],
        "in_tree": [f for f in mapped if "in-tree" in lanes[f].values()],
        "laneless": [f for f, p in copies.items() if p and not lanes[f]],
        "not_installed": [f for f, p in copies.items() if not p],
    }


def run_env(entry: dict, site: str, plugins: str) -> dict:
    """CI's environment for an entry: the checks' ROCKE_* exports and PYTHONPATH dropped, the GPU hidden as the checks
    hide it (rkmon_lib/nogpu.py: HIP_VISIBLE_DEVICES=-1 would abort a test that touches HIP, a false "did not finish"),
    the entry's ENVIRONMENT applied, then the install's site-packages (CI's platform wheel), our pytest plugin and
    RKMON_PYEXTRA (the checks' PyYAML) after its PYTHONPATH."""
    env = nogpu.env(
        {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("ROCKE_") and k != "PYTHONPATH"
        }
    )
    env.update(entry["env"])
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            entry["env"].get("PYTHONPATH", ""),
            site,
            plugins,
            os.environ.get("RKMON_PYEXTRA", ""),
        )
        if p
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["COLUMNS"] = (
        "400"  # the short summary keeps the error text instead of cutting it at 80 columns
    )
    return env


def run(
    cmd: list[str], cwd: str | None, env: dict | None, log: str, limit: float
) -> tuple[int, bool]:
    """Run `cmd` (output to `log`) for at most `limit` seconds; its whole process group is killed on timeout."""
    with open(log, "w") as fh:
        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as e:
            fh.write(f"cannot start {cmd[0]}: {e}\n")
            return 127, False
        try:
            return p.wait(timeout=max(limit, 1)), False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return -9, True


def parse(text: str) -> dict:
    """A pytest log: failing ids (FAILED, ERROR, SUBFAILED) with their error line, passing ids, the ids it reported at
    all (-v progress lines), outcome counts, and whether it finished (its closing summary line). Outcome lines count
    only in the short test summary: -rA also prints captured output, which may hold such words at a line's start.
    """
    failed, passed, seen, counts, done, short = {}, set(), set(), {}, False, False
    for line in text.splitlines():
        if SUMMARY.match(line):
            done = True
            for n, k in COUNT.findall(line):
                k = "error" if k.startswith("error") else k
                counts[k] = counts.get(k, 0) + int(n)
            continue
        if re.match(r"^=+ short test summary info =+$", line):
            short = True
            continue
        if not short:
            m = PROGRESS.match(line)
            if m:
                seen.add(m.group(1))
            continue
        if line.startswith("PASSED "):
            passed.add(line[7:].strip())
            continue
        s = None
        if line.startswith(("FAILED ", "ERROR ")):
            s = line.split(" ", 1)[1]
        elif line.startswith(
            "SUBFAILED"
        ):  # "SUBFAILED[label] <id> - msg", "SUBFAILED(k='v') <id>"
            m = re.search(r"[)\]] (\S+\.py(?:::|$| ))", line)
            s = line[m.start(1) :] if m else None
        if s is None:
            continue
        i = s.find(" - ")
        tid, msg = (s[:i], s[i + 3 :]) if i >= 0 else (s, "")
        failed.setdefault(tid.strip(), msg.strip())
    return {
        "failed": failed,
        "passed": passed,
        "seen": seen | passed | set(failed),
        "counts": counts,
        "done": done,
    }


def canon(cwd: str, tid: str) -> str:
    """A test id relative to the run's directory, made absolute ("<file>::<rest>")."""
    f, sep, rest = tid.partition("::")
    return os.path.normpath(os.path.join(cwd, f)) + sep + rest


def first_error(text: str) -> str:
    for pat in (
        r"^E +(\S.*)$",
        r"^(ERROR: .*)$",
        r"^(Fatal Python error: .*)$",
        r"^(\S*Error: .*)$",
    ):
        m = re.search(pat, text, re.M)
        if m:
            return m.group(1).strip()
    return ""


def child_errors(text: str) -> dict[str, str]:
    """A pytest log's first nested error line under each failing test's name: an in-tree runner case fails with an
    assertion whose message is its child pytest's output, so the child's own `E` lines sit under the runner's
    ("E         E       AssertionError: ...")."""
    out, name = {}, None
    for line in text.splitlines():
        m = re.match(r"^_+ (\S.*?) _+$", line)
        if m or line.startswith("="):
            name = m.group(1) if m else None
            continue
        m = re.match(r"^E\s+E\s+(\S.*)$", line)
        if m and name and name not in out:
            out[name] = m.group(1).strip()
    return out


def in_tree_args(entry: dict, files: list[str]) -> list[str]:
    """The pytest arguments that run installed files as the entry's in-tree runner does: the entry's options, -k
    naming the runner's case for each file (joined to the entry's own -k: pytest keeps only the last), ours, and the
    runners."""
    cases = [in_tree_case(entry, f) for f in files]
    expr = " or ".join(f"[{c}]" for _r, c in cases)
    opts = entry["options"]
    theirs = [opts[i + 1] for i in range(len(opts) - 1) if opts[i] == "-k"]
    opts = [
        o for i, o in enumerate(opts) if o != "-k" and not (i and opts[i - 1] == "-k")
    ]
    return (
        opts
        + ["-k", f"({theirs[-1]}) and ({expr})" if theirs else expr]
        + EXTRA
        + sorted({os.path.relpath(r, entry["cwd"]) for r, _c in cases})
    )


def pytest_cmd(entry: dict, files: list[str] | None) -> list[str]:
    """The pytest command of a run under an entry: the given installed files, or the entry whole (`files` None). Files
    the entry runs through its in-tree runner run as CI runs them, through the runner (in_tree_args).
    """
    cmd = [entry["python"], "-m", "pytest"]
    if files is None:
        # Whole, without the paths the install lacks: they are reported on their own, and pytest would run nothing else.
        return cmd + [x for x in entry["args"] if x not in entry["missing"]] + EXTRA
    if files and all(lane(entry, f) == "in-tree" for f in files):
        return cmd + in_tree_args(entry, files)
    return (
        cmd
        + entry["options"]
        + EXTRA
        + [os.path.relpath(f, entry["cwd"]) for f in files]
    )


def env_file(entry: dict, site: str, plugins: str, path: str) -> str:
    """A shell file that sets run_env's environment for the entry (what it adds, and what it drops), for the re-run
    commands the report prints; `path`."""
    env = run_env(entry, site, plugins)
    drop = sorted(k for k in os.environ if k not in env)
    with open(path, "w") as fh:
        fh.write(f"unset {' '.join(drop)}\n" if drop else "")
        for k in sorted(env):
            if os.environ.get(k) != env[k] or k == "PATH":
                fh.write(f"export {k}={shlex.quote(env[k])}\n")
    return path


def env_data(entry: dict, site: str, plugins: str, path: str) -> dict:
    """run_env's environment for the entry, for the re-run commands through `rk pr-run`: what worker/pr_env.py lists
    of it written to `path` as `rk pr-run --env` reads it (NUL-separated NAME=VALUE), and the names the run sets that
    the list leaves out (COLUMNS, an entry's own ENVIRONMENT) returned, for the command line. The other names are the
    slot's (its HOME, TMPDIR and git's): a re-run's slot has its own."""
    env = run_env(entry, site, plugins)
    with open(path, "wb") as fh:
        fh.write(pr_env.data(env))
    return {
        k: v
        for k, v in sorted(env.items())
        if not pr_env.listed(k) and os.environ.get(k) != v
    }


def rerun_command(cmd: list[str], cwd: str, env: str, via: str = "", extra=None) -> str:
    """CMD again in CWD with the environment in the file ENV: through VIA (`rk pr-run`), with EXTRA (the names its
    file leaves out) on the command line; without it, in a subshell that sources ENV first.
    """
    if via:
        pre = ["env", *(f"{k}={v}" for k, v in extra.items())] if extra else []
        return f"{via} --env {shlex.quote(env)} -C {shlex.quote(cwd)} -- {shlex.join(pre + cmd)}"
    return f"(. {env} && cd {shlex.quote(cwd)} && {shlex.join(cmd)})"


def rerun_lines(
    plan: dict,
    entries: list[dict],
    copies: dict[str, str | None],
    site: str,
    plugins: str,
    out: str,
) -> list[str]:
    """The exact command that re-runs each changed file under each entry that runs it (as run here), or each entry
    run whole, in the entry's environment as run here. RKMON_RERUN_VIA (`rk pr-run`, which the job scripts set) runs
    each one as PR code in a slot of its own, with the environment file out/env_<entry>.0 (env_data) named where the
    job copies the results (RKMON_RESULTS; `out` without it), in the entry's folder; without it, a subshell sources
    out/env_<entry>.sh (env_file) and goes to the entry's folder first."""
    via = os.environ.get("RKMON_RERUN_VIA", "")
    results = os.environ.get("RKMON_RESULTS") or out
    lines = []
    for e in entries:
        mine = [f for f in plan["mapped"] if e["name"] in plan["lanes"][f]]
        if not mine or (
            plan["whole"] and not any(r["entry"] is e for r in plan["runs"])
        ):
            continue
        extra = None
        if via:
            name = f"env_{e['name']}.0"
            extra = env_data(e, site, plugins, os.path.join(out, name))
            env = os.path.join(results, name)
        else:
            env = env_file(e, site, plugins, os.path.join(out, f"env_{e['name']}.sh"))
        if plan["whole"]:
            cmd = rerun_command(pytest_cmd(e, None), e["cwd"], env, via, extra)
            lines.append(f"   {e['name']} (whole): {cmd}")
            continue
        for f in mine:
            cmd = rerun_command(pytest_cmd(e, [copies[f]]), e["cwd"], env, via, extra)
            lines.append(f"   {f} [{e['name']}]: {cmd}")
    return lines


def pytest_run(
    entry: dict,
    files: list[str] | None,
    site: str,
    plugins: str,
    log: str,
    limit: float,
) -> dict:
    """One pytest run under an entry (pytest_cmd). Ids come back absolute, a runner case's as the file it runs; so do
    the skipped tests' (`skips`: missing tool or "", id, reason). A run that did not finish fails each of its files,
    with the reason."""
    cmd = pytest_cmd(entry, files)
    t = time.monotonic()
    rc, timed_out = run(cmd, entry["cwd"], run_env(entry, site, plugins), log, limit)
    text = open(log, errors="replace").read()
    r = parse(text)
    ids = {k: case_file(entry, canon(entry["cwd"], k)) for k in r["seen"]}
    # A runner case fails with the runner's assertion of its child's output: the child's own first error says more.
    nested = child_errors(text) if entry.get("in_tree") else {}
    failed = {
        ids[k]: (
            nested.get(k.rpartition("::")[2], v)
            if ids[k] != canon(entry["cwd"], k)
            else v
        )
        for k, v in r["failed"].items()
    }
    skips = [
        (tool, case_file(entry, canon(entry["cwd"], tid)), why)
        for tool, tid, why in rkmon_skips.parse(text)
    ]
    r.update(
        rc=rc,
        secs=int(time.monotonic() - t),
        log=log,
        entry=entry["name"],
        files=files,
        failed=failed,
        passed={ids[k] for k in r["passed"]},
        seen=set(ids.values()),
        skips=skips,
    )
    # 77: the skip exit an entry declares with SKIP_RETURN_CODE 77 (every selected case skipped here).
    if not r["done"] or rc not in (0, 1, 5, 77):
        why = f"timed out after {int(limit)} s" if timed_out else f"rc={rc}"
        err = first_error(text)
        targets = (
            files
            if files is not None
            else [
                os.path.normpath(os.path.join(entry["cwd"], p.split("::")[0]))
                for p in entry["paths"]
                if p not in entry["missing"]
            ]
        )
        for f in targets:
            r["failed"].setdefault(
                f, f"the run did not finish ({why})" + (f": {err}" if err else "")
            )
        r["done"] = False
    return r


# --- the install -----------------------------------------------------------------------------------------------------


def cmake_error(text: str) -> str:
    """The first CMake error of a log, its indented lines joined: 'file INSTALL cannot find "<path>": ...'."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CMake Error"):
            body = []
            for x in lines[i + 1 : i + 6]:
                if x and not x.startswith(" "):
                    break
                if x.strip():
                    body.append(x.strip())
            return " ".join(body) or line.strip()
    return next((x.strip() for x in reversed(lines) if x.strip()), "")


def install(
    cmake: str, build: str, prefix: str, log: str, limit: float, fail: bool = True
) -> None:
    rc, timed_out = run(
        [cmake, "--install", build, "--prefix", prefix], None, None, log, limit
    )
    if rc != 0:
        why = (
            f"timed out after {int(limit)} s"
            if timed_out
            else cmake_error(open(log, errors="replace").read())
        )
        raise NotRun(f"cmake --install failed (rc={rc}): {why[:220]}", fail=fail)


def read_install(ctest: str, build: str, prefix: str, out: str, tag: str) -> dict:
    """An install's manifest, CTest directory, test root, site-packages (the installed `rocke`) and pytest entries,
    each with the in-tree runners it collects."""
    try:
        manifest = [
            x.strip()
            for x in open(os.path.join(build, "install_manifest.txt"))
            if x.strip()
        ]
    except OSError as e:
        raise NotRun(f"no install manifest in {build} ({e.strerror})") from e
    ctfiles = sorted(
        (p for p in manifest if os.path.basename(p) == "CTestTestfile.cmake"), key=len
    )
    if not ctfiles:
        raise NotRun(
            "the install has no CTestTestfile.cmake (ROCKE_INSTALL_TESTS off?)"
        )
    ctdir = os.path.dirname(ctfiles[0])
    pkgs = sorted(
        (
            p
            for p in manifest
            if p.endswith(os.sep + os.path.join("rocke", "__init__.py"))
        ),
        key=lambda p: (
            os.path.basename(os.path.dirname(os.path.dirname(p))) != "site-packages",
            len(p),
        ),
    )
    if not pkgs:
        raise NotRun(
            "the install has no rocke package (ROCKE_INSTALL_PYTHON_PACKAGE off?)"
        )
    try:
        p = subprocess.run(
            [ctest, "--show-only=json-v1"],
            cwd=ctdir,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise NotRun(f"ctest --show-only=json-v1 did not run ({e})") from e
    with open(os.path.join(out, f"{tag}ctest.json"), "w") as fh:
        fh.write(p.stdout)
    if p.returncode != 0:
        raise NotRun(
            f"ctest --show-only=json-v1 failed (rc={p.returncode}): {p.stderr.strip()[:160]}"
        )
    parent = os.path.dirname(ctdir)
    site = os.path.dirname(os.path.dirname(pkgs[0]))
    return {
        "manifest": manifest,
        "ctdir": ctdir,
        "prefix": prefix,
        "test_root": parent if os.path.isdir(os.path.join(parent, "tests")) else ctdir,
        "site": site,
        "entries": with_in_tree(pytest_entries(p.stdout), manifest, site),
    }


def grace(script: str, log: str, say) -> None:
    """The machine limit before a CPU-heavy step: no heavy CPU in a reviewer-model worker's first minutes
    (worker_grace.py waits; a shell script runs under bash)."""
    if script:
        run(
            (
                [sys.executable, "-B", script]
                if script.endswith(".py")
                else ["bash", script]
            ),
            None,
            None,
            log,
            960,
        )
        last = [x.strip() for x in open(log, errors="replace") if x.strip()]
        if last and "waiting" in " ".join(last):
            say(last[-1])


def mem_available_gb() -> float:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("inf")


# --- develop ---------------------------------------------------------------------------------------------------------


def well_formed(entry: str, tid: str, outcome: str | None = None) -> bool:
    """A line of a todo, an outcomes file or the cache, read as data: an entry name, a printable id of at most 1000
    characters, and (with `outcome`) one of OUTCOMES."""
    return (
        bool(ENTRY_NAME.match(entry))
        and 0 < len(tid) <= 1000
        and tid.isprintable()
        and (outcome is None or outcome in OUTCOMES)
    )


def read_rows(path: str, fields: int, cap: int) -> list[list[str]]:
    """The well-formed lines of a tab-separated file of `fields` fields (`well_formed`), at most `cap`; none when it
    cannot be read."""
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == fields and well_formed(*parts):
                    out.append(parts)
                    if len(out) >= cap:
                        break
    except OSError:
        pass
    return out


def load_cache(path: str) -> dict[tuple[str, str], str]:
    """Develop's outcomes at one SHA, by (entry, id relative to the CTest directory): a line `entry<TAB>id<TAB>outcome`
    each (OUTCOMES), as the cache and a develop job's develop_outcomes.tsv hold them; {} without the file.
    """
    return {(e, i): o for e, i, o in read_rows(path, 3, CACHE_MAX)} if path else {}


def cache_add(src: str, cache: str) -> int:
    """The well-formed lines of `src` (a develop job's develop_outcomes.tsv, read as data) for ids `cache` lacks, added
    to it when their run finished (CACHED): the whole cache written whole (rkmon_lib/whole_file.py), a tree's file
    (perms.TREE_FILE_MODE) as in the monitor's trees. The number of ids added.
    """
    have = load_cache(cache)
    new = {
        (e, i): o
        for e, i, o in read_rows(src, 3, TODO_MAX)
        if o in CACHED and (e, i) not in have
    }
    room = CACHE_MAX - len(have)
    new = dict(sorted(new.items())[: max(room, 0)])
    if not new:
        return 0
    whole_file.write(
        cache,
        "".join(f"{e}\t{i}\t{o}\n" for (e, i), o in sorted({**have, **new}.items())),
        perms.TREE_FILE_MODE,
    )
    return len(new)


class Develop:
    """Develop's outcomes at the merge's develop SHA. A merge's run (`compare`) takes them from the cache, which it reads
    and never writes, and lists the failing ids the cache lacks (`todo`). A develop job of their own gives those
    (`outcomes`): develop's tree built and installed on first need, once per SHA (a stamp in its prefix), and each id
    run under develop's entry.
    """

    def __init__(self, a, out: str, deadline: float, say):
        self.a, self.out, self.deadline, self.say = a, out, deadline, say
        # A develop job's outcomes count only for the ids the cache lacks.
        self.cache = {
            **load_cache(getattr(a, "outcomes", "")),
            **load_cache(getattr(a, "cache", "")),
        }
        self.inst, self.why = None, None
        # (entry, id relative to the CTest directory)
        self.todo: list[tuple[str, str]] = []

    def tree_has(self, src: str | None) -> bool | None:
        if (
            self.a.dev_tree in ("", "-")
            or not os.path.isdir(self.a.dev_tree)
            or not src
        ):
            return None
        return os.path.exists(os.path.join(self.a.dev_tree, src))

    def install(self) -> dict | None:
        if self.inst or self.why:
            return self.inst
        a = self.a
        try:
            if a.dev_tree in ("", "-") or not os.path.isdir(a.dev_tree):
                raise NotRun("no develop tree")
            if not (a.dev_sha and a.dev_build and a.dev_core and a.dev_prefix):
                raise NotRun("no develop build configured")
            stamp = os.path.join(a.dev_prefix, "rkmon_develop_sha")
            if os.path.exists(stamp) and open(stamp).read().strip() == a.dev_sha:
                self.inst = read_install(
                    a.ctest, a.dev_core, a.dev_prefix, self.out, "develop_"
                )
                self.say(
                    f"develop {a.dev_sha[:10]}: its installed tree from an earlier run ({a.dev_prefix})"
                )
                return self.inst
            head = subprocess.run(
                ["git", "-C", a.dev_tree, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout.strip()
            if head != a.dev_sha:
                raise NotRun(
                    f"the develop tree is at {head[:10] or '?'}, not {a.dev_sha[:10]}"
                )
            if mem_available_gb() < a.min_free_gb:
                raise NotRun(
                    f"low memory for develop's build (under {a.min_free_gb} GiB available)"
                )
            grace(a.grace, os.path.join(self.out, "develop_grace.log"), self.say)
            t = time.monotonic()
            left = self.deadline - time.monotonic()
            if left < 120:
                raise NotRun("the time box ran out before develop's build")
            rc, timed_out = run(
                ["bash", "-c", a.dev_build],
                None,
                None,
                os.path.join(self.out, "develop_build.log"),
                min(900, left),
            )
            if rc != 0:
                raise NotRun(
                    f"develop's native build failed ({'timed out' if timed_out else f'rc={rc}'})"
                )
            install(
                a.cmake,
                a.dev_core,
                a.dev_prefix,
                os.path.join(self.out, "develop_install.log"),
                max(60, self.deadline - time.monotonic()),
                fail=False,
            )
            with open(stamp, "w") as fh:
                fh.write(a.dev_sha + "\n")
            self.inst = read_install(
                a.ctest, a.dev_core, a.dev_prefix, self.out, "develop_"
            )
            self.say(
                f"develop {a.dev_sha[:10]}: built and installed for the comparison "
                f"({int(time.monotonic() - t)} s, {a.dev_prefix})"
            )
        except NotRun as e:
            self.why = str(e)
        return self.inst

    def compare(
        self, failing: dict[tuple[str, str], str], pr: dict, known: dict[str, str]
    ) -> dict:
        """(entry, id) -> (label, note) for the merge's failing ids, from the trees and the cache; an id the cache lacks
        is not compared, and goes to `todo`."""
        out = {}
        for (ename, cid), _msg in failing.items():
            f, _, rest = cid.partition("::")
            has = self.tree_has(source_of(f, pr["test_root"], known))
            if has is False:
                out[(ename, cid)] = (PR, "a file the PR adds")
                continue
            if has is None and self.a.dev_tree in ("", "-"):
                out[(ename, cid)] = (NOCOMPARE, "no develop tree")
                continue
            rel = os.path.relpath(f, pr["ctdir"]) + ("::" + rest if rest else "")
            o = self.cache.get((ename, rel))
            if o:
                out[(ename, cid)] = self.label(o)
            else:
                self.todo.append((ename, rel))
                out[(ename, cid)] = (NOCOMPARE, NOT_KNOWN)
        return out

    def outcomes(self, todo: list[tuple[str, str]]) -> dict:
        """(entry, id) -> develop's outcome (OUTCOMES) for each of `todo`, or (NOCOMPARE, why) when develop's run cannot
        tell; develop built and installed first (`install`)."""
        out: dict = {}
        dev = self.install()
        if not dev:
            return {k: (NOCOMPARE, self.why) for k in todo}
        dentries = {e["name"]: e for e in dev["entries"]}
        groups: dict[tuple[str, str], list] = {}
        for ename, rel in todo:
            d = dentries.get(ename)
            dfile = os.path.normpath(os.path.join(dev["ctdir"], rel.partition("::")[0]))
            kind = lane(d, dfile) if d and os.path.exists(dfile) else None
            if not kind:
                out[(ename, rel)] = "nolane"
            else:  # a file develop's entry collects and one its in-tree runner runs never share a run
                groups.setdefault((ename, kind), []).append((rel, dfile))
        for k, ((ename, _kind), items) in enumerate(sorted(groups.items())):
            d = dentries[ename]
            files = sorted({x[1] for x in items})
            left = self.deadline - time.monotonic()
            if left < 30:
                for rel, _f in items:
                    out[(ename, rel)] = (
                        NOCOMPARE,
                        "the time box ran out before develop's run",
                    )
                continue
            r = pytest_run(
                d,
                files,
                dev["site"],
                self.a.plugins,
                os.path.join(self.out, f"develop_run{k}.log"),
                min(max(2 * (d["timeout"] or 300), 300), left),
            )
            for rel, dfile in items:
                did = canon(dev["ctdir"], rel)
                if not r["done"]:
                    # A whole file whose run did not finish at the merge either is develop's too (a crash both
                    # share); a single test cannot be told apart.
                    out[(ename, rel)] = (
                        "unfinished"
                        if "::" not in rel
                        else (
                            NOCOMPARE,
                            f"develop's run of it did not finish (rc={r['rc']})",
                        )
                    )
                    continue
                out[(ename, rel)] = (
                    "failed"
                    if did in r["failed"] or dfile in r["failed"]
                    else (
                        "passed"
                        if did in r["passed"]
                        else "unrun" if did in r["seen"] else "absent"
                    )
                )
        return out

    @staticmethod
    def label(outcome: str) -> tuple[str, str]:
        return {
            "failed": (DEVELOP, "fails on develop too"),
            "passed": (PR, "passes on develop"),
            "unrun": (UNRUN, "develop skips it"),
            "absent": (PR, "a test develop lacks"),
            "nolane": (
                PR,
                "develop's installed lanes do not run it: the PR's enrolment exposes it",
            ),
            "unfinished": (DEVELOP, "develop's run of it does not finish either"),
        }.get(outcome, (NOCOMPARE, f"unknown develop outcome {outcome!r}"))


def develop_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="installed_tests.py develop",
        description="a develop job: develop's installed run of the ids a merge's run could not compare",
    )
    p.add_argument(
        "--out", required=True, help="this job's logs and develop_outcomes.tsv"
    )
    p.add_argument("--todo", required=True, help="a merge's run's develop_todo.tsv")
    p.add_argument(
        "--dev-tree",
        required=True,
        help="develop's rocKE tree at the merge's develop SHA",
    )
    p.add_argument(
        "--dev-sha",
        required=True,
        help="the merge's develop SHA (the develop tree must be at it)",
    )
    p.add_argument(
        "--dev-build",
        required=True,
        help="shell command that builds develop's tree into --dev-core",
    )
    p.add_argument("--dev-core", required=True)
    p.add_argument("--dev-prefix", required=True)
    p.add_argument("--plugins", default=PLUGINS)
    p.add_argument(
        "--grace", default="", help="worker_grace.py, run before develop's build"
    )
    p.add_argument("--min-free-gb", type=float, default=16)
    p.add_argument(
        "--budget", type=float, default=1200, help="the time box, in seconds"
    )
    p.add_argument("--cmake", default="cmake")
    p.add_argument("--ctest", default="ctest")
    return p


def develop_main(argv: list[str]) -> int:
    """`develop` (the module's usage)."""
    a = develop_args().parse_args(argv)
    t0 = time.monotonic()
    os.makedirs(a.out, exist_ok=True)

    def say(line: str = "") -> None:
        print(line, flush=True)

    todo = sorted({(e, i) for e, i in read_rows(a.todo, 2, TODO_MAX)})
    got = Develop(a, a.out, t0 + a.budget, say).outcomes(todo) if todo else {}
    known = {k: v for k, v in got.items() if isinstance(v, str)}
    with open(os.path.join(a.out, OUTCOMES_FILE), "w", encoding="utf-8") as fh:
        fh.writelines(f"{e}\t{i}\t{o}\n" for (e, i), o in sorted(known.items()))
    why = sorted({v[1] for v in got.values() if not isinstance(v, str)})
    for w in why[:4]:
        say(f"not known: {w}")
    say(
        f"INSTALLED_DEVELOP {len(known)} of {len(todo)} outcomes at develop {a.dev_sha[:10]}, "
        f"{int(time.monotonic() - t0)} s" + (f": {why[0]}" if why else "")
    )
    return 0


def cache_add_main(argv: list[str]) -> int:
    """`cache-add` (the module's usage)."""
    if len(argv) != 2:
        sys.stderr.write("usage: installed_tests.py cache-add OUTCOMES CACHE\n")
        return 2
    n = cache_add(argv[0], argv[1])
    print(f"develop's outcomes added to the cache: {n}")
    return 0 if n else 1


# --- the section -----------------------------------------------------------------------------------------------------


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--out", required=True, help="this section's logs")
    p.add_argument(
        "--tree",
        required=True,
        help="the merged rocKE tree (the rocKE folder of the merge's worktree)",
    )
    p.add_argument(
        "--build", required=True, help="the checks' native core build of that tree"
    )
    p.add_argument("--prefix", required=True, help="the scratch install prefix")
    p.add_argument(
        "--build-ok",
        default="",
        help="a file that exists only when this run's native build succeeded",
    )
    p.add_argument(
        "--dev-tree",
        default="-",
        help="develop's rocKE tree at the merge's develop SHA, or -",
    )
    p.add_argument(
        "--cache",
        default="",
        help="develop's outcomes at this develop SHA (read, never written)",
    )
    p.add_argument(
        "--outcomes",
        default="",
        help="a develop job's develop_outcomes.tsv, for the ids the cache lacks",
    )
    p.add_argument("--plugins", default=PLUGINS)
    p.add_argument(
        "--grace",
        default="",
        help="worker_grace.py, run before the install",
    )
    p.add_argument(
        "--budget", type=float, default=1200, help="the time box, in seconds"
    )
    p.add_argument("--cap", type=int, default=CAP)
    p.add_argument("--cmake", default="cmake")
    p.add_argument("--ctest", default="ctest")
    p.add_argument("files", nargs="*")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["develop"]:
        return develop_main(argv[1:])
    if argv[:1] == ["cache-add"]:
        return cache_add_main(argv[1:])
    a = args_parser().parse_args(argv)
    t0 = time.monotonic()
    deadline = t0 + a.budget
    os.makedirs(a.out, exist_ok=True)

    def say(line: str = "", raw: bool = False) -> None:
        print(
            line if raw else line.replace(a.prefix.rstrip("/"), "<install>"), flush=True
        )

    def secs() -> str:
        return f"{int(time.monotonic() - t0)} s"

    files = sorted(set(a.files))
    present = [f for f in files if os.path.isfile(os.path.join(a.tree, f))]
    deleted = [f for f in files if f not in present]
    try:
        if a.build_ok and not os.path.exists(a.build_ok):
            raise NotRun("no native build of this merge (it failed or was skipped)")
        if not os.path.isfile(os.path.join(a.build, "CMakeCache.txt")):
            raise NotRun(f"no native build at {a.build}")
        grace(a.grace, os.path.join(a.out, "grace.log"), say)
        deadline = (
            time.monotonic() + a.budget
        )  # the time box starts after a wait for a young reviewer-model worker
        install(
            a.cmake,
            a.build,
            a.prefix,
            os.path.join(a.out, "install.log"),
            max(60, min(600, a.budget / 2)),
        )
        pr = read_install(a.ctest, a.build, a.prefix, a.out, "")
    except NotRun as e:
        say(f"NOT RUN: {e}")
        say(f"INSTALLED {'fail' if e.fail else 'skip'} NOT RUN, {secs()}: {e}")
        return 0
    say(
        f"installed (<install>): cmake --install {a.build} --prefix {a.prefix}: {len(pr['manifest'])} files; "
        f"{len(pr['entries'])} installed CTest entries run pytest, from {pr['ctdir']}",
        raw=True,
    )
    copies = installed_copies(a.tree, present, pr["manifest"])
    known = {p: f for f, p in copies.items() if p}
    plan = plan_runs(pr["entries"], copies, a.cap)
    via = (
        f" ({len(plan['in_tree'])} through the in-tree lane)" if plan["in_tree"] else ""
    )
    say(
        f"changed test files: {len(files)}: {len(plan['mapped'])} in installed CTest entries{via}, "
        f"{len(plan['laneless'])} installed in no entry, {len(plan['not_installed'])} not installed, "
        f"{len(deleted)} deleted by the PR"
    )
    for e in pr["entries"]:
        for kind, head in (
            ("direct", e["name"]),
            ("in-tree", f"run through the in-tree lane ({e['name']})"),
        ):
            mine = [
                f for f in plan["mapped"] if plan["lanes"][f].get(e["name"]) == kind
            ]
            if mine:
                say(
                    f"   {head}: {' '.join(mine[:6])}{f' (+{len(mine) - 6} more)' if len(mine) > 6 else ''}"
                )
    if plan["laneless"]:
        say(
            f"   installed in no CTest entry (CI never runs them installed): {' '.join(plan['laneless'][:6])}"
        )
        for r in {
            r["runner"]: r for e in pr["entries"] for r in e["in_tree"] if r["why"]
        }.values():
            say(
                f"   (the in-tree runner {source_of(r['runner'], pr['test_root'], known) or r['runner']} names its "
                f"suites in a way this cannot read, so a file of its suites reads as in no entry: {r['why']})"
            )
    if plan["not_installed"]:
        say(
            f"   not installed (no CI lane can run them): {' '.join(plan['not_installed'][:6])}"
        )
    dev = Develop(a, a.out, deadline, say)
    # An entry naming a path the install lacks stops pytest ("file or directory not found"): CI fails the whole entry
    # (a test file deleted or moved while a CMake list still names it).
    broken = {}
    for e in pr["entries"]:
        for p in e["missing"]:
            f = os.path.normpath(os.path.join(e["cwd"], p.split("::")[0]))
            broken[(e["name"], f)] = blame_missing(
                a.tree, a.dev_tree, source_of(f, pr["test_root"], known)
            )
    for (ename, f), (lab, note) in sorted(broken.items()):
        say(
            f"   {ename} names a path the install lacks, so pytest stops and CI fails the entry: "
            f"{source_of(f, pr['test_root'], known) or f} ({note})"
        )
    if plan["whole"]:
        say(
            f"over the cap: {len(present)} changed test files (cap {a.cap}): the installed CTest entries that collect "
            f"them run whole: {' '.join(r['entry']['name'] for r in plan['runs']) or 'none'}"
            + (
                f"; GPU entries not run whole (no gfx942/gfx950 here): "
                f"{' '.join(e['name'] for e in plan['gpu_not_run'])}"
                if plan["gpu_not_run"]
                else ""
            )
        )
        if plan["gpu_only"]:
            say(
                f"   not run here, only GPU entries collect them ({len(plan['gpu_only'])}): "
                f"{' '.join(plan['gpu_only'][:6])}"
            )
    results, timeboxed = [], []
    for k, r in enumerate(plan["runs"]):
        e = r["entry"]
        left = deadline - time.monotonic()
        if left < 30:
            timeboxed.append(r)
            continue
        res = pytest_run(
            e,
            r["files"],
            pr["site"],
            a.plugins,
            os.path.join(a.out, f"run{k}_{e['name']}.log"),
            min(max(2 * (e["timeout"] or 300), 300), left),
        )
        results.append(res)
        c = res["counts"]
        what = (
            "whole"
            if r["files"] is None
            else f"{len(r['files'])} files"
            + (" through the in-tree lane" if r["in_tree"] else "")
        )
        tally = (
            ", ".join(
                f"{c[x]} {x}"
                for x in ("failed", "error", "passed", "skipped")
                if c.get(x)
            )
            or "no result"
        )
        say(f"{e['name']} ({what}, {res['secs']} s, rc={res['rc']}): {tally}")
        for line in open(res["log"], errors="replace"):
            if line.startswith(("rkmon: ", "rkmon-import: ")):
                say("   " + line.rstrip())

    def show(cid: str) -> str:
        f, sep, rest = cid.partition("::")
        return (
            (source_of(f, pr["test_root"], known) or os.path.relpath(f, a.prefix))
            + sep
            + rest
        )

    ran, uncased = set(), set()
    for res in results:
        if res["done"]:
            hit = (
                res["files"]
                if res["files"] is not None
                else [
                    copies[f]
                    for f in plan["mapped"]
                    if res["entry"] in plan["lanes"][f]
                ]
            )
            for f in (known[p] for p in hit if p in known):
                # A file of the in-tree lane ran only when the run reports the runner's case for it: a runner whose
                # case ids are not "<package>:<path>" leaves our -k nothing to select, and pytest just deselects.
                if (
                    plan["lanes"][f].get(res["entry"]) == "in-tree"
                    and copies[f] not in res["seen"]
                ):
                    uncased.add(f)
                else:
                    ran.add(f)
    unran = [f for f in plan["mapped"] if f not in ran and f not in plan["gpu_only"]]
    nocase = [f for f in unran if f in uncased]
    stalled = [f for f in unran if f not in uncased]
    failing = {
        (res["entry"], cid): msg
        for res in results
        for cid, msg in res["failed"].items()
    }
    labels = dev.compare(failing, pr, known) if failing else {}
    if dev.todo:
        with open(os.path.join(a.out, TODO_FILE), "w", encoding="utf-8") as fh:
            fh.writelines(f"{e}\t{i}\n" for e, i in sorted(set(dev.todo)))
    labels.update({k: v for k, v in broken.items()})
    msgs = dict(failing)
    msgs.update({k: "the entry names a path the install lacks" for k in broken})

    groups = {
        lab: sorted(k for k, v in labels.items() if v[0] == lab)
        for lab in (PR, DEVELOP, UNRUN, NOCOMPARE)
    }
    heads = {
        PR: "failing in the installed tree, on the merge only",
        DEVELOP: "fail on develop too, not this PR's doing",
        UNRUN: "not compared, develop skips them",
        NOCOMPARE: "not compared with develop",
    }
    for lab in (PR, UNRUN, NOCOMPARE, DEVELOP):
        if groups[lab]:
            say(f"{heads[lab]} ({len(groups[lab])}):")
            for ename, cid in groups[lab][:8]:
                msg = msgs[(ename, cid)][:240] or "(no message)"
                say(f"   {show(cid)} [{ename}] ({labels[(ename, cid)][1]}): {msg}")
    if timeboxed or stalled:
        say(
            f"NOT RUN (the time box, {int(a.budget)} s, ran out, or the run did not finish): "
            f"{' '.join(stalled[:8]) or ' '.join(r['entry']['name'] for r in timeboxed)}"
        )
    if nocase:
        say(
            f"NOT RUN (the run reported no case of the in-tree runner for them; see its log): {' '.join(nocase[:8])}"
        )
    # The skipped tests of the changed files (a whole entry's run skips others too), each with its reason.
    skips, seen_skip = [], set()
    for res in results:
        for tool, cid, why in res["skips"]:
            if cid.partition("::")[0] in known and (cid, why) not in seen_skip:
                seen_skip.add((cid, why))
                skips.append((tool, show(cid), why))
    with open(os.path.join(a.out, "skipped.tsv"), "w") as fh:
        fh.writelines(
            rkmon_skips.line(tid, why, {tool} if tool else set()) + "\n"
            for tool, tid, why in skips
        )
    for ln in rkmon_skips.report(skips, os.path.join(a.out, "skipped.tsv")):
        say(ln, raw=True)
    rr = rerun_lines(plan, pr["entries"], copies, pr["site"], a.plugins, a.out)
    if rr:
        say(
            "re-run one file as this run did (add -k or a test id after the file):",
            raw=True,
        )
        for ln in rr:
            say(ln, raw=True)
    notrun = rkmon_skips.not_run_text(skips)
    tot: dict[str, int] = {}
    for res in results:
        for x, n in res["counts"].items():
            tot[x] = tot.get(x, 0) + n
    tally = ", ".join(
        f"{tot[x]} {x}" for x in ("passed", "skipped", "failed", "error") if tot.get(x)
    )
    head = f"{len(ran)} run of {len(present)} changed test files, {secs()}"

    def first(lab: str) -> str:
        ename, cid = groups[lab][0]
        msg = msgs[(ename, cid)].replace(a.prefix.rstrip("/"), "<install>")
        more = f" (+{len(groups[lab]) - 1} more)" if len(groups[lab]) > 1 else ""
        return f"{show(cid)}{more} - {msg[:150]}"

    nr = f"; {notrun}" if notrun else ""
    if groups[PR]:
        line = f"fail {head}: {len(groups[PR])} failing (on the merge only): {first(PR)}{nr}"
    elif groups[UNRUN]:
        line = f"fail {head}: {len(groups[UNRUN])} failing (not compared: develop skips them): {first(UNRUN)}{nr}"
    elif groups[NOCOMPARE]:
        ename, cid = groups[NOCOMPARE][0]
        line = (
            f"fail {head}: {len(groups[NOCOMPARE])} failing (not compared with develop: "
            f"{labels[(ename, cid)][1]}): {first(NOCOMPARE)}{nr}"
        )
    elif timeboxed or unran:
        why = " or ".join(
            w
            for w, c in (
                ("time box or unfinished run", timeboxed or stalled),
                ("no in-tree case reported", nocase),
            )
            if c
        )
        line = (
            f"skip {head}: NOT RUN for {len(unran) or len(timeboxed)} ({why})"
            + (f"; {tally}" if tally else "")
            + nr
        )
    elif not ran:
        why = (
            "none is in an installed CTest entry (CI never runs them installed)"
            if present
            else "the PR deletes them all"
        )
        line = f"skip {head}: {why}"
    else:
        dv = f"; {len(groups[DEVELOP])} fail on develop too" if groups[DEVELOP] else ""
        ln = (
            f"; {len(plan['laneless'])} in no installed CTest entry"
            if plan["laneless"]
            else ""
        )
        gp = (
            f"; {len(plan['gpu_only'])} only in GPU entries, not run"
            if plan["gpu_only"]
            else ""
        )
        if notrun:
            line = f"skip {notrun}; {head}: {tally or 'no test outcome'}{dv}{ln}{gp}"
        else:
            sk = f"({rkmon_skips.summary(skips)}) " if skips else ""
            line = f"pass {sk}{head}: {tally or 'no test outcome'}{dv}{ln}{gp}"
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(
            {
                "labels": {f"{k[0]} {show(k[1])}": v for k, v in labels.items()},
                "ran": sorted(ran),
                "mapped": plan["mapped"],
                "in_tree": plan["in_tree"],
                "laneless": plan["laneless"],
                "not_installed": plan["not_installed"],
                "deleted": deleted,
                "whole": plan["whole"],
                "secs": int(time.monotonic() - t0),
                "skipped": [
                    {"id": tid, "reason": why, "missing": tool}
                    for tool, tid, why in skips
                ],
            },
            fh,
            indent=1,
        )
    say("INSTALLED " + line)
    return 0


def blame_missing(tree: str, dev_tree: str, src: str | None) -> tuple[str, str]:
    """Who broke an entry that names a path the install lacks. The PR, when develop has the file and the merge does
    not (deleted or moved while a CMake list still names it), or when only the merge's list names it; develop, when
    develop's platform/CMakeLists.txt names it too (develop's entry is broken the same way).
    """
    if dev_tree in ("", "-") or not os.path.isdir(dev_tree):
        return NOCOMPARE, "no develop tree"
    if (
        src
        and os.path.exists(os.path.join(dev_tree, src))
        and not os.path.exists(os.path.join(tree, src))
    ):
        return PR, "the PR deletes or moves it"
    rel = (
        (src or "").replace("library/", "tests/library/", 1)
        if (src or "").startswith("library/")
        else (src or "").replace("platform/tests/", "tests/", 1)
    )
    try:
        named = (
            bool(src)
            and rel
            in open(
                os.path.join(dev_tree, "platform", "CMakeLists.txt"), errors="replace"
            ).read()
        )
    except OSError:
        named = False
    return (
        (DEVELOP, "develop's list names it too")
        if named
        else (PR, "only the merge's CMake list names it")
    )


if __name__ == "__main__":
    sys.exit(main())
