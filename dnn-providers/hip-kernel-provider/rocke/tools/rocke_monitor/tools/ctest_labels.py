# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CTest labels and tiers, in every review's checks. A test whose add_test() a PR
moves out of the directory that labels it keeps its name and passes, but loses its YAML tier labels in one tree, and
CI's tier runs (`ctest -L quick`, TheRock's test_hipkernelprovider.py) silently stop running it (C++ host tests moved
to a directory where TestCategoryLabels.cmake does not apply, while the installed tree keeps their tiers).

**The trees.** The merge base and the head, each written to a scratch directory (`git archive`: the rocKE tree, the
provider's CMakeLists.txt, src/CMakeLists.txt and test-category YAMLs, and shared/ctest), then configured, not built,
as hip-kernel-provider configures rocKE: its YAML categorization block (shared/ctest/TestCategories.cmake with rocKE's
YAML), then `add_subdirectory(rocke)` as part of the provider. The provider's own configure needs hip, hiprtc and the
hipDNN SDKs, which this host lacks, so a small top-level CMakeLists.txt (SHIM) does those two steps; a tree whose
provider wires rocKE in a way SHIM does not copy is flagged. GPU reference entries need their DVC bundles and are left
out (ROCKE_INSTALL_TEST_GPU_REFERENCES=OFF). The checks' own configure (worker/build_native.py) is rocke/platform
standalone, with no YAML tiers, so this check runs its own: a few seconds a tree.

**The entries.** The build tree: `ctest --show-only=json-v1` in the build directory, with each entry's labels and its
add_test() (file:line, and the directory whose CMakeLists.txt registered it). The installed tree: the provider's
installed file as src/CMakeLists.txt writes it (rocKE's staged add_test() lines, then the YAMLs' labels by test name),
listed by the same ctest.

**The test map.** Each test file the PR adds or changes (test_*.py at the head, against the merge base) -> the head's
CTest entries that run it, in each tree (a listed file, a file under a listed directory that no --ignore drops, or a
file the in-tree runner's suites hold, as installed_tests.py reads them; the installed tree's paths map back to the
source as the install lays it out: tests/library/ <- library/, tests/ <- platform/tests/, rocke/ <-
platform/python/rocke/) -> each entry's tier labels (the categories YAML's) -> the PR-CI lanes that run it: config
`ci_test_tiers` (--ci-tiers), each a CI job, the tree it runs, the tier label it selects (`ctest -L '^<tier>$'`) and
the labels it leaves out (`-LE`). Without lanes the map stops at the tiers.

**Leads** (the section feeds no gate): an entry that loses a tier label (quick, standard, comprehensive, full) in a
tree, with its add_test()'s move when there is one and what the other tree keeps; an entry that leaves a tree; an
entry the PR adds with no tier label in a tree; a test file the PR adds or changes that no PR-CI lane runs. Other
label changes are listed. A head that does not configure while the merge base does is a lead too; a merge base that
does not configure, or a YAML the parser could not read, is NOT RUN.

Usage: python3 -I ctest_labels.py --repo R --head SHA --out DIR [--base REV] [--rocke-path K] [--python PY]
       [--budget 120] [--ci-tiers JSON]
--python: an interpreter with PyYAML for the YAML parser (default: the first of this one and /usr/bin/python3 that
has it). Prints the section, then one last line: "LABELS pass|fail|skip <detail>".
       python3 -I ctest_labels.py --repo R --head SHA [--base REV] [--rocke-path K] --fetch
Fetches the blobs of the clone the check reads, and runs nothing else (rkmon_lib/prtree.py, fetch_main).
"""
import argparse
import fnmatch
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
from rkmon_lib import gitops, prtree  # noqa: E402
from installed_tests import (
    IN_TREE_FILES,
    IN_TREE_RUNNER,
    in_tree_suites,
    split_args,
)  # noqa: E402

BUDGET_S = 120
TIERS = ("quick", "standard", "comprehensive", "full")
TREES = ("build", "installed")
SHOWN = 12  # entries listed per kind of change
INSTALLED_DIR = "rkmon_installed"
YAML_ON = "rkmon: YAML tiers on"
# What a configure log says when the YAML tiers could not be applied: the tiers are then unknown, not absent.
YAML_FAILED = re.compile(
    r"Failed to parse test categories YAML[^\n]*|Python3 not found, cannot parse[^\n]*|"
    r"Test category parser script not found[^\n]*"
)
# Where the install puts the source's tests (installed path prefix -> source prefix, relative to the rocKE tree).
INSTALL_MAP = (
    ("tests/library/", "library/"),
    ("tests/", "platform/tests/"),
    ("rocke/", "platform/python/rocke/"),
)
TEST_FILE = re.compile(r"(^|/)test_[^/]*\.py$")
# The provider wiring SHIM copies (hip-kernel-provider/CMakeLists.txt, src/CMakeLists.txt).
WIRING = {
    "CMakeLists.txt": 'set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "hip_kernel_provider")',
    "src/CMakeLists.txt": '"${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"',
}

SHIM = r"""# rkmon (tools/ctest_labels.py): rocKE configured as the provider configures it, to list its CTest entries.
cmake_minimum_required(VERSION 3.16)
project(rkmon_ctest_labels LANGUAGES NONE)
set(ROCM_LIBRARIES_ROOT "${RKMON_ROOT}")
set(_provider "${RKMON_PROVIDER}")
# hip-kernel-provider/CMakeLists.txt, "YAML-based CTest categorization detection": rocKE's YAML and shared/ctest.
set(ROCKE_ENGINE_CTEST_CATEGORIES_YAML "${_provider}/ROCKE_ENGINE_test_categories_external.yaml")
set(HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML "${_provider}/HIP_MLOPS_ENGINE_test_categories_external.yaml")
set(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED FALSE)
if(EXISTS "${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}"
   AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  set(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED TRUE)
  message(STATUS "rkmon: YAML tiers on")
endif()
# The provider's HIPKERNELPROVIDER_ENABLE_ROCKE block.
set(ROCKE_INSTALL_PROVIDER_CTEST_SUBDIR "hip_kernel_provider")
set(ROCKE_INSTALL_STANDALONE_CTEST OFF)
set(ROCKE_INSTALL_PYTHON_TESTS ON)
set(ROCKE_INSTALL_PLATFORM_PYTEST ON)
enable_testing()
add_subdirectory("${_provider}/rocke" "${CMAKE_BINARY_DIR}/rocke")
# src/CMakeLists.txt: rocKE's staged installed entries in the provider's file, then the YAMLs' labels by test name.
get_property(_staging GLOBAL PROPERTY EXTERNAL_TEST_INSTALL_STAGING_hip_kernel_provider)
set(_installed "${CMAKE_BINARY_DIR}/rkmon_installed/CTestTestfile.cmake")
file(WRITE "${_installed}" "# rkmon: the provider's installed rocKE entries\n${_staging}")
if(HIPKERNELPROVIDER_YAML_CATEGORIZATION_ENABLED AND COMMAND apply_ctest_category_labels)
  if(EXISTS "${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}")
    apply_ctest_category_labels("${HIPKERNELPROVIDER_CTEST_CATEGORIES_YAML}" "${_installed}")
  endif()
  apply_ctest_category_labels("${ROCKE_ENGINE_CTEST_CATEGORIES_YAML}" "${_installed}")
endif()
"""


# --- one tree ---------------------------------------------------------------------------------------------------------


def tree_paths(repo: str, rev: str, K: str) -> list[str]:
    """What a tree's configure reads: the rocKE tree, the provider's CMakeLists and top-level YAMLs, shared/ctest."""
    provider = K.rsplit("/", 1)[0]
    top = gitops.git(
        repo,
        "-c",
        "core.quotepath=false",
        "ls-tree",
        "--name-only",
        "-z",
        rev,
        "--",
        provider + "/",
        lazy=False,
    ).split("\0")
    yamls = sorted(p for p in top if p.endswith((".yaml", ".yml")))
    return prtree.present(
        repo,
        rev,
        [
            K,
            f"{provider}/CMakeLists.txt",
            f"{provider}/src/CMakeLists.txt",
            "shared/ctest",
        ]
        + yamls,
    )


def wiring_notes(provider_dir: pathlib.Path, tag: str) -> list[str]:
    """The provider files that no longer wire rocKE as SHIM copies: the tiers there may not be CI's."""
    out = []
    for name, text in WIRING.items():
        f = provider_dir / name
        if f.exists() and text not in f.read_text(errors="replace"):
            out.append(
                f"the {tag}'s provider {name} lacks `{text}`: its rocKE wiring may differ from this "
                "emulation's, and the tiers from CI's"
            )
    return out


def cmake_error(text: str) -> str:
    """The first CMake error of a configure log, with its place and first message line."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("CMake Error"):
            msg = next((x.strip() for x in lines[i + 1 : i + 6] if x.strip()), "")
            return f"{line.strip()} {msg}".strip()[:240]
    return next((x.strip() for x in reversed(lines) if x.strip()), "no output")[:240]


def configure(
    shim: pathlib.Path,
    build: pathlib.Path,
    root: pathlib.Path,
    provider: str,
    python: str,
    timeout: float,
    log: pathlib.Path,
) -> str | None:
    """Configure one tree through SHIM; None when it configures, else why not."""
    cmd = [
        shutil.which("cmake") or "cmake",
        "-S",
        str(shim),
        "-B",
        str(build),
        f"-DRKMON_ROOT={root}",
        f"-DRKMON_PROVIDER={root / provider}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF",
        f"-DPython3_EXECUTABLE={python}",
    ]
    if shutil.which("ninja"):
        cmd += ["-G", "Ninja"]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        log.write_text(f"timed out after {timeout:.0f} s\n")
        return f"the configure took over {timeout:.0f} s"
    log.write_text(p.stdout + p.stderr)
    return None if p.returncode == 0 else cmake_error(p.stdout + p.stderr)


def _rel(path: str | None, root: pathlib.Path, K: str) -> str | None:
    if not path:
        return None
    for base in (f"{root}/{K}/", f"{root}/"):
        if path.startswith(base):
            return path[len(base) :]
    return path


def registration(graph: dict, node, root: pathlib.Path, K: str) -> dict | None:
    """An entry's add_test(): its file:line and the directory whose CMakeLists.txt registered it, relative to the rocKE
    tree (a file outside it, to the repository); None without a backtrace (the installed tree's entries).
    """
    files, nodes = graph.get("files") or [], graph.get("nodes") or []
    first, directory = None, None
    while isinstance(node, int) and 0 <= node < len(nodes):
        n = nodes[node]
        i = n.get("file", -1)
        f = files[i] if isinstance(i, int) and 0 <= i < len(files) else ""
        if first is None and n.get("line"):
            first = (f, n["line"])
        if directory is None and f.endswith("/CMakeLists.txt"):
            directory = f.rsplit("/", 1)[0]
        node = n.get("parent")
    if first is None:
        return None
    return {
        "file": _rel(first[0], root, K),
        "line": first[1],
        "dir": _rel(directory, root, K),
    }


def entries(ctest_json: str, root: pathlib.Path, K: str) -> dict[str, dict]:
    """name -> {"labels": sorted labels, "at": its registration} from `ctest --show-only=json-v1`. A label spelled
    with escaped semicolons ("ckc\\;python\\;quick", one label to ctest) counts as its parts: `-L` matches them
    alike."""
    try:
        data = json.loads(ctest_json)
    except ValueError as e:
        raise prtree.NotRun(f"ctest --show-only=json-v1 gave no JSON ({e})") from e
    graph = data.get("backtraceGraph") or {}
    out: dict[str, dict] = {}
    for t in data.get("tests") or []:
        props = {x.get("name"): x.get("value") for x in t.get("properties") or []}
        raw = props.get("LABELS") or []
        labels = sorted(
            {
                x
                for lab in ([raw] if isinstance(raw, str) else raw)
                for x in str(lab).split(";")
                if x
            }
        )
        cmd = [
            _rel(x, root, K) if str(x).startswith(f"{root}/") else x
            for x in t.get("command") or []
        ]
        out.setdefault(
            t.get("name") or "?",
            {
                "labels": labels,
                "at": registration(graph, t.get("backtrace"), root, K),
                "cmd": cmd,
                "cwd": _rel(props.get("WORKING_DIRECTORY"), root, K) or "",
            },
        )
    return out


def listing(
    where: pathlib.Path, root: pathlib.Path, K: str, timeout: float
) -> dict[str, dict]:
    try:
        p = subprocess.run(
            [shutil.which("ctest") or "ctest", "--show-only=json-v1"],
            cwd=where,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        raise prtree.NotRun(
            f"ctest --show-only=json-v1 in {where.name} took over {timeout:.0f} s"
        ) from e
    if p.returncode:
        raise prtree.NotRun(
            f"ctest --show-only=json-v1 in {where.name} failed: {p.stderr.strip()[:160]}"
        )
    return entries(p.stdout, root, K)


def yaml_python(given: str | None) -> str:
    """An interpreter that imports yaml (the shared YAML parser needs PyYAML): the given one, else this one or
    /usr/bin/python3; the given one (or this one) when none does, and the configure log then says so.
    """
    for py in ([given] if given else []) + [sys.executable, "/usr/bin/python3"]:
        try:
            if (
                py
                and subprocess.run(
                    [py, "-c", "import yaml"], capture_output=True, timeout=60
                ).returncode
                == 0
            ):
                return py
        except (OSError, subprocess.SubprocessError):
            continue
    return given or sys.executable


# --- base against head ------------------------------------------------------------------------------------------------


def tiers(labels) -> list[str]:
    return [t for t in TIERS if t in set(labels or ())]


def compare(base: dict, head: dict) -> dict:
    """Each tree's counts, added, removed and relabelled entries, and the leads (module docstring)."""
    out: dict = {"trees": {}, "leads": []}
    for tree in TREES:
        b, h = base[tree], head[tree]
        relabelled = []
        for n in sorted(set(b) & set(h)):
            lost = sorted(set(b[n]["labels"]) - set(h[n]["labels"]))
            gained = sorted(set(h[n]["labels"]) - set(b[n]["labels"]))
            if lost or gained:
                relabelled.append({"name": n, "lost": lost, "gained": gained})
        out["trees"][tree] = {
            "base": len(b),
            "head": len(h),
            "added": sorted(set(h) - set(b)),
            "removed": sorted(set(b) - set(h)),
            "relabelled": relabelled,
        }
    for tree in TREES:
        other = TREES[1 - TREES.index(tree)]
        t, b, h = out["trees"][tree], base[tree], head[tree]
        for r in t["relabelled"]:
            lost = tiers(r["lost"])
            if lost:
                peer = head[other].get(r["name"])
                out["leads"].append(
                    {
                        "kind": "tiers",
                        "tree": tree,
                        "name": r["name"],
                        "tiers": lost,
                        "also": [x for x in r["lost"] if x not in TIERS],
                        "other": None if peer is None else tiers(peer["labels"]),
                        "was": b[r["name"]]["at"],
                        "now": h[r["name"]]["at"],
                    }
                )
        for n in t["removed"]:
            out["leads"].append(
                {"kind": "removed", "tree": tree, "name": n, "other": n in head[other]}
            )
        for n in t["added"]:
            if not tiers(h[n]["labels"]):
                out["leads"].append(
                    {
                        "kind": "untiered",
                        "tree": tree,
                        "name": n,
                        "labels": h[n]["labels"],
                        "now": h[n]["at"],
                    }
                )
    return out


# --- the test map -----------------------------------------------------------------------------------------------------


def source_path(arg: str, tree: str) -> str | None:
    """A pytest path argument of an entry as a path in the rocKE tree: the build tree's are (made relative to it when
    listed), the installed tree's map back through INSTALL_MAP; None for one outside them.
    """
    p = os.path.normpath(arg.split("::")[0])
    if p.startswith("..") or os.path.isabs(p):
        return None
    if tree == "build":
        return p
    for inst, src in INSTALL_MAP:
        if p == inst.rstrip("/") or p.startswith(inst):
            return src + p[len(inst) :] if p != inst.rstrip("/") else src.rstrip("/")
    return None


def _under(f: str, d: str) -> bool:
    return f == d or f.startswith(d.rstrip("/") + "/")


def runs_file(
    entry: dict, tree: str, f: str, suites: dict[str, set] | None
) -> str | None:
    """How an entry's pytest runs source file `f`: "direct", "in-tree" (through the in-tree runner it collects), or
    None."""
    cmd = entry.get("cmd") or []
    if len(cmd) < 3 or cmd[1:3] != ["-m", "pytest"]:
        return None
    paths, ignores, _globs, _opts = split_args(cmd[3:], "")
    src = [x for x in (source_path(a, tree) for a in paths) if x]
    ign = [x for x in (source_path(i, tree) for i in ignores) if x]

    def collected(p: str) -> bool:
        return any(_under(p, x) for x in src) and not any(_under(p, x) for x in ign)

    if collected(f):
        return "direct"
    runner = f"platform/tests/in_tree/{IN_TREE_RUNNER}"
    if (
        suites
        and collected(runner)
        and fnmatch.fnmatchcase(os.path.basename(f), IN_TREE_FILES)
    ):
        for pkg_dir, excluded in suites.items():
            if _under(f, pkg_dir) and os.path.basename(f) not in excluded:
                return "in-tree"
    return None


def read_suites(tree_root: pathlib.Path, K: str) -> dict[str, set] | None:
    """The in-tree runner's suites at a tree: package folder (in the rocKE tree) -> the file names it leaves out."""
    runner = tree_root / K / "platform" / "tests" / "in_tree" / IN_TREE_RUNNER
    try:
        packages, excluded = in_tree_suites(str(runner))
    except (OSError, SyntaxError, ValueError):
        return None
    return {"platform/python/" + p.replace(".", "/"): set(excluded) for p in packages}


def test_map(
    files: list[str], head: dict, suites: dict | None, lanes: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Each changed test file's entries, tiers and PR-CI lanes; and a lead for each that no lane runs (none without
    LANES: which lanes run is then unknown)."""
    rows, leads = [], []
    for f in files:
        ents = []
        for tree in TREES:
            for name, e in sorted(head.get(tree, {}).items()):
                how = runs_file(e, tree, f, suites)
                if how:
                    run_by = [
                        ln["jobs"]
                        for ln in lanes
                        if ln.get("tree", "installed") == tree
                        and ln["tier"] in e["labels"]
                        and not set(ln.get("exclude") or ()) & set(e["labels"])
                    ]
                    ents.append(
                        {
                            "tree": tree,
                            "entry": name,
                            "how": how,
                            "tiers": tiers(e["labels"]),
                            "excluded": sorted(
                                set(e["labels"])
                                & {x for ln in lanes for x in ln.get("exclude") or ()}
                            ),
                            "lanes": run_by,
                        }
                    )
        rows.append({"file": f, "entries": ents})
        if lanes and not any(x["lanes"] for x in ents):
            if not ents:
                why = "no CTest entry in either tree runs it"
            else:
                why = "; ".join(
                    f"{x['entry']} ({x['tree']}) has "
                    + (
                        f"the label {' '.join(x['excluded'])}, which the lanes leave "
                        "out"
                        if x["excluded"]
                        else f"tiers {' '.join(x['tiers']) or 'none'}"
                    )
                    for x in ents[:3]
                )
            leads.append({"kind": "unrun", "tree": "installed", "name": f, "why": why})
    return rows, leads


def map_lines(rows: list[dict], lanes: list[dict]) -> list[str]:
    if not rows:
        return []
    out = [
        "test map (test -> CTest entry -> tiers -> PR-CI lane): lanes "
        + (
            "; ".join(
                f"{ln['jobs']} run the {ln.get('tree', 'installed')} tree, -L {ln['tier']}"
                + (f" -LE {' '.join(ln['exclude'])}" if ln.get("exclude") else "")
                for ln in lanes
            )
            or "unknown (config ci_test_tiers names none)"
        )
    ]
    for r in rows[: SHOWN * 2]:
        if not r["entries"]:
            out.append(f"   {r['file']}: in no entry -> no lane")
            continue
        parts = [
            f"{x['tree']} {x['entry']}{' (in-tree runner)' if x['how'] == 'in-tree' else ''} "
            f"[{' '.join(x['tiers']) or 'no tier'}{'; ' + ' '.join(x['excluded']) if x['excluded'] else ''}] -> "
            f"{', '.join(x['lanes']) or 'no lane'}"
            for x in r["entries"][:4]
        ]
        more = (
            f" (+{len(r['entries']) - 4} more entries)" if len(r["entries"]) > 4 else ""
        )
        out.append(f"   {r['file']}: " + "; ".join(parts) + more)
    if len(rows) > SHOWN * 2:
        out.append(f"   +{len(rows) - SHOWN * 2} more test files (ctest_labels.json)")
    return out


def _at(r: dict | None) -> str:
    return f"{r['file']}:{r['line']}" if r else "?"


def describe(lead: dict) -> str:
    tree, other = lead["tree"], TREES[1 - TREES.index(lead["tree"])]
    if lead["kind"] == "removed":
        return (
            f"({tree} tree) {lead['name']} is an entry at the base but not at the head; the {other} tree "
            f"{'still has it' if lead['other'] else 'has none either'}"
        )
    if lead["kind"] == "unrun":
        return f"no PR-CI lane runs {lead['name']}, which the PR adds or changes: {lead['why']}"
    if lead["kind"] == "untiered":
        return (
            f"({tree} tree) {lead['name']} is new with no tier label ({' '.join(lead['labels']) or 'no label'}"
            f"{', add_test() at ' + _at(lead['now']) if lead.get('now') else ''}): `ctest -L <tier>` never runs it"
        )
    s = f"({tree} tree) {lead['name']} loses its tier labels {' '.join(lead['tiers'])}"
    if lead["also"]:
        s += f" (and {' '.join(lead['also'])})"
    was, now = lead.get("was"), lead.get("now")
    if was and now and (was["file"], was["dir"]) != (now["file"], now["dir"]):
        s += f"; its add_test() moved from {_at(was)} to {_at(now)}"
        if was["dir"] != now["dir"]:
            s += f" (directory {was['dir']} -> {now['dir']})"
    if lead["other"] is None:
        s += f"; it is not in the {other} tree"
    else:
        kept = [t for t in lead["tiers"] if t in lead["other"]]
        s += (
            f"; the {other} tree keeps {' '.join(kept)}"
            if kept
            else f"; the {other} tree lacks them too"
        )
    return s


def summary(leads: list[dict]) -> str:
    """The status line's account of the leads: the first kind, its entries grouped."""
    first = leads[0]
    same = [
        x
        for x in leads
        if (x["kind"], x["tree"], x.get("tiers"))
        == (first["kind"], first["tree"], first.get("tiers"))
    ]
    names = ", ".join(x["name"] for x in same[:3]) + (
        f" +{len(same) - 3}" if len(same) > 3 else ""
    )
    if first["kind"] == "tiers":
        moved = any(
            x.get("was") and x.get("now") and x["was"]["file"] != x["now"]["file"]
            for x in same
        )
        kept = all(x["other"] and set(x["tiers"]) <= set(x["other"]) for x in same)
        other = TREES[1 - TREES.index(first["tree"])]
        why = (["add_test moved"] if moved else []) + [
            f"the {other} tree keeps them" if kept else f"not kept in the {other} tree"
        ]
        what = f"{names} lose {' '.join(first['tiers'])} in the {first['tree']} tree ({'; '.join(why)})"
    elif first["kind"] == "removed":
        what = f"{names} left the {first['tree']} tree"
    elif first["kind"] == "unrun":
        what = f"no PR-CI lane runs {names}"
    else:
        what = f"{names} new in the {first['tree']} tree with no tier"
    rest = len(leads) - len(same)
    return f"{len(leads)} lead{'s' if len(leads) > 1 else ''}: {what}" + (
        f"; +{rest} more" if rest else ""
    )


def verdict(res: dict) -> tuple[str, str]:
    if res.get("error"):
        return "skip", f"NOT RUN: {res['error'][:180]}"
    if res.get("head_error"):
        return (
            "fail",
            f"the head does not configure as the provider configures rocKE: {res['head_error'][:140]}",
        )
    if res["leads"]:
        return "fail", summary(res["leads"])
    t = res["trees"]
    return "pass", (
        f"no entry loses a tier, leaves a tree or comes without one (build {t['build']['base']} -> "
        f"{t['build']['head']}, installed {t['installed']['base']} -> {t['installed']['head']} entries)"
    )


def render(res: dict) -> list[str]:
    if res.get("error"):
        return [f"NOT RUN: {res['error']}", " ".join(("LABELS",) + verdict(res))]
    yaml = res.get("yaml") or {}
    tiers_on = (
        "YAML tiers on"
        if all(yaml.values())
        else "YAML tiers "
        + ", ".join(f"{'on' if v else 'off'} at the {k}" for k, v in yaml.items())
    )
    out = [
        f"CTest entries at the merge base {res['mb'][:10]} and the head {res['head'][:10]}, rocKE configured (not "
        f"built) as hip-kernel-provider configures it: {tiers_on}; GPU reference entries left out (their DVC "
        f"bundles are not here); {res.get('secs', 0)} s"
    ]
    out += [f"!! {n}" for n in res.get("notes") or []]
    if res.get("head_error"):
        out.append(
            f"LEAD the head does not configure as the provider configures rocKE, and the merge base does: "
            f"{res['head_error']}"
        )
        out.append(" ".join(("LABELS",) + verdict(res)))
        return out
    for tree in TREES:
        t = res["trees"][tree]
        out.append(
            f"{tree} tree: {t['base']} entries at the base, {t['head']} at the head: {len(t['added'])} added, "
            f"{len(t['removed'])} removed, {len(t['relabelled'])} relabelled"
        )
    out += [f"LEAD {describe(x)}" for x in res["leads"][: 2 * SHOWN]]
    if len(res["leads"]) > 2 * SHOWN:
        out.append(f"LEAD +{len(res['leads']) - 2 * SHOWN} more (ctest_labels.json)")
    listed = {(x["tree"], x["name"]) for x in res["leads"]}
    for tree in TREES:
        t = res["trees"][tree]
        added = [n for n in t["added"] if (tree, n) not in listed]
        if added:
            out.append(
                f"added ({tree} tree): {' '.join(added[:SHOWN])}"
                + (f" +{len(added) - SHOWN} more" if len(added) > SHOWN else "")
            )
        other = [r for r in t["relabelled"] if (tree, r["name"]) not in listed]
        for r in other[:SHOWN]:
            out.append(
                f"relabelled ({tree} tree): {r['name']}: "
                + " ".join(
                    [f"-{x}" for x in r["lost"]] + [f"+{x}" for x in r["gained"]]
                )
            )
        if len(other) > SHOWN:
            out.append(
                f"relabelled ({tree} tree): +{len(other) - SHOWN} more (ctest_labels.json)"
            )
    out += map_lines(res.get("map") or [], res.get("lanes") or [])
    out.append(" ".join(("LABELS",) + verdict(res)))
    return out


# --- the check --------------------------------------------------------------------------------------------------------


def changed_tests(repo: str, mb: str, head: str, K: str) -> list[str]:
    """The test files the PR adds or changes that the head has, relative to the rocKE tree."""
    names = gitops.git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=d",
        mb,
        head,
        "--",
        K + "/",
        lazy=False,
    ).split()
    return sorted(f[len(K) + 1 :] for f in names if TEST_FILE.search(f))


def needs(repo: str, head: str, K: str, base: str = prtree.DEVELOP) -> list[str]:
    """The blobs of the clone the check reads (`--fetch`): what it writes of the merge base's tree and the head's."""
    mb = prtree.merge_base(repo, base, head)
    return [
        b
        for rev in (mb, head)
        for b in prtree.blobs(repo, rev, tree_paths(repo, rev, K))
    ]


def run(
    repo: str,
    head: str,
    out: pathlib.Path,
    K: str,
    base: str = prtree.DEVELOP,
    python: str | None = None,
    budget: float = BUDGET_S,
    lanes: list[dict] | None = None,
) -> dict:
    t0 = time.monotonic()

    def left() -> float:
        return budget - (time.monotonic() - t0)

    res: dict = {
        "head": head,
        "error": None,
        "notes": [],
        "yaml": {},
        "lanes": lanes or [],
    }
    suites = None
    out.mkdir(parents=True, exist_ok=True)
    work = None
    try:
        res["mb"] = mb = prtree.merge_base(repo, base, head)
        if not (shutil.which("cmake") and shutil.which("ctest")):
            raise prtree.NotRun("no cmake or ctest on PATH")
        res["python"] = py = yaml_python(python)
        work = pathlib.Path(
            tempfile.mkdtemp(
                prefix="ctest_labels_", dir=os.environ.get("TMPDIR") or None
            )
        ).resolve()
        shim = work / "shim"
        shim.mkdir()
        (shim / "CMakeLists.txt").write_text(SHIM)
        provider = K.rsplit("/", 1)[0]
        got: dict[str, dict] = {}
        for tag, rev in (("base", mb), ("head", head)):
            if left() < 15:
                raise prtree.NotRun(
                    f"the time budget ({budget:.0f} s) ran out before the {tag}'s configure"
                )
            root = work / tag
            paths = tree_paths(repo, rev, K)
            if K not in paths:
                raise prtree.NotRun(f"the {tag} {rev[:10]} has no {K}")
            prtree.extract(repo, rev, paths, root, timeout=max(30.0, left()))
            res["notes"] += wiring_notes(root / provider, tag)
            log = out / f"configure_{tag}.log"
            err = configure(
                shim,
                work / f"build_{tag}",
                root,
                provider,
                py,
                max(10.0, min(left(), 300.0)),
                log,
            )
            text = log.read_text(errors="replace")
            failed = YAML_FAILED.search(text)
            if failed and err is None:
                raise prtree.NotRun(
                    f"the YAML tiers could not be applied at the {tag} ({failed.group().strip()}): "
                    "the tiers are unknown"
                )
            res["yaml"][tag] = YAML_ON in text
            if tag == "head":
                suites = read_suites(root, K)
            got[tag] = {"error": err}
            if err is None:
                for tree, where in (
                    ("build", work / f"build_{tag}"),
                    ("installed", work / f"build_{tag}" / INSTALLED_DIR),
                ):
                    got[tag][tree] = listing(
                        where, root, K, max(10.0, min(left(), 120.0))
                    )
                (out / f"{tag}_entries.json").write_text(
                    json.dumps({t: got[tag][t] for t in TREES}, indent=1)
                )
        if got["base"]["error"]:
            raise prtree.NotRun(
                f"the merge base does not configure as the provider configures rocKE: "
                f"{got['base']['error']}"
            )
        if got["head"]["error"]:
            res["head_error"] = got["head"]["error"]
        else:
            res.update(compare(got["base"], got["head"]))
            res["map"], unrun = test_map(
                changed_tests(repo, mb, head, K), got["head"], suites, res["lanes"]
            )
            res["leads"] += unrun
    except (
        prtree.NotRun,
        RuntimeError,
        OSError,
        subprocess.SubprocessError,
        gitops.GitError,
    ) as exc:
        res["error"] = str(exc)[:300]
    finally:
        if work:
            shutil.rmtree(work, ignore_errors=True)
    res.setdefault("leads", [])
    res["secs"] = round(time.monotonic() - t0, 1)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument(
        "--out",
        default=None,
        help="the configure logs, each tree's entries and ctest_labels.json",
    )
    ap.add_argument("--base", default=prtree.DEVELOP)
    ap.add_argument("--rocke-path", default="", type=prtree.folder_argument)
    ap.add_argument(
        "--python", default=None, help="an interpreter with PyYAML, for the YAML parser"
    )
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    ap.add_argument(
        "--ci-tiers",
        default="",
        help="config ci_test_tiers as JSON (default: none, and the test map stops at the tiers)",
    )
    ap.add_argument("--fetch", action="store_true", help=prtree.FETCH_HELP)
    a = ap.parse_args(argv)
    if a.fetch:
        return prtree.fetch_main(
            a.repo, lambda: needs(a.repo, a.head, a.rocke_path, a.base)
        )
    if not a.out:
        ap.error("the following arguments are required: --out")
    out = pathlib.Path(a.out)
    lanes = json.loads(a.ci_tiers) if a.ci_tiers.strip() else None
    res = run(a.repo, a.head, out, a.rocke_path, a.base, a.python, a.budget, lanes)
    (out / "ctest_labels.json").write_text(json.dumps(res, indent=1))
    print("\n".join(render(res)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
