#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The Windows simulation: `rk winsim` and the checks' "Windows simulation" section (a lead).

Windows CI lanes break tests and code that Linux passes, for reasons a Linux container can bring to light:
1. **The tests, run as on Windows** (`run`): each test file runs twice, plainly and with pytest_plugins/rkmon_winsim.py
   (`-p rkmon_winsim`: text opens in the Windows code page, CRLF text writes, a NamedTemporaryFile that cannot be
   opened again while open; `os` adds a Windows `os.name`/`os.sep`/`os.linesep`). A test that passes plainly and fails
   under the simulation is a lead, named with the simulation that breaks it (each such test runs once more under each
   simulation alone). A test that fails plainly is not this section's (the tests section has it).
2. **The C/C++ the PR adds, against windows.h's macros** (`scan`): an added identifier that windows.h (or a header it
   pulls in) defines as a macro breaks the Windows build: `far`, `near`, `IN`, `OUT`, `OPTIONAL`, `ERROR`, `DELETE`,
   `CONST`, `VOID`, `interface`, `small` and the A/W function names as a whole word; `min(` and `max(` (function-like:
   `std::max(a, b)` and `numeric_limits<unsigned int>::max()` break, `(std::max)(a, b)` does not) unless the file or
   the tree's CMake defines NOMINMAX. Comments, strings and `#undef`/`#ifdef` lines aside. A name is a lead only in a
   file that sees the macros: one that includes a Windows header (windows.h, windef.h, winsock2.h, rpcndr.h, ...)
   itself or through the tree's own headers (--tree, the rocKE tree at the head or the merge); the others are counted,
   no lead. Without --tree every name is a lead. Git and the tree's text only.

Usage:
  winsim.py run --tree K_DIR [--python PY] [--only SIMS] [--budget S] [--scan-report FILE] [TARGET...]
  winsim.py scan --repo R --head SHA [--base REF] [--rocke-path K] [--tree K_DIR]
TARGETs are test files or node ids relative to the rocKE tree K_DIR. `run` runs PR code (the checks start it through
`prc`); `scan` reads git and the tree only, and its report (`SCANNED`, `HIT` and `UNREACHED` lines) is folded into
a run's output with --scan-report. `rk winsim` (rkmon_lib/lanes.py) writes the tree to a scratch folder, scans and
runs. The last line of `run`: "WINSIM pass|fail|skip <detail>"; its exit: 0 nothing found, 1 a lead, 3 NOT RUN.
"""

import argparse
import collections
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "pytest_plugins"))
from rkmon_lib import gitops, prtree  # noqa: E402
import rkmon_winsim  # noqa: E402  (its names: importing it simulates nothing, pytest_configure does)

PLUGINS = ROOT / "tools" / "pytest_plugins"
BUDGET_S = 300
PER_RUN_S = 180
ATTRIBUTE_CAP = (
    20  # sim-only failures run again under each simulation alone, at most this many
)
RESULT = "rkmon-ci-result:"
SIM_LINE = "rkmon-winsim:"

# --- the macro scan -----------------------------------------------------------------------------------------------
CPP = re.compile(r"\.(?:c|cc|cpp|cxx|c\+\+|h|hh|hpp|hxx|inl|ipp|tpp|hip|cu|cuh)$", re.I)
# Object-like macros of windows.h and the headers it includes (without WIN32_LEAN_AND_MEAN for the last group): an
# identifier of that name anywhere in code turns into the macro's text.
OBJECT_MACROS = {
    "far": "minwindef.h (empty)",
    "near": "minwindef.h (empty)",
    "FAR": "minwindef.h (empty)",
    "NEAR": "minwindef.h (empty)",
    "IN": "minwindef.h (empty)",
    "OUT": "minwindef.h (empty)",
    "OPTIONAL": "minwindef.h (empty)",
    "CONST": "minwindef.h (const)",
    "VOID": "winnt.h (void)",
    "DELETE": "winnt.h (an access-right number)",
    "ERROR": "wingdi.h (0)",
    "TRANSPARENT": "wingdi.h (a number)",
    "OPAQUE": "wingdi.h (a number)",
    "ABSOLUTE": "wingdi.h (a number)",
    "RELATIVE": "wingdi.h (a number)",
    "NO_ERROR": "winerror.h (0)",
    "IGNORE": "winbase.h (0)",
    "INFINITE": "winbase.h (a number)",
    "CALLBACK": "minwindef.h (__stdcall)",
    "WINAPI": "minwindef.h (__stdcall)",
    "APIENTRY": "minwindef.h (__stdcall)",
    "PASCAL": "minwindef.h (__stdcall)",
    "pascal": "minwindef.h (__stdcall)",
    "cdecl": "minwindef.h (empty or __cdecl)",
    "interface": "objbase.h (struct; without WIN32_LEAN_AND_MEAN)",
    "small": "rpcndr.h (char; without WIN32_LEAN_AND_MEAN)",
    "hyper": "rpcndr.h (__int64; without WIN32_LEAN_AND_MEAN)",
    # The A/W function names: each a macro for its ...A or ...W function.
    **{
        n: "winuser.h/winbase.h/wingdi.h (an A/W function name)"
        for n in (
            "CreateFile",
            "DeleteFile",
            "CopyFile",
            "MoveFile",
            "GetObject",
            "GetMessage",
            "SendMessage",
            "PostMessage",
            "LoadImage",
            "DrawText",
            "CreateWindow",
            "GetClassName",
            "GetCurrentTime",
            "CreateEvent",
            "CreateMutex",
            "GetEnvironmentVariable",
            "SetEnvironmentVariable",
            "LoadLibrary",
            "GetModuleFileName",
            "CreateDirectory",
            "RemoveDirectory",
            "FormatMessage",
            "GetUserName",
            "GetComputerName",
            "CreateProcess",
            "OutputDebugString",
            "Yield",
        )
    },
}
FUNCTION_MACROS = {
    "min": "minwindef.h (unless NOMINMAX)",
    "max": "minwindef.h (unless NOMINMAX)",
}
OBJECT_RX = re.compile(
    r"(?<![\w$])(" + "|".join(map(re.escape, OBJECT_MACROS)) + r")(?![\w$])"
)
FUNCTION_RX = re.compile(r"(?<![\w$])(min|max)\s*\(")
STRING = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
PREPROC_SKIP = re.compile(
    r"^\s*#\s*(?:undef|ifn?def|if|elif|pragma|include|error|warning)\b"
)
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _code(line: str, in_block: bool) -> tuple[str, bool]:
    """The line's code: strings, character literals and comments blanked; whether a block comment is still open."""
    out, i = [], 0
    line = STRING.sub(lambda m: " " * len(m.group(0)), line)
    while i < len(line):
        if in_block:
            j = line.find("*/", i)
            if j < 0:
                return "".join(out), True
            i, in_block = j + 2, False
            continue
        if line.startswith("//", i):
            break
        if line.startswith("/*", i):
            in_block, i = True, i + 2
            continue
        out.append(line[i])
        i += 1
    return "".join(out), in_block


def macro_hits(
    path: str, lines: list[tuple[int, str]], nominmax: bool = False
) -> list[str]:
    """`path:line: NAME (header): text` for each windows.h macro name the added lines use as code."""
    hits, block = [], False
    for no, text in lines:
        code, block = _code(text, block)
        if PREPROC_SKIP.match(code):
            continue
        names = [m.group(1) for m in OBJECT_RX.finditer(code)]
        if not nominmax:
            names += [m.group(1) for m in FUNCTION_RX.finditer(code)]
        for name in dict.fromkeys(names):
            where = OBJECT_MACROS.get(name) or FUNCTION_MACROS[name]
            hits.append(
                f"{path}:{no}: {name} is a windows.h macro ({where}): {text.strip()[:120]}"
            )
    return hits


def added_lines(diff: str) -> dict[str, list[tuple[int, str]]]:
    """Each file's added lines with their line numbers, from a `git diff -U0` text."""
    out: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    path, no = None, 0
    for ln in diff.splitlines():
        if ln.startswith("+++ "):
            path = ln[6:] if ln.startswith("+++ b/") else None
            continue
        m = HUNK.match(ln)
        if m:
            no = int(m.group(1))
            continue
        if path and ln.startswith("+"):
            out[path].append((no, ln[1:]))
            no += 1
    return out


def scan(repo: str, head: str, base: str, rocke: str) -> tuple[list[str], int]:
    """The macro hits in the C/C++ lines the PR adds under the rocKE tree, and how many C/C++ files it touches."""
    mb = prtree.merge_base(repo, base, head)
    diff = gitops.git(
        repo,
        "-c",
        "core.quotepath=false",
        "diff",
        "-U0",
        "--no-color",
        "--no-renames",
        mb,
        head,
        "--",
        rocke,
        timeout=600,
    )
    hits, files = [], 0
    for path, lines in sorted(added_lines(diff).items()):
        if not CPP.search(path):
            continue
        files += 1
        try:
            whole = gitops.git(repo, "show", f"{head}:{path}", timeout=120)
        except gitops.GitError:
            whole = ""
        hits += macro_hits(path, lines, nominmax="NOMINMAX" in whole)
    return hits, files


# The Windows headers that define those macros: a file that includes one, itself or through the tree's own headers,
# sees them; any other file does not (a standard or HIP header that pulls one in on Windows is not followed).
WIN_HEADERS = {
    "windows.h",
    "windef.h",
    "minwindef.h",
    "winbase.h",
    "wingdi.h",
    "winuser.h",
    "winnt.h",
    "winsock.h",
    "winsock2.h",
    "rpcndr.h",
    "objbase.h",
}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
NOMINMAX_DEF = re.compile(r"\bNOMINMAX\b")


def windows_reach(tree: pathlib.Path) -> tuple[set[str], bool]:
    """The C/C++ files of `tree` (paths relative to it) that include a Windows header, directly or through the
    tree's own headers; and whether the tree's CMake defines NOMINMAX for its builds."""
    includes, nominmax = {}, False
    for f in tree.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(tree).as_posix()
        if f.name == "CMakeLists.txt" or f.suffix == ".cmake":
            try:
                nominmax = nominmax or bool(
                    NOMINMAX_DEF.search(f.read_text(encoding="utf-8", errors="replace"))
                )
            except OSError:
                pass
        elif CPP.search(f.name):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            includes[rel] = [n.strip() for n in INCLUDE.findall(text)]
    reached = {
        rel
        for rel, names in includes.items()
        if any(n.rsplit("/", 1)[-1].lower() in WIN_HEADERS for n in names)
    }
    for _ in range(20):  # each round reaches one include deeper
        more = {
            rel
            for rel, names in includes.items()
            if rel not in reached
            and any(r == n or r.endswith("/" + n) for n in names for r in reached)
        }
        if not more:
            break
        reached |= more
    return reached, nominmax


def classify(
    hits: list[str], tree: pathlib.Path | None, rocke: str
) -> tuple[list[str], list[str]]:
    """The hits in files that see windows.h's macros (every hit when no tree was read), and the rest. min( and max( do
    not count where the tree's CMake defines NOMINMAX."""
    if tree is None:
        return hits, []
    reached, nominmax = windows_reach(tree)
    k = rocke.rstrip("/") + "/"
    seen, unseen = [], []
    for h in hits:
        path = h.split(":", 1)[0]
        rel = path[len(k) :] if path.startswith(k) else path
        name = h.split(": ", 1)[1].split(" ", 1)[0]
        if rel in reached and not (nominmax and name in FUNCTION_MACROS):
            seen.append(h)
        else:
            unseen.append(h)
    return seen, unseen


# --- the test runs ------------------------------------------------------------------------------------------------
def run_env(tree: pathlib.Path, sims: tuple[str, ...] | None, environ=None) -> dict:
    """The run's environment: the tree's Python and the plugins on PYTHONPATH, UTF-8 mode off (Windows has it off),
    and the simulations when `sims` is given."""
    env = {
        k: v
        for k, v in (os.environ if environ is None else environ).items()
        if k
        not in (
            "PYTHONPATH",
            "PYTEST_ADDOPTS",
            "PYTEST_PLUGINS",
            "RKMON_WINSIM",
            "PYTHONUTF8",
        )
    }
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(PLUGINS)]
            + [str(tree / p) for p in prtree.PY_ROOTS if (tree / p).is_dir()]
        ),
        PYTHONUTF8="0",
        PYTHONDONTWRITEBYTECODE="1",
        RKMON_CI_ABSENT="",
    )
    if sims is not None:
        env["RKMON_WINSIM"] = ",".join(sims)
    return env


def parse(text: str) -> dict:
    """{node id: passed|failed|skipped} from rkmon_ci.py's result lines, the collection errors, and whether the
    simulation said it was on."""
    out = {"results": {}, "errors": [], "sim": None}
    for ln in text.splitlines():
        parts = ln.split("\t")
        if parts[0] == RESULT and len(parts) >= 3:
            out["results"][parts[2]] = parts[1]
        elif parts[0] == SIM_LINE and len(parts) >= 2:
            out["sim"] = parts[1]
        elif ln.startswith("ERROR ") or "ERROR collecting" in ln:
            out["errors"].append(ln.strip()[:160])
    return out


def pytest_run(tree, targets, python, env, timeout, winsim: bool) -> dict:
    tree = tree.resolve()
    first = (tree / targets[0].split("::")[0]).resolve()
    ini = next(
        (
            d
            for d in first.parents
            if (d == tree or tree in d.parents) and (d / "pytest.ini").is_file()
        ),
        tree,
    )
    cmd = [
        python,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "rkmon_ci",
        "-p",
        "no:cacheprovider",
    ]
    if winsim:
        cmd += ["-p", "rkmon_winsim"]
    cmd += [
        os.path.relpath(tree / t.split("::")[0], ini)
        + ("::" + t.split("::", 1)[1] if "::" in t else "")
        for t in targets
    ]
    base = tempfile.mkdtemp(prefix="winsim_tmp_")  # the tests' tmp_path, removed after
    try:
        p = subprocess.run(
            cmd + ["--basetemp", os.path.join(base, "t")],
            cwd=ini,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"over {timeout:.0f} s"}
    except OSError as e:
        return {"error": f"pytest did not start ({e})"}
    finally:
        shutil.rmtree(base, ignore_errors=True)
    got = parse(p.stdout + p.stderr)
    if winsim and got["sim"] is None:
        got["error"] = "the simulation did not load (no rkmon-winsim line)"
    # Node ids relative to the rocKE tree, whatever folder pytest ran in.
    rel = os.path.relpath(ini, tree)
    if rel != ".":
        got["results"] = {f"{rel}/{k}": v for k, v in got["results"].items()}
    return got


def counts(results: dict) -> str:
    c = collections.Counter(results.values())
    return f"{c['passed']} pass, {c['failed']} fail, {c['skipped']} skip"


def run(
    tree: pathlib.Path,
    targets: list[str],
    python: str,
    sims: tuple[str, ...],
    budget: float,
) -> tuple[list, list, list]:
    """Per file: its line; the leads; the files NOT RUN."""
    t0, lines, leads, not_run = time.monotonic(), [], [], []
    files = list(dict.fromkeys(t.split("::")[0] for t in targets))
    for i, f in enumerate(files):
        mine = [t for t in targets if t.split("::")[0] == f]
        left = budget - (time.monotonic() - t0)
        if left < 10:
            lines.append(
                f"NOT RUN (the {budget:.0f} s time box ran out): {' '.join(files[i:])}"
            )
            not_run += files[i:]
            break
        plain = pytest_run(
            tree, mine, python, run_env(tree, None), min(PER_RUN_S, left / 2), False
        )
        sim = pytest_run(
            tree,
            mine,
            python,
            run_env(tree, sims),
            min(PER_RUN_S, max(10, left / 2)),
            True,
        )
        if not sim.get("error") and plain.get("results") and not sim["results"]:
            sim["error"] = "the simulated run reported no test: " + (
                (sim["errors"] or ["no result line"])[0]
            )
        if plain.get("error") or sim.get("error"):
            lines.append(f"{f}: NOT RUN ({plain.get('error') or sim.get('error')})")
            not_run.append(f)
            continue
        broke = [
            t
            for t, o in plain["results"].items()
            if o == "passed" and sim["results"].get(t) == "failed"
        ]
        lines.append(
            f"{f}: plain {counts(plain['results'])}; Windows simulation ({','.join(sims)}) "
            f"{counts(sim['results'])}"
        )
        if sim["errors"] and not plain["errors"]:
            leads.append(
                f"{f} does not collect under the Windows simulation: {sim['errors'][0]}"
            )
        by = {t: [] for t in broke}
        if len(sims) > 1:
            for s in sims:
                left = budget - (time.monotonic() - t0)
                if left < 10 or not broke:
                    break
                one = pytest_run(
                    tree,
                    broke[:ATTRIBUTE_CAP],
                    python,
                    run_env(tree, (s,)),
                    min(PER_RUN_S, left),
                    True,
                )
                for t in broke:
                    if (one.get("results") or {}).get(t) == "failed":
                        by[t].append(s)
        for t in broke:
            leads.append(
                f"{t} passes plainly and fails under the Windows simulation ({','.join(by[t] or sims)})"
            )
    return lines, leads, not_run


def report(
    lines, leads, not_run, hits, scanned: int | None, unseen: list | None = None
) -> int:
    for ln in lines:
        print(ln)
    unseen = unseen or []
    if scanned is not None:
        print(
            f"windows.h macro names in the C/C++ lines the PR adds: {len(hits)} in {scanned} C/C++ files"
            + (
                f"; {len(unseen)} more in files that include no Windows header (no lead), e.g. {unseen[0][:120]}"
                if unseen
                else ""
            )
        )
    for h in hits:
        print(f"LEAD {h}")
    for x in leads:
        print(f"LEAD {x}")
    n = len(leads) + len(hits)
    detail = (
        f"{len(leads)} test lead{'s' if len(leads) != 1 else ''}, "
        f"{len(hits)} macro name{'s' if len(hits) != 1 else ''}"
    )
    if not_run:
        detail += f"; NOT RUN: {' '.join(not_run[:4])}" + (
            f" +{len(not_run) - 4}" if len(not_run) > 4 else ""
        )
    ran = any(": plain " in ln for ln in lines)
    if n:
        print(f"WINSIM fail {detail}: {(hits + leads)[0][:140]}")
        return 1
    if not ran and scanned is None:
        print(f"WINSIM skip nothing run or scanned; {detail}")
        return 3
    if not_run:
        print(f"WINSIM skip {detail}")
        return 3
    print(f"WINSIM pass {detail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("run")
    pa.add_argument("--tree", required=True)
    pa.add_argument("--scan-report", default=None, help="a `scan` report to fold in")
    pa.add_argument("--python", default=sys.executable)
    pa.add_argument(
        "--only",
        default=None,
        help="the simulations, comma-separated (text,crlf,tempfile,os)",
    )
    pa.add_argument("--budget", type=float, default=BUDGET_S)
    pa.add_argument("targets", nargs="*", metavar="TARGET")
    ps = sub.add_parser("scan")
    ps.add_argument("--repo", required=True)
    ps.add_argument("--head", required=True)
    ps.add_argument("--base", default=prtree.DEVELOP)
    ps.add_argument("--rocke-path", default="", type=prtree.folder_argument)
    ps.add_argument(
        "--tree",
        default=None,
        help="the rocKE tree at the head (or the merge): which files see a Windows header",
    )
    a = ap.parse_args(argv)

    if a.cmd == "scan":
        rocke = a.rocke_path.rstrip("/")
        try:
            hits, files = scan(a.repo, a.head, a.base, rocke)
        except (prtree.NotRun, gitops.GitError, subprocess.SubprocessError) as exc:
            print(f"SCAN NOT RUN: {str(exc)[:200]}")
            return 3
        tree = pathlib.Path(a.tree) if a.tree and os.path.isdir(a.tree) else None
        seen, unseen = classify(hits, tree, rocke)
        print(f"SCANNED {files}")
        if tree is None:
            print("REACH not read (no tree): every name counts")
        for h in seen:
            print(f"HIT {h}")
        for h in unseen:
            print(f"UNREACHED {h}")
        return 0

    sims = rkmon_winsim.chosen(a.only)
    if not sims:
        print(
            f"WINSIM skip no simulation named (the simulations: {','.join(rkmon_winsim.KNOWN)})"
        )
        return 3
    python = shutil.which(a.python) or a.python
    tree = pathlib.Path(a.tree)
    hits, scanned, unseen = [], None, []
    if a.scan_report:
        try:
            text = pathlib.Path(a.scan_report).read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            text = ""
        m = re.search(r"^SCANNED (\d+)$", text, re.M)
        scanned = int(m.group(1)) if m else None
        hits = [ln[4:] for ln in text.splitlines() if ln.startswith("HIT ")]
        unseen = [ln[10:] for ln in text.splitlines() if ln.startswith("UNREACHED ")]
        if m is None:
            print(
                f"macro scan: NOT RUN ({(text.strip().splitlines() or ['no report'])[-1][:160]})"
            )
    targets = [t for t in a.targets if (tree / t.split("::")[0]).is_file()]
    for t in a.targets:
        if t not in targets:
            print(f"{t}: not run (the tree has no such file)")
    lines, leads, not_run = (
        run(tree, targets, python, sims, a.budget) if targets else ([], [], [])
    )
    return report(lines, leads, not_run, hits, scanned, unseen)


if __name__ == "__main__":
    sys.exit(main())
