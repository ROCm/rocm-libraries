#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The checks' "CI-condition run": each test file the PR adds or changes, run as CI's test environment would run it.

The checks' own test run has what CI lacks: the built engines on the path, ROCm's tools, maybe more modules. A test
that only runs there reads green here and runs nothing in CI (a module-level `importorskip("torch")` skips every test
of the file; a both-engine case skips without `rocke_engine`; a tool found under a fixed /opt/rocm path is gone). So
each file runs once more:
- the modules of config `ci_absent_modules` (--absent) unimportable (pytest_plugins/rkmon_ci.py), such as torch and
  the built engines, and no build folder on PYTHONPATH (the tree's platform/python and library only);
- ROCm's tools hidden (pytest_plugins/rkmon_norocm.py): PATH without the folders that hold them, ROCM_PATH and its
  kin unset, and /opt/rocm and those folders reading as absent to the test's own lookups (a program the test runs by
  an absolute path still starts: that much is not hidden).
Per file: how many of its tests run, skip and fail there. Leads (the section feeds no gate): a file with tests of
which none runs ("0 of n run": CI runs nothing of it), a test that fails only here, and a skip in a tier or contract
test file (its name says tier or contract), which reads NOT RUN, never a pass.

Usage: python3 -I ci_condition.py --tree K_DIR [--python PY] [--absent M,M] [--budget S] FILE...
FILEs are relative to the rocKE tree K_DIR. The last line: "CICOND pass|fail|skip <detail>".
"""

import argparse
import collections
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True

PLUGINS = ROOT / "tools" / "pytest_plugins"
BUDGET_S = 600
PER_FILE_S = 240
# Tools CI's test image does not have on PATH: a folder of PATH that holds one is dropped.
ROCM_TOOLS = (
    "hipcc",
    "llvm-objdump",
    "ld.lld",
    "llc",
    "amdclang",
    "amdclang++",
    "hipconfig",
    "rocminfo",
)
TIER_OR_CONTRACT = re.compile(r"tier|contract", re.I)
RESULT = "rkmon-ci-result:"


def ci_path(path: str) -> tuple[str, list[str]]:
    """PATH without the folders that hold a ROCm tool, and the folders dropped."""
    keep, dropped = [], []
    for d in path.split(os.pathsep):
        if d and any(os.path.exists(os.path.join(d, t)) for t in ROCM_TOOLS):
            dropped.append(d)
        elif d:
            keep.append(d)
    return os.pathsep.join(keep), dropped


def ci_env(
    tree: pathlib.Path, absent: set[str], environ=None
) -> tuple[dict, list[str]]:
    env = {
        k: v
        for k, v in (os.environ if environ is None else environ).items()
        if k
        not in (
            "PYTHONPATH",
            "PYTEST_ADDOPTS",
            "PYTEST_PLUGINS",
            "ROCM_PATH",
            "ROCM_HOME",
            "HIP_PATH",
        )
        and not k.startswith("ROCKE_")
    }
    hidden = [d for d in ("/opt/rocm",) if os.path.exists(d)]
    for k in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        v = (os.environ if environ is None else environ).get(k)
        if v:
            hidden.append(v)
    env["PATH"], dropped = ci_path(env.get("PATH", ""))
    env.update(
        PYTHONPATH=os.pathsep.join(
            [str(PLUGINS)]
            + [
                str(tree / p)
                for p in ("platform/python", "library")
                if (tree / p).is_dir()
            ]
        ),
        RKMON_CI_ABSENT=",".join(sorted(absent)),
        RKMON_NOROCM_HIDE=os.pathsep.join(hidden + dropped),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env, dropped


def parse(text: str) -> dict:
    """The run's outcomes from rkmon_ci.py's result lines: counts, the skip reasons, the failing ids."""
    out = {"passed": 0, "failed": [], "skipped": collections.Counter(), "errors": []}
    for ln in text.splitlines():
        parts = ln.split("\t")
        if parts[0] != RESULT or len(parts) < 3:
            if ln.startswith("ERROR ") or "ERROR collecting" in ln:
                out["errors"].append(ln.strip()[:160])
            continue
        outcome, tid = parts[1], parts[2]
        if outcome == "passed":
            out["passed"] += 1
        elif outcome == "failed":
            out["failed"].append(tid)
        elif outcome == "skipped":
            out["skipped"][(parts[3] if len(parts) > 3 else "") or "skipped"] += 1
    return out


def run_file(
    tree: pathlib.Path, f: str, python: str, env: dict, timeout: float
) -> dict:
    path = tree / f
    ini = next((d for d in path.parents if (d / "pytest.ini").is_file()), tree)
    cmd = [
        python,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "rkmon_ci",
        "-p",
        "rkmon_norocm",
        "-p",
        "no:cacheprovider",
        os.path.relpath(path, ini),
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
        return {"error": f"over {timeout:.0f} s"}
    except OSError as e:
        return {"error": f"pytest did not start ({e})"}
    got = parse(p.stdout + p.stderr)
    got["rc"] = p.returncode
    return got


def describe(f: str, got: dict) -> tuple[str, list[str], bool]:
    """A file's line, its leads, and whether a skip there reads NOT RUN (a tier or contract test)."""
    if got.get("error"):
        return f"{f}: NOT RUN ({got['error']})", [], False
    n_skip = sum(got["skipped"].values())
    n = got["passed"] + len(got["failed"]) + n_skip
    reasons = "; ".join(f"{c}× {r[:90]}" for r, c in got["skipped"].most_common(2))
    line = (
        f"{f}: {got['passed'] + len(got['failed'])} of {n} run"
        + (f", {n_skip} skip ({reasons})" if n_skip else "")
        + (f", {len(got['failed'])} fail" if got["failed"] else "")
        + (f", collection error: {got['errors'][0]}" if got["errors"] else "")
    )
    leads = []
    if got["errors"]:
        leads.append(f"{f} does not collect in CI's condition: {got['errors'][0]}")
    elif n and not (got["passed"] + len(got["failed"])):
        leads.append(f"0 of {n} run in CI's condition: {f} ({reasons})")
    if got["failed"]:
        leads.append(
            f"{f}: {len(got['failed'])} fail in CI's condition (no torch, no engines, no ROCm tools): "
            + ", ".join(got["failed"][:3])
        )
    tier = bool(n_skip and TIER_OR_CONTRACT.search(os.path.basename(f)))
    return line, leads, tier


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", metavar="FILE")
    ap.add_argument("--tree", required=True, help="the rocKE tree the files are in")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument(
        "--absent",
        default=None,
        help="config ci_absent_modules, comma-separated (default: none)",
    )
    ap.add_argument("--budget", type=float, default=BUDGET_S)
    a = ap.parse_args(argv)
    tree = pathlib.Path(a.tree)
    absent = {m.strip() for m in (a.absent or "").split(",") if m.strip()}
    files = [f for f in a.files if (tree / f).is_file()]
    if not files:
        print("CICOND skip no test file to run")
        return 0
    python = (
        shutil.which(a.python) or a.python
    )  # found on the full PATH, before ROCm's folders leave it
    env, dropped = ci_env(tree, absent)
    print(
        f"CI's condition: {', '.join(sorted(absent)) or 'no module'} unimportable; no build folder on PYTHONPATH; "
        f"ROCm tools hidden ({len(dropped)} PATH folders dropped, ROCM_PATH unset, /opt/rocm reads as absent)"
    )
    t0, leads, not_run, ran, total = time.monotonic(), [], [], 0, 0
    for i, f in enumerate(files):
        left = a.budget - (time.monotonic() - t0)
        if left < 10:
            print(
                f"NOT RUN (the {a.budget:.0f} s time box ran out): {' '.join(files[i:])}"
            )
            not_run += files[i:]
            break
        got = run_file(tree, f, python, env, min(PER_FILE_S, left))
        line, more, tier = describe(f, got)
        print(line)
        leads += more
        if got.get("error") or tier:
            not_run.append(f)
        if not got.get("error"):
            ran += got["passed"] + len(got["failed"])
            total += got["passed"] + len(got["failed"]) + sum(got["skipped"].values())
    for x in leads:
        print(f"LEAD {x}")
    detail = f"{ran} of {total} tests run in CI's condition over {len(files)} files"
    if not_run:
        detail += f"; NOT RUN: {' '.join(not_run[:4])}" + (
            f" +{len(not_run) - 4}" if len(not_run) > 4 else ""
        )
    if leads:
        print(
            f"CICOND fail {len(leads)} lead{'s' if len(leads) > 1 else ''}: {leads[0][:140]}; {detail}"
        )
    elif not ran or not_run:
        print(f"CICOND skip {detail}")
    else:
        print(f"CICOND pass {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
