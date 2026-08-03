#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Cross-platform (Windows + Linux) CI/parent entrypoint for the rocKE engine.
# One command runs: (1) the relative-path contract guard, (2) the byte-identity
# gate, (3) the pytest suite, (4) the same suite again under ROCKE_BACKEND=both
# when the engine extension is importable, (5) ctest if a build dir exists. All
# paths are derived relative to this file so the rocke/platform/ tree is
# copy-able verbatim.
#
# Usage:
#   python rocke/platform/tests/run_all.py [--no-guard] [--no-gate] [--no-pytest]
#       [--no-both] [--only SUBSTR] [--build-root DIR]

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROCKE = Path(__file__).resolve().parents[1]  # tests -> rocKE
TESTS = ROCKE / "tests"
TOOLS = ROCKE / "tools"

# Same sys.path bootstrap as tests/conftest.py, so the runner can read shared
# constants (e.g. backend.CPP_UNPORTED_ARCHES) out of the package it is testing
# instead of restating them here and letting the two drift.
if str(ROCKE / "python") not in sys.path:
    sys.path.insert(0, str(ROCKE / "python"))

# Files that may reference an absolute repo path or a path that escapes rocke/platform/
# break the verbatim-copy contract. Enforce on code/build files only (docs are
# exempt). A clean run is required before the tree is dropped into another repo.
_GUARD_SUFFIXES = {".py", ".cmake", ".toml", ".ini", ".sh", ".cfg"}
_GUARD_NAMES = {"CMakeLists.txt"}
_GUARD_SKIP_DIRS = {".git", "__pycache__", "build", "dsl_docs", "examples"}
# Any ".venv*" dir is a local virtual environment (".venv", ".venv-torch", ...);
# keeping side venvs around for CI parity and local dev must not trip the guard.
_GUARD_SKIP_PREFIXES = (".venv",)
_FORBIDDEN = [
    re.compile(r"/workspace\b"),
    re.compile(r"rocm-libraries(?:-[a-z-]+)?/"),
    re.compile(r"projects/composablekernel"),
    re.compile(r"dnn-providers/"),
]


def relative_path_guard() -> int:
    """Fail if any code/build file under rocke/platform/ references an absolute repo path."""
    violations: list[str] = []
    for path in ROCKE.rglob("*"):
        if not path.is_file():
            continue
        if any(
            part in _GUARD_SKIP_DIRS or part.startswith(_GUARD_SKIP_PREFIXES)
            for part in path.relative_to(ROCKE).parts
        ):
            continue
        if path.suffix not in _GUARD_SUFFIXES and path.name not in _GUARD_NAMES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # this guard file defines the patterns
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat in _FORBIDDEN:
                if pat.search(line):
                    violations.append(
                        f"{path.relative_to(ROCKE)}:{i}: {line.strip()[:100]}"
                    )
    if violations:
        print(
            "RELATIVE-PATH GUARD: FAIL - absolute/repo paths found under rocke/platform/:"
        )
        for v in violations:
            print(f"  {v}")
        return 1
    print("RELATIVE-PATH GUARD: PASS")
    return 0


def differential_pytest_pass() -> int:
    """Re-run pytest with ``ROCKE_BACKEND=both`` (the cross-engine gate).

    The default pass exercises one engine per assertion, so two engines that
    each emit self-consistent but different IR both go green. ``both`` lowers
    through each and raises ``BackendMismatch`` on any byte difference, which
    is the only thing in the suite that can see a divergence the byte-identity
    gate's fixed family list does not cover.

    ``both`` never substitutes the Python result for a kernel the C++ engine
    could not lower, so this pass cannot go green vacuously: a kernel is either
    compared, or it fails, or -- for an arch named in
    ``backend.CPP_UNPORTED_ARCHES`` -- it is reported as a skip by the
    ``BackendCoverageGap`` hook in conftest. The skip count is the size of the
    known gap, and it is printed below so it stays visible.

    Needs the ``rocke_engine`` extension. Without it every kernel would raise
    and the lane would prove nothing, so it is reported as a skipped *pass*
    with a reason rather than run.
    """
    probe = subprocess.run(
        [sys.executable, "-c", "import rocke_engine"],
        capture_output=True,
        cwd=str(TESTS),
    )
    if probe.returncode != 0:
        print(
            "\n== pytest (ROCKE_BACKEND=both): SKIPPED ==\n"
            "   rocke_engine is not importable; build it with "
            "-DROCKE_BUILD_PYBIND=ON and put the build dir on PYTHONPATH."
        )
        return 0
    from rocke.core.backend import CPP_UNPORTED_ARCHES

    print("\n== pytest (ROCKE_BACKEND=both) ==")
    if CPP_UNPORTED_ARCHES:
        print(
            "   no fallback to Python in this lane; skips are the known C++ "
            f"coverage gap (unported arches: {', '.join(CPP_UNPORTED_ARCHES)})"
        )
    else:
        print(
            "   no fallback to Python in this lane, and no arch is exempt "
            "(backend.CPP_UNPORTED_ARCHES is empty), so every kernel here is "
            "either compared byte-for-byte or fails; remaining skips are "
            "environmental (torch / GPU)"
        )
    env = dict(os.environ, ROCKE_BACKEND="both")
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-rs"], cwd=str(TESTS), env=env
    ).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE test/validation runner")
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-pytest", action="store_true")
    ap.add_argument(
        "--no-both",
        action="store_true",
        help="skip the ROCKE_BACKEND=both differential pytest pass",
    )
    ap.add_argument(
        "--only",
        default="",
        help="restrict byte-identity gate to families containing SUBSTR",
    )
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify")
    )
    args = ap.parse_args()

    status = 0

    if not args.no_guard:
        status |= relative_path_guard()

    if not args.no_gate:
        print("\n== byte-identity gate ==")
        gate = [
            sys.executable,
            str(TOOLS / "check_byte_identity.py"),
            "--build-root",
            args.build_root,
        ]
        if args.only:
            gate += ["--only", args.only]
        status |= subprocess.run(gate).returncode

    if not args.no_pytest:
        print("\n== pytest ==")
        status |= subprocess.run(
            [sys.executable, "-m", "pytest", str(TESTS)], cwd=str(TESTS)
        ).returncode

    if not args.no_pytest and not args.no_both:
        status |= differential_pytest_pass()

    build_root = Path(args.build_root)
    # Only ctest when the CTest-registered binaries were actually built (the
    # byte-identity gate builds just `rocke_core`, so a gate-only build dir has the
    # registration file but no test executables -> running ctest there would
    # spuriously fail). Gate on the registered tests only; `rocke_smoke` is an
    # optional build-only target (not an add_test target) so it is not a signal.
    test_bins = [
        build_root / "tests" / b
        for b in ("rocke_ir_serialize_roundtrip", "rocke_tiled_attention_2d_reentrancy")
    ]
    if (build_root / "CTestTestfile.cmake").exists() and any(
        b.exists() for b in test_bins
    ):
        print("\n== ctest ==")
        status |= subprocess.run(
            ["ctest", "--output-on-failure", "--no-tests=ignore"], cwd=str(build_root)
        ).returncode

    print("\nRESULT:", "GREEN" if status == 0 else "RED")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
