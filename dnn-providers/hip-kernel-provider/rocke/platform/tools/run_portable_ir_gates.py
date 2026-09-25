#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The portable-IR definition-of-done in one command: build the shared engine and
# prove that the C++ replay paths reproduce the Python lowerer byte for byte, all
# the way to the object code that actually ships.
#
# WHY THE GATES END AT HSACO. The portable-IR path has two implementations of the
# same lowering -- one in Python, one in C++ -- plus a recipe VM that replays a
# recorded build, and a roller that turns several recorded builds into one
# parametric recipe. The contract between them is byte-identity. Checking it on
# .ll alone is not enough for two reasons: the artifact that ships is HSACO, and
# identical .ll is not evidence that the .ll compiles at all. So the .ll gate is
# kept as the fast, precise signal (it pins SSA names, which HSACO would let
# drift) and every path is then carried through comgr.
#
# No GPU is required: comgr compiles for the target ISA on the host.
#
# This script is what CI runs, and running it locally is the way to reproduce a
# CI failure. Everything it needs beyond a compiler and comgr is derived from
# this file's location, so the rocke/platform/ tree stays copy-able.
#
# Usage:
#   python rocke/platform/tools/run_portable_ir_gates.py
#       [--build-root DIR] [--log-dir DIR] [--lib PATH] [--arch gfx950,gfx942]
#       [--expect-points N] [--skip-tests] [--no-hsaco]

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
PLATFORM = HERE.parent  # tools -> platform
ROCKE = PLATFORM.parent  # platform -> rocke
PYROOT = PLATFORM / "python"
LIBRARY = ROCKE / "library"

# The rolled gate verifies one parametric recipe per axis at several values,
# sampled and held out. Pinning the count here rather than in the CI workflow
# keeps one source of truth: an axis that quietly stops rolling then fails
# everywhere instead of just producing a shorter table under a green tick.
EXPECT_POINTS = 22


def _env(lib: Optional[Path], extra: Optional[dict] = None) -> dict:
    env = dict(os.environ)
    parts = [str(PYROOT), str(LIBRARY)]
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if lib is not None:
        env["ROCKE_ONLINE_LIB"] = str(lib)
    env.update(extra or {})
    return env


def _stream(cmd: List[str], env: dict, log: Optional[Path]) -> int:
    """Run cmd, echoing output live and copying it to a log file.

    Live output keeps a CI job readable while it runs; the log file is what the
    workflow uploads and what the summary quotes from."""
    fh = log.open("w") if log else None
    try:
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(PLATFORM),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if fh:
                fh.write(line)
        return proc.wait()
    finally:
        if fh:
            fh.close()


def build_shared_engine(build_root: Path) -> Optional[Path]:
    """Configure and build librocke.so. Returns its path, or None on failure."""
    print("== building the shared engine ==")
    print(f"   source : {PLATFORM}")
    print(f"   build  : {build_root}")
    configure = [
        "cmake",
        "-S",
        str(PLATFORM),
        "-B",
        str(build_root),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DROCKE_BUILD_SHARED_ENGINE=ON",
    ]
    # Ninja if it is available and this build tree has not already been
    # configured with something else, since generators cannot be swapped.
    if not (build_root / "CMakeCache.txt").exists():
        from shutil import which

        if which("ninja"):
            configure.append("-GNinja")
    try:
        subprocess.run(configure, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build_root),
                "--target",
                "rocke_shared",
                "-j",
                str(os.cpu_count() or 1),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"FATAL: could not build the shared engine: {e}", file=sys.stderr)
        return None

    lib = build_root / "librocke.so"
    if not lib.exists():
        print(f"FATAL: shared engine not produced: {lib}", file=sys.stderr)
        return None
    print(f"   library: {lib}")
    return lib


def describe_environment(lib: Path) -> None:
    """Print what will actually do the work.

    Which comgr compiles the IR decides the LLVM flavor both engines must emit,
    and the engine build-id says which library is loaded. When a gate fails for
    an environmental reason, this preamble is usually the whole diagnosis."""
    probe = (
        "import ctypes\n"
        "from rocke.portable_ir.src import online\n"
        "from rocke.runtime import comgr\n"
        "from rocke.portable_ir.drivers.hsaco_parity import _auto_cap_gb\n"
        "lib = online.load()\n"
        "fn = lib.rocke_build_id; fn.restype = ctypes.c_char_p; fn.argtypes = []\n"
        "ver = comgr.resolved_lib_rocm_version()\n"
        "print('   engine build-id :', fn().decode())\n"
        "print('   comgr           :', comgr.resolved_lib_path())\n"
        "print('   comgr ROCm      :', '.'.join(map(str, ver)) if ver else 'unknown')\n"
        "print('   memory cap      :', str(_auto_cap_gb()) + 'G per compile')\n"
    )
    print("== environment ==")
    print(f"   python          : {sys.version.split()[0]}")
    rc = _stream([sys.executable, "-c", probe], _env(lib), None)
    if rc != 0:
        print("   (probe failed; the gates below will report the real error)")


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE portable-IR parity gates")
    ap.add_argument(
        "--build-root",
        default=str(Path(tempfile.gettempdir()) / "rocke_portable_ir_gates"),
        help="CMake build directory for the shared engine",
    )
    ap.add_argument(
        "--lib",
        default="",
        help="use an existing librocke.so instead of building one",
    )
    ap.add_argument("--log-dir", default="", help="write a log per gate here")
    ap.add_argument("--arch", default="gfx950,gfx942", help="arches for the HSACO gate")
    ap.add_argument(
        "--expect-points",
        type=int,
        default=EXPECT_POINTS,
        help="minimum verified (family, value) points in the rolled gate",
    )
    ap.add_argument(
        "--skip-tests", action="store_true", help="skip the unit tests, run gates only"
    )
    ap.add_argument(
        "--no-hsaco",
        action="store_true",
        help="stop the rolled gate at .ll. Faster, and weaker: use for a quick "
        "local check, never as the gate",
    )
    args = ap.parse_args()

    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    if args.lib:
        lib = Path(args.lib).resolve()
        if not lib.exists():
            print(f"FATAL: --lib does not exist: {lib}", file=sys.stderr)
            return 1
        print(f"== using the shared engine at {lib} ==")
    else:
        built = build_shared_engine(Path(args.build_root))
        if built is None:
            return 1
        lib = built

    describe_environment(lib)

    steps: List[Tuple[str, List[str], dict]] = []
    if not args.skip_tests:
        # The in-package unit tests live under python/rocke/, which pytest does
        # not collect from tests/; test_portable_ir.py drives them, so pointing
        # at both directories covers the lot.
        steps.append(
            (
                "unit_tests",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    str(PYROOT / "rocke" / "portable_ir" / "tests"),
                    str(PLATFORM / "tests" / "portable_ir"),
                ],
                {},
            )
        )
    steps += [
        (
            "parity_matrix",
            [sys.executable, "-u", "-m", "rocke.portable_ir.drivers.parity_matrix"],
            # Some parity emitters ask for backend='cpp' while building their own
            # kernel and warn per kernel when the pybind extension is absent. It
            # is absent by design and is not on this gate's path -- the C++ side
            # under test is librocke.so, reached through ctypes -- so the warning
            # would only bury the verdict.
            {"ROCKE_CPP_QUIET_FALLBACK": "1"},
        ),
        (
            "hsaco_parity",
            [
                sys.executable,
                "-u",
                "-m",
                "rocke.portable_ir.drivers.hsaco_parity",
                "--arch",
                args.arch,
            ],
            {},
        ),
        (
            "roll_hsaco_parity",
            [
                sys.executable,
                "-u",
                "-m",
                "rocke.portable_ir.drivers.roll_hsaco_parity",
                "--expect-points",
                str(args.expect_points),
            ]
            + (["--no-hsaco"] if args.no_hsaco else []),
            {},
        ),
    ]

    results: List[Tuple[str, int, float]] = []
    for name, cmd, extra in steps:
        print(f"\n{'=' * 78}\n== {name}\n{'=' * 78}")
        t0 = time.perf_counter()
        log = (log_dir / f"{name}.log") if log_dir else None
        rc = _stream(cmd, _env(lib, extra), log)
        dt = time.perf_counter() - t0
        results.append((name, rc, dt))
        if rc != 0 and name == "unit_tests":
            # A broken unit suite makes the gate verdicts hard to interpret, but
            # they are still worth having, so keep going and report both.
            print("  (unit tests failed; continuing so the gates still report)")

    print(f"\n{'=' * 78}")
    for name, rc, dt in results:
        print(f"  {'PASS' if rc == 0 else 'FAIL'}  {name:<20} {dt:6.1f}s")
    failed = [n for n, rc, _ in results if rc != 0]
    if failed:
        print(f"\nRESULT: RED - {', '.join(failed)}")
        print(
            "\nIf a kernel that used to compile no longer does, hsaco_parity names "
            "it.\nWhen that is intended, regenerate the pinned set and review the "
            "diff:\n  python3 -m rocke.portable_ir.drivers.hsaco_parity "
            "--update-baseline"
        )
        return 1
    print("\nRESULT: GREEN - both C++ replay paths match Python byte for byte, at")
    print("        .ll and at HSACO, including recipe values never recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
