#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# run_checks.py -- ONE command to run before a PR merge for sanity (guard +
# byte-identity gate + parity + pytest + on-GPU numeric). NOT comprehensive: the
# on-GPU numeric lane only exercises the arch of whatever GPU is visible on this
# machine, so it does not cover every supported arch. Run it on each arch you can.
#
# AUTO-DISCOVERS tests: adding a new test_*.py or a new <family>_emit.{py,c}
# parity pair requires NO edit to this script or any checklist doc -- it is
# picked up automatically, so a new test never has to be registered by hand.
#
# Stages, selected with --steps (default: all, in this order):
#   guard          -- relative-path guard
#   gate           -- byte-identity gate (all flavors unless --flavor pins one)
#   ir-validity    -- compile+link every emitted-IR corpus case
#   arch-domain    -- check corpus demands against every committed domain column
#   reproduction  -- regenerate this host flavor through its dedicated CTest
#   parity         -- platform instance .py/.c parity emitters
#   pytest         -- auto-globs platform/tests + library/tests; the pinned GPU
#                     reference suites run separately and report when skipped
#   numeric        -- on-GPU numeric correctness (needs a HIP device)
#
# --steps picks a subset, e.g. `--steps numeric` (just the correctness lane) or
# `--steps gate,parity`. --op scopes an OPERATOR/family across the stages that
# can filter (parity, pytest, numeric, and the byte-identity gate); the two
# corpus stages and the artifact-reproduction CTest always run in full. This
# preserves the complete-host-gate behavior that predates stage selection while
# avoiding the incompatible family/case-id filter vocabularies. You can run one
# operator's checks, e.g. `--steps numeric --op fmha_bwd`. The pinned GPU
# reference suites are never --op scoped: their test names carry the reference
# operation (e.g. conv, sdpa), not the operator/family. All paths are derived
# from this file so the tree stays copy-able; cross-platform (no bash/nproc/sudo).
#
# Usage:
#   python tools/run_checks.py                       # all stages
#   python tools/run_checks.py --steps numeric       # ONLY the on-GPU correctness lane
#   python tools/run_checks.py --steps gate,parity   # just the byte-identity checks
#   python tools/run_checks.py --op fmha_bwd          # one operator, all stages
#   python tools/run_checks.py --steps numeric --op fmha_bwd  # one operator, correctness only
#   python tools/run_checks.py --flavor llvm22       # pin one LLVM flavor
#   python tools/run_checks.py --list                # list what WOULD run, then exit

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROCKE = Path(__file__).resolve().parents[1]  # tools -> rocke
PLATFORM = ROCKE / "platform"
LIBRARY = ROCKE / "library"
PLATFORM_PYROOT = PLATFORM / "python"
EMIT_COMMON_DIR = PLATFORM / "tests" / "instances" / "parity"

FLAVORS = ("llvm20", "llvm22", "llvm23")
STAGES = (
    "guard",
    "gate",
    "ir-validity",
    "arch-domain",
    "reproduction",
    "parity",
    "pytest",
    "numeric",
)


def _env() -> dict:
    env = dict(os.environ)
    roots = [str(LIBRARY), str(PLATFORM_PYROOT), str(EMIT_COMMON_DIR)]
    env["PYTHONPATH"] = os.pathsep.join(
        roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.setdefault("ROCKE", str(PLATFORM))
    return env


def _has_gpu() -> bool:
    try:
        sys.path.insert(0, str(PLATFORM_PYROOT))
        from rocke.runtime.hip_module import get_device_arch

        return bool(get_device_arch(0))
    except Exception:
        return False


def _discover_pytest_dirs() -> list[Path]:
    # Auto-discover: any dir under platform/tests or library/tests holding a
    # test_*.py. We hand the roots to pytest (it recurses); we only check
    # presence so --list is informative and empty roots warn.
    dirs = []
    for root in (PLATFORM / "tests", LIBRARY / "tests"):
        if root.exists() and any(root.rglob("test_*.py")):
            dirs.append(root)
    return dirs


def _discover_pinned_reference_tests() -> list[Path]:
    # Pinned GPU reference suites (test_<operation>_pinned_reference.py) skip
    # unless this machine has an enrolled processor and an extracted bundle.
    # They run as a separate step with -rs and are never -k scoped, so a run
    # with zero pinned comparisons is reported instead of passing silently.
    return sorted((LIBRARY / "tests").glob("test_*_pinned_reference.py"))


def _unexercised_pinned(junit: Path, suites: list[Path]) -> list[str]:
    # Operations whose pinned suite executed no test, per pytest's junit report.
    # Skipping other processors' cases is normal; one executed test is enough.
    try:
        cases = list(ET.parse(junit).getroot().iter("testcase"))
    except (OSError, ET.ParseError):
        return []
    idle = []
    for suite in suites:
        ran = [
            c
            for c in cases
            if suite.stem in c.get("classname", "").split(".")
            and c.find("skipped") is None
        ]
        if not ran:
            idle.append(
                suite.stem.removeprefix("test_").removesuffix("_pinned_reference")
            )
    return idle


def _run(label: str, cmd: list[str], env: dict, cwd: Path | None = None) -> int:
    print(f"\n== {label} ==\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None).returncode
    print(f"-- {label}: {'PASS' if rc == 0 else 'FAIL (rc=%d)' % rc}", flush=True)
    return rc


def _parse_steps(raw: str) -> list[str]:
    if not raw:
        return list(STAGES)
    picked = [s.strip() for s in raw.split(",") if s.strip()]
    bad = [s for s in picked if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown --steps {bad}; valid: {', '.join(STAGES)}")
    # preserve canonical order regardless of how the user listed them
    return [s for s in STAGES if s in picked]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="rocke check runner (auto-discovering, stage-selectable)"
    )
    ap.add_argument(
        "--steps",
        default="",
        help=f"comma-separated subset of stages to run (default: all). Stages: {', '.join(STAGES)}",
    )
    ap.add_argument(
        "--op",
        default="",
        help="scope to an OPERATOR/family substring across pytest (except the pinned "
        "GPU reference suites), numeric, and the gate",
    )
    ap.add_argument(
        "--flavor",
        choices=FLAVORS,
        default="",
        help="pin one LLVM flavor (default: all supported flavors)",
    )
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_checks")
    )
    ap.add_argument(
        "--list", action="store_true", help="list what would run, then exit"
    )
    args = ap.parse_args()

    steps = _parse_steps(args.steps)
    steps_explicit = bool(args.steps)  # did the user hand-pick stages?
    op = args.op.strip()
    env = _env()
    flavors = (args.flavor,) if args.flavor else FLAVORS
    gpu = _has_gpu()
    pytest_dirs = _discover_pytest_dirs()
    pinned_tests = _discover_pinned_reference_tests()

    if args.list:
        print(
            f"rocke checks that WOULD run (steps: {', '.join(steps)}"
            + (f"; op: {op}" if op else "")
            + "):"
        )
        print(f"  guard        : {'yes' if 'guard' in steps else 'skip'}")
        print(f"  gate flavors : {', '.join(flavors) if 'gate' in steps else 'skip'}")
        print(
            f"  IR validity  : {'yes (full corpus)' if 'ir-validity' in steps else 'skip'}"
        )
        print(
            f"  arch domain  : {'yes (full corpus)' if 'arch-domain' in steps else 'skip'}"
        )
        print(
            "  reproduction : "
            + ("yes (host flavor CTest)" if "reproduction" in steps else "skip")
        )
        if "parity" in steps:
            print(
                "  platform parity: "
                + ("skip (--op has no filter here)" if op else "yes")
            )
        else:
            print("  platform parity: skip")
        pt = ", ".join(str(d.relative_to(ROCKE)) for d in pytest_dirs) or "NONE FOUND"
        print(
            f"  pytest dirs  : {pt + (f' (-k {op})' if op else '') if 'pytest' in steps else 'skip'}"
        )
        pinned = ", ".join(p.name for p in pinned_tests) or "NONE FOUND"
        print(
            f"  pinned refs  : {pinned + ' (never -k scoped)' if 'pytest' in steps else 'skip'}"
        )
        if "numeric" in steps:
            print(
                f"  numeric lane : {'yes (device visible)' if gpu else 'NO device'}"
                + (f" (--only {op})" if op else "")
            )
        else:
            print("  numeric lane : skip")
        return 0

    status = 0

    # guard -- relative-path guard (fast, no build). Disable every other
    # run_all stage and point CTest discovery at an empty temporary directory,
    # so a stale default build cannot turn this stage into a toolchain sweep.
    if "guard" in steps:
        with tempfile.TemporaryDirectory(prefix="rocke_guard_") as guard_root:
            status |= _run(
                "relative-path guard",
                [
                    sys.executable,
                    str(PLATFORM / "tests" / "run_all.py"),
                    "--no-gate",
                    "--no-ir-validity",
                    "--no-arch-domain",
                    "--no-pytest",
                    "--build-root",
                    guard_root,
                ],
                env,
                cwd=PLATFORM,
            )

    # gate -- byte-identity gate at each flavor (builds the C++ engine)
    if "gate" in steps:
        for fl in flavors:
            genv = dict(env, ROCKE_LLVM_FLAVOR=fl)
            cmd = [
                sys.executable,
                str(PLATFORM / "tools" / "check_byte_identity.py"),
                "--build-root",
                args.build_root,
            ]
            if op:
                cmd += ["--only", op]
            status |= _run(f"byte-identity gate [{fl}]", cmd, genv, cwd=PLATFORM)

    # Corpus gates -- these used to ride on run_all.py's default-on behavior in
    # the guard stage. Keep guard cheap without silently deleting either check
    # from the default complete host gate.
    if "ir-validity" in steps:
        status |= _run(
            "emitted-IR validity gate",
            [sys.executable, str(PLATFORM / "tools" / "check_ir_validity.py")],
            env,
            cwd=PLATFORM,
        )

    if "arch-domain" in steps:
        status |= _run(
            "arch-domain warning gate",
            [sys.executable, str(PLATFORM / "tools" / "check_arch_domain.py")],
            env,
            cwd=PLATFORM,
        )

    # Artifact reproduction owns external compiler processes, a long timeout,
    # serial execution, and an explicit CTest skip result. Configure its real
    # registration and select that test alone; do not fold it back into pytest
    # or make the entire native CTest suite a prerequisite.
    if "reproduction" in steps:
        configure_rc = _run(
            "configure arch-domain reproducibility CTest",
            [
                "cmake",
                "-S",
                str(PLATFORM),
                "-B",
                args.build_root,
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DPython3_EXECUTABLE={sys.executable}",
                "-DBUILD_TESTING=ON",
            ],
            env,
        )
        status |= configure_rc
        if configure_rc == 0:
            status |= _run(
                "arch-domain reproducibility CTest",
                [
                    "ctest",
                    "--test-dir",
                    args.build_root,
                    "-C",
                    "Release",
                    "-R",
                    "^rocke_arch_domain_reproducibility$",
                    "--output-on-failure",
                    "--no-tests=error",
                ],
                env,
            )

    # parity -- platform instance .py/.c pairs (covers a fixed set of platform
    # microkernels; has no operator filter, so it is skipped when scoping to --op).
    if "parity" in steps:
        if op:
            print(
                f"\n== parity: SKIPPED (--op {op}; platform instance parity has no operator filter) =="
            )
        else:
            status |= _run(
                "platform instance parity",
                [
                    sys.executable,
                    str(EMIT_COMMON_DIR / "run_parity.py"),
                    "--build-root",
                    args.build_root,
                ],
                env,
                cwd=PLATFORM,
            )

    # pytest -- auto-discovered dirs; GPU-only tests inside self-skip without a device
    unexercised: list[str] = []
    if "pytest" in steps:
        if pytest_dirs:
            base = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            cmd = list(base)
            if op:
                cmd += ["-k", op]
            cmd += [f"--ignore={p}" for p in pinned_tests]
            cmd += [str(d) for d in pytest_dirs]
            status |= _run("pytest", cmd, env, cwd=ROCKE)
            # pinned GPU references -- never -k scoped; -rs prints why they skipped
            if pinned_tests:
                with tempfile.TemporaryDirectory() as tmp:
                    junit = Path(tmp) / "pinned.xml"
                    cmd = base + ["-rs", f"--junitxml={junit}"]
                    cmd += [str(p) for p in pinned_tests]
                    status |= _run(
                        "pytest [pinned GPU references]", cmd, env, cwd=ROCKE
                    )
                    unexercised = _unexercised_pinned(junit, pinned_tests)
        else:
            print("\n== pytest: NO test dirs discovered ==")

    # numeric -- on-GPU numeric correctness (honors --op via the harness's --only)
    if "numeric" in steps:
        numeric = LIBRARY / "tests" / "differential" / "numeric_attention.py"
        if gpu and numeric.exists():
            cmd = [sys.executable, str(numeric)]
            if op:
                cmd += ["--only", op]
            status |= _run("on-GPU numeric", cmd, env, cwd=LIBRARY)
            # gfx950 fp8 KV-decode 3D numeric gate. The dense FMHA lane above is
            # fp16 2D only, so this covers the fp8 long-KV decode -> 3D split-KV
            # cohort. Self-skips (exit 0) off gfx950. Only run it on an unscoped
            # numeric pass: this driver has no --only support, so an --op scope
            # (e.g. fmha_bwd) can neither select nor filter it, and pulling in a
            # gfx950-only decode check under an unrelated operator's scope would
            # defeat the scoping.
            decode3d = (
                LIBRARY
                / "builders"
                / "gfx950"
                / "attention"
                / "decode"
                / "fp8_decode_3d_verify.py"
            )
            if decode3d.exists() and not op:
                status |= _run(
                    "on-GPU numeric (gfx950 fp8 decode 3D)",
                    [sys.executable, str(decode3d)],
                    env,
                    cwd=LIBRARY,
                )
        elif steps_explicit:
            # numeric was explicitly requested; no device is an error, not a skip.
            print("\n== numeric: NO HIP device visible (explicitly requested) ==")
            return 1
        else:
            print("\n== numeric: SKIPPED (no HIP device visible) ==")

    print(
        f"\n{'=' * 48}\nrocke checks: {'ALL PASSED' if status == 0 else 'FAILURES PRESENT'}"
    )
    if unexercised:
        # A skip is not a failure, but a green run must not read as pinned coverage.
        print(
            f"pinned GPU reference comparisons did NOT run for {', '.join(unexercised)} "
            "(reasons above); see TESTING.md to run them"
        )
    print("=" * 48)
    return 1 if status else 0


if __name__ == "__main__":
    raise SystemExit(main())
