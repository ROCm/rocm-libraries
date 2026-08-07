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
#   guard    -- relative-path guard        (platform tests/run_all.py)
#   gate     -- byte-identity gate         (BOTH llvm20 and llvm22 unless --flavor pins one)
#   parity   -- platform instance .py/.c parity emitters
#   pytest   -- auto-globs platform/tests + library/tests
#   numeric  -- on-GPU numeric correctness (needs a HIP device)
#
# --steps picks a subset, e.g. `--steps numeric` (just the correctness lane) or
# `--steps gate,parity`. --op scopes an OPERATOR/family across the stages that
# can filter (parity, pytest, numeric, and the gate) so you can run one
# operator's checks, e.g. `--steps numeric --op fmha_bwd`. All paths are derived
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
from pathlib import Path

ROCKE = Path(__file__).resolve().parents[1]  # tools -> rocke
PLATFORM = ROCKE / "platform"
LIBRARY = ROCKE / "library"
PLATFORM_PYROOT = PLATFORM / "python"
EMIT_COMMON_DIR = PLATFORM / "tests" / "instances" / "parity"

FLAVORS = ("llvm20", "llvm22")
STAGES = ("guard", "gate", "parity", "pytest", "numeric")


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
        help="scope to an OPERATOR/family substring across pytest, numeric, and the gate",
    )
    ap.add_argument(
        "--flavor",
        choices=FLAVORS,
        default="",
        help="pin one LLVM flavor (default: both)",
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

    if args.list:
        print(
            f"rocke checks that WOULD run (steps: {', '.join(steps)}"
            + (f"; op: {op}" if op else "")
            + "):"
        )
        print(f"  guard        : {'yes' if 'guard' in steps else 'skip'}")
        print(f"  gate flavors : {', '.join(flavors) if 'gate' in steps else 'skip'}")
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
        if "numeric" in steps:
            print(
                f"  numeric lane : {'yes (device visible)' if gpu else 'NO device'}"
                + (f" (--only {op})" if op else "")
            )
        else:
            print("  numeric lane : skip")
        return 0

    status = 0

    # guard -- relative-path guard (fast, no build)
    if "guard" in steps:
        status |= _run(
            "relative-path guard",
            [
                sys.executable,
                str(PLATFORM / "tests" / "run_all.py"),
                "--no-gate",
                "--no-pytest",
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
    if "pytest" in steps:
        if pytest_dirs:
            cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            if op:
                cmd += ["-k", op]
            cmd += [str(d) for d in pytest_dirs]
            status |= _run("pytest", cmd, env, cwd=ROCKE)
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
        elif steps_explicit:
            # numeric was explicitly requested; no device is an error, not a skip.
            print("\n== numeric: NO HIP device visible (explicitly requested) ==")
            return 1
        else:
            print("\n== numeric: SKIPPED (no HIP device visible) ==")

    print(
        f"\n{'=' * 48}\nrocke checks: {'ALL PASSED' if status == 0 else 'FAILURES PRESENT'}\n{'=' * 48}"
    )
    return 1 if status else 0


if __name__ == "__main__":
    raise SystemExit(main())
