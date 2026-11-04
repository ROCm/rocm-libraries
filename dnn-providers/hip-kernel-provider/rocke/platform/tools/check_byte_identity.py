#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The engine definition-of-done in one command: build the C++ engine fresh and
# prove its LLVM-IR emission is byte-identical to the Python engine across every
# kernel family. A green run means the dual-backend contract still holds.
#
# When the selected emission profile lacks the LLVM23-generation target-op
# capability, a second lane re-runs the affected families at llvm23. Profiles
# that already carry the capability (llvm23 and llvm24) run them directly.
#
# Cross-platform (Windows + Linux) replacement for the legacy check_byte_identity.sh.
# All paths are derived relative to this file so the rocke/platform/ tree stays copy-able.
#
# Usage:
#   python rocke/platform/tools/check_byte_identity.py [--ir] [--only SUBSTR]
#       [--build-root DIR] [--ref-pyroot DIR] [--ref-shim DIR]

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _hostcaps import available_cpus

HERE = Path(__file__).resolve().parent
ROCKE = HERE.parents[0] if HERE.name != "tools" else HERE.parent  # tools -> rocKE
RUN_DIFF = ROCKE / "tests" / "instances" / "differential" / "run_diff.py"
# The parity family that carries the llvm23-gated gfx1250 ops.
GFX1250_FAMILY = "gfx1250_lowering"
# Families whose every config uses an llvm23-gated op (their kernels move data
# with TDM). Under another flavor both engines reject all of them, which the
# gate reads as ALL_REJECTED, so they are compared in the llvm23 lane only.
TARGET_OP_PROFILE_FAMILIES = (
    "conv_dgrad_patch_wmma",
    "conv_dgrad_smallc_wmma",
    "conv_direct_dgrad_wmma",
    "conv_direct_fwd_wmma",
    "conv_fwd_small_wmma",
)


def _selected_profile_supports_target_ops() -> bool:
    """Resolve the selected lane once and query its structural profile."""
    python_root = ROCKE / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))
    from rocke.core.lower_llvm import (
        _resolve_llvm_flavor,
        _supports_llvm23_target_ops,
    )

    return _supports_llvm23_target_ops(_resolve_llvm_flavor())


def _cxx() -> str | None:
    for c in ("c++", "clang++", "g++", "cl"):
        p = shutil.which(c)
        if p:
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="rocKE byte-identity gate")
    ap.add_argument(
        "--ir", action="store_true", help="also run the IR canonical diff (diagnostic)"
    )
    ap.add_argument(
        "--only",
        default="",
        help="restrict to families containing SUBSTR (comma-separated)",
    )
    ap.add_argument(
        "--build-root", default=str(Path(tempfile.gettempdir()) / "rocke_verify")
    )
    ap.add_argument("--ref-pyroot", default=os.environ.get("ROCKE_REF_PYROOT", ""))
    ap.add_argument("--ref-shim", default=os.environ.get("ROCKE_REF_SHIM", ""))
    args = ap.parse_args()

    build_root = Path(args.build_root)
    archive = build_root / "librocke_core.a"

    print("== building engine archive (fresh) ==")
    print(f"   source : {ROCKE}")
    print(f"   build  : {build_root}")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROCKE),
            "-B",
            str(build_root),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build_root),
            "--target",
            "rocke_core",
            # This process's share, not the machine's size. `os.cpu_count()`
            # reports the host, so on a scheduler-pinned node it asked ninja
            # for 384 compile jobs against an affinity of 2 -- which is how a
            # gate that should take a couple of minutes turns into a thrash.
            "-j",
            str(available_cpus()),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if not archive.exists():
        print(f"FATAL: archive not produced: {archive}", file=sys.stderr)
        return 1
    print(f"   archive: {archive}")

    # Freshness/provenance: print the build-id stamped into the archive. Off the
    # emission path, so this never affects the .ll contract. Best-effort.
    cxx = _cxx()
    if cxx and cxx.endswith("cl") is False:
        probe_src = build_root / "_build_id_probe.cpp"
        probe_bin = build_root / "_build_id_probe"
        probe_src.write_text(
            'extern "C" const char* rocke_build_id(void);\n'
            'extern "C" const char* rocke_engine_version(void);\n'
            "#include <cstdio>\n"
            'int main(){ printf("%s %s\\n", rocke_build_id(), rocke_engine_version()); return 0; }\n'
        )
        try:
            subprocess.run(
                [
                    cxx,
                    "-std=c++20",
                    str(probe_src),
                    str(archive),
                    "-lm",
                    "-o",
                    str(probe_bin),
                ],
                check=True,
                stderr=subprocess.DEVNULL,
            )
            out = subprocess.run([str(probe_bin)], capture_output=True, text=True)
            print(f"   build-id: {out.stdout.strip()}")
        except (subprocess.CalledProcessError, OSError):
            print("   build-id: (probe unavailable)")

    only_args = ["--only", args.only] if args.only else []
    ref_args = []
    if args.ref_pyroot:
        ref_args += ["--pyroot", args.ref_pyroot]
    if args.ref_shim:
        ref_args += ["--shim", args.ref_shim]

    def run_gate(
        label: str,
        dashboard: str,
        extra: list[str],
        *,
        only: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> int:
        print(f"\n== differential gate: {label} ==", flush=True)
        # The full gate is long-running. Inherit the parent's streams and make
        # the child unbuffered so each completed family is visible instead
        # of holding the entire dashboard until the process exits.
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(RUN_DIFF),
                "--archive",
                str(archive),
                "--json",
                dashboard,
                *(only_args if only is None else only),
                *ref_args,
                *extra,
            ],
            env=env,
        )
        # run_diff owns the pass/fail rule (its GATE_PASS/GATE_FAIL partition,
        # which fails closed on any status it does not recognize). Trust the
        # exit code -- a second rule here could only disagree with it.
        return proc.returncode

    # A dashboard per lane, beside the archive it describes: one shared path
    # meant the diagnostic ir run overwrote the gating run's JSON.
    supports_target_ops = _selected_profile_supports_target_ops()
    only_subs = [s for s in args.only.split(",") if s]
    llvm23_families = [
        f
        for f in TARGET_OP_PROFILE_FAMILIES
        if not only_subs or any(s in f for s in only_subs)
    ]
    # Under another flavor the default lane leaves the llvm23-only families
    # out (run_diff passes when that leaves nothing to compare).
    status = run_gate(
        "LLVM-IR (the contract)",
        str(build_root / "dashboard_ll.json"),
        ["--mode", "ll"]
        + (
            []
            if supports_target_ops
            else ["--exclude", ",".join(TARGET_OP_PROFILE_FAMILIES)]
        ),
    )
    # The gfx1250-only ops need the LLVM23-generation target-op profile. Under
    # an older profile both engines reject those configs, which reads as
    # BOTH_REJECTED and passes without comparing a byte. Re-run affected
    # families at llvm23 only when the selected profile lacks that capability.
    llvm23_only = [s for s in only_subs if s in GFX1250_FAMILY]
    if not args.only:
        llvm23_only = [GFX1250_FAMILY]
    llvm23_only += llvm23_families
    if not supports_target_ops and llvm23_only:
        env = dict(os.environ, ROCKE_LLVM_FLAVOR="llvm23")
        status |= run_gate(
            "LLVM-IR, gfx1250 families at llvm23",
            str(build_root / "dashboard_ll_llvm23.json"),
            ["--mode", "ll"],
            only=["--only", ",".join(llvm23_only)],
            env=env,
        )
    if args.ir:
        # Diagnostic, not gating: its exit code is deliberately discarded, so a
        # GATE FAILURES block under this label does not turn the result RED.
        run_gate(
            "IR canonical (diagnostic)",
            str(build_root / "dashboard_ir.json"),
            ["--mode", "ir", "--canonical"],
        )

    print()
    if status == 0:
        print(
            "RESULT: GREEN - engine builds and .ll emission is byte-identical to Python."
        )
    else:
        # Not "the engines disagree" -- they may not have been compared at all
        # (nothing built, nothing enumerated). The GATE FAILURES block says which.
        print("RESULT: RED - the gate rejected this run; see GATE FAILURES above.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
