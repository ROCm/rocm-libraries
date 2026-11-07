#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Developer entrypoint for the rocKE library tests. Ordinary pytest discovery
# is the fast unit tier. Resource-bound exhaustive checks are non-discovered
# modules invoked as explicit stages here and as separate installed CTests.
#
# Usage:
#   python library/tests/run_all.py                 # units + explicit host gates
#   python library/tests/run_all.py --pytest-only   # discovered pytest only
#   python library/tests/run_all.py --gate counted-structural
#   python library/tests/run_all.py --gate counted-compiler --require-compiler-gate
#   python library/tests/run_all.py --gate counted-compiler --gate-arch gfx942 \
#       --allow-partial-gate                 # focused diagnostic, never GREEN
#   python library/tests/run_all.py --gpu           # discovered pytest + GPU tests + host gates
#   python library/tests/run_all.py --gpu-only      # only GPU-marked discovered pytest
#   python library/tests/run_all.py --only dense    # focused pytest; gates omitted

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from ._explicit_gate_policy import (
        preflight_is_disabled as _preflight_is_disabled,
    )
    from ._explicit_gate_policy import selection_is_partial as _selection_is_partial
else:
    from _explicit_gate_policy import preflight_is_disabled as _preflight_is_disabled
    from _explicit_gate_policy import selection_is_partial as _selection_is_partial

TESTS = Path(__file__).resolve().parent
EXPLICIT_GATES = {
    "counted-structural": TESTS / "gates" / "attention_bwd_counted_structural_gate.py",
    "counted-compiler": TESTS / "gates" / "attention_bwd_counted_compiler_gate.py",
}
_OPTIONAL_ISA_ENV = "ROCKE_COUNTED_ISA_OPTIONAL"
_STAGE_OUTPUT_ARGS = ("--junitxml", "--junit-xml")


def _has_stage_output(arguments: list[str]) -> bool:
    """Whether arguments name output that one pytest stage must own."""
    return any(
        argument == option or argument.startswith(option + "=")
        for argument in arguments
        for option in _STAGE_OUTPUT_ARGS
    )


def _gate_environment(
    name: str,
    *,
    require_compiler: bool,
    gate_arch: str | None = None,
    partial: bool = False,
) -> dict[str, str]:
    """Return an isolated gate environment with explicit compiler policy."""
    partial = partial or gate_arch is not None
    environment = dict(os.environ)
    if name in EXPLICIT_GATES:
        environment.pop("PYTEST_ADDOPTS", None)
        environment.pop("ROCKE_EXPLICIT_GATE_PREFLIGHT", None)
        environment["ROCKE_EXPLICIT_GATE_RUN"] = "1"
        environment["ROCKE_EXPLICIT_GATE_PARTIAL"] = "1" if partial else "0"
        if not partial:
            environment.pop("ROCKE_TEST_SHARD", None)
    if name == "counted-compiler":
        # This gate validates the caller-selected ROCm/COMGR toolchain rather
        # than replacing it with torch's bundled compiler.
        environment["ROCKE_BWD_NO_COMGR_PIN"] = "1"
        environment.pop("ROCKE_BWD_ISA_ARCH", None)
        if gate_arch is not None:
            environment["ROCKE_BWD_ISA_ARCH"] = gate_arch
        if require_compiler:
            environment.pop(_OPTIONAL_ISA_ENV, None)
        else:
            environment[_OPTIONAL_ISA_ENV] = "1"
    return environment


@dataclass(frozen=True)
class _StageResult:
    code: int
    skipped: bool = False


def _run(
    label: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    allow_skip: bool = False,
) -> _StageResult:
    print(f"\n== {label} ==")
    print(" ", " ".join(command))
    result = subprocess.run(command, cwd=str(TESTS.parent), env=env).returncode
    if result == 77 and allow_skip:
        print(f"{label}: SKIPPED")
        return _StageResult(0, skipped=True)
    return _StageResult(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="rocKE library validation runner")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="include the `gpu`-marked numeric pytest lane",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="run only the `gpu`-marked discovered pytest lane",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue into resource-bound gates after an earlier stage fails",
    )
    parser.add_argument(
        "--pytest-only",
        action="store_true",
        help="run discovered pytest only; omit explicit gates",
    )
    parser.add_argument(
        "--gate",
        action="append",
        choices=tuple(EXPLICIT_GATES),
        help="run only this explicit gate; repeat to select several",
    )
    parser.add_argument(
        "--require-compiler-gate",
        action="store_true",
        help="fail when the counted ISA gate's qualified toolchain is unavailable; "
        "the default local run reports that gate as skipped",
    )
    parser.add_argument(
        "--allow-partial-gate",
        action="store_true",
        help="allow pytest selection arguments for focused gate debugging; the result "
        "is not a complete gate verdict",
    )
    parser.add_argument(
        "--gate-arch",
        choices=("gfx942", "gfx950"),
        help="select one counted-compiler architecture for an explicitly partial gate run",
    )
    parser.add_argument(
        "--only",
        default="",
        help="pytest -k expression for focused unit iteration; explicit gates omitted",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra pytest arguments, after --, applied to every selected stage",
    )
    args = parser.parse_args()

    if args.pytest_only and args.gate:
        parser.error("--pytest-only and --gate are mutually exclusive")
    if args.gpu_only and args.gate:
        parser.error("--gpu-only and --gate are mutually exclusive")
    if args.gate_arch and (
        not args.allow_partial_gate
        or not args.gate
        or "counted-compiler" not in args.gate
    ):
        parser.error("--gate-arch needs --gate counted-compiler --allow-partial-gate")
    if args.require_compiler_gate and (
        args.pytest_only
        or args.gpu_only
        or args.only
        or (args.gate and "counted-compiler" not in args.gate)
    ):
        parser.error("--require-compiler-gate needs the counted-compiler gate")

    extra = args.pytest_args
    if extra and extra[0] == "--":
        extra = extra[1:]
    filtered = _selection_is_partial(extra)
    preflight_disabled = _preflight_is_disabled(extra)
    gates_selected = bool(args.gate) or not (
        args.pytest_only or args.gpu_only or args.only
    )
    if gates_selected and preflight_disabled:
        parser.error("explicit gates do not allow --noconftest or --confcutdir")
    if gates_selected and filtered and not args.allow_partial_gate:
        parser.error(
            "pytest selection arguments need --allow-partial-gate; a partial run is "
            "not a gate verdict"
        )
    selected_stage_count = (0 if args.gate else 1) + (
        len(args.gate) if args.gate else (2 if gates_selected else 0)
    )
    if selected_stage_count > 1 and _has_stage_output(extra):
        parser.error(
            "JUnit output needs a single selected stage; use --pytest-only or one --gate"
        )

    status = 0
    skipped = False
    run_discovered_pytest = not args.gate
    if run_discovered_pytest:
        command = [sys.executable, "-m", "pytest", str(TESTS)]
        if args.gpu_only:
            command += ["-m", "gpu"]
        elif not args.gpu:
            command += ["-m", "not gpu"]
        if args.only:
            command += ["-k", args.only]
        command += extra
        pytest_result = _run("rocke library discovered pytest", command)
        status = status or pytest_result.code
        skipped |= pytest_result.skipped
        if status and not args.keep_going:
            print(
                "\nRESULT: RED (resource gates not run after discovered-pytest failure)"
            )
            return status

    if args.gate:
        gates = args.gate
    elif args.pytest_only or args.gpu_only or args.only:
        gates = []
    else:
        gates = list(EXPLICIT_GATES)

    for name in gates:
        environment = _gate_environment(
            name,
            require_compiler=args.require_compiler_gate,
            gate_arch=args.gate_arch if name == "counted-compiler" else None,
            partial=filtered or args.gate_arch is not None,
        )
        command = [sys.executable, "-m", "pytest", str(EXPLICIT_GATES[name])]
        command += extra
        gate_result = _run(
            f"rocke library gate: {name}",
            command,
            env=environment,
            allow_skip=name == "counted-compiler" and not args.require_compiler_gate,
        )
        status = status or gate_result.code
        skipped |= gate_result.skipped
        if status and not args.keep_going:
            break

    if status:
        verdict = "RED"
    elif filtered or args.gate_arch or args.only:
        verdict = "PARTIAL"
    elif skipped:
        verdict = "INCOMPLETE"
    else:
        verdict = "GREEN"
    print("\nRESULT:", verdict)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
