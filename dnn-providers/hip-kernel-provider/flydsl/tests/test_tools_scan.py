# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The R-MIR scan and the SGPR def analysis of ``tools/``, on synthetic input. Pure Python."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL / "tools"))
import analyze  # noqa: E402
import rmir  # noqa: E402


def _mir(tmp_path: Path, save_arms: tuple[bool, bool]) -> str:
    """bb0 branches to bb1 and bb2, which each may save slot 0; bb3 restores it and uses it."""
    lines = [
        "bb.0:",
        "  successors: %bb.1(0x40000000), %bb.2(0x40000000)",
        "  %1:sgpr_32 = S_MOV_B32 5",
        "  %3:sgpr_64 = S_MOV_B64 0",
        "  S_CBRANCH_SCC0 %bb.2, implicit $scc",
        "bb.1:",
        "  successors: %bb.3(0x80000000)",
    ]
    save = "  SI_SPILL_S32_SAVE %1:sgpr_32, %stack.0, implicit $exec"
    if save_arms[0]:
        lines.append(save)
    lines += ["  S_BRANCH %bb.3", "bb.2:", "  successors: %bb.3(0x80000000)"]
    if save_arms[1]:
        lines.append(save)
    lines += [
        "bb.3:",
        "  %2:sgpr_32 = SI_SPILL_S32_RESTORE %stack.0, implicit $exec",
        "  S_STORE_DWORD_IMM %2:sgpr_32, %3:sgpr_64, 0",
    ]
    path = tmp_path / "f.mir"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_a_slot_saved_on_only_one_path_does_not_fail_the_gate(tmp_path) -> None:
    # The shipped dkdv kernels restore a slot behind a branch correlated with its save, so a
    # path that never saved the slot says nothing about it: scanning must not stop those rows.
    assert rmir.scan(_mir(tmp_path, (True, False))) == []


def test_a_slot_saved_on_only_one_path_is_reported_as_unrecorded(tmp_path) -> None:
    findings = rmir.unrecorded_slot_findings(_mir(tmp_path, (True, False)))
    assert [f[-1] for f in findings] == [True]  # a use of a restored register


def test_a_slot_saved_on_every_path_has_no_unrecorded_finding(tmp_path) -> None:
    assert rmir.unrecorded_slot_findings(_mir(tmp_path, (True, True))) == []


def test_the_mir_cli_reports_unrecorded_slot_findings_and_exits_zero(tmp_path) -> None:
    import subprocess

    def run(saved: tuple[bool, bool]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                sys.executable,
                str(FLYDSL / "tools" / "rmir.py"),
                "--mir",
                _mir(tmp_path, saved),
            ],
            capture_output=True,
            text=True,
        )

    one_path = run((True, False))
    assert one_path.returncode == 0
    assert "1 unrecorded-slot findings" in one_path.stdout
    every_path = run((True, True))
    assert every_path.returncode == 0
    assert "0 unrecorded-slot findings" in every_path.stdout


def test_a_slot_saved_on_no_path_is_a_finding(tmp_path) -> None:
    # Every predecessor lacks the slot, so none of them defines what the restore reads.
    findings = rmir.scan(_mir(tmp_path, (False, False)))
    assert [f[-1] for f in findings] == [True]


@pytest.mark.parametrize("mnemonic", ["s_cmov_b32", "s_cmov_b64", "s_cmovk_i32"])
def test_a_conditional_move_does_not_define_its_destination(mnemonic) -> None:
    # The destination keeps its old value when the condition is false: it is not a definition.
    assert analyze.defs_of(mnemonic, ["s5", "s6"]) == set()


def _gate_llvm_bin(**env: str) -> str:
    import os
    import subprocess

    clean = {k: v for k, v in os.environ.items() if k not in ("LLVM_BIN", "ROCM_PATH")}
    code = "import os, sdpa_bwd_gate; print(os.environ['LLVM_BIN'])"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=FLYDSL / "tools",
        env={**clean, **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def test_the_gate_takes_its_llvm_tools_from_rocm_path() -> None:
    assert _gate_llvm_bin(ROCM_PATH="/opt/rocm-x") == "/opt/rocm-x/llvm/bin"


def test_an_explicit_llvm_bin_wins_over_rocm_path() -> None:
    assert (
        _gate_llvm_bin(ROCM_PATH="/opt/rocm-x", LLVM_BIN="/elsewhere/bin")
        == "/elsewhere/bin"
    )
