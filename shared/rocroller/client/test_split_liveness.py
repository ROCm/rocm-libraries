#!/usr/bin/env python3

# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test the split_liveness script."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest
from test_gemm_client import SOLUTION_NOT_SUPPORTED_ON_ARCH, gemm, rocm_gfx

split_liveness_script = (
    Path(__file__).parent.parent / "scripts" / "split_liveness.py"
).resolve()

split_liveness = None
if split_liveness is None:
    spec = importlib.util.spec_from_file_location(
        "split_liveness", split_liveness_script
    )
    split_liveness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split_liveness)
    del spec


def test_split_liveness(tmp_path):
    """Test that split_liveness splits a .live file into per-register-type files."""

    arch = rocm_gfx()
    if arch is None:
        pytest.skip("No GPU detected via rocminfo")
    if arch.startswith("gfx12"):
        pytest.skip("Skipping on gfx12 (no ACCVGPR)")

    if not gemm.exists():
        pytest.skip("rocroller-gemm binary not found")
    if not split_liveness_script.exists():
        pytest.skip("split_liveness script not found")

    asm_path = tmp_path / "test_liveness.s"
    env = os.environ.copy()
    env["ROCROLLER_KERNEL_ANALYSIS"] = "1"

    p = subprocess.run(
        [str(gemm), "generate", "--asm", str(asm_path)],
        env=env,
        cwd=str(tmp_path),
        text=True,
        capture_output=True,
    )

    if p.returncode == SOLUTION_NOT_SUPPORTED_ON_ARCH:
        pytest.skip("GEMM solution not supported on this architecture")
    assert (
        p.returncode == 0
    ), f"rocroller-gemm failed\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"

    live_path = Path(str(asm_path) + ".live")
    assert live_path.exists(), f"Expected .live file at {live_path}"
    assert live_path.stat().st_size > 0, "Expected .live file to be non-empty"

    p = subprocess.run(
        [str(split_liveness_script), str(live_path)],
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, f"""
        split_liveness failed
        stdout:
        {p.stdout}
        stderr:
        {p.stderr}
        """

    for reg_type in ["VGPR", "SGPR", "ACCVGPR"]:
        parts = str(live_path).split(".")
        expected_path = Path(parts[0] + "_" + reg_type + "." + ".".join(parts[1:]))

        assert (
            expected_path.exists()
        ), f"Expected split file for {reg_type} at {expected_path}"

        lines = expected_path.read_text().splitlines()
        while lines and not lines[-1].strip():
            lines.pop()

        assert len(lines) > 0, f"{reg_type} file is empty"
        assert len(lines) % 2 == 0, (
            f"{reg_type} file has {len(lines)} lines, expected even"
            " (alternating liveness/instruction)"
        )

        assert reg_type in lines[0], (
            f"Expected '{reg_type}' in first line of {reg_type} file,"
            f" got: {lines[0]!r}"
        )
        assert "Instruction" in lines[1], (
            f"Expected 'Instruction' in second line of {reg_type} file,"
            f" got: {lines[1]!r}"
        )


def test_split_lines():
    """Test the split_lines function from the split_liveness script."""

    input_text = """
    ACCVGPR |VGPR   |SGPR  |Instruction
    :       |v::    |:x    | s_nop 0
    X       |x_:    |v^    | v_mul_u32 v2, v1, v3
    """.strip().splitlines()
    input_text = list([line.strip() for line in input_text])

    expected_output = {
        "ACCVGPR": [
            "ACCVGPR ",
            "Instruction",
            ":       ",
            " s_nop 0",
            "X       ",
            " v_mul_u32 v2, v1, v3",
        ],
        "VGPR": [
            "VGPR   ",
            "Instruction",
            "v::    ",
            " s_nop 0",
            "x_:    ",
            " v_mul_u32 v2, v1, v3",
        ],
        "SGPR": [
            "SGPR  ",
            "Instruction",
            ":x    ",
            " s_nop 0",
            "v^    ",
            " v_mul_u32 v2, v1, v3",
        ],
    }

    assert split_liveness.split_lines(iter(input_text)) == expected_output
