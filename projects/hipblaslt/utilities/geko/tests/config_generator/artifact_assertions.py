################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Assertions on config_generator output layout and file contents."""

from __future__ import annotations

from pathlib import Path

from geko.schemas import GemmType


def gemm_type_string(config: dict) -> str:
    """Return gemm_name-style string from legacy YAML dtype/transpose keys."""
    return GemmType.format_gemm_name(
        config["DataType"],
        config["DestDataType"],
        config["ComputeDataType"],
        config["TRANSA"],
        config["TRANSB"],
    )


def assert_output_artifacts(
    output_dir: str,
    gemm_type: str,
    *,
    expect_shell_scripts: bool = True,
) -> None:
    """Validate generator output under *output_dir* (filesystem path as a string)."""
    root = Path(output_dir)
    assert root.is_dir(), f"Missing output dir {output_dir}"
    assert (root / "MI_finder_log").is_dir(), "MI_finder_log directory missing"

    yamls = sorted(root.glob(f"{gemm_type}_*.yaml"))
    assert len(yamls) >= 1, f"Expected at least one {gemm_type}_*.yaml under {output_dir}"

    if expect_shell_scripts:
        shs = sorted(root.glob(f"{gemm_type}_*.sh"))
        assert len(yamls) == len(shs), (
            f"Mismatched yaml ({len(yamls)}) vs sh ({len(shs)}) entity scripts"
        )

        run_all = root / f"run_{gemm_type}_all.sh"
        assert run_all.is_file(), f"Missing {run_all.name}"
        run_all_text = run_all.read_text(encoding="utf-8", errors="replace")
        assert "#!/bin/bash" in run_all_text
        assert "Auto-generated run-all script" in run_all_text
        for y in yamls:
            assert f"./{y.stem}.sh\n" in run_all_text or f"./{y.stem}.sh" in run_all_text
    else:
        shs = []
        assert not list(root.glob(f"{gemm_type}_*.sh")), (
            f"Expected no {gemm_type}_*.sh when expect_shell_scripts is false"
        )
        run_all = root / f"run_{gemm_type}_all.sh"
        assert not run_all.is_file(), f"Did not expect {run_all.name}"

    log_path = root.parent / f"Config_{gemm_type}.log"
    assert log_path.is_file(), f"Expected Config log beside output dir: {log_path}"

    for ypath in yamls:
        text = ypath.read_text(encoding="utf-8", errors="replace")
        for keyword in (
            "GlobalParameters",
            "BenchmarkProblems",
            "LibraryLogic",
            "ForkParameters",
            "Groups",
        ):
            assert keyword in text, f"{ypath.name} missing section/keyword {keyword!r}"

    if expect_shell_scripts:
        for spath in shs:
            st = spath.read_text(encoding="utf-8", errors="replace")
            assert "#!/bin/bash" in st
            assert spath.stem in st
            assert "YAML=" in st and "yaml" in st.lower()
            assert "tensilelite" in st
