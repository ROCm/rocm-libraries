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

"""Tests for log/workload-driven config (SIZE_OPTION 2, GEMM_LOG_PATH)."""

from __future__ import annotations

from pathlib import Path

import pytest

from geko.config_generator.load_input_config import (
    apply_input_config_defaults,
    gemm_configs_from_gemm_log_path,
    load_prepared_config_from_yaml,
    validate_input_config,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_prepared_requires_arch_for_gemm_log_mode() -> None:
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    with pytest.raises(ValueError, match="No ARCH defined"):
        load_prepared_config_from_yaml(None, gemm_log_path=str(wf))


def test_validate_log_mode_minimal() -> None:
    """Log-driven mode skips TRANSA/dtypes; ARCH is validated before defaults."""
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    assert wf.is_file()
    cfg = {
        "ARCH": "gfx950",
        "GEMM_LOG_PATH": str(wf),
        "SIZE_OPTION": 2,
    }
    validate_input_config(cfg)


def test_validate_size_option_two_requires_path() -> None:
    with pytest.raises(ValueError, match="SIZE_OPTION 2 requires GEMM_LOG_PATH"):
        validate_input_config({"ARCH": "gfx950", "SIZE_OPTION": 2})


def test_gemm_configs_missing_log_file() -> None:
    """Parser raises when the workload file path does not exist."""
    missing = _repo_root() / "nonexistent_workload.yaml"
    assert not missing.is_file()
    with pytest.raises(FileNotFoundError):
        gemm_configs_from_gemm_log_path(missing)


def test_gemm_configs_from_workload_yaml() -> None:
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    gcs = gemm_configs_from_gemm_log_path(wf)
    assert len(gcs) >= 1
    assert all(len(gc.sizes) >= 1 for gc in gcs)


def test_apply_defaults_log_mode_roundtrip() -> None:
    """Defaults fill tuning keys; GemmProblems are not set by apply_input_config_defaults alone."""
    wf = _repo_root() / "tests" / "test_data" / "workload.yaml"
    cfg = {"ARCH": "gfx950", "GEMM_LOG_PATH": str(wf), "SIZE_OPTION": 2}
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    assert "GA" in cfg
    assert cfg.get("GemmProblems") is None
