#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for gemm_rowcolquant_utils.py.

No GPU, no hipcc, no .so compilation required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))

import pytest
from gemm_rowcolquant_utils import (
    RowColQuantKernelConfig,
    RowColQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
)


class TestKernelName:
    def test_fp8_rcr_default_name(self):
        cfg = default_fp8_config()
        name = cfg.name
        assert name.startswith("gemm_rowcolquant_fp8_rcr_")
        assert "cshuffle" in name
        assert "intrawave" in name

    def test_bf8_rcr_name(self):
        assert "gemm_rowcolquant_bf8_rcr" in default_bf8_config().name

    def test_name_contains_persistent_false(self):
        # Persistent slot is always False for rowcolquant
        cfg = default_fp8_config()
        # Format: ..._False_False_True_False_tile
        assert "_False_False_True_False_" in cfg.name

    def test_name_contains_tile_dims(self):
        cfg = default_fp8_config()
        assert "16x64x256" in cfg.name
        assert "1x4x1" in cfg.name
        assert "16x16x128" in cfg.name

    def test_name_only_ascii(self):
        assert default_fp8_config().name.isascii()

    def test_fp8_and_bf8_differ(self):
        assert default_fp8_config().name != default_bf8_config().name

    def test_name_deterministic(self):
        cfg = default_fp8_config()
        assert cfg.name == cfg.name


class TestCodegenConfig:
    def test_variant_key(self):
        d = default_fp8_config().to_codegen_config()
        assert d["variant_keys"] == ["fp8"]

    def test_layout(self):
        d = default_fp8_config().to_codegen_config()
        assert d["layouts"] == ["rcr"]

    def test_tile_params(self):
        d = default_fp8_config().to_codegen_config()
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == 16
        assert tc["tile_n"] == 64
        assert tc["tile_k"] == 256
        assert tc["warp_tile_k"] == 128

    def test_pipeline(self):
        d = default_fp8_config().to_codegen_config()
        assert d["pipeline"] == "compv3"
        assert d["scheduler"] == "intrawave"

    def test_no_quant_groups_in_config(self):
        d = default_fp8_config().to_codegen_config()
        # RowColQuant doesn't use quant_group_k / quant_group_n
        assert "quant_group_k" not in d
        assert "quant_group_n" not in d

    def test_single_tile(self):
        d = default_fp8_config().to_codegen_config()
        assert len(d["tile_configs"]) == 1


class TestRowColQuantGemmProblem:
    def test_qk_a_always_1(self):
        prob = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert prob.QK_A == 1

    def test_qk_b_always_1(self):
        prob = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert prob.QK_B == 1

    def test_k_batch_default(self):
        prob = RowColQuantGemmProblem(M=16, N=64, K=256)
        assert prob.k_batch == 1

    def test_k_batch_explicit(self):
        prob = RowColQuantGemmProblem(M=16, N=64, K=256, k_batch=2)
        assert prob.k_batch == 2

    def test_large_dims(self):
        prob = RowColQuantGemmProblem(M=1024, N=4096, K=8192)
        assert prob.QK_A == 1
        assert prob.QK_B == 1


class TestDefaultConfigs:
    def test_fp8_variant(self):
        assert default_fp8_config().variant_key == "fp8"

    def test_bf8_variant(self):
        assert default_bf8_config().variant_key == "bf8"

    def test_rcr_layout(self):
        for cfg in [default_fp8_config(), default_bf8_config()]:
            assert cfg.layout == "rcr"

    def test_compv3_pipeline(self):
        for cfg in [default_fp8_config(), default_bf8_config()]:
            assert cfg.pipeline == "compv3"

    def test_fp8_warp_tile_k_128(self):
        assert default_fp8_config().warp_tile_k == 128

    def test_bf8_warp_tile_k_128(self):
        assert default_bf8_config().warp_tile_k == 128

    def test_fp8_bf8_names_unique(self):
        assert default_fp8_config().name != default_bf8_config().name

    def test_gfx_arch_stored(self):
        cfg = default_fp8_config(gfx_arch="gfx942")
        assert cfg.gfx_arch == "gfx942"

    def test_custom_tile(self):
        cfg = RowColQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=32, tile_n=128, tile_k=256,
            warp_m=2, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        )
        assert "32x128x256" in cfg.name
        assert "2x4x1" in cfg.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
