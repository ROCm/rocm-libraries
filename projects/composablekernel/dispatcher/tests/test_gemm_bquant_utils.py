#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for gemm_bquant_utils.py (block_scale_gemm operator, gemm_bquant_* prefix).

Distinct from test_grouped_gemm_bquant_utils.py which tests the grouped_gemm_bquant_* naming.
No GPU, no hipcc, no .so compilation required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))

import pytest
from gemm_bquant_utils import (
    GemmBQuantKernelConfig,
    GemmBQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
)


class TestKernelName:
    def test_fp8_rcr_default_name_prefix(self):
        cfg = default_fp8_config()
        name = cfg.name
        # Must start with "gemm_bquant_" (NOT "grouped_gemm_bquant_")
        assert name.startswith("gemm_bquant_fp8_rcr_")
        assert not name.startswith("grouped_gemm_bquant_")

    def test_bf8_rcr_name(self):
        assert "gemm_bquant_bf8_rcr" in default_bf8_config().name
        assert not default_bf8_config().name.startswith("grouped_")

    def test_default_epilogue_is_default(self):
        # block_scale_gemm bquant uses "default" epilogue (DefaultGemm2DEpilogue)
        cfg = default_fp8_config()
        assert "_default_" in cfg.name

    def test_name_contains_tile_dims(self):
        cfg = default_fp8_config()
        assert "16x64x256" in cfg.name
        assert "1x4x1" in cfg.name
        assert "16x16x128" in cfg.name

    def test_preshuffle_false_in_name(self):
        # 7th slot: BPreshuffleQuant = False
        cfg = default_fp8_config()
        assert "_False_" in cfg.name

    def test_preshuffle_true_in_name(self):
        cfg = GemmBQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="default", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_bquant=True,
        )
        assert "_True_" in cfg.name

    def test_name_only_ascii(self):
        assert default_fp8_config().name.isascii()

    def test_fp8_and_bf8_differ(self):
        assert default_fp8_config().name != default_bf8_config().name

    def test_name_deterministic(self):
        cfg = default_fp8_config()
        assert cfg.name == cfg.name

    def test_distinct_from_grouped_gemm_bquant(self):
        from grouped_gemm_bquant_utils import default_fp8_config as grp_fp8
        assert default_fp8_config().name != grp_fp8().name


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

    def test_epilogue_default(self):
        d = default_fp8_config().to_codegen_config()
        assert d["epilogue"] == "default"

    def test_pipeline_compv3(self):
        d = default_fp8_config().to_codegen_config()
        assert d["pipeline"] == "compv3"

    def test_quant_group_k(self):
        d = default_fp8_config(quant_group_k=64).to_codegen_config()
        assert d["quant_group_k"] == 64

    def test_preshuffle_false_default(self):
        d = default_fp8_config().to_codegen_config()
        assert d["preshuffle_bquant"] is False


class TestGemmBQuantGemmProblem:
    def test_qk_b_exact(self):
        prob = GemmBQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128)
        assert prob.QK_B == 2

    def test_qk_b_ceil(self):
        prob = GemmBQuantGemmProblem(M=16, N=64, K=200, quant_group_k=128)
        assert prob.QK_B == 2

    def test_qn_b_default(self):
        prob = GemmBQuantGemmProblem(M=16, N=64, K=256, quant_group_n=1)
        assert prob.QN_B == 64

    def test_qn_b_grouped(self):
        prob = GemmBQuantGemmProblem(M=16, N=64, K=256, quant_group_n=16)
        assert prob.QN_B == 4

    def test_k_batch_default(self):
        prob = GemmBQuantGemmProblem(M=16, N=64, K=256)
        assert prob.k_batch == 1


class TestDefaultConfigs:
    def test_fp8_variant(self):
        assert default_fp8_config().variant_key == "fp8"

    def test_bf8_variant(self):
        assert default_bf8_config().variant_key == "bf8"

    def test_default_rcr_layout(self):
        for cfg in [default_fp8_config(), default_bf8_config()]:
            assert cfg.layout == "rcr"

    def test_default_compv3_pipeline(self):
        for cfg in [default_fp8_config(), default_bf8_config()]:
            assert cfg.pipeline == "compv3"

    def test_fp8_warp_tile_k_128(self):
        assert default_fp8_config().warp_tile_k == 128

    def test_bf8_warp_tile_k_128(self):
        assert default_bf8_config().warp_tile_k == 128

    def test_all_default_no_preshuffle(self):
        for cfg in [default_fp8_config(), default_bf8_config()]:
            assert cfg.preshuffle_bquant is False

    def test_fp8_bf8_names_unique(self):
        assert default_fp8_config().name != default_bf8_config().name

    def test_gfx_arch_stored(self):
        cfg = default_fp8_config(gfx_arch="gfx942")
        assert cfg.gfx_arch == "gfx942"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
