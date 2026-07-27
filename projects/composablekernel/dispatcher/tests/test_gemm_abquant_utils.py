#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for gemm_abquant_utils.py.

No GPU, no hipcc, no .so compilation required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))

import pytest
from gemm_abquant_utils import (
    ABQuantKernelConfig,
    ABQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
)


class TestKernelName:
    def test_fp8_rcr_default_name(self):
        cfg = default_fp8_config()
        name = cfg.name
        assert name.startswith("gemm_abquant_fp8_rcr_")
        assert "cshuffle" in name
        assert "intrawave" in name

    def test_bf8_rcr_name(self):
        assert "gemm_abquant_bf8_rcr" in default_bf8_config().name

    def test_name_contains_gsn(self):
        # gsn{N} is always in name, even for default gsn1
        assert "_gsn1_" in default_fp8_config().name

    def test_name_contains_gsn_when_n_gt_1(self):
        cfg = default_fp8_config(quant_group_n=16)
        assert "_gsn16_" in cfg.name

    def test_name_contains_tile_dims(self):
        cfg = default_fp8_config()
        assert "16x64x256" in cfg.name
        assert "1x4x1" in cfg.name
        assert "16x16x128" in cfg.name

    def test_no_preshuffle_by_default(self):
        cfg = default_fp8_config()
        # preshuffle_a (APreshuffleQuant) and preshuffle_b (BPreshuffleQuant)
        assert "_False_False_gsn" in cfg.name

    def test_preshuffle_a_in_name(self):
        cfg = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_a=True,
        )
        assert "_True_False_gsn" in cfg.name

    def test_preshuffle_b_in_name(self):
        cfg = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_b=True,
        )
        assert "_False_True_gsn" in cfg.name

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

    def test_quant_group_k(self):
        d = default_fp8_config(quant_group_k=64).to_codegen_config()
        assert d["quant_group_k"] == 64

    def test_quant_group_n(self):
        d = default_fp8_config(quant_group_n=16).to_codegen_config()
        assert d["quant_group_n"] == 16

    def test_preshuffle_defaults_false(self):
        d = default_fp8_config().to_codegen_config()
        assert d["preshuffle_a"] is False
        assert d["preshuffle_b"] is False

    def test_preshuffle_a_true(self):
        cfg = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_a=True,
        )
        d = cfg.to_codegen_config()
        assert d["preshuffle_a"] is True
        assert d["preshuffle_b"] is False


class TestABQuantGemmProblem:
    def test_qk_a_exact(self):
        prob = ABQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128)
        assert prob.QK_A == 2

    def test_qk_b_equals_qk_a(self):
        prob = ABQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128)
        assert prob.QK_B == prob.QK_A

    def test_qn_b_exact(self):
        prob = ABQuantGemmProblem(M=16, N=64, K=256, quant_group_n=16)
        assert prob.QN_B == 4  # 64/16

    def test_qn_b_default_no_n_grouping(self):
        prob = ABQuantGemmProblem(M=16, N=64, K=256, quant_group_n=1)
        assert prob.QN_B == 64  # 64/1

    def test_qk_a_ceil(self):
        prob = ABQuantGemmProblem(M=16, N=64, K=200, quant_group_k=128)
        assert prob.QK_A == 2  # ceil(200/128)

    def test_k_batch_default(self):
        prob = ABQuantGemmProblem(M=16, N=64, K=256)
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

    def test_default_quant_group_n_is_1(self):
        assert default_fp8_config().quant_group_n == 1

    def test_custom_quant_group_n(self):
        cfg = default_fp8_config(quant_group_n=16)
        assert cfg.quant_group_n == 16

    def test_all_default_no_preshuffle(self):
        for cfg in [default_fp8_config(), default_bf8_config()]:
            assert cfg.preshuffle_a is False
            assert cfg.preshuffle_b is False

    def test_fp8_bf8_names_unique(self):
        assert default_fp8_config().name != default_bf8_config().name

    def test_gfx_arch_stored(self):
        cfg = default_fp8_config(gfx_arch="gfx942")
        assert cfg.gfx_arch == "gfx942"


class TestNameUniqueness:
    def test_different_group_n_produces_different_names(self):
        cfg1 = default_fp8_config(quant_group_n=1)
        cfg16 = default_fp8_config(quant_group_n=16)
        assert cfg1.name != cfg16.name

    def test_preshuffle_variants_unique(self):
        base = default_fp8_config()
        pa = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_a=True,
        )
        pb = ABQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_b=True,
        )
        names = {base.name, pa.name, pb.name}
        assert len(names) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
