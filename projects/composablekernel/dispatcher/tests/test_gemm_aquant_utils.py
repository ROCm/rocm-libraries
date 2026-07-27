#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for gemm_aquant_utils.py.

No GPU, no hipcc, no .so compilation required.
Tests cover kernel name generation, codegen config round-trips,
and problem dimension helpers.
"""

import sys
from pathlib import Path

# Ensure dispatcher/python and dispatcher/codegen are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codegen"))

import pytest
from gemm_aquant_utils import (
    AQuantKernelConfig,
    AQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
    default_fp8i4_config,
    default_bf8i4_config,
)


class TestKernelName:
    def test_fp8_rcr_default_name(self):
        cfg = default_fp8_config()
        name = cfg.name
        assert name.startswith("gemm_aquant_fp8_rcr_")
        # Verify the format: gemm_aquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_...
        parts = name.split("_")
        assert parts[0] == "gemm"
        assert parts[1] == "aquant"
        assert parts[2] == "fp8"
        assert parts[3] == "rcr"
        assert "cshuffle" in name
        assert "intrawave" in name

    def test_bf8_rcr_name(self):
        cfg = default_bf8_config()
        assert "gemm_aquant_bf8_rcr" in cfg.name

    def test_fp8i4_name(self):
        cfg = default_fp8i4_config()
        assert "gemm_aquant_fp8i4_rcr" in cfg.name

    def test_bf8i4_name(self):
        cfg = default_bf8i4_config()
        assert "gemm_aquant_bf8i4_rcr" in cfg.name

    def test_preshuffle_quant_suffix(self):
        cfg = AQuantKernelConfig(
            variant_key="fp8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_quant=True,
        )
        assert "True" in cfg.name

    def test_no_preshuffle_quant_by_default(self):
        cfg = default_fp8_config()
        # 7th trait slot (preshuffle_quant) should be "False"
        # Format: gemm_aquant_fp8_rcr_compv3_cshuffle_intrawave_False_False_True_False_...
        assert "_False_False_True_False_" in cfg.name

    def test_name_contains_tile_dims(self):
        cfg = default_fp8_config()
        assert "16x64x256" in cfg.name
        assert "1x4x1" in cfg.name
        assert "16x16x128" in cfg.name

    def test_name_only_ascii(self):
        cfg = default_fp8_config()
        assert cfg.name.isascii()

    def test_fp8_and_bf8_names_differ(self):
        assert default_fp8_config().name != default_bf8_config().name

    def test_fp8i4_and_fp8_names_differ(self):
        assert default_fp8i4_config().name != default_fp8_config().name

    def test_name_deterministic(self):
        cfg = default_fp8_config()
        assert cfg.name == cfg.name


class TestCodegenConfig:
    def test_codegen_config_variant_key(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["variant_keys"] == ["fp8"]

    def test_codegen_config_layout(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["layouts"] == ["rcr"]

    def test_codegen_config_tile_params(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == 16
        assert tc["tile_n"] == 64
        assert tc["tile_k"] == 256
        assert tc["warp_m"] == 1
        assert tc["warp_n"] == 4
        assert tc["warp_tile_k"] == 128

    def test_codegen_config_pipeline(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["pipeline"] == "compv3"
        assert d["scheduler"] == "intrawave"

    def test_codegen_config_quant_group(self):
        cfg = default_fp8_config(quant_group_k=64)
        d = cfg.to_codegen_config()
        assert d["quant_group_k"] == 64

    def test_codegen_config_preshuffle_false_default(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["preshuffle_quant"] is False

    def test_codegen_config_preshuffle_true(self):
        cfg = AQuantKernelConfig(
            variant_key="fp8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_quant=True,
        )
        d = cfg.to_codegen_config()
        assert d["preshuffle_quant"] is True

    def test_codegen_config_has_single_tile(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert len(d["tile_configs"]) == 1

    def test_codegen_config_has_single_variant(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert len(d["variant_keys"]) == 1


class TestAQuantGemmProblem:
    def test_qk_a_exact_division(self):
        prob = AQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128)
        assert prob.QK_A == 2  # ceil(256/128)

    def test_qk_a_ceil_division(self):
        prob = AQuantGemmProblem(M=16, N=64, K=200, quant_group_k=128)
        assert prob.QK_A == 2  # ceil(200/128)

    def test_qk_a_single_group(self):
        prob = AQuantGemmProblem(M=16, N=64, K=64, quant_group_k=128)
        assert prob.QK_A == 1  # ceil(64/128)

    def test_qk_a_group64(self):
        prob = AQuantGemmProblem(M=16, N=64, K=256, quant_group_k=64)
        assert prob.QK_A == 4  # ceil(256/64)

    def test_k_batch_default(self):
        prob = AQuantGemmProblem(M=16, N=64, K=256)
        assert prob.k_batch == 1

    def test_k_batch_explicit(self):
        prob = AQuantGemmProblem(M=16, N=64, K=256, k_batch=2)
        assert prob.k_batch == 2


class TestDefaultConfigs:
    def test_fp8_variant(self):
        assert default_fp8_config().variant_key == "fp8"

    def test_bf8_variant(self):
        assert default_bf8_config().variant_key == "bf8"

    def test_fp8i4_variant(self):
        assert default_fp8i4_config().variant_key == "fp8i4"

    def test_bf8i4_variant(self):
        assert default_bf8i4_config().variant_key == "bf8i4"

    def test_all_default_rcr_layout(self):
        for cfg in [
            default_fp8_config(),
            default_bf8_config(),
            default_fp8i4_config(),
            default_bf8i4_config(),
        ]:
            assert cfg.layout == "rcr"

    def test_all_default_compv3_pipeline(self):
        for cfg in [
            default_fp8_config(),
            default_bf8_config(),
            default_fp8i4_config(),
            default_bf8i4_config(),
        ]:
            assert cfg.pipeline == "compv3"

    def test_fp8_warp_tile_k_128(self):
        # get_k_warp_tile<fp8_t, 16>() = 128 on gfx950
        assert default_fp8_config().warp_tile_k == 128

    def test_bf8_warp_tile_k_128(self):
        assert default_bf8_config().warp_tile_k == 128

    def test_fp8i4_warp_tile_k_16(self):
        # pk_int4 is not 8-bit float -> warp_tile_k=16
        assert default_fp8i4_config().warp_tile_k == 16

    def test_bf8i4_warp_tile_k_16(self):
        assert default_bf8i4_config().warp_tile_k == 16

    def test_all_default_preshuffle_false(self):
        for cfg in [
            default_fp8_config(),
            default_bf8_config(),
            default_fp8i4_config(),
            default_bf8i4_config(),
        ]:
            assert cfg.preshuffle_quant is False

    def test_all_four_names_unique(self):
        names = [
            default_fp8_config().name,
            default_bf8_config().name,
            default_fp8i4_config().name,
            default_bf8i4_config().name,
        ]
        assert len(set(names)) == 4

    def test_custom_quant_group_k(self):
        cfg = default_fp8_config(quant_group_k=64)
        # Quant group k is embedded in the config but not in the name
        assert cfg.quant_group_k == 64

    def test_gfx_arch_stored(self):
        cfg = default_fp8_config(gfx_arch="gfx942")
        assert cfg.gfx_arch == "gfx942"


class TestNameUniqueness:
    def test_preshuffle_and_non_preshuffle_differ(self):
        no_pq = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_quant=False,
        )
        with_pq = AQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
            preshuffle_quant=True,
        )
        assert no_pq.name != with_pq.name

    def test_different_quant_group_k_same_name(self):
        # quant_group_k does NOT appear in the kernel name (it's a compile-time baked constant)
        # Different QK values with same tile should produce same name
        cfg128 = default_fp8_config(quant_group_k=128)
        cfg64  = default_fp8_config(quant_group_k=64)
        # Both have the same tile and dtype, quant_group_k differs but name doesn't include it
        # (name is determined by tile shape, dtype, layout, pipeline, etc.)
        assert cfg128.name == cfg64.name  # quant_group_k not in name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
