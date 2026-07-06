#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
CPU-only unit tests for grouped_gemm_bquant_utils.py.

Tests kernel name generation, config serialization, and problem dimension helpers.
No GPU or hipcc required.

Run:
    python3 -m pytest dispatcher/tests/test_grouped_gemm_bquant_utils.py -v
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from grouped_gemm_bquant_utils import (
    BQuantKernelConfig,
    BQuantGemmProblem,
    default_fp8_config,
    default_bf8_config,
)


# =============================================================================
# BQuantKernelConfig.name — byte-exact match with codegen KERNEL_NAME
# =============================================================================


class TestKernelName:

    def test_fp8_rcr_default_name(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_bquant_fp8_rcr_compv3_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_qg1x1x128"
        )

    def test_bf8_rcr_default_name(self):
        cfg = BQuantKernelConfig(
            variant_key="bf8",
            layout="rcr",
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
        )
        assert cfg.name == (
            "grouped_gemm_bquant_bf8_rcr_compv3_cshuffle_intrawave_"
            "16x64x256_1x4x1_16x16x16_qg1x1x128"
        )

    def test_different_quant_groups_produce_different_names(self):
        def make(gk, gn):
            return BQuantKernelConfig(
                variant_key="fp8", layout="rcr",
                pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
                tile_m=16, tile_n=64, tile_k=256,
                warp_m=1, warp_n=4, warp_k=1,
                warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
                quant_group_m=1, quant_group_n=gn, quant_group_k=gk,
            ).name

        names = [make(64, 1), make(128, 1), make(128, 8), make(128, 128)]
        assert len(names) == len(set(names)), "All quant-group variants must have unique names"

    def test_preshuffle_b_suffix(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_b=True,
        )
        assert cfg.name.endswith("_preshuffleb")

    def test_preshuffle_bquant_suffix(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_b=True, preshuffle_bquant=True,
        )
        assert "_preshuffleb_preshufflebq" in cfg.name

    def test_preshuffle_bquant_only_suffix(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=16, tile_n=64, tile_k=256,
            warp_m=1, warp_n=4, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=1, quant_group_k=128,
            preshuffle_bquant=True,
        )
        # preshuffle_bquant without preshuffle_b: _preshufflebq present but not _preshuffleb_
        # Use word-boundary check: the suffix "_preshuffleb" must not appear as a standalone token
        name = cfg.name
        assert "_preshufflebq" in name
        # "_preshuffleb" is a prefix of "_preshufflebq" — check it's not a standalone suffix
        assert not name.endswith("_preshuffleb")

    def test_name_no_spaces(self):
        cfg = default_fp8_config()
        assert " " not in cfg.name

    def test_name_only_valid_chars(self):
        import re
        cfg = default_fp8_config()
        assert re.match(r'^[a-z0-9_]+$', cfg.name), f"Invalid chars in name: {cfg.name}"

    def test_default_fp8_config_name(self):
        cfg = default_fp8_config(quant_group_k=128)
        assert "fp8" in cfg.name
        assert "qg1x1x128" in cfg.name

    def test_default_bf8_config_name(self):
        cfg = default_bf8_config(quant_group_k=128)
        assert "bf8" in cfg.name
        assert "qg1x1x128" in cfg.name


# =============================================================================
# BQuantKernelConfig.to_codegen_config — round-trip shape
# =============================================================================


class TestCodegenConfig:

    def test_codegen_config_contains_correct_variant(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["variant_keys"] == ["fp8"]

    def test_codegen_config_tile_roundtrip(self):
        cfg = BQuantKernelConfig(
            variant_key="fp8", layout="rcr",
            pipeline="compv3", epilogue="cshuffle", scheduler="intrawave",
            tile_m=32, tile_n=128, tile_k=64,
            warp_m=2, warp_n=2, warp_k=1,
            warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
            quant_group_m=1, quant_group_n=4, quant_group_k=64,
        )
        d = cfg.to_codegen_config()
        tc = d["tile_configs"][0]
        assert tc["tile_m"] == 32
        assert tc["tile_n"] == 128
        assert tc["tile_k"] == 64
        assert tc["warp_m"] == 2
        qg = d["quant_groups"][0]
        assert qg["quant_group_k"] == 64
        assert qg["quant_group_n"] == 4

    def test_codegen_config_single_layout(self):
        cfg = default_fp8_config()
        d = cfg.to_codegen_config()
        assert d["layouts"] == ["rcr"]


# =============================================================================
# BQuantGemmProblem dimension helpers
# =============================================================================


class TestBQuantGemmProblem:

    def test_QK_B_exact_multiple(self):
        p = BQuantGemmProblem(M=16, N=64, K=256, quant_group_k=128)
        assert p.QK_B == 2   # 256 / 128

    def test_QK_B_ceil(self):
        p = BQuantGemmProblem(M=16, N=64, K=300, quant_group_k=128)
        assert p.QK_B == math.ceil(300 / 128)  # == 3

    def test_QN_B_exact_multiple(self):
        p = BQuantGemmProblem(M=16, N=64, K=256, quant_group_n=32)
        assert p.QN_B == 2   # 64 / 32

    def test_QN_B_default_one_group(self):
        p = BQuantGemmProblem(M=16, N=64, K=256)
        assert p.QN_B == 64  # default quant_group_n=1 → every column is its own group

    def test_QN_B_ceil(self):
        p = BQuantGemmProblem(M=16, N=65, K=256, quant_group_n=32)
        assert p.QN_B == math.ceil(65 / 32)  # == 3

    def test_default_k_batch(self):
        p = BQuantGemmProblem(M=16, N=64, K=256)
        assert p.k_batch == 1


# =============================================================================
# Name uniqueness across a small sweep
# =============================================================================


class TestNameUniqueness:

    def _make_configs(self):
        configs = []
        for variant in ("fp8", "bf8"):
            for gk in (64, 128):
                for gn in (1, 8):
                    configs.append(BQuantKernelConfig(
                        variant_key=variant,
                        layout="rcr",
                        pipeline="compv3",
                        epilogue="cshuffle",
                        scheduler="intrawave",
                        tile_m=16, tile_n=64, tile_k=256,
                        warp_m=1, warp_n=4, warp_k=1,
                        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
                        quant_group_m=1, quant_group_n=gn, quant_group_k=gk,
                    ))
        return configs

    def test_all_names_unique(self):
        configs = self._make_configs()
        names = [c.name for c in configs]
        assert len(names) == len(set(names)), f"Duplicate names: {[n for n in names if names.count(n) > 1]}"
