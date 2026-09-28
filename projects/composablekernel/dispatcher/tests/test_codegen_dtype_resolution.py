#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
UnifiedGemmCodegen must resolve the arch-validation dtype triple exactly: a
mapped --datatype (fp32) is validated as itself, and an unmapped one (pk_fp4)
raises naming the value instead of being silently validated as fp16.

Run: python3 -m pytest -q tests/test_codegen_dtype_resolution.py
"""

import sys
from pathlib import Path

import pytest

DISPATCHER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))

import unified_gemm_codegen as ugc  # noqa: E402
from arch_filter import ELEMENT_SIZE_MAP  # noqa: E402
from codegen_common import CommonTypeMappings, TileConfig  # noqa: E402


class _RecordingFilter:
    """Stands in for ArchFilter and records the dtypes it is asked about."""

    def __init__(self):
        self.calls = []

    def is_kernel_valid(self, **kwargs):
        self.calls.append(kwargs)
        return True


def _codegen(tmp_path, datatype):
    return ugc.UnifiedGemmCodegen(
        output_dir=tmp_path, datatype=datatype, layout="rcr", gpu_target="gfx942"
    )


def test_unmapped_datatype_raises_naming_value(tmp_path):
    with pytest.raises(ValueError, match="'pk_fp4'"):
        _codegen(tmp_path, "pk_fp4")


def test_show_arch_info_unmapped_datatype_raises():
    with pytest.raises(ValueError, match="'pk_fp4'"):
        ugc._show_arch_info("gfx942", "pk_fp4")


@pytest.mark.skipif(not ugc.HAS_ARCH_FILTER, reason="arch_filter not importable")
def test_fp32_validated_as_fp32(tmp_path):
    codegen = _codegen(tmp_path, "fp32")
    codegen.arch_filter = _RecordingFilter()
    tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 8)

    assert codegen._is_tile_arch_valid(tile)

    (call,) = codegen.arch_filter.calls
    triple = (call["datatype_a"], call["datatype_b"], call["datatype_c"])
    assert triple == ("fp32", "fp32", "fp32")
    assert ELEMENT_SIZE_MAP.get(call["datatype_a"], 2) == 4
    assert ELEMENT_SIZE_MAP.get(call["datatype_b"], 2) == 4


@pytest.mark.parametrize("dtype", CommonTypeMappings.ARCH_VALIDATION_DTYPES)
def test_every_mapped_dtype_resolves_to_itself(dtype):
    a, b, acc = CommonTypeMappings.get_arch_dtype_triple(dtype)
    assert (a, b) == (dtype, dtype)
    assert acc == CommonTypeMappings.get_acc_dtype(dtype)


@pytest.mark.skipif(not ugc.HAS_ARCH_FILTER, reason="arch_filter not importable")
@pytest.mark.parametrize("gpu_target,listed", [("gfx942", True), ("gfx1250", False)])
def test_fp32_needs_listed_warp_tiles(tmp_path, gpu_target, listed):
    # An arch whose table has no fp32 warp-tile entry must reject fp32 tiles
    # rather than let the arch filter pass every warp tile unchecked.
    codegen = ugc.UnifiedGemmCodegen(
        output_dir=tmp_path, datatype="fp32", layout="rcr", gpu_target=gpu_target
    )
    codegen.arch_filter = _RecordingFilter()
    tile = TileConfig(64, 64, 32, 2, 2, 1, 16, 16, 4)

    assert codegen._is_tile_arch_valid(tile, variant=ugc.GemmVariant.STANDARD) is listed
    assert len(codegen.arch_filter.calls) == int(listed)
