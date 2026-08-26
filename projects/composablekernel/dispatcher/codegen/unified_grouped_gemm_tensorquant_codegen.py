#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm TensorQuant Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking vector<QuantGroupedGemmHostArgs> — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_tensorquant_ctypes_lib.cpp

TensorQuant: A and B each have a single scalar scale (one scale per tensor).
AQDataType=BQDataType=float; kernel uses QuantType::TensorQuant.

Naming convention (byte-exact with TensorQuantKernelConfig.name in grouped_gemm_tensorquant_utils.py):
    grouped_gemm_tensorquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {padm}_{padn}_{padk}_{persistent}_{TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}

The generator itself lives in ``grouped_scalar_quant_codegen``: this op and its
sibling rowcolquant are the same generator, differing only in the three values in
``TENSORQUANT_OP`` below.  This module keeps its own name so the emitted
"DO NOT EDIT — regenerate via" line, the CLI and the ``_default_config`` /
``_build_specs`` imports all keep naming the right script.

Reference:
    tile_engine/ops/gemm/grouped_gemm_quant/grouped_gemm_tensorquant/grouped_gemm_tensorquant_instance_builder.py
"""

from typing import List

from codegen_common import make_tensorquant_kernel_name, run_codegen_cli
from grouped_scalar_quant_codegen import (
    ScalarQuantKernelHeaderGenerator,
    ScalarQuantKernelSpec,
    ScalarQuantOp,
    ScalarQuantTileConfig,
    build_specs,
    default_config,
    generate_kernels_generic,
)

# make_tensorquant_kernel_name lives in codegen_common alongside the aquant/bquant/
# abquant builders and is re-exported here so existing
# `from unified_grouped_gemm_tensorquant_codegen import make_tensorquant_kernel_name`
# imports keep working.
__all__ = ["make_tensorquant_kernel_name", "generate_kernels", "main"]


TENSORQUANT_OP = ScalarQuantOp(
    op_name="tensorquant",
    display_name="TensorQuant",
    op_label="GroupedTensorQuant",
    quant_type="ck_tile::QuantType::TensorQuant",
    codegen_script="unified_grouped_gemm_tensorquant_codegen.py",
    description="TensorQuant Grouped GEMM kernel header generator",
    aq_bq_note="AQ layout is RowMajor, BQ layout is ColumnMajor (follows B convention; nominal for single-scalar quant)",
    make_kernel_name=make_tensorquant_kernel_name,
)

# Kept for callers that imported the per-op names.
TensorQuantTileConfig = ScalarQuantTileConfig


class TensorQuantKernelSpec(ScalarQuantKernelSpec):
    op = TENSORQUANT_OP


class TensorQuantKernelHeaderGenerator(ScalarQuantKernelHeaderGenerator):
    op = TENSORQUANT_OP


def _default_config() -> dict:
    return default_config(TENSORQUANT_OP)


def _build_specs(config: dict) -> List["TensorQuantKernelSpec"]:
    return build_specs(TENSORQUANT_OP, TensorQuantKernelSpec, config)


def generate_kernels(output_dir, config=None, parallel: bool = True) -> List:
    """Generate all GroupedTensorQuant kernel headers into output_dir.

    Returns list of generated .hpp paths.
    """
    return generate_kernels_generic(
        op_label="GroupedTensorQuant",
        generator=TensorQuantKernelHeaderGenerator(),
        specs=_build_specs(config or _default_config()),
        output_dir=output_dir,
        parallel=parallel,
    )


def main() -> int:
    return run_codegen_cli(
        description=TENSORQUANT_OP.description,
        op_label=TENSORQUANT_OP.op_label,
        make_generator=TensorQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
