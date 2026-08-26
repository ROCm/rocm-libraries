#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm RowColQuant Code Generator

Generates one .hpp per kernel config for the dispatcher's ctypes path.
Each header defines a SelectedKernel struct with a static launch() method
taking vector<QuantGroupedGemmHostArgs> — compiled per-kernel via force-include:

    hipcc -include <kernel.hpp> -DCK_TILE_SINGLE_KERNEL_INCLUDE grouped_gemm_rowcolquant_ctypes_lib.cpp

RowColQuant: A has per-row scales (AQ shape [M, 1]), B has per-column scales
(BQ shape [1, N]). Both ADataType and BDataType are fp8 or bf8;
AQDataType=BQDataType=float.

Naming convention (byte-exact with RowColQuantKernelConfig.name in grouped_gemm_rowcolquant_utils.py):
    grouped_gemm_rowcolquant_{dtype}_{layout}_{pipeline}_{epilogue}_{scheduler}_
    {padm}_{padn}_{padk}_{persistent}_{TileM}x{TileN}x{TileK}_{WarpM}x{WarpN}x{WarpK}_{WtM}x{WtN}x{WtK}

The generator itself lives in ``grouped_scalar_quant_codegen``: this op and its
sibling tensorquant are the same generator, differing only in the three values in
``ROWCOLQUANT_OP`` below.  This module keeps its own name so the emitted
"DO NOT EDIT — regenerate via" line, the CLI and the ``_default_config`` /
``_build_specs`` imports all keep naming the right script.

Reference:
    tile_engine/ops/gemm/grouped_gemm_quant/grouped_gemm_rowcolquant/grouped_gemm_rowcolquant_instance_builder.py
"""

from typing import List

from codegen_common import make_rowcolquant_kernel_name, run_codegen_cli
from grouped_scalar_quant_codegen import (
    ScalarQuantKernelHeaderGenerator,
    ScalarQuantKernelSpec,
    ScalarQuantOp,
    ScalarQuantTileConfig,
    build_specs,
    default_config,
    generate_kernels_generic,
)

# make_rowcolquant_kernel_name lives in codegen_common alongside the aquant/bquant/
# abquant builders and is re-exported here so existing
# `from unified_grouped_gemm_rowcolquant_codegen import make_rowcolquant_kernel_name`
# imports keep working.
__all__ = ["make_rowcolquant_kernel_name", "generate_kernels", "main"]


ROWCOLQUANT_OP = ScalarQuantOp(
    op_name="rowcolquant",
    display_name="RowColQuant",
    op_label="GroupedRowColQuant",
    quant_type="ck_tile::QuantType::RowColQuant",
    codegen_script="unified_grouped_gemm_rowcolquant_codegen.py",
    description="RowColQuant Grouped GEMM kernel header generator",
    aq_bq_note="AQ is RowMajor [M,1]; BQ is ColMajor [1,N]",
    make_kernel_name=make_rowcolquant_kernel_name,
)

# Kept for callers that imported the per-op names.
RowColQuantTileConfig = ScalarQuantTileConfig


class RowColQuantKernelSpec(ScalarQuantKernelSpec):
    op = ROWCOLQUANT_OP


class RowColQuantKernelHeaderGenerator(ScalarQuantKernelHeaderGenerator):
    op = ROWCOLQUANT_OP


def _default_config() -> dict:
    return default_config(ROWCOLQUANT_OP)


def _build_specs(config: dict) -> List["RowColQuantKernelSpec"]:
    return build_specs(ROWCOLQUANT_OP, RowColQuantKernelSpec, config)


def generate_kernels(output_dir, config=None, parallel: bool = True) -> List:
    """Generate all GroupedRowColQuant kernel headers into output_dir.

    Returns list of generated .hpp paths.
    """
    return generate_kernels_generic(
        op_label="GroupedRowColQuant",
        generator=RowColQuantKernelHeaderGenerator(),
        specs=_build_specs(config or _default_config()),
        output_dir=output_dir,
        parallel=parallel,
    )


def main() -> int:
    return run_codegen_cli(
        description=ROWCOLQUANT_OP.description,
        op_label=ROWCOLQUANT_OP.op_label,
        make_generator=RowColQuantKernelHeaderGenerator,
        build_specs=_build_specs,
        default_config=_default_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
