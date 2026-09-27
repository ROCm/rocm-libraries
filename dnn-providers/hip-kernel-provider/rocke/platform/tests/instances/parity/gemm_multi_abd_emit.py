#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_multi_abd_emit.py -- Python reference emitter for the
# multi-A/B/D GEMM parity harness. Selects one of 6 sampled configs by argv[1]
# (the config index 0..5), builds the GemmMultiAbdSpec, builds the kernel via
# build_gemm_multi_abd(spec, arch="gfx950") and prints
# lower_kernel_to_llvm(kernel, arch="gfx950") to stdout so it can be
# byte-compared with the C emitter gemm_multi_abd_emit.c.
from rocke.instances.common.gemm_multi_abd import (
    GemmMultiAbdSpec,
    build_gemm_multi_abd,
)
from rocke.instances.common.gemm_universal import (
    UniversalGemmSpec,
    TileSpec,
    TraitSpec,
    DataSpec,
)
from _emit_common import run_emit


def _spec(idx: int) -> GemmMultiAbdSpec:
    if idx == 0:
        return GemmMultiAbdSpec(
            base=UniversalGemmSpec(
                name="test",
                tile=TileSpec(tile_m=128, tile_n=128, tile_k=32, warp_m=2, warp_n=2),
                trait=TraitSpec(
                    pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
                ),
                data=DataSpec(
                    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
                ),
            ),
            a_operands=(("A", "fp16"),),
            b_operands=(("B", "fp16"),),
            d_operands=(),
            d_dtype="fp16",
            d_load_kind="vector",
        )
    if idx == 1:
        return GemmMultiAbdSpec(
            base=UniversalGemmSpec(
                name="test",
                tile=TileSpec(tile_m=64, tile_n=64, tile_k=32, warp_m=1, warp_n=1),
                trait=TraitSpec(
                    pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
                ),
                data=DataSpec(
                    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
                ),
            ),
            a_operands=(("A", "fp16"),),
            b_operands=(("B", "fp16"),),
            d_operands=(("D0", "add"),),
            d_dtype="fp16",
            d_load_kind="vector",
        )
    if idx == 2:
        return GemmMultiAbdSpec(
            base=UniversalGemmSpec(
                name="test",
                tile=TileSpec(tile_m=256, tile_n=128, tile_k=64, warp_m=4, warp_n=2),
                trait=TraitSpec(
                    pipeline="compv3", scheduler="intrawave", epilogue="cshuffle"
                ),
                data=DataSpec(
                    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
                ),
            ),
            a_operands=(("A", "fp16"),),
            b_operands=(("B", "fp16"),),
            d_operands=(("D0", "add"), ("D1", "mul")),
            d_dtype="fp16",
            d_load_kind="tiled",
        )
    if idx == 3:
        return GemmMultiAbdSpec(
            base=UniversalGemmSpec(
                name="test",
                tile=TileSpec(tile_m=64, tile_n=128, tile_k=32, warp_m=1, warp_n=2),
                trait=TraitSpec(
                    pipeline="mem", scheduler="intrawave", epilogue="cshuffle"
                ),
                data=DataSpec(
                    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
                ),
            ),
            a_operands=(("A", "fp16"),),
            b_operands=(("B", "fp16"),),
            d_operands=(("D0", "add"),),
            d_dtype="fp16",
            d_load_kind="stock",
        )
    if idx == 4:
        return GemmMultiAbdSpec(
            base=UniversalGemmSpec(
                name="test",
                tile=TileSpec(tile_m=192, tile_n=192, tile_k=64, warp_m=3, warp_n=3),
                trait=TraitSpec(
                    pipeline="compv4", scheduler="interwave", epilogue="cshuffle"
                ),
                data=DataSpec(
                    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
                ),
            ),
            a_operands=(("A", "fp16"),),
            b_operands=(("B", "fp16"),),
            d_operands=(("D0", "mul"),),
            d_dtype="fp16",
            d_load_kind="vector",
        )
    if idx == 5:
        return GemmMultiAbdSpec(
            base=UniversalGemmSpec(
                name="test",
                tile=TileSpec(tile_m=128, tile_n=64, tile_k=16, warp_m=2, warp_n=1),
                trait=TraitSpec(
                    pipeline="compv4", scheduler="intrawave", epilogue="cshuffle"
                ),
                data=DataSpec(
                    dtype_a="fp16", dtype_b="fp16", dtype_c="fp16", dtype_acc="fp32"
                ),
            ),
            a_operands=(("A", "fp16"),),
            b_operands=(("B", "fp16"),),
            d_operands=(("D0", "add"), ("D1", "add"), ("D2", "mul")),
            d_dtype="fp16",
            d_load_kind="vector",
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_gemm_multi_abd,
        usage="usage: gemm_multi_abd_emit.py <config_index 0..5>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
