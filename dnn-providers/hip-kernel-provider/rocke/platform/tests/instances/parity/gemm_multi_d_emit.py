#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gemm_multi_d_emit.py -- Python reference emitter for the
# gemm_multi_d parity harness. Selects one of the sampled GemmMultiDSpec
# configs by argv[1], builds the kernel via build_gemm_multi_d and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared
# with the C emitter gemm_multi_d_emit.c.
from rocke.instances.common.gemm_multi_d import (
    GemmMultiDSpec,
    build_gemm_multi_d,
)
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
)
from _emit_common import run_emit


def _spec(idx: int) -> GemmMultiDSpec:
    if idx == 0:
        return GemmMultiDSpec(
            base=UniversalGemmSpec(
                name="gemmmd1",
                tile=TileSpec(128, 128, 64, 2, 2),
                trait=TraitSpec(epilogue="cshuffle"),
                data=DataSpec(dtype_a="fp16"),
            ),
            d_operands=(("D0", "add"),),
            d_dtype="fp16",
            d_load_kind="vector",
        )
    if idx == 1:
        return GemmMultiDSpec(
            base=UniversalGemmSpec(
                name="gemmmd2",
                tile=TileSpec(64, 64, 64, 1, 2),
                trait=TraitSpec(epilogue="cshuffle"),
                data=DataSpec(),
            ),
            d_operands=(("D0", "add"), ("D1", "mul")),
            d_dtype="fp16",
            d_load_kind="tiled",
        )
    if idx == 2:
        return GemmMultiDSpec(
            base=UniversalGemmSpec(
                name="gemmmd3",
                tile=TileSpec(256, 128, 128, 4, 2),
                trait=TraitSpec(epilogue="cshuffle"),
                data=DataSpec(),
            ),
            d_operands=(("D0", "add"), ("D1", "add"), ("D2", "mul")),
            d_dtype="fp16",
            d_load_kind="vector",
        )
    if idx == 3:
        return GemmMultiDSpec(
            base=UniversalGemmSpec(
                name="gemmmd4",
                tile=TileSpec(192, 192, 64, 2, 2),
                trait=TraitSpec(epilogue="cshuffle"),
                data=DataSpec(),
            ),
            d_operands=(("D0", "mul"), ("D1", "add")),
            d_dtype="fp16",
            d_load_kind="vector",
        )
    if idx == 4:
        return GemmMultiDSpec(
            base=UniversalGemmSpec(
                name="gemmmd5",
                tile=TileSpec(128, 128, 128, 2, 2),
                trait=TraitSpec(epilogue="cshuffle"),
                data=DataSpec(),
            ),
            d_operands=(("D0", "add"), ("D1", "mul"), ("D2", "add"), ("D3", "mul")),
            d_dtype="fp16",
            d_load_kind="tiled",
        )
    if idx == 5:
        return GemmMultiDSpec(
            base=UniversalGemmSpec(
                name="gemmmd6",
                tile=TileSpec(256, 256, 32, 4, 4),
                trait=TraitSpec(epilogue="cshuffle"),
                data=DataSpec(),
            ),
            d_operands=(("D0", "add"),),
            d_dtype="fp16",
            d_load_kind="stock",
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_gemm_multi_d,
        usage="usage: gemm_multi_d_emit.py <config_index 0..5>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
