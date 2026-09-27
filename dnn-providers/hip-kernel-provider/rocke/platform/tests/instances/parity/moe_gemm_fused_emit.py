#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/moe_gemm_fused_emit.py -- Python reference emitter for the
# moe_gemm_fused parity harness. Selects one of N sampled spec configs by
# argv[1], builds the matching Fused* spec, builds the kernel via the matching
# build_moe_* entry and prints lower_kernel_to_llvm(arch='gfx950') to stdout so
# it can be byte-compared with the C emitter moe_gemm_fused_emit.c.
from rocke.instances.common.gemm_universal import TileSpec, TraitSpec
from rocke.instances.common.moe_gemm_fused import (
    FusedGateUpSiluGemmSpec,
    FusedInterleavedGateUpSiluGemmSpec,
    FusedDownReduceGemmSpec,
    build_moe_gate_up_silu_gemm,
    build_moe_interleaved_gate_up_silu_gemm,
    build_moe_down_reduce_gemm,
)
from _emit_common import run_emit


def _build(idx: int):
    if idx == 0:
        spec = FusedGateUpSiluGemmSpec(
            name="moe_gate_up_silu_f16",
            tile=TileSpec(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pad_m=True, pad_n=True),
            dtype="f16",
        )
        return build_moe_gate_up_silu_gemm(spec, arch="gfx950")
    if idx == 1:
        spec = FusedGateUpSiluGemmSpec(
            name="moe_gate_up_silu_grouped_f16",
            tile=TileSpec(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pad_m=True, pad_n=True),
            dtype="f16",
            grouped=True,
        )
        return build_moe_gate_up_silu_gemm(spec, arch="gfx950")
    if idx == 2:
        spec = FusedInterleavedGateUpSiluGemmSpec(
            name="moe_interleaved_gate_up_silu_f16",
            tile=TileSpec(
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pad_m=True, pad_n=True),
            dtype="f16",
        )
        return build_moe_interleaved_gate_up_silu_gemm(spec, arch="gfx950")
    if idx == 3:
        spec = FusedDownReduceGemmSpec(
            name="moe_down_reduce_f16",
            tile=TileSpec(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            trait=TraitSpec(pad_m=True, pad_n=True),
            dtype="f16",
        )
        return build_moe_down_reduce_gemm(spec, arch="gfx950")
    if idx == 4:
        spec = FusedInterleavedGateUpSiluGemmSpec(
            name="moe_interleaved_bf16_grouped",
            tile=TileSpec(
                tile_m=32,
                tile_n=32,
                tile_k=32,
                warp_m=2,
                warp_n=2,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
            ),
            dtype="bf16",
            grouped=True,
        )
        return build_moe_interleaved_gate_up_silu_gemm(spec, arch="gfx950")
    if idx == 5:
        spec = FusedDownReduceGemmSpec(
            name="moe_down_grouped",
            tile=TileSpec(
                tile_m=32,
                tile_n=128,
                tile_k=64,
                warp_m=1,
                warp_n=2,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
            ),
            dtype="f16",
            grouped=True,
        )
        return build_moe_down_reduce_gemm(spec, arch="gfx950")
    raise SystemExit(f"unknown config index {idx}")


def _emit_build(kernel, arch=None):
    return kernel


def main() -> int:
    return run_emit(
        _build,
        _emit_build,
        usage="usage: moe_gemm_fused_emit.py <config_index> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
