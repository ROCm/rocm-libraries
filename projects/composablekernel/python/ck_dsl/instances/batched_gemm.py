# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Batched GEMM instance builder (CK Tile ``16_batched_gemm`` parity).

This is the DSL-side counterpart of CK Tile's
``example/ck_tile/16_batched_gemm``. It re-uses the universal GEMM
kernel body verbatim, in batched mode:

    grid = (N_tiles, M_tiles, batch_count)
    kernel reads block_id_z as batch_idx
    A_offset = batch_idx * stride_a + per-tile A index
    B_offset = batch_idx * stride_b + per-tile B index
    C_offset = batch_idx * stride_c + per-tile C index

The strides are passed as additional ``i32`` kernel arguments so we can
support irregular per-batch layouts (e.g. a 3D ``(B, M, K)`` torch
tensor where ``stride_a = M*K`` packed, or one with extra row padding).

The MFMA / LDS body is the same as ``build_universal_gemm`` (which is
already battle-tested by the GEMM bake-off). This makes batched GEMM
inherit all of universal_gemm's perf knobs:
``pipeline in {mem, compv3, compv4}``, ``epilogue in {default,
cshuffle}``, warp grids up to 4x4, MFMA atoms 16x16x{16,32} and
32x32x{8,16}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from ..core.ir import KernelDef
from .gemm_universal import (
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
    is_valid_spec as is_valid_gemm_spec,
)


@dataclass(frozen=True)
class BatchedGemmSpec:
    """One batched GEMM kernel instance.

    ``batch_size`` is informational (it's an upper bound on the
    ``block_id_z`` dimension; the actual launch grid passes the real
    batch count). It's used by the dispatcher to skip configs that
    won't fit, but the kernel itself doesn't bake it in.
    """

    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=TraitSpec)
    wave_size: int = 64
    block_size: int = 0
    batch_size: int = 0

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def to_universal_spec(self) -> UniversalGemmSpec:
        return UniversalGemmSpec(
            name=self.name,
            tile=self.tile,
            trait=self.trait,
            wave_size=self.wave_size,
            block_size=self.block_size,
            batched=True,
        )

    def kernel_name(self) -> str:
        return self.to_universal_spec().kernel_name()


def is_valid_spec(spec: BatchedGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    return is_valid_gemm_spec(spec.to_universal_spec(), arch=arch)


def build_batched_gemm(spec: BatchedGemmSpec) -> KernelDef:
    """Build the IR for one batched GEMM instance.

    Kernel signature:
      ``(A: ptr, B: ptr, C: ptr, M: i32, N: i32, K: i32,
         stride_a: i32, stride_b: i32, stride_c: i32)``

    Grid layout: ``(ceil_div(N, tile_n), ceil_div(M, tile_m), batch)``.
    Block layout: ``(block_size, 1, 1)``.
    """
    return build_universal_gemm(spec.to_universal_spec())


def batched_gemm_signature(spec: BatchedGemmSpec):
    return [
        {"name": "A", "type": "ptr<f16, global>"},
        {"name": "B", "type": "ptr<f16, global>"},
        {"name": "C", "type": "ptr<f16, global>"},
        {"name": "M", "type": "i32"},
        {"name": "N", "type": "i32"},
        {"name": "K", "type": "i32"},
        {"name": "stride_a", "type": "i32"},
        {"name": "stride_b", "type": "i32"},
        {"name": "stride_c", "type": "i32"},
    ]


def batched_gemm_grid(
    batch: int, m: int, n: int, spec: BatchedGemmSpec
) -> Tuple[int, int, int]:
    t = spec.tile
    return (
        (n + t.tile_n - 1) // t.tile_n,
        (m + t.tile_m - 1) // t.tile_m,
        batch,
    )
