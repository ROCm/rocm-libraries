# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMA-tiled GEMM kernel (production CK Tile pattern).

This is the **first MFMA-based instance** in the kernel set: a real
f16 GEMM that consumes the ``mfma_f32_16x16x16_f16`` atom directly
rather than emitting scalar FMUL / FADD per output cell.

Why it matters:

* CDNA3 (MI300X) ``mfma_f32_16x16x16_f16`` emits **one MFMA per 4
  cycles** for a 16x16x16 matmul; that's 256 MAC ops per atom or
  **64 FLOPS/cycle/lane**. The scalar-inner v1 in
  :mod:`streamk_gemm` emits one FMUL + one FADD per K-element which
  is 2 FLOPS/cycle/lane. The MFMA path is **~32x denser** in
  FLOPS for the same K-loop trip count.
* CK Tile's GEMM hero kernels all bottom out on this atom (or its
  fp8 / bf8 siblings); matching it is the "beat CK Tile" baseline.

The kernel's per-CTA structure:

1. Grid = ``(N // 16, M // 16, 1)`` -- one CTA per 16x16 output tile.
2. Block_size = 64 (one wave64 warp).
3. Each lane owns:
   * **A**: a ``<4 x f16>`` per K-iter (4 K-elements at lane row).
   * **B**: a ``<4 x f16>`` per K-iter (4 K-elements at lane col).
   * **C**: a ``<4 x f32>`` accumulator across the whole K-loop.
4. K-loop: ``scf.for k_blk in [0, K // 16)``: each iter loads the
   16x16x16 A/B slab into per-lane vector regs, fires one MFMA,
   accumulates into C.
5. Epilogue: each lane writes 4 output cells of C[m_tile*16:m_tile*16+16,
   n_tile*16:n_tile*16+16] -- cell ``(m_blk * 4 + i, n_in_atom)``
   where ``m_blk = lane // 16`` and ``n_in_atom = lane % 16``.

Limitations of v1:

* **head shape only 16x16** (one atom); larger M/N tiles need warp-
  level tiling + cshuffle epilogue (v2).
* **f16 only**; bf16 / fp8 / bf8 siblings are one-line extensions
  via the corresponding ``MfmaAtom`` factory.
* **No LDS staging**; all A/B loads come from global memory. The
  L2-cached path is enough for parity vs the scalar inner; LDS
  staging + async DMA is the v2 perf hoist.
* **Persistent + StreamK split-K** lives in :mod:`streamk_gemm`; this
  kernel is the dense one-CTA-per-tile baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import F16, I32, IRBuilder, KernelDef, PtrType
from ..helpers.atoms import MfmaAtom
from ..helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    load_a_row_major_contiguous,
    load_b_col_strided_scalars,
    mfma_k_loop,
    store_acc_to_global,
)
from ..helpers.spec import SignatureBuilder, kernel_name_join


DType = Literal["f16"]


@dataclass(frozen=True)
class MfmaGemmSpec:
    """One concrete MFMA GEMM kernel configuration.

    ``M`` / ``N`` / ``K`` are compile-time so the partitioner can
    statically derive the grid + K-loop trip count. v1 ships a single
    16x16 output tile (one MFMA atom per CTA); the v2 hoist adds
    warp-level tiling (32x32 / 64x64) on the same spec surface.
    """

    M: int
    N: int
    K: int
    dtype: DType = "f16"
    name: str = "ck_dsl_mfma_gemm"

    @property
    def atom(self) -> MfmaAtom:
        return MfmaAtom.f16_16x16x16()

    @property
    def tile_m(self) -> int:
        return self.atom.m

    @property
    def tile_n(self) -> int:
        return self.atom.n

    @property
    def tile_k(self) -> int:
        return self.atom.k

    @property
    def block_size(self) -> int:
        # One wave64 warp per CTA -- the MFMA atom is per-wave.
        return 64

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            f"M{self.M}N{self.N}K{self.K}",
            self.dtype,
            f"atom{self.tile_m}x{self.tile_n}x{self.tile_k}",
        )


def is_valid_spec(spec: MfmaGemmSpec) -> Tuple[bool, str]:
    if spec.dtype != "f16":
        return False, f"v1 ships f16 only, got {spec.dtype!r}"
    if spec.M % spec.tile_m or spec.N % spec.tile_n or spec.K % spec.tile_k:
        return False, (
            f"M / N / K must be divisible by the 16x16x16 atom shape; "
            f"got M={spec.M}, N={spec.N}, K={spec.K}"
        )
    return True, "ok"


def build_mfma_gemm(spec: MfmaGemmSpec) -> KernelDef:
    """Build a 16x16x16-tile f16 MFMA GEMM kernel.

    Kernel signature: ``(A: ptr<f16>, B: ptr<f16>, C: ptr<f16>,
    M: i32, N: i32, K: i32)``.

    Grid: ``(N // 16, M // 16, 1)``. Block: 64 threads (one warp).
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid mfma_gemm spec: {why}")

    atom = spec.atom
    BS = spec.block_size

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    _M = b.param("M", I32)  # noqa: F841 - ABI
    _N = b.param("N", I32)  # noqa: F841 - ABI
    _K = b.param("K", I32)  # noqa: F841 - ABI

    lane = b.thread_id_x()
    bid_n = b.block_id_x()
    bid_m = b.block_id_y()
    m_tile_base = b.mul(bid_m, b.const_i32(atom.m))
    n_tile_base = b.mul(bid_n, b.const_i32(atom.n))

    lane_decode = decode_mfma_lanes(b, atom, lane)
    c_atom_k = b.const_i32(atom.k)

    def _load_a(b, kt):
        return load_a_row_major_contiguous(
            b,
            A=A,
            atom=atom,
            lane_decode=lane_decode,
            m_tile_base=m_tile_base,
            k_tile_base=b.mul(kt, c_atom_k),
            K=spec.K,
        )

    def _load_b(b, kt):
        return load_b_col_strided_scalars(
            b,
            B=Bp,
            atom=atom,
            lane_decode=lane_decode,
            n_tile_base=n_tile_base,
            k_tile_base=b.mul(kt, c_atom_k),
            N=spec.N,
        )

    acc_final = mfma_k_loop(
        b,
        K=spec.K,
        atom=atom,
        load_a=_load_a,
        load_b=_load_b,
    )

    store_acc_to_global(
        b,
        C=C,
        atom=atom,
        lane_decode=lane_decode,
        m_tile_base=m_tile_base,
        n_tile_base=n_tile_base,
        acc=acc_final,
        N=spec.N,
        out_dtype="f16",
    )

    b.ret()
    return b.kernel


def mfma_gemm_grid(spec: MfmaGemmSpec) -> Tuple[int, int, int]:
    n_tiles = spec.N // spec.tile_n
    m_tiles = spec.M // spec.tile_m
    return (n_tiles, m_tiles, 1)


def mfma_gemm_signature(spec: MfmaGemmSpec):
    return (
        SignatureBuilder()
        .ptr("A", spec.dtype)
        .ptr("B", spec.dtype)
        .ptr("C", spec.dtype)
        .scalar("M", "i32")
        .scalar("N", "i32")
        .scalar("K", "i32")
        .build()
    )


__all__ = [
    "MfmaGemmSpec",
    "build_mfma_gemm",
    "is_valid_spec",
    "mfma_gemm_grid",
    "mfma_gemm_signature",
]
