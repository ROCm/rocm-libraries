# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MFMA atoms: shape + lane layout + IR dispatch.

The `MfmaAtom` dataclass collapses everything a kernel author needs to
know about a single matrix-multiply-accumulate intrinsic into one
object:

  - The (m, n, k) shape of the matrix tile this MFMA computes.
  - The per-lane operand widths (a_per_lane, b_per_lane, c_per_lane).
    On wave64 these equal m*k/64, k*n/64, m*n/64 respectively, which
    determines how big a vector each lane has to load and how big the
    accumulator is. This is the number that drives VGPR pressure.
  - The dispatch to the right `IRBuilder` method, hiding the
    `b.mfma_f32_16x16x16_f16` vs `b.mfma_f32_4x4x4_f16` vs ... choice
    behind one `atom.emit(b, a, b, c)` call.
  - The lane -> output (row, col) mapping that the epilogue uses to
    figure out where each accumulator slot belongs in the result tile.
    AMD's output layouts are not uniform across atom shapes (16x16
    atoms have one layout, 32x32 atoms split the M dimension across
    accumulator slot index, 4x4 atoms put a whole independent batch
    on lane >> 2), so this is the most error-prone piece of any GEMM
    or direct-conv epilogue.

CK Tile uses MfmaAtom equivalents per (in_dtype, m, n, k) tuple
(`mfma_type` in `mfma_instr.hpp`). We keep the same structure here so
once we expose bf16 and fp8 we can re-use the same names without
reworking the kernel builders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ..core.ir import (
    IRBuilder,
    Value,
)


@dataclass(frozen=True)
class MfmaAtom:
    """One MFMA intrinsic with all the metadata a kernel author needs.

    Construct via the class methods (`MfmaAtom.f16_16x16x16()`, etc.)
    or via `mfma_atom("f16", 16, 16, 16)` which is the lookup-by-shape
    helper.

    Lane-output mapping convention (for the 4-tuple `lane_to_output`):
      Given a per-lane `lane: i32` (0..63 on wave64) and a per-lane
      accumulator slot index `i` (0..c_per_lane-1), the helper returns
      the (row_offset_within_atom, col_offset_within_atom) of that
      output element.
    """

    m: int
    n: int
    k: int
    a_per_lane: int
    b_per_lane: int
    c_per_lane: int
    dtype_in: str
    dtype_out: str
    """Logical name used in error messages and the manifest schema."""
    name: str

    # ---- factory class methods (the only supported atoms today) ----

    @classmethod
    def f16_16x16x16(cls) -> "MfmaAtom":
        """The legacy CDNA f16 atom. K=16/atom, c_per_lane=4 floats.

        Per-lane layout on wave64:
          A: <4 x half>, B: <4 x half>, C: <4 x float>
        Lane mapping:
          lane = (k_blk * 16 + m_in_atom)
            with k_blk = lane / 16 ∈ {0..3},
                 m_in_atom = lane % 16 ∈ {0..15}
          A lane holds K = [k_blk*4 : k_blk*4 + 4]
          C lane[i] -> output (m_blk * 4 + i, n_in_atom)
            with m_blk = lane / 16, n_in_atom = lane % 16
        """
        return cls(
            m=16,
            n=16,
            k=16,
            a_per_lane=4,
            b_per_lane=4,
            c_per_lane=4,
            dtype_in="f16",
            dtype_out="f32",
            name="mfma_f32_16x16x16_f16",
        )

    @classmethod
    def f16_16x16x32(cls) -> "MfmaAtom":
        """K-packed f16 atom on gfx950+ (CDNA3). K=32/atom in two halves.

        Per-lane layout on wave64:
          A: <8 x half>, B: <8 x half>, C: <4 x float>
        K-pack lane mapping (per runbook §7.2):
          A lane `c4 = lane / 16` holds K = [c4 * 8 : c4 * 8 + 8]
          (NOT the flat-concat layout [c4*4 : c4*4 + 4] + [c4*4 +
          16 : c4*4 + 20]; the wrong packing compiles, runs, and
          validates within 1e-2 but fails at 1e-3).
        Output layout: same as 16x16x16 (`(m_blk*4 + i, n_in_atom)`).
        """
        return cls(
            m=16,
            n=16,
            k=32,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=4,
            dtype_in="f16",
            dtype_out="f32",
            name="mfma_f32_16x16x32_f16",
        )

    @classmethod
    def f16_32x32x8(cls) -> "MfmaAtom":
        """The canonical 32x32 f16 atom (every CK dispatcher default tile uses it).

        Per-lane layout on wave64:
          A: <4 x half>, B: <4 x half>, C: <16 x float>
        Lane mapping:
          lane = (k_blk * 32 + m_in_atom)
            with k_blk = lane / 32 ∈ {0,1},
                 m_in_atom = lane % 32 ∈ {0..31}
          A lane holds K = [k_blk*4 : k_blk*4 + 4]
          C lane[i] -> output:
            row = (i // 4) * 8 + (lane / 32) * 4 + (i % 4)
            col = lane % 32
          (16 outputs per lane spread over a 32x32 tile.)
        """
        return cls(
            m=32,
            n=32,
            k=8,
            a_per_lane=4,
            b_per_lane=4,
            c_per_lane=16,
            dtype_in="f16",
            dtype_out="f32",
            name="mfma_f32_32x32x8_f16",
        )

    @classmethod
    def f16_32x32x16(cls) -> "MfmaAtom":
        """K-packed 32x32 f16 atom on gfx950+. K=16/atom.

        Per-lane layout on wave64:
          A: <8 x half>, B: <8 x half>, C: <16 x float>
        Output layout: same as 32x32x8.
        """
        return cls(
            m=32,
            n=32,
            k=16,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=16,
            dtype_in="f16",
            dtype_out="f32",
            name="mfma_f32_32x32x16_f16",
        )

    @classmethod
    def f16_4x4x4(cls) -> "MfmaAtom":
        """The tiny f16 atom. One MFMA emits 16 independent 4x4x4 matmuls per wave.

        This is what our small-channel direct-conv path uses: 16 groups
        of 4-channel direct convolutions in one MFMA, indexed by
        `batch = lane / 4`.

        Per-lane layout on wave64:
          A: <4 x half>, B: <4 x half>, C: <4 x float>
          batch_idx = lane / 4 ∈ {0..15}
          lane_in_batch = lane % 4 ∈ {0..3}
          A holds the 4 K-elements of row `lane_in_batch` of matrix A
          B holds the 4 K-elements of column `lane_in_batch` of matrix B
          C lane[i] -> output (i, lane_in_batch) of independent 4x4 #batch_idx.
        """
        return cls(
            m=4,
            n=4,
            k=4,
            a_per_lane=4,
            b_per_lane=4,
            c_per_lane=4,
            dtype_in="f16",
            dtype_out="f32",
            name="mfma_f32_4x4x4_f16",
        )

    # ---- emit ----

    def emit(self, b: IRBuilder, a: Value, bb: Value, c: Value) -> Value:
        """Issue one MFMA at this atom's shape."""
        if (self.m, self.n, self.k, self.dtype_in) == (16, 16, 16, "f16"):
            return b.mfma_f32_16x16x16_f16(a, bb, c)
        if (self.m, self.n, self.k, self.dtype_in) == (16, 16, 32, "f16"):
            return b.mfma_f32_16x16x32_f16(a, bb, c)
        if (self.m, self.n, self.k, self.dtype_in) == (32, 32, 8, "f16"):
            return b.mfma_f32_32x32x8_f16(a, bb, c)
        if (self.m, self.n, self.k, self.dtype_in) == (32, 32, 16, "f16"):
            return b.mfma_f32_32x32x16_f16(a, bb, c)
        if (self.m, self.n, self.k, self.dtype_in) == (4, 4, 4, "f16"):
            return b.mfma_f32_4x4x4_f16(a, bb, c)
        raise NotImplementedError(
            f"no MFMA dispatch for atom {self.dtype_in} {self.m}x{self.n}x{self.k}"
        )

    def zero_acc(self, b: IRBuilder) -> Value:
        """Allocate a fresh `<c_per_lane x float>` accumulator (all zeros).

        This is the accumulator initial value that the K-loop carries
        through `scf.for_iter`'s `iter_args`.
        """
        return b.zero_vec_f32(self.c_per_lane)

    # ---- output lane layout ----

    def lane_to_output(
        self,
        b: IRBuilder,
        lane: Value,
        i: int,
    ) -> Tuple[Value, Value]:
        """Per-lane (row_offset, col_offset) of accumulator slot `i` within one atom.

        The result is `(row_in_atom, col_in_atom)` ∈ [0, m) × [0, n).
        Combined with the warp-base offsets (`warp_m_off`, `warp_n_off`)
        and the block-base offsets (`block_m_off`, `block_n_off`) this
        gives the final global row/col.

        16x16 atoms (c_per_lane=4):
          m_blk = lane / 16
          n_in_atom = lane % 16
          row = m_blk * 4 + i
          col = n_in_atom

        32x32 atoms (c_per_lane=16):
          m_blk = lane / 32  (0 or 1)
          n_in_atom = lane % 32
          row = (i // 4) * 8 + m_blk * 4 + (i % 4)
          col = n_in_atom

        4x4 atoms (c_per_lane=4):
          # All 16 batches share the same (row,col) layout within their
          # own 4x4. Caller composes `batch_idx = lane / 4` separately.
          lane_in_batch = lane % 4
          row = i
          col = lane_in_batch
        """
        if (self.m, self.n) == (16, 16):
            c_atom_n = b.const_i32(self.n)
            n_in_atom = b.mod(lane, c_atom_n)
            m_blk = b.div(lane, c_atom_n)
            row = b.add(b.mul(m_blk, b.const_i32(self.c_per_lane)), b.const_i32(i))
            return row, n_in_atom
        if (self.m, self.n) == (32, 32):
            c_atom_n = b.const_i32(self.n)
            n_in_atom = b.mod(lane, c_atom_n)
            m_blk = b.div(lane, c_atom_n)
            rb = i // 4
            ri = i % 4
            row = b.add(
                b.add(b.const_i32(rb * 8), b.mul(m_blk, b.const_i32(4))),
                b.const_i32(ri),
            )
            return row, n_in_atom
        if (self.m, self.n) == (4, 4):
            c4 = b.const_i32(4)
            lane_in_batch = b.mod(lane, c4)
            return b.const_i32(i), lane_in_batch
        raise NotImplementedError(
            f"no lane_to_output dispatch for atom {self.m}x{self.n}"
        )


# ---------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------


MFMA_F16_ATOMS: Tuple[MfmaAtom, ...] = (
    MfmaAtom.f16_4x4x4(),
    MfmaAtom.f16_16x16x16(),
    MfmaAtom.f16_16x16x32(),
    MfmaAtom.f16_32x32x8(),
    MfmaAtom.f16_32x32x16(),
)

_BY_SHAPE: Dict[Tuple[str, int, int, int], MfmaAtom] = {
    (a.dtype_in, a.m, a.n, a.k): a for a in MFMA_F16_ATOMS
}


def mfma_atom(dtype: str, m: int, n: int, k: int) -> MfmaAtom:
    """Lookup an atom by (dtype_in, m, n, k). Raises if unknown."""
    key = (dtype, m, n, k)
    if key not in _BY_SHAPE:
        valid = sorted((a.dtype_in, a.m, a.n, a.k) for a in MFMA_F16_ATOMS)
        raise ValueError(f"no MFMA atom for {key}; valid: {valid}")
    return _BY_SHAPE[key]
