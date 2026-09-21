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

from ..core.arch import ArchTarget, LayoutMap, MmaOp
from ..core.ir import (
    IRBuilder,
    Value,
)
from .distribution import TileDistributionEncoding


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

    # ---- FP8 / BF8 atoms (gfx940+) ----
    #
    # FP8E4M3 and BF8E5M2 share the same MFMA shape catalog: the
    # 16x16x32 atom (small-tile, 4-float accumulator) and the
    # 32x32x16 atom (canonical hero, 16-float accumulator). Both pack
    # 8 fp8 bytes per lane = ``<2 x i32>`` at the LLVM intrinsic
    # boundary. The mixed-precision variants (.fp8.bf8 / .bf8.fp8)
    # are reachable by bitcasting the operand vectors and calling
    # the same intrinsic family; we expose the homogeneous variants
    # as atoms today and treat mixed as a manual op.

    @classmethod
    def fp8_16x16x32(cls) -> "MfmaAtom":
        """FP8 (e4m3) MFMA, 16x16 output, K=32 per atom.

        Per-lane: A = <8 x fp8e4m3>, B = <8 x fp8e4m3>,
        C = <4 x float>.
        Output layout: same as ``f16_16x16x32`` -- 4 floats per lane,
        row = ``(m_blk * 4 + i)``, col = ``n_in_atom``.
        """
        return cls(
            m=16,
            n=16,
            k=32,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=4,
            dtype_in="fp8e4m3",
            dtype_out="f32",
            name="mfma_f32_16x16x32_fp8",
        )

    @classmethod
    def bf8_16x16x32(cls) -> "MfmaAtom":
        """BF8 (e5m2) sibling of :meth:`fp8_16x16x32`."""
        return cls(
            m=16,
            n=16,
            k=32,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=4,
            dtype_in="bf8e5m2",
            dtype_out="f32",
            name="mfma_f32_16x16x32_bf8",
        )

    @classmethod
    def fp8_32x32x16(cls) -> "MfmaAtom":
        """FP8 MFMA at the 32x32 hero tile, K=16 per atom.

        Per-lane: A = <8 x fp8e4m3>, B = <8 x fp8e4m3>,
        C = <16 x float>.
        Output layout: same as ``f16_32x32x16`` -- 16 floats per lane
        spread across 4 row-groups of 4 floats each.
        """
        return cls(
            m=32,
            n=32,
            k=16,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=16,
            dtype_in="fp8e4m3",
            dtype_out="f32",
            name="mfma_f32_32x32x16_fp8",
        )

    @classmethod
    def fp8_16x16x128(cls) -> "MfmaAtom":
        """Unscaled FP8 (e4m3) MFMA hero atom, 16x16 output, K=128 per atom.

        This is the dense, *unscaled* wide-K f8 MFMA — the throughput
        ceiling for fp8 GEMM (4x the K of :meth:`fp8_16x16x32`, so 4x
        fewer K-trips). gfx950 exposes the K=128 f8 MFMA only through the
        ``f8f6f4`` instruction whose LLVM intrinsic is
        ``llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`` (there is NO plain
        ``mfma.f32.16x16x128.fp8.fp8`` — only the sparse ``smfmac`` and the
        scaled ``f8f6f4`` exist). The lowering pins the in-instruction E8M0
        scale operands to 0 (exponent 0 => scale factor 2^0 == 1.0), and
        the format selectors ``cbsz=0`` / ``blgp=0`` (fp8e4m3 for both A
        and B), making the instruction numerically equivalent to a plain
        unscaled fp8 MFMA. This is the block-scale design's intended use:
        the per-block dequant is applied to the f32 accumulator afterward,
        NOT in-instruction.

        Per-lane on wave64: A = ``<32 x fp8e4m3>`` (32 f8 bytes = 256 bits
        = ``<8 x i32>`` at the intrinsic boundary), B same, C =
        ``<4 x float>``. K = 128 f8 elements per atom.
        Output layout: same as :meth:`fp8_16x16x32` -- 4 floats per lane,
        row = ``(m_blk * 4 + i)``, col = ``n_in_atom``.
        """
        return cls(
            m=16,
            n=16,
            k=128,
            a_per_lane=32,
            b_per_lane=32,
            c_per_lane=4,
            dtype_in="fp8e4m3",
            dtype_out="f32",
            name="mfma_f32_16x16x128_fp8",
        )

    @classmethod
    def bf8_32x32x16(cls) -> "MfmaAtom":
        """BF8 (e5m2) sibling of :meth:`fp8_32x32x16`."""
        return cls(
            m=32,
            n=32,
            k=16,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=16,
            dtype_in="bf8e5m2",
            dtype_out="f32",
            name="mfma_f32_32x32x16_bf8",
        )

    # ---- BF16 atoms (gfx940+; same shapes as f16, different intrinsic) ----
    #
    # The bf16 MFMA family is a 1:1 mirror of the f16 family at the
    # (m, n, k) catalog (gfx940 ships the same set). Per-lane vector
    # widths and accumulator footprints are identical. The only
    # differences vs f16 are the LLVM intrinsic name and the input
    # element dtype — both surface through ``MfmaAtom.emit`` dispatch
    # and the ``dtype_in="bf16"`` tag.

    # ---- FP32 atoms (all CDNA gfx9xx) ----
    #
    # mfma_f32_16x16x4f32 / mfma_f32_32x32x2f32: TF32-class scalar MFMA.
    # Each lane presents a single float scalar for A and B (a_per_lane=1,
    # b_per_lane=1); the accumulator layout is identical to the fp16
    # counterparts (c_per_lane=4 / 16 respectively).

    @classmethod
    def f32_16x16x4(cls) -> "MfmaAtom":
        """FP32 MFMA, 16x16 output, K=4 per atom.

        Per-lane layout on wave64:
        A: scalar float, B: scalar float, C: <4 x float>
        lane = k_blk * 16 + m_in_atom  (k_blk ∈ {0,1,2,3}, m_in_atom ∈ {0..15})
        Output layout: same as f16_16x16x16 -- row = m_blk*4 + i, col = lane%16.
        """
        return cls(
            m=16,
            n=16,
            k=4,
            a_per_lane=1,
            b_per_lane=1,
            c_per_lane=4,
            dtype_in="fp32",
            dtype_out="f32",
            name="mfma_f32_16x16x4_f32",
        )

    @classmethod
    def f32_32x32x2(cls) -> "MfmaAtom":
        """FP32 MFMA, 32x32 output, K=2 per atom.

        Per-lane layout on wave64:
        A: scalar float, B: scalar float, C: <16 x float>
        lane = k_blk * 32 + m_in_atom  (k_blk ∈ {0,1}, m_in_atom ∈ {0..31})
        Output layout: same as f16_32x32x8 -- 16 floats per lane.
        """
        return cls(
            m=32,
            n=32,
            k=2,
            a_per_lane=1,
            b_per_lane=1,
            c_per_lane=16,
            dtype_in="fp32",
            dtype_out="f32",
            name="mfma_f32_32x32x2_f32",
        )

    @classmethod
    def bf16_16x16x16(cls) -> "MfmaAtom":
        """BF16 sibling of :meth:`f16_16x16x16` (legacy CDNA atom)."""
        return cls(
            m=16,
            n=16,
            k=16,
            a_per_lane=4,
            b_per_lane=4,
            c_per_lane=4,
            dtype_in="bf16",
            dtype_out="f32",
            name="mfma_f32_16x16x16_bf16",
        )

    @classmethod
    def bf16_16x16x32(cls) -> "MfmaAtom":
        """BF16 sibling of :meth:`f16_16x16x32` (gfx950+ K-packed)."""
        return cls(
            m=16,
            n=16,
            k=32,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=4,
            dtype_in="bf16",
            dtype_out="f32",
            name="mfma_f32_16x16x32_bf16",
        )

    @classmethod
    def bf16_32x32x8(cls) -> "MfmaAtom":
        """BF16 32x32 atom, K=8 per atom (all CDNA gfx9xx).

        Per-lane layout on wave64:
        A: <4 x bfloat>, B: <4 x bfloat>, C: <16 x float>
        Output layout: same as f16_32x32x8.
        """
        return cls(
            m=32,
            n=32,
            k=8,
            a_per_lane=4,
            b_per_lane=4,
            c_per_lane=16,
            dtype_in="bf16",
            dtype_out="f32",
            name="mfma_f32_32x32x8_bf16",
        )

    @classmethod
    def bf16_32x32x16(cls) -> "MfmaAtom":
        """BF16 sibling of :meth:`f16_32x32x16` (gfx950+ K-packed 32x32)."""
        return cls(
            m=32,
            n=32,
            k=16,
            a_per_lane=8,
            b_per_lane=8,
            c_per_lane=16,
            dtype_in="bf16",
            dtype_out="f32",
            name="mfma_f32_32x32x16_bf16",
        )

    @classmethod
    def fp4_16x16x128(cls) -> "MfmaAtom":
        """fp4 MX MFMA atom (gfx950+, P52).

        Per-lane: A = 16 fp4 nibbles packed as i64 (= 8 bytes per
        lane × 2 nibbles); B same; C = ``<4 x float>``. K = 128
        elements per atom — the densest gfx950 MFMA shape.
        """
        return cls(
            m=16,
            n=16,
            k=128,
            a_per_lane=16,
            b_per_lane=16,
            c_per_lane=4,
            dtype_in="fp4",
            dtype_out="f32",
            name="mfma_f32_16x16x128_fp4",
        )

    @classmethod
    def fp6_16x16x96(cls) -> "MfmaAtom":
        """fp6 MX MFMA atom (gfx950+, P52)."""
        return cls(
            m=16,
            n=16,
            k=96,
            a_per_lane=12,
            b_per_lane=12,
            c_per_lane=4,
            dtype_in="fp6",
            dtype_out="f32",
            name="mfma_f32_16x16x96_fp6",
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

    # ---- HotLoop scheduler timing traits (additive; see F0 brief) ----
    #
    # These are *derived properties* (not new constructor fields) so every
    # existing ``MfmaAtom(...)`` factory call stays byte-identical. They surface
    # the per-atom timing constants the v3/v4 HotLoopScheduler needs:
    #   - ``k_per_xdlops`` == the atom's full per-instruction K (CK's KPerXdlops,
    #     == ``self.k``; this is the value the ``C_MFMA_Inst_Cycle`` table and the
    #     ``KGroup`` predicate switch on, blkgemmpipe_scheduler.hpp:34/65-74 and
    #     blockwise_gemm_pipeline_xdlops_base.hpp:72-85).
    #   - ``mfma_cycle`` == the per-shape/dtype MFMA latency in cycles, the exact
    #     ``C_MFMA_Inst_Cycle`` closed form (blkgemmpipe_scheduler.hpp:63-74):
    #         IsF4F6 ? speedup=2 : 1
    #         NPerXDL==16 -> (KPerXDL==128 ? 32 : 16) / speedup
    #         NPerXDL==32 -> (KPerXDL== 64 ? 64 : 32) / speedup
    #     ck_tile's comp_v3 uses the simpler ``NPerXDL==16 ? 16 : 32`` form
    #     (comp_v3.hpp:305) which coincides with the classic-CK table for every
    #     shipped non-F4F6 atom (16x16x{16,32}->16, 32x32x{8,16}->32), so this
    #     single property serves both pipelines.
    #   - ``is_f4f6`` / ``f8_kgroup`` flags drive the speedup and the f8 16-elem
    #     ds_read split.

    @property
    def is_f4f6(self) -> bool:
        """True for the MX fp4 / fp6 atoms (CK ``IsF4F6``, 2x MFMA speed-up)."""
        return self.dtype_in in ("fp4", "fp6")

    @property
    def k_per_xdlops(self) -> int:
        """CK ``KPerXdlops``: the atom's full per-instruction K (== ``self.k``).

        This is the value the ``C_MFMA_Inst_Cycle`` latency table and the
        f8 ``KGroup`` predicate switch on
        (``blkgemmpipe_scheduler.hpp:65-74``,
        ``blockwise_gemm_pipeline_xdlops_base.hpp:79-81``).
        """
        return self.k

    @property
    def mfma_cycle(self) -> int:
        """Per-shape/dtype MFMA latency in cycles (CK ``C_MFMA_Inst_Cycle``).

        Exact port of ``blkgemmpipe_scheduler.hpp:63-74``. ``NPerXDL`` is
        ``self.n`` and ``KPerXDL`` is ``self.k`` (== :attr:`k_per_xdlops`).
        Raises for atom shapes outside the ``NPerXDL in {16, 32}`` table
        (the batched 4x4 atom), matching the C++ which only instantiates
        the scheduler for the 16x16 / 32x32 XDL shapes.
        """
        speedup = 2 if self.is_f4f6 else 1
        if self.n == 16:
            return (32 if self.k == 128 else 16) // speedup
        if self.n == 32:
            return (64 if self.k == 64 else 32) // speedup
        raise NotImplementedError(
            f"no C_MFMA_Inst_Cycle for atom NPerXDL={self.n} "
            f"(only the 16x16 and 32x32 XDL shapes have a latency table)"
        )

    @property
    def f8_kgroup(self) -> int:
        """CK ``KGroup`` for f8 atoms (``..._base.hpp:72-85``).

        On gfx950 an f8 MFMA consumes 32 f8 elements split into 2
        non-contiguous groups of 16 (one ds_read can fetch only 16 f8 at a
        time). Returns 2 for the f8/bf8 atoms whose XDL shape hits the
        gfx950 wide-K f8 instruction (16x16 KPerXdlops==128 or 32x32
        KPerXdlops==64), else 1. The shipped fp8/bf8 atoms here are the
        16x16x32 / 32x32x16 catalog entries; CK's gfx950 f8 MFMA is the
        wide-K (128/64) instruction those map onto, so the split applies
        when the per-XDL K reaches the wide-K threshold. Non-f8 atoms
        always return 1.
        """
        if self.dtype_in not in ("fp8e4m3", "bf8e5m2"):
            return 1
        if (self.m == 16 and self.n == 16 and self.k_per_xdlops == 128) or (
            self.m == 32 and self.n == 32 and self.k_per_xdlops == 64
        ):
            return 2
        return 1

    # ---- emit ----

    def emit(self, b: IRBuilder, a: Value, bb: Value, c: Value) -> Value:
        """Issue one MMA at this atom's shape via the unified contract.

        The atom's :attr:`name` *is* the backend ``op_id`` (the
        :data:`MFMA_ATOMS` catalog and ``rocke.core.arch`` register the same
        strings), so this routes straight through the target-neutral
        :meth:`IRBuilder.mma`. The prior per-shape ``b.mfma_f32_*`` dispatch
        table was a 1:1 wrapper layer over exactly this call — emission is
        byte-identical (same ``op_id`` attribute, same ``c_frag_len``-sized
        result, same SSA name hint) — and routing through ``mma`` lets the
        same atom drive WMMA on RDNA once a WMMA-named atom is added.
        """
        if self.name not in _OP_ID_NAMES:
            raise NotImplementedError(
                f"no MMA dispatch for atom {self.dtype_in} {self.m}x{self.n}x{self.k}"
            )
        return b.mma(self.name, a, bb, c)

    def emit_scaled(
        self,
        b: IRBuilder,
        a: Value,
        bb: Value,
        c: Value,
        a_scale: Value,
        b_scale: Value,
    ) -> Value:
        """Issue a scaled MX MFMA (P15).

        Currently routes every shape through the
        ``mfma_scale_f32_16x16x128_f8f6f4`` intrinsic; the
        per-output-row scale broadcast happens inside the
        instruction. Future expansion: dedicated scaled intrinsics
        for the 32x32 hero shape if AMDGPU exposes them.
        """
        return b.mfma_scale_f32_16x16x128_f8f6f4(a, bb, c, a_scale, b_scale)

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
        m_blk = lane / 32 (0 or 1)
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


@dataclass(frozen=True)
class WmmaAtom:
    """One WMMA intrinsic (RDNA3/3.5 ``gfx11``, wave32) — the RDNA sibling of
    :class:`MfmaAtom`.

    WMMA's wave32 fragment layout differs fundamentally from MFMA's wave64
    layout: both A and B operands are ``<16 x half>`` covering the *full* ``K=16``
    on every lane (no ``k_blk`` split), and the accumulator is ``<8 x float>``.
    Rather than duplicate that lane math here, the layout accessors delegate to
    the arch-target's hardware-verified :class:`~rocke.core.arch.MmaOp` layout
    maps (the SSOT in ``core/arch/target.py``), so there is exactly one source of
    truth for the fragment layout. Like ``MfmaAtom``, :attr:`name` *is* the
    backend ``op_id``, so :meth:`emit` routes straight through
    :meth:`IRBuilder.mma`.
    """

    m: int
    n: int
    k: int
    a_per_lane: int
    b_per_lane: int
    c_per_lane: int
    dtype_in: str
    dtype_out: str
    name: str
    family: str = "wmma"
    wave_size: int = 32

    # ---- factory class methods ----

    @classmethod
    def f16_16x16x16(cls) -> "WmmaAtom":
        """The gfx11 f16 WMMA atom. K=16/atom, c_per_lane=8 floats.

        Per-lane layout on wave32 (hardware-verified gfx1151):
        A: ``<16 x half>`` (row ``lane % 16``, K = 0..15),
        B: ``<16 x half>`` (col ``lane % 16``, K = 0..15),
        C: ``<8 x float>`` (slot ``i`` -> row ``2*i + lane // 16``, col
        ``lane % 16``).
        """
        return cls(
            m=16,
            n=16,
            k=16,
            a_per_lane=16,
            b_per_lane=16,
            c_per_lane=8,
            dtype_in="f16",
            dtype_out="f32",
            name="wmma_f32_16x16x16_f16",
        )

    @classmethod
    def bf16_16x16x16(cls) -> "WmmaAtom":
        """BF16 sibling of :meth:`f16_16x16x16` (same wave32 fragment layout;
        only the element type / intrinsic mangling differ)."""
        return cls(
            m=16,
            n=16,
            k=16,
            a_per_lane=16,
            b_per_lane=16,
            c_per_lane=8,
            dtype_in="bf16",
            dtype_out="f32",
            name="wmma_f32_16x16x16_bf16",
        )

    # ---- emit ----

    def emit(self, b: IRBuilder, a: Value, bb: Value, c: Value) -> Value:
        """Issue one WMMA at this atom's shape via the unified MMA contract.

        Byte-identical to the hand-rolled ``b.mma(op_id, a, b, c)`` the gfx1151
        attention kernels already emit: same ``op_id`` attribute, same
        ``c_per_lane``-sized ``<8 x float>`` result. The LLVM lowering dispatches
        the ``op_id`` to ``Gfx11RdnaBackend.emit_wmma`` on RDNA.
        """
        if self.name not in _WMMA_OP_ID_NAMES:
            raise NotImplementedError(
                f"no WMMA dispatch for atom {self.dtype_in} {self.m}x{self.n}x{self.k}"
            )
        return b.mma(self.name, a, bb, c)

    def zero_acc(self, b: IRBuilder) -> Value:
        """Allocate a fresh ``<c_per_lane x float>`` accumulator (all zeros) for
        the K-loop ``iter_args``."""
        return b.zero_vec_f32(self.c_per_lane)

    # ---- physical lane layout (delegated to the arch-target SSOT) ----

    def mma_op(self, arch: str = "gfx1151") -> MmaOp:
        """The arch-target :class:`MmaOp` backing this atom (the layout-map
        SSOT). ``arch`` selects which gfx target's verified maps to use."""
        op = ArchTarget.from_gfx(arch).mma.by_op_id(self.name)
        if op is None:
            raise ValueError(
                f"arch {arch!r} has no WMMA op {self.name!r}; "
                f"WMMA needs an RDNA3/3.5 (gfx11) target"
            )
        return op

    def a_layout(self, arch: str = "gfx1151") -> LayoutMap:
        """The A-operand ``(row, k)`` lane/slot -> coordinate map."""
        return self.mma_op(arch).a_layout()

    def b_layout(self, arch: str = "gfx1151") -> LayoutMap:
        """The B-operand ``(k, col)`` lane/slot -> coordinate map."""
        return self.mma_op(arch).b_layout()

    def c_layout(self, arch: str = "gfx1151") -> LayoutMap:
        """The accumulator (C/D) ``(row, col)`` lane/slot -> coordinate map."""
        return self.mma_op(arch).c_layout()

    # Convenience alias mirroring ``MmaOp.acc_layout``.
    def acc_layout(self, arch: str = "gfx1151") -> LayoutMap:
        return self.c_layout(arch)


# ---------------------------------------------------------------------
# C-accumulator warp distribution (CWarpDstrEncoding)
# ---------------------------------------------------------------------
#
# CK Tile expresses the MFMA C-fragment lane/register layout as a
# ``tile_distribution_encoding`` rather than as hand-rolled lane->(row,
# col) arithmetic. The canonical encoding lives in
# ``ops/gemm/warp/warp_gemm_attribute_mfma.hpp``:
#
#     using CWarpDstrEncoding = tile_distribution_encoding<
#         sequence<>,                                         // Rs (none)
#         tuple<sequence<kCM0PerLane, kCMLane, kCM1PerLane>,  // M decomposition
#               sequence<kCNLane>>,                           // N decomposition
#         tuple<sequence<1, 2>>,   // single P (lane) -> M-level1 and N-level0
#         tuple<sequence<1, 0>>,   //   at M minor 1 (kCMLane) and N minor 0
#         sequence<1, 1>,          // two Y dims, both on the M axis
#         sequence<0, 2>>;         //   at M minor 0 (kCM0PerLane) and 2 (kCM1PerLane)
#
# where (from ``warp_gemm_attribute_mfma_impl.hpp``):
#   16x16 atoms: kCMLane=4, kCNLane=16, kCM0PerLane=1, kCM1PerLane=4
#   32x32 atoms: kCMLane=2, kCNLane=32, kCM0PerLane=4, kCM1PerLane=4
#
# Reading the encoding (single-warp, wave64) the lane decomposes as
# ``(m_blk, n) = (lane // kCNLane, lane % kCNLane)`` and the per-lane
# accumulator slot ``i`` decomposes as ``(y0, y1)`` (row-major over the
# two Y lengths kCM0PerLane, kCM1PerLane). ``calculate_x`` then yields
#   row = y0 * (kCMLane * kCM1PerLane) + m_blk * kCM1PerLane + y1
#   col = n
# which is exactly :meth:`MfmaAtom.lane_to_output` for every supported
# atom (verified element-by-element against the SSA both helpers emit).


# (kCM0PerLane, kCMLane, kCM1PerLane, kCNLane) for each (m, n) C tile.
# Sourced from the kC* constants in warp_gemm_attribute_mfma_impl.hpp.
_C_WARP_PARAMS: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {
    (16, 16): (1, 4, 4, 16),
    (32, 32): (4, 2, 4, 32),
}


def c_warp_params(atom: "MfmaAtom") -> Tuple[int, int, int, int]:
    """Return ``(kCM0PerLane, kCMLane, kCM1PerLane, kCNLane)`` for ``atom``.

    These are the four CK Tile ``WarpGemmAttributeMfmaImpl`` constants
    that fully describe the MFMA C-fragment layout. ``kCM0PerLane *
    kCM1PerLane`` is the per-lane accumulator count (``c_per_lane``) and
    ``kCMLane * kCNLane`` is the wavefront size that participates in the
    M/N spatial tiling (64 on wave64). Raises for atom shapes CK Tile
    does not give a ``CWarpDstrEncoding`` for (the batched 4x4 atom).
    """
    key = (atom.m, atom.n)
    if key not in _C_WARP_PARAMS:
        raise NotImplementedError(
            f"no CWarpDstrEncoding for atom {atom.m}x{atom.n} "
            f"(only the 16x16 and 32x32 MFMA C tiles are supported)"
        )
    m0, m_lane, m1, n_lane = _C_WARP_PARAMS[key]
    if m0 * m1 != atom.c_per_lane:
        raise ValueError(
            f"atom {atom.name}: kCM0PerLane*kCM1PerLane ({m0 * m1}) "
            f"!= c_per_lane ({atom.c_per_lane})"
        )
    return m0, m_lane, m1, n_lane


def make_c_warp_dstr_encoding(atom: "MfmaAtom") -> TileDistributionEncoding:
    """Build the MFMA C-tile :class:`TileDistributionEncoding` for ``atom``.

    Port of CK Tile's ``CWarpDstrEncoding``
    (``ops/gemm/warp/warp_gemm_attribute_mfma.hpp``) /
    ``make_embed_tile_distribution_encoding`` for the C accumulator. The
    returned encoding describes, for one wavefront, how the (lane,
    per-lane register slot) pair maps onto the (row, col) of the warp's
    output tile -- i.e. it expresses the accumulator layout as a
    :class:`TileDistribution` instead of the hand-rolled lane arithmetic
    in :meth:`MfmaAtom.lane_to_output`.

    Driving it: split the lane as ``(m_blk, n) = (lane // kCNLane, lane
    % kCNLane)`` to form the single P sub-sequence ``[m_blk, n]``, and
    split the accumulator slot ``i`` row-major over the two Y lengths
    (``kCM0PerLane``, ``kCM1PerLane``). Then
    ``make_static_tile_distribution(enc).calculate_x(b, ys=[y0, y1],
    ps=[[m_blk, n]])`` returns ``(row_in_atom, col_in_atom)`` identical
    to :meth:`MfmaAtom.lane_to_output`.

    Raises :class:`NotImplementedError` for atom shapes without a CK
    Tile ``CWarpDstrEncoding`` (the batched 4x4 atom).
    """
    m0, m_lane, m1, n_lane = c_warp_params(atom)
    return TileDistributionEncoding(
        Rs=(),
        Hs=((m0, m_lane, m1), (n_lane,)),
        # Single P (the lane): sub0 -> X-major 1 (M) at minor 1 (kCMLane),
        # sub1 -> X-major 2 (N) at minor 0 (kCNLane).
        Ps2RHs_major=((1, 2),),
        Ps2RHs_minor=((1, 0),),
        # Two Y dims, both on the M axis: minor 0 (kCM0PerLane) and
        # minor 2 (kCM1PerLane).
        Ys2RHs_major=(1, 1),
        Ys2RHs_minor=(0, 2),
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

MFMA_F32_ATOMS: Tuple[MfmaAtom, ...] = (
    MfmaAtom.f32_16x16x4(),
    MfmaAtom.f32_32x32x2(),
)

MFMA_BF16_ATOMS: Tuple[MfmaAtom, ...] = (
    MfmaAtom.bf16_16x16x16(),
    MfmaAtom.bf16_16x16x32(),
    MfmaAtom.bf16_32x32x8(),
    MfmaAtom.bf16_32x32x16(),
)

MFMA_FP8_ATOMS: Tuple[MfmaAtom, ...] = (
    MfmaAtom.fp8_16x16x32(),
    MfmaAtom.fp8_32x32x16(),
    MfmaAtom.bf8_16x16x32(),
    MfmaAtom.bf8_32x32x16(),
    MfmaAtom.fp8_16x16x128(),
)

# P52: MX fp4 / fp6 atoms (gfx950+).
MFMA_MX_ATOMS: Tuple[MfmaAtom, ...] = (
    MfmaAtom.fp4_16x16x128(),
    MfmaAtom.fp6_16x16x96(),
)

# Unified catalog covering every shipped MFMA shape. Used by
# ``mfma_atom("<dtype>", m, n, k)`` to dispatch into the right
# factory; ``MFMA_F16_ATOMS`` / ``MFMA_BF16_ATOMS`` / ``MFMA_FP8_ATOMS``
# are kept as narrower subset accessors for callers that want to walk
# only the fp16, bf16, or fp8/bf8 families.
MFMA_ATOMS: Tuple[MfmaAtom, ...] = (
    MFMA_F16_ATOMS + MFMA_F32_ATOMS + MFMA_BF16_ATOMS + MFMA_FP8_ATOMS + MFMA_MX_ATOMS
)

# Accept aliases on the dtype lookup key: ``fp8`` -> ``fp8e4m3``,
# ``bf8`` -> ``bf8e5m2``. Keeps Triton-ported and CK Tile-ported
# call sites working without translation.
_DTYPE_ALIAS = {
    "fp8": "fp8e4m3",
    "fp8e4m3": "fp8e4m3",
    "bf8": "bf8e5m2",
    "bf8e5m2": "bf8e5m2",
    "f16": "f16",
    "fp16": "f16",
    "f32": "fp32",
    "fp32": "fp32",
    "float": "fp32",
    "bf16": "bf16",
    "bfloat16": "bf16",
}

_BY_SHAPE: Dict[Tuple[str, int, int, int], MfmaAtom] = {
    (a.dtype_in, a.m, a.n, a.k): a for a in MFMA_ATOMS
}

# The set of backend ``op_id`` strings (== ``MfmaAtom.name``) that
# :meth:`MfmaAtom.emit` knows how to issue through :meth:`IRBuilder.mma`.
# Sourced from the catalog so adding a factory + catalog row is the only
# step needed to make a new atom emittable.
_OP_ID_NAMES: frozenset = frozenset(a.name for a in MFMA_ATOMS)


def mfma_atom(dtype: str, m: int, n: int, k: int) -> MfmaAtom:
    """Lookup an atom by (dtype_in, m, n, k). Raises if unknown.

    Accepts ``fp8`` / ``fp8e4m3``, ``bf8`` / ``bf8e5m2``, and
    ``f16`` / ``fp16`` aliases on the dtype key.
    """
    canon = _DTYPE_ALIAS.get(dtype, dtype)
    key = (canon, m, n, k)
    if key not in _BY_SHAPE:
        valid = sorted((a.dtype_in, a.m, a.n, a.k) for a in MFMA_ATOMS)
        raise ValueError(f"no MFMA atom for {key}; valid: {valid}")
    return _BY_SHAPE[key]


# ---------------------------------------------------------------------
# WMMA catalog (RDNA3/3.5, wave32)
# ---------------------------------------------------------------------

WMMA_F16_ATOMS: Tuple[WmmaAtom, ...] = (WmmaAtom.f16_16x16x16(),)
WMMA_BF16_ATOMS: Tuple[WmmaAtom, ...] = (WmmaAtom.bf16_16x16x16(),)
WMMA_ATOMS: Tuple[WmmaAtom, ...] = WMMA_F16_ATOMS + WMMA_BF16_ATOMS

# Backend ``op_id`` strings (== ``WmmaAtom.name``) that :meth:`WmmaAtom.emit`
# knows how to issue through :meth:`IRBuilder.mma`.
_WMMA_OP_ID_NAMES: frozenset = frozenset(a.name for a in WMMA_ATOMS)

_WMMA_BY_SHAPE: Dict[Tuple[str, int, int, int], WmmaAtom] = {
    (a.dtype_in, a.m, a.n, a.k): a for a in WMMA_ATOMS
}


def wmma_atom(dtype: str, m: int, n: int, k: int) -> WmmaAtom:
    """Lookup a WMMA atom by (dtype_in, m, n, k). Raises if unknown.

    Accepts ``f16`` / ``fp16`` and ``bf16`` aliases on the dtype key.
    """
    canon = _DTYPE_ALIAS.get(dtype, dtype)
    key = (canon, m, n, k)
    if key not in _WMMA_BY_SHAPE:
        valid = sorted((a.dtype_in, a.m, a.n, a.k) for a in WMMA_ATOMS)
        raise ValueError(f"no WMMA atom for {key}; valid: {valid}")
    return _WMMA_BY_SHAPE[key]
