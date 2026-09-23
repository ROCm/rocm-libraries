# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Single-launch fused-MoE MEGA-kernel (FP8 e4m3 block-scale).

The fp8 mega-kernel snapshot for this level. A SINGLE
fused kernel computes, per (inter-slice, sorted-m-block) threadgroup, the full
MoE per-expert path with fp8 e4m3 operands and per-128-block f32 scales:

    GEMM0 (gate) + GEMM1 (up)  -- fp8 X . fp8 W -> f32 acc, per-128-group
        dequant fold (group-accumulator pattern, scale applied POST-MFMA)
      -> SiLU(gate_dq) * up_dq in f32
      -> DYNAMIC-QUANTIZE the f32 Hidden to fp8 (per-128-block-along-inter,
         per-row amax/448 scale), stage into a PERSISTENT LDS Hidden_smem,
         and stash the per-block dynamic scales in an LDS scratch.
      -> GEMM2 (down) -- fp8 Hidden . fp8 W_down -> f32 acc, dequant by
         (hidden_dyn_scale * down_scale) -> weighted atomic reduce into Y.

See ``examples/gfx950/fused_mega_moe/docs/BUILD_SPEC_FP8.md`` for the authoritative
build spec. The dequant ordering follows BUILD_SPEC_FP8 Section 1.2 (the
``block_scale_gemm.py`` group-accumulator pattern): within a 128-wide
contraction block the scales are constant, so
``sum_k (a.sa)(b.sb) = sa.sb . sum_k (a.b)`` -- the scale is applied per
K-group, post-MFMA, NOT in-instruction (which would mean the E8M0 trap of
``cvt_scalef32_pk_f32_fp8x4``).

STAGING STATUS (incremental implementation per BUILD_SPEC_FP8 Phase plan):

* STAGE 1 (THIS FILE, current state):
    - ``FusedMegaKernelSpecFp8`` (fp8 spec / signature / grid).
    - the gate+up fp8 GEMM via ``_emit_fp8_gateup_group_gemm`` (fp8 atom,
      fp8 operand loads, per-128-block scale dequant of BOTH accumulators).
    - SiLU(gate)*up in f32.
    - ``_emit_hidden_dyn_quant_stage``: per-(row, 128-inter-block) amax ->
      dynamic scale -> quantize the f32 Hidden to fp8 and stage into the
      persistent LDS ``Hidden_smem`` + stash the per-block scales in
      ``HiddenScale_smem``.
* STAGE 2 (THIS FILE, current state): the fp8 down GEMM reading ``Hidden_smem``
  as the LDS-resident A operand (``make_transposed``-equivalent implicit reshape
  -- the dynamic-quant write addr == the down MFMA A-read addr, same logical
  ``(m, inter)`` cell, BUILD_SPEC_FP8 Section 3.5), with the per-128-group
  group-accumulator dequant by (``HiddenScale_smem[row, blk]`` * ``down_scale``),
  multiply by the sorted per-token weight, and atomic-add the f32 partial into Y
  (padded rows token id -1 / >= tokens are skipped). Tiled over the H_out output
  in ``tile_n_down`` chunks; grid.x split the inter contraction so each TG
  atomic-adds a PARTIAL Y over the whole H_out.

The fp8 MFMA atom + cvt primitives are 100% reused from ``helpers/atoms.py``
and ``core/ir.py`` (never modified). This file builds the gate+up fp8 GEMM and
the dynamic-quant staging from the fp8-aware ``helpers/mfma_gemm_inner.py``
toolkit (which dispatches the fp8 atom directly) rather than the
``UniversalGemmSpec``-bound f16/bf16 helpers (whose ``io_ir_type`` rejects fp8).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from ...core.ir import (
    CACHE_ALL,
    F32,
    FP8E4M3,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ...helpers.asm import mfma_f8f6f4_agpr
from ...helpers.atoms import MfmaAtom
from ...helpers.mfma_gemm_inner import (
    decode_mfma_lanes,
    validate_arch_and_block_size,
    validate_mfma_atom_in_catalog,
)
from ...helpers.quant import quant_max_abs
from ...helpers.tensor_view import TensorDescriptor, TensorView


__all__ = [
    "FusedMegaKernelSpecFp8",
    "build_moe_fused_mega_gemm_fp8",
    "moe_fused_mega_fp8_grid",
    "moe_fused_mega_fp8_signature",
]


# AGPR operand staging: route the gate/up/down K=128 fp8 MFMAs through the
# inline-asm helper (``v_mfma_f32_16x16x128_f8f6f4`` with AGPR srcA/srcB +
# VGPR acc, the hand-tuned asm staging layout) instead of the scaled intrinsic. The
# helper is numerically identical (cbsz/blgp=0 => fp8e4m3, scales pinned to
# the neutral E8M0 0 => factor 1.0) and verified bit-exact vs the intrinsic
# (asm_mfma_parity / asm_mfma_fp8frag). Forcing the fp8 A/B fragments to live
# in the AGPR file across the K-loop is the lever (hand-tuned asm keeps ~192 AGPR of
# operand state; the intrinsic path leaves srcA/srcB placement to the register
# allocator). Flip to True to force the AGPR-source inline-asm MFMA.
#
# DEFAULT False (intrinsic path): the forced-AGPR-source inline-asm route was
# DEBUGGED TO BIT-EXACT CORRECTNESS (hardened parity passes at every hazard_nop
# down to 0) but measured ~25% SLOWER same-session (T1 0.197-0.201 ms asm vs
# 0.159 ms intrinsic, clean A/B/A). Root cause: the ``sideeffect`` inline-asm
# MFMA is OPAQUE to the LLVM machine scheduler + GCNHazardRecognizer, so the
# scheduler can no longer interleave independent VMEM loads / other MFMAs into
# the MFMA latency window -- the exact overlap the AGPR-resident-operand lever
# is meant to enable -- and the sideeffect barrier blocks reordering. The
# intrinsic path ALREADY stages operands into AGPR via the register allocator
# (verified ``v_accvgpr_write_b32`` in ISA) WITHOUT forfeiting the scheduler.
# hand-tuned asm's 192-AGPR staging works because hand-tuned asm hand-schedules the ENTIRE asm
# instruction stream (AGPR lifetime AND interleave together); comgr inline-asm
# gives register-class control but forfeits the scheduler, which dominates here.
# Kept behind a flag (correct, available) rather than reverted.
_USE_ASM_AGPR_MFMA = False

# Trailing ``s_nop`` count on each inline-asm MFMA (see helpers/asm.py). The
# default-8 the helper bakes in is conservative: it is sized for a back-to-back
# dependent accumulation chain on the SAME source AGPRs with no interleaved
# work. The mega-kernel's K-loop emits multiple INDEPENDENT MFMAs back-to-back
# (gate vs up; multiple ni accumulators), so the hardware MFMA pipeline is
# already partially filled and a smaller nop suffices -- recovering the
# throughput the blanket nop would otherwise serialise. This is SWEPT against
# HARDENED parity (NOT a free perf knob: too-small a value silently corrupts
# the accumulator, so every value is re-validated numerically). Overridable
# via the ROCKE_FP8_MFMA_NOP env var for the sweep.
_ASM_MFMA_HAZARD_NOP = int(os.environ.get("ROCKE_FP8_MFMA_NOP", "8"))

# D5 scheduling-cadence sweep knob (additive). Values:
#   "iglp1" -> (KEPT, default) emit ``b.iglp_opt(1)``
#              (MFMASmallGemmSingleWaveOpt) once at the top of the gate/up + down
#              K-loop bodies. The post-RA scheduler then imposes the canned
#              MFMA/DS interleave for each loop region. Won the D5 sweep:
#              same-session best-of-5 T1 ~0.1586 vs "none" ~0.1597 (strictly
#              faster across 3 alternating thermal-controlled pairs, parity PASS,
#              golden digest unchanged). Mutually exclusive with sched_*barrier
#              (so no sched_group_barrier is emitted when this is set).
#   "none"  -> no scheduler hint (pre-D5 baseline; byte-identical IR).
#   "sgb"   -> explicit sched_group_barrier cadence in the active DTLA gate/up +
#              down loops. D5-swept: REGRESSED (~0.166 T1) -> not kept.
_SCHED_CADENCE = os.environ.get("ROCKE_FP8_SCHED", "iglp1").strip().lower()

# D8 XCD-remap sweep knob (additive, file-only, grid shape UNCHANGED at (gx, gy)).
# The HW round-robins linear workgroups across the 8 XCDs (each its own L2
# slice). The default linear (m_block major in y, inter slice in x) order spreads
# the 8 m_blocks (= 8 distinct experts' weight footprints) round-robin across the
# 8 XCDs interleaved with the 28 inter slices, so every XCD touches every expert
# (no L2 locality on the huge per-expert WGate/WUp/WDown). This knob remaps the
# (inter, m_block) tile a physical workgroup owns so that workgroups landing on
# the same XCD share the same m_block (= same expert weights) => per-expert weight
# columns stay resident in that XCD's L2 across the 28 inter slices.
#   0 -> OFF (default; byte-identical IR / digest, the kept-best path)
#   N>0 -> remap with N = XCD count (8 on MI355X). The linear workgroup id
#          ``bx*gy + by`` is re-tiled into (xcd, slot) so each XCD owns a
#          contiguous run of inter slices for ONE m_block at a time.
_XCD_REMAP = int(os.environ.get("ROCKE_FP8_XCD", "8"))


def _emit_loop_cadence_hint(b: IRBuilder) -> None:
    """Emit the D5 per-loop scheduler hint at the TOP of a K-loop body.

    Only ``iglp1`` emits here (it owns the whole-loop schedule and must precede
    the loop body). ``sgb`` emits its cadence inline next to each MFMA instead.
    """
    if _SCHED_CADENCE == "iglp1":
        b.iglp_opt(1)


# sched_group_barrier mask bits.
_SGB_MFMA = 0x008
_SGB_VMEM_READ = 0x020
_SGB_VMEM_WRITE = 0x040
_SGB_DS_READ = 0x100
_SGB_DS_WRITE = 0x200


def _emit_sgb_gateup_dtla(b: IRBuilder, n_mfma: int = 2) -> None:
    """compv4-style cadence for the DTLA gate/up loop body (per ni).

    The DTLA path stages B via ``global_load...lds`` (a VMEM read whose dest is
    LDS) then ``ds_read`` then ``n_mfma`` MFMAs. Impose: 1 VMEM_READ (the staged
    DMA), DS_READ feeding the MFMA, then the MFMAs -- so the in-flight DMA + LDS
    read overlap the MFMA shadow. No-op unless ``_SCHED_CADENCE == 'sgb'``.
    """
    if _SCHED_CADENCE != "sgb":
        return
    b.sched_group_barrier(_SGB_VMEM_READ, 1, 0)
    b.sched_group_barrier(_SGB_DS_READ, n_mfma, 0)
    b.sched_group_barrier(_SGB_MFMA, int(n_mfma), 0)


def _emit_sgb_down_group(b: IRBuilder, n_mfma: int = 1) -> None:
    """compv4-style VMEM<->MFMA cadence for the down loop per-group body.

    The down loop issues a global VMEM W_down load then ``n_mfma`` MFMAs per
    128-group. Impose: 1 VMEM_READ (next group's W_down) under the MFMA(s).
    No-op unless ``_SCHED_CADENCE == 'sgb'``.
    """
    if _SCHED_CADENCE != "sgb":
        return
    b.sched_group_barrier(_SGB_VMEM_READ, 1, 0)
    b.sched_group_barrier(_SGB_MFMA, int(n_mfma), 0)


def _emit_mfma(b: IRBuilder, atom: MfmaAtom, a: Value, bb: Value, acc: Value) -> Value:
    """Issue one K=128 fp8 MFMA, optionally via the AGPR-source inline-asm helper.

    When :data:`_USE_ASM_AGPR_MFMA` is set AND the atom is the K=128 fp8 hero
    atom, route through :func:`mfma_f8f6f4_agpr` (AGPR srcA/srcB). The helper
    bitcasts the ``<32 x fp8e4m3>`` fragment to ``<8 x i32>`` itself. Otherwise
    fall back to the bit-identical scaled-intrinsic atom (``atom.emit``).
    """
    if _USE_ASM_AGPR_MFMA and atom.k == 128 and atom.dtype_in == "fp8e4m3":
        return mfma_f8f6f4_agpr(b, a, bb, acc, hazard_nop=_ASM_MFMA_HAZARD_NOP)
    return atom.emit(b, a, bb, acc)


# Group block along the contraction axis (= 4 fp8_16x16x32 atoms).
GROUP_K = 128
# fp8e4m3 saturating clamp magnitude.
FP8_MAX = quant_max_abs("fp8e4m3")  # 448.0
# hand-tuned asm dynamic-quant amax floor.
AMAX_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedMegaKernelSpecFp8:
    """Single-launch fused-MoE mega-kernel spec (FP8 e4m3 block-scale).

    Tile geometry is IDENTICAL to the f16 :class:`FusedMegaKernelSpec`
    (BUILD_SPEC_FP8 Section 2.0): the fp8 atom is also K=32 and the per-lane
    fragment width is the same, so only the operand element dtype (2B->1B) and
    the MFMA intrinsic change.

    * ``tile_m`` = sorted tokens per m-block (hand-tuned asm ``sub_x``), default 32.
    * ``tile_n_inter`` = inter columns this TG owns (hand-tuned asm ``sub_gu``); the
      GEMM0/1 N extent AND the GEMM2 contraction extent, default 256.
    * ``tile_k_gu`` = K-loop tile along the hidden contraction H for gate/up.
    * MFMA atom = fp8 ``16x16x32`` (e4m3); the 128-wide scale group = 4 atoms.
    """

    name: str
    tile_m: int = 16
    tile_n_inter: int = 256
    tile_k_gu: int = 32
    warp_m: int = 1
    warp_n: int = 4
    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 32
    tile_n_down: int = 256
    tile_k_down: int = 64
    wave_size: int = 64
    block_size: int = 0
    dtype: str = "fp8e4m3"

    def __post_init__(self) -> None:
        if self.block_size == 0:
            object.__setattr__(
                self,
                "block_size",
                self.warp_m * self.warp_n * self.wave_size,
            )

    # -- derived geometry helpers ----------------------------------------

    def gate_up_atom(self) -> MfmaAtom:
        # L6: the unscaled fp8 16x16x128 hero atom (K=128 per MFMA, 4x fewer
        # K-trips than the legacy 16x16x32). The 128-wide scale group now maps
        # to EXACTLY ONE atom (atoms_per_group = GROUP_K // atom.k = 1), so the
        # per-128-block dequant fold aligns naturally with a single MFMA.
        return MfmaAtom.fp8_16x16x128()

    def down_atom(self) -> MfmaAtom:
        return MfmaAtom.fp8_16x16x128()

    @property
    def mfmas_m(self) -> int:
        """Atoms along M per warp for the gate/up tile."""
        return (self.tile_m // self.warp_m) // self.warp_tile_m

    @property
    def mfmas_n(self) -> int:
        """Atoms along N per warp for the gate/up tile."""
        return (self.tile_n_inter // self.warp_n) // self.warp_tile_n

    @property
    def mfmas_m_down(self) -> int:
        """Atoms along M per warp for the down tile (M x H_out_slice)."""
        return (self.tile_m // self.warp_m) // self.warp_tile_m

    @property
    def mfmas_n_down(self) -> int:
        """Atoms along N (= H_out output) per warp for the down tile."""
        return (self.tile_n_down // self.warp_n) // self.warp_tile_n

    def kernel_name(self) -> str:
        return (
            f"{self.name}_moe_fused_mega_fp8_"
            f"m{self.tile_m}n{self.tile_n_inter}k{self.tile_k_gu}"
        )


# ---------------------------------------------------------------------------
# Grid + signature (BUILD_SPEC_FP8 Section 2.7 / 2.8)
# ---------------------------------------------------------------------------


def moe_fused_mega_fp8_grid(
    num_m_blocks: int, inter: int, spec: FusedMegaKernelSpecFp8
) -> Tuple[int, int, int]:
    """Mega-kernel launch grid (unchanged from the f16 kernel).

    ``grid = (ceil(inter / tile_n_inter), num_m_blocks, 1)``. Canonical decode
    (I=7168, tile_n_inter=256): grid.x = 28; grid.y = num_m_blocks = 8.
    """
    sub_gu = spec.tile_n_inter
    gx = (inter + sub_gu - 1) // sub_gu
    return (gx, num_m_blocks, 1)


def moe_fused_mega_fp8_signature(spec: FusedMegaKernelSpecFp8):
    from ...helpers.spec import SignatureBuilder

    return (
        SignatureBuilder()
        .ptr(
            "A", "fp8e4m3"
        )  # quantized activation X (hand-tuned asm: fp8 + input_scale)
        .ptr("WGate", "fp8e4m3")
        .ptr("WUp", "fp8e4m3")
        .ptr("WDown", "fp8e4m3")
        .ptr("AScale", "f32")  # input_scale, per (token-block, H-block-of-128)
        .ptr("WGateScale", "f32")  # fc1_scale gate half, per (E, I-block, H-block)
        .ptr("WUpScale", "f32")  # fc1_scale up half
        .ptr("WDownScale", "f32")  # fc2_scale, per (E, H_out-block, I-block)
        .ptr("SortedTokenIds", "i32")
        .ptr("SortedWeights", "f32")
        .ptr("BlockExpertIds", "i32")
        .ptr("Y", "f32")  # f32 to reuse the f16 atomic epilogue unchanged
        .scalar("M", "i32")
        .scalar("N", "i32")  # = I (inter dim)
        .scalar("K", "i32")  # = H (hidden contraction)
        .scalar("H_out", "i32")  # = H (down output)
        .scalar("stride_a", "i32")
        .scalar("stride_b_gate", "i32")
        .scalar("stride_b_up", "i32")
        .scalar("stride_b_down", "i32")
        .scalar("stride_a_scale", "i32")
        .scalar("stride_gate_scale", "i32")
        .scalar("stride_up_scale", "i32")
        .scalar("stride_down_scale", "i32")
        .scalar("stride_gate_scale_e", "i32")
        .scalar("stride_up_scale_e", "i32")
        .scalar("stride_down_scale_e", "i32")
        .scalar("slot_size", "i32")
        .scalar("tokens", "i32")
        .build()
    )


# ---------------------------------------------------------------------------
# STAGE 1a: gate+up fp8 GEMM with per-128-block dequant
# ---------------------------------------------------------------------------


def _emit_fp8_gateup_group_gemm(
    b: IRBuilder,
    *,
    A: Value,
    WGate: Value,
    WUp: Value,
    AScale: Value,
    WGateScale: Value,
    WUpScale: Value,
    atom: MfmaAtom,
    lane_decode,
    m_tile_base: Value,
    n_tile_base: Value,
    K: Value,
    stride_a_scale: Value,
    stride_gate_scale: Value,
    stride_up_scale: Value,
    tag: str,
) -> Tuple[Value, Value]:
    """Gate + up fp8 GEMM, returning ``(gate_dq, up_dq)`` per-lane f32 vectors.

    Group-accumulator pattern (BUILD_SPEC_FP8 Section 1.2): the outer loop walks
    128-wide groups along the hidden contraction K (= H). Per group, 4
    fp8_16x16x32 atoms accumulate into a FRESH ``group_acc`` (one for gate, one
    for up), then the group is folded into the outer accumulator scaled by
    ``a_scale * b_scale`` -- a single ``v_pk_fma_f32`` per accumulator. The
    A read (the quantized activation) is shared across gate and up; only the
    B-side weight scale differs.

    Scale index math (BUILD_SPEC_FP8 Section 1.3), per-128 along the
    contraction and per-128 along the output:

        a_scale_off = (m_row // GROUP_K_M) * k_scale_count + kg
        b_scale_off = kg * n_scale_count + (n_col // GROUP_K)

    Here the activation amax is per-(token-block, H-block); we use the per-row
    granularity available to the lane (``m_row``) with the H-block index ``kg``,
    and the weight scale is per (output-128, contraction-128).
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4

    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)

    # Per-lane scale offsets vary along the K-group index ``kg`` only.
    a_row_scale_base = b.mul(m_row, stride_a_scale)
    n_blk = b.div(n_col, c_group_k)

    zero = atom.zero_acc(b)
    gate_zero = atom.zero_acc(b)
    up_zero = atom.zero_acc(b)

    # Outer loop over 128-wide contraction groups: num_groups = K // GROUP_K.
    num_groups = b.div(K, c_group_k)
    outer = b.scf_for_iter(
        b.const_i32(0),
        num_groups,
        b.const_i32(1),
        [(f"gate_outer_{tag}", gate_zero), (f"up_outer_{tag}", up_zero)],
        iv_name=f"kg_{tag}",
    )
    with outer as (kg, (gate_outer, up_outer)):
        # Per-group scales (one f32 each).
        a_scale_off = b.add(a_row_scale_base, kg)
        a_scale_v = b.global_load_f32(AScale, a_scale_off)
        gate_scale_off = b.add(b.mul(kg, stride_gate_scale), n_blk)
        up_scale_off = b.add(b.mul(kg, stride_up_scale), n_blk)
        gate_scale_v = b.global_load_f32(WGateScale, gate_scale_off)
        up_scale_v = b.global_load_f32(WUpScale, up_scale_off)
        gate_ab = b.fmul(a_scale_v, gate_scale_v)
        up_ab = b.fmul(a_scale_v, up_scale_v)

        k_group_base = b.mul(kg, c_group_k)

        # Fresh per-group accumulators; 4 fp8 atoms cover the 128-wide group.
        ginner = b.scf_for_iter(
            b.const_i32(0),
            b.const_i32(atoms_per_group),
            b.const_i32(1),
            [(f"g_acc_{tag}", zero), (f"u_acc_{tag}", zero)],
            iv_name=f"kk_{tag}",
        )
        with ginner as (kk, (g_acc, u_acc)):
            k_tile_base = b.add(k_group_base, b.mul(kk, c_atom_k))
            a_frag = _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=k_tile_base,
                K=K,
            )
            gb_frag = _load_b_fp8(
                b,
                B=WGate,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=K,
            )
            ub_frag = _load_b_fp8(
                b,
                B=WUp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=k_tile_base,
                N=K,
            )
            g_new = _emit_mfma(b, atom, a_frag, gb_frag, g_acc)
            u_new = _emit_mfma(b, atom, a_frag, ub_frag, u_acc)
            b.scf_yield(g_new, u_new)
        group_gate = ginner.results[0]
        group_up = ginner.results[1]

        # Fold (post-MFMA, per-group): outer += group * (a_scale * b_scale).
        gate_scale_vec = b.vector_splat(gate_ab, atom.c_per_lane)
        up_scale_vec = b.vector_splat(up_ab, atom.c_per_lane)
        gate_outer_new = b.vector_fma(group_gate, gate_scale_vec, gate_outer)
        up_outer_new = b.vector_fma(group_up, up_scale_vec, up_outer)
        b.scf_yield(gate_outer_new, up_outer_new)

    return outer.results[0], outer.results[1]


def _emit_fp8_gateup_fused_kloop(
    b: IRBuilder,
    *,
    A: Value,
    WGate: Value,
    WUp: Value,
    AScale: Value,
    WGateScale: Value,
    WUpScale: Value,
    atom: MfmaAtom,
    lane_decode,
    m_tile_base: Value,
    n_tile_bases,
    K: Value,
    stride_a_scale: Value,
    stride_gate_scale: Value,
    stride_up_scale: Value,
    tag: str,
    dtla=None,
):
    """Gate + up fp8 GEMM fused across ALL ni cells of one mi row.

    COMBINATION lever (gate+up SW-pipeline + wave-pair odd/even MFMA interleave).
    The legacy ``_emit_fp8_gateup_group_gemm`` ran an INDEPENDENT K-loop per
    (mi, ni) cell, reloading the shared A operand ``mfmas_n`` times and emitting
    the two MFMAs (gate, up) of each cell back-to-back -- a bursty pattern that
    starves the compiler of any load/MFMA overlap window. This fused emitter:

    * Runs ONE outer 128-group K-loop carrying ``2 * len(n_tile_bases)`` outer
      f32 accumulators (gate[ni], up[ni]).
    * Loads the shared A fragment ONCE per atom (m_tile_base, k) and reuses it
      across every ni -- killing the (mfmas_n - 1) redundant A streams.
    * Unrolls the 4-atom inner group in Python (the trip count is the
      compile-time constant ``atoms_per_group``; an scf.for iter-arg cannot carry
      an fp8e4m3 fragment) and REGISTER-DOUBLE-BUFFERS the next atom's A + every
      WGate/WUp fragment (``a_next`` / ``gb_next[ni]`` / ``ub_next[ni]``): the
      next loads are ISSUED before the current MFMAs consume their operands,
      giving an in-flight load under the MFMA.
    * INTERLEAVES the MFMAs across ni in hand-tuned asm wave-pair odd/even order
      (gate[0], up[0], gate[1], up[1], ...) with the next-atom loads spliced in
      the middle of the burst, so the longest run of consecutive MFMAs that
      share no intervening load drops from the legacy 5 toward ~2.

    Returns ``(gate_list, up_list)`` -- per-ni f32 outer accumulators, same
    order as ``n_tile_bases``.
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4
    nni = len(n_tile_bases)

    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    a_row_scale_base = b.mul(m_row, stride_a_scale)

    # Per-ni n_col / n_blk for the weight-scale index math.
    n_cols = [b.add(nb, lane_decode.n_in_atom) for nb in n_tile_bases]
    n_blks = [b.div(nc, c_group_k) for nc in n_cols]

    # Outer iter-args: gate[0..nni), up[0..nni).
    zero = atom.zero_acc(b)
    iter_args = []
    for ni in range(nni):
        iter_args.append((f"g_out_{tag}_{ni}", atom.zero_acc(b)))
    for ni in range(nni):
        iter_args.append((f"u_out_{tag}_{ni}", atom.zero_acc(b)))

    num_groups = b.div(K, c_group_k)
    outer = b.scf_for_iter(
        b.const_i32(0), num_groups, b.const_i32(1), iter_args, iv_name=f"kg_{tag}"
    )
    with outer as (kg, outs):
        _emit_loop_cadence_hint(b)
        gate_outer = list(outs[:nni])
        up_outer = list(outs[nni:])

        # Per-group scales (hoisted alongside the operand prefetch).
        a_scale_off = b.add(a_row_scale_base, kg)
        a_scale_v = b.global_load_f32(AScale, a_scale_off)
        kg_gate = b.mul(kg, stride_gate_scale)
        kg_up = b.mul(kg, stride_up_scale)
        gate_ab = []
        up_ab = []
        for ni in range(nni):
            gsc = b.global_load_f32(WGateScale, b.add(kg_gate, n_blks[ni]))
            usc = b.global_load_f32(WUpScale, b.add(kg_up, n_blks[ni]))
            gate_ab.append(b.fmul(a_scale_v, gsc))
            up_ab.append(b.fmul(a_scale_v, usc))

        k_group_base = b.mul(kg, c_group_k)

        def _a_at(kk):
            kbase = b.add(k_group_base, b.mul(b.const_i32(kk), c_atom_k))
            return _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=kbase,
                K=K,
            )

        def _gb_at(ni, kk):
            kbase = b.add(k_group_base, b.mul(b.const_i32(kk), c_atom_k))
            return _load_b_fp8(
                b,
                B=WGate,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_bases[ni],
                k_tile_base=kbase,
                N=K,
            )

        def _ub_at(ni, kk):
            kbase = b.add(k_group_base, b.mul(b.const_i32(kk), c_atom_k))
            return _load_b_fp8(
                b,
                B=WUp,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_bases[ni],
                k_tile_base=kbase,
                N=K,
            )

        # Fresh per-group accumulators, one gate + one up per ni.
        g_acc = [zero] * nni
        u_acc = [zero] * nni

        if dtla is not None and atoms_per_group == 1:
            # ---- DTLA path (GOAL 1): direct-to-LDS gate+up B operands -------
            # The A activation stays a cheap global->VGPR load (tiny, reused).
            # The dominant WGate/WUp weight streams go global->LDS->MFMA via
            # ``b.global_load_lds``, PING-PONG double-buffered over ni so the DMA
            # for ni+1 is issued (and in flight) while ni's MFMAs run, then the
            # vmcnt drain + ds_read feed the MFMA. Per-wave LDS base + CACHE_ALL
            # are threaded via the ``dtla`` bundle.
            kbase0 = k_group_base
            a_cur = _load_a_fp8(
                b,
                A=A,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                k_tile_base=kbase0,
                K=K,
            )

            def _stage(ni, slot_pair):
                # slot_pair 0/1 -> gate slot 2*pair, up slot 2*pair+1.
                _dtla_stage_b_fp8(
                    b,
                    B=WGate,
                    atom=atom,
                    lane_decode=lane_decode,
                    n_tile_base=n_tile_bases[ni],
                    k_tile_base=kbase0,
                    N=K,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair,
                    wave_lds_base=dtla["base"],
                    lane=dtla["lane"],
                    wave_size=dtla["wave_size"],
                )
                _dtla_stage_b_fp8(
                    b,
                    B=WUp,
                    atom=atom,
                    lane_decode=lane_decode,
                    n_tile_base=n_tile_bases[ni],
                    k_tile_base=kbase0,
                    N=K,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair + 1,
                    wave_lds_base=dtla["base"],
                    lane=dtla["lane"],
                    wave_size=dtla["wave_size"],
                )

            def _read(slot_pair):
                g = _dtla_read_b_fp8(
                    b,
                    atom=atom,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair,
                    lane=dtla["lane"],
                    warp_row_base=dtla["warp_row_base"],
                    wave_size=dtla["wave_size"],
                )
                u = _dtla_read_b_fp8(
                    b,
                    atom=atom,
                    stage_view=dtla["view"],
                    slot=2 * slot_pair + 1,
                    lane=dtla["lane"],
                    warp_row_base=dtla["warp_row_base"],
                    wave_size=dtla["wave_size"],
                )
                return g, u

            # Prime: stage ni=0 into ping slot 0.
            _stage(0, 0)
            # DMAs per _stage call: gate + up, each ceil(b_per_lane/16) chunks.
            chunks_per_frag = (atom.b_per_lane + DTLA_CHUNK - 1) // DTLA_CHUNK
            dmas_per_stage = 2 * chunks_per_frag
            for ni in range(nni):
                pair = ni % 2
                # Issue next ni's DMA into the OTHER slot BEFORE consuming this
                # ni's MFMAs (in flight under the MFMA). Drain only DOWN TO the
                # next stage's outstanding DMA count so the prefetch stays in
                # flight (vmcnt(0) here would serialize and kill the overlap --
                # the regression DTLA-alone trap). VMEM completes ~FIFO so this
                # ni's stage is done once only the next stage's DMAs remain.
                if ni + 1 < nni:
                    _stage(ni + 1, (ni + 1) % 2)
                    b.s_waitcnt(vmcnt=dmas_per_stage)
                else:
                    b.s_waitcnt(vmcnt=0)
                gb, ub = _read(pair)
                g_acc[ni] = _emit_mfma(b, atom, a_cur, gb, g_acc[ni])
                u_acc[ni] = _emit_mfma(b, atom, a_cur, ub, u_acc[ni])
                _emit_sgb_gateup_dtla(b, n_mfma=2)
        else:
            # ---- legacy global->VGPR->MFMA path ----------------------------
            # Prefetch atom 0 operands (A shared + all ni B fragments).
            a_cur = _a_at(0)
            gb_cur = [_gb_at(ni, 0) for ni in range(nni)]
            ub_cur = [_ub_at(ni, 0) for ni in range(nni)]

            for kk in range(atoms_per_group):
                last = kk + 1 >= atoms_per_group
                # Issue the NEXT atom's shared A load up front (in flight while
                # the current atom's MFMAs run).
                if not last:
                    a_next = _a_at(kk + 1)
                # Wave-pair interleave: for each ni, emit gate then up, and
                # splice the next-atom B prefetch for ni before ni's MFMAs.
                for ni in range(nni):
                    if not last:
                        gb_next_ni = _gb_at(ni, kk + 1)
                        ub_next_ni = _ub_at(ni, kk + 1)
                    g_acc[ni] = _emit_mfma(b, atom, a_cur, gb_cur[ni], g_acc[ni])
                    u_acc[ni] = _emit_mfma(b, atom, a_cur, ub_cur[ni], u_acc[ni])
                    if not last:
                        gb_cur[ni] = gb_next_ni
                        ub_cur[ni] = ub_next_ni
                if not last:
                    a_cur = a_next

        # Fold each group accumulator (post-MFMA) by a_scale * b_scale.
        new_gate = []
        new_up = []
        for ni in range(nni):
            gvec = b.vector_splat(gate_ab[ni], atom.c_per_lane)
            uvec = b.vector_splat(up_ab[ni], atom.c_per_lane)
            new_gate.append(b.vector_fma(g_acc[ni], gvec, gate_outer[ni]))
            new_up.append(b.vector_fma(u_acc[ni], uvec, up_outer[ni]))
        b.scf_yield(*(new_gate + new_up))

    res = outer.results
    return list(res[:nni]), list(res[nni:])


def _global_load_fp8_vec(b: IRBuilder, ptr: Value, addr: Value, n: int) -> Value:
    """Coalesced fp8 vector load of ``n`` contiguous bytes at ``addr``.

    ``global_load_vN`` for fp8 caps at n=16 (ds/global payload width); the
    K=128 hero atom needs ``a_per_lane`` == ``b_per_lane`` == 32 fp8 per lane.
    Split into ceil(n/16) consecutive 16-wide loads and concat -- the bytes are
    contiguous so this is two ``global_load_dwordx4`` over the same 32-byte run,
    bit-identical to a single 32-wide load.
    """
    if n <= 16:
        return b.global_load_vN(ptr, addr, FP8E4M3, n)
    acc = None
    off = 0
    while off < n:
        chunk = min(16, n - off)
        a = addr if off == 0 else b.add(addr, b.const_i32(off))
        v = b.global_load_vN(ptr, a, FP8E4M3, chunk)
        acc = v if acc is None else b.vec_concat(acc, v)
        off += chunk
    return acc


def _load_a_fp8(
    b: IRBuilder, *, A, atom, lane_decode, m_tile_base, k_tile_base, K
) -> Value:
    """Per-lane fp8 A load for row-major (M, K) -- K contiguous; scalar loads."""
    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    a_addr = b.add(b.mul(m_row, K), k_base)
    # The a_per_lane fp8 bytes are CONTIGUOUS along K (addr + j) -> one (or, for
    # the K=128 atom's 32-wide fragment, two concatenated) coalesced vector
    # load(s) instead of a_per_lane byte-granular global_load_ubyte.
    # Bit-identical values.
    return _global_load_fp8_vec(b, A, a_addr, atom.a_per_lane)


def _load_b_fp8(
    b: IRBuilder, *, B, atom, lane_decode, n_tile_base, k_tile_base, N
) -> Value:
    """Per-lane fp8 B load for row-major (K, N) -- col-strided scalar loads.

    ``N`` is the weight's K extent (the matrix is stored (out, K) row-major;
    B[k, n] address = (n_tile_base + n_in_atom) * K + (k_base + j)).
    """
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    # W is stored (out_rows, K) row-major -> address of W[n_col, k] = n_col*K + k.
    row_base = b.mul(n_col, N)
    # The b_per_lane fp8 weight bytes are CONTIGUOUS along K (k_base + j) ->
    # one (or, for the K=128 atom's 32-wide fragment, two concatenated)
    # coalesced vector load(s) instead of b_per_lane byte-granular
    # global_load_ubyte. This is the dominant HBM weight traffic; bit-identical
    # values.
    b_addr = b.add(row_base, k_base)
    return _global_load_fp8_vec(b, B, b_addr, atom.b_per_lane)


# ---------------------------------------------------------------------------
# DIRECT-TO-LDS (DTLA) B-operand staging for the gate+up GEMM
# ---------------------------------------------------------------------------
#
# GOAL 1: replace the global->VGPR->MFMA weight load of the dominant gate/up
# weight stream with a global->LDS->MFMA path via the additive
# ``b.global_load_lds`` op (== ``global_load_lds_dwordx4`` ISA, the flat sibling
# of hand-tuned asm's ``buffer_load_dwordx4 ... offen lds``). The DMA bypasses the VGPR
# round-trip.
#
# LANE DISTRIBUTION (the correctness model): ``llvm.amdgcn.global.load.lds``
# takes a WAVE-UNIFORM LDS destination; the HARDWARE spreads the 64 lanes'
# payloads lane-contiguously -- lane L writes its ``size_bytes`` to
# ``lds_dst + L*size_bytes``. So the destination passed to the intrinsic must NOT
# carry a per-lane term (the lane term lives in the per-lane SOURCE pointer). A
# 32-byte fp8 fragment exceeds the 16-byte payload cap, so it is two 16-byte DMA
# chunks landing in two SEPARATE 64*16-byte lane-blocks: chunk c at
# ``slot_base + c*(wave_size*16)``, each lane's source at ``src + c*16``. The
# read-back gathers both half-blocks (rows (slot*chunks + c)*wave_size + L).
#
# CRITICAL coupling (DTLA regresses ALONE): the DMA for the NEXT ni cell is
# issued BEFORE the current cell's MFMAs, so it is in flight under the MFMA. The
# LDS slot is PER-WAVE (base biased by wave_id*wave_bytes) and PING-PONG
# double-buffered over ni so the next DMA does not stomp the fragment the current
# MFMA is still reading. CACHE_ALL keeps the reused weights resident in L2.

# 16-byte direct-to-LDS payload cap (== global_load_lds_dwordx4 on gfx950).
DTLA_CHUNK = 16


def _dtla_stage_b_fp8(
    b: IRBuilder,
    *,
    B: Value,
    atom: MfmaAtom,
    lane_decode,
    n_tile_base: Value,
    k_tile_base: Value,
    N: Value,
    stage_view,
    slot: int,
    wave_lds_base: Value,
    lane: Value,
    wave_size: int,
) -> None:
    """Issue the direct-to-LDS DMA of one lane's ``b_per_lane`` fp8 weight bytes.

    Lane ``L`` streams the SAME ``b_per_lane`` contiguous-along-K fp8 weight bytes
    that :func:`_load_b_fp8` would have loaded into a VGPR, but the HARDWARE
    distributes them: chunk ``c`` lands lane-contiguously at
    ``slot_base + c*wave_size*16 + L*16`` (the destination is wave-uniform; the
    per-lane spread is implicit). The DMA completes on the VMEM counter; the
    caller drains ``s_waitcnt(vmcnt=0)`` before the read-back.
    """
    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.b_per_lane))
    k_base = b.add(k_tile_base, k_lane_start)
    row_base = b.mul(n_col, N)
    src_elem = b.add(row_base, k_base)  # per-lane element (== byte) source offset

    frag_bytes = atom.b_per_lane  # fp8 == 1 byte/elem
    chunks = (frag_bytes + DTLA_CHUNK - 1) // DTLA_CHUNK
    block_bytes = wave_size * DTLA_CHUNK  # one DMA's lane-block footprint
    slot_base_off = slot * chunks * block_bytes

    for c in range(chunks):
        chunk = min(DTLA_CHUNK, frag_bytes - c * DTLA_CHUNK)
        src = src_elem if c == 0 else b.add(src_elem, b.const_i32(c * DTLA_CHUNK))
        # WAVE-UNIFORM destination for chunk c (no per-lane term -- HW spreads).
        dst = b.smem_ptr_add(
            wave_lds_base, b.const_i64(slot_base_off + c * block_bytes)
        )
        b.global_load_lds(B, src, dst, chunk, CACHE_ALL)


def _dtla_read_b_fp8(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    stage_view,
    slot: int,
    lane: Value,
    warp_row_base: Value,
    wave_size: int,
) -> Value:
    """Read back a lane's fp8 B fragment staged in LDS by :func:`_dtla_stage_b_fp8`.

    Mirrors the staging lane-block layout: chunk ``c`` of lane ``L`` lives at row
    ``warp_row_base + (slot*chunks + c)*wave_size + L`` (the 2-D view has a
    16-byte column). Concatenate the per-chunk 16-wide ``ds_read_b128`` reads to
    re-form the full ``b_per_lane`` fragment -- bit-identical to the VGPR load.
    """
    frag = atom.b_per_lane
    chunks = (frag + DTLA_CHUNK - 1) // DTLA_CHUNK
    acc = None
    for c in range(chunks):
        chunk = min(DTLA_CHUNK, frag - c * DTLA_CHUNK)
        row = b.add(
            warp_row_base,
            b.add(b.const_i32((slot * chunks + c) * wave_size), lane),
        )
        v = b.smem_load_vN(stage_view.base, row, b.const_i32(0), dtype=FP8E4M3, n=chunk)
        acc = v if acc is None else b.vec_concat(acc, v)
    return acc


# ---------------------------------------------------------------------------
# STAGE 2: down fp8 GEMM (LDS-A) + per-128-group dequant + weighted atomic Y
# ---------------------------------------------------------------------------


def _load_a_fp8_lds(
    b: IRBuilder, *, a_view, atom, lane_decode, m_tile_base, k_tile_base
) -> Value:
    """Per-lane fp8 A load for the down GEMM, reading the LDS-resident Hidden.

    The down-GEMM A operand is ``Hidden_smem`` (``[tile_m, tile_n_inter]``,
    logical ``(row=m, col=inter)``). The contraction axis is the inter slice, so
    for output row ``m_row`` (= ``m_tile_base + m_in_atom``) the lane reads its
    ``a_per_lane`` contiguous fp8 elements starting at inter column
    ``k_tile_base + k_blk * a_per_lane``. This is the SAME logical ``(m, inter)``
    cell the dynamic-quant Pass C wrote (BUILD_SPEC_FP8 Section 3.5: the implicit
    reshape -- cshuffle/quant write addr == down MFMA A-read addr).
    """
    m_row = b.add(m_tile_base, lane_decode.m_in_atom)
    k_lane_start = b.mul(lane_decode.k_blk, b.const_i32(atom.a_per_lane))
    k_col = b.add(k_tile_base, k_lane_start)
    # smem_load_vN caps fp8 at n=16; the K=128 hero atom needs a 32-wide
    # fragment. Split into ceil(n/16) contiguous 16-wide ds_read_b128 loads and
    # concat -- the inter columns are contiguous in LDS so this is bit-identical
    # to a single 32-wide read.
    n = atom.a_per_lane
    if n <= 16:
        return b.smem_load_vN(a_view.base, m_row, k_col, dtype=FP8E4M3, n=n)
    acc = None
    off = 0
    while off < n:
        chunk = min(16, n - off)
        c = k_col if off == 0 else b.add(k_col, b.const_i32(off))
        v = b.smem_load_vN(a_view.base, m_row, c, dtype=FP8E4M3, n=chunk)
        acc = v if acc is None else b.vec_concat(acc, v)
        off += chunk
    return acc


def _emit_fp8_down_group_gemm(
    b: IRBuilder,
    *,
    a_view,
    WDown: Value,
    WDownScale: Value,
    atom: MfmaAtom,
    lane_decode,
    n_tile_base: Value,
    scale_view,
    inter_slice: int,
    inter_full: Value,
    inter_blk_base: Value,
    stride_down_scale: Value,
    m_row_base: Value,
    tag: str,
) -> Value:
    """Down fp8 GEMM for one warp-atom output cell -> per-lane f32 vector.

    Group-accumulator pattern (BUILD_SPEC_FP8 Section 1.2 / 2.2): the outer loop
    walks 128-wide groups along the inter CONTRACTION (= this TG's inter slice,
    ``tile_n_inter`` = ``inter_slice``). Per group, 4 fp8_16x16x32 atoms
    accumulate into a FRESH ``group_acc``; the group is folded into the outer
    accumulator scaled by ``hidden_dyn_scale * down_b_scale`` -- a single
    ``v_pk_fma_f32``.

    Index conventions (the two-space split):
    * **A (Hidden, LDS):** read at LOCAL inter columns ``[0, inter_slice)``; its
      per-128-block dynamic scale is ``HiddenScale_smem[m_row, local_blk]`` (the
      dynamic scale is only defined over THIS TG's slice).
    * **B (W_down, global):** stored ``(H_out, I_full)`` row-major; the
      contraction column is the GLOBAL inter position ``inter_blk_base*128 + local``
      and the row stride is the FULL inter dim ``inter_full``. The B-side scale
      ``WDownScale`` is indexed per (GLOBAL inter-block, H_out-block):
      ``off = (inter_blk_base + kg) * stride_down_scale + h_out_blk``.

    The W_down per-expert pointer base is folded by the caller; the inter slice
    base is folded here (column ``inter_blk_base*128 + local``), NOT into the
    pointer, so the row stride math stays the full-inter stride.
    """
    c_group_k = b.const_i32(GROUP_K)
    c_atom_k = b.const_i32(atom.k)
    atoms_per_group = GROUP_K // atom.k  # 4

    n_col = b.add(n_tile_base, lane_decode.n_in_atom)
    h_out_blk = b.div(n_col, c_group_k)
    # CORRECTNESS FIX: the down-GEMM A operand row must follow the per-mi atom
    # m-base (m_row_base = down_warp_m_off + mi*atom.m), not a hardcoded 0.
    # HiddenScale is row-uniform-within-block so this term is harmless for the
    # scale read, but threading it keeps the A-read (below) and scale-read on
    # the same row basis. mfmas_m_down=1 (tile_m=16) leaves it == 0; tile_m=32
    # makes mi=1 read Hidden LDS rows 16-31 instead of 0-15.
    m_row = b.add(m_row_base, lane_decode.m_in_atom)

    # Global inter column base for this TG's slice (W_down contraction origin).
    inter_col_base = b.mul(inter_blk_base, c_group_k)

    zero = atom.zero_acc(b)
    outer_zero = atom.zero_acc(b)

    # num_groups = inter_slice // GROUP_K (local inter slice / 128).
    num_groups = b.const_i32(inter_slice // GROUP_K)
    outer = b.scf_for_iter(
        b.const_i32(0),
        num_groups,
        b.const_i32(1),
        [(f"down_outer_{tag}", outer_zero)],
        iv_name=f"dg_{tag}",
    )
    with outer as (kg, (down_outer,)):
        _emit_loop_cadence_hint(b)
        # A-side dynamic scale: HiddenScale_smem[m_row, local-inter-block kg].
        a_scale_v = b.vec_extract(
            b.smem_load_vN(scale_view.base, m_row, kg, dtype=F32, n=1), 0
        )
        # B-side W_down scale: per (GLOBAL inter-block, H_out-block).
        global_blk = b.add(inter_blk_base, kg)
        down_scale_off = b.add(b.mul(global_blk, stride_down_scale), h_out_blk)
        down_scale_v = b.global_load_f32(WDownScale, down_scale_off)
        ab_scale = b.fmul(a_scale_v, down_scale_v)

        local_k_group = b.mul(kg, c_group_k)
        global_k_group = b.add(inter_col_base, local_k_group)

        # L5: software-pipeline the W_down (B) tile -- the dominant HBM stream of
        # this stage. The inner k-group loop has a COMPILE-TIME-CONSTANT trip
        # count (``atoms_per_group`` == 4 fp8_16x16x32 atoms), so it is unrolled
        # in Python and the B=W_down fragments are register double-buffered:
        # the next atom's global B load is ISSUED before the current b_frag is
        # consumed in ``atom.emit``, keeping a load in flight under the MFMA.
        # (An scf.for iter-arg cannot carry the fp8e4m3 fragment -- the LLVM
        # lowering has no loop-carried fp8 vector type -- so the prefetch is
        # expressed by Python unroll instead, same dataflow as the f16
        # ``_emit_moe_down_kloop_lds_a`` b_next pattern. A stays a direct LDS
        # read: cheap, already resident.)
        def _load_b_at(kk_idx):
            return _load_b_fp8(
                b,
                B=WDown,
                atom=atom,
                lane_decode=lane_decode,
                n_tile_base=n_tile_base,
                k_tile_base=b.add(global_k_group, b.mul(b.const_i32(kk_idx), c_atom_k)),
                N=inter_full,
            )

        def _load_a_at(kk_idx):
            return _load_a_fp8_lds(
                b,
                a_view=a_view,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_row_base,
                k_tile_base=b.add(local_k_group, b.mul(b.const_i32(kk_idx), c_atom_k)),
            )

        # Prefetch the first atom's W_down fragment, then pipeline.
        b_cur = _load_b_at(0)
        group_acc = zero
        for kk in range(atoms_per_group):
            a_frag = _load_a_at(kk)
            # Issue the NEXT atom's W_down load (in flight during the MFMA).
            if kk + 1 < atoms_per_group:
                b_next = _load_b_at(kk + 1)
            d_new = _emit_mfma(b, atom, a_frag, b_cur, group_acc)
            group_acc = d_new
            if kk + 1 < atoms_per_group:
                b_cur = b_next

        # D5 sgb: place the next-group W_down VMEM under this group's MFMA(s).
        _emit_sgb_down_group(b, n_mfma=atoms_per_group)

        scale_vec = b.vector_splat(ab_scale, atom.c_per_lane)
        down_outer_new = b.vector_fma(group_acc, scale_vec, down_outer)
        b.scf_yield(down_outer_new)

    return outer.results[0]


def _emit_down_atomic_reduce(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    down_list,
    warp_m_off: Value,
    warp_n_off: Value,
    lane: Value,
    mfmas_m: int,
    mfmas_n: int,
    block_m_off: Value,
    ho_off: Value,
    H_out: Value,
    SortedTokenIds: Value,
    SortedWeights: Value,
    Y: Value,
    tokens: Value,
) -> None:
    """Weighted, token-validity-masked atomic reduce of the down result into Y.

    ``Y[token, h_out] += weight * down_dq`` (f32 atomic add). Padded rows
    (sorted token id < 0 or >= tokens) are skipped. Mirrors the f16
    ``_emit_down_reduce_epilogue_atomic`` but on the raw-atom lane layout used by
    the rest of this fp8 file (``atom.lane_to_output``). The sorted-token bucket
    index is the GLOBAL output row ``block_m_off + warp/atom row``; the output
    column is ``ho_off + warp/atom col`` along H_out.
    """
    c0 = b.const_i32(0)
    # L1: the sorted-token id and routing weight are per-ROW (bucket == row); they
    # do NOT depend on the output column (ni / col_in). The pre-L1 nest reloaded
    # SortedTokenIds + SortedWeights and re-checked validity for EVERY (mi, ni, i)
    # output slot, each `global_load_f32(SortedWeights) -> s_waitcnt vmcnt(0) ->
    # v_mul -> global_atomic_add` forcing one full drain per slot (~12-16 of 25
    # epilogue drains). Hoist the token/weight load + validity check to ONE per
    # (mi, i) row, then batch the per-ni atomics inside a single scf_if. This
    # collapses mfmas_n redundant drains per row into one.
    for mi in range(mfmas_m):
        # Lever G drain reduction: the per-row SortedTokenIds / SortedWeights
        # loads do NOT depend on each other. The pre-lever code loaded the weight
        # INSIDE each row's `scf_if(valid)` block, forcing a separate
        # `global_load_f32 -> vmcnt(0) -> v_mul -> atomic` serialization per row
        # (c_per_lane full drains in the epilogue). Here we ISSUE every row's
        # token + weight load up front into DISTINCT registers, then drain ONCE
        # for the whole batch, then run the per-row validity-masked atomics with
        # the operands already resident. The weight bucket index is always a
        # valid slot (padded-row slots have weights too), so the unconditional
        # hoisted load is safe; the validity check still gates the atomic store.
        rows = []
        for i in range(atom.c_per_lane):
            row_in, col_in = atom.lane_to_output(b, lane, i)
            row = b.add(
                block_m_off,
                b.add(warp_m_off, b.add(b.const_i32(mi * atom.m), row_in)),
            )
            bucket = row
            token = b.global_load_i32(SortedTokenIds, bucket)
            w = b.global_load_f32(SortedWeights, bucket)
            rows.append((i, col_in, token, w))
        # One rolling drain covers all c_per_lane (token,weight) loads instead of
        # one vmcnt(0) per row.
        b.s_waitcnt(vmcnt=0)
        for i, col_in, token, w in rows:
            valid = b.land(b.cmp_ge(token, c0), b.cmp_lt(token, tokens))
            with b.scf_if(valid):
                token_h = b.mul(token, H_out)
                for ni in range(mfmas_n):
                    flat = mi * mfmas_n + ni
                    acc = down_list[flat]
                    col = b.add(
                        ho_off,
                        b.add(warp_n_off, b.add(b.const_i32(ni * atom.n), col_in)),
                    )
                    v = b.vec_extract(acc, i)
                    contrib = b.fmul(w, v)
                    y_off = b.add(token_h, col)
                    b.global_atomic_add(Y, y_off, contrib)


# ---------------------------------------------------------------------------
# STAGE 1b: SiLU(gate)*up + dynamic-quantize Hidden to fp8 in LDS
# ---------------------------------------------------------------------------


def _silu_mul_f32(
    b: IRBuilder, g: Value, u: Value, *, one_f32: Value, c_neg_log2e: Value
) -> Value:
    """f32 SwiGLU chain ``silu(g) * u`` (sigmoid via exp2), op order matched."""
    sig = b.rcp(b.fadd(one_f32, b.exp2(b.fmul(c_neg_log2e, g))))
    silu = b.fmul(g, sig)
    return b.fmul(silu, u)


def _store_hidden_f32_pass(
    b: IRBuilder,
    *,
    atom: MfmaAtom,
    gate_list,
    up_list,
    f32_view,
    warp_m_off: Value,
    warp_n_off: Value,
    lane: Value,
    mfmas_m: int,
    mfmas_n: int,
    one_f32: Value,
    c_neg_log2e: Value,
    c_floor: Value,
) -> Value:
    """Fused Pass A: silu(gate)*up -> f32 LDS scratch AND in-register amax.

    Writes each MFMA output cell to the f32 LDS scratch (still consumed by the
    requantize pass) while accumulating, per lane, the absolute max over every
    cell this lane owns. Because ``tile_n_inter`` is partitioned so that an
    entire warp's N-extent (warp_n_off .. +mfmas_n*atom.n) lands inside ONE
    128-inter scale block, the per-lane partial here belongs to exactly one
    block; the caller reduces the 64 lane partials of a warp (and the 2 warps
    that share a block) to the final per-block amax. Returns the per-lane
    partial amax (f32), already floored at ``AMAX_FLOOR``.
    """
    amax_partial = c_floor
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            flat = mi * mfmas_n + ni
            g_vec = gate_list[flat]
            u_vec = up_list[flat]
            for i in range(atom.c_per_lane):
                row_in, col_in = atom.lane_to_output(b, lane, i)
                row = b.add(warp_m_off, b.add(b.const_i32(mi * atom.m), row_in))
                col = b.add(warp_n_off, b.add(b.const_i32(ni * atom.n), col_in))
                g = b.vec_extract(g_vec, i)
                u = b.vec_extract(u_vec, i)
                h = _silu_mul_f32(b, g, u, one_f32=one_f32, c_neg_log2e=c_neg_log2e)
                f32_view_store(b, f32_view, row, col, h)
                amax_partial = b.fmax(amax_partial, b.fabs(h))
    return amax_partial


def f32_view_store(b: IRBuilder, view, row: Value, col: Value, val: Value) -> None:
    b.smem_store_vN(view.base, [row, col], val, 1)


def f32_view_load(b: IRBuilder, view, row: Value, col: Value) -> Value:
    v = b.smem_load_vN(view.base, row, col, dtype=F32, n=1)
    return b.vec_extract(v, 0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_moe_fused_mega_gemm_fp8(
    spec: FusedMegaKernelSpecFp8, arch: str = "gfx950"
) -> KernelDef:
    """Build the STAGE 1 fp8 fused-MoE mega-kernel.

    Current implementation covers STAGE 1 (gate+up fp8 GEMM with per-128-block
    dequant -> SiLU*up f32 -> dynamic-quant to fp8 staged in the persistent LDS
    ``Hidden_smem`` + per-block scales in ``HiddenScale_smem``). The down GEMM +
    weighted atomic reduce (STAGE 2) is a documented stub at the end of the body.
    """
    ok, why, _ = validate_arch_and_block_size(arch, spec.block_size)
    if not ok:
        raise ValueError(f"invalid fp8 fused-mega spec for {arch}: {why}")
    atom = spec.gate_up_atom()
    # L6: the unscaled fp8 16x16x128 hero atom reuses the (catalog-registered)
    # ``mfma.scale.f32.16x16x128.f8f6f4`` intrinsic with the in-instruction E8M0
    # scales pinned to the neutral value (verified numerically standalone), so it
    # is gfx950-valid even though the plain unscaled 16x16x128 fp8 SHAPE is not a
    # separate JSON catalog row. Only run the per-arch catalog guard for atoms
    # that ARE catalog shapes (the legacy 16x16x32 path); skip it for the hero
    # atom to avoid a spurious NotImplementedError (this is a guard against
    # gfx950-only intrinsics reaching comgr -- which this atom is not).
    if atom.k != 128:
        validate_mfma_atom_in_catalog(atom, arch, where="moe_fused_mega_fp8")

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    # ---- params (BUILD_SPEC_FP8 Section 2.7) ---------------------------
    A = b.param("A", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16)
    WGate = b.param(
        "WGate", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    WUp = b.param(
        "WUp", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    WDown = b.param(
        "WDown", PtrType(FP8E4M3, "global"), noalias=True, readonly=True, align=16
    )
    AScale = b.param("AScale", PtrType(F32, "global"), readonly=True, align=4)
    WGateScale = b.param("WGateScale", PtrType(F32, "global"), readonly=True, align=4)
    WUpScale = b.param("WUpScale", PtrType(F32, "global"), readonly=True, align=4)
    WDownScale = b.param("WDownScale", PtrType(F32, "global"), readonly=True, align=4)
    SortedTokenIds = b.param(
        "SortedTokenIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    SortedWeights = b.param(
        "SortedWeights", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    BlockExpertIds = b.param(
        "BlockExpertIds", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    Y = b.param("Y", PtrType(F32, "global"), align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)  # = I (inter dim)
    K = b.param("K", I32)  # = H (hidden contraction)
    H_out = b.param("H_out", I32)  # = H (down output)
    stride_a = b.param(
        "stride_a", I32
    )  # noqa: F841 -- ABI (A is dense, gather elsewhere)
    stride_b_gate = b.param("stride_b_gate", I32)
    stride_b_up = b.param("stride_b_up", I32)
    stride_b_down = b.param("stride_b_down", I32)  # noqa: F841 -- used in STAGE 2
    stride_a_scale = b.param("stride_a_scale", I32)
    stride_gate_scale = b.param("stride_gate_scale", I32)
    stride_up_scale = b.param("stride_up_scale", I32)
    stride_down_scale = b.param("stride_down_scale", I32)  # noqa: F841 -- STAGE 2
    # Per-expert ELEMENT stride for the weight scale tensors (scales are
    # per-expert, but the scale pointer is NOT folded by _b_base; fold here).
    stride_gate_scale_e = b.param("stride_gate_scale_e", I32)
    stride_up_scale_e = b.param("stride_up_scale_e", I32)
    stride_down_scale_e = b.param("stride_down_scale_e", I32)
    slot_size = b.param("slot_size", I32)  # noqa: F841 -- ABI
    tokens = b.param("tokens", I32)  # noqa: F841 -- used in STAGE 2 epilogue

    tile_m = spec.tile_m
    tile_n = spec.tile_n_inter
    n_blocks = tile_n // GROUP_K

    # LEVER (fuse-quant) invariant: each warp's N-extent must lie inside exactly
    # one 128-inter scale block, AND each block must be covered by an integer
    # number of consecutive warps, so the register-amax combine
    # (block ``blk`` <- warps {warps_per_block*blk .. +warps_per_block-1}) is
    # exact. With warp_m=1 the warp N-extent = warp_n_cols below.
    warp_n_cols = tile_n // spec.warp_n  # mfmas_n * atom.n
    warps_per_block = GROUP_K // warp_n_cols
    if (
        spec.warp_m != 1
        or warp_n_cols * spec.warp_n != tile_n
        or GROUP_K % warp_n_cols != 0
        or warps_per_block * n_blocks != spec.warp_n
    ):
        raise ValueError(
            "fuse-quant lever requires warp_m==1 and warps to tile the "
            f"128-inter blocks evenly (got tile_n={tile_n}, warp_n="
            f"{spec.warp_n}, warp_n_cols={warp_n_cols})"
        )

    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(spec.warp_n)
    c_block_m = b.const_i32(tile_m)
    c_block_n = b.const_i32(tile_n)
    c0 = b.const_i32(0)

    # ---- block/thread prelude -----------------------------------------
    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    m_block_idx = b.block_id_y()
    inter_block_x = None  # set below only when XCD remap is active (OFF = no-op)
    if _XCD_REMAP > 0:
        inter_block_x = b.block_id_x()
        # D8 XCD remap (grid shape unchanged): re-tile the inter-slice id so that
        # workgroups landing on the same XCD (HW round-robins consecutive linear
        # ids across the _XCD_REMAP XCDs) walk a CONTIGUOUS run of inter slices
        # for this m_block, maximising per-XCD L2 residency of the gate/up/down
        # weight columns. gx = ceil(N / tile_n_inter) (tile_n_inter compile-time).
        c_xcd = b.const_i32(_XCD_REMAP)
        c_tn = b.const_i32(spec.tile_n_inter)
        gx = b.div(b.add(N, b.const_i32(spec.tile_n_inter - 1)), c_tn)
        # n_full = (gx // XCD) * XCD: the largest multiple of XCD <= gx. Only the
        # [0, n_full) prefix is XCD-transposed (a TRUE bijection over that range);
        # the [n_full, gx) tail keeps identity. This guarantees the overall map is
        # a permutation of [0, gx) for ANY gx (correctness-preserving): every
        # inter slice computed exactly once, atomic-reduce coverage intact.
        cols = b.div(gx, c_xcd)  # slots per XCD over the full prefix
        n_full = b.mul(cols, c_xcd)
        orig = inter_block_x
        # transpose within the prefix: new = (orig % XCD) * cols + (orig / XCD)
        xcd = b.mod(orig, c_xcd)
        slot = b.div(orig, c_xcd)
        remapped = b.add(b.mul(xcd, cols), slot)
        in_prefix = b.cmp_lt(orig, n_full)
        inter_block_x = b.select(in_prefix, remapped, orig)
    expert_idx = b.global_load_i32(BlockExpertIds, m_block_idx)

    # ---- BYTE-BASE FIX (BUILD_SPEC_FP8 Section 2.4): fp8 weights = 1 byte.
    elem_bytes_b = b.const_i64(1)

    def _b_base(ptr: Value, stride_b: Value) -> Value:
        bytes_off = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_b, I64)), elem_bytes_b
        )
        return b.global_ptr_add(ptr, bytes_off)

    WGate = _b_base(WGate, stride_b_gate)
    WUp = _b_base(WUp, stride_b_up)
    WDown = _b_base(WDown, stride_b_down)

    # Per-expert base for the f32 weight scale tensors (4-byte elements). The
    # scale index math inside the k-loops carries no expert term, so the
    # per-expert slice is selected here on the pointer.
    def _scale_base(ptr: Value, stride_e: Value) -> Value:
        bytes_off = b.mul(
            b.mul(b.sext(expert_idx, I64), b.sext(stride_e, I64)),
            b.const_i64(4),
        )
        return b.global_ptr_add(ptr, bytes_off)

    WGateScale = _scale_base(WGateScale, stride_gate_scale_e)
    WUpScale = _scale_base(WUpScale, stride_up_scale_e)
    WDownScale = _scale_base(WDownScale, stride_down_scale_e)

    block_m_off = b.mul(m_block_idx, c_block_m)
    gu_n_off = b.mul(
        inter_block_x if inter_block_x is not None else b.block_id_x(), c_block_n
    )

    # ---- LDS allocations ----------------------------------------------
    # Persistent fp8 Hidden buffer (half the f16 bytes): silu(gate)*up quantized
    # here, reused as the down-GEMM LDS-resident A operand in STAGE 2.
    Hidden_smem = b.smem_alloc(FP8E4M3, [tile_m, tile_n], name_hint="Hidden_smem")
    # Per-(row, 128-inter-block) dynamic scales for the down dequant (STAGE 2).
    HiddenScale_smem = b.smem_alloc(
        F32, [tile_m, n_blocks], name_hint="HiddenScale_smem"
    )
    # f32 scratch for the exact per-block amax reduction (STAGE 1 only).
    HiddenF32_smem = b.smem_alloc(F32, [tile_m, tile_n], name_hint="HiddenF32_smem")
    # Tiny per-warp amax partials (one f32 per warp). Each warp's N-extent lands
    # inside ONE 128-inter block, so warps {2*blk, 2*blk+1} cover block ``blk``.
    n_warps = spec.warp_m * spec.warp_n
    WarpAmax_smem = b.smem_alloc(F32, [n_warps], name_hint="WarpAmax_smem")

    # ---- DTLA gate+up B staging (GOAL 1) -----------------------------------
    # Per-wave, ping-pong (4 logical slots: gate/up x 2 ni-buffers) direct-to-LDS
    # landing zone. Each slot holds DTLA_CHUNKS lane-blocks of (wave_size x 16B)
    # -- the lane-contiguous spread the global.load.lds HW imposes. Shape
    # [n_warps*DTLA_SLOTS*DTLA_CHUNKS*wave_size, 16] fp8: 4*4*2*64*16 = 32 KiB
    # at the canonical geometry.
    DTLA_SLOTS = 4
    DTLA_CHUNKS = (atom.b_per_lane + DTLA_CHUNK - 1) // DTLA_CHUNK
    bstage_rows = n_warps * DTLA_SLOTS * DTLA_CHUNKS * spec.wave_size
    BStage_smem = b.smem_alloc(
        FP8E4M3, [bstage_rows, DTLA_CHUNK], name_hint="BStage_smem"
    )
    bstage_view = TensorView(
        base=BStage_smem,
        desc=TensorDescriptor.packed((bstage_rows, DTLA_CHUNK), FP8E4M3),
        addr_space="lds",
    )

    f32_view = TensorView(
        base=HiddenF32_smem,
        desc=TensorDescriptor.packed((tile_m, tile_n), F32),
        addr_space="lds",
    )
    fp8_view = TensorView(
        base=Hidden_smem,
        desc=TensorDescriptor.packed((tile_m, tile_n), FP8E4M3),
        addr_space="lds",
    )
    scale_view = TensorView(
        base=HiddenScale_smem,
        desc=TensorDescriptor.packed((tile_m, n_blocks), F32),
        addr_space="lds",
    )

    lane_decode = decode_mfma_lanes(b, atom, lane)
    mfmas_m = spec.mfmas_m
    mfmas_n = spec.mfmas_n
    mfmas_m_down = spec.mfmas_m_down
    mfmas_n_down = spec.mfmas_n_down
    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * atom.m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * atom.n))

    c_neg_log2e = b.const_f32(-1.4426950408889634)
    one_f32 = b.const_f32(1.0)
    c_fp8_max = b.const_f32(FP8_MAX)
    c_floor = b.const_f32(AMAX_FLOOR)

    c_group_k = b.const_i32(GROUP_K)
    c_threads = b.const_i32(spec.block_size)
    _c_n_blocks = b.const_i32(n_blocks)
    c_tile_n = b.const_i32(tile_n)

    def _emit_body() -> None:
        # ---- STAGE 1a: gate + up fp8 GEMM -> f32 (per-128-block dequant) ----
        # Per warp atom (mi, ni) -> one gate_dq / up_dq vector. The contraction
        # group GEMM is run per warp-atom output position; the lane decode and
        # tile bases select the rows/cols this lane owns.
        # COMBINATION lever: one fused K-loop per mi row covering ALL ni cells
        # (shared A read + register-double-buffered B prefetch + wave-pair
        # odd/even MFMA interleave). gate_list/up_list keep the row-major
        # (mi, ni) ordering the downstream Pass A / down stage expects.
        gate_list = []
        up_list = []
        # DTLA bundle (GOAL 1): per-wave LDS base + read-row base for the
        # direct-to-LDS gate+up B staging. The DMA dst i64 = smem_addr_of +
        # warp_id * (DTLA_SLOTS*wave_size*b_per_lane); the read row base =
        # warp_id * DTLA_SLOTS*wave_size -- same byte (packed, b_per_lane/row).
        bstage_base_i64 = b.smem_addr_of(BStage_smem)
        warp_rows = DTLA_SLOTS * DTLA_CHUNKS * spec.wave_size
        warp_wave_bytes = warp_rows * DTLA_CHUNK
        wave_lds_base = b.smem_ptr_add(
            bstage_base_i64,
            b.sext(b.mul(warp_id, b.const_i32(warp_wave_bytes)), I64),
        )
        warp_row_base = b.mul(warp_id, b.const_i32(warp_rows))
        dtla_bundle = {
            "view": bstage_view,
            "base": wave_lds_base,
            "warp_row_base": warp_row_base,
            "lane": lane,
            "wave_size": spec.wave_size,
        }
        for mi in range(mfmas_m):
            m_tile_base = b.add(
                block_m_off, b.add(warp_m_off, b.const_i32(mi * atom.m))
            )
            n_tile_bases = [
                b.add(gu_n_off, b.add(warp_n_off, b.const_i32(ni * atom.n)))
                for ni in range(mfmas_n)
            ]
            g_dqs, u_dqs = _emit_fp8_gateup_fused_kloop(
                b,
                A=A,
                WGate=WGate,
                WUp=WUp,
                AScale=AScale,
                WGateScale=WGateScale,
                WUpScale=WUpScale,
                atom=atom,
                lane_decode=lane_decode,
                m_tile_base=m_tile_base,
                n_tile_bases=n_tile_bases,
                K=K,
                stride_a_scale=stride_a_scale,
                stride_gate_scale=stride_gate_scale,
                stride_up_scale=stride_up_scale,
                tag=f"{mi}",
                dtla=dtla_bundle,
            )
            gate_list.extend(g_dqs)
            up_list.extend(u_dqs)

        # ---- STAGE 1b Pass A (FUSED): SiLU(gate)*up -> f32 LDS + amax -----
        # hand-tuned asm G_dyn_quant granularity = per-token-block (sub_x=32), NOT
        # per-row: ONE dynamic scale per 128-inter-block, reduced over ALL
        # tile_m rows of the block. This row-uniform-within-block scheme is
        # MANDATORY because the down-GEMM fold applies a single per-lane scalar
        # (indexed by the A-input Hidden row ``m_in_atom``) to output slots that
        # span 4 DIFFERENT output rows; only a row-uniform block scale stays
        # correct under that fold (the same constraint as the activation scale,
        # BUILD_SPEC_FP8 OPEN RISK #1). The scale is broadcast to every row of
        # ``scale_view`` so any ``m_in_atom`` read returns the block scale.
        #
        # LEVER (fuse-quant): the per-block amax is reduced from registers as
        # part of Pass A instead of a separate full-block LDS re-read sweep.
        # Each lane returns the abs-max over the cells it owns; a 64-lane
        # butterfly max collapses that to the warp amax, and since an entire
        # warp's N-extent lives in ONE 128-inter block, the two warps that
        # share a block (2*blk, 2*blk+1) are combined below. This removes the
        # 4096-iter per-thread re-read scan and one whole barrier.
        amax_lane = _store_hidden_f32_pass(
            b,
            atom=atom,
            gate_list=gate_list,
            up_list=up_list,
            f32_view=f32_view,
            warp_m_off=warp_m_off,
            warp_n_off=warp_n_off,
            lane=lane,
            mfmas_m=mfmas_m,
            mfmas_n=mfmas_n,
            one_f32=one_f32,
            c_neg_log2e=c_neg_log2e,
            c_floor=c_floor,
        )
        # 64-lane butterfly max over the warp (xor 1,2,4,8,16,32).
        amax_warp = amax_lane
        for xm in (1, 2, 4, 8, 16, 32):
            amax_warp = b.fmax(amax_warp, b.warp_shuffle_xor(amax_warp, xm))
        # Lane 0 of each warp publishes its partial.
        with b.scf_if(b.cmp_eq(lane, c0)):
            b.smem_store_vN(WarpAmax_smem, [warp_id], amax_warp, 1)
        b.sync()

        # ---- STAGE 1b combine: per-block amax from the 2 warps' partials --
        # block ``blk`` is covered by warps {2*blk, 2*blk+1}; combine + scale +
        # broadcast to every row so any ``m_in_atom`` read returns the scale.
        sweep = b.scf_for_iter(
            tid, b.const_i32(n_blocks), c_threads, [], iv_name="cell"
        )
        with sweep as blk:
            w0 = b.mul(blk, b.const_i32(warps_per_block))
            amax = b.vec_extract(b.smem_load_vN(WarpAmax_smem, w0, dtype=F32, n=1), 0)
            for wo in range(1, warps_per_block):
                pw = b.vec_extract(
                    b.smem_load_vN(
                        WarpAmax_smem,
                        b.add(w0, b.const_i32(wo)),
                        dtype=F32,
                        n=1,
                    ),
                    0,
                )
                amax = b.fmax(amax, pw)
            scale = b.fmul(amax, b.rcp(c_fp8_max))  # amax / 448
            row_bc = b.scf_for_iter(c0, c_block_m, b.const_i32(1), [], iv_name="rb")
            with row_bc as rr:
                b.smem_store_vN(scale_view.base, [rr, blk], scale, 1)
                b.scf_yield()
            b.scf_yield()
        b.sync()

        # ---- STAGE 1b Pass C: quantize f32 Hidden -> fp8 LDS --------------
        # Flat thread sweep over all (tile_m * tile_n) cells; each thread reads
        # its f32 value + the block's scale, computes cvt_f32_to_fp8(h * inv),
        # writes fp8 into Hidden_smem at the SAME (row, col) (implicit reshape).
        total_q = tile_m * tile_n
        qsweep = b.scf_for_iter(
            tid, b.const_i32(total_q), c_threads, [], iv_name="qcell"
        )
        with qsweep as qcell:
            row = b.div(qcell, c_tile_n)
            col = b.mod(qcell, c_tile_n)
            blk = b.div(col, c_group_k)
            hv = f32_view_load(b, f32_view, row, col)
            sc = b.smem_load_vN(scale_view.base, row, blk, dtype=F32, n=1)
            sc = b.vec_extract(sc, 0)
            inv = b.rcp(sc)
            q = b.cvt_f32_to_fp8(b.fmul(hv, inv))
            b.smem_store_vN(fp8_view.base, [row, col], q, 1)
            b.scf_yield()
        b.sync()

        # ---- STAGE 2: down fp8 GEMM (LDS-A) -> dequant -> weighted atomic Y --
        # grid.x split the inter contraction; this TG owns the inter slice at
        # gu_n_off and produces a PARTIAL Y over the WHOLE H_out, tiled in
        # tile_n_down chunks. Per output tile, run the LDS-A down group GEMM
        # (contracting this TG's inter slice with the group-accumulator dequant
        # fold by hidden_dyn_scale * down_b_scale) then atomic-add the weighted,
        # token-validity-masked partial into Y.
        #
        # The Hidden A operand (fp8 in Hidden_smem) and its per-128-block dynamic
        # scales (HiddenScale_smem) are already resident from STAGE 1. The
        # contraction extent is the LOCAL inter slice tile_n_inter; A reads local
        # inter columns, W_down reads GLOBAL inter columns (full inter row stride
        # N) at this slice's base. inter_blk_base = gu_n_off // GROUP_K.
        inter_blk_base = b.div(gu_n_off, c_group_k)
        down_for = b.scf_for_iter(
            c0, H_out, b.const_i32(spec.tile_n_down), [], iv_name="ho"
        )
        with down_for as ho:
            down_warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m_down * atom.m))
            down_warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n_down * atom.n))
            down_list = []
            for mi in range(mfmas_m_down):
                for ni in range(mfmas_n_down):
                    # Down output column base (along H_out) for this warp-atom.
                    n_tile_base = b.add(
                        ho,
                        b.add(down_warp_n_off, b.const_i32(ni * atom.n)),
                    )
                    # A-read m-base for this atom: warp m-offset + mi*atom.m so
                    # atom mi reads its own Hidden LDS rows (the correctness fix).
                    m_row_base = b.add(down_warp_m_off, b.const_i32(mi * atom.m))
                    d_dq = _emit_fp8_down_group_gemm(
                        b,
                        a_view=fp8_view,
                        WDown=WDown,
                        WDownScale=WDownScale,
                        atom=atom,
                        lane_decode=lane_decode,
                        n_tile_base=n_tile_base,
                        scale_view=scale_view,
                        inter_slice=tile_n,
                        inter_full=N,
                        inter_blk_base=inter_blk_base,
                        stride_down_scale=stride_down_scale,
                        m_row_base=m_row_base,
                        tag=f"d{mi}_{ni}",
                    )
                    down_list.append(d_dq)
            # Barrier before the next H_out tile reuses Hidden_smem reads
            # (read-only here, but keep the scf.for body well-formed).
            _emit_down_atomic_reduce(
                b,
                atom=atom,
                down_list=down_list,
                warp_m_off=down_warp_m_off,
                warp_n_off=down_warp_n_off,
                lane=lane,
                mfmas_m=mfmas_m_down,
                mfmas_n=mfmas_n_down,
                block_m_off=block_m_off,
                ho_off=ho,
                H_out=H_out,
                SortedTokenIds=SortedTokenIds,
                SortedWeights=SortedWeights,
                Y=Y,
                tokens=tokens,
            )
            b.scf_yield()
        _ = (M, stride_a, slot_size)

    # Empty tail block (BlockExpertIds == -1) skips all work.
    with b.scf_if(b.cmp_ge(expert_idx, c0)):
        _emit_body()

    b.ret()
    return b.kernel
