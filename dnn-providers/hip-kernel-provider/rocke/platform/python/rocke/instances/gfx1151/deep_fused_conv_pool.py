# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1151 (RDNA3.5 / Strix Halo) deep-fused conv + maxpool, real quantization.

Wave32/WMMA sibling of ``instances/gfx950/deep_fused_conv_pool.py``. It computes
the encoder_0 block

    conv0 3x3 (int8) -> Quant(int32->int8) -> ReLU -> Quant(int8->int4)
    -> conv1 1x1 (int4) -> Quant(int32->int4) -> ReLU
    -> 2x2/s2 MaxPool -> Quant(int4->int4) -> packed-int4 output

in one kernel, with no conv0/conv1 intermediate written to HBM. Each CTA owns a
rectangular tile of final pooled outputs (backward-planned: pooled tile ->
conv1 patch -> conv0 region -> input halo), exactly like the gfx950 prototype.

Genuine low-bit storage (no fake-quant). The inputs/weights live in HBM as real
int8 / packed-int4 codes; every Quant node performs the real
``clamp(round(x * inv_scale), qmin, qmax)``. gfx1151 has **no int8/int4 matrix
cores** (the catalog exposes only ``wmma_f32_16x16x16_f16``), so the integer
operands are dequantized to fp16 and fed to fp16 WMMA with fp32 accumulation.
This is **bit-exact** to a native integer MMA for these ranges:

  * conv0 int8 x int8 over ``K_gemm = R*S*C = 72``: |sum| <= 72*127*127 ~ 1.16M
    < 2**24, so the fp32 accumulator is exact.
  * conv1 int4 x int4 over ``K0 = 32``: |sum| <= 32*8*8 = 2048, exact.

The int4 codes [-8, 7] and int8 codes [-127, 127] are exactly representable in
fp16, so only the *storage dtype* of on-chip intermediates is fp16; the numbers
are integer-exact. Same approach the shipped ``instances/common/matmul_nbits``
uses (dequant int4 -> fp16, then fp16 WMMA).

All per-tensor symmetric quant scales fold into four compile-time inverse
multipliers carried on the spec (``m0`` / ``m0b`` / ``m1`` / ``mf``); the conv
operands are therefore raw integer codes (dequant scale 1.0), and the requant
multipliers absorb ``act_scale * weight_scale / out_scale`` at each node.

Packed-int4 output: the maxpool stage gathers a full pooled pixel's 24 channels
from LDS into one thread, so it can assemble three i32 words (8 signed nibbles
each) with i32 bit-ops and store them as i32 -- no per-byte store or i8 constant
needed. The verify harness unpacks with the identical nibble layout.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ...core.ir import F16, I8, I16, I32, IRBuilder, PtrType, Value, VectorType
from ...helpers.geometry import WarpGrid
from ...helpers.schedule import DS_READ, MFMA, SchedulePolicy
from ...helpers.spec import kernel_name_join
from ...runtime.hip_module import Runtime
from ..common._matmul_nbits_common import (
    MatMulNBitsSpec,
    pack_i4_weights_for_matmul_nbits,
)
from ..common.conv_implicit_gemm import ConvProblem, _emit_frag_smem_load
from ..common.gemm_universal import TileSpec
from ..common.manifest_runner.utils import as_u8_buffer, nbytes, require_numpy
from ..gfx950.deep_fused_conv_pool import FusedConvPoolProblem

_WMMA = 16
_WAVE = 32
# Native integer WMMA atom for the conv0 iu8 path (Phase C). int8 A/B fragments
# are <4 x i32> (16 K-bytes packed 4-per-i32); accumulator/result <8 x i32>.
_OP_ID_IU8 = "wmma_i32_16x16x16_iu8"
_OP_ID_IU4 = "wmma_i32_16x16x16_iu4"
_K_PER_I32 = 4  # int8 K-values packed per i32 fragment slot
_I4_PER_I32 = 8  # int4 K-values packed per i32 fragment slot


@dataclass(frozen=True)
class Gfx1151DeepFusedConvPoolSpec:
    """One concrete gfx1151 genuine-int8/int4 deep-fusion configuration."""

    problem: FusedConvPoolProblem
    name: str = "rocke_gfx1151_deep_fused_conv_pool"
    tile_m: int = 256
    tile_n: int = 32
    pool_tile_h: int = 8
    pool_tile_w: int = 8
    warp_m: int = 4
    warp_n: int = 2
    # Optimization toggles (correctness-neutral; for in-process A/B benching).
    vectorize_conv0_a: bool = True
    vectorize_maxpool: bool = True
    early_w1: bool = True
    # Conv0 A operand mode. False -> im2col: materialize a row-major
    # [tile_m, kpad] LDS tile (each input pixel staged R*S times). True ->
    # direct conv: cache the raw input halo footprint once into a small
    # [foot_h*foot_w, C] LDS tile (no R*S redundancy) and gather each WMMA A
    # fragment from it with conv addressing. Drops ~17 KB LDS (a0 is the
    # occupancy blocker) and a full LDS round-trip. Correctness-neutral.
    # Measured +38% over im2col at the full target shape (7.54 vs 5.44 useful
    # TFLOP/s), so it is the default.
    direct_conv0: bool = True
    # Grid dispatch order. False -> grid (1, H_tiles, W_tiles): H is the
    # x-fastest axis. True -> (1, W_tiles, H_tiles): W (the NHWC-contiguous
    # spatial dim) becomes the fast axis, so adjacent workgroups walk
    # contiguous input memory. Correctness-neutral; perf-only.
    w_fast: bool = False
    # --- multi-lever latency-hiding campaign toggles (correctness-neutral) ---
    # L1: waves-per-EU occupancy hint. 0 -> unset (compiler default). >0 emits
    # the "amdgpu-waves-per-eu" launch bound: direct-conv0 freed enough LDS
    # (~26 KB/CTA) that two workgroups can co-reside per CU; raising the hint
    # forces the compiler to cap VGPRs so a 2nd resident WG fits, buying free
    # latency-hiding on this issue/latency-bound kernel. Swept empirically.
    waves_per_eu: int = 0
    # L2: instruction-schedule policy for the two WMMA loops. "mem" (default)
    # emits no hints; "compv3"/"compv4"/"intrawave" emit sched_group_barrier
    # DS_READ/MFMA/VMEM interleave hints to keep the matrix pipe fed across
    # operand-delivery latency. See helpers/schedule.py.
    sched_policy: str = "mem"
    # L3: maxpool tail control flow. False -> scf_if-guarded (only n_pix lanes
    # active). True -> predicated/masked: all lanes compute, the store address
    # is clamped in-range and trailing lanes write a harmless duplicate, cutting
    # branch/divergence overhead. Correctness-neutral.
    mask_maxpool: bool = False
    # Native integer cleanup levers.
    # Q1: vectorize fixed power-of-two RNE/clamp over the whole WMMA accumulator.
    # Measured non-default: cleaner ISA shape but slightly slower than scalar
    # per-slot RNE on gfx1151.
    specialized_rne: bool = False
    # Q2: skip halo predicates for CTAs whose direct-conv input footprint is
    # fully inside the image.
    interior_fastpath: bool = False
    # Q3: specialize direct-conv A-fragment K mapping for C=8/R=S=3 so
    # kg->(r,s,ci) is compile-time arithmetic, not magic-divide VALU.
    static_direct_kmap: bool = False
    # Q4: experimental packed C0 handoff. Even lanes pack their own int4 code
    # with the adjacent odd lane's code, then store one byte for two C0 columns.
    # This halves C0 LDS footprint/loads but adds a cross-lane permute.
    packed_c0_handoff: bool = False
    # Lever 2: lane-local packed C0 handoff. Keep the byte-per-code conv0 scatter,
    # then run a one-time LDS->LDS repack pass that reads two adjacent-K codes from
    # C0 LDS (a plain LDS read -- lane-agnostic, NO ds_bpermute) and writes one
    # packed byte. conv1 then loads A as a bitcast (like W1), deleting the
    # per-fragment-load nibble pack that dominates the conv1 A-path bitops. Unlike
    # packed_c0_handoff (which packs in the scatter and needs a cross-lane permute
    # because the WMMA C-frag scatters adjacent K across lanes), the repack reads
    # from LDS so no permute is needed. Mutually exclusive with packed_c0_handoff.
    #
    # MEASURED NON-LEVER (gfx1151 board, 8x8, full shape): 18.11 ms baseline ->
    # 20.00 ms with repack_c0 (~10% SLOWER), despite cutting ~12.5% of static VALU
    # (v_and_b32 84->4, no ds_bpermute introduced). Why it loses: the per-load
    # nibble pack it removes lives INSIDE the conv1 WMMA k-loop, where the
    # scheduler overlaps it with WMMA/LDS latency (nearly free); the repack
    # replaces that hidden work with a mandatory full-workgroup barrier + an extra
    # C0 LDS round-trip ON the critical path. Same thesis as butterfly: the
    # cross-WMMA handoff barrier is the expensive part, not the VALU. Kept behind
    # the flag as a documented negative result; do not enable.
    repack_c0: bool = False
    # L6: fused conv0->conv1 register handoff via permlanex16 (the gfx11 FMHA
    # C->A transpose). False (default) -> conv0 writes requantized i4 codes to
    # c0_smem, a full-WG barrier, then conv1 re-reads them (the LDS round-trip IS
    # the k0<->m transpose). True -> REORIENT conv0 (swap the b.mma operands so
    # W0 is WMMA-A and the footprint is WMMA-B): the conv0 acc then lands as
    # acc[k0_row, m_col] with lane->m, slot->k0. We requant each acc atom in
    # register, then transpose k0<->m WITHOUT LDS using one permlanex16 per word
    # (swap with the lane^16 partner) + v_perm_b32 byte-interleave + nibble-pack,
    # producing the conv1 A-fragment <2 x i32> directly in registers. This deletes
    # c0_smem, its scatter, and barrier 2 -- the cross-WMMA handoff the roofline
    # probe attributed ~part of the 90% conv1 wall to. The transpose is mandatory
    # (HW-fixed WMMA lane layouts) but CK's gfx11 pipelines pay it in 3 VALU/word
    # via the permute network, NOT an LDS round-trip -- falsifying the repack_c0 /
    # butterfly "ds_bpermute is the only vehicle" assumption. native_int + direct
    # conv0 only (needs the iu8 acc + iu4 conv1 register frags). Mutually
    # exclusive with packed_c0_handoff / repack_c0 (all three target the handoff).
    # Correctness-preserving (same accs, explicit lane transpose). Board A/B pending.
    fused_c0a1: bool = False
    # L4: butterfly register-fusion of conv0 -> conv1. ANALYZED NON-LEVER on
    # gfx1151 WMMA -- rejected by is_valid_spec (no codegen). The idea was to
    # transpose the conv0 WMMA C-fragment in-register straight into conv1's
    # A-fragment, deleting c0_smem + barrier 2. But chaining WMMA where the
    # producer's N becomes the consumer's K is a genuine cross-lane 16x16
    # transpose: the C-frag scatters N across lanes (col = lane%16) while the
    # A-frag needs that same N in the per-lane fragment slots (k = slot). The
    # only wave32 cross-lane vehicle, ds_bpermute, is ITSELF an LDS-unit
    # instruction (it "uses LDS as the shuffle vehicle") and broadcasts a single
    # register per source lane -- it cannot hand different slots to the different
    # destination lanes that read the same source, so a correct transpose needs
    # ~8 bpermutes per output slot (~64-128 LDS-unit ops/warp) to replace the
    # c0_smem path's ~4 ds_reads + one warp-uniform (one-WG) barrier. On this
    # LDS/latency-bound kernel that is a guaranteed large regression: here the
    # LDS round-trip is the CHEAP path. Same anti-staging thesis, opposite sign.
    # (rocprofv3 unavailable on Windows; this is the sanctioned instruction-shape
    # verdict, same as the w_fast / dispatch-order non-lever.)
    butterfly_conv01: bool = False
    # Native integer WMMA path for conv0. False (default) -> the fp16-emulation
    # path: int8 operands are sitofp->trunc to f16, run wmma_f32_16x16x16_f16,
    # accumulate in f32, rint-snap back to int. True -> conv0 uses the native
    # wmma_i32_16x16x16_iu8 atom: int8 operands are staged raw into i8 LDS,
    # loaded as <4 x i32> fragments, and accumulated exactly in i32 (no rint).
    # conv1 stays fp16 in this phase, so conv0's int32 output is requantized and
    # written as f16 codes into c0_smem (the fp16 conv1 consumer is unchanged).
    # Forces the im2col A path (direct/butterfly are not ported). The fp16 path
    # is left byte-identical so a flag A/B benchmark is clean. iu8-only (conv0).
    native_int: bool = False
    # Lever 1: register-level multi-buffered staging. False (default) ->
    # staging emits each global_load immediately followed by its dependent
    # ds_store, so the LLVM backend must insert s_waitcnt vmcnt(0) between every
    # load/store pair -> the four conv0-footprint global loads pay full memory
    # latency back-to-back with zero overlap (confirmed in ISA). True -> the
    # vectorized footprint staging issues ALL per-thread global loads into
    # distinct VGPRs first, then a single vmcnt(0) gates the whole batch, then
    # all ds_stores fire. This lets the four loads coalesce under one wait so
    # their latencies overlap. Correctness-neutral. gfx1151 has no direct
    # global->LDS DMA, so this VGPR-staged batching is the only prefetch vehicle.
    # MEASURED LEVER (gfx1151 board, pt2x64 native-int direct, full shape,
    # interleaved A/B): 14.74 ms -> 13.99 ms (~5.1% faster), bit-exact across 4
    # pairs. Default ON. Only the footprint loads coalesce so far (the W1 b128
    # load is still serialized in _stage_conv1_w1_packed -- a further lever).
    batch_loads: bool = True
    # Lever 3: packed-int16 maxpool reduction. False (default) -> the maxpool
    # 2x2 max runs per-channel in full-width i32 (v_cmp + v_cndmask, ~144 ops
    # for the 24-channel reduction; v_pk_* count = 0 in the ISA). True -> widen
    # the int4 codes to <N x i16> and reduce 4 corners with vector_smax so the
    # gfx11 backend selects v_pk_max_i16 (2 channels/op). The i8->i16 widen
    # replaces the i8->i32 widen the scalar path already pays (the 96 v_bfe_i32),
    # so the win is the reduction collapse; a small i16->i32 re-extract is added
    # at the nibble-pack boundary. Native-int finalpack path only; requires the
    # vectorized maxpool chunk conditions (tile_n % 8 == 0). Correctness-neutral
    # (signed max over identical values). STATIC-ISA / board A/B pending.
    pk_maxpool: bool = False
    # Lever 4: conv1 cross-k-step fragment prefetch. False (default) -> the conv1
    # iu4 GEMM hoists all A/B fragment LDS loads inside each k-step then issues
    # the MMAs, so the k=1 ds_read latency (lgkmcnt) is exposed at the top of the
    # k=1 iteration with nothing to overlap it (k_atoms=2, so this is the only
    # cross-step slack). True -> load ALL k-step A/B fragments up front (k=0 and
    # k=1 together) before any MMA, so the k=1 loads are in flight during the k=0
    # MMAs and their latency hides behind conv1 WMMA compute. Costs ~+18 VGPR
    # (the extra in-flight k=1 frags; ~110 -> ~128, within the wave32 256 budget).
    # Native-int byte-coded conv1 path only (the from_lds variant, not packed).
    # Correctness-neutral (pure load reordering, same MMAs/accs). Board A/B pending.
    conv1_prefetch_k: bool = False
    # Lever 5: conv1 fused k-step schedule group. False (default) -> the conv1
    # iu4 GEMM emits a per-k-step sched_group_barrier(DS_READ, n_ds) +
    # sched_group_barrier(MFMA, n_mma) pair, which pins each k-step's LDS reads
    # and MMAs into separate scheduling regions and BLOCKS the k=1 ds_read from
    # overlapping the k=0 MMAs (this is why the IR-level conv1_prefetch_k hoist
    # was a static no-op -- the per-step hint overrode the emission order). True
    # -> suppress the per-step hints and emit ONE combined group after the loop
    # (DS_READ over all k_atoms loads, then MFMA over all k_atoms MMAs) so the
    # post-RA scheduler may pull every conv1 LDS read ahead of every conv1 MMA,
    # hiding the k=1 LDS latency behind k=0 WMMA compute. Native-int byte-coded
    # conv1 path only (the from_lds variant). Correctness-neutral (scheduling
    # hint only; identical MMAs/accs). Pairs with conv1_prefetch_k. Board A/B pending.
    conv1_sched_fuse: bool = False
    # Lever 6: do conv1 contraction in int8 (iu8 atom) instead of int4 (iu4).
    # False (default) -> the fused_c0a1 handoff squeezes the 16 contiguous-k0
    # int8 byte codes into a packed <2 x i32> int4 A-fragment (8 codes/i32) via
    # two _pack_i4_codes_to_i32 calls, and conv1 uses the iu4 atom with a packed
    # int4 W1 B-fragment. True -> skip the nibble squeeze and hand the int8 byte
    # codes straight through as a <4 x i32> iu8 A-fragment (4 codes/i32), and
    # conv1 uses the iu8 atom with a byte-per-code W1. gfx11 iu8 and iu4 atoms are
    # BOTH 16x16x16, so this is matrix-count-neutral; the only difference is
    # fragment pack-math and W1/LDS footprint. The CODES stay int4-range
    # ([-8,7]), so iu8 and iu4 produce identical integer dot products ->
    # BIT-EXACT, no reference change. native_int + fused_c0a1 only. Board A/B
    # pending. The hypothesis (delete the per-handoff nibble squeeze) carries the
    # repack_c0 caveat: static VALU reduction may not yield wall-time gains if the
    # pack hides under WMMA/LDS latency -- the ISA gate decides before board.
    conv1_int8: bool = False
    # Persistent kernel. False (default) -> one output tile per CTA: the grid is
    # (1, H_tiles, W_tiles) = 16,200 tiny CTAs, and every CTA re-stages the
    # tile-invariant weights W0 (conv0 3x3) and W1 (conv1 1x1) global->LDS even
    # though they are identical for all spatial tiles. True -> launch only
    # `persistent_ctas` CTAs and grid-stride over the tile strip
    # (tile_idx = block_id_x; tile_idx < num_tiles; tile_idx += persistent_ctas),
    # staging W0/W1 into LDS ONCE before the loop (read from global once per CTA,
    # not once per tile). Only X (input footprint) and Y (output) stream per tile.
    # The redundant pre-conv1 barrier is dropped (W1 is staged+synced before the
    # loop and never rewritten; the fused_c0a1 handoff is register-only). An
    # inter-tile barrier at the loop head guards a0_smem/c1_smem reuse across tiles.
    #
    # RISK: the kernel is latency/overhead-bound on the ~8-CU iGPU slice and today
    # leans on the 16,200-deep CTA queue for free inter-CTA prologue/epilogue
    # overlap; collapsing to ~#CU*WG CTAs exposes per-tile footprint-load/store
    # latency unless occupancy (resident WG/CU) is preserved. native_int +
    # direct_conv0 + fused_c0a1 only (the live winning path; the only path with no
    # per-tile c0_smem and a register weight/handoff). Board A/B + ISA gate pending.
    persistent: bool = False
    # Persistent grid size (number of resident CTAs that grid-stride over tiles).
    # Load-bearing perf knob (see `persistent`): under-subscribing exposes conv1
    # barriers/bubbles; target ~ #CU(8) * steady-state-WG/CU. The grid-stride loop
    # covers ALL tiles for any value, so this is perf-only, swept on the board.
    # Default 16 = 8 CU * 2 WG/CU starting point; pair with a waves_per_eu sweep
    # (occupancy is what preserves steady-state latency hiding under persistence).
    persistent_ctas: int = 16
    # Per-node inverse requant multipliers (fold act/weight/out scales).
    m0: float = 0.0625  # conv0 int32 -> int8
    m0b: float = 0.5  # conv0 int8 -> int4
    m1: float = 0.25  # conv1 int32 -> int4
    mf: float = 1.0  # maxpool int4 -> int4

    @property
    def warp_tile_m(self) -> int:
        return _WMMA

    @property
    def warp_tile_n(self) -> int:
        return _WMMA

    @property
    def warp_tile_k(self) -> int:
        return _WMMA

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * _WAVE

    @property
    def kpad(self) -> int:
        """conv0 K_gemm rounded up to a whole number of 16-wide WMMA atoms."""
        kg = self.problem.conv.K_gemm
        return ((kg + _WMMA - 1) // _WMMA) * _WMMA

    @property
    def conv_tile_h(self) -> int:
        return self.pool_tile_h * self.problem.pool_stride_h

    @property
    def conv_tile_w(self) -> int:
        return self.pool_tile_w * self.problem.pool_stride_w

    @property
    def foot_h(self) -> int:
        """Input-halo footprint height for one conv0 output tile (direct mode)."""
        c = self.problem.conv
        return (self.conv_tile_h - 1) * c.sH + (c.Y - 1) * c.dH + 1

    @property
    def foot_w(self) -> int:
        c = self.problem.conv
        return (self.conv_tile_w - 1) * c.sW + (c.X - 1) * c.dW + 1

    def kernel_name(self) -> str:
        parts = [
            self.name,
            self.problem.short(),
            f"t{self.tile_m}x{self.tile_n}",
            f"pt{self.pool_tile_h}x{self.pool_tile_w}",
            f"w{self.warp_m}x{self.warp_n}",
            "wmma16x16x16",
            "directa" if self.direct_conv0 else "im2col",
        ]
        # Lever tags (only when non-default) so A/B configs get distinct kernel
        # names and don't collide in the multi-config compile harness.
        if self.waves_per_eu:
            parts.append(f"wpe{self.waves_per_eu}")
        if self.sched_policy != "mem":
            parts.append(f"sch{self.sched_policy}")
        if self.mask_maxpool:
            parts.append("maskpool")
        if self.specialized_rne:
            parts.append("vecrne")
        if self.interior_fastpath:
            parts.append("interior")
        if self.static_direct_kmap:
            parts.append("statick")
        if self.packed_c0_handoff:
            parts.append("packedc0")
        if self.repack_c0:
            parts.append("repackc0")
        if self.butterfly_conv01:
            parts.append("butterfly")
        if self.native_int:
            parts.append("nativeiu8")
        if self.pk_maxpool:
            parts.append("pkpool")
        if self.conv1_prefetch_k:
            parts.append("pfk")
        if self.conv1_sched_fuse:
            parts.append("schfuse")
        if self.conv1_int8:
            parts.append("conv1iu8")
        if self.fused_c0a1:
            parts.append("fusedc0a1")
        if self.persistent:
            parts.append(f"persist{self.persistent_ctas}")
        parts.append("i8i4_realquant")
        return kernel_name_join(*parts)


def make_deep_fused_conv_pool_spec(
    *,
    n: int = 1,
    h: int,
    w: int,
    c: int,
    k0: int,
    k1: int,
    r: int = 3,
    s: int = 3,
    pool_tile_h: int = 8,
    pool_tile_w: int = 8,
    tile_n: int = 32,
    warp_m: int = 4,
    warp_n: int = 2,
    vectorize_conv0_a: bool = True,
    vectorize_maxpool: bool = True,
    early_w1: bool = True,
    direct_conv0: bool = True,
    w_fast: bool = False,
    waves_per_eu: int = 0,
    sched_policy: str = "mem",
    mask_maxpool: bool = False,
    specialized_rne: bool = False,
    interior_fastpath: bool = False,
    static_direct_kmap: bool = False,
    packed_c0_handoff: bool = False,
    repack_c0: bool = False,
    fused_c0a1: bool = False,
    butterfly_conv01: bool = False,
    native_int: bool = False,
    batch_loads: bool = True,
    pk_maxpool: bool = False,
    conv1_prefetch_k: bool = False,
    conv1_sched_fuse: bool = False,
    conv1_int8: bool = False,
    persistent: bool = False,
    persistent_ctas: int = 16,
    m0: float = 0.0625,
    m0b: float = 0.5,
    m1: float = 0.25,
    mf: float = 1.0,
) -> Gfx1151DeepFusedConvPoolSpec:
    """Build a spec, auto-deriving ``tile_m`` from the pool tile geometry."""
    conv = ConvProblem(
        N=n,
        Hi=h,
        Wi=w,
        C=c,
        K=k0,
        Y=r,
        X=s,
        sH=1,
        sW=1,
        pH=1,
        pW=1,
        dH=1,
        dW=1,
    )
    problem = FusedConvPoolProblem(conv=conv, conv1_k=k1)
    conv_tile_h = pool_tile_h * problem.pool_stride_h
    conv_tile_w = pool_tile_w * problem.pool_stride_w
    tile_m = conv_tile_h * conv_tile_w
    return Gfx1151DeepFusedConvPoolSpec(
        problem=problem,
        tile_m=tile_m,
        tile_n=tile_n,
        pool_tile_h=pool_tile_h,
        pool_tile_w=pool_tile_w,
        warp_m=warp_m,
        warp_n=warp_n,
        vectorize_conv0_a=vectorize_conv0_a,
        vectorize_maxpool=vectorize_maxpool,
        early_w1=early_w1,
        direct_conv0=direct_conv0,
        w_fast=w_fast,
        waves_per_eu=waves_per_eu,
        sched_policy=sched_policy,
        mask_maxpool=mask_maxpool,
        specialized_rne=specialized_rne,
        interior_fastpath=interior_fastpath,
        static_direct_kmap=static_direct_kmap,
        packed_c0_handoff=packed_c0_handoff,
        repack_c0=repack_c0,
        fused_c0a1=fused_c0a1,
        butterfly_conv01=butterfly_conv01,
        native_int=native_int,
        batch_loads=batch_loads,
        pk_maxpool=pk_maxpool,
        conv1_prefetch_k=conv1_prefetch_k,
        conv1_sched_fuse=conv1_sched_fuse,
        conv1_int8=conv1_int8,
        persistent=persistent,
        persistent_ctas=persistent_ctas,
        m0=m0,
        m0b=m0b,
        m1=m1,
        mf=mf,
    )


def is_valid_spec(
    spec: Gfx1151DeepFusedConvPoolSpec, arch: str = "gfx1151"
) -> Tuple[bool, str]:
    if arch not in ("gfx1151", "gfx11-generic"):
        return False, (
            "gfx1151 deep fused conv/pool needs the gfx1151 wave32/WMMA ABI "
            f"(gfx1151 or gfx11-generic); got {arch!r}"
        )
    p = spec.problem
    c = p.conv
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)
    if not target.mma.has_shape(
        family="wmma",
        a_dtype="f16",
        b_dtype="f16",
        c_dtype="fp32",
        m=_WMMA,
        n=_WMMA,
        k=_WMMA,
    ):
        return False, f"WMMA 16x16x16 f16 atom absent on {arch}"
    if target.wave_size != _WAVE:
        return False, f"this kernel is wave32; {arch} is wave{target.wave_size}"
    if (p.pool_y, p.pool_x, p.pool_stride_h, p.pool_stride_w) != (2, 2, 2, 2):
        return False, "only 2x2 stride-2 maxpool is supported"
    if c.N != 1:
        return False, f"tiled schedule supports only N=1 (got N={c.N})"
    if spec.pool_tile_h <= 0 or spec.pool_tile_w <= 0:
        return False, "pool_tile_h and pool_tile_w must be positive"
    conv_tile_h = spec.pool_tile_h * p.pool_stride_h
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    if spec.tile_m != conv_tile_h * conv_tile_w:
        return False, (
            f"tile_m={spec.tile_m} must equal conv tile "
            f"{conv_tile_h}x{conv_tile_w}={conv_tile_h * conv_tile_w}"
        )
    if p.pool_ho % spec.pool_tile_h or p.pool_wo % spec.pool_tile_w:
        return False, (
            f"pool dims ({p.pool_ho},{p.pool_wo}) must be divisible by pool tile "
            f"({spec.pool_tile_h},{spec.pool_tile_w})"
        )
    if c.K > spec.tile_n:
        return (
            False,
            f"one CTA owns all conv0 channels: K0={c.K} > tile_n={spec.tile_n}",
        )
    if p.conv1_channels > spec.tile_n:
        return False, (
            f"one CTA owns all conv1 channels: K1={p.conv1_channels} > tile_n={spec.tile_n}"
        )
    if spec.tile_m % (spec.warp_m * _WMMA):
        return False, "tile_m must divide warp_m * 16"
    if spec.tile_n % (spec.warp_n * _WMMA):
        return False, "tile_n must divide warp_n * 16"
    if c.K % _WMMA:
        return False, f"conv0 channels K0={c.K} must be a multiple of 16 (conv1 K)"
    if not spec.direct_conv0 and (spec.tile_m * spec.kpad) % spec.block_size:
        return False, "tile_m*kpad must divide block_size (A0 staging is untailed)"
    if spec.sched_policy not in ("mem", "compv3", "compv4", "intrawave"):
        return False, f"unknown sched_policy {spec.sched_policy!r}"
    if spec.butterfly_conv01:
        # Analyzed non-lever: chaining WMMA (conv0 N -> conv1 K) is a genuine
        # cross-lane 16x16 transpose, and the only wave32 vehicle (ds_bpermute)
        # is itself an LDS-unit op needing ~64-128 ds-ops/warp to replace the
        # cheaper c0_smem round-trip. Rejected, not implemented. See the
        # butterfly_conv01 field comment for the full instruction-shape verdict.
        return False, (
            "butterfly_conv01 is an analyzed non-lever on gfx1151 WMMA "
            "(cross-lane C->A transpose costs more LDS-unit ops than the "
            "c0_smem round-trip it would replace); not implemented"
        )
    if spec.native_int:
        if target.mma.by_op_id(_OP_ID_IU8) is None:
            return False, f"{_OP_ID_IU8} atom absent on {arch}"
        if target.mma.by_op_id(_OP_ID_IU4) is None:
            return False, f"{_OP_ID_IU4} atom absent on {arch}"
        if spec.direct_conv0 and spec.problem.conv.C % _K_PER_I32:
            return False, (
                "native_int direct conv0 requires C to be divisible by 4 so each "
                "iu8 packed fragment slot stays inside one footprint pixel"
            )
        if spec.butterfly_conv01:
            return False, "native_int is incompatible with butterfly_conv01"
    if spec.interior_fastpath and not (spec.native_int and spec.direct_conv0):
        return False, "interior_fastpath is only implemented for native direct conv0"
    if spec.static_direct_kmap:
        if not (spec.native_int and spec.direct_conv0):
            return (
                False,
                "static_direct_kmap is only implemented for native direct conv0",
            )
        if (c.C, c.Y, c.X) != (8, 3, 3):
            return False, "static_direct_kmap assumes C=8 and Y=X=3"
    if spec.packed_c0_handoff:
        if not spec.native_int:
            return False, "packed_c0_handoff is only implemented for native_int"
        if c.K % 2:
            return False, "packed_c0_handoff requires even conv0 channels"
    if spec.repack_c0:
        if not spec.native_int:
            return False, "repack_c0 is only implemented for native_int"
        if spec.packed_c0_handoff:
            return False, "repack_c0 and packed_c0_handoff are mutually exclusive"
        if c.K % 2:
            return False, "repack_c0 requires even conv0 channels"
    if spec.fused_c0a1:
        if not (spec.native_int and spec.direct_conv0):
            return (
                False,
                "fused_c0a1 is only implemented for native direct conv0",
            )
        if spec.packed_c0_handoff or spec.repack_c0:
            return (
                False,
                "fused_c0a1 is mutually exclusive with packed_c0_handoff/repack_c0",
            )
        # The handoff produces one conv1 A-frag per (m-atom, k0-atom). conv0's N
        # atoms over k0 (mfmas_n) must equal conv1's K atoms (c.K//16), which
        # holds iff K0 is a whole number of 16-wide WMMA atoms.
        if spec.problem.conv.K % _WMMA:
            return False, "fused_c0a1 requires conv0 K0 divisible by 16"
    if spec.conv1_int8:
        if not (spec.native_int and spec.fused_c0a1):
            return False, (
                "conv1_int8 only implemented for the native_int fused_c0a1 "
                "register handoff path"
            )
    if spec.persistent:
        if not (spec.native_int and spec.direct_conv0 and spec.fused_c0a1):
            return False, (
                "persistent requires native_int + direct_conv0 + fused_c0a1 "
                "(the only path with register weight/handoff and no per-tile "
                "c0_smem, so W0/W1 can be hoisted above the grid-stride loop)"
            )
        if spec.persistent_ctas <= 0:
            return False, "persistent_ctas must be positive"
    return True, "ok"


def deep_fused_conv_pool_grid(
    spec: Gfx1151DeepFusedConvPoolSpec,
) -> Tuple[int, int, int]:
    p = spec.problem
    h_tiles = p.pool_ho // spec.pool_tile_h
    w_tiles = p.pool_wo // spec.pool_tile_w
    if spec.persistent:
        # Launch only persistent_ctas resident CTAs; each grid-strides over the
        # flattened tile strip (tile_idx = block_id_x, += persistent_ctas). The
        # loop covers all h_tiles*w_tiles tiles regardless of the CTA count.
        return (spec.persistent_ctas, 1, 1)
    if spec.w_fast:
        return (1, w_tiles, h_tiles)
    return (1, h_tiles, w_tiles)


def _quant_i8(b: IRBuilder, vf32: Value, inv_scale: Value) -> Value:
    """clamp(round(v*inv_scale), -127, 127) -> i8 (round-to-nearest-even)."""
    scaled = b.fmul(vf32, inv_scale)
    clamped = b.clamp_f32(scaled, b.const_f32(-127.0), b.const_f32(127.0))
    return b.cvt_f32_to_i8_sat(clamped)


def _quant_i4(b: IRBuilder, vf32: Value, inv_scale: Value) -> Value:
    """clamp(round(v*inv_scale), -8, 7) -> i8 holding an int4 code."""
    scaled = b.fmul(vf32, inv_scale)
    clamped = b.clamp_f32(scaled, b.const_f32(-8.0), b.const_f32(7.0))
    return b.cvt_f32_to_i8_sat(clamped)


def _i8_to_f32(b: IRBuilder, qi8: Value) -> Value:
    return b.sitofp_f32(b.sext(qi8, I32))


def _neg_i32(b: IRBuilder, x: Value) -> Value:
    return b.sub(b.const_i32(0), x)


def _clamp_i32(b: IRBuilder, x: Value, lo: int, hi: int) -> Value:
    return b.smin(b.smax(x, b.const_i32(lo)), b.const_i32(hi))


def _relu_i32(b: IRBuilder, x: Value) -> Value:
    return b.smax(x, b.const_i32(0))


def _round_shift_rne_i32(b: IRBuilder, x: Value, shift: int) -> Value:
    """Round ``x / 2**shift`` to nearest-even using integer ops only."""
    if shift == 0:
        return x
    sign = b.cmp_lt(x, b.const_i32(0))
    ax = b.select(sign, _neg_i32(b, x), x)
    floor_q = b.lshr(ax, b.const_i32(shift))
    # RNE for positive integers:
    #   (x + half - 1 + (floor(x/d) & 1)) >> shift
    # At exact half, this increments only when the retained quotient is odd.
    bias = b.add(b.const_i32((1 << (shift - 1)) - 1), b.land(floor_q, b.const_i32(1)))
    q = b.lshr(b.add(ax, bias), b.const_i32(shift))
    return b.select(sign, _neg_i32(b, q), q)


def _quant_i8_shift(b: IRBuilder, x: Value, shift: int) -> Value:
    q = _clamp_i32(b, _round_shift_rne_i32(b, x, shift), -127, 127)
    return b.trunc(q, I8)


def _quant_i4_shift(b: IRBuilder, x: Value, shift: int) -> Value:
    q = _clamp_i32(b, _round_shift_rne_i32(b, x, shift), -8, 7)
    return b.trunc(q, I8)


def _splat_i32(b: IRBuilder, value: int, n: int) -> Value:
    return b.vector_splat(b.const_i32(value), n)


def _neg_i32_vec(b: IRBuilder, x: Value) -> Value:
    return b.vector_sub(b.zero_vec(I32, x.type.count), x)


def _clamp_i32_vec(b: IRBuilder, x: Value, lo: int, hi: int) -> Value:
    n = x.type.count
    return b.vector_smin(b.vector_smax(x, _splat_i32(b, lo, n)), _splat_i32(b, hi, n))


def _relu_i32_vec(b: IRBuilder, x: Value) -> Value:
    return b.vector_smax(x, b.zero_vec(I32, x.type.count))


def _round_shift_rne_i32_vec(b: IRBuilder, x: Value, shift: int) -> Value:
    """Vector RNE ``x / 2**shift`` for an ``<N x i32>`` accumulator."""
    if shift == 0:
        return x
    n = x.type.count
    sign = b.vector_cmp("lt", x, b.zero_vec(I32, n))
    ax = b.vector_select(sign, _neg_i32_vec(b, x), x)
    floor_q = b.vector_lshr(ax, _splat_i32(b, shift, n))
    bias = b.vector_add(
        _splat_i32(b, (1 << (shift - 1)) - 1, n),
        b.vector_and(floor_q, _splat_i32(b, 1, n)),
    )
    q = b.vector_lshr(b.vector_add(ax, bias), _splat_i32(b, shift, n))
    return b.vector_select(sign, _neg_i32_vec(b, q), q)


def _quant_i8_shift_vec_i32(b: IRBuilder, x: Value, shift: int) -> Value:
    return _clamp_i32_vec(b, _round_shift_rne_i32_vec(b, x, shift), -127, 127)


def _quant_i4_shift_vec_i32(b: IRBuilder, x: Value, shift: int) -> Value:
    return _clamp_i32_vec(b, _round_shift_rne_i32_vec(b, x, shift), -8, 7)


def _stage_conv0_a(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    x_ptr: Value,
    a_smem: Value,
    grid: WarpGrid,
) -> None:
    """im2col the int8 conv0 activations for this tile into ``a_smem`` as fp16
    integer codes (dequant scale 1.0). Out-of-image halo and K padding -> 0."""
    p = spec.problem
    c = p.conv
    kpad = spec.kpad
    conv_tile_h = spec.pool_tile_h * p.pool_stride_h
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    bs = spec.block_size

    c_ctw = b.const_i32(conv_tile_w)
    c_Wi = b.const_i32(c.Wi)
    c_Hi = b.const_i32(c.Hi)
    c0 = b.const_i32(0)
    h_blk = b.block_id_z() if spec.w_fast else b.block_id_y()
    w_blk = b.block_id_y() if spec.w_fast else b.block_id_z()
    h_base = b.mul(h_blk, b.const_i32(conv_tile_h))
    w_base = b.mul(w_blk, b.const_i32(conv_tile_w))

    # Fast path: all C channels of one (row, r, s) im2col entry are contiguous in
    # both global memory (NHWC) and the LDS column (kg = r*S*C + s*C + ci), and
    # share one validity check (padding depends only on r/s, not the channel).
    # So load C int8 in one VMEM transaction and ds_write_b128 the C f16 codes.
    vec_c = (
        spec.vectorize_conv0_a
        and c.C in (2, 4, 8, 16)
        and kpad % c.C == 0
        and c.K_gemm % c.C == 0
    )
    if vec_c:
        cc = c.C
        groups = kpad // cc  # incl. K-pad groups (zeroed)
        real_groups = c.K_gemm // cc
        c_g = b.const_i32(groups)
        c_total = b.const_i32(spec.tile_m * groups)
        zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))
        for e in range((spec.tile_m * groups + bs - 1) // bs):
            idx = b.add(b.const_i32(e * bs), grid.tid)
            in_range = b.cmp_lt(idx, c_total)
            sidx = b.select(in_range, idx, c0)
            row = b.div(sidx, c_g)
            g = b.mod(sidx, c_g)
            r = b.div(g, b.const_i32(c.X))
            s = b.mod(g, b.const_i32(c.X))
            local_oh = b.div(row, c_ctw)
            local_ow = b.mod(row, c_ctw)
            oh = b.add(h_base, local_oh)
            ow = b.add(w_base, local_ow)
            ih = b.sub(
                b.add(b.mul(oh, b.const_i32(c.sH)), b.mul(r, b.const_i32(c.dH))),
                b.const_i32(c.pH),
            )
            iw = b.sub(
                b.add(b.mul(ow, b.const_i32(c.sW)), b.mul(s, b.const_i32(c.dW))),
                b.const_i32(c.pW),
            )
            h_ok = b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi))
            w_ok = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi))
            g_real = b.cmp_lt(g, b.const_i32(real_groups))
            valid = b.land(g_real, b.land(h_ok, w_ok))
            base_off = b.mul(
                b.add(b.mul(ih, c_Wi), iw), b.const_i32(cc)
            )  # ci=0; C contiguous
            safe_off = b.select(valid, base_off, c0)
            raw = b.global_load_vN(x_ptr, safe_off, I8, cc)
            comps = [
                b.select(
                    valid,
                    b.trunc_f32_to_f16(_i8_to_f32(b, b.vec_extract(raw, i))),
                    zero_h,
                )
                for i in range(cc)
            ]
            vec = b.vec_pack(comps, F16)
            kg = b.mul(g, b.const_i32(cc))
            with b.scf_if(in_range):
                b.smem_store_vN_f16(a_smem, [row, kg], vec, n=cc)
        return

    total = spec.tile_m * kpad
    ept = (total + bs - 1) // bs
    c_kpad = b.const_i32(kpad)
    c_kg = b.const_i32(c.K_gemm)
    c_sc = b.const_i32(c.X * c.C)
    c_cc = b.const_i32(c.C)
    zero_f = b.const_f32(0.0)

    for e in range(ept):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        row = b.div(idx, c_kpad)
        kg = b.mod(idx, c_kpad)
        kg_in = b.cmp_lt(kg, c_kg)

        local_oh = b.div(row, c_ctw)
        local_ow = b.mod(row, c_ctw)
        r = b.div(kg, c_sc)
        rem = b.mod(kg, c_sc)
        s = b.div(rem, c_cc)
        ci = b.mod(rem, c_cc)

        oh = b.add(h_base, local_oh)
        ow = b.add(w_base, local_ow)
        ih = b.sub(
            b.add(b.mul(oh, b.const_i32(c.sH)), b.mul(r, b.const_i32(c.dH))),
            b.const_i32(c.pH),
        )
        iw = b.sub(
            b.add(b.mul(ow, b.const_i32(c.sW)), b.mul(s, b.const_i32(c.dW))),
            b.const_i32(c.pW),
        )
        h_ok = b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi))
        w_ok = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi))
        valid = b.land(kg_in, b.land(h_ok, w_ok))

        in_off = b.add(b.mul(b.add(b.mul(ih, c_Wi), iw), c_cc), ci)
        safe_off = b.select(valid, in_off, c0)
        raw_i8 = b.global_load(x_ptr, safe_off, I8)
        v = b.select(valid, _i8_to_f32(b, raw_i8), zero_f)
        b.smem_store_f16(a_smem, [row, kg], b.trunc_f32_to_f16(v))


def _stage_input_footprint(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    x_ptr: Value,
    inp_smem: Value,
    grid: WarpGrid,
) -> None:
    """Direct-conv mode: cache this CTA's raw int8 input halo footprint into
    ``inp_smem[foot_h*foot_w, C]`` as fp16 codes (each input pixel staged once,
    no R*S im2col redundancy). Out-of-image halo -> 0. The conv im2col expansion
    is then applied implicitly at WMMA A-fragment load time."""
    p = spec.problem
    c = p.conv
    bs = spec.block_size
    foot_w = spec.foot_w
    npix = spec.foot_h * foot_w
    cc = c.C
    c0 = b.const_i32(0)
    c_Wi = b.const_i32(c.Wi)
    c_Hi = b.const_i32(c.Hi)
    c_fw = b.const_i32(foot_w)
    c_npix = b.const_i32(npix)
    zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))

    h_blk = b.block_id_z() if spec.w_fast else b.block_id_y()
    w_blk = b.block_id_y() if spec.w_fast else b.block_id_z()
    # Footprint origin in image coords: top-left input pixel of the halo.
    ih0 = b.sub(
        b.mul(b.mul(h_blk, b.const_i32(spec.conv_tile_h)), b.const_i32(c.sH)),
        b.const_i32(c.pH),
    )
    iw0 = b.sub(
        b.mul(b.mul(w_blk, b.const_i32(spec.conv_tile_w)), b.const_i32(c.sW)),
        b.const_i32(c.pW),
    )

    # Fast path: C channels of one footprint pixel are contiguous in NHWC global
    # and in the LDS row, sharing one validity check. One i8 vector load + b128.
    if spec.vectorize_conv0_a and cc in (2, 4, 8, 16):
        for e in range((npix + bs - 1) // bs):
            idx = b.add(b.const_i32(e * bs), grid.tid)
            in_range = b.cmp_lt(idx, c_npix)
            sidx = b.select(in_range, idx, c0)
            fr = b.div(sidx, c_fw)
            fw = b.mod(sidx, c_fw)
            ih = b.add(ih0, fr)
            iw = b.add(iw0, fw)
            valid = b.land(
                b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi)),
                b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi)),
            )
            off = b.mul(b.add(b.mul(ih, c_Wi), iw), b.const_i32(cc))
            safe_off = b.select(valid, off, c0)
            raw = b.global_load_vN(x_ptr, safe_off, I8, cc)
            comps = [
                b.select(
                    valid,
                    b.trunc_f32_to_f16(_i8_to_f32(b, b.vec_extract(raw, i))),
                    zero_h,
                )
                for i in range(cc)
            ]
            vec = b.vec_pack(comps, F16)
            with b.scf_if(in_range):
                b.smem_store_vN_f16(inp_smem, [idx, c0], vec, n=cc)
        return

    total = npix * cc
    c_total = b.const_i32(total)
    c_cc = b.const_i32(cc)
    zero_f = b.const_f32(0.0)
    for e in range((total + bs - 1) // bs):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        sidx = b.select(in_range, idx, c0)
        pix = b.div(sidx, c_cc)
        ci = b.mod(sidx, c_cc)
        fr = b.div(pix, c_fw)
        fw = b.mod(pix, c_fw)
        ih = b.add(ih0, fr)
        iw = b.add(iw0, fw)
        valid = b.land(
            b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi)),
            b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi)),
        )
        off = b.add(b.mul(b.add(b.mul(ih, c_Wi), iw), c_cc), ci)
        safe_off = b.select(valid, off, c0)
        raw_i8 = b.global_load(x_ptr, safe_off, I8)
        v = b.select(valid, _i8_to_f32(b, raw_i8), zero_f)
        with b.scf_if(in_range):
            b.smem_store_f16(inp_smem, [pix, ci], b.trunc_f32_to_f16(v))


def _stage_input_footprint_int(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    x_ptr: Value,
    inp_smem: Value,
    grid: WarpGrid,
    h_blk: Value = None,
    w_blk: Value = None,
) -> None:
    """Direct native-int conv0: cache the raw int8 input halo footprint.

    Persistent mode threads the loop-carried tile coords in via ``h_blk``/
    ``w_blk``; otherwise they fall back to the per-CTA block ids (byte-identical).
    """
    p = spec.problem
    c = p.conv
    bs = spec.block_size
    foot_w = spec.foot_w
    npix = spec.foot_h * foot_w
    cc = c.C
    c0 = b.const_i32(0)
    c_Wi = b.const_i32(c.Wi)
    c_Hi = b.const_i32(c.Hi)
    c_fw = b.const_i32(foot_w)
    c_npix = b.const_i32(npix)

    if h_blk is None:
        h_blk = b.block_id_z() if spec.w_fast else b.block_id_y()
        w_blk = b.block_id_y() if spec.w_fast else b.block_id_z()
    ih0 = b.sub(
        b.mul(b.mul(h_blk, b.const_i32(spec.conv_tile_h)), b.const_i32(c.sH)),
        b.const_i32(c.pH),
    )
    iw0 = b.sub(
        b.mul(b.mul(w_blk, b.const_i32(spec.conv_tile_w)), b.const_i32(c.sW)),
        b.const_i32(c.pW),
    )

    if spec.vectorize_conv0_a and cc in (2, 4, 8, 16):
        zero_vec = b.zero_vec(I8, cc)
        ept = (npix + bs - 1) // bs

        def emit_vec_body(skip_halo_predicates: bool) -> None:
            if spec.batch_loads:
                # Lever 1: decouple loads from stores. Issue every per-thread
                # global load into its own VGPR first (no intervening store ->
                # the backend coalesces them under a single vmcnt(0)), then run
                # the validity selects, then fire all ds_stores.
                loads = []
                for e in range(ept):
                    idx = b.add(b.const_i32(e * bs), grid.tid)
                    in_range = b.cmp_lt(idx, c_npix)
                    sidx = b.select(in_range, idx, c0)
                    fr = b.div(sidx, c_fw)
                    fw = b.mod(sidx, c_fw)
                    ih = b.add(ih0, fr)
                    iw = b.add(iw0, fw)
                    off = b.mul(b.add(b.mul(ih, c_Wi), iw), b.const_i32(cc))
                    if skip_halo_predicates:
                        raw = b.global_load_vN(x_ptr, off, I8, cc)
                        loads.append((idx, in_range, raw, None))
                    else:
                        valid = b.land(
                            in_range,
                            b.land(
                                b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi)),
                                b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi)),
                            ),
                        )
                        safe_off = b.select(valid, off, c0)
                        raw = b.global_load_vN(x_ptr, safe_off, I8, cc)
                        loads.append((idx, in_range, raw, valid))
                for idx, in_range, raw, valid in loads:
                    code = (
                        raw if valid is None else b.vector_select(valid, raw, zero_vec)
                    )
                    with b.scf_if(in_range):
                        b.smem_store_vN(inp_smem, [idx, c0], code, n=cc)
                return
            for e in range(ept):
                idx = b.add(b.const_i32(e * bs), grid.tid)
                in_range = b.cmp_lt(idx, c_npix)
                sidx = b.select(in_range, idx, c0)
                fr = b.div(sidx, c_fw)
                fw = b.mod(sidx, c_fw)
                ih = b.add(ih0, fr)
                iw = b.add(iw0, fw)
                off = b.mul(b.add(b.mul(ih, c_Wi), iw), b.const_i32(cc))
                if skip_halo_predicates:
                    raw = b.global_load_vN(x_ptr, off, I8, cc)
                    code = raw
                else:
                    valid = b.land(
                        in_range,
                        b.land(
                            b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi)),
                            b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi)),
                        ),
                    )
                    safe_off = b.select(valid, off, c0)
                    raw = b.global_load_vN(x_ptr, safe_off, I8, cc)
                    code = b.vector_select(valid, raw, zero_vec)
                with b.scf_if(in_range):
                    b.smem_store_vN(inp_smem, [idx, c0], code, n=cc)

        if spec.interior_fastpath:
            h_inner = b.land(
                b.cmp_gt(h_blk, c0),
                b.cmp_lt(h_blk, b.const_i32(p.pool_ho // spec.pool_tile_h - 1)),
            )
            w_inner = b.land(
                b.cmp_gt(w_blk, c0),
                b.cmp_lt(w_blk, b.const_i32(p.pool_wo // spec.pool_tile_w - 1)),
            )
            interior = b.land(h_inner, w_inner)
            with b.scf_if(interior):
                emit_vec_body(skip_halo_predicates=True)
            with b.scf_if(b.lnot(interior)):
                emit_vec_body(skip_halo_predicates=False)
        else:
            emit_vec_body(skip_halo_predicates=False)
        return

    total = npix * cc
    c_total = b.const_i32(total)
    c_cc = b.const_i32(cc)
    zero_i8 = b.cvt_f32_to_i8_sat(b.const_f32(0.0))
    for e in range((total + bs - 1) // bs):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        sidx = b.select(in_range, idx, c0)
        pix = b.div(sidx, c_cc)
        ci = b.mod(sidx, c_cc)
        fr = b.div(pix, c_fw)
        fw = b.mod(pix, c_fw)
        ih = b.add(ih0, fr)
        iw = b.add(iw0, fw)
        valid = b.land(
            in_range,
            b.land(
                b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi)),
                b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi)),
            ),
        )
        off = b.add(b.mul(b.add(b.mul(ih, c_Wi), iw), c_cc), ci)
        safe_off = b.select(valid, off, c0)
        raw_i8 = b.global_load(x_ptr, safe_off, I8)
        code = b.select(valid, raw_i8, zero_i8)
        with b.scf_if(in_range):
            b.smem_store_vN(inp_smem, [pix, ci], code, n=1)


def _stage_conv0_w0(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    w0_ptr: Value,
    w0_smem: Value,
    grid: WarpGrid,
) -> None:
    """Load int8 conv0 weights ``W0[K0, K_gemm]`` (KYXC contiguous) into
    ``w0_smem[tile_n, kpad]`` as fp16 codes; padding rows/cols -> 0."""
    p = spec.problem
    c = p.conv
    kpad = spec.kpad
    bs = spec.block_size
    c0 = b.const_i32(0)
    zero_f = b.const_f32(0.0)

    # Fast path: C contiguous channels per (n, r, s) share validity and live in
    # contiguous W0 (KRSC) + LDS columns; one i8 vector load + ds_write_b128.
    vec_c = (
        spec.vectorize_conv0_a
        and c.C in (2, 4, 8, 16)
        and kpad % c.C == 0
        and c.K_gemm % c.C == 0
    )
    if vec_c:
        cc = c.C
        groups = kpad // cc
        real_groups = c.K_gemm // cc
        c_g = b.const_i32(groups)
        c_total = b.const_i32(spec.tile_n * groups)
        c_kg = b.const_i32(c.K_gemm)
        c_k0 = b.const_i32(c.K)
        zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))
        for e in range((spec.tile_n * groups + bs - 1) // bs):
            idx = b.add(b.const_i32(e * bs), grid.tid)
            in_range = b.cmp_lt(idx, c_total)
            sidx = b.select(in_range, idx, c0)
            n = b.div(sidx, c_g)
            g = b.mod(sidx, c_g)
            valid = b.land(b.cmp_lt(n, c_k0), b.cmp_lt(g, b.const_i32(real_groups)))
            off = b.add(b.mul(n, c_kg), b.mul(g, b.const_i32(cc)))
            safe_off = b.select(valid, off, c0)
            raw = b.global_load_vN(w0_ptr, safe_off, I8, cc)
            comps = [
                b.select(
                    valid,
                    b.trunc_f32_to_f16(_i8_to_f32(b, b.vec_extract(raw, i))),
                    zero_h,
                )
                for i in range(cc)
            ]
            vec = b.vec_pack(comps, F16)
            with b.scf_if(in_range):
                b.smem_store_vN_f16(w0_smem, [n, b.mul(g, b.const_i32(cc))], vec, n=cc)
        return

    total = spec.tile_n * kpad
    ept = (total + bs - 1) // bs
    c_kpad = b.const_i32(kpad)
    c_kg = b.const_i32(c.K_gemm)
    c_k0 = b.const_i32(c.K)
    c_total = b.const_i32(total)

    for e in range(ept):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        sidx = b.select(in_range, idx, c0)
        n = b.div(sidx, c_kpad)
        kg = b.mod(sidx, c_kpad)
        valid = b.land(in_range, b.land(b.cmp_lt(n, c_k0), b.cmp_lt(kg, c_kg)))
        off = b.add(b.mul(n, c_kg), kg)
        safe_off = b.select(valid, off, c0)
        raw_i8 = b.global_load(w0_ptr, safe_off, I8)
        v = b.select(valid, _i8_to_f32(b, raw_i8), zero_f)
        with b.scf_if(in_range):
            b.smem_store_f16(w0_smem, [n, kg], b.trunc_f32_to_f16(v))


def _stage_conv0_a_int(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    x_ptr: Value,
    a_smem: Value,
    grid: WarpGrid,
) -> None:
    """Native-int conv0: im2col the int8 conv0 activations for this tile into the
    *i8* LDS tile ``a_smem[tile_m, kpad]`` as raw int8 codes (no f16 conversion).
    Out-of-image halo and K padding -> 0. The fast path stages all contiguous C
    channels of one (row, r, s) im2col entry with one vector LDS store; the GEMM
    later loads 16 contiguous K bytes per row and bitcasts to the ``<4 x i32>``
    WMMA A fragment."""
    p = spec.problem
    c = p.conv
    kpad = spec.kpad
    bs = spec.block_size
    conv_tile_h = spec.pool_tile_h * p.pool_stride_h
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w

    c_ctw = b.const_i32(conv_tile_w)
    c_Wi = b.const_i32(c.Wi)
    c_Hi = b.const_i32(c.Hi)
    c0 = b.const_i32(0)
    zero_i8 = b.cvt_f32_to_i8_sat(b.const_f32(0.0))
    h_blk = b.block_id_z() if spec.w_fast else b.block_id_y()
    w_blk = b.block_id_y() if spec.w_fast else b.block_id_z()
    h_base = b.mul(h_blk, b.const_i32(conv_tile_h))
    w_base = b.mul(w_blk, b.const_i32(conv_tile_w))

    vec_c = (
        spec.vectorize_conv0_a
        and c.C in (2, 4, 8, 16)
        and kpad % c.C == 0
        and c.K_gemm % c.C == 0
    )
    if vec_c:
        cc = c.C
        groups = kpad // cc
        real_groups = c.K_gemm // cc
        c_g = b.const_i32(groups)
        c_total = b.const_i32(spec.tile_m * groups)
        zero_vec = b.zero_vec(I8, cc)
        for e in range((spec.tile_m * groups + bs - 1) // bs):
            idx = b.add(b.const_i32(e * bs), grid.tid)
            in_range = b.cmp_lt(idx, c_total)
            sidx = b.select(in_range, idx, c0)
            row = b.div(sidx, c_g)
            g = b.mod(sidx, c_g)
            r = b.div(g, b.const_i32(c.X))
            s = b.mod(g, b.const_i32(c.X))
            local_oh = b.div(row, c_ctw)
            local_ow = b.mod(row, c_ctw)
            oh = b.add(h_base, local_oh)
            ow = b.add(w_base, local_ow)
            ih = b.sub(
                b.add(b.mul(oh, b.const_i32(c.sH)), b.mul(r, b.const_i32(c.dH))),
                b.const_i32(c.pH),
            )
            iw = b.sub(
                b.add(b.mul(ow, b.const_i32(c.sW)), b.mul(s, b.const_i32(c.dW))),
                b.const_i32(c.pW),
            )
            h_ok = b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi))
            w_ok = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi))
            g_real = b.cmp_lt(g, b.const_i32(real_groups))
            valid = b.land(in_range, b.land(g_real, b.land(h_ok, w_ok)))
            base_off = b.mul(b.add(b.mul(ih, c_Wi), iw), b.const_i32(cc))
            safe_off = b.select(valid, base_off, c0)
            raw = b.global_load_vN(x_ptr, safe_off, I8, cc)
            code = b.vector_select(valid, raw, zero_vec)
            kg = b.mul(g, b.const_i32(cc))
            with b.scf_if(in_range):
                b.smem_store_vN(a_smem, [row, kg], code, n=cc)
        return

    total = spec.tile_m * kpad
    ept = (total + bs - 1) // bs
    c_kpad = b.const_i32(kpad)
    c_kg = b.const_i32(c.K_gemm)
    c_sc = b.const_i32(c.X * c.C)
    c_cc = b.const_i32(c.C)

    for e in range(ept):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        row = b.div(idx, c_kpad)
        kg = b.mod(idx, c_kpad)
        kg_in = b.cmp_lt(kg, c_kg)

        local_oh = b.div(row, c_ctw)
        local_ow = b.mod(row, c_ctw)
        r = b.div(kg, c_sc)
        rem = b.mod(kg, c_sc)
        s = b.div(rem, c_cc)
        ci = b.mod(rem, c_cc)

        oh = b.add(h_base, local_oh)
        ow = b.add(w_base, local_ow)
        ih = b.sub(
            b.add(b.mul(oh, b.const_i32(c.sH)), b.mul(r, b.const_i32(c.dH))),
            b.const_i32(c.pH),
        )
        iw = b.sub(
            b.add(b.mul(ow, b.const_i32(c.sW)), b.mul(s, b.const_i32(c.dW))),
            b.const_i32(c.pW),
        )
        h_ok = b.land(b.cmp_ge(ih, c0), b.cmp_lt(ih, c_Hi))
        w_ok = b.land(b.cmp_ge(iw, c0), b.cmp_lt(iw, c_Wi))
        valid = b.land(kg_in, b.land(h_ok, w_ok))

        in_off = b.add(b.mul(b.add(b.mul(ih, c_Wi), iw), c_cc), ci)
        safe_off = b.select(valid, in_off, c0)
        raw_i8 = b.global_load(x_ptr, safe_off, I8)
        code = b.select(valid, raw_i8, zero_i8)
        b.smem_store_vN(a_smem, [row, kg], code, n=1)


def _stage_conv0_w0_int(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    w0_ptr: Value,
    w0_smem: Value,
    grid: WarpGrid,
) -> None:
    """Native-int conv0: load int8 conv0 weights ``W0[K0, K_gemm]`` (KYXC
    contiguous) into the *i8* LDS tile ``w0_smem[tile_n, kpad]`` as raw int8
    codes; padding rows/cols -> 0. The fast path stores one contiguous C-channel
    group with a vector LDS store."""
    p = spec.problem
    c = p.conv
    kpad = spec.kpad
    bs = spec.block_size
    c0 = b.const_i32(0)
    zero_i8 = b.cvt_f32_to_i8_sat(b.const_f32(0.0))

    total = spec.tile_n * kpad
    ept = (total + bs - 1) // bs
    c_kpad = b.const_i32(kpad)
    c_kg = b.const_i32(c.K_gemm)
    c_k0 = b.const_i32(c.K)
    c_total = b.const_i32(total)

    vec_c = (
        spec.vectorize_conv0_a
        and c.C in (2, 4, 8, 16)
        and kpad % c.C == 0
        and c.K_gemm % c.C == 0
    )
    if vec_c:
        cc = c.C
        groups = kpad // cc
        real_groups = c.K_gemm // cc
        c_g = b.const_i32(groups)
        c_vec_total = b.const_i32(spec.tile_n * groups)
        zero_vec = b.zero_vec(I8, cc)
        for e in range((spec.tile_n * groups + bs - 1) // bs):
            idx = b.add(b.const_i32(e * bs), grid.tid)
            in_range = b.cmp_lt(idx, c_vec_total)
            sidx = b.select(in_range, idx, c0)
            n = b.div(sidx, c_g)
            g = b.mod(sidx, c_g)
            valid = b.land(
                in_range,
                b.land(b.cmp_lt(n, c_k0), b.cmp_lt(g, b.const_i32(real_groups))),
            )
            off = b.add(b.mul(n, c_kg), b.mul(g, b.const_i32(cc)))
            safe_off = b.select(valid, off, c0)
            raw = b.global_load_vN(w0_ptr, safe_off, I8, cc)
            code = b.vector_select(valid, raw, zero_vec)
            with b.scf_if(in_range):
                b.smem_store_vN(w0_smem, [n, b.mul(g, b.const_i32(cc))], code, n=cc)
        return

    for e in range(ept):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        sidx = b.select(in_range, idx, c0)
        n = b.div(sidx, c_kpad)
        kg = b.mod(sidx, c_kpad)
        valid = b.land(in_range, b.land(b.cmp_lt(n, c_k0), b.cmp_lt(kg, c_kg)))
        off = b.add(b.mul(n, c_kg), kg)
        safe_off = b.select(valid, off, c0)
        raw_i8 = b.global_load(w0_ptr, safe_off, I8)
        code = b.select(valid, raw_i8, zero_i8)
        with b.scf_if(in_range):
            b.smem_store_vN(w0_smem, [n, kg], code, n=1)


def _load_frag_iu8_from_lds(
    b: IRBuilder,
    smem: Value,
    frag_rc: Value,
    atom_off: Value,
    k_tile_base: Value,
) -> Value:
    """Load one native-int WMMA operand fragment from a row-major *i8* LDS tile
    ``smem[tile, kpad]``. ``frag_rc`` is the lane's row (A) / col-as-row (B)
    within the atom (``lane % 16``); ``atom_off`` is the atom's base row in the
    tile; ``k_tile_base`` the K-atom column base. Reads the 16 contiguous K
    bytes of that row (one ``ds_read_b128``) and bitcasts to ``<4 x i32>`` -
    slot ``j`` = K bytes [4j..4j+3] little-endian, matching ``_wmma_*_16x16_iu8``."""
    row = b.add(atom_off, frag_rc)
    raw = b.smem_load_vN(smem, row, k_tile_base, dtype=I8, n=_WMMA)  # <16 x i8>
    return b.bitcast(raw, VectorType(I32, _K_PER_I32))


def _pack_i4_codes_to_i32(b: IRBuilder, codes: Value) -> Value:
    """Pack ``<8 x i8>`` int4 codes into one i32, low nibble first.

    ``codes`` holds signed values in [-8, 7] as byte elements. Masking with
    0xF preserves their two's-complement nibble representation for iu4 WMMA.
    """
    word = b.const_i32(0)
    mask = b.const_i32(0xF)
    for i in range(_I4_PER_I32):
        nib = b.land(b.sext(b.vec_extract(codes, i), I32), mask)
        if i:
            nib = b.shl(nib, b.const_i32(4 * i))
        word = b.lor(word, nib)
    return word


def _load_frag_iu4_codes_from_lds(
    b: IRBuilder,
    smem: Value,
    frag_rc: Value,
    atom_off: Value,
    k_tile_base: Value,
) -> Value:
    """Load one iu4 A fragment from byte-per-code C0 LDS.

    C0 is stored as signed int4 codes in i8 lanes to avoid cross-lane packed-byte
    races during the conv0 epilogue. The conv1 A fragment packs those bytes into
    the two ``<2 x i32>`` iu4 operand slots on demand.
    """
    row = b.add(atom_off, frag_rc)
    words = []
    for slot in range(2):
        raw = b.smem_load_vN(
            smem,
            row,
            b.add(k_tile_base, b.const_i32(slot * _I4_PER_I32)),
            dtype=I8,
            n=_I4_PER_I32,
        )
        words.append(_pack_i4_codes_to_i32(b, raw))
    return b.vec_pack(words, I32)


def _load_frag_iu4_packed_from_lds(
    b: IRBuilder,
    smem: Value,
    frag_rc: Value,
    atom_off: Value,
    k_tile_base: Value,
) -> Value:
    """Load one iu4 B fragment from packed-byte W1 LDS.

    W1 is already packed two int4 values per byte, low nibble first. A 16-wide
    K atom is exactly eight bytes, which bitcasts to the ``<2 x i32>`` WMMA
    operand ABI.
    """
    row = b.add(atom_off, frag_rc)
    byte_base = b.div(k_tile_base, b.const_i32(2))
    raw = b.smem_load_vN(smem, row, byte_base, dtype=I8, n=8)  # <8 x i8>
    return b.bitcast(raw, VectorType(I32, 2))


def _wmma_gemm_from_lds_int(
    b: IRBuilder,
    op,
    a_smem: Value,
    b_smem: Value,
    grid: WarpGrid,
    k_total: int,
    policy=None,
) -> List[Value]:
    """Native-int WMMA GEMM ``A[tile_m,k] @ B[tile_n,k].T`` from two row-major
    *i8* LDS tiles (M/N = row, K = column), accumulating in i32. Twin of
    :func:`_wmma_gemm_from_lds`; returns mfmas_m*mfmas_n ``<8 x i32>`` accs."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = k_total // _WMMA
    a_map = op.a_layout()
    b_map = op.b_layout()
    a_row, _a_k = a_map.coord(b, grid.lane, 0)  # a_row = lane % 16
    _b_k, b_col = b_map.coord(b, grid.lane, 0)  # b_col = lane % 16
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    # One ds_read_b128 per fragment (16 i8 = 128 bits).
    n_ds = mfmas_m + mfmas_n

    accs = [b.zero_vec(I32, op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    for kk in range(k_atoms):
        k_tile_base = b.const_i32(kk * _WMMA)
        a_rows = []
        for mi in range(mfmas_m):
            atom_off = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            a_rows.append(
                _load_frag_iu8_from_lds(b, a_smem, a_row, atom_off, k_tile_base)
            )
        b_cols = []
        for ni in range(mfmas_n):
            atom_off = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            b_cols.append(
                _load_frag_iu8_from_lds(b, b_smem, b_col, atom_off, k_tile_base)
            )
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                flat += 1
        _emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n)
    return accs


def _wmma_gemm_conv1_i4_from_lds(
    b: IRBuilder,
    op,
    c0_smem: Value,
    w1_smem: Value,
    grid: WarpGrid,
    k_total: int,
    policy=None,
    prefetch_k: bool = False,
    sched_fuse: bool = False,
) -> List[Value]:
    """Native iu4 conv1 GEMM from C0 byte codes and packed W1 LDS.

    A (C0) is byte-per-int4-code LDS and is packed on fragment load. B (W1) is
    already packed two int4 values per byte, staged directly from HBM.

    With ``prefetch_k`` the A/B fragments for ALL k-steps are loaded up front
    (before any MMA) so the k=1 ds_read latency overlaps the k=0 MMAs; default
    keeps the per-k-step hoist (k=1 loads exposed at the k=1 iteration top).

    With ``sched_fuse`` the per-k-step DS_READ->MFMA sched_group_barrier pair is
    suppressed and ONE combined group (all k_atoms loads, then all k_atoms MMAs)
    is emitted after the loop, so the post-RA scheduler may interleave the k=1
    loads with the k=0 MMAs instead of being pinned per k-step.
    """
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = k_total // _WMMA
    a_map = op.a_layout()
    b_map = op.b_layout()
    a_row, _a_k = a_map.coord(b, grid.lane, 0)  # a_row = lane % 16
    _b_k, b_col = b_map.coord(b, grid.lane, 0)  # b_col = lane % 16
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    # A: two 8-byte LDS loads then pack; B: one 8-byte LDS load.
    n_ds = 2 * mfmas_m + mfmas_n

    def load_a(kk: int) -> List[Value]:
        k_tile_base = b.const_i32(kk * _WMMA)
        out = []
        for mi in range(mfmas_m):
            atom_off = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            out.append(
                _load_frag_iu4_codes_from_lds(b, c0_smem, a_row, atom_off, k_tile_base)
            )
        return out

    def load_b(kk: int) -> List[Value]:
        k_tile_base = b.const_i32(kk * _WMMA)
        out = []
        for ni in range(mfmas_n):
            atom_off = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            out.append(
                _load_frag_iu4_packed_from_lds(b, w1_smem, b_col, atom_off, k_tile_base)
            )
        return out

    accs = [b.zero_vec(I32, op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    # Per-step hint is suppressed when fusing; the combined group is emitted once
    # after the loop instead.
    per_step = None if sched_fuse else policy

    if prefetch_k:
        a_all = [load_a(kk) for kk in range(k_atoms)]
        b_all = [load_b(kk) for kk in range(k_atoms)]
        for kk in range(k_atoms):
            a_rows, b_cols = a_all[kk], b_all[kk]
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                    flat += 1
            _emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n)
    else:
        for kk in range(k_atoms):
            a_rows = load_a(kk)
            b_cols = load_b(kk)
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                    flat += 1
            _emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n)

    if sched_fuse:
        _emit_wmma_k_sched(b, policy, n_ds * k_atoms, mfmas_m * mfmas_n * k_atoms)
    return accs


def _wmma_gemm_conv1_i4_packed_from_lds(
    b: IRBuilder,
    op,
    c0_smem: Value,
    w1_smem: Value,
    grid: WarpGrid,
    k_total: int,
    policy=None,
) -> List[Value]:
    """Native iu4 conv1 GEMM with both A(C0) and B(W1) packed in LDS."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = k_total // _WMMA
    a_map = op.a_layout()
    b_map = op.b_layout()
    a_row, _a_k = a_map.coord(b, grid.lane, 0)
    _b_k, b_col = b_map.coord(b, grid.lane, 0)
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    n_ds = mfmas_m + mfmas_n

    accs = [b.zero_vec(I32, op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    for kk in range(k_atoms):
        k_tile_base = b.const_i32(kk * _WMMA)
        a_rows = []
        for mi in range(mfmas_m):
            atom_off = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            a_rows.append(
                _load_frag_iu4_packed_from_lds(b, c0_smem, a_row, atom_off, k_tile_base)
            )
        b_cols = []
        for ni in range(mfmas_n):
            atom_off = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            b_cols.append(
                _load_frag_iu4_packed_from_lds(b, w1_smem, b_col, atom_off, k_tile_base)
            )
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                flat += 1
        _emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n)
    return accs


def _fuse_c0_to_conv1_a_regs(
    b: IRBuilder,
    op0,
    accs0: Sequence[Value],
    grid: WarpGrid,
    code_fn,
    int8: bool = False,
) -> List[List[Value]]:
    """In-register conv0->conv1 handoff (the fused_c0a1 transpose).

    Consumes the REORIENTED conv0 accumulators (``acc[k0, m]``: lane ``L`` holds
    ``m = m_atom_base + L%16``; slot ``s``, ``half = L//16`` -> ``k0_local =
    2s + half``) and produces the conv1 iu4 A-fragments directly in registers,
    deleting c0_smem, the conv0 scatter, and the handoff barrier.

    For each acc atom ``(mi, kk)`` the 8 i32 slots requant to 8 int4 byte codes
    of ONE k0 parity; the other parity lives on lane ``L^16``. One
    ``permlanex16`` per 4-code word fetches the partner, a half-dependent
    ``select`` orders even/odd k0, ``v_perm_b32`` interleaves the bytes into
    contiguous k0 order, and the existing nibble-pack builds the ``<2 x i32>``
    A-fragment (slot0 = k0_local 0..7, slot1 = 8..15). Lane ``L`` and ``L^16``
    reconstruct identical fragments, satisfying the iu4 A-operand cross-half
    duplication.

    Returns ``a_frags[mi][kk]`` for ``mi in range(mfmas_m)``, ``kk in
    range(mfmas_n)`` (conv0 N atoms over k0 == conv1 K atoms).
    """
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    is_lo = b.cmp_lt(grid.lane, b.const_i32(16))  # half == 0 (wave32)
    out: List[List[Value]] = [[None] * mfmas_n for _ in range(mfmas_m)]
    for mi in range(mfmas_m):
        for kk in range(mfmas_n):
            acc = accs0[mi * mfmas_n + kk]
            codes = [code_fn(b.vec_extract(acc, s)) for s in range(op0.c_frag_len)]
            words = b.bitcast(b.vec_pack(codes, I8), VectorType(I32, 2))
            lo = b.vec_extract(words, 0)  # this half's k0_local {0,2,4,6}+half
            hi = b.vec_extract(words, 1)  # this half's k0_local {8,10,12,14}+half
            plo = b.permlanex16(lo)  # partner half (lane^16)
            phi = b.permlanex16(hi)
            e_lo = b.select(is_lo, lo, plo)  # even k0_local 0,2,4,6
            o_lo = b.select(is_lo, plo, lo)  # odd  k0_local 1,3,5,7
            e_hi = b.select(is_lo, hi, phi)  # even k0_local 8,10,12,14
            o_hi = b.select(is_lo, phi, hi)  # odd  k0_local 9,11,13,15
            # v_perm sources: 0..3 = arg b LSB-first, 4..7 = arg a LSB-first.
            ord0 = b.byte_perm(o_lo, e_lo, 0x05010400)  # k0_local 0,1,2,3
            ord1 = b.byte_perm(o_lo, e_lo, 0x07030602)  # k0_local 4,5,6,7
            ord2 = b.byte_perm(o_hi, e_hi, 0x05010400)  # k0_local 8,9,10,11
            ord3 = b.byte_perm(o_hi, e_hi, 0x07030602)  # k0_local 12,13,14,15
            if int8:
                # iu8 A-fragment is <4 x i32>, slot j = K bytes [4j..4j+3]
                # little-endian. ordN already holds the 4 contiguous k0 byte
                # codes for that slot, so hand them through directly -- no nibble
                # squeeze. Codes stay int4-range, so iu8 reads them bit-exactly.
                out[mi][kk] = b.vec_pack([ord0, ord1, ord2, ord3], I32)
                continue
            s0 = _pack_i4_codes_to_i32(
                b, b.bitcast(b.vec_pack([ord0, ord1], I32), VectorType(I8, 8))
            )
            s1 = _pack_i4_codes_to_i32(
                b, b.bitcast(b.vec_pack([ord2, ord3], I32), VectorType(I8, 8))
            )
            out[mi][kk] = b.vec_pack([s0, s1], I32)
    return out


def _wmma_gemm_conv1_i4_from_regs(
    b: IRBuilder,
    op,
    a_frags: Sequence[Sequence[Value]],
    w1_smem: Value,
    grid: WarpGrid,
    k_total: int,
    policy=None,
    prefetch_k: bool = False,
    sched_fuse: bool = False,
) -> List[Value]:
    """conv1 iu4 GEMM with A from the in-register fused handoff and B (W1) from
    packed LDS. Twin of :func:`_wmma_gemm_conv1_i4_from_lds` minus the c0_smem
    A loads/pack -- only the W1 fragments touch LDS now.

    A is already register-resident (the fused handoff), so the only cross-k-step
    latency to hide is the W1 B-fragment ds_read at k=1. ``prefetch_k`` hoists ALL
    k-step B loads before any MMA so the k=1 ds_read overlaps the k=0 MMAs;
    ``sched_fuse`` replaces the per-k-step sched_group_barrier with one combined
    group after the loop so the post-RA scheduler may pull the k=1 loads ahead of
    the k=0 MMAs (the per-step barrier otherwise pins them per k-step). Both are
    pure load/schedule reordering -- identical MMAs/accs, bit-exact."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = k_total // _WMMA
    b_map = op.b_layout()
    _b_k, b_col = b_map.coord(b, grid.lane, 0)  # b_col = lane % 16
    warp_n_off = grid.warp_n_off(b)
    n_ds = mfmas_n  # one 8-byte W1 load per n-atom per k-step

    def load_b(kk: int) -> List[Value]:
        k_tile_base = b.const_i32(kk * _WMMA)
        out = []
        for ni in range(mfmas_n):
            atom_off = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            out.append(
                _load_frag_iu4_packed_from_lds(b, w1_smem, b_col, atom_off, k_tile_base)
            )
        return out

    accs = [b.zero_vec(I32, op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    per_step = None if sched_fuse else policy

    b_all = [load_b(kk) for kk in range(k_atoms)] if prefetch_k else None
    for kk in range(k_atoms):
        b_cols = b_all[kk] if prefetch_k else load_b(kk)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                accs[flat] = b.mma(op, a_frags[mi][kk], b_cols[ni], accs[flat])
                flat += 1
        _emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n)

    if sched_fuse:
        _emit_wmma_k_sched(b, policy, n_ds * k_atoms, mfmas_m * mfmas_n * k_atoms)
    return accs


def _wmma_gemm_conv1_i8_from_regs(
    b: IRBuilder,
    op,
    a_frags: Sequence[Sequence[Value]],
    w1_smem: Value,
    grid: WarpGrid,
    k_total: int,
    policy=None,
    prefetch_k: bool = False,
    sched_fuse: bool = False,
) -> List[Value]:
    """conv1 iu8 GEMM with A from the in-register fused handoff and B (W1) from
    byte-per-code i8 LDS. Twin of :func:`_wmma_gemm_conv1_i4_from_regs` but the
    iu8 atom: A-fragments are the unsqueezed ``<4 x i32>`` byte codes and W1 is
    one 16-byte row per K atom (one ``ds_read_b128`` per n-atom). The conv0 codes
    stay int4-range, so the iu8 dot products are bit-identical to the iu4 path.
    ``prefetch_k``/``sched_fuse`` are pure load/schedule reordering -- bit-exact."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = k_total // _WMMA
    b_map = op.b_layout()
    _b_k, b_col = b_map.coord(b, grid.lane, 0)  # b_col = lane % 16
    warp_n_off = grid.warp_n_off(b)
    n_ds = mfmas_n  # one 16-byte W1 load (ds_read_b128) per n-atom per k-step

    def load_b(kk: int) -> List[Value]:
        k_tile_base = b.const_i32(kk * _WMMA)
        out = []
        for ni in range(mfmas_n):
            atom_off = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            out.append(
                _load_frag_iu8_from_lds(b, w1_smem, b_col, atom_off, k_tile_base)
            )
        return out

    accs = [b.zero_vec(I32, op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    per_step = None if sched_fuse else policy

    b_all = [load_b(kk) for kk in range(k_atoms)] if prefetch_k else None
    for kk in range(k_atoms):
        b_cols = b_all[kk] if prefetch_k else load_b(kk)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                accs[flat] = b.mma(op, a_frags[mi][kk], b_cols[ni], accs[flat])
                flat += 1
        _emit_wmma_k_sched(b, per_step, n_ds, mfmas_m * mfmas_n)

    if sched_fuse:
        _emit_wmma_k_sched(b, policy, n_ds * k_atoms, mfmas_m * mfmas_n * k_atoms)
    return accs


def _stage_conv1_w1(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    w1_ptr: Value,
    w1_smem: Value,
    grid: WarpGrid,
) -> None:
    """Unpack packed-int4 conv1 weights ``W1[K1, K0/2]`` (2 codes/byte, low
    nibble = even k0) into ``w1_smem[tile_n, K0]`` as fp16 codes; padding -> 0."""
    from ...helpers.i4_dequant import unpack_i4_byte_to_pair_i32

    p = spec.problem
    c = p.conv
    k0 = c.K  # conv1 K
    k1 = p.conv1_channels
    bs = spec.block_size
    bytes_per_row = k0 // 2
    total = spec.tile_n * bytes_per_row  # one thread-element per packed byte
    ept = (total + bs - 1) // bs

    c_bpr = b.const_i32(bytes_per_row)
    c_k1 = b.const_i32(k1)
    c_total = b.const_i32(total)
    c0 = b.const_i32(0)
    zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))

    for e in range(ept):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        sidx = b.select(in_range, idx, c0)
        n = b.div(sidx, c_bpr)
        kb = b.mod(sidx, c_bpr)  # byte column
        valid = b.land(in_range, b.cmp_lt(n, c_k1))
        off = b.add(b.mul(n, c_bpr), kb)
        safe_off = b.select(valid, off, c0)
        byte = b.global_load(w1_ptr, safe_off, I8)
        lo_i32, hi_i32 = unpack_i4_byte_to_pair_i32(b, byte)
        lo_h = b.trunc_f32_to_f16(b.sitofp_f32(lo_i32))
        hi_h = b.trunc_f32_to_f16(b.sitofp_f32(hi_i32))
        lo_h = b.select(valid, lo_h, zero_h)
        hi_h = b.select(valid, hi_h, zero_h)
        k_lo = b.mul(kb, b.const_i32(2))
        k_hi = b.add(k_lo, b.const_i32(1))
        with b.scf_if(in_range):
            b.smem_store_f16(w1_smem, [n, k_lo], lo_h)
            b.smem_store_f16(w1_smem, [n, k_hi], hi_h)


def _stage_conv1_w1_packed(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    w1_ptr: Value,
    w1_smem: Value,
    grid: WarpGrid,
) -> None:
    """Stage packed-int4 W1 bytes into LDS without unpacking to fp16."""
    p = spec.problem
    c = p.conv
    k1 = p.conv1_channels
    bytes_per_row = c.K // 2
    bs = spec.block_size
    c0 = b.const_i32(0)
    zero_vec = b.zero_vec(I8, bytes_per_row)
    c_total = b.const_i32(spec.tile_n)
    c_k1 = b.const_i32(k1)
    c_bpr = b.const_i32(bytes_per_row)

    for e in range((spec.tile_n + bs - 1) // bs):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        n = b.select(in_range, idx, c0)
        valid = b.land(in_range, b.cmp_lt(n, c_k1))
        off = b.mul(n, c_bpr)
        safe_off = b.select(valid, off, c0)
        raw = b.global_load_vN(w1_ptr, safe_off, I8, bytes_per_row)
        packed = b.vector_select(valid, raw, zero_vec)
        with b.scf_if(in_range):
            b.smem_store_vN(w1_smem, [n, c0], packed, n=bytes_per_row)


def _stage_conv1_w1_i8(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    w1_ptr: Value,
    w1_smem: Value,
    grid: WarpGrid,
) -> None:
    """Unpack packed-int4 conv1 weights ``W1[K1, K0/2]`` (2 codes/byte, low
    nibble = even k0) into ``w1_smem[tile_n, K0]`` as sign-extended int4 codes in
    i8 lanes (byte-per-code), for the iu8 conv1 atom; padding -> 0. Twin of
    :func:`_stage_conv1_w1` but stores int8 bytes instead of fp16 codes."""
    from ...helpers.i4_dequant import unpack_i4_byte_to_pair_i32

    p = spec.problem
    c = p.conv
    k0 = c.K  # conv1 K
    k1 = p.conv1_channels
    bs = spec.block_size
    bytes_per_row = k0 // 2
    total = spec.tile_n * bytes_per_row  # one thread-element per packed byte
    ept = (total + bs - 1) // bs

    c_bpr = b.const_i32(bytes_per_row)
    c_k1 = b.const_i32(k1)
    c_total = b.const_i32(total)
    c0 = b.const_i32(0)
    zero_i8 = b.trunc(c0, I8)

    for e in range(ept):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        sidx = b.select(in_range, idx, c0)
        n = b.div(sidx, c_bpr)
        kb = b.mod(sidx, c_bpr)  # byte column
        valid = b.land(in_range, b.cmp_lt(n, c_k1))
        off = b.add(b.mul(n, c_bpr), kb)
        safe_off = b.select(valid, off, c0)
        byte = b.global_load(w1_ptr, safe_off, I8)
        lo_i32, hi_i32 = unpack_i4_byte_to_pair_i32(b, byte)
        lo_i8 = b.select(valid, b.trunc(lo_i32, I8), zero_i8)
        hi_i8 = b.select(valid, b.trunc(hi_i32, I8), zero_i8)
        k_lo = b.mul(kb, b.const_i32(2))
        k_hi = b.add(k_lo, b.const_i32(1))
        with b.scf_if(in_range):
            b.smem_store_vN(w1_smem, [n, k_lo], lo_i8, n=1)
            b.smem_store_vN(w1_smem, [n, k_hi], hi_i8, n=1)


def _emit_wmma_k_sched(b: IRBuilder, policy, n_ds: int, n_mma: int) -> None:
    """L2: per-k-atom DS_READ -> MFMA group hint for the AMDGPU post-RA
    scheduler. Correctness-neutral (scheduling only); no-op when ``policy`` is
    None or hints are off. WMMA lowers under the MFMA sched class on RDNA."""
    if policy is None or not policy.emit_hints:
        return
    b.sched_group_barrier(DS_READ, int(n_ds), 0)
    b.sched_group_barrier(MFMA, int(n_mma), 0)


def _wmma_gemm_from_lds(
    b: IRBuilder,
    op,
    a_smem: Value,
    b_smem: Value,
    grid: WarpGrid,
    k_total: int,
    policy=None,
) -> List[Value]:
    """Generic WMMA GEMM accumulating ``A[tile_m,k] @ B[tile_n,k].T`` from two
    row-major LDS tiles (M/N = row, K = column). Returns mfmas_m*mfmas_n accs."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = k_total // _WMMA
    a_map = op.a_layout()
    b_map = op.b_layout()
    a_row, a_k = a_map.coord(b, grid.lane, 0)
    b_k, b_col = b_map.coord(b, grid.lane, 0)
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    # ds_reads/k-atom: each frag is loaded in 8-wide fp16 chunks.
    ds_per_frag = (op.a_frag_len + 7) // 8
    n_ds = ds_per_frag * (mfmas_m + mfmas_n)

    accs = [b.zero_vec_f32(op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    for kk in range(k_atoms):
        k_tile_base = b.const_i32(kk * _WMMA)
        a_rows = []
        for mi in range(mfmas_m):
            atom_row = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            a_rows.append(
                _emit_frag_smem_load(
                    b, a_smem, a_row, a_k, atom_row, k_tile_base, op.a_frag_len
                )
            )
        b_cols = []
        for ni in range(mfmas_n):
            atom_row = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            b_cols.append(
                _emit_frag_smem_load(
                    b, b_smem, b_col, b_k, atom_row, k_tile_base, op.b_frag_len
                )
            )
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                flat += 1
        _emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n)
    return accs


def _load_conv0_a_frag_from_footprint(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    inp_smem: Value,
    m_row: Value,
    k_base: Value,
    frag_len: int,
) -> Value:
    """Gather one WMMA A operand fragment directly from the cached input
    footprint (direct-conv mode). The lane owns tile-M row ``m_row`` and
    ``frag_len`` K-contiguous implicit-GEMM columns starting at ``k_base``;
    each column ``kg = r*S*C + s*C + ci`` maps to footprint pixel
    ``(local_oh*sH + r*dH, local_ow*sW + s*dW)`` channel ``ci``. K-pad -> 0.

    VALU-reduced: row-dependent ``local_oh/local_ow`` are hoisted out of the
    per-element loop, and div/mod by ``C`` is strength-reduced to shift/mask
    when ``C`` is a power of two."""
    c = spec.problem.conv
    c_ctw = b.const_i32(spec.conv_tile_w)
    c_sc = b.const_i32(c.X * c.C)
    c_fw = b.const_i32(spec.foot_w)
    c_kg = b.const_i32(c.K_gemm)
    zero_h = b.trunc_f32_to_f16(b.const_f32(0.0))

    local_oh = b.div(m_row, c_ctw)
    local_ow = b.mod(m_row, c_ctw)
    oh_base = b.mul(local_oh, b.const_i32(c.sH))
    ow_base = b.mul(local_ow, b.const_i32(c.sW))

    is_pow2_c = c.C > 0 and (c.C & (c.C - 1)) == 0
    c_log2 = (c.C - 1).bit_length() if is_pow2_c else 0

    elems = []
    for i in range(frag_len):
        kg = b.add(k_base, b.const_i32(i))
        kg_ok = b.cmp_lt(kg, c_kg)
        r = b.div(kg, c_sc)
        rem = b.mod(kg, c_sc)
        if is_pow2_c:
            s_col = b.lshr(rem, b.const_i32(c_log2))
            ci = b.land(rem, b.const_i32(c.C - 1))
        else:
            s_col = b.div(rem, b.const_i32(c.C))
            ci = b.mod(rem, b.const_i32(c.C))
        fr = b.add(oh_base, b.mul(r, b.const_i32(c.dH)))
        fw = b.add(ow_base, b.mul(s_col, b.const_i32(c.dW)))
        foot_row = b.add(b.mul(fr, c_fw), fw)
        raw = b.vec_extract(b.smem_load_vN_f16(inp_smem, foot_row, ci, n=1), 0)
        elems.append(b.select(kg_ok, raw, zero_h))
    return b.vec_pack(elems, elems[0].type)


def _load_conv0_a_frag_from_footprint_iu8(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    inp_smem: Value,
    m_row: Value,
    k_base: Value,
) -> Value:
    """Gather and pack one direct-conv A fragment for iu8 WMMA.

    The cached input footprint is raw i8. Each iu8 operand slot packs four
    contiguous K bytes into one i32. For the target C=8 shape, every 4-byte group
    stays within one footprint pixel row.
    """
    c = spec.problem.conv
    c_ctw = b.const_i32(spec.conv_tile_w)
    c_sc = b.const_i32(c.X * c.C)
    c_fw = b.const_i32(spec.foot_w)
    c_kg = b.const_i32(c.K_gemm)

    local_oh = b.div(m_row, c_ctw)
    local_ow = b.mod(m_row, c_ctw)
    oh_base = b.mul(local_oh, b.const_i32(c.sH))
    ow_base = b.mul(local_ow, b.const_i32(c.sW))

    is_pow2_c = c.C > 0 and (c.C & (c.C - 1)) == 0
    c_log2 = (c.C - 1).bit_length() if is_pow2_c else 0
    zero_vec = b.zero_vec(I8, _K_PER_I32)

    words = []
    for slot in range(_K_PER_I32):
        kg = b.add(k_base, b.const_i32(slot * _K_PER_I32))
        kg_ok = b.cmp_lt(kg, c_kg)
        r = b.div(kg, c_sc)
        rem = b.mod(kg, c_sc)
        if is_pow2_c:
            s_col = b.lshr(rem, b.const_i32(c_log2))
            ci = b.land(rem, b.const_i32(c.C - 1))
        else:
            s_col = b.div(rem, b.const_i32(c.C))
            ci = b.mod(rem, b.const_i32(c.C))
        fr = b.add(oh_base, b.mul(r, b.const_i32(c.dH)))
        fw = b.add(ow_base, b.mul(s_col, b.const_i32(c.dW)))
        foot_row = b.add(b.mul(fr, c_fw), fw)
        raw = b.smem_load_vN(inp_smem, foot_row, ci, dtype=I8, n=_K_PER_I32)
        code = b.vector_select(kg_ok, raw, zero_vec)
        words.append(b.bitcast(code, I32))
    return b.vec_pack(words, I32)


def _load_conv0_a_frag_from_footprint_iu8_static(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    inp_smem: Value,
    m_row: Value,
    kk: int,
) -> Value:
    """Static-K-map sibling of ``_load_conv0_a_frag_from_footprint_iu8``.

    Target specialization: C=8, Y=X=3, iu8 slots pack four contiguous C values.
    The flattened K -> (y, x, ci) mapping is compile-time for each kk/slot.
    """
    c = spec.problem.conv
    c_ctw = b.const_i32(spec.conv_tile_w)
    c_fw = b.const_i32(spec.foot_w)
    local_oh = b.div(m_row, c_ctw)
    local_ow = b.mod(m_row, c_ctw)
    oh_base = b.mul(local_oh, b.const_i32(c.sH))
    ow_base = b.mul(local_ow, b.const_i32(c.sW))

    words = []
    for slot in range(_K_PER_I32):
        kg0 = kk * _WMMA + slot * _K_PER_I32
        if kg0 >= c.K_gemm:
            words.append(b.const_i32(0))
            continue
        r = kg0 // (c.X * c.C)
        rem = kg0 % (c.X * c.C)
        s_col = rem // c.C
        ci = rem % c.C
        fr = b.add(oh_base, b.const_i32(r * c.dH))
        fw = b.add(ow_base, b.const_i32(s_col * c.dW))
        foot_row = b.add(b.mul(fr, c_fw), fw)
        raw = b.smem_load_vN(
            inp_smem, foot_row, b.const_i32(ci), dtype=I8, n=_K_PER_I32
        )
        words.append(b.bitcast(raw, I32))
    return b.vec_pack(words, I32)


def _wmma_gemm_conv0_direct(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    op,
    inp_smem: Value,
    w0_smem: Value,
    grid: WarpGrid,
    policy=None,
) -> List[Value]:
    """Direct-conv conv0 WMMA GEMM: A fragments are gathered from the input
    footprint cache (``inp_smem``) via conv addressing; B (W0) from its LDS tile.
    Mirrors ``_wmma_gemm_from_lds`` but with no materialized im2col A tile."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = spec.kpad // _WMMA
    a_map = op.a_layout()
    b_map = op.b_layout()
    a_row, a_k = a_map.coord(b, grid.lane, 0)
    b_k, b_col = b_map.coord(b, grid.lane, 0)
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    # A frags gathered per-element from the footprint (frag_len n=1 ds_reads);
    # B frags via 8-wide chunks. Used for the L2 group-hint counts.
    n_ds = op.a_frag_len * mfmas_m + ((op.b_frag_len + 7) // 8) * mfmas_n

    accs = [b.zero_vec_f32(op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    for kk in range(k_atoms):
        k_tile_base = b.const_i32(kk * _WMMA)
        k_base = b.add(k_tile_base, a_k)
        a_rows = []
        for mi in range(mfmas_m):
            atom_row = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            m_row = b.add(atom_row, a_row)
            a_rows.append(
                _load_conv0_a_frag_from_footprint(
                    b, spec, inp_smem, m_row, k_base, op.a_frag_len
                )
            )
        b_cols = []
        for ni in range(mfmas_n):
            atom_row = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            b_cols.append(
                _emit_frag_smem_load(
                    b, w0_smem, b_col, b_k, atom_row, k_tile_base, op.b_frag_len
                )
            )
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                flat += 1
        _emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n)
    return accs


def _wmma_gemm_conv0_direct_int(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    op,
    inp_smem: Value,
    w0_smem: Value,
    grid: WarpGrid,
    policy=None,
    reorient: bool = False,
) -> List[Value]:
    """Direct native-int conv0: A from raw input footprint, B from i8 W0 LDS.

    With ``reorient`` (the fused_c0a1 handoff) the two WMMA operands are SWAPPED:
    W0 is fed as the A operand and the footprint as the B operand. Both iu8
    fragments share the lane->l%16, slot->k_base ABI so the swap is free, but it
    transposes the accumulator from ``C[m, k0]`` (lane->k0, slot->m) to
    ``C[k0, m]`` (lane->m, slot->k0) -- the distribution the conv1 iu4 A-fragment
    needs, so the k0<->m handoff transpose becomes an in-register permlanex16
    instead of an LDS round-trip. acc atom ``(mi, ni)`` then holds m-atom ``mi``
    on the lanes and k0-atom ``ni`` in the slots; the math is identical
    (``acc[k0,m] = sum_c W0[k0,c]*foot[m,c]``), so requant codes are bit-exact."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    k_atoms = spec.kpad // _WMMA
    a_map = op.a_layout()
    b_map = op.b_layout()
    a_row, a_k = a_map.coord(b, grid.lane, 0)
    _b_k, b_col = b_map.coord(b, grid.lane, 0)
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    # A: four 4-byte LDS loads then bitcast; B: one 16-byte LDS load.
    n_ds = _K_PER_I32 * mfmas_m + mfmas_n

    accs = [b.zero_vec(I32, op.c_frag_len) for _ in range(mfmas_m * mfmas_n)]
    for kk in range(k_atoms):
        k_tile_base = b.const_i32(kk * _WMMA)
        k_base = b.add(k_tile_base, a_k)
        a_rows = []
        for mi in range(mfmas_m):
            atom_row = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            m_row = b.add(atom_row, a_row)
            if spec.static_direct_kmap:
                a_rows.append(
                    _load_conv0_a_frag_from_footprint_iu8_static(
                        b, spec, inp_smem, m_row, kk
                    )
                )
            else:
                a_rows.append(
                    _load_conv0_a_frag_from_footprint_iu8(
                        b, spec, inp_smem, m_row, k_base
                    )
                )
        b_cols = []
        for ni in range(mfmas_n):
            atom_off = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            b_cols.append(
                _load_frag_iu8_from_lds(b, w0_smem, b_col, atom_off, k_tile_base)
            )
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                if reorient:
                    # Swap operands: W0 -> A (row=k0), footprint -> B (col=m).
                    accs[flat] = b.mma(op, b_cols[ni], a_rows[mi], accs[flat])
                else:
                    accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], accs[flat])
                flat += 1
        _emit_wmma_k_sched(b, policy, n_ds, mfmas_m * mfmas_n)
    return accs


def _scatter_codes_to_lds(
    b: IRBuilder,
    op,
    accs: Sequence[Value],
    dst_smem: Value,
    grid: WarpGrid,
    code_fn,
) -> None:
    """Apply ``code_fn(acc_slot_f32) -> f16 code`` to each WMMA accumulator slot
    and store it at its (row, col) in the row-major ``dst_smem`` tile."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    c_map = op.c_layout()
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    flat = 0
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            m_base = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            n_base = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, grid.lane, i)
                row = b.add(m_base, row_off)
                col = b.add(n_base, col_off)
                code_h = code_fn(b.vec_extract(acc, i))
                b.smem_store_f16(dst_smem, [row, col], code_h)


def _scatter_codes_to_i8_lds(
    b: IRBuilder,
    op,
    accs: Sequence[Value],
    dst_smem: Value,
    grid: WarpGrid,
    code_fn,
) -> None:
    """Apply ``code_fn(acc_slot) -> i8 int4 code`` and store byte-per-code LDS."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    c_map = op.c_layout()
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    flat = 0
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            m_base = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            n_base = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, grid.lane, i)
                row = b.add(m_base, row_off)
                col = b.add(n_base, col_off)
                b.smem_store_vN(
                    dst_smem, [row, col], code_fn(b.vec_extract(acc, i)), n=1
                )


def _scatter_vec_codes_to_i8_lds(
    b: IRBuilder,
    op,
    accs: Sequence[Value],
    dst_smem: Value,
    grid: WarpGrid,
    vec_code_fn,
) -> None:
    """Vector-quantize one accumulator vector, then scatter i8 code lanes."""
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    c_map = op.c_layout()
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    flat = 0
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            codes = vec_code_fn(acc)
            m_base = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            n_base = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, grid.lane, i)
                row = b.add(m_base, row_off)
                col = b.add(n_base, col_off)
                b.smem_store_vN(dst_smem, [row, col], b.vec_extract(codes, i), n=1)


def _scatter_packed_i4_codes_to_lds(
    b: IRBuilder,
    op,
    accs: Sequence[Value],
    dst_smem: Value,
    grid: WarpGrid,
    code_fn,
) -> None:
    """Pack adjacent conv0 C columns into one byte and store from even lanes.

    WMMA C maps adjacent columns to adjacent lanes, so the even lane uses
    ds_bpermute to read the odd lane's code, packs high/low nibbles, and stores
    one byte. Odd lanes do not write.
    """
    mfmas_m = grid.mfmas_per_warp_m
    mfmas_n = grid.mfmas_per_warp_n
    c_map = op.c_layout()
    warp_m_off = grid.warp_m_off(b)
    warp_n_off = grid.warp_n_off(b)
    flat = 0
    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c4 = b.const_i32(4)
    c0xf = b.const_i32(0xF)
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            m_base = b.add(warp_m_off, b.const_i32(mi * _WMMA))
            n_base = b.add(warp_n_off, b.const_i32(ni * _WMMA))
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, grid.lane, i)
                row = b.add(m_base, row_off)
                col = b.add(n_base, col_off)
                col_is_even = b.cmp_eq(b.land(col, c1), c0)
                code = b.sext(code_fn(b.vec_extract(acc, i)), I32)
                src_lane = b.add(grid.lane, c1)
                odd_code = b.ds_bpermute(b.shl(src_lane, b.const_i32(2)), code)
                lo = b.land(code, c0xf)
                hi = b.shl(b.land(odd_code, c0xf), c4)
                packed = b.trunc(b.lor(lo, hi), I8)
                with b.scf_if(col_is_even):
                    b.smem_store_vN(
                        dst_smem, [row, b.div(col, b.const_i32(2))], packed, n=1
                    )


def _repack_c0_lds_to_packed(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    c0_smem: Value,
    c0_packed_smem: Value,
    grid: WarpGrid,
) -> None:
    """Lane-local LDS->LDS repack of byte-per-code C0 into packed bytes.

    Each thread reads two adjacent-K int4 codes (a plain LDS read, lane-agnostic,
    NO ds_bpermute) and writes one packed byte (low nibble = even K). Done once
    per packed byte, off the conv1 GEMM critical path, so conv1 can load A as a
    bitcast (``_load_frag_iu4_packed_from_lds``) instead of packing on every
    fragment load.
    """
    k0 = spec.problem.conv.K
    bpr = k0 // 2  # packed bytes per row
    total = spec.tile_m * bpr
    bs = spec.block_size
    ept = (total + bs - 1) // bs
    c_bpr = b.const_i32(bpr)
    c_total = b.const_i32(total)
    c0 = b.const_i32(0)
    c0xf = b.const_i32(0xF)
    c4 = b.const_i32(4)
    for e in range(ept):
        idx = b.add(b.const_i32(e * bs), grid.tid)
        in_range = b.cmp_lt(idx, c_total)
        sidx = b.select(in_range, idx, c0)
        row = b.div(sidx, c_bpr)
        kb = b.mod(sidx, c_bpr)
        k_lo = b.mul(kb, b.const_i32(2))
        pair = b.smem_load_vN(c0_smem, row, k_lo, dtype=I8, n=2)  # <2 x i8>
        lo = b.land(b.sext(b.vec_extract(pair, 0), I32), c0xf)
        hi = b.shl(b.land(b.sext(b.vec_extract(pair, 1), I32), c0xf), c4)
        packed = b.trunc(b.lor(lo, hi), I8)
        with b.scf_if(in_range):
            b.smem_store_vN(c0_packed_smem, [row, kb], packed, n=1)


def _emit_maxpool_finalquant(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    c1_smem: Value,
    y_ptr: Value,
    grid: WarpGrid,
) -> None:
    """One thread per pooled pixel: 2x2 max over int4 codes in ``c1_smem``,
    final int4 quant, pack 24 channels into 3 i32 words, store to ``y_ptr``."""
    p = spec.problem
    out_k = p.conv1_channels
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    n_pix = spec.pool_tile_h * spec.pool_tile_w
    words = (out_k + 7) // 8

    c_ptw = b.const_i32(spec.pool_tile_w)
    c_ctw = b.const_i32(conv_tile_w)
    c_pool_wo = b.const_i32(p.pool_wo)
    c_words = b.const_i32(words)
    c_mf = b.const_f32(spec.mf)
    c_0xf = b.const_i32(0xF)
    neg_inf = b.const_f32(-3.4028234663852886e38)
    h_blk = b.block_id_z() if spec.w_fast else b.block_id_y()
    w_blk = b.block_id_y() if spec.w_fast else b.block_id_z()
    block_ph = b.mul(h_blk, b.const_i32(spec.pool_tile_h))
    block_pw = b.mul(w_blk, b.const_i32(spec.pool_tile_w))

    in_range = b.cmp_lt(grid.tid, b.const_i32(n_pix))

    def _emit_body(lane_idx: Value) -> None:
        local_pho = b.div(lane_idx, c_ptw)
        local_pwo = b.mod(lane_idx, c_ptw)
        gpho = b.add(block_ph, local_pho)
        gpwo = b.add(block_pw, local_pwo)
        pix = b.add(b.mul(gpho, c_pool_wo), gpwo)

        # 2x2 corner conv-tile rows for this pooled pixel (used by both paths).
        corners = []
        for yy in range(2):
            ch_h = b.add(b.mul(local_pho, b.const_i32(2)), b.const_i32(yy))
            for xx in range(2):
                ch_w = b.add(b.mul(local_pwo, b.const_i32(2)), b.const_i32(xx))
                corners.append(b.add(b.mul(ch_h, c_ctw), ch_w))

        # Vectorized fast path: read each corner's channels in 8-wide ds_read_b128
        # chunks (out_k=24 -> 3 chunks) instead of one ds_read_u16 per channel.
        # Requires the rounded-up channel span to fit inside the tile_n columns so
        # the trailing lanes of the last chunk stay in-bounds (values discarded).
        cw = 8
        vec_pool = (
            spec.vectorize_maxpool
            and spec.tile_n % cw == 0
            and ((out_k + cw - 1) // cw) * cw <= spec.tile_n
        )
        chmax = [neg_inf for _ in range(out_k)]
        if vec_pool:
            n_chunks = (out_k + cw - 1) // cw
            for conv_m in corners:
                for ck in range(n_chunks):
                    vecf = b.smem_load_vN_f16(
                        c1_smem, conv_m, b.const_i32(ck * cw), n=cw
                    )
                    for j in range(cw):
                        ch = ck * cw + j
                        if ch >= out_k:
                            break
                        vf = b.cast_to_f32(b.vec_extract(vecf, j))
                        chmax[ch] = b.fmax(chmax[ch], vf)
        else:
            for ch in range(out_k):
                for conv_m in corners:
                    v = b.vec_extract(
                        b.smem_load_vN_f16(c1_smem, conv_m, b.const_i32(ch), n=1), 0
                    )
                    chmax[ch] = b.fmax(chmax[ch], b.cast_to_f32(v))

        word_vals = [b.const_i32(0) for _ in range(words)]
        for ch in range(out_k):
            qf = _quant_i4(b, chmax[ch], c_mf)  # i8 holding int4 code
            nib = b.land(b.sext(qf, I32), c_0xf)
            w = ch // 8
            shift = 4 * (ch % 8)
            if shift:
                nib = b.shl(nib, b.const_i32(shift))
            word_vals[w] = b.lor(word_vals[w], nib)

        base = b.mul(pix, c_words)
        for w in range(words):
            b.global_store(y_ptr, b.add(base, b.const_i32(w)), word_vals[w], align=4)

    if spec.mask_maxpool:
        # L3: branch-free tail. n_pix (= pool_tile_h*pool_tile_w) equals the wave
        # size for the target tile, so the scf_if is already warp-uniform; this
        # path instead clamps out-of-range lanes to the last pooled pixel, which
        # they recompute and re-store with identical words (idempotent), trading
        # the structured branch for redundant compute. Measured as a lever vs
        # non-lever in the campaign writeup.
        sidx = b.select(in_range, grid.tid, b.const_i32(n_pix - 1))
        _emit_body(sidx)
    else:
        with b.scf_if(in_range):
            _emit_body(grid.tid)


def _emit_maxpool_finalpack_i8(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    c1_smem: Value,
    y_ptr: Value,
    grid: WarpGrid,
    h_blk: Value = None,
    w_blk: Value = None,
) -> None:
    """Native integer maxpool over byte int4 codes; final mf=1 pack is no-op.

    Persistent mode threads the loop-carried tile coords in via ``h_blk``/
    ``w_blk``; otherwise they fall back to the per-CTA block ids (byte-identical).
    """
    p = spec.problem
    out_k = p.conv1_channels
    conv_tile_w = spec.pool_tile_w * p.pool_stride_w
    n_pix = spec.pool_tile_h * spec.pool_tile_w
    words = (out_k + 7) // 8

    c_ptw = b.const_i32(spec.pool_tile_w)
    c_ctw = b.const_i32(conv_tile_w)
    c_pool_wo = b.const_i32(p.pool_wo)
    c_words = b.const_i32(words)
    c_0xf = b.const_i32(0xF)
    if h_blk is None:
        h_blk = b.block_id_z() if spec.w_fast else b.block_id_y()
        w_blk = b.block_id_y() if spec.w_fast else b.block_id_z()
    block_ph = b.mul(h_blk, b.const_i32(spec.pool_tile_h))
    block_pw = b.mul(w_blk, b.const_i32(spec.pool_tile_w))
    in_range = b.cmp_lt(grid.tid, b.const_i32(n_pix))

    def _emit_body(lane_idx: Value) -> None:
        local_pho = b.div(lane_idx, c_ptw)
        local_pwo = b.mod(lane_idx, c_ptw)
        gpho = b.add(block_ph, local_pho)
        gpwo = b.add(block_pw, local_pwo)
        pix = b.add(b.mul(gpho, c_pool_wo), gpwo)

        corners = []
        for yy in range(2):
            ch_h = b.add(b.mul(local_pho, b.const_i32(2)), b.const_i32(yy))
            for xx in range(2):
                ch_w = b.add(b.mul(local_pwo, b.const_i32(2)), b.const_i32(xx))
                corners.append(b.add(b.mul(ch_h, c_ctw), ch_w))

        chmax = [b.const_i32(-8) for _ in range(out_k)]
        cw = 8
        vec_pool = (
            spec.vectorize_maxpool
            and spec.tile_n % cw == 0
            and ((out_k + cw - 1) // cw) * cw <= spec.tile_n
        )
        if vec_pool and spec.pk_maxpool:
            # Packed-int16 reduction: widen each corner chunk i8 -> <cw x i16>
            # and max across the 4 corners with vector_smax so the gfx11 backend
            # selects v_pk_max_i16 (2 channels/op). Initialising the accumulator
            # from the first corner (rather than a -8 splat) keeps it bit-exact
            # to the scalar path and avoids an i16 constant. The i8->i16 cast
            # replaces the scalar path's i8->i32 sext; a single i16->i32
            # re-extract per channel feeds the unchanged nibble pack below.
            n_chunks = (out_k + cw - 1) // cw
            for ck in range(n_chunks):
                acc16 = None
                for conv_m in corners:
                    raw = b.smem_load_vN(
                        c1_smem, conv_m, b.const_i32(ck * cw), dtype=I8, n=cw
                    )
                    w16 = b.vector_sext(raw, I16)  # signed widen i8 -> i16
                    acc16 = w16 if acc16 is None else b.vector_smax(acc16, w16)
                for j in range(cw):
                    ch = ck * cw + j
                    if ch >= out_k:
                        break
                    chmax[ch] = b.sext(b.vec_extract(acc16, j), I32)
        elif vec_pool:
            n_chunks = (out_k + cw - 1) // cw
            for conv_m in corners:
                for ck in range(n_chunks):
                    vec = b.smem_load_vN(
                        c1_smem, conv_m, b.const_i32(ck * cw), dtype=I8, n=cw
                    )
                    for j in range(cw):
                        ch = ck * cw + j
                        if ch >= out_k:
                            break
                        v = b.sext(b.vec_extract(vec, j), I32)
                        chmax[ch] = b.select(b.cmp_gt(v, chmax[ch]), v, chmax[ch])
        else:
            for ch in range(out_k):
                for conv_m in corners:
                    v = b.sext(
                        b.vec_extract(
                            b.smem_load_vN(
                                c1_smem, conv_m, b.const_i32(ch), dtype=I8, n=1
                            ),
                            0,
                        ),
                        I32,
                    )
                    chmax[ch] = b.select(b.cmp_gt(v, chmax[ch]), v, chmax[ch])

        word_vals = [b.const_i32(0) for _ in range(words)]
        for ch in range(out_k):
            # mf is 1.0 in the target pipeline. Values are already ReLUed int4
            # codes, so final QuantizeLinear is just nibble packing.
            nib = b.land(chmax[ch], c_0xf)
            w = ch // 8
            shift = 4 * (ch % 8)
            if shift:
                nib = b.shl(nib, b.const_i32(shift))
            word_vals[w] = b.lor(word_vals[w], nib)

        base = b.mul(pix, c_words)
        for w in range(words):
            b.global_store(y_ptr, b.add(base, b.const_i32(w)), word_vals[w], align=4)

    if spec.mask_maxpool:
        sidx = b.select(in_range, grid.tid, b.const_i32(n_pix - 1))
        _emit_body(sidx)
    else:
        with b.scf_if(in_range):
            _emit_body(grid.tid)


def build_deep_fused_conv_pool(
    spec: Gfx1151DeepFusedConvPoolSpec, arch: str = "gfx1151"
):
    """Build the gfx1151 genuine-int8/int4 fused conv0->conv1->maxpool kernel."""
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid gfx1151 deep fused conv/pool spec: {why}")

    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    op = target.mma.op_for_shape(
        family="wmma",
        a_dtype="f16",
        b_dtype="f16",
        c_dtype="fp32",
        m=_WMMA,
        n=_WMMA,
        k=_WMMA,
    )
    # Native integer atoms: conv0 uses iu8, conv1 uses iu4. The fp16 atom remains
    # the non-native path and the shape source for the common warp grid.
    op0 = target.mma.by_op_id(_OP_ID_IU8) if spec.native_int else op
    op1 = target.mma.by_op_id(_OP_ID_IU4) if spec.native_int else op

    p = spec.problem
    c = p.conv
    kpad = spec.kpad

    b = IRBuilder(spec.kernel_name())
    X = b.param("X", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    W0 = b.param("W0", PtrType(I8, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(I32, "global"), noalias=True, writeonly=True, align=16)
    W1 = b.param("W1", PtrType(I8, "global"), noalias=True, readonly=True, align=16)

    grid = WarpGrid.from_atom(
        op,
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=_WMMA,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        wave_size=_WAVE,
    ).bind(b, block_m_axis="y", block_n_axis="x")

    # L1: occupancy launch bound. Setting waves-per-EU forces the compiler to
    # cap VGPRs so a second workgroup can co-reside per CU (direct-conv0 freed
    # the LDS to admit it). max_workgroup_size pins the block size so the bound
    # is interpreted against the real launch shape.
    if spec.waves_per_eu:
        b.kernel.attrs["max_workgroup_size"] = spec.block_size
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    # L2: scheduler hints. SchedulePolicy emits sched_group_barrier DS_READ/MFMA
    # interleave around each WMMA k-atom so the matrix pipe stays fed across
    # operand-delivery latency (the binding constraint). "mem" => no-op.
    policy = SchedulePolicy.for_pipeline(spec.sched_policy)

    # Native-int conv0 stages raw int8 into i8 LDS (loaded as <4 x i32> frags);
    # the fp16 path stages f16 codes. is_valid_spec forces im2col for native_int.
    a0_dtype = I8 if spec.native_int else F16
    if spec.direct_conv0:
        # Small input-halo footprint cache (each pixel once, no R*S redundancy).
        a0_smem = b.smem_alloc(
            a0_dtype, [spec.foot_h * spec.foot_w, c.C], name_hint="INP_smem"
        )
    else:
        a0_smem = b.smem_alloc(a0_dtype, [spec.tile_m, kpad], name_hint="A0_smem")
    w0_smem = b.smem_alloc(a0_dtype, [spec.tile_n, kpad], name_hint="W0_smem")
    c0_dtype = I8 if spec.native_int else F16
    w1_dtype = I8 if spec.native_int else F16
    c0_cols = c.K // 2 if spec.packed_c0_handoff else c.K
    # fused_c0a1 hands conv0->conv1 off entirely in registers (permlanex16
    # transpose), so the C0 LDS tile -- and its scatter + barrier -- are gone.
    c0_smem = (
        None
        if spec.fused_c0a1
        else b.smem_alloc(c0_dtype, [spec.tile_m, c0_cols], name_hint="C0_smem")
    )
    # Lever 2: extra packed-byte C0 buffer (2 codes/byte) produced by a lane-local
    # LDS->LDS repack so conv1 loads A as a bitcast instead of packing on load.
    c0_packed_smem = (
        b.smem_alloc(I8, [spec.tile_m, c.K // 2], name_hint="C0pk_smem")
        if spec.repack_c0
        else None
    )
    # conv1_int8 stores W1 byte-per-code (full K columns) for the iu8 atom; the
    # default packed-int4 path stores 2 codes/byte (K/2 columns).
    if spec.native_int:
        w1_cols = c.K if spec.conv1_int8 else c.K // 2
    else:
        w1_cols = c.K
    w1_smem = b.smem_alloc(w1_dtype, [spec.tile_n, w1_cols], name_hint="W1_smem")
    c1_dtype = I8 if spec.native_int else F16
    c1_smem = b.smem_alloc(c1_dtype, [spec.tile_m, spec.tile_n], name_hint="C1_smem")

    if spec.persistent:
        # Persistent grid-stride variant. is_valid_spec guarantees native_int +
        # direct_conv0 + fused_c0a1, so W0/W1 are tile-invariant and there is no
        # per-tile c0_smem: stage both weights into LDS ONCE before the loop, then
        # grid-stride over the flattened tile strip streaming only X (footprint)
        # and Y (output) per tile.
        def conv0_code_i8(p0: Value) -> Value:
            q0 = _quant_i8_shift(b, p0, 4)  # m0 = 1/16
            q0r = _relu_i32(b, b.sext(q0, I32))
            return _quant_i4_shift(b, q0r, 1)  # m0b = 1/2

        def conv1_code_i8(p1: Value) -> Value:
            q1 = _quant_i4_shift(b, p1, 2)  # m1 = 1/4
            return b.trunc(_relu_i32(b, b.sext(q1, I32)), I8)

        # Weights resident: staged once per CTA (vs once per tile), one barrier.
        _stage_conv0_w0_int(b, spec, W0, w0_smem, grid)
        if spec.conv1_int8:
            _stage_conv1_w1_i8(b, spec, W1, w1_smem, grid)
        else:
            _stage_conv1_w1_packed(b, spec, W1, w1_smem, grid)
        b.sync()

        h_tiles = p.pool_ho // spec.pool_tile_h
        w_tiles = p.pool_wo // spec.pool_tile_w
        num_tiles = h_tiles * w_tiles
        c_wtiles = b.const_i32(w_tiles)
        loop = b.scf_for(
            b.block_id_x(),
            b.const_i32(num_tiles),
            b.const_i32(spec.persistent_ctas),
            iv_name="tile_idx",
        )
        with loop as tile_idx:
            # Scalarize the tile index + coords (uniform across the wave -> SGPR,
            # no per-lane VGPR address math), mirroring CK Tile's
            # amd_wave_read_first_lane(block_id) persistent pattern.
            ti = b.readfirstlane(tile_idx)
            h_blk = b.div(ti, c_wtiles)
            w_blk = b.mod(ti, c_wtiles)

            # Inter-tile guard: the previous tile's maxpool reads c1_smem and its
            # conv0 GEMM read a0_smem; this barrier orders those completions before
            # this tile reuses a0_smem/c1_smem. (On the first iteration it pairs
            # with the pre-loop weight-staging barrier.)
            b.sync()
            _stage_input_footprint_int(
                b, spec, X, a0_smem, grid, h_blk=h_blk, w_blk=w_blk
            )
            b.sync()  # footprint store -> conv0 GEMM read of a0_smem
            accs0 = _wmma_gemm_conv0_direct_int(
                b,
                spec,
                op0,
                a0_smem,
                w0_smem,
                grid,
                policy=policy,
                reorient=True,
            )
            # conv1: register handoff (no scatter/c0_smem/barrier). W1 is already
            # resident and never rewritten, so the pre-conv1 barrier is dropped.
            if spec.conv1_int8:
                a_frags = _fuse_c0_to_conv1_a_regs(
                    b, op0, accs0, grid, conv0_code_i8, int8=True
                )
                accs1 = _wmma_gemm_conv1_i8_from_regs(
                    b,
                    op0,
                    a_frags,
                    w1_smem,
                    grid,
                    c.K,
                    policy=policy,
                    prefetch_k=spec.conv1_prefetch_k,
                    sched_fuse=spec.conv1_sched_fuse,
                )
            else:
                a_frags = _fuse_c0_to_conv1_a_regs(b, op0, accs0, grid, conv0_code_i8)
                accs1 = _wmma_gemm_conv1_i4_from_regs(
                    b,
                    op1,
                    a_frags,
                    w1_smem,
                    grid,
                    c.K,
                    policy=policy,
                    prefetch_k=spec.conv1_prefetch_k,
                    sched_fuse=spec.conv1_sched_fuse,
                )
            _scatter_codes_to_i8_lds(b, op1, accs1, c1_smem, grid, conv1_code_i8)
            b.sync()  # conv1 scatter -> maxpool read of c1_smem
            _emit_maxpool_finalpack_i8(
                b, spec, c1_smem, Y, grid, h_blk=h_blk, w_blk=w_blk
            )

        return b.kernel

    # ---- conv0: int8 -> WMMA -> Quant(i32->i8)->ReLU->Quant(i8->i4)
    # native_int: raw int8 -> i8 LDS -> native iu8 WMMA -> exact i32 acc.
    # fp16 path: int8 -> f16 LDS -> fp16 WMMA -> f32 acc (rint-snapped).
    if spec.native_int and spec.direct_conv0:
        _stage_input_footprint_int(b, spec, X, a0_smem, grid)
        _stage_conv0_w0_int(b, spec, W0, w0_smem, grid)
        b.sync()
        accs0 = _wmma_gemm_conv0_direct_int(
            b,
            spec,
            op0,
            a0_smem,
            w0_smem,
            grid,
            policy=policy,
            reorient=spec.fused_c0a1,
        )
    elif spec.native_int:
        _stage_conv0_a_int(b, spec, X, a0_smem, grid)
        _stage_conv0_w0_int(b, spec, W0, w0_smem, grid)
        b.sync()
        accs0 = _wmma_gemm_from_lds_int(
            b, op0, a0_smem, w0_smem, grid, kpad, policy=policy
        )
    elif spec.direct_conv0:
        _stage_input_footprint(b, spec, X, a0_smem, grid)
        _stage_conv0_w0(b, spec, W0, w0_smem, grid)
        b.sync()
        accs0 = _wmma_gemm_conv0_direct(
            b, spec, op, a0_smem, w0_smem, grid, policy=policy
        )
    else:
        _stage_conv0_a(b, spec, X, a0_smem, grid)
        _stage_conv0_w0(b, spec, W0, w0_smem, grid)
        b.sync()
        accs0 = _wmma_gemm_from_lds(b, op, a0_smem, w0_smem, grid, kpad, policy=policy)

    c_m0 = b.const_f32(spec.m0)
    c_m0b = b.const_f32(spec.m0b)
    zero_f = b.const_f32(0.0)

    def conv0_code_i8(p0: Value) -> Value:
        if spec.native_int:
            q0 = _quant_i8_shift(b, p0, 4)  # m0 = 1/16
            q0r = _relu_i32(b, b.sext(q0, I32))
            return _quant_i4_shift(b, q0r, 1)  # m0b = 1/2
        else:
            # f16 WMMA carries ~7.6e-6 sub-ULP noise even for exact-integer
            # accumulation; the true value is a known exact int32, so snap it
            # before quant to keep round-half-even ties bit-exact to native MMA.
            p0_f32 = b.rint_f32(p0)
            q0 = _quant_i8(b, p0_f32, c_m0)
            q0r = b.fmax(_i8_to_f32(b, q0), zero_f)  # ReLU
            return _quant_i4(b, q0r, c_m0b)

    def conv0_code_f16(p0: Value) -> Value:
        return b.trunc_f32_to_f16(_i8_to_f32(b, conv0_code_i8(p0)))

    def conv0_code_vec_i8(acc: Value) -> Value:
        q0 = _quant_i8_shift_vec_i32(b, acc, 4)  # m0 = 1/16
        q0r = _relu_i32_vec(b, q0)
        q0b = _quant_i4_shift_vec_i32(b, q0r, 1)  # m0b = 1/2
        return b.vector_trunc(q0b, I8)

    # ---- conv1: 1x1 int4 -> int32/fp32 -> Quant(i32->i4)->ReLU
    # accs0 are now in registers; c0_smem / w1_smem are distinct from the conv0
    # operand tiles, so no barrier is needed before producing them. With early_w1
    # the W1 HBM loads are issued before the conv0 epilogue scatter so their
    # latency overlaps the scatter's VALU/LDS work; a single barrier then gates
    # conv1 on both producers.
    if spec.native_int and spec.fused_c0a1:
        # In-register handoff: no scatter, no c0_smem, no handoff barrier. Stage
        # W1 (its global loads overlap the register transpose), build the conv1
        # A-fragments in registers, then ONE barrier gates conv1 on W1 only.
        if spec.conv1_int8:
            # int8 conv1: byte-per-code W1 + unsqueezed <4 x i32> A-frags + iu8
            # atom. Codes stay int4-range, so this is bit-identical to the iu4
            # path -- it only deletes the per-handoff nibble squeeze.
            _stage_conv1_w1_i8(b, spec, W1, w1_smem, grid)
            a_frags = _fuse_c0_to_conv1_a_regs(
                b, op0, accs0, grid, conv0_code_i8, int8=True
            )
            b.sync()
            accs1 = _wmma_gemm_conv1_i8_from_regs(
                b,
                op0,
                a_frags,
                w1_smem,
                grid,
                c.K,
                policy=policy,
                prefetch_k=spec.conv1_prefetch_k,
                sched_fuse=spec.conv1_sched_fuse,
            )
        else:
            _stage_conv1_w1_packed(b, spec, W1, w1_smem, grid)
            a_frags = _fuse_c0_to_conv1_a_regs(b, op0, accs0, grid, conv0_code_i8)
            b.sync()
            accs1 = _wmma_gemm_conv1_i4_from_regs(
                b,
                op1,
                a_frags,
                w1_smem,
                grid,
                c.K,
                policy=policy,
                prefetch_k=spec.conv1_prefetch_k,
                sched_fuse=spec.conv1_sched_fuse,
            )
    elif spec.native_int:
        if spec.early_w1:
            _stage_conv1_w1_packed(b, spec, W1, w1_smem, grid)
            if spec.packed_c0_handoff:
                _scatter_packed_i4_codes_to_lds(
                    b, op0, accs0, c0_smem, grid, conv0_code_i8
                )
            elif spec.specialized_rne:
                _scatter_vec_codes_to_i8_lds(
                    b, op0, accs0, c0_smem, grid, conv0_code_vec_i8
                )
            else:
                _scatter_codes_to_i8_lds(b, op0, accs0, c0_smem, grid, conv0_code_i8)
            b.sync()
        else:
            b.sync()
            if spec.packed_c0_handoff:
                _scatter_packed_i4_codes_to_lds(
                    b, op0, accs0, c0_smem, grid, conv0_code_i8
                )
            elif spec.specialized_rne:
                _scatter_vec_codes_to_i8_lds(
                    b, op0, accs0, c0_smem, grid, conv0_code_vec_i8
                )
            else:
                _scatter_codes_to_i8_lds(b, op0, accs0, c0_smem, grid, conv0_code_i8)
            _stage_conv1_w1_packed(b, spec, W1, w1_smem, grid)
            b.sync()
        if spec.repack_c0:
            _repack_c0_lds_to_packed(b, spec, c0_smem, c0_packed_smem, grid)
            b.sync()
            accs1 = _wmma_gemm_conv1_i4_packed_from_lds(
                b, op1, c0_packed_smem, w1_smem, grid, c.K, policy=policy
            )
        elif spec.packed_c0_handoff:
            accs1 = _wmma_gemm_conv1_i4_packed_from_lds(
                b, op1, c0_smem, w1_smem, grid, c.K, policy=policy
            )
        else:
            accs1 = _wmma_gemm_conv1_i4_from_lds(
                b,
                op1,
                c0_smem,
                w1_smem,
                grid,
                c.K,
                policy=policy,
                prefetch_k=spec.conv1_prefetch_k,
                sched_fuse=spec.conv1_sched_fuse,
            )
    else:
        if spec.early_w1:
            _stage_conv1_w1(b, spec, W1, w1_smem, grid)
            _scatter_codes_to_lds(b, op0, accs0, c0_smem, grid, conv0_code_f16)
            b.sync()
        else:
            b.sync()
            _scatter_codes_to_lds(b, op0, accs0, c0_smem, grid, conv0_code_f16)
            _stage_conv1_w1(b, spec, W1, w1_smem, grid)
            b.sync()
        accs1 = _wmma_gemm_from_lds(b, op, c0_smem, w1_smem, grid, c.K, policy=policy)

    c_m1 = b.const_f32(spec.m1)

    def conv1_code_i8(p1: Value) -> Value:
        if spec.native_int:
            q1 = _quant_i4_shift(b, p1, 2)  # m1 = 1/4
            return b.trunc(_relu_i32(b, b.sext(q1, I32)), I8)
        p1_f32 = b.rint_f32(p1)
        q1 = _quant_i4(b, p1_f32, c_m1)
        q1r = b.fmax(_i8_to_f32(b, q1), zero_f)  # ReLU
        return b.cvt_f32_to_i8_sat(q1r)

    def conv1_code_f16(p1: Value) -> Value:
        return b.trunc_f32_to_f16(_i8_to_f32(b, conv1_code_i8(p1)))

    def conv1_code_vec_i8(acc: Value) -> Value:
        q1 = _quant_i4_shift_vec_i32(b, acc, 2)  # m1 = 1/4
        q1r = _relu_i32_vec(b, q1)
        return b.vector_trunc(q1r, I8)

    if spec.native_int:
        if spec.specialized_rne:
            _scatter_vec_codes_to_i8_lds(
                b, op1, accs1, c1_smem, grid, conv1_code_vec_i8
            )
        else:
            _scatter_codes_to_i8_lds(b, op1, accs1, c1_smem, grid, conv1_code_i8)
    else:
        _scatter_codes_to_lds(b, op, accs1, c1_smem, grid, conv1_code_f16)
    b.sync()

    # ---- maxpool 2x2/s2 -> Quant(i4->i4) -> packed int4 output
    if spec.native_int:
        _emit_maxpool_finalpack_i8(b, spec, c1_smem, Y, grid)
    else:
        _emit_maxpool_finalquant(b, spec, c1_smem, Y, grid)

    return b.kernel


def _q_int_codes(np, scaled_f32, lo: float, hi: float):
    """Clamp then round-to-nearest-even, matching the integer fusion kernels."""
    return np.rint(np.clip(scaled_f32, lo, hi))


def _pack_i4_rows(np, codes):
    """Signed int4 codes ``(rows, cols)`` -> two codes per byte."""
    spec = MatMulNBitsSpec(
        name="deep_fused_w1_pack",
        N=int(codes.shape[0]),
        K=int(codes.shape[1]),
        tile=TileSpec(
            tile_m=16,
            tile_n=16,
            tile_k=16,
            warp_m=1,
            warp_n=1,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
    )
    return pack_i4_weights_for_matmul_nbits(codes, spec).view(np.int8)


def _unpack_deep_fused_y(np, words, channels: int):
    """``(pool_h, pool_w, words)`` int32 output -> signed int4 channel codes."""
    ph, pw, _ = words.shape
    out = np.empty((ph, pw, channels), dtype=np.int32)
    for ch in range(channels):
        word = words[:, :, ch // 8].astype(np.uint32)
        nib = (word >> (4 * (ch % 8))) & 0xF
        signed = nib.astype(np.int32)
        out[:, :, ch] = np.where(signed >= 8, signed - 16, signed)
    return out


def _deep_fused_i8i4_reference(np, X, W0, W1_codes, manifest: dict):
    """Integer-exact reference for gfx1151 deep fused conv0->conv1->pool."""
    N, H, W, C, K0, R, S, sH, sW, pH, pW, dH, dW = [
        int(x) for x in manifest["conv"][:13]
    ]
    K1 = int(manifest["conv1"]["K1"])
    pool_y, pool_x, pool_s_h, pool_s_w = [int(x) for x in manifest["pool"]]
    quant = manifest.get("quant", {})
    m0 = np.float32(quant.get("m0", 0.0625))
    m0b = np.float32(quant.get("m0b", 0.5))
    m1 = np.float32(quant.get("m1", 0.25))
    mf = np.float32(quant.get("mf", 1.0))
    Ho = (H + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (W + 2 * pW - dW * (S - 1) - 1) // sW + 1
    pool_ho = (Ho - pool_y) // pool_s_h + 1
    pool_wo = (Wo - pool_x) // pool_s_w + 1

    Xp = np.pad(
        X.astype(np.int64),
        ((0, 0), (pH, pH), (pW, pW), (0, 0)),
    )
    P0 = np.zeros((N, Ho, Wo, K0), dtype=np.int64)
    for r in range(R):
        for s in range(S):
            x = Xp[
                :,
                r * dH : r * dH + Ho * sH : sH,
                s * dW : s * dW + Wo * sW : sW,
                :,
            ]
            w = W0[:, r, s, :].astype(np.int64)
            P0 += np.einsum("nhwc,kc->nhwk", x, w, optimize=True)

    q0 = _q_int_codes(np, P0.astype(np.float32) * m0, -127.0, 127.0)
    q0_relu = np.maximum(q0, 0.0)
    C0 = _q_int_codes(np, q0_relu * m0b, -8.0, 7.0)

    P1 = np.einsum("nhwk,ok->nhwo", C0, W1_codes.astype(np.float32), optimize=True)
    q1 = _q_int_codes(np, P1 * m1, -8.0, 7.0)
    C1 = np.maximum(q1, 0.0)

    ref = np.empty((pool_ho, pool_wo, K1), dtype=np.int32)
    for ho in range(pool_ho):
        for wo in range(pool_wo):
            h0 = ho * pool_s_h
            w0 = wo * pool_s_w
            patch = C1[0, h0 : h0 + pool_y, w0 : w0 + pool_x, :]
            pooled = patch.max(axis=(0, 1)).astype(np.float32)
            ref[ho, wo, :] = _q_int_codes(np, pooled * mf, -8.0, 7.0).astype(np.int32)
    return ref


def run_deep_fused_conv_pool_i8i4_manifest_problem(
    manifest: dict, shape: Tuple[int, int, int] | None, verify: bool
) -> tuple:
    """gfx1151 deep-fused int8/int4 conv+pool manifest-runner problem."""
    if shape is not None:
        raise ValueError("deep_fused_conv_pool_i8i4 uses manifest shape, not --shape")
    np = require_numpy()
    N, H, W, C, K0, R, S, sH, sW, pH, pW, dH, dW = [
        int(x) for x in manifest["conv"][:13]
    ]
    K1 = int(manifest["conv1"]["K1"])
    pool_y, pool_x, pool_s_h, pool_s_w = [int(x) for x in manifest["pool"]]
    Ho = (H + 2 * pH - dH * (R - 1) - 1) // sH + 1
    Wo = (W + 2 * pW - dW * (S - 1) - 1) // sW + 1
    pool_ho = (Ho - pool_y) // pool_s_h + 1
    pool_wo = (Wo - pool_x) // pool_s_w + 1

    rng = np.random.default_rng(int(manifest.get("seed", 123)))
    X = rng.integers(-3, 4, size=(N, H, W, C), dtype=np.int8)
    W0 = rng.integers(-3, 4, size=(K0, R, S, C), dtype=np.int8)
    W1_codes = rng.integers(-3, 4, size=(K1, K0), dtype=np.int8)
    W1 = _pack_i4_rows(np, W1_codes)
    Y = np.zeros((pool_ho, pool_wo, (K1 + 7) // 8), dtype=np.int32)

    if "grid_explicit" in manifest:
        grid = tuple(int(x) for x in manifest["grid_explicit"])
    else:
        pool_tile_h, pool_tile_w = [int(x) for x in manifest["pool_tile"]]
        grid = (1, pool_ho // pool_tile_h, pool_wo // pool_tile_w)
    block = (int(manifest["threads_per_block"]), 1, 1)
    flop = 2.0 * (N * Ho * Wo * K0 * R * S * C + N * Ho * Wo * K1 * K0)
    bytes_xfer = float(X.nbytes + W0.nbytes + W1.nbytes + Y.nbytes)
    tol = int(manifest.get("verify_tol", 0))

    def make_args(rt: Runtime):
        X_dev = rt.alloc(nbytes(X))
        W0_dev = rt.alloc(nbytes(W0))
        Y_dev = rt.alloc(nbytes(Y))
        W1_dev = rt.alloc(nbytes(W1))
        rt.memcpy_h2d(X_dev, as_u8_buffer(X), nbytes(X))
        rt.memcpy_h2d(W0_dev, as_u8_buffer(W0), nbytes(W0))
        rt.memcpy_h2d(W1_dev, as_u8_buffer(W1), nbytes(W1))
        rt.memset(Y_dev, 0, nbytes(Y))
        return struct.pack("<QQQQ", X_dev, W0_dev, Y_dev, W1_dev), (
            X_dev,
            W0_dev,
            Y_dev,
            W1_dev,
        )

    def check(rt: Runtime, ptrs):
        if not verify:
            return 0.0, 0, pool_ho * pool_wo * K1
        rt.memcpy_d2h(as_u8_buffer(Y), ptrs[2], nbytes(Y))
        got = _unpack_deep_fused_y(np, Y, K1)
        ref = _deep_fused_i8i4_reference(np, X, W0, W1_codes, manifest)
        ref_f = ref.astype(np.float64)
        err = np.abs(got.astype(np.float64) - ref_f)
        bad = err > tol + tol * np.abs(ref_f)
        return (
            float(err.max()) if err.size else 0.0,
            int(np.count_nonzero(bad)),
            got.size,
        )

    return make_args, grid, block, flop, bytes_xfer, check
