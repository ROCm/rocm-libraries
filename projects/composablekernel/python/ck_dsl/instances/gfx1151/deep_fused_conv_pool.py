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

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ...core.ir import F16, I8, I32, IRBuilder, PtrType, Value, VectorType
from ...helpers.geometry import WarpGrid
from ...helpers.schedule import DS_READ, MFMA, SchedulePolicy
from ...helpers.spec import kernel_name_join
from ..common.conv_implicit_gemm import ConvProblem, _emit_frag_smem_load
from ..gfx950.deep_fused_conv_pool import FusedConvPoolProblem

_WMMA = 16
_WAVE = 32
# Native integer WMMA atom for the conv0 iu8 path (Phase C). int8 A/B fragments
# are <4 x i32> (16 K-bytes packed 4-per-i32); accumulator/result <8 x i32>.
_OP_ID_IU8 = "wmma_i32_16x16x16_iu8"
_K_PER_I32 = 4  # int8 K-values packed per i32 fragment slot


@dataclass(frozen=True)
class Gfx1151DeepFusedConvPoolSpec:
    """One concrete gfx1151 genuine-int8/int4 deep-fusion configuration."""

    problem: FusedConvPoolProblem
    name: str = "ck_dsl_gfx1151_deep_fused_conv_pool"
    tile_m: int = 128
    tile_n: int = 32
    pool_tile_h: int = 4
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
        return (self.conv_tile_h - 1) * c.sH + (c.R - 1) * c.dH + 1

    @property
    def foot_w(self) -> int:
        c = self.problem.conv
        return (self.conv_tile_w - 1) * c.sW + (c.S - 1) * c.dW + 1

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
        if self.butterfly_conv01:
            parts.append("butterfly")
        if self.native_int:
            parts.append("nativeiu8")
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
    pool_tile_h: int = 4,
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
    butterfly_conv01: bool = False,
    native_int: bool = False,
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
        R=r,
        S=s,
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
        butterfly_conv01=butterfly_conv01,
        native_int=native_int,
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
        if spec.direct_conv0:
            return False, (
                "native_int conv0 currently supports only the im2col A path "
                "(direct_conv0=False); the footprint-gather path is not ported"
            )
        if spec.butterfly_conv01:
            return False, "native_int is incompatible with butterfly_conv01"
    return True, "ok"


def deep_fused_conv_pool_grid(
    spec: Gfx1151DeepFusedConvPoolSpec,
) -> Tuple[int, int, int]:
    p = spec.problem
    h_tiles = p.pool_ho // spec.pool_tile_h
    w_tiles = p.pool_wo // spec.pool_tile_w
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
            r = b.div(g, b.const_i32(c.S))
            s = b.mod(g, b.const_i32(c.S))
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
    c_sc = b.const_i32(c.S * c.C)
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


def _stage_conv0_w0(
    b: IRBuilder,
    spec: Gfx1151DeepFusedConvPoolSpec,
    w0_ptr: Value,
    w0_smem: Value,
    grid: WarpGrid,
) -> None:
    """Load int8 conv0 weights ``W0[K0, K_gemm]`` (KRSC contiguous) into
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
    Out-of-image halo and K padding -> 0. Scalar element-wise i8 stores (one
    thread-element per (row, kg) slot); the GEMM later loads 16 contiguous K
    bytes per row and bitcasts to the ``<4 x i32>`` WMMA A fragment."""
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

    total = spec.tile_m * kpad
    ept = (total + bs - 1) // bs
    c_kpad = b.const_i32(kpad)
    c_kg = b.const_i32(c.K_gemm)
    c_sc = b.const_i32(c.S * c.C)
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
    """Native-int conv0: load int8 conv0 weights ``W0[K0, K_gemm]`` (KRSC
    contiguous) into the *i8* LDS tile ``w0_smem[tile_n, kpad]`` as raw int8
    codes; padding rows/cols -> 0. Scalar element-wise i8 stores."""
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
    c_sc = b.const_i32(c.S * c.C)
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
    # conv0 atom: native iu8 (int8->i32, exact) when native_int, else the fp16
    # atom shared with conv1. conv1 always stays fp16 in this phase.
    op0 = target.mma.by_op_id(_OP_ID_IU8) if spec.native_int else op

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
            F16, [spec.foot_h * spec.foot_w, c.C], name_hint="INP_smem"
        )
    else:
        a0_smem = b.smem_alloc(a0_dtype, [spec.tile_m, kpad], name_hint="A0_smem")
    w0_smem = b.smem_alloc(a0_dtype, [spec.tile_n, kpad], name_hint="W0_smem")
    c0_smem = b.smem_alloc(F16, [spec.tile_m, spec.tile_n], name_hint="C0_smem")
    w1_smem = b.smem_alloc(F16, [spec.tile_n, c.K], name_hint="W1_smem")
    c1_smem = b.smem_alloc(F16, [spec.tile_m, spec.tile_n], name_hint="C1_smem")

    # ---- conv0: int8 -> WMMA -> Quant(i32->i8)->ReLU->Quant(i8->i4)
    # native_int: raw int8 -> i8 LDS -> native iu8 WMMA -> exact i32 acc.
    # fp16 path: int8 -> f16 LDS -> fp16 WMMA -> f32 acc (rint-snapped).
    if spec.native_int:
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

    def conv0_code(p0: Value) -> Value:
        if spec.native_int:
            # Native iu8 acc is an exact int32; convert straight to f32 (no rint
            # noise to snap). The quant chain then produces the f16 conv1 code.
            p0_f32 = b.sitofp_f32(p0)
        else:
            # f16 WMMA carries ~7.6e-6 sub-ULP noise even for exact-integer
            # accumulation; the true value is a known exact int32, so snap it
            # before quant to keep round-half-even ties bit-exact to native MMA.
            p0_f32 = b.rint_f32(p0)
        q0 = _quant_i8(b, p0_f32, c_m0)
        q0r = b.fmax(_i8_to_f32(b, q0), zero_f)  # ReLU
        q0b = _quant_i4(b, q0r, c_m0b)
        return b.trunc_f32_to_f16(_i8_to_f32(b, q0b))

    # ---- conv1: 1x1 int4 -> fp16 -> WMMA -> Quant(i32->i4)->ReLU
    # accs0 are now in registers; c0_smem / w1_smem are distinct from the conv0
    # operand tiles, so no barrier is needed before producing them. With early_w1
    # the W1 HBM loads are issued before the conv0 epilogue scatter so their
    # latency overlaps the scatter's VALU/LDS work; a single barrier then gates
    # conv1 on both producers.
    if spec.early_w1:
        _stage_conv1_w1(b, spec, W1, w1_smem, grid)
        _scatter_codes_to_lds(b, op0, accs0, c0_smem, grid, conv0_code)
        b.sync()
    else:
        b.sync()
        _scatter_codes_to_lds(b, op0, accs0, c0_smem, grid, conv0_code)
        _stage_conv1_w1(b, spec, W1, w1_smem, grid)
        b.sync()
    accs1 = _wmma_gemm_from_lds(b, op, c0_smem, w1_smem, grid, c.K, policy=policy)

    c_m1 = b.const_f32(spec.m1)

    def conv1_code(p1_f32: Value) -> Value:
        p1_f32 = b.rint_f32(p1_f32)
        q1 = _quant_i4(b, p1_f32, c_m1)
        q1r = b.fmax(_i8_to_f32(b, q1), zero_f)  # ReLU
        return b.trunc_f32_to_f16(q1r)

    _scatter_codes_to_lds(b, op, accs1, c1_smem, grid, conv1_code)
    b.sync()

    # ---- maxpool 2x2/s2 -> Quant(i4->i4) -> packed int4 output
    _emit_maxpool_finalquant(b, spec, c1_smem, Y, grid)

    return b.kernel
