# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Transposed-QK WMMA FMHA-forward kernel for gfx1151 (CK gfx11 `qr_ks_vs` design).

PRODUCTION kernel (the campaign winner for dense D128 fp16 on gfx1151). There is
one config type, :class:`SwapQKCfg`: its defaults are the swept +
hardware-validated winner (wave2, pingpong ``s_setprio`` scheduling,
buffer-descriptor D16 V-gather, dual-subtile gather, the transposed V layout, the
d-outer QK loop, lazy online-softmax rescale, and fast (raw) exp2), so
``SwapQKCfg(head_size=..., num_query_heads=...)`` is the production
configuration and every other field is an opt-in lever for A/B work. Note that
the transposed V layout means callers pass V as ``[B, H, D, S]`` -- see
:func:`swapqk_transpose_v`.

Use :func:`is_valid_spec` as the cheap static gate and
:func:`build_wmma_fmha_swapqk` / :func:`swapqk_grid` to build and launch. Each
field's docstring records what it measured, including the dead-ends, so the
negative results stay attached to the code they describe rather than living only
in a design doc. See ``ALGORITHM.md`` / ``README.md``.

This is the structural change the register-transpose investigation pointed at. The
gather winner (``fmha_multiwave``) computes ``S = Q*K^T`` (query on the accumulator
slots, kv on the lane), which forces (a) a cross-lane 16-lane butterfly softmax and
(b) an LDS round-trip P-transpose every K-tile (the fixed WMMA ``a_map`` needs a full
16-lane gather that ``permlanex16`` cannot do -- the documented ``p_xpose="shuffle"``
dead-end).

Computing the scores **transposed** ``S^T = K*Q^T`` flips both:

  * **query lands on the lane** (``col = lane%16``), kv on the 8 accumulator slots
    (+ the ``lane^16`` half). So the online softmax reduction over kv is an **in-lane
    reduce over 8 slots + ONE ``permlanex16`` cross-half exchange** -- no 16-lane
    butterfly. The running ``m``/``l`` are scalars per lane (one query per lane).
  * the C->operand P-transpose becomes CK's exact ``PermuteWarpGemmCToA``: one
    ``permlanex16`` + two ``v_perm_b32`` per u32, **NO LDS round-trip, no barrier**.
    It works here (unlike on ``S=Q*K^T``) precisely because query is already on the
    lane -- only kv needs the ``lane^16`` reshuffle.

PV is computed **transposed too**: ``O^T = V*P`` (V is the A operand, P the B
operand). That keeps the PV output ``O^T[d, query]`` with query on the lane -- the
SAME distribution as the softmax stats -- so the online rescale ``O *= alpha`` is a
trivial in-lane vector-mul (no cross-lane alpha redistribution). The cost is a
transposed O store in the epilogue (strided in d), paid once per q-block.

V is still the cache-resident column **gather** (gfx1151 has no ``ds_read_tr``); the
transpose here is on P (registers), not V. Ref: CK
``ck_tile/ops/gemm/warp/warp_wmma_gemm_gfx11_utils.hpp::PermuteWarpGemmCToA`` and
``block_fmha_pipeline_qr_ks_vs.hpp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rocke.core.ir import (
    F16,
    F32,
    I16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    VectorType,
)
from rocke.helpers import (
    WmmaAtom,
    WmmaTensor,
    load_wmma_tile,
    make_global_view,
    make_lds_view,
    make_tile_window,
    store_wmma_tile,
    wmma_mma,
)
from rocke.helpers.attention import apply_attention_mask

__all__ = [
    "SwapQKCfg",
    "is_valid_spec",
    "build_wmma_fmha_swapqk",
    "swapqk_grid",
    "swapqk_transpose_v",
    "swapqk_num_work_items",
    "swapqk_persistent_grid",
]

_WMMA_OP_ID = "wmma_f32_16x16x16_f16"
_BLOCK_K = 16
# Lazy-rescale re-anchor threshold in the log2 domain: skip the O/l rescale when
# every lane's (tile_max - m_i) <= this. exp2(8)=256 keeps P in fp32 range.
_LAZY_RESCALE_THRESHOLD = 8.0


@dataclass(frozen=True)
class SwapQKCfg:
    """Configuration for the transposed-QK WMMA FMHA forward.

    The defaults ARE the production configuration: the swept, hardware-validated
    winner for dense D128 fp16 on gfx1151. ``SwapQKCfg(head_size=128,
    num_query_heads=24)`` builds that kernel (199 VGPR, zero spill, zero
    scratch); every remaining field is an opt-in lever kept for A/B work, with
    what it measured recorded in place -- including the dead-ends, so the
    negative results stay attached to the code they describe.

    NOTE the ``v_transposed=True`` default: callers must supply V as [B, H, D, S]
    rather than [B, S, H, D] -- see :func:`swapqk_transpose_v`. That is the
    layout the winning config needs (it collapses the PV V-gather from 16 strided
    d16 loads to 2 dwordx4). Pass ``v_transposed=False`` to build against the
    row-major layout instead, at the cost of a materially slower gather.

    The production tunables are the kv tile (``block_n``), the wave count
    (``n_waves``), the QK ILP (``qk_ilp``) and the non-temporal O store
    (``o_nt``, a large-L lever). Everything experimental (pipeline, q_hoist,
    q_lds, kv_lds, o_f16, d16hi, iglp, static_shape, prefetch_v, v_prefetch,
    bcast_group, k_dual, num_persistent) is off by default.
    """

    head_size: int
    num_query_heads: int
    num_kv_heads: int = 0
    mask_mode: str = "none"  # "none" | "causal"
    # n_waves=2 is the ROBUST choice: best at short L and dominant in the
    # memory-bound regime (more waves = more DRAM memory-level parallelism; at
    # L4096 and L8192 it is roughly 2x and 3x w1 respectively), and only ~3%
    # behind w1 at L2048. Worth +11-15% over the gather kernel in the
    # compute-bound regime.
    n_waves: int = 2
    # q_block (MQ): query-blocking factor. Each wave processes MQ 16-row query
    # tiles per K-loop iteration, REUSING the loaded K fragment (QK) and V
    # fragment (PV) across all MQ groups -> the KV DRAM read is amortized over
    # MQ x more queries, raising arithmetic intensity to lift the L>=4096 kernel
    # off the DRAM roofline (the barrier-free alternative to the coop/LDS
    # dead-ends). Costs MQ live O accumulators (MQ x n_dk x c_frag), so it only
    # fits where O is small: D64 (n_dk=4 -> 32 VGPR/group, 141 VGPR baseline has
    # room for MQ=2) -- at D128 (64 VGPR/group) MQ=2 hits the 256-VGPR wall.
    # Requires the default (non-pipeline) dual-gather PV path.
    # MEASURED WIN (D64 dense H24 B1, w2 bn64 ilp2; correct -- max_abs 3.05e-5):
    # MQ=2 (vgpr 141->246, spill 0) breaks the DRAM roofline and stays
    # compute-bound where the baseline craters -- +11% at L2048, +106% at L4096,
    # +404% at L8192.
    # MQ=4 register-trim attempt (block_n/ilp down cuts spill 268->67 since the
    # p_tiles/scores liveness scales with n_kv_sub=block_n/16): still LOSES to
    # MQ=2 at every L -- MQ4 bn16/ilp1/of16 runs 35-50% behind MQ2 bn64 across
    # L8k/16k/32k. Can't reach spill=0 (the carried f16 O 64 VGPR + f32 PV
    # transient 64 VGPR are irreducible without 2x V traffic), and the bn16
    # needed to shrink spill carries its own throughput penalty. The extra reuse
    # and the register cost fight each other -> MQ=2 is the hard ceiling.
    # MQ=4 spills (381 f32 / 268 even with o_f16 O-carry) and craters: each
    # extra query group costs ~105 VGPR (O + P-tiles + scores/Q transients), and
    # o_f16 only trims the O part (~11 VGPR), so MQ=4 (~456 VGPR) can't fit 256.
    # MQ=2 is the register ceiling at D64 (sustained win to ~L8K, tapering after
    # that); holding it further out needs the persistent KV-stationary kernel,
    # not more MQ. D128 MQ=2 does NOT fit -- vgpr=256 + spill=235 puts it well
    # BELOW the unblocked baseline at L4096, and the wall is far past what an
    # o_f16 O-carry (~64 VGPR saved) could recover. So query-blocking is a D64
    # (and smaller-head) win; D128 L>=4096 stays DRAM-bound, register-wall-
    # limited as throughout the campaign.
    q_block: int = 1
    # waves_per_eu: amdgpu-waves-per-eu hint. Occupancy on gfx1151 wave32 is a
    # STEP function -- VGPRs come from a 1536-entry file in granules of 16, so
    # 199 VGPR occupies a 208 granule = 7 waves/SIMD and the 8th wave needs <=192.
    #
    # MEASURED DEAD-END as an occupancy lever (L16384, mq1/of32/bn64/vt/qk_douter):
    # waves_per_eu=8 does force the allocation under the boundary (199 -> 190) and
    # does report 8 waves/SIMD, but it buys those 9 registers with 24 spills into
    # 100B of scratch and a loop grown to 1105 instructions. The scratch traffic
    # makes it not just slower on average but UNSTABLE across reps (a >2x spread
    # run to run, against a steady baseline). Squeezing the allocator does not
    # create registers; crossing 192 has to come from needing fewer live values,
    # not from telling the allocator to want them less.
    waves_per_eu: Optional[int] = None
    sched_mode: str = "pingpong"  # "none" | "pingpong"
    # iglp: -1 = off; >=0 = emit llvm.amdgcn.iglp_opt(level) at the loop-body top
    # to hand the steady-state schedule to the backend's canned interleave
    # strategy (0 = GEMM/mem<->mfma).
    # DEAD-END (hardware, L2048 dense): every level regresses ~3.5% (levels 0, 1
    # and 2 are within 0.3% of each other, all correct). The loop is already
    # scheduling-optimal via the pingpong s_setprio hand-tuning (the 2.25x
    # lever); the canned iglp strategy conflicts with it and does worse.
    # sched_barrier/sched_group_barrier can't help the d16hi inline-asm path
    # either (they reorder, but don't insert the vmcnt wait the uncounted asm
    # loads need). Kept OFF for reference.
    iglp: int = -1
    qk_ilp: int = 2
    # block_n: kv tile processed per K-loop iteration (multiple of 16). Larger
    # block_n does block_n/16 QK+PV WMMA sub-steps per iteration and rescales the
    # O accumulator only ONCE per block_n keys (vs once per 16) -- amortizes the
    # online-softmax fixed cost + adds WMMA ILP, at the price of block_n/16 live
    # score/P fragments. seqlen_k must be a multiple of block_n. bn64 is the
    # measured winner and only fits spill-free because dual_gather pays for it;
    # bn128 collapses on VGPR.
    block_n: int = 64
    # prefetch_v: software-pipeline the PV V-gather so the strided loads are
    # hidden behind compute. The first fragment's gather is issued BEFORE the
    # softmax (overlaps the softmax VALU); every subsequent (ns,d) fragment is
    # gathered ONE PV-step ahead of the WMMA that consumes it, so its 16 scalar
    # loads stay in flight while the previous step's WMMA runs (the vmcnt wait
    # lands a full step later, when the data is already home). Costs +1 live V
    # fragment.
    #
    # MEASURED DEAD-END (H24 B1 D128): prefetch_v=True REGRESSES at every shape.
    # Compute-bound (L<=2048): the gather is L1-resident, so the AMDGPU backend
    # already hides it; the extra live fragment only adds spills (8->12)
    # -> -8..-10%. Memory-bound (L>=4096): the bottleneck is DRAM MLP, which is
    # hidden by MORE WAVES (n_waves=2 >> n_waves=1 at long L), not by manual
    # prefetch (~neutral to worse). Kept as a lever for the A/B record; OFF by
    # default.
    prefetch_v: bool = False
    # static_shape: bake the Q/K/V/O strides (all pure functions of heads/head_size,
    # known at build time) as compile-time CONSTANTS instead of runtime params.
    # The gather address math is base + kv*stride_v_token repeated 16x per fragment;
    # with a runtime stride the backend emits an s_mul_i32/v_mul_lo_u32 + 64-bit
    # v_lshlrev/v_add_co per element (the ISA showed ~40 s_mul_i32 by the stride reg
    # + 36x v_lshlrev_b64/v_add_co in the K-loop). Constant strides fold those into
    # scaled immediates / a single shift. The params are still declared (ABI/driver
    # packing unchanged) -- just ignored for addressing.
    #
    # MEASURED DEAD-END: static_shape=True REGRESSES (-9% at L2048). The per-kv
    # offset kv*stride_v*2 (~6 KB steps) exceeds the global_load immediate-offset
    # range (+/-4 KB), so constant strides don't fold into scaled immediates --
    # LLVM instead materializes each address with an explicit 64-bit v_add_co
    # (163 vs 36). The runtime-stride path shares the stride multiply in SALU and
    # is cheaper. Kept as a lever; OFF by default.
    static_shape: bool = False
    # buffer_gather: issue the PV V-gather via a buffer descriptor + the D16
    # half-return load (buffer_load_f16_d16 -> raw.ptr.buffer.load.f16). Address =
    # base + voffset + soffset is computed in the MEMORY UNIT (no 64-bit address
    # VALU), and returning `half` directly (not i16+bitcast) keeps the load clause
    # batched (s_clause 121->38) so the loads pipeline like the flat d16 path.
    # MEASURED +2..12% (avg ~+7%) over flat @ L2048, bit-identical (1.53e-5).
    #   FRAGILE: only wins at n_waves=2 AND block_n>=32. At block_n=16 or
    #   n_waves=1 the backend stops batching the buffer loads and throughput
    #   drops to well under half. Enabled by default because the shipped defaults
    #   ARE w2/bn64; an off-sweet-spot sweep will still expose the bad points.
    buffer_gather: bool = True
    # dual_gather: kill the 2x V-load redundancy from the WMMA A-operand's
    # lane 0-15 <-> 16-31 duplication. Instead of lanes 16-31 re-loading subtile
    # d, they load the ADJACENT subtile d+1 (one 32-lane load fetches TWO
    # d-subtiles); a permlanex16 + select then broadcasts each subtile's fragment
    # into both halves. Halves the V load instructions (256->128) at the cost of
    # permlanex16/cndmask VALU -- a win because the kernel is memory-unit-bound.
    # Requires n_dk even (D%32==0).
    # MEASURED: halves V loads (256->128) + s_waitcnt (147->76); small consistent
    # win once loads are no longer strictly binding (~+2-3% at L2048) and lower
    # VGPR/instr. It is also what lets block_n=64 fit spill-free, which is the
    # shipped default.
    dual_gather: bool = True
    # lazy_rescale: skip the O-accumulator rescale (n_dk vector-muls/iter) when the
    # running softmax max is stable -- i.e. every lane's tile_max is within
    # _LAZY_RESCALE_THRESHOLD log2 of m_i, so not re-anchoring is safe (exp2(8)=256
    # can't overflow fp32). gfx950 dense ships this ALWAYS-ON (+~2%, parity-
    # identical). A VALU win here (the max stabilizes after the first few K-tiles,
    # so most iters skip). Implemented as a wave-uniform 0/1-trip scf.for so the
    # multiplies are genuinely skipped (rocke scf_if carries no results).
    lazy_rescale: bool = True
    # fast_exp2: use raw v_exp_f32 (exp2_fast) instead of the IEEE exp2 that the
    # backend guards with a v_cmp/v_cndmask clamp. Safe: online-softmax exp args
    # (m_i - m_new, s - m_new) are always <= 0. gfx950's exp2_fast lever (+11.5%).
    fast_exp2: bool = True
    # pipeline: software-pipeline the QK across K-tiles. Compute tile 0's QK in a
    # prologue and carry the scores as loop iter-args; each iteration runs the
    # CURRENT tile's softmax/P-transpose/PV while issuing the NEXT tile's QK
    # WMMAs -- so the matrix unit stays fed during the softmax VALU (WMMA
    # utilization) instead of idling. Costs n_kv_sub carried score tiles.
    # MEASURED (register-GATED, not a flat dead-end):
    #   * D<=64 (O accumulator <= 32 VGPR): FITS spill-free and WINS +4-5%
    #     (D64 L2048, vgpr 115->185, spill 0). Recommended ON here.
    #   * D=128 (O = 64 VGPR): carrying current+next scores blows the 256-VGPR cap
    #     (vgpr=255 + spills) and more than halves throughput. The D=128 kernel is
    #     register-bound and cannot pipeline; unblocking it via a D-split costs
    #     2x softmax + 1.5x matmul (a confirmed net loss).
    pipeline: bool = False
    # q_hoist: Q is loop-invariant (this wave's 16 query rows). Load all n_dk Q
    # fragments ONCE before the K-loop and pre-scale them by scale_log2, so the
    # QK loop (a) doesn't reload Q every tile -> fewer QK loads + fewer vmcnt(0)
    # drains, and (b) the QK output is already scaled -> drop the per-slot softmax
    # scale-mul (VALU). Costs n_dk live Q fragments (register pressure).
    # MEASURED DEAD-END: regresses ~40%. The 8 hoisted Q fragments (64 VGPR) on
    # top of the 64-VGPR O accumulator blow the 256-VGPR budget (vgpr=256 + 60
    # spills). Same register wall as `pipeline` -- the kernel is register-bound.
    q_hoist: bool = False
    # q_lds: the register-pressure-free version of q_hoist. Stage each wave's 16
    # (pre-scaled) query rows into LDS ONCE, then the QK re-reads Q from LDS
    # (ds_read/lgkmcnt) instead of global (global_load/vmcnt). Moves Q off the
    # V-gather's contended vmcnt AND drops the per-slot softmax scale-mul, WITHOUT
    # holding Q in VGPRs (avoids the q_hoist register wall). Costs W*16*hs*2 bytes
    # LDS (8 KB for w2/D128) + a one-time cooperative staging pass + intra-wave
    # waitcnt.
    # MEASURED DEAD-END: regresses ~2.5x. Same lesson as V (cache-gather beat
    # LDS): on this APU the cache-resident global Q re-read is free (hidden by
    # pingpong), while LDS staging breaks the clause/pipeline structure -- vmcnt(0)
    # drains EXPLODE 20->77 and instr 1215->1689. L1-hit >= LDS here.
    #
    # RE-CONFIRMED at MQ2 (per-wave slab, per-wave s_waitcnt -- NO block barrier,
    # each wave stages+reads only Q_lds[wave_id]): still a ~3x dead-end at every
    # L. The barrier was never the issue -- the per-K-tile ds_read breaks the WMMA
    # operand clause + adds lgkmcnt waits, and (unlike o_nt) Q is NOT a
    # MALL-pressure source: it is ~8 KB/q-block and L1-resident, so it never
    # competed with the MALL-resident KV -> staging it in LDS relieves nothing and
    # only adds overhead. o_nt (stream the write-once O) helps precisely because O
    # DID contend; Q does not.
    q_lds: bool = False
    # kv_lds: PROTOTYPE (large-L / DRAM-bound regime). Keep the ENTIRE swapqk
    # architecture (transposed QK, in-lane softmax, register P-transpose, dual,
    # pingpong) unchanged, but source the per-K-loop K and V tile from a
    # cooperatively-staged LDS copy instead of re-reading it from global every
    # iteration. All W waves in the CTA share one LDS-resident K/V tile, so the
    # DRAM read of that tile is amortized across the CTA's query rows (cuts the
    # cross-wave KV re-reads that make the kernel DRAM-bound at L>=4096, where the
    # per-head KV working set spills L2 to a ~47% hit rate). Costs one cooperative
    # load + 2 s_barriers per K-tile (which partially fight pingpong). Forces the
    # flat LDS V-read path (buffer_gather is a global-only lever).
    # DEAD-END (hardware, dense H24 B1 D128, w2 bn64 ilp2; correct -- max_abs
    # 3.05e-5 == gather): loses badly AND the gap WIDENS with L, the opposite of
    # the hypothesis -- 0.38x at L512, 0.35x at L2048, 0.29x at L4096.
    # Root causes: (1) vgpr 197->256 + spill=16 (coop loader's div/mod addressing
    # + LDS staging), (2) dsld 0->320 -- the flat V read is 16 uncoalesced scalar
    # ds_loads/fragment, far worse than the strided buffer gather it replaces,
    # (3) 2 s_barriers/tile serialize the waves and kill the pingpong 2.25x lever.
    # Crucially the per-tile overhead scales with the K-loop trip count, so it
    # gets RELATIVELY worse as L grows -- the DRAM savings (gld 80->48, only ~w2x
    # since 2 waves share) never approach offsetting it. Confirms the README
    # lesson holds even in the L4096 DRAM-bound regime: on this large-cache APU,
    # barriers + LDS traffic cost more than the KV re-reads they remove.
    kv_lds: bool = False
    # o_f16: carry the O accumulator across the K-loop as f16 (32 VGPR for D128)
    # instead of f32 (64 VGPR), and REORDER the PV to d-pair-outer / ns-inner so
    # each O d-pair is fully accumulated (both kv sub-tiles) then immediately
    # truncated to f16 -- so only the CURRENT d-pair is f32 (16 VGPR) at a time,
    # the rest stay f16. Shrinks the O-accumulator register peak (~64->~40 VGPR)
    # to open headroom for the pipeline (which fits+wins whenever O is small, cf.
    # D64). Costs n_dk f16<->f32 converts/block; f16 carry rounds each block ->
    # precision must be verified. Forces lazy_rescale off (rescale fused into the
    # per-d-pair convert).
    # MEASURED: correct (1.07e-4, within tol) and DOES reclaim VGPR (184->164,
    # -20). But (a) the 16 f16<->f32 converts/block cost more than the 20 freed
    # regs buy -> ~20% slower standalone, and (b) the pipeline needs ~92 VGPR of
    # headroom (its next-QK accumulators), so o_f16+pipeline still spills
    # (vgpr=256 + 111). Net dead-end for D=128; the register relief is real but an
    # order of magnitude short of unblocking the pipeline.
    #
    # ALSO MEASURED as an occupancy lever, and rejected: at bn64/vt it is the only
    # way found to reach 8 waves/SIMD with ZERO spill (vgpr 199 -> 183, crossing
    # the 192 granule), but the loop grows 1082 -> 1251 instructions and it goes
    # PATHOLOGICAL at long sequences -- 3-8x slower at L16384 while still fine at
    # L2048, independent of qk_douter. Whatever that cliff is, it is not the +14%
    # occupancy paying off. Unexplained; do not reach for o_f16 to buy waves at
    # L>=8K without profiling it first.
    o_f16: bool = False
    # d16hi: d16_hi buffer gather (buffer_gather + dual_gather only). Pins
    # buffer_load_d16_b16/_hi_b16 via inline asm (hi tied to lo) so each strided
    # f16 pair packs DIRECTLY into one VGPR's lo/hi lanes, eliminating the ~64
    # v_mov_b16 f16-pack the typed buffer_load_f16_d16 path emits -- the backend
    # NEVER selects the D16-hi buffer form from the intrinsic (verified: minimal
    # llc probe + full-kernel ISA both give buffer_load_u16 + v_mov_b16,
    # regardless of insertelement shape). The flat path already gets
    # global_load_d16_hi_b16 for free, so d16hi is a no-op there.
    #
    # DEAD-END (hardware-validated, L2048 dense H24 B1 D128). ISA is clean either
    # way -- 64 buffer_load_d16_b16 + 64 _hi_b16, 0 buffer_load_u16, v_mov_b16
    # 73->9 -- but the inline-asm loads are OUTSIDE the backend vmcnt model, and
    # every way of adding the mandatory wait loses:
    #   * no fence          -> +2.8% but NaN: the PV permute reads the fragment
    #                          before the loads land (a race).
    #   * coarse vmcnt0_fence (CURRENT, correct 1.53e-5) -> -5.3%:
    #                          one s_waitcnt vmcnt(0) per fragment serialises the
    #                          gather, killing the load/compute overlap the typed
    #                          path gets from backend-managed PARTIAL waits
    #                          (vmcnt(2) interleaved with the PV WMMAs); the 64
    #                          v_mov_b16 saved are cheap + already latency-hidden.
    #   * hand fine-grained partial waits (buffer_load_d16_gather + counting-down
    #                          vmcnt_fence, mimicking the backend schedule)
    #                          -> GPU HANG: with multiple gather blocks pipelined
    #                          the uncounted asm loads make the manual vmcnt(K)
    #                          accounting deadlock-prone. Not safe to enable.
    # Conclusion: intrinsic V loads get free, correct backend software-pipelining
    # + vmcnt tracking that inline-asm d16 loads cannot match. Kept OFF; the
    # current path is the coarse (correct) one for reference only.
    d16hi: bool = False
    # o_nt / q_nt: streaming (non-temporal, cache-bypass) global O-store / Q-load.
    # LARGE-Sq MALL-residency levers (idea 1 / idea 2). At L>=8K the per-head KV
    # is the reused working set we WANT resident in the MALL, but the write-once
    # O output (up to 8 MB/head @ L32K, never re-read) allocates MALL lines and
    # EVICTS KV -- dropping the KV hit rate (the measured cause of the
    # sub-ceiling L16K/32K throughput). ``o_nt`` marks the O epilogue store
    # ``!nontemporal`` so it streams past MALL (no allocate), leaving the whole
    # cache to KV; it is worth +3-14% for large-L head-chunked launches.
    # ``q_nt`` does the same for the Q-fragment load -- an EXPERIMENT knob: Q is
    # re-read every K-tile (reused), so streaming it should HURT, confirming the
    # "keep reused data cached, stream write-once data" separation. KV (K load +
    # V gather) is ALWAYS left default-cached. Pair with the head-chunked launch
    # (concurrent working set <= MALL) for large Sq.
    o_nt: bool = False
    q_nt: bool = False
    # v_transposed: take V pre-transposed as [B, H, D, S] instead of [B, S, H, D].
    # The PV A-operand needs V[kv=0..15, d_col] per lane. In [B,S,H,D] those 16
    # values sit one token-stride (H*D*2 = 6 KB) apart, so the gather is 16
    # separate d16 loads; the 32 lanes DO coalesce (they cover 32 consecutive d,
    # i.e. 64 contiguous bytes), but each instruction only moves 64 B where a
    # dwordx4 moves 512 B. Transposing makes the 16 keys contiguous, so one lane
    # reads them as 2 dwordx4 -- 16 loads collapse to 2 (256 -> 32 vector-memory
    # instructions per K-loop iteration at bn64/D128).
    #
    # The tradeoff is address divergence: the 32 lanes now sit on 32 different
    # d-rows (S*2 bytes apart) instead of one cache line. Measured in isolation
    # with a standalone gather microbenchmark (S=16K/D128/H24, 512 waves):
    # [B,H,D,S] sustains ~2x the bandwidth of [B,S,H,D] -- the instruction saving
    # wins, and the transposed form saturates at low occupancy while the
    # row-major one needs 2-4x the waves to catch up (this kernel is VGPR-capped
    # at 7 waves/SIMD, so it never gets there).
    #
    # MEASURED (H24 B1 D128, mq1/of32/bn64, single-head dispatch, exact: 1.53e-05,
    # identical to the default layout). ISA does what it promises -- buffer_load
    # 256->32, v_mov_b16 145->17, s_waitcnt 163->67, +3 VGPR, 0 spill. Cycle
    # counts (GRBM_GUI_ACTIVE, so clock-independent -- see the throttling note
    # below) and memory-unit busy cycles (TA_TA_BUSY):
    #                cycles      TA busy
    #   * L4096      -20.3%      -24%
    #   * L8192       -5.9%      -10%
    #   * L16384      +0.3%        0%
    # The benefit tracks whether the memory unit RESPONDS at all: at L4096 the
    # gather is the constraint and removing 8/9 of the loads removes a fifth of
    # the runtime; by L16384 TA_TA_BUSY is pinned no matter what the load stream
    # looks like, so there is nothing left to win. At L16K the unit is
    # latency-saturated (holding outstanding L2/MALL returns), not
    # throughput-saturated -- see v_kblock for the experiment that rules out
    # instruction count and cache-line count as the L16K constraint.
    #
    # BEWARE when re-measuring in wall-clock terms: sustained runs power-throttle
    # this part hard within a few seconds, and long-L runs sit deep in that
    # regime. That swing is larger than any of the effects here and it masked the
    # L4096/L8192 wins entirely in an earlier end-to-end sweep. Compare cycles,
    # not wall clock.
    #
    # REQUIRES q_block=1. At q_block=2 the wide loads land on a kernel already
    # pinned at the 256-VGPR cap and spills explode (38 -> 153, scratch 28 ->
    # 114) for -24% at L2048.
    #
    # Requires buffer_gather and is incompatible with kv_lds (an LDS-staged V
    # tile is a separate lever that solves the same problem a different way).
    # Callers must supply V already transposed; see swapqk_transpose_v().
    v_transposed: bool = True
    # v_kblock: with v_transposed, use the KEY-BLOCKED layout [B, H, S/KB, D, KB]
    # instead of the full transpose (KB=0). KB keys stay contiguous per d, and d
    # advances every KB*2 bytes, so one instruction's 32 lanes cover 32*KB*2
    # contiguous bytes instead of 32 rows S*2 bytes apart.
    #
    # This exists to separate the two things the full transpose changes at once.
    # Per K-loop iteration at bn64/D128 the three layouts are:
    #   default   [B,S,H,D]     256 loads,  1 cache line each
    #   transpose [B,H,D,S]      32 loads, 32 cache lines each (rows S*2 apart)
    #   blocked   KB=8           32 loads,  4 cache lines each
    # KB=8 is the discriminating point: the transpose's instruction count AND
    # fewer lines than the default. If the memory unit were bound by either
    # instruction issue or line lookups, KB=8 would be the clear winner.
    #
    # MEASURED (L16384, single-head dispatch, 3 reps, GRBM_GUI_ACTIVE cycles):
    # default and transpose are within 0.3% of each other, and blocked KB8 spends
    # 9.3% MORE cycles than either, while TA_TA_BUSY is FLAT to within 1% across
    # all three. So KB=8 is the WORST of the three, and the memory-unit busy
    # counter does not budge across an 8x swing in instructions and an 8x swing
    # in lines. That rules out both as the L16K constraint (at L4096, where the
    # gather IS binding, the same counter moves -24%). The kernel is
    # memory-LATENCY bound there: TA reads as ~75% busy because it is holding
    # outstanding L2/MALL returns, and issuing fewer/cheaper requests does not
    # shorten the wait. Kept only as the probe that establishes this; KB=0 is the
    # useful setting. KB must divide 16 and give a 4/8/16-byte load, so KB in
    # {2, 4, 8}.
    v_kblock: int = 0
    # v_prefetch: keep N V-gathers in flight across the PV step sequence. The
    # loads for step i+N are issued before step i's permute + WMMAs, so the
    # s_waitcnt for the fragment in hand does not also gate issuing the next
    # requests. Costs 8 VGPR per outstanding step (the RAW <16 x f16>; the
    # permute to a fragment PAIR is deferred until the data is needed, so the
    # in-flight cost is half of what carrying finished fragments would be).
    #
    # This is the lever for the L16K regime, which is memory-LATENCY bound: the
    # SIMD issues on ~16% of cycles and a wave spends ~55 cycles per instruction
    # waiting, while occupancy is hard-capped at 8-9 waves/SIMD by the O
    # accumulator (see v_kblock). More requests per wave is the only remaining
    # way to cover the latency once more waves are unavailable.
    #
    # Distinct from prefetch_v, which only ever ran on the NON-dual_gather path
    # (it sits in an elif after dual_gather, so with the shipped dual_gather=True
    # default it was dead code -- its recorded regression was measured on a path
    # that is not the production one). v_prefetch targets the dual path and
    # composes with v_transposed, where a gather is only 2 loads to issue.
    #
    # MEASURED DEAD-END (H24 B1 D128 L16K, GRBM cycles, o_f32/bn32/vt), relative
    # to depth 0 (187 VGPR, 0.3M L2 misses, 0.2M 128B DRAM reads):
    #   depth 2 -> -2% cycles,  204 VGPR, L2  0.8M miss,  0.7M DRAM reads
    #   depth 3 -> 3.2x cycles, 220 VGPR, L2 12.4M miss, 12.4M
    #   depth 4 -> 3.5x cycles, 220 VGPR, L2 13.5M miss, 13.5M
    # Depth 2 is neutral (within run-to-run spread); depth >=3 falls off a cliff.
    # The cliff is NOT registers or instructions: spill and scratch stay 0, the
    # static mix is unchanged (16 buffer_load / 48 global_load / 32 wmma),
    # s_waitcnt actually DROPS 35 -> 30, and dynamic loads are identical at 8.4M
    # with VALU 10% LOWER. What breaks is locality -- holding steps i..i+depth in
    # flight widens the concurrently-touched footprint past what L0/L1 holds, so
    # lines are evicted before the next step reuses them and DRAM reads go up 40x
    # for the same 8.4M loads.
    #
    # The deeper reason this lever cannot pay: at depth 0 the kernel only pulls
    # ~26 MB from DRAM (~4% of available bandwidth), so the latency the waves are
    # hiding is mostly L0/L1/L2-HIT latency on the vector-memory path, not DRAM
    # latency. Adding requests-in-flight cannot cover that without spending the
    # very cache residency that makes those hits cheap.
    v_prefetch: int = 0
    # NOTE on the dual-gather broadcast cost (248 v_cndmask_b32 + 146
    # v_permlanex16_b32 in dual_gather_finish, together ~16% of the K-loop's issue
    # cycles). Two routes to make it cheaper, both checked against the ISA:
    #
    # (a) One-instruction row swap: NOT on this target. v_permlane16_swap_b32
    #     would turn the 3-op broadcast (1 permlane + 2 selects per dword) into
    #     mov+swap, but llvm-mc rejects it for gfx1151 (it is gfx950/gfx125x).
    #     permlanex16 with a tied vdst under a half-exec mask also costs 3 ops
    #     (mov + 2 permlane), so there is no 2-op formulation on RDNA3.5.
    #
    # (b) VOPD: LEGAL here and largely UNTAKEN -- the open lever.
    #     v_dual_cndmask_b32 :: v_dual_cndmask_b32 assembles for gfx1151, all 248
    #     selects read the SAME mask (one distinct operand, vcc_lo, so the shared-
    #     VCC rule is satisfied by construction), and 120 of them already sit
    #     adjacent to another select. Yet the backend forms only 4
    #     v_dual_cndmask_b32. gcn-create-vopd is on and forcing
    #     -amdgpu-enable-vopd either way is byte-identical, so what blocks the
    #     other ~116 pairs is VOPD's operand rules (src bank / dst parity), not
    #     eligibility. The two selects of one dword are the worst possible
    #     neighbours for those rules: they read the SAME register pair (e, p) with
    #     operands swapped. Emitting selects from DIFFERENT dwords back-to-back
    #     gives the allocator bank freedom. Ceiling if fully paired: 248 -> 128
    #     issue slots, ~157 cyc/iter, ~4.7% of the issue floor.
    #     Watch for: holding all n_i32 permlane results live at once to expose
    #     those runs raises the register peak -- check spill=0 before believing a
    #     win.
    #
    # qk_douter: run the QK loop d-OUTER / kv-inner instead of kv-outer / d-inner.
    #
    # Q[d] does not depend on the kv sub-tile, so d-outer needs one Q fragment
    # live at a time and drops the qk_ilp machinery: the n_kv_sub accumulator
    # chains are mutually independent, so they already supply the ILP acc_ilp was
    # building by hand, and its tail reduction disappears (1102 -> 1082 loop
    # instructions). Distinct from q_hoist, which lifts Q out of the WHOLE K-loop
    # and must keep all n_dk fragments live (64 VGPR at D=128 -> spills).
    #
    # MEASURED WIN (L16384 H24 B1 D128, mq1/of32/bn64/vt, 3 interleaved reps):
    # +3.3% throughput, -4.0% GRBM cycles, -11% TA_TA_BUSY, -12% stall cycles.
    # vgpr 200 -> 199, zero spill, zero scratch, error unchanged (1.53e-05 @
    # L2048, 7.63e-06 @ L16384). Interleave the reps: a sequential A/B/A/B reads
    # BACKWARDS here because the part throttles within seconds and the drift
    # exceeds the effect.
    #
    # The win is NOT the mechanism this was built for. The intent was to delete
    # n_kv_sub*n_dk - n_dk = 24 redundant Q loads per iteration, but the in-loop
    # global_load count does not move (64 at bn64, 32 at bn32) under ANY nesting,
    # and does not move when o_f16 frees 17 VGPR either, so it is not remat under
    # pressure. Grouping the loads by address operand shows what actually
    # changed: kv-outer issues 32 distinct addresses TWICE each, d-outer issues
    # 64 distinct addresses once each. Same instruction count, but the duplicate
    # fetches were spending requests on a texture-address path already 96.9%
    # busy -- which is why TA_TA_BUSY, not the load count, is where the gain
    # shows up.
    #
    # MQ>1 has its own QK loop that already hoists K across query groups; this
    # knob is rejected there rather than silently doing nothing.
    qk_douter: bool = True
    # A NOTE on the register peak, since it is what caps this kernel. qk_douter
    # keeps all n_kv_sub score accumulators AND all n_kv_sub K fragments live (at
    # bn64: 32 + 32 VGPR), which pins it at 199 VGPR = 7 waves/SIMD. The 8th wave is
    # worth +3.9% MEASURED IN ISOLATION (qkdo=0 ilp1 at 192 VGPR/8 waves vs the
    # same code at 193 VGPR/7 waves), and qk_douter itself is worth +7.1%, so the
    # two together would compound. They do not compose: waves_per_eu=8 on the qkdo
    # path gives 190 VGPR but 24 IN-LOOP spills, netting -1.5% -- the wave nearly
    # pays for the spills, but not quite.
    #
    # Source-level grouping does NOT fix this -- tried and reverted: running the d
    # loop over groups of 2 kv sub-tiles (so only 2 accumulators + 2 K fragments are
    # live) left the count at exactly 199, because the scheduler is free to
    # re-interleave the groups and does. Group size 1 degenerates to the kv-outer
    # form (193 VGPR) and loses qk_douter. The peak is set by the scheduler's load
    # batching, not by the source order, so only a real constraint (waves_per_eu) or
    # removing address registers (see buffer_gather, which does this for V) can move
    # it.
    # bcast_group: emit dual_gather_finish's lane broadcast in groups of N dwords
    # (all permlanes of the group, then its d selects, then its d+1 selects).
    # 0/1 = today's per-dword interleave. See the VOPD note above: the two selects
    # of ONE dword read the same register pair with operands swapped, which is the
    # worst case for VOPD's src-bank/dst-parity rules, and only 4 of 64 such pairs
    # actually form. Grouping makes neighbouring selects come from DIFFERENT
    # dwords, so they touch disjoint registers and the allocator has N independent
    # chances to land a legal pair. N=8 (all of n_i32 at D=128) exposes the longest
    # runs but holds 8 permlane results live at once; N=4 halves that pressure,
    # which matters because 199 VGPR is only 9 short of the 208 granule where
    # occupancy drops from 7 waves/SIMD to 6.
    #
    # MEASURED: the packing works exactly as designed, and it is a SHORT-SEQUENCE
    # win that falls off a cliff at long sequences. Default off.
    #   codegen (bn64/vt/qk_douter, N=4): selects 248 -> 20 singles + 118
    #     v_dual_cndmask_b32 pairs (from 4), K-loop 1082 -> 899 instructions,
    #     dynamic VALU -14%, issue floor -4%. Zero spill, zero scratch. N=0 and
    #     N=1 compile identically, confirming this is a pure reorder; error is
    #     1.53e-05 at every N, as it must be.
    #   L2048 bn64: +6.2% (3 interleaved reps). Real win.
    #   L16384 bn64: 11x LOSS. Root cause is a cache-residency collapse, NOT the
    #     occupancy drop: at the real 24-chunk shape L2 misses go 3.75M -> 393M
    #     (105x) and DRAM read 323 MiB -> 47.9 GiB (148x). Grouping lets the
    #     scheduler hoist the gather loads away from their consumers, and the
    #     extra distinct lines in flight blow L2 once 24 heads are resident --
    #     the same failure mode as v_prefetch depth>=3 (0.3M -> 12.4M misses).
    #   L16384 bn32: VGPR stays 179 so occupancy is UNCHANGED (8 waves/SIMD) and
    #     the loop still shrinks 581 -> 500 instructions, yet cycles are FLAT:
    #     issue drops ~5% while stall rises by the same absolute amount. The
    #     freed issue slots convert 1:1 into stall, which says these selects were
    #     never on the critical path -- they were being issued in stall shadow.
    #     Cutting VALU issue is not a lever for this kernel at long L; the memory
    #     path is.
    #
    # NOTE on measuring this: a reduced single-head dispatch reported merely +8%
    # cycles for N=4 at L16384 and completely masked the 11x cliff, which only
    # appears in the full 24-head-chunk shape. Point rocprofv3 at the real
    # head-chunked launch before trusting any long-L verdict. The o_f16 L16384
    # cliff is probably the same effect and is worth re-checking that way.
    bcast_group: int = 0
    # k_dual: the K-side twin of dual_gather. load_wmma_fragment takes the WMMA
    # A-operand row from lane%16, so lanes l and l+16 issue the SAME address and
    # every K load runs at HALF lane efficiency -- at bn64 the 64 K load
    # instructions move 32 KB of traffic for a 16 KB tile. Indexing by the full
    # lane makes the upper half fetch the next sub-tile's 16 keys, so one load
    # covers a kv PAIR, and the existing permlanex16+select broadcast splits it
    # back into two duplicated fragments.
    #
    # Why this is the lever: requests-per-WMMA is pinned at 1.5 for every block_n
    # (32/64/96/128 all measure 1.5), so tiling cannot improve the memory-to-
    # compute ratio -- only reuse can, and MQ=2 does not fit at D=128 (spills 246
    # at bn64, still 40 at bn16+o_f16, because O alone is MQ*n_dk*8 = 128 VGPR).
    # k_dual is the one remaining way to cut requests without more registers: it
    # halves the FLAT loads, taking total in-loop memory instructions from 96 to 64
    # at bn64. It trades the identical 24 VALU per 2 saved loads that dual_gather
    # trades for V, and that trade measured +34% there; bcast_group separately
    # showed this kernel's VALU is issued in stall shadow, so the added broadcast
    # should be closer to free than the arithmetic suggests.
    #
    # Requires qk_douter (the pair-wise kv iteration lives in its d-outer loop) and
    # an even n_kv_sub, i.e. block_n >= 32.
    #
    # MEASURED: correct (max_abs 1.53e-05) and a win at L=2048 (+8%), but an 11x
    # LOSS at L=16384 (3 interleaved reps). The codegen did exactly what it was
    # built to do -- FLAT loads 16.9M -> 8.5M, spill-free, occupancy held, and the
    # doubled broadcast even self-packed into VOPD (42 -> 164 pairs) -- yet
    # TA_TA_BUSY went UP 13% and DRAM read up 13x.
    #
    # WHY, and the general rule: duplicate LANES are free, duplicate INSTRUCTIONS
    # are not. The address unit coalesces the 16 identical addresses inside one
    # global_load_b128 at no cost, so the lane%16 duplication was never costing
    # requests -- halving the instructions bought nothing while making each one span
    # 32 cache lines instead of 16, which costs the residency this kernel lives on.
    # dual_gather is NOT the same case and the symmetry argument was wrong:
    # gather_v_a_frag issues 16 separate load_scalar instructions per V fragment, so
    # V's redundancy is duplicate instructions (uncoalescable, worth 34% to remove).
    # Do not re-attack A-operand lane duplication on a vector-load path.
    k_dual: bool = False
    # num_persistent: >0 replaces the (q_blocks, H, B) grid with a FIXED 1-D grid of
    # this many long-lived CTAs that pull (q_block, head, batch) work-items from a
    # global atomic counter until they are exhausted. 0 = today's one-shot grid.
    #
    # Why this and not another in-loop knob. Three separate in-loop levers
    # (bcast_group, o_f16, k_dual) all won at L=2048 and collapsed at L=16384 via an
    # L2-miss/DRAM explosion, and mem-instructions-per-WMMA is pinned at 1.5 for
    # every block_n, so the binding constraint is CACHE RESIDENCY, not issue slots
    # or request count. A work queue is the first lever that changes *what is
    # resident* rather than trading locality for fewer instructions: with
    # persist_decode="qb_major" a CTA draining adjacent tile ids stays on ONE
    # (head, batch) and reuses that head's K/V out of L2, and deep oversubscription
    # (~24x the CU count) keeps the queue full enough to hide the fetch latency.
    #
    # MQ>1 / q_lds / kv_lds are rejected here: MQ=2 does not fit at D=128 and the
    # LDS paths allocate inside what would become the work-item loop.
    # MEASURED (D128 H24 B1, mq1 of32 bn64 w2 vt dual qkdo): correct (1.53e-05).
    #   L=2048  : pers960 + waves_per_eu=8 .. +24% over one-shot
    #   L=16384 : pers1920 + waves_per_eu=8 . parity with one-shot (3 interleaved
    #             reps; pers960 WITHOUT wpe is ~17% down -- see below)
    # Depth is not optional and the curve is monotone in the CTA count: one CTA per
    # CU starves the queue badly, and throughput climbs steadily out to ~1920 CTAs,
    # more than 3x the shallowest point.
    #
    # COST: the work-item loop is +55 VGPR (199 -> 254, spill-free) because
    # loop-invariant setup hoisted out of it stays live across the whole body, which
    # drops 7 -> 6 waves/SIMD. waves_per_eu=8 buys the wave back (192 VGPR, 19
    # spills, 80 B scratch) and recovers most of the loss; this is the one config
    # where wpe=8 helps, since it corrects that regression rather than pushing past
    # a real peak.
    #
    # The schedule itself is emphatically the right idea -- persist_decode alone is
    # worth 21x at L=16K (qb_major over batch_major) -- but the one-shot grid's
    # in-order dispatch already approximates qb_major, so L2 hit only moves
    # 66.0% -> 67.8% and the net is parity. Default off; use it for short L, or when
    # run-to-run stability matters (pers1920 wpe8 holds a ~16x tighter spread than
    # one-shot).
    num_persistent: int = 0
    # persist_decode: work-item -> (q_group, head, batch) unpack order.
    #   "qb_major"    : q_group fastest -> a CTA stays on one (head,batch): K/V reuse
    #   "batch_major" : spreads batch; only for the reproducible A/B (and it is 21x
    #                   slower at L=16K, which is the measurement that proves the
    #                   schedule's locality is what matters)
    persist_decode: str = "qb_major"
    name: str = "wmma_fmha_swapqk"

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_query_heads

    @property
    def block_size(self) -> int:
        return 32 * self.n_waves

    @property
    def q_rows_per_cta(self) -> int:
        return 16 * self.n_waves * self.q_block

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        return kernel_name_join(
            self.name,
            f"H{self.head_size}",
            f"HQ{self.num_query_heads}",
            f"HK{self.kv_heads}",
            self.mask_mode,
            f"w{self.n_waves}",
            f"vpe{self.waves_per_eu}" if self.waves_per_eu is not None else "vpedef",
            self.sched_mode,
            f"ilp{self.qk_ilp}",
            f"bn{self.block_n}",
            "pfv" if self.prefetch_v else "npf",
            "stat" if self.static_shape else "dyn",
            "buf" if self.buffer_gather else "flat",
            "dual" if self.dual_gather else "single",
            "lazy" if self.lazy_rescale else "eager",
            "fexp" if self.fast_exp2 else "iexp",
            "pipe" if self.pipeline else "nopipe",
            "qh" if self.q_hoist else "noqh",
            "qlds" if self.q_lds else "qglob",
            "kvlds" if self.kv_lds else "kvglob",
            f"qb{self.q_block}",
            "of16" if self.o_f16 else "of32",
            "d16hi" if self.d16hi else "d16lo",
            "ont" if self.o_nt else "oct",
            "qnt" if self.q_nt else "qct",
            (
                ("vt" if not self.v_kblock else f"vk{self.v_kblock}")
                if self.v_transposed
                else "vn"
            ),
            f"vpf{self.v_prefetch}" if self.v_prefetch else "novpf",
            "qkdo" if self.qk_douter else "qkno",
            f"bg{self.bcast_group}" if self.bcast_group else "bgoff",
            "kdual" if self.k_dual else "ksingle",
            (
                f"pers{self.num_persistent}_{self.persist_decode}"
                if self.num_persistent
                else "oneshot"
            ),
            f"iglp{self.iglp}" if self.iglp >= 0 else "noiglp",
        )


def is_valid_spec(cfg: SwapQKCfg, arch: str = "gfx1151") -> "tuple[bool, str]":
    """Cheap static validity gate (mirrors ``wmma_fmha_fwd.is_valid_spec``).

    Only the shape/tile constraints that are cheap to check without building.
    Knob-compatibility rules (``v_transposed`` needs ``buffer_gather``, and so
    on) are enforced by :func:`build_wmma_fmha_swapqk`.
    """
    if arch != "gfx1151":
        return False, f"swapqk is a gfx1151 (RDNA3.5) kernel; got arch={arch!r}"
    # dual_gather pairs adjacent d-subtiles, so n_dk must be even.
    if cfg.head_size <= 0 or (cfg.dual_gather and cfg.head_size % 32 != 0):
        return (
            False,
            f"head_size must be a positive multiple of 32 (got {cfg.head_size})",
        )
    if cfg.head_size % 16 != 0:
        return (
            False,
            f"head_size must be a positive multiple of 16 (got {cfg.head_size})",
        )
    # The backend stops batching the buffer gather below block_n=32, which
    # collapses throughput.
    if cfg.block_n % 16 != 0 or (cfg.buffer_gather and cfg.block_n < 32):
        return (
            False,
            f"block_n must be a multiple of 16 and at least 32 (got {cfg.block_n})",
        )
    if cfg.n_waves not in (1, 2):
        return False, f"n_waves must be 1 or 2 (got {cfg.n_waves})"
    if cfg.mask_mode not in ("none", "causal"):
        return False, f"mask_mode must be 'none' or 'causal' (got {cfg.mask_mode!r})"
    return True, ""


def swapqk_transpose_v(v, kblock: int = 0):
    """Relay V for ``cfg.v_transposed``: [B, S, H, D] -> [B, H, D, S].

    With ``kblock=KB`` (matching ``cfg.v_kblock``) the result is the key-blocked
    [B, H, S/KB, D, KB] form instead. Either way the head stride becomes ``D * S``
    elements (it is ``D`` in the default layout), which matters when a caller
    offsets the V pointer for a head-chunked launch.
    """
    import numpy as np

    if not kblock:
        return np.ascontiguousarray(np.transpose(v, (0, 2, 3, 1)))
    bsz, s, h, d = v.shape
    if s % kblock:
        raise ValueError(f"seqlen {s} must be a multiple of v_kblock {kblock}")
    blocked = v.reshape(bsz, s // kblock, kblock, h, d).transpose(0, 3, 1, 4, 2)
    return np.ascontiguousarray(blocked)


def swapqk_grid(cfg: SwapQKCfg, *, seqlen_q: int, batch: int):
    q_per = cfg.q_rows_per_cta
    if seqlen_q % q_per != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {q_per}")
    return (seqlen_q // q_per, cfg.num_query_heads, batch)


def swapqk_num_work_items(cfg: SwapQKCfg, *, seqlen_q: int, batch: int) -> int:
    """Total (query-block, head, batch) work-items for a persistent launch."""
    q_per = cfg.q_rows_per_cta
    if seqlen_q % q_per != 0:
        raise ValueError(f"seqlen_q {seqlen_q} must be a multiple of {q_per}")
    return (seqlen_q // q_per) * cfg.num_query_heads * batch


def swapqk_persistent_grid(cfg: SwapQKCfg):
    """Fixed 1-D launch grid for num_persistent>0 (independent of problem size)."""
    return (cfg.num_persistent, 1, 1)


def _declare_params(b: IRBuilder, *, persistent: bool = False):
    P = {}
    P["Q"] = b.param("Q", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["K"] = b.param("K", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["V"] = b.param("V", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    P["O"] = b.param(
        "O", PtrType(F16, "global"), noalias=True, writeonly=True, align=16
    )
    if persistent:
        # Work-queue counter: one i32 global slot the host pre-clears to 0 before
        # EACH launch. Declared only when persistent, so the non-persistent arg
        # pack (and every existing harness) keeps its ABI.
        P["Counter"] = b.param(
            "Counter", PtrType(I32, "global"), noalias=True, align=16
        )
    P["scale_log2"] = b.param("scale_log2", F32)
    P["seqlen_q"] = b.param("seqlen_q", I32)
    P["seqlen_k"] = b.param("seqlen_k", I32)
    for nm in (
        "stride_q_token",
        "stride_q_head",
        "stride_k_token",
        "stride_k_head",
        "stride_v_token",
        "stride_v_head",
        "stride_o_token",
        "stride_o_head",
    ):
        P[nm] = b.param(nm, I32)
    return P


def build_wmma_fmha_swapqk(
    cfg: SwapQKCfg,
    arch: str = "gfx1151",
    *,
    seqlen_q: Optional[int] = None,
    batch: int = 1,
) -> KernelDef:
    """``seqlen_q``/``batch`` are only needed when ``cfg.num_persistent > 0``: the
    work-item count and the per-CTA drain bound are compile-time constants there."""
    ok, why = is_valid_spec(cfg, arch)
    if not ok:
        raise ValueError(why)
    atom = WmmaAtom.f16_16x16x16()
    wave = atom.wave_size  # 32
    c_map = atom.c_layout(arch)
    c_frag = atom.c_per_lane  # 8
    a_frag = atom.a_per_lane  # 16
    n_dk = cfg.head_size // 16
    hs = cfg.head_size
    W = cfg.n_waves
    dtype_ir = F16
    PERS = cfg.num_persistent > 0  # needed before the params are declared
    if PERS:
        if cfg.persist_decode not in ("qb_major", "batch_major"):
            raise ValueError(f"bad persist_decode {cfg.persist_decode!r}")
        if cfg.q_block > 1:
            raise ValueError(
                "num_persistent is implemented on the MQ==1 path (the q_block>1 path "
                "builds its own kernel and returns early); MQ=2 does not fit at D=128"
            )
        if cfg.q_lds or cfg.kv_lds:
            raise ValueError(
                "num_persistent is incompatible with q_lds/kv_lds: those allocate LDS "
                "inside what becomes the work-item loop"
            )
        if seqlen_q is None:
            raise ValueError(
                "num_persistent>0 needs seqlen_q at build time (work-item count and "
                "the per-CTA drain bound are baked in)"
            )
        if batch < 1:
            raise ValueError(f"batch must be >= 1, got {batch}")
        _n_batch = batch
        _num_tiles = swapqk_num_work_items(cfg, seqlen_q=seqlen_q, batch=batch)
        # worst-case per-CTA drain count; the in_range guard makes over-counting safe
        _max_iters = (_num_tiles + cfg.num_persistent - 1) // cfg.num_persistent

    b = IRBuilder(cfg.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = cfg.block_size
    if cfg.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = cfg.waves_per_eu
    p = _declare_params(b, persistent=PERS)

    c0 = b.const_i32(0)
    c16 = b.const_i32(16)
    c_wave = b.const_i32(wave)
    tid = b.thread_id_x()
    wave_id = b.div(tid, c_wave)
    lane = b.mod(tid, c_wave)
    col = b.mod(lane, c16)  # lane % 16  == query row within the 16-tile
    lane_lt16 = b.cmp_lt(lane, c16)

    if PERS:
        # ---- persistent work-queue (TOP-FETCH) ----
        # Each of the max_iters iterations does exactly ONE cooperative
        # atomic_add(1) at the TOP and processes the fetched tile iff it is in
        # range. Every CTA fetches exactly max_iters times, so total fetches =
        # num_persistent*max_iters >= num_tiles, and the counter is monotonic:
        # every id in [0, num_tiles) is handed to some iteration and processed
        # exactly once, REGARDLESS of load imbalance. Do not switch to
        # helpers.persistent.persistent_tile_loop, which fetches the NEXT tile at
        # the BOTTOM -- its final fetch per CTA is consumed-but-never-processed,
        # which under imbalance steals a live tile id and silently drops that
        # output tile.
        Counter = p["Counter"]
        c_one = b.const_i32(1)
        _multiwave = cfg.block_size > wave
        _brd = b.smem_alloc(I32, [1], name_hint="pers_brd") if _multiwave else None

        def _fetch_tile():
            """Cooperative atomic tile fetch, broadcast to every thread in the CTA."""
            is_lead = b.cmp_eq(tid, c0)
            if not _multiwave:
                # every lane issues the atomic (only lane 0 increments); ds_bpermute
                # broadcasts lane 0's result wave-internally, so no LDS/barrier and
                # no race (the optimiser elides a single-wave s_barrier).
                inc = b.select(is_lead, c_one, c0)
                return b.ds_bpermute(c0, b.global_atomic_add(Counter, c0, inc))
            with b.scf_if(is_lead):
                v = b.global_atomic_add(Counter, c0, c_one)
                b.smem_store_vN(_brd, [c0], v, 1)
            # LDS-only barrier publishes brd without a full vmcnt drain, so the
            # previous work-item's V-gathers / O-stores keep flowing across the
            # boundary instead of stalling the memory pipeline once per work-item.
            b.sync_lds_only()
            return b.vec_extract(b.smem_load_vN(_brd, c0, dtype=I32, n=1), 0)

        # The work-item body is the ENTIRE rest of the kernel (~1100 lines), so
        # these two scf regions are entered and exited MANUALLY instead of with
        # `with`: the equivalent `with` would re-indent the whole body and make the
        # diff unreviewable. _pers_close() at the bottom is the matching exit.
        _ploop = b.scf_for_iter(
            c0, b.const_i32(_max_iters), c_one, iter_args=[], iv_name="pers_iter"
        )
        _ploop.__enter__()
        _tile = _fetch_tile()
        _pif = b.scf_if(b.cmp_lt(_tile, b.const_i32(_num_tiles)))
        _pif.__enter__()

        c_nqb = b.const_i32(seqlen_q // cfg.q_rows_per_cta)
        c_qh = b.const_i32(cfg.num_query_heads)
        if cfg.persist_decode == "qb_major":
            # q_group fastest: a CTA draining adjacent ids stays on one
            # (head, batch) and reuses that head's K/V out of L2.
            q_group = b.mod(_tile, c_nqb)
            _hb = b.div(_tile, c_nqb)
            head = b.mod(_hb, c_qh)
            batch = b.div(_hb, c_qh)
        else:
            batch = b.mod(_tile, b.const_i32(_n_batch))
            _hq = b.div(_tile, b.const_i32(_n_batch))
            head = b.mod(_hq, c_qh)
            q_group = b.div(_hq, c_qh)

        def _pers_close():
            _pif.__exit__(None, None, None)
            _ploop.__exit__(None, None, None)

    else:
        q_group = b.block_id_x()
        head = b.block_id_y()
        batch = b.block_id_z()

        def _pers_close():
            return None

    qh, kvh = cfg.num_query_heads, cfg.kv_heads
    kv_head = head if kvh == qh else b.div(head, b.const_i32(qh // kvh))

    seqlen_q = p["seqlen_q"]
    seqlen_k = p["seqlen_k"]
    if cfg.static_shape:
        # strides are pure functions of the (build-time) head config -> constants.
        # Layout [.., token, head, dim]: token stride = n_heads*D, head stride = D.
        sq, sqh = b.const_i32(qh * hs), b.const_i32(hs)
        sk, skh = b.const_i32(kvh * hs), b.const_i32(hs)
        sv, svh = b.const_i32(kvh * hs), b.const_i32(hs)
        so, soh = b.const_i32(qh * hs), b.const_i32(hs)
    else:
        sq, sqh = p["stride_q_token"], p["stride_q_head"]
        sk, skh = p["stride_k_token"], p["stride_k_head"]
        sv, svh = p["stride_v_token"], p["stride_v_head"]
        so, soh = p["stride_o_token"], p["stride_o_head"]
    scale_log2 = p["scale_log2"]
    Q, K, V, O = p["Q"], p["K"], p["V"], p["O"]  # noqa: E741

    neg_inf = b.const_f32(-1e30)
    zero_f = b.const_f32(0.0)

    # Q/K: (head, token, dim), dim contiguous -> WMMA operands are contiguous d-slices.
    Q_view = make_global_view(
        Q, shape=(qh, 1, hs), dtype=dtype_ir, strides=(sqh, sq, 1)
    )
    K_view = make_global_view(
        K, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(skh, sk, 1)
    )
    V_view = make_global_view(
        V, shape=(kvh, 1, hs), dtype=dtype_ir, strides=(svh, sv, 1)
    )
    # O^T view: (head, dim, token) so store_wmma_tile's (row=d, col=query) lands on
    # O[query, d]. dim is contiguous (stride 1), token strided (so).
    O_T_view = make_global_view(
        O, shape=(qh, hs, 1), dtype=dtype_ir, strides=(soh, 1, so)
    )

    MQ = cfg.q_block  # query-blocking factor (16-row query tiles per wave)
    if MQ > 1 and cfg.pipeline:
        raise ValueError("q_block>1 is incompatible with pipeline")
    if cfg.v_transposed:
        if not cfg.buffer_gather:
            raise ValueError("v_transposed requires buffer_gather")
        if cfg.kv_lds:
            raise ValueError("v_transposed is incompatible with kv_lds")
        if cfg.v_kblock not in (0, 2, 4, 8):
            raise ValueError(f"v_kblock must be 0, 2, 4 or 8 (got {cfg.v_kblock})")
    elif cfg.v_kblock:
        raise ValueError("v_kblock requires v_transposed")
    if cfg.bcast_group < 0:
        raise ValueError(f"bcast_group must be >= 0, got {cfg.bcast_group}")
    if cfg.bcast_group and not cfg.dual_gather:
        raise ValueError(
            "bcast_group reorders the dual_gather broadcast; it is a no-op "
            "without dual_gather"
        )
    if cfg.k_dual:
        if not cfg.qk_douter:
            raise ValueError("k_dual requires qk_douter (pairs kv in its d-outer loop)")
        if (cfg.block_n // 16) % 2:
            raise ValueError(
                f"k_dual needs an even n_kv_sub (block_n>=32), got n_kv_sub="
                f"{cfg.block_n // 16} at block_n={cfg.block_n}"
            )
        if cfg.kv_lds:
            raise ValueError("k_dual reads K from global; incompatible with kv_lds")
    if cfg.qk_douter and MQ > 1:
        raise ValueError(
            "qk_douter applies to the MQ==1 QK loop; the q_block>1 path has its "
            "own K-hoisting loop (would silently be a no-op)"
        )
    if cfg.v_prefetch:
        if cfg.v_prefetch < 0:
            raise ValueError("v_prefetch must be >= 0")
        if not cfg.dual_gather:
            raise ValueError("v_prefetch requires dual_gather (see prefetch_v)")
        if cfg.pipeline:
            raise ValueError("v_prefetch is incompatible with pipeline")
    q_rows_per_cta = b.const_i32(cfg.q_rows_per_cta)
    cta_row0 = b.mul(q_group, q_rows_per_cta)
    # this wave owns MQ contiguous 16-row query tiles.
    wave_base = b.add(cta_row0, b.mul(wave_id, b.const_i32(16 * MQ)))
    batch_tok_q = b.mul(batch, seqlen_q)
    batch_tok_k = b.mul(batch, seqlen_k)
    # per query-group (g) row bases; MQ==1 reduces to the original single group.
    q_pos_base_g = [b.add(wave_base, b.const_i32(g * 16)) for g in range(MQ)]
    q_token_base_g = [b.add(qpb, batch_tok_q) for qpb in q_pos_base_g]
    # Q windows are loop-invariant (each group's 16 query rows); the B operand.
    qwin_g = [
        make_tile_window(Q_view, (1, 16, hs), origin=(head, qtb, c0))
        for qtb in q_token_base_g
    ]
    # single-group aliases (used by the MQ==1 fast path unchanged).
    q_pos_base = q_pos_base_g[0]
    q_token_base = q_token_base_g[0]
    qwin = qwin_g[0]

    pingpong = cfg.sched_mode == "pingpong"

    # ---- CK PermuteWarpGemmCToA: C-dist P (8 f16) -> operand fragment (16 f16). ----
    # query already sits on lane%16, so only kv needs the lane^16 reshuffle + 2x2
    # f16 interleave. Byte selectors are swapped for the upper 16 lanes (CK trick).
    sel0 = b.select(lane_lt16, b.const_i32(0x05040100), b.const_i32(0x01000504))
    sel1 = b.select(lane_lt16, b.const_i32(0x07060302), b.const_i32(0x03020706))

    def permx16_f32(v):
        return b.bitcast(b.permlanex16(b.bitcast(v, I32)), F32)

    def p_transpose_reg(ps):
        outs = []
        for m in range(c_frag // 2):
            lo = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m], dtype_ir), I16), I32)
            hi = b.zext(b.bitcast(b.cast_f32_to(ps[2 * m + 1], dtype_ir), I16), I32)
            v = b.lor(lo, b.shl(hi, c16))  # {kv 4m | kv 4m+2}  (own parity)
            w = b.permlanex16(v)  # partner (lane^16): other kv parity
            outs.append(b.perm_b32(w, v, sel0))  # {kv 4m,   4m+1}
            outs.append(b.perm_b32(w, v, sel1))  # {kv 4m+2, 4m+3}
        packed = b.vec_pack(outs, I32)
        return b.vec_bitcast(packed, VectorType(dtype_ir, a_frag))

    # ---- iter-args: m (scalar) | l (scalar) | acc (n_dk O^T tiles) ----
    # o_f16 carries acc as <c_frag x f16>; otherwise <c_frag x f32> (zero_acc).
    iter_args = [("m", neg_inf), ("l", zero_f)]
    for d in range(n_dk):
        acc0 = b.zero_vec(dtype_ir, c_frag) if cfg.o_f16 else atom.zero_acc(b)
        iter_args.append((f"acc{d}", acc0))

    def unpack(state):
        """Returns (m, l, acc_raw) where acc_raw are the raw carried vectors
        (f16 if o_f16, else f32). Callers wrap into WmmaTensor as needed."""
        m_i = state[0]
        l_i = state[1]
        acc_raw = list(state[2 : 2 + n_dk])
        return m_i, l_i, acc_raw

    block_n = cfg.block_n
    n_kv_sub = block_n // 16  # 16-wide kv WMMA sub-tiles per K-loop iteration
    c_block_n = b.const_i32(block_n)
    loop_stop = b.div(seqlen_k, c_block_n)
    if cfg.mask_mode == "causal":
        # CTA owns q rows up to cta_row0 + 16*W - 1; a kv block kt is needed iff
        # kt*block_n <= max q pos. Round up + 1 (over-inclusion is masked, safe).
        causal_stop = b.add(
            b.div(b.add(cta_row0, b.const_i32(16 * W)), c_block_n), b.const_i32(1)
        )
        loop_stop = b.select(b.cmp_lt(causal_stop, loop_stop), causal_stop, loop_stop)

    def k_window(k_tile_base):
        return make_tile_window(
            K_view, (1, 16, hs), origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0)
        )

    def k_window32(k_tile_base):
        """32-key K window: one 32-lane load spans TWO kv sub-tiles (see k_dual)."""
        return make_tile_window(
            K_view, (1, 32, hs), origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0)
        )

    def v_window(k_tile_base):
        return make_tile_window(
            V_view, (1, 16, hs), origin=(kv_head, b.add(batch_tok_k, k_tile_base), c0)
        )

    def gather_v_a_frag(vwin, d):
        # PV A-operand V[kv=0..15, d_col] (column gather, cache-resident).
        d_col = b.add(b.const_i32(d * 16), col)
        v_a = b.undef_vec(dtype_ir, a_frag)  # fully overwritten by the 16 loads
        for j in range(a_frag):
            v_a = b.vec_insert(v_a, vwin.load_scalar(b, c0, b.const_i32(j), d_col), j)
        return v_a

    # ---- buffer-descriptor gather (address in the memory unit, no VALU) ----
    c2 = b.const_i32(2)
    if cfg.buffer_gather:
        v_rsrc = b.buffer_rsrc(V, b.const_i32(0x7FFFFFFF))
        sv2 = b.mul(sv, c2)  # bytes per kv step (loop-invariant)
        soff_list = [b.mul(b.const_i32(j), sv2) for j in range(a_frag)]  # hoisted
        kvh_off = b.mul(kv_head, svh)  # element base for this (kv) head, per-CTA

    KB = cfg.v_kblock
    if cfg.v_transposed:
        # Both transposed forms put the head base at kv_head*(hs*S) and both are
        # addressed from k_base = batch*S + k_local (the [B,S,H,D] convention the
        # callers use), so the batch term needs scaling up by kvh*hs.
        v_t_cta = b.add(
            b.mul(kv_head, b.mul(b.const_i32(hs), seqlen_k)),
            b.mul(
                batch_tok_k,
                b.const_i32(hs * (kvh - 1) if KB else kvh * hs - 1),
            ),
        )
        c16b = b.const_i32(16)  # byte offset of the second dwordx4 (kv 8..15)

    def _load_col_transposed(k_base, d_col):
        """Contiguous-in-k V read: the lane's 16 keys are 32 consecutive bytes,
        so the whole A-fragment is 2 dwordx4. voffset is per-lane and fully
        loop-invariant (d_col and S don't move), so only the scalar soffset
        advances with the K-loop."""
        voff = b.mul(b.add(v_t_cta, b.mul(d_col, seqlen_k)), c2)
        soff = b.mul(k_base, c2)
        halves = [
            b.buffer_load_vN_f16(v_rsrc, voff, soff, 4),  # kv 0..7
            b.buffer_load_vN_f16(v_rsrc, voff, b.add(soff, c16b), 4),  # kv 8..15
        ]
        words = [
            b.vec_extract(b.vec_bitcast(h, VectorType(I32, 4)), i)
            for h in halves
            for i in range(4)
        ]
        return b.vec_bitcast(b.vec_pack(words, I32), VectorType(dtype_ir, a_frag))

    def _load_col_blocked(k_base, d_col):
        """V as [B, H, S/KB, D, KB]: KB keys contiguous per d, tiled along S.

        Keeps the full transpose's wide loads (KB keys in one dwordxN) but puts
        adjacent d only KB*2 bytes apart, so the 32 lanes of one instruction span
        32*KB*2 bytes instead of 32 separate S-strided rows. At KB=8 that is 512
        contiguous bytes -- 4 cache lines per instruction rather than 32.
        """
        n_load = a_frag // KB  # loads per 16-key fragment
        dwords = (KB * 2) // 4
        voff = b.mul(b.mul(d_col, b.const_i32(KB)), c2)  # per-lane, KB*2 B apart
        # k_base is a multiple of 16 and KB divides 16, so this tile index is exact.
        tile0 = b.div(k_base, b.const_i32(KB))
        base_e = b.add(v_t_cta, b.mul(tile0, b.const_i32(hs * KB)))
        parts = [
            b.buffer_load_vN_f16(
                v_rsrc,
                voff,
                b.mul(b.add(base_e, b.const_i32(i * hs * KB)), c2),
                dwords,
            )
            for i in range(n_load)
        ]
        words = [
            b.vec_extract(b.vec_bitcast(p, VectorType(I32, dwords)), i)
            for p in parts
            for i in range(dwords)
        ]
        return b.vec_bitcast(b.vec_pack(words, I32), VectorType(dtype_ir, a_frag))

    _load_v_wide = _load_col_blocked if KB else _load_col_transposed

    def gather_v_a_frag_buf(k_base, d):
        # k_base = batch_tok_k + k_block_base + ns*16 (uniform i32). Per-lane
        # voffset selects V[kv=0, d_col]; the buffer HW adds soffset = kv*stride_v.
        d_col = b.add(b.const_i32(d * 16), col)
        if cfg.v_transposed:
            return _load_v_wide(k_base, d_col)
        elem0 = b.add(b.add(kvh_off, b.mul(k_base, sv)), d_col)
        voff = b.mul(elem0, c2)  # bytes, per-lane
        v_a = b.undef_vec(dtype_ir, a_frag)  # fully overwritten by the 16 loads
        for j in range(a_frag):
            # D16 half-return load -> backend packs lo/hi (buffer_load_short_d16
            # /_d16_hi) like the flat global_load_d16 path, no v_mov_b16 pack.
            v_a = b.vec_insert(
                v_a, b.buffer_load_f16_d16(v_rsrc, voff, soff_list[j]), j
            )
        return v_a

    # ---- A1: dual-subtile half-packed gather (halves the V load count) ----
    n_i32 = a_frag // 2  # 8 dwords per <16 x f16> fragment

    def _load_col(k_base, vwin, d_col):
        """Gather V[kv=0..15, d_col] (per-lane d_col) via buffer or flat path.
        kv_lds forces the flat path so it reads the shared-LDS window."""
        if cfg.v_transposed:
            return _load_v_wide(k_base, d_col)
        if cfg.buffer_gather and not cfg.kv_lds:
            elem0 = b.add(b.add(kvh_off, b.mul(k_base, sv)), d_col)
            voff = b.mul(elem0, c2)

            if cfg.d16hi:
                # d16_hi buffer gather: buffer_load_d16_b16/_hi_b16 (inline asm)
                # pack each strided pair DIRECTLY into a VGPR lo/hi, killing the
                # ~64 v_mov_b16 the typed buffer_load_f16_d16 path emits. The asm
                # loads are outside the backend vmcnt model, so vmcnt0_fence ties
                # the dwords through a verbatim s_waitcnt vmcnt(0) barrier (else
                # the PV permute reads the fragment before the loads land -> NaN).
                dwords = [
                    b.buffer_load_d16_pack(
                        v_rsrc, voff, soff_list[2 * m], soff_list[2 * m + 1]
                    )
                    for m in range(a_frag // 2)
                ]
                dwords = b.vmcnt0_fence(dwords)
                return b.vec_bitcast(
                    b.vec_pack(dwords, I32), VectorType(dtype_ir, a_frag)
                )

            def _load(j):
                return b.buffer_load_f16_d16(v_rsrc, voff, soff_list[j])

        else:

            def _load(j):
                return vwin.load_scalar(b, c0, b.const_i32(j), d_col)

        v_a = b.undef_vec(dtype_ir, a_frag)  # fully overwritten by the 16 loads
        for j in range(a_frag):
            v_a = b.vec_insert(v_a, _load(j), j)
        return v_a

    def dual_gather_issue(k_base, vwin, d):
        """Issue only the loads for subtiles (d, d+1): lanes 0-15 fetch subtile d,
        lanes 16-31 subtile d+1. Returns the raw <16 x f16> (8 VGPRs), so a caller
        that wants several gathers in flight pays 8 registers per outstanding
        step instead of the 16 a finished fragment pair costs."""
        # per-lane d_col = (d + lane//16)*16 + lane%16  -> lo half=d, hi half=d+1
        d_col = b.add(b.const_i32(d * 16), b.add(b.mul(b.div(lane, c16), c16), col))
        return _load_col(k_base, vwin, d_col)

    def dual_gather_finish(loaded):
        """permlanex16 + select broadcast each subtile into both lane-halves (the
        layout the WMMA A-operand's lane^16 duplication requires)."""
        li = b.vec_bitcast(loaded, VectorType(I32, n_i32))
        fd, fd1 = [None] * n_i32, [None] * n_i32
        # Emit in groups of `bcast_group` dwords: all permlanes of the group, then
        # its subtile-d selects, then its subtile-d+1 selects. group==1 reproduces
        # the per-dword interleave exactly, so bcast_group 0 and 1 must compile to
        # identical code -- that equality is the check that this is a pure reorder.
        group = cfg.bcast_group if cfg.bcast_group else 1
        for lo in range(0, n_i32, group):
            idx = range(lo, min(lo + group, n_i32))
            e = [b.vec_extract(li, i) for i in idx]
            # value held by lane^16 (the other subtile)
            p = [b.permlanex16(x) for x in e]
            for j, i in enumerate(idx):
                fd[i] = b.select(lane_lt16, e[j], p[j])  # subtile d, both halves
            for j, i in enumerate(idx):
                fd1[i] = b.select(lane_lt16, p[j], e[j])  # subtile d+1, both halves
        frag_d = b.vec_bitcast(b.vec_pack(fd, I32), VectorType(dtype_ir, a_frag))
        frag_d1 = b.vec_bitcast(b.vec_pack(fd1, I32), VectorType(dtype_ir, a_frag))
        return frag_d, frag_d1

    def dual_gather(k_base, vwin, d):
        return dual_gather_finish(dual_gather_issue(k_base, vwin, d))

    def _tree(vals, op):
        # log-depth reduction (shorter loop-carried m/l critical path).
        while len(vals) > 1:
            nxt = [op(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
            if len(vals) % 2:
                nxt.append(vals[-1])
            vals = nxt
        return vals[0]

    ilp = max(1, cfg.qk_ilp)

    # q_hoist: load + pre-scale Q once (loop-invariant). QK output is then already
    # scaled, so the softmax drops the per-slot scale-mul.
    q_hoisted = None
    if cfg.q_hoist:
        scale_f16 = b.cast_f32_to(scale_log2, F16)
        scale_vec = b.zero_vec(dtype_ir, a_frag)
        for i in range(a_frag):
            scale_vec = b.vec_insert(scale_vec, scale_f16, i)
        q_hoisted = []
        for d in range(n_dk):
            qf = load_wmma_tile(
                b, qwin, atom, lane, role="b", k_offset=d * 16, lead=[c0]
            )
            q_hoisted.append(
                WmmaTensor(atom, "b", b.vector_mul(qf.value, scale_vec), arch)
            )

    # q_lds: stage this wave's MQ*16 PRE-SCALED query rows into a PER-WAVE LDS
    # slab once, then the QK re-reads Q from LDS instead of re-fetching it from
    # global every K-tile. Because each wave stages + reads ONLY its own slab
    # (Q_lds[wave_id]), the publish is a per-wave ``s_waitcnt(lgkmcnt=0)`` -- NO
    # cross-wave block barrier (nothing to serialize the waves / fight pingpong).
    # MQ-aware: group g's 16 rows live at Q_lds[wave_id, g*16 : g*16+16].
    Q_lds = None
    if cfg.q_lds:
        _QPAD = 8  # f16 bank-pad on the d-row (QK reads consecutive query rows)
        Q_lds = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(W, MQ * 16, hs),
            strides=(MQ * 16 * (hs + _QPAD), hs + _QPAD, 1),
            name_hint="Qsh",
        )
        _sf16 = b.cast_f32_to(scale_log2, F16)
        _sv8 = b.zero_vec(dtype_ir, 8)
        for _i in range(8):
            _sv8 = b.vec_insert(_sv8, _sf16, _i)
        _chunks = (16 * hs) // (wave * 8)  # vec8 chunks/lane per 16-row group
        for _g in range(MQ):
            for _i in range(_chunks):
                _c = b.add(lane, b.const_i32(_i * wave))
                _base = b.mul(_c, b.const_i32(8))
                _row = b.div(_base, b.const_i32(hs))
                _colc = b.mod(_base, b.const_i32(hs))
                _v8 = Q_view.load_vec(
                    b, [head, b.add(q_token_base_g[_g], _row), _colc], n=8
                )
                Q_lds.store_vec(
                    b,
                    [wave_id, b.add(b.const_i32(_g * 16), _row), _colc],
                    b.vector_mul(_v8, _sv8),
                    8,
                )
        b.s_waitcnt(lgkmcnt=0)  # intra-wave: this wave reads only its own Q_lds slab

    def q_lds_read(d, g=0):
        row = b.add(b.const_i32(g * 16), col)
        lo = Q_lds.load_vec(b, [wave_id, row, b.const_i32(d * 16)], n=8)
        hi = Q_lds.load_vec(b, [wave_id, row, b.const_i32(d * 16 + 8)], n=8)
        return WmmaTensor(atom, "b", b.vec_concat(lo, hi), arch)

    # ---- kv_lds: cooperative K/V tile staging in shared LDS (large-L prototype) ----
    K_lds = V_lds = None
    if cfg.kv_lds:
        _KVPAD = 8  # bank-pad on the d row (K/V read consecutive d per token)
        _kv_strides = (block_n * (hs + _KVPAD), hs + _KVPAD, 1)
        K_lds = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(1, block_n, hs),
            strides=_kv_strides,
            name_hint="Ksh",
        )
        V_lds = make_lds_view(
            b,
            dtype=dtype_ir,
            shape=(1, block_n, hs),
            strides=_kv_strides,
            name_hint="Vsh",
        )
        _nthreads = wave * W
        _tot = block_n * hs
        if _tot % (_nthreads * 8) != 0:
            raise ValueError(
                f"kv_lds coop loader needs block_n*hs ({_tot}) divisible by "
                f"n_threads*8 ({_nthreads * 8}); block_n={block_n} hs={hs} W={W}"
            )
        _kv_chunks = _tot // (_nthreads * 8)
        _c8 = b.const_i32(8)
        _c_hs = b.const_i32(hs)

    def coop_load_kv(k_block_base):
        """All W waves cooperatively stream this K-tile's K and V (block_n x hs)
        from global -> shared LDS once; every wave then reads from LDS. Two
        barriers/tile: before overwrite (prev readers done) + after store (tile
        visible to all waves)."""
        b.sync_lds_only()  # prev iter's LDS readers finish before we overwrite
        kbase_tok = b.add(batch_tok_k, k_block_base)
        for i in range(_kv_chunks):
            c = b.add(tid, b.const_i32(i * (wave * W)))
            base = b.mul(c, _c8)
            row = b.div(base, _c_hs)
            colc = b.mod(base, _c_hs)
            gtok = b.add(kbase_tok, row)
            k8 = K_view.load_vec(b, [kv_head, gtok, colc], n=8)
            K_lds.store_vec(b, [c0, row, colc], k8, 8)
            v8 = V_view.load_vec(b, [kv_head, gtok, colc], n=8)
            V_lds.store_vec(b, [c0, row, colc], v8, 8)
        b.sync_lds_only()  # freshly-staged tile visible to all waves

    def k_lds_read(ns, d):
        # WMMA "a" fragment = 16 consecutive d-values at row = kv = ns*16 + lane%16.
        # LDS ds_read caps at vec8, so read it as 2 vec8 + concat (cf. q_lds_read).
        row = b.add(b.const_i32(ns * 16), col)
        lo = K_lds.load_vec(b, [c0, row, b.const_i32(d * 16)], n=8)
        hi = K_lds.load_vec(b, [c0, row, b.const_i32(d * 16 + 8)], n=8)
        return WmmaTensor(atom, "a", b.vec_concat(lo, hi), arch)

    def v_window_lds(ns):
        return make_tile_window(
            V_lds, (1, 16, hs), origin=(c0, b.const_i32(ns * 16), c0)
        )

    def _k_frag(kwin, ns, d):
        if cfg.kv_lds:
            return k_lds_read(ns, d)  # K from shared LDS (2x vec8)
        return load_wmma_tile(b, kwin, atom, lane, role="a", k_offset=d * 16, lead=[c0])

    def _k_frag_dual(k_block_base, ns, d):
        """K for kv sub-tiles (ns, ns+1) out of ONE 32-lane load.

        load_wmma_fragment takes the A-operand row from lane%16, so lanes l and
        l+16 issue the SAME address: every K load runs at half lane efficiency,
        moving 32 KB of traffic for a 16 KB tile at bn64. Indexing by the full lane
        instead makes the upper half fetch the NEXT sub-tile's 16 keys, and the
        same permlanex16+select broadcast dual_gather already uses for V splits the
        pair back into two duplicated fragments. This is the K-side twin of
        dual_gather, and it trades the identical 24 VALU per 2 saved loads.
        """
        kwin = k_window32(b.add(k_block_base, b.const_i32(ns * 16)))
        raw = kwin.load_vec(b, c0, lane, b.const_i32(d * 16), n=a_frag)
        f0, f1 = dual_gather_finish(raw)
        return WmmaTensor(atom, "a", f0, arch), WmmaTensor(atom, "a", f1, arch)

    def _q_frag(d):
        if cfg.q_hoist:
            return q_hoisted[d]
        if cfg.q_lds:
            return q_lds_read(d)
        return load_wmma_tile(
            b,
            qwin,
            atom,
            lane,
            role="b",
            k_offset=d * 16,
            lead=[c0],
            nontemporal=cfg.q_nt,
        )

    def compute_qk(k_block_base):
        """S^T = K @ Q^T for all n_kv_sub sub-tiles -> list of score WmmaTensors."""
        if pingpong:
            b.s_setprio(1)
        subs = []
        if cfg.qk_douter:
            # d-outer / kv-inner: Q[d] is invariant in ns, so loading it in the
            # OUTER loop issues n_dk Q loads per K-tile instead of
            # n_kv_sub*n_dk. The n_kv_sub accumulator chains are mutually
            # independent, so they supply the ILP that qk_ilp exists to create
            # and no separate acc_ilp / tail reduction is needed.
            kwins = (
                [None] * n_kv_sub
                if cfg.kv_lds or cfg.k_dual
                else [
                    k_window(b.add(k_block_base, b.const_i32(ns * 16)))
                    for ns in range(n_kv_sub)
                ]
            )
            subs = [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(n_kv_sub)]
            for d in range(n_dk):
                q_tile = _q_frag(d)
                if cfg.k_dual:
                    # one load per kv PAIR instead of one per sub-tile
                    for ns in range(0, n_kv_sub, 2):
                        k0, k1 = _k_frag_dual(k_block_base, ns, d)
                        subs[ns] = wmma_mma(b, k0, q_tile, subs[ns])
                        subs[ns + 1] = wmma_mma(b, k1, q_tile, subs[ns + 1])
                    continue
                for ns in range(n_kv_sub):
                    subs[ns] = wmma_mma(b, _k_frag(kwins[ns], ns, d), q_tile, subs[ns])
        else:
            for ns in range(n_kv_sub):
                kwin = None
                if not cfg.kv_lds:
                    kwin = k_window(b.add(k_block_base, b.const_i32(ns * 16)))
                acc_ilp = [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(ilp)]
                for d in range(n_dk):
                    acc_ilp[d % ilp] = wmma_mma(
                        b, _k_frag(kwin, ns, d), _q_frag(d), acc_ilp[d % ilp]
                    )
                sc = acc_ilp[0]
                for si in range(1, ilp):
                    sc = WmmaTensor(
                        atom, "c", b.vector_add(sc.value, acc_ilp[si].value), arch
                    )
                subs.append(sc)
        if pingpong:
            b.s_setprio(0)
        return subs

    # ================= q_block (MQ>1): query-blocked, shared K/V loads =========
    # Self-contained path (eager rescale, f32 O, dual-gather). Reuses the same
    # helper closures; only the state is MQ-group-indexed and the K (QK) / V (PV)
    # fragment loads are hoisted so all MQ groups share them -> KV DRAM amortized
    # MQ x. MQ==1 falls through to the tuned single-group code below untouched.
    if MQ > 1:
        _exp2 = b.exp2_fast if cfg.fast_exp2 else b.exp2
        gs = 2 + n_dk  # iter-arg stride per group: m, l, n_dk O tiles
        qb_iter = []
        for g in range(MQ):
            qb_iter.append((f"qm{g}", neg_inf))
            qb_iter.append((f"ql{g}", zero_f))
            for d in range(n_dk):
                o0 = b.zero_vec(dtype_ir, c_frag) if cfg.o_f16 else atom.zero_acc(b)
                qb_iter.append((f"qo{g}_{d}", o0))
        kloop = b.scf_for_iter(
            b.const_i32(0), loop_stop, b.const_i32(1), iter_args=qb_iter, iv_name="kt"
        )
        with kloop as (kt, state):
            m_i = [state[g * gs] for g in range(MQ)]
            l_i = [state[g * gs + 1] for g in range(MQ)]
            accs = [list(state[g * gs + 2 : g * gs + 2 + n_dk]) for g in range(MQ)]
            k_block_base = b.mul(kt, c_block_n)
            k_bases = [
                b.add(b.add(batch_tok_k, k_block_base), b.const_i32(ns * 16))
                for ns in range(n_kv_sub)
            ]
            vwins = [
                v_window(b.add(k_block_base, b.const_i32(ns * 16)))
                for ns in range(n_kv_sub)
            ]

            # ---- QK: load each K fragment ONCE, reuse across MQ query groups ----
            if pingpong:
                b.s_setprio(1)
            subs = [[None] * n_kv_sub for _ in range(MQ)]
            for ns in range(n_kv_sub):
                kwin = k_window(b.add(k_block_base, b.const_i32(ns * 16)))
                acc = [
                    [WmmaTensor.zero_acc(b, atom, arch=arch) for _ in range(ilp)]
                    for _ in range(MQ)
                ]
                for d in range(n_dk):
                    k_tile = load_wmma_tile(
                        b, kwin, atom, lane, role="a", k_offset=d * 16, lead=[c0]
                    )
                    for g in range(MQ):
                        if cfg.q_lds:
                            q_tile = q_lds_read(d, g)
                        else:
                            q_tile = load_wmma_tile(
                                b,
                                qwin_g[g],
                                atom,
                                lane,
                                role="b",
                                k_offset=d * 16,
                                lead=[c0],
                                nontemporal=cfg.q_nt,
                            )
                        acc[g][d % ilp] = wmma_mma(b, k_tile, q_tile, acc[g][d % ilp])
                for g in range(MQ):
                    sc = acc[g][0]
                    for si in range(1, ilp):
                        sc = WmmaTensor(
                            atom, "c", b.vector_add(sc.value, acc[g][si].value), arch
                        )
                    subs[g][ns] = sc
            if pingpong:
                b.s_setprio(0)

            # ---- per-group online softmax + register P-transpose ----
            p_tiles = [None] * MQ
            alpha_vec = [None] * MQ
            m_new = [None] * MQ
            l_new = [None] * MQ
            for g in range(MQ):
                s_sub = []
                for ns in range(n_kv_sub):
                    kv_base = b.add(k_block_base, b.const_i32(ns * 16))
                    row = []
                    for i in range(c_frag):
                        kv_rel, q_rel = subs[g][ns].coord(b, lane, i)
                        s_i = subs[g][ns].slot(b, i)
                        if not cfg.q_lds:  # else scale pre-baked into the LDS Q
                            s_i = b.fmul(s_i, scale_log2)
                        s_i = apply_attention_mask(
                            b,
                            s_i,
                            mask_mode=cfg.mask_mode,
                            k_idx=b.add(kv_base, kv_rel),
                            query_pos=b.add(q_pos_base_g[g], q_rel),
                            sliding_window=0,
                        )
                        row.append(s_i)
                    s_sub.append(row)
                all_s = [v for r in s_sub for v in r]
                local_max = _tree(list(all_s), b.fmax)
                tile_max = b.fmax(local_max, permx16_f32(local_max))
                mn = b.fmax(m_i[g], tile_max)
                al = _exp2(b.fsub(m_i[g], mn))
                ps = [
                    [_exp2(b.fsub(s_sub[ns][i], mn)) for i in range(c_frag)]
                    for ns in range(n_kv_sub)
                ]
                all_p = [v for r in ps for v in r]
                local_sum = _tree(list(all_p), b.fadd)
                tile_sum = b.fadd(local_sum, permx16_f32(local_sum))
                m_new[g] = mn
                l_new[g] = b.fadd(b.fmul(l_i[g], al), tile_sum)
                av = b.zero_vec_f32(c_frag)
                for i in range(c_frag):
                    av = b.vec_insert(av, al, i)
                alpha_vec[g] = av
                p_tiles[g] = [
                    WmmaTensor(atom, "b", p_transpose_reg(ps[ns]), arch)
                    for ns in range(n_kv_sub)
                ]

            # ---- PV: rescale O[g] by alpha[g], then share each V fragment across groups ----
            if pingpong:
                b.s_setprio(1)
            new_accs = [[None] * n_dk for _ in range(MQ)]
            if cfg.o_f16:
                # d-pair-outer: only the current d-pair is upgraded to f32 (per
                # group); carried O stays f16 -> halves the live O register peak,
                # which is what lets a higher MQ fit at D64/D128.
                for dp in range(0, n_dk, 2):
                    t0 = [
                        WmmaTensor(
                            atom,
                            "c",
                            b.vector_mul(b.vec_ext_to_f32(accs[g][dp]), alpha_vec[g]),
                            arch,
                        )
                        for g in range(MQ)
                    ]
                    t1 = [
                        WmmaTensor(
                            atom,
                            "c",
                            b.vector_mul(
                                b.vec_ext_to_f32(accs[g][dp + 1]), alpha_vec[g]
                            ),
                            arch,
                        )
                        for g in range(MQ)
                    ]
                    for ns in range(n_kv_sub):
                        frag_d, frag_d1 = dual_gather(k_bases[ns], vwins[ns], dp)
                        a0 = WmmaTensor(atom, "a", frag_d, arch)
                        a1 = WmmaTensor(atom, "a", frag_d1, arch)
                        for g in range(MQ):
                            t0[g] = wmma_mma(b, a0, p_tiles[g][ns], t0[g])
                            t1[g] = wmma_mma(b, a1, p_tiles[g][ns], t1[g])
                    for g in range(MQ):
                        new_accs[g][dp] = b.vec_trunc_f32_to_f16(t0[g].value)
                        new_accs[g][dp + 1] = b.vec_trunc_f32_to_f16(t1[g].value)
            else:
                new_accs = [
                    [
                        WmmaTensor(atom, "c", accs[g][d], arch).scale(b, alpha_vec[g])
                        for d in range(n_dk)
                    ]
                    for g in range(MQ)
                ]
                for ns in range(n_kv_sub):
                    for dp in range(0, n_dk, 2):
                        frag_d, frag_d1 = dual_gather(k_bases[ns], vwins[ns], dp)
                        a0 = WmmaTensor(atom, "a", frag_d, arch)
                        a1 = WmmaTensor(atom, "a", frag_d1, arch)
                        for g in range(MQ):
                            new_accs[g][dp] = wmma_mma(
                                b, a0, p_tiles[g][ns], new_accs[g][dp]
                            )
                            new_accs[g][dp + 1] = wmma_mma(
                                b, a1, p_tiles[g][ns], new_accs[g][dp + 1]
                            )
            if pingpong:
                b.s_setprio(0)

            yields = []
            for g in range(MQ):
                yields.append(m_new[g])
                yields.append(l_new[g])
                if cfg.o_f16:
                    yields.extend(new_accs[g])  # already raw f16 values
                else:
                    yields.extend(a.value for a in new_accs[g])
            b.scf_yield(*yields)

        res = kloop.results
        for g in range(MQ):
            l_f = res[g * gs + 1]
            inv_l = b.select(b.fcmp("oeq", l_f, zero_f), zero_f, b.rcp(l_f))

            def _rescale(bld, val, slot, row, colv, _inv=inv_l):
                return bld.fmul(val, _inv)

            accs_g = res[g * gs + 2 : g * gs + 2 + n_dk]
            for d in range(n_dk):
                owin = make_tile_window(
                    O_T_view,
                    (1, 16, 16),
                    origin=(head, b.const_i32(d * 16), q_token_base_g[g]),
                )
                ov = b.vec_ext_to_f32(accs_g[d]) if cfg.o_f16 else accs_g[d]
                store_wmma_tile(
                    b,
                    owin,
                    WmmaTensor(atom, "c", ov, arch),
                    lane,
                    col_offset=0,
                    lead=[c0],
                    align=2,
                    transform=_rescale,
                    nontemporal=cfg.o_nt,
                )
        b.ret()
        return b.kernel

    if cfg.pipeline:
        # prologue: tile 0's QK, carried as iter-args (current-tile scores).
        for ns, sc in enumerate(compute_qk(c0)):
            iter_args.append((f"sc{ns}", sc.value))

    kloop = b.scf_for_iter(
        b.const_i32(0), loop_stop, b.const_i32(1), iter_args=iter_args, iv_name="kt"
    )
    with kloop as (kt, state):
        m_i, l_i, accs = unpack(state)
        k_block_base = b.mul(kt, c_block_n)
        if cfg.iglp >= 0:
            b.iglp_opt(cfg.iglp)

        # kv_lds: cooperatively stage this K-tile's K and V into shared LDS
        # BEFORE the QK/PV read them (all W waves then share the one copy).
        if cfg.kv_lds:
            coop_load_kv(k_block_base)

        # ---- QK: S^T = K @ Q^T. Pipelined -> consume the carried current-tile
        # scores and issue the NEXT tile's QK now (overlaps this tile's softmax/
        # P-transpose/PV, keeping the WMMA unit fed). Non-pipelined -> compute inline.
        next_subs = None
        if cfg.pipeline:
            sub_scores = [
                WmmaTensor(atom, "c", state[2 + n_dk + ns], arch)
                for ns in range(n_kv_sub)
            ]
            kt_n = b.add(kt, b.const_i32(1))
            kt_n = b.select(b.cmp_ge(kt_n, loop_stop), kt, kt_n)
            next_subs = compute_qk(b.mul(kt_n, c_block_n))
        else:
            sub_scores = compute_qk(k_block_base)
        if pingpong:
            b.s_setprio(0)

        # ---- flattened PV step order + V gather dispatch (flat window | buffer SRD) ----
        # kv_lds sources V from the shared-LDS tile via the flat window path
        # (buffer_gather is a global-only lever, so it's bypassed here).
        if cfg.kv_lds:
            vwins = [v_window_lds(ns) for ns in range(n_kv_sub)]
        else:
            vwins = [
                v_window(b.add(k_block_base, b.const_i32(ns * 16)))
                for ns in range(n_kv_sub)
            ]
        k_bases = [
            b.add(b.add(batch_tok_k, k_block_base), b.const_i32(ns * 16))
            for ns in range(n_kv_sub)
        ]

        def do_gather(ns, d):
            if cfg.buffer_gather and not cfg.kv_lds:
                return gather_v_a_frag_buf(k_bases[ns], d)
            return gather_v_a_frag(vwins[ns], d)

        pv_steps = [(ns, d) for ns in range(n_kv_sub) for d in range(n_dk)]

        # PREFETCH: issue the first V gather BEFORE the softmax so its strided
        # loads overlap the softmax VALU (memory<->VALU overlap).
        v_next = None
        if cfg.prefetch_v:
            ns0, d0 = pv_steps[0]
            v_next = do_gather(ns0, d0)

        # ---- online softmax over ALL block_n keys (n_kv_sub*8 in-lane slots + 1 permlanex16) ----
        s_sub = []  # s_sub[ns] = list of c_frag scaled+masked scores
        for ns in range(n_kv_sub):
            kv_base = b.add(k_block_base, b.const_i32(ns * 16))
            row = []
            for i in range(c_frag):
                kv_rel, q_rel = sub_scores[ns].coord(b, lane, i)  # (row=kv, col=query)
                s_i = sub_scores[ns].slot(b, i)
                if not (cfg.q_hoist or cfg.q_lds):  # else scale pre-baked into Q
                    s_i = b.fmul(s_i, scale_log2)
                s_i = apply_attention_mask(
                    b,
                    s_i,
                    mask_mode=cfg.mask_mode,
                    k_idx=b.add(kv_base, kv_rel),
                    query_pos=b.add(q_pos_base, q_rel),
                    sliding_window=0,
                )
                row.append(s_i)
            s_sub.append(row)

        all_s = [v for row in s_sub for v in row]
        local_max = _tree(list(all_s), b.fmax)
        tile_max = b.fmax(local_max, permx16_f32(local_max))
        # lazy: if every lane's tile_max is within threshold of m_i, don't
        # re-anchor (m_new = m_i -> alpha = 1) and skip the O rescale below.
        skip_rescale = None
        if cfg.lazy_rescale:
            below = b.select(
                b.fcmp(
                    "ole", b.fsub(tile_max, m_i), b.const_f32(_LAZY_RESCALE_THRESHOLD)
                ),
                b.const_i32(1),
                c0,
            )
            skip_rescale = b.cmp_ne(b.wave_all(below), c0)  # wave-uniform i1
            m_new = b.select(skip_rescale, m_i, b.fmax(m_i, tile_max))
        else:
            m_new = b.fmax(m_i, tile_max)
        _exp2 = b.exp2_fast if cfg.fast_exp2 else b.exp2
        alpha = _exp2(b.fsub(m_i, m_new))
        ps_sub = [
            [_exp2(b.fsub(s_sub[ns][i], m_new)) for i in range(c_frag)]
            for ns in range(n_kv_sub)
        ]
        all_p = [v for row in ps_sub for v in row]
        local_sum = _tree(list(all_p), b.fadd)
        tile_sum = b.fadd(local_sum, permx16_f32(local_sum))
        l_new = b.fadd(b.fmul(l_i, alpha), tile_sum)

        # ---- alpha (rescale factor) + P operand tiles (both O-carry paths) ----
        alpha_vec = b.zero_vec_f32(c_frag)
        for i in range(c_frag):
            alpha_vec = b.vec_insert(alpha_vec, alpha, i)
        p_tiles = [
            WmmaTensor(atom, "b", p_transpose_reg(ps_sub[ns]), arch)
            for ns in range(n_kv_sub)
        ]

        def pv_pipelined(steps, consume):
            """Run the PV steps with ``v_prefetch`` V-gathers outstanding.

            The loads for step i+depth are issued BEFORE step i's permute and
            WMMAs, so the s_waitcnt for the fragment in hand does not also gate
            issuing the next requests. That raises requests-in-flight per wave,
            which is the lever when the kernel is memory-latency bound rather
            than issue- or bandwidth-bound (see v_kblock).
            """
            depth = min(max(1, cfg.v_prefetch), len(steps))
            fifo = [
                dual_gather_issue(k_bases[ns], vwins[ns], dp)
                for ns, dp in steps[:depth]
            ]
            for i, (ns, dp) in enumerate(steps):
                raw = fifo.pop(0)
                if i + depth < len(steps):
                    n1, d1 = steps[i + depth]
                    fifo.append(dual_gather_issue(k_bases[n1], vwins[n1], d1))
                fd, fd1 = dual_gather_finish(raw)
                consume(ns, dp, fd, fd1)

        if cfg.o_f16:
            # f16-carry, d-pair-outer PV: upgrade one d-pair's f16 carry to f32,
            # fuse the alpha rescale, accumulate BOTH kv sub-tiles, truncate back
            # to f16. Only the current d-pair is f32 -> small O register peak.
            if pingpong:
                b.s_setprio(1)
            new_acc_vals = [None] * n_dk
            if cfg.v_prefetch and cfg.dual_gather:
                # d-pair outer, kv inner (same order as the loop below), so a
                # d-pair's f32 upgrade stays live only across its own kv steps.
                pair = {}

                def _consume_f16(ns, dp, fd, fd1):
                    if ns == 0:
                        pair[dp] = (
                            WmmaTensor(
                                atom,
                                "c",
                                b.vector_mul(b.vec_ext_to_f32(accs[dp]), alpha_vec),
                                arch,
                            ),
                            WmmaTensor(
                                atom,
                                "c",
                                b.vector_mul(b.vec_ext_to_f32(accs[dp + 1]), alpha_vec),
                                arch,
                            ),
                        )
                    t0, t1 = pair[dp]
                    t0 = wmma_mma(b, WmmaTensor(atom, "a", fd, arch), p_tiles[ns], t0)
                    t1 = wmma_mma(b, WmmaTensor(atom, "a", fd1, arch), p_tiles[ns], t1)
                    pair[dp] = (t0, t1)
                    if ns == n_kv_sub - 1:
                        new_acc_vals[dp] = b.vec_trunc_f32_to_f16(t0.value)
                        new_acc_vals[dp + 1] = b.vec_trunc_f32_to_f16(t1.value)

                pv_pipelined(
                    [(ns, dp) for dp in range(0, n_dk, 2) for ns in range(n_kv_sub)],
                    _consume_f16,
                )
                if pingpong:
                    b.s_setprio(0)
            else:
                for dp in range(0, n_dk, 2):
                    t0 = WmmaTensor(
                        atom,
                        "c",
                        b.vector_mul(b.vec_ext_to_f32(accs[dp]), alpha_vec),
                        arch,
                    )
                    t1 = WmmaTensor(
                        atom,
                        "c",
                        b.vector_mul(b.vec_ext_to_f32(accs[dp + 1]), alpha_vec),
                        arch,
                    )
                    for ns in range(n_kv_sub):
                        frag_d, frag_d1 = dual_gather(k_bases[ns], vwins[ns], dp)
                        t0 = wmma_mma(
                            b, WmmaTensor(atom, "a", frag_d, arch), p_tiles[ns], t0
                        )
                        t1 = wmma_mma(
                            b, WmmaTensor(atom, "a", frag_d1, arch), p_tiles[ns], t1
                        )
                    new_acc_vals[dp] = b.vec_trunc_f32_to_f16(t0.value)
                    new_acc_vals[dp + 1] = b.vec_trunc_f32_to_f16(t1.value)
                if pingpong:
                    b.s_setprio(0)
        else:
            accs_wt = [WmmaTensor(atom, "c", v, arch) for v in accs]
            # rescale the O^T accumulators by alpha ONCE per block_n keys.
            if cfg.lazy_rescale:
                # wave-uniform 0/1-trip loop: run the n_dk rescale muls only when
                # the max re-anchored (skip_rescale False) -> 0-trip skips them.
                n_res = b.select(skip_rescale, c0, b.const_i32(1))
                rloop = b.scf_for_iter(
                    c0,
                    n_res,
                    b.const_i32(1),
                    iter_args=[(f"ra{d}", accs_wt[d].value) for d in range(n_dk)],
                    iv_name="rsc",
                )
                with rloop as (_rsc, rstate):
                    out = [
                        WmmaTensor(atom, "c", rstate[d], arch).scale(b, alpha_vec).value
                        for d in range(n_dk)
                    ]
                    b.scf_yield(*out)
                new_accs = [WmmaTensor(atom, "c", v, arch) for v in rloop.results]
            else:
                new_accs = [accs_wt[d].scale(b, alpha_vec) for d in range(n_dk)]

            # ---- PV: O^T += V @ P per kv sub-tile (register P-transpose, no LDS) ----
            if pingpong:
                b.s_setprio(1)
            if cfg.dual_gather and cfg.v_prefetch:

                def _consume_f32(ns, dp, fd, fd1):
                    new_accs[dp] = wmma_mma(
                        b, WmmaTensor(atom, "a", fd, arch), p_tiles[ns], new_accs[dp]
                    )
                    new_accs[dp + 1] = wmma_mma(
                        b,
                        WmmaTensor(atom, "a", fd1, arch),
                        p_tiles[ns],
                        new_accs[dp + 1],
                    )

                pv_pipelined(
                    [(ns, dp) for ns in range(n_kv_sub) for dp in range(0, n_dk, 2)],
                    _consume_f32,
                )
            elif cfg.dual_gather:
                for ns in range(n_kv_sub):
                    for dp in range(0, n_dk, 2):
                        frag_d, frag_d1 = dual_gather(k_bases[ns], vwins[ns], dp)
                        new_accs[dp] = wmma_mma(
                            b,
                            WmmaTensor(atom, "a", frag_d, arch),
                            p_tiles[ns],
                            new_accs[dp],
                        )
                        new_accs[dp + 1] = wmma_mma(
                            b,
                            WmmaTensor(atom, "a", frag_d1, arch),
                            p_tiles[ns],
                            new_accs[dp + 1],
                        )
            elif cfg.prefetch_v:
                for idx, (ns, d) in enumerate(pv_steps):
                    v_cur = v_next
                    if idx + 1 < len(pv_steps):
                        n1, d1 = pv_steps[idx + 1]
                        v_next = do_gather(n1, d1)
                    new_accs[d] = wmma_mma(
                        b, WmmaTensor(atom, "a", v_cur, arch), p_tiles[ns], new_accs[d]
                    )
            else:
                for ns, d in pv_steps:
                    new_accs[d] = wmma_mma(
                        b,
                        WmmaTensor(atom, "a", do_gather(ns, d), arch),
                        p_tiles[ns],
                        new_accs[d],
                    )
            if pingpong:
                b.s_setprio(0)
            new_acc_vals = [a.value for a in new_accs]

        yields = [m_new, l_new, *new_acc_vals]
        if cfg.pipeline:
            yields.extend(s.value for s in next_subs)
        b.scf_yield(*yields)

    m_f, l_f, accs_f = unpack(kloop.results)

    # ---- Epilogue: O^T[d, query] -> O[query, d], rescaled by 1/l. ----
    l_safe = l_f
    zmask = b.fcmp("oeq", l_safe, zero_f)
    inv_l = b.select(zmask, zero_f, b.rcp(l_safe))

    def _rescale(bld, val, slot, row, colv, _inv=inv_l):
        return bld.fmul(val, _inv)

    for d in range(n_dk):
        owin = make_tile_window(
            O_T_view, (1, 16, 16), origin=(head, b.const_i32(d * 16), q_token_base)
        )
        acc_wt = WmmaTensor(
            atom,
            "c",
            b.vec_ext_to_f32(accs_f[d]) if cfg.o_f16 else accs_f[d],
            arch,
        )
        store_wmma_tile(
            b,
            owin,
            acc_wt,
            lane,
            col_offset=0,
            lead=[c0],
            align=2,
            transform=_rescale,
            nontemporal=cfg.o_nt,
        )
    _pers_close()
    b.ret()
    return b.kernel
