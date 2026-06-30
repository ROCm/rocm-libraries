# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 (CDNA3) tiled-2D unified-attention kernel (narrow-atom variant).

This is the gfx942 sibling of ``instances/gfx950/attention_tiled_2d.py``. It
mirrors the same online-softmax tiled algorithm but uses only ISA features that
exist on CDNA3, where the gfx950 kernel cannot run:

  * **MFMA atom**: gfx942 lacks the wide-K ``mfma_f32_16x16x32`` /
    ``mfma_f32_32x32x16`` f16/bf16 atoms. Both QK and PV are built on the
    **narrow ``16x16x16``** atom (selected up front via
    ``ArchTarget.mma.select_largest_k(..., k_max=16)``; the multi-arch policy
    forbids down-splitting a K=32 atom). The K-step is 16 head-dim elements per
    atom, so ``QK_K_ITERS = HD // 16`` and ``PV_K_ITERS = T // 16``. bf16 routes
    through ``mfma_16x16x16_for_dtype`` (the ``_1k`` intrinsic, valid on gfx942).
  * **V B-operand**: gfx942 has no ``ds_read_b64_tr_b16`` (``ds_read_*_tr_*``).
    The PV ``B`` operand is built from ordinary strided LDS loads that reproduce
    the exact per-lane layout ``ds_read_tr16_b64`` produces for a ``16x16x16``
    atom: lane ``l = 16*k_chunk + n`` (``k_chunk in 0..3``, ``n in 0..15``) holds
    ``V_lds[v_buf, k*16 + k_chunk*4 + j, n_tile*16 + n]`` for ``j in 0..3``. This
    is the dense ``helpers/mfma_attention.py`` precedent's B-operand index math.
  * **LDS budget**: the LDS gate derives capacity from
    ``ArchTarget.from_gfx(arch).lds_capacity_bytes`` (gfx942 → 65536 B), not a
    hardcoded constant. The formula is identical to gfx950's; gfx942 is
    tile-limited, not starved (the combo kernel runs ~24 KB on the decode shapes;
    only the largest T=128/HD=128 configs exceed 64 KB and the gate rejects them).

Scope on gfx942: the **default** QK→softmax→PV path only, for f16 and bf16. The
gfx950-only experimental flags (fp8 K/V cache, 32x32x16 migration, native-fp8
MFMA) are rejected up front because they depend on wide-K / transpose-read ISA
that gfx942 does not have. gfx950 keeps its own file, byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Optional, Tuple

from ...core.ir import (
    BF16,
    CACHE_ALL,
    CACHE_GLOBAL,
    CACHE_STREAM,
    F16,
    F32,
    FP8E4M3,
    I32,
    I64,
    IRBuilder,
    KernelDef,
    PtrType,
    NON_TEMPORAL,
    Type,
    Value,
    VectorType,
)
from ...helpers.atoms import MfmaAtom, make_c_warp_dstr_encoding
from ...helpers.attention import (
    apply_softcap_log2,
    binary_search_seq_idx,
    dequant_fp8x8_to_dtype,
    mfma_16x16x16_for_dtype,
    mfma_16x16x32_for_dtype,
    mfma_32x32x8_for_dtype,
    mfma_32x32x16_for_dtype,
    pv32_v_load_paired,
    warp_xor_reduce_max,
    warp_xor_reduce_max_32lane,
    warp_xor_reduce_sum,
    warp_xor_reduce_sum_32lane,
)
from ...helpers.distribution import make_static_tile_distribution
from ...helpers.layouts import TransposeLdsReader
from ...helpers.transforms import TensorDescriptor, embed, indirect, unmerge


MFMA_M = 16
MFMA_N = 16


# CK-Tile C-accumulator warp distribution for the 32x32x16 MFMA atom. The
# hand-rolled ``mfma_32x32x16_c_row/col`` lane arithmetic (in
# ``helpers/attention.py``) is exactly the ``calculate_x`` of CK Tile's
# ``CWarpDstrEncoding`` (``warp_gemm_attribute_mfma.hpp``): for one wavefront
# the lane splits as ``(m_blk, n) = (lane // 32, lane % 32)`` and the per-lane
# ``<16 x f32>`` accumulator slot ``i`` splits row-major over the two M Y dims
# ``(kCM0PerLane, kCM1PerLane) = (4, 4)``. The distribution's ``calculate_x``
# then yields ``row = (i // 4) * 8 + m_blk * 4 + (i % 4)`` and ``col = n`` --
# the same mapping the scalar helpers produce, but emitted by the tile-
# distribution algebra instead of open-coded ``div``/``mul``/``add`` (Phase-C
# "adopt the vocabulary" adoption; the C layout is dtype-independent so the
# f16 atom drives it for both fp16 and bf16). The 3D tiled kernel still imports
# the scalar helpers from ``helpers/attention.py`` directly; this distribution-
# driven form is local to the 2D kernel.
_C32_DIST = make_static_tile_distribution(
    make_c_warp_dstr_encoding(MfmaAtom.f16_32x32x16())
)


def _mfma_32x32_c_row(b, lane, elem_idx: int):
    """MFMA-local output row for a ``32x32x16`` C element ``elem_idx`` (0..15).

    Drives CK Tile's C-warp ``TileDistribution.calculate_x`` instead of the
    hand-rolled ``row = (i//4)*8 + (lane//32)*4 + (i%4)`` lane arithmetic.
    """
    if not (0 <= elem_idx < 16):
        raise ValueError(f"mfma_32x32x16 elem_idx must be 0..15, got {elem_idx}")
    m_blk = b.div(lane, b.const_i32(32))
    n = b.mod(lane, b.const_i32(32))
    y0 = b.const_i32(elem_idx // 4)
    y1 = b.const_i32(elem_idx % 4)
    row, _col = _C32_DIST.calculate_x(b, ys=[y0, y1], ps=[[m_blk, n]])
    return row


def _mfma_32x32_c_col(b, lane, n_tile32: int = 0):
    """MFMA-local output col for ``32x32x16`` C elements in N-tile ``n_tile32``.

    The per-lane column is the N coordinate of CK Tile's C-warp distribution
    (``lane % 32``); ``n_tile32 * 32`` is the 32-column tile base added on top.
    Every per-lane accumulator slot shares this column, so the element index
    does not enter the formula.
    """
    m_blk = b.div(lane, b.const_i32(32))
    n = b.mod(lane, b.const_i32(32))
    _row, col = _C32_DIST.calculate_x(
        b, ys=[b.const_i32(0), b.const_i32(0)], ps=[[m_blk, n]]
    )
    if n_tile32 == 0:
        return col
    return b.add(b.const_i32(n_tile32 * 32), col)


# Backwards-compatible aliases. The 3D tiled kernel currently imports these
# from this module; once that import is removed in a follow-up these aliases
# can go away too. Promoted helpers live in ``rocke.helpers.attention``.
_apply_softcap = apply_softcap_log2
_binary_search_seq_idx = binary_search_seq_idx
_mfma_16x16x16 = mfma_16x16x16_for_dtype
_mfma_16x16x32 = mfma_16x16x32_for_dtype
_mfma_32x32x8 = mfma_32x32x8_for_dtype
_mfma_32x32x16 = mfma_32x32x16_for_dtype
_warp_xor_reduce_max = warp_xor_reduce_max
_warp_xor_reduce_max_32lane = warp_xor_reduce_max_32lane
_warp_xor_reduce_sum = warp_xor_reduce_sum
_warp_xor_reduce_sum_32lane = warp_xor_reduce_sum_32lane


# AMDGPU sched.group.barrier instruction-class mask bits (T8 DSL-novel).
_SGB_MFMA = 0x008  # MFMA / WMMA
_SGB_DS_READ = 0x100  # ds_read (LDS load)


def _sched_group_pin_mfma_step(b, ds_reads: int, mfma_count: int, sgid: int) -> None:
    """Pin one K-step of the MFMA k-loop as a scheduler-ordered block."""
    b.sched_group_barrier(_SGB_DS_READ, int(ds_reads), int(sgid))
    b.sched_group_barrier(_SGB_MFMA, int(mfma_count), int(sgid))


@dataclass(frozen=True)
class UnifiedAttention2DTiledSpec:
    head_size: int
    block_size: int
    num_query_heads: int
    num_kv_heads: int
    dtype: str
    use_sinks: bool
    sliding_window: int
    has_softcap: bool
    use_alibi: bool = False
    use_qq_bias: bool = False
    num_seqs: int = 0
    # Number of wave64 warps per CTA. `BLOCK_M = num_warps * 16` rows are
    # processed per CTA, with each warp owning its own 16-row slice. The
    # online softmax stays per-warp (no cross-warp reduction); the savings
    # come from amortising the Q load, async K/V loads, P_lds publish, and
    # cshuffle epilogue across more lanes. Default `1` preserves the
    # original single-warp behaviour bit-for-bit.
    num_warps: int = 1
    # AMDGPU occupancy hint (``"amdgpu-waves-per-eu"``). Attention is
    # register-pressure-bound; setting this to 2 or 3 forces the
    # backend to tighten VGPR allocation in exchange for higher
    # occupancy. ``None`` keeps the LLVM heuristic.
    waves_per_eu: Optional[int] = None
    # FP8 K/V cache. When ``"fp8e4m3"``, the kernel takes K/V cache
    # pointers as ``ptr<fp8e4m3, global>`` (1 byte/element), uses a
    # sync per-thread load that emits ``cvt_fp8_to_f32 -> fmul k_scale
    # -> cast_f32_to_bf16`` and writes the working dtype (bf16) into
    # LDS. The async DMA path is disabled on this lane because
    # ``raw_ptr_buffer_load_lds`` cannot intercept the scale step. Q is
    # still passed in the working dtype (``self.dtype``), and the rest
    # of the kernel (MFMA, softmax, epilogue) is unchanged.
    kv_storage_dtype: Optional[str] = None
    # FP8 K-in-LDS path (ULP-identical to default, faster). When True
    # (and ``kv_storage_dtype='fp8e4m3'``) the kernel stages K as raw
    # fp8 in LDS instead of dequant-then-store-bf16. Specifically:
    #   - K_lds is allocated as fp8 -> 8 KB instead of 16 KB at T=64/HD=64
    #     (frees 8 KB LDS per CTA; HBM->LDS DMA bytes are halved too).
    #   - The K loader uses async DMA fp8 bytes -> fp8 K_lds (skips the
    #     sync per-thread global_load + fp8->f32 + fmul k_scale + cast
    #     bf16 + smem_store chain we ran before).
    #   - The QK MFMA reads <8 x fp8> from K_lds per lane and dequants
    #     IN-REGISTER via ``cvt_pk_f32_fp8x4`` + ``fmul(k_scale)`` +
    #     ``cvt_pk_bf16_f32`` to produce the same <8 x bf16> B operand
    #     a bf16 K_lds load would have produced. The MFMA itself is still
    #     ``mfma_f32_16x16x32_bf16`` -- Q stays bf16, no Q quantisation.
    #
    # Bit-identical to:
    #   - Our current default path (it stores bf16 K_lds = bf16(fp8 *
    #     k_scale) and then reads bf16; the new path reads fp8 and
    #     applies the same dequant before MFMA -- the bf16 fed to MFMA
    #     is the same).
    #   - Triton's `unified_attention.py` fp8 path
    #     (``K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype);
    #     S = qk_scale * tl.dot(Q, K)``).
    # ULP-identical wrt Triton 2D fp8 by construction.
    #
    # A previous incarnation of this flag also cast Q to fp8 and used the
    # native fp8 MFMA. That was abandoned: fp8e4m3 has only 3 mantissa
    # bits, so Q quantisation produced max_abs 0.5-2.7 vs Triton's <0.02
    # baseline -- nowhere near ULP. The flag is retained for the LDS
    # win and the bf16 math is preserved.
    use_fp8_mfma_qk: bool = False
    # Native fp8 PV MFMA. When True, V remains in raw fp8 LDS and
    # softmax probabilities are quantised to fp8 (P*240) before PV.
    # This avoids quantising Q/K for the QK softmax path, so it is the
    # safer FlyDSL-inspired subset: exact bf16 QK logits, native fp8 PV.
    use_fp8_mfma_pv: bool = False
    # Experimental in-place improvement for the existing 16x16x32 path:
    # keep softmax P in registers and permute the MFMA-C distribution into
    # the PV MFMA-A distribution, instead of publishing P to P_lds and
    # reading it back. This targets the same fundamental hot-loop overhead
    # as the 32x32 rewrite, but applies to the current production geometry.
    # v1 is intentionally restricted to bf16/no-bias/no-window until parity
    # is proven broadly.
    use_register_pv: bool = False
    # ``T`` (per-CTA-iter KV-tile size in tokens). When ``None``, the
    # kernel uses ``T = block_size`` (one paged-KV cache block per
    # iter, matching the AITER decode path). Setting ``tile_size > block_size``
    # makes each iter walk multiple consecutive ``block_table`` entries
    # and amortizes the outer loop overhead — this is what unlocks
    # Triton-class prefill throughput (Triton 2D uses ``TILE_SIZE=64``
    # with ``BLOCK_M=128``). ``tile_size`` must be a positive multiple
    # of ``block_size`` (so the descriptor's multi-block decomposition
    # is well-defined) and ``T * head_size >= num_warps * 64 * 8``
    # (the async-DMA call carries one wave's lane-contiguous payload).
    tile_size: Optional[int] = None
    # Per-warp M-dimension tile size. Default is one ``MFMA_M`` atom
    # (16 rows) per warp. Setting this to 32 stacks two ``MFMA_M=16``
    # atoms per warp so each warp's QK / PV phase processes twice the
    # rows -- matching Triton's prefill config (``BLOCK_M=128`` with
    # ``num_warps=4`` ⇒ each warp owns 32 rows). The accumulator,
    # ``m`` / ``l`` running stats, and mask/softmax loops then iterate
    # over ``REGS_PER_LANE = block_m_per_warp / 4`` register slots
    # per lane instead of 4. The LDS budget grows with ``BLOCK_M``
    # (``Q_lds``, ``P_lds``, ``Acc_lds``), so ``block_m_per_warp=32``
    # crosses MI355X's 3 → 2 WGs/CU threshold for the prefill workload.
    # Only ``{16, 32}`` are supported; 32 requires ``num_warps``
    # in ``{1, 2, 4}`` so total threads stay within the 1024 CTA cap.
    #
    # **Measured (MI355X, bf16, HD=64, BS=32, T=64)**: ``block_m_per_warp=32``
    # with ``num_warps=4`` (BLOCK_M=128) was 1.6-2.0× SLOWER than the
    # default on every prefill shape we tested, because the doubled
    # ``Q_lds`` + ``P_lds`` + ``Acc_lds`` push the kernel from 3 → 2
    # WGs/CU. The per-CTA throughput gain from bigger BLOCK_M is
    # cancelled by the occupancy loss. The knob is kept exposed for
    # future workloads (e.g. HD=128 or shapes with different LDS
    # budgets) where the trade-off might flip. See
    # the out-of-tree ``probe_blockm32_perf.py`` for the sweep.
    block_m_per_warp: int = 16
    # gfx950-only knob: selects the 32x32 wide-K MFMA geometry (M=32, N=32,
    # K-step=16, <16 x f32> accumulator). gfx942 has no f16/bf16 32x32x16
    # atom, so the gfx942 variant rejects this in ``__post_init__`` and only
    # the narrow 16x16x16 path is built. The field is retained so the spec
    # stays a structural superset of the gfx950 spec.
    use_mfma_32x32: bool = False
    # gfx942 (CDNA3) wide-fragment 32x32 path using the **32x32x8** f16 MFMA
    # atom (``mfma_f32_32x32x8_f16``), which -- unlike the gfx950-only
    # 32x32x16 atom -- IS legal on gfx942. It uses the SAME 32x32 C-output
    # distribution (``_C32_DIST``), mask, softmax, P_lds bridge, and epilogue
    # as the 32x32x16 path; only the K-step (8 vs 16) and the per-lane A/B
    # operand widths (<4 x half> vs <8 x half>) differ. The PV V B-operand is
    # built from NAIVE strided LDS reads (no ds_read_tr16_b64 -- gfx950-only),
    # accepting today's bank conflicts (a conflict-free V feed is a follow-on).
    # This is the correctness-first foundation for the gfx942 flash pipeline,
    # gated behind ``HIPDNN_GFX942_FLASH_PIPELINE`` for D128 fp16. fp16-only
    # (gfx942 has no bf16 32x32x8 atom catalog entry).
    use_mfma_32x32x8: bool = False
    # Experimental orientation for the 32x32 migration. The first 32x32
    # prototype computed ``S = Q @ K^T`` and therefore still needed a
    # cross-lane reduction over K columns. Triton/CK Tile's efficient
    # shape is better understood as computing ``S^T = K @ Q^T``:
    # one lane owns one query column and 16 key positions, so the softmax
    # K-axis mostly lives in registers. This flag tracks that orientation
    # independently while it is brought up.
    use_transposed_qk_32x32: bool = False
    # Transposed 32x32 softmax has one online-softmax state per query lane,
    # not one per output-dimension accumulator register. Keep a single m/l
    # loop-carried state and broadcast alpha across the 16 output regs.
    use_transposed_scalar_state: bool = False
    # Hoist query-row invariants used by the transposed 32x32 score/mask path
    # out of the per-reg/per-tile score loop.
    use_transposed_invariant_hoist: bool = False
    # Compute query-row mask invariants once per KV iteration for the
    # transposed 32x32 path, instead of once per score register.
    use_transposed_mask_once: bool = False
    # Experiment: orient PV so each 32-lane half consumes only P rows it owns
    # and read matching V rows through two half-local transpose LDS reads. This
    # targets the transposed path's lane^32 P fetches.
    use_transposed_half_local_pv: bool = False
    # Opt-in scheduling cleanup for the 32x32 path: skip the legacy 16x16 Q
    # register gather and its Q_ALIAS_K drain barrier. R4 consumes Q32_reg
    # exclusively, so the old Q_reg values are dead on that path.
    use_mfma32_skip_legacy_qreg: bool = False
    # No-SW/no-bias transposed-softmax experiment: collapse the causal and
    # prefix masks into one per-score compare against min(causal, prefix_tail)
    # and hoist the MFMA row base used by all 16 score registers in a lane.
    use_transposed_mask_limit: bool = False
    # Experimental two-KV-tile online-softmax group for the transposed 32x32 R4
    # path. The prototype computes S/P for two consecutive KV tiles against one
    # shared m_new, then runs PV for both tiles while scaling the loop-carried
    # output accumulator only once. Kept opt-in until parity/perf are proven.
    use_grouped_kv2_softmax: bool = False
    # Prototype fast path for the paged-KV byte descriptor on the hot R4 shape:
    # bf16, h64kv8, HD=64, BS=32, T=64, nw=4. In that geometry each async DMA
    # call covers exactly one 32-token cache block, so the K/V loaders can emit
    # two direct block-table loads per tile and simple shift/add byte arithmetic
    # instead of the generic TensorDescriptor transform DAG.
    use_fast_paged_kv_desc: bool = False
    # 64-bit paged-KV addressing. The default load path puts the full byte
    # offset (incl. ``physical_block * block_stride``) in a 32-bit buffer
    # voffset, which overflows for paged caches > 2 GiB (~65 K bf16 / ~131 K
    # fp8 blocks) and silently corrupts loads. When True, the per-block
    # ``physical_block`` offset is folded into a 64-bit buffer *base* (small
    # within-block voffset) so any cache size is addressable. It costs a
    # per-call ``make_buffer_rsrc`` so the dispatcher only enables it when the
    # cache actually exceeds the 2 GiB cap (see _enable_i64_kv_addr).
    use_i64_kv_addr: bool = False
    # Experimental schedule: issue the current V async copy immediately after
    # the iter-start K drain/barrier, before QK. This gives V the whole QK plus
    # softmax window to arrive; the next-K prefetch is still issued after QK so
    # the partial wait before PV can leave only next K pending.
    use_early_v_schedule: bool = False
    # Backend residency experiment for accumulator-touching 32x32 attention:
    # request zero AGPR allocation so LLVM selects VGPR-form MFMA and avoids
    # AGPR<->VGPR copies around the online-softmax/PV accumulator scaling.
    use_agpr_alloc_zero: bool = False
    # Conflict-free transposed V feed (Stream B). On the transposed-x8 PV path the
    # naive A-operand (V^T) read issues 4 strided ``ds_read_u16`` over 4 distinct
    # V_lds rows at one head-dim column -> 32 wave-half lanes collapse onto the
    # same LDS bank set (the rocprof bank-conflict bound, ~7.25 conflicts/LDS).
    # When True, V is stored TRANSPOSED ``[HD, T + pad]`` (head-dim outer, token
    # contiguous, kKPack-padded) so the 4 tokens a lane needs are CONTIGUOUS ->
    # one wide ``ds_read_b64`` (bank-spread). The transpose is paid once on the
    # LDS store (sync ``global_load + shaped smem_store``). f16-only, requires the
    # transposed-x8 orientation (use_mfma_32x32x8 + use_transposed_qk_32x32). This
    # is a KERNEL-SIGNATURE field (tagged "cfv" in kernel_name) so it produces a
    # distinct compiled/cached kernel -- a bare env var would alias the cache.
    use_conflict_free_v: bool = False
    # Conflict-free transposed V feed, STORE-PATH vehicle (c) (Stream 2 GRIND).
    # Same conflict-free ``V_lds[HD, T + pad]`` layout + ``ds_read_b64`` consumer
    # as ``use_conflict_free_v`` above, but a DIFFERENT V *store* mechanism. The
    # ``use_conflict_free_v`` store is vehicle (a) -- a SYNCHRONOUS token-strided
    # ``global_load`` gather (V_T_VEC scalar loads at stride HD per item); that
    # VMEM gather stalls on HBM and is ~5x slower (rocprof: ~30 TF vs wide4's
    # 183). The conflict-free LDS read was already right; only the store vehicle
    # was wrong. This field switches the store to the CK/flash way (gfx942
    # playbook vehicle (c)): each lane ``global_load_vN``-reads V in its NATURAL
    # ``[token, dim]`` layout (CONTIGUOUS dim run -> coalesced VMEM, pipelined),
    # transposes 2x2 f16 blocks IN-REGISTER with ``perm_b32`` (``v_perm_b32``,
    # in-thread, no cross-lane, no ``lgkmcnt``; CK ``transpose_vectors`` masks
    # 0x01000504 / 0x03020706), then writes the now-A-consumable ``[HD,T+pad]``
    # layout with ONE contiguous ``ds_write_b{32,64}`` per token group. The
    # micro-tile loop is RUNTIME-rolled (a fully-unrolled transpose hit a 45-min
    # JIT IR-explosion -- gfx942 playbook). f16-only, requires the transposed-x8
    # orientation (same gate as use_conflict_free_v) and is MUTUALLY EXCLUSIVE
    # with it. KERNEL-SIGNATURE field (tagged "cfvst" in kernel_name) so it
    # produces a distinct compiled/cached kernel. DEFAULT OFF -> gfx950 and every
    # existing kernel are byte-identical.
    use_conflict_free_v_store: bool = False
    # cfvst sub-knobs that affect IR shape. Keep them signature-tracked instead
    # of env-only, otherwise A/B variants can alias a cached code object.
    use_conflict_free_v_store_split: bool = True
    use_conflict_free_v_ck_vlds: bool = True
    # K single-buffer (OCCUPANCY lever, gfx942). The default path double-buffers
    # K_lds (``[2, T, HD]``) so the next-tile async-DMA K[i+1] overlaps softmax+PV
    # of tile i. On the transposed-x8 D128 fp16 path that double buffer is the LDS
    # peak: K_lds(2*T*HD*2=32KB) + V_lds(T*HD*2=16KB) = 48KB > the 32KB
    # 2-wg/CU threshold (STEP-0 evidence: Acc_lds is already backend-aliased into
    # the loop region, so dropping it / acc->AGPR does NOT cross the threshold;
    # only cutting LOOP LDS does). When True, K_lds is single-buffered
    # (``[1, T, HD]`` = 16KB) so loop LDS = 16+16 = 32KB -> 2 wg/CU on gfx942.
    # Correctness: the next-K async DMA into the shared slot races the tail of
    # QK[i]'s LDS reads via raw_ptr_buffer_load_lds lgkmcnt accounting; we close
    # the race by issuing next-K only AFTER a full post-QK ``s_waitcnt(lgkmcnt=0)
    # + sync()`` (all QK K-reads retired before the shared-slot DMA write). The
    # lost per-CTA prefetch overlap is compensated by cross-CTA latency hiding
    # from 2 wg/CU. KERNEL-SIGNATURE field (tagged "k1buf" in kernel_name) so it
    # produces a distinct compiled/cached kernel. f16-only, requires the
    # transposed-x8 orientation + BLOCK_M <= T (so Q still aliases the single K
    # slot). DEFAULT OFF -> gfx950 and every existing kernel are byte-identical.
    use_k_single_buffer: bool = False
    # Experimental CK-style sliced K staging for transposed-x8 D128 cfvst.
    # Stages 32-head-dim K slices in a 3-slot ring instead of double-buffering
    # full [T, HD] K tiles. This is the LDS prerequisite for NumPrefetchK/V=3
    # on MI300X; v1 syncs per slice before adding overlap.
    use_k_sliced_ring: bool = False
    # Use CK Tile's explicit LDS buffer sequence for the sliced-K pipeline
    # instead of the conservative round-robin slot map. This is opt-in because
    # the sequence can reuse the previously consumed slot and therefore needs
    # a carefully placed LDS-read drain before the overwrite.
    use_k_sliced_ldsseq: bool = False
    # Whole-loop scheduler hint experiment. Emits ``iglp_opt(1)`` at the top of
    # the KV-loop body, letting the AMDGPU post-RA scheduler choose a canned
    # MFMA/DS/VMEM interleave without explicit sched_group_barrier fences.
    use_iglp_opt: bool = False
    # T8 (DSL-novel): pin each MFMA k-step of the QK and PV loops as an explicit
    # sched_group_barrier-ordered block (DS_READ then MFMA) so the AMDGPU post-RA
    # scheduler interleaves ds_read into the MFMA window 1:1, matching CK Tile's
    # hand-tuned schedule. Requires the transposed-x8 path; mutually exclusive
    # with iglp_opt (which owns the whole-loop schedule).
    use_qk_pv_sched_group_barrier: bool = False
    # Gather Q directly from global into Q32 registers instead of using K_lds as
    # one-time Q scratch. This is tested independently from sliced K because it
    # can remove a barrier/prologue even when full-tile K buffering remains.
    use_q_direct_global: bool = False
    # Cache policy for async K/V buffer->LDS loads. Gfx942 interprets these aux
    # bits as sc0/nt/swz/sc1, so this stays explicit and benchmarked.
    kv_cache_policy: str = "stream"
    # Use flat/global ``global_load_lds`` for K staging instead of buffer-resource
    # ``raw_ptr_buffer_load_lds``. This avoids constructing/using a buffer SRD for
    # K and exercises the gfx942-native direct global->LDS path. V remains on its
    # existing path (cfvst uses sync global loads, non-cfv uses buffer DMA).
    use_global_load_lds_k: bool = False
    # Launch-grid ordering experiment: default grid is (kv_head, q_block). When
    # True the launcher uses (q_block, kv_head) and the kernel swaps block IDs so
    # workgroups for one KV head walk q-blocks contiguously, improving L2/XCD
    # locality for long-prefill.
    use_q_major_grid: bool = False

    def __post_init__(self):
        # gfx942 (CDNA3) variant: the narrow ``16x16x16`` default path only.
        # The wide-K (32x32x16 migration) and transpose-read / native-fp8
        # experimental knobs all depend on gfx950-only ISA (wide-K MFMA atoms,
        # ``ds_read_*_tr_*``) that does not exist on gfx942; reject them up front
        # rather than emit IR that comgr cannot select. The fp8 K/V cache path
        # uses ``ds_read_tr_b8`` for PV and is likewise gfx950-only here.
        # ``use_transposed_qk_32x32`` is NO LONGER unconditionally gfx942-
        # unsupported: paired with the gfx942-legal 32x32x8 atom
        # (``use_mfma_32x32x8``) it computes S^T = K @ Q^T and consumes P^T
        # directly from registers as the PV B-operand -- dropping the P_lds
        # round-trip (Stream C). It is still rejected when paired with the
        # gfx950-only 32x32x16 atom (``use_mfma_32x32``), and the dependent
        # transposed-softmax VALU knobs (scalar_state/mask_once/...) remain
        # gfx950-only here. So gate the transposed flag below (allow only the
        # x8 pairing) rather than blanket-rejecting it.
        # ``use_mfma_32x32`` (the K=16 32x32x16 atom) stays rejected on gfx942.
        # Although arch_specs.json lists ``mfma_f32_32x32x16_bf16`` under gfx942
        # (#8348) and the LLVM lowerer emits the intrinsic, the gfx942 backend
        # (ROCm 7.2) ``Cannot select`` ``llvm.amdgcn.mfma.f32.32x32x16.bf16`` --
        # it is a CDNA4/gfx950-only instruction (the catalog entry is wrong).
        # The gfx942-legal wide bf16/fp16 path is the K=8 ``use_mfma_32x32x8``
        # atom (``mfma_f32_32x32x8_{f16,bf16}`` -> ``v_mfma_f32_32x32x8_*``,
        # both verified-selectable on gfx942), gated separately below.
        _unsupported = [
            name
            for name, on in (
                ("use_mfma_32x32", self.use_mfma_32x32),
                ("use_transposed_half_local_pv", self.use_transposed_half_local_pv),
                ("use_mfma32_skip_legacy_qreg", self.use_mfma32_skip_legacy_qreg),
                ("use_grouped_kv2_softmax", self.use_grouped_kv2_softmax),
                # ``use_agpr_alloc_zero`` is NOT an ISA feature -- it is a backend
                # codegen hint ("amdgpu-agpr-alloc"="0,0" + -amdgpu-mfma-vgpr-form)
                # that asks LLVM to keep MFMA accumulators in VGPR instead of
                # spilling them to AGPR. It is legal on the gfx942 wide x8
                # transposed path (the only 32x32 MFMA path gfx942 has); the
                # detailed pairing guard below enforces that. So it is gated
                # there, not blanket-rejected here.
                ("use_fp8_mfma_qk", self.use_fp8_mfma_qk),
                ("use_fp8_mfma_pv", self.use_fp8_mfma_pv),
            )
            if on
        ]
        if _unsupported:
            raise ValueError(
                "gfx942 tiled-2D attention supports only the narrow 16x16x16 "
                "default path and the K=8 32x32x8 wide path; these are not "
                f"available on gfx942: {', '.join(_unsupported)}"
            )
        # gfx942 transposed orientation is legal in the x8 pairing (S^T = K @ Q^T
        # via the 32x32x8 f16/bf16 atom + register P^T -> PV). The x16 pairing is
        # rejected above (the K=16 atom can't select on gfx942).
        if self.use_transposed_qk_32x32 and not self.use_mfma_32x32x8:
            raise ValueError(
                "gfx942: use_transposed_qk_32x32 requires use_mfma_32x32x8 "
                "(the K=8 f16/bf16 wide atom)"
            )
        if self.kv_storage_dtype is not None:
            raise ValueError(
                "gfx942 tiled-2D attention does not support the fp8 K/V cache "
                "(its PV path uses ds_read_tr_b8, a gfx950-only transpose read)"
            )
        if self.num_warps not in (1, 2, 4, 8):
            raise ValueError(
                f"num_warps must be 1, 2, 4, or 8 (got {self.num_warps}). "
                f"Other counts would need new MFMA distribution logic. "
                f"num_warps=8 (BLOCK_M=128, THREADS=512) matches the BLOCK_M "
                f"the production Triton 2D kernel uses for high-q prefill "
                f"shapes; both are well within MI355X's 1024-thread CTA cap."
            )
        if self.block_m_per_warp not in (16, 32):
            raise ValueError(
                f"block_m_per_warp must be 16 or 32 (got {self.block_m_per_warp})."
            )
        if self.block_m_per_warp == 32 and self.num_warps not in (1, 2, 4, 8):
            raise ValueError(
                f"block_m_per_warp=32 requires num_warps in {{1,2,4,8}} "
                f"(got {self.num_warps}); other values need new MFMA/launch "
                f"distribution logic."
            )
        if self.use_mfma_32x32:
            if self.block_m_per_warp != 32:
                raise ValueError(
                    "use_mfma_32x32 requires block_m_per_warp=32: "
                    "one 32-row MFMA atom per warp"
                )
            if self.tile_size_eff % 32 != 0:
                raise ValueError(
                    "use_mfma_32x32 requires tile_size to be a multiple of 32"
                )
            if self.head_size % 16 != 0:
                raise ValueError(
                    "use_mfma_32x32 requires head_size to be a multiple of 16"
                )
        if self.use_mfma_32x32x8:
            # gfx942-legal 32x32x8 wide-fragment path. Same 32x32 C layout as
            # the (gfx950-only) 32x32x16 path, but K=8 per atom -- the atom IS
            # present on gfx942 for BOTH fp16 (mfma_f32_32x32x8_f16) and bf16
            # (mfma_f32_32x32x8_bf16, the .1k intrinsic -> v_mfma_f32_32x32x8_bf16,
            # selectable on CDNA3; verified on ROCm 7.2 gfx942). The K=16 bf16
            # atom (mfma_f32_32x32x16_bf16) is CDNA4/gfx950-only, so bf16 wide
            # flash on gfx942 uses THIS K=8 atom.
            if self.use_mfma_32x32:
                raise ValueError(
                    "use_mfma_32x32x8 and use_mfma_32x32 are mutually exclusive "
                    "(K=8 gfx942 atom vs K=16 gfx950-only atom)"
                )
            if self.dtype not in ("fp16", "bf16"):
                raise ValueError("use_mfma_32x32x8 requires fp16 or bf16")
            if self.block_m_per_warp != 32:
                raise ValueError(
                    "use_mfma_32x32x8 requires block_m_per_warp=32: "
                    "one 32-row MFMA atom per warp"
                )
            if self.tile_size_eff % 32 != 0:
                raise ValueError(
                    "use_mfma_32x32x8 requires tile_size to be a multiple of 32"
                )
            if self.head_size % 32 != 0:
                raise ValueError(
                    "use_mfma_32x32x8 requires head_size to be a multiple of 32"
                )
            if self.use_transposed_qk_32x32:
                # x8-transposed (Stream C): S^T = K @ Q^T, P^T consumed from
                # registers as the PV B-operand (no P_lds bridge). Same C
                # distribution / mask / softmax / epilogue as the x8 plain
                # path; only the QK/PV MFMA orientation differs. The scalar
                # state / mask-once / mask-limit VALU rewrites are pure register
                # scheduling and are legal for the x8 path too; half-local PV
                # remains excluded because x8 already consumes P^T registers.
                if self.dtype not in ("fp16", "bf16"):
                    raise ValueError(
                        "x8-transposed (use_transposed_qk_32x32 + "
                        "use_mfma_32x32x8) requires fp16 or bf16"
                    )
                if self.head_size % 32 != 0:
                    raise ValueError(
                        "x8-transposed requires head_size to be a multiple of 32"
                    )
                if self.block_m_per_warp != 32:
                    raise ValueError("x8-transposed requires block_m_per_warp=32")
                if self.tile_size_eff % 32 != 0:
                    raise ValueError(
                        "x8-transposed requires tile_size to be a multiple of 32"
                    )
                _x8t_unsupported = [
                    name
                    for name, on in (
                        (
                            "use_transposed_half_local_pv",
                            self.use_transposed_half_local_pv,
                        ),
                    )
                    if on
                ]
                if _x8t_unsupported:
                    raise ValueError(
                        "gfx942 x8-transposed does not support the gfx950-only "
                        "transposed-softmax PV sub-knobs: "
                        f"{', '.join(_x8t_unsupported)}"
                    )
        if self.use_transposed_qk_32x32 and not (
            self.use_mfma_32x32 or self.use_mfma_32x32x8
        ):
            raise ValueError(
                "use_transposed_qk_32x32 requires use_mfma_32x32 or use_mfma_32x32x8"
            )
        if self.use_transposed_scalar_state and not self.use_transposed_qk_32x32:
            raise ValueError(
                "use_transposed_scalar_state requires use_transposed_qk_32x32"
            )
        if self.use_transposed_invariant_hoist and not self.use_transposed_qk_32x32:
            raise ValueError(
                "use_transposed_invariant_hoist requires use_transposed_qk_32x32"
            )
        if self.use_transposed_mask_once and not self.use_transposed_qk_32x32:
            raise ValueError(
                "use_transposed_mask_once requires use_transposed_qk_32x32"
            )
        if self.use_transposed_half_local_pv and not self.use_transposed_qk_32x32:
            raise ValueError(
                "use_transposed_half_local_pv requires use_transposed_qk_32x32"
            )
        if self.use_conflict_free_v and not (
            self.use_mfma_32x32x8 and self.use_transposed_qk_32x32
        ):
            raise ValueError(
                "use_conflict_free_v requires the transposed-x8 PV orientation "
                "(use_mfma_32x32x8 + use_transposed_qk_32x32)"
            )
        if self.use_conflict_free_v_store and not (
            self.use_mfma_32x32x8 and self.use_transposed_qk_32x32
        ):
            raise ValueError(
                "use_conflict_free_v_store requires the transposed-x8 PV "
                "orientation (use_mfma_32x32x8 + use_transposed_qk_32x32)"
            )
        if self.use_conflict_free_v_store and self.use_conflict_free_v:
            raise ValueError(
                "use_conflict_free_v_store and use_conflict_free_v are mutually "
                "exclusive (they are two store vehicles for the same transposed "
                "V_lds layout)"
            )
        if self.use_k_single_buffer:
            if not (self.use_mfma_32x32x8 and self.use_transposed_qk_32x32):
                raise ValueError(
                    "use_k_single_buffer requires the transposed-x8 PV orientation "
                    "(use_mfma_32x32x8 + use_transposed_qk_32x32)"
                )
            if self.dtype not in ("fp16", "bf16"):
                raise ValueError("use_k_single_buffer requires fp16 or bf16")
            _block_m = self.num_warps * self.block_m_per_warp
            _t = self.tile_size if self.tile_size is not None else self.block_size
            if _block_m > _t:
                raise ValueError(
                    "use_k_single_buffer requires BLOCK_M <= tile_size so Q still "
                    "aliases the single K slot (BLOCK_M="
                    f"{_block_m}, tile_size={_t})"
                )
        if self.use_k_sliced_ring:
            if not (
                self.use_mfma_32x32x8
                and self.use_transposed_qk_32x32
                and self.use_conflict_free_v_store
            ):
                raise ValueError(
                    "use_k_sliced_ring requires the transposed-x8 cfvst path"
                )
            if (
                self.dtype not in ("fp16", "bf16")
                or self.head_size not in (64, 128)
                or self.head_size % 32 != 0
                or self.tile_size_eff not in (64, 128)
            ):
                raise ValueError(
                    "use_k_sliced_ring requires fp16/bf16, head_size in {64,128} "
                    "(HD %% 32 == 0 for the 32-wide K slices), T in {64,128}"
                )
        if self.use_k_sliced_ldsseq and not self.use_k_sliced_ring:
            raise ValueError("use_k_sliced_ldsseq requires use_k_sliced_ring")
        if self.use_q_direct_global:
            if not (self.use_mfma_32x32x8 and self.use_transposed_qk_32x32):
                raise ValueError("use_q_direct_global currently targets transposed-x8")
        if self.use_qk_pv_sched_group_barrier:
            if not (self.use_mfma_32x32x8 and self.use_transposed_qk_32x32):
                raise ValueError(
                    "use_qk_pv_sched_group_barrier requires the transposed-x8 path (use_mfma_32x32x8 + use_transposed_qk_32x32)"
                )
            if self.dtype not in ("fp16", "bf16"):
                raise ValueError("use_qk_pv_sched_group_barrier requires fp16 or bf16")
            if self.use_iglp_opt:
                raise ValueError(
                    "use_qk_pv_sched_group_barrier is mutually exclusive with use_iglp_opt (iglp_opt owns the loop schedule)"
                )
        if self.num_warps == 8 and self.block_m_per_warp == 32:
            if not (self.use_q_direct_global and self.use_conflict_free_v_store):
                raise ValueError(
                    "8-wave mw32 v1 requires direct-Q + cfvst to stay within LDS"
                )
        if self.kv_cache_policy not in ("all", "global", "stream", "nt"):
            raise ValueError(
                "kv_cache_policy must be one of 'all', 'global', 'stream', 'nt' "
                f"(got {self.kv_cache_policy!r})"
            )
        if self.use_global_load_lds_k and self.kv_storage_dtype is not None:
            raise ValueError("use_global_load_lds_k v1 supports bf16/fp16 KV only")
        if self.use_mfma32_skip_legacy_qreg and not self.use_mfma_32x32:
            raise ValueError("use_mfma32_skip_legacy_qreg requires use_mfma_32x32")
        if self.use_transposed_mask_limit:
            if not (
                (self.use_mfma_32x32 or self.use_mfma_32x32x8)
                and self.use_transposed_qk_32x32
                and self.use_transposed_scalar_state
                and self.use_transposed_mask_once
            ):
                raise ValueError(
                    "transposed softmax VALU opts require the R4_s1mask path"
                )
            if self.sliding_window > 0:
                raise ValueError(
                    "transposed softmax VALU opts require no sliding window"
                )
            if self.has_softcap or self.use_alibi or self.use_qq_bias:
                raise ValueError(
                    "transposed softmax VALU opts do not support softcap, ALiBi, or QQ bias"
                )
            if self.use_grouped_kv2_softmax:
                raise ValueError(
                    "transposed softmax VALU opts do not support grouped_kv2"
                )
        if self.use_grouped_kv2_softmax:
            if self.dtype != "bf16":
                raise ValueError(
                    "use_grouped_kv2_softmax v1 is restricted to dtype='bf16'"
                )
            if not (self.use_mfma_32x32 and self.use_transposed_qk_32x32):
                raise ValueError(
                    "use_grouped_kv2_softmax requires the transposed 32x32 R4 path"
                )
            if self.block_m_per_warp != 32:
                raise ValueError("use_grouped_kv2_softmax requires block_m_per_warp=32")
            if self.sliding_window > 0:
                raise ValueError(
                    "use_grouped_kv2_softmax v1 requires no sliding window"
                )
            if self.has_softcap or self.use_alibi or self.use_qq_bias:
                raise ValueError(
                    "use_grouped_kv2_softmax v1 does not support softcap, ALiBi, or QQ bias"
                )
            if self.kv_storage_dtype is not None:
                raise ValueError("use_grouped_kv2_softmax v1 does not support FP8 KV")
        if self.use_early_v_schedule:
            if self.use_grouped_kv2_softmax:
                raise ValueError("use_early_v_schedule does not support grouped_kv2")
            if self.kv_storage_dtype is not None:
                raise ValueError("use_early_v_schedule v1 does not support FP8 KV")
        if self.use_fast_paged_kv_desc:
            if not (
                self.dtype == "bf16"
                and self.kv_storage_dtype is None
                and self.head_size == 64
                and self.block_size == 32
                and self.tile_size_eff == 64
                and self.num_query_heads == 64
                and self.num_kv_heads == 8
                and self.num_warps == 4
            ):
                raise ValueError(
                    "use_fast_paged_kv_desc is restricted to bf16 h64kv8 "
                    "HD=64 BS=32 T=64 num_warps=4"
                )
        if self.use_agpr_alloc_zero:
            _agpr0_r4_s1mask_hlpv = (
                self.use_mfma_32x32
                and self.use_transposed_qk_32x32
                and self.use_transposed_scalar_state
                and self.use_transposed_mask_once
                and self.use_transposed_half_local_pv
            )
            # Wide K=8 bf16/fp16 transposed flash path (the D128 ship geometry).
            # The accumulator-residency win is orthogonal to the softmax-VALU
            # opts of the R4_s1mask_hlpv family: any 32x32 MFMA path that touches
            # the C accumulator across the online-softmax loop benefits from
            # VGPR-form MFMA. Accept the x8 transposed orientation directly so the
            # D128 wide config can request zero-AGPR codegen.
            _agpr0_wide_x8 = self.use_mfma_32x32x8 and self.use_transposed_qk_32x32
            if not (_agpr0_r4_s1mask_hlpv or _agpr0_wide_x8):
                raise ValueError(
                    "use_agpr_alloc_zero currently targets the R4_s1mask_hlpv "
                    "path or the wide x8 transposed path "
                    "(use_mfma_32x32x8 + use_transposed_qk_32x32)"
                )
        if self.kv_storage_dtype is not None and self.kv_storage_dtype != "fp8e4m3":
            raise ValueError(
                f"kv_storage_dtype must be None or 'fp8e4m3' (got {self.kv_storage_dtype!r})"
            )
        if self.use_fp8_mfma_qk and self.kv_storage_dtype != "fp8e4m3":
            raise ValueError("use_fp8_mfma_qk requires kv_storage_dtype='fp8e4m3'")
        if self.use_fp8_mfma_pv and self.kv_storage_dtype != "fp8e4m3":
            raise ValueError("use_fp8_mfma_pv requires kv_storage_dtype='fp8e4m3'")
        if self.use_register_pv:
            if self.use_mfma_32x32:
                raise ValueError(
                    "use_register_pv currently targets the existing 16x16x32 path; "
                    "the 32x32 path has a separate register-P migration"
                )
            if self.dtype != "bf16":
                raise ValueError("use_register_pv v1 is restricted to dtype='bf16'")
            if self.kv_storage_dtype is not None:
                raise ValueError("use_register_pv v1 does not support fp8 K/V cache")
            if self.use_sinks or self.sliding_window > 0 or self.has_softcap:
                raise ValueError(
                    "use_register_pv v1 requires no sinks, no sliding window, and no softcap"
                )
            if self.use_alibi or self.use_qq_bias:
                raise ValueError("use_register_pv v1 does not support ALiBi or QQ bias")
        if self.tile_size is not None:
            if self.tile_size <= 0 or self.tile_size % self.block_size != 0:
                raise ValueError(
                    f"tile_size must be a positive multiple of block_size "
                    f"(got tile_size={self.tile_size}, block_size={self.block_size})"
                )

    @property
    def num_queries_per_kv(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    @property
    def block_m(self) -> int:
        return self.block_m_per_warp * self.num_warps

    @property
    def regs_per_lane(self) -> int:
        """Number of accumulator register slots per lane per N-tile.

        For ``MFMA_M=16``, the 16x16 MFMA distribution gives each lane
        4 row slots (rows ``lane_rg*4..lane_rg*4+3`` within a 16-row
        atom). With ``block_m_per_warp=32``, we stack two MFMA atoms
        per warp ⇒ 8 row slots per lane.

        For the 32x32x16 migration, a single warp owns all 32 rows in
        one MFMA atom and each lane carries 16 accumulator elements in
        CK Tile's C distribution. Those 16 row slots are the state we
        need for ``m``/``l``/``P`` while we remove the old P_lds roundtrip.
        """
        if self.use_mfma_32x32 or self.use_mfma_32x32x8:
            return 16
        return self.block_m_per_warp // 4  # 4 for M=16, 8 for M=32

    @property
    def block_q(self) -> int:
        return self.block_m // self.num_queries_per_kv

    @property
    def tile_size_eff(self) -> int:
        """Effective per-iter KV-tile size in tokens."""
        return self.tile_size if self.tile_size is not None else self.block_size

    @property
    def n_blocks_per_tile(self) -> int:
        """How many paged-KV cache blocks one kernel iter consumes."""
        return self.tile_size_eff // self.block_size

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.dtype == "fp16" else BF16

    @property
    def binary_search_iters(self) -> int:
        # AITER/Triton uses a true while-loop binary search. Our IR currently
        # lowers this as a fixed-trip scf.for, so specialize the trip count to
        # the known problem batch size instead of always paying 32 iterations.
        # Keep 32 as a conservative fallback for direct unit-test specs that do
        # not provide `num_seqs`.
        if self.num_seqs <= 0:
            return 32
        return max(1, int(math.ceil(math.log2(self.num_seqs + 1))))

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        # Value-carrying optionals (sw{N}, w{N}) become plain
        # conditional strings; kernel_name_join drops empty ones.
        # Value-less flags go through the `flags=` map so they get
        # rendered in iteration order with leading underscores.
        return kernel_name_join(
            "rocke_uattn2d_tiled",
            f"d{self.head_size}",
            f"b{self.block_size}",
            f"t{self.tile_size_eff}" if self.n_blocks_per_tile != 1 else "",
            f"h{self.num_query_heads}kv{self.num_kv_heads}",
            self.dtype,
            f"kv{self.kv_storage_dtype}" if self.kv_storage_dtype else "",
            "" if not self.use_sinks else "sinks",
            f"sw{self.sliding_window}" if self.sliding_window > 0 else "",
            "softcap" if self.has_softcap else "",
            "alibi" if self.use_alibi else "",
            "qqb" if self.use_qq_bias else "",
            f"w{self.num_warps}" if self.num_warps != 1 else "",
            f"wpe{self.waves_per_eu}" if self.waves_per_eu is not None else "",
            f"mw{self.block_m_per_warp}" if self.block_m_per_warp != 16 else "",
            "mfma32" if self.use_mfma_32x32 else "",
            "mfma32x8" if self.use_mfma_32x32x8 else "",
            "stqk" if self.use_transposed_qk_32x32 else "",
            "s1" if self.use_transposed_scalar_state else "",
            "mask1" if self.use_transposed_mask_once else "",
            "hoist" if self.use_transposed_invariant_hoist else "",
            "hlpv" if self.use_transposed_half_local_pv else "",
            "skipqreg" if self.use_mfma32_skip_legacy_qreg else "",
            "mlim" if self.use_transposed_mask_limit else "",
            "gkv2" if self.use_grouped_kv2_softmax else "",
            "fastkvdesc" if self.use_fast_paged_kv_desc else "",
            "earlyv" if self.use_early_v_schedule else "",
            "qdir" if self.use_q_direct_global else "",
            "qsgb" if self.use_qk_pv_sched_group_barrier else "",
            f"kvcp{self.kv_cache_policy}" if self.kv_cache_policy != "stream" else "",
            "gldlds" if self.use_global_load_lds_k else "",
            "qgrid" if self.use_q_major_grid else "",
            "agpr0" if self.use_agpr_alloc_zero else "",
            "fp8mfma" if self.use_fp8_mfma_qk else "",
            "fp8pv" if self.use_fp8_mfma_pv else "",
            "regpv" if self.use_register_pv else "",
            "cfv" if self.use_conflict_free_v else "",
            "cfvst" if self.use_conflict_free_v_store else "",
            (
                "cfvnosplit"
                if self.use_conflict_free_v_store
                and not self.use_conflict_free_v_store_split
                else ""
            ),
            (
                "cfvplainv"
                if self.use_conflict_free_v_store
                and not self.use_conflict_free_v_ck_vlds
                else ""
            ),
            "ksring" if self.use_k_sliced_ring else "",
            "ldsseq" if self.use_k_sliced_ldsseq else "",
            "iglp1" if self.use_iglp_opt else "",
            "k1buf" if self.use_k_single_buffer else "",
        )


def supports_tiled_2d(
    *,
    head_size: int,
    block_size: int,
    dtype: str,
    num_queries_per_kv: int,
    use_alibi: bool,
    use_qq_bias: bool,
    use_fp8: bool,
    q_dtype,
    num_warps: int = 1,
    block_m_per_warp: int = 16,
    kv_storage_dtype: Optional[str] = None,
    tile_size: Optional[int] = None,
    arch: str = "gfx942",
    use_mfma_32x32x8: bool = False,
    use_transposed_qk_32x32: bool = False,
    use_k_single_buffer: bool = False,
    use_conflict_free_v_store: bool = False,
    use_k_sliced_ring: bool = False,
) -> Tuple[bool, str]:
    # The gfx942 variant runs the narrow 16x16x16 default path. The arch gate
    # admits gfx942 (narrow atom present + non-transpose V pipeline selectable)
    # and still rejects targets that have neither wide-K nor the narrow atom.
    # See ``instances/common/attention_arch.py``.
    from ..common.attention_arch import validate_tiled_attention_arch

    arch_ok, arch_reason = validate_tiled_attention_arch(arch)
    if not arch_ok:
        return False, arch_reason
    # The fp8 K/V cache path needs ds_read_tr_b8 (gfx950-only); the gfx942
    # variant supports only the bf16/fp16 default path.
    if kv_storage_dtype is not None:
        return (
            False,
            "gfx942 tiled 2D kernel does not support the fp8 K/V cache "
            "(ds_read_tr_b8 is gfx950-only)",
        )
    if dtype not in ("fp16", "bf16"):
        return False, f"tiled 2D kernel currently supports fp16/bf16 (got {dtype!r})"
    if head_size not in (64, 128, 256):
        return (
            False,
            f"tiled 2D kernel only supports head_size in {{64,128,256}} (got {head_size})",
        )
    if head_size % 32 != 0:
        return (
            False,
            f"tiled 2D kernel requires head_size divisible by 32 (got {head_size})",
        )
    if block_size not in (16, 32, 64):
        return (
            False,
            f"tiled 2D kernel only supports block_size in {{16,32,64}} (got {block_size})",
        )
    if num_queries_per_kv > 16 or num_queries_per_kv < 1:
        return (
            False,
            f"tiled 2D kernel needs 1<=num_queries_per_kv<=16 (got {num_queries_per_kv})",
        )
    block_m = 16 * num_warps
    if block_m % num_queries_per_kv != 0:
        return (
            False,
            f"tiled 2D kernel needs num_queries_per_kv to divide BLOCK_M={block_m} "
            f"(num_warps={num_warps}, got num_queries_per_kv={num_queries_per_kv})",
        )
    # FP8 K/V cache is supported via ``kv_storage_dtype="fp8e4m3"`` plus
    # ``use_fp8=True`` (the latter is what the upstream selector flips on
    # for the FP8 path).
    if kv_storage_dtype is not None and kv_storage_dtype != "fp8e4m3":
        return (
            False,
            f"tiled 2D kernel: unsupported kv_storage_dtype {kv_storage_dtype!r}",
        )
    if use_fp8 and kv_storage_dtype is None:
        return (
            False,
            "tiled 2D kernel: use_fp8=True requires kv_storage_dtype='fp8e4m3'",
        )
    if q_dtype is not None and q_dtype not in ("fp16", "bf16"):
        return False, f"tiled 2D kernel: unsupported q_dtype {q_dtype!r}"
    if tile_size is not None:
        if tile_size <= 0 or tile_size % block_size != 0:
            return (
                False,
                f"tiled 2D kernel: tile_size={tile_size} must be a positive "
                f"multiple of block_size={block_size}",
            )
        # The async DMA call carries THREADS*2 lane-contiguous halves on
        # gfx942 (CDNA3 caps load-to-LDS at 1 dword = 4 bytes = 2 halves per
        # lane); the per-tile KV slab must hold at least that much, otherwise
        # the wave under-fills the LDS slab and corrupts the partial buffer.
        halves_per_lane = 2
        threads = num_warps * 64
        if tile_size * head_size < threads * halves_per_lane:
            return (
                False,
                f"tiled 2D kernel: tile_size*head_size={tile_size * head_size} too "
                f"small for num_warps={num_warps} (need >= {threads * halves_per_lane})",
            )
        # Per-wave window must fit within one block. Each wave (64 lanes)
        # owns ``WAVE * 2 // head_size`` consecutive tokens within a call
        # (gfx942: 2 halves per lane per call).
        # If a wave straddles two blocks, the per-lane block_table lookup
        # diverges within the wave -- the multi-block descriptor's
        # ``global_load_i32`` becomes lane-divergent (per-lane VMEM) and
        # the async DMA's lane-contiguous LDS layout no longer matches
        # the physical block, so the wave under-fills the slab. Per-WAVE
        # uniformity (waves land in different blocks but each wave is
        # entirely in one block) is allowed; this is the
        # ``num_warps=8, HD=64, BS=32, T=64`` Triton-class config.
        per_wave_tokens = (64 * 2) // head_size
        if per_wave_tokens > block_size:
            return (
                False,
                f"tiled 2D kernel: per-wave tokens {per_wave_tokens} exceeds "
                f"block_size={block_size}; would need lane-divergent block lookup",
            )
    # LDS-budget gate (ahead-of-time compilability). The kernel stages its
    # tiles in LDS (the smem_alloc calls in build_unified_attention_2d_tiled);
    # comgr CODEGEN (CODEGEN_BC_TO_RELOCATABLE) rejects a kernel whose static
    # group segment exceeds the target LDS capacity (gfx950: 163840 B / 160 KB,
    # arch_specs.json; arch is already gated to the tiled-attention family
    # above). Rejecting here yields a clean reason instead of a comgr abort.
    # The footprint mirrors the fp16/bf16 allocations (bpe=2):
    #   K_lds  = 2*T*HD*bpe   (double-buffered async load)
    #   V_lds  =   T*HD*bpe   (single-buffered)
    #   P_lds  = BLOCK_M*(T+8)*bpe  (score tile; +8 pad)
    #   Q_lds  = BLOCK_M*HD*bpe, 0 when it aliases K_lds (BLOCK_M <= 2*T)
    #   Acc_lds= BLOCK_M*OUT_STRIPE*bpe,  OUT_STRIPE = 32 if HD<=64 else HD
    # where BLOCK_M = num_warps*block_m_per_warp, T = tile_size_eff. This is
    # LDS-only and conservative (assumes P_lds present -- the bf16 REGISTER_PV
    # path drops it -- and ignores register/AGPR pressure, which can flip a
    # couple of borderline mfma32 configs), so it never admits an unbuildable
    # combo. FP8 staging has extra LDS allocations and a 1-byte KV dtype, so it
    # is accounted separately; gate only the validated fp16/bf16 path here.
    if not use_fp8:
        # Per-arch LDS capacity from the catalog (gfx942 -> 65536 B / 64 KB),
        # NOT a hardcoded gfx950 constant. The footprint formula is identical
        # to gfx950's; gfx942 is tile-limited rather than starved, so the only
        # effect is that the largest T/HD combos (which exceed 64 KB) are
        # rejected here with a clean reason instead of a comgr CODEGEN abort.
        from ...core.arch import ArchTarget

        try:
            _LDS_CAPACITY_BYTES = ArchTarget.from_gfx(arch).lds_capacity_bytes
        except KeyError:
            _LDS_CAPACITY_BYTES = 65536
        _BPE = 2  # fp16/bf16
        _t_eff = tile_size if tile_size is not None else block_size
        _block_m = num_warps * block_m_per_warp
        if use_mfma_32x32x8 and use_transposed_qk_32x32:
            if use_k_sliced_ring:
                # Overlapped sliced-K ring (CK Tile geometry): K is staged as a
                # 3-slot ring of 32-head-dim slices (not double-buffered at full
                # HD), Q is fed direct-from-global (no Q_lds), and V uses the
                # conflict-free CK packed layout. This is what lets T=128 fit the
                # 64 KB cap where the naive 2*T*HD K buffer would not.
                _K_SLICE_HD = 32
                _K_SLICE_SLOTS = 3
                _k_bytes = _K_SLICE_SLOTS * _t_eff * _K_SLICE_HD * _BPE
                # CK conflict-free transposed-V slot map (mirrors V_T_CK_SLOTS in
                # build_unified_attention_2d_tiled): kgroups*ngroups*group_stride.
                _v_kpack = 8
                _v_pixels = 64
                _v_nper_row = _v_pixels // _v_kpack
                _v_ngroups = head_size // _v_nper_row
                _v_kgroups = _t_eff // _v_kpack
                _v_bytes = _v_kgroups * _v_ngroups * (_v_pixels + _v_kpack) * _BPE
                _lds_x8 = _k_bytes + _v_bytes
                if _lds_x8 > _LDS_CAPACITY_BYTES:
                    return (
                        False,
                        f"gfx942 transposed-x8 sliced-K ring estimated LDS "
                        f"{_lds_x8} B exceeds the {arch} {_LDS_CAPACITY_BYTES} B "
                        f"LDS budget",
                    )
                return True, "supported"
            k_slots = 1 if use_k_single_buffer else 2
            q_lds = 0 if _block_m <= 2 * _t_eff else _block_m * head_size * _BPE
            v_pad = 8 if use_conflict_free_v_store else 0
            _lds_x8 = (
                k_slots * _t_eff * head_size * _BPE
                + (_t_eff + v_pad) * head_size * _BPE
                + q_lds
            )
            if _lds_x8 > _LDS_CAPACITY_BYTES:
                return (
                    False,
                    f"gfx942 transposed-x8 estimated LDS {_lds_x8} B exceeds "
                    f"the {arch} {_LDS_CAPACITY_BYTES} B LDS budget",
                )
            return True, "supported"
        _out_stripe = 32 if head_size <= 64 else head_size
        _lds = (
            2 * _t_eff * head_size * _BPE  # K_lds (double-buffered)
            + _t_eff * head_size * _BPE  # V_lds
            + _block_m * (_t_eff + 8) * _BPE  # P_lds
            + (0 if _block_m <= 2 * _t_eff else _block_m * head_size * _BPE)  # Q_lds
            + _block_m * _out_stripe * _BPE  # Acc_lds
        )
        if _lds > _LDS_CAPACITY_BYTES:
            return (
                False,
                f"tiled 2D kernel: estimated LDS {_lds} B (BLOCK_M={_block_m}, T={_t_eff}, "
                f"HD={head_size}) exceeds the {arch} {_LDS_CAPACITY_BYTES} B LDS budget; "
                f"comgr CODEGEN would fail",
            )
    return True, "supported"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_unified_attention_2d_tiled(
    spec: UnifiedAttention2DTiledSpec, *, arch: str = "gfx942"
) -> KernelDef:
    """Emit the gfx942 narrow-atom tiled MFMA 2D unified-attention kernel.

    Algorithm (per CTA = 1 wave64 = 64 lanes):

    1. Find `seq_idx` via the AITER binary-search-on-`cu_q`.
    2. Compute Q-block-local index; early-exit if it's a padding block.
    3. Cooperatively stage Q[16, 128] from global to LDS (zero-fill for
       rows that map to padding queries or out-of-range heads).
    4. Loop over KV tiles (`tile_start..tile_end`):
       4a. Look up `physical_block = block_tables[seq_idx, tile_idx]`.
       4b. Cooperatively stage K, V (each [T, 128]) from cache to LDS,
           zero-filling per-tile rows outside `max_seq_prefix_len`.
       4c. Compute `S = Q @ K^T` via `_mfma_16x16x16` (`HD/16` K-iters
           per N-tile, with `T/16` N-tiles).
       4d. Apply `qk_scale`, optional `softcap`, mask (causal, sliding
           window, padding rows, padding heads).
       4e. Online softmax: per-row max via `ds_bpermute` butterfly (lanes
           in 16-lane groups share their 4-row state). Compute P=exp2(S-m)
           in registers and stash it in LDS for the PV MFMA A operand.
       4f. `acc *= alpha`, `acc += P @ V` via `_mfma_16x16x16` (8 N-tiles,
           `T/16` K-iters). The V `B` operand is assembled from strided LDS
           reads of `V_lds[T, HD]` that reproduce the 16x16x16 MFMA B
           distribution directly, since gfx942 has no `ds_read_tr16_b64`
           transpose read.
    5. Normalise `acc /= L` per row, stage into Acc_lds, and store fp16
       output via 8-half vector writes.
    """

    # gfx942 runs the narrow 16x16x16 default path. The arch gate admits gfx942
    # (the narrow f16/bf16 atom exists and the non-transpose V pipeline is
    # selectable) and rejects targets without it; reject cleanly before any IR
    # is emitted. See ``instances/common/attention_arch.py``.
    from ..common.attention_arch import require_tiled_attention_arch

    require_tiled_attention_arch(arch)

    if spec.dtype not in ("fp16", "bf16"):
        raise NotImplementedError("tiled 2D kernel supports fp16/bf16")
    dtype = spec.dtype_ir

    # Select the narrow 16x16x16 atom up front from the arch catalog (the
    # multi-arch policy forbids down-splitting a K=32 atom; we SELECT narrow).
    # bf16 routes through the ``_1k`` intrinsic via mfma_16x16x16_for_dtype.
    from ...core.arch import ArchTarget

    _a_dt = "f16" if spec.dtype == "fp16" else "bf16"
    _qk_atom = ArchTarget.from_gfx(arch).mma.select_largest_k(
        family="mma", a_dtype=_a_dt, b_dtype=_a_dt, c_dtype="fp32", m=16, n=16, k_max=16
    )
    if _qk_atom is None or _qk_atom.k != 16:
        raise NotImplementedError(
            f"gfx942 tiled 2D kernel requires a 16x16x16 {_a_dt} MFMA atom on {arch}"
        )

    HD = spec.head_size
    T = spec.tile_size_eff
    BS = spec.block_size
    N_BLOCKS_PER_TILE = spec.n_blocks_per_tile
    BLOCK_M = spec.block_m
    BLOCK_Q = spec.block_q
    NQK = spec.num_queries_per_kv
    NUM_KV = spec.num_kv_heads
    NUM_QH = spec.num_query_heads
    SLIDING_WINDOW = spec.sliding_window
    USE_SOFTCAP = spec.has_softcap
    USE_SINKS = spec.use_sinks
    USE_ALIBI = spec.use_alibi
    USE_QQ_BIAS = spec.use_qq_bias
    TRANSPOSED_SCALAR_STATE = spec.use_transposed_scalar_state
    TRANSPOSED_INVARIANT_HOIST = spec.use_transposed_invariant_hoist
    TRANSPOSED_MASK_ONCE = spec.use_transposed_mask_once
    TRANSPOSED_HALF_LOCAL_PV = spec.use_transposed_half_local_pv
    SKIP_LEGACY_QREG = spec.use_mfma32_skip_legacy_qreg
    TRANSPOSED_MASK_LIMIT = spec.use_transposed_mask_limit
    GROUPED_KV2 = spec.use_grouped_kv2_softmax
    FAST_PAGED_KV_DESC = spec.use_fast_paged_kv_desc
    I64_KV_ADDR = spec.use_i64_kv_addr
    EARLY_V_SCHEDULE = spec.use_early_v_schedule
    # FP8 K/V cache: when set, K/V cache pointers are ``ptr<fp8e4m3, global>``
    # (one byte per element), the async DMA path is disabled, and a sync
    # per-thread load + ``cvt_fp8_to_f32 * k_scale -> cast<bf16>`` chain
    # populates the same LDS slabs as the bf16 path. ``KV_BYTES`` flips
    # the byte-stride math for the paged-KV descriptor.
    KV_FP8 = spec.kv_storage_dtype == "fp8e4m3"
    FP8_MFMA_QK = KV_FP8 and spec.use_fp8_mfma_qk
    FP8_MFMA_PV = KV_FP8 and spec.use_fp8_mfma_pv
    # Native fp8 QK (32x32 combo with raw fp8 K, fp8-quantized Q, native fp8
    # MFMA) was implemented and measured: a LOSE-LOSE here -- slower AND less
    # accurate. The kernel is HBM/latency-bound, so the fp8->bf16 dequant it
    # would eliminate is already hidden behind memory latency (removing hidden
    # VALU buys nothing, same lesson as the mask-skip and fp8-in-LDS probes),
    # and quantizing Q to fp8 costs accuracy. The accurate sync-dequant path
    # is optimal; native fp8 QK stays disabled.
    FP8_NATIVE_QK = False
    REGISTER_PV = spec.use_register_pv
    TRANSPOSED_QK_32X32 = spec.use_transposed_qk_32x32
    CONFLICT_FREE_V = spec.use_conflict_free_v
    # Store-path conflict-free V (vehicle (c)): reuses the SAME transposed
    # ``V_lds[HD, T+pad]`` layout + conflict-free ``ds_read_b64`` consumer as
    # ``CONFLICT_FREE_V`` (so ``TRANSPOSED_V`` below is driven by EITHER flag),
    # but swaps the store fill from the sync HBM gather to register-load +
    # ``perm_b32`` in-register transpose + contiguous LDS store.
    CONFLICT_FREE_V_STORE = spec.use_conflict_free_v_store
    K_SINGLE_BUF = spec.use_k_single_buffer
    K_SLICED_RING = spec.use_k_sliced_ring
    K_SLICED_LDSSEQ = spec.use_k_sliced_ldsseq
    USE_IGLP_OPT = spec.use_iglp_opt
    USE_QK_PV_SCHED_GROUP_BARRIER = spec.use_qk_pv_sched_group_barrier
    USE_GLOBAL_LOAD_LDS_K = spec.use_global_load_lds_k
    # DIAGNOSTIC ONLY (not signature-gated -- toggle re-JITs via env): read the
    # transposed cfv [HD,T+pad] V via 4 scalar n=1 reads instead of one n=4
    # ds_read_b64. Isolates whether the wide read lowering is the cfv fault.
    _CFV_SCALAR_READ = os.environ.get("HIPDNN_GFX942_CFV_SCALAR_READ", "0") == "1"
    # DIAGNOSTIC: store-path vehicle (c) -- read the HBM dim-pair via 2 scalar
    # loads + pack instead of one global_load_vN. Isolates the vectorized load.
    _CFV_STORE_SCALAR_LOAD = (
        os.environ.get("HIPDNN_GFX942_CFV_STORE_SCALAR_LOAD", "0") == "1"
    )
    # DIAGNOSTIC: store-path vehicle (c) -- element-wise scatter (no perm).
    # Proves the (dim,token) mapping + coverage independent of the perm.
    _CFV_STORE_SCATTER = os.environ.get("HIPDNN_GFX942_CFV_STORE_SCATTER", "0") == "1"
    _CFV_STORE_PREZERO = os.environ.get("HIPDNN_GFX942_CFV_STORE_PREZERO", "0") == "1"
    _CFV_STORE_SEPOFF = os.environ.get("HIPDNN_GFX942_CFV_STORE_SEPOFF", "0") == "1"
    _CFV_STORE_SPLIT = spec.use_conflict_free_v_store_split
    KV_BYTES = 1 if KV_FP8 else 2
    kv_io_dtype = FP8E4M3 if KV_FP8 else dtype
    kv_cache_aux = {
        "all": CACHE_ALL,
        "global": CACHE_GLOBAL,
        "stream": CACHE_STREAM,
        "nt": NON_TEMPORAL,
    }[spec.kv_cache_policy]

    # gfx942 32x32x8 wide-fragment path (correctness-first flash foundation).
    # ``use_mfma_32x32`` (K=16) stays rejected in the spec; ``use_mfma_32x32x8``
    # (K=8) is the gfx942-legal wide atom. ``USE_MFMA_32X32`` is the *umbrella*
    # predicate driving all K-INDEPENDENT 32x32 branches (acc storage, mask,
    # softmax, alpha/L, P_lds bridge, epilogue, Q32 gather control flow). The
    # 32x32 C-output distribution is identical for K=8 and K=16. ``USE_MFMA_32X32X8``
    # drives the K-SPECIFIC geometry (K-step 8, <4 x half> operands, atom select,
    # naive strided V B-operand).
    USE_MFMA_32X32X8 = spec.use_mfma_32x32x8
    USE_MFMA_32X32 = spec.use_mfma_32x32 or USE_MFMA_32X32X8
    if USE_MFMA_32X32:
        # 32x32 score/output geometry: 32-column N-tiles, one M=32 atom/warp.
        #   - per-lane C: <16 x f32>
        #   - N tile: 32 columns
        #   - K step: 8 (32x32x8) head-dim elements per atom
        QK_MFMA_N = 32
        QK_K_STEP = 8 if USE_MFMA_32X32X8 else 16
        PV_K_STEP = QK_K_STEP
        QK_K_ITERS = HD // QK_K_STEP
        QK_N_TILES = T // QK_MFMA_N
        PV_K_ITERS = T // PV_K_STEP
        PV_N_TILES = HD // 32
    else:
        # gfx942 QK/PV geometry: narrow 16x16x16 atoms only.
        #   - per-lane C: <4 x f32>
        #   - N tile: 16 columns
        #   - K step: 16 head-dim elements (no wide-K atom on gfx942)
        # So ``QK_K_ITERS = HD // 16`` and ``PV_K_ITERS = T // 16``; the PV V
        # B-operand is built from strided LDS loads (no ds_read_tr16_b64).
        QK_MFMA_N = MFMA_N
        QK_K_STEP = 16
        PV_K_STEP = 16
        QK_K_ITERS = HD // QK_K_STEP
        QK_N_TILES = T // QK_MFMA_N
        PV_K_ITERS = T // PV_K_STEP
        PV_N_TILES = HD // MFMA_N

    NUM_WARPS = spec.num_warps
    WAVE = 64
    THREADS = NUM_WARPS * WAVE
    # CDNA3 (gfx942) async global->LDS DMA width cap.
    #
    # ``llvm.amdgcn.raw.ptr.buffer.load.lds`` on CDNA3 can only legalise a
    # per-lane DWORD (4-byte) load-to-LDS. The wider b96/b128 load-to-LDS
    # forms (dwords 3 and 4) are a CDNA4 (gfx950)-only feature; emitting a
    # 16-byte (dwords=4) LDS DMA on gfx942 makes the backend abort with
    # ``LLVM ERROR: Do not know how to expand this operator's operand!``.
    # The tiled-2D loaders were templated from the gfx950 async-copy width
    # (16 bytes/lane), so on gfx942 we clamp every async-LDS-DMA to 1 dword
    # and issue proportionally more calls to move the same bytes. The LDS
    # deposit stays lane-contiguous, so the [T, HD] K_lds / V_lds layout
    # (and the strided-V B-operand math that depends on it) is unchanged.
    #
    # There is no catalog field for this today: ``memory.buffer_load_max_dwords``
    # describes the *register* vector buffer-load width (4 on both gfx942 and
    # gfx950) and does NOT capture the narrower load-*to-LDS* limit, so we
    # encode the gfx942-specific value here rather than touching core/arch/.
    ASYNC_LDS_MAX_DWORDS = 1
    ASYNC_LDS_MAX_BYTES_PER_LANE = ASYNC_LDS_MAX_DWORDS * 4
    BLOCK_M_PER_WARP = spec.block_m_per_warp
    # Number of stacked MFMA-M=16 atoms per warp's M dimension. For
    # ``block_m_per_warp=16`` this is 1 (the original kernel); for
    # ``block_m_per_warp=32`` it's 2, so each warp does two stacked
    # ``mfma_f32_16x16x*`` atoms per QK / PV step.
    M_ATOMS_PER_WARP = BLOCK_M_PER_WARP // MFMA_M
    # Per-lane accumulator register slot count per N-tile. The 16x16
    # MFMA distribution gives each lane 4 row slots within one 16-row
    # atom (rows ``lane_rg*4..lane_rg*4+3``); stacking ``M_ATOMS_PER_WARP``
    # atoms gives ``4 * M_ATOMS_PER_WARP`` slots per lane per N-tile.
    REGS_PER_LANE = spec.regs_per_lane  # 4 for M=16, 8 for M=32
    SOFTMAX_STATE_SLOTS = (
        1
        if USE_MFMA_32X32 and TRANSPOSED_QK_32X32 and TRANSPOSED_SCALAR_STATE
        else REGS_PER_LANE
    )

    name = spec.kernel_name()
    b = IRBuilder(name)
    b.kernel.attrs["max_workgroup_size"] = THREADS
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    if spec.use_agpr_alloc_zero:
        b.kernel.attrs["agpr_alloc"] = (0, 0)

    # ---------------- parameter declarations ----------------
    output = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr",
        PtrType(kv_io_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(kv_io_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sinks = b.param("sink_ptr", PtrType(dtype, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    alibi_slopes_ptr = b.param(
        "alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4
    )
    qq_bias_ptr = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale_p = b.param("scale", F32)
    k_scale_p = b.param("k_scale", F32)
    v_scale_p = b.param("v_scale", F32)
    _out_scale = b.param("out_scale", F32)
    softcap_p = b.param("softcap", F32)
    num_seqs_p = b.param("num_seqs", I32)
    bt_stride_p = b.param("block_table_stride", I32)
    qq_bias_stride0_p = b.param("qq_bias_stride_0", I32)

    if spec.use_q_major_grid:
        q_block_global_idx = b.block_id_x()
        kv_head_idx = b.block_id_y()
    else:
        kv_head_idx = b.block_id_x()
        q_block_global_idx = b.block_id_y()
    tid = b.thread_id_x()

    # Wave decomposition. For NUM_WARPS=1 this collapses to `wave_id=0,
    # lane=tid`, exactly the single-warp behaviour. For NUM_WARPS>1 each
    # wave owns rows `[wave_id*16, (wave_id+1)*16)` of the M dimension.
    if NUM_WARPS == 1:
        lane = tid
        wave_row_base = b.const_i32(0)
    else:
        lane = b.mod(tid, b.const_i32(WAVE))
        wave_id = b.div(tid, b.const_i32(WAVE))
        wave_row_base = b.mul(wave_id, b.const_i32(BLOCK_M_PER_WARP))

    # ---------------- seq lookup ----------------
    seq_idx = binary_search_seq_idx(
        b,
        cu_q,
        q_block_global_idx,
        num_seqs_p,
        block_q=BLOCK_Q,
        iterations=spec.binary_search_iters,
    )
    cu_q_start = b.global_load_i32(cu_q, seq_idx)
    cu_q_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    cur_batch_q_len = b.sub(cu_q_stop, cu_q_start)
    q_block_start_idx = b.add(b.div(cu_q_start, b.const_i32(BLOCK_Q)), seq_idx)
    q_block_local_idx = b.sub(q_block_global_idx, q_block_start_idx)
    seq_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(seq_len, cur_batch_q_len)

    qb_start_pos = b.mul(q_block_local_idx, b.const_i32(BLOCK_Q))
    with b.scf_if(b.cmp_ge(qb_start_pos, cur_batch_q_len)):
        b.ret()

    # ---------------- LDS layout ----------------
    # Q is loaded once. K and V are double-buffered [2, T, HD] in natural
    # row-major layout (async DMA deposits lane-contiguous). The PV MFMA
    # B operand is fetched via `ds_read_b64_tr_b16` with per-lane addresses
    # following CK Tile's `TransposeLDSLayout<M=16,K=16,B=1>` (single read
    # for K=16, 2 reads for K=32). This collapses the 4-8 scalar
    # `ds_read_u16` per atom (16-way bank conflicted) into 1-2 wide
    # transpose reads with the MFMA B distribution baked in.
    # Epilogue staging buffer: ``Acc_lds`` re-uses LDS across multiple
    # ``OUT_STRIPE_COLS``-wide stripes (one stripe = ``OUT_STRIPE_COLS /
    # MFMA_N`` consecutive PV N-tiles). Two regimes:
    #
    # 1. ``HD <= 64``: use 32-col stripes. The big LDS savings (e.g. NW=4
    #    T=64 HD=64 drops 16 KiB → 2 KiB) crosses MI355X's 2 → 3 WGs/CU
    #    threshold and is worth a couple of extra sync barriers per CTA.
    #
    # 2. ``HD >= 128``: use full-HD stripes. The old F32 Acc_lds was
    #    ``BLOCK_M * HD * 4`` bytes; the new dtype-only Acc_lds is
    #    ``BLOCK_M * HD * 2`` (still a 2× LDS reduction, but inside the
    #    same WG/CU class). Splitting into 32-col stripes here would add
    #    3-7 extra sync barriers per CTA without an occupancy gain --
    #    that's a measurable regression for decode workloads (the
    #    HD=128 + BLOCK_M=16 NW=1 path takes only ~6 µs of MFMA + small
    #    amounts of LDS work, so a few hundred extra cycles of barrier
    #    is visible).
    if HD <= 64:
        OUT_STRIPE_COLS = 32
    else:
        OUT_STRIPE_COLS = HD
    OUT_STRIPES = HD // OUT_STRIPE_COLS
    assert (
        HD % OUT_STRIPE_COLS == 0
    ), f"HD={HD} must split evenly into {OUT_STRIPE_COLS}-col stripes"
    # ---- LDS pad swizzle (Q_lds, P_lds) ----
    # Q_lds and P_lds are read with a pattern where 16 lanes (one MFMA
    # 16x16 row-group) hit DIFFERENT rows but the SAME column. With a
    # natural [BLOCK_M, HD] row-major layout and row stride = HD halves
    # = HD*2 bytes, all 16 lanes land on the SAME 32-byte LDS bank cycle
    # → 16-way bank conflict.
    #
    # Padding each row by 8 halves (16 bytes) breaks the bank alignment:
    # row r now occupies bytes ``r * (HD*2 + 16)`` so the bank index
    # ``(r * (HD/2 + 4)) % 32`` cycles every ~8 rows instead of every 1.
    # That converts the worst case from 16-way to 2-way bank conflict.
    # An internal conv kernel optimization study measured +43% throughput on
    # MI355X gfx950 from this same trick.
    #
    # We pad by exactly 16 bytes (8 halves) -- not 4 -- to preserve
    # 16-byte alignment for ``ds_write_b128``/``ds_read_b128`` (vec8).
    # A 4-byte pad makes row 1's start at byte ``HD*2 + 4`` which is
    # only 4-byte aligned for HD=64, downgrading the vec8 store to
    # scalar and erasing the LDS savings.
    #
    # K_lds and V_lds cannot be padded the same way because the async
    # DMA (``raw_ptr_buffer_load_lds``) writes lane-contiguous bytes; a
    # padded row stride would corrupt the layout. V_lds reads use
    # ``ds_read_tr16_b64`` which is bank-conflict-free by design. K_lds
    # reads (in QK) still go through regular ``smem_load_vN`` — converting
    # them to ``ds_read_tr16`` is a follow-up.
    # ---- K/V LDS buffering ----
    # **K is double-buffered** (correctness requirement -- see note below).
    # **V is single-buffered** (safe because V[i+1] is issued in iter i+1
    # AFTER the iter-start ``vmcnt=0, lgkmcnt=0`` drain, which guarantees
    # PV[i]'s LDS reads have retired before V[i+1] can write the shared
    # slot. Saves 8 KiB LDS per CTA).
    #
    # **Q is held in per-lane VGPRs across the kvloop** when its size fits
    # in K_lds[0] (``BLOCK_M * HD <= T * HD`` i.e. ``BLOCK_M <= T``). The
    # prologue cooperatively writes Q into ``K_lds[0]`` (treating it as
    # scratch), syncs, then each lane gathers its MFMA A-operand into 16
    # VGPRs and the K[0] prefetch overwrites the K_lds[0] slot. This
    # eliminates Q's permanent LDS allocation entirely (saving another 8
    # KiB on the NW=4 prefill config). The QK MFMA reads Q from registers,
    # eliminating the per-iter 16-way bank-conflicted Q_lds reads.
    #
    # When ``BLOCK_M > T`` (only the NW=8, BLOCK_M=128 config in our
    # supported set) Q doesn't fit in K_lds[0] and we fall back to a
    # dedicated Q_lds allocation.
    #
    # The K single-buffer variant was tried (see git history) and silently
    # produced wrong results on prefill q>=1k: even with ``wait_K[i] →
    # QK[i] → issue V[i] → issue K[i+1] → softmax → wait_V[i] → PV[i]``,
    # the async-DMA write of ``K[i+1]`` to the shared K slot can race with
    # the tail end of QK[i]'s LDS reads through ``raw_ptr_buffer_load_lds``'s
    # lgkmcnt accounting, corrupting the working tile. K must stay
    # double-buffered.
    # K_lds dtype: bf16 for the standard path (sync FP8 dequant or
    # async bf16 DMA both target a bf16 working slab). When the FP8
    # native-MFMA mode is on (`use_fp8_mfma_qk`), K_lds holds raw fp8
    # bytes (no dequant); this halves the per-buffer footprint
    # (T*HD instead of T*HD*2 bytes) so the double-buffer fits in
    # 8 KB instead of 16 KB.
    K_LDS_DTYPE = FP8E4M3 if FP8_MFMA_QK else dtype
    V_LDS_DTYPE = FP8E4M3 if FP8_MFMA_PV else dtype
    P_LDS_DTYPE = FP8E4M3 if FP8_MFMA_PV else dtype
    Q_BYTES = BLOCK_M * HD * 2
    K_SLICE_HD = 32
    K_SLICE_SLOTS = 3
    K_SLICED_ACTIVE = K_SLICED_RING and USE_MFMA_32X32X8 and TRANSPOSED_QK_32X32
    # K_BUF_BYTES depends on the K_LDS_DTYPE (1 byte for fp8, 2 for bf16).
    K_LDS_ELEM_BYTES = 1 if K_LDS_DTYPE == FP8E4M3 else 2
    K_BUF_BYTES = (T * K_SLICE_HD if K_SLICED_ACTIVE else T * HD) * K_LDS_ELEM_BYTES
    # Number of K_lds double-buffer slots. Default 2 (next-tile prefetch overlaps
    # softmax+PV). The OCCUPANCY lever single-buffers K (1 slot) to cut loop LDS
    # from 48KB -> 32KB and reach 2 wg/CU; the next-K issue is then ordered after
    # a post-QK LDS-read drain to close the shared-slot DMA race.
    K_BUFS = K_SLICE_SLOTS if K_SLICED_ACTIVE else (1 if K_SINGLE_BUF else 2)
    K_TOTAL_BYTES = K_BUFS * K_BUF_BYTES
    # Q can alias K_lds when (a) the dtypes match and (b) it fits in the
    # full K_lds region (both slots). When ``BLOCK_M <= T`` Q fits in one
    # slot (rows 0..BLOCK_M of K_lds[0]); when ``BLOCK_M > T`` Q spills
    # into K_lds[1] (rows T..2T map to K_lds[1, row-T, :]). The gather +
    # store use ``(buf, row, col)`` indexing in both cases.
    # Aliasing requires same dtype because the Q_lds writes use bf16 stores;
    # for the fp8-MFMA path K_lds is fp8 so a dedicated Q_lds slab is needed.
    Q_DIRECT_GLOBAL = K_SLICED_ACTIVE or spec.use_q_direct_global
    Q_ALIAS_K = (
        (not Q_DIRECT_GLOBAL) and (K_LDS_DTYPE == dtype) and Q_BYTES <= K_TOTAL_BYTES
    )
    Q_USES_DUAL_SLOT = Q_ALIAS_K and BLOCK_M > T
    if K_SLICED_ACTIVE:
        K_lds = b.smem_alloc(K_LDS_DTYPE, [K_BUFS, T, K_SLICE_HD], name_hint="KldsS")
    else:
        K_lds = b.smem_alloc(K_LDS_DTYPE, [K_BUFS, T, HD], name_hint="Klds")
    V_BUFS = 1  # single-buffer V (race-free: see comment above)
    # ---- V_lds bank-deconflict swizzle (HIPDNN_GFX942_SWIZZLE_VLDS) ----
    # The plain-atom PV B-operand is read with 4 strided ``ds_read_u16`` loads
    # over the row-major ``V_lds[T, HD]`` tile (``_strided_v_b_operand`` below).
    # For each read the 32 lanes of a wave-half map onto only 8 LDS banks
    # (16-col atom window = 8 banks) -> a hard 4-way bank conflict, which the
    # gfx942 rocprof trace flagged as the dominant stall (57% of busy cycles
    # on the V read, ~11 conflicts/LDS-instruction). gfx942 has no
    # ``ds_read_tr16`` transpose read to avoid it.
    #
    # The conflict is structural: bank = ((row*HD + col) >> 1) & 31, and for
    # HD in {64, 128} the ``row*HD`` term is always 0 mod 32 banks, so the row
    # never reaches the bank and 32 lanes collapse onto 8 banks. The fix is to
    # give consecutive rows a *padded* row stride so they land on different
    # banks. V_lds is written by a hardware-contiguous async DMA (no per-lane
    # dest control), so the pad can only be inserted at a DMA *call* boundary.
    # Each call writes ``V_ROWS_PER_CALL = THREADS*2 / HD`` whole rows. Padding
    # at the call boundary moves the per-row bank far enough to break the
    # 4-way collision down to the irreducible 2-way (the col-parity pair, which
    # share one 4-byte bank word) ONLY when a call spans <= 2 rows -- i.e. the
    # 4 colliding rows (lane_rg*4 apart) straddle a padded boundary. With
    # 8 rows/call (the D64 nw=4 config) all 4 colliding rows fall inside one
    # contiguous call, so the boundary pad cannot separate them; the swizzle is
    # auto-disabled there (and on the fp8 / 32x32 / register-pv paths, whose
    # V-lane map differs). The read inverts the same per-call pad with cheap
    # shift/and on the already-live row value (VGPR-neutral, no per-element
    # address recompute).
    # The contiguous DMA interleaves (wave, call, lane): for the bf16/f16
    # path each wave deposits ``WAVE * (ASYNC_LDS_MAX_BYTES_PER_LANE//2)``
    # halves per call lane-contiguously, and the per-call/per-wave bases stack
    # so the physical LDS half equals the logical ``token*HD + dim``. The
    # swizzle's per-row pad only reduces cleanly to a ``row -> (group, within)``
    # split when **one wave deposits exactly one V row per call** -- i.e.
    # ``WAVE * halves_per_lane == HD`` -- so that ``within = wave`` and
    # ``group = call``. Require that (true for the D128 nw=2 ship config:
    # 64*2 == 128). When a wave spans several rows per call (e.g. D64 nw=4,
    # HD=64: 8 rows/call) the row pad cannot be expressed as a per-call/per-wave
    # base offset, so the swizzle is auto-disabled there.
    _V_HALVES_PER_LANE = ASYNC_LDS_MAX_BYTES_PER_LANE // 2
    _V_ROW_PER_WAVE_CALL = (WAVE * _V_HALVES_PER_LANE) == HD
    V_ROWS_PER_CALL = NUM_WARPS  # one row per wave per call when eligible
    # LDS-budget headroom guard: the swizzle adds ``N_GROUPS * V_ROWS_PER_CALL *
    # PAD`` halves to V_lds. ``supports_tiled_2d`` sized V_lds at the natural
    # (unpadded) figure, so disable the swizzle if the pad would push the total
    # estimate past the gfx942 LDS budget (otherwise a borderline config would
    # comgr-abort only when the env flag is set). Mirrors the gate's formula.
    try:
        from ...core.arch import ArchTarget as _ArchTarget

        _LDS_CAP = _ArchTarget.from_gfx(arch).lds_capacity_bytes
    except Exception:  # noqa: BLE001
        _LDS_CAP = 65536
    _swz_extra_bytes = (T // max(V_ROWS_PER_CALL, 1)) * V_ROWS_PER_CALL * 8 * 2
    _out_stripe_b = 32 if HD <= 64 else HD
    _lds_natural = (
        2 * T * HD * 2
        + T * HD * 2
        + BLOCK_M * (T + 8) * 2
        + (0 if BLOCK_M <= 2 * T else BLOCK_M * HD * 2)
        + BLOCK_M * _out_stripe_b * 2
    )
    _swz_fits = (_lds_natural + _swz_extra_bytes) <= _LDS_CAP
    # ---- Stream B: conflict-free transposed V LDS (flash level 3) ----
    # On the transposed-x8 PV path (TRANSPOSED_X8) the A operand is V^T. The
    # naive feed reads it from a row-major ``V_lds[T, HD]`` tile with 4 strided
    # ``ds_read_u16`` over 4 distinct token rows at one head-dim column -> the 32
    # wave-half lanes (varying head-dim) collapse onto the same LDS bank set (the
    # rocprof bank-conflict bound, ~7.25 conflicts/LDS-inst). CK-Tile's gfx942
    # ``qr_ks_vs`` recipe lays V K-contiguous so the 4 tokens a lane needs are
    # CONTIGUOUS -> ONE wide ``ds_read_b64``, bank-spread by a kKPack pad. We
    # adopt that: store V TRANSPOSED ``[HD, T + V_T_PAD]`` (head-dim outer, token
    # contiguous). gfx942 async DMA cannot transpose HBM->LDS, so the transpose
    # is paid ONCE on the LDS store via a sync ``global_load + shaped
    # smem_store`` pass (CK technique #3, mirroring ``_issue_v_fp8_mfma_stripe``
    # which already fills a non-natural V_lds). f16-only, requires the
    # transposed-x8 orientation; signature-gated so it caches distinctly.
    _v_t_pad = 8
    _v_t_extra_bytes = HD * _v_t_pad * 2
    _v_t_fits = (_lds_natural + _v_t_extra_bytes) <= _LDS_CAP
    # ACCURATE transposed-x8 footprint for the cfv-store gate. The generic
    # ``_lds_natural`` above over-counts the transposed-x8 path (it assumes
    # P_lds present, K always double-buffered, full Acc_lds) and walls the wide
    # (nw=4 / T=128) geometry the FLASH_WIDE=4 baseline actually runs. On
    # TRANSPOSED_X8: P_lds=0 (P^T in registers); Acc_lds aliases the loop-dead
    # K/V region (~0 to the binding peak); Q_lds=0 when BLOCK_M<=2*T; K is single-
    # buffered when BLOCK_M<=T else double. V_lds is the transposed [HD,T+pad].
    # Mirrors compile_service's ``_lds_bytes_transposed_x8``.
    _x8_k_slots = 1 if (BLOCK_M <= T) else 2
    _x8_q_lds = 0 if (BLOCK_M <= 2 * T) else BLOCK_M * HD * 2
    _v_t_fits_x8 = (
        _x8_k_slots * T * HD * 2 + (T + _v_t_pad) * HD * 2 + _x8_q_lds
    ) <= _LDS_CAP
    # Both conflict-free-V vehicles share the SAME transposed [HD,T+pad] V_lds
    # layout + consumer; they differ only in the STORE fill. TRANSPOSED_V gates
    # the shared layout/consumer; TRANSPOSED_V_STORE picks the store vehicle
    # (True = vehicle (c), register-load + perm_b32 transpose; False = the
    # vehicle (a) sync gather of the original CONFLICT_FREE_V path).
    # The store path (vehicle (c)) uses the accurate transposed-x8 footprint so
    # the wide (nw=4 / T=128) baseline geometry is admitted; the legacy gather
    # (vehicle (a)) keeps the original conservative ``_v_t_fits`` byte-identically.
    _v_t_fits_eff = _v_t_fits_x8 if CONFLICT_FREE_V_STORE else _v_t_fits
    TRANSPOSED_V = (
        (CONFLICT_FREE_V or CONFLICT_FREE_V_STORE)
        and USE_MFMA_32X32X8
        and TRANSPOSED_QK_32X32
        and not FP8_MFMA_PV
        and not FP8_MFMA_QK
        and not KV_FP8
        and not FAST_PAGED_KV_DESC
        and V_LDS_DTYPE == dtype
        # Conflict-free V (cfv/cfvst) is legal for both 2-byte element types:
        # the in-register 2x2 transpose (perm_b32 on raw i32 words) and the
        # ds_read_b64 / ds_write_b{32,64} LDS feed are byte-identical for fp16
        # and bf16. (The K=16 bf16 atom is gfx950-only, but cfvst rides the
        # gfx942-legal K=8 32x32x8 path, so bf16 is fine here.)
        and dtype in (F16, BF16)
        and HD in (64, 128)
        and (HD % 8 == 0)
        and (T * HD) % THREADS == 0
        and _v_t_fits_eff
    )
    # Store-vehicle selector: vehicle (c) (register-load + in-register perm_b32
    # transpose) when the store-path flag drove TRANSPOSED_V; the 2x2 f16
    # transpose needs T and HD both even (HD%8==0 already; T%2 below).
    TRANSPOSED_V_STORE = TRANSPOSED_V and CONFLICT_FREE_V_STORE and (T % 2 == 0)
    SWIZZLE_VLDS = (
        os.environ.get("HIPDNN_GFX942_SWIZZLE_VLDS", "1") == "1"
        and not FP8_MFMA_PV
        and not FP8_MFMA_QK
        and not USE_MFMA_32X32
        and not REGISTER_PV
        and not TRANSPOSED_V
        and V_LDS_DTYPE == dtype
        and _V_ROW_PER_WAVE_CALL
        and NUM_WARPS in (1, 2)
        and (T % V_ROWS_PER_CALL == 0)
        and HD in (64, 128)
        and not FAST_PAGED_KV_DESC  # FAST path uses a distinct per-call voff
        and _swz_fits  # don't blow the LDS budget the static gate sized natural
    )
    # Pad of 8 halves (16 bytes) per V row: shifts each row's bank base by
    # (8 >> 1) = 4 banks, the value the static probe shows breaks the 4-way
    # collision down to the irreducible 2-way (col-parity pair). 16 bytes keeps
    # the contiguous per-wave-row DMA store 16-byte aligned.
    V_LDS_PAD = 8 if SWIZZLE_VLDS else 0
    # Padded per-row stride and per-call (= V_ROWS_PER_CALL rows) group stride,
    # both in halves. ``V_GROUP_SHIFT`` splits a row into (group, within=wave)
    # with a cheap shift/and on the already-live row value.
    V_LDS_STRIDE = HD + V_LDS_PAD
    V_GROUP_STRIDE = V_ROWS_PER_CALL * V_LDS_STRIDE
    V_GROUP_SHIFT = V_ROWS_PER_CALL.bit_length() - 1 if SWIZZLE_VLDS else 0
    # Transposed-V column stride (in halves): the K=token axis is padded by
    # ``V_T_PAD`` so consecutive head-dim rows land on a rotated bank base (CK's
    # ``+kKPack``). T=64 f16 = 128 B = 32 banks * 4 -> one full bank sweep; the
    # pad rotates each head-dim row's bank base by (8>>1)=4 banks.
    V_T_PAD = _v_t_pad if TRANSPOSED_V else 0
    V_T_STRIDE = T + V_T_PAD
    # CK Tile V LDS descriptor equivalent for the transposed feed:
    #
    #   base dims   [K/kKPack, N/NPerRow, NPerRow, kKPack]
    #   base stride [(N/NPerRow)*(PixelsPerRow+kKPack),
    #                PixelsPerRow+kKPack, kKPack, 1]
    #
    # where logical N=head-dim and K=token. The simple [HD, T+pad] layout is
    # correct but not the production bank layout; use the CK-packed slot map by
    # default, and keep a kill-switch for quick A/B.
    V_T_CK_LAYOUT = TRANSPOSED_V and spec.use_conflict_free_v_ck_vlds
    V_T_KPACK = 8
    V_T_PIXELS_PER_ROW = 64  # 32 LDS banks * 4 bytes / sizeof(f16)
    V_T_NPER_ROW = V_T_PIXELS_PER_ROW // V_T_KPACK
    V_T_NGROUPS = HD // V_T_NPER_ROW
    V_T_KGROUPS = T // V_T_KPACK
    V_T_GROUP_STRIDE = V_T_PIXELS_PER_ROW + V_T_KPACK
    V_T_CK_SLOTS = V_T_KGROUPS * V_T_NGROUPS * V_T_GROUP_STRIDE
    if TRANSPOSED_V:
        # Conflict-free transposed V LDS: ``[V_BUFS, HD, T + V_T_PAD]``. Logical
        # ``V[token, dim]`` lives at ``V_lds[0, dim, token]`` (K=token contiguous
        # along the fast/inner axis, padded by ``V_T_PAD``). The transposed-x8 PV
        # A-operand read for a fixed head-dim row is a contiguous ``ds_read_b64``
        # over consecutive tokens.
        if V_T_CK_LAYOUT:
            V_lds = b.smem_alloc(
                V_LDS_DTYPE, [V_BUFS, V_T_CK_SLOTS], name_hint="VldsTck"
            )
        else:
            V_lds = b.smem_alloc(
                V_LDS_DTYPE, [V_BUFS, HD, V_T_STRIDE], name_hint="VldsT"
            )
    elif FP8_MFMA_PV:
        # Native-fp8 PV uses ds_read_b64_tr_b8. The validated lane mapping
        # (HIP probe in /tmp/probe_tr_b8_stripe.hip) is:
        #   for lane L in 16-lane group G = L/16, position l = L%16:
        #     - K-row picked per slot S = G*8 + S    (S in 0..7)
        #     - N-col picked            = l % 8       (lanes 8..15 of a group
        #                                              redundantly select N=0..7)
        #     - vaddr provided by lane L =
        #         (G*8 + (l/2)) * row_stride_bytes
        #   and the instruction requires LDS row_stride = 16 bytes.
        # So each call gives K=32 x N=8 (half of an MFMA f32_16x16x32_fp8
        # B operand). To get N=16, two calls + a per-lane select on
        # (lane%16 < 8) are needed. The natural V LDS layout for this is
        # [N_STRIPES = HD/16, T, 16] so each stripe is row-stride 16 and
        # the second read just adds +8 to the base address.
        N_STRIPES = HD // 16
        V_lds = b.smem_alloc(
            V_LDS_DTYPE, [V_BUFS, N_STRIPES, T, 16], name_hint="VldsStripe"
        )
    elif SWIZZLE_VLDS:
        # Bank-deconflicted V_lds: flat ``[V_BUFS, N_GROUPS * V_GROUP_STRIDE]``.
        # Each DMA call fills one group (``V_ROWS_PER_CALL`` rows, one row per
        # wave), with every row stored at the padded stride ``V_LDS_STRIDE``;
        # the V_LDS_PAD-half gap after each row rotates that row's bank base.
        # Reads recover the flat slot via ``_v_load1`` below.
        V_N_GROUPS = T // V_ROWS_PER_CALL
        V_N_SLOTS = V_N_GROUPS * V_GROUP_STRIDE
        V_lds = b.smem_alloc(V_LDS_DTYPE, [V_BUFS, V_N_SLOTS], name_hint="Vlds")
    else:
        V_lds = b.smem_alloc(V_LDS_DTYPE, [V_BUFS, T, HD], name_hint="Vlds")

    def _v_t_slot(dim: Value, tok: Value) -> Value:
        """CK Tile transposed-V slot for logical ``V_lds[dim, token]``."""
        k_group = b.div(tok, b.const_i32(V_T_KPACK))
        k_inner = b.mod(tok, b.const_i32(V_T_KPACK))
        n_group = b.div(dim, b.const_i32(V_T_NPER_ROW))
        n_inner = b.mod(dim, b.const_i32(V_T_NPER_ROW))
        return b.add(
            b.add(
                b.mul(
                    k_group,
                    b.const_i32(V_T_NGROUPS * V_T_GROUP_STRIDE),
                ),
                b.mul(n_group, b.const_i32(V_T_GROUP_STRIDE)),
            ),
            b.add(b.mul(n_inner, b.const_i32(V_T_KPACK)), k_inner),
        )

    def _v_t_store(dim: Value, tok: Value, value: Value, n: int) -> None:
        if V_T_CK_LAYOUT:
            b.smem_store_vN(V_lds, [b.const_i32(0), _v_t_slot(dim, tok)], value, n)
        else:
            b.smem_store_vN(V_lds, [b.const_i32(0), dim, tok], value, n)

    def _v_t_load(dim: Value, tok: Value, *, n: int) -> Value:
        v_buf0 = b.const_i32(0)
        if V_T_CK_LAYOUT:
            return b.smem_load_vN(V_lds, v_buf0, _v_t_slot(dim, tok), dtype=dtype, n=n)
        return b.smem_load_vN(V_lds, v_buf0, dim, tok, dtype=dtype, n=n)

    def _v_load1(v_buf: Value, v_row: Value, v_n_col: Value) -> Value:
        """Load a single V_lds element ``V[v_row, v_n_col]`` as ``<1 x dtype>``.

        When ``SWIZZLE_VLDS`` the physical store is the bank-rotated flat
        layout, so the (row, col) logical coordinate maps to its padded flat
        slot: ``group*V_GROUP_STRIDE + within*V_LDS_STRIDE + col`` with
        ``group = row >> V_GROUP_SHIFT`` and ``within = row & (rows-1)``. Both
        are cheap bit ops on the already-live ``v_row`` (no per-element address
        recompute -> VGPR-neutral). When the swizzle is off this is exactly the
        natural ``V_lds[v_buf, v_row, v_n_col]`` 3D access.
        """
        if not SWIZZLE_VLDS:
            return b.smem_load_vN(V_lds, v_buf, v_row, v_n_col, dtype=dtype, n=1)
        group = b.lshr(v_row, b.const_i32(V_GROUP_SHIFT))
        within = b.land(v_row, b.const_i32(V_ROWS_PER_CALL - 1))
        slot = b.add(
            b.add(
                b.mul(group, b.const_i32(V_GROUP_STRIDE)),
                b.mul(within, b.const_i32(V_LDS_STRIDE)),
            ),
            v_n_col,
        )
        return b.smem_load_vN(V_lds, v_buf, slot, dtype=dtype, n=1)

    # The x8-transposed path (Stream C) keeps P^T in registers and consumes it
    # directly as the PV B-operand -- the P_lds bridge is never written or read,
    # so skip its allocation entirely. That is the whole point: dropping the
    # P_lds round-trip is the LDS-traffic / L2-pressure win this orientation
    # buys. (REGISTER_PV is the bf16 16x16 register-P path; orthogonal.)
    TRANSPOSED_X8 = USE_MFMA_32X32X8 and TRANSPOSED_QK_32X32
    if not REGISTER_PV and not TRANSPOSED_X8:
        # P_lds row stride padding (16 bytes = 8 halves) to eliminate 4-way
        # LDS bank conflict on the softmax `ds_write_b16` stores. With row
        # stride T*2 = 128 bytes = exactly 32 banks, lanes 0, 16, 32, 48
        # (same lane_col but different lane_rg → different rows) all hit
        # bank 0 because (row*128) % 128 == 0 for any row. Padding by
        # 8 halves shifts row-N bank to bank-(N*16)%32, breaking the
        # alias. Cost: +16 bytes per row × BLOCK_M rows = up to 2 KiB LDS
        # at BLOCK_M=128. Profiling (rocprofv2 SQ_LDS_BANK_CONFLICT) showed
        # ~12× LDS conflict cycles per LDS instruction on the FP8 mid-
        # prefill kernel (25.7M conflicts vs 2.0M LDS ops) before the
        # padding; the pad makes those writes single-cycle.
        # In native fp8-MFMA mode P_lds holds quantised fp8 probabilities.
        # Keep the same 16-byte row-padding distance by using 16 bytes / 1 byte.
        P_LDS_PAD = 16 if FP8_MFMA_PV else 8
        P_lds = b.smem_alloc(P_LDS_DTYPE, [BLOCK_M, T + P_LDS_PAD], name_hint="Plds")
    # ---- FP8 K/V staging (round-2 async-DMA path) ----
    # When KV_FP8, the loader is split into two phases:
    #   1. async-DMA raw fp8 bytes from HBM into the fp8 staging slab
    #      (K_fp8_lds / V_fp8_lds) — same hardware primitive as the bf16
    #      async load, just with fp8 byte-stride descriptors and one
    #      call/wave for T=64/HD=64 (fp8 is half the bytes of bf16 so we
    #      need half the calls).
    #   2. After ``s_waitcnt vmcnt=0; s_barrier``, sync dequant from the
    #      fp8 staging slab into the existing K_lds / V_lds (bf16) — each
    #      thread reads 8 fp8, applies ``cvt_fp8_to_f32 * scale -> bf16``,
    #      writes 8 bf16. The dequant is LDS-to-LDS (no HBM stall) and
    #      runs entirely from the same WG (no cross-WG synchronisation).
    #
    # The bf16 K_lds and V_lds layouts are unchanged so the rest of the
    # kernel (Q gather, QK MFMA, P_lds publish, PV MFMA, epilogue) doesn't
    # know about FP8. Only allocate the fp8 staging slabs when actually
    # using the async-FP8 path (round 2 v1) -- otherwise they'd waste LDS
    # and starve occupancy on the round-1 sync FP8 path.
    if KV_FP8 and spec.use_fp8_mfma_qk:
        K_fp8_lds = b.smem_alloc(FP8E4M3, [2, T, HD], name_hint="Kfp8lds")
        V_fp8_lds = b.smem_alloc(FP8E4M3, [1, T, HD], name_hint="Vfp8lds")
    if Q_DIRECT_GLOBAL:
        # Sliced-K ring is intentionally smaller than a full Q tile; Q is
        # gathered directly from global into Q32_reg below and never uses LDS.
        Q_lds = K_lds
    elif Q_ALIAS_K:
        # Reuse K_lds[0] (and K_lds[1] if BLOCK_M > T) as Q-load scratch.
        # Q is gathered to VGPRs after, then the K-prefetch overwrites
        # the slot(s).
        Q_lds = K_lds
    else:
        Q_lds = b.smem_alloc(dtype, [BLOCK_M, HD], name_hint="Qlds")
    # NOTE: Acc_lds bank-conflict padding was tested (8 bytes = 4 halves
    # to break the 4-way conflict on the epilogue per-element stores)
    # but the bench showed mostly noise (-49% to +95% on the same bucket
    # depending on cu_seqlens variance, overall break-even on win count).
    # Acc_lds is used only in the epilogue (one set of writes/reads per
    # CTA, vs P_lds's per-iter writes), so the bank-conflict cost is a
    # one-shot overhead and the pad's LDS budget cost doesn't pay back.
    # Keep Acc_lds tight.
    Acc_lds = b.smem_alloc(dtype, [BLOCK_M, OUT_STRIPE_COLS], name_hint="Aclds")
    # NOTE: A staged Acc_lds epilogue for the transposed 32x32 path was
    # tested (one 32-d stripe per MFMA tile, bank-padded to 17 dwords/row
    # for HD>=128). It regressed on every tested shape (HD=64/128, single
    # and multi-batch) by 15-100% vs the per-lane direct global-store
    # epilogue, because the transposed accumulator already produces
    # 32 adjacent-token vec16-per-lane stores per cycle that the memory
    # subsystem coalesces well, and the LDS bounce + barrier added more
    # cost than the gather it saved. Keep the direct scalar epilogue.

    # ---- CK Tile `TransposeLDSLayout<M=16, K=*, B=1>` lane formulas ----
    # ``TransposeLdsReader`` materializes the per-lane row / col SSA
    # values once and exposes ``row(k_offset, read)`` for use inside
    # the PV K iteration loop. These are per-warp formulas, so the
    # bind site uses the in-warp ``lane`` id (not the global thread id).
    pv_tr_reader = TransposeLdsReader(K=PV_K_STEP, M=16).bind(b, lane)
    tr_col_lane = pv_tr_reader.col

    # ---------------- constants ----------------
    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    qk_scale = b.fmul(scale_p, rcp_ln2)
    if FP8_NATIVE_QK:
        # Native fp8 QK: K is RAW fp8 (k_scale NOT folded in LDS) and Q is
        # raw fp8, so the f32 accumulator is in fp8-code units. Fold k_scale
        # into qk_scale post-MFMA: S_phys = (Qfp8 . Kfp8) * k_scale, then
        # * softmax_scale / ln(2). (Q was quantized from a unit-scale bf16,
        # so it carries no separate scale.)
        qk_scale = b.fmul(qk_scale, k_scale_p)
    # In the fp8-K-LDS path the QK MFMA dequants K -> bf16 in register
    # using ``k_scale``; the MFMA accumulator is already in physical
    # units (Q is bf16, K is bf16 = fp8 * k_scale). qk_scale therefore
    # stays the same as the default bf16 path -- just softmax_scale /
    # ln(2) -- and we do NOT fold k_scale into qk_scale here.
    pv_fp8_scale = b.fdiv(v_scale_p, b.const_f32(240.0)) if FP8_MFMA_PV else None
    sw_const = b.const_i32(int(SLIDING_WINDOW))
    z8 = b.zero_vec(dtype, 8)

    # ---------------- Q -> LDS (cooperative vec8 chunks) ----------------
    # General distribution:
    #   total vec8 chunks = BLOCK_M * HD / 8
    #   each wave64 lane handles (BLOCK_M * HD / 8) / 64 chunks.
    # This gives 4 chunks/thread for HD=128 and 8 chunks/thread for HD=256.
    Q_VECS_PER_ROW = HD // 8
    Q_VECS_PER_THREAD = (BLOCK_M * Q_VECS_PER_ROW) // THREADS
    # Coordinate transform for Q (and the symmetric output buffer):
    # ``(token, head, dim)`` packed contiguously. The element-unit
    # descriptor is reused below for the output store too.
    q_desc = TensorDescriptor.naive(
        "Q",
        # The runtime extents (total_q, num_query_heads, head_size) are
        # only used by the validity predicate (which we don't request
        # here). Use generous compile-time bounds so the descriptor's
        # row-major stride product matches the kernel's layout
        # assumptions exactly.
        lengths=[1 << 30, NUM_QH, HD],
        coord_names=("token", "head", "dim"),
    )
    if not Q_DIRECT_GLOBAL:
        for li in range(Q_VECS_PER_THREAD):
            q_vid = b.add(b.mul(b.const_i32(li), b.const_i32(THREADS)), tid)
            Q_row = b.div(q_vid, b.const_i32(Q_VECS_PER_ROW))
            Q_col = b.mul(b.mod(q_vid, b.const_i32(Q_VECS_PER_ROW)), b.const_i32(8))
            q_pos_t = b.add(qb_start_pos, b.div(Q_row, b.const_i32(NQK)))
            qh_t = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(Q_row, b.const_i32(NQK))
            )
            qmask_t = b.land(
                b.cmp_lt(q_pos_t, cur_batch_q_len), b.cmp_lt(qh_t, b.const_i32(NUM_QH))
            )
            q_pos_safe = b.select(qmask_t, q_pos_t, b.const_i32(0))
            qh_safe = b.select(qmask_t, qh_t, b.const_i32(0))
            q_off_base, _ = q_desc.offset(
                b,
                token=b.add(cu_q_start, q_pos_safe),
                head=qh_safe,
                dim=b.const_i32(0),
            )
            v8 = b.global_load_vN(query, b.add(q_off_base, Q_col), dtype, 8, align=16)
            # When Q is aliased to K_lds (single shared scratch), the store
            # goes through ``[buf, row_in_buf, col]`` indexing of the K_lds
            # 3D buffer. ``buf = Q_row // T`` selects which K_lds slot,
            # ``row_in_buf = Q_row % T`` is the row within that slot.
            # For single-slot (BLOCK_M <= T), buf is statically 0 and
            # row_in_buf == Q_row, so the div/mod fold to constants.
            if Q_ALIAS_K:
                if Q_USES_DUAL_SLOT:
                    q_buf = b.div(Q_row, b.const_i32(T))
                    q_row_in_buf = b.mod(Q_row, b.const_i32(T))
                else:
                    q_buf = b.const_i32(0)
                    q_row_in_buf = Q_row
                q_store_idx = [q_buf, q_row_in_buf, Q_col]
            else:
                q_store_idx = [Q_row, Q_col]
            b.smem_store_vN(
                Q_lds,
                q_store_idx,
                b.vector_select(b.vector_splat(qmask_t, 8), v8, z8),
                8,
            )
        b.sync()
    # The per-lane Q → VGPR gather is deferred until after ``lane_rg`` /
    # ``lane_col`` / ``wave_row_base`` are materialized below; see
    # the ``Q_reg`` block right after the softmax-state allocation.

    # ---------------- KV tile loop bounds ----------------
    bm1_div_nqk = (BLOCK_M - 1) // NQK
    msp_raw = b.add(b.add(context_len, qb_start_pos), b.const_i32(bm1_div_nqk + 1))
    max_seq_prefix_len = b.select(b.cmp_lt(msp_raw, seq_len), msp_raw, seq_len)
    num_tiles = b.div(b.add(max_seq_prefix_len, b.const_i32(T - 1)), b.const_i32(T))

    if SLIDING_WINDOW > 0:
        qpos_hi_raw = b.add(qb_start_pos, b.const_i32(bm1_div_nqk))
        cur_q_minus1 = b.sub(cur_batch_q_len, b.const_i32(1))
        qpos_hi = b.select(
            b.cmp_lt(qpos_hi_raw, cur_q_minus1), qpos_hi_raw, cur_q_minus1
        )
        first_allowed_key = b.add(
            b.sub(b.add(context_len, qb_start_pos), sw_const), b.const_i32(1)
        )
        last_allowed_key = b.add(context_len, qpos_hi)
        tile_start_raw = b.div(first_allowed_key, b.const_i32(T))
        tile_start = b.select(
            b.cmp_lt(tile_start_raw, b.const_i32(0)), b.const_i32(0), tile_start_raw
        )
        tile_end_raw = b.add(b.div(last_allowed_key, b.const_i32(T)), b.const_i32(1))
        tile_end = b.select(b.cmp_lt(tile_end_raw, num_tiles), tile_end_raw, num_tiles)
    else:
        tile_start = b.const_i32(0)
        tile_end = num_tiles

    # ---------------- online softmax registers ----------------
    # Each lane owns 4 row slots within its warp's BLOCK_M_PER_WARP=16 rows
    # (rows = wave_row_base + (lane/16)*4 + r for r in 0..3) when viewed
    # through the MFMA acc distribution. We keep `(m, l)` per row slot and
    # the 8 PV N-tile accumulators in iter_args of the KV loop. The MFMA
    # distribution is a per-warp construct, so the indexing uses `lane`
    # (== tid%64), not `tid`.
    lane_rg = b.div(lane, b.const_i32(16))
    lane_col = b.mod(lane, b.const_i32(16))
    lane_half32 = b.div(lane, b.const_i32(32))
    lane_col32 = b.mod(lane, b.const_i32(32))
    lane_col_div4 = b.div(lane_col, b.const_i32(4))
    lane_col_mod4 = b.mod(lane_col, b.const_i32(4))
    lane_rg_is0 = b.cmp_eq(lane_rg, b.const_i32(0))
    lane_rg_is1 = b.cmp_eq(lane_rg, b.const_i32(1))
    lane_rg_is2 = b.cmp_eq(lane_rg, b.const_i32(2))

    # ---- Per-lane row map ----
    # For ``block_m_per_warp=16`` the lane owns 4 row slots within one
    # 16-row atom (rows ``lane_rg*4 + r`` for ``r in 0..3``). For
    # ``block_m_per_warp=32`` the lane owns 8 row slots across 2
    # stacked atoms: reg ``r`` maps to ``(atom_idx, in_atom) =
    # (r // 4, r % 4)`` and the in-warp row is
    # ``atom_idx * 16 + lane_rg * 4 + in_atom``.
    def _in_warp_row(r: int) -> Value:
        atom_idx = r // 4
        in_atom = r % 4
        return b.add(
            b.mul(lane_rg, b.const_i32(4)),
            b.const_i32(atom_idx * 16 + in_atom),
        )

    def _state_row(r: int) -> Value:
        """Row represented by softmax state slot ``r`` in the active layout."""
        if USE_MFMA_32X32 and not TRANSPOSED_QK_32X32:
            return _mfma_32x32_c_row(b, lane, r)
        return _in_warp_row(r)

    def _bit2(v: Value, bit: int) -> Value:
        return b.land(b.lshr(v, b.const_i32(bit)), b.const_i32(1))

    def _select_lane_rg(v0: Value, v1: Value, v2: Value, v3: Value) -> Value:
        return b.select(
            lane_rg_is0, v0, b.select(lane_rg_is1, v1, b.select(lane_rg_is2, v2, v3))
        )

    def _permute_p_c_to_a16(p_regs_f32: list[Value]) -> list[Value]:
        """Convert one 16-col P tile from 16x16 MFMA-C regs to PV-A regs.

        Current production geometry (not the 32x32 migration) has QK C
        indexed by lane fields ``(A=lane/16, B=(lane%16)/4, C=lane%4)``
        and per-lane register ``R``:

            P[row=A*4+R, col=B*4+C]

        PV's A operand wants:

            P[row=B*4+C, col=A*4+R]

        so the transform is ``(A,B,C,R) -> (B,A,R,C)``. This is the same
        register-P idea CK Tile uses to avoid an LDS bridge between two
        GEMMs. Two bit-level transposes swap ``C`` with ``R`` inside a
        lane quad; then lane-field swaps exchange ``A`` and ``B``.
        """

        vals = p_regs_f32
        for bit in (0, 1):
            lane_bit = _bit2(lane_col_mod4, bit)
            reg_bit = 1 << bit
            old = vals
            vals = []
            for reg in range(4):
                partner = b.warp_shuffle_xor(old[reg ^ reg_bit], reg_bit)
                same_bit = b.cmp_eq(lane_bit, b.const_i32((reg >> bit) & 1))
                vals.append(b.select(same_bit, old[reg], partner))

        for bit, lane_xor in ((0, 20), (1, 40)):
            a_bit = _bit2(lane_rg, bit)
            b_bit = _bit2(lane_col_div4, bit)
            swap = b.cmp_ne(a_bit, b_bit)
            vals = [b.select(swap, b.warp_shuffle_xor(v, lane_xor), v) for v in vals]
        return vals

    def _pack_p_a16(p_regs_f32: list[Value]) -> Value:
        return b.vec_pack(
            [b.cast_f32_to(v, dtype) for v in _permute_p_c_to_a16(p_regs_f32)], dtype
        )

    def _pack_p_a32(p_regs0_f32: list[Value], p_regs1_f32: list[Value]) -> Value:
        vals0 = _permute_p_c_to_a16(p_regs0_f32)
        vals1 = _permute_p_c_to_a16(p_regs1_f32)
        lo: list[Value] = []
        hi: list[Value] = []
        for j in range(4):
            v0 = vals0[j]
            v1 = vals1[j]
            v0_x16 = b.warp_shuffle_xor(v0, 16)
            v0_x32 = b.warp_shuffle_xor(v0, 32)
            v0_x48 = b.warp_shuffle_xor(v0, 48)
            v1_x16 = b.warp_shuffle_xor(v1, 16)
            v1_x32 = b.warp_shuffle_xor(v1, 32)
            v1_x48 = b.warp_shuffle_xor(v1, 48)
            lo.append(_select_lane_rg(v0, v0_x48, v1_x32, v1_x16))
            hi.append(_select_lane_rg(v0_x16, v0_x32, v1_x48, v1))
        return b.vec_pack([b.cast_f32_to(v, dtype) for v in (lo + hi)], dtype)

    if USE_SINKS:
        m_inits = []
        if USE_MFMA_32X32 and TRANSPOSED_QK_32X32 and TRANSPOSED_SCALAR_STATE:
            row = b.add(wave_row_base, lane_col32)
            qh = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
            )
            qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
            sink_h = b.global_load(sinks, qh, dtype, align=2)
            sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
            m_inits.append(b.select(qh_in, sink_f, neg_inf))
        else:
            for r in range(REGS_PER_LANE):
                row = b.add(wave_row_base, _state_row(r))
                qh = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK))
                )
                qh_in = b.cmp_lt(qh, b.const_i32(NUM_QH))
                sink_h = b.global_load(sinks, qh, dtype, align=2)
                sink_f = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
                m_inits.append(b.select(qh_in, sink_f, neg_inf))
    else:
        m_inits = [neg_inf for _ in range(SOFTMAX_STATE_SLOTS)]
    l_inits = [one_f for _ in range(SOFTMAX_STATE_SLOTS)]

    # Acc storage. Old path: one vec_f32(4) per (N-tile, M-atom).
    # 32x32 path: one vec_f32(16) per 32-column output N-tile; each warp
    # owns exactly one M=32 atom, matching CK Tile's M32N32K16 C layout.
    PV32_N_TILES = HD // 32
    ACC_N_TILES = PV32_N_TILES if USE_MFMA_32X32 else PV_N_TILES
    ACC_M_ATOMS = 1 if USE_MFMA_32X32 else M_ATOMS_PER_WARP
    acc_zero = b.zero_vec_f32(16) if USE_MFMA_32X32 else b.zero_vec_f32(4)
    acc_inits = [acc_zero for _ in range(ACC_N_TILES * ACC_M_ATOMS)]

    def _acc_idx(n: int, atom: int) -> int:
        return n * ACC_M_ATOMS + atom

    iter_args = []
    for r in range(SOFTMAX_STATE_SLOTS):
        iter_args.append((f"m{r}", m_inits[r]))
        iter_args.append((f"l{r}", l_inits[r]))
    for n in range(ACC_N_TILES):
        for atom in range(ACC_M_ATOMS):
            iter_args.append(
                (
                    f"acc{n}a{atom}" if ACC_M_ATOMS > 1 else f"acc{n}",
                    acc_inits[_acc_idx(n, atom)],
                )
            )

    # ---- Pre-loop: build K/V buffer descriptors and pre-fetch tile 0.
    # The buffer rsrc bounds OOB voffsets to return zero. We size it large
    # so valid block offsets never trip the check.
    big_bytes = b.const_i32(0x7FFF0000)
    key_rsrc = b.buffer_rsrc(key, big_bytes)
    value_rsrc = b.buffer_rsrc(value, big_bytes)

    # Async load contract (bf16 K/V path): on gfx942 each lane writes
    # ``ASYNC_LDS_MAX_BYTES_PER_LANE`` (= 4 bytes = 2 halves) lane-contiguous
    # in LDS per call (CDNA3 caps load-to-LDS at 1 dword; see
    # ASYNC_LDS_MAX_DWORDS above). One call writes THREADS*2 halves = a
    # contiguous slice of the natural [T, HD] tile, so we issue 4x the calls
    # the gfx950 16-byte template did while preserving the lane-contiguous
    # [T, HD] LDS layout. Works for HD=128 and HD=256 unchanged.
    KV_HALVES_PER_LANE = ASYNC_LDS_MAX_BYTES_PER_LANE // 2  # bf16: 2 bytes/half
    KV_HALVES_PER_CALL = THREADS * KV_HALVES_PER_LANE
    assert (T * HD) % KV_HALVES_PER_CALL == 0
    kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL
    bytes_per_call = KV_HALVES_PER_CALL * 2
    # For fp8-MFMA mode K_lds is sized in fp8 (1 byte/element). On gfx942 the
    # async DMA is capped at ``ASYNC_LDS_MAX_BYTES_PER_LANE`` (4 bytes) per
    # lane (CDNA3 1-dword load-to-LDS limit; see ASYNC_LDS_MAX_DWORDS above),
    # so 4 fp8 elements per lane vs the 2 bf16 halves the bf16 path moves.
    K_FP8_MFMA = FP8_MFMA_QK
    PV_FP8_MFMA = FP8_MFMA_PV
    if K_FP8_MFMA or PV_FP8_MFMA:
        # Pick the largest dwords (and therefore widest per-lane payload)
        # the tile bytes support, clamped to the gfx942 load-to-LDS max.
        # The AMDGPU async-DMA intrinsic ``raw.ptr.buffer.load.lds`` accepts
        # dwords in {1, 3, 4}; on CDNA3 only 1 (4 bytes/lane) legalises for
        # the load-*to-LDS* form, so the candidate list is clamped here.
        tile_bytes = T * HD  # 1 byte per fp8 element
        fp8_dword_candidates = [
            (d, by)
            for (d, by) in [(4, 16), (3, 12), (1, 4)]
            if d <= ASYNC_LDS_MAX_DWORDS
        ]
        for dwords_try, bytes_per_lane in fp8_dword_candidates:
            payload = THREADS * bytes_per_lane
            if tile_bytes >= payload and tile_bytes % payload == 0:
                K_FP8_DWORDS = dwords_try
                K_BYTES_PER_LANE = bytes_per_lane
                break
        else:
            raise AssertionError(
                f"fp8-mfma K loader: T*HD={tile_bytes} cannot be covered by "
                f"any supported async-DMA payload (THREADS={THREADS}, "
                f"max dwords={ASYNC_LDS_MAX_DWORDS})"
            )
        K_ELEMS_PER_CALL = THREADS * K_BYTES_PER_LANE
        K_BYTES_PER_CALL = K_ELEMS_PER_CALL  # 1 byte per fp8 element
        k_fp8_calls_per_tile = tile_bytes // K_ELEMS_PER_CALL
    else:
        K_FP8_DWORDS = ASYNC_LDS_MAX_DWORDS
        K_BYTES_PER_LANE = ASYNC_LDS_MAX_BYTES_PER_LANE
        k_fp8_calls_per_tile = 0  # unused
    # Byte strides for the paged-KV cache. ``KV_BYTES`` is 2 for bf16, 1
    # for fp8e4m3. The async DMA reads bytes verbatim (no implicit cast),
    # so for the FP8 K/V path the loader switches to a sync per-thread
    # dequant chain below; the byte-stride math here is shared so the
    # paged_kv_desc compiles consistently in both branches.
    kv_stride_blk_b = BS * NUM_KV * HD * KV_BYTES
    kv_stride_tok_b = NUM_KV * HD * KV_BYTES
    kv_stride_h_b = HD * KV_BYTES
    kv_block_bytes_c = b.const_i32(kv_stride_blk_b)  # one-block buffer bound

    # Per-lane starting half index. With the gfx942 1-dword DMA each lane
    # contributes ``KV_HALVES_PER_LANE`` (= 2) contiguous halves per call,
    # so the lane base advances by that many halves (not the gfx950 8).
    lane_half_base = b.mul(tid, b.const_i32(KV_HALVES_PER_LANE))

    K_lds_addr = b.smem_addr_of(K_lds)
    V_lds_addr = b.smem_addr_of(V_lds)
    bytes_per_buf = T * HD * 2  # one [T, HD] *working-dtype* (bf16) slab

    zero_soff = b.const_i32(0)

    # Bytes one wave's lanes write per call. `raw.ptr.buffer.load.lds`
    # writes `dwords * 4` bytes per lane lane-contiguous starting at
    # the wave-uniform `lds_dst`. Each wave issues its own instruction
    # but they share the LDS pointer unless we add a wave offset; with
    # NUM_WARPS=1 this collapses to zero.
    # Per-wave LDS slab stride for one async call. Each wave's 64 lanes
    # deposit ``ASYNC_LDS_MAX_BYTES_PER_LANE`` bytes lane-contiguous, so the
    # wave offset advances by ``WAVE * ASYNC_LDS_MAX_BYTES_PER_LANE`` (= 256
    # bytes on gfx942). Pairing this with the ``bytes_per_call`` per-call
    # stride keeps the LDS deposit an exact identity copy of the linear
    # [T, HD] tile for every (wave, call, lane) — across all NUM_WARPS and
    # call counts — which the [T, HD] K_lds/V_lds readers rely on.
    WAVE_BYTES = WAVE * ASYNC_LDS_MAX_BYTES_PER_LANE
    if NUM_WARPS == 1:
        wave_lds_offset_i64 = b.const_i64(0)
    else:
        # ``wave_lds_offset_i32`` is wave-uniform (it derives from ``wave_id``,
        # which is constant across a wave's lanes). Pin it to SGPR via
        # ``to_sgpr_u32`` so the register allocator doesn't re-materialise
        # it as a per-lane VGPR each time the unrolled K/V-load loops
        # consume it (saves a ``v_readfirstlane_b32`` per use). See
        # ``dsl_docs/primitives/wave_and_cross_lane.md`` ("Wave-uniform
        # LDS base hoist" section).
        wave_lds_offset_i32 = b.to_sgpr_u32(b.mul(wave_id, b.const_i32(WAVE_BYTES)))
        wave_lds_offset_i64 = b.zext(wave_lds_offset_i32, I64)

    # V's DMA dest base when the bank-deconflict swizzle is on: each wave owns
    # one padded V row per call, so the wave's base advances by the padded row
    # stride ``WAVE_BYTES + V_LDS_PAD*2`` (bytes) instead of ``WAVE_BYTES``.
    # Wave-uniform -> pin to SGPR (same rationale as the K/V wave offset above).
    V_WAVE_BYTES = WAVE_BYTES + (V_LDS_PAD * 2 if SWIZZLE_VLDS else 0)
    # Per-call dest stride for V (bytes). With the swizzle each call writes
    # ``V_ROWS_PER_CALL`` padded rows, so the call base steps by the padded
    # group stride; without it, the natural contiguous ``bytes_per_call``.
    V_BYTES_PER_CALL_SWZ = bytes_per_call + (
        V_ROWS_PER_CALL * V_LDS_PAD * 2 if SWIZZLE_VLDS else 0
    )
    if not SWIZZLE_VLDS or NUM_WARPS == 1:
        v_wave_lds_offset_i64 = (
            b.const_i64(0) if NUM_WARPS == 1 else wave_lds_offset_i64
        )
    else:
        v_wave_lds_offset_i32 = b.to_sgpr_u32(b.mul(wave_id, b.const_i32(V_WAVE_BYTES)))
        v_wave_lds_offset_i64 = b.zext(v_wave_lds_offset_i32, I64)

    # ---- Paged KV byte descriptor (full transform DAG) ----
    # The paged-KV cache is laid out ``[num_blocks, BS, NUM_KV, HD]`` with
    # *byte* strides. The kernel addresses it via a chain of coordinate
    # transforms.
    #
    # **Single-block tile** (``N_BLOCKS_PER_TILE == 1``, ``T == BS``):
    #
    #   1. ``indirect(tile_idx -> physical_block)`` does the
    #      ``physical_block = block_tables[seq_idx*bt_stride + tile_idx]``
    #      table lookup.
    #   2. ``unmerge(linear_half -> (token, dim))`` splits the
    #      cooperative ``THREADS*8`` half count into ``(token_in_tile,
    #      head_dim)`` (token range ``[0, BS)`` here).
    #
    # **Multi-block tile** (``N_BLOCKS_PER_TILE > 1``, ``T == N_B*BS``,
    # used for prefill workloads to match Triton's ``TILE_SIZE=64`` while
    # the paged cache still has ``BS=32``):
    #
    #   1. ``unmerge(linear_half -> (block_within_tile, token, dim))``
    #      where ``block_within_tile in [0, N_B)`` and ``token in [0, BS)``.
    #   2. ``embed((tile_idx, block_within_tile) -> linear_block_idx)``
    #      with strides ``(N_B, 1)`` so
    #      ``linear_block_idx = tile_idx * N_B + block_within_tile``.
    #   3. ``indirect(linear_block_idx -> physical_block)`` looks up
    #      ``block_tables[seq_base + linear_block_idx]`` once per
    #      sub-block (per-wave-uniform when ``per_call_tokens <= BS``;
    #      we enforce this in ``supports_tiled_2d``).
    #
    # In both cases the naive base ``(physical_block, token, kv_head, dim)``
    # with byte strides ``(BS*NUM_KV*HD, NUM_KV*HD, HD, 1) * KV_BYTES``
    # produces the final byte offset. Calling
    # ``paged_kv_desc.offset(b, tile_idx=, linear_half=, kv_head=)``
    # transparently picks up the multi-block lookup; loaders are unchanged.
    # ``seq_base`` indexes the per-sequence offset into the global
    # block_tables; it's CTA-wide-uniform (depends only on ``seq_idx`` and
    # ``bt_stride_p``, both CTA constants). Pin to SGPR so the per-iter
    # ``indirect()`` table lookup inside the paged-KV descriptor doesn't
    # re-materialise the base into a VGPR.
    seq_base = b.to_sgpr_u32(b.mul(seq_idx, bt_stride_p))
    # Block-table bounds. The paged-KV descriptor over-fetches
    # ``N_BLOCKS_PER_TILE`` entries per tile from ``block_tables`` (one
    # ``tile_idx`` worth, regardless of how many of those entries the
    # current seq actually owns). For short ``kv_len[i]`` (e.g.
    # decode with ``kv_len = 1`` and ``T = 64``) the over-fetched
    # indices land past the end of ``block_tables`` -- the values read
    # are uninitialised memory that, with bad luck, are >= num_blocks
    # and route the downstream ``buffer_load`` to an unmapped page,
    # crashing with "Memory access fault by GPU". GUARD the indirect
    # lookup with the total block-table footprint so OOB lookups
    # return ``0`` (a guaranteed-valid block id). Downstream causal /
    # in_prefix masks already discard the bogus tokens.
    block_table_max_idx = b.to_sgpr_u32(b.mul(num_seqs_p, bt_stride_p))
    if FAST_PAGED_KV_DESC:
        # Two logical blocks per tile (T=64 = 2 * BS=32). On gfx942 the
        # 1-dword async DMA needs ``FAST_CALLS_PER_BLOCK`` calls to drain one
        # block instead of the single 16-byte call the gfx950 template used,
        # so a tile now takes ``2 * FAST_CALLS_PER_BLOCK`` calls. Each block
        # is BS*HD halves; ``KV_HALVES_PER_CALL`` halves are moved per call.
        assert (BS, T, HD, NUM_KV, KV_BYTES, NUM_WARPS) == (32, 64, 64, 8, 2, 4)
        assert (BS * HD) % KV_HALVES_PER_CALL == 0
        FAST_CALLS_PER_BLOCK = (BS * HD) // KV_HALVES_PER_CALL
        assert kv_calls_per_tile == 2 * FAST_CALLS_PER_BLOCK

        def _fast_paged_kv_blocks(kv_tile_idx: Value) -> tuple[Value, Value]:
            logical_block0 = b.mul(kv_tile_idx, b.const_i32(2))
            logical_block1 = b.add(logical_block0, b.const_i32(1))
            idx0 = b.add(seq_base, logical_block0)
            idx1 = b.add(seq_base, logical_block1)
            block0 = b.masked_global_load(
                block_tables,
                idx0,
                b.cmp_lt(idx0, block_table_max_idx),
                b.const_i32(0),
                dtype=I32,
                align=4,
            )
            block1 = b.masked_global_load(
                block_tables,
                idx1,
                b.cmp_lt(idx1, block_table_max_idx),
                b.const_i32(0),
                dtype=I32,
                align=4,
            )
            return b.to_sgpr_u32(block0), b.to_sgpr_u32(block1)

        def _fast_paged_kv_voff(call: int, block0: Value, block1: Value):
            # ``FAST_CALLS_PER_BLOCK`` consecutive calls drain one logical
            # block; calls ``[0, CPB)`` -> block0, ``[CPB, 2*CPB)`` -> block1.
            # ``physical`` is uniform per call. Within the block the half
            # index is ``(call_in_block)*KV_HALVES_PER_CALL + lane_half_base``.
            # Returns (i64 block base byte offset, within-block i32 voffset)
            # when I64_KV_ADDR (cache may exceed the 2 GiB i32-voffset cap),
            # else the legacy single i32 voffset (fast path, base = key).
            physical = block0 if call < FAST_CALLS_PER_BLOCK else block1
            call_in_block = call % FAST_CALLS_PER_BLOCK
            half_in_block = b.add(
                b.const_i32(call_in_block * KV_HALVES_PER_CALL), lane_half_base
            )
            token = b.lshr(half_in_block, b.const_i32(6))  # half_in_block / HD
            dim = b.land(half_in_block, b.const_i32(63))  # half_in_block % HD
            token_b = b.shl(token, b.const_i32(10))  # 8 * 64 * 2
            head_b = b.shl(kv_head_idx, b.const_i32(7))  # 64 * 2
            dim_b = b.shl(dim, b.const_i32(1))
            within = b.add(b.add(token_b, head_b), dim_b)  # < 32768
            if I64_KV_ADDR:
                base_i64 = b.shl(b.zext(physical, I64), b.const_i64(15))
                return base_i64, within
            block_b = b.shl(physical, b.const_i32(15))  # 32 * 8 * 64 * 2 (i32)
            return None, b.add(block_b, within)

    else:
        _kv_base = TensorDescriptor.naive(
            "paged_kv_bytes",
            # ``lengths`` here is just informational (validity propagation
            # is driven by the transforms above, not by these bounds).
            lengths=[1 << 24, BS, NUM_KV, HD],
            strides=[kv_stride_blk_b, kv_stride_tok_b, kv_stride_h_b, KV_BYTES],
            coord_names=("physical_block", "token", "kv_head", "dim"),
        )
        if N_BLOCKS_PER_TILE == 1:
            paged_kv_desc = _kv_base.transform(
                indirect(
                    "tile_idx",
                    into="physical_block",
                    table=block_tables,
                    base=seq_base,
                    max_idx=block_table_max_idx,
                ),
                unmerge("linear_half", into=("token", "dim"), dims=(T, HD)),
            )
        else:
            paged_kv_desc = _kv_base.transform(
                unmerge(
                    "linear_half",
                    into=("block_within_tile", "token", "dim"),
                    dims=(N_BLOCKS_PER_TILE, BS, HD),
                ),
                embed(
                    ("tile_idx", "block_within_tile"),
                    into="linear_block_idx",
                    strides=(N_BLOCKS_PER_TILE, 1),
                ),
                indirect(
                    "linear_block_idx",
                    into="physical_block",
                    table=block_tables,
                    base=seq_base,
                    max_idx=block_table_max_idx,
                ),
            )

    def _issue_k_load_runtime(kv_tile_idx: Value, buf_idx: Value) -> None:
        """Issue async K loads for one tile into K_lds[buf_idx].

        CK's QRKSVSAsync pipeline deliberately makes K the early-prefetch
        stream: QK can start as soon as K is visible, while V is still not
        needed until after softmax. Keeping K and V as independent streams
        avoids waiting on V before QK.

        Multi-warp: each wave's `raw.ptr.buffer.load.lds` writes a
        lane-contiguous 1 KiB slab starting at `lds_dst`. To keep the
        waves from stomping on each other we offset `lds_dst` by
        `wave_id * WAVE_BYTES`; combined with each wave's natural voff
        offset (lanes 64..127 have `tid*8 / HD` advanced by T/NUM_WARPS),
        the cooperative load fills the full `[T, HD]` slab correctly.
        """
        buf_off_i32 = b.mul(buf_idx, b.const_i32(bytes_per_buf))
        buf_off_i64 = b.zext(buf_off_i32, I64)
        K_buf_base = b.smem_ptr_add(K_lds_addr, buf_off_i64)
        K_wave_base = b.smem_ptr_add(K_buf_base, wave_lds_offset_i64)
        if FAST_PAGED_KV_DESC:
            fast_block0, fast_block1 = _fast_paged_kv_blocks(kv_tile_idx)
        for call in range(kv_calls_per_tile):
            k_rsrc = key_rsrc
            k_ptr = key
            if FAST_PAGED_KV_DESC:
                base_i64, voff = _fast_paged_kv_voff(call, fast_block0, fast_block1)
                if base_i64 is not None:  # I64_KV_ADDR: per-block 64-bit base
                    k_ptr = b.global_ptr_add(key, base_i64)
                    k_rsrc = b.buffer_rsrc(k_ptr, kv_block_bytes_c)
            else:
                linear_half = b.add(
                    b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base
                )
                if I64_KV_ADDR:
                    base_i64, voff, _ = paged_kv_desc.offset_i64_split(
                        b,
                        "physical_block",
                        tile_idx=kv_tile_idx,
                        linear_half=linear_half,
                        kv_head=kv_head_idx,
                    )
                    k_ptr = b.global_ptr_add(key, base_i64)
                    k_rsrc = b.buffer_rsrc(k_ptr, kv_block_bytes_c)
                else:
                    voff, _ = paged_kv_desc.offset(
                        b,
                        tile_idx=kv_tile_idx,
                        linear_half=linear_half,
                        kv_head=kv_head_idx,
                    )
            k_dst = b.smem_ptr_add(K_wave_base, b.const_i64(call * bytes_per_call))
            if USE_GLOBAL_LOAD_LDS_K:
                b.global_load_lds(
                    k_ptr,
                    voff,
                    k_dst,
                    ASYNC_LDS_MAX_BYTES_PER_LANE,
                    coherency=kv_cache_aux,
                )
            else:
                b.async_buffer_load_lds_addr(
                    k_rsrc,
                    k_dst,
                    voff,
                    zero_soff,
                    ASYNC_LDS_MAX_DWORDS,
                    coherency=kv_cache_aux,
                )

    K_SLICE_CALLS_PER_TILE = (T * K_SLICE_HD) // KV_HALVES_PER_CALL

    def _issue_k_slice_load_runtime(
        kv_tile_idx: Value, slice_idx: int, slot_idx: int
    ) -> None:
        """Issue one 32-dim K slice into the sliced K ring."""
        slot_off_i64 = b.const_i64(slot_idx * K_BUF_BYTES)
        K_slot_base = b.smem_ptr_add(K_lds_addr, slot_off_i64)
        K_wave_base = b.smem_ptr_add(K_slot_base, wave_lds_offset_i64)
        for call in range(K_SLICE_CALLS_PER_TILE):
            linear_local = b.add(b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base)
            token = b.div(linear_local, b.const_i32(K_SLICE_HD))
            dim_local = b.mod(linear_local, b.const_i32(K_SLICE_HD))
            dim = b.add(b.const_i32(slice_idx * K_SLICE_HD), dim_local)
            linear_half = b.add(b.mul(token, b.const_i32(HD)), dim)
            k_rsrc = key_rsrc
            k_ptr = key
            if I64_KV_ADDR:
                base_i64, voff, _ = paged_kv_desc.offset_i64_split(
                    b,
                    "physical_block",
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half,
                    kv_head=kv_head_idx,
                )
                k_ptr = b.global_ptr_add(key, base_i64)
                k_rsrc = b.buffer_rsrc(k_ptr, kv_block_bytes_c)
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half,
                    kv_head=kv_head_idx,
                )
            k_dst = b.smem_ptr_add(K_wave_base, b.const_i64(call * bytes_per_call))
            if USE_GLOBAL_LOAD_LDS_K:
                b.global_load_lds(
                    k_ptr,
                    voff,
                    k_dst,
                    ASYNC_LDS_MAX_BYTES_PER_LANE,
                    coherency=kv_cache_aux,
                )
            else:
                b.async_buffer_load_lds_addr(
                    k_rsrc,
                    k_dst,
                    voff,
                    zero_soff,
                    ASYNC_LDS_MAX_DWORDS,
                    coherency=kv_cache_aux,
                )

    def _issue_v_load_runtime(kv_tile_idx: Value, buf_idx: Value) -> None:
        """Issue async V loads for one tile into V_lds[0] (single-buffered).

        ``buf_idx`` is ignored -- V is single-buffered (safe because PV[i]
        reads retire before V[i+1] is issued in iter i+1, after the
        iter-start full drain). Saves 8 KiB LDS per CTA vs the original
        double-buffer V layout.
        """
        # V is single-buffered; ignore buf_idx, always write slot 0. The wave
        # base uses the V-specific (swizzle-aware) wave offset so each wave's
        # padded V row lands in its bank-rotated slot when SWIZZLE_VLDS is on.
        V_wave_base = b.smem_ptr_add(V_lds_addr, v_wave_lds_offset_i64)
        if FAST_PAGED_KV_DESC:
            fast_block0, fast_block1 = _fast_paged_kv_blocks(kv_tile_idx)
        for call in range(kv_calls_per_tile):
            v_rsrc = value_rsrc
            if FAST_PAGED_KV_DESC:
                base_i64, voff = _fast_paged_kv_voff(call, fast_block0, fast_block1)
                if base_i64 is not None:
                    v_rsrc = b.buffer_rsrc(
                        b.global_ptr_add(value, base_i64), kv_block_bytes_c
                    )
            else:
                linear_half = b.add(
                    b.const_i32(call * KV_HALVES_PER_CALL), lane_half_base
                )
                if I64_KV_ADDR:
                    base_i64, voff, _ = paged_kv_desc.offset_i64_split(
                        b,
                        "physical_block",
                        tile_idx=kv_tile_idx,
                        linear_half=linear_half,
                        kv_head=kv_head_idx,
                    )
                    v_rsrc = b.buffer_rsrc(
                        b.global_ptr_add(value, base_i64), kv_block_bytes_c
                    )
                else:
                    voff, _ = paged_kv_desc.offset(
                        b,
                        tile_idx=kv_tile_idx,
                        linear_half=linear_half,
                        kv_head=kv_head_idx,
                    )
            v_dst = b.smem_ptr_add(
                V_wave_base, b.const_i64(call * V_BYTES_PER_CALL_SWZ)
            )
            # CACHE_STREAM (SLC): V is consumed once per iter and never
            # re-read within this kernel; see _issue_k_load_runtime for
            # the rationale.
            b.async_buffer_load_lds_addr(
                v_rsrc,
                v_dst,
                voff,
                zero_soff,
                ASYNC_LDS_MAX_DWORDS,
                coherency=kv_cache_aux,
            )

    # ---- Stream B (DECISIVE pivot): transposed-V VECTOR-store loader ----
    # The transposed ``V_lds[HD, T+V_T_PAD]`` (head-dim outer, token inner) is the
    # conflict-free PV feed (the PV A-operand read is then ONE wide contiguous
    # ``ds_read_b64`` over consecutive tokens). gfx942 async DMA cannot transpose
    # HBM->LDS, so the transpose is paid on the LDS store side.
    #
    # PROVEN DEAD VEHICLE (do NOT restore): the prior fill loaded 8 contiguous
    # head-dims ``V[token, col..col+7]`` and SCATTERED them to 8 strided rows
    # ``V_lds[col+i, token]`` with 8 scalar ``smem_store_vN(..., n=1)`` =
    # sub-dword ``ds_write_b16``. That corrupts head-dims >=8 (12/5/9; d>=8
    # garbage/inf, non-deterministic) on the transposed-x8 path -- two independent
    # implementations failed identically. The ds_write_b16 scatter is the hazard.
    #
    # THE VECTOR-STORE FILL (this implementation): invert the per-thread tiling so
    # the LDS STORE is contiguous. Each thread owns one head-dim row ``col`` and a
    # group of ``V_T_VEC`` CONSECUTIVE tokens ``token_base..token_base+V_T_VEC-1``;
    # it reads those tokens' value at head-dim ``col`` from HBM (token-strided
    # global loads -- gathers in HBM are fine, no LDS hazard), packs them into one
    # ``<V_T_VEC x half>`` and writes them with ONE vector ``smem_store_vN(V_lds,
    # [0, col, token_base], vec, n=V_T_VEC)``. Because token is the INNER/fast
    # axis of ``V_lds[HD, T+pad]``, those V_T_VEC slots are CONTIGUOUS in LDS ->
    # the store lowers to a single ``ds_write_b{32,64,128}`` (NO sub-dword
    # ds_write_b16). This is the brief's mandated pivot: vector ds_write, not
    # scatter. ``paged_kv_desc.offset`` returns a BYTE offset because the async
    # DMA path consumes it directly; sync ``global_load`` consumes element
    # indices, so sync V fills must divide by ``KV_BYTES`` before loading.
    V_T_VEC = 4 if (T % 4 == 0) else (2 if (T % 2 == 0) else 1)
    if TRANSPOSED_V:
        # token-groups along the inner axis: V_T_VEC consecutive tokens per group.
        _v_t_token_groups = T // V_T_VEC
        # Total (col, token-group) work items spread across the workgroup.
        _v_t_total_items = HD * _v_t_token_groups
        # Items each thread fills (loop trip count). May not divide evenly when
        # THREADS > items; guard the tail with item < total below.
        V_T_ITEMS_PER_THREAD = (_v_t_total_items + THREADS - 1) // THREADS
        _v_t_need_item_guard = (_v_t_total_items % THREADS) != 0

    def _issue_v_transposed(kv_tile_idx: Value) -> None:
        """VECTOR-store transpose of V into ``V_lds[0, col, token]`` (f16).

        Per work item: ``item = call*THREADS + tid`` maps to head-dim row
        ``col = item // token_groups`` and token group ``g = item % token_groups``
        (token_base = g*V_T_VEC). Reads V_T_VEC consecutive tokens at head-dim
        ``col`` from HBM and writes them as ONE contiguous vector ds_write.
        """
        _tg = b.const_i32(_v_t_token_groups)
        # Zero of the working dtype for masked (out-of-range) V tokens.
        zero_h = b.cast_f32_to(b.const_f32(0.0), dtype)
        for call in range(V_T_ITEMS_PER_THREAD):
            item = b.add(b.mul(b.const_i32(call), b.const_i32(THREADS)), tid)
            if _v_t_need_item_guard:
                # Tail items past the work set must not store (would scatter OOB
                # of V_lds). Clamp to item 0 (a redundant valid write).
                item = b.select(
                    b.cmp_lt(item, b.const_i32(_v_t_total_items)),
                    item,
                    b.const_i32(0),
                )
            col = b.div(item, _tg)
            g = b.mod(item, _tg)
            token_base = b.mul(g, b.const_i32(V_T_VEC))
            # Gather V_T_VEC consecutive tokens at head-dim ``col`` from HBM.
            # HBM is [token, dim] row-major, so token t at dim col lives at
            # linear (token_base + j) * HD + col -- stride HD between the j's.
            # THE FIX (root cause of the L3 12/9/5 garbage + perf MEMORY FAULT):
            # tokens past the valid KV length of the LAST (partial) tile are NOT
            # real V. The naive async feed reads them through a BOUNDED
            # ``buffer_rsrc`` (size ``kv_block_bytes_c``) so out-of-range lanes
            # HARDWARE-mask to 0; this manual scalar gather had NO such bound, so
            # the invalid V_lds slots held garbage (often inf/nan from
            # unmapped/stale memory). Softmax correctly sets P=0 for those masked
            # tokens, BUT the PV MFMA then computes ``P(0) * V(inf) = nan`` -> the
            # whole O^T row goes nan/inf (observed: first mismatch at index 1,
            # worst diff inf, ~70% wrong), and on large multi-tile shapes the
            # unbounded load FAULTS (faulting addr ...c000 in the _cfv kernel).
            # The descriptor carries NO validity predicate for this geometry
            # (``valid_j`` is None), so reproduce the buffer-size bound EXPLICITLY:
            # the tile-global KV position is ``kv_tile_idx*T + token_j``; it is a
            # real token iff it is < ``max_seq_prefix_len`` (this CTA's valid KV
            # length). Out-of-range -> clamp the offset to 0 (no fault) AND select
            # 0 for the value (no nan) -- exactly the zeroing the bounded async
            # DMA gave us for free.
            tile_tok_base = b.mul(kv_tile_idx, b.const_i32(T))
            elems = []
            for j in range(V_T_VEC):
                token_j = b.add(token_base, b.const_i32(j))
                linear_j = b.add(b.mul(token_j, b.const_i32(HD)), col)
                voff_j, valid_j = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_j,
                    kv_head=kv_head_idx,
                )
                in_range = b.cmp_lt(b.add(tile_tok_base, token_j), max_seq_prefix_len)
                if valid_j is not None:
                    in_range = b.land(in_range, valid_j)
                safe_voff_j = b.select(in_range, voff_j, b.const_i32(0))
                safe_elem_j = (
                    b.div(safe_voff_j, b.const_i32(KV_BYTES))
                    if KV_BYTES != 1
                    else safe_voff_j
                )
                v1 = b.global_load(value, safe_elem_j, dtype, align=2)
                v1 = b.select(in_range, v1, zero_h)
                elems.append(v1)
            v_vec = b.vec_pack(elems, dtype)
            # ONE contiguous vector store: token is the inner axis, so
            # [0, col, token_base] + n=V_T_VEC writes V_T_VEC adjacent LDS slots
            # -> a single ds_write_b{32,64,128} (no sub-dword scatter).
            _v_t_store(col, token_base, v_vec, V_T_VEC)

    # ---- Stream 2 (GRIND): conflict-free V STORE-PATH vehicle (c) ----
    # gfx942 playbook's only conflict-free transposed-operand vehicle. The
    # vehicle (a) ``_issue_v_transposed`` above is correct but ~5x slower: its
    # HBM read is a SYNCHRONOUS token-strided ``global_load`` gather (V_T_VEC
    # scalar loads at stride HD per item) that stalls on HBM (rocprof: ~30 TF).
    # The LDS *store* and the LDS *read* were already conflict-free; only the
    # HBM access pattern was wrong.
    #
    # This loader makes the HBM read NATURAL and the transpose IN-REGISTER, the
    # CK ``transpose_vectors`` / flash way:
    #   * Partition the (token, dim) tile into 2x2 f16 blocks. Block (tg, dg)
    #     covers tokens {2*tg, 2*tg+1} x dims {2*dg, 2*dg+1}.
    #   * Read each of the 2 token rows as ONE 2-half vector load over a
    #     CONTIGUOUS dim pair: ``x0 = (V[t0,d0], V[t0,d1])``,
    #     ``x1 = (V[t1,d0], V[t1,d1])`` -- coalesced VMEM (adjacent lanes / the
    #     dim axis are contiguous in HBM), no token-stride gather.
    #   * Transpose the 2x2 with ``perm_b32`` (in-thread VALU ``v_perm_b32``, no
    #     cross-lane, no lgkmcnt). With src0=x0(t0), src1=x1(t1) the 8 concat
    #     bytes are [t1.d0, t1.d1, t0.d0, t0.d1]; the CK masks give:
    #       perm(x0,x1, 0x01000504) -> (V[t0,d0], V[t1,d0])  = dim-row d0
    #       perm(x0,x1, 0x03020706) -> (V[t0,d1], V[t1,d1])  = dim-row d1
    #     i.e. each output i32 holds 2 CONSECUTIVE tokens at a fixed dim.
    #   * Write each output as ONE contiguous 2-half ``ds_write_b32`` into
    #     ``V_lds[0, d, 2*tg]`` (token is the inner/fast axis) -- conflict-free.
    # The block loop is RUNTIME-rolled (``scf_for`` over block index, stride
    # THREADS) so the IR stays O(1) regardless of tile size (gfx942 playbook:
    # a fully-unrolled transpose hit a 45-min JIT IR-explosion).
    if TRANSPOSED_V_STORE:
        _v_t2_tok_pairs = T // 2
        _v_t2_dim_pairs = HD // 2
        _v_t2_total_blocks = _v_t2_tok_pairs * _v_t2_dim_pairs

    def _cfvst_load_v_regs(kv_tile_idx: Value):
        """Issue cfvst VMEM loads and keep each thread's V tile in VGPRs."""
        zero_h = b.cast_f32_to(b.const_f32(0.0), dtype)
        zero_i32 = b.const_i32(0)
        tile_tok_base = b.mul(kv_tile_idx, b.const_i32(T))
        _dp = b.const_i32(_v_t2_dim_pairs)

        def _load_token_row_pair(t_row: Value, d0: Value) -> Value:
            """Load <2 x f16> = (V[t_row, d0], V[t_row, d0+1]) as i32, bounded.

            Contiguous dim pair -> coalesced 2-half VMEM load. Out-of-range
            tokens (past this CTA's valid KV length) are masked to 0 (matching
            the bounded async DMA's hardware zero-fill) so masked P*V stays 0,
            never P(0)*V(inf)=nan.
            """
            linear = b.add(b.mul(t_row, b.const_i32(HD)), d0)
            voff, valid = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear,
                kv_head=kv_head_idx,
            )
            in_range = b.cmp_lt(b.add(tile_tok_base, t_row), max_seq_prefix_len)
            if valid is not None:
                in_range = b.land(in_range, valid)
            safe_voff = b.select(in_range, voff, zero_i32)
            safe_elem = (
                b.div(safe_voff, b.const_i32(KV_BYTES)) if KV_BYTES != 1 else safe_voff
            )
            if _CFV_STORE_SEPOFF:
                # ISOLATION: compute the (t_row, d0+1) offset via a SEPARATE
                # descriptor call instead of voff+1. Tests the dim-contiguity
                # assumption (does the paged descriptor have dim element-stride
                # 1?). Vehicle (a) always uses separate per-element offsets.
                linear1 = b.add(
                    b.mul(t_row, b.const_i32(HD)), b.add(d0, b.const_i32(1))
                )
                voff1, valid1 = paged_kv_desc.offset(
                    b, tile_idx=kv_tile_idx, linear_half=linear1, kv_head=kv_head_idx
                )
                safe_voff1 = b.select(in_range, voff1, zero_i32)
                safe_elem1 = (
                    b.div(safe_voff1, b.const_i32(KV_BYTES))
                    if KV_BYTES != 1
                    else safe_voff1
                )
                e0 = b.global_load(value, safe_elem, dtype, align=2)
                e1 = b.global_load(value, safe_elem1, dtype, align=2)
                e0 = b.select(in_range, e0, zero_h)
                e1 = b.select(in_range, e1, zero_h)
                v2 = b.vec_pack([e0, e1], dtype)
            elif _CFV_STORE_SCALAR_LOAD:
                # ISOLATION: read the 2 dims as 2 scalar n=1 loads + manual pack
                # (rules out the vectorized global_load_vN lowering). Same values.
                e0 = b.global_load(value, safe_elem, dtype, align=2)
                e1 = b.global_load(
                    value, b.add(safe_elem, b.const_i32(1)), dtype, align=2
                )
                e0 = b.select(in_range, e0, zero_h)
                e1 = b.select(in_range, e1, zero_h)
                v2 = b.vec_pack([e0, e1], dtype)
            else:
                v2 = b.global_load_vN(value, safe_elem, dtype, 2, align=4)
                # Zero masked rows (whole pair belongs to one token).
                zero_vec = b.vec_pack([zero_h, zero_h], dtype)
                v2 = b.select(in_range, v2, zero_vec)
            return b.bitcast(v2, I32)

        def _cfvst_block_coords(blk: Value):
            # block (tg, dg): tokens {2*tg, 2*tg+1}, dims {2*dg, 2*dg+1}.
            # Keep dim-pair as the fastest-varying coordinate so adjacent lanes
            # issue coalesced VMEM loads over the natural V[token, dim] layout.
            tg = b.div(blk, _dp)
            dg = b.mod(blk, _dp)
            t0 = b.mul(tg, b.const_i32(2))
            t1 = b.add(t0, b.const_i32(1))
            d0 = b.mul(dg, b.const_i32(2))
            d1 = b.add(d0, b.const_i32(1))
            return d0, d1, t0, t1

        payload = []
        for call in range(V_T_ITEMS_PER_THREAD):
            blk = b.add(b.mul(b.const_i32(call), b.const_i32(THREADS)), tid)
            if _v_t_need_item_guard:
                blk = b.select(
                    b.cmp_lt(blk, b.const_i32(_v_t2_total_blocks)),
                    blk,
                    b.const_i32(0),
                )
            d0, d1, t0, t1 = _cfvst_block_coords(blk)
            x0 = _load_token_row_pair(t0, d0)  # (V[t0,d0], V[t0,d1])
            x1 = _load_token_row_pair(t1, d0)  # (V[t1,d0], V[t1,d1])
            payload.append((d0, d1, t0, t1, x0, x1))
        return payload

    def _cfvst_store_v_regs(payload) -> None:
        """Permute loaded cfvst VGPRs and publish the transposed V LDS tile."""
        zero_h = b.cast_f32_to(b.const_f32(0.0), dtype)
        if _CFV_STORE_PREZERO:
            # DIAGNOSTIC: pre-zero every V_lds slot (dim x token) so any
            # uncovered slot reads 0 instead of garbage. If this fixes the
            # error, the store has a coverage gap.
            _pz = b.scf_for(
                tid, b.const_i32(HD * T), b.const_i32(THREADS), iv_name="vpz"
            )
            with _pz as pz:
                _pd = b.div(pz, b.const_i32(T))
                _ptk = b.mod(pz, b.const_i32(T))
                _v_t_store(_pd, _ptk, zero_h, 1)
            b.sync()

        for d0, d1, t0, t1, x0, x1 in payload:
            if _CFV_STORE_SCATTER:
                # DIAGNOSTIC: element-wise scatter (no perm) -- store each of
                # the 4 loaded V values directly at its transposed slot.
                x0v = b.bitcast(x0, VectorType(dtype, 2))  # (V[t0,d0],V[t0,d1])
                x1v = b.bitcast(x1, VectorType(dtype, 2))  # (V[t1,d0],V[t1,d1])
                v_t0_d0 = b.vec_extract(x0v, 0)
                v_t0_d1 = b.vec_extract(x0v, 1)
                v_t1_d0 = b.vec_extract(x1v, 0)
                v_t1_d1 = b.vec_extract(x1v, 1)
                _v_t_store(d0, t0, v_t0_d0, 1)
                _v_t_store(d0, t1, v_t1_d0, 1)
                _v_t_store(d1, t0, v_t0_d1, 1)
                _v_t_store(d1, t1, v_t1_d1, 1)
            else:
                # 2x2 transpose: each output i32 = 2 consecutive tokens at one dim.
                row_d0 = b.perm_b32(x0, x1, b.const_i32(0x01000504))  # (t0,t1)@d0
                row_d1 = b.perm_b32(x0, x1, b.const_i32(0x03020706))  # (t0,t1)@d1
                # ONE contiguous 2-half ds_write per dim row (token is inner).
                _v_t_store(d0, t0, b.bitcast(row_d0, VectorType(dtype, 2)), 2)
                _v_t_store(d1, t0, b.bitcast(row_d1, VectorType(dtype, 2)), 2)

    def _issue_v_transposed_store(kv_tile_idx: Value) -> None:
        """Register-load + ``perm_b32`` transpose + contiguous LDS store of V
        into ``V_lds[0, dim, token]`` (vehicle (c), f16). See block comment."""
        _cfvst_store_v_regs(_cfvst_load_v_regs(kv_tile_idx))

    # ---------------- FP8 K/V cache: async DMA loader (round 2) ----------------
    # Two-phase split that mirrors the bf16 path's HW DMA pipeline:
    #   1. `_issue_kv_fp8_async_load` issues `raw.ptr.buffer.load.lds`
    #      writing fp8 bytes directly into K_fp8_lds / V_fp8_lds. On gfx942
    #      this is the same capped 1-dword (4 bytes/lane) width as the bf16
    #      path (CDNA3 load-to-LDS limit); each lane moves 4 fp8 elements
    #      vs 2 bf16 halves, so fp8 needs half the bf16 call count.
    #   2. `_dequant_fp8_lds_to_bf16` runs in the kv loop, after the
    #      ``s_waitcnt vmcnt=0; s_barrier`` that publishes the fp8
    #      bytes. Each thread reads 8 fp8 from the fp8 slab, applies
    #      ``cvt_fp8_to_f32 * scale -> bf16``, writes 8 bf16 to the
    #      regular K_lds / V_lds slab. The dequant is LDS-to-LDS so it
    #      runs at LDS throughput (no HBM stall).
    #
    # This replaces the round-1 sync path (`_issue_fp8_dequant_loads`)
    # which issued `global_load_vN(FP8, n=8)` per chunk and blocked the
    # wave on every load. For long-prefill no-SW shapes that visit
    # 9-17 kv-tiles per CTA, the sync stall was the dominant cost
    # (5000 µs vs Triton's 50 µs measured on
    # `n254q5880k10999_fp8kv` in round 1).
    if KV_FP8 and spec.use_fp8_mfma_qk:
        # The async FP8 loader is gated on use_fp8_mfma_qk. We pick the
        # largest dwords (and per-lane payload) that the tile bytes
        # support, mirroring the K_FP8_DWORDS / K_BYTES_PER_LANE pick
        # we did for `_issue_k_fp8_mfma_async` above. This lets the
        # selector enable use_fp8_mfma_qk for small-T shapes (e.g. the
        # FP8 SW long-prefill choice T=BS=32) without an assert trip.
        FP8_DWORDS_PER_LANE = K_FP8_DWORDS
        FP8_BYTES_PER_LANE = K_BYTES_PER_LANE
        FP8_ELEMS_PER_LANE = FP8_BYTES_PER_LANE  # 1 byte per fp8 element
        FP8_ELEMS_PER_CALL = THREADS * FP8_ELEMS_PER_LANE
        FP8_CALLS_PER_TILE = (T * HD) // FP8_ELEMS_PER_CALL
        assert FP8_CALLS_PER_TILE >= 1 and (T * HD) % FP8_ELEMS_PER_CALL == 0, (
            f"fp8 async loader: T*HD={T * HD} not coverable by THREADS*{FP8_BYTES_PER_LANE}"
            f"={FP8_ELEMS_PER_CALL} (T={T}, HD={HD}, THREADS={THREADS})"
        )
        # Wave-uniform LDS offset (same idea as the bf16 path); each
        # wave's lanes write a contiguous WAVE*16-byte slab in LDS.
        FP8_WAVE_BYTES = WAVE * FP8_BYTES_PER_LANE
        FP8_BYTES_PER_CALL = FP8_ELEMS_PER_CALL  # 1 byte per fp8 element
        FP8_BYTES_PER_BUF = T * HD  # 1 byte per fp8 element

        # Lane base in ELEMENTS (== bytes for fp8) for this wave's call.
        lane_fp8_base = b.mul(tid, b.const_i32(FP8_ELEMS_PER_LANE))

        if NUM_WARPS == 1:
            wave_fp8_offset_i64 = b.const_i64(0)
        else:
            wave_fp8_offset_i32 = b.to_sgpr_u32(
                b.mul(wave_id, b.const_i32(FP8_WAVE_BYTES))
            )
            wave_fp8_offset_i64 = b.zext(wave_fp8_offset_i32, I64)

        K_fp8_lds_addr = b.smem_addr_of(K_fp8_lds)
        V_fp8_lds_addr = b.smem_addr_of(V_fp8_lds)

        def _issue_kv_fp8_async_load(
            kv_tile_idx: Value, buf_idx: Value, slot: str
        ) -> None:
            """Issue async DMA of fp8 K or V bytes into the staging slab.

            `slot` is ``"K"`` (uses K_fp8_lds[buf_idx]) or ``"V"`` (uses
            V_fp8_lds[0]; V is single-buffered, buf_idx ignored).
            """
            if slot == "K":
                rsrc = key_rsrc
                buf_off_i32 = b.mul(buf_idx, b.const_i32(FP8_BYTES_PER_BUF))
                buf_off_i64 = b.zext(buf_off_i32, I64)
                buf_base = b.smem_ptr_add(K_fp8_lds_addr, buf_off_i64)
            else:
                rsrc = value_rsrc
                # V is single-buffered; ignore buf_idx
                buf_base = V_fp8_lds_addr
            wave_base = b.smem_ptr_add(buf_base, wave_fp8_offset_i64)
            for call in range(FP8_CALLS_PER_TILE):
                # paged_kv_desc returns a BYTE offset; for fp8, linear_half
                # is interpreted as fp8-ELEMENT index (KV_BYTES=1 so element
                # offset == byte offset within a tile).
                linear_elem = b.add(
                    b.const_i32(call * FP8_ELEMS_PER_CALL), lane_fp8_base
                )
                call_rsrc = rsrc
                if I64_KV_ADDR:
                    base_i64, voff, _ = paged_kv_desc.offset_i64_split(
                        b,
                        "physical_block",
                        tile_idx=kv_tile_idx,
                        linear_half=linear_elem,
                        kv_head=kv_head_idx,
                    )
                    src_ptr = key if slot == "K" else value
                    call_rsrc = b.buffer_rsrc(
                        b.global_ptr_add(src_ptr, base_i64), kv_block_bytes_c
                    )
                else:
                    voff, _ = paged_kv_desc.offset(
                        b,
                        tile_idx=kv_tile_idx,
                        linear_half=linear_elem,
                        kv_head=kv_head_idx,
                    )
                lds_dst = b.smem_ptr_add(
                    wave_base, b.const_i64(call * FP8_BYTES_PER_CALL)
                )
                b.async_buffer_load_lds_addr(
                    call_rsrc,
                    lds_dst,
                    voff,
                    zero_soff,
                    FP8_DWORDS_PER_LANE,
                    coherency=kv_cache_aux,
                )

        # The dequant step distributes T*HD elements across THREADS in
        # 8-element chunks (matches the existing fp8_chunks_per_thread
        # layout). Each thread reads 8 fp8 from K_fp8_lds, applies the
        # dequant chain, and writes 8 bf16 to K_lds. The K_lds layout is
        # exactly what the bf16 async DMA produces, so the rest of the
        # kernel (Q gather, QK MFMA, PV MFMA) is identical to bf16.
        fp8_dequant_elems_per_chunk = 8
        fp8_dequant_total_chunks = (T * HD) // fp8_dequant_elems_per_chunk
        assert fp8_dequant_total_chunks % THREADS == 0, (
            f"fp8 dequant: total chunks {fp8_dequant_total_chunks} must be "
            f"divisible by THREADS={THREADS}"
        )
        fp8_dequant_chunks_per_thread = fp8_dequant_total_chunks // THREADS
        fp8_cols_per_row = HD // fp8_dequant_elems_per_chunk

        def _dequant_fp8_lds_to_bf16(
            buf_idx: Value, scale: Value, fp8_lds, bf16_lds, bf16_buf: Value
        ) -> None:
            """LDS->LDS dequant: fp8 -> f32 * scale -> bf16.

            `buf_idx` selects the fp8 source buffer (for K, the active
            double-buffer slot; for V, always 0).
            `bf16_buf` selects the bf16 destination buffer (for K, always
            0 since FP8 path single-buffers the bf16 K_lds; for V, 0).
            """
            for c in range(fp8_dequant_chunks_per_thread):
                chunk_id = b.add(b.mul(b.const_i32(c), b.const_i32(THREADS)), tid)
                row = b.div(chunk_id, b.const_i32(fp8_cols_per_row))
                col = b.mul(
                    b.mod(chunk_id, b.const_i32(fp8_cols_per_row)),
                    b.const_i32(fp8_dequant_elems_per_chunk),
                )
                fp8_vec = b.smem_load_vN(
                    fp8_lds,
                    buf_idx,
                    row,
                    col,
                    dtype=FP8E4M3,
                    n=fp8_dequant_elems_per_chunk,
                )
                dequanted = []
                for i in range(fp8_dequant_elems_per_chunk):
                    fp8_v = b.vec_extract(fp8_vec, i)
                    f32_v = b.fmul(b.cvt_fp8_to_f32(fp8_v), scale)
                    dequanted.append(b.cast_f32_to(f32_v, dtype))
                packed = b.vec_pack(dequanted, dtype)
                b.smem_store_vN(
                    bf16_lds,
                    [bf16_buf, row, col],
                    packed,
                    fp8_dequant_elems_per_chunk,
                )

    # ---------------- FP8 K/V cache: sync dequant loader (round 1, kept as fallback) ----------------
    # Each thread loads one byte per fp8 element from HBM, dequantises
    # (cvt_fp8_to_f32 * scale), casts to the working bf16/fp16 dtype, and
    # stores 8 elements at a time to LDS. The total bytes per tile match
    # the async path's working-dtype LDS layout, so the rest of the
    # kernel reads K_lds / V_lds in the working dtype unchanged.
    #
    # Layout: distribute T*HD elements across THREADS threads such that
    # each thread processes a contiguous run of ``elems_per_thread`` fp8
    # bytes from HBM, lane-contiguous, then writes them as bf16 to LDS at
    # the same linear offset. We pick the chunk size to be 8 so the LDS
    # store is one ``smem_store_vN(..., n=8)``.
    fp8_elems_per_chunk = 8
    fp8_total_chunks = (T * HD) // fp8_elems_per_chunk
    assert fp8_total_chunks % THREADS == 0, (
        f"fp8 loader: total chunks {fp8_total_chunks} must be divisible by "
        f"THREADS={THREADS} (T={T}, HD={HD})"
    )
    fp8_chunks_per_thread = fp8_total_chunks // THREADS

    def _issue_fp8_dequant_loads(
        kv_tile_idx: Value, buf_idx: Value, lds_token: str
    ) -> None:
        """Sync per-thread fp8 -> f32 -> *scale -> bf16/fp16 -> LDS.

        ``lds_token`` is either ``"K"`` or ``"V"``; selects the LDS slab
        and the per-tensor scale parameter.

        Per-chunk serial structure: each iteration issues one VMEM load,
        does the dequant chain, then publishes to LDS. The AMDGPU
        compiler's instruction scheduler interleaves the back-to-back
        chunks' loads, dequant, and stores.

        Cvt chain: uses ``cvt_pk_f32_fp8x4`` (lowers to two packed
        ``v_cvt_pk_f32_fp8`` instructions per 4 fp8 inputs) instead of
        the per-element ``cvt_fp8_to_f32`` (which the compiler does NOT
        fuse into the packed variant on its own — ISA inspection of the
        round-1 sync loader showed 48 separate ``v_cvt_f32_fp8_e32``
        instructions instead of 24 ``v_cvt_pk_f32_fp8`` for the same
        kernel). The packed primitive matches AITER's ``to_float_fp8x4``
        helper in ``csrc/include/attention_common.cuh`` which is what
        AITER's production paged-attention FP8 K/V path uses.

        For each ``fp8_elems_per_chunk = 8``-fp8 chunk we issue two
        packed cvts (4 fp8 each) — 4 ``v_cvt_pk_f32_fp8`` total per
        chunk vs the previous 8 scalar ``v_cvt_f32_fp8`` (50% fewer
        cvt-class instructions).
        """
        scale = k_scale_p if lds_token == "K" else v_scale_p
        lds = K_lds if lds_token == "K" else V_lds
        src = key if lds_token == "K" else value
        assert fp8_elems_per_chunk == 8, (
            f"FP8 dequant loader expects 8-elem chunks (got "
            f"{fp8_elems_per_chunk}); the packed-cvt path needs the chunk "
            f"to split cleanly into 2× <4 x fp8> sub-chunks."
        )
        for call in range(fp8_chunks_per_thread):
            chunk_id = b.add(
                b.mul(b.const_i32(call), b.const_i32(THREADS)),
                tid,
            )
            row = b.div(chunk_id, b.const_i32(HD // fp8_elems_per_chunk))
            col = b.mul(
                b.mod(chunk_id, b.const_i32(HD // fp8_elems_per_chunk)),
                b.const_i32(fp8_elems_per_chunk),
            )
            linear_half_first = b.add(
                b.mul(row, b.const_i32(HD)),
                col,
            )
            if I64_KV_ADDR:
                # Flat per-lane global load: full i64 byte/element offset so
                # the per-thread physical_block * stride cannot overflow the
                # cache addressing for paged caches > 2 GiB.
                voff, _ = paged_kv_desc.offset_i64(
                    b,
                    "physical_block",
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half_first,
                    kv_head=kv_head_idx,
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_half_first,
                    kv_head=kv_head_idx,
                )
            fp8_vec = b.global_load_vN(
                src, voff, FP8E4M3, n=fp8_elems_per_chunk, align=fp8_elems_per_chunk
            )
            # Split <8 x fp8> into 2x <4 x fp8> via extract+pack so we
            # can apply the packed cvt (the LLVM backend collapses the
            # extract+pack chain back into a bitcast-equivalent shuffle
            # once it sees the cvt.pk.f32.fp8 operand is whole-i32).
            lo_quad = b.vec_pack(
                [b.vec_extract(fp8_vec, i) for i in range(4)],
                FP8E4M3,
            )
            hi_quad = b.vec_pack(
                [b.vec_extract(fp8_vec, i) for i in range(4, 8)],
                FP8E4M3,
            )
            # CORRECTNESS: ``v_cvt_scalef32_pk_f32_fp8`` interprets its
            # ``scale`` operand as an E8M0 microscaling factor (only the
            # f32 exponent bits matter, mantissa is discarded). For
            # arbitrary per-tensor scales like ``k_scale = max_abs/448 ≈
            # 0.011`` the hardware silently rounds to the nearest
            # power-of-two (0.0078 instead of 0.011), giving outputs
            # ~0.71x of expected. Power-of-two scales happen to work
            # bit-correctly by coincidence, which is why the original
            # bench-driven "win" measurement looked correct on shapes
            # whose extracted ``k_scale`` rounded toward the right
            # exponent.
            #
            # The correct sequence is unfused: ``cvt_pk_f32_fp8`` (no
            # scaling) followed by a regular ``v_pk_mul`` against the
            # f32 scale. We pay ~4 extra packed muls per inner-loop
            # iter; that's the cost of being correct for non-pow2
            # scales. A future fast path can detect pow2 scales and
            # opt into the fused intrinsic, but the production trace
            # has only non-pow2 scales.
            lo_f32x4 = b.cvt_pk_f32_fp8x4(lo_quad)
            hi_f32x4 = b.cvt_pk_f32_fp8x4(hi_quad)
            lo_scaled = []
            hi_scaled = []
            for i in range(4):
                lo_scaled.append(b.fmul(b.vec_extract(lo_f32x4, i), scale))
                hi_scaled.append(b.fmul(b.vec_extract(hi_f32x4, i), scale))
            lo_f32x4 = b.vec_pack(lo_scaled, F32)
            hi_f32x4 = b.vec_pack(hi_scaled, F32)
            dequanted = []
            for i in range(4):
                dequanted.append(b.cast_f32_to(b.vec_extract(lo_f32x4, i), dtype))
            for i in range(4):
                dequanted.append(b.cast_f32_to(b.vec_extract(hi_f32x4, i), dtype))
            packed = b.vec_pack(dequanted, dtype)
            b.smem_store_vN(lds, [buf_idx, row, col], packed, fp8_elems_per_chunk)
        # Caller is expected to issue a `b.sync()` (LDS visibility) when
        # appropriate. The sync loader has no in-flight async work, so the
        # consumer side only needs the LDS barrier, not VMEM waitcnt.

    def _issue_k_fp8_mfma_async(kv_tile_idx: Value, buf_idx: Value) -> None:
        """fp8-K-LDS path: async DMA raw fp8 K bytes into fp8 K_lds.

        K_lds is allocated as fp8 (8 KB for double-buf at T=64 HD=64 vs
        16 KB bf16). The async DMA writes ``K_BYTES_PER_LANE`` bytes per
        lane per call; on gfx942 the dwords selector is clamped to the
        CDNA3 load-to-LDS max (1 dword = 4 bytes/lane), issuing more calls
        per tile to move the same bytes. QK reads K_lds as fp8 and dequants
        in-register with k_scale to the bf16 input the standard bf16 MFMA
        expects.
        """
        buf_off_i32 = b.mul(buf_idx, b.const_i32(T * HD))  # fp8: 1 byte/elem
        buf_off_i64 = b.zext(buf_off_i32, I64)
        K_buf_base = b.smem_ptr_add(K_lds_addr, buf_off_i64)
        # Per-wave LDS offset in BYTES.
        if NUM_WARPS == 1:
            wave_fp8_off_i64 = b.const_i64(0)
        else:
            wave_fp8_off_i32 = b.to_sgpr_u32(
                b.mul(wave_id, b.const_i32(WAVE * K_BYTES_PER_LANE))
            )
            wave_fp8_off_i64 = b.zext(wave_fp8_off_i32, I64)
        K_wave_base = b.smem_ptr_add(K_buf_base, wave_fp8_off_i64)
        # Lane base in fp8 ELEMENTS (== bytes for fp8).
        lane_fp8_base = b.mul(tid, b.const_i32(K_BYTES_PER_LANE))
        for call in range(k_fp8_calls_per_tile):
            linear_elem = b.add(b.const_i32(call * K_ELEMS_PER_CALL), lane_fp8_base)
            k_rsrc = key_rsrc
            if I64_KV_ADDR:
                base_i64, voff, _ = paged_kv_desc.offset_i64_split(
                    b,
                    "physical_block",
                    tile_idx=kv_tile_idx,
                    linear_half=linear_elem,
                    kv_head=kv_head_idx,
                )
                k_rsrc = b.buffer_rsrc(
                    b.global_ptr_add(key, base_i64), kv_block_bytes_c
                )
            else:
                voff, _ = paged_kv_desc.offset(
                    b,
                    tile_idx=kv_tile_idx,
                    linear_half=linear_elem,
                    kv_head=kv_head_idx,
                )
            k_dst = b.smem_ptr_add(K_wave_base, b.const_i64(call * K_BYTES_PER_CALL))
            b.async_buffer_load_lds_addr(
                k_rsrc,
                k_dst,
                voff,
                zero_soff,
                K_FP8_DWORDS,
                coherency=kv_cache_aux,
            )

    def _issue_v_fp8_mfma_async(kv_tile_idx: Value) -> None:
        """Native-fp8 PV path: async DMA raw fp8 V bytes into V_lds[0].

        This mirrors `_issue_k_fp8_mfma_async` but V is single-buffered.
        V is consumed once by PV after the softmax/P publish phase; the
        loop's existing `s_waitcnt(vmcnt=0, lgkmcnt=0); sync` before PV
        makes the raw fp8 bytes visible before the `ds_read_b64_tr_b8`
        transpose reads.
        """
        V_buf_base = V_lds_addr
        if NUM_WARPS == 1:
            wave_fp8_off_i64 = b.const_i64(0)
        else:
            wave_fp8_off_i32 = b.to_sgpr_u32(
                b.mul(wave_id, b.const_i32(WAVE * K_BYTES_PER_LANE))
            )
            wave_fp8_off_i64 = b.zext(wave_fp8_off_i32, I64)
        V_wave_base = b.smem_ptr_add(V_buf_base, wave_fp8_off_i64)
        lane_fp8_base = b.mul(tid, b.const_i32(K_BYTES_PER_LANE))
        for call in range(k_fp8_calls_per_tile):
            linear_elem = b.add(b.const_i32(call * K_ELEMS_PER_CALL), lane_fp8_base)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_elem,
                kv_head=kv_head_idx,
            )
            v_dst = b.smem_ptr_add(V_wave_base, b.const_i64(call * K_BYTES_PER_CALL))
            b.async_buffer_load_lds_addr(
                value_rsrc,
                v_dst,
                voff,
                zero_soff,
                K_FP8_DWORDS,
                coherency=kv_cache_aux,
            )

    def _issue_v_fp8_mfma_stripe(kv_tile_idx: Value) -> None:
        """Load raw fp8 V and store it into V_lds as
        [V_BUFS=1, N_STRIPES=HD/16, T, 16].

        Each thread loads 8 contiguous fp8 values from V[token, col..col+7]
        in HBM, then writes them as one 8-byte LDS vec store into the
        owning stripe at V_lds[0, col/16, token, col%16].

        Because our chunk size is exactly 8 fp8 (= half a 16-byte stripe
        row), col%16 is always 0 or 8 — both 8-byte-aligned, so the LDS
        vec store stays b64 and there is no scalar-byte regression.
        """
        for call in range(fp8_chunks_per_thread):
            chunk_id = b.add(
                b.mul(b.const_i32(call), b.const_i32(THREADS)),
                tid,
            )
            token = b.div(chunk_id, b.const_i32(HD // fp8_elems_per_chunk))
            col = b.mul(
                b.mod(chunk_id, b.const_i32(HD // fp8_elems_per_chunk)),
                b.const_i32(fp8_elems_per_chunk),
            )
            linear_first = b.add(b.mul(token, b.const_i32(HD)), col)
            voff, _ = paged_kv_desc.offset(
                b,
                tile_idx=kv_tile_idx,
                linear_half=linear_first,
                kv_head=kv_head_idx,
            )
            fp8_vec = b.global_load_vN(
                value, voff, FP8E4M3, n=fp8_elems_per_chunk, align=fp8_elems_per_chunk
            )
            stripe_idx = b.div(col, b.const_i32(16))
            col_in_stripe = b.mod(col, b.const_i32(16))
            b.smem_store_vN(
                V_lds,
                [b.const_i32(0), stripe_idx, token, col_in_stripe],
                fp8_vec,
                fp8_elems_per_chunk,
            )

    def _issue_k(tile_idx: Value, buf_idx: Value) -> None:
        """Issue a K load into the appropriate LDS slab.

        Three paths:
        - bf16: async DMA to bf16 K_lds.
        - FP8 sync dequant (round 1): per-thread `global_load_vN(FP8)` +
          dequant chain + LDS store to bf16 K_lds. Implicitly pipelined
          across chunks by the AMDGPU backend.
        - FP8 native MFMA (round 2 v2, `use_fp8_mfma_qk=True`): async DMA
          of raw fp8 bytes to fp8 K_lds. No dequant; the MFMA op reads
          fp8 directly and the k_scale is applied to the f32 accumulator
          via the folded post-MFMA `qk_scale` constant.
        """
        if K_SLICED_ACTIVE:
            return
        if K_FP8_MFMA:
            _issue_k_fp8_mfma_async(tile_idx, buf_idx)
        elif KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, buf_idx, "K")
        else:
            _issue_k_load_runtime(tile_idx, buf_idx)

    def _issue_v(tile_idx: Value, buf_idx: Value) -> None:
        """Issue a V load (single-buffered). See `_issue_k` for FP8 notes.

        ``V_lds`` is allocated with ``V_BUFS=1`` so only slot 0 is valid.
        The caller passes ``cur_buf`` (which alternates 0/1 across iters
        for the K double-buffer), but for V we MUST pin it to slot 0 --
        passing ``buf_idx=1`` into the FP8 sync loader produces an OOB
        LDS store at ``V_lds[1, ...]`` which lands on top of the
        subsequent LDS slabs (P_lds, Acc_lds). For BLOCK_M=64 T=64 the
        clobber happens to land exactly on P_lds and is overwritten by
        the softmax publish before PV reads it (so the bug is invisible).
        For BLOCK_M in {16, 32} P_lds is smaller (2 KB or 4 KB), the OOB
        write extends past P_lds into Acc_lds (and whatever follows),
        and the kernel produces large output errors (max_abs ~9-11 vs
        the FP8 noise floor of ~0.3). The bf16 path is unaffected
        because ``_issue_v_load_runtime`` ignores ``buf_idx`` and uses
        ``V_lds_addr`` directly. Fix: always pin slot 0 for V.
        """
        if PV_FP8_MFMA:
            _issue_v_fp8_mfma_stripe(tile_idx)
        elif KV_FP8:
            _issue_fp8_dequant_loads(tile_idx, b.const_i32(0), "V")
        elif TRANSPOSED_V_STORE:
            _issue_v_transposed_store(tile_idx)
        elif TRANSPOSED_V:
            _issue_v_transposed(tile_idx)
        else:
            _issue_v_load_runtime(tile_idx, buf_idx)

    def _read_k8_mfma_operand(
        buf_idx: Value, k_row: Value, k_off: Value, frag: int = 8
    ) -> Value:
        """Read ``frag`` K elements from K_lds as the bf16 MFMA operand.

        ``frag`` is 8 for the K=16 (32x32x16) atom and 4 for the K=8
        (32x32x8) atom -- each lane owns half of K per MFMA. For the
        fp8-in-LDS path (``K_FP8_MFMA``) K_lds holds raw fp8 bytes
        (half the LDS of bf16 -> higher occupancy). Dequant in register --
        ``cvt_pk_f32_fp8`` + ``* k_scale`` + cast -- to the exact bf16 the
        bf16 K_lds path would have held (bit-identical), then feed the
        standard bf16 MFMA. (Unfused cvt + explicit f32 multiply: the fused
        ``cvt_scalef32`` truncates non-pow2 scales; see the 16x16 path.)
        Otherwise read bf16 directly. Shared by the 16x16 and 32x32 QK
        paths so both get the fp8 LDS-footprint win. (fp8 is rejected on
        gfx942, so the x8-transposed path always takes the plain bf16 read.)
        """
        if not K_FP8_MFMA:
            return b.smem_load_vN(K_lds, buf_idx, k_row, k_off, dtype=dtype, n=frag)
        if FP8_NATIVE_QK:
            # Native fp8 QK: hand the raw fp8 K straight to the fp8 MFMA --
            # no dequant. (k_scale is folded into qk_scale post-MFMA.)
            return b.smem_load_vN(K_lds, buf_idx, k_row, k_off, dtype=FP8E4M3, n=frag)
        k_fp8 = b.smem_load_vN(K_lds, buf_idx, k_row, k_off, dtype=FP8E4M3, n=frag)
        return dequant_fp8x8_to_dtype(b, k_fp8, k_scale_p, dtype)

    # ---- Per-lane Q → VGPR gather (eliminates per-iter Q LDS reads) ----
    # Each lane reads its MFMA-A operand slice of Q (16 halves per atom)
    # into VGPRs ONCE per CTA. Subsequent QK iterations use ``Q_reg``
    # directly instead of paying the 16-way bank-conflicted Q_lds read
    # ``num_tiles`` times. Per lane VGPR cost: 8 halves × QK_K_ITERS ×
    # M_ATOMS_PER_WARP = up to 32 halves = 16 VGPRs for
    # ``BLOCK_M_PER_WARP=32``.
    Q_reg = [[None] * QK_K_ITERS for _ in range(M_ATOMS_PER_WARP)]
    if not (Q_DIRECT_GLOBAL or (USE_MFMA_32X32 and SKIP_LEGACY_QREG)):
        for atom in range(M_ATOMS_PER_WARP):
            q_row_atom = b.add(wave_row_base, b.add(b.const_i32(atom * 16), lane_col))
            # Map Q_row → (buf, row_in_buf) for the K_lds-aliased case.
            if Q_ALIAS_K:
                if Q_USES_DUAL_SLOT:
                    q_buf_atom = b.div(q_row_atom, b.const_i32(T))
                    q_row_in_buf_atom = b.mod(q_row_atom, b.const_i32(T))
                else:
                    q_buf_atom = b.const_i32(0)
                    q_row_in_buf_atom = q_row_atom
            for k in range(QK_K_ITERS):
                # 16x16x16 A (Q) operand per lane: row = lane%16, K =
                # k*16 + lane_rg*4 + 0..3 (<4 x dtype>). K-step is 16 on gfx942
                # (no wide-K atom), so the per-lane column base is lane_rg*4.
                q_col_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                q_load_idx_args = (
                    (q_buf_atom, q_row_in_buf_atom, q_col_off)
                    if Q_ALIAS_K
                    else (q_row_atom, q_col_off)
                )
                Q_reg[atom][k] = b.smem_load_vN(
                    Q_lds, *q_load_idx_args, dtype=dtype, n=4
                )

        if Q_ALIAS_K:
            # Drain the per-lane Q-gather LDS reads BEFORE issuing K[0] async
            # write to the same K_lds[0] slot. Without this, the async DMA's
            # LDS write can race with the in-flight ds_read, corrupting Q
            # for the first QK iter. One-time cost at kernel start.
            b.s_waitcnt(lgkmcnt=0)
            b.sync()

    # ---- Per-lane Q gather for CK Tile/Triton 32x32x16 QK geometry ----
    #
    # This is the main-kernel migration path for long-prefill. It is not a
    # separate fallback kernel. Once the downstream mask/softmax/PV sections
    # consume the 32x32 accumulator layout, this replaces the 16x16x32
    # QK body above for long-prefill shapes.
    #
    # CK Tile traits for M32N32K16:
    #   A (Q) per lane: <8 x dtype>
    #     row = wave_row_base + (lane % 32)
    #     k   = k_iter*16 + (lane/32)*8 + [0..7]
    #   B (K^T) per lane: <8 x dtype>
    #     n   = n_tile*32 + (lane % 32)
    #     k   = k_iter*16 + (lane/32)*8 + [0..7]
    #   C per lane: <16 x f32>
    #     row = wave_row_base + ((elem//4)*8 + (lane/32)*4 + elem%4)
    #     col = tile_off + n_tile*32 + (lane % 32)
    #
    # The key structural win is that each lane owns 16 scores in one column
    # rather than 4 scores. Softmax can reduce most values in-lane with
    # chained v_max3 and only then use 32-lane-half shuffles. The old
    # layout needed 4-stage row-group shuffles for every row slot.
    if USE_MFMA_32X32:
        # K: both fp8 modes are supported now -- the sync-dequant loader
        # writes bf16 K_lds (read directly), and the fp8-in-LDS loader
        # (``use_fp8_mfma_qk``) keeps raw fp8 in K_lds, which
        # ``_read_k8_mfma_operand`` dequants in-register to bf16 before the
        # 32x32 MFMA (half the K LDS footprint -> higher occupancy).
        # V: the 32x32 PV reads V_lds as bf16, so native-fp8 PV (V kept fp8
        # in LDS) is not wired into the transposed PV yet.
        if FP8_MFMA_PV:
            raise NotImplementedError(
                "32x32x16 PV needs bf16 V in LDS; disable use_fp8_mfma_pv "
                "(it is broken / slower anyway) for the fp8 combo"
            )
        Q32_reg = [None] * QK_K_ITERS
        lane_half = b.div(lane, b.const_i32(32))
        lane_col32 = b.mod(lane, b.const_i32(32))
        q32_row = b.add(wave_row_base, lane_col32)
        if Q_ALIAS_K:
            if Q_USES_DUAL_SLOT:
                q32_buf = b.div(q32_row, b.const_i32(T))
                q32_row_in_buf = b.mod(q32_row, b.const_i32(T))
            else:
                q32_buf = b.const_i32(0)
                q32_row_in_buf = q32_row
        # 32x32x8 A (Q) per lane: <4 x half>, row = lane%32,
        #   K = k*8 + (lane//32)*4 + [0..3]  (k_blk = lane//32).
        # 32x32x16 A (Q) per lane: <8 x half>, K = k*16 + (lane//32)*8 + [0..7].
        Q32_FRAG = 4 if USE_MFMA_32X32X8 else 8
        Q32_HALF_STRIDE = 4 if USE_MFMA_32X32X8 else 8
        for k in range(QK_K_ITERS):
            q32_col = b.add(
                b.const_i32(k * QK_K_STEP),
                b.mul(lane_half, b.const_i32(Q32_HALF_STRIDE)),
            )
            if Q_DIRECT_GLOBAL:
                q32_pos = b.add(qb_start_pos, b.div(q32_row, b.const_i32(NQK)))
                q32_h = b.add(
                    b.mul(kv_head_idx, b.const_i32(NQK)),
                    b.mod(q32_row, b.const_i32(NQK)),
                )
                q32_mask = b.land(
                    b.cmp_lt(q32_pos, cur_batch_q_len),
                    b.cmp_lt(q32_h, b.const_i32(NUM_QH)),
                )
                q32_pos_safe = b.select(q32_mask, q32_pos, b.const_i32(0))
                q32_h_safe = b.select(q32_mask, q32_h, b.const_i32(0))
                q32_off, _ = q_desc.offset(
                    b,
                    token=b.add(cu_q_start, q32_pos_safe),
                    head=q32_h_safe,
                    dim=q32_col,
                )
                q32_raw = b.global_load_vN(query, q32_off, dtype, Q32_FRAG, align=8)
                q32 = b.vector_select(
                    b.vector_splat(q32_mask, Q32_FRAG),
                    q32_raw,
                    b.zero_vec(dtype, Q32_FRAG),
                )
            else:
                q32_idx_args = (
                    (q32_buf, q32_row_in_buf, q32_col)
                    if Q_ALIAS_K
                    else (q32_row, q32_col)
                )
                q32 = b.smem_load_vN(Q_lds, *q32_idx_args, dtype=dtype, n=Q32_FRAG)
            if FP8_NATIVE_QK:
                # Quantize the Q operand to fp8 once so the QK MFMA can run
                # native fp8xfp8 (no per-tile K dequant). Q values are unit-
                # scale bf16, well within fp8 e4m3 range.
                q32 = b.vec_pack(
                    [
                        b.cvt_f32_to_fp8(b.cast_to_f32(b.vec_extract(q32, i)))
                        for i in range(8)
                    ],
                    FP8E4M3,
                )
            Q32_reg[k] = q32
        if Q_ALIAS_K:
            # Q32_reg was gathered AFTER the original Q_reg drain above.
            # When Q_lds aliases K_lds, the upcoming K[0] async prefetch
            # writes into the same LDS slabs. Drain these 32x32 Q reads too
            # before issuing K[0], otherwise random Q (but not all-zero Q)
            # suffers sparse large errors from read/write races.
            b.s_waitcnt(lgkmcnt=0)
            b.sync()

    # ULP-correct fp8-K LDS path -- NO Q quantisation, Q stays bf16.
    # K_FP8_MFMA is True iff K_lds holds raw fp8 bytes (saving 8 KB of
    # LDS at T=64/HD=64 and halving the K HBM->LDS DMA). The QK MFMA
    # dequants K fp8 -> bf16 IN-REGISTER and runs the standard bf16
    # ``mfma_f32_16x16x32_bf16``. The bf16 fed to MFMA is bit-identical
    # to the bf16 the current default path stores in bf16 K_lds, so
    # outputs are ULP-identical to default and to Triton's fp8 path
    # (which also stays in bf16 for ``S = qk_scale * tl.dot(Q, K_bf16)``).
    # An earlier variant of this flag also quantised Q to fp8 and used
    # the native fp8 MFMA; that variant lost ULP correctness and was
    # dropped. See the spec docstring for the history.
    # NOTE: above re-cast also requires Q_lds to still hold the bf16
    # Q. For Q_ALIAS_K the K[0] prefetch (issued below) would
    # overwrite Q_lds, so we DELAY the K issue until after this cast.
    # The Q_ALIAS_K drain barrier above already ran; the K issue is
    # the next thing scheduled, so this ordering is naturally
    # correct in the IR.

    # Prefetch tile_start's K into buffer 0 BEFORE the loop.
    _issue_k(tile_start, b.const_i32(0))

    # ---------------- KV tile loop ----------------
    # Double-buffered: we carry ``cur_buf`` (the buffer that holds tile i's
    # data) through the loop. At iter i we:
    #   1. Wait for current K (prefetched by the previous iteration, or the
    #      pre-loop prologue).
    #   2. Compute QK.
    #   3. Issue current V, then next K, and run softmax while both are in
    #      flight. Since current V is older than next K in the VMEM/LGKM
    #      queues, a partial wait with ``kv_calls_per_tile`` pending leaves
    #      next K in flight while making current V visible for PV.
    #   4. ``s_barrier`` to make tile i's data visible to all reads.
    #   5. Wait for current V, publish P_lds, then run PV.
    #   6. Yield ``(m, l, acc, nxt_buf)`` so the next iter consumes nxt_buf.
    cur_buf_init = b.const_i32(0)
    iter_args.append(("cur_buf", cur_buf_init))

    # ---------------- LICM hoist: per-reg invariants ----------------
    # ``qp_r``, ``qh_r``, ``row_ok``, ``causal_lim``, and ``alibi_per_row``
    # depend only on CTA-level constants (``wave_row_base``, ``lane_rg``,
    # ``qb_start_pos``, ``kv_head_idx``, ``cur_batch_q_len``, ``context_len``,
    # ``NUM_QH``, ``NQK``). Computing them BEFORE the kvloop avoids paying
    # them per kv tile. LLVM LICM should hoist them automatically, but
    # explicit hoisting makes the IR's hot path leaner and eliminates a
    # source of compiler-scheduling variability. The per-reg arrays
    # ``hoist_*[reg]`` for ``reg in range(REGS_PER_LANE)`` are indexed by
    # the per-lane row slot.
    hoist_row = []
    hoist_qp_r = []
    hoist_qh_r = []
    hoist_row_ok = []
    hoist_causal_lim = []
    for reg in range(REGS_PER_LANE):
        if USE_MFMA_32X32:
            row = b.add(wave_row_base, _mfma_32x32_c_row(b, lane, reg))
        else:
            row = b.add(wave_row_base, _in_warp_row(reg))
        qp_r = b.add(qb_start_pos, b.div(row, b.const_i32(NQK)))
        qh_r = b.add(b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(row, b.const_i32(NQK)))
        row_ok = b.land(
            b.cmp_lt(qp_r, cur_batch_q_len), b.cmp_lt(qh_r, b.const_i32(NUM_QH))
        )
        causal_lim = b.add(context_len, qp_r)
        hoist_row.append(row)
        hoist_qp_r.append(qp_r)
        hoist_qh_r.append(qh_r)
        hoist_row_ok.append(row_ok)
        hoist_causal_lim.append(causal_lim)

    if USE_ALIBI:
        hoist_alibi = []
        for reg in range(REGS_PER_LANE):
            qh_r = hoist_qh_r[reg]
            qh_ok = b.cmp_lt(qh_r, b.const_i32(NUM_QH))
            slope = b.masked_global_load(
                alibi_slopes_ptr, qh_r, qh_ok, b.const_f32(0.0), dtype=F32, align=4
            )
            hoist_alibi.append(slope)
    else:
        hoist_alibi = None

    if USE_MFMA_32X32 and TRANSPOSED_QK_32X32 and TRANSPOSED_INVARIANT_HOIST:
        st_q_row = b.add(wave_row_base, _mfma_32x32_c_col(b, lane, 0))
        st_qp = b.add(qb_start_pos, b.div(st_q_row, b.const_i32(NQK)))
        st_qh = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)), b.mod(st_q_row, b.const_i32(NQK))
        )
        st_row_ok = b.land(
            b.cmp_lt(st_qp, cur_batch_q_len), b.cmp_lt(st_qh, b.const_i32(NUM_QH))
        )
        st_causal_lim = b.add(context_len, st_qp)
    else:
        st_q_row = None
        st_qp = None
        st_qh = None
        st_row_ok = None
        st_causal_lim = None

    # Transposed-32x32 ALiBi slope hoist. ``st_qh`` is per-lane (constant
    # across reg/n), so the slope is per-lane too. We load it once outside
    # the kvloop and reuse it for every (reg, n) inside the
    # transposed-softmax block. When ``TRANSPOSED_INVARIANT_HOIST`` is
    # off and ``TRANSPOSED_MASK_ONCE`` is on we recompute per-iter
    # inside the kvloop instead.
    if (
        USE_ALIBI
        and USE_MFMA_32X32
        and TRANSPOSED_QK_32X32
        and TRANSPOSED_INVARIANT_HOIST
    ):
        st_qh_ok = b.cmp_lt(st_qh, b.const_i32(NUM_QH))
        st_alibi_slope = b.masked_global_load(
            alibi_slopes_ptr, st_qh, st_qh_ok, b.const_f32(0.0), dtype=F32, align=4
        )
    else:
        st_alibi_slope = None

    kv_step = b.const_i32(2 if GROUPED_KV2 else 1)

    # NOTE: the per-tile body lives in a local ``_emit_kv_body`` closure. An
    # earlier experiment drove it from TWO phases (a "full tile" phase with
    # the causal mask elided -- provably a no-op since ``select(true, s, -inf)
    # == s`` -- and a masked boundary phase) to cut the per-element mask VALU.
    # It was bit-exact but ~7% SLOWER: this kernel is latency/occupancy-bound,
    # not VALU-throughput-bound, so duplicating the ~1100-line body across two
    # loops cost more in I-cache / code size than the masking VALU it saved.
    # Kept as a single masked loop. ``skip_mask`` is wired through (always
    # False here) so the experiment can be re-tried behind a flag if a future
    # change makes the kernel throughput-bound.
    def _emit_kv_body(kv_tile_iv, carry, skip_mask):
        if USE_IGLP_OPT:
            b.iglp_opt(1)
        m_vals = [carry[2 * r] for r in range(SOFTMAX_STATE_SLOTS)]
        l_vals = [carry[2 * r + 1] for r in range(SOFTMAX_STATE_SLOTS)]
        ml_count = 2 * SOFTMAX_STATE_SLOTS
        acc_vals = [
            carry[ml_count + n * ACC_M_ATOMS + a]
            for n in range(ACC_N_TILES)
            for a in range(ACC_M_ATOMS)
        ]

        # acc_vals is flat indexed by (n * M_ATOMS_PER_WARP + atom).
        def _acc_get(n: int, atom: int) -> Value:
            return acc_vals[n * ACC_M_ATOMS + atom]

        cur_buf = carry[ml_count + ACC_N_TILES * ACC_M_ATOMS]
        # Single-buffered K: there is only slot 0, so the "next" buffer IS the
        # current one. (cur_buf is also pinned to 0 by the const-0 carry init +
        # the const-0 yield below.) Double-buffered K alternates 0<->1.
        nxt_buf = cur_buf if K_SINGLE_BUF else b.sub(b.const_i32(1), cur_buf)
        tile_off = b.mul(kv_tile_iv, b.const_i32(T))
        if GROUPED_KV2:
            tile1_iv_raw = b.add(kv_tile_iv, b.const_i32(1))
            tile1_in_range = b.cmp_lt(tile1_iv_raw, tile_end)
            safe_tile1 = b.select(tile1_in_range, tile1_iv_raw, kv_tile_iv)
            tile_off_g1 = b.add(tile_off, b.const_i32(T))
        else:
            tile1_iv_raw = None
            tile1_in_range = None
            safe_tile1 = None
            tile_off_g1 = None

        if USE_MFMA_32X32 and TRANSPOSED_QK_32X32 and TRANSPOSED_MASK_ONCE:
            st_q_row_iter = b.add(wave_row_base, _mfma_32x32_c_col(b, lane, 0))
            st_qp_iter = b.add(qb_start_pos, b.div(st_q_row_iter, b.const_i32(NQK)))
            st_qh_iter = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)),
                b.mod(st_q_row_iter, b.const_i32(NQK)),
            )
            st_row_ok_iter = b.land(
                b.cmp_lt(st_qp_iter, cur_batch_q_len),
                b.cmp_lt(st_qh_iter, b.const_i32(NUM_QH)),
            )
            st_causal_lim_iter = b.add(context_len, st_qp_iter)
            if USE_ALIBI:
                st_qh_iter_ok = b.cmp_lt(st_qh_iter, b.const_i32(NUM_QH))
                st_alibi_slope_iter = b.masked_global_load(
                    alibi_slopes_ptr,
                    st_qh_iter,
                    st_qh_iter_ok,
                    b.const_f32(0.0),
                    dtype=F32,
                    align=4,
                )
            else:
                st_alibi_slope_iter = None
        else:
            st_qp_iter = None
            st_row_ok_iter = None
            st_causal_lim_iter = None
            st_alibi_slope_iter = None

        # Prepare the clamped tile index for the next-K prefetch we will issue
        # after QK. The final iteration intentionally prefetches the current
        # tile again into the alternate buffer; this keeps the schedule uniform.
        if GROUPED_KV2:
            next_tile_iv_raw = b.add(kv_tile_iv, b.const_i32(2))
        else:
            next_tile_iv_raw = b.add(kv_tile_iv, b.const_i32(1))
        in_range_next = b.cmp_lt(next_tile_iv_raw, tile_end)
        safe_next_tile = b.select(in_range_next, next_tile_iv_raw, kv_tile_iv)

        # Wait for current K. There should be no in-flight next-K work here;
        # the previous iteration waited all async loads before PV.
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.sync()
        if EARLY_V_SCHEDULE:
            # V_lds is single-buffered. The iter-start full drain guarantees
            # the previous PV's V reads retired, so current V can be issued
            # before QK and overlap with QK + softmax.
            _issue_v(kv_tile_iv, cur_buf)
        if GROUPED_KV2:
            # Prototype schedule: while QK0 reads cur_buf, prefetch QK1 into
            # nxt_buf. This keeps the two K tiles resident simultaneously but
            # intentionally preserves the existing single-buffered V path.
            _issue_k(safe_tile1, nxt_buf)

        # ---- FP8 path: dequant placeholder (reverted in round-2) ----
        # Round-2 v1 attempted async DMA fp8 -> LDS dequant -> K_lds here
        # but the extra barriers regressed perf by ~10%. The sync FP8
        # loader (`_issue_fp8_dequant_loads`) is dispatched directly via
        # `_issue_k`/`_issue_v` and handles its own VMEM + dequant + LDS
        # in one chunk-pipelined sequence, so no extra work is needed at
        # this point in the loop for FP8.

        # ---- S = Q @ K^T (per-warp MFMA) ----
        # Q is in LDS only; we re-read it per iter -- the compiler hoists the
        # LDS reads across iterations when alignment lets it (Q never changes
        # after the prelude). Each warp reads its own ``BLOCK_M_PER_WARP``-row
        # slice of Q[BLOCK_M, HD] at rows
        # ``[wave_row_base, wave_row_base + BLOCK_M_PER_WARP)``.
        # For ``BLOCK_M_PER_WARP=32`` we read Q for both atoms (rows
        # ``wave_row_base + lane_col`` for atom 0 and ``wave_row_base + 16 +
        # lane_col`` for atom 1), then run the inner MFMA loop twice -- once
        # per atom -- sharing the same K read across both atoms.
        # Q comes from pre-loop VGPR gather (``Q_reg[atom][k]``); no per-iter
        # Q_lds reads, no per-iter Q LDS bank conflict.
        A_kits = Q_reg
        # S_n[atom][n] = vec_f32(4) -- per-atom, per-N-tile accumulator.
        # ``sched_group_barrier`` hints were tried here (mirroring the CK
        # Tile ``compv4`` GEMM pattern) but **regressed prefill_q64 by
        # ~50%** -- the hints constrain the scheduler in a way that doesn't
        # fit attention's mask + softmax + PV pattern, where the post-RA
        # scheduler's default heuristics already produce good interleave.
        # Consistent with the conv-kernel optimization study finding that
        # "compiler scheduling hints don't work on gfx950" (per an
        # internal conv kernel optimization study). Leaving them out.
        if USE_MFMA_32X32:
            if TRANSPOSED_QK_32X32:
                # Transposed-score orientation: compute S^T = K @ Q^T.
                #
                # A operand = K tile rows (KV tokens), B operand = Q rows
                # (queries). The resulting C tile has:
                #   row = KV position inside the tile
                #   col = query row inside the warp's 32-row Q tile
                #
                # This orientation matches the Triton/CK Tile efficient
                # reduction shape: softmax over K for one query column can
                # be reduced mostly in-lane across the 16 C registers.
                # B operand of the transposed MFMA wants Q[q_row=L%32,
                # k_dim=(L/32)*8 + i + k*16]. This is the SAME per-lane
                # layout that ``Q32_reg[k]`` already holds (loaded in the
                # pre-loop Q gather above) -- the K-axis distribution is
                # identical between the non-transposed and transposed
                # operands of M32N32K16 MFMA. We MUST consume Q32_reg
                # rather than re-reading ``Q_lds`` here: when
                # ``Q_ALIAS_K`` is true (the bf16 path always sets it),
                # the K[0] async prefetch issued before the kvloop has
                # already overwritten Q_lds with K data. Reading Q_lds
                # would silently feed K values into the B operand and
                # silently corrupt every transposed S^T.
                # K-step parameterization for the transposed QK MFMA. For
                # the gfx942-legal 32x32x8 atom each lane owns K = k*8 +
                # (lane/32)*4 + [0..3] (<4 x half>); for the gfx950-only
                # 32x32x16 atom it owns K = k*16 + (lane/32)*8 + [0..7]
                # (<8 x half>). The A (K^T) and B (Q^T) operands share the
                # SAME K distribution, so ``Q32_reg`` (gathered with the
                # parameterized ``Q32_HALF_STRIDE`` above) is reused verbatim.
                TQK_FRAG = 4 if USE_MFMA_32X32X8 else 8
                TQK_HALF_STRIDE = 4 if USE_MFMA_32X32X8 else 8
                if K_SLICED_ACTIVE:
                    # CK-style sliced-K schedule, depth-3:
                    #   prologue issues slices 0 and 1
                    #   per slice kg, issue kg+2 into round-robin slot
                    #   partial-wait leaving only the newest slice in flight
                    #   consume kg while kg+2 streams into a distinct slot
                    #
                    # The wait value is exactly one slice prefetch worth of
                    # VMEM calls. This is the same correctness invariant as the
                    # streaming beat-CK kernel: partial waits are FIFO-count
                    # fences, so the slot map must be pure kg % 3. The live set
                    # {kg, kg+1, kg+2} then occupies three distinct slots.
                    ST32_n = [b.zero_vec_f32(16) for _ in range(QK_N_TILES)]
                    k_groups = HD // K_SLICE_HD
                    k_steps_per_group = K_SLICE_HD // QK_K_STEP

                    def _kslot(group_idx: int) -> int:
                        if K_SLICED_LDSSEQ and k_groups == 4:
                            return (1, 2, 0, 1)[group_idx]
                        if K_SLICED_LDSSEQ and k_groups == 2:
                            return (1, 2)[group_idx]
                        return group_idx % K_SLICE_SLOTS

                    _issue_k_slice_load_runtime(kv_tile_iv, 0, _kslot(0))
                    if k_groups > 1:
                        _issue_k_slice_load_runtime(kv_tile_iv, 1, _kslot(1))
                    for kg in range(k_groups):
                        slot = _kslot(kg)
                        if kg + 2 < k_groups:
                            next_slot = _kslot(kg + 2)
                            if (
                                K_SLICED_LDSSEQ
                                and kg > 0
                                and next_slot == _kslot(kg - 1)
                            ):
                                # CK's LdsSeq can reuse the slice consumed by the
                                # previous kg. Drain LDS reads before overwriting
                                # that slot; the VMEM prefetch still overlaps the
                                # current slice's compute after the partial wait.
                                b.s_waitcnt(lgkmcnt=0)
                                b.s_barrier_bare()
                            _issue_k_slice_load_runtime(kv_tile_iv, kg + 2, next_slot)
                        # Leave one newer slice's VMEM stream in flight whenever
                        # such a slice exists; fully drain for the final slice.
                        if kg + 1 < k_groups:
                            b.s_waitcnt(vmcnt=K_SLICE_CALLS_PER_TILE, lgkmcnt=0)
                        else:
                            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
                        b.s_barrier_bare()
                        if USE_IGLP_OPT:
                            b.sched_barrier(0)
                        for n in range(QK_N_TILES):
                            k_row_t = b.add(b.const_i32(n * 32), lane_col32)
                            acc32 = ST32_n[n]
                            for kk in range(k_steps_per_group):
                                k = kg * k_steps_per_group + kk
                                k_off_t = b.add(
                                    b.const_i32(kk * QK_K_STEP),
                                    b.mul(lane_half32, b.const_i32(TQK_HALF_STRIDE)),
                                )
                                A_k_t = b.smem_load_vN(
                                    K_lds,
                                    b.const_i32(slot),
                                    k_row_t,
                                    k_off_t,
                                    dtype=dtype,
                                    n=TQK_FRAG,
                                )
                                B_q_t = Q32_reg[k]
                                acc32 = _mfma_32x32x8(b, dtype, A_k_t, B_q_t, acc32)
                            ST32_n[n] = acc32
                else:
                    ST32_n = [None] * QK_N_TILES
                    for n in range(QK_N_TILES):
                        acc32 = b.zero_vec_f32(16)
                        # K row becomes the M dimension of the transposed score
                        # tile. K_lds layout is [T, HD], so A is K_lds[row, k].
                        k_row_t = b.add(b.const_i32(n * 32), lane_col32)
                        for k in range(QK_K_ITERS):
                            k_off_t = b.add(
                                b.const_i32(k * QK_K_STEP),
                                b.mul(lane_half32, b.const_i32(TQK_HALF_STRIDE)),
                            )
                            A_k_t = _read_k8_mfma_operand(
                                cur_buf, k_row_t, k_off_t, frag=TQK_FRAG
                            )
                            B_q_t = Q32_reg[k]
                            if USE_MFMA_32X32X8:
                                acc32 = _mfma_32x32x8(b, dtype, A_k_t, B_q_t, acc32)
                            else:
                                acc32 = _mfma_32x32x16(b, dtype, A_k_t, B_q_t, acc32)
                            if USE_QK_PV_SCHED_GROUP_BARRIER:
                                _sched_group_pin_mfma_step(
                                    b,
                                    ds_reads=max(1, TQK_FRAG // 4),
                                    mfma_count=1,
                                    sgid=0,
                                )
                        ST32_n[n] = acc32
                if GROUPED_KV2:
                    # Second score tile for the opt-in grouped online-softmax
                    # experiment. For an odd tile count, safe_tile1 duplicates
                    # tile0 but tile_off_g1 is still beyond max_seq_prefix_len,
                    # so the mask below turns every tile1 probability into zero.
                    b.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    b.sync()
                    ST32_n_g1 = [None] * QK_N_TILES
                    for n in range(QK_N_TILES):
                        acc32 = b.zero_vec_f32(16)
                        k_row_t = b.add(b.const_i32(n * 32), lane_col32)
                        for k in range(QK_K_ITERS):
                            k_off_t = b.add(
                                b.const_i32(k * QK_K_STEP),
                                b.mul(lane_half32, b.const_i32(TQK_HALF_STRIDE)),
                            )
                            A_k_t = _read_k8_mfma_operand(
                                nxt_buf, k_row_t, k_off_t, frag=TQK_FRAG
                            )
                            B_q_t = Q32_reg[k]
                            if USE_MFMA_32X32X8:
                                acc32 = _mfma_32x32x8(b, dtype, A_k_t, B_q_t, acc32)
                            else:
                                acc32 = _mfma_32x32x16(b, dtype, A_k_t, B_q_t, acc32)
                        ST32_n_g1[n] = acc32

                # Transposed softmax scaffold. ST32_n[n][reg] holds
                # S^T[k, q] for one query column (lane_col32) and 16 K
                # positions per lane. The counterpart lane (lane ^ 32)
                # owns the complementary 16 K positions for the same
                # query column. Therefore row-wise softmax over K needs:
                #
                #   1. local max/sum over all regs and all N-tiles
                #   2. one cross-half exchange with lane^32
                #
                # This is the Triton-like reduction shape; it replaces the
                # previous 5-stage per-row 32-lane butterfly.
                st_local_max = neg_inf
                st_scores = {}
                st_groups = [(ST32_n, tile_off)]
                if GROUPED_KV2:
                    st_groups.append((ST32_n_g1, tile_off_g1))
                if TRANSPOSED_MASK_LIMIT and not skip_mask:
                    row_ok = st_row_ok if TRANSPOSED_INVARIANT_HOIST else st_row_ok_iter
                    causal_lim = (
                        st_causal_lim
                        if TRANSPOSED_INVARIANT_HOIST
                        else st_causal_lim_iter
                    )
                    prefix_tail = b.sub(max_seq_prefix_len, b.const_i32(1))
                    valid_tail = b.select(
                        b.cmp_lt(causal_lim, prefix_tail), causal_lim, prefix_tail
                    )
                    st_row_half_base = b.mul(lane_half32, b.const_i32(4))
                    # VALU reduction for the per-element mask (algebraically
                    # identical to ``land(row_ok, col_abs <= valid_tail)``):
                    #   * Fold the per-lane ``row_ok`` into the threshold once
                    #     (row invalid -> threshold = -BIG -> always masked),
                    #     dropping a per-element ``v_and``.
                    #   * Pre-subtract the compile-time ``row_off`` from the
                    #     threshold per reg so the per-element ``col_abs`` add
                    #     folds away: ``col_abs_base + row_off <= valid_tail``
                    #     becomes ``col_abs_base <= valid_tail - row_off``.
                    # Net: per element drops from {v_add, v_cmp, v_and} to a
                    # single {v_cmp}; the row_off rows are 16 constants so the
                    # pre-subtract is 16 v_sub / tile (hoisted out of the n loop).
                    _NEG_BIG = b.const_i32(-(1 << 30))
                    valid_tail_eff = b.select(row_ok, valid_tail, _NEG_BIG)
                    st_mask_thresh = [
                        b.sub(valid_tail_eff, b.const_i32((reg // 4) * 8 + (reg % 4)))
                        for reg in range(16)
                    ]
                else:
                    valid_tail = None
                    st_mask_thresh = None
                    st_row_half_base = None

                # Transposed-32x32 ALiBi/QQ-bias plumbing. ``slope`` is
                # per-lane (depends on ``st_qh`` / ``st_qh_iter`` which is
                # already invariant across (n, reg) in this lane); we reuse
                # the hoisted ``st_alibi_slope`` (or the per-iter
                # ``st_alibi_slope_iter``) so the computation is a single
                # ``v_pk_mul_f32`` chain per (n, reg).
                if USE_ALIBI:
                    if TRANSPOSED_INVARIANT_HOIST:
                        slope_for_lane = st_alibi_slope
                    elif TRANSPOSED_MASK_ONCE:
                        slope_for_lane = st_alibi_slope_iter
                    else:
                        # Fallback: load per-lane slope inline. Same shape
                        # as the hoisted versions above; matches AITER
                        # ``unified_attention.py:317``.
                        q_row_t_alibi = b.add(
                            wave_row_base, _mfma_32x32_c_col(b, lane, 0)
                        )
                        qh_t_alibi = b.add(
                            b.mul(kv_head_idx, b.const_i32(NQK)),
                            b.mod(q_row_t_alibi, b.const_i32(NQK)),
                        )
                        slope_for_lane = b.masked_global_load(
                            alibi_slopes_ptr,
                            qh_t_alibi,
                            b.cmp_lt(qh_t_alibi, b.const_i32(NUM_QH)),
                            b.const_f32(0.0),
                            dtype=F32,
                            align=4,
                        )
                else:
                    slope_for_lane = None

                for group_idx, (st_regs, group_tile_off) in enumerate(st_groups):
                    for n in range(QK_N_TILES):
                        if TRANSPOSED_MASK_LIMIT and not skip_mask:
                            col_abs_base = b.add(
                                b.add(group_tile_off, b.const_i32(n * 32)),
                                st_row_half_base,
                            )
                        else:
                            col_abs_base = None
                        for reg in range(16):
                            if TRANSPOSED_MASK_LIMIT and skip_mask:
                                # Causal full-tile peel: this tile is entirely
                                # within the causal+prefix bound for every row,
                                # so no per-element masking is needed (the select
                                # would be select(true, s, -inf) == s).
                                m_ok = None
                            elif TRANSPOSED_MASK_LIMIT:
                                # col_abs_base + row_off <= valid_tail  <=>
                                # col_abs_base <= (valid_tail_eff - row_off),
                                # with row_ok already folded into the threshold.
                                m_ok = b.cmp_le(col_abs_base, st_mask_thresh[reg])
                            else:
                                k_local = b.add(
                                    b.const_i32(n * 32),
                                    _mfma_32x32_c_row(b, lane, reg),
                                )
                                if TRANSPOSED_INVARIANT_HOIST:
                                    qp_r = st_qp
                                    row_ok = st_row_ok
                                    causal_lim = st_causal_lim
                                elif TRANSPOSED_MASK_ONCE:
                                    qp_r = st_qp_iter
                                    row_ok = st_row_ok_iter
                                    causal_lim = st_causal_lim_iter
                                else:
                                    q_row_t = b.add(
                                        wave_row_base, _mfma_32x32_c_col(b, lane, 0)
                                    )
                                    qp_r = b.add(
                                        qb_start_pos, b.div(q_row_t, b.const_i32(NQK))
                                    )
                                    qh_r = b.add(
                                        b.mul(kv_head_idx, b.const_i32(NQK)),
                                        b.mod(q_row_t, b.const_i32(NQK)),
                                    )
                                    row_ok = b.land(
                                        b.cmp_lt(qp_r, cur_batch_q_len),
                                        b.cmp_lt(qh_r, b.const_i32(NUM_QH)),
                                    )
                                col_abs = b.add(group_tile_off, k_local)
                                if not (
                                    TRANSPOSED_INVARIANT_HOIST or TRANSPOSED_MASK_ONCE
                                ):
                                    causal_lim = b.add(context_len, qp_r)
                                causal_ok = b.cmp_le(col_abs, causal_lim)
                                in_prefix = b.cmp_lt(col_abs, max_seq_prefix_len)
                                m_ok = b.land(b.land(row_ok, causal_ok), in_prefix)
                                if SLIDING_WINDOW > 0:
                                    dist = b.sub(causal_lim, col_abs)
                                    m_ok = b.land(m_ok, b.cmp_lt(dist, sw_const))
                            s_raw = b.vec_extract(st_regs[n], reg)
                            s_scaled = b.fmul(s_raw, qk_scale)
                            if USE_SOFTCAP:
                                s_scaled = b.fmul(
                                    _apply_softcap(b, s_scaled, softcap_p), rcp_ln2
                                )
                            # ``m_ok is None`` => causal full-tile peel: no mask.
                            score = (
                                s_scaled
                                if m_ok is None
                                else b.select(m_ok, s_scaled, neg_inf)
                            )
                            if USE_ALIBI:
                                # ``slope * (key_pos - context_len) * RCP_LN2``;
                                # mirrors AITER ``unified_attention.py:317``.
                                pos_off = b.sub(col_abs, context_len)
                                pos_f = b.sitofp_f32(pos_off)
                                add_term = b.fmul(
                                    b.fmul(slope_for_lane, pos_f), rcp_ln2
                                )
                                score = b.fadd(score, add_term)
                            if USE_QQ_BIAS:
                                # ``qq_bias[qp_r, key_pos - context_len]``;
                                # mirrors AITER ``unified_attention.py:319-330``.
                                krp = b.sub(col_abs, context_len)
                                krp_ok = b.land(
                                    b.cmp_ge(krp, b.const_i32(0)),
                                    b.cmp_lt(krp, qq_bias_stride0_p),
                                )
                                qq_ok = b.land(row_ok, krp_ok)
                                qp_safe = b.select(row_ok, qp_r, b.const_i32(0))
                                qq_idx = b.add(b.mul(qp_safe, qq_bias_stride0_p), krp)
                                qq_v = b.masked_global_load(
                                    qq_bias_ptr,
                                    qq_idx,
                                    qq_ok,
                                    b.const_f32(0.0),
                                    dtype=F32,
                                    align=4,
                                )
                                score = b.fadd(score, b.fmul(qq_v, rcp_ln2))
                            st_scores[(group_idx, n, reg)] = score
                            st_local_max = b.fmax(st_local_max, score)
                st_remote_max = b.warp_shuffle_xor(st_local_max, 32)
                st_tile_max = b.fmax(st_local_max, st_remote_max)
                st_m_raw = b.fmax(m_vals[0], st_tile_max)
                st_ok = b.fcmp("ogt", st_m_raw, neg_inf)
                st_m_new = b.select(st_ok, st_m_raw, zero_f)
                cfvst_payload = (
                    _cfvst_load_v_regs(kv_tile_iv)
                    if TRANSPOSED_V_STORE and _CFV_STORE_SPLIT
                    else None
                )
                PT32_groups = [
                    [[None] * 16 for _ in range(QK_N_TILES)]
                    for _ in range(len(st_groups))
                ]
                st_l_local = zero_f
                for group_idx in range(len(st_groups)):
                    for n in range(QK_N_TILES):
                        for reg in range(16):
                            p_t = b.exp2(
                                b.fsub(st_scores[(group_idx, n, reg)], st_m_new)
                            )
                            PT32_groups[group_idx][n][reg] = p_t
                            st_l_local = b.fadd(st_l_local, p_t)
                PT32_n = PT32_groups[0]
                if GROUPED_KV2:
                    PT32_n_g1 = PT32_groups[1]
                st_l_remote = b.warp_shuffle_xor(st_l_local, 32)
                st_l_sum = b.fadd(st_l_local, st_l_remote)
                if TRANSPOSED_V_STORE and _CFV_STORE_SPLIT:
                    _cfvst_store_v_regs(cfvst_payload)
                # The transposed orientation has one softmax state per query
                # column/lane, shared across all 16 output-dimension regs.
                # ``m_new`` / ``l_local`` are intentionally broadcast across
                # ``REGS_PER_LANE`` so the downstream alpha/L-update code
                # (shared with the non-transposed path) keeps producing one
                # state per query column. The alpha = exp2(m_old - m_new) and
                # full ``l_new = alpha * l_old + l_sum`` updates are applied
                # by that shared downstream block, not here.
                m_new = [st_m_new for _ in range(SOFTMAX_STATE_SLOTS)]
                l_local = [st_l_sum for _ in range(SOFTMAX_STATE_SLOTS)]
                # In the transposed orientation we already have masked /
                # scaled / softcapped scores in ``st_scores`` and softmax
                # probabilities in ``PT32_n``. We must NOT recompute
                # ``S32_n`` and run the non-transposed mask + softmax
                # block below: it would (1) duplicate compute, and (2)
                # overwrite the transposed ``m_new`` / ``l_local`` /
                # ``masked`` with values keyed on the non-transposed
                # (row=q, col=k) layout. The downstream alpha + l + PV
                # code then expects the transposed layout, so leaving the
                # non-transposed softmax in would silently scramble alpha
                # and the running L per query column.
                S32_n = None
            else:
                # Non-transposed 32x32 QK (S = Q @ K^T): ``S32_n[n] =
                # vec_f32(16)``, one 32x32 C tile per warp.  Downstream
                # softmax/PV consume this via the non-transposed mask block.
                #
                # B (K^T) per lane:
                #   32x32x8 : <4 x half>, col = n*32 + lane%32,
                #             K = k*8  + (lane//32)*4 + [0..3]
                #   32x32x16: <8 x half>, K = k*16 + (lane//32)*8 + [0..7]
                B32_FRAG = 4 if USE_MFMA_32X32X8 else 8
                B32_HALF_STRIDE = 4 if USE_MFMA_32X32X8 else 8
                S32_n = [None] * QK_N_TILES
                for n in range(QK_N_TILES):
                    acc32 = b.zero_vec_f32(16)
                    k_row32 = b.add(b.const_i32(n * 32), lane_col32)
                    for k in range(QK_K_ITERS):
                        kc_off32 = b.add(
                            b.const_i32(k * QK_K_STEP),
                            b.mul(lane_half, b.const_i32(B32_HALF_STRIDE)),
                        )
                        B32_v = b.smem_load_vN(
                            K_lds, cur_buf, k_row32, kc_off32, dtype=dtype, n=B32_FRAG
                        )
                        if USE_MFMA_32X32X8:
                            acc32 = _mfma_32x32x8(b, dtype, Q32_reg[k], B32_v, acc32)
                        else:
                            acc32 = _mfma_32x32x16(b, dtype, Q32_reg[k], B32_v, acc32)
                    S32_n[n] = acc32
        else:
            S_n = [[None] * QK_N_TILES for _ in range(M_ATOMS_PER_WARP)]
            for n in range(QK_N_TILES):
                acc_per_atom = [b.zero_vec_f32(4) for _ in range(M_ATOMS_PER_WARP)]
                for k in range(QK_K_ITERS):
                    # 16x16x16 B (K^T) operand per lane: col = n*16 + lane%16,
                    # K = k*16 + lane_rg*4 + 0..3 (<4 x dtype>). K-step is 16 on
                    # gfx942 (no wide-K atom), so the per-lane column base is
                    # lane_rg*4 and each load is <4 x dtype>.
                    kc_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                    k_row = b.add(b.const_i32(n * 16), lane_col)
                    B_v = b.smem_load_vN(
                        K_lds, cur_buf, k_row, kc_off, dtype=dtype, n=4
                    )
                    for atom in range(M_ATOMS_PER_WARP):
                        acc_per_atom[atom] = _mfma_16x16x16(
                            b, dtype, A_kits[atom][k], B_v, acc_per_atom[atom]
                        )
                for atom in range(M_ATOMS_PER_WARP):
                    S_n[atom][n] = acc_per_atom[atom]

        # Now that QK no longer needs VMEM, start current V first and next K
        # second. This ordering is what lets the partial wait before PV leave
        # only next K pending.
        if K_SINGLE_BUF:
            # OCCUPANCY lever: K is single-buffered, so next-K reuses the SAME
            # slot QK just read. The next-K async DMA would race the tail of
            # this iteration's QK K-reads (lgkmcnt accounting on
            # raw_ptr_buffer_load_lds) and silently corrupt the tile. Drain ALL
            # outstanding LDS reads + barrier BEFORE issuing the next-K DMA into
            # the shared slot; the DMA then overlaps softmax + PV + epilogue, and
            # the next iter's start-of-loop ``s_waitcnt(vmcnt=0,lgkmcnt=0); sync``
            # waits for it before QK. V is also single-buffered: issue it after
            # the same drain so its store does not race the K-read drain.
            b.s_waitcnt(lgkmcnt=0)
            b.sync()
            if TRANSPOSED_V_STORE and _CFV_STORE_SPLIT:
                # Store-path cfv split its V load/store across the transposed
                # softmax work above, so only next-K remains to issue here.
                pass
            elif not EARLY_V_SCHEDULE:
                # When EARLY_V is set, V was already issued at iter start; do not
                # re-issue (it would double-load and race the in-flight V).
                _issue_v(kv_tile_iv, cur_buf)
            _issue_k(safe_next_tile, nxt_buf)
        elif GROUPED_KV2:
            _issue_v(kv_tile_iv, cur_buf)
            # Both K buffers have been consumed by QK0/QK1. Refill cur_buf
            # with the next group's first K tile and carry cur_buf forward.
            _issue_k(safe_next_tile, cur_buf)
        elif EARLY_V_SCHEDULE:
            _issue_k(safe_next_tile, nxt_buf)
        elif TRANSPOSED_V_STORE and _CFV_STORE_SPLIT:
            # Store-path cfv has already published V_lds during the transposed
            # softmax block. Keep the next-K prefetch schedule unchanged.
            _issue_k(safe_next_tile, nxt_buf)
        else:
            _issue_v(kv_tile_iv, cur_buf)
            _issue_k(safe_next_tile, nxt_buf)

        # ---- mask / scale / softcap / alibi / qq-bias ----
        # ALiBi and QQ-bias mirror Triton's apply-before-mask-result semantics:
        # we fold them into the unmasked S, then the select-with-(-inf) below
        # zeroes them out for invalid cells (finite + (-inf) = (-inf) in IEEE
        # for the mask path so result is identical to Triton's
        # "S = where(mask, S, -inf); S += bias" formulation).
        # All per-reg loops iterate over REGS_PER_LANE = 4 (block_m_per_warp=16)
        # or 8 (block_m_per_warp=32); reg `r` maps to in-warp-row
        # `(r // 4) * 16 + lane_rg * 4 + (r % 4)` and to the QK acc atom
        # ``S_n[r // 4][n]``.
        # ``alibi_per_row``, ``qp_r``, ``qh_r``, ``row_ok``, ``causal_lim``
        # are all CTA-constants -- pulled in from the pre-loop hoist above.
        alibi_per_row = hoist_alibi if USE_ALIBI else None
        if USE_MFMA_32X32 and TRANSPOSED_QK_32X32:
            # Transposed orientation: ``masked`` / ``m_new`` / ``l_local``
            # were already produced by the in-place transposed softmax
            # above. The downstream alpha + l + PV update uses them
            # directly. ``P_lds`` is intentionally not written because the
            # transposed PV consumes ``PT32_n`` from registers.
            pass
        elif USE_MFMA_32X32:
            masked = {}
            for reg in range(REGS_PER_LANE):
                qp_r = hoist_qp_r[reg]
                row_ok = hoist_row_ok[reg]
                causal_lim = hoist_causal_lim[reg]
                for n in range(QK_N_TILES):
                    col_abs = b.add(tile_off, _mfma_32x32_c_col(b, lane, n))
                    causal_ok = b.cmp_le(col_abs, causal_lim)
                    in_prefix = b.cmp_lt(col_abs, max_seq_prefix_len)
                    m_ok = b.land(b.land(row_ok, causal_ok), in_prefix)
                    if SLIDING_WINDOW > 0:
                        dist = b.sub(causal_lim, col_abs)
                        m_ok = b.land(m_ok, b.cmp_lt(dist, sw_const))
                    s_raw = b.vec_extract(S32_n[n], reg)
                    s_scaled = b.fmul(s_raw, qk_scale)
                    if USE_SOFTCAP:
                        s_scaled = b.fmul(
                            _apply_softcap(b, s_scaled, softcap_p), rcp_ln2
                        )
                    score = b.select(m_ok, s_scaled, neg_inf)
                    if USE_ALIBI:
                        pos_off = b.sub(col_abs, context_len)
                        pos_f = b.sitofp_f32(pos_off)
                        add_term = b.fmul(b.fmul(alibi_per_row[reg], pos_f), rcp_ln2)
                        score = b.fadd(score, add_term)
                    if USE_QQ_BIAS:
                        krp = b.sub(col_abs, context_len)
                        krp_ok = b.land(
                            b.cmp_ge(krp, b.const_i32(0)),
                            b.cmp_lt(krp, qq_bias_stride0_p),
                        )
                        qq_ok = b.land(row_ok, krp_ok)
                        qp_safe = b.select(row_ok, qp_r, b.const_i32(0))
                        qq_idx = b.add(b.mul(qp_safe, qq_bias_stride0_p), krp)
                        qq_v = b.masked_global_load(
                            qq_bias_ptr,
                            qq_idx,
                            qq_ok,
                            b.const_f32(0.0),
                            dtype=F32,
                            align=4,
                        )
                        score = b.fadd(score, b.fmul(qq_v, rcp_ln2))
                    masked[(n, reg)] = score

            m_new = []
            s_local = {}
            for reg in range(REGS_PER_LANE):
                local_max = neg_inf
                for n in range(QK_N_TILES):
                    v = masked[(n, reg)]
                    s_local[(reg, n)] = v
                    local_max = b.fmax(local_max, v)
                tile_max = _warp_xor_reduce_max_32lane(b, local_max)
                full_max_raw = b.fmax(m_vals[reg], tile_max)
                ok = b.fcmp("ogt", full_max_raw, neg_inf)
                m_new.append(b.select(ok, full_max_raw, zero_f))

            l_local = []
            for reg in range(REGS_PER_LANE):
                row = hoist_row[reg]
                sum_p = zero_f
                for n in range(QK_N_TILES):
                    p = b.exp2(b.fsub(s_local[(reg, n)], m_new[reg]))
                    col = _mfma_32x32_c_col(b, lane, n)
                    b.smem_store_vN(P_lds, [row, col], b.cast_f32_to(p, dtype), 1)
                    sum_p = b.fadd(sum_p, p)
                l_local.append(_warp_xor_reduce_sum_32lane(b, sum_p))

        else:
            masked = {}
            for reg in range(REGS_PER_LANE):
                atom = reg // 4
                in_atom = reg % 4
                qp_r = hoist_qp_r[reg]
                row_ok = hoist_row_ok[reg]
                causal_lim = hoist_causal_lim[reg]
                for n in range(QK_N_TILES):
                    col_abs = b.add(
                        b.add(tile_off, b.mul(b.const_i32(n), b.const_i32(16))),
                        lane_col,
                    )
                    causal_ok = b.cmp_le(col_abs, causal_lim)
                    in_prefix = b.cmp_lt(col_abs, max_seq_prefix_len)
                    m_ok = b.land(b.land(row_ok, causal_ok), in_prefix)
                    if SLIDING_WINDOW > 0:
                        dist = b.sub(causal_lim, col_abs)
                        m_ok = b.land(m_ok, b.cmp_lt(dist, sw_const))
                    s_raw = b.vec_extract(S_n[atom][n], in_atom)
                    s_scaled = b.fmul(s_raw, qk_scale)
                    if USE_SOFTCAP:
                        s_scaled = b.fmul(
                            _apply_softcap(b, s_scaled, softcap_p), rcp_ln2
                        )
                    score = b.select(m_ok, s_scaled, neg_inf)
                    if USE_ALIBI:
                        # Triton order: mask first, then add ALiBi. For invalid
                        # cells this is `-inf + finite == -inf`, avoiding any
                        # pre-mask finite arithmetic from leaking into reductions.
                        pos_off = b.sub(col_abs, context_len)
                        pos_f = b.sitofp_f32(pos_off)
                        add_term = b.fmul(b.fmul(alibi_per_row[reg], pos_f), rcp_ln2)
                        score = b.fadd(score, add_term)
                    if USE_QQ_BIAS:
                        # qq_bias[qp_r, key_rel_pos] with key_rel_pos = col - ctx.
                        # Valid range 0 <= key_rel_pos < qq_bias_stride_0 AND
                        # qp_r is a non-padding query position. The padding-row
                        # guard is required because qb_start_pos can exceed
                        # cur_batch_query_len for the last Q-block of a sequence;
                        # without it `qq_bias_ptr + qp_r*stride0 + krp` can run
                        # off the end of the tensor for tail blocks.
                        krp = b.sub(col_abs, context_len)
                        krp_ok = b.land(
                            b.cmp_ge(krp, b.const_i32(0)),
                            b.cmp_lt(krp, qq_bias_stride0_p),
                        )
                        qq_ok = b.land(row_ok, krp_ok)
                        qp_safe = b.select(row_ok, qp_r, b.const_i32(0))
                        qq_idx = b.add(b.mul(qp_safe, qq_bias_stride0_p), krp)
                        qq_v = b.masked_global_load(
                            qq_bias_ptr,
                            qq_idx,
                            qq_ok,
                            b.const_f32(0.0),
                            dtype=F32,
                            align=4,
                        )
                        score = b.fadd(score, b.fmul(qq_v, rcp_ln2))
                    masked[(n, reg)] = score

            # ---- per-row max via cross-lane butterfly ----
            # Each lane has REGS_PER_LANE floats (one per row in its row-group(s)),
            # repeated for every N-tile. Local lane-max across N-tiles, then 4-stage
            # XOR butterfly across the 16 lanes in the row-group.
            m_new = []
            s_local = {}  # (reg, n) -> the lane's masked score (still owned per-lane)
            for reg in range(REGS_PER_LANE):
                local_max = neg_inf
                for n in range(QK_N_TILES):
                    v = masked[(n, reg)]
                    s_local[(reg, n)] = v
                    local_max = b.fmax(local_max, v)
                tile_max = _warp_xor_reduce_max(b, local_max)
                # Online softmax update (FlashAttention/Triton): see docstring.
                full_max_raw = b.fmax(m_vals[reg], tile_max)
                ok = b.fcmp("ogt", full_max_raw, neg_inf)
                m_new.append(b.select(ok, full_max_raw, zero_f))

            # ---- compute P = exp2(S - m_new) and l_local = sum(P) per row ----
            # P_lds[row, col] = exp2(S[row, col] - m_new[row]). Each warp publishes
            # its own BLOCK_M_PER_WARP-row slice (16 or 32 rows). Row coords come
            # from the pre-loop hoist.
            p_regs_f32 = [[None] * QK_N_TILES for _ in range(REGS_PER_LANE)]
            l_local = []
            for reg in range(REGS_PER_LANE):
                sum_p = zero_f
                for n in range(QK_N_TILES):
                    p = b.exp2(b.fsub(s_local[(reg, n)], m_new[reg]))
                    p_regs_f32[reg][n] = p
                    if not REGISTER_PV:
                        row = hoist_row[reg]
                        col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
                        if PV_FP8_MFMA:
                            # FlyDSL pa_decode_fp8 pattern: quantise probabilities
                            # into fp8 with a fixed 240x scale, then fold the
                            # reciprocal plus v_scale into the PV result.
                            p_q = b.cvt_f32_to_fp8(b.fmul(p, b.const_f32(240.0)))
                            b.smem_store_vN(P_lds, [row, col], p_q, 1)
                        else:
                            b.smem_store_vN(
                                P_lds, [row, col], b.cast_f32_to(p, dtype), 1
                            )
                    sum_p = b.fadd(sum_p, p)
                l_local.append(_warp_xor_reduce_sum(b, sum_p))

        # alpha and L update (still per-lane registers; matches FA-2 paper)
        alpha_regs = [
            b.exp2(b.fsub(m_vals[r], m_new[r])) for r in range(SOFTMAX_STATE_SLOTS)
        ]
        new_l_vals = [
            b.fadd(b.fmul(l_vals[r], alpha_regs[r]), l_local[r])
            for r in range(SOFTMAX_STATE_SLOTS)
        ]
        if GROUPED_KV2:
            # Simplicity-first prototype: wait for V0 and the next group's K
            # prefetch together. This proves the grouped softmax/PV dataflow;
            # a later performance pass can restore the one-tile partial wait.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sync()
        elif KV_FP8:
            # FP8 sync loader has no in-flight async work (the sync loader
            # completes before returning). The full LDS-visibility sync
            # below ensures PV's V_lds reads see the just-loaded V bytes.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sync()
        elif TRANSPOSED_V_STORE and _CFV_STORE_SPLIT and not K_SLICED_ACTIVE:
            # Split cfvst has already loaded/stored current V and then issued
            # next K. Drain only the older current-V VMEM/LDS work and leave the
            # just-issued next-K async DMA in flight across PV, matching CK Tile's
            # partial wait + bare barrier pattern.
            b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
            b.s_barrier_bare()
        elif TRANSPOSED_V_STORE and _CFV_STORE_SPLIT:
            # Sliced-K v1 does not yet prefetch next K after softmax, so there is
            # no younger VMEM work to preserve. Fully publish current V.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sync()
        elif TRANSPOSED_V:
            # Transposed-V sync loader: ``global_load`` (vmcnt) followed by
            # ``smem_store`` (lgkmcnt). The next-K async DMA is still pending on
            # its own counters, but the partial-count trick below assumes an
            # async-V deposit; the transpose fill uses a different instruction
            # mix, so drain BOTH fully + barrier to publish the transposed V_lds
            # before any PV A-operand read.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.sync()
        else:
            # Wait for current V while leaving next K pending. Current V was
            # issued before next K, so `kv_calls_per_tile` pending operations are
            # exactly the next-K stream. Apply the same idea to lgkmcnt so we do
            # not wait for the next-K LDS writes before PV.
            b.s_waitcnt(vmcnt=kv_calls_per_tile, lgkmcnt=kv_calls_per_tile)
            b.sync()

        # ---- acc *= alpha, acc += P @ V ----
        # For ``M_ATOMS_PER_WARP=2`` we stack two MFMAs in M per (n, k) atom:
        # both atoms share the same B (V) operand, but read different P rows
        # (atom 0: rows wave_row_base..wave_row_base+15; atom 1: rows
        # wave_row_base+16..wave_row_base+31). Each atom has its own
        # accumulator + per-reg alpha.
        #
        # NOTE: an ``s_setprio(1)`` wrap around this cluster was tested
        # (a warp-specialized design),
        # which raises this wave's instruction-issue priority so the SIMD
        # scheduler favours us through the MFMA-dominated PV section. On
        # multi-seq bf16 prefill (n>=100 q=1000) this caused CATASTROPHIC
        # regressions (up to +1282% / 12.8x slower) — priority inversion
        # starved the other waves competing for the same SIMD. That reference
        # gets away with this on its hand-unrolled single-batch kernel
        # because it tightly controls inter-wave interleaving via the
        # cluster pattern + stagger barriers; our DSL-lowered loop has
        # different scheduling pressure. Disabled.
        new_acc = [None] * (ACC_N_TILES * ACC_M_ATOMS)
        if USE_MFMA_32X32:
            if TRANSPOSED_QK_32X32:
                # Transposed PV: O^T = V^T @ P^T.
                #
                # A operand = V^T[M=d, K=kv-token]
                # B operand = P^T[K=kv-token, N=query]
                # C output = O^T[d, query]
                #
                # The P^T register layout from ST32 is cheap to consume:
                # for each needed K row, the value is either local
                # PT32_n[p_tile][reg] or the same register in lane^32.
                v_buf = b.const_i32(0)
                use_hi = b.cmp_eq(lane_half32, b.const_i32(1))

                def _apply_transposed_pv_regs(acc32: Value, n: int, p_regs) -> Value:
                    v_dim32 = b.add(b.const_i32(n * 32), lane_col32)
                    if USE_MFMA_32X32X8:
                        # gfx942-legal K=8 transposed PV: O^T = V^T @ P^T,
                        # P^T consumed DIRECTLY from registers (NO P_lds).
                        #   A = V^T[M=d, K=kv-token] per lane (<4 x half>):
                        #     M=d  = n*32 + lane%32 (== v_dim32)
                        #     K    = k*8 + (lane/32)*4 + [0..3]
                        #     V_lds is row-major [T, HD]; 4 distinct strided
                        #     single-element reads (naive V -- Stream B's job
                        #     to make conflict-free).
                        #   B = P^T[K=kv-token, N=query] per lane (<4 x half>):
                        #     K    = k*8 + (lane/32)*4 + [0..3]
                        #     The probability for K-row r lives in
                        #     ``p_regs[r//32][reg]`` of the lane whose 32-half
                        #     owns row (r%32); cross-half rows are fetched with
                        #     a single ``lane ^ 32`` exchange (the cheap reshape
                        #     that beats the plain orientation's serializing
                        #     C->A cross-lane reduction).
                        for k in range(T // 8):
                            # A = V^T[M=d, K=token], 4 tokens per lane:
                            #   token = k*8 + (lane/32)*4 + [0..3]
                            #   head-dim row = v_dim32 (n*32 + lane%32)
                            if TRANSPOSED_V:
                                # CK technique #1: V is stored K-contiguous in
                                # ``V_lds[HD, T+pad]``, so those 4 tokens are
                                # CONTIGUOUS along the inner (token) axis at the
                                # lane's head-dim row ``v_dim32`` -> ONE wide
                                # ``ds_read_b64`` (<4 x half>) replaces the 4
                                # strided ``ds_read_u16``. Bank-spread because
                                # adjacent lanes (varying ``v_dim32``) land in
                                # ``V_T_PAD``-rotated rows. The token base is the
                                # SAME ``k*8 + lane_half32*4`` the naive feed used,
                                # and the n=4 contiguous read covers tokens
                                # base..base+3 -> element-for-element identical to
                                # the naive A operand (same MFMA K mapping).
                                tok_base = b.add(
                                    b.const_i32(k * 8),
                                    b.mul(lane_half32, b.const_i32(4)),
                                )
                                if _CFV_SCALAR_READ:
                                    # ISOLATION: read the SAME [HD,T+pad] cfv
                                    # layout but via 4 scalar n=1 reads (same
                                    # values/order as the n=4 wide read). If this
                                    # is CORRECT while n=4 is WRONG, the bug is the
                                    # n=4 ds_read_b64 lowering over the padded
                                    # row; if it still fails, it's the
                                    # layout/MFMA mapping, not the read width.
                                    _av = []
                                    for _kk in range(4):
                                        _t = b.add(tok_base, b.const_i32(_kk))
                                        _v1 = _v_t_load(v_dim32, _t, n=1)
                                        _av.append(b.vec_extract(_v1, 0))
                                    A_v_t = b.vec_pack(_av, dtype)
                                else:
                                    A_v_t = _v_t_load(v_dim32, tok_base, n=4)
                            else:
                                a_v_elems = []
                                for kk in range(4):
                                    v_row = b.add(
                                        b.const_i32(k * 8 + kk),
                                        b.mul(lane_half32, b.const_i32(4)),
                                    )
                                    v1 = b.smem_load_vN(
                                        V_lds, v_buf, v_row, v_dim32, dtype=dtype, n=1
                                    )
                                    a_v_elems.append(b.vec_extract(v1, 0))
                                A_v_t = b.vec_pack(a_v_elems, dtype)
                            b_p_elems = []
                            for kk in range(4):
                                # Low half wants K-row k0; high half wants k1.
                                k0 = k * 8 + kk
                                k1 = k * 8 + 4 + kk
                                p_tile0 = k0 // 32
                                p_tile1 = k1 // 32
                                row0 = k0 % 32
                                row1 = k1 % 32
                                owner_half0 = (row0 % 8) // 4
                                owner_half1 = (row1 % 8) // 4
                                reg0 = (row0 // 8) * 4 + (row0 % 4)
                                reg1 = (row1 // 8) * 4 + (row1 % 4)
                                p0 = p_regs[p_tile0][reg0]
                                p1 = p_regs[p_tile1][reg1]
                                if owner_half0 == 1:
                                    p0 = b.warp_shuffle_xor(p0, 32)
                                if owner_half1 == 0:
                                    p1 = b.warp_shuffle_xor(p1, 32)
                                p_val = b.select(use_hi, p1, p0)
                                b_p_elems.append(b.cast_f32_to(p_val, dtype))
                            B_p_t = b.vec_pack(b_p_elems, dtype)
                            acc32 = _mfma_32x32x8(b, dtype, A_v_t, B_p_t, acc32)
                            if USE_QK_PV_SCHED_GROUP_BARRIER:
                                _sched_group_pin_mfma_step(
                                    b, ds_reads=4, mfma_count=1, sgid=1
                                )
                        return acc32
                    for k in range(T // 16):
                        if TRANSPOSED_HALF_LOCAL_PV:
                            # Half-local K orientation. Each 32-lane half
                            # consumes only the P rows already local to that
                            # half: half0 -> {0..3, 8..11}, half1 ->
                            # {4..7, 12..15}. Use matching half-local
                            # transpose LDS reads for V so V and P share the
                            # same permuted K order.
                            A_v_t = pv32_v_load_paired(
                                b,
                                V_lds=V_lds,
                                v_buf=v_buf,
                                n=n,
                                k=k,
                                lane_half32=lane_half32,
                                lane_col32=lane_col32,
                                dtype=dtype,
                            )
                            b_p_elems = []
                            for kk in range(8):
                                local_in_group = kk % 4
                                band = kk // 4
                                p_tile = (k * 16 + band * 8 + local_in_group) // 32
                                row_static = (k * 16 + band * 8 + local_in_group) % 32
                                preg = (row_static // 8) * 4 + (row_static % 4)
                                b_p_elems.append(
                                    b.cast_f32_to(p_regs[p_tile][preg], dtype)
                                )
                            B_p_t = b.vec_pack(b_p_elems, dtype)
                            acc32 = _mfma_32x32x16(b, dtype, A_v_t, B_p_t, acc32)
                        else:
                            # Issue all 8 V scalar loads first so the LDS port
                            # can pipeline them back-to-back (the compiler is
                            # free to interleave the address compute, but the
                            # data dependencies don't force serialisation).
                            a_v_elems = []
                            for kk in range(8):
                                k_static = k * 16 + kk
                                v_row = b.add(
                                    b.const_i32(k_static),
                                    b.mul(lane_half32, b.const_i32(8)),
                                )
                                v1 = b.smem_load_vN(
                                    V_lds, v_buf, v_row, v_dim32, dtype=dtype, n=1
                                )
                                a_v_elems.append(b.vec_extract(v1, 0))
                            # Then assemble the P operand. Each kk picks (k0,
                            # k1) and may xor to fetch the cross-half register.
                            # All xors are independent and can issue in
                            # parallel with the V loads.
                            b_p_elems = []
                            for kk in range(8):
                                k_static = k * 16 + kk
                                k0 = k_static
                                k1 = k_static + 8
                                p_tile0 = k0 // 32
                                p_tile1 = k1 // 32
                                row0 = k0 % 32
                                row1 = k1 % 32
                                owner_half0 = (row0 % 8) // 4
                                owner_half1 = (row1 % 8) // 4
                                reg0 = (row0 // 8) * 4 + (row0 % 4)
                                reg1 = (row1 // 8) * 4 + (row1 % 4)
                                p0 = p_regs[p_tile0][reg0]
                                p1 = p_regs[p_tile1][reg1]
                                # PT32 registers store P^T[k, q] in lanes whose
                                # column is the query row q. The K row only
                                # selects which 32-lane half/register owns the
                                # value; it does NOT change the source column.
                                if owner_half0 == 1:
                                    p0 = b.warp_shuffle_xor(p0, 32)
                                if owner_half1 == 0:
                                    p1 = b.warp_shuffle_xor(p1, 32)
                                p_val = b.select(use_hi, p1, p0)
                                b_p_elems.append(b.cast_f32_to(p_val, dtype))
                            A_v_t = b.vec_pack(a_v_elems, dtype)
                            B_p_t = b.vec_pack(b_p_elems, dtype)
                            acc32 = _mfma_32x32x16(b, dtype, A_v_t, B_p_t, acc32)
                    return acc32

                for n in range(ACC_N_TILES):
                    scaled = []
                    old_acc = _acc_get(n, 0)
                    alpha_t = alpha_regs[0]
                    for reg in range(REGS_PER_LANE):
                        e = b.vec_extract(old_acc, reg)
                        scaled.append(
                            b.fmul(
                                e,
                                alpha_t if TRANSPOSED_SCALAR_STATE else alpha_regs[reg],
                            )
                        )
                    acc32 = b.vec_pack(scaled, F32)
                    acc32 = _apply_transposed_pv_regs(acc32, n, PT32_n)
                    new_acc[n] = acc32
                if GROUPED_KV2:
                    # V_lds is intentionally still single-buffered. Finish the
                    # tile0 LDS reads before overwriting it with tile1's V, then
                    # accumulate tile1's PV into the already scaled block acc.
                    b.s_waitcnt(lgkmcnt=0)
                    b.sync()
                    _issue_v(safe_tile1, nxt_buf)
                    b.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    b.sync()
                    for n in range(ACC_N_TILES):
                        new_acc[n] = _apply_transposed_pv_regs(new_acc[n], n, PT32_n_g1)
            elif USE_MFMA_32X32X8:
                # gfx942 32x32x8 PV (O = P @ V), P_lds bridge + NAIVE strided V.
                #
                # A = P[M,K] per lane (<4 x half>):
                #   row = wave_row_base + lane%32
                #   k   = k_iter*8 + (lane/32)*4 + [0..3]
                #   P_lds is row-major [BLOCK_M, T] -> one contiguous <4 x half>.
                # B = V[K,N] per lane (<4 x half>):
                #   col = n_tile*32 + lane%32
                #   k   = k_iter*8 + (lane/32)*4 + [0..3]
                #   V_lds is row-major [T, HD]. Each of the 4 K rows is a
                #   DISTINCT V_lds row at the SAME column -> 4 strided single-
                #   element reads (``_v_load1``). This reproduces the 32x32x8 B
                #   distribution without a transpose read; it carries today's
                #   bank conflicts (a conflict-free V feed is Stream B's job).
                v_buf = b.const_i32(0)
                p_row32 = b.add(wave_row_base, lane_col32)
                for n in range(ACC_N_TILES):
                    scaled = []
                    old_acc = _acc_get(n, 0)
                    for reg in range(REGS_PER_LANE):
                        e = b.vec_extract(old_acc, reg)
                        scaled.append(b.fmul(e, alpha_regs[reg]))
                    acc32 = b.vec_pack(scaled, F32)
                    v_col32 = b.add(b.const_i32(n * 32), lane_col32)
                    for k in range(PV_K_ITERS):  # T // 8
                        p_off32 = b.add(
                            b.const_i32(k * 8), b.mul(lane_half32, b.const_i32(4))
                        )
                        A_p32 = b.smem_load_vN(
                            P_lds, p_row32, p_off32, dtype=dtype, n=4
                        )
                        # Naive strided V B-operand: 4 distinct K rows.
                        k_row_base = b.add(
                            b.const_i32(k * 8), b.mul(lane_half32, b.const_i32(4))
                        )
                        B_v32 = b.zero_vec(dtype, 4)
                        for j in range(4):
                            v_row = b.add(k_row_base, b.const_i32(j))
                            elem = b.vec_extract(_v_load1(v_buf, v_row, v_col32), 0)
                            B_v32 = b.vec_insert(B_v32, elem, j)
                        acc32 = _mfma_32x32x8(b, dtype, A_p32, B_v32, acc32)
                    new_acc[n] = acc32
            else:
                # Transitional 32x32x16 PV consumer (gfx950-only ds_read_tr16
                # transpose read; unreachable on gfx942 -- use_mfma_32x32 is
                # rejected in the spec). Kept for structural parity with gfx950.
                #
                # PV32 operand layouts:
                #   A = P[M,K] per lane:
                #       row = wave_row_base + lane%32
                #       k   = k_iter*16 + (lane/32)*8 + [0..7]
                #   B = V[K,N] per lane:
                #       k   = k_iter*16 + (lane/32)*8 + [0..7]
                #       col = n_tile*32 + lane%32
                v_buf = b.const_i32(0)
                for n in range(ACC_N_TILES):
                    scaled = []
                    old_acc = _acc_get(n, 0)
                    for reg in range(REGS_PER_LANE):
                        e = b.vec_extract(old_acc, reg)
                        scaled.append(b.fmul(e, alpha_regs[reg]))
                    acc32 = b.vec_pack(scaled, F32)
                    for k in range(T // 16):
                        p_off32 = b.add(
                            b.const_i32(k * 16), b.mul(lane_half32, b.const_i32(8))
                        )
                        p_row32 = b.add(wave_row_base, lane_col32)
                        A_p32 = b.smem_load_vN(
                            P_lds, p_row32, p_off32, dtype=dtype, n=8
                        )
                        col_group16 = b.mul(
                            b.div(lane_col32, b.const_i32(16)), b.const_i32(16)
                        )
                        tr_col32 = b.add(
                            col_group16,
                            b.mul(b.mod(lane_col32, b.const_i32(4)), b.const_i32(4)),
                        )
                        tr_row_base32 = b.add(
                            b.add(
                                b.const_i32(k * 16),
                                b.mul(lane_half32, b.const_i32(8)),
                            ),
                            b.mod(b.div(lane_col32, b.const_i32(4)), b.const_i32(4)),
                        )
                        B32_r0 = b.ds_read_tr16_b64(
                            V_lds,
                            v_buf,
                            tr_row_base32,
                            b.add(b.const_i32(n * 32), tr_col32),
                            dtype=dtype,
                        )
                        B32_r1 = b.ds_read_tr16_b64(
                            V_lds,
                            v_buf,
                            b.add(tr_row_base32, b.const_i32(4)),
                            b.add(b.const_i32(n * 32), tr_col32),
                            dtype=dtype,
                        )
                        B_v32 = b.vec_concat(B32_r0, B32_r1)
                        acc32 = _mfma_32x32x16(b, dtype, A_p32, B_v32, acc32)
                    new_acc[n] = acc32
        for n in range(0 if USE_MFMA_32X32 else PV_N_TILES):
            # Per-atom: scale the inherited acc by per-row alpha, then add P @ V.
            acc_per_atom: list[Value] = []
            for atom in range(M_ATOMS_PER_WARP):
                scaled_comps = []
                for in_atom in range(4):
                    reg = atom * 4 + in_atom
                    e = b.vec_extract(_acc_get(n, atom), in_atom)
                    scaled_comps.append(b.fmul(e, alpha_regs[reg]))
                acc_per_atom.append(b.vec_pack(scaled_comps, F32))

            # gfx942 has no ds_read_b64_tr_b16, so the PV ``B`` operand is built
            # from ordinary strided LDS loads that reproduce the EXACT per-lane
            # layout ds_read_tr16_b64 would produce for a 16x16x16 atom over the
            # V_lds tile at (row base = k*16, col base = n*16):
            #   lane l = 16*k_chunk + n_col  (k_chunk = lane/16, n_col = lane%16)
            #   B[j] = V_lds[v_buf, k*16 + k_chunk*4 + j, n*16 + n_col], j in 0..3
            # k_chunk = lane_rg, n_col = lane_col. Each of the 4 elements is a
            # different LDS row (stride HD), so this is 4 strided ds_read_u16
            # loads -- the same B operand the transpose read delivers, without
            # the gfx950-only transpose intrinsic. (See helpers/mfma_attention.py
            # 786-823 for the dense-kernel B-operand index precedent.)
            v_n_col = b.add(b.mul(b.const_i32(n), b.const_i32(16)), lane_col)
            v_k_chunk_base = b.mul(lane_rg, b.const_i32(4))

            def _strided_v_b_operand(k_iter: int) -> Value:
                """<4 x dtype> PV B operand for K-iter ``k_iter`` via strided LDS.

                Each of the 4 elements is a distinct V row. ``_v_load1`` applies
                the bank-deconflict slot mapping when ``HIPDNN_GFX942_SWIZZLE_VLDS``
                is on (foldable shift/and on ``v_row``); off, it is the natural
                ``V_lds[v_buf, v_row, v_n_col]`` read.
                """
                bv = b.zero_vec(dtype, 4)
                for j in range(4):
                    v_row = b.add(
                        b.const_i32(k_iter * 16 + j),
                        v_k_chunk_base,
                    )
                    elem = b.vec_extract(_v_load1(v_buf, v_row, v_n_col), 0)
                    bv = b.vec_insert(bv, elem, j)
                return bv

            # V is single-buffered; the V_lds buffer index is always 0.
            v_buf = b.const_i32(0)
            for k in range(PV_K_ITERS):
                if False:
                    # gfx950-only wide-K (K=32) and native-fp8 PV paths are not
                    # reachable on gfx942 (PV_K_STEP is always 16 and fp8 K/V is
                    # rejected in the spec/builder); kept structurally for parity
                    # with the gfx950 file but compiled out.
                    p_off = b.add(b.const_i32(k * 32), b.mul(lane_rg, b.const_i32(8)))
                    row_r0 = pv_tr_reader.row(b, k_offset=k * 32, read=0)
                    row_r1 = pv_tr_reader.row(b, k_offset=k * 32, read=1)
                    if PV_FP8_MFMA:
                        stripe_const = b.const_i32(n)
                        k_row_per_lane = b.add(
                            b.mul(lane_rg, b.const_i32(8)),
                            b.div(lane_col, b.const_i32(2)),
                        )
                        k_row_for_iter = b.add(
                            b.const_i32(k * 32),
                            k_row_per_lane,
                        )
                        B_v8_lo = b.ds_read_tr_b8(
                            V_lds,
                            v_buf,
                            stripe_const,
                            k_row_for_iter,
                            b.const_i32(0),
                            dtype=FP8E4M3,
                        )
                        B_v8_hi = b.ds_read_tr_b8(
                            V_lds,
                            v_buf,
                            stripe_const,
                            k_row_for_iter,
                            b.const_i32(8),
                            dtype=FP8E4M3,
                        )
                        # Per-lane select on (lane_col < 8): low-half
                        # lanes keep B_v8_lo (N = 0..7), high-half lanes
                        # keep B_v8_hi (N = 8..15).
                        lo_mask = b.cmp_lt(lane_col, b.const_i32(8))
                        B_v8 = b.vector_select(
                            b.vector_splat(lo_mask, 8),
                            B_v8_lo,
                            B_v8_hi,
                        )
                        for atom in range(M_ATOMS_PER_WARP):
                            p_row = b.add(
                                wave_row_base, b.add(b.const_i32(atom * 16), lane_col)
                            )
                            A_p8 = b.smem_load_vN(
                                P_lds, p_row, p_off, dtype=FP8E4M3, n=8
                            )
                            raw = b.mfma_f32_16x16x32_fp8(A_p8, B_v8, b.zero_vec_f32(4))
                            comps = []
                            for ii in range(4):
                                old = b.vec_extract(acc_per_atom[atom], ii)
                                add = b.fmul(b.vec_extract(raw, ii), pv_fp8_scale)
                                comps.append(b.fadd(old, add))
                            acc_per_atom[atom] = b.vec_pack(comps, F32)
                    else:
                        n_col_base = b.add(
                            b.mul(b.const_i32(n), b.const_i32(16)), tr_col_lane
                        )
                        B_r0 = b.ds_read_tr16_b64(
                            V_lds, v_buf, row_r0, n_col_base, dtype=dtype
                        )
                        B_r1 = b.ds_read_tr16_b64(
                            V_lds, v_buf, row_r1, n_col_base, dtype=dtype
                        )
                        B_v = b.vec_concat(B_r0, B_r1)
                        for atom in range(M_ATOMS_PER_WARP):
                            if REGISTER_PV:
                                A_p = _pack_p_a32(
                                    [p_regs_f32[atom * 4 + r][2 * k] for r in range(4)],
                                    [
                                        p_regs_f32[atom * 4 + r][2 * k + 1]
                                        for r in range(4)
                                    ],
                                )
                            else:
                                # P_lds row for this atom: each warp's atom_idx slice
                                # of P_lds[BLOCK_M_PER_WARP, T] -- the in-warp row is
                                # ``atom * 16 + lane_col``.
                                p_row = b.add(
                                    wave_row_base,
                                    b.add(b.const_i32(atom * 16), lane_col),
                                )
                                A_p = b.smem_load_vN(
                                    P_lds, p_row, p_off, dtype=dtype, n=8
                                )
                            acc_per_atom[atom] = _mfma_16x16x32(
                                b, dtype, A_p, B_v, acc_per_atom[atom]
                            )
                else:
                    # K=16 (gfx942 narrow atom): build the full <4 x dtype> PV B
                    # operand from 4 strided LDS loads instead of a single
                    # gfx950-only ds_read_b64_tr_b16. ``_strided_v_b_operand``
                    # reproduces the transpose read's exact per-lane (row, col)
                    # mapping for this 16x16x16 atom.
                    p_off = b.add(b.const_i32(k * 16), b.mul(lane_rg, b.const_i32(4)))
                    B_v = _strided_v_b_operand(k)
                    for atom in range(M_ATOMS_PER_WARP):
                        if REGISTER_PV:
                            A_p = _pack_p_a16(
                                [p_regs_f32[atom * 4 + r][k] for r in range(4)]
                            )
                        else:
                            p_row = b.add(
                                wave_row_base, b.add(b.const_i32(atom * 16), lane_col)
                            )
                            A_p = b.smem_load_vN(P_lds, p_row, p_off, dtype=dtype, n=4)
                        acc_per_atom[atom] = _mfma_16x16x16(
                            b, dtype, A_p, B_v, acc_per_atom[atom]
                        )
            for atom in range(M_ATOMS_PER_WARP):
                new_acc[n * M_ATOMS_PER_WARP + atom] = acc_per_atom[atom]

        yields = []
        for r in range(SOFTMAX_STATE_SLOTS):
            yields.append(m_new[r])
            yields.append(new_l_vals[r])
        for n in range(ACC_N_TILES):
            for atom in range(ACC_M_ATOMS):
                yields.append(new_acc[n * ACC_M_ATOMS + atom])
        yields.append(cur_buf if GROUPED_KV2 else nxt_buf)
        b.scf_yield(*yields)

    # ---- drive the (one or two) KV-loop phases ----
    # ``_phases`` is a single (tile_start, tile_end, skip_mask=False) entry for
    # every path except the no-SW transposed combo, which splits into a
    # full-tile phase (skip_mask=True) followed by a boundary phase. The
    # softmax/acc/buffer carry is threaded from one phase into the next via
    # ``scf_for_iter`` results so the second loop resumes exactly where the
    # first left off (same K/V double-buffer slot, same online-softmax state).
    kvloop = b.scf_for_iter(tile_start, tile_end, kv_step, iter_args, iv_name="kv_tile")
    with kvloop as (kv_tile_iv, carry):
        _emit_kv_body(kv_tile_iv, carry, False)

    # ---------------- epilogue ----------------
    # The loop issues a uniform "next K" async load every iteration, including
    # the final iteration where that load is intentionally never consumed. The
    # partial wait before PV leaves that final prefetch in flight. CK Tile
    # kernels always close outstanding async-copy groups before the CTA exits;
    # do the same here so no raw global->LDS operation can outlive the kernel
    # and corrupt later launches in the same process.
    b.s_waitcnt(vmcnt=0, lgkmcnt=0)
    b.sync()

    final = kvloop.results
    l_final = [final[2 * r + 1] for r in range(SOFTMAX_STATE_SLOTS)]
    ml_count_final = 2 * SOFTMAX_STATE_SLOTS
    # acc_final indexed by (n * ACC_M_ATOMS + atom)
    acc_final = [
        final[ml_count_final + n * ACC_M_ATOMS + atom]
        for n in range(ACC_N_TILES)
        for atom in range(ACC_M_ATOMS)
    ]

    def _acc_final_get(n: int, atom: int) -> Value:
        return acc_final[n * ACC_M_ATOMS + atom]

    # Per-row reciprocal of L (computed once, reused across stripes).
    rcp_l = [b.rcp(l_final[r]) for r in range(SOFTMAX_STATE_SLOTS)]
    l_nonzero = [b.fcmp("ogt", l_final[r], zero_f) for r in range(SOFTMAX_STATE_SLOTS)]

    if USE_MFMA_32X32:
        if TRANSPOSED_QK_32X32:
            # Per-lane direct scalar global stores. Tested both this and a
            # coalesced Acc_lds-staged variant; the scalar form was
            # consistently faster on bf16 hd64/hd128 prefill (1.05-1.36x).
            # Why: each lane already owns 16 d positions at one fixed
            # q_row, so the 32 lanes in a half-wave naturally produce 32
            # adjacent-token, 16-element stores per cycle. Hardware
            # already coalesces those into ~32 transactions per CTA. The
            # staged form added an LDS bounce plus barriers that cost
            # more than the (now redundant) gather it saved.
            q_row_t = b.add(wave_row_base, lane_col32)
            op_pos_t = b.add(qb_start_pos, b.div(q_row_t, b.const_i32(NQK)))
            op_qh_t = b.add(
                b.mul(kv_head_idx, b.const_i32(NQK)),
                b.mod(q_row_t, b.const_i32(NQK)),
            )
            op_mask_t = b.land(
                b.cmp_lt(op_pos_t, cur_batch_q_len),
                b.cmp_lt(op_qh_t, b.const_i32(NUM_QH)),
            )
            out_base_t, _ = q_desc.offset(
                b,
                token=b.add(cu_q_start, op_pos_t),
                head=op_qh_t,
                dim=b.const_i32(0),
            )
            inv_l_t = b.rcp(l_final[0])
            l_nonzero_t = b.fcmp("ogt", l_final[0], zero_f)
            for n in range(ACC_N_TILES):
                acc32 = _acc_final_get(n, 0)
                for reg in range(REGS_PER_LANE):
                    out_col_t = b.add(
                        b.const_i32(n * 32), _mfma_32x32_c_row(b, lane, reg)
                    )
                    v = b.vec_extract(acc32, reg)
                    normalized = b.fmul(v, inv_l_t)
                    final_h = b.cast_f32_to(
                        b.select(l_nonzero_t, normalized, zero_f), dtype
                    )
                    with b.scf_if(op_mask_t):
                        b.global_store(
                            output, b.add(out_base_t, out_col_t), final_h, align=2
                        )
            return b.kernel

        # Coalesced correctness-first epilogue for the M32N32K16 accumulator
        # layout. Stage one 32-column stripe into Acc_lds, then reuse the
        # vec8 cooperative global-store pattern. This avoids the very slow
        # scalar global stores from the first end-to-end scaffold.
        OUT_VEC32 = 8
        OUT_PER_THREAD_HALVES32 = (BLOCK_M * 32) // THREADS
        assert OUT_PER_THREAD_HALVES32 % OUT_VEC32 == 0
        OUT_CHUNKS_PER_THREAD32 = OUT_PER_THREAD_HALVES32 // OUT_VEC32
        OUT_THREADS_PER_ROW32 = 32 // (OUT_CHUNKS_PER_THREAD32 * OUT_VEC32)
        OUT_ROW_BASE32 = b.div(tid, b.const_i32(OUT_THREADS_PER_ROW32))
        OUT_col_base32 = b.mul(
            b.mod(tid, b.const_i32(OUT_THREADS_PER_ROW32)),
            b.const_i32(OUT_CHUNKS_PER_THREAD32 * OUT_VEC32),
        )
        op_pos32_base = b.add(qb_start_pos, b.div(OUT_ROW_BASE32, b.const_i32(NQK)))
        op_qh32_base = b.add(
            b.mul(kv_head_idx, b.const_i32(NQK)),
            b.mod(OUT_ROW_BASE32, b.const_i32(NQK)),
        )
        op_mask32_base = b.land(
            b.cmp_lt(op_pos32_base, cur_batch_q_len),
            b.cmp_lt(op_qh32_base, b.const_i32(NUM_QH)),
        )
        out_base32_base, _ = q_desc.offset(
            b,
            token=b.add(cu_q_start, op_pos32_base),
            head=op_qh32_base,
            dim=b.const_i32(0),
        )
        for n in range(ACC_N_TILES):
            acc32 = _acc_final_get(n, 0)
            for reg in range(REGS_PER_LANE):
                row = b.add(wave_row_base, _mfma_32x32_c_row(b, lane, reg))
                col_in_stripe = lane_col32
                v = b.vec_extract(acc32, reg)
                normalized = b.fmul(v, rcp_l[reg])
                final_h = b.cast_f32_to(
                    b.select(l_nonzero[reg], normalized, zero_f), dtype
                )
                b.smem_store_vN(Acc_lds, [row, col_in_stripe], final_h, 1)
            b.sync()
            for chunk in range(OUT_CHUNKS_PER_THREAD32):
                col_in_stripe = b.add(OUT_col_base32, b.const_i32(chunk * OUT_VEC32))
                v8h = b.smem_load_vN(
                    Acc_lds, OUT_ROW_BASE32, col_in_stripe, dtype=dtype, n=OUT_VEC32
                )
                out_col = b.add(b.const_i32(n * 32), col_in_stripe)
                with b.scf_if(op_mask32_base):
                    b.global_store_vN(
                        output,
                        b.add(out_base32_base, out_col),
                        v8h,
                        OUT_VEC32,
                        align=16,
                    )
            if n + 1 < ACC_N_TILES:
                b.sync()
        return b.kernel

    # ---------------- striped epilogue ----------------
    # Loop in ``OUT_STRIPES`` stripes, each covering ``OUT_STRIPE_COLS = 32``
    # consecutive output columns (= 2 PV N-tiles). For each stripe we:
    #   1. Cast and normalise each warp's MFMA acc slice (4 floats per
    #      N-tile per lane), store as the working dtype into Acc_lds.
    #   2. Sync so the cooperative output store sees every warp's writes.
    #   3. Cooperative vec8 output store from Acc_lds into the global
    #      output buffer at the right stripe column base.
    #   4. Sync so the next stripe can safely overwrite Acc_lds.
    #
    # ``Acc_lds`` is only [BLOCK_M, OUT_STRIPE_COLS] of the working dtype,
    # so the per-CTA epilogue LDS is ``BLOCK_M * 32 * 2`` bytes -- a 75-87%
    # reduction vs the previous ``BLOCK_M * HD * 4`` F32 buffer. That LDS
    # savings is what gives MI355X room for 3 WGs/CU on prefill workloads
    # (the documented Triton ``num_warps=4`` BLOCK_M=128 configuration runs
    # at 3-4 WGs/CU; we now match that occupancy class).
    N_TILES_PER_STRIPE = OUT_STRIPE_COLS // MFMA_N
    assert PV_N_TILES % N_TILES_PER_STRIPE == 0
    # Cooperative store distribution for one stripe ([BLOCK_M, OUT_STRIPE_COLS]
    # of dtype). Per stripe: total halves = BLOCK_M * OUT_STRIPE_COLS; per
    # thread = BLOCK_M * OUT_STRIPE_COLS / THREADS. We unroll
    # ``OUT_CHUNKS_PER_THREAD`` consecutive 16-byte ``vec8`` stores per thread
    # so each thread always writes one row's slice of the stripe contiguously.
    # For BLOCK_M=16 NW=1 HD=128 (decode) THREADS=64, full-HD stripe:
    #   16*128/64 = 32 halves/thread = 4 vec8 chunks per thread per stripe.
    # For BLOCK_M=64 NW=4 HD=64 (prefill) THREADS=256, 32-col stripe:
    #   64*32/256 = 8 halves/thread = 1 vec8 chunk per thread per stripe.
    OUT_VEC = 8
    OUT_PER_THREAD_HALVES = (BLOCK_M * OUT_STRIPE_COLS) // THREADS
    assert OUT_PER_THREAD_HALVES % OUT_VEC == 0 and OUT_PER_THREAD_HALVES > 0, (
        f"Expected a positive multiple of vec{OUT_VEC} halves per thread per "
        f"stripe (got {OUT_PER_THREAD_HALVES} for BLOCK_M={BLOCK_M} "
        f"STRIPE_COLS={OUT_STRIPE_COLS} THREADS={THREADS})"
    )
    OUT_CHUNKS_PER_THREAD = OUT_PER_THREAD_HALVES // OUT_VEC
    OUT_THREADS_PER_ROW = OUT_STRIPE_COLS // (OUT_CHUNKS_PER_THREAD * OUT_VEC)
    OUT_ROWS_PER_ITER = THREADS // OUT_THREADS_PER_ROW
    assert OUT_ROWS_PER_ITER == BLOCK_M, (
        f"Stripe cooperative-store assumes one row per thread group "
        f"(got OUT_ROWS_PER_ITER={OUT_ROWS_PER_ITER}, BLOCK_M={BLOCK_M})"
    )
    OUT_ROW_BASE = b.div(tid, b.const_i32(OUT_THREADS_PER_ROW))
    OUT_col_base_in_stripe = b.mul(
        b.mod(tid, b.const_i32(OUT_THREADS_PER_ROW)),
        b.const_i32(OUT_CHUNKS_PER_THREAD * OUT_VEC),
    )

    # Compute (op_pos, op_qh, op_mask, out_base) once per CTA -- these
    # depend only on OUT_row, which is loop-invariant across stripes.
    op_pos = b.add(qb_start_pos, b.div(OUT_ROW_BASE, b.const_i32(NQK)))
    op_qh = b.add(
        b.mul(kv_head_idx, b.const_i32(NQK)),
        b.mod(OUT_ROW_BASE, b.const_i32(NQK)),
    )
    op_mask = b.land(
        b.cmp_lt(op_pos, cur_batch_q_len), b.cmp_lt(op_qh, b.const_i32(NUM_QH))
    )
    out_base, _ = q_desc.offset(
        b,
        token=b.add(cu_q_start, op_pos),
        head=op_qh,
        dim=b.const_i32(0),
    )

    for stripe in range(OUT_STRIPES):
        n_start = stripe * N_TILES_PER_STRIPE
        # ---- stage 1: write this stripe's N-tiles into Acc_lds ----
        # For ``M_ATOMS_PER_WARP=2`` each warp writes 2 stacked 16-row tiles
        # per N-tile. The reg loop iterates over all REGS_PER_LANE = 4*M_ATOMS
        # row slots; reg ``r`` decomposes into ``(atom=r//4, in_atom=r%4)``
        # for both the row offset and the per-atom accumulator pick.
        for n_local in range(N_TILES_PER_STRIPE):
            n = n_start + n_local
            for reg in range(REGS_PER_LANE):
                atom = reg // 4
                in_atom = reg % 4
                row = b.add(wave_row_base, _in_warp_row(reg))
                # Column within the stripe = n_local*16 + lane_col
                col_in_stripe = b.add(b.const_i32(n_local * MFMA_N), lane_col)
                v = b.vec_extract(_acc_final_get(n, atom), in_atom)
                normalized = b.fmul(v, rcp_l[reg])
                final_h = b.cast_f32_to(
                    b.select(l_nonzero[reg], normalized, zero_f), dtype
                )
                b.smem_store_vN(Acc_lds, [row, col_in_stripe], final_h, 1)
        b.sync()
        # ---- stage 2: cooperative vec8 store(s) from Acc_lds to global ----
        for chunk in range(OUT_CHUNKS_PER_THREAD):
            col_in_stripe = b.add(OUT_col_base_in_stripe, b.const_i32(chunk * OUT_VEC))
            v8h = b.smem_load_vN(
                Acc_lds, OUT_ROW_BASE, col_in_stripe, dtype=dtype, n=OUT_VEC
            )
            out_col = b.add(b.const_i32(stripe * OUT_STRIPE_COLS), col_in_stripe)
            with b.scf_if(op_mask):
                b.global_store_vN(
                    output, b.add(out_base, out_col), v8h, OUT_VEC, align=16
                )
        # ---- stage 3: sync so the next stripe can overwrite Acc_lds ----
        if stripe + 1 < OUT_STRIPES:
            b.sync()

    return b.kernel
