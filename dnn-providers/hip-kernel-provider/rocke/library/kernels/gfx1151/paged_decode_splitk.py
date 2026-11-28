# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paged split-K decode attention for gfx1151 (RDNA3.5, wave32).

Two-launch FlashDecoding pipeline over vLLM's *paged* KV cache:

1. ``build_paged_decode_splitk_segment`` -- one wave32 CTA per
   ``(request, kv_head, split)``. Walks its slice of the block table and
   emits the unnormalised ``(m, l, acc)`` triple per fused Q head.
2. ``build_paged_decode_splitk_reduce`` -- one CTA per
   ``(request, kv_head)``; merges the per-split triples into ``O``.

Three things separate this from :mod:`kernels.common.fmha_splitkv_decode`,
which is wave64 and gated on the MFMA atom catalog:

* **GQA fusion.** The grid is keyed on the *kv* head, so one CTA serves all
  ``gqa_fuse`` query heads of that kv head and reads K/V **once** instead of
  ``gqa_fuse`` times. Decode attention is ~4 FLOP/byte, so this is the whole
  ballgame -- the per-Q-head grid burns 4x the DRAM traffic for the same math.
* **wave32**, with a tunable split of the wave between the head_size axis and
  the key axis -- see ``d_lanes`` on :class:`PagedDecodeCfg`.
* **Paged addressing** against the layout vLLM's ROCm backend actually
  writes (probed, not inferred from ``split_kv_cache``'s ``.view()``)::

      K[blk, kv_head, D/x, slot, x]   x = 16 / element_size = 8 for f16
      V[blk, kv_head, D, slot]        <- slot is the FASTEST-varying dim

  V is therefore *already transposed*: for a fixed ``d``, consecutive tokens
  are contiguous, so the PV gather reads ``block_n`` tokens in one
  ``global_load_dwordx4`` per ``d`` and the wave consumes the block's V slab
  with zero over-fetch. K's x-split means a lane owning ``d in [ept*t,
  ept*t+ept)`` sits wholly inside one x-group (guaranteed by the
  ``x % ept == 0`` check), so its K slice is one contiguous vector load too.

The arithmetic intensity is two orders of magnitude below the gfx1151 ridge
point, so the body is deliberately plain VALU FMA -- no WMMA. At ``Sq == 1``
a WMMA tile would be 1/16 utilised for no bandwidth benefit.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

from rocke.core.ir import F32, I32, IRBuilder, KernelDef, PtrType, Value
from rocke.helpers.attention import warp_xor_reduce_sum
from rocke.helpers.io import io_ir_type, load_vec_as_f32, pack_f32_to
from rocke.helpers.spec import kernel_name_join, validate_field_types


__all__ = [
    "PagedDecodeCfg",
    "build_paged_decode_splitk_segment",
    "build_paged_decode_splitk_reduce",
    "is_valid_spec",
    "paged_decode_segment_grid",
    "paged_decode_reduce_grid",
    "paged_decode_workspace_shapes",
    "choose_num_splits",
]


# One wave per CTA. gfx1151 is wave32; the whole body (lane-owns-D
# distribution, 5-stage butterfly) is written against that width.
WAVE = 32

# Vector widths ``global_load_vN`` covers. ``ept`` and ``block_n`` must both
# land here or the gather degrades to scalar loads and the whole point of the
# transposed-V layout is lost.
_VEC_WIDTHS = (2, 4, 8)

# Sentinel for "this split saw no keys". Deliberately finite: the reduce
# computes ``exp2(m_seg - overall_max)``, and a true -inf on BOTH sides would
# produce NaN before the guarding select could discard it. Scores of masked
# *keys* do use a real -inf (``exp2(-inf - finite) == 0``, exactly right).
_NEG_BIG = -1e30


@dataclass(frozen=True)
class PagedDecodeCfg:
    head_size: int = 128
    num_q_heads: int = 32
    num_kv_heads: int = 8
    dtype: str = "f16"
    # Tokens per PHYSICAL kv block. Never spell this ``block_size``: the
    # sibling SwapQKCfg already uses that name for the CTA thread count.
    kv_block_size: int = 16
    num_splits: int = 8
    # Tokens per inner sub-tile == the width of one V vector load. Capped by
    # keys_per_sub, which d_lanes sets, so this default tracks the d_lanes
    # default rather than standing on its own: at d_lanes=8 a sub-tile is 4
    # keys wide and an 8-wide load would straddle two key partitions. The
    # launcher derives it the same way when passed 0.
    block_n: int = 4
    # Lanes that cooperate on ONE dot product. The wave's remaining
    # WAVE/d_lanes lanes work on disjoint keys and are folded together once
    # per segment, so this knob trades cross-lane reduce cost against
    # register pressure and nothing else:
    #
    #   ds_swizzle per key = d_lanes * log2(d_lanes) / (kv_block_size / 2)
    #   VGPRs for Q + acc  = 2 * gqa_fuse * head_size / d_lanes
    #
    # At the production shape that is 20 swizzles / 115 VGPRs (4 waves/SIMD) at
    # d_lanes=32, 8 / 152 (3 waves) at 16, and 3 / 172 (2 waves) at 8.
    #
    # 8 > 16 > 32 monotonically, measured 2026-10-18 at heads_per_cta=0 with
    # min-of-15-reps graph replay: 8 beats 16 by 6.3% at B=1, 2.0% at B=4 and
    # 1.6% at B=16, with no (B, Sk) point in the sweep where it loses. An
    # earlier sweep had 16 winning everywhere; that was true of the kernel as it
    # stood before block_n, unroll and waves_per_eu existed, and block_n is now
    # derived from d_lanes, so the occupancy arm is a different function. Treat
    # any d_lanes conclusion older than the last edit to this file as void and
    # re-sweep.
    d_lanes: int = 8
    # Block-loop unroll: physical kv blocks consumed per loop iteration. All U
    # block-table lookups are issued ahead of the first block's math, and the
    # tail is masked, so any U is correct for any block count. Costs code size
    # and live registers, buys memory-level parallelism per wave.
    unroll: int = 1
    # Q heads per CTA. 0 means the full GQA ratio (K/V read once). A divisor of
    # the ratio splits the fusion: the grid grows by gqa_fuse/heads_per_cta, each
    # CTA carries proportionally fewer accumulator VGPRs (more waves per SIMD),
    # and K/V are re-read once per head group (L2/MALL hits at best).
    heads_per_cta: int = 0
    # ``amdgpu-waves-per-eu`` occupancy hint for the segment kernel; 0 = none.
    waves_per_eu: int = 0
    # 0: the reduce walks the splits in three scf loops. 1: all splits are
    # unrolled straight-line so every workspace load is issued up front.
    reduce_unroll: int = 0
    name: str = "rocke_paged_decode_splitk"

    def __post_init__(self) -> None:
        # No bool fields today, but every int here is either baked into the
        # kernel name or used to size a loop, and ``unroll=True`` would pass
        # an isinstance check and build a 1-deep unroll under a name that
        # claims otherwise. Guard the whole field list so it stays correct as
        # fields are added.
        validate_field_types(self)

    @property
    def gqa_fuse(self) -> int:
        """Q heads of one kv head: the GQA ratio."""
        return self.num_q_heads // self.num_kv_heads

    @property
    def heads_eff(self) -> int:
        """Q heads actually served by one segment CTA."""
        return self.heads_per_cta or self.gqa_fuse

    @property
    def head_groups(self) -> int:
        """Segment CTAs per (request, kv_head, split) along the head axis."""
        return self.gqa_fuse // self.heads_eff

    @property
    def key_subs(self) -> int:
        """Disjoint key partitions inside one wave."""
        return WAVE // self.d_lanes

    @property
    def keys_per_sub(self) -> int:
        """Keys of a physical block owned by one key partition."""
        return self.kv_block_size // self.key_subs

    @property
    def ept(self) -> int:
        """head_size elements owned by one lane in the segment kernel."""
        return self.head_size // self.d_lanes

    @property
    def ept_reduce(self) -> int:
        """head_size elements per lane in the reduce kernel.

        Always ``head_size / WAVE``: the reduce reads a finished ``[.., D]``
        row out of the workspace and has no key axis to trade against, so it
        spreads D over the whole wave regardless of ``d_lanes``.
        """
        return self.head_size // WAVE

    @property
    def x(self) -> int:
        """K cache x-split width: ``16 / element_size``, 8 for any 2-byte dtype."""
        return 8

    def kernel_name(self, phase: str) -> str:
        # Non-default knobs only, so default-config kernel names are unchanged.
        extra = []
        if phase == "seg":
            if self.unroll != 1:
                extra.append(f"U{self.unroll}")
            if self.heads_eff != self.gqa_fuse:
                extra.append(f"HC{self.heads_eff}")
            if self.waves_per_eu:
                extra.append(f"WE{self.waves_per_eu}")
        elif self.reduce_unroll:
            extra.append("RU")
        return kernel_name_join(
            self.name,
            phase,
            f"H{self.head_size}",
            f"HQ{self.num_q_heads}",
            f"HK{self.num_kv_heads}",
            self.dtype,
            f"KB{self.kv_block_size}",
            f"BN{self.block_n}",
            f"DL{self.d_lanes}",
            f"S{self.num_splits}",
            *extra,
        )


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _chunk_widths(n: int) -> List[int]:
    """Split an ``n``-element lane slice into widths ``global_load_vN`` covers.

    ``ept`` exceeds the widest vector load once ``d_lanes`` drops below
    ``head_size / 8``, so a lane's Q slice, K slice and accumulator row each
    become a short run of maximal loads rather than a single one.
    """
    out: List[int] = []
    rem = n
    while rem:
        w = max(w for w in _VEC_WIDTHS if w <= rem)
        out.append(w)
        rem -= w
    return out


def is_valid_spec(cfg: PagedDecodeCfg, arch: str = "gfx1151") -> Tuple[bool, str]:
    if arch != "gfx1151":
        return False, f"paged_decode_splitk targets gfx1151, got {arch!r}"
    if cfg.dtype not in ("f16", "fp16", "bf16"):
        return False, f"dtype {cfg.dtype!r} not in {{f16, bf16}}"
    if cfg.head_size % WAVE != 0:
        return False, f"head_size {cfg.head_size} must be a multiple of {WAVE}"
    if not _is_pow2(cfg.d_lanes) or not 2 <= cfg.d_lanes <= WAVE:
        return False, f"d_lanes {cfg.d_lanes} must be a power of two in [2, {WAVE}]"
    if cfg.head_size % cfg.d_lanes != 0:
        return False, f"d_lanes {cfg.d_lanes} must divide head_size {cfg.head_size}"
    if cfg.ept_reduce not in _VEC_WIDTHS:
        return False, (
            f"ept_reduce {cfg.ept_reduce} (= head_size/{WAVE}) must be in "
            f"{_VEC_WIDTHS}: the reduce kernel writes O as one vector per lane"
        )
    if not _is_pow2(cfg.ept) or cfg.ept < 2:
        return False, (
            f"ept {cfg.ept} (= head_size/d_lanes) must be a power of two >= 2 so "
            "the per-lane K slice splits into whole vector loads"
        )
    if cfg.ept % cfg.x and cfg.x % cfg.ept:
        return False, (
            f"ept {cfg.ept} and the K x-split width {cfg.x} must divide one "
            "another; otherwise a lane's d-slice straddles a partial x-group and "
            "stops being a contiguous run"
        )
    if cfg.kv_block_size % cfg.key_subs != 0:
        return False, (
            f"key_subs {cfg.key_subs} (= {WAVE}/d_lanes) must divide "
            f"kv_block_size {cfg.kv_block_size}, so every key partition owns the "
            "same number of slots of a physical block"
        )
    if cfg.num_kv_heads <= 0 or cfg.num_q_heads % cfg.num_kv_heads != 0:
        return False, (
            f"num_q_heads {cfg.num_q_heads} must be a positive multiple of "
            f"num_kv_heads {cfg.num_kv_heads}"
        )
    if not _is_pow2(cfg.kv_block_size) or cfg.kv_block_size % 16 != 0:
        return False, (
            f"kv_block_size {cfg.kv_block_size} must be a power of two and a "
            "multiple of 16 (vLLM's ROCm backend rejects anything else)"
        )
    if not _is_pow2(cfg.num_splits) or cfg.num_splits > 128:
        return False, f"num_splits {cfg.num_splits} must be a power of two <= 128"
    if cfg.block_n not in _VEC_WIDTHS:
        return False, (
            f"block_n {cfg.block_n} must be in {_VEC_WIDTHS}: it is the width of "
            "the transposed-V vector load"
        )
    if cfg.keys_per_sub % cfg.block_n != 0:
        return False, (
            f"block_n {cfg.block_n} must divide keys_per_sub {cfg.keys_per_sub} "
            f"(= kv_block_size {cfg.kv_block_size} / key_subs {cfg.key_subs}) so a "
            "sub-tile stays inside one key partition of one physical block"
        )
    if cfg.unroll not in (1, 2, 4, 8):
        return False, f"unroll {cfg.unroll} must be one of (1, 2, 4, 8)"
    if cfg.heads_per_cta < 0 or cfg.gqa_fuse % cfg.heads_eff != 0:
        return False, (
            f"heads_per_cta {cfg.heads_per_cta} must be 0 (= full) or divide the "
            f"GQA ratio {cfg.gqa_fuse}"
        )
    if not 0 <= cfg.waves_per_eu <= 16:
        return False, f"waves_per_eu {cfg.waves_per_eu} must be in [0, 16]"
    if cfg.reduce_unroll not in (0, 1):
        return False, f"reduce_unroll {cfg.reduce_unroll} must be 0 or 1"
    if cfg.reduce_unroll and cfg.num_splits > 64:
        return False, "reduce_unroll=1 unrolls every split; num_splits must be <= 64"
    return True, "ok"


# ---------------------------------------------------------------------------
# Parameter packs
# ---------------------------------------------------------------------------


def _declare_segment_params(b: IRBuilder, cfg: PagedDecodeCfg) -> Dict[str, Value]:
    dt = io_ir_type(cfg.dtype)
    P: Dict[str, Value] = {}
    P["Q"] = b.param("Q", PtrType(dt, "global"), noalias=True, readonly=True, align=16)
    P["KCache"] = b.param(
        "KCache", PtrType(dt, "global"), noalias=True, readonly=True, align=16
    )
    P["VCache"] = b.param(
        "VCache", PtrType(dt, "global"), noalias=True, readonly=True, align=16
    )
    # align=4, not 16: a caller may hand over a ROW of the [num_reqs, bt_stride]
    # table rather than its base, and row i only inherits element alignment.
    P["BlockTable"] = b.param(
        "BlockTable", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    P["seq_lens"] = b.param(
        "seq_lens", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    P["ws_m"] = b.param("ws_m", PtrType(F32, "global"), noalias=True, align=4)
    P["ws_l"] = b.param("ws_l", PtrType(F32, "global"), noalias=True, align=4)
    P["ws_acc"] = b.param("ws_acc", PtrType(F32, "global"), noalias=True, align=16)
    P["scale_log2"] = b.param("scale_log2", F32)
    P["stride_q_seq"] = b.param("stride_q_seq", I32)
    P["stride_q_head"] = b.param("stride_q_head", I32)
    P["bt_stride"] = b.param("bt_stride", I32)
    P["bt_num_entries"] = b.param("bt_num_entries", I32)
    # Page count of the KV cache, so a block id read out of the table can be
    # clamped into range before it is scaled into a cache offset.
    P["num_blocks"] = b.param("num_blocks", I32)
    return P


def _declare_reduce_params(b: IRBuilder, cfg: PagedDecodeCfg) -> Dict[str, Value]:
    dt = io_ir_type(cfg.dtype)
    P: Dict[str, Value] = {}
    P["ws_m"] = b.param(
        "ws_m", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    P["ws_l"] = b.param(
        "ws_l", PtrType(F32, "global"), noalias=True, readonly=True, align=4
    )
    P["ws_acc"] = b.param(
        "ws_acc", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    P["O"] = b.param("O", PtrType(dt, "global"), noalias=True, writeonly=True, align=16)
    P["stride_o_seq"] = b.param("stride_o_seq", I32)
    P["stride_o_head"] = b.param("stride_o_head", I32)
    return P


# ---------------------------------------------------------------------------
# Phase 1 -- segment kernel
# ---------------------------------------------------------------------------


def build_paged_decode_splitk_segment(
    spec: PagedDecodeCfg, *, arch: str = "gfx1151"
) -> KernelDef:
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid paged_decode_splitk spec: {why}")

    D = spec.head_size
    BS = spec.kv_block_size
    BN = spec.block_n
    HK = spec.num_kv_heads
    GF = spec.heads_eff  # Q heads in this CTA
    GFT = spec.gqa_fuse  # Q heads per kv head (workspace / Q layout)
    HG = spec.head_groups
    U = spec.unroll
    EPT = spec.ept
    X = spec.x
    NS = spec.num_splits
    DL = spec.d_lanes
    KSUB = spec.key_subs
    KPS = spec.keys_per_sub
    dl_log2 = DL.bit_length() - 1
    bs_log2 = BS.bit_length() - 1
    ns_log2 = NS.bit_length() - 1

    b = IRBuilder(spec.kernel_name("seg"))
    b.kernel.attrs["max_workgroup_size"] = WAVE
    if spec.waves_per_eu:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu
    p = _declare_segment_params(b, spec)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    zero_f = b.const_f32(0.0)
    neg_big = b.const_f32(_NEG_BIG)
    neg_inf = b.const_f32(float("-inf"))

    seq_idx = b.block_id_x()
    grid_y = b.block_id_y()
    if HG == 1:
        kv_head = grid_y
        head0 = None
    else:
        # Grid y enumerates (kv_head, head group): head groups of one kv head
        # are adjacent, so they launch together and share K/V through L2.
        kv_head = b.div(grid_y, b.const_i32(HG))
        head0 = b.mul(b.mod(grid_y, b.const_i32(HG)), b.const_i32(GF))
    seg_idx = b.block_id_z()
    tid = b.thread_id_x()

    # seq_lens arrives through a global load, and AMDGPU treats every
    # addrspace(1) load as a divergence source. Without this promotion the
    # split bounds -- and therefore the whole outer loop -- go vector, and the
    # block loop lowers as a divergent loop over uniform data.
    seqlen_k = b.to_sgpr_u32(b.global_load_i32(p["seq_lens"], seq_idx))
    # A request cannot own more keys than its row of the block table covers.
    # Rows are contiguous, so a seq_lens entry that overruns its row would walk
    # into the NEXT request's pages -- in bounds, so it faults nothing, and the
    # values fold silently into this request's softmax. Clamping at the source
    # keeps num_kv_blocks and key_limit consistent: the loop simply never
    # reaches the uncovered blocks. Truncating a bad length is visible in the
    # output; reading a neighbour's keys is not.
    seqlen_k = b.smin(seqlen_k, b.shl(p["bt_stride"], b.const_i32(bs_log2)))

    # Block-ALIGNED splits: work is partitioned in units of physical kv blocks,
    # never raw keys. That buys one block-table lookup per outer iteration and
    # guarantees a sub-tile never straddles a block, so all the intra-block
    # address math folds to compile-time constants.
    num_kv_blocks = b.lshr(b.add(seqlen_k, b.const_i32(BS - 1)), b.const_i32(bs_log2))
    blocks_per_seg = b.lshr(
        b.add(num_kv_blocks, b.const_i32(NS - 1)), b.const_i32(ns_log2)
    )
    blk_start = b.mul(seg_idx, blocks_per_seg)
    blk_end = b.smin(b.add(blk_start, blocks_per_seg), num_kv_blocks)
    # blk_start >= blk_end means an empty split: the loop runs zero times and
    # the initial (m = -1e30, l = 0, acc = 0) falls straight through to the
    # workspace, where the reduce's guard discards it.

    # ---- lane -> (key partition, head_dim slice) map ----
    # tid splits as (key_sub, d_lane) with d_lane in the LOW dl_log2 bits, so
    # the QK reduce is an XOR butterfly over masks 1..DL/2 -- all below 32, so
    # all ds_swizzle -- and the once-per-segment cross-partition fold uses the
    # masks above it. Lane (s, dl) owns d in [dl*EPT, dl*EPT+EPT) and, within
    # each physical block, keys [s*KPS, s*KPS+KPS).
    if KSUB == 1:
        d_lane = tid
        key_sub = c0
    else:
        d_lane = b.mod(tid, b.const_i32(DL))
        key_sub = b.lshr(tid, b.const_i32(dl_log2))

    d0 = b.mul(d_lane, b.const_i32(EPT))

    # K is [blk, kv_head, D/X, slot, X]. A lane's d-slice is either inside one
    # x-group (EPT <= X) or a whole run of them (EPT >= X); both are checked
    # divisibilities, so in each case the slice is one or more contiguous runs
    # of width min(EPT, X).
    if EPT <= X:
        lanes_per_group = X // EPT
        k_group = b.div(d_lane, b.const_i32(lanes_per_group))
        k_in_group = b.mul(
            b.mod(d_lane, b.const_i32(lanes_per_group)), b.const_i32(EPT)
        )
        k_widths = [EPT]
    else:
        k_group = b.mul(d_lane, b.const_i32(EPT // X))
        k_in_group = c0
        k_widths = [X] * (EPT // X)
    k_lane_off = b.add(b.mul(k_group, b.const_i32(BS * X)), k_in_group)

    # V is [blk, kv_head, D, slot]; the lane's slot window is its key partition.
    v_lane_off = b.mul(d0, b.const_i32(BS))
    if KSUB > 1:
        sub_slot = b.mul(key_sub, b.const_i32(KPS))
        v_lane_off = b.add(v_lane_off, sub_slot)
        k_lane_off = b.add(k_lane_off, b.mul(sub_slot, b.const_i32(X)))

    # CTA-uniform: this kv head's slab inside a physical block, and the row of
    # the [num_reqs, bt_stride] block table this request owns.
    kv_head_off = b.mul(kv_head, b.const_i32(D * BS))
    blk_scale = b.const_i32(HK * D * BS)
    bt_row = b.mul(seq_idx, p["bt_stride"])
    bt_max = p["bt_num_entries"]
    blk_max = b.sub(p["num_blocks"], c1)

    scale_log2 = p["scale_log2"]

    # ---- Q: one vector load per fused head, resident in registers ----
    q_head0 = b.mul(kv_head, b.const_i32(GFT))
    if head0 is not None:
        q_head0 = b.add(q_head0, head0)
    q_row_base = b.add(
        b.mul(seq_idx, p["stride_q_seq"]),
        b.mul(q_head0, p["stride_q_head"]),
    )
    q_widths = _chunk_widths(EPT)
    q_lane = []
    for g in range(GF):
        q_row = b.add(q_row_base, b.mul(b.const_i32(g), p["stride_q_head"]))
        vals, off = [], 0
        for w in q_widths:
            vals += load_vec_as_f32(
                b,
                p["Q"],
                (
                    b.add(q_row, d0)
                    if off == 0
                    else b.add(q_row, b.add(d0, b.const_i32(off)))
                ),
                dtype=spec.dtype,
                n=w,
            )
            off += w
        q_lane.append(vals)

    iter_args = [(f"m{g}", neg_big) for g in range(GF)]
    iter_args += [(f"l{g}", zero_f) for g in range(GF)]
    for g in range(GF):
        iter_args += [(f"a{g}_{k}", zero_f) for k in range(EPT)]

    # Keys at or past this bound are masked. With U > 1 the last iteration can
    # step past blk_end into the NEXT split's blocks, whose keys are < seqlen_k
    # and would otherwise be counted twice.
    key_limit = (
        seqlen_k if U == 1 else b.smin(seqlen_k, b.shl(blk_end, b.const_i32(bs_log2)))
    )

    def _block(m, l, acc, cache_base, key_base):  # noqa: E741
        """Online-softmax update over one physical kv block."""
        for sub in range(KPS // BN):
            tok0 = sub * BN

            # All BN K slices are issued before the first dot consumes one, so
            # the sub-tile's VMEM latency overlaps instead of serialising
            # behind each ds_swizzle butterfly. The dots below then run on
            # values that are already in flight.
            k_slices = []
            for t in range(BN):
                vals, off = [], 0
                for w in k_widths:
                    vals += load_vec_as_f32(
                        b,
                        p["KCache"],
                        b.add(
                            cache_base,
                            b.add(k_lane_off, b.const_i32((tok0 + t) * X + off * BS)),
                        ),
                        dtype=spec.dtype,
                        n=w,
                    )
                    off += w
                k_slices.append(vals)

            scores = [[None] * BN for _ in range(GF)]
            for t in range(BN):
                # Only the tail block of the whole sequence is partial, but the
                # predicate is data-dependent so it cannot be hoisted. Mask the
                # SCORE (not K): an out-of-range slot holds stale cache bytes
                # that may be inf/NaN, and zeroing K would still give it
                # softmax weight exp2(0 - m) against a garbage V.
                valid = b.cmp_lt(b.add(key_base, b.const_i32(tok0 + t)), key_limit)
                for g in range(GF):
                    partial = zero_f
                    for k in range(EPT):
                        partial = b.fma(q_lane[g][k], k_slices[t][k], partial)
                    dot = warp_xor_reduce_sum(b, partial, stages=dl_log2)
                    scores[g][t] = b.select(valid, b.fmul(dot, scale_log2), neg_inf)

            # Sub-tile online softmax: one max + one rescale per BN keys
            # instead of per key, so acc is touched BN times less often.
            m_new, alpha, l_new, probs = [], [], [], []
            for g in range(GF):
                tile_max = scores[g][0]
                for t in range(1, BN):
                    tile_max = b.fmax(tile_max, scores[g][t])
                mn = b.fmax(m[g], tile_max)
                al = b.exp2(b.fsub(m[g], mn))
                pg, lsum = [], zero_f
                for t in range(BN):
                    pv = b.exp2(b.fsub(scores[g][t], mn))
                    pg.append(pv)
                    lsum = b.fadd(lsum, pv)
                m_new.append(mn)
                alpha.append(al)
                l_new.append(b.fma(l[g], al, lsum))
                probs.append(pg)

            # V is stored slot-fastest, so one dwordx4 per d covers BN tokens
            # and the wave walks the block's V slab with zero over-fetch.
            v_slice = [
                load_vec_as_f32(
                    b,
                    p["VCache"],
                    b.add(cache_base, b.add(v_lane_off, b.const_i32(k * BS + tok0))),
                    dtype=spec.dtype,
                    n=BN,
                )
                for k in range(EPT)
            ]

            acc_new = []
            for g in range(GF):
                row = []
                for k in range(EPT):
                    a = b.fmul(acc[g][k], alpha[g])
                    for t in range(BN):
                        a = b.fma(probs[g][t], v_slice[k][t], a)
                    row.append(a)
                acc_new.append(row)

            m, l, acc = m_new, l_new, acc_new
        return m, l, acc

    loop = b.scf_for_iter(
        blk_start,
        blk_end,
        c1 if U == 1 else b.const_i32(U),
        iter_args=iter_args,
        iv_name="blk",
    )
    with loop as (blk, st):
        m = list(st[0:GF])
        l = list(st[GF : 2 * GF])  # noqa: E741 - online-softmax denominator
        acc = [list(st[2 * GF + g * EPT : 2 * GF + (g + 1) * EPT]) for g in range(GF)]

        # Every block-table lookup of the iteration is issued before the first
        # block's math. Blocks past blk_end (unroll tail) read page 0 -- always
        # a valid page -- and are masked out through key_limit.
        bases = []
        for u in range(U):
            blk_u = blk if u == 0 else b.add(blk, b.const_i32(u))
            bt_idx = b.add(bt_row, blk_u)
            in_table = b.cmp_lt(bt_idx, bt_max)
            if U > 1:
                in_table = b.land(in_table, b.cmp_lt(blk_u, blk_end))
            raw_blk = b.masked_global_load(
                p["BlockTable"],
                bt_idx,
                in_table,
                c0,  # block 0 is always a valid page
                I32,
                align=4,
            )
            # to_sgpr_u32 on the RAW block id is not an optimisation. The id
            # comes from an addrspace(1) load, which AMDGPU treats as divergent,
            # so without the promotion every K and V gather in this loop gets
            # wrapped in a 32-iteration waterfall. Promote first, scale after,
            # so the multiply-add lands on the SALU and the gathers keep a
            # uniform base.
            phys = b.to_sgpr_u32(raw_blk)
            # The block table is caller-supplied memory; a stale or corrupt
            # entry would otherwise scale straight into an out-of-range cache
            # offset and read arbitrary device memory. Clamping into [0,
            # num_blocks-1] keeps the gather inside the allocation -- the
            # result is garbage for a garbage table, but it is contained
            # garbage. Two SALU ops on an already-uniform value, off the
            # critical path of the gather.
            phys = b.smax(b.smin(phys, blk_max), c0)
            cache_base = b.add(b.mul(phys, blk_scale), kv_head_off)
            key0 = b.shl(blk_u, b.const_i32(bs_log2))
            key_base = key0 if KSUB == 1 else b.add(key0, sub_slot)
            bases.append((cache_base, key_base))

        for cache_base, key_base in bases:
            m, l, acc = _block(m, l, acc, cache_base, key_base)

        b.scf_yield(*(m + l + [v for row in acc for v in row]))

    res = loop.results
    m_final = list(res[0:GF])
    l_final = list(res[GF : 2 * GF])
    acc_final = [
        list(res[2 * GF + g * EPT : 2 * GF + (g + 1) * EPT]) for g in range(GF)
    ]

    # ---- fold the KSUB key partitions together ----
    # Each partition ran an independent online softmax over disjoint keys, so
    # this is the same merge the reduce kernel does across splits -- just held
    # in registers and paid once per segment instead of once per key, which is
    # the entire reason the QK reduce could shrink to log2(DL) stages.
    # m starts at the finite -1e30 sentinel, so an empty partition yields
    # exp2(-1e30 - finite) == 0 rather than the NaN a true -inf would give.
    for stage in range(KSUB.bit_length() - 1):
        mask = DL << stage
        for g in range(GF):
            m_other = b.warp_shuffle_xor(m_final[g], mask)
            l_other = b.warp_shuffle_xor(l_final[g], mask)
            m_new = b.fmax(m_final[g], m_other)
            a_self = b.exp2(b.fsub(m_final[g], m_new))
            a_other = b.exp2(b.fsub(m_other, m_new))
            l_final[g] = b.fma(l_final[g], a_self, b.fmul(l_other, a_other))
            acc_final[g] = [
                b.fma(
                    v,
                    a_self,
                    b.fmul(b.warp_shuffle_xor(v, mask), a_other),
                )
                for v in acc_final[g]
            ]
            m_final[g] = m_new

    # Workspace: ws_m/ws_l [B, HK, splits, GF], ws_acc [B, HK, splits, GF, D].
    unit = b.mul(
        b.add(
            b.mul(b.add(b.mul(seq_idx, b.const_i32(HK)), kv_head), b.const_i32(NS)),
            seg_idx,
        ),
        b.const_i32(GFT),
    )
    if head0 is not None:
        unit = b.add(unit, head0)
    is_lead = b.cmp_eq(tid, c0)
    with b.scf_if(is_lead):
        for g in range(GF):
            slot = b.add(unit, b.const_i32(g))
            b.global_store(p["ws_m"], slot, m_final[g], align=4)
            b.global_store(p["ws_l"], slot, l_final[g], align=4)
    # After the fold every key partition holds the same value, so without this
    # guard KSUB lanes would issue identical stores to the same address.
    acc_widths = _chunk_widths(EPT)
    guard = contextlib.nullcontext() if KSUB == 1 else b.scf_if(b.cmp_eq(key_sub, c0))
    with guard:
        for g in range(GF):
            acc_base = b.add(b.mul(b.add(unit, b.const_i32(g)), b.const_i32(D)), d0)
            off = 0
            for w in acc_widths:
                b.global_store_vN(
                    p["ws_acc"],
                    acc_base if off == 0 else b.add(acc_base, b.const_i32(off)),
                    b.vec_pack(acc_final[g][off : off + w], F32),
                    w,
                    align=w * 4,
                )
                off += w

    b.ret()
    return b.kernel


# ---------------------------------------------------------------------------
# Phase 2 -- reduce kernel
# ---------------------------------------------------------------------------


def build_paged_decode_splitk_reduce(
    spec: PagedDecodeCfg, *, arch: str = "gfx1151"
) -> KernelDef:
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid paged_decode_splitk spec: {why}")

    D = spec.head_size
    HK = spec.num_kv_heads
    GF = spec.gqa_fuse
    # Not spec.ept: the reduce has no key axis to trade lanes against, so it
    # always spreads D over the whole wave whatever d_lanes the segment used.
    EPT = spec.ept_reduce
    NS = spec.num_splits

    b = IRBuilder(spec.kernel_name("reduce"))
    b.kernel.attrs["max_workgroup_size"] = WAVE
    p = _declare_reduce_params(b, spec)

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_ns = b.const_i32(NS)
    zero_f = b.const_f32(0.0)
    neg_big = b.const_f32(_NEG_BIG)

    seq_idx = b.block_id_x()
    kv_head = b.block_id_y()
    tid = b.thread_id_x()
    d0 = b.mul(tid, b.const_i32(EPT))

    # Splits of one (request, kv_head) are GF apart within a split and
    # HK-major otherwise, so stepping the split index costs one add of a
    # compile-time stride.
    base_unit = b.mul(
        b.mul(b.add(b.mul(seq_idx, b.const_i32(HK)), kv_head), c_ns),
        b.const_i32(GF),
    )
    split_stride = b.const_i32(GF)

    o_row_base = b.add(
        b.mul(seq_idx, p["stride_o_seq"]),
        b.mul(b.mul(kv_head, b.const_i32(GF)), p["stride_o_head"]),
    )

    def _store(g: int, out: List[Value]) -> None:
        o_row = b.add(o_row_base, b.mul(b.const_i32(g), p["stride_o_head"]))
        b.global_store_vN(
            p["O"],
            b.add(o_row, d0),
            pack_f32_to(b, out, dtype=spec.dtype),
            EPT,
            align=EPT * 2,
        )

    for g in range(GF):
        slot0 = b.add(base_unit, b.const_i32(g))

        if spec.reduce_unroll:
            # Straight-line over the splits: every m, l and acc load is
            # independent, so all of them are in flight before the first
            # dependent exp2. The same guards as the loop form discard an
            # empty split's -1e30 sentinel.
            slots = [b.add(slot0, b.const_i32(s * GF)) for s in range(NS)]
            ms = [b.global_load_f32(p["ws_m"], sl) for sl in slots]
            ls = [b.global_load_f32(p["ws_l"], sl) for sl in slots]
            parts = [
                b.global_load_vN(
                    p["ws_acc"],
                    b.add(b.mul(sl, b.const_i32(D)), d0),
                    F32,
                    EPT,
                    align=EPT * 4,
                )
                for sl in slots
            ]
            overall_max = neg_big
            for mv in ms:
                overall_max = b.fmax(overall_max, mv)
            factors = [
                b.select(
                    b.fcmp("ogt", mv, neg_big),
                    b.exp2(b.fsub(mv, overall_max)),
                    zero_f,
                )
                for mv in ms
            ]
            den = zero_f
            for lv, fv in zip(ls, factors):
                den = b.fadd(den, b.fmul(lv, fv))
            inv_l = b.select(b.fcmp("oeq", den, zero_f), zero_f, b.rcp(den))
            row = [zero_f] * EPT
            for part, fv in zip(parts, factors):
                row = [b.fma(b.vec_extract(part, k), fv, row[k]) for k in range(EPT)]
            _store(g, [b.fmul(row[k], inv_l) for k in range(EPT)])
            continue

        def _slot(sv: Value) -> Value:
            return b.add(slot0, b.mul(sv, split_stride))

        # Three passes, not a streaming merge. A streaming online-softmax
        # merge computes 0 * inf -> NaN whenever every split for one
        # (request, head) was empty; the max/expsum/acc split lets each pass
        # discard the -1e30 sentinel with a select instead.
        # Every iter-arg name carries the g suffix: all GF fused heads share
        # one function scope, so a bare "mx" would be defined GF times and
        # LLVM rejects the duplicate local value.
        mx_loop = b.scf_for_iter(
            c0, c_ns, c1, [(f"mx{g}", neg_big)], iv_name=f"s_mx{g}"
        )
        with mx_loop as (sv, (mx,)):
            b.scf_yield(b.fmax(mx, b.global_load_f32(p["ws_m"], _slot(sv))))
        overall_max = mx_loop.results[0]

        sum_loop = b.scf_for_iter(
            c0, c_ns, c1, [(f"den{g}", zero_f)], iv_name=f"s_sum{g}"
        )
        with sum_loop as (sv, (den,)):
            slot = _slot(sv)
            ms = b.global_load_f32(p["ws_m"], slot)
            ls = b.global_load_f32(p["ws_l"], slot)
            factor = b.select(
                b.fcmp("ogt", ms, neg_big), b.exp2(b.fsub(ms, overall_max)), zero_f
            )
            b.scf_yield(b.fadd(den, b.fmul(ls, factor)))
        overall_expsum = sum_loop.results[0]
        inv_l = b.select(
            b.fcmp("oeq", overall_expsum, zero_f), zero_f, b.rcp(overall_expsum)
        )

        acc_args = [(f"r{g}_{k}", zero_f) for k in range(EPT)]
        acc_loop = b.scf_for_iter(c0, c_ns, c1, acc_args, iv_name=f"s_acc{g}")
        with acc_loop as (sv, running):
            slot = _slot(sv)
            ms = b.global_load_f32(p["ws_m"], slot)
            factor = b.select(
                b.fcmp("ogt", ms, neg_big), b.exp2(b.fsub(ms, overall_max)), zero_f
            )
            part = b.global_load_vN(
                p["ws_acc"],
                b.add(b.mul(slot, b.const_i32(D)), d0),
                F32,
                EPT,
                align=EPT * 4,
            )
            b.scf_yield(
                *[b.fma(b.vec_extract(part, k), factor, running[k]) for k in range(EPT)]
            )

        _store(g, [b.fmul(acc_loop.results[k], inv_l) for k in range(EPT)])

    b.ret()
    return b.kernel


# ---------------------------------------------------------------------------
# Host-side helpers
# ---------------------------------------------------------------------------


def paged_decode_segment_grid(cfg: PagedDecodeCfg, batch: int) -> Tuple[int, int, int]:
    return (batch, cfg.num_kv_heads * cfg.head_groups, cfg.num_splits)


def paged_decode_reduce_grid(cfg: PagedDecodeCfg, batch: int) -> Tuple[int, int, int]:
    return (batch, cfg.num_kv_heads, 1)


def paged_decode_workspace_shapes(cfg: PagedDecodeCfg, batch: int):
    """``(ws_m/ws_l shape, ws_acc shape)`` -- both f32, allocate once and cache."""
    ml = (batch, cfg.num_kv_heads, cfg.num_splits, cfg.gqa_fuse)
    return ml, ml + (cfg.head_size,)


# MEASURED, not derived from the CU count. A sweep of num_splits over
# batch x {1,2,4,8,16,32,64,128} found the optimum at a near-constant total
# phase-1 CTA count of ~256, and the shape of the curve is the same at every
# batch (us, min of 5 reps, Sk=4096, Hk=8):
#
#   B    s=1     s=2     s=4     s=8    s=16    s=32    s=64   best CTAs
#   1  1062.7   534.3   272.4   144.5    86.8    53.3    56.0   8*32 = 256
#   4  1157.1   607.4   389.6   315.1   339.0   405.0   420.0  32*8 = 256
#   8  1208.4   775.4   614.1   632.0   770.3   765.1   845.9  64*4 = 256
#  16  1558.5  1209.9  1240.7  1507.6  1450.4  1494.4  1661.4 128*2 = 256
#  32  2395.1  2451.9  2983.5  2717.5  2821.8  2931.8  3233.3 256*1 = 256
#
# 256 is ~6.4 CTAs per CU on this 40-CU part. The first cut of this heuristic
# targeted 40 -- one wave per CU -- and left half the bandwidth on the floor
# (~116 GB/s vs the ~220 GB/s ceiling at 256 CTAs), because a single wave
# cannot keep enough loads outstanding to cover DRAM latency.
#
# Re-measured after d_lanes moved to 16. The per-CTA optimum does drift down to
# ~128, because a wave that spends a quarter as many cycles on cross-lane
# reduces issues its loads sooner and needs less oversubscription to hide DRAM
# latency. It is still right to target 256: holding 256 costs at most 2.8%
# (at B=8) across B in {4, 8, 16, 32}, whereas targeting 128 costs 25% at B=1,
# where 8 base CTAs mean even 32 splits barely reaches 256.
_TARGET_CTAS = 256


def choose_num_splits(
    batch: int, num_kv_heads: int, *, max_splits: int = 32, target: int = _TARGET_CTAS
) -> int:
    """Smallest power of two that brings the phase-1 grid to ~``target`` CTAs.

    Split-K trades extra partial-write traffic and a longer reduce for
    occupancy, so it pays only until the grid is deep enough to hide memory
    latency and costs after that. Returns 1 once ``batch * num_kv_heads``
    already clears the target on its own -- that path has to stay fast, since
    the win this kernel is chasing is a high-batch win.
    """
    ctas = max(1, batch * num_kv_heads)
    if ctas >= target:
        return 1
    want = -(-target // ctas)
    n = 1
    while n < want and n < max_splits:
        n *= 2
    return min(n, max_splits)


# Where the long-context branch starts. The threshold is a cliff, not a taper:
# at B=1 the branch is worth +2.8% at Sk=16384 and costs **31%** at Sk=8192,
# because 8192 still fits the regime the plain heuristic was tuned for. Measured
# 2026-10-19, two independent interleaved runs agreeing to 0.4 points.
_HEADS_SPLIT_MIN_SEQLEN = 16384


def choose_decode_cfg(
    batch: int, max_seq_len: int, num_kv_heads: int, gqa_fuse: int
) -> Tuple[int, int]:
    """``(num_splits, heads_per_cta)`` for a decode step of this shape.

    ``heads_per_cta=1`` gives up the GQA fusion, so each kv head is read once
    per Q head instead of once per group. That is only affordable when the
    duplicate reads land in MALL, and only worth it when there are too few base
    CTAs to fill the machine any other way -- which is the single-request
    long-context case. At ``batch >= 4`` the base grid is already wide and
    splitting the fusion costs ~13% at B=16, so it stays gated on ``batch == 1``.

    ``num_splits`` then falls out of the same ~256-CTA target as always, with
    the head axis counted in, so the two knobs cannot both inflate the grid.
    That coupling is doing most of the work: at B=1 the resulting drop from 32
    splits to 8 is worth +2.8%/+1.4% at Sk=16384/32768 on its own, and giving up
    the fusion adds a further +0.5-0.9%. Both effects wash out by Sk=65536,
    where every arrangement that reaches the CTA target measures the same.
    """
    split_heads = batch == 1 and max_seq_len >= _HEADS_SPLIT_MIN_SEQLEN
    heads_per_cta = 1 if split_heads else 0
    head_groups = gqa_fuse if split_heads else 1
    return choose_num_splits(batch, num_kv_heads * head_groups), heads_per_cta
