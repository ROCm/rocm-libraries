#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Python reference for the gfx950 dense attention builder. Each index selects
# a spec and runs the shared parity driver: LLVM (`ll`), serialized IR (`ir`),
# verifier diagnostics (`verify`) or HIP source identity (`hip-case`).
# For example, from rocke/:
#   PYTHONPATH=library:platform/python python3 library/tests/parity/attention_dense_gfx950_emit.py 72 ir
#
# Keep indices stable: the native emitter will need the same input at each
# index. A successful Python emission alone is not a byte-identity comparison.
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    build_attention_dense,
)

# Shared base problem: GQA-4, D128, bf16, top-left causal, tile-aligned lengths.
_BASE = dict(
    batch=1,
    seqlen_q=512,
    seqlen_kv=512,
    num_query_heads=8,
    num_kv_heads=2,
    head_size=128,
    dtype="bf16",
    causal=True,
)
_PERSIST = dict(persistent=True, num_persistent=256)
_WIDE = dict(persistent=True, num_persistent=256, wide_lds_dma=True)
# IGLP owns the loop schedule, so the manual fence and template must be off.
_IGLP_OFF_MANUAL = dict(pv_sched_fence=False, pv_sched_group_template=False)
# A query block whose window ends before the keys start: rows with no key.
_EMPTY_WINDOW = dict(seqlen_q=256, seqlen_kv=64, sliding_window=64)
_PAGED = dict(
    paged=True,
    block_size=16,
    num_kv_blocks=64,
    sliding_window=256,
)

_CONFIGS = (
    # ---- grid body: problem shape and dtype ----
    {},  # 0: default (runtime-shape cohort)
    dict(causal=False, dtype="fp16"),  # 1
    dict(head_size=64),  # 2: D64 (K group pad)
    dict(head_size=64, dtype="fp16", causal=False, num_query_heads=4, num_kv_heads=4),
    dict(num_kv_heads=1),  # 4: MQA
    # 5: batch/lengths are runtime arguments here; the window size is specialized.
    dict(batch=4, seqlen_q=1024, seqlen_kv=1024, sliding_window=512),
    # ---- grid body: tile geometry ----
    dict(block_m=128),  # 6
    dict(block_n=128),  # 7
    dict(head_size=64, block_n=32),  # 8
    dict(block_m=128, block_n=32, head_size=64),  # 9
    # ---- grid body: band ----
    dict(sliding_window=256),  # 10
    dict(causal=False, sliding_window=128),  # 11: left-only window
    dict(right_bound=64),  # 12: lookahead
    dict(sliding_window=256, right_bound=64),  # 13: two-sided band
    dict(seqlen_q=256, causal_bottom_right=True),  # 14: moving diagonal
    dict(causal_bottom_right=True),  # 15: equal lengths, unshifted
    dict(**_EMPTY_WINDOW),  # 16
    # ---- grid body: ragged (k-bound mask) ----
    dict(seqlen_q=500, seqlen_kv=500, ragged=True),  # 17
    dict(seqlen_q=300, seqlen_kv=500, ragged=True, causal=False),  # 18
    dict(seqlen_q=200, seqlen_kv=500, ragged=True, causal_bottom_right=True),  # 19
    dict(seqlen_q=500, seqlen_kv=500, ragged=True, head_size=64, dtype="fp16"),
    # ---- grid body: sinks / LSE ----
    dict(use_sinks=True),  # 21
    dict(use_sinks=True, **_EMPTY_WINDOW),  # 22
    dict(emit_lse=True),  # 23
    dict(emit_lse=True, use_sinks=True, **_EMPTY_WINDOW),  # 24
    dict(emit_lse=True, seqlen_q=500, seqlen_kv=500, ragged=True),  # 25
    dict(emit_lse=True, causal=False, head_size=64, dtype="fp16"),  # 26
    # ---- grid body: varlen / paged ----
    dict(varlen=True, batch=2),  # 27
    dict(varlen=True, batch=2, emit_lse=True),  # 28
    dict(**_PAGED),  # 29
    dict(emit_lse=True, **_PAGED),  # 30
    # ---- grid body: codegen knobs ----
    dict(o_store_width=2),  # 31
    dict(o_store_width=1),  # 32
    dict(lazy_rescale=False),  # 33
    dict(lazy_rescale_threshold=4.0),  # 34
    dict(use_exp2_fast=False),  # 35
    dict(exp_per_pv_step=2),  # 36
    dict(partial_vmcnt_prefetch=False),  # 37
    dict(pv_priority=0),  # 38
    dict(pv_priority=3),  # 39
    dict(pv_sched_fence=False),  # 40
    dict(pv_sched_fence_mask=0x8),  # 41
    dict(pv_sched_group_template=False),  # 42
    dict(pv_sched_group_ds_read=4),  # 43
    dict(iglp_mode=1, **_IGLP_OFF_MANUAL),  # 44
    dict(iglp_mode=0, **_IGLP_OFF_MANUAL),  # 45
    dict(pv_loop_order="k_major"),  # 46
    dict(causal_diag_split=False),  # 47
    dict(lds_k_row_pad=16, lds_v_row_pad=16),  # 48
    dict(head_size=64, lds_k_group_pad=0),  # 49
    dict(waves_per_eu=1),  # 50
    # ---- persistent body ----
    dict(**_PERSIST),  # 51: persistent auto decode
    dict(persist_decode="hkv_major", **_PERSIST),  # 52
    dict(persistent=True, num_persistent=4, persist_decode="gqa_pair"),  # 53
    dict(persistent=True, num_persistent=8, persist_decode="gqa_pair_2phase"),
    dict(interleave=True, **_PERSIST),  # 55
    dict(causal=False, dtype="fp16", **_PERSIST),  # 56
    dict(sliding_window=256, **_PERSIST),  # 57
    dict(right_bound=64, **_PERSIST),  # 58
    dict(use_sinks=True, emit_lse=True, **_PERSIST),  # 59
    dict(use_sinks=True, emit_lse=True, **_EMPTY_WINDOW, **_PERSIST),  # 60
    dict(seqlen_q=500, seqlen_kv=500, ragged=True, **_PERSIST),  # 61
    dict(head_size=64, dtype="fp16", **_PERSIST),  # 62
    dict(block_m=128, **_PERSIST),  # 63
    dict(lazy_rescale=False, **_PERSIST),  # 64
    dict(exp_per_pv_step=2, **_PERSIST),  # 65
    dict(pv_loop_order="k_major", o_store_width=2, **_PERSIST),  # 66
    # ---- persistent body, wide LDS DMA ----
    dict(**_WIDE),  # 67: iglp 1, k_major by default
    dict(emit_lse=True, use_sinks=True, **_WIDE),  # 68
    dict(causal=False, **_WIDE),  # 69
    dict(pv_loop_order="d_major", iglp_mode=0, **_WIDE),  # 70
    dict(iglp_mode=-1, pv_sched_fence=True, **_WIDE),  # 71
    # Aligned grid cases: exercise shipped features without relying on how a
    # future builder handles the last partial tile (Sq = Sk = 512).
    dict(kv_segments=2),  # 72: per-segment output, max, and exp-sum workspaces
    dict(kv_storage_dtype="fp8e4m3"),  # 73: FP8 K/V pointers and two dequant scales
    dict(_PAGED, head_size=64, sliding_window=0),  # 74: D64 paged, plain causal
    dict(nonpersist_decode="hq_minor_swz"),  # 75: ordinary grid
    dict(persist_decode="hq_minor_swz", **_PERSIST),  # 76: persistent grid
)


def spec_for_index(idx: int) -> Gfx950AttentionDenseSpec:
    if not 0 <= idx < len(_CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    return Gfx950AttentionDenseSpec(**{**_BASE, **_CONFIGS[idx]})


def main() -> int:
    from _emit_common import run_emit

    return run_emit(spec_for_index, build_attention_dense)


if __name__ == "__main__":
    raise SystemExit(main())
