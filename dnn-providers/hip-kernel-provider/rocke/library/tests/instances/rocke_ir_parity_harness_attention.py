# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# rocke_ir_parity_harness_attention.py -- Attention cases for the IR parity
# golden gate.
#
# Extracted from platform/tests/instances/rocke_ir_parity_harness.py so that
# the platform harness has zero library (kernels/builders/dispatch) imports.
# The platform harness merges these cases at runtime via a guarded import.

from __future__ import annotations


def attn_problem(**kw):
    from kernels.common.attention_unified import UnifiedAttentionProblem

    return UnifiedAttentionProblem(**kw)


def build_attn_2d(name, pkw):
    def _build():
        from kernels.common.attention_unified import (
            UnifiedAttention2DSpec,
            build_unified_attention_2d,
        )

        return build_unified_attention_2d(
            UnifiedAttention2DSpec(attn_problem(**pkw), name=name)
        )

    return _build


def build_attn_3d(name, pkw, segs):
    def _build():
        from kernels.common.attention_unified import (
            UnifiedAttention3DSpec,
            build_unified_attention_3d,
        )

        return build_unified_attention_3d(
            UnifiedAttention3DSpec(attn_problem(**pkw), name=name, num_segments=segs)
        )

    return _build


def build_attn_reduce(name, pkw, segs):
    def _build():
        from kernels.common.attention_unified import (
            UnifiedAttentionReduceSpec,
            build_unified_attention_reduce,
        )

        return build_unified_attention_reduce(
            UnifiedAttentionReduceSpec(
                attn_problem(**pkw), num_segments=segs, name=name
            )
        )

    return _build


def attention_cases(add):
    """Add the 6 unified_attention IR parity cases via *add*.

    *add* is the same callable used in the platform harness::

        def add(family, case_id, arch, build): ...

    Returns the list of case dicts (same as the platform harness convention)
    for callers that prefer to collect rather than side-effect.
    """
    collected: list = []

    def _add(family, case_id, arch, build):
        entry = {"family": family, "case_id": case_id, "arch": arch, "build": build}
        collected.append(entry)
        add(family, case_id, arch, build)

    p_decode = dict(
        total_q=4,
        num_seqs=4,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=64,
        dtype="fp16",
    )
    p_prefill = dict(
        total_q=64,
        num_seqs=2,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=128,
        block_size=16,
        max_seqlen_q=32,
        max_seqlen_k=128,
        dtype="bf16",
        sliding_window=32,
        softcap=10.0,
        use_sinks=True,
    )
    _add(
        "unified_attention",
        "ua/gfx942/2d_decode_fp16",
        "gfx942",
        build_attn_2d("irhash_ua_2d_decode", p_decode),
    )
    _add(
        "unified_attention",
        "ua/gfx950/2d_prefill_bf16_sw",
        "gfx950",
        build_attn_2d("irhash_ua_2d_prefill", p_prefill),
    )
    _add(
        "unified_attention",
        "ua/gfx1151/2d_decode_fp16",
        "gfx1151",
        build_attn_2d("irhash_ua_2d_decode", p_decode),
    )
    _add(
        "unified_attention",
        "ua/gfx950/3d_prefill",
        "gfx950",
        build_attn_3d("irhash_ua_3d_prefill", p_prefill, 8),
    )
    _add(
        "unified_attention",
        "ua/gfx942/reduce_prefill",
        "gfx942",
        build_attn_reduce("irhash_ua_reduce_prefill", p_prefill, 8),
    )
    _add(
        "unified_attention",
        "ua/gfx1201/reduce_decode",
        "gfx1201",
        build_attn_reduce("irhash_ua_reduce_decode", p_decode, 4),
    )
    return collected


def cases():
    """Return the attention IR-parity case dicts (family/case_id/arch/build).

    Mirrors the platform harness `cases()` contract so the shared golden
    machinery (`rocke_ir_parity_harness.run/build_golden/check_golden`) can be
    driven over the attention cases via its `cases_fn` hook.
    """
    return attention_cases(lambda *a, **k: None)
