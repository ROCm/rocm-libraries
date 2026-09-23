# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Audit HIP debug lowering parity for attention kernels (library layer).

This module is the library-layer companion to
``rocke.examples.common.hip_lowering_parity`` (platform layer).  It provides
``_attention_cases(arch)`` which builds ``Case`` instances for all FMHA /
Sage / Sparse / UnifiedAttention kernel variants imported from the ``kernels``
package.

Platform code must NOT import from ``kernels``; this module isolates those
imports in the library layer.  It imports ``Case``, ``AuditResult``,
``_selected``, ``_compile_hip_source``, and ``audit_cases`` from the platform
module (library → platform is legal per the one-way layering rule).

Usage::

    python -m builders.common.hip_lowering_attention_parity
    python -m builders.common.hip_lowering_attention_parity --compile-hip
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from rocke.helpers import QkScaleSpec  # noqa: E402
from kernels import (  # noqa: E402
    FmhaAppendKvSpec,
    FmhaBwdSpec,
    FmhaCommonSpec,
    FmhaFwdFp8Spec,
    FmhaFwdHeadGroupingSpec,
    FmhaFwdPagedPrefillSpec,
    FmhaFwdSplitKvDecodeSpec,
    FmhaFwdVarlenSpec,
    FmhaShape,
    JengaSparseSpec,
    SageAttentionSpec,
    UnifiedAttention2DTiledSpec,
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    VsaSparseSpec,
    build_fmha_bwd,
    build_fmha_fwd_appendkv,
    build_fmha_fwd_fp8,
    build_fmha_fwd_head_grouping,
    build_fmha_fwd_paged_prefill,
    build_fmha_fwd_splitkv_decode_reduce,
    build_fmha_fwd_splitkv_decode_segment,
    build_fmha_fwd_varlen,
    build_jenga_sparse_attention,
    build_sage_attention,
    build_unified_attention_2d_tiled,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
    build_vsa_sparse_attention,
)

# Import shared audit infrastructure from platform (library→platform is legal).
from rocke.examples.common.hip_lowering_parity import (  # noqa: E402
    Case,
    AuditResult,
    _selected,
    _compile_hip_source,
    audit_cases,
)


def _fmha_common(dtype: str = "f16") -> FmhaCommonSpec:
    return FmhaCommonSpec(
        FmhaShape(
            head_size=128,
            num_query_heads=4,
            num_kv_heads=2,
            block_size_q=16,
            block_size_k=64,
        ),
        dtype=dtype,
    )


def _attention_cases(arch: str = "gfx950") -> List[Case]:
    """Return all attention ``Case`` instances for the given arch.

    These are the cases that were previously gated by
    ``make_cases(include_attention=True)`` in the platform harness.
    """
    common = _fmha_common()
    common_bf16 = _fmha_common("bf16")
    qks = QkScaleSpec(layout="per_head", stride_batch=4, stride_head=1)

    cases: List[Case] = []
    cases.extend(
        [
            Case(
                "fmha_varlen",
                "attention",
                lambda: build_fmha_fwd_varlen(
                    FmhaFwdVarlenSpec(
                        common=common, max_seqlen_q=64, max_seqlen_k=64, batch=2
                    )
                ),
            ),
            Case(
                "fmha_appendkv",
                "attention",
                lambda: build_fmha_fwd_appendkv(
                    FmhaAppendKvSpec(common=common, batch=2)
                ),
            ),
            Case(
                "fmha_paged_prefill",
                "attention",
                lambda: build_fmha_fwd_paged_prefill(
                    FmhaFwdPagedPrefillSpec(
                        common=common, page_block_size=16, max_blocks_per_seq=8, batch=2
                    )
                ),
            ),
            Case(
                "fmha_splitkv_segment",
                "attention",
                lambda: build_fmha_fwd_splitkv_decode_segment(
                    FmhaFwdSplitKvDecodeSpec(common=common, batch=2, num_segments=4)
                ),
            ),
            Case(
                "fmha_splitkv_reduce",
                "attention",
                lambda: build_fmha_fwd_splitkv_decode_reduce(
                    FmhaFwdSplitKvDecodeSpec(common=common, batch=2, num_segments=4)
                ),
            ),
            Case(
                "fmha_head_grouping",
                "attention",
                lambda: build_fmha_fwd_head_grouping(
                    FmhaFwdHeadGroupingSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "fmha_bwd",
                "attention",
                lambda: build_fmha_bwd(
                    FmhaBwdSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "fmha_fwd_fp8",
                "attention",
                lambda: build_fmha_fwd_fp8(
                    FmhaFwdFp8Spec(
                        common=common, kv_dtype="fp8e4m3", seqlen_q=16, seqlen_k=64
                    )
                ),
            ),
            Case(
                "sage_fp16_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="fp16_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "sage_fp8_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="fp8_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "jenga_sparse",
                "attention",
                lambda: build_jenga_sparse_attention(
                    JengaSparseSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "vsa_sparse",
                "attention",
                lambda: build_vsa_sparse_attention(
                    VsaSparseSpec(common=common, seqlen_q=64, seqlen_k=64)
                ),
            ),
            Case(
                "uattn_2d_tiled",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                    )
                ),
            ),
            Case(
                "uattn_2d_tiled_bf16",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="bf16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                    )
                ),
            ),
            Case(
                "uattn_3d_tiled",
                "attention",
                lambda: build_unified_attention_3d_tiled(
                    UnifiedAttention3DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                        num_segments=4,
                    )
                ),
            ),
            Case(
                "uattn_reduce_tiled",
                "attention",
                lambda: build_unified_attention_reduce_tiled(
                    UnifiedAttentionReduceTiledSpec(
                        head_size=128,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        num_segments=4,
                    )
                ),
            ),
        ]
    )
    cases.extend(
        [
            Case(
                "fmha_varlen.bf16",
                "attention",
                lambda: build_fmha_fwd_varlen(
                    FmhaFwdVarlenSpec(
                        common=common_bf16, max_seqlen_q=64, max_seqlen_k=64, batch=2
                    )
                ),
            ),
            Case(
                "fmha_fwd_fp8.bf8",
                "attention",
                lambda: build_fmha_fwd_fp8(
                    FmhaFwdFp8Spec(
                        common=common, kv_dtype="bf8e5m2", seqlen_q=16, seqlen_k=64
                    )
                ),
            ),
            Case(
                "sage_i8_fp8_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="i8_fp8_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "sage_i4_fp8_bf16",
                "attention",
                lambda: build_sage_attention(
                    SageAttentionSpec(
                        common=common,
                        quant_mode="i4_fp8_bf16",
                        q_scale=qks,
                        k_scale=qks,
                        seqlen_q=64,
                        seqlen_k=64,
                    )
                ),
            ),
            Case(
                "uattn_2d_tiled_softcap",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=True,
                    )
                ),
            ),
            Case(
                "uattn_2d_tiled_alibi",
                "attention",
                lambda: build_unified_attention_2d_tiled(
                    UnifiedAttention2DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="fp16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                        use_alibi=True,
                    )
                ),
            ),
            Case(
                "uattn_3d_tiled_bf16",
                "attention",
                lambda: build_unified_attention_3d_tiled(
                    UnifiedAttention3DTiledSpec(
                        head_size=128,
                        block_size=64,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="bf16",
                        use_sinks=False,
                        sliding_window=0,
                        has_softcap=False,
                        num_segments=4,
                    )
                ),
            ),
            Case(
                "uattn_reduce_tiled_bf16",
                "attention",
                lambda: build_unified_attention_reduce_tiled(
                    UnifiedAttentionReduceTiledSpec(
                        head_size=128,
                        num_query_heads=4,
                        num_kv_heads=2,
                        dtype="bf16",
                        num_segments=4,
                    )
                ),
            ),
        ]
    )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="all", help="substring or group filter")
    parser.add_argument(
        "--compile-hip", action="store_true", help="compile HIP source to HSACO"
    )
    parser.add_argument("--compile-timeout-s", type=int, default=180)
    parser.add_argument("--arch", default="gfx950")
    parser.add_argument("--emit-dir", type=Path, default=None)
    args = parser.parse_args()

    cases = _selected(_attention_cases(arch=args.arch), args.case)
    if not cases:
        print(f"no cases matched {args.case!r}")
        return 2

    results = audit_cases(
        cases,
        compile_hip=args.compile_hip,
        arch=args.arch,
        emit_dir=args.emit_dir,
        compile_timeout_s=args.compile_timeout_s,
    )
    for r in results:
        compile_status = ""
        if args.compile_hip:
            compile_status = f" hipcc={'OK' if r.hip_compile_ok else 'FAIL'}"
        status = "OK" if r.ok else "FAIL"
        print(
            f"{status:4} {r.group:9} {r.name:28} "
            f"llvm={'OK' if r.llvm_ok else 'FAIL'} hip={'OK' if r.hip_ok else 'FAIL'} "
            f"chars={r.hip_chars}{compile_status}"
        )
        if r.error:
            print(f"     {r.error}")

    failures = [r for r in results if not r.ok]
    print(
        f"SUMMARY total={len(results)} ok={len(results) - len(failures)} fail={len(failures)}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
