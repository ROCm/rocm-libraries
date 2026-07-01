#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/sage_attention_emit.py -- Python reference emitter for the Sage
# attention forward (instance_sage_attention) parity harness. Selects one of the
# sampled configs by argv[1] (the config index 0..5), builds the
# SageAttentionSpec, builds the kernel via build_sage_attention(arch='gfx950')
# and prints lower_kernel_to_llvm(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter sage_attention_emit.c.
from rocke.instances.common.sage_attention import (
    SageAttentionSpec,
    build_sage_attention,
)
from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.helpers.qk_scale import QkScaleSpec
from _emit_common import run_emit


def _spec(idx: int) -> SageAttentionSpec:
    if idx == 0:
        shape = FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8)
        common = FmhaCommonSpec(shape=shape, dtype="f16", mask_mode="none")
        qs = QkScaleSpec(
            layout="per_block",
            scale_block=16,
            stride_batch=128,
            stride_head=8,
            stride_block=1,
        )
        ks = QkScaleSpec(
            layout="per_block",
            scale_block=64,
            stride_batch=128,
            stride_head=8,
            stride_block=1,
        )
        return SageAttentionSpec(
            common=common,
            quant_mode="fp16_bf16",
            q_scale=qs,
            k_scale=ks,
            seqlen_q=16,
            seqlen_k=64,
        )
    if idx == 1:
        shape = FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8)
        common = FmhaCommonSpec(shape=shape, dtype="bf16", mask_mode="none")
        qs = QkScaleSpec(
            layout="per_block",
            scale_block=16,
            stride_batch=128,
            stride_head=8,
            stride_block=1,
        )
        ks = QkScaleSpec(
            layout="per_block",
            scale_block=64,
            stride_batch=128,
            stride_head=8,
            stride_block=1,
        )
        return SageAttentionSpec(
            common=common,
            quant_mode="fp8_bf16",
            q_scale=qs,
            k_scale=ks,
            seqlen_q=16,
            seqlen_k=64,
        )
    if idx == 2:
        shape = FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8)
        common = FmhaCommonSpec(shape=shape, dtype="f16", mask_mode="none")
        qs = QkScaleSpec(
            layout="per_head",
            scale_block=0,
            stride_batch=8,
            stride_head=1,
            stride_block=1,
        )
        ks = QkScaleSpec(
            layout="per_head",
            scale_block=0,
            stride_batch=8,
            stride_head=1,
            stride_block=1,
        )
        return SageAttentionSpec(
            common=common,
            quant_mode="i8_fp8_bf16",
            q_scale=qs,
            k_scale=ks,
            seqlen_q=16,
            seqlen_k=64,
        )
    if idx == 3:
        shape = FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=8)
        common = FmhaCommonSpec(shape=shape, dtype="bf16", mask_mode="none")
        qs = QkScaleSpec(
            layout="per_block",
            scale_block=16,
            stride_batch=128,
            stride_head=8,
            stride_block=1,
        )
        ks = QkScaleSpec(
            layout="per_block",
            scale_block=64,
            stride_batch=128,
            stride_head=8,
            stride_block=1,
        )
        return SageAttentionSpec(
            common=common,
            quant_mode="i4_fp8_bf16",
            q_scale=qs,
            k_scale=ks,
            seqlen_q=32,
            seqlen_k=128,
        )
    if idx == 4:
        shape = FmhaShape(head_size=256, num_query_heads=16, num_kv_heads=8)
        common = FmhaCommonSpec(shape=shape, dtype="f16", mask_mode="causal")
        qs = QkScaleSpec(
            layout="per_block",
            scale_block=32,
            stride_batch=256,
            stride_head=16,
            stride_block=1,
        )
        ks = QkScaleSpec(
            layout="per_block",
            scale_block=64,
            stride_batch=256,
            stride_head=16,
            stride_block=1,
        )
        return SageAttentionSpec(
            common=common,
            quant_mode="fp16_bf16",
            q_scale=qs,
            k_scale=ks,
            seqlen_q=64,
            seqlen_k=64,
        )
    if idx == 5:
        shape = FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=8)
        common = FmhaCommonSpec(shape=shape, dtype="bf16", mask_mode="none")
        qs = QkScaleSpec(
            layout="per_head",
            scale_block=0,
            stride_batch=8,
            stride_head=1,
            stride_block=1,
        )
        ks = QkScaleSpec(
            layout="per_head",
            scale_block=0,
            stride_batch=8,
            stride_head=1,
            stride_block=1,
        )
        return SageAttentionSpec(
            common=common,
            quant_mode="fp8_bf16",
            q_scale=qs,
            k_scale=ks,
            seqlen_q=32,
            seqlen_k=128,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_sage_attention,
        usage="usage: sage_attention_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
