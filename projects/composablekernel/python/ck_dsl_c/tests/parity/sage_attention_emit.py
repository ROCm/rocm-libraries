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
import sys

from ck_dsl.instances.common.sage_attention import (
    SageAttentionSpec,
    build_sage_attention,
)
from ck_dsl.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from ck_dsl.helpers.qk_scale import QkScaleSpec
from ck_dsl import lower_kernel_to_llvm


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
    if len(sys.argv) < 2:
        sys.stderr.write("usage: sage_attention_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    spec = _spec(idx)
    kernel = build_sage_attention(spec, arch="gfx950")
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
