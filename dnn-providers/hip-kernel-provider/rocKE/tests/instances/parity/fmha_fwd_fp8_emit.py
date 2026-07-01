#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_fwd_fp8_emit.py -- Python reference emitter for the FP8 FMHA
# forward parity harness. Selects one of 6 sampled configs by argv[1] (the config
# index 0..5), builds the FmhaFwdFp8Spec, builds the kernel via
# build_fmha_fwd_fp8(arch="gfx950") and prints lower_kernel_to_llvm(arch='gfx950')
# to stdout so it can be byte-compared with the C emitter fmha_fwd_fp8_emit.c.
from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.instances.common.fmha_fwd_fp8 import (
    FmhaFwdFp8Spec,
    build_fmha_fwd_fp8,
)
from _emit_common import run_emit


def _spec(idx: int) -> FmhaFwdFp8Spec:
    if idx == 0:
        return FmhaFwdFp8Spec(
            common=FmhaCommonSpec(
                shape=FmhaShape(
                    head_size=64,
                    num_query_heads=4,
                    num_kv_heads=4,
                    block_size_q=16,
                    block_size_k=64,
                ),
                dtype="f16",
                mask_mode="none",
            ),
            kv_dtype="fp8e4m3",
            seqlen_q=16,
            seqlen_k=64,
        )
    if idx == 1:
        return FmhaFwdFp8Spec(
            common=FmhaCommonSpec(
                shape=FmhaShape(
                    head_size=64,
                    num_query_heads=2,
                    num_kv_heads=2,
                    block_size_q=16,
                    block_size_k=64,
                ),
                dtype="f16",
                mask_mode="causal",
            ),
            kv_dtype="fp8e4m3",
            seqlen_q=32,
            seqlen_k=128,
        )
    if idx == 2:
        return FmhaFwdFp8Spec(
            common=FmhaCommonSpec(
                shape=FmhaShape(
                    head_size=128,
                    num_query_heads=8,
                    num_kv_heads=4,
                    block_size_q=16,
                    block_size_k=64,
                ),
                dtype="bf16",
                mask_mode="none",
            ),
            kv_dtype="bf8e5m2",
            seqlen_q=16,
            seqlen_k=64,
        )
    if idx == 3:
        return FmhaFwdFp8Spec(
            common=FmhaCommonSpec(
                shape=FmhaShape(
                    head_size=256,
                    num_query_heads=4,
                    num_kv_heads=1,
                    block_size_q=16,
                    block_size_k=64,
                ),
                dtype="f16",
                mask_mode="sliding_window",
                sliding_window=32,
            ),
            kv_dtype="fp8e4m3",
            seqlen_q=48,
            seqlen_k=256,
            waves_per_eu=4,
        )
    if idx == 4:
        return FmhaFwdFp8Spec(
            common=FmhaCommonSpec(
                shape=FmhaShape(
                    head_size=32,
                    num_query_heads=16,
                    num_kv_heads=16,
                    block_size_q=16,
                    block_size_k=64,
                ),
                dtype="f16",
                mask_mode="none",
            ),
            kv_dtype="fp8e4m3",
            seqlen_q=64,
            seqlen_k=256,
        )
    if idx == 5:
        return FmhaFwdFp8Spec(
            common=FmhaCommonSpec(
                shape=FmhaShape(
                    head_size=64,
                    num_query_heads=8,
                    num_kv_heads=2,
                    block_size_q=16,
                    block_size_k=64,
                ),
                dtype="bf16",
                mask_mode="none",
            ),
            kv_dtype="bf8e5m2",
            seqlen_q=32,
            seqlen_k=512,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_fmha_fwd_fp8,
        usage="usage: fmha_fwd_fp8_emit.py <config_index 0..5>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
