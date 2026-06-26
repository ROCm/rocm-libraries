#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_head_grouping_emit.py -- Python reference emitter for the
# GQA/MQA head-grouped FMHA forward parity harness. Selects one of 6 sampled
# configs by argv[1] (the config index 0..5), builds the FmhaFwdHeadGroupingSpec,
# builds the kernel via build_fmha_fwd_head_grouping(arch="gfx950") and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter fmha_head_grouping_emit.c.
from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.instances.common.fmha_head_grouping import (
    FmhaFwdHeadGroupingSpec,
    build_fmha_fwd_head_grouping,
)
from _emit_common import run_emit


def _spec(idx: int) -> FmhaFwdHeadGroupingSpec:
    if idx == 0:
        return FmhaFwdHeadGroupingSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=32, num_kv_heads=8),
                dtype="f16",
                mask_mode="none",
            ),
            seqlen_q=16,
            seqlen_k=16,
        )
    if idx == 1:
        return FmhaFwdHeadGroupingSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=16, num_kv_heads=1),
                dtype="f16",
                mask_mode="causal",
            ),
            seqlen_q=32,
            seqlen_k=32,
        )
    if idx == 2:
        return FmhaFwdHeadGroupingSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=4),
                dtype="bf16",
                mask_mode="none",
            ),
            seqlen_q=32,
            seqlen_k=32,
        )
    if idx == 3:
        return FmhaFwdHeadGroupingSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=256, num_query_heads=32, num_kv_heads=4),
                dtype="f16",
                mask_mode="sliding_window",
                sliding_window=512,
            ),
            seqlen_q=16,
            seqlen_k=16,
        )
    if idx == 4:
        return FmhaFwdHeadGroupingSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=2),
                dtype="bf16",
                mask_mode="none",
            ),
            seqlen_q=48,
            seqlen_k=64,
        )
    if idx == 5:
        return FmhaFwdHeadGroupingSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=24, num_kv_heads=8),
                dtype="f16",
                mask_mode="causal",
            ),
            seqlen_q=64,
            seqlen_k=64,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_fmha_fwd_head_grouping,
        usage="usage: fmha_head_grouping_emit.py <config_index 0..5>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
