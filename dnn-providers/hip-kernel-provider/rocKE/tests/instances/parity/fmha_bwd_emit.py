#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_bwd_emit.py -- Python reference emitter for the
# FMHA-backward parity harness. Selects one of 6 sampled FmhaBwdSpec configs by
# argv[1] (0..5), builds via build_fmha_bwd and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter fmha_bwd_emit.c.
from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.instances.common.fmha_bwd import FmhaBwdSpec, build_fmha_bwd
from _emit_common import run_emit


def _spec(idx: int) -> FmhaBwdSpec:
    if idx == 0:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=4, num_kv_heads=4),
                dtype="f16",
                mask_mode="none",
            ),
            seqlen_q=16,
            seqlen_k=16,
        )
    if idx == 1:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=4),
                dtype="f16",
                mask_mode="none",
            ),
            seqlen_q=32,
            seqlen_k=64,
        )
    if idx == 2:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=256, num_query_heads=16, num_kv_heads=8),
                dtype="bf16",
                mask_mode="causal",
            ),
            seqlen_q=64,
            seqlen_k=64,
        )
    if idx == 3:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=2, num_kv_heads=2),
                dtype="f16",
                mask_mode="sliding_window",
                sliding_window=16,
            ),
            seqlen_q=128,
            seqlen_k=128,
        )
    if idx == 4:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=192, num_query_heads=8, num_kv_heads=2),
                dtype="bf16",
                mask_mode="none",
            ),
            seqlen_q=256,
            seqlen_k=256,
        )
    if idx == 5:
        return FmhaBwdSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=1, num_kv_heads=1),
                dtype="f16",
                mask_mode="causal",
            ),
            seqlen_q=128,
            seqlen_k=256,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_fmha_bwd,
        usage="usage: fmha_bwd_emit.py <config_index 0..5>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
