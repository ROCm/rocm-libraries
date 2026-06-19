#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_varlen_emit.py -- Python reference emitter for the
# varlen-FMHA-fwd parity harness. Selects one of 6 sampled FmhaFwdVarlenSpec
# configs by argv[1] (0..5), builds via build_fmha_fwd_varlen and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared
# with the C emitter fmha_varlen_emit.c.
import sys

from ck_dsl.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from ck_dsl.instances.common.fmha_varlen import (
    FmhaFwdVarlenSpec,
    build_fmha_fwd_varlen,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> FmhaFwdVarlenSpec:
    if idx == 0:
        return FmhaFwdVarlenSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8),
                dtype="f16",
                mask_mode="none",
            ),
            max_seqlen_q=128,
            max_seqlen_k=256,
            batch=4,
        )
    if idx == 1:
        return FmhaFwdVarlenSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=8, num_kv_heads=8),
                dtype="bf16",
                mask_mode="causal",
            ),
            max_seqlen_q=256,
            max_seqlen_k=512,
            batch=2,
        )
    if idx == 2:
        return FmhaFwdVarlenSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=128, num_query_heads=16, num_kv_heads=4),
                dtype="f16",
                mask_mode="none",
            ),
            max_seqlen_q=64,
            max_seqlen_k=128,
            batch=8,
        )
    if idx == 3:
        return FmhaFwdVarlenSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=256, num_query_heads=4, num_kv_heads=4),
                dtype="bf16",
                mask_mode="sliding_window",
                sliding_window=64,
            ),
            max_seqlen_q=512,
            max_seqlen_k=512,
            batch=1,
        )
    if idx == 4:
        return FmhaFwdVarlenSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=64, num_query_heads=12, num_kv_heads=12),
                dtype="fp16",
                mask_mode="none",
            ),
            max_seqlen_q=256,
            max_seqlen_k=256,
            batch=16,
        )
    if idx == 5:
        return FmhaFwdVarlenSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(head_size=192, num_query_heads=12, num_kv_heads=12),
                dtype="f16",
                mask_mode="causal",
            ),
            max_seqlen_q=192,
            max_seqlen_k=384,
            batch=8,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: fmha_varlen_emit.py <config_index 0..5>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_fmha_fwd_varlen(spec, arch="gfx950")
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
