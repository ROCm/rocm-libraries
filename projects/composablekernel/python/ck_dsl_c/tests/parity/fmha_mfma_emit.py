#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fmha_mfma_emit.py -- Python reference emitter for the tiled
# FMHA-forward (fmha_mfma) instance parity harness. Selects one of the sampled
# configs by argv[1] (the config index), builds the FmhaMfmaSpec, builds the
# kernel via build_fmha_fwd_mfma(arch='gfx950') and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter fmha_mfma_emit.c.
import sys

from ck_dsl.instances.common.fmha_mfma import FmhaMfmaSpec, build_fmha_fwd_mfma
from ck_dsl.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


def _spec(idx: int) -> FmhaMfmaSpec:
    if idx == 0:
        shape = FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8)
        common = FmhaCommonSpec(
            shape=shape, dtype="f16", mask_mode="none", sliding_window=0, scale_log2=0.0
        )
        return FmhaMfmaSpec(common=common, seqlen_q=256, seqlen_k=256)
    if idx == 1:
        shape = FmhaShape(head_size=128, num_query_heads=16, num_kv_heads=16)
        common = FmhaCommonSpec(
            shape=shape, dtype="f16", mask_mode="none", sliding_window=0, scale_log2=0.0
        )
        return FmhaMfmaSpec(common=common, seqlen_q=512, seqlen_k=512)
    if idx == 2:
        shape = FmhaShape(head_size=64, num_query_heads=8, num_kv_heads=8)
        common = FmhaCommonSpec(
            shape=shape,
            dtype="f16",
            mask_mode="causal",
            sliding_window=0,
            scale_log2=0.0,
        )
        return FmhaMfmaSpec(common=common, seqlen_q=256, seqlen_k=1024)
    if idx == 3:
        shape = FmhaShape(head_size=256, num_query_heads=32, num_kv_heads=32)
        common = FmhaCommonSpec(
            shape=shape,
            dtype="f16",
            mask_mode="sliding_window",
            sliding_window=512,
            scale_log2=0.0,
        )
        return FmhaMfmaSpec(common=common, seqlen_q=512, seqlen_k=2048)
    if idx == 4:
        shape = FmhaShape(head_size=192, num_query_heads=12, num_kv_heads=12)
        common = FmhaCommonSpec(
            shape=shape, dtype="f16", mask_mode="none", sliding_window=0, scale_log2=0.0
        )
        return FmhaMfmaSpec(common=common, seqlen_q=128, seqlen_k=512)
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: fmha_mfma_emit.py <config_index 0..4>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_fmha_fwd_mfma(spec, arch="gfx950")
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
