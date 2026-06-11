#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fused_moe_emit.py -- Python reference emitter for the fused_moe
# parity harness. Selects one of N sampled FusedMoeSpec configs by argv[1] and
# one of the five MoE-specific builders by argv[2] (the "phase"), builds the
# kernel and prints lower_kernel_to_llvm(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter fused_moe_emit.c.
import sys

from ck_dsl.instances.common.fused_moe import (
    FusedMoeSpec,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_silu_mul_packed,
    build_moe_static_scatter_gather,
    build_moe_topk_weighted_reduce,
)
from ck_dsl import lower_kernel_to_llvm


# Each tuple mirrors the C-side config table exactly:
#   (tokens, experts, topk, hidden, intermediate, dtype, block_size, vec)
CONFIGS = [
    (4, 4, 2, 128, 512, "f16", 256, 4),
    (1, 8, 2, 1024, 2048, "f16", 256, 4),
    (256, 16, 4, 4096, 16384, "bf16", 256, 8),
    (128, 32, 2, 2048, 8192, "f16", 512, 4),
    (512, 64, 8, 8192, 32768, "bf16", 1024, 8),
    (16, 4, 1, 256, 1024, "f16", 64, 2),
]

BUILDERS = {
    "gather": build_moe_gather,
    "silu_mul": build_moe_silu_mul,
    "silu_mul_packed": build_moe_silu_mul_packed,
    "static_scatter_gather": build_moe_static_scatter_gather,
    "topk_weighted_reduce": build_moe_topk_weighted_reduce,
}


def _spec(idx: int) -> FusedMoeSpec:
    t, e, k, h, i, dt, bs, v = CONFIGS[idx]
    return FusedMoeSpec(
        tokens=t,
        experts=e,
        topk=k,
        hidden=h,
        intermediate=i,
        dtype=dt,
        block_size=bs,
        vec=v,
    )


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("usage: fused_moe_emit.py <config_index> <phase>\n")
        return 2
    idx = int(sys.argv[1])
    phase = sys.argv[2]
    if idx < 0 or idx >= len(CONFIGS):
        sys.stderr.write(f"unknown config index {idx}\n")
        return 2
    if phase not in BUILDERS:
        sys.stderr.write(f"unknown phase {phase}\n")
        return 2
    kernel = BUILDERS[phase](_spec(idx))
    text = lower_kernel_to_llvm(kernel, arch="gfx950")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
