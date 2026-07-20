#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/fused_moe_emit.py -- Python reference emitter for the fused_moe
# parity harness. Selects one of N sampled FusedMoeSpec configs by argv[1] and
# one of the five MoE-specific builders by argv[2] (the "phase"), builds the
# kernel and prints _native_lower(arch='gfx950') to stdout so it can be
# byte-compared with the C emitter fused_moe_emit.c.
import sys

from rocke.instances.common.fused_moe import (
    FusedMoeSpec,
    build_moe_gather,
    build_moe_silu_mul,
    build_moe_silu_mul_packed,
    build_moe_static_scatter_gather,
    build_moe_topk_weighted_reduce,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# Each tuple mirrors the C-side config table exactly:
#   (tokens, experts, topk, hidden, intermediate, dtype, block_size, vec)
# WIDE adversarial stress set: edge shapes (M/E/K=1, primes, very large),
# all dtypes/block_sizes/vec widths, and the _effective_vec degeneration path
# (H==block_size -> VEC collapses to 1 -> scalar fallback bodies).
CONFIGS = [
    # --- original sampled set ---
    (4, 4, 2, 128, 512, "f16", 256, 4),
    (1, 8, 2, 1024, 2048, "f16", 256, 4),
    (256, 16, 4, 4096, 16384, "bf16", 256, 8),
    (128, 32, 2, 2048, 8192, "f16", 512, 4),
    (512, 64, 8, 8192, 32768, "bf16", 1024, 8),
    (16, 4, 1, 256, 1024, "f16", 64, 2),
    # --- M/E/K = 1 corners ---
    (1, 1, 1, 64, 64, "f16", 64, 2),
    (1, 1, 1, 64, 64, "bf16", 64, 2),
    (1, 2, 1, 128, 128, "fp16", 64, 2),
    # --- prime tokens/experts/topk (non-divisibility-constrained axes) ---
    (7, 13, 3, 256, 256, "f16", 64, 2),
    (101, 17, 5, 512, 1536, "bf16", 256, 4),
    (3, 3, 3, 192, 384, "f16", 64, 2),
    (97, 5, 5, 320, 640, "bf16", 64, 2),
    # --- VEC degeneration: H==block_size -> effective vec collapses to 1 ---
    (4, 4, 2, 64, 64, "f16", 64, 8),  # H=BS -> VEC 8->1
    (4, 4, 2, 128, 128, "bf16", 128, 4),  # H=BS -> VEC 4->1
    (4, 4, 2, 256, 256, "f16", 256, 8),  # H=BS -> VEC 8->1
    # --- VEC partial degeneration (BS*vec doesn't divide H but BS*2 does) ---
    (8, 4, 2, 128, 384, "f16", 64, 8),  # 64*8=512 !| 128 -> VEC->2
    (8, 4, 2, 192, 192, "bf16", 64, 4),  # 64*4=256 !| 192 -> VEC->2 (192/128? no)
    # --- asymmetric H vs I (different VEC per phase) ---
    (16, 8, 2, 64, 2048, "f16", 64, 8),
    (16, 8, 2, 4096, 64, "bf16", 64, 8),
    # --- large block sizes ---
    (32, 8, 2, 512, 512, "f16", 512, 2),
    (32, 8, 2, 1024, 1024, "bf16", 1024, 2),
    (32, 8, 2, 2048, 2048, "f16", 1024, 8),
    # --- topk == experts (boundary) ---
    (10, 6, 6, 384, 768, "bf16", 128, 4),
    (5, 5, 5, 640, 1280, "f16", 128, 8),
    # --- very large dims ---
    (1024, 128, 8, 16384, 16384, "bf16", 1024, 8),
    (2048, 256, 4, 8192, 28672, "f16", 512, 4),
    # --- dtype sweep on same shape ---
    (64, 8, 2, 1024, 4096, "f16", 256, 8),
    (64, 8, 2, 1024, 4096, "bf16", 256, 8),
    (64, 8, 2, 1024, 4096, "fp16", 256, 8),
    # --- vec sweep on same shape ---
    (64, 8, 2, 768, 1536, "f16", 64, 2),
    (64, 8, 2, 768, 1536, "f16", 64, 4),
    (64, 8, 2, 768, 1536, "f16", 64, 8),  # 64*8=512 !| 768 -> VEC->4
    # --- H/I prime-ish multiples of block_size ---
    (4, 4, 2, 64 * 7, 64 * 11, "bf16", 64, 2),  # 448, 704
    (4, 4, 2, 128 * 3, 128 * 5, "f16", 128, 4),  # 384, 640
    # --- topk=1 large ---
    (300, 32, 1, 2048, 8192, "bf16", 256, 8),
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
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: fused_moe_emit.py <config_index> [<phase>|<ll|ir|verify>]\n"
        )
        return 2
    idx = int(sys.argv[1])

    # argv[2] may be a phase or a mode ("ll"/"ir"/"verify").
    # If it looks like a mode, treat it as mode with default phase "gather".
    # If it looks like a phase, treat it as phase with default mode "ll".
    # If absent, default phase="gather", mode="ll".
    phase = "gather"
    mode = "ll"
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        if arg2 in ("ir", "verify", "ll"):
            mode = arg2
        elif arg2 in BUILDERS:
            phase = arg2
        else:
            sys.stderr.write(f"unknown phase {arg2}\n")
            return 2

    if idx < 0 or idx >= len(CONFIGS):
        sys.stderr.write(f"unknown config index {idx}\n")
        return 2

    kernel = BUILDERS[phase](_spec(idx))
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
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
