#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python parity emitter for local rank-staged MoE reductions."""

from __future__ import annotations

import sys

from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify
from rocke.instances.common.moe_rank_reduce import (
    MoeRankReduceRMSNormSpec,
    MoeRankReduceScatterSpec,
    build_moe_rank_reduce_rmsnorm,
    build_moe_rank_reduce_scatter,
)


def _kernel(index: int):
    if index == 0:
        return build_moe_rank_reduce_rmsnorm(
            MoeRankReduceRMSNormSpec(width=3584, world_size=8)
        )
    if index == 1:
        return build_moe_rank_reduce_rmsnorm(
            MoeRankReduceRMSNormSpec(
                width=2048,
                world_size=4,
                dtype="f16",
                block_size=128,
                vec=4,
                fp32_internal=True,
            )
        )
    if index == 2:
        return build_moe_rank_reduce_scatter(
            MoeRankReduceScatterSpec(width=7168, world_size=8)
        )
    raise SystemExit(f"unknown config index {index}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: moe_rank_reduce_emit.py <config_index> [ll|ir|verify]\n"
        )
        return 2
    kernel = _kernel(int(sys.argv[1]))
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode == "ll":
        sys.stdout.write(_lower_kernel_to_llvm_python(kernel, arch="gfx950"))
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write("".join(f"{diag}\n" for diag in verify(kernel)))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
