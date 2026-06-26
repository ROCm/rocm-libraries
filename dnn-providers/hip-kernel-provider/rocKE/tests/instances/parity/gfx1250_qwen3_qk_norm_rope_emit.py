#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_qwen3_qk_norm_rope_emit.py -- Python reference emitter for
# the gfx1250 fused QK-norm + RoPE parity harness. Selects one of the sampled
# Qwen3QkNormRopeSpec configs by argv[1], builds via build_qwen3_qk_norm_rope and
# prints lower_kernel_to_llvm(arch='gfx1250') to stdout for byte-comparison.
from rocke.instances.gfx1250.qwen3_qk_norm_rope import (
    Qwen3QkNormRopeSpec,
    build_qwen3_qk_norm_rope,
)
from _emit_common import run_emit


def _spec(idx: int) -> Qwen3QkNormRopeSpec:
    if idx == 0:
        return Qwen3QkNormRopeSpec(num_heads=32)
    if idx == 1:
        return Qwen3QkNormRopeSpec(num_heads=4)
    if idx == 2:
        return Qwen3QkNormRopeSpec(num_heads=32, dtype="fp16")
    if idx == 3:
        return Qwen3QkNormRopeSpec(num_heads=8, rope_layout="interleaved")
    if idx == 4:
        return Qwen3QkNormRopeSpec(num_heads=16, head_dim=128, block_size=128)
    if idx == 5:
        return Qwen3QkNormRopeSpec(
            num_heads=4, head_dim=128, dtype="fp16", rope_layout="interleaved"
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_qwen3_qk_norm_rope(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_qwen3_qk_norm_rope_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
