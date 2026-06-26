#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_qwen3_sampler_emit.py -- Python reference emitter for the
# gfx1250 greedy-sampler parity harness. Selects one of the sampled
# Qwen3GreedySamplerSpec configs by argv[1], builds via build_qwen3_greedy_sampler
# and prints lower_kernel_to_llvm(arch='gfx1250') to stdout for byte-comparison.
from rocke.instances.gfx1250.qwen3_sampler import (
    Qwen3GreedySamplerSpec,
    build_qwen3_greedy_sampler,
)
from _emit_common import run_emit


def _spec(idx: int) -> Qwen3GreedySamplerSpec:
    if idx == 0:
        return Qwen3GreedySamplerSpec()
    if idx == 1:
        return Qwen3GreedySamplerSpec(logits_dtype="bf16")
    if idx == 2:
        return Qwen3GreedySamplerSpec(logits_dtype="fp16")
    if idx == 3:
        return Qwen3GreedySamplerSpec(block_size=128)
    if idx == 4:
        return Qwen3GreedySamplerSpec(logits_dtype="bf16", block_size=512)
    if idx == 5:
        return Qwen3GreedySamplerSpec(logits_dtype="fp32", block_size=64)
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_qwen3_greedy_sampler(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_qwen3_sampler_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
