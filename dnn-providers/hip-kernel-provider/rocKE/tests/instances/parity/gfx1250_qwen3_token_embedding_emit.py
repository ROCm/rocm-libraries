#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1250_qwen3_token_embedding_emit.py -- Python reference emitter
# for the gfx1250 token-embedding parity harness. Selects one of the sampled
# Qwen3TokenEmbeddingSpec configs by argv[1], builds via
# build_qwen3_token_embedding and prints lower_kernel_to_llvm(arch='gfx1250') to
# stdout so it can be byte-compared with the C emitter.
from rocke.instances.gfx1250.qwen3_token_embedding import (
    Qwen3TokenEmbeddingSpec,
    build_qwen3_token_embedding,
)
from _emit_common import run_emit


def _spec(idx: int) -> Qwen3TokenEmbeddingSpec:
    if idx == 0:
        return Qwen3TokenEmbeddingSpec()
    if idx == 1:
        return Qwen3TokenEmbeddingSpec(hidden=4096)
    if idx == 2:
        return Qwen3TokenEmbeddingSpec(dtype="fp16")
    if idx == 3:
        return Qwen3TokenEmbeddingSpec(vec=4)
    if idx == 4:
        return Qwen3TokenEmbeddingSpec(hidden=2560, dtype="fp16", vec=2)
    if idx == 5:
        return Qwen3TokenEmbeddingSpec(block_size=128, vec=1)
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch="gfx1250"):
    return build_qwen3_token_embedding(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: gfx1250_qwen3_token_embedding_emit.py <config_index 0..5>\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
