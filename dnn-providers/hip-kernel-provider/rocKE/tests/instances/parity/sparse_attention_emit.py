#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/sparse_attention_emit.py -- Python reference emitter for the
# sparse-attention forward parity harness. Selects one of 6 sampled configs by
# argv[1] (0..5), builds either a JengaSparseSpec via build_jenga_sparse_attention
# or a VsaSparseSpec via build_vsa_sparse_attention (arch="gfx950") and prints
# lower_kernel_to_llvm(arch='gfx950') to stdout so it can be byte-compared with
# the C emitter sparse_attention_emit.c.
from rocke.instances.common._fmha_common import FmhaCommonSpec, FmhaShape
from rocke.instances.common.sparse_attention import (
    JengaSparseSpec,
    VsaSparseSpec,
    build_jenga_sparse_attention,
    build_vsa_sparse_attention,
)
from _emit_common import run_emit


def _kernel(idx: int):
    if idx == 0:
        spec = JengaSparseSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(64, 8, 8),
                dtype="f16",
            ),
            seqlen_q=32,
            seqlen_k=128,
            block_q=1,
            block_k=64,
        )
        return build_jenga_sparse_attention(spec, arch="gfx950")
    if idx == 1:
        spec = JengaSparseSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(128, 16, 16),
                dtype="bf16",
            ),
            seqlen_q=64,
            seqlen_k=256,
            block_q=2,
            block_k=64,
        )
        return build_jenga_sparse_attention(spec, arch="gfx950")
    if idx == 2:
        spec = VsaSparseSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(64, 8, 8),
                dtype="f16",
            ),
            seqlen_q=32,
            seqlen_k=128,
            block_q=1,
            block_k=64,
            max_blocks_per_q=16,
        )
        return build_vsa_sparse_attention(spec, arch="gfx950")
    if idx == 3:
        spec = VsaSparseSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(128, 16, 16),
                dtype="f16",
            ),
            seqlen_q=64,
            seqlen_k=256,
            block_q=2,
            block_k=64,
            max_blocks_per_q=32,
            use_wave_ballot_scatter=True,
        )
        return build_vsa_sparse_attention(spec, arch="gfx950")
    if idx == 4:
        spec = JengaSparseSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(256, 32, 8),
                dtype="f16",
            ),
            seqlen_q=96,
            seqlen_k=512,
            block_q=4,
            block_k=128,
        )
        return build_jenga_sparse_attention(spec, arch="gfx950")
    if idx == 5:
        spec = VsaSparseSpec(
            common=FmhaCommonSpec(
                shape=FmhaShape(256, 32, 32),
                dtype="bf16",
            ),
            seqlen_q=128,
            seqlen_k=1024,
            block_q=8,
            block_k=64,
            max_blocks_per_q=24,
            use_wave_ballot_scatter=False,
        )
        return build_vsa_sparse_attention(spec, arch="gfx950")
    raise SystemExit(f"unknown config index {idx}")


def _emit_build(kernel, arch=None):
    return kernel


def main() -> int:
    return run_emit(
        _kernel,
        _emit_build,
        usage="usage: sparse_attention_emit.py <config_index 0..5> [mode]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
