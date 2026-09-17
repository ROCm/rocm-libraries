#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/elementwise_emit.py -- Python reference emitter for the
# elementwise parity harness. Selects one of N sampled ElementwiseSpec configs
# by argv[1] (the config index), builds the kernel via build_elementwise and
# prints lower_kernel_to_llvm(kernel, arch='gfx950') to stdout so it can be
# byte-compared with the C emitter elementwise_emit.c.
from rocke.instances.common.elementwise import (
    ElementwiseSpec,
    build_elementwise,
)
from _emit_common import run_emit


def _spec(idx: int) -> ElementwiseSpec:
    if idx == 0:
        return ElementwiseSpec(op="relu", dtype="f16", block_size=256, vec=8)
    if idx == 1:
        return ElementwiseSpec(op="relu", dtype="bf16", block_size=256, vec=8)
    if idx == 2:
        return ElementwiseSpec(op="add", dtype="f16", block_size=128, vec=4)
    if idx == 3:
        return ElementwiseSpec(op="add", dtype="f16", block_size=512, vec=2)
    if idx == 4:
        return ElementwiseSpec(op="silu", dtype="bf16", block_size=64, vec=8)
    if idx == 5:
        return ElementwiseSpec(op="gelu_tanh", dtype="f16", block_size=1024, vec=4)
    if idx == 6:
        # gfx1151 (RDNA3.5) coverage: elementwise has no MMA/wave dependence, so
        # the same kernel is lowered for the RDNA datalayout/triple/waitcnt.
        return (
            ElementwiseSpec(op="relu", dtype="f16", block_size=256, vec=8),
            "gfx1151",
        )
    if idx == 7:
        # gfx1201 (RDNA4) coverage.
        return (
            ElementwiseSpec(op="relu", dtype="f16", block_size=256, vec=8),
            "gfx1201",
        )
    raise SystemExit(f"unknown config index {idx}")


def _build(spec, arch=None):
    return build_elementwise(spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: elementwise_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
