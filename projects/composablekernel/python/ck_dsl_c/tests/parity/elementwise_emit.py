#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/elementwise_emit.py -- Python reference emitter for the
# elementwise parity harness. Selects one of N sampled ElementwiseSpec configs
# by argv[1] (the config index), builds the kernel via build_elementwise and
# prints lower_kernel_to_llvm(kernel, arch='gfx950') to stdout so it can be
# byte-compared with the C emitter elementwise_emit.c.
import sys

from ck_dsl.instances.common.elementwise import (
    ElementwiseSpec,
    build_elementwise,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify


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
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: elementwise_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec = _spec(idx)
    kernel = build_elementwise(spec)
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
