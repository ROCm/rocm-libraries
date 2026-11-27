#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batchnorm2d_flat_emit.py -- Python reference emitter for the
# flat elementwise BatchNorm2D inference parity harness. Selects one of N
# configs by argv[1] (the config index) and prints _native_lower(arch='gfx950')
# to stdout so it can be byte-compared with the C emitter
# batchnorm2d_flat_emit.c.
import sys

from rocke.instances.common.batchnorm2d import (
    BatchNorm2DFlatInferSpec,
    build_batchnorm2d_flat_infer,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# (block_size, vec, unroll, aligned, dtype)
CONFIGS = [
    (256, 8, 1, True, "f16"),  # 0  defaults
    (256, 8, 1, True, "bf16"),  # 1
    (512, 4, 2, False, "f16"),  # 2  unaligned (row-straddling groups)
    (128, 2, 4, False, "bf16"),  # 3  unaligned, unroll 4
    (1024, 1, 1, True, "f16"),  # 4  vec=1 (scalar load/store)
    (256, 8, 4, False, "f16"),  # 5  unaligned, vec 8, unroll 4
    (128, 1, 2, True, "bf16"),  # 6  vec=1, unroll 2
]


def _spec(idx: int) -> BatchNorm2DFlatInferSpec:
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    bs, v, u, al, dt = CONFIGS[idx]
    return BatchNorm2DFlatInferSpec(
        block_size=bs, vec=v, unroll=u, aligned=al, dtype=dt
    )


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            f"usage: batchnorm2d_flat_emit.py <config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n"
        )
        return 2
    if sys.argv[1] == "--count":
        sys.stdout.write(str(len(CONFIGS)) + "\n")
        return 0
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    if mode not in ("ll", "ir", "verify"):
        sys.stderr.write(f"unknown mode {mode}\n")
        return 2
    kernel = build_batchnorm2d_flat_infer(_spec(idx))
    if mode == "ll":
        sys.stdout.write(_native_lower(kernel, arch="gfx950"))
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
