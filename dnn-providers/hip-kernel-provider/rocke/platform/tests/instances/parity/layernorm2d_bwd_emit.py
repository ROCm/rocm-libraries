#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/layernorm2d_bwd_emit.py -- Python reference emitter for the
# LayerNorm2D backward parity harness. Selects one of N configs by argv[1]
# (the config index) and prints _native_lower(arch='gfx950') to stdout so it
# can be byte-compared with the C emitter layernorm2d_bwd_emit.c.
import sys

from rocke.instances.common.layernorm2d_bwd import (
    LayerNorm2DBwdSpec,
    build_layernorm2d_bwd,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# (part, block_size, vec, dtype, fast_reduce, unroll)
CONFIGS = [
    ("dx", 256, 4, "f16", False, 4),  # 0
    ("dx", 64, 8, "bf16", False, 4),  # 1
    ("dx", 128, 2, "bf16", True, 4),  # 2
    ("dx", 1024, 8, "f16", True, 4),  # 3
    ("dx", 512, 4, "fp16", False, 4),  # 4
    ("dwdb", 64, 2, "bf16", False, 4),  # 5
    ("dwdb", 64, 8, "bf16", False, 1),  # 6
    ("dwdb", 256, 4, "f16", False, 8),  # 7
    ("dwdb", 128, 2, "f16", False, 2),  # 8
    ("dwdb_reduce", 64, 2, "bf16", False, 4),  # 9
    ("dwdb_reduce", 256, 8, "f16", False, 1),  # 10
    ("dwdb_reduce", 128, 4, "bf16", False, 8),  # 11
]


def _spec(idx: int) -> LayerNorm2DBwdSpec:
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    part, bs, v, dt, fr, u = CONFIGS[idx]
    return LayerNorm2DBwdSpec(
        part=part, block_size=bs, vec=v, dtype=dt, fast_reduce=fr, unroll=u
    )


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            f"usage: layernorm2d_bwd_emit.py <config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n"
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
    kernel = build_layernorm2d_bwd(_spec(idx))
    if mode == "ll":
        sys.stdout.write(_native_lower(kernel, arch="gfx950"))
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
