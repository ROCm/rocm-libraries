#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batchnorm2d_dyn_split_emit.py -- Python reference emitter for
# the dynamic (AOT-generic) cross-CTA split batchnorm2d parity harness. Selects
# one of N configs by argv[1] (the config index) and prints
# _native_lower(arch='gfx950') to stdout so it can be byte-compared with the C
# emitter batchnorm2d_dyn_split_emit.c.
#
# Each config picks ONE kernel of the pair: "reduce"
# (build_batchnorm2d_split_dyn_reduce) or "apply"
# (build_batchnorm2d_split_dyn_apply).
import sys

from rocke.instances.common.batchnorm2d import (
    BatchNorm2DSplitDynSpec,
    build_batchnorm2d_split_dyn_apply,
    build_batchnorm2d_split_dyn_reduce,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# (kind, block_size, vec, dtype, fast_reduce, save_mean_invstd)
CONFIGS = [
    ("reduce", 256, 8, "f16", False, True),  # 0  defaults
    ("apply", 256, 8, "f16", False, True),  # 1  defaults
    ("reduce", 512, 4, "bf16", True, True),  # 2  fast_reduce
    ("apply", 512, 4, "bf16", True, True),  # 3  fast_reduce
    ("apply", 128, 8, "f16", False, False),  # 4  no saved mean/invstd
    ("reduce", 128, 4, "f32", False, True),  # 5  f32
    ("apply", 128, 4, "f32", False, True),  # 6  f32
    ("reduce", 256, 2, "f32", True, True),  # 7  f32 vec 2 + fast_reduce
    ("apply", 256, 2, "f32", True, False),  # 8  f32 vec 2 + fast_reduce, no smv
]


def _spec_and_kind(idx: int):
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    kind, bs, v, dt, fr, smv = CONFIGS[idx]
    spec = BatchNorm2DSplitDynSpec(
        block_size=bs, vec=v, dtype=dt, fast_reduce=fr, save_mean_invstd=smv
    )
    return spec, kind


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "usage: batchnorm2d_dyn_split_emit.py "
            f"<config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n"
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
    spec, kind = _spec_and_kind(idx)
    kernel = (
        build_batchnorm2d_split_dyn_reduce(spec)
        if kind == "reduce"
        else build_batchnorm2d_split_dyn_apply(spec)
    )
    if mode == "ll":
        sys.stdout.write(_native_lower(kernel, arch="gfx950"))
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
