#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batchnorm2d_nhwc_emit.py -- Python reference emitter for the
# dynamic NHWC batchnorm2d kernels (build_batchnorm2d_nhwc: infer, training
# forward and backward; kinds "infer", "fused", "reduce", "apply"). Selects
# one config by argv[1] and prints the .ll so it can be byte-compared with the
# C emitter batchnorm2d_nhwc_emit.c (CONFIGS in lockstep).
from _emit_common import run_emit

from rocke.instances.common.batchnorm2d import (
    BatchNorm2DNhwcSpec,
    build_batchnorm2d_nhwc,
)

# (mode, kind, block_size, tx, vec, unroll, dtype)
CONFIGS = [
    ("infer", "infer", 256, 4, 8, 2, "f16"),  # 0
    ("infer", "infer", 256, 64, 4, 2, "bf16"),  # 1
    ("infer", "infer", 256, 32, 8, 16, "f16"),  # 2
    ("train", "fused", 256, 8, 8, 4, "f16"),  # 3 defaults
    ("train", "fused", 1024, 16, 4, 4, "bf16"),  # 4
    ("train", "reduce", 1024, 32, 2, 16, "bf16"),  # 5
    ("train", "apply", 1024, 32, 2, 16, "bf16"),  # 6
    ("train", "reduce", 512, 8, 8, 16, "f16"),  # 7
    ("train", "apply", 512, 8, 8, 16, "f16"),  # 8
    ("bwd", "fused", 512, 16, 2, 4, "f16"),  # 9
    ("bwd", "fused", 1024, 8, 4, 4, "bf16"),  # 10
    ("bwd", "reduce", 1024, 64, 4, 4, "bf16"),  # 11
    ("bwd", "apply", 1024, 64, 4, 4, "bf16"),  # 12
    ("bwd", "reduce", 512, 16, 8, 8, "f16"),  # 13
    ("bwd", "apply", 512, 16, 8, 8, "f16"),  # 14
    ("bwd", "reduce", 1024, 32, 2, 2, "bf16"),  # 15
    ("bwd", "apply", 1024, 32, 2, 2, "bf16"),  # 16
    ("train", "reduce", 256, 8, 4, 4, "bf16"),  # 17
    ("train", "apply", 256, 8, 4, 4, "bf16"),  # 18
]


class _Sel:
    def __init__(self, kind, spec):
        self.kind, self.spec = kind, spec


def _spec(idx: int):
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    mode, kind, bs, tx, vec, unroll, dt = CONFIGS[idx]
    spec = BatchNorm2DNhwcSpec(
        mode=mode, block_size=bs, tx=tx, vec=vec, unroll=unroll, dtype=dt
    )
    return _Sel(kind, spec)


def _build(sel, arch="gfx950"):
    return build_batchnorm2d_nhwc(sel.spec, sel.kind)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage=f"usage: batchnorm2d_nhwc_emit.py <config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
