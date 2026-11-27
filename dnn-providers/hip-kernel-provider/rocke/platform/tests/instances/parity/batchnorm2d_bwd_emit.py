#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batchnorm2d_bwd_emit.py -- Python reference emitter for the
# dynamic NCHW batchnorm2d backward kernels: the single launch
# (build_batchnorm2d_bwd_dynamic, kind "bwd") and the split pair
# (build_batchnorm2d_bwd_split_reduce / _apply, kinds "reduce" / "apply").
# Selects one config by argv[1] and prints the .ll so it can be byte-compared
# with the C emitter batchnorm2d_bwd_emit.c (CONFIGS in lockstep).
from _emit_common import run_emit

from rocke.instances.common.batchnorm2d import (
    BatchNorm2DBwdSpec,
    BatchNorm2DBwdSplitSpec,
    build_batchnorm2d_bwd_dynamic,
    build_batchnorm2d_bwd_split_apply,
    build_batchnorm2d_bwd_split_reduce,
)

# (kind, block_size, vec, dtype, channels_per_block, n_groups, fast_reduce)
# channels_per_block / n_groups apply to kind "bwd" only.
CONFIGS = [
    ("bwd", 256, 4, "f16", 1, 1, False),  # 0 defaults
    ("bwd", 256, 4, "bf16", 1, 1, False),  # 1
    ("bwd", 1024, 8, "f16", 1, 1, False),  # 2
    ("bwd", 128, 2, "bf16", 1, 1, True),  # 3
    ("bwd", 512, 8, "f16", 1, 2, False),  # 4 n_groups
    ("bwd", 256, 4, "bf16", 1, 8, True),  # 5
    ("bwd", 1024, 4, "f16", 2, 1, False),  # 6 channels_per_block
    ("bwd", 128, 8, "bf16", 8, 1, True),  # 7
    ("bwd", 512, 2, "f16", 4, 1, True),  # 8
    ("bwd", 256, 8, "f16", 1, 4, False),  # 9
    ("reduce", 256, 8, "f16", 1, 1, False),  # 10 split defaults
    ("apply", 256, 8, "f16", 1, 1, False),  # 11
    ("reduce", 512, 8, "bf16", 1, 1, True),  # 12
    ("apply", 512, 8, "bf16", 1, 1, True),  # 13
    ("reduce", 128, 4, "f16", 1, 1, True),  # 14
    ("apply", 128, 4, "f16", 1, 1, True),  # 15
    ("reduce", 512, 4, "bf16", 1, 1, False),  # 16
    ("apply", 512, 4, "bf16", 1, 1, False),  # 17
    ("bwd", 256, 4, "f32", 1, 1, False),  # 18 f32
    ("bwd", 128, 2, "f32", 1, 2, True),  # 19 f32
    ("reduce", 256, 4, "f32", 1, 1, False),  # 20 f32
    ("apply", 256, 4, "f32", 1, 1, False),  # 21 f32
    ("reduce", 512, 2, "f32", 1, 1, True),  # 22 f32
    ("apply", 512, 2, "f32", 1, 1, True),  # 23 f32
]


class _Sel:
    def __init__(self, kind, spec):
        self.kind, self.spec = kind, spec


def _spec(idx: int):
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    kind, bs, vec, dt, cpb, ng, fr = CONFIGS[idx]
    if kind == "bwd":
        spec = BatchNorm2DBwdSpec(
            block_size=bs,
            vec=vec,
            dtype=dt,
            channels_per_block=cpb,
            n_groups=ng,
            fast_reduce=fr,
        )
    else:
        spec = BatchNorm2DBwdSplitSpec(block_size=bs, vec=vec, dtype=dt, fast_reduce=fr)
    return _Sel(kind, spec)


def _build(sel, arch="gfx950"):
    if sel.kind == "bwd":
        return build_batchnorm2d_bwd_dynamic(sel.spec)
    if sel.kind == "reduce":
        return build_batchnorm2d_bwd_split_reduce(sel.spec)
    return build_batchnorm2d_bwd_split_apply(sel.spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage=f"usage: batchnorm2d_bwd_emit.py <config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
