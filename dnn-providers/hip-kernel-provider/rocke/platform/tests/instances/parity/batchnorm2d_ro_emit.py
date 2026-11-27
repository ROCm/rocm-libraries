# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python reference emitter for the read-once (persistent grid) batchnorm2d
kernels (BatchNorm2DRoSpec, build_batchnorm2d_ro_train / _bwd). Byte-compared
with batchnorm2d_ro_emit.c; the config tables must match index for index.

Configs cover training and backward, f16 / bf16, both block sizes, vec 4 / 8,
the register depth range, the wave-prologue block reduction and the training
kernel without the saved mean / inverse std outputs, plus the spec defaults.
"""

from rocke.instances.common.batchnorm2d_ro import (
    BatchNorm2DRoSpec,
    build_batchnorm2d_ro_bwd,
    build_batchnorm2d_ro_train,
)

from _emit_common import run_emit

# (kind, block_size, vec, regs, dtype, fast_reduce, save_mean_invstd)
CONFIGS = [
    ("train", 256, 8, 16, "f16", False, True),  # 0 spec defaults
    ("bwd", 256, 8, 16, "f16", False, True),  # 1 spec defaults, backward
    ("train", 512, 4, 2, "bf16", False, True),  # 2
    ("bwd", 512, 4, 32, "bf16", False, True),  # 3
    ("train", 256, 8, 4, "f16", True, True),  # 4
    ("bwd", 256, 4, 8, "f16", True, True),  # 5
    ("train", 256, 8, 2, "f16", False, False),  # 6
    ("train", 512, 8, 32, "f16", False, True),  # 7
    ("bwd", 256, 8, 2, "bf16", False, True),  # 8
    ("train", 256, 4, 16, "f32", False, True),  # 9 f32
    ("bwd", 256, 4, 16, "f32", False, True),  # 10 f32
    ("train", 512, 2, 32, "f32", False, True),  # 11 f32
    ("bwd", 512, 2, 32, "f32", False, True),  # 12 f32
]


def _spec(idx):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    kind, bs, vec, regs, dtype, fr, smv = CONFIGS[idx]
    spec = BatchNorm2DRoSpec(
        block_size=bs,
        vec=vec,
        regs=regs,
        dtype=dtype,
        fast_reduce=fr,
        save_mean_invstd=smv,
    )
    return (kind, spec)


def _build(sel, *, arch):
    kind, spec = sel
    return (
        build_batchnorm2d_ro_bwd(spec)
        if kind == "bwd"
        else build_batchnorm2d_ro_train(spec)
    )


def main() -> int:
    return run_emit(lambda idx: (_spec(idx), "gfx950"), _build)


if __name__ == "__main__":
    raise SystemExit(main())
