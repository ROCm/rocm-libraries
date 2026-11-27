# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Python reference emitter for the multi-channel-per-wave batchnorm2d kernels
(BatchNorm2DMcwSpec, build_batchnorm2d_mcw). Byte-compared with
batchnorm2d_mcw_emit.c; the config tables must match index for index.

Configs cover training and backward, f16 / bf16, a single wave (no LDS
exchange) and several waves, vec 1 / 2 / 4 / 8, several channels per wave and
several slots, plus the spec defaults.
"""

from rocke.instances.common.batchnorm2d import BatchNorm2DMcwSpec, build_batchnorm2d_mcw

from _emit_common import run_emit

# (mode, waves, cpw, npw, slots, vec, dtype)
CONFIGS = [
    ("train", 2, 1, 8, 1, 4, "f16"),  # 0
    ("train", 1, 1, 8, 1, 4, "f16"),  # 1
    ("train", 4, 2, 2, 2, 1, "f16"),  # 2
    ("train", 16, 1, 1, 1, 8, "bf16"),  # 3
    ("train", 8, 4, 2, 2, 8, "f16"),  # 4
    ("train", 1, 1, 2, 1, 2, "f16"),  # 5
    ("bwd", 8, 1, 2, 1, 8, "bf16"),  # 6
    ("bwd", 1, 2, 8, 2, 1, "f16"),  # 7
    ("bwd", 8, 4, 2, 4, 1, "f16"),  # 8
    ("bwd", 4, 8, 1, 16, 2, "f16"),  # 9
    ("bwd", 16, 4, 8, 1, 4, "f16"),  # 10
    ("bwd", 4, 1, 8, 1, 8, "bf16"),  # 11
    ("train", 4, 1, 4, 1, 1, "f16"),  # 12
    ("train", 4, 1, 32, 1, 1, "f16"),  # 13 spec defaults
    ("bwd", 4, 1, 32, 1, 1, "f16"),  # 14 spec defaults, backward
    ("train", 8, 1, 2, 2, 4, "f32"),  # 15 f32
    ("bwd", 8, 1, 2, 2, 4, "f32"),  # 16 f32
    ("train", 16, 1, 1, 16, 1, "f32"),  # 17 f32
    ("bwd", 16, 1, 1, 16, 1, "f32"),  # 18 f32
    ("train", 1, 1, 2, 16, 1, "f32"),  # 19 f32
    ("bwd", 1, 1, 2, 16, 1, "f32"),  # 20 f32
    ("train", 1, 4, 2, 8, 4, "f32"),  # 21 f32
    ("bwd", 1, 2, 2, 16, 1, "f32"),  # 22 f32
]


def _spec(idx):
    if not 0 <= idx < len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    mode, waves, cpw, npw, slots, vec, dtype = CONFIGS[idx]
    return BatchNorm2DMcwSpec(
        mode=mode, waves=waves, cpw=cpw, npw=npw, slots=slots, vec=vec, dtype=dtype
    )


def _build(spec, *, arch):
    return build_batchnorm2d_mcw(spec)


def main() -> int:
    return run_emit(_spec, _build)


if __name__ == "__main__":
    raise SystemExit(main())
