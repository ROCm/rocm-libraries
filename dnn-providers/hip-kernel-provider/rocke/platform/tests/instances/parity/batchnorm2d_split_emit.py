#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/batchnorm2d_split_emit.py -- Python reference emitter for the
# cross-CTA spatial-split batchnorm2d parity STRESS harness. Selects one of N
# configs by argv[1] (the config index) and prints _native_lower(arch='gfx950')
# to stdout so it can be byte-compared with the C emitter
# batchnorm2d_split_emit.c.
#
# Each config picks ONE of the two kernels in the split-reduction pair (see
# rocke.instances.common.batchnorm2d's "Cross-CTA spatial-split reduction"
# section): "reduce" (build_batchnorm2d_reduce_partial) or "combine"
# (build_batchnorm2d_combine_apply). The config table encodes which.
import sys

from rocke.instances.common.batchnorm2d import (
    BatchNorm2DSplitSpec,
    build_batchnorm2d_combine_apply,
    build_batchnorm2d_reduce_partial,
)

try:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
except ImportError:  # pragma: no cover - older reference tree
    from rocke import lower_kernel_to_llvm as _native_lower
from rocke.core.ir_serialize import serialize
from rocke.core.verify import verify


# (kind, n, c, h, w, split_hw, block_size, vec, dtype, channels_per_block, fast_reduce)
CONFIGS = [
    # --- basic reduce / combine pair, single channel per CTA ---
    ("reduce", 1, 16, 64, 64, 4, 64, 2, "f16", 1, False),  # 0
    ("combine", 1, 16, 64, 64, 4, 64, 2, "f16", 1, False),  # 1
    ("reduce", 1, 16, 64, 64, 4, 64, 2, "bf16", 1, False),  # 2
    ("combine", 1, 16, 64, 64, 4, 64, 2, "bf16", 1, False),  # 3
    # --- split_hw sweep ---
    ("reduce", 1, 8, 32, 32, 2, 64, 2, "f16", 1, False),  # 4
    ("combine", 1, 8, 32, 32, 2, 64, 2, "f16", 1, False),  # 5
    ("reduce", 1, 8, 32, 32, 8, 128, 4, "f16", 1, False),  # 6
    ("combine", 1, 8, 32, 32, 8, 128, 4, "f16", 1, False),  # 7
    # --- HW not divisible by split_hw (last split shorter) ---
    ("reduce", 1, 16, 14, 14, 3, 64, 2, "f16", 1, False),  # 8
    ("combine", 1, 16, 14, 14, 3, 64, 2, "f16", 1, False),  # 9
    # --- N > 1 ---
    ("reduce", 2, 16, 32, 32, 4, 64, 2, "f16", 1, False),  # 10
    ("combine", 2, 16, 32, 32, 4, 64, 2, "f16", 1, False),  # 11
    # --- channels_per_block > 1 ---
    ("reduce", 1, 16, 32, 32, 4, 64, 2, "f16", 2, False),  # 12
    ("combine", 1, 16, 32, 32, 4, 64, 2, "f16", 2, False),  # 13
    # --- fast_reduce ---
    ("reduce", 1, 16, 32, 32, 4, 64, 2, "f16", 1, True),  # 14
    ("combine", 1, 16, 32, 32, 4, 64, 2, "f16", 1, True),  # 15
    # --- large-H*W / small-C motivating shape (scaled down for parity speed) ---
    ("reduce", 1, 16, 128, 128, 16, 256, 4, "f16", 1, False),  # 16
    ("combine", 1, 16, 128, 128, 16, 256, 4, "f16", 1, False),  # 17
]


def _spec_and_kind(idx: int):
    if idx < 0 or idx >= len(CONFIGS):
        raise SystemExit(f"unknown config index {idx}")
    kind, n, c, h, w, split_hw, bs, v, dt, cpb, fr = CONFIGS[idx]
    spec = BatchNorm2DSplitSpec(
        n=n,
        c=c,
        h=h,
        w=w,
        split_hw=split_hw,
        block_size=bs,
        vec=v,
        dtype=dt,
        channels_per_block=cpb,
        fast_reduce=fr,
    )
    return spec, kind


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            f"usage: batchnorm2d_split_emit.py <config_index 0..{len(CONFIGS) - 1}> [ll|ir|verify]\n"
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
        build_batchnorm2d_reduce_partial(spec)
        if kind == "reduce"
        else build_batchnorm2d_combine_apply(spec)
    )
    if mode == "ll":
        text = _native_lower(kernel, arch="gfx950")
        sys.stdout.write(text)
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    else:  # verify
        sys.stdout.write("".join(str(d) + "\n" for d in verify(kernel)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
