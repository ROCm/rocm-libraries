#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1151_deep_fused_conv_pool_emit.py -- Python reference emitter
# for the gfx1151 (RDNA3.5 / Strix Halo, wave32, WMMA 16x16x16) deep fused
# conv0 -> conv1 -> maxpool kernel. Selects one of N sampled spec configs by
# argv[1], builds the Gfx1151DeepFusedConvPoolSpec via the gfx1151 shim's
# make_deep_fused_conv_pool_spec, builds the kernel via
# build_deep_fused_conv_pool (arch="gfx1151") and prints
# lower_kernel_to_llvm(arch="gfx1151") to stdout so it can be byte-compared with
# the C emitter gfx1151_deep_fused_conv_pool_emit.c.
import sys

from ck_dsl.instances.gfx1151.deep_fused_conv_pool import (
    make_deep_fused_conv_pool_spec,
    build_deep_fused_conv_pool,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify

_ARCH = "gfx1151"


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`. All gfx1151 / wave32 WMMA."""
    cfgs = {
        0: dict(
            n=1, h=64, w=128, c=8, k0=16, k1=16, r=3, s=3, pool_tile_h=4, pool_tile_w=8
        ),
        1: dict(
            n=1, h=80, w=80, c=8, k0=16, k1=24, r=3, s=3, pool_tile_h=4, pool_tile_w=8
        ),
        2: dict(
            n=1, h=56, w=112, c=8, k0=16, k1=16, r=3, s=3, pool_tile_h=2, pool_tile_w=4
        ),
        3: dict(
            n=1, h=112, w=112, c=8, k0=16, k1=16, r=3, s=3, pool_tile_h=4, pool_tile_w=8
        ),
        4: dict(
            n=1, h=56, w=56, c=8, k0=24, k1=16, r=3, s=3, pool_tile_h=4, pool_tile_w=8
        ),
        5: dict(
            n=1, h=112, w=224, c=8, k0=16, k1=32, r=3, s=3, pool_tile_h=4, pool_tile_w=8
        ),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return make_deep_fused_conv_pool_spec(**cfgs[idx]), _ARCH


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx1151_deep_fused_conv_pool_emit.py <config_index>\n")
        return 2
    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"
    spec, arch = _spec(idx)
    kernel = build_deep_fused_conv_pool(spec, arch=arch)
    if mode == "ll":
        text = lower_kernel_to_llvm(kernel, arch=arch)
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
