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
from rocke.instances.gfx1151.deep_fused_conv_pool import (
    make_deep_fused_conv_pool_spec,
    build_deep_fused_conv_pool,
)
from _emit_common import run_emit

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
    return run_emit(
        _spec,
        build_deep_fused_conv_pool,
        usage="usage: gfx1151_deep_fused_conv_pool_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
