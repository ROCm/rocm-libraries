#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx1201_deep_fused_conv_pool_emit.py -- Python reference emitter
# for the gfx1201 (RDNA4, wave32, WMMA 16x16x16) arch shim over the deep fused
# conv0 -> conv1 -> maxpool kernel. Selects one of N sampled spec configs by
# argv[1], builds the Gfx1201DeepFusedConvPoolSpec via the gfx1201 shim's
# make_deep_fused_conv_pool_spec, builds the kernel via build_deep_fused_conv_pool
# (re-exported common driver, arch="gfx1201") and prints
# lower_kernel_to_llvm(arch="gfx1201") to stdout so it can be byte-compared with
# the C emitter gfx1201_deep_fused_conv_pool_emit.c.
from rocke.instances.gfx1201.deep_fused_conv_pool import (
    make_deep_fused_conv_pool_spec,
    build_deep_fused_conv_pool,
)
from _emit_common import run_emit

_ARCH = "gfx1201"


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`. All gfx1201 / wave32 WMMA."""
    if idx == 0:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=64,
                w=128,
                c=8,
                k0=16,
                k1=16,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
            ),
            _ARCH,
        )
    if idx == 1:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=80,
                w=80,
                c=16,
                k0=16,
                k1=16,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
            ),
            _ARCH,
        )
    if idx == 2:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=80,
                w=80,
                c=8,
                k0=16,
                k1=16,
                r=3,
                s=3,
                pool_tile_h=2,
                pool_tile_w=4,
            ),
            _ARCH,
        )
    if idx == 3:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=96,
                w=96,
                c=8,
                k0=16,
                k1=16,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
            ),
            _ARCH,
        )
    if idx == 4:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=32,
                w=64,
                c=8,
                k0=16,
                k1=16,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
            ),
            _ARCH,
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_deep_fused_conv_pool,
        usage="usage: gfx1201_deep_fused_conv_pool_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
