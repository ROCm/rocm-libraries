#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/deep_fused_conv_pool_emit.py -- Python reference emitter for the
# deep fused conv0 -> conv1 -> maxpool parity harness. Selects one of N sampled
# spec configs by argv[1], builds the DeepFusedConvPoolSpec via
# make_deep_fused_conv_pool_spec, builds the kernel via
# build_deep_fused_conv_pool(spec, arch=<cfg arch>) and prints
# lower_kernel_to_llvm(arch=<cfg arch>) to stdout so it can be byte-compared with
# the C emitter deep_fused_conv_pool_emit.c.
from rocke.instances.common.deep_fused_conv_pool import (
    make_deep_fused_conv_pool_spec,
    build_deep_fused_conv_pool,
)
from _emit_common import run_emit


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`."""
    if idx == 0:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=112,
                w=112,
                c=64,
                k0=64,
                k1=64,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                wave_size=64,
            ),
            "gfx950",
        )
    if idx == 1:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=56,
                w=56,
                c=128,
                k0=128,
                k1=128,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                wave_size=64,
            ),
            "gfx950",
        )
    if idx == 2:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=28,
                w=28,
                c=256,
                k0=256,
                k1=256,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                wave_size=64,
            ),
            "gfx950",
        )
    if idx == 3:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=112,
                w=112,
                c=64,
                k0=64,
                k1=64,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
                wave_size=32,
            ),
            "gfx1201",
        )
    if idx == 4:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=56,
                w=56,
                c=32,
                k0=32,
                k1=32,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                wave_size=64,
                cache_input_footprint=True,
            ),
            "gfx950",
        )
    if idx == 5:
        return (
            make_deep_fused_conv_pool_spec(
                n=1,
                h=28,
                w=28,
                c=64,
                k0=64,
                k1=64,
                r=3,
                s=3,
                pool_tile_h=4,
                pool_tile_w=8,
                tile_n=32,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=32,
                warp_tile_n=32,
                warp_tile_k=16,
                wave_size=64,
                direct_conv0_from_input_cache=True,
            ),
            "gfx950",
        )
    if idx == 6:
        # Passing config that BOTH the gate accepts AND the emitter supports
        # (16x16x16 MFMA atom has a verified layout map; the 32x32x16 atom the
        # original configs use does not -- Python itself raises NotImplemented
        # for those). Proves the emit path is byte-faithful, not just reject.
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
                tile_n=16,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
                wave_size=64,
            ),
            "gfx950",
        )
    if idx == 7:
        # gfx1201 WMMA passing/emittable config.
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
                tile_n=16,
                tile_k=16,
                warp_m=2,
                warp_n=1,
                warp_tile_m=16,
                warp_tile_n=16,
                warp_tile_k=16,
                wave_size=32,
            ),
            "gfx1201",
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_deep_fused_conv_pool,
        usage="usage: deep_fused_conv_pool_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
