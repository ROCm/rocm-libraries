#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/gfx950_deep_fused_conv_pool_emit.py -- Python reference emitter
# for the gfx950 (CDNA, wave64, MFMA) arch shim over the deep fused
# conv0 -> conv1 -> maxpool kernel. Selects one of N sampled spec configs by
# argv[1], builds the Gfx950DeepFusedConvPoolSpec, builds the kernel via
# build_deep_fused_conv_pool (re-exported common driver, arch="gfx950") and
# prints lower_kernel_to_llvm(arch="gfx950") to stdout so it can be
# byte-compared with the C emitter gfx950_deep_fused_conv_pool_emit.c.
#
# WHY NOT THE gfx950 make_deep_fused_conv_pool_spec FACTORY:
#   The gfx950 public factory hard-pins the CDNA MFMA 32x32x16 atom
#   (warp_tile_m=warp_tile_n=32). That atom has NO verified MMA fragment layout
#   map, so build_deep_fused_conv_pool raises NotImplementedError before any IR
#   is emitted -- every "config" would be a no-emit reject, masking real
#   C-vs-Python divergence. The only gfx950 (wave64) geometry that actually
#   LOWERS is the MFMA 16x16x16 atom (same atom the common emit.py idx-6 uses to
#   exercise the gfx950 emit path). The gfx950 shim differs from the common spec
#   in `name` only, so we build the common spec with the wave64 16x16x16 geometry
#   + the pinned gfx950 kernel name, then re-wrap it as the gfx950 spec -- exactly
#   what the gfx950 factory does internally, minus the non-emittable 32x32 pin.
#   The v1 "one CTA owns all conv channels" gate (K <= tile_n) is satisfied with
#   tile_n=16 and K0=K1=16.
import sys
from dataclasses import fields

from ck_dsl.instances.gfx950.deep_fused_conv_pool import (
    Gfx950DeepFusedConvPoolSpec,
    build_deep_fused_conv_pool,
)
from ck_dsl.instances.common.deep_fused_conv_pool import (
    DeepFusedConvPoolSpec,
    make_deep_fused_conv_pool_spec as _make_common_spec,
)
from ck_dsl import lower_kernel_to_llvm
from ck_dsl.core.ir_serialize import serialize
from ck_dsl.core.verify import verify

_ARCH = "gfx950"
_GFX950_NAME = "ck_dsl_gfx950_deep_fused_conv_pool"


def _mk(**kwargs) -> Gfx950DeepFusedConvPoolSpec:
    """Build a gfx950 spec on the emittable wave64 MFMA 16x16x16 atom.

    Mirrors the gfx950 factory (pin name + wave64, re-wrap the common spec) but
    selects the 16x16x16 warp tile -- the only wave64 atom with a verified MMA
    fragment layout map -- so the kernel actually lowers.
    """
    base = _make_common_spec(
        name=_GFX950_NAME,
        wave_size=64,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=16,
        **kwargs,
    )
    return Gfx950DeepFusedConvPoolSpec(
        **{f.name: getattr(base, f.name) for f in fields(DeepFusedConvPoolSpec)}
    )


def _spec(idx: int):
    """Return (spec, arch) for config index `idx`. All gfx950 / wave64 MFMA
    16x16x16, tile_n=16 >= K (one CTA owns all conv channels)."""
    cfgs = {
        0: dict(
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
        ),
        1: dict(
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
            tile_n=16,
            tile_k=16,
            warp_m=2,
            warp_n=1,
        ),
        2: dict(
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
            tile_n=16,
            tile_k=16,
            warp_m=2,
            warp_n=1,
        ),
        3: dict(
            n=1,
            h=56,
            w=56,
            c=8,
            k0=16,
            k1=16,
            r=3,
            s=3,
            pool_tile_h=4,
            pool_tile_w=4,
            tile_n=16,
            tile_k=16,
            warp_m=2,
            warp_n=1,
            cache_input_footprint=True,
        ),
        4: dict(
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
            tile_n=16,
            tile_k=16,
            warp_m=2,
            warp_n=1,
            direct_conv0_from_input_cache=True,
        ),
    }
    if idx not in cfgs:
        raise SystemExit(f"unknown config index {idx}")
    return _mk(**cfgs[idx]), _ARCH


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gfx950_deep_fused_conv_pool_emit.py <config_index>\n")
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
