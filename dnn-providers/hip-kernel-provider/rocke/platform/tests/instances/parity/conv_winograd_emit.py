#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_winograd_emit.py -- Python reference emitter for the
# Winograd convolution parity harness.
#
# Selects one of N sampled spec configs by argv[1], builds a WinogradConvSpec,
# builds all three transform kernels (data / filter / output) via the
# build_winograd_* builders, and prints lower_kernel_to_llvm(arch=<cfg arch>)
# for the selected kernel to stdout so it can be byte-compared with the C
# emitter conv_winograd_emit.c.
#
# argv[2] selects the sub-kernel within the config:
#   "data"   -- data transform kernel   (default)
#   "filter" -- filter transform kernel
#   "output" -- output transform kernel
#
# NOTE: The C counterpart (conv_winograd_emit.c) is a placeholder stub until
# the C++ engine mirror is implemented. Until then, run_diff.py will report
# this family as UNSUPPORTED on the C side, which is the gate-passing status
# for new families not yet mirrored in C++.

from rocke.instances.common.conv_winograd import (
    WinogradConvSpec,
    WinogradProblem,
    build_winograd_data_transform,
    build_winograd_filter_transform,
    build_winograd_output_transform,
)
from _emit_common import run_emit
import sys


def _spec(idx: int):
    """Return (spec, arch, sub_kernel) for config index `idx`.

    Sub-kernel is one of "data", "filter", "output"; it is baked into the
    config index so run_diff.py can treat each transform as a separate config.
    Each logical problem contributes three consecutive indices:
        idx 0,1,2  -- F(4,3) N8 H56 W56 C64 K64  data/filter/output
        idx 3,4,5  -- F(2,3) N8 H56 W56 C64 K64  data/filter/output
        idx 6,7,8  -- F(4,3) N4 H28 W28 C128 K128  data/filter/output
        idx 9,10,11-- F(4,3) N1 H7 W7 C512 K512  data/filter/output
    """
    _sub = ["data", "filter", "output"]

    if 0 <= idx <= 2:
        p = WinogradProblem(N=8, Hi=56, Wi=56, C=64, K=64, pH=1, pW=1)
        spec = WinogradConvSpec(
            problem=p, out_tile=4, block_c=32, block_k=32, block_nhw=4
        )
        return spec, "gfx950", _sub[idx % 3]

    if 3 <= idx <= 5:
        p = WinogradProblem(N=8, Hi=56, Wi=56, C=64, K=64, pH=1, pW=1)
        spec = WinogradConvSpec(
            problem=p, out_tile=2, block_c=32, block_k=32, block_nhw=4
        )
        return spec, "gfx950", _sub[idx % 3]

    if 6 <= idx <= 8:
        p = WinogradProblem(N=4, Hi=28, Wi=28, C=128, K=128, pH=1, pW=1)
        spec = WinogradConvSpec(
            problem=p, out_tile=4, block_c=32, block_k=32, block_nhw=4
        )
        return spec, "gfx950", _sub[idx % 3]

    if 9 <= idx <= 11:
        p = WinogradProblem(N=1, Hi=7, Wi=7, C=512, K=512, pH=1, pW=1)
        spec = WinogradConvSpec(
            problem=p, out_tile=4, block_c=32, block_k=32, block_nhw=1
        )
        return spec, "gfx942", _sub[idx % 3]

    return None


def _build(idx: int):
    """Return (kernel, arch) for the given config index."""
    result = _spec(idx)
    if result is None:
        return None
    spec, arch, sub = result
    builders = {
        "data": build_winograd_data_transform,
        "filter": build_winograd_filter_transform,
        "output": build_winograd_output_transform,
    }
    kernel = builders[sub](spec, arch=arch)
    return kernel, arch


def _spec_only(idx: int):
    result = _spec(idx)
    if result is None:
        return None
    spec, arch, sub = result
    return spec, arch


def _build_kernel(spec, arch):
    # run_emit calls build_fn(spec, arch=arch); we ignore spec here since
    # _build() already encodes the sub-kernel selection by idx.
    raise NotImplementedError("use _build() directly")


# run_emit expects spec_fn(idx) -> (spec, arch) and build_fn(spec, arch=arch).
# For Winograd we bypass this by providing a wrapped spec_fn that returns the
# fully-built kernel directly.
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("usage: conv_winograd_emit.py <config_index> [ll|ir|verify]\n")
        raise SystemExit(2)

    idx = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "ll"

    result = _build(idx)
    if result is None:
        # Signal "unknown config" — run_diff stops enumeration here.
        sys.stderr.write(f"unknown config {idx}\n")
        raise SystemExit(1)

    kernel, arch = result

    try:
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as _native_lower
    except ImportError:
        from rocke import lower_kernel_to_llvm as _native_lower

    from rocke.core.ir_serialize import serialize
    from rocke.core.verify import verify

    if mode == "ll":
        sys.stdout.write(_native_lower(kernel, arch=arch))
    elif mode == "ir":
        sys.stdout.write(serialize(kernel))
    elif mode == "verify":
        sys.stdout.write(verify(kernel, arch=arch))
    else:
        sys.stderr.write(f"unknown mode {mode}\n")
        raise SystemExit(2)
