#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/conv_dgrad_halo_zero_emit.py -- Python reference emitter for
# the dgrad dx_interior shell zero-fill (conv_dgrad_halo_zero) parity
# harness. Selects one of N sampled spec configs by argv[1], builds the kernel
# via build_dgrad_halo_zero and prints the LLVM IR, to be byte-compared with
# the C emitter conv_dgrad_halo_zero_emit.c.
#
# Config index map (keep in lockstep with the C emitter):
#   0  block 256 (the default), gfx942
#   1  block 128, gfx942
#   2  block 512, gfx942
#   3  block 256, gfx950
from kernels.common.conv_dgrad_interior import DgradHaloZeroSpec, build_dgrad_halo_zero
from _emit_common import run_emit

_CFGS = {
    0: (256, "gfx942"),
    1: (128, "gfx942"),
    2: (512, "gfx942"),
    3: (256, "gfx950"),
}


def _spec(idx: int):
    if idx not in _CFGS:
        raise SystemExit(f"unknown config index {idx}")
    block, arch = _CFGS[idx]
    return DgradHaloZeroSpec(block=block), arch


def _build(spec, arch=None):
    return build_dgrad_halo_zero(spec)


def main() -> int:
    return run_emit(
        _spec,
        _build,
        usage="usage: conv_dgrad_halo_zero_emit.py <config_index>\n",
    )


if __name__ == "__main__":
    raise SystemExit(main())
