#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_jit_ew_emit.py -- Python reference emitter for the
# problem-specialized elementwise kernels of hipTensor's JIT
# (jit_ew.build_ew_jit with the geometry as constants): every JIT class of
# the generic rows (32- / 64-bit offsets, complex, math) at one geometry
# drawn from a 64-bit LCG seeded by the config index. C twin:
# hiptensor_aot_jit_ew_emit.c.
from kernels.common.hiptensor_aot.ew import EwSpec
from kernels.common.hiptensor_aot.jit_ew import EwJitSpec, build_ew_jit
from _emit_common import run_emit
from _hiptensor_aot_rows import geometry, jit_rows

_ROWS = jit_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    p, vec, unit = _ROWS[idx]
    spec = EwJitSpec(EwSpec(**p), vec, unit)
    return (spec, geometry([n for n, _ in spec.axes()], idx)), "gfx950"


def _build(sg, arch=None):
    spec, geom = sg
    return build_ew_jit(spec, geom)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_jit_ew_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
