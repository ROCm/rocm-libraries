#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_jit_ew_emit.py -- Python reference emitter for the
# problem-specialized elementwise kernels of hipTensor's JIT
# (jit_ew.build_ew_jit with the geometry as constants): every JIT class of
# the generic rows (32- / 64-bit offsets, complex, math) at three geometry
# points (_hiptensor_aot_rows.geometry_point: values drawn from a 64-bit LCG
# seeded by the class index; 0 / 1 / 2; the edges of the i32 / i64 constants
# with negative magic numbers and offsets past 2^31), each lowered for its
# own target. Config index = point * <classes> + class. C twin:
# hiptensor_aot_jit_ew_emit.c.
from kernels.common.hiptensor_aot.ew import EwSpec
from kernels.common.hiptensor_aot.jit_ew import EwJitSpec, build_ew_jit
from _emit_common import run_emit
from _hiptensor_aot_rows import GEOMETRY_POINTS, geometry_point, jit_rows

_ROWS = jit_rows()


def _spec(idx: int):
    point, cls = divmod(idx, len(_ROWS))
    if idx < 0 or point >= len(GEOMETRY_POINTS):
        raise SystemExit(f"unknown config index {idx}")
    p, vec, unit = _ROWS[cls]
    spec = EwJitSpec(EwSpec(**p), vec, unit)
    return (spec, geometry_point(spec.axes(), cls, point)), GEOMETRY_POINTS[point]


def _build(sg, arch=None):
    spec, geom = sg
    return build_ew_jit(spec, geom)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_jit_ew_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
