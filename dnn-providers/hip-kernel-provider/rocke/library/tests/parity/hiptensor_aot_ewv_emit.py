#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_ewv_emit.py -- Python reference emitter for the
# vector class kernels of hipTensor's kernel pack (jit_ew.build_ew_jit with
# the geometry as kernel arguments): every pack class. C twin:
# hiptensor_aot_ewv_emit.c.
from kernels.common.hiptensor_aot.ew import EwSpec
from kernels.common.hiptensor_aot.jit_ew import EwJitSpec, build_ew_jit
from _emit_common import run_emit
from _hiptensor_aot_rows import ewv_rows

_ROWS = ewv_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    p, vec, unit = _ROWS[idx]
    return EwJitSpec(EwSpec(**p), vec, unit)


def _build(spec, arch=None):
    return build_ew_jit(spec)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_ewv_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
