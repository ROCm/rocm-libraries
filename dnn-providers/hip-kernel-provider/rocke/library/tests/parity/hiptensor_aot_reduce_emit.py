#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_reduce_emit.py -- Python reference emitter for the
# reduction kernels of hipTensor's kernel pack
# (kernels.common.hiptensor_aot.reduce): every pack row (row / col / final,
# 32- / 64-bit offsets, math). C twin: hiptensor_aot_reduce_emit.c.
from kernels.common.hiptensor_aot.reduce import ReduceSpec, build_reduce
from _emit_common import run_emit
from _hiptensor_aot_rows import reduce_rows

_ROWS = reduce_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    return ReduceSpec(**_ROWS[idx])


def _build(spec, arch=None):
    return build_reduce(spec)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_reduce_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
