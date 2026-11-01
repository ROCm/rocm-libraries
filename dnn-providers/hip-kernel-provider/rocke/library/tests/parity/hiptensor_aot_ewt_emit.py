#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_ewt_emit.py -- Python reference emitter for the
# tiled (transposing) elementwise kernel of hipTensor's kernel pack
# (kernels.common.hiptensor_aot.ewt): every pack row. C twin:
# hiptensor_aot_ewt_emit.c.
from kernels.common.hiptensor_aot.ewt import EwtSpec, build_ewt
from _emit_common import run_emit
from _hiptensor_aot_rows import ewt_rows

_ROWS = ewt_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    return EwtSpec(**_ROWS[idx])


def _build(spec, arch=None):
    return build_ewt(spec)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_ewt_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
