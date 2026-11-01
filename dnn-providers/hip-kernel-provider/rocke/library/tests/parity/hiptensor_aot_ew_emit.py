#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_ew_emit.py -- Python reference emitter for the
# shape-generic elementwise kernel of hipTensor's kernel pack
# (kernels.common.hiptensor_aot.ew). Config idx selects one pack row (every
# row of the pack: 32- / 64-bit offsets, mixed types, complex, math); the C
# emitter hiptensor_aot_ew_emit.c builds the same row with
# rocke_build_htaot_ew.
from kernels.common.hiptensor_aot.ew import EwSpec, build_ew
from _emit_common import run_emit
from _hiptensor_aot_rows import ew_rows

_ROWS = ew_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    return EwSpec(**_ROWS[idx])


def _build(spec, arch=None):
    return build_ew(spec)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_ew_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
