#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_rreduce_emit.py -- Python reference emitter for the tuned
# reduction classes of hipTensor's kernel pack (kernels.common.hiptensor_aot.rreduce): a fixed
# set covering the builder's paths, every class of the packs of hipTensor's targets, and the
# finalize classes (_hiptensor_aot_rows.rreduce_rows). C twin: hiptensor_aot_rreduce_emit.c.
from kernels.common.hiptensor_aot.rreduce import (
    RreduceFinalSpec,
    RreduceSpec,
    build_rreduce,
    build_rreduce_final,
)
from _emit_common import run_emit
from _hiptensor_aot_rows import rreduce_rows

_ROWS = rreduce_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    kind, f, arch = _ROWS[idx]
    spec = RreduceSpec(**f) if kind == "class" else RreduceFinalSpec(**f)
    return spec, arch


def _build(spec, arch=None):
    if isinstance(spec, RreduceSpec):
        return build_rreduce(spec)
    return build_rreduce_final(spec)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_rreduce_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
