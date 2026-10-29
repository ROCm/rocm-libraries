#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_ewf_emit.py -- Python reference emitter for the flat elementwise
# kernels of hipTensor's kernel pack (ewf.build_ewf): every plain row, without and with
# streaming accesses. C twin: hiptensor_aot_ewf_emit.c.
from kernels.common.hiptensor_aot.ewf import EwfSpec, build_ewf
from _emit_common import run_emit
from _hiptensor_aot_rows import ewf_rows

_ROWS = ewf_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    p, nt = _ROWS[idx]
    return EwfSpec(
        arity=p["arity"], a=p["a"], d=p["d"], b=p.get("b"), c=p.get("c"), nt=nt
    )


def _build(spec, arch=None):
    return build_ewf(spec)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_ewf_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
