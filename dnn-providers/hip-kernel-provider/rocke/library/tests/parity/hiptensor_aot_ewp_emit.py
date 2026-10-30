#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_ewp_emit.py -- Python reference emitter for the tuned tiled
# elementwise kernels of hipTensor's kernel pack (ewp.build_ewp): the knob coverage set and every
# class of the packs. C twin: hiptensor_aot_ewp_emit.c.
from kernels.common.hiptensor_aot.ewp import EwpSpec, build_ewp
from _emit_common import run_emit
from _hiptensor_aot_rows import ewp_rows

_ROWS = ewp_rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    return EwpSpec(**_ROWS[idx])


def _build(spec, arch=None):
    return build_ewp(spec)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_ewp_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
