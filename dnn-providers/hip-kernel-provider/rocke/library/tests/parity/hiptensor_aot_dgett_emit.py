#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# tests/parity/hiptensor_aot_dgett_emit.py -- Python reference emitter for the
# direct GETT kernels of hipTensor's kernel pack
# (kernels.common.hiptensor_aot.dgett): every pack descriptor (gfx950, gfx942,
# gfx90a x real / complex x compute type x load directions x math x 32- /
# 64-bit offsets; a descriptor the target cannot build is rejected by both
# engines). C twin: hiptensor_aot_dgett_emit.c.
from kernels.common.hiptensor_aot.dgett import DgettSpec, build_dgett
from _emit_common import run_emit

ARCHES = ("gfx950", "gfx942", "gfx90a")


def _rows():
    out = []
    for arch in ARCHES:
        for cplx in (False, True):
            for comp in ("f32", "f64") if cplx else ("f16", "bf16", "f32", "f64"):
                for a_fast in ("k", "r"):
                    for b_fast in ("k", "r"):
                        for math in (False,) if cplx else (False, True):
                            for idx64 in (False, True):
                                out.append(
                                    (
                                        DgettSpec(
                                            comp=comp,
                                            cplx=cplx,
                                            a_fast=a_fast,
                                            b_fast=b_fast,
                                            math=math,
                                            idx64=idx64,
                                        ),
                                        arch,
                                    )
                                )
    return out


_ROWS = _rows()


def _spec(idx: int):
    if not 0 <= idx < len(_ROWS):
        raise SystemExit(f"unknown config index {idx}")
    return _ROWS[idx]


def _build(spec, arch=None):
    return build_dgett(spec, arch=arch)


def main() -> int:
    return run_emit(
        _spec, _build, usage="usage: hiptensor_aot_dgett_emit.py <config_index>\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
