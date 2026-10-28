#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Representative MX FP4 TDM/LDS GEMM parity emitter. Keep its spec order
# index-for-index with gfx1250_mxfp4_tdm_gemm_emit.c.
from rocke.instances.gfx1250.mxfp4_tdm_gemm import (
    MxFp4TdmGemmSpec,
    build_mxfp4_tdm_gemm,
)

from _emit_common import run_emit


def _spec(idx: int) -> MxFp4TdmGemmSpec:
    if idx == 0:
        return MxFp4TdmGemmSpec(
            "mxfp4_tdm_gemm", 64, 32, 256, tile_m=64, tile_n=32
        )
    raise SystemExit(f"unknown config index {idx}")


def main() -> int:
    return run_emit(
        _spec,
        build_mxfp4_tdm_gemm,
        usage="usage: gfx1250_mxfp4_tdm_gemm_emit.py <config_index 0..0> [mode]\n",
        arch="gfx1250",
    )


if __name__ == "__main__":
    raise SystemExit(main())
