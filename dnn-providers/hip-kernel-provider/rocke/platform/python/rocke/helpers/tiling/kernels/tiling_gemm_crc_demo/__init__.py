# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CRC interleaved GEMM demo package.

Kernel code lives in ``crc_interleaved_gemm.py``; this re-exports the public API so
``from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo import build_crc_gemm`` keeps working.
Docs + renders: ``docs/`` (``design_report.md``, ``viz/``). Throwaway scripts: ``tmp/``.
"""
from .crc_interleaved_gemm import (
    benchmark_crc,
    build_crc_gemm,
    run_and_verify_crc,
)

__all__ = ["build_crc_gemm", "run_and_verify_crc", "benchmark_crc"]
