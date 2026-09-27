# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compatibility entry point for gfx1151 f16 WMMA GEMM verification."""

from __future__ import annotations

import importlib


def main() -> int:
    verifier = importlib.import_module(
        "rocke.examples.gfx1151.gemm.scripts.01_f16_verify"
    )
    return verifier.main()


if __name__ == "__main__":
    raise SystemExit(main())
