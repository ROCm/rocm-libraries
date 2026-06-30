# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx942 Universal GEMM showcase.

Thin per-arch wrapper over the shared ``examples/common/universal_gemm_verify``
harness: pins ``--arch gfx942`` and lets the catalog pick the gfx942-legal MFMA
atom (16x16x16 / 32x32x8 — never the gfx950-only wide atoms). Demonstrates the
hybrid-layout pattern: arch showcases add intent, not duplicated plumbing.

  PYTHONPATH=Python python3 -m rocke.examples.gfx942.gemm_demo
"""

from __future__ import annotations

import sys

from rocke.examples.common import universal_gemm_verify


def main() -> int:
    argv = sys.argv[1:]
    if "--arch" not in argv:
        argv = ["--arch", "gfx942", *argv]
    sys.argv = [sys.argv[0], *argv]
    return universal_gemm_verify.main()


if __name__ == "__main__":
    raise SystemExit(main())
