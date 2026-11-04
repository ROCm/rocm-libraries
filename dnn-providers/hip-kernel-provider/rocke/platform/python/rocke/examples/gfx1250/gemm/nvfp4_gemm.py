# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""NVFP4 example: packed FP4, E4M3 block scales, and FP32 tensor factors.

Use --dtype-c bf16 or fp16 and --compile-route comgr or hip. The shared
NVFP4 verifier supplies bounded input fixtures and checks the output.
"""

from .nvfp4_gemm_verify import main

if __name__ == "__main__":
    raise SystemExit(main())
