#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Worker script for running BATCHED GEMM kernels in an isolated subprocess.

The batched worker is the GEMM worker (``run_one_gemm_kernel.py``): a problem
with ``batch_count`` ({"batch_count":.., "M":.., "N":.., "K":..}) runs on
``batched_gemm_utils.GpuBatchedGemmRunner``, with the same input and output
JSON, ``verify`` (an fp32 per-batch ``A @ B`` reference on the inputs the
kernel reads) and the adaptive search's options (``check_identity``, the host
cache, ``prepare``). An item's or the payload's ``arch`` is the target the
.so was built for; it fixes the fp8/bf8 host encoding instead of probing the
local GPU.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from run_one_gemm_kernel import main  # noqa: E402

if __name__ == "__main__":
    main()
