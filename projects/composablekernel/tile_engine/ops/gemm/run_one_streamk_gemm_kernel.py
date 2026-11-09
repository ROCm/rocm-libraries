#!/usr/bin/env python3
"""Worker script for running Stream-K GEMM kernels in an isolated subprocess.

Stream-K is a single-problem GEMM with the same C ABI as regular GEMM, so this
worker is the GEMM worker (``run_one_gemm_kernel.py``): the same input and
output JSON, ``verify`` and the search's options (``check_identity``, the host
cache, ``arch``, ``prepare``) included. The Stream-K specifics live inside the
force-included kernel and ``streamk_gemm_ctypes_lib.cpp``, which reads the
workgroup budget (``gemm_utils.STREAMK_BUDGET_ENV``, percent of CUs x
occupancy) at each launch: from the worker's environment
(``streamk_gemm_full_benchmark.py --budget-pct``) or from an item's ``env``
(``gemm_full_benchmark.py --variant gemm_streamk --budget-pct``, one budget
per item).

Atomic reduction rounds each partial update to the output precision; many
contributors can exceed the default --verify tolerance (2e-2). Linear/tree
reduction combines partials in fp32 before converting the output.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from run_one_gemm_kernel import main  # noqa: E402

if __name__ == "__main__":
    main()
