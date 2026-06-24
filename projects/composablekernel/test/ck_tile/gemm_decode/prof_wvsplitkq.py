#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Minimal wvSplitKQ-only launch loop for rocprofv3 side-by-side profiling.

Mirrors wvsplitk_msweep.py's _wvsplitkq_run but strips everything else (no
gemm_a8w8, no per-iter alloc, no timing) so the only repeated dispatch rocprof
sees in the timed region is wvSplitKQ. Filter the counter CSV by Kernel_Name
~ /wvSplit/ to drop the one-time quant/setup kernels.

  rocprofv3 --pmc MeanOccupancyPerCU -f csv -d OUT -- \
    python3 prof_wvsplitkq.py 40        # N,K,M via env (default 7168/7168/1)
"""
import importlib.util
import os
import sys

import torch  # noqa: E402,F401 - resolve libtorch symbols first

# Reuse the harness's _import_aiter (meta-path shim that dodges the broken
# aiter.utility.mx_types eager import) so this matches the working msweep path.
_HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wvsplitk_msweep.py")
_spec = importlib.util.spec_from_file_location("_wvh", _HARNESS)
_wvh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wvh)
aiter = _wvh._import_aiter(os.environ.get("AITER_DIR", "/home/AMD/samremes/dev/aiter"))
from aiter import dtypes  # noqa: E402

N = int(os.environ.get("N", 7168))
K = int(os.environ.get("K", 7168))
M = int(os.environ.get("M", 1))
ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 40

dev = torch.device("cuda")
fp8 = dtypes.fp8
cu = torch.cuda.get_device_properties(0).multi_processor_count

Abf = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
Bbf = torch.randn((N, K), dtype=torch.bfloat16, device=dev) * 0.1
Aq, x_scale = aiter.per_tensor_quant(Abf, quant_dtype=fp8)
Bq, w_scale = aiter.per_tensor_quant(Bbf, quant_dtype=fp8)
Aqc = Aq.contiguous()
out = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

for _ in range(5):  # warmup (excluded by averaging over the long timed region)
    aiter.wvSplitKQ(Bq, Aqc, out, w_scale, x_scale, cu)
torch.cuda.synchronize()

for _ in range(ITERS):  # profiled region: wvSplitKQ only
    aiter.wvSplitKQ(Bq, Aqc, out, w_scale, x_scale, cu)
torch.cuda.synchronize()
print(f"# wvSplitKQ x{ITERS} done: M={M} N={N} K={K} cu={cu}", file=sys.stderr)
