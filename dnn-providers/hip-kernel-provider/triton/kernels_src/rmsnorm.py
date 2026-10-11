# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RMSNorm Triton kernel (one program per row, no bias, no inverse-RMS output)."""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(x_ptr, scale_ptr, y_ptr, n_cols, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    offs = row * n_cols + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1 / tl.sqrt(var + eps)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * scale
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
