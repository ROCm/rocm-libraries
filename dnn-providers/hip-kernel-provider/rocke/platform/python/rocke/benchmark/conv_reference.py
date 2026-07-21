# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""torch-based reference convolution for verify paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def conv_reference(A: torch.Tensor, B: torch.Tensor, p) -> torch.Tensor:
    """Compute a float32 reference output for a forward convolution problem.

    Both 2-D (NHWC input, KHWC weight) and 3-D (NDHWC input, KDHWC weight)
    problems are supported.  The computation is always done in float32.
    The output layout matches the kernel convention: NHWC for 2-D, NDHWC for 3-D.

    Args:
        A: Input tensor, shape (N, H, W, C) or (N, D, H, W, C), any dtype.
        B: Weight tensor, shape (K, Y, X, C) or (K, Z, Y, X, C), any dtype.
        p: ConvProblem instance carrying stride/padding/dilation/groups.

    Returns:
        Reference output as a float32 torch.Tensor.
    """
    if not p.is_3d:
        A_t = A.float().cuda().permute(0, 3, 1, 2)  # NHWC -> NCHW
        B_t = B.float().cuda().permute(0, 3, 1, 2)  # KHWC -> KCHW
        return (
            F.conv2d(
                A_t,
                B_t,
                stride=(p.sH, p.sW),
                padding=(p.pH, p.pW),
                dilation=(p.dH, p.dW),
                groups=p.groups,
            )
            .permute(0, 2, 3, 1)  # NCHW -> NHWC
            .contiguous()
        )
    else:
        A_t = A.float().cuda().permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
        B_t = B.float().cuda().permute(0, 4, 1, 2, 3)  # KDHWC -> KCDHW
        return (
            F.conv3d(
                A_t,
                B_t,
                stride=(p.sD, p.sH, p.sW),
                padding=(p.pD, p.pH, p.pW),
                dilation=(p.dD, p.dH, p.dW),
                groups=p.groups,
            )
            .permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC
            .contiguous()
        )
