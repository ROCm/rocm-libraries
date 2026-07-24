# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reference convolutions for verify paths.

Two implementations are provided:

* ``conv_reference``        — torch-based (``F.conv2d`` / ``F.conv3d``), works
  on any GPU target.
* ``conv_reference_gfx1250`` — hand-written numpy reference for gfx1250.
  Accumulates in float32 with no GPU driver dependency.  Only supports 2-D
  NHWC forward convolution.
"""

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


def conv_reference_gfx1250(
    A: torch.Tensor,
    B: torch.Tensor,
    p,
) -> torch.Tensor:
    """Hand-written float32 reference for 2-D NHWC forward convolution.

    Implements the convolution directly in numpy so there is no dependency on
    torch.nn or any GPU driver.  Accumulation is always in float32 to keep
    the reference numerically clean even for large filter footprints.

    Supports grouped convolution (``p.groups > 1``) and arbitrary
    stride / padding / dilation.  3-D problems are not supported; call
    ``conv_reference`` instead.

    Args:
        A: Input tensor, shape ``(N, Hi, Wi, C)``, any dtype.
        B: Weight tensor, shape ``(K, Y, X, C)``, any dtype.
        p: ``ConvProblem`` instance (carries stride / padding / dilation /
           groups).

    Returns:
        Float32 torch.Tensor of shape ``(N, Ho, Wo, K)`` on CPU.
    """
    import numpy as np

    A_np = A.float().cpu().numpy()  # (N, Hi, Wi, C)
    B_np = B.float().cpu().numpy()  # (K, Y, X, C)

    N, Hi, Wi, C = A_np.shape
    K = B_np.shape[0]
    Ho, Wo = p.Ho, p.Wo
    g = p.groups
    Cg = C // g  # input channels per group
    Kg = K // g  # output channels per group

    out = np.zeros((N, Ho, Wo, K), dtype=np.float32)

    for grp in range(g):
        a_grp = A_np[:, :, :, grp * Cg : (grp + 1) * Cg]  # (N, Hi, Wi, Cg)
        b_grp = B_np[grp * Kg : (grp + 1) * Kg]  # (Kg, Y, X, Cg)
        for n in range(N):
            for ho in range(Ho):
                for wo in range(Wo):
                    for y in range(p.Y):
                        hi = ho * p.sH - p.pH + y * p.dH
                        if hi < 0 or hi >= Hi:
                            continue
                        for x in range(p.X):
                            wi = wo * p.sW - p.pW + x * p.dW
                            if wi < 0 or wi >= Wi:
                                continue
                            # a_vec: (Cg,)  b_slice: (Kg, Cg)
                            a_vec = a_grp[n, hi, wi, :].astype(np.float32)
                            b_slice = b_grp[:, y, x, :].astype(np.float32)
                            out[n, ho, wo, grp * Kg : (grp + 1) * Kg] += b_slice @ a_vec

    return torch.from_numpy(out.astype(np.float32))
