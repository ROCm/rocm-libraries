# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""torch-based reference convolution for verify paths."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def wgrad_reference(X: torch.Tensor, dY: torch.Tensor, p) -> torch.Tensor:
    """Compute a float32 reference weight gradient for a convolution problem.

    Uses ``torch.nn.grad.conv2d_weight`` / ``conv3d_weight`` so the result is
    numerically identical to what autograd would produce.  The output layout
    matches the wgrad kernel convention: KYXC for 2-D, KZYXC for 3-D.

    Args:
        X:  Input activations, shape (N, H, W, C) or (N, D, H, W, C), any dtype.
        dY: Output gradient, shape (N, Ho, Wo, K) or (N, Do, Ho, Wo, K), any dtype.
        p:  ConvProblem carrying stride/padding/dilation/groups.

    Returns:
        Weight gradient as a float32 torch.Tensor in KYXC / KZYXC layout.
    """
    if not p.is_3d:
        X_t = X.float().cuda().permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        dY_t = dY.float().cuda().permute(0, 3, 1, 2).contiguous()  # NHWK -> NKHW
        dW_nchw = torch.nn.grad.conv2d_weight(
            X_t,
            weight_size=(p.K, p.C // p.groups, p.Y, p.X),
            grad_output=dY_t,
            stride=(p.sH, p.sW),
            padding=(p.pH, p.pW),
            dilation=(p.dH, p.dW),
            groups=p.groups,
        )
        # KCHW -> KHWC (KYXC)
        return dW_nchw.permute(0, 2, 3, 1).contiguous()
    else:
        X_t = X.float().cuda().permute(0, 4, 1, 2, 3).contiguous()  # NDHWC -> NCDHW
        dY_t = dY.float().cuda().permute(0, 4, 1, 2, 3).contiguous()  # NDHWK -> NKDHW
        dW_ncdhw = torch.nn.grad.conv3d_weight(
            X_t,
            weight_size=(p.K, p.C // p.groups, p.Z, p.Y, p.X),
            grad_output=dY_t,
            stride=(p.sD, p.sH, p.sW),
            padding=(p.pD, p.pH, p.pW),
            dilation=(p.dD, p.dH, p.dW),
            groups=p.groups,
        )
        # KCDHW -> KDHWC (KZYXC)
        return dW_ncdhw.permute(0, 2, 3, 4, 1).contiguous()


def dgrad_reference(dY: torch.Tensor, W: torch.Tensor, p) -> torch.Tensor:
    """Compute a float32 reference input gradient for a convolution problem.

    Uses ``torch.nn.grad.conv2d_input`` so the result is numerically identical
    to what autograd would produce.  The output layout matches the dgrad kernel
    convention: NHWC for 2-D.

    Args:
        dY: Output gradient, shape (N, Ho, Wo, K), any dtype.
        W:  Weight tensor, shape (K, Y, X, C), any dtype.
        p:  ConvProblem carrying stride/padding/dilation/groups.

    Returns:
        Input gradient as a float32 torch.Tensor in NHWC layout.
    """
    dY_t = dY.float().cuda().permute(0, 3, 1, 2).contiguous()  # NHWK -> NKHW
    W_t = W.float().cuda().permute(0, 3, 1, 2).contiguous()  # KYXC -> KCYX
    dX_nchw = torch.nn.grad.conv2d_input(
        input_size=(p.N, p.C, p.Hi, p.Wi),
        weight=W_t,
        grad_output=dY_t,
        stride=(p.sH, p.sW),
        padding=(p.pH, p.pW),
        dilation=(p.dH, p.dW),
        groups=p.groups,
    )
    return dX_nchw.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


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
