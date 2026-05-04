# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Stride computation helpers and convolution output-dimension formula."""

import math
from typing import List, Optional


def nchw_strides(N: int, C: int, H: int, W: int) -> List[int]:
    return [C * H * W, H * W, W, 1]


def nhwc_strides(N: int, C: int, H: int, W: int) -> List[int]:
    # Dims stay as [N, C, H, W]; strides reflect NHWC memory order
    return [H * W * C, 1, W * C, C]


def ncdhw_strides(N: int, C: int, D: int, H: int, W: int) -> List[int]:
    return [C * D * H * W, D * H * W, H * W, W, 1]


def ndhwc_strides(N: int, C: int, D: int, H: int, W: int) -> List[int]:
    # Dims stay as [N, C, D, H, W]; strides reflect NDHWC memory order
    return [D * H * W * C, 1, H * W * C, W * C, C]


def _input_strides(
    layout: str, N: int, C: int, H: int, W: int, D: Optional[int] = None
) -> List[int]:
    """Return strides for an input tensor given its memory layout."""
    if D is not None:
        if layout == "NDHWC":
            return ndhwc_strides(N, C, D, H, W)
        return ncdhw_strides(N, C, D, H, W)
    if layout == "NHWC":
        return nhwc_strides(N, C, H, W)
    return nchw_strides(N, C, H, W)


def _weight_strides(
    K: int, Cg: int, R: int, S: int, D: Optional[int] = None, layout: str = "NCHW"
) -> List[int]:
    """Weight strides for dims [K, Cg, R, S] (or [K, Cg, D, R, S] for 3D).

    NCHW/NCDHW → row-major KCRS / KCDRS (Cg innermost after spatial).
    NHWC/NDHWC → KRSC / KDRSC (Cg is the fastest-moving dimension).
    """
    if D is not None:
        if layout in ("NHWC", "NDHWC"):
            # KDRSC: stride[K]=D*R*S*Cg, stride[Cg]=1, stride[D]=R*S*Cg,
            #        stride[R]=S*Cg, stride[S]=Cg
            return [D * R * S * Cg, 1, R * S * Cg, S * Cg, Cg]
        return [Cg * D * R * S, D * R * S, R * S, S, 1]
    if layout in ("NHWC",):
        # KRSC: stride[K]=R*S*Cg, stride[Cg]=1, stride[R]=S*Cg, stride[S]=Cg
        return [R * S * Cg, 1, S * Cg, Cg]
    return [Cg * R * S, R * S, S, 1]


def conv_out_dim(dim_in: int, pad: int, dilation: int, kernel: int, stride: int) -> int:
    return math.floor((dim_in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)
