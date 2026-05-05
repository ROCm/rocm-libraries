# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Stride computation helpers and convolution output-dimension formula."""

import math
from typing import List


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


_VALID_2D_LAYOUTS = {"NCHW", "NHWC"}
_VALID_3D_LAYOUTS = {"NCDHW", "NDHWC"}


def _input_strides(
    layout: str, N: int, C: int, H: int, W: int, D: int = 0
) -> List[int]:
    """Return strides for an input tensor given its memory layout."""
    if D > 0:
        if layout not in _VALID_3D_LAYOUTS:
            raise ValueError(
                f"Unsupported 3D layout {layout!r}, expected one of {sorted(_VALID_3D_LAYOUTS)}"
            )
        if layout == "NDHWC":
            return ndhwc_strides(N, C, D, H, W)
        return ncdhw_strides(N, C, D, H, W)
    if layout not in _VALID_2D_LAYOUTS:
        raise ValueError(
            f"Unsupported 2D layout {layout!r}, expected one of {sorted(_VALID_2D_LAYOUTS)}"
        )
    if layout == "NHWC":
        return nhwc_strides(N, C, H, W)
    return nchw_strides(N, C, H, W)


def _weight_strides(
    K: int, Cg: int, R: int, S: int, D: int = 0, layout: str = "NCHW"
) -> List[int]:
    """Weight strides for dims [K, Cg, R, S] (or [K, Cg, D, R, S] for 3D).

    NCHW/NCDHW → row-major KCRS / KCDRS (Cg innermost after spatial).
    NHWC/NDHWC → KRSC / KDRSC (Cg is the fastest-moving dimension).
    """
    if D > 0:
        if layout not in _VALID_3D_LAYOUTS:
            raise ValueError(
                f"Unsupported 3D weight layout {layout!r}, expected one of {sorted(_VALID_3D_LAYOUTS)}"
            )
        if layout == "NDHWC":
            return [D * R * S * Cg, 1, R * S * Cg, S * Cg, Cg]
        return [Cg * D * R * S, D * R * S, R * S, S, 1]
    if layout not in _VALID_2D_LAYOUTS:
        raise ValueError(
            f"Unsupported 2D weight layout {layout!r}, expected one of {sorted(_VALID_2D_LAYOUTS)}"
        )
    if layout == "NHWC":
        return [R * S * Cg, 1, S * Cg, Cg]
    return [Cg * R * S, R * S, S, 1]


def conv_out_dim(dim_in: int, pad: int, dilation: int, kernel: int, stride: int) -> int:
    return math.floor((dim_in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)
