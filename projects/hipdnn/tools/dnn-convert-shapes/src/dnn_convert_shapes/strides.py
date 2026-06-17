# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Stride computation helpers and convolution output-dimension formula."""

import enum
import math
from typing import List


class Layout(enum.Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCDHW = "NCDHW"
    NDHWC = "NDHWC"

    @classmethod
    def parse(cls, value: str | None, default: "Layout") -> "Layout":
        if value is None:
            return default
        try:
            return cls(value)
        except ValueError:
            raise ValueError(
                f"Unknown layout {value!r}, "
                f"expected one of {[l.value for l in cls]}"
            )


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


_VALID_2D_LAYOUTS = {Layout.NCHW, Layout.NHWC}
_VALID_3D_LAYOUTS = {Layout.NCDHW, Layout.NDHWC}


def _input_strides(
    layout: Layout, N: int, C: int, H: int, W: int, D: int = 0
) -> List[int]:
    """Return strides for an input tensor given its memory layout."""
    if D > 0:
        if layout not in _VALID_3D_LAYOUTS:
            raise ValueError(
                f"Unsupported 3D layout {layout!r}, expected one of {sorted(l.value for l in _VALID_3D_LAYOUTS)}"
            )
        match layout:
            case Layout.NDHWC:
                return ndhwc_strides(N, C, D, H, W)
            case Layout.NCDHW:
                return ncdhw_strides(N, C, D, H, W)
    if layout not in _VALID_2D_LAYOUTS:
        raise ValueError(
            f"Unsupported 2D layout {layout!r}, expected one of {sorted(l.value for l in _VALID_2D_LAYOUTS)}"
        )
    match layout:
        case Layout.NHWC:
            return nhwc_strides(N, C, H, W)
        case Layout.NCHW:
            return nchw_strides(N, C, H, W)


def _weight_strides(
    K: int, Cg: int, R: int, S: int, D: int = 0, layout: Layout = Layout.NCHW
) -> List[int]:
    """Weight strides for dims [K, Cg, R, S] (or [K, Cg, D, R, S] for 3D).

    NCHW/NCDHW → row-major KCRS / KCDRS (Cg innermost after spatial).
    NHWC/NDHWC → KRSC / KDRSC (Cg is the fastest-moving dimension).
    """
    if D > 0:
        if layout not in _VALID_3D_LAYOUTS:
            raise ValueError(
                f"Unsupported 3D weight layout {layout!r}, expected one of {sorted(l.value for l in _VALID_3D_LAYOUTS)}"
            )
        match layout:
            case Layout.NDHWC:
                return [D * R * S * Cg, 1, R * S * Cg, S * Cg, Cg]
            case Layout.NCDHW:
                return [Cg * D * R * S, D * R * S, R * S, S, 1]
    if layout not in _VALID_2D_LAYOUTS:
        raise ValueError(
            f"Unsupported 2D weight layout {layout!r}, expected one of {sorted(l.value for l in _VALID_2D_LAYOUTS)}"
        )
    match layout:
        case Layout.NHWC:
            return [R * S * Cg, 1, S * Cg, Cg]
        case Layout.NCHW:
            return [Cg * R * S, R * S, S, 1]


def conv_out_dim(dim_in: int, pad: int, dilation: int, kernel: int, stride: int) -> int:
    return math.floor((dim_in + 2 * pad - dilation * (kernel - 1) - 1) / stride + 1)
