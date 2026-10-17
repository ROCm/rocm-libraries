# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Source isolation and requests shared by convolution reference adapters."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..contract import Case

if TYPE_CHECKING:
    from dispatch.grouped_convolution import ConvGroupedRequest


def validate_library_root(library_root: str) -> None:
    """Keep production imports inside the worker's selected source tree."""
    import kernels
    import dispatch

    root = Path(library_root).resolve()
    for package in (kernels, dispatch):
        if not Path(package.__file__).resolve().is_relative_to(root):
            raise RuntimeError(
                "worker imported production packages outside the selected library"
            )


def convolution_request(case: Case, architecture: str) -> ConvGroupedRequest:
    """Translate the forward reference contract into a production request."""
    from dispatch.grouped_convolution import ConvGroupedRequest

    return ConvGroupedRequest(
        N=case.N,
        C=case.C,
        K=case.K,
        Hi=case.Hi,
        Wi=case.Wi,
        Y=case.Y,
        X=case.X,
        G=case.groups,
        stride_h=case.sH,
        stride_w=case.sW,
        pad_h=case.pH,
        pad_w=case.pW,
        dilation_h=case.dH,
        dilation_w=case.dW,
        dtype=case.dtype,
        arch=architecture,
    )
