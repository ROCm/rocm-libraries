# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Source isolation and requests shared by convolution reference adapters."""

from __future__ import annotations

import os
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
        algorithm="implicit_gemm",
    )


def compile_reference_kernel(kernel, architecture: str, llvm_flavor: str | None = None):
    """Compile an instance using its named IR flavor, restoring worker settings.

    Registered AOT identities include the lowering flavor. A newer installed
    compiler may accept that flavor, but must not silently select another IR
    dialect for the same identity. Reference workers compile serially.
    """
    from rocke import compile_kernel

    if llvm_flavor is None:
        return compile_kernel(kernel, arch=architecture, backend="python")
    previous = os.environ.get("ROCKE_LLVM_FLAVOR")
    os.environ["ROCKE_LLVM_FLAVOR"] = llvm_flavor
    try:
        return compile_kernel(kernel, arch=architecture, backend="python")
    finally:
        if previous is None:
            os.environ.pop("ROCKE_LLVM_FLAVOR", None)
        else:
            os.environ["ROCKE_LLVM_FLAVOR"] = previous
