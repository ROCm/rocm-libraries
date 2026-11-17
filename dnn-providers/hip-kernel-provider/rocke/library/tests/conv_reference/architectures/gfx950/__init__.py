# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 forward cohort using production implicit-GEMM dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from reference_common.cases import index_cases

from ...contract import Case
from ...implementations import implicit_gemm_fwd
from ..common import convolution_request, validate_library_root

if TYPE_CHECKING:
    from rocke.runtime import KernelLauncher

NAME = "gfx950"
CASES = tuple(
    Case(name=name, dtype=dtype, **geometry)
    for name, geometry in (
        ("padded", {}),
        ("pointwise", {"Y": 1, "X": 1, "pH": 0, "pW": 0}),
        ("stride2", {"sH": 2, "sW": 2}),
        ("dilation2", {"dH": 2, "dW": 2, "pH": 2, "pW": 2}),
        ("groups2", {"groups": 2}),
        ("asymmetric", {"Hi": 7, "Wi": 11, "N": 2}),
        ("multi-tile", {"K": 96}),
        ("odd-channels", {"K": 33}),
        ("odd-input-channels", {"C": 33}),
        ("stem", {"C": 3, "Hi": 16, "Wi": 16}),
        ("large", {"N": 2, "Hi": 14, "Wi": 14, "C": 64, "K": 128}),
        ("long-k", {"Y": 5, "X": 5, "pH": 2, "pW": 2, "sH": 2, "sW": 2}),
        ("grouped-odd", {"C": 66, "K": 66, "groups": 2}),
        ("depthwise", {"C": 64, "K": 64, "groups": 64}),
        ("anisotropic", {"Y": 1, "X": 7, "pH": 0, "pW": 3, "sW": 2}),
        ("out-1x1", {"Hi": 3, "Wi": 3, "pH": 0, "pW": 0}),
    )
    for dtype in ("fp16", "bf16")
)
CASE_BY_ID = index_cases(CASES, label="convolution")


def prepare(case: Case, library_root: str) -> tuple[KernelLauncher, dict]:
    """Select the current kernel through gfx950's production dispatch policy."""
    validate_library_root(library_root)
    from dispatch.grouped_convolution import dispatch_conv_grouped

    request = convolution_request(case, NAME)
    selected = dispatch_conv_grouped(request)
    return implicit_gemm_fwd.prepare(case, request, selected)


def launch(prepared: tuple[KernelLauncher, dict], buffers: dict) -> None:
    implicit_gemm_fwd.launch(prepared, buffers)


def exported_kernel(prepared: tuple[KernelLauncher, dict]) -> tuple[bytes, dict]:
    return implicit_gemm_fwd.exported_kernel(prepared)
