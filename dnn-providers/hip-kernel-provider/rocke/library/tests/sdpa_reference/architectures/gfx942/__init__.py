# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Qualified gfx942 SDPA cohort and current-source dispatch adapter."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from reference_common.cases import index_cases

from ...contract import Case
from ...implementations import dense
from ..common import attention_request, validate_library_root

if TYPE_CHECKING:
    from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec

NAME = "gfx942"

CASES = tuple(
    Case(*row)
    for row in (
        ("fp16", 128, 16, 4, False, True),
        ("fp16", 128, 16, 4, True, True),
        ("bf16", 128, 16, 4, True, True),
        ("bf16", 128, 16, 4, False, True),
        ("fp16", 64, 16, 16, False, True),
        ("bf16", 64, 16, 4, True, True),
        ("fp16", 128, 16, 16, False, True),
        ("fp16", 128, 16, 4, False, False),
    )
)
CASE_BY_ID = index_cases(CASES, label="SDPA")


def prepare(case: Case, library_root: str) -> Gfx942AttentionDenseSpec:
    """Select the current kernel through gfx942's production dispatch policy."""
    validate_library_root(library_root)
    import dispatch.attention as attention

    request = attention_request(case, NAME)
    # Recognize published baseline sources by their own API, so a renamed
    # current API fails on its name rather than inside the historical path.
    if not hasattr(attention, "tuning_spec_with_knobs") and hasattr(
        attention, "dense_spec_for_request"
    ):
        # Published baseline sources select this knob through the request.
        request = replace(request, dense_persistent="on" if case.persistent else "off")
        return attention.dense_spec_for_request(request)
    return attention.tuning_spec_with_knobs(
        request,
        "gfx942_dense",
        {"persistent": case.persistent},
    ).kernel_spec


def launch(spec: Gfx942AttentionDenseSpec, buffers: dict, scale: float) -> None:
    """Use the same public launch entry as the existing gfx942 numeric tests."""
    from kernels.gfx942 import attention_dense

    dense.launch(attention_dense, spec, buffers, scale)


def exported_kernel(spec: Gfx942AttentionDenseSpec) -> tuple[bytes, dict]:
    """Describe the launcher that actually executed during qualification."""
    from kernels.gfx942 import attention_dense

    return dense.exported_kernel(attention_dense, spec, NAME)
