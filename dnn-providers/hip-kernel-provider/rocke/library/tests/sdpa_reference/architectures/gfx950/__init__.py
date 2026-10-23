# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Base dense gfx950 SDPA cohort and current-source dispatch adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from reference_common.cases import index_cases

from ...contract import Case
from ...implementations import dense
from ..common import attention_request, validate_library_root

if TYPE_CHECKING:
    from kernels.gfx950.attention_dense import Gfx950AttentionDenseSpec

NAME = "gfx950"

CASES = tuple(
    Case(*row)
    for row in (
        ("fp16", 128, 16, 4, False, True),
        ("fp16", 128, 16, 4, True, True),
        ("bf16", 128, 16, 4, True, True),
        ("bf16", 128, 16, 4, False, True),
        ("fp16", 64, 16, 16, False, True),
        ("fp16", 128, 16, 16, True, True),
        ("bf16", 64, 16, 4, True, True),
    )
)
CASE_BY_ID = index_cases(CASES, label="SDPA")


def prepare(case: Case, library_root: str) -> Gfx950AttentionDenseSpec:
    """Select the current kernel through gfx950's production dispatch policy."""
    validate_library_root(library_root)
    from dispatch.attention import attention_tuning_spec

    # Match the existing gfx950 numeric cohort's named production candidates.
    if not case.persistent:
        spec_id = "gfx950_dense_grid"
    elif case.head_dim == 128 and case.causal:
        spec_id = "gfx950_dense_persist_widedma"
    else:
        spec_id = "gfx950_dense_persist"
    return attention_tuning_spec(
        attention_request(case, NAME),
        spec_id,
    ).kernel_spec


def launch(spec: Gfx950AttentionDenseSpec, buffers: dict, scale: float) -> None:
    """Use the same public launch entry as the existing gfx950 numeric tests."""
    from kernels.gfx950 import attention_dense

    dense.launch(attention_dense, spec, buffers, scale)


def exported_kernel(spec: Gfx950AttentionDenseSpec) -> tuple[bytes, dict]:
    """Describe the launcher that actually executed during qualification."""
    from kernels.gfx950 import attention_dense

    return dense.exported_kernel(attention_dense, spec, NAME)
