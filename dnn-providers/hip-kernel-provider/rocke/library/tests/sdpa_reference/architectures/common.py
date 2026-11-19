# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Source isolation and request construction shared by SDPA reference adapters.

Production imports stay lazy so workers select their source tree before loading
kernel or dispatch modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..contract import Case

if TYPE_CHECKING:
    from dispatch.attention import AttentionRequest


def validate_library_root(library_root: str) -> None:
    """Reject production kernels imported from outside the worker's source tree."""
    import kernels

    if (
        not Path(kernels.__file__)
        .resolve()
        .is_relative_to(Path(library_root).resolve())
    ):
        raise RuntimeError("worker imported kernels outside the selected library")


def attention_request(case: Case, architecture: str) -> AttentionRequest:
    """Translate the reference tensor and mask contract into a production request."""
    from dispatch.attention import AttentionRequest

    return AttentionRequest(
        batch=case.batch,
        nhead_q=case.query_heads,
        nhead_k=case.kv_heads,
        seqlen_q=case.sequence_length,
        seqlen_k=case.sequence_length,
        hdim_q=case.head_dim,
        hdim_v=case.head_dim,
        arch=architecture,
        mask_type=1 if case.causal else 0,
        dtype=case.dtype,
    )
