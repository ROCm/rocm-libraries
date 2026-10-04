# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Qualified gfx942 SDPA cohort and current-source dispatch adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...contract import Case

if TYPE_CHECKING:
    from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec

NAME = "gfx942"
FAMILIES = ("gfx942", "gfx94x")

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
CASE_BY_ID = {case.id: case for case in CASES}


def prepare(case: Case, library_root: str) -> Gfx942AttentionDenseSpec:
    """Select the current kernel through gfx942's production dispatch policy."""
    import kernels
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import _dense_spec
    from kernels.gfx942.attention_dense import _as_gfx942_spec

    if (
        not Path(kernels.__file__)
        .resolve()
        .is_relative_to(Path(library_root).resolve())
    ):
        raise RuntimeError("worker imported kernels outside the selected library")
    spec = _as_gfx942_spec(
        _dense_spec(
            AttentionRequest(
                batch=case.batch,
                nhead_q=case.query_heads,
                nhead_k=case.kv_heads,
                seqlen_q=case.sequence_length,
                seqlen_k=case.sequence_length,
                hdim_q=case.head_dim,
                hdim_v=case.head_dim,
                arch="gfx942",
                mask_type=1 if case.causal else 0,
                dtype=case.dtype,
                algorithm="attention_dense",
                dense_persistent="on" if case.persistent else "off",
            )
        )
    )
    return spec


def launch(spec: Gfx942AttentionDenseSpec, buffers: dict, scale: float) -> None:
    """Use the same public launch entry as the existing gfx942 numeric tests."""
    from kernels.gfx942.attention_dense import run_attention_dense_torch

    run_attention_dense_torch(
        spec=spec,
        q=buffers["q"],
        k=buffers["k"],
        v=buffers["v"],
        out=buffers["out"],
        scale=scale,
    )


def exported_kernel(spec: Gfx942AttentionDenseSpec) -> tuple[bytes, dict]:
    """Describe the launcher that actually executed during qualification."""
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from kernels.gfx942.attention_dense import (
        _DENSE_LAUNCHER_CACHE,
        attention_dense_block,
        attention_dense_grid,
    )

    launcher = _DENSE_LAUNCHER_CACHE[attention_dense_cache_key(spec, arch=NAME)]
    return launcher._hsaco, {
        "name": launcher.kernel_name,
        "signature": launcher.signature,
        "grid": attention_dense_grid(spec),
        "block": attention_dense_block(spec),
        "runtime_shape": spec.runtime_shape,
    }
