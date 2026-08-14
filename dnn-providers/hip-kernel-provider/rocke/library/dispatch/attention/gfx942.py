# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 attention candidates (CDNA3, wave64, narrow 16x16x16 MFMA)."""

from __future__ import annotations

from typing import Tuple

from kernels.common.attention_unified import supports_native_unified_attention
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    AttentionSpec,
    FAMILY,
    _problem,
    _request_errors,
    _selector_matches,
)


def _make_gfx942_dense_pipe_candidate() -> KernelCandidate:
    """Fast gfx942 fp16 prefill kernel — transposed-x8 flash with ring-sliced K.

    Registered at priority 5 so it outranks the generic unified_2d candidate
    (priority 10) whenever both would match the same gfx942 fp16 2D problem.
    The registry sorts ascending (lower = higher precedence).
    Callers can also force this path explicitly via algorithm="dense_pipe".
    """
    spec_id = "gfx942_dense_pipe"
    name = "attention_gfx942_dense_pipe"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem)
        if not ok:
            return False, why
        if problem.select_path() != "2d":
            return False, "problem routes to 3D, not 2D"
        from kernels.common.attention_unified import _enable_gfx942_fp16_flash

        if not _enable_gfx942_fp16_flash(problem):
            return False, "gfx942 fp16 flash not eligible for this shape"
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_gfx942_dense_pipe",
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="dense_pipe",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("fp16",),
            shapes=(
                ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            # ``_enable_gfx942_fp16_flash`` is the real narrowing; nothing here
            # claims a feature it turns down, so the full set stays declared.
            supports_features=ATTENTION_FEATURES,
        ),
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def register(registry: CandidateRegistry) -> None:
    registry.register(_make_gfx942_dense_pipe_candidate())
