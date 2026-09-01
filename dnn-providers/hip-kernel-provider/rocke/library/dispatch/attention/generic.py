# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Multi-arch attention candidates: the two unified paths and D256 decode.

This module holds the candidates whose ``Capability`` names more than one arch.
That is a different claim from "portable": each one still lists its arches
explicitly, because there is no family wildcard (section 8.1, step 7). A new
target inherits nothing here until someone opts it in.
"""

from __future__ import annotations

from typing import Tuple

from kernels.common.attention_unified import supports_native_unified_attention
from rocke.core.arch import known_arches
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
    UNIFIED_DTYPES,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
    AttentionSpec,
    FAMILY,
    _problem,
    _request_errors,
    _selector_matches,
)

# The unified paths are portable, so they declare every known target rather than
# a wave-size cohort. This looks like an over-broad claim and is not: these
# candidates select a *path*, and the concrete kernel behind it is chosen
# downstream by ``attention_unified`` on the running device -- the wave64 MFMA
# variants on gfx942/gfx950, the wave32 WMMA variant on gfx1250, and the
# arch-neutral scalar rocKE kernel (no MMA atom at all) everywhere else. They
# are therefore the second documented exception to wave-size consistency,
# alongside norm2d. Narrowing them to the wave64 MFMA targets would drop the
# gfx1250 and RDNA coverage that the scalar and WMMA backends actually provide.
_UNIFIED_CAPABILITY = Capability(
    arches=known_arches(),
    dtypes=UNIFIED_DTYPES,
    shapes=(
        ShapeRange("hdim_q", allowed=UNIFIED_HEAD_SIZES),
        ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
    ),
    supports_features=ATTENTION_FEATURES,
)


def _make_candidate(*, path: str, priority: int) -> KernelCandidate:
    spec_id = f"unified_{path}"
    name = f"attention_unified_{path}"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem, arch=req.arch)
        if not ok:
            return False, why
        if problem.select_path() != path:
            return False, (
                f"problem routes to {problem.select_path()!r} path, not {path!r}"
            )
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path=path,
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            use_fp8=problem.use_fp8,
            fp8_fnuz=problem.fp8_fnuz,
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm=f"unified_{path}",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=priority,
        capability=_UNIFIED_CAPABILITY,
        _supports=support,
        select_spec=select,
        signature=lambda _spec: (),
        grid=lambda spec, req: (0, 0, 0),  # geometry deferred (see common.py)
        block=lambda spec: (0, 0, 0),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def _make_d256_decode_candidate() -> KernelCandidate:
    """D256 bf16 decode candidate for gfx950 and gfx942 — 3D split-KV path.

    Registered at priority 5 so it outranks the generic unified_3d candidate
    (priority 10) for eligible D256 bf16 decode shapes. Callers can also force
    this path explicitly via algorithm="d256_decode".

    Lives here rather than under an arch module because it serves two arches
    from one definition; splitting it per arch would duplicate the cohort gate.
    """
    from kernels.common.attention_unified import _d256_decode_cohort

    spec_id = "d256_decode"
    name = "attention_d256_decode"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        problem = _problem(req)
        ok, why = supports_native_unified_attention(problem, arch=req.arch)
        if not ok:
            return False, why
        if not _d256_decode_cohort(problem):
            return False, "problem is not in the D256 bf16 decode cohort"
        if problem.select_path() != "3d":
            return False, (
                f"problem routes to {problem.select_path()!r} path, not '3d'"
            )
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        return AttentionSpec(
            path="3d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_d256_decode",
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="d256_decode",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx942", "gfx950"),
            dtypes=("bf16",),
            shapes=(
                ShapeRange("hdim_q", allowed=(256,)),
                ShapeRange("kv_block_size", allowed=UNIFIED_BLOCK_SIZES),
            ),
            supports_features=frozenset({"causal"}),
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
    registry.extend(
        (
            # 2d and 3d are mutually exclusive per problem (select_path returns
            # one), so priority only orders the two when both could match --
            # which they cannot. Equal priority keeps the registry order stable.
            _make_candidate(path="2d", priority=10),
            _make_candidate(path="3d", priority=10),
        )
    )
    registry.register(_make_d256_decode_candidate())
