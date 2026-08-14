# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 attention candidates (CDNA4, wave64, 32x32 MFMA + dense persistent)."""

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
    UNIFIED_BLOCK_SIZES,
    AttentionRequest,
    AttentionSpec,
    FAMILY,
    _problem,
    _request_errors,
    _selector_matches,
)

# block_n (KV tile) the dense candidate ships; 64 is the resource-efficient
# peak (see AttentionDenseSpec.block_n).
_DENSE_BLOCK_N = 64


def _dense_spec(req: OperatorRequest):
    """Build the ``AttentionDenseSpec`` for a request at its best-performing
    config. Persistent ("auto") turns on the grid-stride ~970-TFLOPS variant once
    there is enough work to fill the persistent grid (``nqb*Hq*B >= num_persistent``
    -- the large-Sq prefill regime); ``persist_decode`` / ``lazy_rescale`` default
    to the L2-locality hkv-major decode and always-on lazy rescale. Non-tile-
    multiple self-attention lengths use the on-chip ragged path (no host pad)."""
    from kernels.gfx950.attention_dense import AttentionDenseSpec, _BLOCK_M

    assert isinstance(req, AttentionRequest)
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    bn = _DENSE_BLOCK_N
    # on-chip ragged padding for ragged self-attention lengths (seqlen_q==seqlen_kv,
    # not a 256/block_n multiple). Cross-attention ragged is left to the validator.
    ragged = (sq == sk) and ((sq % _BLOCK_M != 0) or (sk % bn != 0))
    nqb = (sq + _BLOCK_M - 1) // _BLOCK_M
    work = nqb * int(req.nhead_q) * int(req.batch)
    np = int(req.dense_num_persistent)
    mode = req.dense_persistent.strip().lower()
    if mode == "on":
        persistent = True
    elif mode == "off":
        persistent = False
    elif mode == "auto":
        persistent = work >= np  # enough work to fill the persistent grid
    else:
        raise ValueError(
            f"dense_persistent must be 'auto'/'on'/'off', got {req.dense_persistent!r}"
        )
    return AttentionDenseSpec(
        batch=int(req.batch),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=int(req.hdim_q),
        causal=(int(req.mask_type) != 0),
        dtype=req.dtype.lower(),
        block_n=bn,
        persistent=persistent,
        num_persistent=np,
        persist_decode=req.dense_persist_decode.strip().lower(),
        ragged=ragged,
    )


def dense_spec_for_request(req: AttentionRequest):
    """Public builder: the launch-ready ``AttentionDenseSpec`` for ``req`` at its
    best config (see :func:`_dense_spec`). Pair with ``run_attention_dense_torch``
    to execute the dispatched dense candidate."""
    return _dense_spec(req)


def _make_gfx950_attention_dense_candidate() -> KernelCandidate:
    """Dense CK-1 persistent flash-attn prefill on gfx950 (bf16/fp16, causal/full).

    OPT-IN ONLY: matches solely when the request explicitly names
    ``algorithm="attention_dense"`` (or ``spec_id="gfx950_attention_dense"``), so it
    never auto-overrides the generic unified_2d path (no change to default routing).
    When selected it uses the persistent best-config from :func:`_dense_spec`
    (grid-stride + hkv-major + lazy for large Sq); the concrete kernel_name (incl.
    ``persist``/``hkvmaj``/``lazyrs``/``ragged``) is surfaced on the spec so the
    launched kernel is the fast path, not the default grid. End-to-end launch is
    ``run_attention_dense_torch(spec=dense_spec_for_request(req), ...)``.
    """
    spec_id = "gfx950_attention_dense"
    name = "attention_gfx950_dense"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        # Opt-in: never selected under algorithm/spec_id "auto".
        if req.algorithm.strip().lower() != "attention_dense" and (
            req.spec_id.strip().lower() != spec_id
        ):
            return False, "attention_dense is opt-in (algorithm='attention_dense')"
        from kernels.gfx950.attention_dense import supports_attention_dense

        try:
            spec = _dense_spec(req)
        except ValueError as e:
            return False, str(e)
        ok, why = supports_attention_dense(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        dense_spec = _dense_spec(req)
        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_dense",
            # surface the concrete persistent/hkvmaj/lazyrs/ragged kernel so the
            # dispatched spec names the fast path it will actually launch.
            kernel_name_override=dense_spec.kernel_name(),
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="attention_dense",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=3,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("bf16", "fp16"),
            # Dense: no sliding-window, no sinks. Causal is a mask, not a
            # feature this path turns down.
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


def _make_gfx950_d256_candidate() -> KernelCandidate:
    """Fast gfx950 bf16 head_size-256 prefill kernel — 32x32 transposed stack
    with FA3-style softmax<->MFMA interleave (mode2/g4) + slab-padded K_lds.

    Registered at priority 5 so it outranks the generic unified_2d candidate
    (priority 10) for the gfx950 bf16 D256 prefill cohort. The registry sorts
    ascending (lower = higher precedence); gfx950-only, so it never competes
    with the gfx942 dense_pipe candidate. Callers can also force this path
    explicitly via algorithm="d256_gfx950".

    The cohort is the single source of truth
    ``kernels.common.attention_unified._d256_gfx950_cohort`` — the same predicate
    the orchestrator's ``_d256_gfx950_fast`` override uses — so dispatch selection
    and the built spec cannot drift. Only the arch gate differs (request arch
    here vs resolved device arch there).
    """
    spec_id = "gfx950_d256"
    name = "attention_gfx950_d256"

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
        from kernels.common.attention_unified import _d256_gfx950_cohort

        if not _d256_gfx950_cohort(problem):
            return False, "not the gfx950 bf16 D256 prefill fast-path cohort"
        return True, "ok"

    def select(req: OperatorRequest) -> AttentionSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, AttentionRequest)
        problem = _problem(req)
        from kernels.common.attention_unified import _d256_gfx950_spec_overrides

        return AttentionSpec(
            path="2d",
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            name="rocke_attention_gfx950_d256",
            tiled_overrides=tuple(sorted(_d256_gfx950_spec_overrides().items())),
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="d256_gfx950",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=5,
        capability=Capability(
            arches=("gfx950",),
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
    registry.register(_make_gfx950_attention_dense_candidate())
    registry.register(_make_gfx950_d256_candidate())
