# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 MLA candidates.

One candidate today: the chunked bf16 prefill forward kernel. Registration is an
explicit ``register(registry)`` call rather than an import side effect, so the
registry contents stay a readable list and a test can assemble a registry from a
subset of arch modules.
"""

from __future__ import annotations

from typing import Tuple

from kernels.mla.mla_prefill_gfx942 import (
    MlaPrefillSpec,
    build_mla_prefill_fwd,
    mla_prefill_block,
    mla_prefill_fwd_grid,
    mla_prefill_fwd_signature,
    supports_mla_prefill,
)
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    FAMILY,
    MLA_ABI_VERSION,
    MLA_FEATURES,
    MLARequest,
    _request_errors,
    _selector_matches,
    num_q_blocks_for,
)


def _spec_for(req: MLARequest) -> MlaPrefillSpec:
    """Build the bring-up spec for ``req``.

    Every codegen lever (``block_q``, ``block_k``, ``r_kv_tile``, ``num_warps``)
    is left at its :class:`MlaPrefillSpec` default. Choosing them per request is
    a tuning decision that needs a resource model, and there is none yet; a
    dispatcher that silently picked one would make the shipped configuration
    depend on the request shape before anything had measured that.
    """
    return MlaPrefillSpec(
        num_heads=req.num_heads,
        d_nope=req.d_nope,
        d_rope=req.d_rope,
        d_v=req.d_v,
        r_kv=req.kv_lora_rank,
        page_block_size=req.page_block_size,
        block_k=req.page_block_size,
        dtype=req.dtype.lower(),
    )


def _make_gfx942_mla_prefill_candidate() -> KernelCandidate:
    """The chunked paged-KV MLA prefill forward kernel.

    Capability declares the geometry the kernel was built for; the residual
    predicate is :func:`supports_mla_prefill`, which owns the MFMA-atom and
    staging-divisibility admission the capability cannot state as data.
    """
    name = "mla_prefill_fwd_gfx942"
    spec_id = "gfx942_mla_prefill_fwd"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, MLARequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        try:
            spec = _spec_for(req)
        except ValueError as exc:
            # ``MlaPrefillSpec.__post_init__`` rejects specs impossible on any
            # arch. Report it rather than propagate: a dispatcher asking "can
            # you serve this?" wants a no, not an exception.
            return False, f"spec rejected: {exc}"
        ok, why = supports_mla_prefill(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> MlaPrefillSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MLARequest)
        return _spec_for(req)

    def grid(spec: MlaPrefillSpec, req: OperatorRequest) -> Tuple[int, int, int]:
        assert isinstance(req, MLARequest)
        return mla_prefill_fwd_grid(
            spec, num_q_blocks=num_q_blocks_for(req, spec.block_q)
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="mla_prefill_chunked",
        spec_id=spec_id,
        abi_version=MLA_ABI_VERSION,
        priority=10,
        capability=Capability(
            arches=("gfx942",),
            dtypes=("bf16",),
            shapes=(
                ShapeRange("num_heads", min=1),
                ShapeRange("d_nope", allowed=(128,)),
                ShapeRange("d_rope", allowed=(64,)),
                ShapeRange("d_v", allowed=(128,)),
                ShapeRange("hdim_qk", allowed=(192,)),
                ShapeRange("kv_lora_rank", allowed=(512,)),
                ShapeRange("page_block_size", allowed=(16,)),
                ShapeRange("total_q", min=1),
                ShapeRange("num_seqs", min=1),
            ),
            supports_features=MLA_FEATURES,
            # Not merely *supported* -- required. The kernel applies the
            # bottom-right causal mask unconditionally, so a non-causal request
            # must be refused rather than served a masked result it did not ask
            # for.
            requires_features=MLA_FEATURES,
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: tuple(mla_prefill_fwd_signature(spec)),
        grid=grid,
        block=mla_prefill_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, arch: build_mla_prefill_fwd(spec, arch=arch),
    )
    return candidate


def register(registry: CandidateRegistry) -> None:
    registry.register(_make_gfx942_mla_prefill_candidate())


__all__ = ["register"]
