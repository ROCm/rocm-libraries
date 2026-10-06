# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 MLA candidates.

One candidate today: the chunked bf16 prefill forward kernel. Registration is an
explicit ``register(registry)`` call rather than an import side effect, so the
registry contents stay a readable list and a test can assemble a registry from a
subset of arch modules.
"""

from __future__ import annotations

import dataclasses
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


# Shipped one-wave-per-head layout: a workgroup holds _HEADS_PER_WG heads x
# _Q_TILES_PER_WG tiles of 16 queries, one wave per (head, tile).
_HEADS_PER_WG = 2
_Q_TILES_PER_WG = 2

# CUs per XCD on CDNA3 (MI300A has 6 XCDs, MI300X 8).
_CUS_PER_XCD = 38


def _resolve_num_xcds(req: MLARequest) -> int:
    """XCD count for the workgroup reorder, or 0 to keep launch order.

    Taken from ``req.num_cus`` when the caller sets it, else from the live
    device -- but only when the build box is the requested arch, so a
    cross-compile never bakes another part's count into the kernel. A CU count
    that is not a whole number of CDNA3 XCDs (a partitioned device, an
    unknown part) keeps launch order; a single XCD needs no reorder.
    """
    n = int(req.num_cus)
    if n <= 0:
        try:
            from rocke.runtime.hip_module import get_device_arch, get_device_num_cus

            if (get_device_arch() or "").split(":")[0] == req.arch.split(":")[0]:
                n = int(get_device_num_cus() or 0)
        except Exception:
            n = 0
    if n <= 0 or n % _CUS_PER_XCD:
        return 0
    xcds = n // _CUS_PER_XCD
    return xcds if xcds > 1 else 0


def _spec_for(req: MLARequest) -> MlaPrefillSpec:
    """Build the spec dispatch ships for ``req``.

    The one-wave-per-head layout with two heads x two 16-query tiles per
    workgroup: each pair of waves shares its head's W_UK / W_UV fetches and
    epilogue staging, halving the per-row weight traffic of four heads x one
    tile, and each staged key tile still feeds 64 query rows. (One head x four
    tiles cuts the weights further but measured a large regression on some
    chunked shapes.) The
    workgroup order is made XCD-aware when the device's XCD count is known
    (:func:`_resolve_num_xcds`). If the head count does not divide or
    admission refuses the layout, the remaining levers stay at their
    :class:`MlaPrefillSpec` defaults, the single-head layout. Nothing else is
    tuned per request.
    """
    base = MlaPrefillSpec(
        num_heads=req.num_heads,
        d_nope=req.d_nope,
        d_rope=req.d_rope,
        d_v=req.d_v,
        r_kv=req.kv_lora_rank,
        page_block_size=req.page_block_size,
        block_k=req.page_block_size,
        dtype=req.dtype.lower(),
    )
    if req.num_heads % _HEADS_PER_WG == 0:
        packed = dataclasses.replace(
            base,
            heads_per_wg=_HEADS_PER_WG,
            q_tiles_per_wg=_Q_TILES_PER_WG,
            num_warps=_HEADS_PER_WG * _Q_TILES_PER_WG,
            block_q=16 * _Q_TILES_PER_WG,
            num_xcds=_resolve_num_xcds(req),
        )
        if supports_mla_prefill(packed, arch=req.arch)[0]:
            return packed
    return base


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
