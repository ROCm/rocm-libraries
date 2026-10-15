# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 MLA candidates.

One candidate today: the chunked bf16 prefill forward kernel. It takes the
query already projected by ``W_UQ`` and RoPE-rotated; neither step is in-tree,
and the pre-kernel that does both is a follow-up. Registration is an
explicit ``register(registry)`` call rather than an import side effect, so the
registry contents stay a readable list and a test can assemble a registry from a
subset of arch modules.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

from kernels.mla.mla_prefill_gfx942 import (
    MLA_MAX_NUM_SEQS,
    MlaPrefillSpec,
    build_mla_prefill_fwd,
    check_mla_prefill_extents,
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

# CUs per XCD on CDNA3 (a 228-CU gfx942 has 6 XCDs, a 304-CU one 8).
_CUS_PER_XCD = 38

# The agpr_acc layout (32 rows per wave, one workgroup per CU) is selected only
# for key ranges at least this long, and only when its workgroups fill every
# CU at least once. Shorter ranges are dominated by the per-workgroup absorb
# and W_UV, which a second workgroup per CU hides; fewer workgroups than CUs
# leave CUs idle that the 16-row layout, with twice the workgroups, uses.
_AGPR_MIN_KEYS = 4096
# Query rows per wave in the agpr_acc layout (16 in the default layout).
_AGPR_ROWS_PER_WAVE = 32
# Test hook: True / False forces the agpr_acc choice, None applies the rule.
_AGPR_FORCE = None


def _resolve_num_cus(req: MLARequest) -> int:
    """Device CU count, or 0 when unknown.

    Taken from ``req.num_cus`` when the caller sets it, else from the live
    device -- but only when the build box is the requested arch, so a
    cross-compile never bakes another part's count into the kernel.
    """
    n = int(req.num_cus)
    if n > 0:
        return n
    try:
        from rocke.runtime.hip_module import get_device_arch, get_device_num_cus

        if (get_device_arch() or "").split(":")[0] == req.arch.split(":")[0]:
            return int(get_device_num_cus() or 0)
    except Exception:
        pass
    return 0


def _num_xcds(num_cus: int) -> int:
    """XCD count for the workgroup reorder, or 0 to keep launch order.

    A CU count that is not a whole number of CDNA3 XCDs (a partitioned device,
    an unknown part) keeps launch order; a single XCD needs no reorder.
    """
    if num_cus <= 0 or num_cus % _CUS_PER_XCD:
        return 0
    xcds = num_cus // _CUS_PER_XCD
    return xcds if xcds > 1 else 0


def _use_agpr_layout(req: MLARequest, num_cus: int) -> bool:
    """Long key ranges with at least one workgroup per CU; see _AGPR_MIN_KEYS."""
    if _AGPR_FORCE is not None:
        return _AGPR_FORCE
    if num_cus <= 0 or req.max_seqlen_k < _AGPR_MIN_KEYS:
        return False
    # The agpr layout's real launch: num_q_blocks_for counts the packed varlen
    # blocks (total_q // block_q + num_seqs), so ragged batches are not
    # under-counted against the CU count.
    q_blocks = num_q_blocks_for(req, _AGPR_ROWS_PER_WAVE * _Q_TILES_PER_WG)
    wgs = q_blocks * (req.num_heads // _HEADS_PER_WG)
    return wgs >= num_cus


def _spec_for(req: MLARequest) -> tuple[MlaPrefillSpec, tuple[bool, str]]:
    """Build the spec dispatch ships for ``req``, with its admission verdict.

    The one-wave-per-head layout with two heads x two query tiles per
    workgroup: each pair of waves shares its head's W_UK / W_UV fetches and
    epilogue staging, and each staged key tile feeds every row of the
    workgroup. (One head x four tiles cuts the weights further but measured a
    large regression on some chunked shapes.) Two variants:

    * ``agpr_acc`` -- 32 rows per wave, the latent accumulator in AGPRs, one
      workgroup per CU, DMA'd double-buffered key tiles -- for long key ranges
      that fill the device (:func:`_use_agpr_layout`); its admission refuses
      the llvm20 flavor;
    * otherwise 16 rows per wave at two workgroups per CU.

    The workgroup order is XCD-aware when the device's XCD count is known. If
    the head count is odd, the remaining levers stay at their
    :class:`MlaPrefillSpec` defaults, the single-head layout. Nothing else is
    tuned per request.

    Admission runs once per candidate spec here, and the verdict of the spec
    returned is handed back so :func:`support` does not run it again.
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
    if req.num_heads % _HEADS_PER_WG:
        return base, supports_mla_prefill(base, arch=req.arch)
    num_cus = _resolve_num_cus(req)

    def packed(agpr: bool) -> MlaPrefillSpec:
        return dataclasses.replace(
            base,
            heads_per_wg=_HEADS_PER_WG,
            q_tiles_per_wg=_Q_TILES_PER_WG,
            num_warps=_HEADS_PER_WG * _Q_TILES_PER_WG,
            block_q=(_AGPR_ROWS_PER_WAVE if agpr else 16) * _Q_TILES_PER_WG,
            agpr_acc=agpr,
            num_xcds=_num_xcds(num_cus),
        )

    if _use_agpr_layout(req, num_cus):
        spec = packed(True)
        verdict = supports_mla_prefill(spec, arch=req.arch)
        # llvm20 refuses agpr_acc; fall back to 16 rows per wave.
        if verdict[0]:
            return spec, verdict
    spec = packed(False)
    return spec, supports_mla_prefill(spec, arch=req.arch)


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
        if int(req.num_kv_blocks) == 0:
            # Refused rather than skipped: without the pool size the paged
            # c_kv extent admission cannot run, and an over-2 GiB pool would
            # wrap the kernel's 32-bit page offsets silently.
            return False, (
                "num_kv_blocks is required (pass c_kv.shape[0], the paged KV "
                "pool size); 0 leaves the paged c_kv extent unchecked"
            )
        try:
            spec, (ok, why) = _spec_for(req)
        except ValueError as exc:
            # ``MlaPrefillSpec.__post_init__`` rejects specs impossible on any
            # arch. Report it rather than propagate: a dispatcher asking "can
            # you serve this?" wants a no, not an exception.
            return False, f"spec rejected: {exc}"
        if not ok:
            return False, why
        # Per request, not per spec: the extents are runtime arguments.
        ok, why = check_mla_prefill_extents(
            spec, total_q=req.total_q, num_kv_blocks=req.num_kv_blocks
        )
        return (True, "ok") if ok else (False, why)

    def select(req: OperatorRequest) -> MlaPrefillSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        assert isinstance(req, MLARequest)
        return _spec_for(req)[0]

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
                # The in-kernel seq_idx binary search runs a fixed
                # MLA_BINARY_SEARCH_ITERS iterations, which resolve at most
                # 2**iters - 1 sequences.
                ShapeRange("num_seqs", min=1, max=MLA_MAX_NUM_SEQS),
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
