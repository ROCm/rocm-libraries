# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 MLA candidates.

One kernel today: the chunked bf16 prefill forward kernel. It takes the
query already projected by ``W_UQ`` and RoPE-rotated; neither step is in-tree,
and the pre-kernel that does both is a follow-up. It is registered three
times: the production candidate ships the layout the agpr_acc rule picks, and
two opt-in candidates each pin one fixed two-heads layout (``agpr_acc`` and
``row16``) so sweeps and tuning records can name a layout that does not move
when the rule or the LLVM flavor changes. Registration is an
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

# agpr_acc (32 rows per wave, one workgroup per CU) is selected iff the batch
# averages at least _AGPR_MIN_AVG_ROWS query rows per sequence
# (total_q / num_seqs) *and* its row-carrying workgroups fill at least
# _AGPR_MIN_ROW_FILL of the CUs. The rule was chosen from gated sweeps of both
# layouts on 304-CU (the primary target) and 228-CU gfx942 as the only rule
# tested that never selected the slower layout on the 304-CU part; see the PR
# description. Key ranges were measured down to 512 on the 304-CU part and down
# to 2048 on the 228-CU part with no dependence on key length, so the rule
# ignores it; shorter key ranges are unmeasured.
#
# The fill term counts a lower bound on agpr_acc workgroups that carry rows,
# max(ceil(total_q / block_q), num_seqs) per head pair, rather than the launched
# grid, whose per-sequence padding workgroups exit at once. On the 304-CU part,
# at H 16, 32 and 64, grids with 128 such workgroups lost with agpr_acc and grids
# with 160 won, so half the CUs splits them. It only binds at low head counts,
# which are unmeasured on the 228-CU part. An unknown CU count fails it, so a
# cross-compile without num_cus keeps the default layout.
#
# It is conservative: at H=128 the agpr_acc layout also wins with 128 to 319
# rows per sequence, and the rule leaves those wins unused. The reason is
# padding-workgroup placement. The agpr_acc grid launches one padding workgroup
# per sequence per head group, and where those land across the shader engines
# can strand CUs and push a grid that fits into a second round. That makes
# ragged batches layout-sensitive in a way no request field shows dispatch.
# Dispatching the padding workgroups last is a tracked follow-up; once it lands,
# re-sweep this threshold.
_AGPR_MIN_AVG_ROWS = 320
# (numerator, denominator): row-carrying agpr_acc workgroups >= this x num_cus.
_AGPR_MIN_ROW_FILL = (1, 2)
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


def _use_agpr_layout(
    req: MLARequest,
    num_cus: int | None = None,
    *,
    min_avg_rows: int = _AGPR_MIN_AVG_ROWS,
    min_row_fill: tuple[int, int] = _AGPR_MIN_ROW_FILL,
) -> bool:
    """Enough rows per sequence and enough row-carrying workgroups.

    See _AGPR_MIN_AVG_ROWS. ``num_cus`` defaults to :func:`_resolve_num_cus`;
    unknown (<= 0) fails the fill term.
    """
    if _AGPR_FORCE is not None:
        return _AGPR_FORCE
    total_q, num_seqs = int(req.total_q), int(req.num_seqs)
    # total_q / num_seqs >= min_avg_rows, without the division.
    if total_q < int(min_avg_rows) * num_seqs:
        return False
    if num_cus is None:
        num_cus = _resolve_num_cus(req)
    if num_cus <= 0:
        return False
    block_q = _AGPR_ROWS_PER_WAVE * _Q_TILES_PER_WG
    rows_wgs = max(-(-total_q // block_q), num_seqs) * (
        int(req.num_heads) // _HEADS_PER_WG
    )
    fill_num, fill_den = min_row_fill
    return rows_wgs * int(fill_den) >= int(num_cus) * int(fill_num)


# Fixed two-heads layouts the opt-in candidates pin; None is the rule's pick.
_LAYOUTS = ("agpr_acc", "row16")
# Opt-in candidate name -> the layout it pins.
_LAYOUT_OF = {f"mla_prefill_fwd_gfx942_{layout}": layout for layout in _LAYOUTS}


def _spec_for(
    req: MLARequest, *, layout: str | None = None
) -> tuple[MlaPrefillSpec, tuple[bool, str]]:
    """Build the spec dispatch ships for ``req``, with its admission verdict.

    The one-wave-per-head layout with two heads x two query tiles per
    workgroup: each pair of waves shares its head's W_UK / W_UV fetches and
    epilogue staging, and each staged key tile feeds every row of the
    workgroup. (One head x four tiles cuts the weights further but measured a
    large regression on some chunked shapes.) Two variants:

    * ``agpr_acc`` -- 32 rows per wave, the latent accumulator in AGPRs, one
      workgroup per CU, DMA'd double-buffered key tiles -- for batches with
      enough query rows per sequence (:func:`_use_agpr_layout`); its admission
      refuses the llvm20 flavor;
    * otherwise 16 rows per wave at two workgroups per CU.

    The workgroup order is XCD-aware when the device's XCD count is known. If
    the head count is odd, the remaining levers stay at their
    :class:`MlaPrefillSpec` defaults, the single-head layout. Every layout
    takes ``absorbed_query`` from the request, and the single-head layout's
    admission refuses it. Nothing else is tuned per request.

    Admission runs once per candidate spec here, and the verdict of the spec
    returned is handed back so :func:`support` does not run it again.

    ``layout`` pins one two-heads layout instead of applying the rule:
    ``"agpr_acc"`` (refused at llvm20, with no fallback) or ``"row16"`` (16
    rows per wave). Odd head counts have only the single-head layout, so a
    pinned layout is refused there.
    """
    if layout is not None and layout not in _LAYOUTS:
        raise ValueError(f"unknown MLA layout {layout!r}; expected one of {_LAYOUTS}")
    base = MlaPrefillSpec(
        num_heads=req.num_heads,
        d_nope=req.d_nope,
        d_rope=req.d_rope,
        d_v=req.d_v,
        r_kv=req.kv_lora_rank,
        page_block_size=req.page_block_size,
        block_k=req.page_block_size,
        dtype=req.dtype.lower(),
        # Carried into every layout below (they derive from ``base``); the
        # single-head layout does not implement it, so its admission refuses.
        absorbed_query=req.absorbed_query,
    )
    if req.num_heads % _HEADS_PER_WG:
        if layout is not None:
            why = (
                f"odd num_heads={req.num_heads} runs the single-head layout; "
                f"the {layout} two-heads layout needs an even head count"
            )
            return base, (False, why)
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

    def admitted(agpr: bool) -> tuple[MlaPrefillSpec, tuple[bool, str]]:
        spec = packed(agpr)
        return spec, supports_mla_prefill(spec, arch=req.arch)

    if layout is not None:
        return admitted(layout == "agpr_acc")
    if _use_agpr_layout(req, num_cus):
        spec, verdict = admitted(True)
        # llvm20 refuses agpr_acc; fall back to 16 rows per wave.
        if verdict[0]:
            return spec, verdict
    return admitted(False)


def _make_gfx942_mla_prefill_candidate(*, layout: str | None = None) -> KernelCandidate:
    """The chunked paged-KV MLA prefill forward kernel.

    Capability declares the geometry the kernel was built for; the residual
    predicate is :func:`supports_mla_prefill`, which owns the MFMA-atom and
    staging-divisibility admission the capability cannot state as data.

    ``layout`` makes an opt-in candidate pinned to that two-heads layout
    (see :func:`_spec_for`): a pin always gets that layout or a refusal, and
    ``algorithm="auto"`` never selects it. The family sweep entry points skip
    it when the production candidate already ships that layout for the request
    (:func:`sweep_skips`), so the sweep holds each layout once.
    """
    suffix = f"_{layout}" if layout else ""
    name = "mla_prefill_fwd_gfx942" + suffix
    spec_id = "gfx942_mla_prefill_fwd" + suffix

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
            spec, (ok, why) = _spec_for(req, layout=layout)
        except (TypeError, ValueError) as exc:
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
        return _spec_for(req, layout=layout)[0]

    def grid(spec: MlaPrefillSpec, req: OperatorRequest) -> Tuple[int, int, int]:
        assert isinstance(req, MLARequest)
        return mla_prefill_fwd_grid(
            spec, num_q_blocks=num_q_blocks_for(req, spec.block_q)
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="mla_prefill_chunked" + suffix,
        spec_id=spec_id,
        abi_version=MLA_ABI_VERSION,
        priority=20 if layout else 10,
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
            # ``causal`` is not merely *supported* -- required. The kernel
            # applies the bottom-right causal mask unconditionally, so a
            # non-causal request must be refused rather than served a masked
            # result it did not ask for. ``absorbed_query`` is optional.
            requires_features=frozenset({"causal"}),
        ),
        _supports=support,
        select_spec=select,
        signature=lambda spec: tuple(mla_prefill_fwd_signature(spec)),
        grid=grid,
        block=mla_prefill_block,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        build=lambda spec, arch: build_mla_prefill_fwd(spec, arch=arch),
        opt_in=layout is not None,
    )
    return candidate


def sweep_skips(req: OperatorRequest, candidate: KernelCandidate) -> bool:
    """True when an unpinned sweep should skip ``candidate`` for ``req``.

    An opt-in fixed-layout candidate whose layout the production candidate
    already ships would repeat that combo. This cannot live in the opt-in's
    ``sweep_space``: the registry falls back to ``select_spec`` when a sweep
    space is empty, so the combo would come back. A request that pins a
    candidate is never filtered.
    """
    layout = _LAYOUT_OF.get(candidate.name)
    if layout is None or not isinstance(req, MLARequest) or _request_errors(req):
        return False
    if any(str(v).strip().lower() != "auto" for v in (req.algorithm, req.spec_id)):
        return False
    try:
        shipped, (ok, _why) = _spec_for(req)
    except ValueError:
        return False
    return (
        ok
        and shipped.heads_per_wg == _HEADS_PER_WG
        and (shipped.agpr_acc == (layout == "agpr_acc"))
    )


def register(registry: CandidateRegistry) -> None:
    registry.register(_make_gfx942_mla_prefill_candidate())
    for layout in _LAYOUTS:
        registry.register(_make_gfx942_mla_prefill_candidate(layout=layout))


__all__ = ["register", "sweep_skips"]
