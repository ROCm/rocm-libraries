# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1151 attention candidates (wave32 RDNA3.5, 16x16x16 WMMA).

``kernels/gfx1151/wmma_fmha_swapqk.py`` is the production dense-attention
forward kernel for Strix Halo, but it was reachable only through the standalone
verify scripts in ``builders/gfx1151/attention``; dispatch had no gfx1151 entry
at all. This module is the registration that makes it dispatchable.

Like the gfx1250 WMMA candidate it declares real launch geometry rather than
deferring to the device: the grid is ``(seqlen_q // q_rows_per_cta,
num_query_heads // gqa_fuse, batch)`` and the argument list is fixed.

**V layout is a caller contract.** ``SwapQKCfg.v_transposed`` defaults to True,
which makes V a ``[B, Hk, D, Sk]`` tensor rather than ``[B, Sk, Hk, D]``. The
dispatcher does not relayout tensors -- ``select_spec`` hands back the kernel's
own ``SwapQKCfg`` and the caller feeds it. Callers relay with
``swapqk_transpose_v``; :func:`bindings.bind_swapqk_torch` shape-checks V so a
row-major tensor fails loudly instead of returning plausible garbage.
"""

from __future__ import annotations

from typing import Tuple

from kernels.gfx1151.wmma_fmha_swapqk import (
    SwapQKCfg,
    build_wmma_fmha_swapqk,
    is_valid_spec as _swapqk_is_valid,
    swapqk_grid,
)
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DimRelation,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)

from .common import (
    AttentionRequest,
    FAMILY,
    _request_errors,
    _selector_matches,
)

ATTENTION_GFX1151_ABI = "rocke-attention-gfx1151/v1"

# The kernel is fp16-only: `dtype_ir` is hardcoded to F16 and SwapQKCfg carries
# no dtype field, so there is nothing to read this from.
DTYPES = ("fp16",)

# Take the tile geometry FROM the config rather than transcribing it, so a
# default that moves takes the declared coverage with it. head_size and
# num_query_heads are required fields; every other default is the swept winner.
_DEFAULT_CFG = SwapQKCfg(head_size=128, num_query_heads=1)

# dual_gather pairs adjacent d-subtiles, so head_size must be even in units of
# the 16-wide WMMA contraction.
_HDIM_MULTIPLE = 32 if _DEFAULT_CFG.dual_gather else 16
# swapqk_grid refuses a seqlen_q remainder rather than launching a partial tile.
_Q_ROWS_PER_CTA = _DEFAULT_CFG.q_rows_per_cta
# The kv loop bound is `seqlen_k / block_n` by plain integer division, so a
# remainder is silently dropped -- a correctness constraint, not a tiling one.
_BLOCK_N = _DEFAULT_CFG.block_n

# Declared coverage, restating the is_valid_spec and grid-helper gates as DATA.
_SWAPQK_CAP = Capability(
    arches=("gfx1151",),
    dtypes=DTYPES,
    shapes=(
        ShapeRange("hdim_q", min=_HDIM_MULTIPLE, multiple_of=_HDIM_MULTIPLE),
        ShapeRange("seqlen_q", min=_Q_ROWS_PER_CTA, multiple_of=_Q_ROWS_PER_CTA),
        ShapeRange("seqlen_k", min=_BLOCK_N, multiple_of=_BLOCK_N),
    ),
    relations=(
        DimRelation("hdim_q", "==", "hdim_v"),  # single head_size arg
        DimRelation("nhead_q", "multiple_of", "nhead_k"),  # GQA grouping
    ),
    # Causal only. SwapQKCfg's mask_mode vocabulary is "none"/"causal" and it
    # has no sink or window field at all, so declaring either feature would
    # admit the request and compile plain causal for it.
    supports_features=frozenset({"causal"}),
)


def _swapqk_cfg(req: OperatorRequest) -> SwapQKCfg:
    assert isinstance(req, AttentionRequest)
    return SwapQKCfg(
        head_size=int(req.hdim_q),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        mask_mode="causal" if int(req.mask_type) != 0 else "none",
    )


def _args_signature() -> Tuple[dict, ...]:
    """The fixed kernel ABI, byte-identical to the gfx1250 WMMA forward."""
    ptr = {"type": "ptr<f16, global>", "size_bytes": 8}
    i32 = {"type": "i32", "size_bytes": 4}
    strides = (
        "stride_q_token",
        "stride_q_head",
        "stride_k_token",
        "stride_k_head",
        "stride_v_token",
        "stride_v_head",
        "stride_o_token",
        "stride_o_head",
    )
    return (
        *({"name": n, **ptr} for n in ("Q", "K", "V", "O")),
        {"name": "scale_log2", "type": "f32", "size_bytes": 4},
        *({"name": n, **i32} for n in ("seqlen_q", "seqlen_k", *strides)),
    )


def _make_swapqk_candidate() -> KernelCandidate:
    """gfx1151 transposed-QK WMMA FMHA forward — a standalone kernel.

    OPT-IN ONLY, matching the ``attention_gfx950_dense`` and
    ``attention_gfx1250_wmma`` precedents: selected solely when the request
    names ``algorithm="wmma_fmha_swapqk"`` or ``spec_id="gfx1151_swapqk"``.
    Registering it makes it reachable, which is what this phase is for; making
    it the gfx1151 *default* is a separate, measured decision across the shape
    space. gfx1151 prefill routes to ``unified_2d`` today, so flipping default
    routing here would swap a benchmarked path for an unbenchmarked one on the
    strength of a registration.
    """
    spec_id = "gfx1151_swapqk"
    name = "attention_gfx1151_swapqk"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if req.algorithm.strip().lower() != "wmma_fmha_swapqk" and (
            req.spec_id.strip().lower() != spec_id
        ):
            return False, (
                "gfx1151 swapqk FMHA is opt-in "
                "(algorithm='wmma_fmha_swapqk'); default gfx1151 prefill "
                "routes to unified_2d"
            )
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        # Capability already cleared arch / dtype / head_size / seqlen / mask.
        # Only the residual knob-compatibility gates remain.
        ok, why = _swapqk_is_valid(_swapqk_cfg(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> SwapQKCfg:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return _swapqk_cfg(req)

    def grid(spec: SwapQKCfg, req: OperatorRequest):
        assert isinstance(req, AttentionRequest)
        return swapqk_grid(spec, seqlen_q=int(req.seqlen_q), batch=int(req.batch))

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_swapqk_torch

        return bind_swapqk_torch(request, spec, tensors, **kwargs)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="wmma_fmha_swapqk",
        spec_id=spec_id,
        abi_version=ATTENTION_GFX1151_ABI,
        priority=5,
        capability=_SWAPQK_CAP,
        _supports=support,
        select_spec=select,
        build=build_wmma_fmha_swapqk,
        grid=grid,
        block=lambda spec: (spec.block_size, 1, 1),
        signature=lambda _spec: _args_signature(),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        bind_torch=bind_torch,
    )
    return candidate


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    candidate = _make_swapqk_candidate()
    route.register(candidate)
    execution.register(candidate)
