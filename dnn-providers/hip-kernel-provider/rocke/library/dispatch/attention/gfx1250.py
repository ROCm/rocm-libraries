# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 attention candidates (wave32, 16x16x32 WMMA).

``kernels/gfx1250/wmma_attention_fwd.py`` has had a complete spec, validator,
builder, and grid helper for some time; nothing referenced them, so the kernel
was unreachable. This module is the registration that makes it dispatchable.

It is also the first attention candidate that can state its launch geometry.
The unified 2D/3D candidates report ``grid=(0, 0, 0)`` and an empty signature
because their geometry is chosen downstream on the running device; this kernel
is self-contained, its grid is ``(seqlen_q // 16, num_query_heads, batch)``, and
its argument list is fixed, so both are declared here for real.
"""

from __future__ import annotations

from typing import Tuple

from kernels.gfx1250.wmma_attention_fwd import (
    BLOCK_M,
    DTYPES,
    WMMA_K,
    WmmaAttentionFwdSpec,
    build_wmma_attention_fwd,
    is_valid_spec as _wmma_fwd_is_valid,
    wmma_attention_fwd_grid,
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

ATTENTION_GFX1250_ABI = "rocke-attention-gfx1250/v1"

# Declared coverage, restating the spec's __post_init__ and is_valid_spec gates
# as DATA -- but taking the numbers from the kernel rather than transcribing
# them. That matters concretely here: WmmaAttentionFwdSpec RAISES from
# __post_init__ on a bad dtype / head_size / mask_mode, so the prefilter has to
# reject those before select_spec ever constructs one, and a prefilter that
# quietly disagrees with the gate it mirrors is worse than no prefilter.
_WMMA_FWD_CAP = Capability(
    arches=("gfx1250",),
    dtypes=DTYPES,
    shapes=(
        # head_size rides the WMMA contraction; seqlen_q rides the M tile,
        # which the grid helper refuses a remainder of.
        ShapeRange("hdim_q", min=WMMA_K, multiple_of=WMMA_K),
        ShapeRange("seqlen_q", min=BLOCK_M, multiple_of=BLOCK_M),
    ),
    relations=(
        DimRelation("hdim_q", "==", "hdim_v"),  # single head_size arg
        DimRelation("nhead_q", "multiple_of", "nhead_k"),  # GQA grouping
    ),
    # Causal only. The spec's mask_mode vocabulary is "none"/"causal", and
    # apply_attention_mask reads sliding_window solely under a "sliding_window"
    # mode this spec cannot express -- so the field is inert and a window would
    # be silently dropped. Declaring the feature here would admit that request
    # and compile plain causal for it. No sinks either, for the same reason.
    supports_features=frozenset({"causal"}),
)


def _wmma_fwd_spec(req: OperatorRequest) -> WmmaAttentionFwdSpec:
    assert isinstance(req, AttentionRequest)
    return WmmaAttentionFwdSpec(
        head_size=int(req.hdim_q),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        dtype="fp16",
        mask_mode="causal" if int(req.mask_type) != 0 else "none",
        sliding_window=int(req.sliding_window),
    )


def _args_signature() -> Tuple[dict, ...]:
    """The fixed kernel ABI, mirroring ``_declare_params`` in the kernel."""
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


def _make_wmma_fwd_candidate() -> KernelCandidate:
    """gfx1250 WMMA FMHA forward — a standalone kernel, not a unified path.

    OPT-IN ONLY, matching the ``attention_gfx950_dense`` precedent: it is
    selected solely when the request names ``algorithm="wmma_attention_fwd"``
    or ``spec_id="gfx1250_wmma_fwd"``. Registering it makes it reachable, which
    is what this phase is for; making it the *default* on gfx1250 is a separate,
    measured decision. gfx1250 fp16 prefill routes to ``unified_2d`` today, and
    that path -- not this one -- is what the gfx1250 prefill benchmark exercises,
    so flipping default routing here would swap a benchmarked path for an
    unbenchmarked one on the strength of a registration.
    """
    spec_id = "gfx1250_wmma_fwd"
    name = "attention_gfx1250_wmma"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        if req.algorithm.strip().lower() != "wmma_attention_fwd" and (
            req.spec_id.strip().lower() != spec_id
        ):
            return False, (
                "gfx1250 WMMA FMHA is opt-in "
                "(algorithm='wmma_attention_fwd'); default gfx1250 prefill "
                "routes to unified_2d"
            )
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        # Capability already cleared arch / dtype / head_size / mask, so the
        # spec construction below cannot raise. Only residual checks here.
        ok, why = _wmma_fwd_is_valid(_wmma_fwd_spec(req), arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> WmmaAttentionFwdSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return _wmma_fwd_spec(req)

    def grid(spec: WmmaAttentionFwdSpec, req: OperatorRequest):
        assert isinstance(req, AttentionRequest)
        return wmma_attention_fwd_grid(
            spec, seqlen_q=int(req.seqlen_q), batch=int(req.batch)
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="wmma_attention_fwd",
        spec_id=spec_id,
        abi_version=ATTENTION_GFX1250_ABI,
        priority=5,
        capability=_WMMA_FWD_CAP,
        _supports=support,
        select_spec=select,
        build=build_wmma_attention_fwd,
        grid=grid,
        block=lambda spec: (spec.block_size, 1, 1),  # one wave32 per CTA
        signature=lambda _spec: _args_signature(),
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
    )
    return candidate


def register(registry: CandidateRegistry) -> None:
    registry.register(_make_wmma_fwd_candidate())
