# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Executable gfx1151 WMMA attention selection for dense and packed layouts."""

from __future__ import annotations

from typing import Tuple

from kernels.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    build_wmma_fmha_fwd,
    is_valid_spec as _wmma_fwd_is_valid,
    wmma_fmha_fwd_grid,
    wmma_fmha_fwd_signature,
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
    AttentionMaskType,
    AttentionRequest,
    FAMILY,
    _request_errors,
    _selector_matches,
)

ATTENTION_GFX1151_ABI = "rocke-attention-gfx1151/v1"

_WMMA_FWD_CAP = Capability(
    arches=("gfx1151",),
    dtypes=("fp16", "bf16"),
    shapes=(ShapeRange("hdim_q", allowed=(64, 128, 256)),),
    relations=(
        DimRelation("hdim_q", "==", "hdim_v"),  # single head_size arg
        DimRelation("nhead_q", "multiple_of", "nhead_k"),  # GQA grouping
    ),
    supports_features=frozenset(
        {
            "causal",
            "causal_bottom_right",
            "sliding_window",
            "sinks",
            "fp8",
            "softcap",
            "alibi",
            "qq_bias",
            "layout_dense",
            "layout_ragged",
            "layout_paged",
        }
    ),
)


def _resolve_layout(req: AttentionRequest) -> str:
    """Resolve the standalone WMMA default without changing other candidates."""
    layout = req.layout.strip().lower()
    return "dense" if layout == "auto" else layout


def _wmma_fwd_spec(req: OperatorRequest) -> WmmaFmhaFwdSpec:
    assert isinstance(req, AttentionRequest)
    mask_type = AttentionMaskType(int(req.mask_type))
    layout = _resolve_layout(req)
    seqlen_q = int(req.seqlen_q)
    seqlen_k = int(req.seqlen_k)
    query_tail = layout != "dense" or bool(seqlen_q % 16)
    kv_tail = layout != "dense" or bool(seqlen_k % 16)
    transposed = (
        req.dtype.strip().lower() == "fp16"
        and layout == "dense"
        and int(req.hdim_q) in (64, 128)
        and mask_type
        in (
            AttentionMaskType.NO_MASK,
            AttentionMaskType.TOP_LEFT_CAUSAL,
            AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        )
        and not query_tail
        and seqlen_k % 32 == 0
        and not (
            req.use_fp8
            or req.sliding_window
            or req.use_softcap
            or req.use_sinks
            or req.use_alibi
            or req.use_qq_bias
        )
    )
    wide = transposed and seqlen_q >= 512 and seqlen_q % 32 == 0 and seqlen_k % 64 == 0
    maximum_length = max(seqlen_q, seqlen_k)
    tuned_small_head = (
        req.dtype.strip().lower() == "fp16"
        and int(req.hdim_q) == 64
        and not (
            req.sliding_window
            or req.use_fp8
            or req.use_softcap
            or req.use_sinks
            or req.use_alibi
            or req.use_qq_bias
        )
        and maximum_length <= (1 << 30)
        and (
            (
                layout == "ragged"
                and mask_type
                in (
                    AttentionMaskType.TOP_LEFT_CAUSAL,
                    AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                )
            )
            or (layout == "dense" and (query_tail or kv_tail))
        )
    )
    query_groups = ((seqlen_q + 15) // 16) * int(req.nhead_q) * int(req.batch)
    tiled_values = (
        layout == "dense"
        and int(req.hdim_q) == 256
        and seqlen_k >= 128
        and query_groups <= 128
        and not (
            req.sliding_window
            or req.use_fp8
            or req.use_softcap
            or req.use_sinks
            or req.use_alibi
            or req.use_qq_bias
        )
    )
    short_query = seqlen_q <= 16
    value_tile_size = 0
    if tiled_values:
        value_tile_size = (
            32 if mask_type == AttentionMaskType.NO_MASK and not short_query else 64
        )
    # A window covering both advertised maxima removes no causally-visible key.
    # The admission bound also keeps context/window arithmetic within I32.
    return WmmaFmhaFwdSpec(
        head_size=int(req.hdim_q),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        dtype=req.dtype.strip().lower(),
        mask_mode="none" if mask_type == AttentionMaskType.NO_MASK else "causal",
        # Equal maxima can still describe unequal packed sequence lengths.
        causal_bottom_right=mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        sliding_window=(
            maximum_length
            if tuned_small_head and mask_type != AttentionMaskType.NO_MASK
            else int(req.sliding_window)
        ),
        v_lds_stage=tuned_small_head
        or (tiled_values and (short_query or mask_type == AttentionMaskType.NO_MASK)),
        value_tile_size=value_tile_size,
        scheduler_strategy="max-ilp" if tuned_small_head else None,
        query_tail=query_tail,
        kv_tail=kv_tail,
        use_softcap=bool(req.use_softcap),
        use_sinks=bool(req.use_sinks),
        use_alibi=bool(req.use_alibi),
        use_qq_bias=bool(req.use_qq_bias),
        layout=layout,
        page_block_size=int(req.kv_block_size) if layout == "paged" else 0,
        kv_dtype="fp8e4m3" if bool(req.use_fp8) else "",
        transposed_qk=transposed,
        block_n=64 if wide else 32,
        num_waves=2 if wide else 1,
    )


def _make_wmma_fwd_candidate() -> KernelCandidate:
    """Select a complete WMMA spec for auto dispatch or an explicit pin."""
    spec_id = "gfx1151_wmma_fmha_fwd"
    name = "attention_gfx1151_wmma"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        layout = _resolve_layout(req)
        if layout not in ("dense", "ragged", "paged"):
            return False, f"unsupported attention layout {req.layout!r}"
        if bool(req.use_fp8) and bool(req.fp8_fnuz):
            return False, (
                "gfx1151 WMMA FP8 KV is OCP e4m3fn only; fp8_fnuz is not "
                "representable by this kernel"
            )
        if layout == "paged":
            block = int(req.kv_block_size)
            if block <= 0 or block & (block - 1):
                return False, (
                    f"paged attention requires a positive power-of-two "
                    f"kv_block_size (got {block})"
                )
        try:
            mask_type = AttentionMaskType(int(req.mask_type))
        except ValueError:
            return False, f"unsupported mask_type {req.mask_type!r}"
        if (
            mask_type == AttentionMaskType.SLIDING_WINDOW
            and int(req.sliding_window) <= 0
        ):
            return False, "sliding-window mask_type requires sliding_window > 0"
        try:
            spec = _wmma_fwd_spec(req)
        except ValueError as error:
            return False, str(error)
        ok, why = _wmma_fwd_is_valid(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest) -> WmmaFmhaFwdSpec:
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return _wmma_fwd_spec(req)

    def grid(spec: WmmaFmhaFwdSpec, req: OperatorRequest):
        assert isinstance(req, AttentionRequest)
        return wmma_fmha_fwd_grid(
            spec, seqlen_q=int(req.seqlen_q), batch=int(req.batch)
        )

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_gfx1151_attention_torch

        return bind_gfx1151_attention_torch(request, spec, tensors, **kwargs)

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="wmma_fmha_fwd",
        spec_id=spec_id,
        abi_version=ATTENTION_GFX1151_ABI,
        priority=5,
        capability=_WMMA_FWD_CAP,
        _supports=support,
        select_spec=select,
        build=build_wmma_fmha_fwd,
        grid=grid,
        block=lambda spec: (spec.block_size, 1, 1),  # one wave32 per CTA
        signature=wmma_fmha_fwd_signature,
        sweep_space=lambda req: (select(req),) if candidate.admits(req)[0] else (),
        bind_torch=bind_torch,
    )
    return candidate


def register(registry: CandidateRegistry) -> None:
    registry.register(_make_wmma_fwd_candidate())
