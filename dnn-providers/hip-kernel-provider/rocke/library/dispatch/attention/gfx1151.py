# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Executable gfx1151 WMMA attention selection for dense and packed layouts."""

from __future__ import annotations

import dataclasses
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
    _device_num_cus,
    _request_errors,
    _selector_matches,
)

# v5 composes the runtime-shape-generic base ABI (including runtime windows and
# runtime-gated LSE) with unequal Q/V widths, causal tile skipping, additive
# attention bias, output-column tiling, and the expanded multiple-of-16 catalog.
# Callers compiled against any earlier header must be rebuilt.
ATTENTION_GFX1151_ABI = "rocke-attention-gfx1151/v5"

# Compute units of the reference gfx1151 part. The output-column tiling gate was
# tuned at this size, so it is the fallback when no count is given or visible.
_GFX1151_DEFAULT_NUM_CUS = 40
# Workgroups per CU the tiled-value path is expected to keep in flight; the
# tiling gate admits a request while its query groups fit this many per CU
# (``query_groups * 5 <= num_cus * 16``, i.e. 128 groups on the 40-CU part).
_TILED_VALUE_GROUPS_PER_16_CUS = 16
_TILED_VALUE_GROUP_WEIGHT = 5


def _resolve_gfx1151_num_cus(req: AttentionRequest) -> int:
    """Compute units used by CU-aware gfx1151 selection."""
    n = int(req.num_cus)
    if n > 0:
        return n
    try:
        from rocke.runtime.hip_module import get_device_arch

        if get_device_arch() == "gfx1151":
            live = _device_num_cus()
            if live and live > 0:
                return 2 * int(live)
    except Exception:
        pass
    return _GFX1151_DEFAULT_NUM_CUS


_WMMA_FWD_CAP = Capability(
    arches=("gfx1151",),
    dtypes=("fp16", "bf16"),
    shapes=(
        ShapeRange(frozenset({"hdim_q", "hdim_v"}), min=16, max=256, multiple_of=16),
    ),
    relations=(DimRelation("nhead_q", "multiple_of", "nhead_k"),),  # GQA grouping
    supports_features=frozenset(
        {
            "causal",
            "causal_bottom_right",
            "sliding_window",
            "window_bounds",
            "noncausal_window",
            "lse",
            "sinks",
            "fp8",
            "softcap",
            "alibi",
            "qq_bias",
            "window_right",
            "attn_bias",
            "layout_dense",
            "layout_bhsd",
            "layout_ragged",
            "layout_paged",
        }
    ),
)


def _resolve_layout(req: AttentionRequest) -> str:
    """Resolve the standalone WMMA default without changing other candidates."""
    layout = req.layout.strip().lower()
    return "dense" if layout == "auto" else layout


def _window_bounds(req: AttentionRequest) -> tuple[int, int]:
    """Resolve explicit bounds and the legacy left-width/right-bound API."""
    if req.window_left is not None:
        left = int(req.window_left)
    elif int(req.sliding_window) > 0:
        left = int(req.sliding_window) - 1
    else:
        left = -1
    if req.window_right is not None:
        right = int(req.window_right)
    elif left >= 0:
        right = 0
    else:
        right = -1
    return left, right


def _wmma_fwd_spec(req: OperatorRequest) -> WmmaFmhaFwdSpec:
    assert isinstance(req, AttentionRequest)
    mask_type = AttentionMaskType(int(req.mask_type))
    layout = _resolve_layout(req)
    seqlen_q = int(req.seqlen_q)
    seqlen_k = int(req.seqlen_k)
    equal_heads = int(req.hdim_q) == int(req.hdim_v)
    query_tail = layout != "dense" or bool(seqlen_q % 16)
    kv_tail = layout != "dense" or bool(seqlen_k % 16)
    windowed = (
        mask_type == AttentionMaskType.SLIDING_WINDOW
        or int(req.sliding_window) > 0
        or req.window_left is not None
        or (req.window_right is not None and int(req.window_right) >= 0)
    )
    transposed = (
        req.dtype.strip().lower() in ("fp16", "bf16")
        and layout == "dense"
        and equal_heads
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
            or windowed
            or req.use_softcap
            or req.use_sinks
            or req.use_alibi
            or req.use_qq_bias
            or req.use_attn_bias
        )
    )
    wide = transposed and seqlen_q >= 512 and seqlen_q % 32 == 0 and seqlen_k % 64 == 0
    tuned_small_head = (
        req.dtype.strip().lower() == "fp16"
        and int(req.hdim_q) == 64
        and equal_heads
        and not (
            windowed
            or req.use_fp8
            or req.use_softcap
            or req.use_sinks
            or req.use_alibi
            or req.use_qq_bias
            or req.use_attn_bias
        )
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
        and equal_heads
        and seqlen_k >= 128
        and query_groups * _TILED_VALUE_GROUP_WEIGHT
        <= _resolve_gfx1151_num_cus(req) * _TILED_VALUE_GROUPS_PER_16_CUS
        and not (
            windowed
            or req.use_fp8
            or req.use_softcap
            or req.use_sinks
            or req.use_alibi
            or req.use_qq_bias
            or req.use_attn_bias
        )
    )
    short_query = seqlen_q <= 16
    value_tile_size = 0
    if tiled_values:
        value_tile_size = (
            32 if mask_type == AttentionMaskType.NO_MASK and not short_query else 64
        )
    # Runtime windows already bound the loop; the transposed path bounds it at
    # the diagonal by construction.
    causal_tile_skip = (
        mask_type
        in (
            AttentionMaskType.TOP_LEFT_CAUSAL,
            AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
        )
        and not windowed
        and not transposed
    )
    return WmmaFmhaFwdSpec(
        head_size=int(req.hdim_q),
        v_head_size=0 if equal_heads else int(req.hdim_v),
        dtype=req.dtype.strip().lower(),
        mask_mode=(
            "window"
            if windowed
            else ("none" if mask_type == AttentionMaskType.NO_MASK else "causal")
        ),
        causal_tile_skip=causal_tile_skip,
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
        use_attn_bias=bool(req.use_attn_bias),
        bias_dtype=req.attn_bias_dtype.strip().lower(),
        layout=layout,
        page_block_size=int(req.kv_block_size) if layout == "paged" else 0,
        kv_dtype="fp8e4m3" if bool(req.use_fp8) else "",
        transposed_qk=transposed,
        block_n=64 if wide else 32,
        num_waves=2 if wide else 1,
    )


def _sweep_variants(spec: WmmaFmhaFwdSpec, req: AttentionRequest):
    """Valid, launchable single-knob variants of ``spec`` (baseline excluded).

    Each knob changes only performance, never results, so the autotuner can
    time them against the heuristic baseline.
    """
    seqlen_q = int(req.seqlen_q)
    seqlen_k = int(req.seqlen_k)
    variants = []

    def replace(base, **changes):
        try:
            return dataclasses.replace(base, **changes)
        except ValueError:
            return None

    if spec.transposed_qk:
        for block_n in (32, 64):
            for num_waves in (1, 2):
                if seqlen_k % block_n or seqlen_q % (16 * num_waves):
                    continue
                variants.append(replace(spec, block_n=block_n, num_waves=num_waves))
    else:
        variants.append(replace(spec, v_lds_stage=not spec.v_lds_stage))
        if spec.mask_mode == "causal":
            variants.append(replace(spec, causal_tile_skip=not spec.causal_tile_skip))
        if spec.layout == "dense" and spec.v_dim >= 128:
            for tile in (0, 32, 64, 128):
                if tile < spec.v_dim:
                    variants.append(replace(spec, value_tile_size=tile))
        if spec.scheduler_strategy is None:
            variants.append(replace(spec, scheduler_strategy="max-ilp"))
        else:
            variants.append(replace(spec, scheduler_strategy=None))
    out = []
    for variant in variants:
        if variant is None or variant == spec or variant in out:
            continue
        try:
            ok, _ = _wmma_fwd_is_valid(variant, arch=req.arch)
            if ok:
                wmma_fmha_fwd_grid(
                    variant,
                    seqlen_q=seqlen_q,
                    num_query_heads=int(req.nhead_q),
                    batch=int(req.batch),
                )
        except ValueError:
            continue
        if ok:
            out.append(variant)
    return tuple(out)


def _make_wmma_fwd_candidate() -> KernelCandidate:
    """Select a complete WMMA spec for auto dispatch or an explicit pin."""
    spec_id = "gfx1151_wmma_fmha_fwd"
    name = "attention_gfx1151_wmma"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req, allow_unequal_head_dims=True)
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
        if mask_type == AttentionMaskType.SLIDING_WINDOW:
            left, right = _window_bounds(req)
            if left == -1 and right == -1:
                return False, "window attention requires at least one finite bound"
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

    def sweep(req: OperatorRequest):
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, AttentionRequest)
        baseline = _wmma_fwd_spec(req)
        return (baseline,) + _sweep_variants(baseline, req)

    def grid(spec: WmmaFmhaFwdSpec, req: OperatorRequest):
        assert isinstance(req, AttentionRequest)
        return wmma_fmha_fwd_grid(
            spec,
            seqlen_q=int(req.seqlen_q),
            num_query_heads=int(req.nhead_q),
            batch=int(req.batch),
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
        block=lambda spec: (spec.block_size, 1, 1),  # num_waves wave32s per CTA
        signature=wmma_fmha_fwd_signature,
        sweep_space=sweep,
        bind_torch=bind_torch,
    )
    return candidate


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    candidate = _make_wmma_fwd_candidate()
    route.register(candidate)
    execution.register(candidate)
