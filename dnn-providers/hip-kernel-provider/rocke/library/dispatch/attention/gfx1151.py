# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Executable RDNA WMMA attention selection for dense and packed layouts.

One kernel set per RDNA generation: requests name ``gfx11-generic`` (code
objects load on gfx1100-gfx1153) or ``gfx12-generic`` (gfx1200/gfx1201). A
request may still name ``gfx1151`` or ``gfx1201`` to build processor-specific
objects. The transposed-QK profile uses the gfx11 WMMA operand layout; gfx12
targets run the standard path for those shapes.

``attention_gfx1151_wmma`` is the public auto route on every RDNA arch. Each
arch also registers an opt-in tuned candidate, ``attention_<arch>_wmma_fwd``
(``spec_id`` ``<arch>_wmma_fwd``, ``-`` spelled ``_``), whose default is the
same policy spec and whose knob space (:mod:`.wmma_rules`) is what tuning
sweeps, ``tuning_id`` pins and registry-driven AOT catalogs enumerate.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Optional, Tuple

from kernels.gfx1151.wmma_fmha_fwd import (
    WmmaFmhaFwdSpec,
    build_wmma_fmha_fwd,
    is_valid_spec as _wmma_fwd_is_valid,
    wmma_fmha_fwd_grid,
    uses_gfx11_operand_layout,
    wmma_fmha_fwd_signature,
)
from rocke.core.arch import base_arch_from_target_id, generic_arch_from_target_id
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DimRelation,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
)
from rocke.dispatch.tuning import make_tuned_candidate

from .common import (
    AttentionMaskType,
    AttentionRequest,
    FAMILY,
    _device_num_cus,
    _request_errors,
    _selector_matches,
    _tuning_problem,
)
from .wmma_rules import WMMA_PATH, WmmaProblem, WmmaSpace, wmma_variants

# v6 composes the runtime-shape-generic base ABI (including runtime windows and
# runtime-gated LSE) with unequal Q/V widths, causal tile skipping, additive
# attention bias, output-column tiling, opt-in runtime head widths and V head
# counts, and the expanded multiple-of-16 catalog. Callers compiled against any
# earlier header must be rebuilt.
ATTENTION_GFX1151_ABI = "rocke-attention-gfx1151/v6"

# Compute units of the reference gfx1151 part, where the output-column tiling
# gate was tuned. Used only when the request names no count and device 0 cannot
# run the requested object (cross-compiles and AOT packaging).
_DEFAULT_NUM_CUS = 40
# Workgroups per CU the tiled-value path is expected to keep in flight; the
# tiling gate admits a request while its query groups fit this many per CU
# (``query_groups * 5 <= num_cus * 16``, i.e. 128 groups on the 40-CU part).
_TILED_VALUE_GROUPS_PER_16_CUS = 16
_TILED_VALUE_GROUP_WEIGHT = 5


def _resolve_num_cus(req: AttentionRequest) -> int:
    """Compute units used by CU-aware selection.

    The live count of device 0 is used only when that device runs the requested
    code object: any member of the family for a generic request, the named
    processor otherwise. A cross-compile never bakes in another part's size.
    """
    n = int(req.num_cus)
    if n > 0:
        return n
    try:
        from rocke.runtime.hip_module import get_device_target_id

        device = get_device_target_id()
        family = generic_arch_from_target_id(req.arch)
        if not device or not family:
            return _DEFAULT_NUM_CUS
        if req.arch == family:
            matches = generic_arch_from_target_id(device) == family
        else:
            matches = base_arch_from_target_id(device) == req.arch
        if matches:
            live = _device_num_cus()
            if live and live > 0:
                return 2 * int(live)
    except Exception:
        pass
    return _DEFAULT_NUM_CUS


def _tuning_num_cus(req: AttentionRequest) -> int:
    """The CU count a tuned candidate's base reads: the request's own, else the
    reference part's. Never the live device, so a pinned or enumerated spec is
    a pure function of the request (see ``make_tuned_candidate``)."""
    n = int(req.num_cus)
    return n if n > 0 else _DEFAULT_NUM_CUS


# Every RDNA target the WMMA forward kernel is registered for.
RDNA_WMMA_ARCHES = ("gfx11-generic", "gfx1151", "gfx12-generic", "gfx1201")
WMMA_FWD_ALGORITHM = "wmma_fmha_fwd"

_WMMA_FWD_CAP = Capability(
    arches=RDNA_WMMA_ARCHES,
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
            # RDNA WMMA objects can serve runtime head widths from one bucket.
            "runtime_head_dims",
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
    causal = int(req.mask_type) in (
        AttentionMaskType.TOP_LEFT_CAUSAL,
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    )
    if req.window_right is not None:
        right = int(req.window_right)
    elif left >= 0 or causal:
        # A causal mask keeps its diagonal even with an unbounded left side
        # (cuDNN's causal + window_left=-1 is plain causal, not "no mask").
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
    # The transposed-QK body is a gfx11-layout kernel; the layout also decides
    # whether staging FP8 V through LDS pays (gfx11) or not (gfx12).
    gfx11_layout = uses_gfx11_operand_layout(req.arch, req.dtype.strip().lower())
    query_tail = layout != "dense" or bool(seqlen_q % 16)
    kv_tail = layout != "dense" or bool(seqlen_k % 16)
    windowed = (
        mask_type == AttentionMaskType.SLIDING_WINDOW
        or int(req.sliding_window) > 0
        or req.window_left is not None
        or (req.window_right is not None and int(req.window_right) >= 0)
    )
    # A runtime-head-width bucket object (hdim_q/hdim_v are its maxima) runs the
    # standard body untiled, and the small-head V-LDS tuning measured on exact
    # D64 objects does not carry over to it (the shipped buckets stage V
    # through registers).
    runtime_head_dims = bool(req.runtime_head_dims)
    transposed = (
        not runtime_head_dims
        and gfx11_layout
        and req.dtype.strip().lower() in ("fp16", "bf16")
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
    # gfx11-generic objects schedule under the family's smaller register model;
    # they prefer narrower K tiles in wide transposed launches and wider D256
    # noncausal output-column tiles than processor-specific gfx1151 builds.
    generic_target = generic_arch_from_target_id(req.arch) == req.arch
    tuned_small_head = (
        not runtime_head_dims
        and req.dtype.strip().lower() == "fp16"
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
    fp8 = bool(req.use_fp8)
    head = int(req.hdim_q)
    query_groups = ((seqlen_q + 15) // 16) * int(req.nhead_q) * int(req.batch)
    # Output-column tiling multiplies the CTAs of a launch too small to fill the
    # device: D256 always, and D128 when FP8 KV decode keeps the softmax cheap.
    tiled_values = (
        not runtime_head_dims
        and layout == "dense"
        and equal_heads
        and ((head == 256 and not fp8) or (head == 128 and fp8))
        and seqlen_k >= 128
        and not (
            windowed
            or req.use_softcap
            or req.use_sinks
            or req.use_alibi
            or req.use_qq_bias
            or req.use_attn_bias
        )
        and query_groups * _TILED_VALUE_GROUP_WEIGHT
        <= _resolve_num_cus(req) * _TILED_VALUE_GROUPS_PER_16_CUS
    )
    short_query = seqlen_q <= 16
    value_tile_size = 0
    if tiled_values:
        value_tile_size = (
            32
            if head == 128
            or (
                mask_type == AttentionMaskType.NO_MASK
                and not short_query
                and not (generic_target and gfx11_layout)
            )
            else 64
        )
    # The max-ILP machine scheduler keeps more V gathers in flight on the
    # standard path. FP8 KV (software decode) and noncausal output tiles keep
    # the default scheduler.
    max_ilp = (
        not transposed
        and not fp8
        and not (tiled_values and mask_type == AttentionMaskType.NO_MASK)
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
        or (
            tiled_values
            and head == 256
            and (short_query or mask_type == AttentionMaskType.NO_MASK)
        )
        or (fp8 and gfx11_layout and not tiled_values),
        value_tile_size=value_tile_size,
        scheduler_strategy="max-ilp" if max_ilp else None,
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
        block_n=64 if wide and not generic_target else 32,
        num_waves=2 if wide else 1,
        runtime_head_dims=runtime_head_dims,
    )


def _wmma_problem(req: AttentionRequest, spec: WmmaFmhaFwdSpec) -> WmmaProblem:
    return WmmaProblem(
        spec=spec,
        seqlen_q=int(req.seqlen_q),
        seqlen_k=int(req.seqlen_k),
        num_query_heads=int(req.nhead_q),
        batch=int(req.batch),
    )


def tuned_wmma_selectors(arch: str) -> Tuple[str, str]:
    """``(candidate name, spec_id)`` of ``arch``'s tuned WMMA forward candidate."""
    tag = arch.replace("-", "_")
    return f"attention_{tag}_wmma_fwd", f"{tag}_wmma_fwd"


@lru_cache(maxsize=None)
def _wmma_space(arch: str) -> WmmaSpace:
    return WmmaSpace(
        abi=ATTENTION_GFX1151_ABI,
        arch=arch,
        path=WMMA_PATH,
        variant_id="fwd",
        candidate_name=tuned_wmma_selectors(arch)[0],
    )


def _wmma_request_refusal(req: AttentionRequest) -> Optional[str]:
    """Why the WMMA forward kernel cannot serve ``req`` (beyond the capability
    and the structural request checks), or None."""
    layout = _resolve_layout(req)
    if layout not in ("dense", "ragged", "paged"):
        return f"unsupported attention layout {req.layout!r}"
    if bool(req.use_fp8) and bool(req.fp8_fnuz):
        return (
            "gfx1151 WMMA FP8 KV is OCP e4m3fn only; fp8_fnuz is not "
            "representable by this kernel"
        )
    if layout == "paged":
        block = int(req.kv_block_size)
        if block <= 0 or block & (block - 1):
            return (
                f"paged attention requires a positive power-of-two "
                f"kv_block_size (got {block})"
            )
    try:
        mask_type = AttentionMaskType(int(req.mask_type))
    except ValueError:
        return f"unsupported mask_type {req.mask_type!r}"
    if mask_type == AttentionMaskType.SLIDING_WINDOW:
        left, right = _window_bounds(req)
        if left == -1 and right == -1:
            return "window attention requires at least one finite bound"
    return None


def _wmma_request_errors(req: OperatorRequest):
    return _request_errors(req, allow_unequal_head_dims=True)


def _make_wmma_fwd_candidate() -> KernelCandidate:
    """Select a complete WMMA spec for auto dispatch or an explicit pin."""
    spec_id = "gfx1151_wmma_fmha_fwd"
    name = "attention_gfx1151_wmma"

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _wmma_request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        why = _wmma_request_refusal(req)
        if why is not None:
            return False, why
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
        # The policy spec, then the production variants of the arch's tuned
        # space around it (each knob changes performance, never results).
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, AttentionRequest)
        return wmma_variants(
            _wmma_space(req.arch), _wmma_problem(req, _wmma_fwd_spec(req))
        )

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
        algorithm=WMMA_FWD_ALGORITHM,
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


def _make_tuned_wmma_fwd_candidate(arch: str) -> KernelCandidate:
    """``arch``'s opt-in tuned WMMA forward candidate.

    Pinned by ``algorithm="wmma_fmha_fwd"`` and its own ``spec_id``; its default
    (``tuning_id="auto"``) is the auto route's policy spec for the request, with
    the CU count taken from the request (else the reference part's), never the
    live device. ``tuning_id``/``tuning_knobs`` name one point of the arch's
    WMMA space; ``runtime_head_dims`` requests build bucket objects.
    """
    name, spec_id = tuned_wmma_selectors(arch)

    def base(req: AttentionRequest) -> WmmaProblem:
        pure = dataclasses.replace(req, num_cus=_tuning_num_cus(req))
        return _wmma_problem(req, _wmma_fwd_spec(pure))

    def precheck(req: AttentionRequest) -> Tuple[bool, str]:
        why = _wmma_request_refusal(req)
        return (False, why) if why is not None else (True, "ok")

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_gfx1151_attention_torch

        return bind_gfx1151_attention_torch(
            request, spec.kernel_spec, tensors, **kwargs
        )

    return make_tuned_candidate(
        name=name,
        family=FAMILY,
        algorithm=WMMA_FWD_ALGORITHM,
        spec_id=spec_id,
        abi_version=ATTENTION_GFX1151_ABI,
        priority=5,
        capability=dataclasses.replace(_WMMA_FWD_CAP, arches=(arch,)),
        space=_wmma_space(arch),
        base=base,
        request_errors=_wmma_request_errors,
        precheck=precheck,
        signature=lambda spec: wmma_fmha_fwd_signature(spec.kernel_spec),
        build=lambda spec, arch_: spec.build(arch_),
        bind_torch=bind_torch,
        grid=lambda spec, req: spec.launch_grid(_tuning_problem(req)),
    )


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    for candidate in (_make_wmma_fwd_candidate(),) + tuple(
        _make_tuned_wmma_fwd_candidate(arch) for arch in RDNA_WMMA_ARCHES
    ):
        route.register(candidate)
        execution.register(candidate)
