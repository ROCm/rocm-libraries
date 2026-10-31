# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Executable RDNA WMMA attention selection (gfx11 / gfx12) for dense and
packed layouts.

One kernel set per RDNA generation: requests name ``gfx11-generic`` (code
objects load on gfx1100-gfx1154) or ``gfx12-generic`` (gfx1200/gfx1201). A
request may still name ``gfx1151`` or ``gfx1201`` to build processor-specific
objects.

Two kernels serve every RDNA request:

* the multi-wave prefill (``kernels.gfx1151.wmma_fmha_fwd_v2``), one object
  per (dtype, head-width bucket, mask class, bias kind, feature flags), auto
  route ``attention_rdna_wmma_prefill``;
* the split-KV, GQA-packed decode (``kernels.gfx1151.wmma_fmha_decode``, decode
  kernel plus its merge kernel), auto route ``attention_rdna_wmma_decode``,
  which outranks the prefill exactly when :func:`rdna_attention_use_decode`
  says so.

Each arch also registers an opt-in tuned candidate per kernel,
``attention_<arch>_wmma_prefill`` / ``attention_<arch>_wmma_decode`` (``spec_id``
without the ``attention_`` prefix, ``-`` spelled ``_``), whose default is the
auto route's spec and whose knob space (:mod:`.wmma_rules`) is what tuning
sweeps, ``tuning_id`` pins and registry-driven AOT catalogs enumerate.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

from kernels.gfx1151.wmma_fmha_decode import (
    MERGE_THREADS,
    WmmaFmhaDecodeSpec,
    build_wmma_fmha_decode,
    build_wmma_fmha_decode_merge,
    is_valid_spec as _decode_is_valid,
    resolve_wmma_fmha_decode_spec,
    wmma_fmha_decode_bucket,
    wmma_fmha_decode_fused_merge,
    wmma_fmha_decode_grid,
    wmma_fmha_decode_kv_range,
    wmma_fmha_decode_merge_grid,
    wmma_fmha_decode_num_splits,
    wmma_fmha_decode_pack_groups,
    wmma_fmha_decode_row_tiles,
    wmma_fmha_decode_signature,
    wmma_fmha_decode_workspace,
)
from kernels.gfx1151.wmma_fmha_fwd_v2 import (
    WmmaFmhaFwdV2Spec,
    build_wmma_fmha_fwd_v2,
    is_gfx12_family,
    is_valid_spec as _prefill_is_valid,
    wmma_fmha_fwd_v2_grid,
    wmma_fmha_fwd_v2_signature,
)
from rocke.core.arch import base_arch_from_target_id, generic_arch_from_target_id
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DimRelation,
    KernelCandidate,
    OperatorRequest,
    ShapeRange,
    normalize_selector,
)
from rocke.dispatch.tuning import make_tuned_candidate

from .common import (
    AttentionMaskType,
    AttentionRequest,
    FAMILY,
    _device_num_cus,
    _request_errors,
    _selector_matches,
)
from .wmma_rules import (
    WMMA_DECODE_PATH,
    WMMA_PATH,
    WmmaProblem,
    WmmaSpace,
    wmma_variants,
)

# v7 retires the single-wave forward kernel: the prefill takes ``batch`` and a
# flat 1-D grid, head widths are multiples of 8 inside a bucket, and decode-
# shaped requests run the split-KV decode with its workspace and merge kernel.
# Callers compiled against any earlier header must be rebuilt.
ATTENTION_GFX1151_ABI = "rocke-attention-gfx1151/v7"

# Compute units of the reference gfx1151 part. Used only when the request names
# no count and device 0 cannot run the requested object (cross-compiles and AOT
# packaging).
_DEFAULT_NUM_CUS = 40


def _resolve_num_cus(req: AttentionRequest) -> int:
    """Compute units used by CU-aware selection and the decode split count.

    The live count of device 0 is used only when that device runs the requested
    code object: any member of the family for a generic request, the named
    processor otherwise. A cross-compile never bakes in another part's size.
    HIP reports RDNA multiprocessors as WGPs (two CUs each).
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


# Every RDNA target the WMMA kernels are registered for.
RDNA_WMMA_ARCHES = ("gfx11-generic", "gfx1151", "gfx12-generic", "gfx1201")
PREFILL_ALGORITHM = "wmma_fmha_fwd_v2"
DECODE_ALGORITHM = "wmma_fmha_decode"

_RDNA_FEATURES = frozenset(
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
        # Every object takes its head widths (multiples of 8 within the bucket)
        # and its V head count at launch.
        "runtime_head_dims",
    }
)

# Both kernels take runtime head widths in multiples of 8 (16-byte rows).
_RDNA_CAP = Capability(
    arches=RDNA_WMMA_ARCHES,
    dtypes=("fp16", "bf16"),
    shapes=(
        ShapeRange(frozenset({"hdim_q", "hdim_v"}), min=8, max=256, multiple_of=8),
    ),
    relations=(DimRelation("nhead_q", "multiple_of", "nhead_k"),),  # GQA grouping
    supports_features=_RDNA_FEATURES,
)

# Prefill head-width buckets. Q/K and V/O are bucketed independently; the
# 192-wide Q/K bucket exists only beside a V/O bucket of at most 128 (MLA-style
# Q192/V128 runs one wave per 16 rows on gfx12 instead of a 256-wide pair).
_PREFILL_QK_BUCKETS = (64, 96, 128, 192, 256)
_PREFILL_V_BUCKETS = (64, 96, 128, 256)


def prefill_head_buckets(head_dim_q: int, head_dim_v: int) -> Tuple[int, int]:
    """``(Q/K bucket, V/O bucket)`` of the prefill object serving these widths.

    The smallest V bucket in 64/96/128/256 holding ``head_dim_v``, and the
    smallest Q bucket in 64/96/128/192/256 holding ``head_dim_q`` (192 only when
    the V bucket is at most 128). Raises ``ValueError`` above 256.
    """
    hv = next((b for b in _PREFILL_V_BUCKETS if head_dim_v <= b), None)
    if hv is None or head_dim_q > 256:
        raise ValueError(f"head widths {head_dim_q}/{head_dim_v} exceed 256")
    hq = next(
        b for b in _PREFILL_QK_BUCKETS if head_dim_q <= b and (b != 192 or hv <= 128)
    )
    return hq, hv


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


def _windowed(req: AttentionRequest) -> bool:
    return (
        int(req.mask_type) == AttentionMaskType.SLIDING_WINDOW
        or int(req.sliding_window) > 0
        or req.window_left is not None
        or (req.window_right is not None and int(req.window_right) >= 0)
    )


def mask_arguments(req: AttentionRequest) -> Tuple[bool, int, int]:
    """``(bottom_right, window_left, window_right)``: the runtime mask both
    kernels take (``-1`` = unbounded side, ``window_right = 0`` = causal)."""
    mask_type = AttentionMaskType(int(req.mask_type))
    if _windowed(req):
        left, right = _window_bounds(req)
    elif mask_type in (
        AttentionMaskType.TOP_LEFT_CAUSAL,
        AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
    ):
        left, right = -1, 0
    else:
        left, right = -1, -1
    bottom_right = mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL or (
        mask_type == AttentionMaskType.SLIDING_WINDOW and bool(req.window_bottom_right)
    )
    return bool(bottom_right), left, right


def _bias_kind(req: AttentionRequest) -> Optional[str]:
    if not bool(req.use_attn_bias):
        return None
    return "f32" if req.attn_bias_dtype.strip().lower() == "f32" else "q"


def _wmma_family(arch: str) -> str:
    return "gfx12" if is_gfx12_family(arch) else "gfx11"


# ----------------------------------------------------------------- decode rule
def rdna_attention_use_decode(
    *,
    seqlen_q: int,
    visible_keys: int,
    num_query_heads: int,
    num_kv_heads: int,
    batch: int,
    wmma_family: str,
    num_cus: int,
) -> bool:
    """Whether the split-KV decode kernel runs a request rather than the
    prefill. A pure function the hipDNN native mirrors exactly.

    ``visible_keys`` is the key range any query row sees (``hi - lo`` of
    ``wmma_fmha_decode_kv_range``: the window's span, else ``seqlen_k``; the
    maxima for packed layouts), ``wmma_family`` is ``"gfx11"`` or ``"gfx12"``
    and ``num_cus`` counts CUs (twice HIP's RDNA multiprocessor count).

    The decode kernel packs the ``g = Hq / Hkv`` query heads of a K/V head and
    their positions into 16-row tiles and splits the keys across workgroups;
    the prefill runs one 128-row CTA per (query tile, query head). Decode wins
    while ``g * Sq`` fills few tiles. Each decode tile re-reads its K/V head,
    so on gfx11 512 packed rows lose once the key stream is long for the
    device (at least 64 keys per CU, and 512 per CU over the batch); on gfx12
    the prefill's tiles win at Sq 64 once short key ranges make the problem
    compute bound. Past 64 rows per head the prefill wins unless its CTAs
    would leave most of the device idle.

    The thresholds are the decode/prefill crossover points: the 2048-key gfx12
    threshold and the prefill-underfill exception mark where the prefill
    kernel overtakes split-KV decode.
    """
    g = max(1, num_query_heads // max(num_kv_heads, 1))
    if seqlen_q > 64:
        prefill_ctas = batch * num_query_heads * -(-seqlen_q // 128)
        return 4 * prefill_ctas < num_cus
    if wmma_family == "gfx12":
        return seqlen_q <= 32 or g == 1 or visible_keys >= 2048
    return (
        g * seqlen_q < 512
        or visible_keys < 64 * num_cus
        or batch * visible_keys < 512 * num_cus
    )


# ---------------------------------------------------------------- spec policy
def _prefill_spec(req: AttentionRequest) -> WmmaFmhaFwdV2Spec:
    """The auto route's prefill object for ``req``: the shipped geometry
    (``block_m`` 128, or 64 over a wave pair; 32-key tiles) with every derived
    field explicit, so the spec names its kernel and its tuning point."""
    layout = _resolve_layout(req)
    hq, hv = prefill_head_buckets(int(req.hdim_q), int(req.hdim_v))
    # gfx12 keeps Q/K up to 256 in one wave while V/O fits 128.
    pair = max(hq, hv) > 128 and not (is_gfx12_family(req.arch) and hv <= 128)
    mask_type = AttentionMaskType(int(req.mask_type))
    masked = _windowed(req) or mask_type != AttentionMaskType.NO_MASK
    return WmmaFmhaFwdV2Spec(
        dtype=req.dtype.strip().lower(),
        head_dim=hq,
        head_dim_v=hv,
        mask="causal" if masked else "none",
        bias=_bias_kind(req),
        use_softcap=bool(req.use_softcap),
        use_sinks=bool(req.use_sinks),
        use_alibi=bool(req.use_alibi),
        use_qq_bias=bool(req.use_qq_bias),
        layout=layout,
        page_block_size=int(req.kv_block_size) if layout == "paged" else 0,
        kv_dtype="fp8e4m3" if bool(req.use_fp8) else "",
        block_m=64 if pair else 128,
        wave_pair=pair,
    )


def _decode_spec(req: AttentionRequest) -> WmmaFmhaDecodeSpec:
    """The auto route's decode object for ``req``, resolved for its arch."""
    bucket = wmma_fmha_decode_bucket(
        head_dim_q=int(req.hdim_q), head_dim_v=int(req.hdim_v)
    )
    if bucket is None:
        raise ValueError(
            f"no decode bucket serves head widths {req.hdim_q}/{req.hdim_v}"
        )
    layout = _resolve_layout(req)
    spec = WmmaFmhaDecodeSpec(
        head_size=bucket,
        dtype=req.dtype.strip().lower(),
        bias=_bias_kind(req) or "",
        softcap=bool(req.use_softcap),
        sinks=bool(req.use_sinks),
        alibi=bool(req.use_alibi),
        qq_bias=bool(req.use_qq_bias),
        kv_dtype="fp8e4m3" if bool(req.use_fp8) else "",
        layout=layout,
        page_block_size=int(req.kv_block_size) if layout == "paged" else 0,
    )
    return resolve_wmma_fmha_decode_spec(spec, arch=req.arch)


def _wmma_problem(req: AttentionRequest, spec) -> WmmaProblem:
    return WmmaProblem(
        spec=spec,
        batch=int(req.batch),
        seqlen_q=int(req.seqlen_q),
        seqlen_k=int(req.seqlen_k),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
    )


# ---------------------------------------------------------------- decode plan
@dataclass(frozen=True)
class DecodeLaunchPlan:
    """Host-side launch geometry of one decode problem.

    ``num_splits`` comes from :func:`wmma_fmha_decode_num_splits` over the
    visible key range with ``num_cus // 2`` multiprocessors (RDNA WGPs, what
    HIP's ``multiProcessorCount`` reports). ``fused`` merges in the last split
    (WS_CNT passed); otherwise more than one split launches the merge kernel.
    ``workspace`` is :func:`wmma_fmha_decode_workspace`'s byte layout.
    """

    num_splits: int
    fused: bool
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    merge_grid: Tuple[int, int, int]
    merge_block: Tuple[int, int, int]
    workspace: dict


def decode_launch_plan(
    spec: WmmaFmhaDecodeSpec,
    *,
    batch: int,
    num_query_heads: int,
    num_kv_heads: int,
    num_v_heads: int,
    seqlen_q: int,
    seqlen_k: int,
    head_dim_v: int,
    bottom_right: bool,
    window_left: int,
    window_right: int,
    num_cus: int,
) -> DecodeLaunchPlan:
    """The decode launch for one problem (``spec`` resolved for its arch)."""
    heads = dict(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_v_heads=num_v_heads,
    )
    groups = wmma_fmha_decode_pack_groups(**heads)
    tiles = wmma_fmha_decode_row_tiles(seqlen_q=seqlen_q, **heads)
    lo, hi = wmma_fmha_decode_kv_range(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        bottom_right=bottom_right,
        window_left=window_left,
        window_right=window_right,
    )
    splits = wmma_fmha_decode_num_splits(
        spec,
        batch=batch,
        pack_groups=groups,
        row_tiles=tiles,
        kv_len=hi - lo,
        num_cus=max(1, int(num_cus) // 2),
    )
    return DecodeLaunchPlan(
        num_splits=splits,
        fused=wmma_fmha_decode_fused_merge(splits),
        grid=wmma_fmha_decode_grid(
            spec, batch=batch, pack_groups=groups, row_tiles=tiles, num_splits=splits
        ),
        block=(spec.block_size, 1, 1),
        merge_grid=wmma_fmha_decode_merge_grid(
            batch=batch, num_query_heads=num_query_heads, seqlen_q=seqlen_q
        ),
        merge_block=(MERGE_THREADS, 1, 1),
        workspace=wmma_fmha_decode_workspace(
            spec,
            batch=batch,
            seqlen_q=seqlen_q,
            head_dim_v=head_dim_v,
            num_splits=splits,
            **heads,
        ),
    )


def request_decode_plan(
    req: AttentionRequest,
    spec: WmmaFmhaDecodeSpec,
    *,
    num_v_heads: Optional[int] = None,
    num_cus: Optional[int] = None,
) -> DecodeLaunchPlan:
    """:func:`decode_launch_plan` for ``req`` (V heads default to K heads)."""
    bottom_right, left, right = mask_arguments(req)
    return decode_launch_plan(
        spec,
        batch=int(req.batch),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        num_v_heads=int(req.nhead_k if num_v_heads is None else num_v_heads),
        seqlen_q=int(req.seqlen_q),
        seqlen_k=int(req.seqlen_k),
        head_dim_v=int(req.hdim_v),
        bottom_right=bottom_right,
        window_left=left,
        window_right=right,
        num_cus=_resolve_num_cus(req) if num_cus is None else num_cus,
    )


def build_decode_kernels(spec: WmmaFmhaDecodeSpec, arch: str):
    """``(decode, merge)`` kernels of one decode object."""
    return (
        build_wmma_fmha_decode(spec, arch=arch),
        build_wmma_fmha_decode_merge(spec, arch=arch),
    )


# ---------------------------------------------------------------- candidates
def tuned_wmma_selectors(arch: str, kind: str) -> Tuple[str, str]:
    """``(candidate name, spec_id)`` of ``arch``'s tuned ``kind`` candidate
    (``"prefill"`` or ``"decode"``)."""
    if kind not in ("prefill", "decode"):
        raise ValueError(f"kind must be prefill or decode, got {kind!r}")
    tag = arch.replace("-", "_")
    return f"attention_{tag}_wmma_{kind}", f"{tag}_wmma_{kind}"


@lru_cache(maxsize=None)
def _wmma_space(arch: str, kind: str) -> WmmaSpace:
    return WmmaSpace(
        abi=ATTENTION_GFX1151_ABI,
        arch=arch,
        path=WMMA_PATH if kind == "prefill" else WMMA_DECODE_PATH,
        variant_id=kind,
        candidate_name=tuned_wmma_selectors(arch, kind)[0],
    )


def _wmma_request_refusal(req: AttentionRequest) -> Optional[str]:
    """Why neither RDNA kernel can serve ``req`` (beyond the capability and
    the structural request checks), or None."""
    layout = _resolve_layout(req)
    if layout not in ("dense", "ragged", "paged"):
        return f"unsupported attention layout {req.layout!r}"
    if bool(req.use_fp8) and bool(req.fp8_fnuz):
        return (
            "RDNA WMMA FP8 KV is OCP e4m3fn only; fp8_fnuz is not "
            "representable by these kernels"
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


def _prefill_refusal(req: AttentionRequest) -> Optional[str]:
    try:
        spec = _prefill_spec(req)
    except ValueError as error:
        return str(error)
    ok, why = _prefill_is_valid(spec, arch=req.arch)
    return None if ok else why


def _decode_refusal(req: AttentionRequest) -> Optional[str]:
    try:
        spec = _decode_spec(req)
    except ValueError as error:
        return str(error)
    ok, why = _decode_is_valid(spec, arch=req.arch)
    return None if ok else why


def _auto_selected(req: AttentionRequest) -> bool:
    return (
        normalize_selector(req.algorithm) == "auto"
        and normalize_selector(req.spec_id) == "auto"
    )


def _visible_keys(req: AttentionRequest) -> int:
    bottom_right, left, right = mask_arguments(req)
    lo, hi = wmma_fmha_decode_kv_range(
        seqlen_q=int(req.seqlen_q),
        seqlen_k=int(req.seqlen_k),
        bottom_right=bottom_right,
        window_left=left,
        window_right=right,
    )
    return hi - lo


def _decode_rule(req: AttentionRequest) -> bool:
    return rdna_attention_use_decode(
        seqlen_q=int(req.seqlen_q),
        visible_keys=_visible_keys(req),
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        batch=int(req.batch),
        wmma_family=_wmma_family(req.arch),
        num_cus=_resolve_num_cus(req),
    )


def _bind_prefill(request, spec, tensors, **kwargs):
    from .bindings import bind_rdna_prefill_torch

    return bind_rdna_prefill_torch(request, spec, tensors, **kwargs)


def _bind_decode(request, spec, tensors, **kwargs):
    from .bindings import bind_rdna_decode_torch

    return bind_rdna_decode_torch(request, spec, tensors, **kwargs)


def _make_auto_candidate(kind: str) -> KernelCandidate:
    """The auto route for ``kind``; also selected by pinning its algorithm.

    The decode route admits an unpinned request only when
    :func:`rdna_attention_use_decode` picks it (and outranks the prefill
    then), and never a ``runtime_head_dims`` request: a decode bucket serves
    Q/K widths only from its own minimum up, while the request promises every
    width up to the bucket. A request pinning ``algorithm="wmma_fmha_decode"``
    runs it on any shape it can serve.
    """
    decode = kind == "decode"
    name = f"attention_rdna_wmma_{kind}"
    make_spec = _decode_spec if decode else _prefill_spec
    refusal = _decode_refusal if decode else _prefill_refusal

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _wmma_request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        ok, why = _selector_matches(req, candidate)
        if not ok:
            return False, why
        why = _wmma_request_refusal(req) or refusal(req)
        if why is not None:
            return False, why
        if decode and _auto_selected(req):
            if bool(req.runtime_head_dims):
                return False, "runtime_head_dims buckets run on the prefill kernel"
            if not _decode_rule(req):
                return False, "the decode/prefill rule picks the prefill kernel"
        return True, "ok"

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return make_spec(req)

    def sweep(req: OperatorRequest):
        # The policy spec, then the production variants of the arch's tuned
        # space around it (each knob changes performance, never results).
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, AttentionRequest)
        return wmma_variants(
            _wmma_space(req.arch, kind), _wmma_problem(req, make_spec(req))
        )

    if decode:

        def grid(spec, req):
            return request_decode_plan(req, spec).grid

        build = build_decode_kernels
        signature = wmma_fmha_decode_signature
        block = lambda spec: (spec.block_size, 1, 1)  # noqa: E731
    else:

        def grid(spec, req):
            return wmma_fmha_fwd_v2_grid(
                spec,
                batch=int(req.batch),
                seqlen_q=int(req.seqlen_q),
                num_query_heads=int(req.nhead_q),
            )[0]

        build = lambda spec, arch: build_wmma_fmha_fwd_v2(spec, arch=arch)  # noqa: E731
        signature = wmma_fmha_fwd_v2_signature
        block = lambda spec: (spec.threads, 1, 1)  # noqa: E731

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm=DECODE_ALGORITHM if decode else PREFILL_ALGORITHM,
        spec_id=f"rdna_{DECODE_ALGORITHM if decode else PREFILL_ALGORITHM}",
        abi_version=ATTENTION_GFX1151_ABI,
        # The decode route only admits (unpinned) what its rule assigns it.
        priority=4 if decode else 5,
        capability=_RDNA_CAP,
        _supports=support,
        select_spec=select,
        build=build,
        grid=grid,
        block=block,
        signature=signature,
        sweep_space=sweep,
        bind_torch=_bind_decode if decode else _bind_prefill,
    )
    return candidate


def _make_tuned_candidate(arch: str, kind: str) -> KernelCandidate:
    """``arch``'s opt-in tuned ``kind`` candidate.

    Pinned by the kernel's ``algorithm`` and its own ``spec_id``; its default
    (``tuning_id="auto"``) is the auto route's spec for the request (the
    decode/prefill rule is not consulted: the pin chooses the kernel).
    ``tuning_id``/``tuning_knobs`` name one point of the arch's space.
    """
    decode = kind == "decode"
    name, spec_id = tuned_wmma_selectors(arch, kind)
    make_spec = _decode_spec if decode else _prefill_spec
    refusal = _decode_refusal if decode else _prefill_refusal

    def base(req: AttentionRequest) -> WmmaProblem:
        return _wmma_problem(req, make_spec(req))

    def precheck(req: AttentionRequest) -> Tuple[bool, str]:
        why = _wmma_request_refusal(req) or refusal(req)
        return (False, why) if why is not None else (True, "ok")

    def bind_torch(request, spec, tensors, **kwargs):
        bind = _bind_decode if decode else _bind_prefill
        return bind(request, spec.kernel_spec, tensors, **kwargs)

    if decode:

        def grid(spec, req):
            # The binding's own plan (live CU count), so the reported grid is
            # the launched one.
            return request_decode_plan(req, spec.kernel_spec).grid

        signature = lambda spec: wmma_fmha_decode_signature(
            spec.kernel_spec
        )  # noqa: E731
    else:

        def grid(spec, req):
            return wmma_fmha_fwd_v2_grid(
                spec.kernel_spec,
                batch=int(req.batch),
                seqlen_q=int(req.seqlen_q),
                num_query_heads=int(req.nhead_q),
            )[0]

        signature = lambda spec: wmma_fmha_fwd_v2_signature(
            spec.kernel_spec
        )  # noqa: E731

    return make_tuned_candidate(
        name=name,
        family=FAMILY,
        algorithm=DECODE_ALGORITHM if decode else PREFILL_ALGORITHM,
        spec_id=spec_id,
        abi_version=ATTENTION_GFX1151_ABI,
        priority=5,
        capability=dataclasses.replace(_RDNA_CAP, arches=(arch,)),
        space=_wmma_space(arch, kind),
        base=base,
        request_errors=_wmma_request_errors,
        precheck=precheck,
        signature=signature,
        build=lambda spec, arch_: spec.build(arch_),
        bind_torch=bind_torch,
        grid=grid,
    )


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    candidates = (
        _make_auto_candidate("decode"),
        _make_auto_candidate("prefill"),
    ) + tuple(
        _make_tuned_candidate(arch, kind)
        for kind in ("prefill", "decode")
        for arch in RDNA_WMMA_ARCHES
    )
    for candidate in candidates:
        route.register(candidate)
        execution.register(candidate)
