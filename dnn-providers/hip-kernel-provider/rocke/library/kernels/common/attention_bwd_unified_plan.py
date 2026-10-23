# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host plan, exact predicate, workspace and declaration of the unified backward.

The unified family (THD queries over a paged KV cache) is a rocKE-native entry:
it is not declared to hipDNN, which keeps declining paged problems. Everything
here is a pure function of request fields and in-repo tables: no device query,
no allocation, no kernel build.

* :class:`AttnBwdUnifiedRequest`: one problem (shapes, per-tensor strides,
  paged-cache geometry, block-table format, mask, the caller's injectivity
  assertion, optional device-fill hints).
* :func:`attn_bwd_unified_support`: the exact predicate, ordered reason codes
  (:data:`REASON_CODES`), first failure wins.
* :func:`attn_bwd_unified_workspace_layout` / ``..._bytes``: the caller
  workspace (256-byte packed fp32 sub-buffers).
* :func:`attn_bwd_unified_tile_choice`: the page class and tile of a page
  size (tuned configuration, the largest shipped ``kN0`` dividing the page, a
  "sub" instance, or the one-wave ``kN0 = 16`` fallback).
* :func:`attn_bwd_unified_plan`: instances, runtime values, the launch
  sequence and every kernarg value.
* :func:`attn_bwd_unified_declaration`: the versioned declaration (version 2).

Launch sequence: direct page mode ``prep -> main -> convert(dQ)``; atomic page
mode ``prep -> page_zero -> main -> convert(dQ) -> page_convert(dK) ->
page_convert(dV)``. The prep and dQ convert are the general family's THD aux
kernels (``rocke.attn_bwd.v3``) with ``OFF_Q = cu_seqlens_q``; the main kernel
and the page roles use ``rocke.attn_bwd_unified.v1``.

Page size: any positive multiple of ``PAGE_GRANULE`` (16) tokens; a power of
two is not required (48, 80, ... take the ``"multi"`` class when a shipped
``kN0`` divides them, else the ``"sub"`` class or the one-wave fallback, see
:func:`attn_bwd_unified_tile_choice`). Every other page size is declined
``PAGE_SIZE``.

Caller contract on device data (the predicate sees shapes only):
``max_seqlen_q`` / ``max_seqlen_kv`` bound every query span / kv length;
``cu_seqlens_q`` is non-decreasing inside ``[0, total_q]``; every block id of
a live block is in ``[0, num_blocks)``; ``block_table_injective = True``
asserts that distinct (sequence, live logical block) pairs map to distinct
physical blocks. :func:`kernels.common.attention_bwd_unified_run.check_attn_bwd_unified_inputs`
checks all of it on host-visible data and rejects a violation. On the device
the defined behaviour for out-of-contract data is: a kv length outside
``[0, max_seqlen_kv]`` is clamped into it, and a block id outside ``[0,
num_blocks)`` is never dereferenced (its K / V rows read as zeros and its
dK / dV slots are not written; the main kernel passes the cache's
``page_rows`` to :class:`._attention_bwd_paged.PagedKv`, which turns the guard
on). The plan passes ``page_rows = num_blocks * page`` to the main kernel in
both page modes for that guard.

Every decline code has a kind (``Verdict.kind``, see
:data:`kernels.common.attention_bwd_plan.DECLINE_KINDS`); a
``not_yet_implemented`` decline's detail starts with ``NOT_YET_IMPLEMENTED:``.
An accepted request plans: the predicate resolves the plan and declines
``INSTANCE_UNAVAILABLE`` when it cannot be built.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional

from rocke.helpers.attention_band import I32_FAR
from rocke.helpers.spec import align_up, ceil_div

from kernels.common.attention_bwd import (
    BWD_ARCHS,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    attn_bwd_arch_facts,
    attn_bwd_params,
    static_unit_mults,
)
from kernels.common.attention_bwd_aux import (
    attn_bwd_aux_block_threads,
    attn_bwd_convert_grid,
    attn_bwd_page_grid,
    attn_bwd_prep_grid,
)
from kernels.common.attention_bwd_plan import (
    DECLINE_KINDS,
    NOT_YET_IMPLEMENTED,
    REQ_SCOPE_D256_BWD,
    SCALE_PLACEMENTS,
    AttnBwdPlan,
    LaunchStep,
    Verdict,
    _FrozenStrides,
    _overlaps,
    _scale_mults,
    attn_bwd_stage_vec,
    check_field_types,
    decline_verdict,
)
from kernels.common.attention_bwd_unified import (
    ATTN_BWD_UNIFIED_ABI,
    PAGE_GRANULE,
    UNIFIED_KNOBS,
    AttnBwdUnifiedSpec,
    attn_bwd_unified_params,
    validate_attn_bwd_unified_spec,
)

__all__ = [
    "ATTN_BWD_UNIFIED_DECLARATION_VERSION",
    "FEATURE_GROUPS",
    "REASON_CODES",
    "TUNED_PAGE_TILES",
    "AttnBwdUnifiedPolicy",
    "AttnBwdUnifiedRequest",
    "attn_bwd_unified_declaration",
    "attn_bwd_unified_perf_class",
    "attn_bwd_unified_plan",
    "attn_bwd_unified_support",
    "attn_bwd_unified_tile_choice",
    "attn_bwd_unified_tile_configs",
    "attn_bwd_unified_workspace_bytes",
    "attn_bwd_unified_workspace_layout",
    "shipped_tuning",
    "shipped_unified_policy",
]

ATTN_BWD_UNIFIED_DECLARATION_VERSION = 2
LOG2E = 1.4426950408889634
GENERAL_ABI = "rocke.attn_bwd.v3"

FEATURE_DECLINES = (
    "DROPOUT",
    "BIAS_IN_BACKWARD",
    "DBIAS",
    "ALIBI",
    "FP8",
    "DETERMINISTIC",
    "UNSUPPORTED_FEATURE",
)
REASON_CODES = (
    "ARCH_UNSUPPORTED",
    "ARCH_NOT_VALIDATED",
    *FEATURE_DECLINES,
    "SCALE_DEVICE_TENSOR",
    "DTYPE",
    "OUT_DTYPE",
    "LSE_FORMAT",
    "DQK_NE_DV",
    "HEAD_DIM_256_BWD",
    "HEAD_DIM",
    "GQA_RATIO",
    "BAND_BOUND",
    "STRIDE_FORMAT",
    "STRIDE_D",
    "STRIDE_NEGATIVE",
    "OUTPUT_OVERLAP",
    "STRIDE_ALIGN",
    "INDEX_RANGE",
    "GRID_LIMIT",
    "SEQ_LEN_FORMAT",
    "RAGGED_FORMAT",
    "KV_CACHE_LAYOUT",
    "PAGE_SIZE",
    "BLOCK_TABLE_FORMAT",
    "BLOCK_TABLE_NOT_INJECTIVE",
    "SINKS",
    "SOFTCAP",
    "EMPTY",
    "INSTANCE_UNAVAILABLE",
    "NOT_YET_DECLARED",
)

# Functional groups, each declared once its correctness gate has passed on
# the arch: direct scatter (injective tables) and the atomic page mode (any
# table, including shared prefix pages).
FEATURE_GROUPS = ("direct_pages", "atomic_pages")
SHIPPED_DECLARED_GROUPS: frozenset = frozenset(FEATURE_GROUPS)
# Targets with a hardware run of the unified suite (both page modes).
VALIDATED_ARCHS: frozenset = frozenset({"gfx942", "gfx950"})
RUNTIME_KNOBS = ("g_split", "lb_order", "scale_placement", "page_mode")
LB_ORDERS = {"natural": 0, "reverse": 1}
MASK_KINDS = ("none", "causal_tl", "causal_br", "window")
HEAD_MODES = ("mha", "gqa")
PAGE_CLASS_TAGS = ("multi", "sub", "fallback")
PERF_STATUSES = ("verified", "unverified", "below_baseline")

I32_MAX = 2**31 - 1
_GRID_YZ_MAX = 65535
_WS_ALIGN = 256
_Q_SIDE = ("q", "o", "do", "dq")
_CACHES = ("k", "v", "dk", "dv")
_TENSORS = _Q_SIDE + _CACHES

DECLINED_FEATURES = tuple(
    {"feature": feature, "code": code, "kind": DECLINE_KINDS[code]}
    for feature, code in (
        ("dropout", "DROPOUT"),
        ("bias / attn_mask in backward", "BIAS_IN_BACKWARD"),
        ("dBias", "DBIAS"),
        ("ALiBi", "ALIBI"),
        ("fp8 caches", "FP8"),
        ("deterministic backward", "DETERMINISTIC"),
        ("qq-bias and other features", "UNSUPPORTED_FEATURE"),
        ("attention sinks", "SINKS"),
        ("logit softcap", "SOFTCAP"),
        ("head dim 256", "HEAD_DIM_256_BWD"),
    )
)


# ---------------------------------------------------------------------------
# request, policy
# ---------------------------------------------------------------------------


def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _frozen_strides(mapping):
    """Read-only, hashable ``{tensor: tuple}`` map (malformed values kept as
    given for the ``STRIDE_FORMAT`` decline)."""
    return None if mapping is None else _FrozenStrides(mapping)


_REQ_BOOLS = (
    "bottom_right",
    "block_table_injective",
    "scale_is_device_tensor",
    "has_bias",
    "want_dbias",
    "deterministic",
    "alibi",
    "fp8",
    "sinks",
    "qq_bias",
)
_REQ_INTS = (
    "num_seqs",
    "h_q",
    "h_kv",
    "d_qk",
    "d_v",
    "total_q",
    "max_seqlen_q",
    "max_seqlen_kv",
    "page",
    "num_blocks",
    "max_blocks_per_seq",
    "left_bound",
    "right_bound",
    "tensor_alignment",
)
_REQ_OPT_INTS = ("bt_stride", "num_cus", "num_xcds")


@dataclass(frozen=True)
class AttnBwdUnifiedRequest:
    """Everything the unified predicate knows about one problem.

    Shapes: ``num_seqs`` sequences; ``total_q`` query tokens (the rows of
    ``Q, O, dO, dQ`` and ``LSE``); per-sequence query spans and kv lengths
    bounded by ``max_seqlen_q`` / ``max_seqlen_kv``; caches of ``num_blocks``
    pages of ``page`` tokens; a block table of ``max_blocks_per_seq`` int32
    entries per sequence at row stride ``bt_stride`` (``None``: packed).

    ``strides`` maps ``q o do dq`` to ``(token, head, D)`` and ``k v dk dv``
    to ``(block, token, head, D)`` element strides; a missing entry is packed
    (``[total_q, h_q, D]`` / ``[num_blocks, page, h_kv, D]``). ``lse_strides``
    are ``(token, head)`` (``()``: packed ``[total_q, h_q]``).

    Mask: the default is the per-sequence bottom-right causal band
    (``right_bound = 0``, ``bottom_right = True``); ``left_bound`` /
    ``right_bound`` ``-1`` are unbounded.
    """

    num_seqs: int
    h_q: int
    h_kv: int
    d_qk: int
    d_v: int
    total_q: int
    max_seqlen_q: int
    max_seqlen_kv: int
    page: int
    num_blocks: int
    max_blocks_per_seq: int
    dtype: str = "bf16"
    k_dtype: Optional[str] = None  # None = dtype (also for v, o, do)
    v_dtype: Optional[str] = None
    o_dtype: Optional[str] = None
    do_dtype: Optional[str] = None
    dq_dtype: Optional[str] = None
    dk_dtype: Optional[str] = None
    dv_dtype: Optional[str] = None
    strides: Optional[Mapping[str, tuple]] = None
    cache_dims: Optional[Mapping[str, tuple]] = None  # None = the canonical dims
    lse_dims: tuple = ()  # () = (total_q, h_q); (total_q, h_q, 1) is accepted
    lse_strides: tuple = ()  # () = packed
    lse_dtype: str = "fp32"
    bt_stride: Optional[int] = None  # None = max_blocks_per_seq
    block_table_dtype: str = "int32"
    block_table_dims: tuple = ()  # () = (num_seqs, max_blocks_per_seq)
    cu_seqlens_dtype: str = "int32"
    seq_lens_dtype: str = "int32"
    left_bound: int = -1
    right_bound: int = 0
    bottom_right: bool = True
    block_table_injective: bool = False
    tensor_alignment: int = 16  # bytes, smallest over every tensor base pointer
    scale: Optional[float] = None  # None = 1 / sqrt(d_qk)
    scale_is_device_tensor: bool = False
    has_bias: bool = False
    want_dbias: bool = False
    dropout: float = 0.0
    deterministic: bool = False
    alibi: bool = False
    fp8: bool = False
    sinks: bool = False
    softcap: float = 0.0
    qq_bias: bool = False
    other_features: tuple = ()
    num_cus: Optional[int] = None
    num_xcds: Optional[int] = None

    def __post_init__(self) -> None:
        # Flags are bools and extents ints (TypeError otherwise), as in the
        # general request.
        check_field_types(self, _REQ_BOOLS, _REQ_INTS, _REQ_OPT_INTS)
        object.__setattr__(self, "strides", _frozen_strides(self.strides))
        object.__setattr__(self, "cache_dims", _frozen_strides(self.cache_dims))
        for f in ("lse_dims", "lse_strides", "block_table_dims", "other_features"):
            object.__setattr__(self, f, tuple(getattr(self, f)))

    # -- derived views -------------------------------------------------------

    def tensor_strides(self, name: str) -> tuple:
        if self.strides is not None and name in self.strides:
            return self.strides[name]
        d = self.d_qk
        if name in _Q_SIDE:
            return (max(1, self.h_q) * d, d, 1)
        return (self.page * max(1, self.h_kv) * d, max(1, self.h_kv) * d, d, 1)

    def tensor_dims(self, name: str) -> tuple:
        if name in _Q_SIDE:
            return (self.total_q, self.h_q, self.d_qk)
        if self.cache_dims is not None and name in self.cache_dims:
            return self.cache_dims[name]
        return (self.num_blocks, self.page, self.h_kv, self.d_qk)

    def used_strides(self) -> Iterator[int]:
        """The strides (other than D) of every tensor that address more than
        one index, tensor by tensor (lazy: a caller may stop at the first)."""
        for name in _TENSORS:
            yield from _used_strides(self, name)

    @property
    def group(self) -> int:
        return self.h_q // self.h_kv if self.h_kv > 0 else 0

    @property
    def page_rows(self) -> int:
        return self.num_blocks * self.page

    @property
    def table_stride(self) -> int:
        return self.max_blocks_per_seq if self.bt_stride is None else self.bt_stride

    def resolved_scale(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        return 1.0 / math.sqrt(self.d_qk) if self.d_qk > 0 else 1.0


def _no_tuning(req, arch, page_class) -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True)
class AttnBwdUnifiedPolicy:
    """What ships for the unified family.

    * ``declared_groups``: functional groups (:data:`FEATURE_GROUPS`) whose
      correctness gate passed;
    * ``validated_archs``: targets with a hardware run of the unified suite
      (others are declined ``ARCH_NOT_VALIDATED``);
    * ``perf_status``: ``{perf_class: status}`` of the unified-vs-general
      gate (a missing class is ``unverified``); published, never a decline;
    * ``instance_unavailable``: unsalted main kernel names that failed the
      resource gate;
    * ``tuning``: ``(req, arch, page_class) -> {knob: value}`` spec knobs and
      runtime values (:data:`RUNTIME_KNOBS`); ``page_class`` is ``"multi"``
      or ``"sub"`` per the request's page size and the arch's default tile.
    """

    declared_groups: frozenset = SHIPPED_DECLARED_GROUPS
    validated_archs: frozenset = VALIDATED_ARCHS
    perf_status: Mapping = field(default_factory=dict)
    instance_unavailable: frozenset = frozenset()
    tuning: Callable[..., Mapping[str, Any]] = _no_tuning


# Tuned tile configuration per (arch, head size, page): rule (1) of the tile
# choice, from the paired unified-route measurements (provenance and
# selection rule: platform/dsl_docs/instances/attention_bwd_unified.md
# section 3). Keys are request shape only; values name a shipped
# configuration of attn_bwd_unified_tile_configs. Pages not listed follow
# rules (2) - (4). Every entry was measured on a paired cell of
# benchmarks/common/attention_bwd_unified_bench.py (cell named per entry).
TUNED_PAGE_TILES: Mapping[tuple[str, int, int], str] = MappingProxyType(
    {
        # pages below the default kN0: the default tile in the sub class
        # serves them better than the smaller multi tile of rule (2)
        ("gfx942", 32, 64): "sub64",  # mha_d32_page64
        ("gfx942", 64, 64): "sub64",  # mha_d64_page64_nomask
        ("gfx950", 32, 64): "sub64",  # mha_d32_page64
        ("gfx950", 64, 64): "sub64",  # mha_d64_page64_nomask
        ("gfx950", 128, 64): "sub64",  # mha_d128_page64
        ("gfx950", 64, 128): "sub128",  # mha_d64_page128
        # d128: the four-wave kN0 = 64 tile
        ("gfx942", 128, 64): "k64",  # mha_d128_page64
        ("gfx942", 128, 128): "k64",  # gqa_d128_page128
        ("gfx942", 128, 256): "k64",  # decode_like_gqa_page256
        ("gfx950", 128, 128): "k64",  # gqa_d128_page128
    }
)


def shipped_tuning(req, arch: str, page_class: str) -> Mapping[str, Any]:
    """The shipped tuning hook: the tuned tile of ``(arch, d, page)``, if any."""
    name = TUNED_PAGE_TILES.get((arch, req.d_qk, req.page))
    if name is None:
        return {}
    cfgs = attn_bwd_unified_tile_configs(arch, req.d_qk, attn_bwd_stage_vec(req))
    return dict(cfgs[name][1]) if name in cfgs else {}


def shipped_unified_policy() -> AttnBwdUnifiedPolicy:
    return AttnBwdUnifiedPolicy(tuning=shipped_tuning)


# ---------------------------------------------------------------------------
# helpers over request fields
# ---------------------------------------------------------------------------


def _align(x: int) -> int:
    return align_up(x, _WS_ALIGN)


def _decline(code: str, detail: str) -> Verdict:
    return decline_verdict(code, detail, REASON_CODES)


def _mask_kind(req: AttnBwdUnifiedRequest) -> str:
    left, right = req.left_bound, req.right_bound
    if left < 0 and right < 0:
        return "none"
    if left < 0 and right == 0:
        return "causal_br" if req.bottom_right else "causal_tl"
    return "window"


def _feature_decline(req: AttnBwdUnifiedRequest) -> Optional[Verdict]:
    checks = (
        ("DROPOUT", req.dropout != 0.0, f"dropout {req.dropout}"),
        ("BIAS_IN_BACKWARD", req.has_bias, "bias in backward"),
        ("DBIAS", req.want_dbias, "dBias requested"),
        ("ALIBI", req.alibi, "ALiBi"),
        ("FP8", req.fp8, "fp8 cache"),
        ("DETERMINISTIC", req.deterministic, "deterministic backward"),
        (
            "UNSUPPORTED_FEATURE",
            req.qq_bias or bool(req.other_features),
            f"qq_bias={req.qq_bias} other={req.other_features}",
        ),
    )
    for code, hit, detail in checks:
        if hit:
            return _decline(code, detail)
    return None


def _strides_wellformed(req: AttnBwdUnifiedRequest) -> Optional[str]:
    for n in _TENSORS:
        st = req.tensor_strides(n)
        want = 3 if n in _Q_SIDE else 4
        if not isinstance(st, tuple) or len(st) != want or not all(map(_is_int, st)):
            return f"{n} strides {st!r} (want {want} integers)"
    lse = req.lse_strides
    if lse and (len(lse) != 2 or not all(map(_is_int, lse))):
        return f"LSE strides {lse!r} (want (token, head))"
    return None


def _lse_layout(req: AttnBwdUnifiedRequest) -> tuple[Optional[str], tuple[int, int]]:
    """``(decline detail or None, (token, head) element strides)``."""
    if req.lse_dtype != "fp32":
        return f"LSE dtype {req.lse_dtype} (fp32 natural-log only)", (0, 0)
    dims = req.lse_dims
    want = (req.total_q, req.h_q)
    if dims and tuple(dims) not in (want, want + (1,)):
        return f"LSE dims {dims} not {want} (optionally with a trailing 1)", (0, 0)
    if not req.lse_strides:
        return None, (req.h_q, 1)
    st = req.lse_strides
    if len(st) != 2 or not all(map(_is_int, st)) or min(st) < 0:
        return f"LSE strides {st!r}", (0, 0)
    return None, (int(st[0]), int(st[1]))


def _used_strides(req: AttnBwdUnifiedRequest, name: str) -> list[int]:
    """Strides (other than D) of ``name`` that address more than one index."""
    st = req.tensor_strides(name)
    dims = req.tensor_dims(name)
    return [s for s, n in zip(st[:-1], dims[:-1]) if n > 1]


def _index_range(req: AttnBwdUnifiedRequest) -> Optional[str]:
    for n in _TENSORS:
        for s in req.tensor_strides(n):
            if s > I32_MAX:
                return f"{n} stride {s} exceeds i32"
    if req.table_stride > I32_MAX:
        return f"block-table row stride {req.table_stride} exceeds i32"
    if req.page_rows > I32_MAX:
        return f"num_blocks * page = {req.page_rows} >= 2^31"
    if req.max_blocks_per_seq * req.page >= I32_FAR:
        return (
            f"max_blocks_per_seq * page = {req.max_blocks_per_seq * req.page} >= 2^30"
        )
    if max(req.max_seqlen_q, req.max_seqlen_kv) >= I32_FAR:
        return "sequence bound >= 2^30"
    if req.total_q > I32_MAX:
        return f"total_q {req.total_q} exceeds i32"
    # in-slab (i32) indices: a page of one cache, a q tile of the q side
    for n in _CACHES:
        _b, tok, _h, _d = req.tensor_strides(n)
        if req.page * tok + req.d_qk > I32_MAX:
            return f"{n}: page * token stride exceeds i32"
    for n in _Q_SIDE:
        t, _h, _d = req.tensor_strides(n)
        if 64 * t + req.d_qk > I32_MAX:
            return f"{n}: a q tile of token stride {t} exceeds i32"
    return None


# ---------------------------------------------------------------------------
# tile choice
# ---------------------------------------------------------------------------

# Shipped sub pages ("sub" class instances of the default tile; a page is
# shipped only where it is below the default kN0, so 128 exists only on the
# kN0 = 256 tiles).
SHIPPED_SUB_PAGES = (16, 32, 64, 128)
# The one-wave 16 x 16 tile: kN0 = 16 divides every legal page.
_FALLBACK_CDNA = dict(
    waves=1,
    block_m=16,
    block_n=16,
    block_k4=16,
    warp_grid_g4=(1, 1),
    atom_g02="16x16x16",
    atom_g13="16x16x16",
    atom_g4="16x16x16",
    kv_residency="lds",
    kt_source="lds",
    transpose_source="lds_plain",
    pt_route="relabel",
    sched="none",
)


def attn_bwd_unified_tile_configs(arch: str, head_size: int, stage_vec: int) -> dict:
    """Shipped unified tile configurations: ``{name: (page class, knobs)}``.

    ``"default"`` (the arch start point) and, on CDNA, ``"k64"`` (a four-wave
    ``kN0 = 64`` tile) are ``"multi"`` configurations; ``"sub16"`` /
    ``"sub32"`` / ``"sub64"`` are the default tile in the ``"sub"`` class;
    ``"fallback"`` is the one-wave ``kN0 = 16`` tile. A configuration whose
    spec is illegal on the arch is left out.
    """
    return {
        name: (cls, dict(knobs))
        for name, cls, knobs in _tile_configs(arch, head_size, stage_vec)
    }


@functools.cache
def _tile_configs(arch: str, head_size: int, stage_vec: int) -> tuple:
    facts = attn_bwd_arch_facts(arch)
    cands = {"default": ("multi", {})}
    if facts.matrix_path == "mfma":
        km, g4 = (32, (2, 2)) if head_size == 32 else (16, (1, 4))
        cands["k64"] = (
            "multi",
            dict(waves=4, block_m=km, block_n=64, block_k4=32, warp_grid_g4=g4),
        )
        for p in SHIPPED_SUB_PAGES:
            cands[f"sub{p}"] = ("sub", {"sub_page": p})
        cands["fallback"] = ("multi", dict(_FALLBACK_CDNA))
    else:
        cands["fallback"] = ("multi", {})
    out = []
    for name, (cls, knobs) in cands.items():
        try:
            spec = AttnBwdUnifiedSpec(head_size=head_size, stage_vec=stage_vec, **knobs)
            validate_attn_bwd_unified_spec(spec, arch)
        except ValueError:
            continue
        out.append((name, cls, MappingProxyType(knobs)))
    return tuple(out)


def _page_legal(arch, head_size, dtype, stage_vec, page, knobs) -> bool:
    """Whether the spec of ``knobs`` is legal on ``arch`` and serves ``page``."""
    try:
        spec = AttnBwdUnifiedSpec(
            head_size=head_size, dtype=dtype, stage_vec=stage_vec, **knobs
        )
        kn = validate_attn_bwd_unified_spec(spec, arch).block_n
    except ValueError:
        return False
    sp = knobs.get("sub_page")
    return sp == page if sp is not None else page % kn == 0


def attn_bwd_unified_tile_choice(
    arch: str,
    head_size: int,
    dtype: str,
    stage_vec: int,
    page: int,
    tuned: Optional[Mapping[str, Any]] = None,
) -> tuple[str, dict]:
    """``(page class tag, spec knobs)`` serving page size ``page``.

    Rules in order: (1) the tuned knobs when they are legal for the page
    (``page % kN0 == 0``, or a ``sub_page`` equal to the page); (2) the shipped
    ``"multi"`` configuration with the largest ``kN0`` dividing the page;
    (3) a shipped ``"sub"`` instance with ``sub_page == page``; (4) the
    one-wave ``kN0 = 16`` fallback (tag ``"fallback"``). Under rules (2)-(4)
    the tuned knobs that the chosen configuration does not set (for example
    ``edge_tiles``) are kept when the merged spec is legal. Raises
    ``ValueError`` for a page that is not a multiple of 16.
    """
    if page <= 0 or page % PAGE_GRANULE:
        raise ValueError(f"PAGE_SIZE: page {page} is not a multiple of {PAGE_GRANULE}")
    tuned = dict(tuned or {})
    if tuned and _page_legal(arch, head_size, dtype, stage_vec, page, tuned):
        return ("sub" if tuned.get("sub_page") is not None else "multi"), tuned
    cfgs = attn_bwd_unified_tile_configs(arch, head_size, stage_vec)
    choice = None
    best = None
    for name, (cls, knobs) in cfgs.items():
        if cls != "multi" or name == "fallback":
            continue
        kn = validate_attn_bwd_unified_spec(
            AttnBwdUnifiedSpec(
                head_size=head_size, dtype=dtype, stage_vec=stage_vec, **knobs
            ),
            arch,
        ).block_n
        if page % kn == 0 and (best is None or kn > best[0]):
            best = (kn, knobs)
    if best is not None:
        choice = ("multi", best[1])
    if choice is None:
        for _name, (cls, knobs) in cfgs.items():
            if cls == "sub" and knobs["sub_page"] == page:
                choice = ("sub", knobs)
                break
    if choice is None:
        choice = ("fallback", cfgs["fallback"][1])
    tag, knobs = choice
    if tuned:
        merged = {**tuned, **knobs}
        if _page_legal(arch, head_size, dtype, stage_vec, page, merged):
            return tag, merged
    return tag, dict(knobs)


def _default_page_class(arch, req: AttnBwdUnifiedRequest, stage_vec: int) -> str:
    """The page class of the untuned choice (the tuning hook's argument)."""
    tag, _ = attn_bwd_unified_tile_choice(
        arch, req.d_qk, req.dtype, stage_vec, req.page
    )
    return "sub" if tag == "sub" else "multi"


def attn_bwd_unified_perf_class(
    req: AttnBwdUnifiedRequest, arch: str, page_class: str, page_mode: str
) -> tuple:
    """``(arch, "unified", d, dtype, mask kind, head mode, page class, page mode)``."""
    heads = "mha" if req.group == 1 else "gqa"
    return (
        arch,
        "unified",
        req.d_qk,
        req.dtype,
        _mask_kind(req),
        heads,
        page_class,
        page_mode,
    )


# ---------------------------------------------------------------------------
# predicate
# ---------------------------------------------------------------------------


def _instance_key(req: AttnBwdUnifiedRequest, arch: str, stage_vec: int, policy) -> str:
    """Unsalted base name of the main instance the plan would use (the
    ``"sub"`` class instances carry their page in the base name)."""
    page_class = _default_page_class(arch, req, stage_vec)
    tuned = {
        k: v
        for k, v in dict(policy.tuning(req, arch, page_class) or {}).items()
        if k in UNIFIED_KNOBS or k == "sub_page"
    }
    _tag, knobs = attn_bwd_unified_tile_choice(
        arch, req.d_qk, req.dtype, stage_vec, req.page, tuned
    )
    return AttnBwdUnifiedSpec(
        head_size=req.d_qk,
        dtype=req.dtype,
        page_mode="direct" if req.block_table_injective else "atomic",
        mask_class="none" if _mask_kind(req) == "none" else "band",
        stage_vec=stage_vec,
        sub_page=knobs.get("sub_page"),
    ).kernel_name()


def attn_bwd_unified_support(
    req: AttnBwdUnifiedRequest,
    arch: str,
    *,
    policy: Optional[AttnBwdUnifiedPolicy] = None,
) -> Verdict:
    """Exact host predicate of the unified family; first failure wins.

    An accepted request plans: after the request checks the predicate
    resolves the plan exactly as :func:`attn_bwd_unified_plan` does (the
    policy's tuning included) and declines ``INSTANCE_UNAVAILABLE`` when that
    fails; the declared-group check then uses the page mode the plan chose.
    """
    policy = shipped_unified_policy() if policy is None else policy
    v = _request_checks(req, arch, policy)
    if v is not None:
        return v
    try:
        plan = _resolve_plan(req, arch, policy)
    except ValueError as exc:
        return _decline("INSTANCE_UNAVAILABLE", f"no buildable instance: {exc}")
    return _policy_checks(plan, policy)


def _request_checks(
    req: AttnBwdUnifiedRequest, arch: str, policy: AttnBwdUnifiedPolicy
) -> Optional[Verdict]:
    """The predicate's checks up to the resource-gate list (``None``: passed)."""
    if arch not in BWD_ARCHS:
        return _decline("ARCH_UNSUPPORTED", f"{arch} is not a backward target")
    if arch not in policy.validated_archs:
        return _decline(
            "ARCH_NOT_VALIDATED", f"{arch} has no hardware run of the unified suite"
        )
    v = _feature_decline(req)
    if v is not None:
        return v
    if req.scale_is_device_tensor:
        return _decline("SCALE_DEVICE_TENSOR", "scale is a device-buffer tensor")
    ins = [req.dtype, req.k_dtype, req.v_dtype, req.o_dtype, req.do_dtype]
    ins = [req.dtype if x is None else x for x in ins]
    if req.dtype not in BWD_DTYPES or any(x != req.dtype for x in ins):
        return _decline("DTYPE", f"inputs {ins} not all fp16 or all bf16")
    outs = [
        req.dtype if x is None else x
        for x in (req.dq_dtype, req.dk_dtype, req.dv_dtype)
    ]
    if any(x != req.dtype for x in outs):
        return _decline("OUT_DTYPE", f"outputs {outs} differ from {req.dtype}")
    why, _ = _lse_layout(req)
    if why is not None:
        return _decline("LSE_FORMAT", why)
    if req.d_qk != req.d_v:
        return _decline("DQK_NE_DV", f"d_qk={req.d_qk} d_v={req.d_v}")
    if req.d_qk == 256:
        return _decline("HEAD_DIM_256_BWD", REQ_SCOPE_D256_BWD)
    if req.d_qk not in BWD_HEAD_SIZES:
        return _decline("HEAD_DIM", f"head dim {req.d_qk} not in {BWD_HEAD_SIZES}")
    if req.h_kv > 0 and req.h_q % req.h_kv:
        return _decline("GQA_RATIO", f"h_q={req.h_q} not a multiple of h_kv={req.h_kv}")
    left, right = req.left_bound, req.right_bound
    if not (-1 <= left < I32_FAR and -1 <= right < I32_FAR):
        return _decline("BAND_BOUND", f"left={left} right={right}")
    why = _strides_wellformed(req)
    if why is not None:
        return _decline("STRIDE_FORMAT", why)
    for n in _TENSORS:
        if req.tensor_strides(n)[-1] != 1:
            return _decline("STRIDE_D", f"{n} D stride {req.tensor_strides(n)[-1]}")
    for n in _TENSORS:
        if any(s < 0 for s in req.tensor_strides(n)):
            return _decline("STRIDE_NEGATIVE", f"{n} strides {req.tensor_strides(n)}")
    for n in ("dq", "dk", "dv"):
        dims = req.tensor_dims(n)
        if min(dims) > 0 and _overlaps(dims, req.tensor_strides(n)):
            return _decline("OUTPUT_OVERLAP", f"{n} strides self-overlap")
    if req.tensor_alignment < 2:
        return _decline(
            "STRIDE_ALIGN", f"base alignment {req.tensor_alignment} B < element size"
        )
    why = _index_range(req)
    if why:
        return _decline("INDEX_RANGE", why)
    # grid y = h_kv * g_split (g_split >= 1), grid z = num_seqs; grid x is
    # bounded by max_seqlen_kv < 2^30 (INDEX_RANGE)
    if req.h_kv > _GRID_YZ_MAX or req.num_seqs > _GRID_YZ_MAX:
        return _decline(
            "GRID_LIMIT", f"h_kv={req.h_kv} or num_seqs={req.num_seqs} > 65535"
        )
    if req.seq_lens_dtype != "int32":
        return _decline("SEQ_LEN_FORMAT", f"seq_lens_kv dtype {req.seq_lens_dtype}")
    if req.max_seqlen_kv < 0:
        return _decline("SEQ_LEN_FORMAT", "max_seqlen_kv < 0")
    if req.cu_seqlens_dtype != "int32":
        return _decline("RAGGED_FORMAT", f"cu_seqlens_q dtype {req.cu_seqlens_dtype}")
    if req.max_seqlen_q < 0 or req.total_q < 0:
        return _decline("RAGGED_FORMAT", "negative query extent")
    canon = (req.num_blocks, req.page, req.h_kv, req.d_qk)
    for n in _CACHES:
        if tuple(req.tensor_dims(n)) != canon:
            return _decline(
                "KV_CACHE_LAYOUT", f"{n} cache dims {req.tensor_dims(n)} != {canon}"
            )
    for g, src in (("dk", "k"), ("dv", "v")):
        if req.tensor_strides(g) != req.tensor_strides(src):
            return _decline(
                "KV_CACHE_LAYOUT", f"{g} cache strides differ from {src} cache strides"
            )
    if req.page <= 0 or req.page % PAGE_GRANULE:
        return _decline(
            "PAGE_SIZE", f"page {req.page} is not a multiple of {PAGE_GRANULE}"
        )
    bt_dims = req.block_table_dims or (req.num_seqs, req.max_blocks_per_seq)
    if req.block_table_dtype != "int32" or len(bt_dims) != 2:
        return _decline(
            "BLOCK_TABLE_FORMAT",
            f"block table {req.block_table_dtype} dims {bt_dims} (int32 2-D only)",
        )
    if tuple(bt_dims) != (req.num_seqs, req.max_blocks_per_seq):
        return _decline("BLOCK_TABLE_FORMAT", f"block table dims {bt_dims}")
    if req.num_seqs > 1 and req.table_stride < req.max_blocks_per_seq:
        return _decline("BLOCK_TABLE_FORMAT", f"row stride {req.table_stride}")
    if req.max_seqlen_kv > req.max_blocks_per_seq * req.page:
        return _decline(
            "BLOCK_TABLE_FORMAT",
            f"max_seqlen_kv {req.max_seqlen_kv} exceeds the table's "
            f"{req.max_blocks_per_seq} pages of {req.page}",
        )
    if not req.block_table_injective and "atomic_pages" not in policy.declared_groups:
        return _decline(
            "BLOCK_TABLE_NOT_INJECTIVE",
            "the atomic page mode is not declared on this arch; supply an "
            "injective block table",
        )
    if req.sinks:
        return _decline("SINKS", "attention sinks")
    if req.softcap:
        return _decline("SOFTCAP", f"softcap {req.softcap}")
    if (
        min(req.num_seqs, req.h_q, req.h_kv, req.d_qk, req.total_q) <= 0
        or min(req.max_seqlen_q, req.max_seqlen_kv, req.num_blocks) <= 0
        or req.max_blocks_per_seq <= 0
    ):
        return _decline(
            "EMPTY", "a sequence, head, token, block or head-dim extent is 0"
        )
    key = _instance_key(req, arch, attn_bwd_stage_vec(req), policy)
    if key in policy.instance_unavailable:
        return _decline("INSTANCE_UNAVAILABLE", f"{key} failed the resource gate")
    return None


def _policy_checks(plan: AttnBwdPlan, policy: AttnBwdUnifiedPolicy) -> Verdict:
    """The declared-group check of the page mode the plan resolved (a tuned
    ``g_split > 1`` or ``page_mode = "atomic"`` takes the atomic mode on an
    injective table)."""
    group = f"{plan.runtime['page_mode']}_pages"
    if group not in policy.declared_groups:
        return _decline("NOT_YET_DECLARED", f"feature group {group} not declared yet")
    return Verdict(True, "OK", "")


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


def attn_bwd_unified_workspace_layout(
    req: AttnBwdUnifiedRequest, *, page_mode: str = "atomic"
) -> dict:
    """``{name: (byte offset, bytes)}`` of every workspace sub-buffer, packed.

    ``A(x)`` rounds up to 256 bytes; every array is fp32::

        WS_DQ    A(4 * total_q * h_q * D)
        WS_LSE2  A(4 * total_q * h_q)
        WS_DSUM  A(4 * total_q * h_q)
        WS_PDK   A(4 * num_blocks * page * h_kv * D)   atomic page mode only
        WS_PDV   A(4 * num_blocks * page * h_kv * D)   atomic page mode only

    The q-side rows are exactly ``total_q`` (a dimension of Q); the page
    workspace is indexed by physical slot. The plan uses the atomic page mode
    whenever the table is not asserted injective or ``g_split > 1``.
    """
    if page_mode not in ("atomic", "direct"):
        raise ValueError(f"page_mode={page_mode!r}")
    d = req.d_qk
    sizes = [
        ("WS_DQ", 4 * req.total_q * req.h_q * d),
        ("WS_LSE2", 4 * req.total_q * req.h_q),
        ("WS_DSUM", 4 * req.total_q * req.h_q),
    ]
    if page_mode == "atomic":
        sizes.append(("WS_PDK", 4 * req.page_rows * req.h_kv * d))
        sizes.append(("WS_PDV", 4 * req.page_rows * req.h_kv * d))
    out, off = {}, 0
    for name, nbytes in sizes:
        out[name] = (off, nbytes)
        off += _align(nbytes)
    return out


def attn_bwd_unified_workspace_bytes(
    req: AttnBwdUnifiedRequest, *, page_mode: str = "atomic"
) -> int:
    layout = attn_bwd_unified_workspace_layout(req, page_mode=page_mode)
    return sum(_align(n) for _, n in layout.values())


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _freeze(mapping):
    return MappingProxyType(dict(mapping))


def attn_bwd_unified_plan(
    req: AttnBwdUnifiedRequest,
    arch: str,
    *,
    policy: Optional[AttnBwdUnifiedPolicy] = None,
) -> AttnBwdPlan:
    """Resolve instances, runtime values, launches and workspace for ``req``.

    Raises ``ValueError`` (with the reason code) when the predicate declines,
    in the predicate's order (request checks, plan resolution as
    ``INSTANCE_UNAVAILABLE``, declared groups). The plan's ``route`` is
    ``"unified"`` and its ``route_tag`` the page class (``"multi"``,
    ``"sub"`` or ``"fallback"``).
    """
    policy = shipped_unified_policy() if policy is None else policy
    verdict = _request_checks(req, arch, policy)
    if verdict is not None:
        raise ValueError(f"{verdict.code}: {verdict.detail}")
    try:
        plan = _resolve_plan(req, arch, policy)
    except ValueError as exc:
        raise ValueError(f"INSTANCE_UNAVAILABLE: no buildable instance: {exc}") from exc
    verdict = _policy_checks(plan, policy)
    if not verdict.ok:
        raise ValueError(f"{verdict.code}: {verdict.detail}")
    return plan


def _resolve_plan(
    req: AttnBwdUnifiedRequest, arch: str, policy: AttnBwdUnifiedPolicy
) -> AttnBwdPlan:
    """The plan of a request that passed the request checks (``ValueError``
    when no instance serves it under ``policy``)."""
    stage_vec = attn_bwd_stage_vec(req)
    tuned = dict(
        policy.tuning(req, arch, _default_page_class(arch, req, stage_vec)) or {}
    )
    spec_fields = set(UNIFIED_KNOBS) | {"sub_page"}
    unknown = set(tuned) - spec_fields - set(RUNTIME_KNOBS)
    if unknown:
        raise ValueError(f"tuning hook returned unknown knobs {sorted(unknown)}")
    runtime_in = {k: tuned.pop(k) for k in RUNTIME_KNOBS if k in tuned}
    placement = runtime_in.get("scale_placement", "fold_ds")
    if placement not in SCALE_PLACEMENTS:
        raise ValueError(f"scale_placement={placement!r} not in {SCALE_PLACEMENTS}")
    # The main kernel is compiled for the placement (its unit multipliers are
    # not applied, the general family's rule) unless the hook pins "runtime";
    # another pin is refused.
    static_scale = tuned.pop("static_scale", placement)
    if static_scale not in (placement, "runtime"):
        raise ValueError(
            f"static_scale={static_scale!r} does not match "
            f"scale_placement={placement!r}"
        )

    g_split = int(runtime_in.get("g_split", 1))
    group = req.group
    if g_split < 1 or group % g_split:
        raise ValueError(f"g_split={g_split} must divide the head group {group}")
    want_mode = runtime_in.get("page_mode")
    if want_mode not in (None, "atomic", "direct"):
        raise ValueError(f"page_mode={want_mode!r} not in ('atomic', 'direct')")
    if want_mode == "direct" and not req.block_table_injective:
        raise ValueError("page_mode='direct' needs block_table_injective")
    direct = req.block_table_injective and g_split == 1 and want_mode != "atomic"
    page_mode = "direct" if direct else "atomic"
    mask_kind = _mask_kind(req)
    tag, knobs = attn_bwd_unified_tile_choice(
        arch, req.d_qk, req.dtype, stage_vec, req.page, tuned
    )
    knobs = dict(knobs, static_scale=static_scale)
    spec = AttnBwdUnifiedSpec(
        head_size=req.d_qk,
        dtype=req.dtype,
        page_mode=page_mode,
        mask_class="none" if mask_kind == "none" else "band",
        stage_vec=stage_vec,
        **knobs,
    )
    spec = validate_attn_bwd_unified_spec(spec, arch)
    if (
        "head_pack" not in knobs
        and group > 1
        and req.max_seqlen_q * group <= spec.block_m
    ):
        # Short query spans: the group's query heads share the M rows of one
        # q tile instead of leaving them mostly masked.
        spec = validate_attn_bwd_unified_spec(
            dataclasses.replace(spec, head_pack=True), arch
        )
    km, kn = spec.block_m, spec.block_n
    if spec.head_pack and req.max_seqlen_q * group > km:
        raise ValueError("head_pack needs G * max_seqlen_q <= block_m")
    pack_heads = group if spec.head_pack else 0
    facts = attn_bwd_arch_facts(arch)
    # Direct scatter covers every slot of the last live page: the grid spans
    # roundup(max_seqlen_kv, page) keys; the atomic mode only the keys.
    cover = req.page if direct else kn
    n_kv_tiles = ceil_div(align_up(req.max_seqlen_kv, cover), kn)
    if req.h_kv * g_split > _GRID_YZ_MAX:
        raise ValueError(f"GRID_LIMIT: h_kv * g_split = {req.h_kv * g_split} > 65535")
    lb_name = runtime_in.get("lb_order", "natural")
    if lb_name not in LB_ORDERS:
        raise ValueError(
            f"lb_order={lb_name!r} not in {tuple(LB_ORDERS)} (mirror pairing is "
            "not built: the main kernel runs one kv tile per CTA)"
        )
    scale = req.resolved_scale()
    mults = _scale_mults(placement, scale, not direct)
    unit = static_unit_mults(spec.static_scale, spec.general_spec().dkv_mode)
    if any(mults[n] != 1.0 for n in unit):
        raise AssertionError("a multiplier the main kernel does not apply is not 1.0")
    layout = attn_bwd_unified_workspace_layout(req, page_mode=page_mode)
    ws_bytes = attn_bwd_unified_workspace_bytes(req, page_mode=page_mode)
    q_t, q_h, _ = req.tensor_strides("q")
    do_t, do_h, _ = req.tensor_strides("do")
    o_t, o_h, _ = req.tensor_strides("o")
    dq_t, dq_h, _ = req.tensor_strides("dq")
    k_st = req.tensor_strides("k")
    v_st = req.tensor_strides("v")
    lse_t, lse_h = _lse_layout(req)[1]
    left, right = req.left_bound, req.right_bound
    br = int(req.bottom_right)

    main_scalars = {
        "scale_log2": scale * LOG2E,
        "ds_mult": mults["ds_mult"],
        "dk_mult": mults["dk_mult"],
        "dq_mult": mults["dq_mult"],
        "q_t": q_t,
        "q_h": q_h,
        "do_t": do_t,
        "do_h": do_h,
        "kc_blk": k_st[0],
        "kc_tok": k_st[1],
        "kc_h": k_st[2],
        "vc_blk": v_st[0],
        "vc_tok": v_st[1],
        "vc_h": v_st[2],
        "bt_stride": req.table_stride,
        "h_q": req.h_q,
        "h_k": req.h_kv,
        "G": group,
        "num_seqs": req.num_seqs,
        "max_seqlen_q": req.max_seqlen_q,
        "max_seqlen_kv": req.max_seqlen_kv,
        "max_blocks_per_seq": req.max_blocks_per_seq,
        "page": req.page,
        "left": left,
        "right": right,
        "bottom_right": br,
        "ws_rows_q": req.total_q,
        # num_blocks * page in both page modes: the page workspace rows
        # (atomic mode) and the bound of the main kernel's block-id guard
        "page_rows": req.page_rows,
        "g_split": g_split,
        "n_kv_tiles": n_kv_tiles,
        "lb_order": LB_ORDERS[lb_name],
        "pair": 0,
        "xcd_n": 0,
        "xcd_chunk": 0,
        "pack_heads": pack_heads,
        "use_worklist": 0,
    }
    main_ptrs = {
        "Q": "q",
        "KC": "k_cache",
        "VC": "v_cache",
        "dO": "do",
        "DKC": "dk_cache" if direct else None,
        "DVC": "dv_cache" if direct else None,
        "WS_LSE2": "WS_LSE2",
        "WS_DSUM": "WS_DSUM",
        "WS_DQ": "WS_DQ",
        "WS_PDK": None if direct else "WS_PDK",
        "WS_PDV": None if direct else "WS_PDV",
        "CU_SEQLENS_Q": "cu_seqlens_q",
        "SEQ_LENS_KV": "seq_lens_kv",
        "BLOCK_TABLE": "block_table",
        "WORKLIST": None,
    }
    prep_spec = spec.aux_spec("prep")
    conv_spec = spec.aux_spec("convert")
    page_spec = spec.aux_spec("page_zero")
    common_len = {"has_len": 0, "len_stride": 0, "off64": 0}
    prep_scalars = {
        "o_b": 0,
        "o_h": o_h,
        "do_b": 0,
        "do_h": do_h,
        "l_b": 0,
        "l_h": lse_h,
        "o_t": o_t,
        "do_t": do_t,
        "l_t": lse_t,
        "h_q": req.h_q,
        "h_k": req.h_kv,
        "h_v": req.h_kv,
        "S_q_max": req.max_seqlen_q,
        "S_kv_max": req.max_seqlen_kv,
        **common_len,
        "q_mult": 1,
        "q_div": 1,
        "kv_mult": 1,
        "kv_div": 1,
        "ws_rows_q": req.total_q,
        "ws_rows_kv": 0,
        "ws_seg": 0,
        "zero_kv": 0,
        "zero_dq": 1,
        "use_worklist": 0,
        "n_batch": req.num_seqs,
        "wl_kn0": 0,
    }
    prep_ptrs = {
        "O": "o",
        "dO": "do",
        "LSE": "lse",
        "SEQ_Q": None,
        "SEQ_KV": None,
        "OFF_Q": "cu_seqlens_q",
        "OFF_KV": None,
        "WS_LSE2": "WS_LSE2",
        "WS_DSUM": "WS_DSUM",
        "WS_DQ": "WS_DQ",
        "WS_DK": None,
        "WS_DV": None,
        "WORKLIST": None,
    }
    launches = [
        LaunchStep(
            "prep",
            prep_spec.kernel_name("prep"),
            attn_bwd_prep_grid(
                prep_spec,
                arch=arch,
                s_q_max=req.max_seqlen_q,
                s_kv_max=0,
                h_q=req.h_q,
                h_k=req.h_kv,
                h_v=req.h_kv,
                batch=req.num_seqs,
                zero_kv=False,
            ),  # fmt: skip
            attn_bwd_aux_block_threads(prep_spec, arch=arch),
            _freeze(prep_scalars),
            _freeze(prep_ptrs),
        )
    ]
    page_ints = {
        "h_kv": req.h_kv,
        "num_seqs": req.num_seqs,
        "max_blocks_per_seq": req.max_blocks_per_seq,
        "bt_stride": req.table_stride,
        "page": req.page,
        "page_rows": req.page_rows,
    }
    page_grid = attn_bwd_page_grid(
        page_spec, arch=arch, max_blocks_per_seq=req.max_blocks_per_seq,
        page=req.page, h_kv=req.h_kv, num_seqs=req.num_seqs,
    )  # fmt: skip
    page_block = attn_bwd_aux_block_threads(page_spec, arch=arch)
    if not direct:
        launches.append(
            LaunchStep(
                "page_zero",
                page_spec.kernel_name("page_zero"),
                page_grid,
                page_block,
                _freeze(page_ints),
                _freeze(
                    {
                        "SEQ_LENS_KV": "seq_lens_kv",
                        "BLOCK_TABLE": "block_table",
                        "WS_PDK": "WS_PDK",
                        "WS_PDV": "WS_PDV",
                    }
                ),
            )
        )
    launches.append(
        LaunchStep(
            "main",
            spec.kernel_name(),
            (n_kv_tiles, req.h_kv * g_split, req.num_seqs),
            spec.waves * facts.wave_size,
            _freeze(main_scalars),
            _freeze(main_ptrs),
        )
    )
    conv_scalars = {
        "mult": mults["conv_dq"],
        "d_b": 0,
        "d_h": dq_h,
        "H": req.h_q,
        "d_t": dq_t,
        "S_max": req.max_seqlen_q,
        **common_len,
        "tok_mult": 1,
        "tok_div": 1,
        "ws_rows": req.total_q,
        "ws_seg": 0,
    }
    launches.append(
        LaunchStep(
            "convert",
            conv_spec.kernel_name("convert"),
            attn_bwd_convert_grid(
                conv_spec,
                arch=arch,
                s_max=req.max_seqlen_q,
                heads=req.h_q,
                batch=req.num_seqs,
            ),  # fmt: skip
            attn_bwd_aux_block_threads(conv_spec, arch=arch),
            _freeze(conv_scalars),
            _freeze({"SRC": "WS_DQ", "DST": "dq", "SEQ": None, "OFF": "cu_seqlens_q"}),
        )
    )
    if not direct:
        for src, dst, mult, st in (
            ("WS_PDK", "dk_cache", mults["conv_dk"], k_st),
            ("WS_PDV", "dv_cache", 1.0, v_st),
        ):
            scal = {
                "mult": mult,
                **page_ints,
                "d_blk": st[0],
                "d_tok": st[1],
                "d_h": st[2],
            }
            launches.append(
                LaunchStep(
                    "page_convert",
                    page_spec.kernel_name("page_convert"),
                    page_grid,
                    page_block,
                    _freeze(scal),
                    _freeze(
                        {
                            "SRC": src,
                            "SEQ_LENS_KV": "seq_lens_kv",
                            "BLOCK_TABLE": "block_table",
                            "DST": dst,
                        }
                    ),
                )
            )
    for step in launches:
        names = [n for n, _ in unified_stage_params(step.stage)]
        got = list(step.pointers) + list(step.scalars)
        if sorted(got) != sorted(names):
            raise AssertionError(f"{step.stage} kernargs out of sync with the ABI")
    page_class = tag
    runtime = {
        "g_split": g_split,
        "lb_order": lb_name,
        "pair": 0,
        "xcd_n": 0,
        "xcd_chunk": 0,
        "pack_heads": pack_heads,
        "use_worklist": False,
        "scale_placement": placement,
        "scale": scale,
        "page_mode": page_mode,
        "page_class": page_class,
    }
    total_ctas = n_kv_tiles * req.h_kv * g_split * req.num_seqs
    if req.num_cus is None:
        fill = "unknown"
    else:
        fill = (
            "under"
            if total_ctas < req.num_cus
            else ("full" if total_ctas < 2 * req.num_cus else "over")
        )
    return AttnBwdPlan(
        arch=arch,
        route="unified",
        route_tag=page_class,
        perf_class=attn_bwd_unified_perf_class(req, arch, page_class, page_mode),
        fill_bucket=fill,
        specs=_freeze(
            {
                "main": spec,
                "prep": prep_spec,
                "convert": conv_spec,
                "page_zero": page_spec,
                "page_convert": page_spec,
            }
        ),
        runtime=_freeze(runtime),
        launches=tuple(launches),
        workspace_bytes=ws_bytes,
        workspace_layout=_freeze(layout),
    )


def unified_stage_params(stage: str) -> tuple[tuple[str, str], ...]:
    """Kernargs of a launch stage of the unified plan (main and page roles:
    ``rocke.attn_bwd_unified.v1``; prep / convert: ``rocke.attn_bwd.v3``)."""
    if stage in ("prep", "convert"):
        return attn_bwd_params(stage)
    return attn_bwd_unified_params(stage)


# ---------------------------------------------------------------------------
# declaration
# ---------------------------------------------------------------------------


def _classes(arch: str):
    for d in BWD_HEAD_SIZES:
        for dt in BWD_DTYPES:
            for mk in MASK_KINDS:
                for hm in HEAD_MODES:
                    for pc in PAGE_CLASS_TAGS:
                        for pm in ("direct", "atomic"):
                            yield (arch, "unified", d, dt, mk, hm, pc, pm)


_CLASS_FIELDS = (
    "arch",
    "route",
    "d",
    "dtype",
    "mask",
    "heads",
    "page_class",
    "page_mode",
)


def attn_bwd_unified_declaration(
    arch: str, *, policy: Optional[AttnBwdUnifiedPolicy] = None
) -> dict:
    """Versioned, machine-readable declaration of the unified family on ``arch``.

    A rocKE-level document (paged problems are not declared to hipDNN). The
    performance status of each class is the unified-vs-general relative gate
    (``unverified`` until measured); it is published, never a decline.
    """
    policy = shipped_unified_policy() if policy is None else policy
    declared = arch in BWD_ARCHS and arch in policy.validated_archs
    classes = []
    for pc in _classes(arch):
        row = dict(zip(_CLASS_FIELDS, pc))
        row["status"] = policy.perf_status.get(pc, "unverified")
        classes.append(row)
    return {
        "family": "attention_bwd_unified",
        "version": ATTN_BWD_UNIFIED_DECLARATION_VERSION,
        "arch": arch,
        "declared": declared,
        "abi": {
            "main": ATTN_BWD_UNIFIED_ABI,
            "page_zero": ATTN_BWD_UNIFIED_ABI,
            "page_convert": ATTN_BWD_UNIFIED_ABI,
            "prep": GENERAL_ABI,
            "convert": GENERAL_ABI,
        },
        "hipdnn": "not declared (hipDNN declines paged KV with PAGED)",
        "features": {
            "dtypes": list(BWD_DTYPES),
            "head_sizes": list(BWD_HEAD_SIZES),
            "page_sizes": f"every positive multiple of {PAGE_GRANULE} (a power "
            "of two is not required); any other page is declined PAGE_SIZE",
            "page_classes": {
                "multi": "page % block_n == 0: one block-table lookup per kv tile",
                "sub": "block_n % page == 0 for a shipped sub page "
                f"{list(SHIPPED_SUB_PAGES)}: block_n / page lookups per kv tile",
                "fallback": "the one-wave block_n = 16 tile (a page no other "
                "shipped tile serves)",
            },
            "page_modes": {
                "direct": "block_table_injective = True: dK / dV scattered into "
                "the gradient caches; deterministic",
                "atomic": "default: fp32 page workspace, atomics, page convert; "
                "correct with shared pages; non-deterministic in the last bits",
            },
            "masks": list(MASK_KINDS),
            "heads": "MHA, GQA, MQA (h_q a multiple of h_kv)",
            "query_layout": "THD [total_q, h_q, D], int32 cu_seqlens_q, any "
            "(token, head) strides",
            "kv_layout": "paged [num_blocks, page, h_kv, D], int32 seq_lens_kv and "
            "block_table, any (block, token, head) strides; dK / dV caches with "
            "the K / V strides and block table",
            "lse": "natural-log fp32 [total_q, h_q] (optionally [.., 1])",
        },
        "output_conventions": {
            "referenced_pages": "every slot of every page a live block references "
            "is written",
            "past_length_slots": "slots of the last live page past len_kv are exact "
            "zeros (unless a sequence sharing the page adds to them, atomic mode)",
            "unreferenced_pages": "untouched",
            "dq_rows": "rows outside every query span are untouched",
        },
        "determinism": {
            "dq": "non-deterministic in the last bits (fp32 atomics)",
            "dk_dv_direct": "deterministic",
            "dk_dv_atomic": "non-deterministic in the last bits",
        },
        "feature_groups": {g: g in policy.declared_groups for g in FEATURE_GROUPS},
        "reason_codes": list(REASON_CODES),
        "reason_kinds": {c: DECLINE_KINDS[c] for c in REASON_CODES},
        "not_yet_implemented_prefix": f"{NOT_YET_IMPLEMENTED}:",
        "device_data_contract": {
            "seq_lens_kv": "in [0, max_seqlen_kv]; on the device a value outside "
            "is clamped into it (host check: SEQ_LEN_RANGE)",
            "cu_seqlens_q": "non-decreasing in [0, total_q], spans at most "
            "max_seqlen_q (host check: RAGGED_RANGE)",
            "block_table": "live entries in [0, num_blocks); on the device an "
            "entry outside is never dereferenced: its K / V rows read as zeros "
            "and its dK / dV slots are not written (host check: "
            "BLOCK_TABLE_RANGE)",
        },
        "declined": [dict(x) for x in DECLINED_FEATURES],
        "perf_classes": classes,
    }
