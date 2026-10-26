# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host plan, exact predicate, workspace and declaration of the general backward.

Everything here is a pure function of request fields (and, for the policy
hooks, of in-repo tables): no device query, no allocation, no kernel build.

* :class:`AttnBwdRequest` holds everything a capability check knows about one
  SDPA backward problem (shapes, per-tensor element strides, dtypes, stats
  format, mask, lengths, ragged offsets, optional device-fill hints).
* :func:`attn_bwd_support` is the exact predicate: ordered reason codes, the
  first failing check wins (:data:`REASON_CODES`).
* :func:`attn_bwd_workspace_bytes` and :func:`attn_bwd_workspace_layout` give
  the caller workspace (256-byte packed sub-buffers, fp32).
* :func:`attn_bwd_plan` resolves the instance specs, runtime knob values,
  launch sequence and kernarg values for a supported request.
* :func:`attn_bwd_declaration` publishes the versioned, machine-readable
  declaration including the performance status per (route, class).

Shipping policy (:class:`AttnBwdPolicy`): which functional feature groups have
passed their correctness gate, the performance-status table (information only
unless a policy opts into the performance gate), instances that failed the
resource gate, the dense pack and the tuned knobs. :func:`shipped_policy` is
the kernel-level default (every declared group, nothing else); the routing
layer (``dispatch/attention_bwd/routing.py``) adds the per-arch tuned tables
and the dense pack; tests pass their own.

Every decline code has a kind (:data:`DECLINE_KINDS`, ``Verdict.kind``):
``out_of_scope`` (a permanent decision of this entry), ``not_yet_implemented``
(a feature the family can add; its detail starts with
``NOT_YET_IMPLEMENTED:``, the convention of the forward families),
``request_limit`` (a malformed request, or one past the kernels' index, grid
or format limits) or ``policy`` (the shipping policy declines).

The request is checked for types at construction: every flag must be a
``bool`` and every extent, bound and count an integer (``TypeError``
otherwise), so a truthy non-bool never selects a code path by accident.
"""

from __future__ import annotations

import contextvars
import dataclasses
import itertools
import math
import numbers
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Optional

from rocke.helpers.attention_band import I32_FAR
from rocke.helpers.spec import align_up, ceil_div

from kernels.common._attention_bwd_worklist import worklist_words
from kernels.common.attention_bwd_aux import (
    CONVERT_VECS,
    attn_bwd_aux_block_threads,
    attn_bwd_convert_grid,
    attn_bwd_prep_grid,
)
from kernels.common.attention_bwd import (
    ATTN_BWD_ABI,
    BWD_ARCHS,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    DKV_SINK_NOT_BUILT,
    DKV_SINKS,
    SPLIT_DQ_ARCHS,
    AttnBwdSpec,
    attn_bwd_arch_facts,
    attn_bwd_default_geometry,
    attn_bwd_params,
    static_unit_mults,
    validate_attn_bwd_spec,
)
from kernels.common.attention_bwd_dq import (
    DQ_TUNING_FIELDS,
    attn_bwd_dq_grid,
    dq_rows_per_head,
    dq_static_unit_mults,
    validate_attn_bwd_dq_spec,
)

__all__ = [
    "ATTN_BWD_DECLARATION_VERSION",
    "DECLARATION_NOTES",
    "DECLARED_ARCHS",
    "DECLINE_KINDS",
    "DECLINE_KIND_NAMES",
    "DEFERRAL_REGISTER",
    "DQ_KNOBS",
    "FEATURE_GROUPS",
    "REASON_CODES",
    "AttnBwdPlan",
    "AttnBwdPolicy",
    "AttnBwdRequest",
    "LaunchStep",
    "Verdict",
    "attn_bwd_declaration",
    "attn_bwd_feature_groups",
    "attn_bwd_narrow_width",
    "attn_bwd_perf_class",
    "attn_bwd_plan",
    "attn_bwd_request_verdict",
    "attn_bwd_route",
    "attn_bwd_stage_vec",
    "attn_bwd_support",
    "attn_bwd_tuning_key",
    "attn_bwd_workspace_bytes",
    "attn_bwd_workspace_layout",
    "dense_bwd_eligibility",
    "shipped_policy",
]

ATTN_BWD_DECLARATION_VERSION = 4
LOG2E = 1.4426950408889634

# gfx1201 is declared on its hardware validation run (the backward suite on a
# gfx1201 device); the forward-chain tests skip there because the forward
# probe shell has no gfx12 body (recorded in DECLARATION_NOTES).
DECLARED_ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
NOT_VALIDATED_ARCHS: tuple = ()
RDNA_ARCHS = ("gfx1151", "gfx1201")

# Scope statement of the d = 256 decline (the decline detail and the
# declaration both cite it).
REQ_SCOPE_D256_BWD = (
    "d = 256 backward is not implemented yet: the backward kernels serve head "
    "dims 32, 64 and 128 (d = 256 attention is served by the forward family "
    "only)"
)

# Kinds of decline (see the module docstring), published per reason code.
DECLINE_KIND_NAMES = ("out_of_scope", "not_yet_implemented", "request_limit", "policy")
_NYI = "not_yet_implemented"
_SCOPE = "out_of_scope"
_LIMIT = "request_limit"
_POLICY = "policy"
# The kind of every reason code of the general entry and of the unified
# (paged) entry (:mod:`kernels.common.attention_bwd_unified_plan`).
DECLINE_KINDS: Mapping[str, str] = MappingProxyType(
    {
        "ARCH_UNSUPPORTED": _SCOPE,
        "ARCH_NOT_VALIDATED": _POLICY,
        "DROPOUT": _NYI,
        "BIAS_IN_BACKWARD": _SCOPE,
        "DBIAS": _SCOPE,
        "ALIBI": _NYI,
        "PAGED": _SCOPE,
        "FP8": _NYI,
        "DETERMINISTIC": _NYI,
        "UNSUPPORTED_FEATURE": _NYI,
        "SCALE_DEVICE_TENSOR": _NYI,
        "DTYPE": _NYI,
        "OUT_DTYPE": _NYI,
        "LSE_FORMAT": _LIMIT,
        "DQK_NE_DV": _NYI,
        "HEAD_DIM_256_BWD": _NYI,
        "HEAD_DIM": _NYI,
        "GQA_RATIO": _LIMIT,
        "BAND_BOUND": _LIMIT,
        "STRIDE_FORMAT": _LIMIT,
        "STRIDE_D": _NYI,
        "STRIDE_NEGATIVE": _NYI,
        "OUTPUT_OVERLAP": _LIMIT,
        "STRIDE_ALIGN": _LIMIT,
        "INDEX_RANGE": _LIMIT,
        "GRID_LIMIT": _LIMIT,
        "SEQ_LEN_FORMAT": _LIMIT,
        "RAGGED_FORMAT": _LIMIT,
        "EMPTY": _LIMIT,
        "INSTANCE_UNAVAILABLE": _POLICY,
        "PERF_BELOW_BASELINE": _POLICY,
        "PERF_UNVERIFIED": _POLICY,
        "NOT_YET_DECLARED": _POLICY,
        # unified (paged) entry only
        "KV_CACHE_LAYOUT": _NYI,
        "PAGE_SIZE": _LIMIT,
        "BLOCK_TABLE_FORMAT": _LIMIT,
        "BLOCK_TABLE_NOT_INJECTIVE": _POLICY,
        "SINKS": _NYI,
        "SOFTCAP": _NYI,
    }
)
# Prefix of the detail of a not_yet_implemented decline.
NOT_YET_IMPLEMENTED = "NOT_YET_IMPLEMENTED"

# The feature declines of this family, published in the declaration: what is
# declined, by which reason code, whether the decline is permanent
# (``out_of_scope``) or a feature not built yet (``not_yet_implemented``), and
# the consequence for hipDNN (which falls back to its other engines).
DEFERRAL_REGISTER = (
    {
        "feature": "head dim 256 (backward)",
        "reason_code": "HEAD_DIM_256_BWD",
        "kind": _NYI,
        "decision": "declined: not implemented yet",
        "rationale": REQ_SCOPE_D256_BWD,
        "consequence": "d = 256 models are servable for inference but not "
        "trainable through this provider; their training attention stays on "
        "CK / AOTriton",
    },
    {
        "feature": "dropout",
        "reason_code": "DROPOUT",
        "kind": _NYI,
        "decision": "declined: not implemented yet",
        "rationale": "the backward kernels have no dropout mask or random "
        "number path",
        "consequence": "attention-dropout training falls back to CK / AOTriton",
    },
    {
        "feature": "bias / attn_mask tensor in the backward graph",
        "reason_code": "BIAS_IN_BACKWARD",
        "kind": _SCOPE,
        "decision": "declined: the backward contract has no bias input",
        "rationale": "the additive bias is a forward-only input of the "
        "attention contract this family implements",
        "consequence": "F.sdpa(attn_mask=...) under autograd falls back to "
        "CK / AOTriton on every target",
    },
    {
        "feature": "dBias output",
        "reason_code": "DBIAS",
        "kind": _SCOPE,
        "decision": "declined: the backward contract has no dBias output",
        "rationale": "the backward contract this family implements returns "
        "dQ, dK and dV only",
        "consequence": "graphs that need dBias train on CK / AOTriton",
    },
    {
        "feature": "ALiBi",
        "reason_code": "ALIBI",
        "kind": _NYI,
        "decision": "declined: not implemented yet",
        "rationale": "the backward kernels have no positional-bias term",
        "consequence": "ALiBi training attention falls back to CK / AOTriton",
    },
    {
        "feature": "paged KV cache (general entry)",
        "reason_code": "PAGED",
        "kind": _SCOPE,
        "decision": "declined by the general entry: paged KV is served by the "
        "rocKE-native unified entry",
        "rationale": "the general entry takes dense or THD tensors; the "
        "unified entry (attention_bwd_unified_plan) serves paged caches and "
        "is not declared to hipDNN",
        "consequence": "a paged backward through hipDNN takes CK / AOTriton",
    },
    {
        "feature": "fp8 tensors",
        "reason_code": "FP8",
        "kind": _NYI,
        "decision": "declined: not implemented yet",
        "rationale": "the backward kernels read and write fp16 / bf16 only",
        "consequence": "fp8 training attention falls back to CK / AOTriton",
    },
    {
        "feature": "deterministic backward",
        "reason_code": "DETERMINISTIC",
        "kind": _NYI,
        "decision": "declined by the shipped policy",
        "rationale": "the atomic-free backward (split dQ with direct or slice "
        "dK / dV) exists on gfx942 / gfx950 as an opt-in policy "
        "(AttnBwdPolicy.deterministic) and is not built on the RDNA targets",
        "consequence": "a deterministic backward request falls back to CK / "
        "AOTriton under the shipped policy; dQ of this provider's default "
        "plan is nondeterministic in the last bits",
    },
    {
        "feature": "block masks, sink tokens, max / sum_exp outputs and other "
        "features",
        "reason_code": "UNSUPPORTED_FEATURE",
        "kind": _NYI,
        "decision": "declined: not implemented yet",
        "rationale": "the backward kernels build none of these features",
        "consequence": "such graphs train on CK / AOTriton",
    },
)

DECLARATION_NOTES = (
    "gfx1201: declared on the hardware run of the backward suite; the "
    "forward-to-backward chain tests are skipped there (the forward probe "
    "shell has no gfx12 body); the frozen-fixture chain test runs",
)

# Ordered reason codes (first failure wins). Row 3 of the order is the group of
# out-of-scope features, checked in the listed order.
FEATURE_DECLINES = (
    "DROPOUT",
    "BIAS_IN_BACKWARD",
    "DBIAS",
    "ALIBI",
    "PAGED",
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
    "EMPTY",
    "INSTANCE_UNAVAILABLE",
    "PERF_BELOW_BASELINE",
    "PERF_UNVERIFIED",
    "NOT_YET_DECLARED",
)

# Functional feature groups, in the order their correctness gates are passed.
# A request needs every group whose feature it uses; it is declined
# NOT_YET_DECLARED while any of them is not in the policy's declared set.
FEATURE_GROUPS = (
    "base",  # fp16/bf16, d64/d128, MHA, dense strides (multiples of 8), fixed lengths, no mask
    "masks_and_stats",  # any mask, explicit scale, d32, s_q == 1
    "grouped_heads",  # GQA / MQA with h_k == h_v
    "split_kv_heads",  # h_k != h_v (atomic dK / dV)
    "padding",  # SEQ_LEN tensors with padding_mask
    "ragged",  # THD offsets
    "narrow_access",  # stage_vec = 1 (strides not multiples of 8 or alignment < 16 B)
)

# Functional groups whose correctness gate has passed on every declared arch.
SHIPPED_DECLARED_GROUPS: frozenset = frozenset(
    {"base", "masks_and_stats", "grouped_heads", "split_kv_heads", "padding", "ragged"}
    | {"narrow_access"}
)

PERF_STATUSES = ("verified", "unverified", "below_baseline")
MASK_KINDS = ("none", "causal_tl", "causal_br", "window")
HEAD_MODES = ("mha", "gqa", "hk_ne_hv")
LENGTH_RELATIONS = ("square", "unequal", "s_q_1")
LENGTH_MODES = ("fixed", "padded", "thd")
ROUTES = ("general", "dense")
SCALE_PLACEMENTS = ("fold_ds", "dq_before_atomic", "convert")
# interleave (0, n-1, 1, n-2, ...) needs the decode of a main spec with
# lb_interleave = True; the plan sets it.
LB_ORDERS = {"natural": 0, "reverse": 1, "mirror": 2, "interleave": 3}
RUNTIME_KNOBS = ("g_split", "lb_order", "xcd_chunk", "scale_placement", "use_worklist")
# Tuning values of the prep / convert kernels (aux recompiles; the main kernel
# is unchanged): rows per CTA as a multiple of the aux default ("1x" = the
# default, "2x", "4x") or an explicit row count, and the convert's elements per
# lane (attention_bwd_aux.CONVERT_VECS).
AUX_KNOBS = ("prep_rows_per_cta", "convert_rows_per_cta", "convert_vec")
AUX_ROW_MULTIPLES = {"1x": 1, "2x": 2, "4x": 4}
# Tuning values of the split dQ kernel (dq_mode = "split" only; the main
# kernel is unchanged): "dq_<field>" sets AttnBwdDqSpec.<field>, and
# "dq_xcd_chunk" is its XCD chunk (0, "gx", "2gx", "4gx" (multiples of the dQ
# grid x) or 64; with the request's num_xcds the plan builds the dQ kernel
# with xcd_remap).
DQ_KNOBS = tuple(f"dq_{f}" for f in DQ_TUNING_FIELDS) + ("dq_xcd_chunk",)
# Tuning buckets appended to the performance class in the tuned-table key
# (attn_bwd_tuning_key): kv-tile count at the class's default kN0 against
# these upper bounds, and grid fill against num_cus. "*" in a table key
# matches any bucket.
KV_TILE_BUCKETS = (1, 4, 16, 64)
FILL_BUCKETS = ("under", "full", "over", "unknown")
ANY_BUCKET = "*"
# Mirror pairing (lb_order = "mirror"): a main kernel built with lb_mirror runs
# kv tile t and then kv tile n_kv_tiles - 1 - t in one CTA; the plan sets the
# switch whenever it asks for pairing.
MIRROR_PAIRING_BUILT = True

I32_MAX = 2**31 - 1
_GRID_YZ_MAX = 65535
_WS_ALIGN = 256
_TENSORS = ("q", "k", "v", "o", "do", "dq", "dk", "dv")
_Q_SIDE = ("q", "o", "do", "dq")
_KV_SIDE = ("k", "v", "dk", "dv")
_OUTPUTS = ("dq", "dk", "dv")


# ---------------------------------------------------------------------------
# request, verdict, policy
# ---------------------------------------------------------------------------


class _FrozenStrides(dict):
    """Read-only, hashable ``{tensor: (B, H, S, D)}`` stride map.

    Keys are stored sorted and values as tuples (integer entries as ``int``),
    so two requests that differ only in insertion order or list-vs-tuple
    spelling are equal and hash alike. Malformed values (a non-integer entry,
    a non-sequence) are kept as given, never coerced, so the predicate declines
    them ``STRIDE_FORMAT``. A ``dict`` subclass (not a mapping proxy) so
    ``dataclasses.asdict``, copy and pickle work.
    """

    def __init__(self, items=()):
        pairs = dict(items)
        super().__init__(
            (str(k), _stride_value(pairs[k])) for k in sorted(pairs, key=str)
        )

    def _readonly(self, *args, **kwargs):
        raise TypeError("request strides are read-only")

    __setitem__ = __delitem__ = __ior__ = _readonly
    update = pop = popitem = clear = setdefault = _readonly

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted(self.items())))

    def __reduce__(self):
        return (type(self), (dict(self),))


def _stride_value(v):
    try:
        t = tuple(v)
    except TypeError:
        return v  # not a sequence: kept for the STRIDE_FORMAT decline
    return tuple(int(x) if _is_int(x) else x for x in t)


def _freeze(mapping):
    return MappingProxyType(dict(mapping)) if mapping is not None else None


def _freeze_strides(mapping):
    return _FrozenStrides(mapping) if mapping is not None else None


def check_field_types(obj, bools, ints, opt_ints) -> None:
    """``TypeError`` unless every field in ``bools`` is a ``bool`` (exactly:
    ``type(x) is bool``, so ``0`` / ``1`` / ``numpy.bool_`` are refused), every
    field in ``ints`` an integer that is not a bool, and every field in
    ``opt_ints`` ``None`` or such an integer."""
    for name in bools:
        v = getattr(obj, name)
        if type(v) is not bool:
            raise TypeError(f"{name} must be a bool, got {v!r}")
    for name in ints:
        v = getattr(obj, name)
        if not _is_int(v):
            raise TypeError(f"{name} must be an int, got {v!r}")
    for name in opt_ints:
        v = getattr(obj, name)
        if v is not None and not _is_int(v):
            raise TypeError(f"{name} must be None or an int, got {v!r}")


_REQ_BOOLS = (
    "bottom_right",
    "causal",
    "causal_bottom_right",
    "padding",
    "has_seq_len_q",
    "has_seq_len_kv",
    "ragged_same_table",
    "scale_is_device_tensor",
    "has_bias",
    "want_dbias",
    "deterministic",
    "alibi",
    "paged",
    "fp8",
)
_REQ_INTS = (
    "b",
    "h_q",
    "h_k",
    "h_v",
    "s_q",
    "s_kv",
    "d_qk",
    "d_v",
    "left_bound",
    "right_bound",
    "seq_len_stride",
    "ragged_offset_multiplier",
    "tensor_alignment",
)
_REQ_OPT_INTS = ("max_total_q", "max_total_kv", "num_cus", "num_xcds")


def _json_value(v):
    if isinstance(v, Mapping):
        return {str(k): _json_value(v[k]) for k in sorted(v, key=str)}
    if isinstance(v, (tuple, list)):
        return [_json_value(x) for x in v]
    return v


@dataclass(frozen=True)
class AttnBwdRequest:
    """Everything a capability check knows about one SDPA backward problem.

    Shapes: ``b`` batches (THD: number of sequences), ``s_q`` / ``s_kv`` the
    sequence dims (THD: the per-sequence maximum). ``strides`` maps each of
    ``q k v o do dq dk dv`` to its ``(B, H, S, D)`` element strides (THD:
    ``(ignored, H, token, D)``); a missing tensor is dense BHSD (dense layout)
    or ``[T, H, D]`` packed (THD). No pointers, no device state.
    """

    b: int
    h_q: int
    h_k: int
    h_v: int
    s_q: int
    s_kv: int
    d_qk: int
    d_v: int
    dtype: str = "fp16"
    k_dtype: Optional[str] = None  # None = dtype (also for v, o, do)
    v_dtype: Optional[str] = None
    o_dtype: Optional[str] = None
    do_dtype: Optional[str] = None
    dq_dtype: Optional[str] = None
    dk_dtype: Optional[str] = None
    dv_dtype: Optional[str] = None
    strides: Optional[Mapping[str, tuple[int, int, int, int]]] = None
    lse_dims: tuple = ()  # () = the canonical dims for the layout
    lse_strides: tuple = ()  # () = packed
    lse_dtype: str = "fp32"
    layout: str = "dense"  # "dense" | "thd"
    left_bound: int = -1
    right_bound: int = -1
    bottom_right: bool = False
    causal: bool = False  # deprecated booleans
    causal_bottom_right: bool = False
    padding: bool = False
    has_seq_len_q: bool = False
    has_seq_len_kv: bool = False
    seq_len_dtype: str = "int32"
    seq_len_dims: tuple = ()  # () = (B, 1, 1, 1)
    seq_len_stride: int = 1
    ragged_offset_dtype: str = "int32"
    ragged_offset_dims: tuple = ()  # () = (B + 1,)
    ragged_offset_multiplier: int = 1
    ragged_same_table: bool = True
    # THD workspace row bounds. Used only when both are given: the rows are then
    # the storage tokens and each bound must cover the end token of every
    # segment; otherwise the rows are per-sequence slots, B * s_q / B * s_kv.
    max_total_q: Optional[int] = None
    max_total_kv: Optional[int] = None
    tensor_alignment: int = 16  # bytes, smallest over all tensor base pointers
    scale: Optional[float] = None  # None = 1 / sqrt(d_qk)
    scale_is_device_tensor: bool = False
    has_bias: bool = False
    want_dbias: bool = False
    dropout: float = 0.0
    deterministic: bool = False
    alibi: bool = False
    paged: bool = False
    fp8: bool = False
    other_features: tuple = ()  # block mask, sink tokens, max / sum_exp outputs, ...
    num_cus: Optional[int] = None  # device fill hint; None = no grid-fill heuristics
    num_xcds: Optional[int] = None  # None = no XCD remap
    # Caller assertion that no physical KV page is shared between sequences.
    # Reserved for the paged (unified) entry; the general predicate ignores it.
    block_table_injective: Optional[bool] = None

    def __post_init__(self) -> None:
        check_field_types(self, _REQ_BOOLS, _REQ_INTS, _REQ_OPT_INTS)
        if self.block_table_injective is not None:
            check_field_types(self, ("block_table_injective",), (), ())
        object.__setattr__(self, "strides", _freeze_strides(self.strides))
        object.__setattr__(self, "other_features", tuple(self.other_features))
        for f in ("lse_dims", "lse_strides", "seq_len_dims", "ragged_offset_dims"):
            object.__setattr__(self, f, tuple(getattr(self, f)))

    def normalized(self) -> dict:
        """Plain JSON form (sorted stride map, lists for tuples) for hashing and logs."""
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name in ("scale", "dropout") and v is not None:
                v = float(v)
            out[f.name] = _json_value(v)
        return out

    # -- derived views -------------------------------------------------------

    @property
    def is_thd(self) -> bool:
        return self.layout == "thd"

    def tensor_heads(self, name: str) -> int:
        return {
            "q": self.h_q,
            "o": self.h_q,
            "do": self.h_q,
            "dq": self.h_q,
            "k": self.h_k,
            "dk": self.h_k,
            "v": self.h_v,
            "dv": self.h_v,
        }[name]

    def tensor_seq(self, name: str) -> int:
        return self.s_q if name in _Q_SIDE else self.s_kv

    def tensor_strides(self, name: str) -> tuple[int, int, int, int]:
        if self.strides is not None and name in self.strides:
            st = self.strides[name]
            if (
                not isinstance(st, tuple)
                or len(st) != 4
                or not all(_is_int(x) for x in st)
            ):
                # attn_bwd_support declines these STRIDE_FORMAT first
                raise ValueError(f"STRIDE_FORMAT: {name} strides {st!r}")
            return tuple(int(x) for x in st)
        # A zero extent keeps the packed strides of extent 1, so an empty tensor
        # gets well-formed strides and is declined EMPTY, not on a stride rule.
        h, s, d = (
            max(1, self.tensor_heads(name)),
            max(1, self.tensor_seq(name)),
            self.d_qk,
        )
        if self.is_thd:
            return (0, d, h * d, 1)
        return (h * s * d, s * d, d, 1)

    def used_strides(self) -> Iterator[int]:
        """The B / H / S strides of every tensor that address more than one
        index, tensor by tensor (lazy: a caller may stop at the first)."""
        for name in _TENSORS:
            yield from _used_strides(self, name)

    def resolved_scale(self) -> float:
        if self.scale is not None:
            return float(self.scale)
        return 1.0 / math.sqrt(self.d_qk) if self.d_qk > 0 else 1.0

    def resolved_band(self) -> tuple[int, int, bool]:
        """``(left, right, bottom_right)`` after the deprecated booleans."""
        if self.causal:
            return -1, 0, False
        if self.causal_bottom_right:
            return -1, 0, True
        return int(self.left_bound), int(self.right_bound), bool(self.bottom_right)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: str  # a member of REASON_CODES, or "OK"
    detail: str = ""

    @property
    def kind(self) -> str:
        """The decline's kind (:data:`DECLINE_KINDS`); ``""`` when accepted."""
        return "" if self.ok else DECLINE_KINDS[self.code]


def _no_tuning(req, arch, route, perf_class) -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True)
class AttnBwdPolicy:
    """What ships: declared feature groups, performance policy and tables.

    * ``declared_groups``: functional groups whose correctness gate passed.
    * ``perf_status``: ``{perf_class: status}`` keyed by
      :func:`attn_bwd_perf_class` (which starts with arch and route); a missing
      entry is ``unverified``. RDNA targets carry no performance claim
      (``degraded``). The status is information: the plan records it
      (``AttnBwdPlan.perf_status``) and the declaration publishes it per class;
      it never changes the route or the knobs.
    * ``perf_gate``: decline on the performance status, ``PERF_BELOW_BASELINE``
      for a ``below_baseline`` class and ``PERF_UNVERIFIED`` for an
      ``unverified`` one unless ``waive_unverified``. Off by default (the
      shipped policy serves every request the predicate accepts); RDNA targets
      are never declined on performance.
    * ``waive_unverified``: under ``perf_gate``, serve ``unverified`` classes
      (an explicit waiver of the performance requirement); no effect without
      ``perf_gate``.
    * ``instance_unavailable``: base kernel names (``AttnBwdSpec.kernel_name``
      of the unsalted base key) whose instance failed the resource gate.
    * ``dense_pack``: ``{(arch, dense_key)}`` served by the dense family.
    * ``tuning``: hook ``(req, arch, route, perf_class) -> {knob: value}``
      returning spec knob overrides and runtime values (``RUNTIME_KNOBS``).
    * ``tuned_table``: ``{tuning key: {knob: value}}`` keyed by
      :func:`attn_bwd_tuning_key` (the performance class plus the kv-tile and
      grid-fill buckets; ``ANY_BUCKET`` matches any bucket). The plan takes
      the most specific match, then lets the ``tuning`` hook override it. An
      empty table (the default) changes nothing.
    * ``not_validated_archs``: backward targets without a hardware validation
      run, declined ``ARCH_NOT_VALIDATED`` (the shipped set is empty: every
      target in ``DECLARED_ARCHS`` has one).
    * ``deterministic``: serve ``deterministic`` requests (else declined
      ``DETERMINISTIC``) on the split-dQ targets with the atomic-free
      backward: the plan sets ``dq_mode = "split"`` and, where dK / dV are
      in atomic mode, ``dkv_sink = "slices"`` (over any tuned value). Off in
      the shipped policy.
    """

    declared_groups: frozenset = SHIPPED_DECLARED_GROUPS
    waive_unverified: bool = False
    perf_status: Mapping = field(default_factory=dict)
    instance_unavailable: frozenset = frozenset()
    dense_pack: frozenset = frozenset()
    tuning: Callable[..., Mapping[str, Any]] = _no_tuning
    not_validated_archs: frozenset = frozenset(NOT_VALIDATED_ARCHS)
    tuned_table: Mapping = field(default_factory=dict)
    deterministic: bool = False
    perf_gate: bool = False

    def status_of(self, arch: str, route: str, perf_class: tuple) -> str:
        if arch in RDNA_ARCHS:
            return "degraded"
        if tuple(perf_class[:2]) != (arch, route):
            raise ValueError(f"class {perf_class} does not belong to ({arch}, {route})")
        return self.perf_status.get(tuple(perf_class), "unverified")


def shipped_policy() -> AttnBwdPolicy:
    """The shipped kernel-level policy: :class:`AttnBwdPolicy` defaults.

    Every declared feature group (:data:`SHIPPED_DECLARED_GROUPS`), no
    performance gate (the status is information only, and no per-class status
    is claimed), no resource-gate exclusion, no dense pack, no tuned knobs, the
    deterministic backward not served. It serves every request the predicate
    accepts on every declared arch, gfx942 and gfx950 included. The routing
    layer (``dispatch/attention_bwd/routing.py``) adds the per-arch tuned
    tables and the dense pack on top of it; nothing here reads them.
    """
    return AttnBwdPolicy()


# ---------------------------------------------------------------------------
# helpers over request fields
# ---------------------------------------------------------------------------


def _align(x: int) -> int:
    return align_up(x, _WS_ALIGN)


def _dkv_atomic(req: AttnBwdRequest, g_split: int = 1) -> bool:
    return req.h_k != req.h_v or g_split > 1


def _dkv_slices(
    req: AttnBwdRequest, g_split: int, dkv_sink: str = "atomic"
) -> tuple[int, int]:
    """``(n_k, n_v)``: fp32 dK / dV workspace slices per kv head (1, 1 for the
    atomic sink). A CTA owns ``G / g_split`` query heads of one K head and one
    V head, so ``(h_q / h_k) / (G / g_split)`` CTAs share each K head (one
    slice each), likewise for V."""
    _check_dkv_sink(dkv_sink)
    if dkv_sink != "slices" or not _dkv_atomic(req, g_split):
        return 1, 1
    group = _group(req, True)
    if not group or g_split < 1 or group % g_split:
        raise ValueError(f"g_split={g_split} must divide the head group {group}")
    per = group // g_split
    return (req.h_q // req.h_k) // per, (req.h_q // req.h_v) // per


def _check_dkv_sink(dkv_sink: str) -> None:
    if dkv_sink not in DKV_SINKS:
        raise ValueError(f"dkv_sink={dkv_sink!r} not in {DKV_SINKS}")


def _group(req: AttnBwdRequest, atomic: bool) -> int:
    gk = req.h_q // req.h_k if req.h_k else 0
    gv = req.h_q // req.h_v if req.h_v else 0
    return math.gcd(gk, gv) if atomic else gk


def _units(req: AttnBwdRequest, atomic: bool) -> int:
    if not atomic:
        return req.h_k
    g = _group(req, atomic)
    return req.h_q // g if g else 0


def _mask_kind(req: AttnBwdRequest) -> str:
    left, right, br = req.resolved_band()
    if left == -1 and right == -1:
        return "none"
    if left == -1 and right == 0:
        return "causal_br" if br else "causal_tl"
    return "window"


def _length_mode(req: AttnBwdRequest) -> str:
    if req.is_thd:
        return "thd"
    return "padded" if req.padding else "fixed"


def _ws_by_token(req: AttnBwdRequest) -> bool:
    """THD workspace rows indexed by storage token (both caller bounds given).

    Otherwise (every batched request, and THD without both ``max_total_*``)
    the rows of sequence ``b`` are the slot ``[b * S_max, b * S_max + len)``,
    so ``B * S_max`` rows bound every request whatever the storage holds
    before, between or after the sequences (no device query). With both
    bounds the rows are the storage tokens ``[tok0, tok0 + len)`` and the
    caller asserts ``max_total_* >= tok0 + len`` for every sequence (the end
    of the last storage segment, not the sum of the lengths).
    """
    return req.is_thd and req.max_total_q is not None and req.max_total_kv is not None


def _rows(req: AttnBwdRequest) -> tuple[int, int]:
    """Workspace rows ``(rows_q, rows_kv)``: ``B*S`` (fixed / padded, and THD by
    sequence slot) or the caller's ``max_total_*`` (THD by storage token)."""
    if _ws_by_token(req):
        return int(req.max_total_q), int(req.max_total_kv)
    return req.b * req.s_q, req.b * req.s_kv


# Enumeration budget of the exact overlap test (combinations of the outer
# index differences); a layout that needs more is declined conservatively.
_OVERLAP_BUDGET = 1 << 16


def _overlaps_sorted(dims, strides) -> bool:
    """Sufficient non-overlap test: each stride clears the extent of the
    smaller ones (``False`` proves the layout disjoint)."""
    extent = 1
    for s, n in sorted((s, n) for n, s in zip(dims, strides) if n > 1):
        if s < extent:
            return True
        extent += (n - 1) * s
    return False


def _overlaps(dims, strides) -> bool:
    """Whether two distinct indices of a non-negatively strided tensor share
    an element (exact for the backward's output layouts).

    The sorted test proves most layouts disjoint. Otherwise, with the unit
    stride dim ``u`` (the head dim; extent 1 when absent) and one more dim
    ``j`` solved in closed form, the remaining dims' index differences are
    enumerated (at most :data:`_OVERLAP_BUDGET` combinations, else ``True``,
    a conservative decline): an overlap is a
    nonzero difference vector ``delta`` (``|delta_i| < n_i``) with
    ``|sum_{i != u} delta_i s_i| < n_u``. Interleaved rows (for example
    head stride 96 and token stride 64 with ``D = 32``) are disjoint and
    pass.
    """
    if not _overlaps_sorted(dims, strides):
        return False
    live = [(int(n), int(s)) for n, s in zip(dims, strides) if n > 1]
    if any(s == 0 for _n, s in live):
        return True
    units = [k for k, (_n, s) in enumerate(live) if s == 1]
    if units:
        nu = live[units[0]][0]
        rest = [x for k, x in enumerate(live) if k != units[0]]
    else:
        nu, rest = 1, live  # an extent-1 unit dim: offsets must coincide
    if not rest:
        return False
    j = max(range(len(rest)), key=lambda k: rest[k][0])
    nj, sj = rest[j]
    others = [x for k, x in enumerate(rest) if k != j]
    combos = 1
    for n, _s in others:
        combos *= 2 * n - 1
    if combos > _OVERLAP_BUDGET:
        return True
    for delta in itertools.product(*(range(-(n - 1), n) for n, _s in others)):
        x = sum(d * s for d, (_n, s) in zip(delta, others))
        # delta_j with |x + delta_j * sj| <= nu - 1 and |delta_j| < nj
        lo = max(-(nj - 1), -((nu - 1 + x) // sj))
        hi = min(nj - 1, (nu - 1 - x) // sj)
        if not any(delta):
            # every other difference is 0: delta_j must be nonzero
            if max(lo, 1) <= hi or lo <= min(hi, -1):
                return True
        elif lo <= hi:
            return True
    return False


def _used_strides(req: AttnBwdRequest, name: str) -> list[int]:
    """B/H/S strides of ``name`` that address more than one index."""
    sb, sh, ss, _ = req.tensor_strides(name)
    out = []
    if not req.is_thd and req.b > 1:
        out.append(sb)
    if req.tensor_heads(name) > 1:
        out.append(sh)
    if req.tensor_seq(name) > 1 or req.is_thd:
        out.append(ss)
    return out


def attn_bwd_stage_vec(req) -> int:
    """8 when every used stride (``req.used_strides()``) is a multiple of 8 and
    every base is 16 B aligned, else 1.

    The one staging-width rule of the general request and the unified request
    (:class:`kernels.common.attention_bwd_unified_plan.AttnBwdUnifiedRequest`).
    """
    if req.tensor_alignment < 16:
        return 1
    return 1 if any(s % 8 for s in req.used_strides()) else 8


def attn_bwd_narrow_width(req: AttnBwdRequest) -> int:
    """Widest user-tensor access (elements: 4, 2 or 1) a ``stage_vec = 1``
    request allows: every used B/H/S stride a multiple of it and every base
    aligned to its byte size (the ``narrow_vec`` cap's ceiling)."""
    for w in (4, 2):
        if req.tensor_alignment < 2 * w:
            continue
        if all(s % w == 0 for s in req.used_strides()):
            return w
    return 1


def _rows_per_cta(arch: str, d: int) -> int:
    # The prep / convert default: one block of max(wave, 64) threads, D / 8
    # lanes per row (see attention_bwd_aux.attn_bwd_aux_rows_per_cta).
    wave = attn_bwd_arch_facts(arch).wave_size
    return max(1, max(wave, 64) // max(1, d // 8))


def _aux_rows(value, knob: str, arch: str, d: int) -> Optional[int]:
    """Aux ``rows_per_cta`` of a tuning value: ``None`` (the default rows) for
    ``"1x"`` or an explicit count equal to the default, else the row count.

    ``d`` is the head size in units of 8-element lanes (``D * 8 / convert_vec``
    for the convert), so the multiples scale the kernel's own default. The aux
    module checks the block (whole waves, at most 1024 threads).
    """
    base = _rows_per_cta(arch, d)
    if isinstance(value, str):
        if value not in AUX_ROW_MULTIPLES:
            raise ValueError(f"{knob}={value!r} not in {tuple(AUX_ROW_MULTIPLES)}")
        rows = base * AUX_ROW_MULTIPLES[value]
    elif _is_int(value) and int(value) > 0:
        rows = int(value)
    else:
        raise ValueError(f"{knob}={value!r} is not a multiple name or a row count")
    return None if rows == base else rows


def attn_bwd_feature_groups(req: AttnBwdRequest) -> frozenset:
    """Functional feature groups ``req`` needs (see :data:`FEATURE_GROUPS`)."""
    groups = {"base"}
    left, right, _ = req.resolved_band()
    default_scale = req.scale is None or (
        req.d_qk > 0 and float(req.scale) == 1.0 / math.sqrt(req.d_qk)
    )
    if left != -1 or right != -1 or not default_scale or req.d_qk == 32 or req.s_q == 1:
        groups.add("masks_and_stats")
    if req.h_k != req.h_v:
        groups.add("split_kv_heads")
    elif req.h_k != req.h_q:
        groups.add("grouped_heads")
    if req.padding:
        groups.add("padding")
    if req.is_thd:
        groups.add("ragged")
    if attn_bwd_stage_vec(req) == 1:
        groups.add("narrow_access")
    return frozenset(groups)


# ---------------------------------------------------------------------------
# routing and performance classes
# ---------------------------------------------------------------------------


def _dense_layout_of(req: AttnBwdRequest, names) -> Optional[str]:
    kinds = set()
    for n in names:
        h, s, d = req.tensor_heads(n), req.tensor_seq(n), req.d_qk
        st = req.tensor_strides(n)
        if st == (h * s * d, s * d, d, 1):
            kinds.add("bhsd")
        elif st == (s * h * d, d, h * d, 1):
            kinds.add("bshd")
        else:
            return None
    return kinds.pop() if len(kinds) == 1 else None


def dense_bwd_eligibility(req: AttnBwdRequest, arch: str) -> tuple[bool, str]:
    """Dense-route eligibility (routing only, never a decline) and its tag.

    Returns ``(eligible, tag)``; ``tag`` is ``"dense_eligible"`` or the internal
    routing tag naming the first failing condition.
    """
    if arch not in ("gfx942", "gfx950"):
        return False, "dense_ineligible_arch"
    q_layout = _dense_layout_of(req, _Q_SIDE)
    kv_layout = _dense_layout_of(req, _KV_SIDE)
    # The dense kernel bakes one layout for every tensor, so a Q side and a
    # K/V side in different dense layouts is not dense-eligible.
    if (
        req.is_thd
        or q_layout is None
        or kv_layout is None
        or q_layout != kv_layout
        or req.tensor_alignment < 16
    ):
        return False, "dense_ineligible_layout"
    if req.padding or req.has_seq_len_q or req.has_seq_len_kv:
        return False, "dense_ineligible_lengths"
    if req.h_k != req.h_v:
        return False, "dense_ineligible_heads"
    if req.d_qk not in (64, 128):
        return False, "dense_ineligible_head_size"
    return True, "dense_eligible"


def _dense_key(req: AttnBwdRequest) -> tuple:
    left, right, br = req.resolved_band()
    return (
        req.s_q,
        req.s_kv,
        req.h_q,
        req.h_k,
        req.d_qk,
        req.dtype,
        _dense_layout_of(req, _Q_SIDE),
        _mask_kind(req),
        (left, right, br),
    )


def attn_bwd_route(
    req: AttnBwdRequest, arch: str, *, policy: Optional[AttnBwdPolicy] = None
) -> tuple[str, str]:
    """``(route, tag)``: ``"dense"`` iff eligible and its key is in the pack."""
    policy = shipped_policy() if policy is None else policy
    ok, tag = dense_bwd_eligibility(req, arch)
    if not ok:
        return "general", tag
    if (arch, _dense_key(req)) not in policy.dense_pack:
        return "general", "dense_ineligible_no_tuned_config"
    return "dense", tag


def attn_bwd_perf_class(
    req: AttnBwdRequest, arch: str, route: str = "general"
) -> tuple:
    """Performance class of ``req`` on ``route`` (request fields only).

    General route: ``(arch, route, d, dtype, mask kind, head mode, length
    relation, length mode, stage_vec)``. Dense route: the exact dense key.
    """
    if route == "dense":
        return (arch, route) + _dense_key(req)
    if req.h_k != req.h_v:
        head = "hk_ne_hv"
    elif req.h_k != req.h_q:
        head = "gqa"
    else:
        head = "mha"
    if req.s_q == 1:
        rel = "s_q_1"
    elif req.s_q == req.s_kv:
        rel = "square"
    else:
        rel = "unequal"
    return (
        arch,
        route,
        req.d_qk,
        req.dtype,
        _mask_kind(req),
        head,
        rel,
        _length_mode(req),
        attn_bwd_stage_vec(req),
    )


def _kv_tile_bucket(n_tiles: int) -> str:
    for bound in KV_TILE_BUCKETS:
        if n_tiles <= bound:
            return f"t{bound}"
    return f"t>{KV_TILE_BUCKETS[-1]}"


def attn_bwd_tuning_key(
    req: AttnBwdRequest, arch: str, route: str = "general"
) -> tuple:
    """Tuned-table key of ``req``: the performance class plus two buckets.

    ``(*attn_bwd_perf_class(req, arch, route), kv-tile bucket, fill bucket)``.
    Both buckets use request fields only, so a table cannot depend on its own
    choice: the kv-tile count is ``ceil(s_kv / kN0)`` at the default kN0 of
    the class (:func:`attn_bwd_default_geometry`), the bucket the smallest
    bound of ``KV_TILE_BUCKETS`` that holds it (``t1``, ``t4``, ``t16``,
    ``t64``, else ``t>64``); the fill bucket compares the default grid
    ``B * units * kv tiles`` (no head split) with ``num_cus`` (``under``,
    ``full``, ``over``; ``unknown`` without ``num_cus``). For a supported
    request only.
    """
    pclass = attn_bwd_perf_class(req, arch, route)
    kn = attn_bwd_default_geometry(
        arch,
        req.d_qk,
        req.dtype,
        attn_bwd_stage_vec(req),
        "thd" if req.is_thd else "batched",
    )["block_n"]
    n_tiles = ceil_div(req.s_kv, kn)
    grid = req.b * _units(req, _dkv_atomic(req)) * n_tiles
    return pclass + (_kv_tile_bucket(n_tiles), _fill_bucket(req, grid))


def _table_knobs(table: Mapping, key: tuple) -> dict:
    """The most specific entry of a tuned table for ``key`` (``{}`` if none):
    exact, then the fill bucket as ``ANY_BUCKET``, then the kv-tile bucket,
    then both."""
    if not table:
        return {}
    base, kv, fill = key[:-2], key[-2], key[-1]
    for k in ((kv, fill), (kv, ANY_BUCKET), (ANY_BUCKET, fill), (ANY_BUCKET,) * 2):
        hit = table.get(base + k)
        if hit is not None:
            return dict(hit)
    return {}


# ---------------------------------------------------------------------------
# predicate
# ---------------------------------------------------------------------------


def decline_verdict(code: str, detail: str, codes: tuple) -> Verdict:
    """The decline ``code`` (a member of ``codes``), its detail prefixed with
    ``NOT_YET_IMPLEMENTED:`` when the code's kind is ``not_yet_implemented``."""
    if code not in codes:
        raise AssertionError(f"unknown reason code {code}")
    if DECLINE_KINDS[code] == _NYI:
        detail = f"{NOT_YET_IMPLEMENTED}: {detail}"
    return Verdict(False, code, detail)


def _decline(code: str, detail: str) -> Verdict:
    return decline_verdict(code, detail, REASON_CODES)


def _feature_decline(
    req: AttnBwdRequest, *, deterministic_ok: bool = False
) -> Optional[Verdict]:
    """First out-of-scope feature of ``req``; ``deterministic_ok`` admits a
    deterministic request (the policy serves it atomic-free)."""
    checks = (
        ("DROPOUT", req.dropout != 0.0, f"dropout={req.dropout}"),
        (
            "BIAS_IN_BACKWARD",
            req.has_bias,
            "bias / attn_mask tensor in the backward graph",
        ),
        ("DBIAS", req.want_dbias, "dBias requested"),
        ("ALIBI", req.alibi, "ALiBi requested"),
        (
            "PAGED",
            req.paged,
            "paged KV is served only by the rocKE-native unified entry",
        ),
        ("FP8", req.fp8, "fp8 tensors requested"),
        (
            "DETERMINISTIC",
            req.deterministic and not deterministic_ok,
            "deterministic backward requested",
        ),
        (
            "UNSUPPORTED_FEATURE",
            bool(req.other_features),
            f"features {req.other_features}",
        ),
    )
    for code, hit, detail in checks:
        if hit:
            return _decline(code, detail)
    return None


def _is_int(x) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _stride_format(req: AttnBwdRequest) -> Optional[str]:
    """Why a caller-supplied tensor stride tuple is malformed, else ``None``.

    Every tensor takes four integer strides ``(B, H, S, D)`` (THD: the B entry
    is ignored); any other rank or a non-integer entry is declined, never
    truncated or indexed past its end.
    """
    for n in _TENSORS:
        if req.strides is None or n not in req.strides:
            continue
        st = req.strides[n]
        try:
            st = tuple(st)
        except TypeError:
            return f"{n} strides {st!r} are not a sequence"
        if len(st) != 4:
            return f"{n} strides {st} have rank {len(st)}, need 4 (B, H, S, D)"
        if not all(_is_int(x) for x in st):
            return f"{n} strides {st} have a non-integer entry"
    return None


def _lse_layout(req: AttnBwdRequest) -> tuple[Optional[str], tuple[int, int, int]]:
    """``(why, (l_b, l_h, l_t))``: the LSE form of ``req`` and the element
    strides the prep reads it with; ``why`` is ``None`` for an accepted form,
    else the reason it is declined (the strides are then ``(0, 0, 0)``).

    Accepted forms (the stride tuple has the rank of the dims; ``()`` dims are
    the canonical dims of the layout, ``()`` strides their packed strides):

    * dense: ``[B, H_q, S_q, 1]``, strides ``(b, h, t, 1)``;
    * THD: ``[T_q, H_q, 1]``, strides ``(t, h, 1)`` (packed: ``t = H_q``,
      ``h = 1``); or the batched shape ``[B, H_q, S_q, 1]`` with explicit
      ``(b, h, t, 1)`` strides whose B stride is unused (tokens are located
      through the ragged offsets, so the token stride is the S stride).

    The stride of a trailing extent-1 dim never moves an address and may be
    any non-negative value. Everything else (another rank, other dims, a
    negative or non-integer entry, a THD batched shape without strides) is
    declined, never replaced by a default.
    """
    none = (0, 0, 0)
    dims, st = req.lse_dims, req.lse_strides
    if req.lse_dtype != "fp32":
        return f"LSE dtype {req.lse_dtype}, not fp32", none
    if not all(_is_int(x) for x in dims + st):
        return f"LSE dims {dims} / strides {st} hold a non-integer entry", none
    if any(s < 0 for s in st):
        return f"LSE strides {st} hold a negative stride", none
    batched = (req.b, req.h_q, req.s_q, 1)
    if not req.is_thd:
        if dims not in ((), batched):
            return f"dense LSE dims {dims}, want () or {batched}", none
        if len(st) not in (0, 4):
            return (
                f"dense LSE strides {st} of rank {len(st)}, want () or the "
                "rank-4 (b, h, t, 1) strides of [B, H_q, S_q, 1]",
                none,
            )
        if not st:
            return None, (req.h_q * req.s_q, req.s_q, 1)
        return None, (int(st[0]), int(st[1]), int(st[2]))
    if dims == batched:
        if len(st) != 4:
            return (
                f"THD LSE in the batched shape {dims} needs its rank-4 "
                f"(b, h, t, 1) strides, got {st} (the packed default is the "
                "[T_q, H_q, 1] form only)",
                none,
            )
        return None, (0, int(st[1]), int(st[2]))
    if dims != () and not (len(dims) == 3 and dims[1:] == (req.h_q, 1)):
        return (
            f"THD LSE dims {dims}, want (), (T_q, {req.h_q}, 1) or {batched}",
            none,
        )
    if dims and dims[0] < 0:
        return f"THD LSE dims {dims} hold a negative token count", none
    if len(st) not in (0, 3):
        return (
            f"THD LSE strides {st} of rank {len(st)}, want () or the rank-3 "
            "(t, h, 1) strides of [T_q, H_q, 1]",
            none,
        )
    if not st:
        return None, (0, 1, req.h_q)
    return None, (0, int(st[1]), int(st[0]))


def _band_ok(req: AttnBwdRequest) -> tuple[bool, str]:
    for name, v in (("left_bound", req.left_bound), ("right_bound", req.right_bound)):
        if v < -1 or v >= I32_FAR:
            return False, f"{name}={v} outside [-1, 2^30)"
    if req.causal and req.causal_bottom_right:
        return False, "both deprecated causal booleans set"
    if (req.causal or req.causal_bottom_right) and (
        req.left_bound != -1 or req.right_bound != -1
    ):
        return False, "a causal boolean together with explicit bounds"
    return True, ""


def _lse_kernarg_strides(req: AttnBwdRequest) -> tuple[int, int, int]:
    """``(l_b, l_h, l_t)`` element strides of the LSE as the prep reads them
    (see :func:`_lse_layout`); raises for a form the predicate declines."""
    why, strides = _lse_layout(req)
    if why is not None:
        raise ValueError(f"LSE_FORMAT: {why}")
    return strides


_I64_LIMIT = 2**63


def _wide_range(req: AttnBwdRequest) -> Optional[str]:
    """The i64 kernargs and the i64 byte offsets the kernels form.

    Batch and head strides are i64 kernargs: each must fit, and the largest
    element offset of each tensor (dense: every index; THD: the head index,
    the token base being device data) times the element size must fit the
    kernels' i64 byte offset.
    """
    for n in _TENSORS:
        sb, sh, st, _sd = req.tensor_strides(n)
        packed = (sh,) if req.is_thd else (sb, sh)
        if any(x >= _I64_LIMIT for x in packed):
            return f"{n}: batch / head stride does not fit the i64 kernarg"
        far = (req.tensor_heads(n) - 1) * sh + req.d_qk
        if not req.is_thd:
            far += (req.b - 1) * sb + (req.tensor_seq(n) - 1) * st
        if far * 2 >= _I64_LIMIT:
            return f"{n}: largest byte offset >= 2^63"
    lb, lh, lt = _lse_kernarg_strides(req)
    if lb >= _I64_LIMIT or lh >= _I64_LIMIT:
        return "lse: batch / head stride does not fit the i64 kernarg"
    far = (req.h_q - 1) * lh + (0 if req.is_thd else (req.b - 1) * lb + req.s_q * lt)
    if (far + 1) * 4 >= _I64_LIMIT:
        return "lse: largest byte offset >= 2^63"
    return None


def _index_range(req: AttnBwdRequest, atomic_kv: bool) -> Optional[str]:
    d = req.d_qk
    for n in _TENSORS:
        s_max = req.tensor_seq(n)
        ss = abs(req.tensor_strides(n)[2])
        if s_max * ss + d >= 2**31:
            return f"{n}: S_max * stride_s + D >= 2^31"
    # the LSE token stride is an i32 kernarg and indexes rows in i32 in-slab
    if req.s_q * abs(_lse_kernarg_strides(req)[2]) + 1 >= 2**31:
        return "lse: S_q * stride_s + 1 >= 2^31"
    why = _wide_range(req)
    if why:
        return why
    rows_q, rows_kv = _rows(req)
    if rows_q >= 2**31:
        return f"workspace rows_q={rows_q} >= 2^31"
    if atomic_kv and rows_kv >= 2**31:
        return f"workspace rows_kv={rows_kv} >= 2^31"
    for s_max in (req.s_q, req.s_kv):
        if s_max * d >= 2**31:
            return "S_max * D >= 2^31 (in-slab workspace index)"
        if s_max >= I32_FAR:
            return "S_max >= 2^30"
    return None


def _grid_limit(req: AttnBwdRequest, arch: str, atomic_kv: bool) -> Optional[str]:
    if req.b > _GRID_YZ_MAX:
        return f"B={req.b} > 65535"
    units = _units(req, atomic_kv)
    if units > _GRID_YZ_MAX:
        return f"main units={units} > 65535"
    prep_y = req.h_q + ((req.h_k + req.h_v) if atomic_kv else 0)
    if prep_y > _GRID_YZ_MAX:
        return f"prep grid y={prep_y} > 65535"
    r = _rows_per_cta(arch, req.d_qk)
    if ceil_div(max(req.s_q, req.s_kv), r) >= 2**31:
        return "prep / convert grid x >= 2^31"
    return None


def _seq_len_ok(req: AttnBwdRequest) -> tuple[bool, str]:
    has_any = req.has_seq_len_q or req.has_seq_len_kv
    if has_any and not req.padding:
        return False, "SEQ_LEN tensors without padding_mask"
    if req.padding and not (req.has_seq_len_q and req.has_seq_len_kv):
        return False, "padding_mask without both SEQ_LEN tensors"
    if has_any:
        if req.seq_len_dtype != "int32":
            return False, f"SEQ_LEN dtype {req.seq_len_dtype}"
        if req.seq_len_dims not in ((), (req.b, 1, 1, 1)):
            return False, f"SEQ_LEN dims {req.seq_len_dims}"
        if not (0 <= req.seq_len_stride <= I32_MAX):
            return False, f"SEQ_LEN stride {req.seq_len_stride}"
    return True, ""


def _ragged_ok(req: AttnBwdRequest) -> tuple[bool, str]:
    if not req.is_thd:
        return True, ""
    if req.ragged_offset_dtype not in ("int32", "int64"):
        return False, f"offset dtype {req.ragged_offset_dtype}"
    if req.ragged_offset_dims not in ((), (req.b + 1,)):
        return False, f"offset dims {req.ragged_offset_dims}"
    if not req.ragged_same_table:
        return False, "q-side or kv-side tensors not on one offset table"
    if not (1 <= req.ragged_offset_multiplier <= I32_MAX):
        return False, f"multiplier {req.ragged_offset_multiplier} outside [1, 2^31)"
    for n in ("q", "k"):
        div = req.tensor_strides(n)[2]
        if not (1 <= div <= I32_MAX):
            return False, f"{n} token stride {div} outside [1, 2^31)"
    for n in (*_Q_SIDE, *_KV_SIDE):
        if req.tensor_strides(n)[2] < 1:
            return False, f"{n} token stride must be positive"
    # A token bound of 0 for a non-empty problem leaves ws_rows = 0, and the
    # clamped workspace row (min(row, ws_rows - 1)) would land before the buffer.
    # An empty problem is left to the EMPTY check.
    if not _has_zero_extent(req):
        for name in ("max_total_q", "max_total_kv"):
            v = getattr(req, name)
            if v is not None and v < 1:
                return False, f"{name}={v} must be at least 1"
    return True, ""


def _has_zero_extent(req: AttnBwdRequest) -> bool:
    return min(req.b, req.h_q, req.h_k, req.h_v, req.s_q, req.s_kv, req.d_qk) == 0


def _instance_key(req: AttnBwdRequest) -> str:
    """Unsalted base kernel name of the main instance serving ``req``."""
    return AttnBwdSpec(
        head_size=req.d_qk,
        dtype=req.dtype,
        seq_mode="thd" if req.is_thd else "batched",
        dkv_mode="atomic" if _dkv_atomic(req) else "direct",
        stage_vec=attn_bwd_stage_vec(req),
        mask_class="none" if _mask_kind(req) == "none" else "band",
    ).kernel_name("main")


def attn_bwd_support(
    req: AttnBwdRequest, arch: str, *, policy: Optional[AttnBwdPolicy] = None
) -> Verdict:
    """Exact host predicate; ordered reason codes, first failure wins.

    An accepted request plans. After the request checks the predicate resolves
    the plan of the route the policy gives the request (:func:`attn_bwd_plan`
    for the general route, the dense family's plan for the dense route), the
    policy's tuned knobs included, and declines ``INSTANCE_UNAVAILABLE`` when
    the routed plan does not build: a spec the family validator refuses
    (among them the recorded resource-gate failures, ``RESOURCE_REFUSED``) or
    a tuned value the request cannot take. One case goes to the dense family:
    a dense-eligible request whose general plan fails only through the
    policy's tuned knobs (the general plan without them builds) is
    configured for the dense route (the dense planner's forced route), and
    the dense family's plan decides. Under the shipped and routed policies the
    general plan uses defaults (never on the resource-gate list) or
    table entries (each checked to plan), so the routed plan is the one that
    builds.
    """
    policy = shipped_policy() if policy is None else policy
    v = _request_checks(req, arch, policy)
    if v is not None:
        return v
    if _RESOLVING.get():
        # a planner called while the predicate resolves its plan (the dense
        # family's): the request checks are its whole question
        return Verdict(True, "OK", "")
    why = _unplannable(req, arch, policy)
    if why is not None:
        return _decline("INSTANCE_UNAVAILABLE", f"no buildable instance: {why}")
    return _policy_checks(req, arch, policy)


def attn_bwd_request_verdict(
    req: AttnBwdRequest, arch: str, *, policy: Optional[AttnBwdPolicy] = None
) -> Verdict:
    """The predicate without the plan resolution: the request checks and the
    policy checks. For a planner of a forced route (one that builds its own
    family's plan whatever the routing would choose), which then owns the
    plannability of its route; every other caller asks
    :func:`attn_bwd_support`."""
    policy = shipped_policy() if policy is None else policy
    v = _request_checks(req, arch, policy)
    return v if v is not None else _policy_checks(req, arch, policy)


# Set while the predicate resolves a plan (re-entrant calls of the predicate
# from that planner run the request checks only).
_RESOLVING: contextvars.ContextVar = contextvars.ContextVar(
    "attn_bwd_resolving", default=False
)


def _unplannable(
    req: AttnBwdRequest, arch: str, policy: AttnBwdPolicy
) -> Optional[str]:
    """``None`` when the routed family's plan of ``req`` builds, else why not.

    The dense family's plan is resolved for the dense route. On the general
    route a failure declines when the general family cannot serve the request
    at all (its plan without the policy's tuned knobs fails too: a refused
    default spec, a request limit). When only the policy's tuned knobs fail
    the general plan of a dense-eligible request, those knobs configure the
    dense family (the dense planner's forced route), and the dense family's
    plan decides.
    """
    route, _tag = attn_bwd_route(req, arch, policy=policy)
    token = _RESOLVING.set(True)
    try:
        general = None
        if route == "general":
            try:
                _resolve_plan(req, arch, policy)
                return None
            except ValueError as exc:
                general = f"general route: {exc}"
            if not dense_bwd_eligibility(req, arch)[0]:
                return general
            untuned = dataclasses.replace(policy, tuning=_no_tuning, tuned_table={})
            try:
                _resolve_plan(req, arch, untuned)
            except ValueError:
                return general  # the family itself cannot serve it
        from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

        try:
            attn_dense_bwd_plan(req, arch, policy=policy)
            return None
        except ValueError as exc:
            return "; ".join(x for x in (general, f"dense route: {exc}") if x)
    finally:
        _RESOLVING.reset(token)


def _request_checks(
    req: AttnBwdRequest, arch: str, policy: AttnBwdPolicy
) -> Optional[Verdict]:
    """The predicate's checks up to the resource-gate list (``None``: passed)."""
    if arch not in BWD_ARCHS:
        return _decline("ARCH_UNSUPPORTED", f"{arch} is not a backward target")
    if arch in policy.not_validated_archs:
        return _decline("ARCH_NOT_VALIDATED", f"{arch} has no hardware validation run")
    v = _feature_decline(
        req, deterministic_ok=policy.deterministic and arch in SPLIT_DQ_ARCHS
    )
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
    if req.h_k > 0 and req.h_v > 0 and (req.h_q % req.h_k or req.h_q % req.h_v):
        return _decline(
            "GQA_RATIO", f"h_q={req.h_q} not a multiple of h_k={req.h_k}, h_v={req.h_v}"
        )
    ok, detail = _band_ok(req)
    if not ok:
        return _decline("BAND_BOUND", detail)
    why = _stride_format(req)
    if why is not None:
        return _decline("STRIDE_FORMAT", why)
    for n in _TENSORS:
        if req.tensor_strides(n)[3] != 1:
            return _decline("STRIDE_D", f"{n} D stride {req.tensor_strides(n)[3]}")
    for n in _TENSORS:
        if any(s < 0 for s in req.tensor_strides(n)):
            return _decline("STRIDE_NEGATIVE", f"{n} strides {req.tensor_strides(n)}")
    rows_q, rows_kv = _rows(req)
    for n in _OUTPUTS:
        sb, sh, ss, sd = req.tensor_strides(n)
        h = req.tensor_heads(n)
        if req.is_thd:
            toks = rows_q if n == "dq" else rows_kv
            dims, st = (toks, h, req.d_qk), (ss, sh, sd)
        else:
            dims, st = (req.b, h, req.tensor_seq(n), req.d_qk), (sb, sh, ss, sd)
        if min(dims) == 0:
            continue  # an empty tensor has no elements that could overlap
        if _overlaps(dims, st):
            return _decline(
                "OUTPUT_OVERLAP", f"{n} strides {req.tensor_strides(n)} self-overlap"
            )
    if req.tensor_alignment < 2:
        return _decline(
            "STRIDE_ALIGN", f"base alignment {req.tensor_alignment} B < element size"
        )
    atomic_kv = _dkv_atomic(req)
    why = _index_range(req, atomic_kv)
    if why:
        return _decline("INDEX_RANGE", why)
    why = _grid_limit(req, arch, atomic_kv)
    if why:
        return _decline("GRID_LIMIT", why)
    ok, detail = _seq_len_ok(req)
    if not ok:
        return _decline("SEQ_LEN_FORMAT", detail)
    ok, detail = _ragged_ok(req)
    if not ok:
        return _decline("RAGGED_FORMAT", detail)
    if _has_zero_extent(req):
        return _decline("EMPTY", "a batch, head, sequence or head-dim extent is 0")
    key = _instance_key(req)
    if key in policy.instance_unavailable:
        return _decline("INSTANCE_UNAVAILABLE", f"{key} failed the resource gate")
    return None


def _policy_checks(req: AttnBwdRequest, arch: str, policy: AttnBwdPolicy) -> Verdict:
    """The predicate's checks after the plan resolved: performance gate and
    declared feature groups."""
    if policy.perf_gate:
        route, _ = attn_bwd_route(req, arch, policy=policy)
        pclass = attn_bwd_perf_class(req, arch, route)
        status = policy.status_of(arch, route, pclass)
        if status == "below_baseline":
            return _decline("PERF_BELOW_BASELINE", f"class {pclass} below baseline")
        if status == "unverified" and not policy.waive_unverified:
            return _decline("PERF_UNVERIFIED", f"class {pclass} has no measured cell")
    missing = sorted(
        attn_bwd_feature_groups(req) - set(policy.declared_groups),
        key=FEATURE_GROUPS.index,
    )
    if missing:
        return _decline(
            "NOT_YET_DECLARED", f"feature groups not declared yet: {missing}"
        )
    return Verdict(True, "OK", "")


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------


def attn_bwd_workspace_layout(
    req: AttnBwdRequest,
    *,
    g_split: int = 1,
    dq_mode: str = "atomic",
    worklist: bool = False,
    block_n: int = 16,
    dkv_sink: str = "atomic",
) -> dict:
    """``{name: (byte offset, bytes)}`` of every workspace sub-buffer, packed.

    Order and sizes (``A(x)`` rounds up to 256 bytes; every array fp32)::

        WS_DQ    A(4 * rows_q * h_q * D)        dq_mode == "atomic" only
        WS_LSE2  A(4 * rows_q * h_q)
        WS_DSUM  A(4 * rows_q * h_q)
        WS_DK    A(4 * rows_kv * h_k * D * n_k) h_k != h_v or g_split > 1
        WS_DV    A(4 * rows_kv * h_v * D * n_v) same condition
        WORKLIST A(8 * (1 + B * ceil(S_kv / kN0)))   work list only (i32 words)

    ``n_k = n_v = 1`` with the atomic sink; with ``dkv_sink = "slices"`` they
    are the slice counts ``n_k = (h_q / h_k) / (G / g_split)`` and ``n_v =
    (h_q / h_v) / (G / g_split)`` (one slice per CTA sharing a kv head; ``G``
    the atomic head group), so each array is ``4 * rows_kv * D * (h_q / (G /
    g_split))`` bytes.

    ``rows_q, rows_kv`` are ``B * S`` (fixed and padded: the tensor dims) or,
    for THD, the caller's ``max_total_q`` / ``max_total_kv`` when both are
    supplied (rows indexed by storage token, kernarg ``ws_seg = 0``), else
    ``B * S_max`` (rows indexed by sequence slot ``b * S_max + i``, kernarg
    ``ws_seg = 1``). Either way the reported size is an upper bound for every
    request the predicate admits (with the caller bounds: under the caller's
    assertion that they cover the end token of every segment). The work list
    holds a two-word header and two words per item; ``B * ceil(S_kv / kN0)``
    (the main grid's CTAs per head unit, ``block_n = kN0``) bounds the live
    items for every offset table, because each sequence length is clamped to
    ``S_kv``, while the sum of the offset spans may exceed a caller's
    ``max_total_kv``.
    """
    d = req.d_qk
    rows_q, rows_kv = _rows(req)
    sizes = []
    if dq_mode == "atomic":
        sizes.append(("WS_DQ", 4 * rows_q * req.h_q * d))
    sizes.append(("WS_LSE2", 4 * rows_q * req.h_q))
    sizes.append(("WS_DSUM", 4 * rows_q * req.h_q))
    if _dkv_atomic(req, g_split):
        n_k, n_v = _dkv_slices(req, g_split, dkv_sink)
        sizes.append(("WS_DK", 4 * rows_kv * req.h_k * d * n_k))
        sizes.append(("WS_DV", 4 * rows_kv * req.h_v * d * n_v))
    if worklist:
        words = worklist_words(req.b, ceil_div(req.s_kv, block_n))
        sizes.append(("WORKLIST", 4 * words))
    out, off = {}, 0
    for name, nbytes in sizes:
        out[name] = (off, nbytes)
        off += _align(nbytes)
    return out


def attn_bwd_workspace_bytes(
    req: AttnBwdRequest,
    *,
    g_split: int = 1,
    dq_mode: str = "atomic",
    worklist: bool = False,
    block_n: int = 16,
    dkv_sink: str = "atomic",
) -> int:
    """Total caller workspace in bytes (sum of the 256-byte aligned sub-buffers)."""
    layout = attn_bwd_workspace_layout(
        req,
        g_split=g_split,
        dq_mode=dq_mode,
        worklist=worklist,
        block_n=block_n,
        dkv_sink=dkv_sink,
    )
    return sum(_align(n) for _, n in layout.values())


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LaunchStep:
    """One kernel launch: stage, kernel name, grid, block, kernarg values.

    ``scalars`` maps every non-pointer kernarg of the stage to its value;
    ``pointers`` maps every pointer kernarg to its role (a tensor name, a
    workspace sub-buffer name, or ``None`` for an unused, zero pointer).
    """

    stage: str
    kernel: str
    grid: tuple[int, int, int]
    block: int
    scalars: Mapping[str, Any]
    pointers: Mapping[str, Optional[str]]


@dataclass(frozen=True)
class AttnBwdPlan:
    arch: str
    route: str
    route_tag: str
    perf_class: tuple
    fill_bucket: str
    specs: Mapping[str, Any]  # "main" -> resolved AttnBwdSpec
    runtime: Mapping[str, Any]
    launches: tuple[LaunchStep, ...]
    workspace_bytes: int
    workspace_layout: Mapping[str, tuple[int, int]]
    tuning_key: tuple = ()  # attn_bwd_tuning_key of the request
    # the policy's performance status of perf_class (information only: it
    # never changes the plan); "" where the route records none (unified)
    perf_status: str = ""

    @property
    def atomic_free(self) -> bool:
        """No launch of the plan issues a data atomic: dQ from the split dQ
        kernel, and dK / dV direct or through workspace slices. Every output
        is then a fixed-order fp32 sum, bit for bit the same run to run."""
        main = self.specs["main"]
        if main.dq_mode != "split":
            return False
        return main.dkv_mode == "direct" or main.dkv_sink == "slices"

    @property
    def deterministic(self) -> bool:
        """Bitwise reproducible run to run (see :attr:`atomic_free`)."""
        return self.atomic_free


def _check_worklist(req: AttnBwdRequest, lb_name: str, n_kv_tiles: int) -> None:
    """Refuse a work list the kernels cannot serve (``ValueError``)."""
    if not req.is_thd:
        raise ValueError(
            "use_worklist serves THD requests: a batched CTA past len_kv still "
            "stores the zero rows of its tile, so every grid CTA runs"
        )
    if lb_name != "natural":
        raise ValueError(
            f"use_worklist fixes the CTA order; lb_order={lb_name!r} needs the "
            "grid decode (use lb_order='natural')"
        )
    if worklist_words(req.b, n_kv_tiles) > I32_MAX:
        raise ValueError(
            "use_worklist: the list of B * ceil(S_kv / kN0) items is not "
            "indexable in i32 words"
        )


def _fill_bucket(req: AttnBwdRequest, total_ctas: int) -> str:
    if req.num_cus is None:
        return "unknown"
    if total_ctas < req.num_cus:
        return "under"
    if total_ctas < 2 * req.num_cus:
        return "full"
    return "over"


def _scale_mults(placement: str, scale: float, atomic_kv: bool) -> dict:
    if placement == "fold_ds":
        return {
            "ds_mult": scale,
            "dk_mult": 1.0,
            "dq_mult": 1.0,
            "conv_dq": 1.0,
            "conv_dk": 1.0,
        }
    if placement == "convert":
        return {
            "ds_mult": 1.0,
            "dk_mult": 1.0 if atomic_kv else scale,
            "dq_mult": 1.0,
            "conv_dq": scale,
            "conv_dk": scale if atomic_kv else 1.0,
        }
    if placement == "dq_before_atomic":
        return {
            "ds_mult": 1.0,
            "dk_mult": 1.0 if atomic_kv else scale,
            "dq_mult": scale,
            "conv_dq": 1.0,
            "conv_dk": scale if atomic_kv else 1.0,
        }
    raise ValueError(f"scale_placement={placement!r} not in {SCALE_PLACEMENTS}")


def attn_bwd_plan(
    req: AttnBwdRequest, arch: str, *, policy: Optional[AttnBwdPolicy] = None
) -> AttnBwdPlan:
    """Resolve instances, runtime knobs, launches and workspace for ``req``.

    Raises ``ValueError`` (with the reason code) when the predicate declines,
    in the predicate's order: the request checks, then the plan resolution
    (``INSTANCE_UNAVAILABLE`` with the reason the plan failed), then the
    policy checks. Only the general route is planned here; a dense route is
    reported in the plan's ``route`` field by the routing layer that owns the
    dense family.
    """
    policy = shipped_policy() if policy is None else policy
    verdict = _request_checks(req, arch, policy)
    if verdict is not None:
        raise ValueError(f"{verdict.code}: {verdict.detail}")
    try:
        plan = _resolve_plan(req, arch, policy)
    except ValueError as exc:
        raise ValueError(f"INSTANCE_UNAVAILABLE: no buildable instance: {exc}") from exc
    verdict = _policy_checks(req, arch, policy)
    if not verdict.ok:
        raise ValueError(f"{verdict.code}: {verdict.detail}")
    return plan


def _resolve_plan(req: AttnBwdRequest, arch: str, policy: AttnBwdPolicy) -> AttnBwdPlan:
    """The plan of a request that passed the request checks (``ValueError``
    when no instance serves it under ``policy``)."""
    route, tag = attn_bwd_route(req, arch, policy=policy)
    pclass = attn_bwd_perf_class(req, arch, route)
    tkey = attn_bwd_tuning_key(req, arch, route)
    tuned = _table_knobs(policy.tuned_table, tkey)
    tuned.update(policy.tuning(req, arch, route, pclass) or {})
    spec_fields = set(AttnBwdSpec.__dataclass_fields__) - {
        "head_size",
        "dtype",
        "seq_mode",
        "dkv_mode",
        "stage_vec",
        "mask_class",
        "name",
    }
    unknown = (
        set(tuned) - spec_fields - set(RUNTIME_KNOBS) - set(AUX_KNOBS) - set(DQ_KNOBS)
    )
    if unknown:
        raise ValueError(f"tuning hook returned unknown knobs {sorted(unknown)}")
    runtime_in = {k: tuned.pop(k) for k in RUNTIME_KNOBS if k in tuned}
    aux_in = {k: tuned.pop(k) for k in AUX_KNOBS if k in tuned}
    dq_in = {k: tuned.pop(k) for k in DQ_KNOBS if k in tuned}
    placement = runtime_in.get("scale_placement", "fold_ds")
    if placement not in SCALE_PLACEMENTS:
        raise ValueError(f"scale_placement={placement!r} not in {SCALE_PLACEMENTS}")
    # The main kernel is compiled for the placement (its unit multipliers are
    # not applied) unless the hook pins "runtime"; another pin is refused.
    static_scale = tuned.setdefault("static_scale", placement)
    if static_scale not in (placement, "runtime"):
        raise ValueError(
            f"static_scale={static_scale!r} does not match "
            f"scale_placement={placement!r}"
        )

    g_split = int(runtime_in.get("g_split", 1))
    atomic_kv = _dkv_atomic(req, g_split)
    group = _group(req, atomic_kv)
    if g_split < 1 or group % g_split:
        raise ValueError(f"g_split={g_split} must divide the head group {group}")
    if req.deterministic:
        # the atomic-free backward (the predicate admitted the request under
        # policy.deterministic): split dQ, and slice dK / dV in atomic mode.
        # The determinism request wins over a tuned value (a tuned table keys
        # on performance classes, not on the flag), so the plan never fails
        # on a request the predicate admitted.
        tuned["dq_mode"] = "split"
        if atomic_kv:
            tuned["dkv_sink"] = "slices"
    if atomic_kv:
        sink = tuned.get("dkv_sink", "atomic")
        _check_dkv_sink(sink)
        if sink in DKV_SINK_NOT_BUILT:
            raise ValueError(
                f"dkv_sink={sink!r} is not built: {DKV_SINK_NOT_BUILT[sink]}"
            )
    stage_vec = attn_bwd_stage_vec(req)
    if "narrow_vec" in tuned:
        # a cap: the narrow class takes the widest access its strides and
        # bases allow up to it (the wide class has no narrow access at all)
        cap = tuned.pop("narrow_vec")
        if cap not in (1, 2, 4):
            raise ValueError(f"narrow_vec={cap!r} not in (1, 2, 4)")
        if stage_vec == 1:
            width = min(int(cap), attn_bwd_narrow_width(req))
            if width != 1:
                tuned["narrow_vec"] = width
    mask_kind = _mask_kind(req)
    seq_mode = "thd" if req.is_thd else "batched"
    dma_note = None
    if tuned.get("global_path") == "dma" and stage_vec != 8:
        # The LDS DMA copies 16-byte vectors: narrow-access requests stage
        # through registers (the ring depth is kept).
        tuned["global_path"], dma_note = "vgpr", "stage_vec=1"
    spec = AttnBwdSpec(
        head_size=req.d_qk,
        dtype=req.dtype,
        seq_mode=seq_mode,
        dkv_mode="atomic" if atomic_kv else "direct",
        stage_vec=stage_vec,
        mask_class="none" if mask_kind == "none" else "band",
        **tuned,
    )
    spec = validate_attn_bwd_spec(spec, arch)
    if "head_pack" not in tuned and group > 1 and req.s_q * group <= spec.block_m:
        # Short query spans (s_q = 1 and friends): the group's query heads share
        # the M rows of one q tile instead of leaving them mostly masked. The
        # DMA ring stages one head per step, so packing moves the staging to
        # the register path (the ring depth is kept).
        packed = {"head_pack": True}
        if spec.global_path == "dma":
            packed["global_path"], dma_note = "vgpr", "head_pack"
        spec = validate_attn_bwd_spec(dataclasses.replace(spec, **packed), arch)
    km, kn = spec.block_m, spec.block_n
    if spec.head_pack and req.s_q * group > km:
        raise ValueError("head_pack needs G * s_q <= block_m")
    pack_heads = group if spec.head_pack else 0

    facts = attn_bwd_arch_facts(arch)
    n_kv_tiles = ceil_div(req.s_kv, kn)
    units = _units(req, atomic_kv)
    if units * g_split > _GRID_YZ_MAX:
        raise ValueError(
            f"GRID_LIMIT: main units * g_split = {units * g_split} > 65535"
        )
    lb_name = runtime_in.get("lb_order", "natural")
    if lb_name not in LB_ORDERS:
        raise ValueError(f"lb_order={lb_name!r} not in {tuple(LB_ORDERS)}")
    if lb_name == "interleave" and not spec.lb_interleave:
        # a decode without the switch would read the value as natural
        spec = validate_attn_bwd_spec(
            dataclasses.replace(spec, lb_interleave=True), arch
        )
    pair = 1 if lb_name == "mirror" else 0
    gx = ceil_div(n_kv_tiles, 2) if pair else n_kv_tiles
    if pair:
        if mask_kind == "none":
            raise ValueError("mirror pairing needs a causal or window mask")
        # The pairing decode visits every kv tile once for any grid (num_cus
        # never reaches the kernel), so this is a no-gain guard, not a
        # correctness rule. The CTA that pairs the heaviest kv tile does at
        # least that tile's work, the longest unpaired CTA: when the unpaired
        # grid runs in one wave (at most num_cus CTAs) pairing cannot shorten
        # it. A larger unpaired
        # grid runs in waves with a ragged tail, which pairing evens out even
        # when the paired grid itself is under num_cus.
        if req.num_cus is None or n_kv_tiles * units * g_split * req.b <= req.num_cus:
            raise ValueError(
                "mirror pairing needs num_cus and an unpaired grid that overfills "
                "it (more than num_cus CTAs)"
            )
        if not MIRROR_PAIRING_BUILT:
            raise ValueError(
                "mirror pairing is not built: the main kernel runs one kv tile per CTA"
            )
        if not spec.lb_mirror:
            # a decode without the switch runs one kv tile per CTA
            spec = validate_attn_bwd_spec(
                dataclasses.replace(spec, lb_mirror=True), arch
            )
    if req.num_xcds:
        xcd_n = int(req.num_xcds)
        chunk_in = runtime_in.get("xcd_chunk", 0)
        # symbolic multiples of the main grid x (the registry's sweep values)
        chunk_in = {"gx": gx, "2gx": 2 * gx, "4gx": 4 * gx}.get(chunk_in, chunk_in)
        xcd_chunk = int(chunk_in) or 0
        if xcd_chunk and xcd_chunk not in (gx, 2 * gx, 4 * gx, 64):
            raise ValueError(f"xcd_chunk={xcd_chunk} not in (gx, 2*gx, 4*gx, 64)")
        if not xcd_chunk:
            xcd_n = 0
    else:
        if runtime_in.get("xcd_chunk"):
            raise ValueError("xcd_chunk needs num_xcds")
        xcd_n, xcd_chunk = 0, 0
    if xcd_n and not spec.xcd_remap:
        # only a kernel built with the remap reads xcd_n / xcd_chunk
        spec = validate_attn_bwd_spec(dataclasses.replace(spec, xcd_remap=True), arch)
    use_worklist = runtime_in.get("use_worklist", False)
    if type(use_worklist) is not bool:
        raise ValueError(f"use_worklist={use_worklist!r} must be a bool")
    if use_worklist:
        _check_worklist(req, lb_name, n_kv_tiles)
        if not spec.worklist:
            spec = validate_attn_bwd_spec(
                dataclasses.replace(spec, worklist=True), arch
            )
    scale = req.resolved_scale()
    mults = _scale_mults(placement, scale, atomic_kv)
    unit = static_unit_mults(spec.static_scale, spec.dkv_mode)
    if any(mults[n] != 1.0 for n in unit):
        raise AssertionError("a multiplier the main kernel does not apply is not 1.0")
    split = spec.dq_mode == "split"
    if dq_in and not split:
        raise ValueError(f"dQ kernel knobs {sorted(dq_in)} need dq_mode='split'")
    dq_spec = dq_grid = None
    dq_group = _group(req, True)  # query heads sharing one K head and one V head
    dq_xcd_n = dq_xcd_chunk = 0
    if split:
        dq_spec = validate_attn_bwd_dq_spec(
            spec.dq_spec(**{k[3:]: v for k, v in dq_in.items() if k != "dq_xcd_chunk"}),
            arch,
        )
        dq_rows_per_head(dq_spec, dq_group)  # head_loop = "group": G | kM0
        dq_grid = attn_bwd_dq_grid(
            dq_spec, s_q_max=req.s_q, h_q=req.h_q, group=dq_group, batch=req.b
        )
        if dq_grid[1] > _GRID_YZ_MAX or dq_grid[0] >= 2**31:
            raise ValueError(f"GRID_LIMIT: dQ grid {dq_grid}")
        chunk_in = dq_in.get("dq_xcd_chunk", 0)
        if req.num_xcds:
            gxd = dq_grid[0]
            chunk_in = {"gx": gxd, "2gx": 2 * gxd, "4gx": 4 * gxd}.get(
                chunk_in, chunk_in
            )
            dq_xcd_chunk = int(chunk_in) or 0
            if dq_xcd_chunk and dq_xcd_chunk not in (gxd, 2 * gxd, 4 * gxd, 64):
                raise ValueError(
                    f"dq_xcd_chunk={dq_xcd_chunk} not in (gx, 2*gx, 4*gx, 64)"
                )
            dq_xcd_n = int(req.num_xcds) if dq_xcd_chunk else 0
        elif chunk_in:
            raise ValueError("dq_xcd_chunk needs num_xcds")
        if dq_xcd_n and not dq_spec.xcd_remap:
            dq_spec = validate_attn_bwd_dq_spec(
                dataclasses.replace(dq_spec, xcd_remap=True), arch
            )

    # slices: each CTA stores its fp32 dK / dV partial to its own workspace
    # slice (no atomics, no zero fill); reduce launches replace the converts
    slices = atomic_kv and spec.dkv_sink == "slices"
    ws_kw = dict(g_split=g_split, dq_mode=spec.dq_mode)
    if slices:
        ws_kw["dkv_sink"] = "slices"
    if use_worklist:
        ws_kw.update(worklist=True, block_n=kn)
    layout = attn_bwd_workspace_layout(req, **ws_kw)
    ws_bytes = attn_bwd_workspace_bytes(req, **ws_kw)
    rows_q, rows_kv = _rows(req)
    left, right, br = req.resolved_band()
    off64 = 1 if req.ragged_offset_dtype == "int64" else 0
    thd = req.is_thd
    has_len = 1 if req.padding else 0
    q_div = req.tensor_strides("q")[2] if thd else 1
    kv_div = req.tensor_strides("k")[2] if thd else 1
    mult = req.ragged_offset_multiplier if thd else 1

    def st(n):
        sb, sh, ss, _ = req.tensor_strides(n)
        return (0 if thd else sb), sh, ss

    common_len = {
        "has_len": has_len,
        "len_stride": req.seq_len_stride if has_len else 0,
        "off64": off64,
    }
    # THD rows by sequence slot unless both caller bounds are given; batched
    # rows are always slots (the flag is not read there).
    ws_seg = int(thd and not _ws_by_token(req))
    main_scalars = {
        "scale_log2": scale * LOG2E,
        "ds_mult": mults["ds_mult"],
        "dk_mult": mults["dk_mult"],
        "dq_mult": mults["dq_mult"],
    }
    for n, p in (
        ("q", "q"),
        ("k", "k"),
        ("v", "v"),
        ("do", "do"),
        ("dk", "dk"),
        ("dv", "dv"),
    ):
        main_scalars.update(dict(zip((f"{p}_b", f"{p}_h", f"{p}_t"), st(n))))
    gk = req.h_q // req.h_k
    gv = req.h_q // req.h_v
    main_scalars.update(
        {
            "h_q": req.h_q,
            "h_k": req.h_k,
            "h_v": req.h_v,
            "G": group,
            "gk": gk,
            "gv": gv,
            "S_q_max": req.s_q,
            "S_kv_max": req.s_kv,
            **common_len,
            "q_mult": mult,
            "q_div": q_div,
            "kv_mult": mult,
            "kv_div": kv_div,
            "left": left,
            "right": right,
            "bottom_right": int(br),
            "ws_rows_q": rows_q,
            "ws_rows_kv": rows_kv if atomic_kv else 0,
            "ws_seg": ws_seg,
            "g_split": g_split,
            "n_kv_tiles": n_kv_tiles,
            "lb_order": LB_ORDERS[lb_name],
            "pair": pair,
            "xcd_n": xcd_n,
            "xcd_chunk": xcd_chunk,
            "pack_heads": pack_heads,
            "use_worklist": int(use_worklist),
        }
    )
    ws = {k: k for k in layout}
    main_ptrs = {
        "Q": "q",
        "K": "k",
        "V": "v",
        "dO": "do",
        "dK": None if atomic_kv else "dk",
        "dV": None if atomic_kv else "dv",
        "WS_LSE2": "WS_LSE2",
        "WS_DSUM": "WS_DSUM",
        "WS_DQ": ws.get("WS_DQ"),
        "WS_DK": ws.get("WS_DK"),
        "WS_DV": ws.get("WS_DV"),
        "SEQ_Q": "seq_len_q" if has_len else None,
        "SEQ_KV": "seq_len_kv" if has_len else None,
        "OFF_Q": "offsets_q" if thd else None,
        "OFF_KV": "offsets_kv" if thd else None,
        "WORKLIST": ws.get("WORKLIST"),
    }
    prep_scalars = {}
    for n, p in (("o", "o"), ("do", "do")):
        prep_scalars.update(dict(zip((f"{p}_b", f"{p}_h", f"{p}_t"), st(n))))
    prep_scalars.update(dict(zip(("l_b", "l_h", "l_t"), _lse_kernarg_strides(req))))
    prep_scalars.update(
        {
            "h_q": req.h_q,
            "h_k": req.h_k,
            "h_v": req.h_v,
            "S_q_max": req.s_q,
            "S_kv_max": req.s_kv,
            **common_len,
            "q_mult": mult,
            "q_div": q_div,
            "kv_mult": mult,
            "kv_div": kv_div,
            "ws_rows_q": rows_q,
            "ws_rows_kv": rows_kv if atomic_kv else 0,
            "ws_seg": ws_seg,
            "zero_kv": int(atomic_kv and not slices),
            "zero_dq": int(spec.dq_mode == "atomic"),
            "use_worklist": int(use_worklist),
            "n_batch": req.b,
            "wl_kn0": kn if use_worklist else 0,
        }
    )
    prep_ptrs = {
        "O": "o",
        "dO": "do",
        "LSE": "lse",
        "SEQ_Q": main_ptrs["SEQ_Q"],
        "SEQ_KV": main_ptrs["SEQ_KV"],
        "OFF_Q": main_ptrs["OFF_Q"],
        "OFF_KV": main_ptrs["OFF_KV"],
        "WS_LSE2": "WS_LSE2",
        "WS_DSUM": "WS_DSUM",
        "WS_DQ": ws.get("WS_DQ"),
        "WS_DK": ws.get("WS_DK"),
        "WS_DV": ws.get("WS_DV"),
        "WORKLIST": ws.get("WORKLIST"),
    }
    conv_vec = aux_in.get("convert_vec", 8)
    if conv_vec not in CONVERT_VECS:
        raise ValueError(f"convert_vec={conv_vec!r} not in {CONVERT_VECS}")
    prep_spec = spec.aux_spec(
        "prep",
        rows_per_cta=_aux_rows(
            aux_in.get("prep_rows_per_cta", "1x"), "prep_rows_per_cta", arch, req.d_qk
        ),
    )
    conv_spec = spec.aux_spec(
        "convert",
        rows_per_cta=_aux_rows(
            aux_in.get("convert_rows_per_cta", "1x"),
            "convert_rows_per_cta",
            arch,
            req.d_qk // conv_vec * 8,
        ),
        convert_vec=conv_vec,
    )
    aux_block = attn_bwd_aux_block_threads(prep_spec, arch=arch)

    launches = [
        LaunchStep(
            "prep",
            prep_spec.kernel_name("prep"),
            attn_bwd_prep_grid(
                prep_spec,
                arch=arch,
                s_q_max=req.s_q,
                s_kv_max=req.s_kv,
                h_q=req.h_q,
                h_k=req.h_k,
                h_v=req.h_v,
                batch=req.b,
                zero_kv=atomic_kv and not slices,
            ),
            aux_block,
            _freeze(prep_scalars),
            _freeze(prep_ptrs),
        ),
        LaunchStep(
            "main",
            spec.kernel_name("main"),
            (gx, units * g_split, req.b),
            spec.waves * facts.wave_size,
            _freeze(main_scalars),
            _freeze(main_ptrs),
        ),
    ]

    def convert(src, dst, heads, s_max, tens, m, side, n_slices=0):
        """A convert launch, or with ``n_slices`` a slice reduce."""
        sb, sh, ss = st(tens)
        scal = {
            "mult": m,
            "H": heads,
            "d_b": sb,
            "d_h": sh,
            "d_t": ss,
            "S_max": s_max,
            **common_len,
            "tok_mult": mult,
            "tok_div": q_div if side == "q" else kv_div,
            "ws_rows": rows_q if side == "q" else rows_kv,
            "ws_seg": ws_seg,
        }
        ptrs = {
            "SRC": src,
            "DST": tens,
            "SEQ": ("seq_len_q" if side == "q" else "seq_len_kv") if has_len else None,
            "OFF": ("offsets_q" if side == "q" else "offsets_kv") if thd else None,
        }
        stage = "reduce" if n_slices else "convert"
        if n_slices:
            # slice i of kv head h starts i * rows * H * D floats after slice 0
            scal["s_stride"] = scal["ws_rows"] * heads * req.d_qk
            scal["n_slices"] = n_slices
        return LaunchStep(
            stage,
            conv_spec.kernel_name(stage),
            attn_bwd_convert_grid(
                conv_spec, arch=arch, s_max=s_max, heads=heads, batch=req.b
            ),
            attn_bwd_aux_block_threads(conv_spec, arch=arch),
            _freeze(scal),
            _freeze(ptrs),
        )

    if split:
        # The Q-parallel dQ kernel replaces the dQ workspace, its zero fill and
        # its convert: the convert multiplier folds into dq_mult.
        dq_scalars = {
            "scale_log2": scale * LOG2E,
            "ds_mult": mults["ds_mult"],
            "dq_mult": mults["dq_mult"] * mults["conv_dq"],
        }
        for n in ("q", "k", "v", "do", "dq"):
            dq_scalars.update(dict(zip((f"{n}_b", f"{n}_h", f"{n}_t"), st(n))))
        dq_scalars.update(
            {
                "h_q": req.h_q,
                "h_k": req.h_k,
                "h_v": req.h_v,
                "G": dq_group,
                "S_q_max": req.s_q,
                "S_kv_max": req.s_kv,
                **common_len,
                "q_mult": mult,
                "q_div": q_div,
                "kv_mult": mult,
                "kv_div": kv_div,
                "left": left,
                "right": right,
                "bottom_right": int(br),
                "ws_rows_q": rows_q,
                "ws_seg": ws_seg,
                "xcd_n": dq_xcd_n,
                "xcd_chunk": dq_xcd_chunk,
            }
        )
        if any(
            dq_scalars[n] != 1.0 for n in dq_static_unit_mults(dq_spec.static_scale)
        ):
            raise AssertionError("a multiplier the dQ kernel does not apply is not 1.0")
        dq_ptrs = {
            "Q": "q",
            "K": "k",
            "V": "v",
            "dO": "do",
            "dQ": "dq",
            "WS_LSE2": "WS_LSE2",
            "WS_DSUM": "WS_DSUM",
            "SEQ_Q": main_ptrs["SEQ_Q"],
            "SEQ_KV": main_ptrs["SEQ_KV"],
            "OFF_Q": main_ptrs["OFF_Q"],
            "OFF_KV": main_ptrs["OFF_KV"],
        }
        launches.append(
            LaunchStep(
                "dq",
                dq_spec.kernel_name(),
                dq_grid,
                dq_spec.waves * facts.wave_size,
                _freeze(dq_scalars),
                _freeze(dq_ptrs),
            )
        )
    else:
        launches.append(
            convert("WS_DQ", "dq", req.h_q, req.s_q, "dq", mults["conv_dq"], "q")
        )
    if atomic_kv:
        n_k, n_v = _dkv_slices(req, g_split, sink) if slices else (0, 0)
        launches.append(
            convert("WS_DK", "dk", req.h_k, req.s_kv, "dk", mults["conv_dk"], "kv", n_k)
        )
        launches.append(convert("WS_DV", "dv", req.h_v, req.s_kv, "dv", 1.0, "kv", n_v))
    for step in launches:
        names = [n for n, _ in attn_bwd_params(step.stage)]
        got = list(step.pointers) + list(step.scalars)
        if sorted(got) != sorted(names):
            raise AssertionError(f"{step.stage} kernargs out of sync with the ABI")

    runtime = {
        "g_split": g_split,
        "lb_order": lb_name,
        "pair": pair,
        "xcd_n": xcd_n,
        "xcd_chunk": xcd_chunk,
        "pack_heads": pack_heads,
        "use_worklist": use_worklist,
        "scale_placement": placement,
        "scale": scale,
        "prep_rows_per_cta": aux_in.get("prep_rows_per_cta", "1x"),
        "convert_rows_per_cta": aux_in.get("convert_rows_per_cta", "1x"),
        "convert_vec": conv_vec,
    }
    specs = {"main": spec, "prep": prep_spec, "convert": conv_spec}
    if split:
        # (an atomic-dQ plan records exactly the keys it always did)
        runtime.update(
            {"dq_mode": "split", "dq_xcd_n": dq_xcd_n, "dq_xcd_chunk": dq_xcd_chunk}
        )
        specs["dq"] = dq_spec
    if dma_note is not None:
        # a tuned global_path = "dma" ran on the register path; the reason
        runtime["dma_downgrade"] = dma_note
    if slices:
        # the reduce is the convert spec's kernel of stage "reduce"
        runtime["dkv_sink"] = "slices"
        specs["reduce"] = conv_spec
    return AttnBwdPlan(
        arch=arch,
        route=route,
        route_tag=tag,
        perf_class=pclass,
        fill_bucket=_fill_bucket(req, gx * units * g_split * req.b),
        specs=_freeze(specs),
        runtime=_freeze(runtime),
        launches=tuple(launches),
        workspace_bytes=ws_bytes,
        workspace_layout=_freeze(layout),
        tuning_key=tkey,
        perf_status=policy.status_of(arch, route, pclass),
    )


# ---------------------------------------------------------------------------
# declaration
# ---------------------------------------------------------------------------


def _general_classes(arch: str):
    for d in BWD_HEAD_SIZES:
        for dt in BWD_DTYPES:
            for mk in MASK_KINDS:
                for hm in HEAD_MODES:
                    for rel in LENGTH_RELATIONS:
                        for lm in LENGTH_MODES:
                            for sv in (8, 1):
                                yield (arch, "general", d, dt, mk, hm, rel, lm, sv)


_CLASS_FIELDS = (
    "arch",
    "route",
    "d",
    "dtype",
    "mask",
    "heads",
    "length_relation",
    "length_mode",
    "stage_vec",
)


def attn_bwd_declaration(arch: str, *, policy: Optional[AttnBwdPolicy] = None) -> dict:
    """Versioned, machine-readable declaration for ``arch`` (JSON-serialisable)."""
    policy = shipped_policy() if policy is None else policy
    if arch in policy.not_validated_archs:
        status = "not_validated"
    elif arch in DECLARED_ARCHS:
        status = "declared"
    else:
        status = "unsupported"
    perf = []
    if status == "declared":
        for cls in _general_classes(arch):
            entry = dict(zip(_CLASS_FIELDS, cls))
            entry["perf"] = policy.status_of(arch, "general", cls)
            perf.append(entry)
        for parch, key in sorted(policy.dense_pack, key=repr):
            if parch == arch:
                cls = (arch, "dense") + tuple(key)
                perf.append(
                    {
                        "route": "dense",
                        "key": list(map(_jsonable, key)),
                        "perf": policy.status_of(arch, "dense", cls),
                    }
                )
    return {
        "family": "attention_bwd_general",
        "version": ATTN_BWD_DECLARATION_VERSION,
        "abi": ATTN_BWD_ABI,
        "arch": arch,
        "status": status,
        "rdna_status": "degraded_performance" if arch in RDNA_ARCHS else "n/a",
        "axes": {
            "dtype": list(BWD_DTYPES),
            "head_dim": list(BWD_HEAD_SIZES),
            "head_dim_declined": {
                "256": {
                    "code": "HEAD_DIM_256_BWD",
                    "kind": DECLINE_KINDS["HEAD_DIM_256_BWD"],
                    "reason": REQ_SCOPE_D256_BWD,
                }
            },
            "layout": ["dense_strided", "thd"],
            "length_mode": list(LENGTH_MODES),
            "heads": list(HEAD_MODES),
            "mask": list(MASK_KINDS),
            "s_q_1": True,
            "stage_vec": [8, 1],
            "ragged_offset_dtype": ["int32", "int64"],
            "max_total_optional": True,
            "max_total_used": "only when both max_total_q and max_total_kv "
            "are supplied",
        },
        "reason_codes": list(REASON_CODES),
        "reason_kinds": {c: DECLINE_KINDS[c] for c in REASON_CODES},
        "not_yet_implemented_prefix": f"{NOT_YET_IMPLEMENTED}:",
        "feature_groups": list(FEATURE_GROUPS),
        "declared_feature_groups": [
            g for g in FEATURE_GROUPS if g in policy.declared_groups
        ],
        "lse": {
            "base": "e",
            "dtype": "fp32",
            "layout": "[B,H_q,S_q,1] | [T_q,H_q,1]",
            "strides": "() = packed, or one non-negative integer per dim: dense "
            "(b,h,t,1); thd (t,h,1), or (b,h,t,1) with b unused for the batched "
            "shape (no packed default); any other form is declined LSE_FORMAT",
            "dead_row": "-inf (NaN also treated as dead)",
        },
        "workspace": {
            "align": _WS_ALIGN,
            "rows": "fixed/padded: B*S; thd: max_total_q / max_total_kv when both "
            "are supplied (rows indexed by storage token; each bound must cover "
            "the end token of every segment), else B*S_max (rows indexed by "
            "sequence slot b*S_max+i; an upper bound for any storage layout)",
            "terms": [
                "WS_DQ A(4*rows_q*h_q*D) if dq atomic",
                "WS_LSE2 A(4*rows_q*h_q)",
                "WS_DSUM A(4*rows_q*h_q)",
                "WS_DK A(4*rows_kv*h_k*D*n_k) if h_k != h_v or g_split > 1",
                "WS_DV A(4*rows_kv*h_v*D*n_v) if h_k != h_v or g_split > 1",
                "WORKLIST A(8*(1 + B*ceil(S_kv/kN0))) if work list (THD)",
            ],
            "slices": "n_k = n_v = 1 with dkv_sink='atomic' (the default); with "
            "dkv_sink='slices' n_k = (h_q/h_k)/(G/g_split) and n_v = "
            "(h_q/h_v)/(G/g_split), G = gcd(h_q/h_k, h_q/h_v)",
        },
        "determinism": {
            "dq": "nondeterministic in the last bits (fp32 atomics) with "
            "dq_mode='atomic' (the default); deterministic (one fixed-order "
            "register sum, no atomics) with dq_mode='split'",
            "dk_dv": "deterministic in direct mode (h_k == h_v, g_split == 1); "
            "in atomic mode nondeterministic in the last bits with "
            "dkv_sink='atomic' (the default), deterministic with "
            "dkv_sink='slices' (per-CTA fp32 slices, plain stores, summed in "
            "slice order by the reduce kernel; no atomics)",
            "atomic_free": "dq_mode='split' with direct dK / dV or "
            "dkv_sink='slices': no data atomic in any launch, every output "
            "bit for bit reproducible (AttnBwdPlan.deterministic); a "
            "deterministic request is served this way only under "
            "AttnBwdPolicy.deterministic, on " + ", ".join(SPLIT_DQ_ARCHS),
        },
        "perf_policy": {
            "gate": "declines" if policy.perf_gate else "information_only",
            "unverified": (
                ("waived" if policy.waive_unverified else "declined")
                if policy.perf_gate
                else "served"
            ),
            "below_baseline": "declined" if policy.perf_gate else "served",
        },
        "perf": perf,
        "deferral_register": [dict(row) for row in DEFERRAL_REGISTER],
        "notes": list(DECLARATION_NOTES),
    }


def _jsonable(x):
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    return x
