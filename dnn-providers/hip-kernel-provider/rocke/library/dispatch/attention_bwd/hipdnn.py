# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Translate a hipDNN ``SdpaBackward`` graph, in its flatbuffer JSON form, into an
:class:`~kernels.common.attention_bwd_plan.AttnBwdRequest`.

The input is what the hipDNN flatbuffer SDK's JSON helpers write
(``utilities/json/SdpaBackwardAttributes.hpp`` and ``TensorAttributes.hpp``):
one node with ``inputs`` / ``outputs`` / ``parameters`` maps, plus the graph's
``tensors`` list. This module is the reference of the native hipDNN engine
``hipkernel:AttentionGeneralBwd``: its C++ lowering applies the same rules in
the same order, and its verdict is :func:`engine_verdict` of the lowered
request.

The lowering declines (:class:`HipdnnLoweringError`, the engine's
``LOWERING`` code) only graphs it cannot read: a missing or malformed
operand, a ragged offset it does not consume, an unreadable offset table or
band. Everything else maps onto request fields, and the general backward
predicate decides.

* Band: hipDNN's ``left_bound = L`` keeps ``L`` keys ending at the diagonal;
  rocKE's ``left`` is the FlashAttention window ``L - 1``.
* THD: a tensor is ragged when it names a ``ragged_offset_tensor_uid``. Its
  sequence ``b`` starts at element ``offsets[b] * ragged_offset_multiplier``;
  the request carries every multiplier
  (``AttnBwdRequest.ragged_offset_multipliers``). Q, O, dO, dQ (and a ragged
  stats) share Q's table, K, V, dK, dV share K's.
* Stats: ragged on Q's table, or dense ``[B, H_q, S_q, 1]`` (declined under
  THD by the predicate, ``LSE_FORMAT``).
* Alignment: the request's ``tensor_alignment`` is the smallest alignment the
  eight data tensors guarantee, each the largest power of two dividing its
  declared ``alignment`` (hipDNN's default 16 when absent).
"""

from __future__ import annotations

import dataclasses
import functools
import math
import struct
from collections.abc import Mapping
from typing import Any

from rocke.numeric.sdpa_reference import resolve_hipdnn_band

from kernels.common.attention_bwd import STATIC_SCALES, attn_bwd_params
from kernels.common.attention_bwd_aux import attn_bwd_aux_rows_per_cta
from kernels.common.attention_bwd_plan import (
    _KV_SIDE,
    _Q_SIDE,
    SHIPPED_DECLARED_GROUPS,
    AttnBwdPlan,
    AttnBwdPolicy,
    AttnBwdRequest,
    attn_bwd_plan,
    attn_bwd_support,
    attn_bwd_table_key,
    shipped_policy,
)
from kernels.common.attention_bwd_plan import _TENSORS as _DATA

from ..sdpa.hipdnn import HIPDNN_DTYPES
from .routing import ARCH_TUNING, arch_tuned_policy

__all__ = [
    "ENCODINGS",
    "ENGINE_ARCHS",
    "ENGINE_NAME",
    "LOWERING",
    "MAIN_TOKENS",
    "PREP_TOKENS",
    "WS_REGIONS",
    "HipdnnLoweringError",
    "engine_policy",
    "engine_verdict",
    "from_hipdnn_graph",
    "main_fields",
    "plan_record",
    "plan_table_key",
    "prep_fields",
    "tensor_roles",
    "to_hipdnn_graph",
    "tuned_keys",
    "tuning_key_id",
    "tuning_key_string",
]

ENGINE_NAME = "hipkernel:AttentionGeneralBwd"
ENGINE_ARCHS = ("gfx942", "gfx950")
# The engine's own reason code, after the predicate's: a graph the lowering
# cannot read.
LOWERING = "LOWERING"
NODE_TYPE = "SdpaBackwardAttributes"

# Required operands: (role, group, key). ``lse`` is the node's stats tensor.
_REQUIRED = (
    ("q", "inputs", "q_tensor_uid"),
    ("k", "inputs", "k_tensor_uid"),
    ("v", "inputs", "v_tensor_uid"),
    ("o", "inputs", "o_tensor_uid"),
    ("do", "inputs", "do_tensor_uid"),
    ("lse", "inputs", "stats_tensor_uid"),
    ("dq", "outputs", "dq_tensor_uid"),
    ("dk", "outputs", "dk_tensor_uid"),
    ("dv", "outputs", "dv_tensor_uid"),
)
# hipDNN's tensor alignment when a graph declares none.
DEFAULT_ALIGNMENT = 16
_OPTIONAL_INPUTS = (
    "scale_tensor_uid",
    "attn_mask_tensor_uid",
    "seq_len_q_tensor_uid",
    "seq_len_kv_tensor_uid",
    "seed_tensor_uid",
    "offset_tensor_uid",
    "dropout_mask_tensor_uid",
    "dropout_scale_tensor_uid",
    "dropout_scale_inv_tensor_uid",
)
_DROPOUT_INPUTS = (
    "seed_tensor_uid",
    "offset_tensor_uid",
    "dropout_mask_tensor_uid",
    "dropout_scale_tensor_uid",
    "dropout_scale_inv_tensor_uid",
)
# The probability recorded for dropout requested through tensors only.
_DROPOUT_SENTINEL = 1.0
# Smallest stats alignment the prep's fp32 reads take.
_LSE_MIN_ALIGNMENT = 4


class HipdnnLoweringError(ValueError):
    """The graph cannot be lowered to a backward request (code ``LOWERING``)."""


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


@functools.cache
def engine_policy() -> AttnBwdPolicy:
    """The engine's policy: the per-arch general tuned tables over the shipped
    policy, without the padding group (no padding mask through hipDNN) and
    without the dense pack."""
    policy = dataclasses.replace(
        arch_tuned_policy(shipped_policy()),
        declared_groups=SHIPPED_DECLARED_GROUPS - {"padding"},
    )
    if (
        policy.perf_gate
        or policy.instance_unavailable
        or policy.not_validated_archs
        or policy.dense_pack
        or policy.deterministic
    ):
        raise AssertionError("the engine policy reads another shipped policy")
    return policy


# ---------------------------------------------------------------------------
# graph access
# ---------------------------------------------------------------------------


def _node(graph_json: Mapping, node_index: int) -> Mapping:
    nodes = graph_json.get("nodes") if isinstance(graph_json, Mapping) else None
    if not isinstance(nodes, list) or len(nodes) != 1 or node_index != 0:
        raise HipdnnLoweringError("not an SdpaBackward graph")
    node = nodes[0]
    if not isinstance(node, Mapping) or node.get("type") != NODE_TYPE:
        raise HipdnnLoweringError("not an SdpaBackward graph")
    return node


def _group(node: Mapping, name: str) -> Mapping:
    group = node.get(name)
    if group is None:
        return {}
    if not isinstance(group, Mapping):
        raise HipdnnLoweringError(f"node {name} is not a map")
    return group


def _table(graph_json: Mapping) -> dict[int, Mapping]:
    tensors = graph_json.get("tensors")
    if not isinstance(tensors, list):
        raise HipdnnLoweringError("graph tensors is not a list")
    out = {}
    for entry in tensors:
        uid = entry.get("uid") if isinstance(entry, Mapping) else None
        if not _is_int(uid):
            raise HipdnnLoweringError(f"tensor entry {entry!r} has no integer uid")
        out[int(uid)] = entry
    return out


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _uid(group: Mapping, key: str) -> int | None:
    v = group.get(key)
    if v is None:
        return None
    if not _is_int(v):
        raise HipdnnLoweringError(f"{key} {v!r} is not an integer uid")
    return int(v)


def _flag(group: Mapping, key: str) -> bool:
    v = group.get(key)
    if v is None:
        return False
    if not isinstance(v, bool):
        raise HipdnnLoweringError(f"{key} must be a JSON boolean, got {v!r}")
    return v


def _ints(entry: Mapping, key: str, uid: int) -> tuple:
    v = entry.get(key)
    if not isinstance(v, list) or not all(_is_int(x) for x in v):
        raise HipdnnLoweringError(f"tensor uid {uid} {key} {v!r} is not an int list")
    return tuple(int(x) for x in v)


def _dtype_name(entry: Mapping) -> str:
    """rocKE spelling of a tensor's hipDNN ``data_type`` (an unknown name is
    kept lower-case, so the predicate declines it by name)."""
    key = str(entry.get("data_type")).lower()
    mapped = HIPDNN_DTYPES.get(key, key)
    return "unset" if mapped is None else mapped


def _alignment(entry: Mapping, uid: int) -> int:
    v = entry.get("alignment", DEFAULT_ALIGNMENT)
    if not _is_int(v):
        raise HipdnnLoweringError(f"tensor uid {uid} alignment {v!r} is not an int")
    v = int(v)
    # The byte alignment the declaration guarantees: its largest power-of-two divisor
    # (a multiple of 24 is only 8-byte aligned).
    return v & -v if v >= 1 else v


def _multiplier(entry: Mapping, uid: int) -> int:
    v = entry.get("ragged_offset_multiplier", 1)
    if v is None:
        return 1
    if not _is_int(v):
        raise HipdnnLoweringError(
            f"tensor uid {uid} ragged_offset_multiplier {v!r} is not an int"
        )
    return int(v)


def tensor_roles(graph_json: Mapping, *, node_index: int = 0) -> dict:
    """``{role: uid or None}`` of a backward graph: the nine operands
    (``q k v o do lse dq dk dv``) and Q's / K's offset tables
    (``offsets_q``, ``offsets_kv``)."""
    node = _node(graph_json, node_index)
    out = {}
    for role, group, key in _REQUIRED:
        out[role] = _uid(_group(node, group), key)
    table = _table(graph_json)
    for role, lead in (("offsets_q", "q"), ("offsets_kv", "k")):
        entry = table.get(out[lead]) if out[lead] is not None else None
        out[role] = None if entry is None else _uid(entry, "ragged_offset_tensor_uid")
    return out


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


def from_hipdnn_graph(
    graph_json: Mapping[str, Any],
    arch: str,
    *,
    num_cus: int | None,
    node_index: int = 0,
) -> AttnBwdRequest:
    """Lower a one-node ``SdpaBackward`` graph to an :class:`AttnBwdRequest`.

    ``num_cus`` is the device's compute-unit count (the tuning key's fill
    bucket). ``arch`` is not read by the lowering (the engine's arch rule is
    :func:`engine_verdict`'s). Raises :class:`HipdnnLoweringError` for a graph
    the lowering cannot read, first failure first.
    """
    del arch
    node = _node(graph_json, node_index)
    inputs, outputs = _group(node, "inputs"), _group(node, "outputs")
    params = _group(node, "parameters")
    table = _table(graph_json)

    def entry_of(uid: int, what: str) -> Mapping:
        found = table.get(uid)
        if found is None:
            raise HipdnnLoweringError(f"{what} uid {uid} names no tensor")
        return found

    # operands
    ent: dict[str, Mapping] = {}
    uids: dict[str, int] = {}
    groups = {"inputs": inputs, "outputs": outputs}
    for role, group, key in _REQUIRED:
        uid = _uid(groups[group], key)
        if uid is None:
            raise HipdnnLoweringError(f"required {key} is not set")
        e = entry_of(uid, key)
        if _flag(e, "virtual") or _flag(e, "is_runtime_pass_by_value"):
            raise HipdnnLoweringError(f"{role} (uid {uid}) is virtual or pass-by-value")
        uids[role], ent[role] = uid, e
    dims = {r: _ints(ent[r], "dims", uids[r]) for r in ent}
    strides = {r: _ints(ent[r], "strides", uids[r]) for r in ent}
    for r in ent:
        if len(dims[r]) != 4 or len(strides[r]) != 4:
            raise HipdnnLoweringError(
                f"{r} (uid {uids[r]}) dims {dims[r]} / strides {strides[r]} not rank 4"
            )
    for r in ent:
        if min(dims[r]) < 0:
            raise HipdnnLoweringError(f"{r} (uid {uids[r]}) dims {dims[r]} negative")
    _check_dims(dims)

    # ragged offsets
    offsets = {}  # role -> offset uid
    for r in (*_DATA, "lse"):
        off = _uid(ent[r], "ragged_offset_tensor_uid")
        if off is None:
            continue
        o = entry_of(off, f"{r} ragged offset")
        if _flag(o, "virtual") or _flag(o, "is_runtime_pass_by_value"):
            raise HipdnnLoweringError(
                f"{r} offset table uid {off} is virtual or pass-by-value"
            )
        if _uid(o, "ragged_offset_tensor_uid") is not None:
            raise HipdnnLoweringError(f"{r} offset table uid {off} is itself ragged")
        o_dims, o_st = _ints(o, "dims", off), _ints(o, "strides", off)
        if len(o_dims) != 4 or len(o_st) != 4:
            raise HipdnnLoweringError(f"{r} offset table uid {off} is not rank 4")
        if o_dims[0] > 1 and o_st[0] != 1:
            raise HipdnnLoweringError(
                f"{r} offset table uid {off} stride {o_st[0]} is not 1"
            )
        offsets[r] = off
    consumed = {uids[r] for r in ent}
    for uid, e in sorted(table.items()):
        if uid not in consumed and _uid(e, "ragged_offset_tensor_uid") is not None:
            raise HipdnnLoweringError(
                f"tensor uid {uid} is ragged; only the backward operands are read "
                "through an offset table"
            )

    if _alignment(ent["lse"], uids["lse"]) < _LSE_MIN_ALIGNMENT:
        raise HipdnnLoweringError("stats alignment below 4 bytes")

    try:
        diagonal, left, right = resolve_hipdnn_band(
            left_bound=params.get("left_bound"),
            right_bound=params.get("right_bound"),
            diagonal_alignment=params.get("diagonal_alignment"),
            causal_mask=_flag(params, "causal_mask"),
            causal_mask_bottom_right=_flag(params, "causal_mask_bottom_right"),
        )
    except (TypeError, ValueError) as exc:
        raise HipdnnLoweringError(f"band: {exc}") from exc

    # request fields
    q, k, v = dims["q"], dims["k"], dims["v"]
    kw: dict[str, Any] = {
        "b": q[0],
        "h_q": q[1],
        "h_k": k[1],
        "h_v": v[1],
        "s_q": q[2],
        "s_kv": k[2],
        "d_qk": q[3],
        "d_v": v[3],
        "num_cus": num_cus,
        "left_bound": -1 if left == -1 else left - 1,
        "right_bound": right,
        "bottom_right": diagonal == "bottom_right",
    }
    dts = {r: _dtype_name(ent[r]) for r in ent}
    kw["dtype"] = dts["q"]
    for r in _DATA[1:]:
        kw[f"{r}_dtype"] = dts[r]
    kw["fp8"] = any(dts[r].startswith("fp8") for r in _DATA)
    kw["lse_dtype"] = dts["lse"]
    compute = node.get("compute_data_type")
    if compute is not None and str(compute).lower() != "float":
        kw["other_features"] = ("compute_data_type",)

    thd = any(r in offsets for r in _DATA)
    if thd:
        kw["layout"] = "thd"
        kw["strides"] = {r: (0, *strides[r][1:]) for r in _DATA}
        kw.update(_ragged_fields(dims["q"][0], offsets, ent, uids, table))
    else:
        kw["strides"] = {r: strides[r] for r in _DATA}
    kw["lse_dims"] = dims["lse"]
    kw["lse_strides"] = strides["lse"]
    kw["lse_ragged"] = "lse" in offsets
    kw["tensor_alignment"] = min(_alignment(ent[r], uids[r]) for r in _DATA)

    kw.update(_scale_fields(inputs, params, entry_of))
    kw["has_bias"] = _uid(inputs, "attn_mask_tensor_uid") is not None
    kw["want_dbias"] = _uid(outputs, "dbias_tensor_uid") is not None
    kw["alibi"] = _flag(params, "alibi_mask")
    probability = params.get("dropout_probability")
    if probability is not None and not isinstance(probability, (int, float)):
        raise HipdnnLoweringError(f"dropout_probability {probability!r}")
    p = 0.0 if probability is None else float(probability)
    if p != 0.0:
        kw["dropout"] = p
    elif any(_uid(inputs, key) is not None for key in _DROPOUT_INPUTS):
        kw["dropout"] = _DROPOUT_SENTINEL
    for key in _OPTIONAL_INPUTS:
        uid = _uid(inputs, key)
        if uid is not None:
            entry_of(uid, key)
    dbias = _uid(outputs, "dbias_tensor_uid")
    if dbias is not None:
        entry_of(dbias, "dbias_tensor_uid")
    kw.update(_padding_fields(inputs, params, entry_of))
    return AttnBwdRequest(**kw)


def _check_dims(dims: Mapping[str, tuple]) -> None:
    q, k, v = dims["q"], dims["k"], dims["v"]
    b, h_q, s_q, d_qk = q
    want = {
        "k": (b, k[1], k[2], d_qk),
        "v": (b, v[1], k[2], v[3]),
        "o": (b, h_q, s_q, v[3]),
        "do": (b, h_q, s_q, v[3]),
        "dq": q,
        "dk": k,
        "dv": v,
    }
    for r, w in want.items():
        if dims[r] != w:
            raise HipdnnLoweringError(f"{r} dims {dims[r]}, want {w}")
    if dims["lse"][:3] != (b, h_q, s_q):
        raise HipdnnLoweringError(
            f"stats dims {dims['lse']}, want [{b}, {h_q}, {s_q}, 1]"
        )


def _ragged_fields(b, offsets, ent, uids, table) -> dict:
    tables_q = {offsets.get(r) for r in _Q_SIDE}
    tables_kv = {offsets.get(r) for r in _KV_SIDE}
    same = (
        all(r in offsets for r in _DATA)
        and len(tables_q) == 1
        and len(tables_kv) == 1
        and offsets.get("lse", offsets["q"]) == offsets["q"]
    )
    used = []
    for r in (*_DATA, "lse"):
        if r in offsets and offsets[r] not in used:
            used.append(offsets[r])
    dtypes = []
    for off in used:
        name = _dtype_name(table[off])
        if name not in dtypes:
            dtypes.append(name)
    dims = (b + 1,)
    for off in used:
        o_dims = _ints(table[off], "dims", off)
        if o_dims != (b + 1, 1, 1, 1):
            dims = o_dims
            break
    return {
        "ragged_same_table": same,
        "ragged_offset_dtype": dtypes[0] if len(dtypes) == 1 else "mixed",
        "ragged_offset_dims": dims,
        "ragged_offset_multipliers": {r: _multiplier(ent[r], uids[r]) for r in offsets},
    }


_ONE_BYTE_DTYPES = frozenset(
    (
        "uint8",
        "int8",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e4m3_fnuz",
        "fp8_e5m2_fnuz",
        "fp8_e8m0",
    )
)


def _scale_fields(inputs, params, entry_of) -> dict:
    uid = _uid(inputs, "scale_tensor_uid")
    if uid is not None:
        e = entry_of(uid, "scale_tensor_uid")
        value = e.get("value")
        if _flag(e, "is_runtime_pass_by_value") or value is None:
            return {"scale_is_device_tensor": True}
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HipdnnLoweringError(f"scale tensor value {value!r} is not a number")
        dtype = str(e.get("data_type")).lower()
        if dtype in _ONE_BYTE_DTYPES or e.get("value_type") == "Float8Value":
            # hipDNN stores a 1-byte constant as its encoded byte, not its value.
            raise HipdnnLoweringError(f"scale tensor of 1-byte dtype {dtype!r}")
        return {"scale": float(value)}
    value = params.get("attn_scale_value")
    if value is None:
        # hipDNN (cuDNN) applies no scaling when neither is given.
        return {"scale": 1.0}
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HipdnnLoweringError(f"attn_scale_value {value!r} is not a number")
    return {"scale": float(value)}


def _padding_fields(inputs, params, entry_of) -> dict:
    out: dict[str, Any] = {"padding": _flag(params, "padding_mask")}
    forms = []
    for side in ("q", "kv"):
        uid = _uid(inputs, f"seq_len_{side}_tensor_uid")
        if uid is None:
            continue
        out[f"has_seq_len_{side}"] = True
        e = entry_of(uid, f"seq_len_{side}_tensor_uid")
        dims, st = _ints(e, "dims", uid), _ints(e, "strides", uid)
        forms.append((_dtype_name(e), dims, st[0] if st else 0))
    if forms:
        dtype, dims, stride = forms[0]
        if any(f != forms[0] for f in forms):
            dtype = "mixed"
        out.update(seq_len_dtype=dtype, seq_len_dims=dims, seq_len_stride=stride)
    return out


# ---------------------------------------------------------------------------
# inverse (corpus seeds)
# ---------------------------------------------------------------------------

_TO_HIPDNN = {"fp16": "half", "bf16": "bfloat16", "fp32": "float"}
for _name, _rk in HIPDNN_DTYPES.items():
    if _rk is not None:
        _TO_HIPDNN.setdefault(_rk, _name)
_DEFAULT_UIDS = {
    "q": 0,
    "k": 1,
    "v": 2,
    "o": 3,
    "do": 4,
    "lse": 5,
    "dq": 6,
    "dk": 7,
    "dv": 8,
    "offsets_q": 9,
    "offsets_kv": 10,
    "offsets_q2": 11,
    "scale": 12,
    "attn_mask": 13,
    "dbias": 14,
    "seq_len_q": 15,
    "seq_len_kv": 16,
}


def _hipdnn_dtype(name: str) -> str:
    if name not in _TO_HIPDNN:
        raise ValueError(f"dtype {name!r} has no hipDNN spelling")
    return _TO_HIPDNN[name]


def to_hipdnn_graph(req: AttnBwdRequest, *, uids: dict | None = None) -> dict:
    """The hipDNN ``SdpaBackward`` graph JSON of ``req`` (the inverse of
    :func:`from_hipdnn_graph` up to its canonical form: explicit strides and
    per-tensor multipliers).

    Raises ``ValueError`` for a request hipDNN cannot express (paged,
    deterministic, ``max_total_*``, XCD hints, unknown features, a band
    combining the deprecated booleans with other settings, a bound below
    ``-1``).
    """
    u = dict(_DEFAULT_UIDS, **(uids or {}))
    if req.paged or req.deterministic:
        raise ValueError("paged / deterministic requests have no hipDNN graph")
    if req.max_total_q is not None or req.max_total_kv is not None:
        raise ValueError("max_total_* has no hipDNN spelling")
    if req.num_xcds is not None:
        raise ValueError("num_xcds has no hipDNN spelling")
    if req.fp8 and not any(
        str(getattr(req, f) or req.dtype).startswith("fp8")
        for f in ("dtype", "k_dtype", "v_dtype", "o_dtype", "do_dtype")
    ):
        raise ValueError("fp8 without an fp8 tensor has no hipDNN spelling")
    if set(req.other_features) - {"compute_data_type"}:
        raise ValueError(f"features {req.other_features} have no hipDNN spelling")
    if (req.causal or req.causal_bottom_right) and (
        (req.causal and req.causal_bottom_right)
        or req.left_bound != -1
        or req.right_bound != -1
    ):
        raise ValueError("the deprecated causal booleans with another band setting")
    left, right, br = req.resolved_band()
    if left < -1 or right < -1:
        raise ValueError(f"bounds ({left}, {right}) have no hipDNN spelling")
    thd = req.is_thd
    dts = {
        "q": req.dtype,
        "k": req.k_dtype,
        "v": req.v_dtype,
        "o": req.o_dtype,
        "do": req.do_dtype,
        "dq": req.dq_dtype,
        "dk": req.dk_dtype,
        "dv": req.dv_dtype,
    }
    tensors = []

    def add(name, uid, dtype, dims, strides, **extra):
        entry = {
            "uid": uid,
            "name": name,
            "data_type": _hipdnn_dtype(dtype),
            "dims": [int(x) for x in dims],
            "strides": [int(x) for x in strides],
            "virtual": False,
            "is_runtime_pass_by_value": False,
            "alignment": 16,
            "ragged_offset_multiplier": 1,
        }
        entry.update(extra)
        tensors.append(entry)
        return entry

    lse_dims, lse_st = _lse_graph_form(req)
    if thd and req.lse_ragged is False and tuple(req.lse_dims) != lse_dims:
        raise ValueError("a dense per-batch THD LSE needs the [B, H_q, S_q, 1] dims")
    lse_ragged = thd and req.lse_ragged is not False
    mults = _graph_multipliers(req, lse_st[2] if lse_ragged else None) if thd else {}
    for n in _DATA:
        h = req.tensor_heads(n)
        s = req.tensor_seq(n)
        d = req.d_qk if n in ("q", "k", "dq", "dk") else req.d_v
        sb, sh, ss, sd = req.tensor_strides(n)
        extra: dict[str, Any] = {"alignment": int(req.tensor_alignment)}
        if thd:
            sb = s * ss
            side_q = n in _Q_SIDE
            table = "offsets_q" if side_q else "offsets_kv"
            if n == "o" and not req.ragged_same_table:
                table = "offsets_q2"
            extra.update(
                ragged_offset_tensor_uid=u[table], ragged_offset_multiplier=mults[n]
            )
        name = n.upper() if n != "do" else "dO"
        add(
            name, u[n], dts[n] or req.dtype, (req.b, h, s, d), (sb, sh, ss, sd), **extra
        )

    lse_extra: dict[str, Any] = {}
    if lse_ragged:
        lse_extra.update(
            ragged_offset_tensor_uid=u["offsets_q"],
            ragged_offset_multiplier=mults["lse"],
        )
    elif req.lse_ragged and not thd:
        lse_extra.update(ragged_offset_tensor_uid=u["offsets_q"])
    add("stats", u["lse"], req.lse_dtype, lse_dims, lse_st, **lse_extra)

    if thd or req.lse_ragged:
        odims = req.ragged_offset_dims or (req.b + 1,)
        if len(odims) == 1:
            odims = (odims[0], 1, 1, 1)
        if len(odims) != 4:
            raise ValueError(f"offset dims {odims} have no hipDNN spelling")
        q_dt = kv_dt = req.ragged_offset_dtype
        if q_dt == "mixed":
            q_dt, kv_dt = "int32", "int64"
        ost = (1, 1, 1, 1)
        add("offsets_q", u["offsets_q"], q_dt, odims, ost)
        if thd:
            add("offsets_kv", u["offsets_kv"], kv_dt, odims, ost)
            if not req.ragged_same_table:
                add("offsets_q2", u["offsets_q2"], q_dt, odims, ost)

    inputs: dict[str, Any] = {key: None for key in _OPTIONAL_INPUTS}
    for role, group, key in _REQUIRED:
        if group == "inputs":
            inputs[key] = u[role]
    outputs = {
        "dq_tensor_uid": u["dq"],
        "dk_tensor_uid": u["dk"],
        "dv_tensor_uid": u["dv"],
        "dbias_tensor_uid": None,
    }
    params: dict[str, Any] = {
        "alibi_mask": req.alibi,
        "padding_mask": req.padding,
        "causal_mask": False,
        "causal_mask_bottom_right": False,
        "dropout_probability": float(req.dropout) if req.dropout else None,
        "attn_scale_value": None,
        "left_bound": None if left == -1 else left + 1,
        "right_bound": None if right == -1 else right,
        "diagonal_alignment": "BOTTOM_RIGHT" if br else "TOP_LEFT",
    }
    scalar = ((1, 1, 1, 1), (1, 1, 1, 1))
    if req.scale_is_device_tensor:
        inputs["scale_tensor_uid"] = u["scale"]
        add("attn_scale", u["scale"], "fp32", *scalar, is_runtime_pass_by_value=True)
    else:
        # rocKE's default (1/sqrt(d)) must be explicit: hipDNN's is 1.0.
        params["attn_scale_value"] = float(req.resolved_scale())
    bias_dims = (req.b, req.h_q, req.s_q, req.s_kv)
    bias_st = (req.h_q * req.s_q * req.s_kv, req.s_q * req.s_kv, req.s_kv, 1)
    if req.has_bias:
        inputs["attn_mask_tensor_uid"] = u["attn_mask"]
        add("bias", u["attn_mask"], req.dtype, bias_dims, bias_st)
    if req.want_dbias:
        outputs["dbias_tensor_uid"] = u["dbias"]
        add("dbias", u["dbias"], req.dtype, bias_dims, bias_st)
    sl_dims = req.seq_len_dims or (req.b, 1, 1, 1)
    sl_st = (req.seq_len_stride,) + (1,) * (len(sl_dims) - 1)
    sl_dt = req.seq_len_dtype if req.seq_len_dtype != "mixed" else "int32"
    for side, has in (("q", req.has_seq_len_q), ("kv", req.has_seq_len_kv)):
        if has:
            inputs[f"seq_len_{side}_tensor_uid"] = u[f"seq_len_{side}"]
            add(f"seq_len_{side}", u[f"seq_len_{side}"], sl_dt, sl_dims, sl_st)
    node = {
        "name": "SdpaBwd_0",
        "type": NODE_TYPE,
        "compute_data_type": (
            "half" if "compute_data_type" in req.other_features else "float"
        ),
        "inputs": inputs,
        "outputs": outputs,
        "parameters": params,
    }
    return {"name": "SdpaBackward", "nodes": [node], "tensors": tensors}


def _graph_multipliers(req: AttnBwdRequest, lse_t: int | None) -> dict:
    """hipDNN (element) multipliers of a THD request's ragged tensors.

    The request's own offsets units when every tensor's element offset is
    whole (a tensor at ``offsets * m / div`` tokens of its side has ``m *
    token stride / div``), else offsets in tokens (each multiplier its
    tensor's token stride): the same problem, with other offset values.
    """
    given = dict(req.ragged_offset_multipliers or {})
    strides = {n: req.tensor_strides(n)[2] for n in _DATA}
    if lse_t is not None:
        strides["lse"] = lse_t
    out = {}
    for n, ts in strides.items():
        lead = "k" if n in _KV_SIDE else "q"
        m = given.get(lead, req.ragged_offset_multiplier)
        div = req.tensor_strides(lead)[2]
        if n in given:
            out[n] = given[n]
        elif div > 0 and (m * ts) % div == 0:
            out[n] = m * ts // div
        else:
            return dict(strides)
    return out


def _lse_graph_form(req: AttnBwdRequest) -> tuple[tuple, tuple]:
    """hipDNN dims and strides of the request's LSE (always ``[B, H_q, S_q, 1]``)."""
    dims = (req.b, req.h_q, req.s_q, 1)
    st = tuple(req.lse_strides)
    if len(req.lse_dims) == 4 or (not req.lse_dims and not req.is_thd):
        if not st:
            st = (req.h_q * req.s_q, req.s_q, 1, 1)
        return tuple(req.lse_dims) or dims, st
    if len(req.lse_dims) not in (0, 3) or len(st) not in (0, 3):
        raise ValueError(f"LSE dims {req.lse_dims} have no hipDNN spelling")
    t, h = (st[0], st[1]) if st else (req.h_q, 1)
    return dims, (req.s_q * t, h, t, 1)


# ---------------------------------------------------------------------------
# verdict and plan record
# ---------------------------------------------------------------------------


def tuning_key_string(key: tuple | None) -> str:
    """The KMD spelling of a tuned-table key (``""``: the class default)."""
    return "" if not key else "|".join(str(x) for x in key)


def tuned_keys(arch: str) -> tuple[tuple[int, tuple, str], ...]:
    """``(id, key, string)`` of every general tuned-table key of ``arch``:
    ids are 1-based positions in string order (0 is the class default)."""
    mod = ARCH_TUNING.get(arch)
    keys = [] if mod is None else list(mod.GENERAL_TUNED_TABLE)
    ordered = sorted(keys, key=tuning_key_string)
    return tuple((i + 1, k, tuning_key_string(k)) for i, k in enumerate(ordered))


def tuning_key_id(arch: str, key: tuple | None) -> int:
    if not key:
        return 0
    for i, k, _s in tuned_keys(arch):
        if k == key:
            return i
    raise KeyError(f"{key!r} is not a {arch} tuned key")


def plan_table_key(plan: AttnBwdPlan) -> tuple | None:
    """The tuned-table key the plan's knobs came from (``None``: defaults)."""
    return attn_bwd_table_key(engine_policy().tuned_table, plan.tuning_key)


def engine_verdict(
    req: AttnBwdRequest, arch: str
) -> tuple[str, str, AttnBwdPlan | None]:
    """``(code, detail, plan)`` of the engine for ``req`` on ``arch``: ``"OK"``
    with the plan, or the first decline's code without one."""
    if arch not in ENGINE_ARCHS:
        return (
            "ARCH_UNSUPPORTED",
            f"{ENGINE_NAME} ships {' and '.join(ENGINE_ARCHS)}",
            None,
        )
    policy = engine_policy()
    try:
        plan = attn_bwd_plan(req, arch, policy=policy)
        code, detail = "OK", ""
    except ValueError as exc:
        plan = None
        code, _, detail = str(exc).partition(": ")
    verdict = attn_bwd_support(req, arch, policy=policy)
    if verdict.ok != (plan is not None) or (plan is None and verdict.code != code):
        raise AssertionError(
            f"attn_bwd_support ({verdict.code}) and attn_bwd_plan ({code}) disagree"
        )
    return code, detail, plan


# workspace region names of the engine (attn_bwd_workspace_layout names)
WS_REGIONS = {
    "WS_DQ": "dq_acc",
    "WS_LSE2": "lse2",
    "WS_DSUM": "dsum",
    "WS_DK": "dk_acc",
    "WS_DV": "dv_acc",
    "WORKLIST": "worklist",
}
ENCODINGS = {
    "dtype": {"fp16": 1, "bf16": 2},
    "seq_mode": {"batched": 0, "thd": 1},
    "dkv_mode": {"direct": 0, "atomic": 1},
    "mask_class": {"none": 0, "band": 1},
    "dq_mode": {"atomic": 0, "split": 1},
    "main_static_scale": {s: i for i, s in enumerate(STATIC_SCALES)},
}
PREP_TOKENS = (
    "dtype",
    "head_size",
    "seq_mode",
    "stage_vec",
    "worklist",
    "rows_per_cta",
    "narrow_vec",
)
MAIN_TOKENS = (
    "dtype",
    "head_size",
    "seq_mode",
    "stage_vec",
    "dkv_mode",
    "mask_class",
    "dq_mode",
    "head_pack",
    "lb_mirror",
    "lb_interleave",
    "tuning_key",
    "main_block_n",
    "main_worklist",
    "main_static_scale",
)


def _encode(field: str, value) -> int:
    if field in ENCODINGS:
        return ENCODINGS[field][value]
    if isinstance(value, bool):
        return int(value)
    return int(value)


def prep_fields(plan: AttnBwdPlan) -> dict:
    """The prep KMD fields of the plan's prep kernel (raw values)."""
    p = plan.specs["prep"]
    return {
        "dtype": p.dtype,
        "head_size": p.head_size,
        "seq_mode": p.seq_mode,
        "stage_vec": p.stage_vec,
        "worklist": p.worklist,
        "rows_per_cta": attn_bwd_aux_rows_per_cta(p, arch=plan.arch),
        "narrow_vec": p.narrow_vec,
    }


def main_fields(plan: AttnBwdPlan) -> dict:
    """The main KMD fields of the plan's main kernel (raw values; the tuning
    key as its string)."""
    m = plan.specs["main"]
    return {
        "dtype": m.dtype,
        "head_size": m.head_size,
        "seq_mode": m.seq_mode,
        "stage_vec": m.stage_vec,
        "dkv_mode": m.dkv_mode,
        "mask_class": m.mask_class,
        "dq_mode": m.dq_mode,
        "head_pack": bool(m.head_pack),
        "lb_mirror": bool(m.lb_mirror),
        "lb_interleave": bool(m.lb_interleave),
        "tuning_key": tuning_key_string(plan_table_key(plan)),
        "main_block_n": m.block_n,
        "main_worklist": bool(m.worklist),
        "main_static_scale": m.static_scale,
    }


def _launch_name(step) -> tuple[str, str]:
    if step.stage == "prep":
        return "prep", ""
    if step.stage == "main":
        return "main", "main"
    if step.stage == "dq":
        return "main", "dq"
    if step.stage == "convert":
        dst = step.pointers["DST"]
        return "main", {"dq": "dq", "dk": "convert_dk", "dv": "convert_dv"}[dst]
    raise ValueError(f"stage {step.stage!r} is not an engine launch")


def _pointer_role(value: str | None) -> str | None:
    if value is None:
        return None
    if value in WS_REGIONS:
        return f"ws.{WS_REGIONS[value]}"
    return value


def _f32_bits(x: float) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', float(x)))[0]:08x}"


def plan_record(plan: AttnBwdPlan, req: AttnBwdRequest, arch: str) -> dict:
    """The engine's expected outcome of an accepted request: graph-match tokens,
    workspace layout and every launch with its kernargs in ABI order."""
    del req
    if plan.arch != arch:
        raise ValueError(f"plan of {plan.arch}, not {arch}")
    total = int(plan.workspace_bytes)
    regions = {
        WS_REGIONS[name]: [int(off), int(nbytes)]
        for name, (off, nbytes) in plan.workspace_layout.items()
    }
    tokens = {"ws.bytes": total, "ws.scratch_offset": total}
    for name, (off, _n) in regions.items():
        tokens[f"ws.{name}.offset"] = off
    for f, v in prep_fields(plan).items():
        tokens[f"attention_general_bwd.prep.{f}"] = _encode(f, v)
    mf = main_fields(plan)
    for f, v in mf.items():
        if f == "tuning_key":
            v = tuning_key_id(arch, plan_table_key(plan))
        tokens[f"attention_general_bwd.main.{f}"] = _encode(f, v)
    launches = []
    for step in plan.launches:
        slot, launch = _launch_name(step)
        kernargs = []
        for name, kind in attn_bwd_params(step.stage):
            if kind == "Q":
                kernargs.append([name, kind, _pointer_role(step.pointers[name])])
            elif kind == "f":
                kernargs.append([name, kind, _f32_bits(step.scalars[name])])
            else:
                value = step.scalars[name]
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{name}: non-finite integer kernarg")
                kernargs.append([name, kind, int(value)])
        launches.append(
            {
                "slot": slot,
                "launch": launch,
                "kernel": step.kernel,
                "grid": [int(x) for x in step.grid],
                "block": int(step.block),
                "kernargs": kernargs,
            }
        )
    return {
        "code": "OK",
        "tokens": tokens,
        "workspace": {"bytes": total, "scratch_offset": total, "regions": regions},
        "launches": launches,
    }
