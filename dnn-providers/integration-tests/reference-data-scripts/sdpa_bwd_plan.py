#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declarations of the SDPA backward test plan.

This module is the machine-readable half of ``docs/test-plans/sdpa-bwd.md``.
The bundle generator (``generate_sdpa_bwd_bundles.py``), its tests and the
exhaustive tool read it:

* the levels, the baseline switches and the window semantics (SEM-001);
* the data types, tensor uids, topologies and the stride kinds of every layout;
* the band model (W, R, alignment) and the deprecated causal flags;
* the cost model, the per-case ceilings and the cumulative budgets;
* the legality layers (F, WF, EXT, U), copied from the frontend rules;
* the oracle gates, the purpose rule and the value domains;
* the tolerance classes, their assignment rule and the magnitude estimator;
* the all-pairs axes and the pairwise-coverage check;
* the feature inventory, the shape catalog (S, H, X, M, L) and the golden set.

It contains no engine knowledge. A plan item is a plain ``dict`` built by
``new``; it becomes one case per data type.

Band conventions (plan §1.2):

* ``W`` is the number of keys kept per query row, including the diagonal key;
  ``None`` is unbounded left.
* ``R`` is the number of keys kept right of the diagonal; ``None`` is
  unbounded and 0 is causal.
* Row ``i`` has its diagonal key at ``i + off``, ``off = 0`` (TOP_LEFT) or
  ``Skv - Sq`` (BOTTOM_RIGHT). Key ``j`` is kept when ``j <= i + off + R`` and
  ``j >= i + off - W + 1``.
* The graph's ``left_bound`` is ``W`` since SEM-001 (#14483); before it,
  develop kept ``left_bound + 1`` keys and the graph needed ``W - 1``.
  ``WINDOW_SEMANTICS`` selects the one the generator writes; case ids name
  ``W``, so they do not change when it flips.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

# --------------------------------------------------------------------------
# Levels and baseline
# --------------------------------------------------------------------------

LEVELS = ("quick", "pr", "nightly", "full")
LEVEL_DIR = {
    "quick": "quick",
    "pr": "standard",
    "nightly": "comprehensive",
    "full": "full",
}
NOTRUN = "notrun"  # declared, not written: the case waits for a gate (plan §4.1)

# The generator's baseline is develop, which has #14312 (O and Stats derived
# from the forward in the fill) and #14483 (SEM-001: left_bound = L keeps L
# keys). Each switch flips when its pull request lands.
WINDOW_SEMANTICS = "after_sem001"  # "develop": left_bound = W - 1; "after_sem001": W
WRITE_EXPECTED_REJECTION = False  # #14346 adds metadata.expected_rejection
WRITE_GOLDEN_POINTERS = False  # gate GT1: golden data lands after #14358's lane

# --------------------------------------------------------------------------
# Data types, tensors and topologies
# --------------------------------------------------------------------------

DTYPE_NAME = {
    "bf16": "bfloat16",
    "fp16": "half",
    "fp32": "float",
    "fp8_e4m3": "fp8_e4m3",
}
DTYPE_TOKEN = {"bf16": "bfp16", "fp16": "fp16", "fp32": "fp32"}  # case-id token
BYTES = {
    "bfloat16": 2,
    "half": 2,
    "float": 4,
    "fp8_e4m3": 1,
    "int32": 4,
    "int64": 8,
    "boolean": 1,
}

IO = ("Q", "K", "V", "O", "dO", "dQ", "dK", "dV")
OUTPUTS = ("dQ", "dK", "dV", "dBias")
# 0-15 follow #14358's generate_sdpa_bwd_golden.py, so its golden bins
# (tensor<uid>.bin) attach unchanged.
# fmt: off
UID = {
    "K": 0, "O": 1, "Q": 2, "dK": 3, "dQ": 4, "dV": 5, "V": 6, "dO": 7, "Stats": 8,
    "Bias": 9, "SeqLenQ": 10, "SeqLenKv": 11, "DropoutMask": 12, "DropoutScale": 13,
    "DropoutScaleInv": 14, "dBias": 15, "Scale": 16, "Seed": 17, "Offset": 18,
    "RaggedOffsetQ": 19, "RaggedOffsetK": 20, "RaggedOffsetV": 21, "RaggedOffsetO": 22,
}

# Template directories under <level>/SdpaBackward/ (plan §4) and the
# optional tensors each one adds to Default.
TOPOLOGIES = OrderedDict([
    ("Default", ()),
    ("AdditiveMask", ("Bias",)),
    ("AdditiveMaskDBias", ("Bias", "dBias")),
    ("DBiasOnly", ("dBias",)),
    ("ScaleTensor", ("Scale",)),  # a compile-time constant 0.5; runtime cases patch it
    ("DropoutScaleInv", ("DropoutScaleInv",)),
    ("SeqLenQOnly", ("SeqLenQ",)),
    ("DropoutMask", ("DropoutMask", "DropoutScale", "DropoutScaleInv")),
    ("Philox", ("Seed", "Offset")),
    ("Padding", ("SeqLenQ", "SeqLenKv")),
    ("Ragged", ("RaggedOffsetQ", "RaggedOffsetK", "RaggedOffsetV", "RaggedOffsetO")),
    ("FwdBwdChained", ()),  # SdpaFwd -> SdpaBackward; O and Stats are virtual
])
# fmt: on

# --------------------------------------------------------------------------
# Case model
# --------------------------------------------------------------------------

ISQ = "inv_sqrt"  # attn_scale_value = 1 / sqrt(Dqk)

GEOMETRY = ("B", "Hq", "Hk", "Hv", "Sq", "Skv", "Dqk", "Dv")
# The default shape (§3): g = 1 and Dqk = Dv (deviation DV-1).
DEFAULT = dict(B=2, Hq=3, Hk=3, Hv=3, Sq=72, Skv=120, Dqk=128, Dv=128)

CASE_DEFAULTS = dict(
    set="S",  # size class: S | L | M | H | X
    level=None,  # quick | pr | nightly | full | notrun
    purpose=None,  # declared claim | probe | regression; the purpose rule decides probes
    items=(),  # plan rows, feature ids and shape ids the case covers
    dtypes=("bf16",),
    align="TL",
    W=None,
    R=None,
    scale=ISQ,  # ISQ | "unset" | float | {"tensor": "runtime" | "const", "value": v}
    layout="bhsd",  # a LAYOUTS token
    tensor_layout=None,  # {tensor: stride kind}, applied after `layout`
    stats="packed",  # a STATS token
    inputs=None,  # {q, k, v, do, bias: [lo, hi] | {"kind": "fixed", "value": v} | text}
    topology=None,  # None: Default (ScaleTensor with a scale tensor)
    attrs=None,  # causal_mask, causal_mask_bottom_right, alibi_mask, dropout_probability,
    # right_bound_raw (a rejection probe's raw bound), explicit_unbounded
    bias=None,  # {dims: token | [dims], dtype: io | fp32 | fp16 | bf16 | bool, stride}
    dbias=False,
    dbias_stride=None,
    dropout=None,  # {p, scale, scale_inv, mask_dims}
    padding=None,  # {seq_len_q, seq_len_kv, padding_mask}
    ragged=None,
    dtype_over=None,  # {tensor: dtype}
    dims_over=None,  # {tensor: dv_half | sq_half}
    alignment=16,  # declared byte alignment of every tensor
    oracle=None,  # None: ready; else a GATES key
    rejection=False,  # rejection probe (layer U)
    defect=None,  # open engine defect the probe pins (§4.2)
    regression_of=None,  # escaped bug the case pins
    golden=None,  # GL row
    seed=0,
    repeat=None,
)


def new(**kw):
    """A plan item with every field present."""
    it = dict(CASE_DEFAULTS)
    it.update(DEFAULT)
    it.update(kw)
    it["items"] = list(it["items"])
    it["dtypes"] = list(it["dtypes"])
    return it


# --------------------------------------------------------------------------
# Band
# --------------------------------------------------------------------------


def mask_kind(W, R):
    if W is None and R is None:
        return "none"
    if W is None:
        return "causal" if R == 0 else "right_only"
    if R is None:
        return "left_only"
    return "window" if R == 0 else "band"


def effective_band(c):
    """(W, R, align) after the deprecated flags (SEM-002): causal_mask sets
    R = R if R >= 0 else 0 and keeps the alignment; causal_mask_bottom_right
    also sets BOTTOM_RIGHT."""
    W, R, al = c.get("W"), c.get("R"), c.get("align", "TL")
    at = c.get("attrs") or {}
    if at.get("causal_mask") or at.get("causal_mask_bottom_right"):
        R = R if (R is not None and R >= 0) else 0
        if at.get("causal_mask_bottom_right"):
            al = "BR"
    return W, R, al


def _clamp_sum(x, n):
    """The sum of min(max(y, 0), n) over the integers y <= x."""
    if x <= 0:
        return 0
    if x <= n:
        return x * (x + 1) // 2
    return n * (n + 1) // 2 + (x - n) * n


def _pairs_upto(t, Sq, Skv):
    """(i, j) pairs, 0 <= i < Sq, 0 <= j < Skv, with j - i <= t."""
    return _clamp_sum(t + Sq, Skv) - _clamp_sum(t, Skv)


def kept_pairs(Sq, Skv, W=None, R=None, align="TL"):
    """Kept (q, k) pairs of one (batch, head)."""
    if W is None and R is None:
        return Sq * Skv
    off = 0 if align == "TL" else Skv - Sq
    hi = Sq * Skv if R is None else _pairs_upto(off + R, Sq, Skv)
    lo = 0 if W is None else _pairs_upto(off - W, Sq, Skv)
    return max(0, hi - lo)


def bounds_written(c, semantics=None):
    """(left_bound, right_bound) the graph writes. A case with
    attrs.explicit_unbounded writes -1 for both; a rejection probe writes its
    raw right bound."""
    at = c.get("attrs") or {}
    if at.get("explicit_unbounded"):
        return -1, -1
    W = c.get("W")
    sem = semantics or WINDOW_SEMANTICS
    lb = None if W is None else (W - 1 if sem == "develop" else W)
    return lb, at.get("right_bound_raw", c.get("R"))


def left_bound_written(c):
    """The left bound under both semantics."""
    if (c.get("attrs") or {}).get("explicit_unbounded"):
        return {"develop": -1, "after_sem001": -1}
    W = c.get("W")
    return None if W is None else {"develop": W - 1, "after_sem001": W}


# --------------------------------------------------------------------------
# Layouts and strides
# --------------------------------------------------------------------------


def packed(dims):
    st = [1] * len(dims)
    for i in range(len(dims) - 2, -1, -1):
        st[i] = st[i + 1] * dims[i + 1]
    return st


# fmt: off
# Strides of a rank-4 [B, H, S, D] IO tensor for each stride kind.
STRIDE_KINDS = {
    "bhsd":          lambda B, H, S, D: [H * S * D, S * D, D, 1],
    "bshd":          lambda B, H, S, D: [S * H * D, D, H * D, 1],
    "rowpad8":       lambda B, H, S, D: [H * S * (D + 8), S * (D + 8), D + 8, 1],
    "rowpad1":       lambda B, H, S, D: [H * S * (D + 1), S * (D + 1), D + 1, 1],
    "dstride2":      lambda B, H, S, D: [H * S * D * 2, S * D * 2, D * 2, 2],
    "sd_transposed": lambda B, H, S, D: [H * S * D, S * D, 1, S],
    "fa_transpose":  lambda B, H, S, D: [2 * H * D, D, 2 * B * H * D, 1],
    "qkv3":          lambda B, H, S, D: [S * 3 * H * D, D, 3 * H * D, 1],
    "bs3hd":         lambda B, H, S, D: [S * 3 * H * D, D, 3 * H * D, 1],
    "perm_bsh":      lambda B, H, S, D: [S * H * D, D, H * D, 1],
    "perm_hbs":      lambda B, H, S, D: [S * D, B * S * D, D, 1],
    "perm_hsb":      lambda B, H, S, D: [D, S * B * D, B * D, 1],
    "perm_sbh":      lambda B, H, S, D: [H * D, D, B * H * D, 1],
    "perm_shb":      lambda B, H, S, D: [D, B * D, H * B * D, 1],
    "gaps":          lambda B, H, S, D: [H * (S + 5) * (D + 7) + 3, (S + 5) * (D + 7), D + 7, 1],
    "overlap_rows":  lambda B, H, S, D: [H * (S * (D // 2) + D // 2), S * (D // 2) + D // 2, D // 2, 1],
    "bstride0":      lambda B, H, S, D: [0, S * D, D, 1],
}

# IO layout token -> stride kind per tensor (tensors not listed: bhsd)
LAYOUTS = {
    "bhsd": {},
    "bshd": {t: "bshd" for t in IO},
    "do_bshd": {"dO": "bshd"},
    "grads_bshd": {"dQ": "bshd", "dK": "bshd", "dV": "bshd"},
    "rowpad8": {t: "rowpad8" for t in IO},
    "rowpad1": {t: "rowpad1" for t in IO},
    "bs3hd": {t: "bs3hd" for t in ("Q", "K", "V", "dQ", "dK", "dV")},
    "perm6": dict(zip(IO, ("perm_bsh", "perm_hbs", "perm_hsb", "perm_sbh", "perm_shb", "bhsd",
                           "perm_bsh", "perm_hbs"))),
    "gaps": {t: "gaps" for t in IO},
}

# Stats layouts: (dims, strides) from B, Hq, Sq
STATS = {
    "packed":     lambda B, H, S: ([B, H, S, 1], [H * S, S, 1, 1]),
    "rank3":      lambda B, H, S: ([B, H, S], [H * S, S, 1]),
    "headpad64":  lambda B, H, S: ([B, H, S, 1], [H * (S + 64), S + 64, 1, 1]),
    "bsh":        lambda B, H, S: ([B, H, S, 1], [S * H, 1, H, 1]),
    "seqstride2": lambda B, H, S: ([B, H, S, 1], [H * S * 2, S * 2, 2, 1]),
    "batchpad":   lambda B, H, S: ([B, H, S, 1], [H * S + 64, S, 1, 1]),
}

# Bias dims tokens (right-aligned broadcast to [B, Hq, Sq, Skv])
BIAS_DIMS = {
    "11":   lambda B, H, Sq, Skv: [1, 1, Sq, Skv],
    "1h":   lambda B, H, Sq, Skv: [1, H, Sq, Skv],
    "b1":   lambda B, H, Sq, Skv: [B, 1, Sq, Skv],
    "full": lambda B, H, Sq, Skv: [B, H, Sq, Skv],
    "r1":   lambda B, H, Sq, Skv: [Skv],
    "r2":   lambda B, H, Sq, Skv: [Sq, Skv],
    "r3":   lambda B, H, Sq, Skv: [H, Sq, Skv],
    "row":  lambda B, H, Sq, Skv: [1, 1, Sq, 1],
    "col":  lambda B, H, Sq, Skv: [1, 1, 1, Skv],
}
BIAS_DTYPE = {"fp32": "float", "fp16": "half", "bf16": "bfloat16", "bool": "boolean"}
# fmt: on


def strides_for(kind, B, H, S, D):
    return STRIDE_KINDS[kind](B, H, S, D)


def bias_dims(tok, B, Hq, Sq, Skv):
    return (
        list(tok) if isinstance(tok, (list, tuple)) else BIAS_DIMS[tok](B, Hq, Sq, Skv)
    )


def bias_strides(kind, dims):
    """packed, a padded row pitch (Skv + 8), or transposed (Sq, Skv swapped)."""
    if kind in (None, "packed") or len(dims) < 2:
        return packed(dims)
    st = [1] * len(dims)
    if kind == "rowpad8":
        st[-2] = dims[-1] + 8
        acc = st[-2] * dims[-2]
    elif kind == "transposed":
        st[-1], st[-2] = dims[-2], 1
        acc = dims[-2] * dims[-1]
    else:
        raise KeyError(kind)
    for i in range(len(dims) - 3, -1, -1):
        st[i] = acc
        acc *= dims[i]
    return st


def _tensor(name, dims, strides, layout, dtype):
    return {
        "uid": UID[name],
        "dims": dims,
        "strides": strides,
        "layout": layout,
        "data_type": dtype,
    }


def tensor_table(c, dt):
    """Every tensor of the case: name -> {uid, dims, strides, layout,
    data_type, alignment}; a scale tensor also has pass_by_value (and value)."""
    B, Hq, Hk, Hv, Sq, Skv, Dqk, Dv = (c[k] for k in GEOMETRY)
    dims = {
        "Q": [B, Hq, Sq, Dqk],
        "K": [B, Hk, Skv, Dqk],
        "V": [B, Hv, Skv, Dv],
        "O": [B, Hq, Sq, Dv],
        "dO": [B, Hq, Sq, Dv],
        "dQ": [B, Hq, Sq, Dqk],
        "dK": [B, Hk, Skv, Dqk],
        "dV": [B, Hv, Skv, Dv],
    }
    for t, how in (c.get("dims_over") or {}).items():
        d = list(dims[t])
        d[3 if how == "dv_half" else 2] //= 2  # dv_half | sq_half
        dims[t] = d
    kinds = {t: "bhsd" for t in IO}
    kinds.update(LAYOUTS[c.get("layout", "bhsd")])
    kinds.update(c.get("tensor_layout") or {})
    io = DTYPE_NAME[dt]
    over = c.get("dtype_over") or {}
    out = OrderedDict()
    for t in IO:
        out[t] = _tensor(
            t,
            dims[t],
            strides_for(kinds[t], *dims[t]),
            kinds[t],
            DTYPE_NAME.get(over.get(t), io),
        )
    st = c.get("stats", "packed")
    sd, ss = STATS[st](B, Hq, Sq)
    out["Stats"] = _tensor(
        "Stats", sd, ss, st, "bfloat16" if over.get("Stats") == "bf16" else "float"
    )
    one = [1, 1, 1, 1]
    b = c.get("bias")
    if b:
        bd = bias_dims(b["dims"], B, Hq, Sq, Skv)
        bdt = io if b["dtype"] == "io" else BIAS_DTYPE[b["dtype"]]
        bk = b.get("stride", "packed")
        out["Bias"] = _tensor("Bias", bd, bias_strides(bk, bd), bk, bdt)
        if c.get("dbias"):
            dk = c.get("dbias_stride") or "packed"
            out["dBias"] = _tensor("dBias", bd, bias_strides(dk, bd), dk, bdt)
    elif c.get("dbias"):
        bd = [1, Hq, Sq, Skv]
        out["dBias"] = _tensor("dBias", bd, packed(bd), "packed", "float")
    s = c.get("scale")
    if isinstance(s, dict):  # a runtime value comes from metadata.inputs
        out["Scale"] = _tensor("Scale", [1], [1], "scalar", "float")
        if s["tensor"] == "runtime":
            out["Scale"]["pass_by_value"] = "runtime"
        else:
            out["Scale"].update(pass_by_value="constant", value=s["value"])
    pad = c.get("padding")
    if pad:
        out["SeqLenQ"] = _tensor("SeqLenQ", [B, 1, 1, 1], one, "packed", "int32")
        if pad.get("seq_len_kv") is not None:
            out["SeqLenKv"] = _tensor("SeqLenKv", [B, 1, 1, 1], one, "packed", "int32")
    if c.get("dropout"):
        md = bias_dims(c["dropout"]["mask_dims"], B, Hq, Sq, Skv)
        out["DropoutMask"] = _tensor("DropoutMask", md, packed(md), "packed", "float")
        out["DropoutScale"] = _tensor("DropoutScale", one, one, "packed", "float")
        out["DropoutScaleInv"] = _tensor("DropoutScaleInv", one, one, "packed", "float")
    topo = c.get("topology")
    if topo == "DropoutScaleInv":
        out["DropoutScaleInv"] = _tensor("DropoutScaleInv", one, one, "packed", "float")
    if topo == "Philox":
        out["Seed"] = _tensor("Seed", one, one, "packed", "int64")
        out["Offset"] = _tensor("Offset", one, one, "packed", "int64")
    if c.get("ragged"):
        for n in ("RaggedOffsetQ", "RaggedOffsetK", "RaggedOffsetV", "RaggedOffsetO"):
            out[n] = _tensor(n, [B + 1, 1, 1, 1], one, "packed", "int32")
    for t in out.values():
        t["alignment"] = c.get("alignment", 16)
    return out


def fp32_bytes(tensors):
    """fp32-equivalent bytes of every tensor (the scale scalar excluded)."""
    return sum(4 * math.prod(t["dims"]) for n, t in tensors.items() if n != "Scale")


def topology_of(c):
    if c.get("topology"):
        return c["topology"]
    return "ScaleTensor" if isinstance(c.get("scale"), dict) else "Default"


def scale_value(c):
    s = c.get("scale", ISQ)
    if s == ISQ:
        return 1.0 / math.sqrt(c["Dqk"])
    if s == "unset":
        return 1.0
    return float(s["value"] if isinstance(s, dict) else s)


# --------------------------------------------------------------------------
# Cost model (§3, §4.4; RFC 0022 §7): one model for the case list and the golden
# set. Wall time of one case with the CPU reference as the oracle:
#   pilot (binding): t = 4.6e-8 * Dv * T + 8.7e-8 * (3*Dqk + Dv)/4 * U + 0.14 s
#   alt:  t = 27.9e-9 * f * (U*(3*Dqk + Dv) + T*Dv) + 1e-8 * (Dqk + Dv)/2 * T + 0.08 s,
#         f = 1.10 for fp16
#   GPU-reference era (lower bound): 1e-8 * (Dqk + Dv)/2 * T + 0.08 s
# T = N_pairs = B*Hq*Sq*Skv; U = N_live = B*Hq*kept pairs. A repeated case
# costs repeat x t. The ceilings and budgets use max(pilot, alt).
# --------------------------------------------------------------------------

PILOT_T, PILOT_U, PILOT_C = 4.6e-8, 8.7e-8, 0.14
ALT_K, ALT_ENG, ALT_C = 27.9e-9, 1.0e-8, 0.08
FP16_REF_FACTOR = 1.10
GPU_FWD_NS_PER_MAC, FWD_EFF_THREADS = 30e-9, 20

MIB = 1 << 20
# Per-case ceilings (seconds, fp32-equivalent bytes) and cumulative budgets
# (seconds, every case of the level and the levels below accepted).
CEIL_S = {"quick": 1.5, "pr": 5.0, "nightly": 75.0, "full": 300.0}
CEIL_BYTES = {"quick": 4 * MIB, "pr": 16 * MIB, "nightly": 64 * MIB, "full": None}
CUM_S = {"quick": 20.0, "pr": 120.0, "nightly": 480.0, "full": 1800.0}
RESERVE = 0.05  # each cumulative budget keeps 5 % for model error


def pilot_seconds(T, U, Dqk, Dv):
    return PILOT_T * Dv * T + PILOT_U * (3 * Dqk + Dv) / 4.0 * U + PILOT_C


def alt_seconds(T, U, Dqk, Dv, fp16=False):
    ref = ALT_K * (U * (3 * Dqk + Dv) + T * Dv) * (FP16_REF_FACTOR if fp16 else 1.0)
    return ref + ALT_ENG * (Dqk + Dv) / 2.0 * T + ALT_C


def shape_cost(B, Hq, Hk, Hv, Sq, Skv, Dqk, Dv, W=None, R=None, align="TL", U=None):
    if U is None:
        U = B * Hq * kept_pairs(Sq, Skv, W, R, align)
    T = B * Hq * Sq * Skv
    io = 2 * (
        B * Hq * Sq * Dqk + B * Hk * Skv * Dqk + B * Hv * Skv * Dv + B * Hq * Sq * Dv
    )
    return {
        "n_pairs": T,
        "n_live": U,
        "flop": 2 * U * (3 * Dqk + 2 * Dv),
        "bytes": 4 * (io + B * Hq * Sq),
        "s": pilot_seconds(T, U, Dqk, Dv),
        "s_alt": alt_seconds(T, U, Dqk, Dv),
        "s_alt_fp16": alt_seconds(T, U, Dqk, Dv, fp16=True),
        "s_gpuref": 1.0e-8 * (Dqk + Dv) / 2 * T + 0.08,
        "fwd_derivation_s": T * (Dqk + Dv) * GPU_FWD_NS_PER_MAC / FWD_EFF_THREADS,
    }


def case_cost(c):
    """Cost of a case. Under padding the live pairs are counted per batch from
    the sequence lengths (bottom-right offset per batch, SEM-008); N_pairs
    stays dense (the reference visits every score position)."""
    W, R, al = effective_band(c)
    pad = c.get("padding")
    U = None
    if pad and pad.get("seq_len_kv") is not None and len(pad["seq_len_q"]) == c["B"]:
        U = sum(
            c["Hq"] * kept_pairs(sq, skv, W, R, al) if sq and skv else 0
            for sq, skv in zip(pad["seq_len_q"], pad["seq_len_kv"])
        )
    return shape_cost(
        *(c[k] for k in GEOMETRY),
        W,
        R,
        al,
        U=U,
    )


def seconds(c, dt):
    """(pilot, alt, GPU-reference era) seconds, repeat included; the alt
    model uses its fp16 factor for fp16."""
    cc = case_cost(c)
    rpt = c.get("repeat") or 1
    alt = cc["s_alt_fp16"] if dt == "fp16" else cc["s_alt"]
    return round(cc["s"] * rpt, 3), round(alt * rpt, 3), round(cc["s_gpuref"] * rpt, 3)


def budget_seconds(pilot, alt):
    """The per-case and cumulative budget rule (§3, §4.4)."""
    return max(pilot, alt)


def fits_level(cost):
    """Lowest level whose per-case time and byte ceilings hold (pilot)."""
    for lv in LEVELS:
        cb = CEIL_BYTES[lv]
        if cost["s"] <= CEIL_S[lv] and (cb is None or cost["bytes"] <= cb):
            return lv
    return None


# --------------------------------------------------------------------------
# Legality (§2.8). The generator writes claim, regression and probe cases in
# WF or EXT, rejection probes in U, and nothing outside F_active.
#   F         frontend-legal on develop (rules R0-R12, tensor validate)
#   F_R13     F plus R13 (left_bound -1 or >= 1), from #14483 (key after_14287:
#             the closed PR #14483 replaced)
#   F_R13R14  F_R13 plus R14 (right_bound >= -1), from #14358
#   WF        well-formed: F_R13R14 plus the cuDNN tensor tables and support rules
#   EXT       outside WF, with semantics hipDNN defines
#   U         legal, outside WF and EXT: undefined (rejection probes)
# The functions read a case record (the generator's record()).
# --------------------------------------------------------------------------

SEMANTICS = ("develop", "after_14287", "after_14358")

# fmt: off
# (reason, rule, source). The first failing rule names the case's reason.
WF_RULES = [
    ("O/dO dims", "O and dO dims are [B, Hq, Sq, Dv]", "Attention.md:693-705; sdpa_node.h:1334-1336"),
    ("Stats rank/dtype", "Stats is FLOAT [B, Hq, Sq, 1]", "SEM-011; Attention.md:469"),
    ("gradient dims", "dQ, dK, dV dims equal Q, K, V dims", "Attention.md:701-705"),
    ("IO dtype not FP16/BF16", "IO dtype is HALF or BFLOAT16", "Attention.md:693-705"),
    ("mixed IO dtypes", "Q, K, V, O, dO, dQ, dK, dV share one dtype", "Attention.md:693-705"),
    ("innermost stride != 1", "innermost stride 1 on every rank-4 IO tensor", "sdpa_node.h:1284-1288"),
    ("overlapping output strides", "dQ, dK, dV and dBias strides do not overlap",
     "TensorAttributes.hpp:613-655 (no rule)"),
    ("bias rank/broadcast/dtype", "bias rank 4, dim 0 in {B, 1}, dim 1 in {Hq, 1}, HALF or BFLOAT16",
     "Attention.md:452; sdpa_node.h:1448-1450"),
    ("dBias without matching bias", "dBias only with a bias, with the bias dims and dtype", "SEM-014"),
    ("both deprecated flags", "not causal_mask and causal_mask_bottom_right together", "PlanUtils.hpp:46-55"),
    ("ALiBi outside TL causal", "alibi_mask only with R_eff = 0 and TOP_LEFT", "sdpa_node.h:1443-1445; SEM-021"),
    ("dropout_probability without RNG state",
     "0 <= dropout_probability < 1 with seed and offset and no dropout mask", "sdpa_node.h:1524-1530"),
    ("dropout scale without a mask", "dropout scale and scale_inv only with a dropout mask",
     "Attention.md:455-459; SEM-015"),
    ("one seq_len", "seq_len_q and seq_len_kv both or neither", "SEM-008"),
    ("seq_len without padding_mask", "seq_len tensors only with padding_mask", "sdpa_node.h:1474-1476"),
    ("Sq = Skv = 1", "not Sq = Skv = 1", "sdpa_node.h:1339-1341"),
    ("BR band with Sq > Skv", "no BOTTOM_RIGHT band with Sq > Skv without padding", "sdpa_node.h:1484-1486"),
    ("BR band with bias/ALiBi/dropout", "no BOTTOM_RIGHT band with bias, ALiBi or dropout",
     "sdpa_node.h:1488-1490"),
    ("window with Sq > Skv", "no window with Sq > Skv without padding", "sdpa_node.h:1505-1507"),
    ("right_bound < 0 when set", "right_bound >= 0 when set", "sdpa_node.h:1514-1516"),
    ("head dim not a multiple of 8", "Dqk and Dv are multiples of 8", "Attention.md:123"),
    ("FLOAT dropout mask", "dropout mask is HALF or BFLOAT16", "Attention.md:455-459"),
]

# (reason, when hipDNN defines it, source). A WF failure whose reason is not
# here, or whose condition fails, is U.
EXT_RULES = [
    ("IO dtype not FP16/BF16", "every IO tensor FLOAT (FP32 extension)", "SdpaBwdSignatureKey.hpp:144-170"),
    ("innermost stride != 1", "always: strides define the meaning (no stride rule)", "TensorAttributes.hpp:613-655"),
    ("Stats rank/dtype", "FLOAT rank 3 [B, Hq, Sq]", "#14358 SdpaBwdPlan.hpp:24-33"),
    ("bias rank/broadcast/dtype",
     "rank 1-4; each dim, right-aligned to [B, Hq, Sq, Skv], is the full size or 1; FLOAT, HALF or BFLOAT16",
     "SdpaBwdNode.hpp:192-222 (R8)"),
    ("BR band with bias/ALiBi/dropout", "bias or dropout, not ALiBi: bias, then band", "SEM-006"),
    ("seq_len without padding_mask", "always: treated as padding", "SEM-008"),
    ("BR band with Sq > Skv", "always: dead rows add 0", "SEM-010"),
    ("window with Sq > Skv", "always: dead rows add 0", "SEM-010"),
    ("Sq = Skv = 1", "always", "hipDNN keeps it legal"),
    ("right_bound < 0 when set", "right_bound = -1 (unset)", "PlanUtils.hpp:37-42"),
    ("head dim not a multiple of 8", "always: engines decide", "SdpaBwdNode.hpp:118-125"),
    ("FLOAT dropout mask", "always: the references need FLOAT", "SEM-015"),
    ("dropout_probability without RNG state", "dropout_probability = 0", "develop references: no-op"),
]
# fmt: on


def _rec_left_bound(r, semantics):
    lb = r.get("left_bound_written")
    if isinstance(lb, dict):
        return lb["develop"] if semantics == "develop" else lb["after_sem001"]
    return lb


def _rec_band(r):
    a = r.get("attributes") or {}
    R = r.get("right_bound")
    al = r["diagonal_alignment"]
    if a.get("causal_mask") or a.get("causal_mask_bottom_right"):
        R = R if (R is not None and R >= 0) else 0
        if a.get("causal_mask_bottom_right"):
            al = "BOTTOM_RIGHT"
    return r.get("W"), R, al


def frontend_legal(r, semantics="develop"):
    """F (develop), F_R13 (after_14287) or F_R13R14 (after_14358): (ok, reason)."""
    T = r["strides"]
    for n in ("Q", "K", "V", "O", "dO", "Stats", "dQ", "dK", "dV"):
        if n not in T:
            return False, f"R1: {n} missing"
    if not (len(T["Q"]["dims"]) == len(T["K"]["dims"]) == len(T["V"]["dims"]) == 4):
        return False, "R2"
    B, Hq, Sq, Dqk = T["Q"]["dims"]
    Bk, Hk, Skv, Dk = T["K"]["dims"]
    Bv, Hv, Skv2, _ = T["V"]["dims"]
    if not (B == Bk == Bv and Dqk == Dk and Skv == Skv2):
        return False, "R3-R5"
    if not (Hk > 0 and Hv > 0 and Hq % Hk == 0 and Hq % Hv == 0):
        return False, "R6"
    if T["dO"]["dims"] != T["O"]["dims"] or len(T["dO"]["dims"]) != 4:
        return False, "R7"
    b = T.get("Bias")
    if b:
        d = b["dims"]
        if (
            len(d) > 4
            or d[-1] not in (Skv, 1)
            or (len(d) >= 2 and d[-2] not in (Sq, 1))
        ):
            return False, "R8"
    m = T.get("DropoutMask")
    if m:
        d = m["dims"]
        if (
            len(d) != 4
            or d[0] not in (B, 1)
            or d[1] not in (Hq, 1)
            or d[2:] != [Sq, Skv]
        ):
            return False, "R9"
    pad = r.get("padding")
    if pad and pad.get("padding_mask", True):
        for n in ("SeqLenQ", "SeqLenKv"):
            t = T.get(n)
            if not t or t["dims"] != [B, 1, 1, 1] or t["data_type"] != "int32":
                return False, "R10"
    s = T.get("Scale")
    if s and (
        r["scale"]["mode"] == "attn_scale_value"
        or s.get("pass_by_value") not in ("runtime", "constant")
    ):
        return False, "R11/R12"
    for n, t in T.items():
        if (
            len(t["dims"]) != len(t["strides"])
            or any(x <= 0 for x in t["dims"])
            or t.get("alignment", 16) < 1
        ):
            return False, f"tensor validate ({n})"
    if semantics in ("after_14287", "after_14358"):
        lb = _rec_left_bound(r, semantics)
        if lb is not None and (lb == 0 or lb < -1):
            return False, "R13"
    if semantics == "after_14358":
        rb = r.get("right_bound_written")
        if rb is not None and rb < -1:
            return False, "R14"
    return True, ""


def _overlaps(t):
    """Two logical elements of a tensor share an address (positive strides)."""
    dims, st = t["dims"], t["strides"]
    extent = 1
    for i in sorted(range(len(dims)), key=lambda i: st[i]):
        if dims[i] == 1:
            continue
        if st[i] < extent:
            return True
        extent = st[i] * (dims[i] - 1) + extent
    return False


def _bias_broadcast_ok(b, B, Hq, Sq, Skv):
    full = [B, Hq, Sq, Skv][4 - len(b["dims"]) :]
    return all(d in (f, 1) for d, f in zip(b["dims"], full))


def well_formed(r):
    """WF (WF_RULES): (ok, reason)."""
    T = r["strides"]
    B, Hq, Sq, Dqk = T["Q"]["dims"]
    Skv, Dv = T["V"]["dims"][2], T["V"]["dims"][3]
    a = r.get("attributes") or {}
    W, R, al = _rec_band(r)
    io = T["Q"]["data_type"]
    b, db, mask = T.get("Bias"), T.get("dBias"), T.get("DropoutMask")
    p = a.get("dropout_probability")
    br_band = R is not None and al == "BOTTOM_RIGHT"
    padded = "SeqLenQ" in T
    rb = r.get("right_bound_written")
    bias_ok = (
        len(b["dims"]) == 4 and b["dims"][0] in (B, 1) and b["dims"][1] in (Hq, 1)
        if b
        else True
    )
    checks = {
        "O/dO dims": T["O"]["dims"] == [B, Hq, Sq, Dv]
        and T["dO"]["dims"] == [B, Hq, Sq, Dv],
        "Stats rank/dtype": T["Stats"]["dims"] == [B, Hq, Sq, 1]
        and T["Stats"]["data_type"] == "float",
        "gradient dims": all(
            T["d" + n]["dims"] == T[n]["dims"] for n in ("Q", "K", "V")
        ),
        "IO dtype not FP16/BF16": io in ("half", "bfloat16"),
        "mixed IO dtypes": all(T[n]["data_type"] == io for n in IO),
        "innermost stride != 1": all(T[n]["strides"][3] == 1 for n in IO),
        "overlapping output strides": not any(
            _overlaps(T[n]) for n in OUTPUTS if n in T
        ),
        "bias rank/broadcast/dtype": not b
        or (bias_ok and b["data_type"] in ("half", "bfloat16")),
        "dBias without matching bias": not db
        or bool(b and db["dims"] == b["dims"] and db["data_type"] == b["data_type"]),
        "both deprecated flags": not (
            a.get("causal_mask") and a.get("causal_mask_bottom_right")
        ),
        "ALiBi outside TL causal": not a.get("alibi_mask")
        or (R == 0 and al == "TOP_LEFT"),
        "dropout_probability without RNG state": p is None
        or (0 <= p < 1 and "Seed" in T and "Offset" in T and not mask),
        "dropout scale without a mask": not (
            ("DropoutScaleInv" in T or "DropoutScale" in T) and not mask
        ),
        "one seq_len": ("SeqLenQ" in T) == ("SeqLenKv" in T),
        "seq_len without padding_mask": not (
            padded and not (r.get("padding") or {}).get("padding_mask", True)
        ),
        "Sq = Skv = 1": not (Sq == Skv == 1),
        "BR band with Sq > Skv": not (br_band and Sq > Skv and not padded),
        "BR band with bias/ALiBi/dropout": not (
            br_band and (b or a.get("alibi_mask") or mask)
        ),
        "window with Sq > Skv": not (W is not None and Sq > Skv and not padded),
        "right_bound < 0 when set": rb is None or rb >= 0,
        "head dim not a multiple of 8": not (Dqk % 8 or Dv % 8),
        "FLOAT dropout mask": not (
            mask and mask["data_type"] not in ("half", "bfloat16")
        ),
    }
    for reason, _rule, _src in WF_RULES:
        if not checks[reason]:
            return False, reason
    return True, ""


def extension(r, why):
    """EXT: outside WF, defined by hipDNN. `why` is the WF failure reason."""
    T = r["strides"]
    a = r.get("attributes") or {}
    B, Hq, Sq, _ = T["Q"]["dims"]
    Skv = T["K"]["dims"][2]
    b = T.get("Bias")
    ext = {
        "IO dtype not FP16/BF16": all(T[n]["data_type"] == "float" for n in IO),
        "innermost stride != 1": True,
        "Stats rank/dtype": T["Stats"]["data_type"] == "float"
        and len(T["Stats"]["dims"]) == 3,
        "bias rank/broadcast/dtype": bool(b)
        and b["data_type"] in ("float", "half", "bfloat16")
        and len(b["dims"]) <= 4
        and _bias_broadcast_ok(b, B, Hq, Sq, Skv),
        "BR band with bias/ALiBi/dropout": not a.get("alibi_mask"),
        "seq_len without padding_mask": True,
        "BR band with Sq > Skv": True,
        "window with Sq > Skv": True,
        "Sq = Skv = 1": True,
        "right_bound < 0 when set": r.get("right_bound_written") == -1,
        "head dim not a multiple of 8": True,
        "FLOAT dropout mask": True,
        "dropout_probability without RNG state": a.get("dropout_probability") == 0,
    }
    return bool(ext.get(why, False))


def layer(r, semantics="develop"):
    """(layer, reason): illegal | WF | EXT | U."""
    ok, why = frontend_legal(r, semantics)
    if not ok:
        return "illegal", why
    wf, why = well_formed(r)
    if wf:
        return "WF", ""
    return ("EXT" if extension(r, why) else "U"), why


def legality_violations(r):
    """Violations of the write rule for one record (§2.8)."""
    out = []
    lay, why = layer(r, "develop")
    rej = r.get("expected_rejection")
    if lay == "illegal":
        out.append(("ILLEGAL", why))
    elif rej and lay != "U":
        out.append(("REJECTION PROBE NOT IN U", f"{lay} {why}"))
    elif not rej and lay == "U":
        out.append(("U CASE NOT A REJECTION PROBE", why))
    if r.get("purpose") in ("claim", "regression") and lay not in ("WF", "EXT"):
        out.append(("CLAIM OR REGRESSION OUTSIDE WF AND EXT", f"{lay} {why}"))
    for sem in ("after_14287", "after_14358"):
        if layer(r, sem)[0] == "illegal" and not rej:
            out.append(("ILLEGAL UNDER " + sem, ""))
    return out


# --------------------------------------------------------------------------
# Oracle gates (§4.1). A case whose oracle is not ready has level notrun and
# is written once its gate opens. Each gate names what opens it and its
# tracking item; "to file" means no ticket exists yet.
# --------------------------------------------------------------------------

READY = ("ready", "ready [I]", "rejection")  # [I]: inferred; the first run confirms it
UNASSIGNED_GATE = "no non-vacuous class (needs the per-tensor validator HG6)"

# fmt: off
GATES = {
    "14358": ("#14358: GPU backward reference; dBias and rank-3 Stats in the references",
              "ALMIOPEN-2103, ALMIOPEN-2695"),
    "14358+derivation": ("#14358 and the #14312 derivation extension (ALiBi)", "#14358; derivation: to file"),
    "14358+derivation+HG4": ("#14358, the #14312 derivation extension and HG4 (per-batch lists, Bernoulli masks)",
                             "ALMIOPEN-2861, ALMIOPEN-2862 (partial); derivation: to file"),
    "14358+derivation+HG20": ("#14358, the derivation extension and HG20 (ragged fill)", "to file"),
    "14358+I4": ("#14358 and I4 (structured -inf injection)", "ALMIOPEN-2948 (design only); to file"),
    "14358+gpufwd": ("#14358 and a GPU forward derivation (X-size shapes)", "ALMIOPEN-2855 (partial)"),
    "14358+timing": ("#14358 and a timing run that re-sizes the shape", "to file"),
    "I3": ("I3: base pointers placed at exactly the declared alignment", "ALMIOPEN-2948 (design only); to file"),
    "I4": ("I4: structured value injection", "ALMIOPEN-2948 (design only); to file"),
    "HG9": ("HG9: exact-zero check", "to file"),
    "HG18": ("HG18: finite-only validator", "to file"),
    "harness: zero strides rejected": ("HG17: zero input strides", "to file"),
    "none": ("no oracle (Philox: the backward schema has no rng_dump)", "to file"),
    "none (decline-only probe mode)": ("a decline-only probe mode that does not allocate", "to file"),
    "unit": ("descriptor-only: a unit test, not a sweep case", "#14075"),
    UNASSIGNED_GATE: ("HG6: per-tensor magnitude-relative validator", "ALMIOPEN-2704 (partial); to file"),
}
# fmt: on


def oracle_of(c):
    if c.get("rejection"):
        return "rejection"
    return c.get("oracle") or "ready"


# --------------------------------------------------------------------------
# Purpose (RFC 0022 §9). A probe exercises a value no engine implements, an
# open defect or a rejection rule; a claim is a mainstream graph; a
# regression pins an escaped bug.
# --------------------------------------------------------------------------

CLAIM_HDIMS = (128, 192)
WINDOW_MASKS = ("window", "band", "left_only", "right_only")


def probe_reasons(c, dt):
    why = []
    if dt == "fp32":
        why.append("fp32")
    if not (c["Hq"] == c["Hk"] == c["Hv"]):
        why.append("heads")
    if c["Dqk"] != c["Dv"]:
        why.append("dqk!=dv")
    if c["Dqk"] not in CLAIM_HDIMS:
        why.append("hdim")
    if (
        c.get("bias")
        or c.get("dbias")
        or c.get("dropout")
        or c.get("padding")
        or c.get("ragged")
    ):
        why.append("optional tensor")
    if (c.get("attrs") or {}).get("alibi_mask"):
        why.append("alibi")
    if c.get("topology") in ("FwdBwdChained", "Philox", "SeqLenQOnly"):
        why.append("topology")
    if c.get("alignment", 16) < 16:
        why.append("alignment")
    if c.get("stats", "packed") != "packed" or c.get("dtype_over"):
        why.append("stats/dtype")
    if c["Dqk"] == 192 and mask_kind(*effective_band(c)[:2]) in WINDOW_MASKS:
        why.append("d192 x window")
    s = c.get("scale")
    if not isinstance(s, dict) and s not in (ISQ, "unset") and float(s) <= 0:
        why.append("scale<=0")
    if dt == "fp16" and c["Skv"] <= 2:
        why.append("fp16 Skv<=2 (DEF-03)")
    if c.get("purpose") == "probe" and not why:
        why.append("declared probe")
    return why


def purpose(c, dt):
    if c.get("rejection"):
        return "probe"
    if c.get("purpose") == "regression":
        return "regression"
    if c.get("defect") and not (c["defect"] == "DEF-03" and dt == "bf16"):
        return "probe"
    return "probe" if probe_reasons(c, dt) else "claim"


# --------------------------------------------------------------------------
# Value domains (§2: V1-V10). The fill recipes are free (uniform in [lo, hi])
# and fixed. Q, K, V and dO default to U[-1,1]; the fill derives O and Stats
# from the forward (#14312), so they never have a recipe.
# --------------------------------------------------------------------------

FIXED0 = {"kind": "fixed", "value": 0.0}
FIXED133120 = {"kind": "fixed", "value": 133120.0}
# fmt: off
DOMAIN = {
    "v01": {"v": [0, 1]},  # the D-term regime: D = rowsum(dO*O) is large
    "peaked": {"q": [-4, 4], "k": [-4, 4]},
    "x4": {"q": [-4, 4], "k": [-4, 4], "v": [-4, 4], "do": [-4, 4]},
    "x6": {"q": [-6, 6], "k": [-6, 6], "v": [-6, 6], "do": [-6, 6]},
    "fa_overflow": {"q": [-8.66, 8.66], "k": [-5.2, 5.2], "v": [-5.2, 5.2]},
    "u01": {"q": [0, 1], "k": [0, 1], "v": [0, 1], "do": [0, 1]},
    "qk0": {"q": FIXED0, "k": FIXED0},
    "do0": {"do": FIXED0},
    "qkv133120": {"q": FIXED133120, "k": FIXED133120, "v": FIXED133120},
    "bias_neg": {"bias": [-4, 0]},
}
# fmt: on
INPUT_NAME = {"q": "Q", "k": "K", "v": "V", "do": "dO", "bias": "Bias"}


def domain(name):
    """A copy of a named value domain."""
    return json.loads(json.dumps(DOMAIN[name]))


# --------------------------------------------------------------------------
# Tolerance classes (§6.1). A case id ends with its class token; each engine
# TOML has one row per class and dtype. Rule order: F32, K1, U, S, P, DL, D.
# The class must hold the guard (atol <= 0.1 x the smallest non-zero max|ref|
# over dQ, dK, dV) and, where the kernel model ran, the margin (atol >=
# atol_needed: margin >= 2.5 against an FP32-accumulating reference). If the
# rule's class fails, the case moves to a neighbour class; if none fits, dO is
# scaled by 2^k (gradients are linear in dO; the softmax does not change). A
# case no class fits waits for the per-tensor validator HG6.
# --------------------------------------------------------------------------

# fmt: off
CLASSES = OrderedDict([
    ("D", ("told", {"bf16": (0.0025, 0.016), "fp16": (0.00075, 0.004)}, "diffuse softmax")),
    ("DL", ("toldl", {"bf16": (0.0012, 0.016)}, "diffuse softmax, long rows (bf16, Skv > 1024)")),
    ("P", ("tolp", {"bf16": (0.012, 0.016), "fp16": (0.003, 0.004)},
           "peaked softmax: causal, narrow window, Sq > Skv bottom-right, long Sq accumulation")),
    ("S", ("tols", {"bf16": (0.0003, 0.016), "fp16": (0.0002, 0.004)},
           "short query side, Sq <= 17 (decode-like); tiny dK, dV")),
    ("U", ("tolu", {"bf16": (0.025, 0.016), "fp16": (0.003, 0.004)},
           "unscaled or high-spread logits (lambda >= 2): unset scale, scale ~1, peaked value domain")),
    ("K1", ("tolk1", {"bf16": (0.0003, 0.016), "fp16": (0.0003, 0.004)},
            "Skv <= 2: dQ and dK cancel to about 0 at Skv = 1; dV is a sum of dO rows")),
    ("F32", ("tolf32", {"fp32": (1e-5, 1e-5)}, "fp32 IO (uncalibrated)")),
])
# fmt: on
ASSIGNMENT_ORDER = ("F32", "K1", "U", "S", "P", "DL", "D")
TOKEN = {k: v[0] for k, v in CLASSES.items()}
GUARD = 0.1  # atol <= GUARD x min over non-zero gradients of max|ref|
MARGIN = 2.5  # same-precision kernel model vs the FP32-accumulating reference
KERNEL_MODEL_CAP = (
    2.5e9  # N_pairs x max(Dqk, Dv) above which the kernel model is not run
)
ESTIMATE_CAP = 2e7  # min(B, 2) x Hq x Sq x Skv above which the class is provisional


def tolerance(cls, dt):
    """(atol, rtol) of a class for an IO dtype, or None."""
    return CLASSES[cls][1].get(dt)


def escalation_order(dt):
    """Classes sorted by atol, tightest first (the neighbours of a class)."""
    names = ("S", "K1", "DL", "D", "P", "U")
    cs = [c for c in names if tolerance(c, dt)]
    return sorted(cs, key=lambda c: (tolerance(c, dt)[0], names.index(c)))


def rule_class(c, dt, est):
    """The class the rule order gives, before the guard and the margin."""
    if dt == "fp32":
        return "F32"
    if c["Skv"] <= 2:
        return "K1"  # Skv = 1: dQ = dK = 0, the guard runs on dV; Skv = 2: on every gradient
    if est["lambda"] >= 2:
        return "U"
    if c["Sq"] <= 17:
        return "S"
    if est["max_ref"]["dV"] >= 0.5:
        return "P"
    if dt == "bf16" and c["Skv"] > 1024:
        return "DL"
    return "D"


def assign_class(c, dt, est):
    """(class, info, dO scale). info["unassigned"] is set when no class holds
    the guard and the margin."""
    mr = est["max_ref"]
    nz = [v for v in mr.values() if v > 1e-9]
    rule = rule_class(c, dt, est)
    info = {
        "rule": rule,
        "min_max_ref": min(nz) if nz else 0.0,
        "max_ref": mr,
        "lambda": est["lambda"],
    }
    if rule == "F32":
        info["guard"] = 1e-5 / min(nz) if nz else None
        return "F32", info, 1.0
    if not nz:
        info["guard"] = "exempt: every gradient is zero by construction"
        return rule, info, 1.0
    g1 = GUARD * min(nz)
    n1 = est.get("atol_needed")

    def fits(cls):
        a = tolerance(cls, dt)[0]
        return a <= g1 * (1 + 1e-12) and (n1 is None or a >= n1)

    order = escalation_order(dt)
    if rule != "K1":
        order = [x for x in order if x != "K1"]
    if fits(rule):
        info["guard"] = tolerance(rule, dt)[0] / (10 * g1)
        return rule, info, 1.0
    i = order.index(rule)
    a0 = tolerance(rule, dt)[0]
    for cls in (
        order[i + 1 :] if (n1 is not None and a0 < n1) else list(reversed(order[:i]))
    ):
        if fits(cls):
            info.update(guard=tolerance(cls, dt)[0] / (10 * g1), escalated_from=rule)
            return cls, info, 1.0
    if c.get("no_scale"):
        info["unassigned"] = True
        return rule, info, 1.0
    for cls in [rule] + [x for x in order if x != rule]:
        a = tolerance(cls, dt)[0]
        lo = a / g1
        hi = a / n1 if n1 else lo * 64
        ks = [k for k in range(-12, 13) if lo <= 2.0**k <= hi]
        if lo > hi or not ks:
            continue
        mid = math.sqrt(lo * hi)
        f = 2.0 ** min(ks, key=lambda k: abs(k - math.log2(mid)))
        info.update(guard=a / (10 * g1 * f), dO_scale=f)
        if cls != rule:
            info["escalated_from"] = rule
        return cls, info, f
    info["unassigned"] = True
    return rule, info, 1.0


def modellable(c):
    """The estimator models the case (no dropout, padding, ragged offsets,
    injected or boolean inputs)."""
    return not (
        c.get("dropout")
        or c.get("padding")
        or c.get("ragged")
        or isinstance((c.get("inputs") or {}).get("bias"), str)
        or (c.get("bias") or {}).get("dtype") == "bool"
    )


def estimable(c):
    """The class rests on an estimate: the case is modellable and small
    enough; otherwise its class is provisional."""
    return (
        modellable(c) and min(c["B"], 2) * c["Hq"] * c["Sq"] * c["Skv"] <= ESTIMATE_CAP
    )


def kernel_model(c, dt, runnable):
    """The estimate runs the same-precision kernel model (the margin check)."""
    T = c["B"] * c["Hq"] * c["Sq"] * c["Skv"]
    return T * max(c["Dqk"], c["Dv"]) <= KERNEL_MODEL_CAP and dt != "fp32" and runnable


def provisional_class(dt):
    """The class of a case without an estimate (rejection probes included)."""
    return "F32" if dt == "fp32" else "D"


def estimate_params(c, dt, km):
    """What the magnitude estimate depends on (its fingerprint)."""
    W, R, al = effective_band(c)
    p = {k: c[k] for k in GEOMETRY}
    p.update(dtype=dt, align=al, W=W, R=R, scale=scale_value(c), kernel_model=km)
    p["inputs"] = {
        k: v for k, v in (c.get("inputs") or {}).items() if not isinstance(v, str)
    }
    if c.get("bias") and c["bias"]["dtype"] != "bool":
        b = c["bias"]
        p["bias_dims"] = bias_dims(b["dims"], c["B"], c["Hq"], c["Sq"], c["Skv"])
        p["bias_dtype"] = dt if b["dtype"] == "io" else b["dtype"]
    # magnitudes are per (batch, head) slab: estimate on at most 2 batches
    if p["B"] > 2 and p["B"] * p["Hq"] * p["Sq"] * p["Skv"] > 4e6:
        p["B"] = 2
        if p.get("bias_dims") and len(p["bias_dims"]) == 4 and p["bias_dims"][0] > 2:
            p["bias_dims"] = [2] + p["bias_dims"][1:]
    return p


def fingerprint(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


CALIBRATION_FILE = Path(__file__).with_name("sdpa_bwd_calibration.json")


def load_calibration(path=CALIBRATION_FILE):
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))["estimates"]


def save_calibration(entries, path=CALIBRATION_FILE):
    doc = {
        "note": (
            "Magnitude estimates behind the SdpaBackward tolerance classes (plan §6.1). "
            "Written by generate_sdpa_bwd_bundles.py --calibrate (NumPy); read by every run. "
            "Key: <plan item>/<dtype>; fp: fingerprint of the estimate's inputs."
        ),
        "estimates": OrderedDict(sorted(entries.items())),
    }
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")


def estimate(params, seed=1234):
    """NumPy magnitude estimate of one case (--calibrate only).

    Draws Q, K, V, dO and the bias from the domain with a fixed seed, rounds
    them to the IO dtype, derives O and the LSE with an FP64 forward (O rounded
    to the IO dtype, as #14312 does) and computes the reference gradients:
    max|ref| per gradient, the logit spread lambda and, with the kernel model,
    the atol that a same-precision kernel (P and dS rounded to the IO dtype)
    needs for margin MARGIN against an FP32-accumulating reference."""
    import numpy as np

    def rnd(x, dt):
        x = np.asarray(x, dtype=np.float64)
        if dt == "fp16":
            return x.astype(np.float16).astype(np.float64)
        if dt == "bf16":
            u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
            u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
            return u.view(np.float32).astype(np.float64)
        if dt == "fp32":
            return x.astype(np.float32).astype(np.float64)
        return x

    def draw(rng, shape, spec):
        spec = [-1.0, 1.0] if spec is None else spec
        if isinstance(spec, dict):
            return np.full(shape, float(spec["value"]))
        return rng.uniform(spec[0], spec[1], size=shape)

    def sigma(spec):
        spec = [-1.0, 1.0] if spec is None else spec
        return 0.0 if isinstance(spec, dict) else (spec[1] - spec[0]) / np.sqrt(12.0)

    p = params
    B, Hq, Hk, Hv, Sq, Skv, Dqk, Dv = (p[k] for k in GEOMETRY)
    dt, sc, ins = p["dtype"], float(p["scale"]), p.get("inputs") or {}
    rng = np.random.default_rng(seed)
    Q = rnd(draw(rng, (B, Hq, Sq, Dqk), ins.get("q")), dt)
    K = rnd(draw(rng, (B, Hk, Skv, Dqk), ins.get("k")), dt)
    V = rnd(draw(rng, (B, Hv, Skv, Dv), ins.get("v")), dt)
    dO = rnd(draw(rng, (B, Hq, Sq, Dv), ins.get("do")), dt)
    bias = None
    if p.get("bias_dims"):
        b = rnd(
            draw(rng, tuple(p["bias_dims"]), ins.get("bias", [-1.0, 1.0])),
            p.get("bias_dtype", "fp32"),
        )
        while b.ndim < 4:
            b = b[None]
        bias = np.broadcast_to(b, (B, Hq, Sq, Skv))
    off = 0 if p["align"] == "TL" else Skv - Sq
    i = np.arange(Sq)[:, None] + off
    j = np.arange(Skv)[None, :]
    keep = np.ones((Sq, Skv), dtype=bool)
    if p["R"] is not None:
        keep &= j <= i + p["R"]
    if p["W"] is not None:
        keep &= j >= i - p["W"] + 1
    hk = np.arange(Hq) // (Hq // Hk)
    hv = np.arange(Hq) // (Hq // Hv)
    Kx, Vx = K[:, hk], V[:, hv]
    S = sc * (Q @ np.swapaxes(Kx, -1, -2))
    if bias is not None:
        S = S + bias
    Sm = np.where(keep, S, -np.inf)
    mx = Sm.max(-1, keepdims=True)
    alive = np.isfinite(mx)
    mx = np.where(alive, mx, 0.0)
    E = np.where(keep, np.exp(Sm - mx), 0.0)
    den = E.sum(-1, keepdims=True)
    P = np.where(alive, E / np.where(den > 0, den, 1.0), 0.0)
    lse = np.where(alive, mx + np.log(np.where(den > 0, den, 1.0)), -np.inf)
    O = rnd(P @ Vx, dt)
    lse32 = lse.astype(np.float32).astype(np.float64)
    Pb = np.where(
        keep & alive, np.exp(np.where(keep, S, 0.0) - np.where(alive, lse32, 0.0)), 0.0
    )
    dP = dO @ np.swapaxes(Vx, -1, -2)
    Dt = (dO * O).sum(-1, keepdims=True)
    dS = np.where(keep, Pb * (dP - Dt), 0.0)

    def reduce_heads(kx, vx, f=np.float64):
        k = np.zeros((B, Hk, Skv, Dqk), dtype=f)
        v = np.zeros((B, Hv, Skv, Dv), dtype=f)
        for h in range(Hq):
            k[:, hk[h]] += kx[:, h]
            v[:, hv[h]] += vx[:, h]
        return k, v

    dK, dV = reduce_heads(
        sc * (np.swapaxes(dS, -1, -2) @ Q), np.swapaxes(Pb, -1, -2) @ dO
    )
    ref = {"dQ": rnd(sc * (dS @ Kx), dt), "dK": rnd(dK, dt), "dV": rnd(dV, dt)}
    out = {
        "lambda": float(sc * np.sqrt(Dqk) * sigma(ins.get("q")) * sigma(ins.get("k"))),
        "max_ref": {k: float(np.abs(v).max()) for k, v in ref.items()},
        "atol_needed": None,
    }
    rt = {"bf16": 1.6e-2, "fp16": 4e-3}.get(dt)
    if p.get("kernel_model") and rt:
        f = np.float32
        Q32, Kx32, Vx32, dO32, O32 = (x.astype(f) for x in (Q, Kx, Vx, dO, O))
        S32 = (Q32 @ np.swapaxes(Kx32, -1, -2)) * f(sc)
        if bias is not None:
            S32 = S32 + bias.astype(f)
        with np.errstate(invalid="ignore", over="ignore"):
            lse_f = np.where(alive, lse32, 0.0).astype(f)
            P32 = np.where(
                keep & alive, np.exp(np.where(keep, S32, f(0)) - lse_f), f(0)
            ).astype(f)
        dP32 = dO32 @ np.swapaxes(Vx32, -1, -2)
        D32 = (dO32 * O32).sum(-1, keepdims=True, dtype=f)
        dS32 = np.where(keep, (P32 * (dP32 - D32)).astype(f), f(0))

        def grads(Pm, dSm):
            k, v = reduce_heads(
                (np.swapaxes(dSm, -1, -2) @ Q32) * f(sc),
                np.swapaxes(Pm, -1, -2) @ dO32,
                f,
            )
            return {
                "dQ": rnd((dSm @ Kx32) * f(sc), dt),
                "dK": rnd(k, dt),
                "dV": rnd(v, dt),
            }

        r32 = grads(P32, dS32)
        model = grads(rnd(P32, dt).astype(f), rnd(dS32, dt).astype(f))
        need = 0.0
        for g in ("dQ", "dK", "dV"):
            if r32[g].size:
                need = max(
                    need,
                    float(
                        (MARGIN * np.abs(model[g] - r32[g]) - rt * np.abs(r32[g])).max()
                    ),
                )
        out["atol_needed"] = max(need, 0.0)
    return out


# --------------------------------------------------------------------------
# All-pairs axes (§4.3). A value is claim (an engine claims or is expected to
# claim it) or probe (no engine implements it). By the end of nightly every
# required pair meets in a runnable case, and every pair of claim values
# meets in a claim-only case.
# --------------------------------------------------------------------------


def _axis(name, feature, *values):
    return {
        "name": name,
        "feature": feature,
        "values": [{"value": v, "class": c, "note": n} for v, c, n in values],
    }


# fmt: off
AXES = {
    "version": 1,
    "axes": [
        _axis("dtype", "DT1", ("bf16", "claim", "IO dtype BF16"), ("fp16", "claim", "IO dtype FP16")),
        _axis("seq", "G7", ("lt", "claim", "Sq < Skv"), ("eq", "claim", "Sq = Skv"), ("gt", "claim", "Sq > Skv")),
        _axis("mask", "A3,A4,A5", ("none", "claim", "no band"), ("causal", "claim", "R = 0"),
              ("window", "claim", "W keys, R = 0"), ("band", "claim", "W keys, R > 0"),
              ("left_only", "claim", "W keys, R unbounded"), ("right_only", "claim", "R > 0, W unbounded")),
        _axis("align", "A2", ("TL", "claim", "TOP_LEFT"), ("BR", "claim", "BOTTOM_RIGHT")),
        _axis("heads", "G2,G3,G4", ("mha", "claim", "Hq = Hk = Hv"), ("gqa2", "probe", "g = 2 (4/2/2)"),
              ("gqa3", "probe", "g = 3, Hq not a power of two (6/2/2)"),
              ("mqa", "probe", "Hk = Hv = 1, g = 4 (4/1/1)"), ("hk_ne_hv", "probe", "Hk != Hv (4/2/1)")),
        _axis("hdim", "G8,G9", ("d128", "claim", "Dqk = Dv = 128"), ("d192", "claim", "Dqk = Dv = 192"),
              ("d64", "probe", "Dqk = Dv = 64"), ("d256", "probe", "Dqk = Dv = 256"),
              ("d192x128", "probe", "Dqk = 192, Dv = 128")),
        _axis("layout", "T18,AW12,G10", ("bhsd", "claim", "packed BHSD"),
              ("bshd", "claim", "packed BSHD, all 8 IO tensors"), ("do_bshd", "claim", "dO BSHD, the rest BHSD"),
              ("grads_bshd", "claim", "dQ, dK, dV BSHD"),
              ("rowpad8", "claim", "sequence stride D + 8 on all 8 IO tensors")),
        _axis("scale", "A1,T12", ("inv_sqrt", "claim", "attn_scale_value = 1/sqrt(Dqk)"),
              ("unset", "claim", "unset (= 1.0)"), ("other", "claim", "attn_scale_value = 0.35"),
              ("rt_pbv", "claim", "runtime pass-by-value scale tensor = 0.5"),
              ("const_pbv", "claim", "compile-time constant pass-by-value scale tensor = 0.5")),
        _axis("vdom", "V1", ("default", "claim", "Q, K, V, dO in U[-1,1]"), ("v01", "claim", "V in U[0,1]"),
              ("peaked", "claim", "Q, K in U[-4,4]")),
        _axis("bias", "T10,G11,DT4", ("none", "claim", "no bias"), ("b11_f32", "probe", "bias [1,1,Sq,Skv] FLOAT"),
              ("bfull_io", "probe", "bias [B,Hq,Sq,Skv] in the IO dtype")),
    ],
    # a scale tensor with a bias needs a combined topology (register entry R-AP1)
    "forbidden": [{"scale": ["rt_pbv", "const_pbv"], "bias": ["b11_f32", "bfull_io"]}],
    # claim values no engine supports together: a case holding one is a probe
    "probe_combinations": [{"hdim": "d192", "mask": ["window", "band", "left_only", "right_only"]}],
    # BOTTOM_RIGHT equals TOP_LEFT without a band or at Sq = Skv (alias, AW1)
    "not_required": [{"align": "BR", "mask": "none"}, {"align": "BR", "seq": "eq"}],
    "dont_care": [{"when": {"mask": "none"}, "axis": "align", "canonical": "TL"},
                  {"when": {"seq": "eq"}, "axis": "align", "canonical": "TL"}],
}

# axis value -> case parameters of the all-pairs cases (the core base, B 2)
AP_SEQ = {"lt": (40, 72), "eq": (56, 56), "gt": (72, 40)}
AP_HEADS = {"mha": (3, 3, 3), "gqa2": (4, 2, 2), "gqa3": (6, 2, 2), "mqa": (4, 1, 1), "hk_ne_hv": (4, 2, 1)}
AP_HDIM = {"d128": (128, 128), "d192": (192, 192), "d64": (64, 64), "d256": (256, 256), "d192x128": (192, 128)}
AP_MASK = {"none": (None, None), "causal": (None, 0), "window": (16, 0), "band": (9, 8), "left_only": (16, None),
           "right_only": (None, 8)}
AP_SCALE = {"inv_sqrt": ISQ, "unset": "unset", "other": 0.35, "rt_pbv": {"tensor": "runtime", "value": 0.5},
            "const_pbv": {"tensor": "const", "value": 0.5}}
AP_BIAS = {"none": None, "b11_f32": {"dims": "11", "dtype": "fp32"}, "bfull_io": {"dims": "full", "dtype": "io"}}
# fmt: on


def axis_names(axes=AXES):
    return [a["name"] for a in axes["axes"]]


def axis_values(name, axes=AXES):
    return [v["value"] for a in axes["axes"] if a["name"] == name for v in a["values"]]


def is_claim(name, value, axes=AXES):
    for a in axes["axes"]:
        for v in a["values"]:
            if a["name"] == name and v["value"] == value:
                return v["class"] == "claim"
    raise KeyError((name, value))


def _match(cond, case):
    return all(
        case.get(k) in (v if isinstance(v, list) else [v]) for k, v in cond.items()
    )


def forbidden(case, axes=AXES):
    """The (possibly partial) case holds a forbidden combination."""
    return any(
        all(k in case for k in rule) and _match(rule, case)
        for rule in axes["forbidden"]
    )


def probe_combination(case, axes=AXES):
    return any(
        all(k in case for k in rule) and _match(rule, case)
        for rule in axes["probe_combinations"]
    )


def pair_required(a, va, b, vb, axes=AXES):
    probe = {a: va, b: vb}
    rules = axes["forbidden"] + axes["not_required"]
    return not any(set(rule) <= {a, b} and _match(rule, probe) for rule in rules)


def required_pairs(axes=AXES):
    names = axis_names(axes)
    return {
        ((a, va), (b, vb))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        for va in axis_values(a, axes)
        for vb in axis_values(b, axes)
        if pair_required(a, va, b, vb, axes)
    }


def claim_pairs(req, axes=AXES):
    """Required pairs of two claim values that are not a probe combination."""
    return {
        p
        for p in req
        if all(is_claim(n, v, axes) for n, v in p)
        and not probe_combination(dict(p), axes)
    }


def normalize(case, axes=AXES):
    """Apply the don't-care rules (alignment is TL where it cannot matter)."""
    c = dict(case)
    for rule in axes["dont_care"]:
        if _match(rule["when"], c):
            c[rule["axis"]] = rule["canonical"]
    return c


def pairs_of(case, axes=AXES):
    names = [n for n in axis_names(axes) if case.get(n) is not None]
    return {
        ((a, case[a]), (b, case[b]))
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }


def claim_only(case, purpose_="claim", axes=AXES):
    if purpose_ not in ("claim", "regression") or probe_combination(case, axes):
        return False
    return all(
        is_claim(n, case[n], axes) for n in axis_names(axes) if case.get(n) is not None
    )


def pairwise_check(cases, axes=AXES):
    """cases: (axis values, purpose). Proves that (a) every required pair is
    covered, (b) every claim pair is covered by a claim-only case and (c) no
    case holds a forbidden combination."""
    req = required_pairs(axes)
    claim_req = claim_pairs(req, axes)
    cov, cov_claim, bad = set(), set(), []
    for c, pur in cases:
        if forbidden(c, axes):
            bad.append(c)
        cov |= pairs_of(c, axes)
        if claim_only(c, pur, axes):
            cov_claim |= pairs_of(c, axes)
    return {
        "required": len(req),
        "covered": len(req & cov),
        "uncovered": sorted(req - cov),
        "claim_required": len(claim_req),
        "claim_covered": len(claim_req & cov_claim),
        "claim_uncovered": sorted(claim_req - cov_claim),
        "forbidden": bad,
    }


def axis_projection(c, dt):
    """The case's value on every axis, or None when the case is outside the
    axes (another topology, deprecated encoding, other tensors or dtypes)."""
    at = c.get("attrs") or {}
    if (
        c.get("dbias")
        or c.get("dropout")
        or c.get("padding")
        or at.get("alibi_mask")
        or c.get("dims_over")
    ):
        return None
    if topology_of(c) not in ("Default", "AdditiveMask", "ScaleTensor"):
        return None
    if (
        c.get("alignment", 16) != 16
        or c.get("dtype_over")
        or dt not in ("bf16", "fp16")
    ):
        return None
    seq = "lt" if c["Sq"] < c["Skv"] else ("eq" if c["Sq"] == c["Skv"] else "gt")
    mk = mask_kind(c.get("W"), c.get("R"))
    al = "TL" if (mk == "none" or seq == "eq") else c.get("align", "TL")
    if at.get("causal_mask") or at.get("causal_mask_bottom_right"):
        mk = al = None  # the deprecated encoding is outside the explicit band axes
    Hq, Hk, Hv = c["Hq"], c["Hk"], c["Hv"]
    if Hq == Hk == Hv:
        heads = "mha"
    elif Hk == Hv == 1:
        heads = "mqa"
    elif Hk != Hv:
        heads = "hk_ne_hv"
    else:
        heads = {2: "gqa2", 3: "gqa3"}.get(Hq // Hk)
    hd = {v: k for k, v in AP_HDIM.items()}.get((c["Dqk"], c["Dv"]))
    lay = c.get("layout", "bhsd")
    if lay not in ("bhsd", "bshd", "do_bshd", "grads_bshd", "rowpad8") or c.get(
        "tensor_layout"
    ):
        lay = None
    if c.get("stats", "packed") != "packed":
        lay = None
    s = c.get("scale", ISQ)
    if isinstance(s, dict):
        sc = "rt_pbv" if s["tensor"] == "runtime" else "const_pbv"
    elif s in (ISQ, "unset"):
        sc = "inv_sqrt" if s == ISQ else "unset"
    else:
        sc = "other" if float(s) > 0 else None
    # dO's magnitude is a free scale (§6.1): a symmetric dO range is the default domain
    ins = {
        k: v
        for k, v in (c.get("inputs") or {}).items()
        if k != "bias" and not (k == "do" and isinstance(v, list) and v[0] == -v[1])
    }
    vd = (
        "default"
        if not ins
        else next((n for n in ("v01", "peaked") if ins == DOMAIN[n]), None)
    )
    b = c.get("bias")
    if not b:
        bias = "none"
    elif b.get("stride", "packed") != "packed":
        bias = None
    else:
        bias = {("11", "fp32"): "b11_f32", ("full", "io"): "bfull_io"}.get(
            (str(b["dims"]), b["dtype"])
        )
    return {
        "dtype": dt,
        "seq": seq,
        "mask": mk,
        "align": al,
        "heads": heads,
        "hdim": hd,
        "layout": lay,
        "scale": sc,
        "vdom": vd,
        "bias": bias,
    }


# --------------------------------------------------------------------------
# Feature inventory (§2). Plan items name these ids; every id has a case or a
# disposition.
# --------------------------------------------------------------------------

# IV: frontend rules (frontend unit tests). IVP: proposed rules (rejection
# probes until they land).
FEATURES = (
    [f"T{i}" for i in range(1, 25)]
    + [f"A{i}" for i in range(1, 14)]
    + [f"G{i}" for i in range(1, 15)]
    + [f"DT{i}" for i in range(1, 7)]
    + [f"GO{i}" for i in range(1, 4)]
    + [f"AW{i}" for i in range(1, 15)]
    + [f"V{i}" for i in range(1, 11)]
    + [f"E{i}" for i in range(1, 7)]
    + [f"IV{i}" for i in range(0, 19)]
    + [f"IVP{i}" for i in range(1, 16)]
)

# Every case carries these (the plain op's tensors and attributes).
BASE_ITEMS = ("T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9", "T22", "A11", "V7", "V10")

# fmt: off
# Features with no case at any level, and why.
FEATURE_DISPOSITION = {
    "A12": "n/a: node name has no semantic effect",
    "A13": "unit: R0 (IV0)",
    "DT5": "unit: graph io/intermediate/compute types (bundles state every tensor dtype)",
    "GO1": "not run: harness gap HG16 (no per-execute override path)",
    "GO3": "CI: SDPA must be built ON in every lane (REG-11)",
    "AW14": "not run: harness gap HG17 (one buffer per uid)",
    "V4": "not run: NaN/Inf injection and a NaN-aware validator (HG4)",
    "V8": "DEF-02a with repeat 3; repeat-and-compare needs I5 (HG5)",
    "E1": "not run: repeat with an uncleared workspace needs I5 (HG5)",
    "E2": "not run: plan reuse with new pointers and scale values needs I5 (HG5)",
    "E3": "unit: lowering and lifting round trips",
    "E5": "n/a: the generator enforces legality (§2.8)",
    "E6": "out of scope (engine knobs)",
    **{f"IVP{i}": "rejection probe until the rule lands; then a frontend unit test" for i in range(1, 16)},
    **{f"IV{i}": "frontend unit test (§2.9)" for i in range(19)},
}
# fmt: on


# --------------------------------------------------------------------------
# Shape catalog (§3). Every shape overrides DEFAULT (B2 H3 72x120 D128, no
# band, BHSD, scale 1/sqrt(Dqk), U[-1,1]); f: the features it isolates; dt:
# its data types (bf16 when absent).
# --------------------------------------------------------------------------

_H1 = dict(B=1, Hq=1, Hk=1, Hv=1)
_H2 = dict(B=1, Hq=2, Hk=2, Hv=2)
BF16, FP16, BOTH = ("bf16",), ("fp16",), ("bf16", "fp16")


def bias_spec(dims, dtype="fp32"):
    """A bias: dims is a BIAS_DIMS token or explicit dims; dtype io | fp32 | fp16 | bf16 | bool."""
    return {"dims": dims, "dtype": dtype}


# fmt: off
# S: one feature changed from the default.
S_SHAPES = {
    "S00": dict(f=("G1", "G2", "G5", "G6", "G7", "G8", "DT1", "T18"), dt=BOTH),
    "S01": dict(f=("DT1",), dt=("fp32",)),
    "S02": dict(B=1, f=("G1",)),
    "S03": dict(B=5, f=("G1",), dt=FP16),
    "S04": dict(Hq=1, Hk=1, Hv=1, f=("G2",), dt=FP16),
    "S05": dict(Hq=4, Hk=4, Hv=4, f=("G2",)),
    "S11": dict(Hq=5, Hk=1, Hv=1, f=("G3",), dt=FP16),                  # MQA group 5
    "S13": dict(Skv=72, f=("G7",), dt=BOTH),
    "S14": dict(Sq=120, Skv=72, f=("G7",)),
    "S15": dict(Dqk=64, Dv=64, f=("G8",)),
    "S16": dict(Dqk=256, Dv=256, f=("G8",), dt=FP16),
    "S17": dict(Dqk=192, Dv=192, f=("G8",)),
    "S18": dict(Dqk=192, f=("G9",), dt=FP16),                           # MLA 192/128
    "S19": dict(Dqk=64, f=("G9",)),                                     # Dv > Dqk
    "S20": dict(Dqk=96, Dv=96, f=("G8",), dt=FP16),
    "S21": dict(Dqk=72, Dv=72, f=("G8",)),
    "S22": dict(Dqk=80, Dv=80, f=("G8",), dt=FP16),
    "S23": dict(Dqk=32, Dv=32, f=("G8",)),
    "S24": dict(Dqk=512, Dv=512, f=("G8",)),
    "S25": dict(R=0, f=("A3", "A2", "AW3"), dt=BOTH),                   # causal TL
    "S26": dict(R=0, align="BR", f=("A3", "A2"), dt=BOTH),              # causal BR
    "S27": dict(W=32, R=0, f=("A4", "A2"), dt=BOTH),
    "S28": dict(W=32, R=0, align="BR", f=("A4", "A2"), dt=BOTH),
    "S29": dict(W=17, R=16, f=("A5", "A2"), dt=FP16),                   # band +-16
    "S30": dict(W=32, f=("A4",)),                                       # left bound only
    "S31": dict(R=16, f=("A3",), dt=FP16),                              # right bound only
    "S32": dict(W=17, R=16, align="BR", f=("A5", "A2")),
    "S37": dict(scale=1.0, f=("A1",)),
    "S38": dict(scale=0.1352, f=("A1",), dt=FP16),                      # YaRN mscale
    "S40": dict(layout="bshd", f=("T18",), dt=BOTH),
    "S41": dict(layout="do_bshd", f=("AW12", "T18")),
    "S42": dict(layout="grads_bshd", f=("AW12", "G10"), dt=FP16),
    "S43": dict(layout="rowpad8", f=("T18",)),
    "S44": dict(layout="bs3hd", f=("T18",), dt=FP16),                   # packed-QKV strides
    "S46": dict(layout="rowpad1", f=("T23", "E4"), oracle="I3"),        # pointer exactly at the alignment
    "S47": dict(f=("T23", "E4"), oracle="I3"),                          # pointer less aligned than declared
    "S57": dict(Sq=64, Skv=128, f=("G5", "G6")),                        # tile-exact
}

# H: hazard breakpoints (sequence tails, tiles, head dims, groups, layouts,
# masks, scale and value regimes).
H_SHAPES = {
    "H02": dict(**_H2, Sq=203, Skv=113, f=("G5", "G6", "G7")),
    "H03": dict(**_H1, Sq=255, Skv=256, f=("G5", "G7")),
    "H04": dict(**_H1, Sq=256, Skv=255, f=("G6", "G7")),
    "H05": dict(**_H1, Sq=511, Skv=512, f=("G5", "G7")),
    "H06": dict(**_H1, Sq=512, Skv=511, f=("G6", "G7"), dt=FP16),
    "H07": dict(**_H1, Sq=257, Skv=257, f=("G5", "G6"), dt=FP16),
    "H08": dict(Sq=1, f=("G5", "AW4")),
    "H09": dict(**_H2, Sq=16, f=("G5",)),                               # short-Sq path boundary
    "H10": dict(**_H2, Sq=17, f=("G5",), dt=FP16),
    "H11": dict(Skv=1, f=("G6", "AW4")),
    "H12": dict(**_H2, Skv=2, f=("G6", "AW4"), dt=FP16),
    "H17": dict(**_H2, Skv=191, f=("G6",)),
    "H18": dict(**_H2, Skv=192, f=("G6",), dt=FP16),
    "H19": dict(**_H2, Skv=193, R=0, f=("G6",)),                        # KV-tile tail, causal grid
    "H20": dict(**_H2, Skv=385, R=0, align="BR", f=("G6",), dt=FP16),
    "H21": dict(**_H2, Sq=63, Skv=129, f=("G5", "G6")),
    "H22": dict(**_H2, Sq=65, Skv=127, f=("G5", "G6"), dt=FP16),
    "H23": dict(**_H2, Sq=516, Skv=253, f=("G7",)),
    "H24": dict(**_H2, Sq=192, Skv=64, R=0, align="BR", f=("AW2",), dt=BOTH),  # 128 dead rows
    "H25": dict(**_H2, Sq=301, Skv=190, R=0, align="BR", f=("AW2",), dt=FP16),
    "H26": dict(**_H1, Sq=4096, Skv=128, R=0, align="BR", f=("AW2",)),
    "H27": dict(**_H1, Sq=128, Skv=768, R=0, f=("AW3",)),              # 640 keys never attended
    "H28": dict(**_H2, Sq=1, Skv=512, f=("G7",), dt=FP16),
    "H29": dict(**_H1, Sq=4096, Skv=64, f=("V6",)),
    "H30": dict(**_H1, Sq=64, Skv=4096, f=("V6",), dt=FP16),
    "H31": dict(**_H1, Sq=512, Skv=128, f=("G7",)),
    "H32": dict(**_H1, Sq=513, Skv=513, R=0, f=("G5", "G6")),
    "H33": dict(**_H2, Dqk=59, Dv=59, f=("G8",)),
    "H34": dict(**_H2, Dqk=111, Dv=111, f=("G8",), dt=FP16),
    "H35": dict(**_H2, Dqk=100, Dv=100, f=("G8",)),
    "H36": dict(**_H2, Dqk=40, Dv=40, f=("G8",), dt=FP16),
    "H37": dict(**_H2, Dqk=8, Dv=8, f=("G8",)),
    "H38": dict(**_H2, Dqk=1, Dv=1, f=("G8",)),
    "H39": dict(**_H2, Dqk=160, Dv=160, f=("G8",), dt=FP16),
    "H40": dict(**_H2, Dqk=224, Dv=224, f=("G8",)),
    "H41": dict(**_H2, Dqk=384, Dv=384, f=("G8",), dt=FP16),
    "H42": dict(**_H2, Dqk=513, Dv=513, f=("G8",)),
    "H43": dict(**_H2, Dqk=40, Dv=88, f=("G9",)),
    "H44": dict(**_H2, Dqk=256, Dv=108, f=("G9",), dt=FP16),
    "H45": dict(**_H2, Dqk=120, Dv=8, f=("G9",)),
    "H46": dict(B=1, Hq=7, Hk=1, Hv=1, f=("G3",)),
    "H47": dict(B=1, Hq=6, Hk=1, Hv=1, f=("G3",), dt=FP16),
    "H49": dict(B=1, Hq=16, Hk=1, Hv=1, Sq=64, Skv=64, f=("G3",), dt=FP16),
    "H50": dict(B=1, Hq=32, Hk=8, Hv=8, Sq=16, Skv=16, f=("G3",), purpose="regression", regression_of="#11543"),
    "H58": dict(Hq=2, Hk=2, Hv=2, Sq=97, Skv=97, tensor_layout={"dO": "fa_transpose"}, f=("AW12", "T18"), dt=BOTH),
    "H60": dict(layout="perm6", f=("AW12",)),
    "H61": dict(layout="gaps", f=("AW12", "T18"), dt=FP16),
    "H62": dict(Hq=2, Hk=2, Hv=2, tensor_layout={"K": "bstride0", "V": "bstride0"}, f=("T21",),
                oracle="harness: zero strides rejected"),
    "H65": dict(**_H2, W=1, R=0, align="BR", f=("A4", "A2", "AW4"), purpose="regression",
                regression_of="SEM-001 (#14483; window anchoring SEM-004)"),
    "H66": dict(**_H2, Sq=256, Skv=256, W=17, R=0, f=("A4",)),
    "H67": dict(**_H2, Sq=256, Skv=256, W=64, R=0, f=("A4",), dt=FP16),
    "H68": dict(**_H2, Sq=256, Skv=256, W=65, R=0, f=("A4",)),
    "H69": dict(**_H2, W=120, R=0, f=("A4", "AW10"), dt=FP16),         # W = Skv: equals causal TL
    "H71": dict(**_H2, Sq=256, Skv=256, W=33, R=96, f=("A5",), dt=FP16),
    "H72": dict(**_H2, Sq=96, Skv=64, W=17, R=0, align="BR", f=("A4", "AW2")),
    "H73": dict(**_H2, R=120, f=("A3", "AW10")),                        # right_bound >= Skv: equals none
    "H74": dict(**_H2, attrs={"explicit_unbounded": True}, f=("A3", "A4"), dt=FP16),
    "H75": dict(**_H2, R=0, align="BR", bias=bias_spec("1h"), topology="AdditiveMask", f=("AW5",)),
    "H76": dict(**_H2, bias=bias_spec("r1"), topology="AdditiveMask", f=("G11",), dt=FP16),
    "H77": dict(**_H2, bias=bias_spec("row"), topology="AdditiveMask", f=("G11",)),
    "H78": dict(**_H2, bias=bias_spec("col"), topology="AdditiveMask", f=("G11",), dt=FP16),
    "H83": dict(**_H2, scale=0.02, f=("A1",)),
    "H84": dict(**_H2, scale=4.0, f=("A1",), dt=FP16),
    "H85": dict(B=9, Sq=16, Skv=64, f=("G1",)),
    "H86": dict(**_H2, inputs=DOMAIN["x6"], f=("V1",)),
    "H87": dict(Sq=1, Skv=1024, scale=0.125, inputs=DOMAIN["qkv133120"], f=("V1", "V4"), oracle="HG18"),
    "H88": dict(Hq=5, Hk=5, Hv=5, Sq=17, Skv=17, inputs=DOMAIN["fa_overflow"], f=("V1",), dt=FP16),
    "H89": dict(**_H2, inputs=DOMAIN["v01"], f=("V1",)),
    "H90": dict(**_H2, inputs=DOMAIN["x4"], scale=0.05, f=("V1", "V9"), dt=FP16),
    "H91": dict(**_H2, inputs=DOMAIN["do0"], f=("V1",), oracle="HG9"),
    "H92": dict(**_H2, inputs=DOMAIN["qk0"], f=("V1",), dt=FP16),
    "H93": dict(**_H2, f=("V5",), oracle="I4"),                         # scores < -87 in one row
    "H94": dict(**_H1, Sq=1023, Skv=1024, f=("G5", "G7")),
    "H95": dict(**_H1, Sq=1024, Skv=1023, f=("G6", "G7"), dt=FP16),
}

# X: stress (full, or not run until a GPU-reference era gate).
X_SHAPES = {
    "X01": dict(**_H1, Sq=46341, Skv=46341, W=128, R=0, f=("T24",), oracle="14358+gpufwd"),
    "X02": dict(B=1, Hq=40, Hk=40, Hv=40, Sq=75600, Skv=75600, f=("T24",), oracle="14358+gpufwd"),
    "X03": dict(B=1, Hq=128, Hk=128, Hv=128, Sq=65536, Skv=64, f=("T24",), oracle="none (decline-only probe mode)"),
    "X04": dict(Hq=65, Hk=65, Hv=65, Sq=65536, Skv=65536, W=256, R=0, f=("T24",), oracle="14358+gpufwd"),
    "X05": dict(**_H1, Sq=65536, Skv=64, f=("V6",)),
    "X06": dict(**_H1, Sq=64, Skv=65536, f=("V6",), dt=FP16),
    "X07": dict(**_H1, Sq=65536, Skv=1, f=("V6", "AW9"), dt=("fp16", "bf16"), defect="DEF-03"),
    "X08": dict(B=65537, Hq=1, Hk=1, Hv=1, Sq=16, Skv=16, Dqk=64, Dv=64, f=("G1",)),
    "X09": dict(**_H2, Sq=16, Skv=65537, f=("G6",)),
    "X10": dict(**_H2, Sq=16, Skv=65537, R=0, attrs={"alibi_mask": True}, f=("A8", "G6"), oracle="14358+derivation"),
    "X11": dict(**_H1, Sq=4096, Skv=4096, R=0, f=("V6",)),
    "X12": dict(**_H1, Sq=16384, Skv=16384, R=0, f=("V6",), oracle="14358+timing"),
    "X13": dict(B=78, Hq=16, Hk=16, Hv=16, Sq=64, Skv=64, Dqk=80, Dv=80, f=("G1",)),
    "X14": dict(**_H1, Sq=1, Skv=1, f=("T24",), oracle="unit"),        # int32 narrowing: descriptor-only
}
# fmt: on


def catalog_shape(sid):
    """The plan item of an S, H or X shape."""
    kw = dict({"S": S_SHAPES, "H": H_SHAPES, "X": X_SHAPES}[sid[0]][sid])
    feats = kw.pop("f")
    return new(id=sid, set=sid[0], items=list(feats), dtypes=kw.pop("dt", BF16), **kw)


# M: model layers (sources in the plan, §3.3). keep_g keeps the GQA ratio;
# keep_window keeps the model window; fixed_kv keeps the text Skv.
# fmt: off
M_MODELS = OrderedDict([
    ("M01", dict(B=1, Hq=32, Hk=8, Hv=8, Sq=8192, Skv=8192, R=0)),                      # Llama-3.1-8B
    ("M02", dict(B=1, Hq=8, Hk=1, Hv=1, Sq=8192, Skv=8192, R=0)),                       # Llama-3.3-70B TP8
    ("M03", dict(B=1, Hq=16, Hk=1, Hv=1, Sq=4096, Skv=65536, R=0, align="BR", keep_g=True)),  # 405B CP
    ("M04", dict(B=1, Hq=28, Hk=4, Hv=4, Sq=4096, Skv=4096, R=0)),                      # Qwen2.5-7B
    ("M05", dict(B=4, Hq=14, Hk=2, Hv=2, Sq=4096, Skv=4096, Dqk=64, Dv=64, R=0)),       # Qwen2.5-0.5B
    ("M06", dict(B=1, Hq=64, Hk=4, Hv=4, Sq=32768, Skv=32768, R=0)),                    # Qwen3-235B
    ("M07", dict(B=1, Hq=32, Hk=2, Hv=2, Sq=8192, Skv=8192, Dqk=256, Dv=256, R=0)),     # Qwen3.5-397B
    ("M08", dict(B=1, Hq=24, Hk=4, Hv=4, Sq=8192, Skv=8192, Dqk=256, Dv=256, R=0)),     # Qwen3.5-27B
    ("M09", dict(B=1, Hq=128, Hk=128, Hv=128, Sq=4096, Skv=4096, Dqk=192, Dv=128, R=0, scale=0.1352)),  # DSv3
    ("M10", dict(B=1, Hq=64, Hk=64, Hv=64, Sq=32768, Skv=32768, Dqk=192, Dv=128, R=0, scale=0.1352)),  # Kimi-K2
    ("M11", dict(B=1, Hq=20, Hk=20, Hv=20, Sq=4096, Skv=4096, Dqk=256, Dv=256, R=0)),   # GLM-4.7-Flash
    ("M12", dict(B=1, Hq=96, Hk=8, Hv=8, Sq=4096, Skv=4096, R=0)),                      # GLM-4.5
    ("M13", dict(B=1, Hq=32, Hk=8, Hv=8, Sq=8192, Skv=8192, W=4096, R=0)),              # Mistral-7B SWA
    ("M14", dict(B=1, Hq=32, Hk=32, Hv=32, Sq=8192, Skv=8192, W=4096, R=0)),            # OLMo-3-7B SWA
    ("M15", dict(B=1, Hq=16, Hk=8, Hv=8, Sq=32768, Skv=32768, Dqk=256, Dv=256, W=1024, R=0)),  # Gemma-3-12B
    ("M16", dict(B=1, Hq=4, Hk=1, Hv=1, Sq=32768, Skv=32768, Dqk=256, Dv=256, W=512, R=0)),    # Gemma-3-1B
    ("M17", dict(B=1, Hq=32, Hk=16, Hv=16, Sq=32768, Skv=32768, R=0, scale=round(168**-0.5, 8))),  # Gemma-3-27B
    ("M18", dict(B=1, Hq=32, Hk=4, Hv=4, Sq=32768, Skv=32768, Dqk=512, Dv=512, R=0, scale=1.0)),   # Gemma-4
    ("M19", dict(B=1, Hq=32, Hk=16, Hv=16, Sq=32768, Skv=32768, Dqk=256, Dv=256, W=1024, R=0, scale=1.0)),
    ("M20", dict(B=1, Hq=64, Hk=8, Hv=8, Sq=4096, Skv=4096, Dqk=64, Dv=64, W=128, R=0, keep_window=True)),  # gpt-oss
    ("M21", dict(B=1, Hq=64, Hk=8, Hv=8, Sq=4096, Skv=4096, Dqk=64, Dv=64, R=0)),       # gpt-oss dense
    ("M22", dict(B=1, Hq=64, Hk=2, Hv=2, Sq=8192, Skv=8192, R=0)),                      # Nemotron-3-Ultra
    ("M23", dict(B=4, Hq=40, Hk=8, Hv=8, Sq=8192, Skv=8192, R=0)),                      # Llama-4 Scout chunked
    ("M24", dict(B=2, Hq=32, Hk=32, Hv=32, Sq=4096, Skv=4096, Dqk=96, Dv=96, R=0)),     # Phi-3-mini
    ("M25", dict(B=1, Hq=64, Hk=1, Hv=1, Sq=8192, Skv=8192, Dqk=512, Dv=512, W=128, R=0, keep_window=True)),  # DSv4
    ("M26", dict(B=64, Hq=12, Hk=12, Hv=12, Sq=197, Skv=197, Dqk=64, Dv=64)),           # ViT-B/16
    ("M27", dict(B=32, Hq=16, Hk=16, Hv=16, Sq=577, Skv=577, Dqk=64, Dv=64)),           # CLIP ViT-L/14-336
    ("M28", dict(B=32, Hq=16, Hk=16, Hv=16, Sq=729, Skv=729, Dqk=72, Dv=72)),           # SigLIP so400m
    ("M29", dict(B=1, Hq=16, Hk=16, Hv=16, Sq=4988, Skv=4988, Dqk=80, Dv=80)),          # Qwen2.5-VL ViT
    ("M30", dict(B=8, Hq=20, Hk=20, Hv=20, Sq=1500, Skv=1500, Dqk=64, Dv=64)),          # Whisper encoder
    ("M31", dict(B=8, Hq=20, Hk=20, Hv=20, Sq=448, Skv=1500, Dqk=64, Dv=64, fixed_kv=True)),  # Whisper cross
    ("M32", dict(B=1, Hq=38, Hk=38, Hv=38, Sq=4429, Skv=4429, Dqk=64, Dv=64)),          # SD3.5 MMDiT
    ("M33", dict(B=1, Hq=24, Hk=24, Hv=24, Sq=4608, Skv=4608)),                         # FLUX.1-dev
    ("M34", dict(B=1, Hq=40, Hk=40, Hv=40, Sq=32760, Skv=32760)),                       # Wan2.1 self
    ("M35", dict(B=1, Hq=40, Hk=40, Hv=40, Sq=32760, Skv=512, fixed_kv=True)),          # Wan2.1 cross
    ("M36", dict(B=1, Hq=8, Hk=1, Hv=1, Sq=8192, Skv=4096)),                            # CP ring step, lower
    ("M37", dict(B=1, Hq=8, Hk=1, Hv=1, Sq=4096, Skv=8192)),                            # CP ring step, upper
    ("M38", dict(B=1, Hq=12, Hk=12, Hv=12, Sq=8192, Skv=8192, Dqk=64, Dv=64, W=65, R=64, keep_window=True)),
    ("M39", dict(B=8, Hq=20, Hk=20, Hv=20, Sq=448, Skv=448, Dqk=64, Dv=64, R=0)),       # Whisper decoder
    ("M40", dict(B=4, Hq=32, Hk=8, Hv=8, Sq=3000, Skv=3000, R=0, seqlens=[3000, 1500, 2600, 1092], oracle="14358")),
    ("M41", dict(B=1, Hq=32, Hk=32, Hv=32, Sq=4096, Skv=4096, R=0)),                    # MHA D128 causal
    ("M42", dict(B=2, Hq=128, Hk=128, Hv=128, Sq=2048, Skv=2048, Dqk=192, Dv=128)),     # MLA, no mask
    ("M43", dict(B=2, Hq=64, Hk=8, Hv=8, Sq=2048, Skv=2048)),                           # GQA g8, no mask
    ("M44", dict(B=2, Hq=10, Hk=10, Hv=10, Sq=1024, Skv=77, Dqk=64, Dv=64, fixed_kv=True)),  # SDXL cross
    ("M45", dict(B=1, Hq=32, Hk=32, Hv=32, Sq=17556, Skv=17556)),                       # LTX-2 video
    ("M46", dict(B=1, Hq=128, Hk=1, Hv=1, Sq=2048, Skv=2048, Dqk=192, Dv=128, R=0)),    # MQA with Dqk != Dv
    ("M47", dict(B=2, Hq=128, Hk=1, Hv=1, Sq=2048, Skv=2048, Dqk=512, Dv=512, W=128, R=0)),  # DSv4-Pro window
])
M_PR = ("M01", "M03", "M07", "M09", "M20", "M28", "M31", "M36", "M41", "M44", "M45")  # with a PR variant
M_NO_NIGHTLY = ("M47",)  # 129 MiB even at S128: full only
M_NO_MODEL_LAYER = ("M44",)  # its nightly variant is the model layer
# Corpus graphs store causal TL as causal_mask=true (plus left_bound).
M_CORPUS_ENCODED = ("M01", "M02", "M06", "M07", "M09", "M10", "M13", "M20", "M23", "M25", "M41", "M46", "M47")
M_PR_TARGET_S, M_NIGHTLY_TARGET_S = 2.5, 6.0  # budget-driven targets below the ceilings
# Hand-sized variants: they keep the model's GQA ratio or window.
M_VARIANTS = {
    "M03": {"pr": dict(B=1, Hq=8, Hk=1, Hv=1, Sq=32, Skv=512),          # g16 -> 8, Sq:Skv 1:16 kept
            "nightly": dict(B=1, Hq=16, Hk=1, Hv=1, Sq=32, Skv=512)},   # g16 and 1:16 kept
    "M07": {"pr": dict(B=1, Hq=16, Hk=1, Hv=1, Sq=64, Skv=64)},         # g16 kept, S -> 64
    "M25": {"nightly": dict(B=1, Hq=16, Hk=1, Hv=1, Sq=128, Skv=128, W=32)},  # W = S/4 still binds
    "M44": {"nightly": {}},                                              # the model layer fits nightly
    "M40": {"pr": dict(B=4, Hq=8, Hk=2, Hv=2, Sq=130, Skv=130, seqlens=[61, 130, 47, 18]),  # packed documents
            "nightly": dict(B=5, Hq=4, Hk=1, Hv=1, Sq=700, Skv=700, seqlens=[700, 301, 512, 1, 535])},
}
M_FULL_CPU = {  # CPU-reference anchors in full beyond the model layers
    "M15": dict(B=1, Hq=2, Hk=1, Hv=1, Sq=2048, Skv=2048, W=1024),     # model window kept, g2
    "M01": dict(B=1, Hq=4, Hk=1, Hv=1, Sq=2048, Skv=2048),              # g4 at S = 2048 causal
}
# fmt: on


def _m_cost(e):
    W, R, al = e.get("W"), e.get("R"), e.get("align", "TL")
    args = (e["B"], e["Hq"], e["Hk"], e["Hv"], e["Sq"], e["Skv"], e["Dqk"], e["Dv"])
    if e.get("seqlens"):  # one batch entry per document, the mask per document
        return shape_cost(
            *args, U=e["Hq"] * sum(kept_pairs(d, d, W, R, al) for d in e["seqlens"])
        )
    return shape_cost(*args, W, R, al)


def _shrink_len(s):
    """Halve a length, keeping s mod 64 while the base is >= 128."""
    res = s % 64
    nb = ((s - res) // 2) // 64 * 64
    return max(s // 2, 1) if nb == 0 else nb + res


def _halve_seq(r, orig, keep_window):
    if r.get("fixed_kv"):
        nsq, nskv = _shrink_len(r["Sq"]), r["Skv"]
    elif orig["Sq"] == orig["Skv"]:
        nsq = nskv = _shrink_len(r["Sq"])
    else:
        ratio = orig["Skv"] / orig["Sq"]
        if ratio >= 1:
            nsq = _shrink_len(r["Sq"])
            nskv = max(int(round(nsq * ratio)), 1)
        else:
            nskv = _shrink_len(r["Skv"])
            nsq = max(int(round(nskv / ratio)), 1)
    if (nsq, nskv) == (r["Sq"], r["Skv"]) or min(nsq, nskv) < 16:
        return False
    r["Sq"], r["Skv"] = nsq, nskv
    if r.get("W") is not None and not keep_window and min(nsq, nskv) < 2 * r["W"]:
        r["W"] = max(min(nsq, nskv) // 4, 1)
    return True


def reduce_model(m, target_s, s_floor, level, keep_window=False):
    """Model-layer reduction (§3.3). Keeps the model's GQA ratio g as long as
    possible: B -> 1; heads -> (2g, 2, 2) (MHA -> 2/2/2); halve S (keep
    Sq:Skv, S mod 64 and a fixed text Skv) down to s_floor; Hkv -> 1 keeping
    g; halve S down to 64; halve g; halve S. A halved S shrinks a window that
    no longer binds to S/4 unless keep_window is set."""
    r = dict(m)
    cap = CEIL_BYTES[level] or 10**18

    def ok():
        c = _m_cost(r)
        return c["s"] <= target_s and c["bytes"] <= cap

    def short():
        return min(r["Sq"], r["Skv"] if not r.get("fixed_kv") else 10**9)

    if ok():
        return r
    r["B"] = 1
    g = r["Hq"] // r["Hk"]
    if ok():
        return r
    if not m.get("keep_g") and r["Hk"] > 2:
        r["Hq"], r["Hk"], r["Hv"] = (2, 2, 2) if g == 1 else (2 * g, 2, 2)
    while not ok() and short() > s_floor and _halve_seq(r, m, keep_window):
        pass
    if not ok() and r["Hk"] > 1 and g > 1:
        r["Hq"], r["Hk"], r["Hv"] = g, 1, 1
    while not ok() and short() > 64 and _halve_seq(r, m, keep_window):
        pass
    while (
        not ok() and r["Hq"] > 1 and r["Hq"] % 2 == 0 and r["Hq"] // max(r["Hk"], 1) > 1
    ):
        r["Hq"] //= 2
    while not ok() and _halve_seq(r, m, keep_window):
        pass
    return r


def _m_key(v):
    keys = GEOMETRY + ("W", "R", "align", "scale")
    return tuple(v.get(k) for k in keys) + (tuple(v.get("seqlens") or ()),)


def _m_record(mid, suffix, v):
    """A model variant: its features from its geometry, and a gate when no
    CPU-reference level can afford it."""
    f = ["G2"]
    if v["Hq"] != v["Hk"]:
        f.append("G3")
    if (v["Dqk"], v["Dv"]) != (128, 128):
        f.append("G8" if v["Dqk"] == v["Dv"] else "G9")
    if v["Sq"] != v["Skv"]:
        f.append("G7")
    if v.get("W") is not None:
        f.append("A4")
    if v.get("R") not in (None, 0):
        f.append("A5")
    if v.get("align", "TL") == "BR":
        f.append("A2")
    if v.get("scale", ISQ) != ISQ:
        f.append("A1")
    if v.get("seqlens"):
        f += ["A9", "G13"]
    oracle = v.get("oracle", "ready")
    cost = _m_cost(v)
    if fits_level(cost) is None and oracle == "ready":
        oracle = (
            "14358+gpufwd"
            if cost["fwd_derivation_s"] > CEIL_S["full"]
            else "14358+timing"
        )
    return dict(v, id=f"{mid}.{suffix}", features=f, oracle=oracle)


def model_variants():
    """Every model's variants in catalog order: the model layer (.f), the PR
    (.pr) and nightly (.n) variants and the full CPU anchors (.fc). A nightly
    variant equal to a PR variant is dropped (the cascade runs it); so is a
    variant equal to an earlier model's variant at the same level."""
    out, seen, pr_keys = [], set(), set()
    for mid, m in M_MODELS.items():
        mm = dict(DEFAULT, align="TL", W=None, R=None, scale=ISQ)
        mm.update(m)
        if mid not in M_NO_MODEL_LAYER:
            out.append(_m_record(mid, "f", mm))
        hand = M_VARIANTS.get(mid, {})
        variants = []
        if mid in M_PR or "pr" in hand:
            pr = (
                dict(mm, **hand["pr"])
                if "pr" in hand
                else reduce_model(mm, M_PR_TARGET_S, 64, "pr")
            )
            variants.append(("pr", pr))
        if mid not in M_NO_NIGHTLY:
            keep = m.get("keep_window", False)
            if "nightly" in hand:
                variants.append(("nightly", dict(mm, **hand["nightly"])))
            else:
                variants.append(
                    (
                        "nightly",
                        reduce_model(mm, M_NIGHTLY_TARGET_S, 128, "nightly", keep),
                    )
                )
        if mid in M_FULL_CPU:
            variants.append(("fullcpu", dict(mm, **M_FULL_CPU[mid])))
        for lvl, v in variants:
            k = _m_key(v)
            if (lvl == "nightly" and k in pr_keys) or (lvl, k) in seen:
                continue
            seen.add((lvl, k))
            if lvl == "pr":
                pr_keys.add(k)
            out.append(
                _m_record(mid, {"pr": "pr", "nightly": "n", "fullcpu": "fc"}[lvl], v)
            )
    return out


# L: one feature each at medium size. Base: the default at 4x length
# (288 x 480); the builder halves B, then Hq (never below 2), then S, until the
# case fits L_TARGET_S (pilot model).
L_TARGET_S = 5.0
L_BASE = dict(B=2, Hq=3, Hk=3, Hv=3, Sq=288, Skv=480, Dqk=128, Dv=128)
RUNTIME_SCALE = {"tensor": "runtime", "value": 0.5}
# fmt: off
L_SHAPES = OrderedDict([
    ("L00", ("G5 G6 G7", {}, "bf16")),
    ("L01", ("A3 AW3", dict(R=0), "fp16")),
    ("L02", ("A2 A3", dict(R=0, align="BR"), "bf16")),
    ("L03", ("A4", dict(W=128, R=0), "fp16")),
    ("L04", ("A4 A2", dict(W=128, R=0, align="BR"), "bf16")),
    ("L05", ("A5", dict(W=65, R=64), "fp16")),
    ("L06", ("AW2 G7", dict(Sq=480, Skv=288, R=0, align="BR"), "bf16")),  # 192 dead rows
    ("L07", ("G7", dict(Sq=480, Skv=480, R=0), "fp16")),
    ("L08", ("G3", dict(Hq=3, Hk=1, Hv=1, R=0), "bf16")),
    ("L09", ("G3", dict(Hq=6, Hk=3, Hv=3, R=0), "fp16")),
    ("L10", ("G4", dict(Hq=4, Hk=2, Hv=1, R=0), "bf16")),
    ("L11", ("G8", dict(Dqk=192, Dv=192, R=0), "fp16")),
    ("L12", ("G9", dict(Dqk=192, Dv=128, R=0), "bf16")),
    ("L13", ("G8", dict(Dqk=256, Dv=256, R=0), "fp16")),
    ("L14", ("G8", dict(Dqk=64, Dv=64), "bf16")),
    ("L15", ("T18", dict(R=0, layout="bshd"), "fp16")),
    ("L16", ("AW12", dict(R=0, layout="do_bshd"), "bf16")),
    ("L17", ("AW12 G10", dict(R=0, layout="grads_bshd"), "fp16")),
    ("L18", ("T18", dict(R=0, layout="rowpad8"), "bf16")),
    ("L19", ("A1", dict(R=0, scale="unset"), "fp16")),
    ("L20", ("T12 T22", dict(R=0, scale=RUNTIME_SCALE), "bf16")),
    ("L21", ("T10 G11", dict(R=0, bias=bias_spec("11")), "fp16")),
    ("L22", ("G11 DT4", dict(bias=bias_spec("full", "io")), "bf16")),
    ("L23", ("AW13 AW5", dict(R=0, align="BR", bias=bias_spec("1h", "fp16")), "bf16")),
    ("L24", ("A6 AW7", dict(W=128, attrs={"causal_mask": True}), "fp16")),  # deprecated flag + window
    ("L25", ("AW4 A4", dict(W=1, R=0), "bf16")),
    ("L26", ("AW10", dict(W=287, R=0), "fp16")),                          # one key short of covering
    ("L27", ("V1", dict(inputs=DOMAIN["v01"]), "bf16")),
    ("L28", ("V1 V7", dict(inputs=DOMAIN["peaked"]), "fp16")),
    ("L29", ("A4 A5", dict(W=128), "bf16")),
    ("L30", ("A3 A5", dict(R=64), "fp16")),
    ("L31", ("AW11 G3", dict(Hq=4, Hk=2, Hv=2, W=128, R=0), "bf16")),
    ("L32", ("DT1", dict(R=0), "fp32")),
    # fixed size: odd batch and heads; the decode-shaped backward
    ("L33", ("G1 G2", dict(B=3, Hq=5, Hk=5, Hv=5, Sq=128, Skv=192, R=0, fixed=True), "fp16")),
    ("L34", ("G5 G7 V6", dict(B=2, Hq=3, Hk=3, Hv=3, Sq=1, Skv=4096, fixed=True), "bf16")),
    ("L35", ("G3 AW11", dict(Hq=8, Hk=1, Hv=1, R=0, align="BR"), "fp16")),
])
# fmt: on


def l_shape(lid):
    """The L item: the features of S on the medium base, reduced to fit."""
    feats, ch, dt = L_SHAPES[lid]
    ch = dict(ch)
    fixed = ch.pop("fixed", False)
    it = new(id=lid, set="L", dtypes=(dt,), **dict(L_BASE, **ch))
    if it.get("bias"):
        it["topology"] = "AdditiveMask"
    while not fixed and case_cost(it)["s"] > L_TARGET_S:
        if it["B"] > 1:
            it["B"] //= 2
        elif it["Hq"] > 2 and it["Hk"] == it["Hq"] and it["Hv"] == it["Hq"]:
            it["Hq"] = it["Hk"] = it["Hv"] = max(2, it["Hq"] // 2)
        else:
            it["Sq"] = max(64, it["Sq"] // 2) if it["Sq"] > 1 else 1
            it["Skv"] = max(64, it["Skv"] // 2)
            if it.get("W") and it["W"] >= it["Skv"]:
                it["W"] = it["Skv"] - 1
    items = feats.split()
    it["items"] = items + (["A3"] if it.get("R") == 0 and "A3" not in items else [])
    return it


# Golden set (§5): golden-backed cases. #14358's generator writes their
# tensors (lane GT1); until then they run against the CPU reference. GL-09 and
# GL-16b are not selected: a Stats input holds -inf on dead rows, which RFC
# 0022 §12.2 forbids (the SEM-010 amendment is pending).
_GL = dict(B=1, Hq=2, Hk=2, Hv=2, Sq=113, Skv=203, Dqk=128, Dv=128)
_GL64 = dict(B=2, Hq=4, Hk=2, Hv=2, Sq=64, Skv=96, Dqk=64, Dv=64)
_CM = {"causal_mask": True, "causal_mask_bottom_right": False}
_DROP = {"p": 0.2, "scale": 1.25, "scale_inv": 0.8, "mask_dims": "full"}
_PADKV = {"seq_len_q": [64, 64, 64], "seq_len_kv": [96, 71, 64], "padding_mask": False}
_GT1_DROPOUT = "GT1, the #14312 derivation extension for dropout and HG4 (the live counterpart's gate)"
_GT1_PADDING = "GT1, the #14312 derivation extension for padding and HG4 (the live counterpart's gate)"
# fmt: off
GOLDEN = OrderedDict([
    # id: (level, purpose, criteria, class, fields)
    ("GL-01", ("quick", "claim", "G-A G-E", "D", dict(_GL, regression_of="#8469"))),
    ("GL-02", ("pr", "claim", "G-A", "D", dict(_GL, dt="fp16"))),
    ("GL-03", ("pr", "claim", "G-A", "D", dict(_GL, layout="bshd"))),
    ("GL-04", ("pr", "probe", "G-A G-C G-E", "D", dict(_GL, align="BR", R=0, bias=bias_spec([1, 1, 113, 203], "bf16"),
                                                      topology="AdditiveMask", regression_of="#13474"))),
    ("GL-05", ("pr", "probe", "G-A G-E", "D", dict(_GL, dt="fp16", bias=bias_spec([1, 2, 113, 203], "fp16"),
                                                  topology="AdditiveMask", regression_of="#13474"))),
    ("GL-06", ("pr", "claim", "G-C", "P", dict(_GL, R=0))),
    ("GL-07", ("pr", "claim", "G-C", "D", dict(_GL, align="BR", R=0))),
    ("GL-08", ("pr", "claim", "G-C", "P", dict(_GL, Sq=203, Skv=113, align="BR", R=90))),
    ("GL-10", ("pr", "claim", "G-C G-E", "P", dict(_GL, align="BR", W=2, R=0, regression_of="SEM-001 (#14483)"))),
    ("GL-11", ("pr", "regression", "G-C G-E", "U", dict(_GL, scale="unset", regression_of="#12926"))),
    ("GL-12", ("pr", "probe", "G-C", "P", dict(_GL64, align="BR", R=0))),
    ("GL-13", ("pr", "regression", "G-E G-C", "D", dict(_GL, align="BR", attrs=_CM, regression_of="#13065"))),
    ("GL-14", ("pr", "probe", "G-B", "P", dict(_GL64, align="BR", R=0, bias=bias_spec([1, 4, 64, 96], "bf16"),
                                              dbias=True,
                                              topology="AdditiveMaskDBias", oracle="14358",
                                              gate="GT1 (live dBias cases open at #14358 too)"))),
    ("GL-15", ("pr", "probe", "G-B", "P", dict(_GL64, align="BR", R=0, inputs=DOMAIN["u01"], dropout=_DROP,
                                              topology="DropoutMask", oracle="14358+derivation+HG4",
                                              gate=_GT1_DROPOUT))),
    ("GL-16a", ("pr", "probe", "G-B", "P", dict(_GL64, B=3, Hq=2, align="BR", R=0, padding=_PADKV,
                                               topology="Padding", oracle="14358+derivation+HG4", gate=_GT1_PADDING))),
    ("GL-17", ("pr", "claim", "G-D", "U", dict(_GL, inputs=DOMAIN["peaked"]))),
    ("GL-18", ("pr", "claim", "G-D G-A", "D", dict(_GL, dt="fp16", layout="bshd", inputs=DOMAIN["v01"]))),
    ("GL-19", ("nightly", "claim", "G-D", "P", dict(_H1, Sq=896, Skv=64, dt="fp16", set="H"))),
    ("GL-20", ("nightly", "probe", "G-F G-C", "P", dict(_H2, Sq=192, Skv=192, Dqk=192, R=0, scale=0.1352337788608801,
                                                       set="M"))),
    ("GL-21", ("nightly", "probe", "G-F G-E G-C", "P", dict(B=1, Hq=8, Hk=1, Hv=1, Sq=192, Skv=192, Dqk=64, Dv=64,
                                                           W=128, attrs=_CM, regression_of="#13065", set="M"))),
])
# fmt: on
GOLDEN_NOT_SELECTED = {
    "GL-09": "BR causal at 203x113 with 90 dead rows per head: LSE = -inf input",
    "GL-16b": "padding on both sides with per-batch dead rows: LSE = -inf input",
}


def golden_case(gid):
    level, purpose_, criteria, cls, fields = GOLDEN[gid]
    kw = dict(fields)
    return new(
        id=gid,
        level=level,
        purpose=purpose_,
        items=[gid] + criteria.split(),
        dtypes=(kw.pop("dt", "bf16"),),
        tol_class=cls,
        golden=gid,
        golden_pointer_gate=kw.pop("gate", "GT1"),
        seed=42,
        **kw,
    )
