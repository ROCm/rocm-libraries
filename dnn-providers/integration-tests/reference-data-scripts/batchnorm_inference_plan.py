#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declarations of the batchnorm inference test plan.

Axes, legality, the shape catalog (S, L, M, H, X), ceilings, ReLU variants,
value recipes and case purposes of ``docs/test-plans/batchnorm-inference.md``.
The bundle generator (``generate_batchnorm_inference_bundles.py``) and the
exhaustive tool read this module; nothing here writes files.

A case is a plain dict. Spatial rank ``rank`` counts the spatial dims only, so
rank 2 in the plan (``[N, C]``) is ``rank == 0`` here and rank 6 is ``rank == 4``.
"""

from __future__ import annotations

import hashlib
import json
import math

LEVELS = ("quick", "standard", "comprehensive", "full")  # bundle directory names

RANKS = (1, 2, 3)  # core spatial ranks; 0 and 4 are probes (F1)
LAYOUTS = ("cf", "cl")  # channels-first, channels-last
CORE_DTYPES = ("float", "half", "bfloat16")
HALF_DTYPES = ("half", "bfloat16")
PAIRS = tuple((lay, dt) for lay in LAYOUTS for dt in CORE_DTYPES)
UPAIRS = tuple((lay, dt) for lay in LAYOUTS for dt in HALF_DTYPES)

SHORT = {"float": "fp32", "half": "fp16", "bfloat16": "bf16"}
BYTES = {"float": 4, "half": 2, "bfloat16": 2}
DTYPE_MAX = {
    "float": 3.4028234663852886e38,
    "half": 65504.0,
    "bfloat16": 3.3895313892515355e38,
}

# (x, scale and bias, mean and inv_variance, y, compute) signatures the references register
# (GpuBatchnormFwdInfSignatureKey.hpp, BatchnormFwdInferenceSignatureKey.hpp).
SIGNATURES = (
    ("float", "float", "float", "float", "float"),
    ("half", "float", "float", "half", "float"),
    ("bfloat16", "float", "float", "bfloat16", "float"),
    ("half", "float", "float", "float", "float"),
    ("bfloat16", "float", "float", "float", "float"),
    ("half", "half", "half", "half", "float"),
    ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "float"),
    ("half", "half", "half", "half", "half"),
    ("bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"),
)

TOPOLOGIES = ("plain", "relu")
BUNDLE_DIR = {
    "plain": ("BatchnormInference", "Default"),
    "relu": ("BatchnormInferencePointwise", "Default"),
}

# relu_fwd fields (lower clip, lower clip slope, upper clip). 0.1 and 0.01 are written
# as their float32 values, as the captured bundles stored them.
F32_0P1 = 0.10000000149011612
F32_0P01 = 0.009999999776482582
ACTS = {
    "relu": (None, None, None),
    "relu_lo0": (0.0, None, None),
    "relu6": (None, None, 6.0),
    "clamp": (F32_0P1, None, 0.5),
    "relu_slope0": (None, 0.0, None),
    "relu_lo0p1": (F32_0P1, None, None),
    "leaky": (None, F32_0P01, None),
}
CLAIM_ACTS = (
    "relu",
    "relu_lo0",
    "relu6",
    "clamp",
)  # rotated as ACTS[k % 4] in the plan
CLIPS = ("relu_lo0", "relu6", "clamp")  # F20 claimed clips
PROBE_ACTS = ("relu_lo0p1", "leaky")

# Elements of x per case, by level (section 7). None: no ceiling.
CEILINGS = {
    "quick": 1 << 16,
    "standard": 1 << 20,
    "comprehensive": 1 << 23,
    "full": None,
}

# F9 non-packed strides: padding added to one dim's extent before the next-slower
# stride is computed. np_row pads the fastest dim (rows of channels-first, the
# channel vector of channels-last); np_chan pads the outermost spatial dim, which
# pads the channel stride of channels-first and the batch stride of channels-last.
NONPACKED = ("np_row", "np_chan")
NP_PAD = {"np_row": 3, "np_chan": 1}

VALUE_REGIMES = ("default", "constant", "zero_scale", "tinyvar", "large", "cancel")
VALUE_TOKEN = {
    "default": "",
    "constant": "const",
    "zero_scale": "zscale",
    "tinyvar": "tinyvar",
    "large": "large",
    "cancel": "cancel",
}

DEFAULTS = dict(
    topo="plain",
    xl="cf",  # x layout
    yl=None,  # y layout; None: same as x (F8 otherwise)
    xt="float",  # x and io type
    yt=None,  # y type (F14 upcast); None: same as x
    pt="float",  # statistic and affine type (F15)
    comp="float",  # batchnorm node compute type (F16)
    inter_t="float",  # fused virtual y type (F21)
    np=None,  # F9 non-packed kind, on x and y
    stats_rank="full",  # "full", or "2" for [1, C] statistics (F12)
    act=None,  # ReLU variant of the fused graph
    vals="default",  # F17 value regime
)


def new(size, ids, rank, n, c, sp, **kw):
    case = dict(DEFAULTS)
    case.update(size=size, ids=tuple(ids), rank=rank, n=n, c=c, sp=tuple(sp))
    case.update(kw)
    return finalize(case)


def finalize(c):
    c = dict(c)
    if c["yl"] is None:
        c["yl"] = c["xl"]
    if c["yt"] is None:
        c["yt"] = c["xt"]
    if c["topo"] == "relu" and c["act"] is None:
        c["act"] = "relu"
    if c["topo"] == "plain":
        c["act"] = None
        c["inter_t"] = "float"
    if c["rank"] == 0:
        c["xl"] = c["yl"] = "cf"  # [N, C] has one layout
    return c


# --------------------------------------------------------------------------
# Tensors
# --------------------------------------------------------------------------


def x_dims(c):
    return [c["n"], c["c"], *c["sp"]]


def stat_dims(c):
    if c["stats_rank"] == "2":
        return [1, c["c"]]
    return [1, c["c"]] + [1] * c["rank"]


def strides(dims, layout, np=None):
    """Strides of ``dims`` (N, C, spatial...) channels-first ('cf') or channels-last
    ('cl'), packed unless ``np`` names a padding of section 6.3 (F9)."""
    r = len(dims) - 2
    spatial = list(range(r + 1, 1, -1))  # fastest first
    order = spatial + [1, 0] if layout == "cf" else [1] + spatial + [0]
    pad = [0] * len(dims)
    if np == "np_row":
        pad[order[0]] = NP_PAD[np]
    elif np == "np_chan" and r:
        pad[2] = NP_PAD[np]
    st = [0] * len(dims)
    acc = 1
    for dim in order:
        st[dim] = acc
        acc *= dims[dim] + pad[dim]
    return st


def stat_strides(c):
    return strides(stat_dims(c), "cf")


def numel(dims):
    return math.prod(dims)


def elements(c):
    return numel(x_dims(c))


def tensor_bytes(c):
    """Bytes of x and y as allocated (padded strides included)."""
    xd = x_dims(c)
    span = 1 + sum((d - 1) * s for d, s in zip(xd, strides(xd, c["xl"], c["np"])))
    yspan = 1 + sum((d - 1) * s for d, s in zip(xd, strides(xd, c["yl"], c["np"])))
    return span * BYTES[c["xt"]] + yspan * BYTES[c["yt"]]


def fits(c, level):
    ceiling = CEILINGS[level]
    return ceiling is None or elements(c) <= ceiling


# --------------------------------------------------------------------------
# Legality (BatchnormInferenceNode::pre_validate_node)
# --------------------------------------------------------------------------


def legal(c):
    """None if the frontend accepts the graph, else the rule it breaks (section 6.8)."""
    xd = x_dims(c)
    if len(xd) < 2:
        return "R2 x has fewer than 2 dims"
    sd = stat_dims(c)
    if len(sd) < 2:
        return "R3 statistic with fewer than 2 dims"
    if sd[0] != 1 or sd[1] != c["c"] or any(d != 1 for d in sd[2:]):
        return "R4-R6 statistic shape"
    if min(xd) < 1:
        return "empty dim"
    if c["topo"] == "relu" and c["act"] not in ACTS:
        return f"unknown ReLU variant {c['act']}"
    if c["topo"] == "plain" and c["act"] is not None:
        return "activation on the plain graph"
    for t in (c["xt"], c["yt"], c["pt"], c["comp"], c["inter_t"]):
        if t not in CORE_DTYPES:
            return f"unknown data type {t}"
    return None


def signature(c):
    return (c["xt"], c["pt"], c["pt"], c["yt"], c["comp"])


# --------------------------------------------------------------------------
# Purpose (section 9)
# --------------------------------------------------------------------------


def probe_reasons(c):
    why = []
    if c["rank"] not in RANKS:
        why.append("F1 rank 2" if c["rank"] == 0 else "F1 rank 6")
    if c["xl"] != c["yl"]:
        why.append("F8 mixed layouts")
    if c["np"]:
        why.append(f"F9 non-packed strides ({c['np']})")
    if "X3" in c["ids"]:
        why.append("F11 64-bit indexing")
    if c["stats_rank"] != "full":
        why.append("F12 statistic rank below x rank")
    if c["pt"] != "float":
        why.append("F15 statistics in the io type")
    if c["comp"] != "float":
        why.append("F16 node compute type")
    if c["act"] in PROBE_ACTS:
        why.append(f"F20 {c['act']}")
    if c["topo"] == "relu" and c["inter_t"] != "float":
        why.append("F21 io-type intermediate")
    return why


def purpose(c):
    if probe_reasons(c):
        return "probe"
    return "regression" if c.get("regression") else "claim"


# --------------------------------------------------------------------------
# Value recipes (section 6.5). role -> ("free", lo, hi) or ("fixed", value).
# --------------------------------------------------------------------------

HARNESS_DEFAULT = dict(
    x=("free", -1.0, 1.0),
    scale=("free", -1.0, 1.0),
    bias=("free", -1.0, 1.0),
    mean=("free", -0.1, 0.1),
    inv_variance=("free", 0.5, 1.5),
)


def large_bound(t):
    """|x| bound so that |y| reaches about a quarter of the type's maximum:
    |y| <= |scale| * (|x| + |mean|) * inv_variance + |bias| <= 1.5 * (a + 0.1) + 1."""
    a = DTYPE_MAX[t] / 4 / 1.5
    return float(f"{a:.2e}")


def value_recipes(c):
    rec = dict(HARNESS_DEFAULT)
    v = c["vals"]
    if v == "constant":
        rec = dict(
            x=("fixed", 1.5),
            scale=("fixed", 0.5),
            bias=("fixed", 0.25),
            mean=("fixed", 0.5),
            inv_variance=("fixed", 2.0),
        )  # y = 0.5 * (1.5 - 0.5) * 2 + 0.25 = 1.25, exact in every type
    elif v == "zero_scale":
        rec["scale"] = ("fixed", 0.0)
    elif v == "tinyvar":
        rec["inv_variance"] = ("free", 100.0, 1000.0)
    elif v == "large":
        a = min(large_bound(c["xt"]), large_bound(c["yt"]))
        rec["x"] = ("free", -a, a)
    elif v == "cancel":
        rec["x"] = ("free", 0.99, 1.01)
        rec["mean"] = ("fixed", 1.0)
    return rec


# --------------------------------------------------------------------------
# Graph identity
# --------------------------------------------------------------------------

LABELS = ("ids", "size", "regression", "src")


def canon(c):
    """Stable key of the graph and its data: everything except the labels."""
    return json.dumps(
        {k: v for k, v in sorted(c.items()) if k not in LABELS}, sort_keys=True
    )


def short_hash(c):
    return hashlib.sha1(canon(c).encode()).hexdigest()[:6]


# --------------------------------------------------------------------------
# Shape catalog (section 7)
# --------------------------------------------------------------------------

S_DEFAULT_SP = {0: (), 1: (64,), 2: (16, 16), 3: (8, 8, 8), 4: (4, 4, 4, 4)}
S_BASE = dict(n=2, c=16)

# id -> (features, change from the defaults, ranks)
S_SHAPES = {
    "S1": ("core", {}, (1, 2, 3)),
    "S2": ("F2", dict(n=1), (1, 2, 3)),
    "S3": ("F3", dict(c=1), (1, 2, 3)),
    "S4": ("F3", dict(c=3), (1, 2, 3)),
    "S5": ("F3", dict(c=13), (1, 2, 3)),
    "S6": ("F4", dict(sp=7), (1, 2, 3)),
    "S7a": ("F4", dict(sp={2: (16, 1), 3: (8, 1, 8)}), (2, 3)),
    "S7b": ("F4", dict(sp={2: (1, 16)}), (2,)),
    "S8": ("F4", dict(sp=1), (1, 2, 3)),
    "S9": ("F5", dict(n=1, c=256, sp=1), (1, 2, 3)),
}
S_FEATURES = {sid: f for sid, (f, _, _) in S_SHAPES.items()}


def _spatial(change, rank, default):
    if change is None:
        return default
    if isinstance(change, dict):
        return change[rank]
    return (change,) * rank


def shape_S(sid, rank, add=(), **kw):
    _, change, ranks = S_SHAPES[sid]
    if rank in RANKS and rank not in ranks:
        raise ValueError(f"{sid} has no rank {rank}")
    n = change.get("n", S_BASE["n"])
    c = change.get("c", S_BASE["c"])
    sp = _spatial(change.get("sp"), rank, S_DEFAULT_SP[rank])
    ids = [sid] + ([] if S_FEATURES[sid] == "core" else [S_FEATURES[sid]])
    return new("S", ids + list(add), rank, n, c, sp, **kw)


def s_ranks(sid):
    return S_SHAPES[sid][2]


L_BASE = {1: (8, 64, (2048,)), 2: (8, 64, (56, 56)), 3: (4, 32, (8, 28, 28))}
L_ODD_SP = {1: (2047,), 2: (55, 55), 3: (7, 27, 27)}
L_SHAPES = {
    "L1": ("core", {}),
    "L4": ("F3", dict(c=3)),
    "L5": ("F3", dict(c=13)),
    "L6": ("F4", dict(sp=L_ODD_SP)),
}


def shape_L(lid, rank, **kw):
    feat, change = L_SHAPES[lid]
    n, c, sp = L_BASE[rank]
    c = change.get("c", c)
    if "sp" in change:
        sp = change["sp"][rank]
    ids = [lid] + ([] if feat == "core" else [feat])
    return new("L", ids, rank, n, c, sp, **kw)


# id -> (N, C, spatial, source)
M_SHAPES = {
    "M1": (8, 64, (112, 112), "ResNet-50 stem batchnorm"),
    "M2": (8, 256, (56, 56), "ResNet-50 stage-1 batchnorm"),
    "M3": (32, 2048, (7, 7), "ResNet-50 stage-4 batchnorm"),
    "M4": (8, 96, (56, 56), "MobileNetV2 depthwise-block batchnorm"),
    "M5": (2, 32, (32, 32, 32), "3D U-Net batchnorm"),
    "M6": (2, 64, (8, 56, 56), "I3D batchnorm"),
    "M7": (8, 256, (128,), "Conformer convolution-module batchnorm (1D)"),
}


def shape_M(mid, **kw):
    n, c, sp, src = M_SHAPES[mid]
    return new("M", [mid], len(sp), n, c, sp, src=src, **kw)


# id -> (feature, change from the S defaults, applies at every core rank)
H_SHAPES = {
    "H1": ("F3", dict(c=15)),
    "H2": ("F3", dict(c=17)),
    "H3": ("F3", dict(c=31)),
    "H4": ("F3", dict(c=33)),
    "H5": ("F3", dict(c=63)),
    "H6": ("F3", dict(c=65)),
    "H7": ("F4", dict(sp=15)),
    "H8": ("F4", dict(sp=17)),
    "H9": ("F2", dict(n=3)),
    "H10": ("F2", dict(n=64)),
    "H11": ("F3", dict(c=2048, sp=1)),
}


def shape_H(hid, rank, **kw):
    feat, change = H_SHAPES[hid]
    n = change.get("n", S_BASE["n"])
    c = change.get("c", S_BASE["c"])
    sp = _spatial(change.get("sp"), rank, S_DEFAULT_SP[rank])
    return new("H", [hid, feat], rank, n, c, sp, **kw)


X_SHAPES = {
    "X1": (32, 64, (112, 112), "F2", {}),
    "X2": (32, 256, (56, 56), "F2", dict(xl="cl")),
    "X3": (1, 1, (46341, 46341), "F11", {}),
    "X4": (1, 16, (1048576,), "F4", {}),
}


def shape_X(xid, **kw):
    n, c, sp, feat, fixed = X_SHAPES[xid]
    args = dict(fixed)
    args.update(kw)
    ids = [xid, feat] + (["F7"] if xid == "X2" else [])
    return new("X", ids, len(sp), n, c, sp, **args)


def rank_probe(rank, **kw):
    """F1: [N, C] (rank 2) or 4D spatial (rank 6) at the S defaults."""
    return new(
        "S", ["S1", "F1"], rank, S_BASE["n"], S_BASE["c"], S_DEFAULT_SP[rank], **kw
    )


# --------------------------------------------------------------------------
# Golden baseline (section 12): id -> (level, case builder). Each is also a
# claim case of its level; the generator merges the G id into that case.
# --------------------------------------------------------------------------


def golden_cases():
    s = shape_S
    return {
        "G1": ("quick", s("S1", 1, xl="cf", xt="float")),
        "G2": ("quick", s("S1", 1, xl="cl", xt="half")),
        "G3": ("quick", s("S1", 2, xl="cf", xt="bfloat16")),
        "G4": ("quick", s("S1", 2, xl="cl", xt="float")),
        "G5": ("quick", s("S1", 3, xl="cf", xt="half")),
        "G6": ("quick", s("S1", 3, xl="cl", xt="bfloat16")),
        "G7": ("quick", s("S1", 2, topo="relu", xl="cf", xt="float")),
        "G8": ("quick", s("S1", 2, topo="relu", act="relu6", xl="cl", xt="half")),
        "G9": ("quick", s("S1", 1, topo="relu", act="clamp", xl="cf", xt="bfloat16")),
        "G10": (
            "standard",
            s("S1", 2, topo="relu", act="relu_slope0", xl="cf", xt="float"),
        ),
        "G11": ("quick", s("S1", 2, xl="cf", xt="half", yt="float")),
        "G12": ("standard", s("S9", 2, xl="cl", xt="bfloat16")),
        "G13": ("comprehensive", s("S1", 2, xl="cf", xt="half", vals="tinyvar")),
        "G14": ("comprehensive", s("S1", 2, xl="cl", xt="bfloat16", vals="large")),
        "G15": ("standard", shape_M("M7", xl="cf", xt="half")),
    }


# --------------------------------------------------------------------------
# Trade-off register (section 10). The level rules do not build these products,
# so nothing is trimmed after ids are assigned; the register is printed by --report.
# --------------------------------------------------------------------------

TRADEOFFS = {
    "T1": "Every claimed ReLU variant x every S and M shape x every pair",
    "T2": "Upcast x every S shape before nightly",
    "T3": "F15 and F16 x every S shape",
    "T4": "F8 and F9 x every S shape",
    "T5": "Leaky x every S, L and M shape",
    "T6": "H x every pair in 1D and 3D",
    "T7": "Any 3-way product beyond variant x layout x data type",
    "T8": "A separate BatchnormFwdInference sweep",
    "T9": "F22 epilogues other than relu_fwd",
    "T10": "The captured full-tier clip products of BatchnormInferencePointwise",
}

# Every plan item a case may name; --report fails on a declared item with no case.
FEATURES = tuple(
    f"F{i}" for i in (1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 20, 21)
)
PLAN_ITEMS = (
    tuple(S_SHAPES)
    + tuple(L_SHAPES)
    + tuple(M_SHAPES)
    + tuple(H_SHAPES)
    + tuple(X_SHAPES)
    + FEATURES
    + tuple(f"G{i}" for i in range(1, 16))
)
