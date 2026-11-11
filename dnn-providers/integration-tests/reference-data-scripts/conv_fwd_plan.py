#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declarations of the convolution-forward test plan.

This module is the machine-readable half of
``docs/test-plans/convolution-fwd.md``. It holds what both the bundle
generator (``generate_conv_fwd_bundles.py``) and the exhaustive tool read:

* the case model and its geometry rules (output size, strides, FLOP, bytes);
* the legality rules, which copy ``ConvolutionFpropNode::pre_validate_node``;
* the shape catalog (S, L, M, R, H, X) and the per-level size ceilings;
* the activation list, the value domains and the purpose rule;
* the products that are generated and then trimmed (``TRIMS``), each with the
  trigger that brings it back.

It has no engine matrix. The one piece of engine evidence is ``TRIMS``: it
records which products no engine executed when they were measured, and each
trim names the trigger that brings its product back. A case is a plain
``dict``; ``finalize`` resolves every default so that two spellings of one
graph compare equal.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------

LEVELS = ("quick", "standard", "comprehensive", "full")  # bundle directory names
LEVEL_NAMES = {
    "quick": "quick",
    "standard": "PR",
    "comprehensive": "nightly",
    "full": "full",
}

RANKS = (1, 2, 3)  # core spatial ranks; 4 is a probe (S23)
LAYOUTS = ("cf", "cl")  # channels-first, channels-last
CORE_DTYPES = ("float", "half", "bfloat16")
PAIRS = tuple(
    (lay, dt) for dt in CORE_DTYPES for lay in LAYOUTS
)  # 6 layout x dtype pairs

SHORT = {
    "float": "fp32",
    "half": "fp16",
    "bfloat16": "bf16",
    "double": "fp64",
    "int8": "i8",
    "int32": "i32",
    "fp8_e4m3": "f8e4m3",
    "fp8_e5m2": "f8e5m2",
    "fp8_e4m3_fnuz": "f8e4m3fnuz",
    "fp8_e5m2_fnuz": "f8e5m2fnuz",
}
BYTES = {"float": 4, "half": 2, "bfloat16": 2, "double": 8, "int8": 1, "int32": 4}
DTYPE_MAX = {"float": 3.4e38, "half": 65504.0, "bfloat16": 3.38e38}

# F19: io types beyond the core three. (x, w, y, node compute type)
EXTRA_TYPES = {
    "fp8_e4m3": ("fp8_e4m3", "fp8_e4m3", "float", "float"),
    "fp8_e5m2": ("fp8_e5m2", "fp8_e5m2", "float", "float"),
    "fp8_e4m3_fnuz": ("fp8_e4m3_fnuz", "fp8_e4m3_fnuz", "float", "float"),
    "fp8_e5m2_fnuz": ("fp8_e5m2_fnuz", "fp8_e5m2_fnuz", "float", "float"),
    "int8": ("int8", "int8", "int32", "int32"),
    "double": ("double", "double", "double", "double"),
}
# F20: mixed precision. (x, w, y, node compute type)
MIXED_TYPES = {
    "half_in_float_out": ("half", "half", "float", "float"),
    "bf16_in_float_out": ("bfloat16", "bfloat16", "float", "float"),
    "x_half_w_float": ("half", "float", "float", "float"),
    "x_float_w_half": ("float", "half", "float", "float"),
    "half_compute_half": ("half", "half", "half", "half"),
}

# F26: activations. name -> (operation, relu_lower_clip, relu_lower_clip_slope,
# relu_upper_clip, elu_alpha, swish_beta, softplus_beta)
ACTS = {
    "relu": ("relu_fwd", None, None, None, None, None, None),
    "relu_lo": ("relu_fwd", 0.1, None, None, None, None, None),
    "relu6": ("relu_fwd", None, None, 6.0, None, None, None),
    "relu_clamp": ("relu_fwd", 0.1, None, 0.5, None, None, None),
    "relu_leaky": ("relu_fwd", None, 0.1, None, None, None, None),
    "sigmoid": ("sigmoid_fwd", None, None, None, None, None, None),
    "tanh": ("tanh_fwd", None, None, None, None, None, None),
    "elu": ("elu_fwd", None, None, None, 1.0, None, None),
    "gelu": ("gelu_fwd", None, None, None, None, None, None),
    "gelu_tanh": ("gelu_approx_tanh_fwd", None, None, None, None, None, None),
    "swish": ("swish_fwd", None, None, None, None, 1.0, None),
    "softplus": ("softplus_fwd", None, None, None, None, None, 1.0),
    "identity": ("identity", None, None, None, None, None, None),
}
RELU_FAMILY = ("relu", "relu_lo", "relu6", "relu_clamp", "relu_leaky")
# Equivalent encodings of a ReLU-family activation. They are not rotated with ACTS;
# each appears only in the regression cases that pin it (plan section 4.2).
# relu_slope0: a relu_lower_clip_slope of 0 must run as standard ReLU (#14363).
ACT_ENCODINGS = {"relu_slope0": ("relu_fwd", None, 0.0, None, None, None, None)}
ACT_PARAMS = {**ACTS, **ACT_ENCODINGS}

# Fused topologies. Each is one bundle directory under ConvolutionFwdPointwise/.
TOPOLOGIES = (
    "conv",  # ConvolutionFwd/Default
    "act",  # conv -> act
    "bias_act",  # conv -> bias -> act
    "scale_bias_act",  # conv -> scale -> bias -> relu
    "add_relu",  # conv -> add(residual) -> relu
    "bias_add_relu",  # conv -> bias -> add(residual) -> relu
    "act_add",  # conv -> relu -> add(residual)
)
FUSED = TOPOLOGIES[1:]
BUNDLE_DIR = {
    "conv": ("ConvolutionFwd", "Default"),
    "act": ("ConvolutionFwdPointwise", "Default"),
    "bias_act": ("ConvolutionFwdPointwise", "Variant2"),
    "scale_bias_act": ("ConvolutionFwdPointwise", "Variant3"),
    "add_relu": ("ConvolutionFwdPointwise", "Variant4"),
    "bias_add_relu": ("ConvolutionFwdPointwise", "Variant5"),
    "act_add": ("ConvolutionFwdPointwise", "Variant6"),
}
HAS_BIAS = ("bias_act", "scale_bias_act", "bias_add_relu")
HAS_SCALE = ("scale_bias_act",)
HAS_RESIDUAL = ("add_relu", "bias_add_relu", "act_add")
HAS_ACT_PLACEHOLDERS = ("act", "bias_act")  # other fused topologies fix relu_fwd

# F21: value domains the current fill recipes can express (kind free / fixed).
VALUE_CLASSES = ("uniform", "zero", "const", "large", "tiny")

# Per-level ceilings for each case: (FLOP, bytes of fp32 x + w + y). None = unbounded.
CEILINGS = {
    "quick": (20e6, 1 << 20),
    "standard": (200e6, 16 << 20),
    "comprehensive": (8e9, 64 << 20),
    "full": (None, None),
}

# --------------------------------------------------------------------------
# Case model
# --------------------------------------------------------------------------

DEFAULTS = dict(
    topo="conv",
    xl="cf",
    wl="cf",
    yl="cf",  # layout of x, w, y
    xe=(),
    we=(),
    ye=(),  # per-dim extra stride padding (non-packed); () = packed
    xt="float",
    wt=None,
    yt=None,
    comp=None,  # data types; None resolves in finalize
    conv_mode="CROSS_CORRELATION",
    act="relu",
    bias="channel",
    bias_t=None,
    bias_first=False,  # F25
    scale="const",  # F27: const | runtime | channel
    conv_y_virtual=True,  # F30
    inter_t=None,  # F31: virtual intermediate type; None resolves
    override=False,  # F35: is_override_shape_enabled
    vals="uniform",
    ids=(),  # plan item ids this case implements
    size="S",
    alignment=None,  # F17: declared alignment; the harness cannot place it yet (I3)
)


def _tup(v, r):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,) * r


def new(rank, n, c, k, g, ins, f, s=1, d=1, pre=0, post=0, **kw):
    """Build a finalized case from per-dimension values (scalars broadcast)."""
    case = dict(DEFAULTS)
    case.update(
        rank=rank,
        n=n,
        c=c,
        k=k,
        g=g,
        ins=_tup(ins, rank),
        f=_tup(f, rank),
        s=_tup(s, rank),
        d=_tup(d, rank),
        pre=_tup(pre, rank),
        post=_tup(post, rank),
    )
    case.update(kw)
    return finalize(case)


def finalize(c):
    c = dict(c)
    c["wt"] = c["wt"] or c["xt"]
    if c["yt"] is None:
        c["yt"] = c["xt"]
    if c["comp"] is None:
        c["comp"] = {"int8": "int32", "double": "double"}.get(c["xt"], "float")
    if c["bias_t"] is None:
        c["bias_t"] = c["xt"]
    if c["inter_t"] is None:
        c["inter_t"] = c["yt"] if c["topo"] == "conv" else "float"
    for key in ("xe", "we", "ye", "ids"):
        c[key] = tuple(c[key])
    return c


def out_dim(i, f, s, d, pre, post):
    return (i + pre + post - (d * (f - 1) + 1)) // s + 1


def outs(c):
    return tuple(
        out_dim(c["ins"][i], c["f"][i], c["s"][i], c["d"][i], c["pre"][i], c["post"][i])
        for i in range(c["rank"])
    )


def legal(c):
    """Legality, copied from ConvolutionFpropNode::pre_validate_node. The
    generator never writes a graph for which this returns a reason."""
    r = c["rank"]
    if c["c"] % c["g"] or c["k"] % c["g"]:
        return "C or K not divisible by groups"
    for key in ("f", "s", "d", "pre", "post", "ins"):
        if len(c[key]) != r:
            return f"{key} count != rank"
    if min(c["s"]) < 1 or min(c["d"]) < 1:
        return "stride or dilation < 1"
    if min(c["pre"]) < 0 or min(c["post"]) < 0:
        return "negative padding"
    for i in range(r):
        if c["ins"][i] + c["pre"][i] + c["post"][i] < c["d"][i] * (c["f"][i] - 1) + 1:
            return "filter larger than padded input"
    if c["n"] < 1 or c["c"] < 1 or c["k"] < 1:
        return "empty dimension"
    return None


def x_dims(c):
    return [c["n"], c["c"], *c["ins"]]


def w_dims(c):
    return [c["k"], c["c"] // c["g"], *c["f"]]


def y_dims(c):
    return [c["n"], c["k"], *outs(c)]


def strides(dims, layout, extra=()):
    """Strides for ``dims`` laid out channels-first ('cf') or channels-last
    ('cl'). ``extra`` pads each dimension's extent when computing the stride of
    the next-slower dimension, which yields non-packed (padded or sliced)
    tensors. Dims index 0 is N (or K), 1 is C, 2.. are spatial."""
    r = len(dims) - 2
    order = (
        list(range(r + 1, 1, -1)) + [1, 0]
        if layout == "cf"
        else [1] + list(range(r + 1, 1, -1)) + [0]
    )
    st = [0] * len(dims)
    acc = 1
    for pos, dim in enumerate(order):
        st[dim] = acc
        e = extra[dim] if extra else 0
        acc *= dims[dim] + e
    return st


def fastest_dim(layout, rank):
    return rank + 1 if layout == "cf" else 1


def numel(dims):
    return math.prod(dims)


def flops(c):
    return (
        2
        * c["n"]
        * c["k"]
        * math.prod(outs(c))
        * (c["c"] // c["g"])
        * math.prod(c["f"])
    )


def fp32_bytes(c):
    return 4 * (numel(x_dims(c)) + numel(w_dims(c)) + numel(y_dims(c)))


def accum_len(c):
    return (c["c"] // c["g"]) * math.prod(c["f"])


def fits(c, level):
    fl, by = CEILINGS[level]
    return (fl is None or flops(c) <= fl) and (by is None or fp32_bytes(c) <= by)


def canon(c):
    """Stable key of the graph: everything that changes the graph or its data,
    nothing that only labels the case."""
    skip = {"ids", "size", "src", "regression"}
    return json.dumps(
        {k: v for k, v in sorted(c.items()) if k not in skip}, sort_keys=True
    )


def short_hash(c):
    return hashlib.sha1(canon(c).encode()).hexdigest()[:6]


# --------------------------------------------------------------------------
# Purpose (RFC 0022 section 9). Engine independent: a probe exists to exercise a
# feature that is unsupported, awkward or a hazard; a claim is a mainstream graph.
# --------------------------------------------------------------------------


def probe_reasons(c):
    why = []
    if c["conv_mode"] != "CROSS_CORRELATION":
        why.append("F13 conv_mode")
    if c["pre"] != c["post"]:
        why.append("F11 asymmetric padding")
    if c["rank"] not in RANKS:
        why.append("F1 4D spatial")
    if c["xe"] or c["we"] or c["ye"]:
        why.append("F16 non-packed strides")
    if not (c["xl"] == c["wl"] == c["yl"]):
        why.append("F15 mixed layouts")
    if c["alignment"]:
        why.append("F17 alignment")
    core_dt = c["xt"] in CORE_DTYPES and c["wt"] == c["xt"]
    if not core_dt or c["yt"] != c["xt"]:
        if c["topo"] == "conv" or not core_dt or c["yt"] != c["xt"]:
            why.append("F19/F20 data types")
    if c["comp"] != "float" and c["xt"] in CORE_DTYPES:
        why.append("F20 compute type")
    if c["override"]:
        why.append("F35 override shape")
    if c["vals"] != "uniform":
        why.append("F21 value domain")
    if c["size"] in ("H", "X") and "X3" not in c["ids"]:
        if c["size"] == "H":
            why.append("H hazard")
    if "X3" in c["ids"]:
        why.append("F18 64-bit indexing")
    topo = c["topo"]
    if topo in ("scale_bias_act", "add_relu", "bias_add_relu", "act_add"):
        why.append("F27-F29 topology")
    if topo in HAS_ACT_PLACEHOLDERS and c["act"] not in RELU_FAMILY + tuple(
        ACT_ENCODINGS
    ):
        why.append("F26 non-ReLU activation")
    if topo in HAS_BIAS and (
        c["bias"] != "channel" or c["bias_t"] != c["xt"] or c["bias_first"]
    ):
        why.append("F25 bias variant")
    if topo in HAS_SCALE and c["scale"] != "const":
        why.append("F27 scale kind")
    if topo != "conv" and not c["conv_y_virtual"]:
        why.append("F30 materialised conv output")
    if topo != "conv" and c["inter_t"] != "float":
        why.append("F31 intermediate type")
    return why


def purpose(c):
    """probe, regression (a case that pins an escaped bug; claimed like a claim case)
    or claim."""
    if probe_reasons(c):
        return "probe"
    return "regression" if c.get("regression") else "claim"


# --------------------------------------------------------------------------
# Shape catalog. S/L are feature functions over a base; M, H, X are tables.
# --------------------------------------------------------------------------

S_IDS = (
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12a",
    "S12b",
    "S13a",
    "S13b",
    "S14",
    "S15",
    "S16",
    "S17",
    "S18",
    "S19",
    "S20",
    "S21",
    "S22",
    "S23",
    "S24",
    "S25",
    "S26",
    "S27",
    "S28",
    "S29",
    "S30",
    "S31",
    "S32",
    "S33",
    "S34",
    "S35",
    "S36",
    "S37",
    "S38",
    "S39",
    "S40",
    "S41",
    "S42",
    "S43",
    "S44",
    "S45",
    "S46",
    "S47",
    "S48",
    "S49",
    "S50",
    "S51",
    "S52",
    "S53",
    "S54",
    "S55",
    "S56",
    "S57",
    "S58",
    "S59",
    "S60",
    "S61",
    "S62",
)
S_QUICK = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")
# Classes defined in 2D only (plan section 3.1).
_S_2D_ONLY = tuple(f"S{i}" for i in range(47, 63))
# S shapes the fused topologies and the feature probes rotate over. S47-S62 run in
# plain convolution only, so they stay out of these rotations; that also keeps the
# ids of every case generated before they were added.
S_CROSSED = S_IDS[: S_IDS.index("S46") + 1]

# Bases. 'rect' = rectangular input (S6); 'odd' = odd input (S14); 'big' = S16 input.
_S_LADDER = {
    1: [dict(ins=(16,), rect=None, odd=(15,), big=(32,))],
    2: [dict(ins=(16, 16), rect=(16, 8), odd=(15, 15), big=(32, 32))],
    3: [
        dict(ins=(16, 16, 16), rect=(16, 8, 4), odd=(15, 15, 15), big=(32, 32, 32)),
        dict(ins=(16, 8, 8), rect=(16, 8, 4), odd=(15, 15, 15), big=(32, 32, 32)),
        dict(ins=(16, 8, 4), rect=(16, 8, 4), odd=(15, 15, 15), big=(32, 32, 32)),
        dict(ins=(8, 8, 4), rect=(8, 4, 4), odd=(7, 7, 7), big=(16, 16, 16)),
    ],
    4: [dict(ins=(6, 6, 4, 4), rect=None, odd=None, big=None)],
}
# N 2 by default: batch 1 is a degenerate default (RFC 0022 section 7); H15 keeps N 1.
_S_BASE = dict(n=2, c=16, k=32, g=1, small=True)
_L_BASE = {
    1: dict(
        n=4,
        c=128,
        k=128,
        g=1,
        small=False,
        ins=(1024,),
        rect=None,
        odd=(1023,),
        big=(1024,),
    ),
    2: dict(
        n=4,
        c=128,
        k=128,
        g=1,
        small=False,
        ins=(56, 56),
        rect=(56, 28),
        odd=(55, 55),
        big=(56, 56),
    ),
    3: dict(
        n=2,
        c=32,
        k=32,
        g=1,
        small=False,
        ins=(32, 32, 32),
        rect=(32, 32, 16),
        odd=(31, 31, 31),
        big=(32, 32, 32),
    ),
}


def _feature(sid, r, b):
    """Case kwargs of shape ``sid`` on base ``b`` (None when the shape does not
    apply to rank ``r``). Features change one thing from the defaults."""
    n, c, k, ins, small = b["n"], b["c"], b["k"], b["ins"], b["small"]
    kw = dict(rank=r, n=n, c=c, k=k, g=1, ins=ins, f=3, s=1, d=1, pre=1, post=1)

    def per(v):  # a per-dimension pattern truncated to the rank
        return tuple(v[:r])

    if sid == "S1":
        kw.update(f=1, pre=0, post=0)
    elif sid == "S2":
        kw.update(pre=0, post=0)
    elif sid == "S3":
        pass
    elif sid == "S4":
        kw.update(s=2)
    elif sid == "S5":
        kw.update(d=2, pre=2, post=2)
    elif sid == "S6":
        if b["rect"] is None:
            return None
        kw.update(ins=b["rect"])
    elif sid == "S7":
        kw.update(g=2)
    elif sid == "S8":
        kw.update(g=4, s=2, d=2)
        if small:
            kw.update(n=2, c=32, k=64)
    elif sid == "S9":
        kw.update(n=8, f=1, pre=0, post=0)
    elif sid == "S10":
        kw.update(f=5, pre=2, post=2)
    elif sid == "S11":
        kw.update(c=3, f=7, s=2, pre=3, post=3)
        if small:
            kw.update(k=16)
    elif sid in ("S12a", "S12b"):
        if r == 1 or (r == 3 and sid == "S12b"):
            return None
        if r == 2 and sid == "S12a":
            kw.update(f=(1, 3), pre=(0, 1), post=(0, 1))
        elif r == 2:
            kw.update(f=(3, 1), pre=(1, 0), post=(1, 0))
        else:
            kw.update(f=(1, 3, 3), pre=(0, 1, 1), post=(0, 1, 1))
    elif sid in ("S13a", "S13b"):
        kw.update(g=c, k=c if sid == "S13a" else 2 * c)
    elif sid == "S14":
        kw.update(c=13 if small else 61, k=17 if small else 67, ins=b["odd"])
    elif sid == "S15":
        if r == 1:
            return None
        kw.update(
            s=per((1, 2, 1)), d=per((2, 1, 2)), pre=per((1, 0, 1)), post=per((1, 0, 1))
        )
    elif sid == "S16":
        kw.update(s=3, d=3, pre=3, post=3, ins=b["big"])
    elif sid == "S17":
        kw.update(c=1)
        if small:
            kw.update(k=16)
    elif sid == "S18":
        kw.update(f=2, s=2, pre=0, post=1)
    elif sid == "S19":
        kw.update(f=ins, pre=0, post=0)
    elif sid == "S20":
        i = tuple(v if v % 3 else v + 1 for v in ins)
        kw.update(ins=i, s=3, pre=0, post=0)
    elif sid == "S21":
        kw.update(f=2)
    elif sid == "S22":
        kw.update(pre=4, post=4)
    elif sid == "S23":
        return None  # handled by shape_S for rank 4
    # S24-S46: classes real models use (plan section 3.1, from the corpus of 3.3)
    elif sid == "S24":  # grouped, strided (ResNeXt, RegNet)
        kw.update(g=4, s=2)
    elif sid == "S25":  # depthwise, strided (MobileNet, EfficientNet)
        kw.update(g=c, k=c, s=2)
    elif sid == "S26":  # depthwise 5x5, strided
        if r != 2:
            return None
        kw.update(g=c, k=c, f=5, s=2, pre=2, post=2)
    elif sid == "S27":  # depthwise 7x7 (ConvNeXt)
        if r != 2:
            return None
        kw.update(g=c, k=c, f=7, pre=3, post=3)
    elif sid == "S28":  # depthwise multiplier 4
        kw.update(g=c, k=4 * c)
    elif (
        sid == "S29"
    ):  # patchify 2, stride 2, no pad (ConvNeXt downsample, UNet, nnU-Net)
        kw.update(f=2, s=2, pre=0, post=0)
    elif sid == "S30":  # patchify 4, stride 4 (ConvNeXt stem)
        if r != 2:
            return None
        kw.update(f=4, s=4, pre=0, post=0)
    elif sid == "S31":  # patchify 16, stride 16 (ViT patch embedding)
        if r != 2:
            return None
        kw.update(c=3, f=16, s=16, pre=0, post=0, ins=b["big"])
    elif sid == "S32":  # temporal 3x1x1 ((2+1)D video)
        if r != 3:
            return None
        kw.update(f=(3, 1, 1), pre=(1, 0, 0), post=(1, 0, 0))
    elif sid == "S33":  # spatial 1x3x3, stride (1, 2, 2)
        if r != 3:
            return None
        kw.update(f=(1, 3, 3), s=(1, 2, 2), pre=(0, 1, 1), post=(0, 1, 1))
    elif sid == "S34":  # pointwise downsample: 1x1, stride 2
        kw.update(f=1, s=2, pre=0, post=0)
    elif sid == "S35":  # vocoder: filter 7, dilation 3 (HiFi-GAN)
        if r != 1:
            return None
        kw.update(f=7, d=3, pre=9, post=9)
    elif sid == "S36":  # vocoder: filter 11, dilation 5
        if r != 1:
            return None
        kw.update(f=11, d=5, pre=25, post=25)
    elif sid == "S37":  # dilation 4 (DeepLab)
        kw.update(d=4, pre=4, post=4)
    elif sid == "S38":  # atrous dilation 12 (DeepLab ASPP)
        if r != 2:
            return None
        kw.update(d=12, pre=12, post=12)
    elif sid == "S39":  # 3x3, stride 2, no pad (Inception)
        kw.update(s=2, pre=0, post=0)
    elif sid == "S40":  # 5x5, no pad (Inception)
        if r != 2:
            return None
        kw.update(f=5, pre=0, post=0)
    elif sid == "S41":  # 1D upsampling-style filter 16, stride 8
        if r != 1:
            return None
        kw.update(f=16, s=8, pre=4, post=4)
    elif sid == "S42":  # 1D filter 4, stride 2
        if r != 1:
            return None
        kw.update(f=4, s=2, pre=1, post=1)
    elif sid == "S43":  # F7 non-square 3x5
        if r != 2:
            return None
        kw.update(f=(3, 5), pre=(1, 2), post=(1, 2))
    elif sid == "S44":  # F11 pre-only padding
        kw.update(f=2, s=2, pre=1, post=0)
    elif sid == "S45":  # F7 filter larger than the input, padded to fit (output 1)
        kw.update(f=tuple(i + 2 for i in ins), pre=1, post=1)
    elif sid == "S46":  # F8 stride 4 with filter 3
        kw.update(s=4)
    # S47-S62: 2D depthwise, grouped and non-square-image classes that S1-S46 and
    # the corpus miss (plan section 3.1). Each keeps the class's defining channel
    # count, stride, dilation or padding at L size and only grows the image and,
    # where C is not the feature, C and K.
    elif r != 2 and sid in _S_2D_ONLY:
        return None
    elif sid == "S47":  # depthwise, odd C below 16, 7x7
        kw.update(c=5, k=5, g=5, f=7, pre=3, post=3, ins=(9, 9) if small else ins)
    elif sid == "S48":  # depthwise, odd C, odd image, 3x3
        kw.update(c=7, k=7, g=7, ins=(13, 13) if small else b["odd"])
    elif sid == "S49":  # depthwise stride 2, odd C, odd image
        kw.update(c=5, k=5, g=5, s=2, ins=(17, 17) if small else b["odd"])
    elif sid == "S50":  # depthwise at C >= 64 on a tall non-square image (row loop)
        cc = 64 if small else c
        kw.update(c=cc, k=cc, g=cc, ins=(70, 40) if small else (112, 56))
    elif sid == "S51":  # depthwise stride 2, non-square image, even height
        cc = 96 if small else c
        kw.update(c=cc, k=cc, g=cc, s=2, ins=(16, 18) if small else (56, 60))
    elif sid == "S52":  # depthwise stride 2, odd image, C >= 64
        cc = 96 if small else c
        kw.update(c=cc, k=cc, g=cc, s=2, ins=(17, 17) if small else b["odd"])
    elif sid == "S53":  # depthwise, dilated
        cc = 32 if small else c
        kw.update(
            c=cc, k=cc, g=cc, d=2, pre=2, post=2, ins=(15, 15) if small else b["odd"]
        )
    elif sid == "S54":  # depthwise, unequal strides
        cc = 32 if small else c
        kw.update(c=cc, k=cc, g=cc, s=(1, 2))
    elif sid == "S55":  # depthwise, no padding
        cc = 32 if small else c
        kw.update(c=cc, k=cc, g=cc, pre=0, post=0)
    elif sid == "S56":  # depthwise, padding larger than "same"
        cc = 32 if small else c
        kw.update(c=cc, k=cc, g=cc, pre=2, post=2)
    elif sid == "S57":  # depthwise filter above 7x7 with C < 64
        kw.update(
            c=5, k=5, g=5, f=17, pre=8, post=8, ins=(56, 24) if small else (112, 48)
        )
    elif sid == "S58":  # grouped (not depthwise), dilated, stride 1
        kw.update(
            g=2, k=c, f=5, d=2, pre=4, post=4, ins=(15, 15) if small else b["odd"]
        )
    elif sid == "S59":  # non-square filter, unequal stride, non-square image
        kw.update(
            c=32 if small else c,
            k=64 if small else k,
            f=(3, 5),
            s=(1, 2),
            pre=(1, 2),
            post=(1, 2),
            ins=(11, 17) if small else (55, 57),
        )
    elif sid == "S60":  # dense, strided, non-square odd image
        kw.update(
            c=32 if small else c,
            k=64 if small else k,
            s=2,
            ins=(17, 19) if small else (55, 57),
        )
    elif sid == "S61":  # dense, dilated, non-square image
        cc = 32 if small else c
        kw.update(c=cc, k=cc, d=2, pre=2, post=2, ins=(15, 17) if small else (55, 57))
    elif sid == "S62":  # depthwise multiplier 2, odd C, batch 3
        kw.update(n=3, c=5, k=10, g=5, ins=(12, 12) if small else ins)
    else:
        raise KeyError(sid)
    return kw


def shape_S(sid, r, level):
    """First S-ladder entry of rank ``r`` on which ``sid`` fits ``level``."""
    if sid == "S23":
        if r != 4:
            return None
        kw = dict(
            rank=4, n=1, c=16, k=32, g=1, ins=(6, 6, 4, 4), f=3, s=1, d=1, pre=1, post=1
        )
        return kw if fits(new(**kw), level) else None
    if r == 4:
        return None
    for ladder in _S_LADDER[r]:
        kw = _feature(sid, r, {**_S_BASE, **ladder})
        if kw is None:
            return None
        if legal(new(**kw)) is None and fits(new(**kw), level):
            return kw
    return None


def shape_L(sid, r, level="comprehensive"):
    """L shape: the features of S on a medium base. The generator halves N, then
    C and K (keeping G and depthwise ratios), until the case fits the ceiling."""
    if sid == "S23" or r not in _L_BASE:
        return None
    kw = _feature(sid, r, _L_BASE[r])
    if kw is None:
        return None
    while True:
        c = new(**kw)
        if legal(c) is not None:
            return None
        if fits(c, level):
            return kw
        if kw["n"] > 1:
            kw["n"] = max(1, kw["n"] // 2)
        elif (
            kw["c"] >= 2 * kw["g"]
            and kw["k"] >= 2 * kw["g"]
            and kw["c"] % 2 == 0
            and kw["k"] % 2 == 0
        ):
            was_depthwise = kw["g"] == kw["c"]
            kw["c"] //= 2
            kw["k"] //= 2
            if was_depthwise:
                kw["g"] = kw["c"]
        else:
            return None


# M: model layers (source cited in the plan). rank, n, c, k, g, ins, f, s, pre/post
M_SHAPES = {
    "M1": dict(rank=2, n=8, c=3, k=64, g=1, ins=(224, 224), f=7, s=2, pre=3, post=3),
    "M2": dict(rank=2, n=8, c=64, k=64, g=1, ins=(56, 56), f=3, pre=1, post=1),
    "M3": dict(rank=2, n=8, c=256, k=64, g=1, ins=(56, 56), f=1),
    "M4": dict(rank=2, n=8, c=512, k=1024, g=1, ins=(28, 28), f=1, s=2),
    "M5": dict(rank=2, n=8, c=128, k=128, g=32, ins=(56, 56), f=3, pre=1, post=1),
    "M6": dict(rank=2, n=8, c=96, k=96, g=96, ins=(56, 56), f=3, pre=1, post=1),
    "M7a": dict(
        rank=2, n=8, c=192, k=192, g=1, ins=(17, 17), f=(1, 7), pre=(0, 3), post=(0, 3)
    ),
    "M7b": dict(
        rank=2, n=8, c=192, k=192, g=1, ins=(17, 17), f=(7, 1), pre=(3, 0), post=(3, 0)
    ),
    "M8": dict(rank=2, n=8, c=240, k=240, g=240, ins=(28, 28), f=5, pre=2, post=2),
    "M9": dict(rank=3, n=2, c=32, k=32, g=1, ins=(32, 32, 32), f=3, pre=1, post=1),
    "M10": dict(
        rank=3, n=2, c=3, k=64, g=1, ins=(16, 112, 112), f=7, s=2, pre=3, post=3
    ),
    "M11": dict(rank=1, n=8, c=256, k=256, g=256, ins=(128,), f=31, pre=15, post=15),
    "M12": dict(rank=1, n=4, c=1, k=512, g=1, ins=(16000,), f=10, s=5),
}
M_PR = ("M2", "M9", "M11")  # the only cases that may exceed the PR ceiling

# R: real model geometries from the dnn-benchmarking convolution corpus (plan
# section 3.3). The pick, its selection rule and its provenance (repository
# commit and archive md5) are data in conv_fwd_corpus.json; the cases keep the
# data type and layout the model used.
CORPUS_FILE = Path(__file__).with_name("conv_fwd_corpus.json")


def corpus():
    """The R-set document: source pin, selection rule and the picked cases."""
    with open(CORPUS_FILE, encoding="utf-8") as fh:
        return json.load(fh)


R_SHAPES = {e["id"]: e for e in corpus()["cases"]}


def shape_R(rid):
    """Case kwargs of R shape ``rid``: geometry, layout and data type."""
    e = R_SHAPES[rid]
    geo = {k: e[k] for k in ("rank", "n", "c", "k", "g")}
    geo.update({k: tuple(e[k]) for k in ("ins", "f", "s", "d", "pre", "post")})
    lay = e["layout"]
    return dict(geo, xl=lay, wl=lay, yl=lay, xt=e["dtype"])


def r_source(rid):
    """Provenance note of R shape ``rid`` (written into the case metadata)."""
    e = R_SHAPES[rid]
    return (
        f"dnn-benchmarking corpus {e['shape_id']}; models: {', '.join(e['models'])}"
        f"; family: {', '.join(e['family'])}"
    )


# H: hazard breakpoints. One change from the S defaults (N 2, C 16, K 32, 16 per
# dimension, filter 3, pad 1). Powers of 2 +- 1, dims not divisible by 4 or 8,
# and batch sizes that real models use.
H_SHAPES = {
    "H1": dict(c=15, k=32),
    "H2": dict(c=17, k=33),
    "H3": dict(c=31, k=31),
    "H4": dict(c=33, k=65),
    "H5": dict(c=63, k=64),
    "H6": dict(c=65, k=63),
    "H7": dict(c=16, k=7),
    "H8": dict(c=16, k=1),
    "H9": dict(n=3),
    "H10": dict(c=12, k=12, g=3),
    "H11": dict(c=15, k=20, g=5),
    "H12": dict(ins=31),
    "H13": dict(ins=33),
    "H14": dict(ins=17),
    "H15": dict(n=1),
    "H16": dict(n=32),
    "H17": dict(n=64),
    "H18": dict(c=4),
}


def shape_H(hid, r, level):
    h = H_SHAPES[hid]
    for ladder in _S_LADDER.get(r, []):
        base = {**_S_BASE, **ladder}
        kw = dict(
            rank=r,
            n=h.get("n", _S_BASE["n"]),
            c=h.get("c", 16),
            k=h.get("k", 32),
            g=h.get("g", 1),
            ins=_tup(h["ins"], r) if "ins" in h else base["ins"],
            f=3,
            s=1,
            d=1,
            pre=1,
            post=1,
        )
        if "ins" in h and r == 3:
            kw["ins"] = (h["ins"], 8, 4) if level == "quick" else _tup(h["ins"], r)
        c = new(**kw)
        if legal(c) is None and fits(c, level):
            return kw
    return None


# X: stress (full only)
X_SHAPES = {
    "X1": dict(rank=2, n=256, c=64, k=64, g=1, ins=(56, 56), f=3, pre=1, post=1),
    "X2": dict(rank=3, n=2, c=64, k=64, g=1, ins=(64, 64, 64), f=3, pre=1, post=1),
    "X3": dict(
        rank=2, n=1, c=1, k=1, g=1, ins=(46341, 46341), f=1
    ),  # y has > 2^31 elements
    "X4": dict(
        rank=2, n=1, c=2048, k=64, g=1, ins=(14, 14), f=3, pre=1, post=1
    ),  # accumulation 18,432
}

# F23: long accumulation (nightly)
ACCUM_SHAPES = {
    "A1": dict(
        rank=2, n=1, c=512, k=64, g=1, ins=(14, 14), f=3, pre=1, post=1
    ),  # 4,608
    "A2": dict(
        rank=3, n=1, c=128, k=32, g=1, ins=(8, 8, 8), f=3, pre=1, post=1
    ),  # 3,456
}


# --------------------------------------------------------------------------
# Trimmed products (plan section 4.3, trade-off register T8, T9, T11, T12; T10 was
# lifted). The level rules generate the complete products; the generator then
# removes these cases, so the cases that remain keep their ids, seeds and claims.
# T8, T9 and T11 remove products that no engine executed on the measured
# architectures (gfx942 and gfx90a, see plan section 4.5); activations that an
# engine executes are never trimmed. T12 keeps
# the geometry classes S24-S62 to plain convolution in nightly and full: an
# epilogue or a tensor feature does not depend on the geometry class (T1, T4).
# Each trim names the trigger that brings its product back.
# --------------------------------------------------------------------------

# Activations at least one engine executed in the act topology (plan section 4.5).
EXECUTED_ACTS = ("relu", "relu6", "relu_clamp", "sigmoid", "tanh")
UNEXECUTED_TOPOLOGIES = ("scale_bias_act", "add_relu", "bias_add_relu", "act_add")
# Geometry classes added from the corpus analysis and the engine review (plan
# section 3.1), and their L forms.
GEOMETRY_CLASS_SHAPES = tuple(f"{p}{i}" for p in "SL" for i in range(24, 63))

TRIMS = {
    "T8": "full: act x every S and L shape for activations no engine executes",
    "T9": "comprehensive and full: plain repeats of the four fused topologies no engine executes",
    "T11": "full: F13, F19 and F35 paired with every S shape",
    "T12": "comprehensive and full: geometry classes S24-S62 crossed with fused topologies or features",
}


def trim(c, level):
    """The trim (key of TRIMS) that removes case ``c`` from ``level``, or None.
    T10 (fp16 and bf16 bias_act in nightly and full) was lifted once the generator
    typed the bias node by the bias tensor; its key is not reused."""
    tagged = any(i.startswith("F") for i in c["ids"])
    topo = c["topo"]
    if level == "full" and topo == "act" and c["act"] not in EXECUTED_ACTS:
        return "T8"
    late = level in ("comprehensive", "full")
    if late and topo in UNEXECUTED_TOPOLOGIES and not tagged:
        return "T9"
    if (
        late
        and (topo != "conv" or tagged)
        and any(i in GEOMETRY_CLASS_SHAPES for i in c["ids"])
    ):
        return "T12"
    if (
        level == "full"
        and topo == "conv"
        and any(i in ("F13", "F19", "F35") for i in c["ids"])
    ):
        return "T11"
    return None


# --------------------------------------------------------------------------
# Value domains (F21). The fill recipes the harness has are 'free' (uniform in
# [lo, hi]) and 'fixed' (constant). NaN, Inf, subnormal injection and sparsity
# need harness work (RFC 0022 I4) and are not generated.
# --------------------------------------------------------------------------


def value_recipes(c):
    """Fill recipe of x and w for the case's value class."""
    cls, t = c["vals"], c["xt"]
    if t not in DTYPE_MAX:
        return None
    if cls == "uniform":
        return dict(x=("free", -1.0, 1.0), w=("free", -1.0, 1.0))
    if cls == "zero":
        return dict(x=("fixed", 0.0), w=("free", -1.0, 1.0))
    if cls == "const":
        return dict(x=("fixed", 1.0), w=("fixed", 1.0))
    if cls == "large":
        # Non-negative, so the sum cannot cancel: outputs sit near the dtype maximum
        # and a rounding error stays small relative to them.
        a = float(f"{math.sqrt(0.5 * DTYPE_MAX[t] / accum_len(c)):.3g}")
        return dict(x=("free", 0.0, a), w=("free", 0.0, a))
    if cls == "tiny":
        a = 3e-3 if t == "half" else 1e-20
        return dict(x=("free", -a, a), w=("free", -a, a))
    raise KeyError(cls)
