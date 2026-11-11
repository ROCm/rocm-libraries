#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declarations of the convolution backward-data (dgrad) test plan.

This module is the machine-readable half of
``docs/test-plans/convolution-bwd.md``. It holds what the bundle generator
(``generate_conv_bwd_bundles.py``) and the exhaustive tool read:

* the case model of a dgrad graph (dy, w -> dx) and its fused topologies;
* the legality rules, which copy ``ConvolutionDgradNode::pre_validate_node``;
* the dgrad shape catalog (S, L, M, R, H, A, X) and the per-level ceilings;
* the data types, backward activations, value domains and the purpose rule;
* the products that are generated and then trimmed (``TRIMS``), each with the
  trigger that brings it back.

The catalog is built around what dgrad computes, not around the forward
convolution: a dx element sums over the K/G channels of dy and over the filter
taps that land on it, so the reduction is over K, the output channels are C,
and stride and dilation decide which taps reach which dx element. A case is a
plain ``dict``; ``finalize`` resolves every default so that two spellings of one
graph compare equal. There is no engine matrix; the engine evidence is
``TRIMS``.
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

RANKS = (1, 2, 3)  # core spatial ranks; 4 is a probe
LAYOUTS = ("cf", "cl")  # channels-first, channels-last
CORE_DTYPES = ("float", "half", "bfloat16")
PAIRS = tuple((lay, dt) for dt in CORE_DTYPES for lay in LAYOUTS)

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

# F19: io types beyond the core three. (dy, w, dx, node compute type)
EXTRA_TYPES = {
    "fp8_e4m3": ("fp8_e4m3", "fp8_e4m3", "float", "float"),
    "fp8_e5m2": ("fp8_e5m2", "fp8_e5m2", "float", "float"),
    "fp8_e4m3_fnuz": ("fp8_e4m3_fnuz", "fp8_e4m3_fnuz", "float", "float"),
    "fp8_e5m2_fnuz": ("fp8_e5m2_fnuz", "fp8_e5m2_fnuz", "float", "float"),
    "int8": ("int8", "int8", "int32", "int32"),
    "double": ("double", "double", "double", "double"),
}
# F20: mixed precision. (dy, w, dx, node compute type)
MIXED_TYPES = {
    "half_in_float_out": ("half", "half", "float", "float"),
    "bf16_in_float_out": ("bfloat16", "bfloat16", "float", "float"),
    "dy_half_w_float": ("half", "float", "float", "float"),
    "dy_float_w_half": ("float", "half", "float", "float"),
    "half_compute_half": ("half", "half", "half", "half"),
}

# F26: backward activations. name -> (operation, relu_lower_clip,
# relu_lower_clip_slope, relu_upper_clip, elu_alpha, swish_beta, softplus_beta).
# Each takes the incoming gradient as in_0 and the forward activation input as in_1.
ACTS = {
    "relu": ("relu_bwd", None, None, None, None, None, None),
    "relu6": ("relu_bwd", None, None, 6.0, None, None, None),
    "relu_leaky": ("relu_bwd", None, 0.1, None, None, None, None),
    "sigmoid": ("sigmoid_bwd", None, None, None, None, None, None),
    "tanh": ("tanh_bwd", None, None, None, None, None, None),
    "elu": ("elu_bwd", None, None, None, 1.0, None, None),
    "gelu": ("gelu_bwd", None, None, None, None, None, None),
    "gelu_tanh": ("gelu_approx_tanh_bwd", None, None, None, None, None, None),
    "swish": ("swish_bwd", None, None, None, None, 1.0, None),
    "softplus": ("softplus_bwd", None, None, None, None, None, 1.0),
}
RELU_FAMILY = ("relu", "relu6", "relu_leaky")

# Topologies. Each is one bundle directory.
TOPOLOGIES = (
    "dgrad",  # ConvolutionBwd/Default
    "dact",  # dgrad -> act_bwd(., x)            (previous layer's activation)
    "pre_dact",  # act_bwd(dy, y) -> dgrad       (this layer's activation)
    "add",  # dgrad -> add(dx_residual)          (gradient of a skip connection)
    "add_drelu",  # dgrad -> add(dx_residual) -> relu_bwd(., x)
)
FUSED = TOPOLOGIES[1:]
BUNDLE_DIR = {
    "dgrad": ("ConvolutionBwd", "Default"),
    "dact": ("ConvolutionBwdPointwise", "Default"),
    "pre_dact": ("ConvolutionBwdPointwise", "Variant2"),
    "add": ("ConvolutionBwdPointwise", "Variant3"),
    "add_drelu": ("ConvolutionBwdPointwise", "Variant4"),
}
HAS_ACT_PLACEHOLDERS = ("dact", "pre_dact")  # add_drelu fixes relu_bwd

# F21: value domains the fill recipes can express.
VALUE_CLASSES = ("uniform", "zero", "const", "large", "tiny")

# Per-level ceilings for each case: (FLOP, bytes of fp32 dy + w + dx). None = unbounded.
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
    topo="dgrad",
    dyl="cf",
    wl="cf",
    dxl="cf",  # layout of dy, w, dx
    dye=(),
    we=(),
    dxe=(),  # per-dim extra stride padding (non-packed); () = packed
    dyt="float",
    wt=None,
    dxt=None,
    comp=None,  # data types; None resolves in finalize
    conv_mode="CROSS_CORRELATION",
    act="relu",
    inter_virtual=True,  # F30
    inter_t=None,  # F31: virtual intermediate type; None resolves
    override=False,  # F35: is_override_shape_enabled
    vals="uniform",
    ids=(),
    size="S",
    alignment=None,  # F17: not placeable yet (I3)
)


def _tup(v, r):
    return tuple(v) if isinstance(v, (tuple, list)) else (v,) * r


def new(rank, n, c, k, g, ins, f, s=1, d=1, pre=0, post=0, **kw):
    """Build a finalized case from per-dimension values (scalars broadcast).
    ``ins`` is the dx spatial size; dy's follows from the output formula."""
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
    c["wt"] = c["wt"] or c["dyt"]
    if c["dxt"] is None:
        c["dxt"] = c["dyt"]
    if c["comp"] is None:
        c["comp"] = {"int8": "int32", "double": "double"}.get(c["dyt"], "float")
    if c["inter_t"] is None:
        c["inter_t"] = c["dxt"] if c["topo"] == "dgrad" else "float"
    for key in ("dye", "we", "dxe", "ids"):
        c[key] = tuple(c[key])
    return c


def out_dim(i, f, s, d, pre, post):
    return (i + pre + post - (d * (f - 1) + 1)) // s + 1


def outs(c):
    """dy spatial dims."""
    return tuple(
        out_dim(c["ins"][i], c["f"][i], c["s"][i], c["d"][i], c["pre"][i], c["post"][i])
        for i in range(c["rank"])
    )


def legal(c):
    """Legality, copied from ConvolutionDgradNode::pre_validate_node: equal
    ranks, dy[1] == w[0], dx[0] == dy[0], dx[1] % w[1] == 0 and w[0] % G == 0,
    one parameter per spatial dim, stride and dilation >= 1, padding >= 0, and
    dx + pre + post >= the dilated filter. dy's dims are computed from the output
    formula, so the formula check holds by construction. dx dims are always set:
    the frontend never infers them."""
    r = c["rank"]
    if c["n"] < 1 or c["c"] < 1 or c["k"] < 1 or c["g"] < 1:
        return "empty dimension"
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
            return "filter larger than padded dx"
    return None


def dx_dims(c):
    return [c["n"], c["c"], *c["ins"]]


def w_dims(c):
    return [c["k"], c["c"] // c["g"], *c["f"]]


def dy_dims(c):
    return [c["n"], c["k"], *outs(c)]


def strides(dims, layout, extra=()):
    """Strides for ``dims`` laid out channels-first ('cf') or channels-last
    ('cl'). ``extra`` pads each dimension's extent when computing the stride of
    the next-slower dimension (non-packed tensors)."""
    r = len(dims) - 2
    order = (
        list(range(r + 1, 1, -1)) + [1, 0]
        if layout == "cf"
        else [1] + list(range(r + 1, 1, -1)) + [0]
    )
    st = [0] * len(dims)
    acc = 1
    for dim in order:
        st[dim] = acc
        acc *= dims[dim] + (extra[dim] if extra else 0)
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
    return 4 * (numel(dx_dims(c)) + numel(w_dims(c)) + numel(dy_dims(c)))


def taps_per_dx(c):
    """Most filter taps that reach one dx element: along a dimension with
    stride s and dilation d a dx position meets one tap in every s / gcd(s, d)."""
    return math.prod(
        -(-f // (s // math.gcd(s, d))) for f, s, d in zip(c["f"], c["s"], c["d"])
    )


def accum_len(c):
    """Products summed into one dx element: K/G times the taps that reach it."""
    return (c["k"] // c["g"]) * taps_per_dx(c)


def fits(c, level):
    fl, by = CEILINGS[level]
    return (fl is None or flops(c) <= fl) and (by is None or fp32_bytes(c) <= by)


def canon(c):
    """Stable key of the graph: everything that changes the graph or its data."""
    skip = {"ids", "size", "src"}
    return json.dumps(
        {k: v for k, v in sorted(c.items()) if k not in skip}, sort_keys=True
    )


def short_hash(c):
    return hashlib.sha1(canon(c).encode()).hexdigest()[:6]


# --------------------------------------------------------------------------
# Dgrad behaviour classes (plan section 2.1). The corpus analysis and the
# coverage check classify graphs with these; the catalog below is built to hit
# every class.
# --------------------------------------------------------------------------


def stride_class(c):
    """How the filter splits into stride phases, per dimension, merged:
    s1; f%s=0 (every phase gets f/s taps); f%s!=0 (unequal phases);
    f<s (some phases empty: dx elements no tap reaches); gcd>1 (stride and
    dilation share a factor: some phases empty)."""
    out = set()
    for f, s, d in zip(c["f"], c["s"], c["d"]):
        if s == 1:
            out.add("s1")
        elif math.gcd(s, d) > 1:
            out.add("gcd>1")
        elif f < s:
            out.add("f<s")
        elif f % s == 0:
            out.add("f%s=0")
        else:
            out.add("f%s!=0")
    return "/".join(sorted(out))


def tpad_class(c):
    """Padding seen by dgrad as a transposed convolution: d*(f-1) - pre.
    0: full (dx border sums fewer taps); > 0: the usual case; < 0: padding larger
    than the dilated filter, so dy's border rows reach no dx element."""
    if c["pre"] != c["post"]:
        return "asym"
    t = [d * (f - 1) - p for f, d, p in zip(c["f"], c["d"], c["pre"])]
    if any(x < 0 for x in t):
        return "tp<0"
    if all(x == 0 for x in t):
        return "tp0"
    return "tp>0"


def remainder(c):
    """True when (dx + pads - dilated filter) % stride != 0: dx's last rows lie
    past the last window and no tap reaches them."""
    return any(
        (i + a + b - (d * (f - 1) + 1)) % s
        for i, a, b, d, f, s in zip(
            c["ins"], c["pre"], c["post"], c["d"], c["f"], c["s"]
        )
    )


def channel_class(v):
    if v in (1, 2, 3, 4):
        return str(v)
    if v % 8:
        return "odd<16" if v < 16 else ("non8<=64" if v <= 64 else "non8>64")
    return "8-32" if v <= 32 else ("40-256" if v <= 256 else ">256")


def group_class(c):
    if c["g"] == 1:
        return "dense"
    return "dw" if c["g"] == c["c"] else "grouped"


def signature(c):
    """Behaviour signature of a dgrad graph (plan section 3.3): rank, group
    class, C/G and K/G classes, stride class, transposed padding, remainder.
    A 1D graph carried as 2D with H = 1 counts as 1D."""
    rank, f, s, d = c["rank"], c["f"], c["s"], c["d"]
    pre, post, ins = c["pre"], c["post"], c["ins"]
    if rank == 2 and ins[0] == 1 and f[0] == 1 and outs(c)[0] == 1:
        c = dict(
            c,
            rank=1,
            f=f[1:],
            s=s[1:],
            d=d[1:],
            pre=pre[1:],
            post=post[1:],
            ins=ins[1:],
        )
    return (
        c["rank"],
        group_class(c),
        channel_class(c["c"] // c["g"]),
        channel_class(c["k"] // c["g"]),
        stride_class(c),
        tpad_class(c),
        remainder(c),
    )


# --------------------------------------------------------------------------
# Purpose (RFC 0021 section 9)
# --------------------------------------------------------------------------


def probe_reasons(c):
    why = []
    if c["conv_mode"] != "CROSS_CORRELATION":
        why.append("F13 conv_mode")
    if c["pre"] != c["post"]:
        why.append("F11 asymmetric padding")
    if c["rank"] not in RANKS:
        why.append("F1 4D spatial")
    if c["dye"] or c["we"] or c["dxe"]:
        why.append("F16 non-packed strides")
    if not (c["dyl"] == c["wl"] == c["dxl"]):
        why.append("F15 mixed layouts")
    if c["alignment"]:
        why.append("F17 alignment")
    if c["dyt"] not in CORE_DTYPES or c["wt"] != c["dyt"] or c["dxt"] != c["dyt"]:
        why.append("F19/F20 data types")
    if c["comp"] != "float" and c["dyt"] in CORE_DTYPES:
        why.append("F20 compute type")
    if c["override"]:
        why.append("F35 override shape")
    if c["vals"] != "uniform":
        why.append("F21 value domain")
    if c["size"] == "H":
        why.append("H hazard")
    if "X3" in c["ids"]:
        why.append("F18 64-bit indexing")
    topo = c["topo"]
    if topo in ("pre_dact", "add", "add_drelu"):
        why.append("F28/F29/F36 topology")
    if topo in HAS_ACT_PLACEHOLDERS and c["act"] not in RELU_FAMILY:
        why.append("F26 non-ReLU activation")
    if topo != "dgrad" and not c["inter_virtual"]:
        why.append("F30 materialised intermediate")
    if topo != "dgrad" and c["inter_t"] != "float":
        why.append("F31 intermediate type")
    return why


def purpose(c):
    return "probe" if probe_reasons(c) else "claim"


# --------------------------------------------------------------------------
# Shape catalog (plan section 3). S shapes change one dgrad feature from the
# defaults; L shapes are the same classes at medium size. Each entry is
# (ranks, small kwargs, medium kwargs); kwargs override the base of the rank.
# A medium value of None means "same as small". Groups:
#   S1-S16   basics and transposed padding
#   S17-S31  stride phases: how stride and dilation split the filter
#   S32-S42  channel classes: output C/G and reduction K/G
#   S43-S52  grouped convolution
#   S53-S64  depthwise
#   S65-S72  image size and tiling (dy and dx extents)
#   S73-S76  rank-specific classes (1D audio, 4D)
#   S77-S80  output heads and latent convs: few reduction channels
# --------------------------------------------------------------------------

# Bases. N 2: no default dimension of 1 (RFC 0022 section 7). C 16, K 32: the
# reduction (K) differs from the output channels (C).
_S_BASE = {
    1: dict(n=2, c=16, k=32, g=1, ins=(16,)),
    2: dict(n=2, c=16, k=32, g=1, ins=(16, 16)),
    3: dict(n=2, c=16, k=32, g=1, ins=(16, 16, 16)),
    4: dict(n=1, c=16, k=32, g=1, ins=(6, 6, 4, 4)),
}
# 3D sizes tried in order until a case fits the level ceiling.
_S_3D_LADDER = ((16, 16, 16), (16, 8, 8), (16, 8, 4), (8, 8, 4))
_L_BASE = {
    1: dict(n=4, c=64, k=128, g=1, ins=(1024,)),
    2: dict(n=4, c=64, k=128, g=1, ins=(56, 56)),
    3: dict(n=2, c=32, k=32, g=1, ins=(32, 32, 32)),
}

R123, R12, R2, R23, R1, R3 = (1, 2, 3), (1, 2), (2,), (2, 3), (1,), (3,)


def _dw(c, mult=1, **kw):
    return dict(c=c, k=mult * c, g=c, **kw)


def _grp(g, cg, kg, **kw):
    return dict(c=g * cg, k=g * kg, g=g, **kw)


# id -> (ranks, small, medium). Spatial values given per rank are truncated or
# broadcast by the builder; "ins" given as a dict maps rank -> size.
S_CLASSES = {
    # --- basics and transposed padding --------------------------------------
    "S1": (R123, dict(f=1, pre=0), None),  # 1x1: dgrad is a GEMM over K
    "S2": (R123, dict(pre=0), None),  # full transposed padding
    "S3": (R123, dict(), None),  # 3x3 "same"
    "S4": (R123, dict(f=5, pre=2), None),
    "S5": (R12, dict(f=7, pre=3), None),
    "S6": (R123, dict(d=2, pre=2), None),  # dilation, stride 1
    "S7": (
        R23,
        dict(ins={2: (16, 8), 3: (16, 8, 4)}),
        dict(ins={2: (56, 28), 3: (32, 32, 16)}),
    ),
    "S8": (R123, dict(n=8, f=1, pre=0), None),
    "S9": (R123, dict(pre=4), None),  # pad > f-1: transposed padding < 0
    "S10": (R123, dict(f=2, s=2, pre=0, post=1), None),  # asymmetric (probe)
    "S11": (R123, dict(f=2, s=2, pre=1, post=0), None),  # pre-only (probe)
    "S12": (R123, dict(f=2, pre=1), None),  # even filter: dy larger than dx
    "S13": (R123, dict(f="ins", pre=0), None),  # filter = dx: dy spatial 1
    "S14": (R123, dict(f="ins+2", pre=1), None),  # filter larger than dx
    "S15": (R23, dict(s=(1, 2, 1), d=(2, 1, 2), pre=(1, 0, 1)), None),
    "S16a": (R23, dict(f=(1, 3, 3), pre=(0, 1, 1)), None),  # 2D 1x3, 3D 1x3x3
    "S16b": (R23, dict(f=(3, 1, 1), pre=(1, 0, 0)), None),  # 2D 3x1, 3D 3x1x1
    "S16c": (R2, dict(f=(3, 5), pre=(1, 2)), None),
    # --- stride phases -------------------------------------------------------
    "S17": (R123, dict(s=2), None),  # f3 s2 p1, even dx: unequal phases + remainder
    "S18": (R123, dict(s=2, ins="odd"), None),  # f3 s2 p1, odd dx: no remainder
    "S19": (R123, dict(f=2, s=2, pre=0), None),  # patchify 2: one tap per dx
    "S20": (R12, dict(f=4, s=2, pre=1), None),  # f%s=0, two taps per dim
    "S21": (R2, dict(f=4, s=4, pre=0), None),  # patchify 4
    "S22": (R2, dict(c=3, f=16, s=16, pre=0, ins=(32, 32)), dict(c=3, ins=(224, 224))),
    "S23": (
        R123,
        dict(f=1, s=2, pre=0),
        None,
    ),  # f<s: 3 of 4 dx elements get no tap (2D)
    "S24": (R123, dict(s=4), None),  # f3 s4: f<s
    "S25": (R123, dict(s=3, pre=0), None),  # f3 s3 on 16: dx's last row meets no window
    "S26": (R123, dict(s=2, d=2, pre=2), None),  # gcd(s, d) = 2: half the phases empty
    "S27": (R12, dict(s=2, d=3, pre=3), None),  # coprime stride and dilation
    "S28": (R123, dict(c=3, f=7, s=2, pre=3), None),  # strided stem, C 3
    "S29": (R123, dict(s=2, pre=0), None),  # f3 s2 no pad
    "S30": (R2, dict(f=7, s=2, pre=3), None),  # f%s!=0 with a large filter
    "S31": (R3, dict(f=(1, 3, 3), s=(1, 2, 2), pre=(0, 1, 1)), None),
    # --- channel classes: C is dgrad's output, K its reduction ---------------
    "S32": (R123, dict(c=1), None),  # one dx channel
    "S33": (R23, dict(c=3, k=64), dict(c=3, k=64)),  # RGB input gradient
    "S34": (
        R2,
        dict(c=4, k=320),
        dict(c=4, k=320),
    ),  # small C, large K (latent conv_in)
    "S35": (R12, dict(k=1), None),  # one dy channel: reduction over taps only
    "S36": (R23, dict(c=128, k=3), dict(c=128, k=3)),  # wide dx, K 3 (image conv_out)
    "S37": (
        R2,
        dict(c=64, k=2, f=1, pre=0),
        dict(c=256, k=2, f=1, pre=0),
    ),  # K 2 (binary head)
    "S38": (R123, dict(c=13, k=17, ins="odd"), dict(c=61, k=67, ins="odd")),
    "S39": (R2, dict(c=24, k=24), dict(c=24, k=24)),  # C = K = 24, dense
    "S40": (R2, dict(c=4, k=4), dict(c=4, k=4)),  # both below the vector width
    "S41": (R2, dict(c=16, k=64, f=1, pre=0), dict(c=64, k=256, f=1, pre=0)),  # K = 4C
    "S42": (R2, dict(c=64, k=16, f=1, pre=0), dict(c=256, k=64, f=1, pre=0)),  # C = 4K
    # --- grouped (not depthwise) ---------------------------------------------
    "S43": (R2, _grp(4, 4, 4), _grp(32, 4, 4)),  # C/G = K/G = 4 (ResNeXt 32x4d)
    "S44": (R2, _grp(8, 8, 8), _grp(32, 8, 8)),  # C/G = K/G = 8 (ResNeXt 32x8d)
    "S45": (R2, _grp(4, 16, 16), _grp(8, 16, 16)),  # C/G = K/G = 16
    "S46": (R2, _grp(2, 24, 24), _grp(4, 24, 24)),  # C/G = K/G = 24
    "S47": (R2, _grp(2, 32, 32), _grp(4, 32, 32)),  # C/G = K/G = 32
    "S48": (R2, _grp(2, 4, 320), _grp(2, 4, 320)),  # grouped small C/G, large K/G
    "S49": (R123, _grp(2, 8, 16), _grp(4, 16, 32)),  # K/G = 2 C/G
    "S50": (R123, _grp(4, 8, 8, s=2), _grp(32, 4, 4, s=2)),  # grouped, strided
    "S51": (
        R2,
        _grp(2, 8, 8, f=5, d=2, pre=4, ins=(15, 15)),
        _grp(4, 16, 16, f=5, d=2, pre=4, ins=(55, 55)),
    ),
    "S52": (R2, _grp(4, 8, 1), _grp(16, 8, 1)),  # K/G = 1: one dy channel per group
    # --- depthwise (G = C) ---------------------------------------------------
    "S53": (R123, _dw(16), _dw(64)),
    "S54": (R123, _dw(16, 2), _dw(64, 2)),  # multiplier 2
    "S55": (R12, _dw(16, 4), _dw(32, 4)),  # multiplier 4
    "S56": (R123, _dw(16, s=2), _dw(64, s=2)),
    "S57": (R2, _dw(16, f=5, s=2, pre=2), _dw(96, f=5, s=2, pre=2)),
    "S58": (R2, _dw(16, f=7, pre=3), _dw(64, f=7, pre=3)),  # ConvNeXt
    "S59": (
        R2,
        _dw(64, f=7, pre=0),
        _dw(64, f=7, pre=0, ins=(28, 28)),
    ),  # windowed, pad 0
    "S60": (R2, _dw(5, f=7, pre=3, ins=(9, 9)), _dw(5, f=7, pre=3)),  # odd C below 16
    "S61": (R2, _dw(64, ins=(70, 40)), _dw(64, ins=(112, 56))),  # tall image
    "S62": (R2, _dw(32, d=2, pre=2, ins=(15, 15)), _dw(64, d=2, pre=2, ins=(55, 55))),
    "S63": (R2, _dw(32, s=(1, 2)), _dw(64, s=(1, 2))),
    "S64": (R1, _dw(64, f=31, pre=15, ins=(128,)), _dw(256, f=31, pre=15, ins=(512,))),
    # --- image size and tiling ----------------------------------------------
    "S65": (
        R2,
        dict(c=32, k=32, ins=(8, 8)),
        dict(c=128, k=128, ins=(8, 8)),
    ),  # dy width 8
    "S66": (
        R2,
        dict(c=32, k=32, ins=(4, 4)),
        dict(c=256, k=256, ins=(4, 4)),
    ),  # dy width 4
    "S67": (
        R2,
        dict(f=5, pre=2, ins=(16, 48)),
        dict(c=64, k=64, f=5, pre=2, ins=(48, 48)),
    ),
    "S68": (
        R2,
        dict(n=1, c=32, k=32, ins=(32, 109)),
        dict(n=2, c=64, k=64, ins=(64, 109)),
    ),
    "S69": (R2, dict(ins=(17, 19)), dict(ins=(55, 57))),  # odd non-square dx
    "S70": (
        R2,
        dict(f=1, pre=0, ins=(1, 1), n=8),
        dict(n=32, c=256, k=512, f=1, pre=0, ins=(1, 1)),
    ),
    "S71": (
        R2,
        dict(ins=(1, 256), f=(1, 3), pre=(0, 1)),
        dict(ins=(1, 4096), f=(1, 3), pre=(0, 1)),
    ),
    "S72": (
        R2,
        dict(s=2, ins=(17, 19)),
        dict(s=2, ins=(55, 57)),
    ),  # strided, odd non-square
    # --- rank-specific -------------------------------------------------------
    "S73": (R1, dict(f=7, d=3, pre=9), None),  # vocoder dilation
    "S74": (R1, dict(f=11, d=5, pre=25, ins=(64,)), dict(f=11, d=5, pre=25)),
    "S75": (
        R1,
        dict(c=1, k=32, f=10, s=5, pre=0, ins=(160,)),
        dict(c=1, k=512, f=10, s=5, pre=0, ins=(16000,)),
    ),
    "S76": ((4,), dict(), None),  # 4D spatial (probe)
    # --- output heads and latent convs: few reduction channels, 1x1 ---------
    "S77": (
        R23,
        dict(c=128, k=3, f=1, pre=0),
        dict(c=256, k=3, f=1, pre=0),
    ),  # RGB head
    "S78": (
        R2,
        dict(c=128, k=21, f=1, pre=0),
        dict(c=256, k=21, f=1, pre=0),
    ),  # class head
    "S79": (
        R2,
        dict(c=128, k=4, f=3, pre=1),
        dict(c=128, k=4, f=3, pre=1),
    ),  # latent conv_out
    "S80": (R2, dict(c=4, k=4, f=1, pre=0), dict(c=4, k=4, f=1, pre=0)),  # quant conv
}
S_IDS = tuple(S_CLASSES)
# Quick: one class from each group that every engine family meets.
S_QUICK = ("S1", "S3", "S6", "S8", "S17", "S19", "S23", "S44", "S53")
# Classes the fused topologies and feature probes rotate over: basics and
# stride phases. Channel, grouped, depthwise, tiling and rank-specific classes
# (S32-S76) select the dgrad kernel itself, not an epilogue or a per-tensor
# feature, so they run in plain dgrad only (plan section 4.3, D3).
S_CROSSED = tuple(f"S{i}" for i in range(1, 32) if f"S{i}" in S_CLASSES) + (
    "S16a",
    "S16b",
    "S16c",
)


def _per_dim(v, r):
    """A scalar broadcasts; a pattern longer than the rank is truncated to it
    (a 3D pattern (1, 3, 3) is (1, 3) in 2D); a shorter one has no form."""
    if not isinstance(v, tuple):
        return (v,) * r
    return v[:r] if len(v) >= r else None


def _resolve(sid, r, small):
    """Case kwargs of class ``sid`` at rank ``r`` on the small or medium base,
    or None when the class has no form of that rank. The medium form takes the
    class's medium values when it declares them; otherwise it keeps the small
    form's channel, group and batch values and takes the medium image."""
    ranks, s_kw, l_kw = S_CLASSES[sid]
    if r not in ranks:
        return None
    base = (_S_BASE if small else _L_BASE)[r]
    if small or l_kw is not None:
        over = dict(s_kw if small else l_kw)
    else:
        over = {k: v for k, v in s_kw.items() if k != "ins" or isinstance(v, str)}
    kw = dict(base, f=3, s=1, d=1, pre=1)
    kw.update(over)
    ins = kw["ins"]
    if isinstance(ins, dict):
        ins = ins[r]
    elif ins == "odd":
        ins = tuple(v - 1 for v in base["ins"])
    kw["ins"] = _per_dim(ins, r)
    if kw["f"] == "ins":
        kw["f"] = kw["ins"]
    elif kw["f"] == "ins+2":
        kw["f"] = tuple(i + 2 for i in kw["ins"])
    for key in ("ins", "f", "s", "d", "pre"):
        kw[key] = _per_dim(kw[key], r)
        if kw[key] is None:
            return None
    kw["post"] = _per_dim(kw.get("post", kw["pre"]), r)
    kw["rank"] = r
    return kw


def shape_S(sid, r, level):
    """Small form of class ``sid`` at rank ``r`` that fits ``level`` (3D tries
    smaller images until it fits), or None."""
    kw = _resolve(sid, r, True)
    if kw is None:
        return None
    candidates = [kw]
    if r == 3 and kw["ins"] == _S_BASE[3]["ins"]:
        candidates = [dict(kw, ins=i) for i in _S_3D_LADDER]
    spec_f = S_CLASSES[sid][1].get("f")
    for cand in candidates:
        if spec_f in ("ins", "ins+2"):  # S13, S14: the filter follows the image
            extra = 2 if spec_f == "ins+2" else 0
            cand = dict(cand, f=tuple(i + extra for i in cand["ins"]))
        c = new(**cand)
        if legal(c) is None and fits(c, level):
            return cand
    return None


def shape_L(sid, r, level="comprehensive"):
    """Medium form of class ``sid``. Until it fits: halve N; then halve C and K
    of a dense case whose channels are not the class's feature; halve the group
    count of a grouped case (per-group channels kept); halve C of a depthwise
    case (multiplier kept)."""
    if r not in _L_BASE:
        return None
    kw = _resolve(sid, r, False)
    if kw is None:
        return None
    _, s_kw, l_kw = S_CLASSES[sid]
    channels_fixed = bool({"c", "k"} & set(l_kw or s_kw))
    while True:
        c = new(**kw)
        if legal(c) is not None:
            return None
        if fits(c, level):
            return kw
        if kw["n"] > 1:
            kw["n"] //= 2
        elif (
            kw["g"] == 1
            and not channels_fixed
            and kw["c"] % 2 == 0
            and kw["k"] % 2 == 0
        ):
            kw["c"] //= 2
            kw["k"] //= 2
        elif 1 < kw["g"] < kw["c"] and kw["g"] % 2 == 0:
            per_c, per_k = kw["c"] // kw["g"], kw["k"] // kw["g"]
            kw["g"] //= 2
            kw["c"], kw["k"] = per_c * kw["g"], per_k * kw["g"]
        elif kw["g"] == kw["c"] > 8 and kw["c"] % 2 == 0:
            mult = kw["k"] // kw["c"]
            kw["c"] //= 2
            kw["g"] = kw["c"]
            kw["k"] = mult * kw["c"]
        else:
            return None


# M: dgrad layers of models trained today (source cited in the plan).
M_SHAPES = {
    "M1": dict(
        rank=2, n=8, c=64, k=64, g=1, ins=(56, 56), f=3, pre=1, post=1
    ),  # ResNet-50 3x3
    "M2": dict(
        rank=2, n=8, c=64, k=256, g=1, ins=(56, 56), f=1
    ),  # ResNet-50 1x1 expand
    "M3": dict(
        rank=2, n=8, c=512, k=1024, g=1, ins=(28, 28), f=1, s=2
    ),  # ResNet-50 downsample
    "M4": dict(
        rank=2, n=8, c=3, k=64, g=1, ins=(224, 224), f=7, s=2, pre=3, post=3
    ),  # stem
    "M5": dict(
        rank=2, n=8, c=128, k=128, g=32, ins=(56, 56), f=3, pre=1, post=1
    ),  # ResNeXt
    "M6": dict(
        rank=2, n=8, c=96, k=96, g=96, ins=(56, 56), f=3, pre=1, post=1
    ),  # MobileNetV2 dw
    "M7": dict(
        rank=2, n=8, c=128, k=128, g=128, ins=(56, 56), f=7, pre=3, post=3
    ),  # ConvNeXt dw
    "M8": dict(
        rank=2, n=2, c=4, k=320, g=1, ins=(64, 64), f=3, pre=1, post=1
    ),  # SD UNet conv_in
    "M9": dict(
        rank=2, n=1, c=128, k=3, g=1, ins=(256, 256), f=3, pre=1, post=1
    ),  # SD VAE conv_out
    "M10": dict(
        rank=3, n=2, c=32, k=32, g=1, ins=(32, 32, 32), f=3, pre=1, post=1
    ),  # UNet3D
    "M11": dict(
        rank=1, n=8, c=256, k=256, g=256, ins=(128,), f=31, pre=15, post=15
    ),  # Conformer
    "M12": dict(
        rank=2, n=8, c=256, k=256, g=1, ins=(14, 14), f=3, pre=1, post=1
    ),  # ResNet-50 stage 3
}
M_PR = ("M1", "M8", "M11")  # the only cases that may exceed the PR ceiling


# H: hazard breakpoints, one change from the S defaults (N 2, C 16, K 32,
# 16 per dimension, filter 3, pad 1). C hazards hit dgrad's output vector width,
# K hazards its reduction.
H_SHAPES = {
    "H1": dict(c=15, k=32),
    "H2": dict(c=17, k=33),
    "H3": dict(c=31, k=31),
    "H4": dict(c=33, k=65),
    "H5": dict(c=63, k=64),
    "H6": dict(c=65, k=63),
    "H7": dict(c=7),  # output channels below the vector width
    "H8": dict(k=7),  # reduction channels below the vector width
    "H9": dict(n=3),
    "H10": dict(n=1),
    "H11": dict(n=32),
    "H12": dict(n=64),
    "H13": dict(ins=31),
    "H14": dict(ins=33),
    "H15": dict(ins=17),
    "H16": dict(c=12, k=12, g=3),
    "H17": dict(c=15, k=20, g=5),
}


def shape_H(hid, r, level):
    h = H_SHAPES[hid]
    base = _S_BASE[r]
    sizes = [base["ins"]] if r != 3 else list(_S_3D_LADDER)
    for ins in sizes:
        kw = dict(
            rank=r,
            n=h.get("n", base["n"]),
            c=h.get("c", base["c"]),
            k=h.get("k", base["k"]),
            g=h.get("g", 1),
            ins=_tup(h["ins"], r) if "ins" in h else ins,
            f=3,
            s=1,
            d=1,
            pre=1,
            post=1,
        )
        if "ins" in h and r == 3 and level == "quick":
            kw["ins"] = (h["ins"], 8, 4)
        c = new(**kw)
        if legal(c) is None and fits(c, level):
            return kw
    return None


# A: long accumulation over K (F23, nightly).
ACCUM_SHAPES = {
    "A1": dict(
        rank=2, n=1, c=64, k=512, g=1, ins=(14, 14), f=3, pre=1, post=1
    ),  # 4,608
    "A2": dict(
        rank=3, n=1, c=32, k=128, g=1, ins=(8, 8, 8), f=3, pre=1, post=1
    ),  # 3,456
}

# X: stress (full only).
X_SHAPES = {
    "X1": dict(rank=2, n=256, c=64, k=64, g=1, ins=(56, 56), f=3, pre=1, post=1),
    "X2": dict(rank=3, n=2, c=64, k=64, g=1, ins=(64, 64, 64), f=3, pre=1, post=1),
    "X3": dict(rank=2, n=1, c=1, k=1, g=1, ins=(46341, 46341), f=1),  # > 2^31 elements
    "X4": dict(
        rank=2, n=1, c=64, k=2048, g=1, ins=(14, 14), f=3, pre=1, post=1
    ),  # A 18,432
}


# R: real model dgrad geometries from the dnn-benchmarking corpus (plan section
# 3.3); pick, selection rule and pin are data in conv_bwd_corpus.json.
CORPUS_FILE = Path(__file__).with_name("conv_bwd_corpus.json")


def corpus():
    with open(CORPUS_FILE, encoding="utf-8") as fh:
        return json.load(fh)


R_SHAPES = {e["id"]: e for e in corpus()["cases"]}


def shape_R(rid):
    e = R_SHAPES[rid]
    geo = {k: e[k] for k in ("rank", "n", "c", "k", "g")}
    geo.update({k: tuple(e[k]) for k in ("ins", "f", "s", "d", "pre", "post")})
    lay = e["layout"]
    return dict(geo, dyl=lay, wl=lay, dxl=lay, dyt=e["dtype"])


def r_source(rid):
    e = R_SHAPES[rid]
    return (
        f"dnn-benchmarking corpus {e['shape_id']}; models: {', '.join(e['models'])}"
        f"; family: {', '.join(e['family'])}"
    )


# --------------------------------------------------------------------------
# Trimmed products (plan section 4.3). The level rules generate the complete
# products; the generator then removes these cases, so the cases that remain
# keep their ids, seeds and claims. Both trims remove products no engine
# executed on the measured architectures (plan section 4.5), and each names the
# trigger that brings its product back.
# --------------------------------------------------------------------------

# Fused topologies no engine executed: remove a topology to bring back its
# nightly and full repeats.
UNEXECUTED_TOPOLOGIES = ("dact", "pre_dact", "add", "add_drelu")
# Plain-dgrad features no engine executed: remove one to bring back its pairing
# with every S class in full.
UNEXECUTED_FEATURES = ("F13", "F19", "F35")

TRIMS = {
    "T9": "comprehensive and full: plain repeats of the fused topologies no engine executes",
    "T11": "full: F13, F19 and F35 paired with every S class",
}


def trim(c, level):
    """The trim (key of TRIMS) that removes case ``c`` from ``level``, or None.
    Feature-tagged fused cases (F30, F31) are never trimmed."""
    tagged = any(i.startswith("F") for i in c["ids"])
    late = level in ("comprehensive", "full")
    if late and c["topo"] in UNEXECUTED_TOPOLOGIES and not tagged:
        return "T9"
    if (
        level == "full"
        and c["topo"] == "dgrad"
        and any(i in UNEXECUTED_FEATURES for i in c["ids"])
    ):
        return "T11"
    return None


# --------------------------------------------------------------------------
# Value domains (F21)
# --------------------------------------------------------------------------


def value_recipes(c):
    """Fill recipe of dy and w for the case's value class."""
    cls, t = c["vals"], c["dyt"]
    if t not in DTYPE_MAX:
        return None
    if cls == "uniform":
        return dict(dy=("free", -1.0, 1.0), w=("free", -1.0, 1.0))
    if cls == "zero":
        return dict(dy=("fixed", 0.0), w=("free", -1.0, 1.0))
    if cls == "const":
        return dict(dy=("fixed", 1.0), w=("fixed", 1.0))
    if cls == "large":
        # Non-negative, so the sum cannot cancel: outputs sit near the dtype maximum.
        a = float(f"{math.sqrt(0.5 * DTYPE_MAX[t] / accum_len(c)):.3g}")
        return dict(dy=("free", 0.0, a), w=("free", 0.0, a))
    if cls == "tiny":
        a = 3e-3 if t == "half" else 1e-20
        return dict(dy=("free", -a, a), w=("free", -a, a))
    raise KeyError(cls)
