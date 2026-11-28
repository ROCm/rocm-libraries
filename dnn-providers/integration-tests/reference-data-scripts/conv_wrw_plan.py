#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declarations of the convolution backward-weights (wrw, wgrad) test plan.

This module is the machine-readable half of
``docs/test-plans/convolution-wrw.md``. It holds what the bundle generator
(``generate_conv_wrw_bundles.py``) and the exhaustive tool read:

* the case model of a wrw graph (x, dy -> dw);
* the legality rules, which copy ``ConvolutionWgradNode::pre_validate_node``;
* the wrw shape catalog (S, L, M, R, H, A, X) and the per-level ceilings;
* the data types, value domains and the purpose rule;
* the products that are generated and then trimmed (``TRIMS``), each with the
  trigger that brings it back.

The catalog is built around what wrw computes, not around the forward or
data-gradient convolution. A dw element sums over the whole batch and every dy
position: ``A = N x prod(dy spatial)`` products, so the reduction length is the
batch times the output image and reaches millions, while the output (K x C/G x
filter) can be one element or a million. Engines pick their kernels on exactly
these properties: how long the reduction is against how large the output is
(split-K and atomics), the filter shape (Winograd, 1x1 and depthwise kernels),
the stride against the filter (x elements no window reads), padding that reaches
past the filter (dw taps that read only padding), and the channel counts. A
case is a plain ``dict``; ``finalize`` resolves every default so that two
spellings of one graph compare equal. There is no engine matrix; the engine
evidence is ``TRIMS``.
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

# F19: io types beyond the core three. (x, dy, dw, node compute type)
EXTRA_TYPES = {
    "fp8_e4m3": ("fp8_e4m3", "fp8_e4m3", "float", "float"),
    "fp8_e5m2": ("fp8_e5m2", "fp8_e5m2", "float", "float"),
    "fp8_e4m3_fnuz": ("fp8_e4m3_fnuz", "fp8_e4m3_fnuz", "float", "float"),
    "fp8_e5m2_fnuz": ("fp8_e5m2_fnuz", "fp8_e5m2_fnuz", "float", "float"),
    "int8": ("int8", "int8", "int32", "int32"),
    "double": ("double", "double", "double", "double"),
}
# F20: mixed precision. (x, dy, dw, node compute type)
MIXED_TYPES = {
    "half_in_float_out": ("half", "half", "float", "float"),
    "bf16_in_float_out": ("bfloat16", "bfloat16", "float", "float"),
    "x_half_dy_float": ("half", "float", "float", "float"),
    "x_float_dy_half": ("float", "half", "float", "float"),
    "half_compute_half": ("half", "half", "half", "half"),
}

# The positive value class stops at this A (2^20). The GPU reference sums one dw
# element in one thread, sequentially in fp32; for 4.2 million products in [0, 1]
# that sum stagnates (about 1.9e-3 relative below the exact sum, reproduced with a
# sequential fp32 sum in NumPy), which is the oracle's error and not the engine's.
POSITIVE_MAX_A = 1 << 20

# F21: value domains the fill recipes can express. The recipes are per operand
# (``value_recipes``): the fills of x and dy differ in the classes that set one
# of them (zero), and every class sets both.
VALUE_CLASSES = ("uniform", "zero", "const", "positive", "large", "tiny")

# Per-level ceilings for each case: (FLOP, bytes of fp32 x + dy + dw). None = unbounded.
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
    xl="cf",
    dyl="cf",
    dwl="cf",  # layout of x, dy, dw
    xe=(),
    dye=(),
    dwe=(),  # per-dim extra stride padding (non-packed); () = packed
    xt="float",
    dyt=None,
    dwt=None,
    comp=None,  # data types; None resolves in finalize
    conv_mode="CROSS_CORRELATION",
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
    ``ins`` is the x spatial size; dy's follows from the output formula."""
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
    c["dyt"] = c["dyt"] or c["xt"]
    if c["dwt"] is None:
        c["dwt"] = c["xt"]
    if c["comp"] is None:
        c["comp"] = {"int8": "int32", "double": "double"}.get(c["xt"], "float")
    for key in ("xe", "dye", "dwe", "ids"):
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
    """Legality, copied from ConvolutionWgradNode::pre_validate_node: equal
    ranks, dy[1] == dw[0], x[1] % dw[1] == 0 and dw[0] % G == 0, one parameter
    per spatial dim, stride and dilation >= 1, padding >= 0, and
    x + pre + post >= the dilated filter. dy's dims are computed from the output
    formula, so the formula check holds by construction. dw dims are always set:
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
            return "filter larger than padded x"
    return None


def x_dims(c):
    return [c["n"], c["c"], *c["ins"]]


def dw_dims(c):
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
    return 4 * (numel(x_dims(c)) + numel(dw_dims(c)) + numel(dy_dims(c)))


def accum_len(c):
    """Products summed into one dw element: the batch times the dy volume. Taps
    at the border read padding, which adds zero, so this is an upper bound."""
    return c["n"] * math.prod(outs(c))


def dw_numel(c):
    return numel(dw_dims(c))


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
# Wrw behaviour classes (plan section 2.1). The corpus analysis and the
# coverage check classify graphs with these; the catalog below is built to hit
# every class.
# --------------------------------------------------------------------------


def stride_class(c):
    """How a window moves over x, per dimension, merged: s1; f=s (windows tile x
    without overlap); f>s (windows overlap); f<s (x elements between windows are
    read by no tap); gcd>1 (stride and dilation share a factor: the taps of one
    window skip every s/gcd-th x element)."""
    out = set()
    for f, s, d in zip(c["f"], c["s"], c["d"]):
        if s == 1:
            out.add("s1")
        elif math.gcd(s, d) > 1:
            out.add("gcd>1")
        elif f < s:
            out.add("f<s")
        elif f == s:
            out.add("f=s")
        else:
            out.add("f>s")
    return "/".join(sorted(out))


def pad_class(c):
    """Padding against the dilated filter. p0: no padding (every product reads
    x); p<=span: the usual case (border taps sum fewer products); p>span: padding
    wider than the dilated filter, so dy positions at the border read only
    padding and add nothing. asym: pre differs from post."""
    if c["pre"] != c["post"]:
        return "asym"
    if not any(c["pre"]):
        return "p0"
    if any(p > d * (f - 1) for f, d, p in zip(c["f"], c["d"], c["pre"])):
        return "p>span"
    return "p<=span"


def empty_taps(c):
    """True when some filter tap reads only padding for every dy position: that
    dw element is exactly 0 (the filter is larger than x, or padding reaches
    past the image)."""
    for i in range(c["rank"]):
        n_out = outs(c)[i]
        for r in range(c["f"][i]):
            first = -c["pre"][i] + r * c["d"][i]
            last = (n_out - 1) * c["s"][i] - c["pre"][i] + r * c["d"][i]
            if last < 0 or first >= c["ins"][i]:
                return True
    return False


def remainder(c):
    """True when (x + pads - dilated filter) % stride != 0: x's last rows lie
    past the last window and no tap reads them."""
    return any(
        (i + a + b - (d * (f - 1) + 1)) % s
        for i, a, b, d, f, s in zip(
            c["ins"], c["pre"], c["post"], c["d"], c["f"], c["s"]
        )
    )


def a_class(c):
    """Reduction length A: engines split a long reduction across workgroups
    (split-K, atomics) and use a different kernel for a short one."""
    a = accum_len(c)
    return "A<=1k" if a <= 1024 else ("A<=64k" if a <= 65536 else "A>64k")


def channel_class(v):
    if v in (1, 2, 3, 4):
        return str(v)
    if v % 8:
        return "odd<16" if v < 16 else ("non8<=64" if v <= 64 else "non8>64")
    return "8-32" if v <= 32 else ("40-256" if v <= 256 else ">256")


def signature_channel(v):
    """Channel class of the behaviour signature: 1 to 4 individually (the
    vector-width cases), then one class."""
    return str(v) if v <= 4 else ">4"


def group_class(c):
    if c["g"] == 1:
        return "dense"
    return "dw" if c["g"] == c["c"] else "grouped"


def filter_class(c):
    """Filter shape. Engines have dedicated kernels for 1x1, for 3x3 (Winograd)
    and for a few non-square and large shapes."""
    f = c["f"]
    if all(v == 1 for v in f):
        return "1x1"
    if all(v == 3 for v in f):
        return "3x3"
    if len(set(f)) > 1:
        return "nonsq"
    return "sq5" if f[0] <= 5 else "sq>5"


def signature(c):
    """Behaviour signature of a wrw graph (plan section 3.3): rank, group
    class, C/G and K/G classes, filter class, stride class, padding class and
    reduction class. A 1D graph carried as 2D with H = 1 counts as 1D."""
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
        signature_channel(c["c"] // c["g"]),
        signature_channel(c["k"] // c["g"]),
        filter_class(c),
        stride_class(c),
        pad_class(c),
        a_class(c),
    )


# --------------------------------------------------------------------------
# Purpose (RFC 0022 section 9)
# --------------------------------------------------------------------------


def probe_reasons(c):
    why = []
    if c["conv_mode"] != "CROSS_CORRELATION":
        why.append("F13 conv_mode")
    if c["pre"] != c["post"]:
        why.append("F11 asymmetric padding")
    if c["rank"] not in RANKS:
        why.append("F1 4D spatial")
    if c["xe"] or c["dye"] or c["dwe"]:
        why.append("F16 non-packed strides")
    if not (c["xl"] == c["dyl"] == c["dwl"]):
        why.append("F15 mixed layouts")
    if c["alignment"]:
        why.append("F17 alignment")
    if c["xt"] not in CORE_DTYPES or c["dyt"] != c["xt"] or c["dwt"] != c["xt"]:
        why.append("F19/F20 data types")
    if c["comp"] != "float" and c["xt"] in CORE_DTYPES:
        why.append("F20 compute type")
    if c["override"]:
        why.append("F35 override shape")
    if c["vals"] != "uniform":
        why.append("F21 value domain")
    if c["size"] == "H":
        why.append("H hazard")
    if "X3" in c["ids"]:
        why.append("F18 64-bit indexing")
    return why


def purpose(c):
    if probe_reasons(c):
        return "probe"
    return "regression" if c["size"] == "B" else "claim"


# --------------------------------------------------------------------------
# Shape catalog (plan section 3). S shapes change one wrw feature from the
# defaults; L shapes are the same classes at medium size. Each entry is
# (ranks, small kwargs, medium kwargs); kwargs override the base of the rank.
# A medium value of None means "same as small". Groups:
#   S1-S16, S84, S85   basics, padding reads and filter shapes
#   S17-S31, S86-S88   stride against the filter: skipped x elements
#   S32-S42            channel classes: dw rows K and columns C/G
#   S43-S52            grouped convolution
#   S53-S64            depthwise
#   S65-S72            image size and tiling (the dy extent)
#   S73-S76            rank-specific classes (1D audio, 4D)
#   S77-S83            output heads, latent convs and one dy channel
# --------------------------------------------------------------------------

# Bases. N 2: no default dimension of 1 (RFC 0022 section 7). C 16, K 32: the
# dw rows (K) differ from the columns (C/G).
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

# Long-reduction base (the LA forms): the batch and the image are sized so that
# A = N x dy volume is above 64k, where engines split the reduction.
_LA_BASE = {
    1: dict(n=8, c=16, k=16, g=1, ins=(16384,)),
    2: dict(n=16, c=16, k=16, g=1, ins=(224, 224)),
    3: dict(n=4, c=8, k=8, g=1, ins=(32, 32, 32)),
}
LA_MIN = 65537  # the smallest A of an LA form: above 2^16

R123, R12, R2, R23, R1, R3 = (1, 2, 3), (1, 2), (2,), (2, 3), (1,), (3,)


def _dw(c, mult=1, **kw):
    return dict(c=c, k=mult * c, g=c, **kw)


def _grp(g, cg, kg, **kw):
    return dict(c=g * cg, k=g * kg, g=g, **kw)


# id -> (ranks, small, medium). Spatial values given per rank are truncated or
# broadcast by the builder; "ins" given as a dict maps rank -> size.
S_CLASSES = {
    # --- basics, padding reads and filter shapes ------------------------------
    "S1": (R123, dict(f=1, pre=0), None),  # 1x1: dw is a GEMM over N x dy
    "S2": (R123, dict(pre=0), None),  # no padding: every product reads x
    "S3": (R123, dict(), None),  # 3x3 "same"
    "S4": (R123, dict(f=5, pre=2), None),
    "S5": (R12, dict(f=7, pre=3), None),
    "S6": (R123, dict(d=2, pre=2), None),  # dilation, stride 1
    "S7": (
        R23,
        dict(ins={2: (16, 8), 3: (16, 8, 4)}),
        dict(ins={2: (56, 28), 3: (32, 32, 16)}),
    ),
    "S8": (R123, dict(n=8, f=1, pre=0), None),  # batch
    "S9": (
        R123,
        dict(pre=4),
        None,
    ),  # pad > d(f-1): border dy positions read only padding
    "S10": (R123, dict(f=2, s=2, pre=0, post=1), None),  # asymmetric (probe)
    "S11": (R123, dict(f=2, s=2, pre=1, post=0), None),  # pre-only (probe)
    "S12": (R123, dict(f=2, pre=1), None),  # even filter: dy larger than x
    "S13": (R123, dict(f="ins", pre=0), None),  # filter = x: dy spatial 1, A = N
    "S14": (R123, dict(f="ins+2", pre=1), None),  # filter larger than x: empty taps
    "S15": (R23, dict(s=(1, 2, 1), d=(2, 1, 2), pre=(1, 0, 1)), None),
    "S16a": (R23, dict(f=(1, 3, 3), pre=(0, 1, 1)), None),  # 2D 1x3, 3D 1x3x3
    "S16b": (R23, dict(f=(3, 1, 1), pre=(1, 0, 0)), None),  # 2D 3x1, 3D 3x1x1
    "S16c": (R2, dict(f=(3, 5), pre=(1, 2)), None),
    "S84": (
        R12,
        dict(d=12, pre=12),
        None,
    ),  # dilation 12: the dilated filter is 25 wide
    "S85": (
        R123,
        dict(pre=8),
        None,
    ),  # pad 8 on a 3-wide filter: most dy positions read padding
    "S86": (R2, dict(f=(1, 7), pre=(0, 3)), None),  # 1x7: Winograd multipass wrw shape
    "S87": (R2, dict(f=(7, 1), pre=(3, 0)), None),  # 7x1
    # --- stride against the filter: x elements no window reads ---------------
    "S17": (R123, dict(s=2), None),  # f3 s2 p1, even x: overlapping windows + remainder
    "S18": (R123, dict(s=2, ins="odd"), None),  # f3 s2 p1, odd x: no remainder
    "S19": (R123, dict(f=2, s=2, pre=0), None),  # patchify 2: windows tile x
    "S20": (R12, dict(f=4, s=2, pre=1), None),  # f = 2s
    "S21": (R2, dict(f=4, s=4, pre=0), None),  # patchify 4
    "S22": (R2, dict(c=3, f=16, s=16, pre=0, ins=(32, 32)), dict(c=3, ins=(224, 224))),
    "S23": (
        R123,
        dict(f=1, s=2, pre=0),
        None,
    ),  # f<s: 3 of 4 x elements read by no tap (2D)
    "S24": (R123, dict(s=4), None),  # f3 s4: f<s with padding
    "S25": (
        R123,
        dict(s=3, pre=0),
        None,
    ),  # f3 s3 on 16: x's last row read by no window
    "S26": (R123, dict(s=2, d=2, pre=2), None),  # gcd(s, d) = 2
    "S27": (R12, dict(s=2, d=3, pre=3), None),  # coprime stride and dilation
    "S28": (R123, dict(c=3, f=7, s=2, pre=3), None),  # strided stem, C 3
    "S29": (R123, dict(s=2, pre=0), None),  # f3 s2 no pad
    "S30": (R2, dict(f=7, s=2, pre=3), None),  # large filter, stride 2
    "S31": (R3, dict(f=(1, 3, 3), s=(1, 2, 2), pre=(0, 1, 1)), None),
    "S89": (
        R3,
        dict(f=1, pre=0, ins=(4, 4, 4)),
        None,
    ),  # 1x1x1 on a small volume: A 128
    "S90": (  # (2+1)D video, strided in two dimensions, on a small volume: A 128
        R3,
        dict(f=(1, 3, 3), s=(1, 2, 2), pre=(0, 1, 1), ins=(4, 8, 8)),
        None,
    ),
    "S88": (  # DeepSpeech2 21x11 conv: tall non-square filter, per-dimension stride
        R2,
        dict(c=8, k=16, f=(21, 11), s=(2, 1), pre=(10, 5), ins=(48, 32)),
        dict(c=1, k=32, f=(21, 11), s=(2, 1), pre=(10, 5), ins=(161, 700)),
    ),
    # --- channel classes: K rows and C/G columns of dw -----------------------
    "S32": (R123, dict(c=1), None),  # one x channel
    "S33": (R23, dict(c=3, k=64), dict(c=3, k=64)),  # RGB input
    "S34": (
        R2,
        dict(c=4, k=320),
        dict(c=4, k=320),
    ),  # small C, large K (latent conv_in)
    "S35": (R123, dict(k=1), None),  # one dy channel: dw is one row
    "S36": (R23, dict(c=128, k=3), dict(c=128, k=3)),  # wide C, K 3 (image conv_out)
    "S37": (
        R2,
        dict(c=64, k=2, f=1, pre=0),
        dict(c=256, k=2, f=1, pre=0),
    ),  # binary head
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
    "S50": (R123, _grp(4, 8, 8, s=2), _grp(16, 8, 8, s=2)),  # grouped, strided
    "S51": (
        R2,
        _grp(2, 8, 8, f=5, d=2, pre=4, ins=(15, 15)),
        _grp(4, 16, 16, f=5, d=2, pre=4, ins=(55, 55)),
    ),
    "S52": (R123, _grp(4, 8, 1), _grp(16, 8, 1)),  # K/G = 1: one dy channel per group
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
    # --- image size and tiling (the dy extent) -------------------------------
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
    "S69": (R2, dict(ins=(17, 19)), dict(ins=(55, 57))),  # odd non-square x
    "S70": (
        R2,
        dict(f=1, pre=0, ins=(1, 1), n=8),
        dict(n=32, c=256, k=512, f=1, pre=0, ins=(1, 1)),
    ),  # a pure GEMM: dw = dy^T x over the batch
    "S71": (
        R2,
        dict(ins=(1, 256), f=(1, 3), pre=(0, 1)),
        dict(ins=(1, 4096), f=(1, 3), pre=(0, 1)),
    ),  # extreme aspect (1D carried as 2D)
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
    ),  # wav2vec feature extractor
    "S76": ((4,), dict(), None),  # 4D spatial (probe)
    # --- output heads and latent convs: few dy channels ----------------------
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
    # --- one dy channel (K/G = 1) with padding reads, strides and groups -----
    "S81": (R123, dict(k=1, pre=2), None),  # K/G 1, pad f-1
    "S82": (R123, dict(k=1, s=2), None),  # K/G 1, f > s, 3 taps at stride 2
    "S83": (R123, _grp(4, 8, 1, s=2, d=2, pre=2), None),  # K/G 1, grouped, gcd(s, d) 2
}
S_IDS = tuple(S_CLASSES)
# Classes with an LA form: those that do not fix the batch or the image extent
# themselves, so that the long reduction can be put under them (S7, S13, S14,
# S18 and the image-size classes keep their extent; the 4D probe has no form).
LA_IDS = tuple(
    sid
    for sid, (ranks, s_kw, _) in S_CLASSES.items()
    if "ins" not in s_kw
    and "n" not in s_kw
    and s_kw.get("f") not in ("ins", "ins+2")
    and ranks != (4,)
)
# Quick and the PR core: one class from each group that every engine family meets.
S_QUICK = ("S1", "S3", "S6", "S8", "S17", "S19", "S23", "S44", "S53")
# Classes the feature probes rotate over: basics and stride. Channel, grouped,
# depthwise, tiling and rank-specific classes (S32-S83) select the wrw kernel
# itself, not a per-tensor feature, so a probe does not need them (plan section
# 4.3, D3).
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


def _resolve(sid, r, form):
    """Case kwargs of class ``sid`` at rank ``r`` on the small (``"S"``), medium
    (``"L"``) or long-reduction (``"LA"``) base, or None when the class has no
    form of that rank. The medium form takes the class's medium values when it
    declares them; otherwise it keeps the small form's channel, group and batch
    values and takes the medium image. The long-reduction form keeps the class's
    filter, stride, dilation, padding, channel and group values and takes the
    base's batch and image."""
    ranks, s_kw, l_kw = S_CLASSES[sid]
    if r not in ranks:
        return None
    small = form == "S"
    base = {"S": _S_BASE, "L": _L_BASE, "LA": _LA_BASE}[form][r]
    if form == "LA":
        over = {k: v for k, v in s_kw.items() if k != "ins"}
    elif small or l_kw is not None:
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
    kw = _resolve(sid, r, "S")
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
    kw = _resolve(sid, r, "L")
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


def shape_LA(sid, r, level="comprehensive"):
    """Long-reduction form of class ``sid``: the class's kernel-selecting values
    under a batch and an image that make A above 2^16. Halves N until the case
    fits; None when the class has no such form (the filter does not fit the
    image, or A stays below ``LA_MIN`` at the ceiling)."""
    kw = _resolve(sid, r, "LA")
    if kw is None:
        return None
    while True:
        c = new(**kw)
        if legal(c) is not None:
            return None
        if fits(c, level):
            return kw if accum_len(c) >= LA_MIN else None
        if kw["n"] <= 1:
            return None
        kw["n"] //= 2


# M: wrw layers of models trained today (source cited in the plan). The layers
# of the dgrad plan, with the batch of a training step: wrw reduces over it.
M_SHAPES = {
    "M1": dict(rank=2, n=8, c=64, k=64, g=1, ins=(56, 56), f=3, pre=1, post=1),
    "M2": dict(rank=2, n=8, c=64, k=256, g=1, ins=(56, 56), f=1),
    "M3": dict(rank=2, n=8, c=512, k=1024, g=1, ins=(28, 28), f=1, s=2),
    "M4": dict(rank=2, n=8, c=3, k=64, g=1, ins=(224, 224), f=7, s=2, pre=3, post=3),
    "M5": dict(rank=2, n=8, c=128, k=128, g=32, ins=(56, 56), f=3, pre=1, post=1),
    "M6": dict(rank=2, n=8, c=96, k=96, g=96, ins=(56, 56), f=3, pre=1, post=1),
    "M7": dict(rank=2, n=8, c=128, k=128, g=128, ins=(56, 56), f=7, pre=3, post=3),
    "M8": dict(rank=2, n=2, c=4, k=320, g=1, ins=(64, 64), f=3, pre=1, post=1),
    "M9": dict(rank=2, n=1, c=128, k=3, g=1, ins=(256, 256), f=3, pre=1, post=1),
    "M10": dict(rank=3, n=2, c=32, k=32, g=1, ins=(32, 32, 32), f=3, pre=1, post=1),
    "M11": dict(rank=1, n=8, c=256, k=256, g=256, ins=(128,), f=31, pre=15, post=15),
    "M12": dict(rank=2, n=8, c=256, k=256, g=1, ins=(14, 14), f=3, pre=1, post=1),
}
M_PR = ("M1", "M4", "M11")  # the only cases that may exceed the PR ceiling


# H: hazard breakpoints, one change from the S defaults (N 2, C 16, K 32,
# 16 per dimension, filter 3, pad 1). K hazards hit dw's row tiles, C hazards
# its columns, and N and the extent the reduction N x dy volume.
H_SHAPES = {
    "H1": dict(c=15, k=32),
    "H2": dict(c=17, k=33),
    "H3": dict(c=31, k=31),
    "H4": dict(c=33, k=65),
    "H5": dict(c=63, k=64),
    "H6": dict(c=65, k=63),
    "H7": dict(c=7),  # columns below the vector width
    "H8": dict(k=7),  # rows below the vector width
    "H9": dict(n=3),
    "H10": dict(n=1),
    "H11": dict(n=32),
    "H12": dict(n=64),
    "H13": dict(ins=31),
    "H14": dict(ins=33),
    "H15": dict(ins=17),
    "H16": dict(c=12, k=12, g=3),
    "H17": dict(c=15, k=20, g=5),
    "H18": dict(ins=7),  # A = 2 x 49 = 98: a reduction that is not a multiple of 8
    "H19": dict(n=5, ins=9),  # A = 405: odd
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


# A: the reduction ladder (F23). A = N x dy volume; the channels stay small so
# that the cost is the reduction, and so that the output is small against it
# (the split-K regime). Each has the A it reaches in the comment.
ACCUM_SHAPES = {
    "A1": dict(rank=2, n=4, c=16, k=16, g=1, ins=(32, 32), f=3, pre=1, post=1),  # 4,096
    "A2": dict(
        rank=2, n=8, c=16, k=16, g=1, ins=(64, 64), f=3, pre=1, post=1
    ),  # 32,768
    "A3": dict(
        rank=2, n=16, c=16, k=16, g=1, ins=(112, 112), f=3, pre=1, post=1
    ),  # 200,704
    "A4": dict(
        rank=2, n=32, c=8, k=8, g=1, ins=(128, 128), f=3, pre=1, post=1
    ),  # 524,288
    "A5": dict(
        rank=2, n=1, c=1, k=1, g=1, ins=(2048, 2048), f=1
    ),  # 4.2M, dw has one element
    "A6": dict(
        rank=1, n=4, c=16, k=16, g=1, ins=(65536,), f=3, pre=1, post=1
    ),  # 262,144
    "A7": dict(
        rank=3, n=2, c=8, k=8, g=1, ins=(48, 48, 48), f=3, pre=1, post=1
    ),  # 221,184
    "A8": dict(
        rank=2, n=1, c=1, k=1, g=1, ins=(4096, 4096), f=3, pre=1, post=1
    ),  # 16.8M
    "A9": dict(rank=2, n=64, c=8, k=8, g=1, ins=(224, 224), f=3, pre=1, post=1),  # 3.2M
    # The opposite corner: a reduction of 1 to 16 and a large output.
    "A10": dict(
        rank=2, n=1, c=256, k=256, g=1, ins=(4, 4), f=3, pre=1, post=1
    ),  # A 16, dw 590k
    "A11": dict(
        rank=2, n=1, c=1024, k=1024, g=1, ins=(1, 1), f=1
    ),  # A 1: an outer product
}

# B: reproducers of escaped wrw defects. Each is the shape of a case that failed
# (ROCm/rocm-libraries#8029, #8030): a split-K kernel adds the partial results of a
# long reduction into a small output in a data type of its own, and the sum
# differed run to run; the bundles that held these cases before this plan are the
# reason the shapes are here. 3D and 2D, N 8, C 16, one dy channel, 1x1 filter.
REGRESSION_SHAPES = {
    "B1": dict(rank=3, n=8, c=16, k=1, g=1, ins=(16, 16, 16), f=1),  # A 32,768, dw 16
    "B2": dict(rank=2, n=8, c=16, k=1, g=1, ins=(16, 16), f=1),  # A 2,048, dw 16
}

# X: stress (full only).
X_SHAPES = {
    "X1": dict(rank=2, n=256, c=64, k=64, g=1, ins=(56, 56), f=3, pre=1, post=1),
    "X2": dict(rank=3, n=2, c=64, k=64, g=1, ins=(64, 64, 64), f=3, pre=1, post=1),
    # More than 2^31 elements of x and dy, and A above 2^31 into one dw element.
    "X3": dict(rank=2, n=1, c=1, k=1, g=1, ins=(46341, 46341), f=1),
    "X4": dict(
        rank=2, n=1, c=64, k=2048, g=1, ins=(14, 14), f=3, pre=1, post=1
    ),  # A 196, dw 1.2M
    "X5": dict(rank=2, n=128, c=32, k=32, g=1, ins=(112, 112), f=1),  # A 1.6M, dw 1,024
}


# R: real model wrw geometries from the dnn-benchmarking corpus (plan section
# 3.3); pick, selection rule and pin are data in conv_wrw_corpus.json.
CORPUS_FILE = Path(__file__).with_name("conv_wrw_corpus.json")


def corpus():
    with open(CORPUS_FILE, encoding="utf-8") as fh:
        return json.load(fh)


R_SHAPES = {e["id"]: e for e in corpus()["cases"]} if CORPUS_FILE.exists() else {}


def shape_R(rid):
    e = R_SHAPES[rid]
    geo = {k: e[k] for k in ("rank", "n", "c", "k", "g")}
    geo.update({k: tuple(e[k]) for k in ("ins", "f", "s", "d", "pre", "post")})
    lay = e["layout"]
    return dict(geo, xl=lay, dyl=lay, dwl=lay, xt=e["dtype"])


def r_source(rid):
    e = R_SHAPES[rid]
    return (
        f"dnn-benchmarking corpus {e['shape_id']}; models: {', '.join(e['models'])}"
        f"; family: {', '.join(e['family'])}"
    )


# --------------------------------------------------------------------------
# Trimmed products (plan section 4.3). The level rules generate the complete
# products; the generator then removes these cases, so the cases that remain
# keep their ids, seeds and claims. The trim removes products no engine
# executed on the measured architectures (plan section 4.5), and names the
# trigger that brings its product back.
# --------------------------------------------------------------------------

# Features no engine executed: remove one to bring back its pairing with every
# S class in full.
UNEXECUTED_FEATURES = ("F13", "F19", "F35")

TRIMS = {
    "T11": "full: F13, F19 and F35 paired with every S class",
}


def trim(c, level):
    """The trim (key of TRIMS) that removes case ``c`` from ``level``, or None."""
    if level == "full" and any(i in UNEXECUTED_FEATURES for i in c["ids"]):
        return "T11"
    return None


# --------------------------------------------------------------------------
# Value domains (F21)
# --------------------------------------------------------------------------


def value_recipes(c):
    """Fill recipe of dy and x for the case's value class, keyed by operand
    role. The generator maps the roles to tensor uids. ``None`` for a data type
    the classes cannot scale (fp8, int8, double)."""
    cls, t = c["vals"], c["xt"]
    if t not in DTYPE_MAX:
        return None
    a_len = accum_len(c)
    if cls == "uniform":
        return dict(dy=("free", -1.0, 1.0), x=("free", -1.0, 1.0))
    if cls == "zero":
        # dy zero: dw must be exactly 0 whatever x holds.
        return dict(dy=("fixed", 0.0), x=("free", -1.0, 1.0))
    if cls == "const":
        # Both 1.0: dw counts the products that read x, so a wrong tap or a
        # dropped split of the reduction is an integer error.
        if a_len > 0.5 * DTYPE_MAX[t]:
            return None
        return dict(dy=("fixed", 1.0), x=("fixed", 1.0))
    if cls == "positive":
        if a_len > POSITIVE_MAX_A:
            return None
        # Non-negative at unit scale: the sum cannot cancel, so it grows with A
        # (about A / 4) and exposes accumulation error and overflow of the
        # partial sums. Scaled down where A / 4 would pass the data type's maximum.
        a = float(f"{min(1.0, math.sqrt(0.4 * DTYPE_MAX[t] / a_len)):.3g}")
        return dict(dy=("free", 0.0, a), x=("free", 0.0, a))
    if cls == "large":
        # Non-negative, so the sum cannot cancel: outputs sit near the dtype maximum.
        a = float(f"{math.sqrt(0.5 * DTYPE_MAX[t] / a_len):.3g}")
        return dict(dy=("free", 0.0, a), x=("free", 0.0, a))
    if cls == "tiny":
        # Small, and far enough above the data type's smallest normal number
        # (1.2e-38 for fp32 and bf16, 6.1e-5 for fp16) that almost no product of
        # two such values falls below it: a kernel that flushes sub-normal
        # products to zero then differs from the reference by a few percent per
        # element, which no tolerance can grade (a = 1e-18 gave 1 to 3 percent).
        a = 3e-1 if t == "half" else 1e-15
        return dict(dy=("free", -a, a), x=("free", -a, a))
    raise KeyError(cls)
