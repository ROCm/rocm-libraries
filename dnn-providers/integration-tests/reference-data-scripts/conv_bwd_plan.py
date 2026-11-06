#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Declarations of the convolution backward-data (dgrad) test plan.

This module is the machine-readable half of
``docs/test-plans/convolution-bwd.md``. It holds what the bundle generator
(``generate_conv_bwd_bundles.py``) and the exhaustive tool read:

* the case model of a dgrad graph (dy, w -> dx) and its fused topologies;
* the data types, activations, value domains and the purpose rule;
* the products that are generated and then trimmed (``TRIMS``), each with the
  trigger that brings it back.

The geometry is the forward convolution's: dgrad computes the gradient of
``y = conv(x, w)`` with respect to x, so its parameters (N, C, K, G, spatial,
filter, stride, dilation, padding), its legality rules, its FLOP count and its
shape catalog (S, L, M, H, A, X) are those of ``conv_fwd_plan.py``, which this
module imports. The forward x is dgrad's output dx and the forward y is dgrad's
input dy. Only the R set differs: it comes from the corpus's backward-data
graphs (``conv_bwd_corpus.json``).

It has no engine matrix. The one piece of engine evidence is ``TRIMS``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import conv_fwd_plan as F

# --------------------------------------------------------------------------
# Axes shared with the forward plan
# --------------------------------------------------------------------------

LEVELS = F.LEVELS
LEVEL_NAMES = F.LEVEL_NAMES
RANKS = F.RANKS
LAYOUTS = F.LAYOUTS
CORE_DTYPES = F.CORE_DTYPES
PAIRS = F.PAIRS
SHORT = F.SHORT
BYTES = F.BYTES
DTYPE_MAX = F.DTYPE_MAX
CEILINGS = F.CEILINGS

# Geometry: the forward x is dx, the forward y is dy.
legal = F.legal
outs = F.outs
dx_dims = F.x_dims
w_dims = F.w_dims
dy_dims = F.y_dims
strides = F.strides
fastest_dim = F.fastest_dim
numel = F.numel
flops = F.flops
fp32_bytes = F.fp32_bytes
fits = F.fits
canon = F.canon
short_hash = F.short_hash

S_IDS = F.S_IDS
S_QUICK = F.S_QUICK
S_CROSSED = F.S_CROSSED
shape_S = F.shape_S
shape_L = F.shape_L
shape_H = F.shape_H
H_SHAPES = F.H_SHAPES
M_SHAPES = F.M_SHAPES
M_PR = F.M_PR
X_SHAPES = F.X_SHAPES
ACCUM_SHAPES = F.ACCUM_SHAPES
GEOMETRY_CLASS_SHAPES = F.GEOMETRY_CLASS_SHAPES

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
HAS_RESIDUAL = ("add", "add_drelu")
HAS_SAVED_X = ("dact", "add_drelu")  # forward input of the previous activation
HAS_SAVED_Y = ("pre_dact",)  # forward input of this layer's activation

VALUE_CLASSES = F.VALUE_CLASSES

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


def new(rank, n, c, k, g, ins, f, s=1, d=1, pre=0, post=0, **kw):
    """Build a finalized case from per-dimension values (scalars broadcast)."""
    case = dict(DEFAULTS)
    case.update(
        rank=rank,
        n=n,
        c=c,
        k=k,
        g=g,
        ins=F._tup(ins, rank),
        f=F._tup(f, rank),
        s=F._tup(s, rank),
        d=F._tup(d, rank),
        pre=F._tup(pre, rank),
        post=F._tup(post, rank),
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


def accum_len(c):
    """Most products summed into one dx element: K/G times the filter taps that
    land on it. Along a dimension with stride s and dilation d, a dx position
    meets one tap in every s / gcd(s, d)."""
    taps = 1
    for f, s, d in zip(c["f"], c["s"], c["d"]):
        taps *= -(-f // (s // math.gcd(s, d)))
    return (c["k"] // c["g"]) * taps


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
# R: real model backward-data geometries from the dnn-benchmarking corpus
# (plan section 3.3). Pick, selection rule and pin are data in
# conv_bwd_corpus.json; the cases keep the model's data type and layout.
# --------------------------------------------------------------------------

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
# keep their ids, seeds and claims. T9 and T11 remove products that no engine
# executed on the measured architectures (gfx90a and gfx942, plan section 4.5);
# T12 is a design trade-off (the forward plan's T12). Each trim names the trigger
# that brings its product back. The ids follow the forward plan's register; T8
# and T10 have no dgrad form.
# --------------------------------------------------------------------------

# Fused topologies no engine executed (plan section 4.5). An engine that executes
# one brings its nightly and full repeats back: remove it from this tuple.
UNEXECUTED_TOPOLOGIES = ("dact", "pre_dact", "add", "add_drelu")
# Plain-dgrad features no engine executed. An engine that executes one brings back
# its pairing with every S shape in full: remove it from this tuple.
UNEXECUTED_FEATURES = ("F13", "F19", "F35")

TRIMS = {
    "T9": "comprehensive and full: plain repeats of the fused topologies no engine executes",
    "T11": "full: F13, F19 and F35 paired with every S shape",
    "T12": "comprehensive and full: geometry classes S24-S62 crossed with fused topologies or features",
}


def trim(c, level):
    """The trim (key of TRIMS) that removes case ``c`` from ``level``, or None.
    Feature-tagged fused cases (F30, F31) are never trimmed."""
    tagged = any(i.startswith("F") for i in c["ids"])
    late = level in ("comprehensive", "full")
    if late and c["topo"] in UNEXECUTED_TOPOLOGIES and not tagged:
        return "T9"
    if (
        late
        and (c["topo"] != "dgrad" or tagged)
        and any(i in GEOMETRY_CLASS_SHAPES for i in c["ids"])
    ):
        return "T12"
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
