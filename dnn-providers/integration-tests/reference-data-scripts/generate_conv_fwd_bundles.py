#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the convolution-forward bundles from the test plan.

Reads the declarations in ``conv_fwd_plan.py`` and the level rules below, and
writes, for each level (quick, standard, comprehensive, full), one template
sweep per topology under ``integration-test-bundles/<level>/``:

    ConvolutionFwd/Default                  conv
    ConvolutionFwdPointwise/Default         conv -> act
    ConvolutionFwdPointwise/Variant2        conv -> bias -> act
    ConvolutionFwdPointwise/Variant3        conv -> scale -> bias -> relu
    ConvolutionFwdPointwise/Variant4        conv -> add(residual) -> relu
    ConvolutionFwdPointwise/Variant5        conv -> bias -> add(residual) -> relu
    ConvolutionFwdPointwise/Variant6        conv -> relu -> add(residual)

Each directory gets ``graph.template.json``, ``sweep.json`` and an empty
``support.json`` (claims are harvested with ``--write-support-claims``, never
written here). The generator is deterministic: the same plan gives the same
bytes. Every case carries a ``purpose`` (claim | probe) and the plan items it
implements (``plan_items``) in its metadata. See
``docs/test-plans/convolution-fwd.md``.

Usage:
    generate_conv_fwd_bundles.py --bundle-dir <integration-test-bundles> [--report] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import conv_fwd_plan as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import support_sidecar  # noqa: E402

GENERATOR = "generate_conv_fwd_bundles.py"
SCALE_CONST = 0.5
# A constant tensor value must carry the type of its tensor (the backend rejects a mismatch).
VALUE_TYPE = {
    "float": "Float32Value",
    "half": "Float16Value",
    "bfloat16": "BFloat16Value",
    "double": "Float64Value",
}


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


def ph(name):
    return "${case." + name + "}"


# role -> uid, per topology
UIDS = {
    "conv": dict(y=0, w=1, x=2),
    "act": dict(y=0, out=1, w=2, x=3),
    "bias_act": dict(y=0, t1=1, out=2, bias=3, w=4, x=5),
    "scale_bias_act": dict(y=0, t1=1, t2=2, out=3, scale=4, bias=5, w=6, x=7),
    "add_relu": dict(y=0, t1=1, out=2, res=3, w=4, x=5),
    "bias_add_relu": dict(y=0, t1=1, t2=2, out=3, bias=4, res=5, w=6, x=7),
    "act_add": dict(y=0, t1=1, out=2, res=3, w=4, x=5),
}
NAMES = dict(
    y="ConvolutionFprop_0::Y", w="w", x="x", bias="bias", scale="scale", res="residual"
)
GRAPH_NAME = {"conv": "ConvolutionForwardTest"}


def _tensor(uid, name, virtual):
    return {
        "data_type": ph("data_type"),
        "dims": ph("dims"),
        "name": name,
        "strides": ph("strides"),
        "uid": uid,
        "virtual": virtual,
    }


def _conv_node(u):
    return {
        "compute_data_type": ph("attributes.conv_compute"),
        "inputs": {"w_tensor_uid": u["w"], "x_tensor_uid": u["x"]},
        "name": "ConvolutionFprop_0",
        "outputs": {"y_tensor_uid": u["y"]},
        "parameters": {
            "conv_mode": ph("attributes.conv_mode"),
            "dilation": ph("attributes.dilation"),
            "post_padding": ph("attributes.post_padding"),
            "pre_padding": ph("attributes.pre_padding"),
            "stride": ph("attributes.stride"),
        },
        "type": "ConvolutionFwdAttributes",
    }


def _pw(name, op, in0, in1, out, act_placeholders=False, in0_ph=None, in1_ph=None):
    inputs = {
        "axis_tensor_uid": None,
        "elu_alpha": None,
        "in_0_tensor_uid": in0,
        "in_1_tensor_uid": in1,
        "in_2_tensor_uid": None,
        "operation": op,
        "relu_lower_clip": None,
        "relu_lower_clip_slope": None,
        "relu_upper_clip": None,
        "softplus_beta": None,
        "swish_beta": None,
    }
    if act_placeholders:
        for key, path in (
            ("operation", "act_operation"),
            ("relu_lower_clip", "act_relu_lower_clip"),
            ("relu_lower_clip_slope", "act_relu_lower_clip_slope"),
            ("relu_upper_clip", "act_relu_upper_clip"),
            ("elu_alpha", "act_elu_alpha"),
            ("swish_beta", "act_swish_beta"),
            ("softplus_beta", "act_softplus_beta"),
        ):
            inputs[key] = ph("attributes." + path)
    if in0_ph:
        inputs["in_0_tensor_uid"] = ph("attributes." + in0_ph)
    if in1_ph:
        inputs["in_1_tensor_uid"] = ph("attributes." + in1_ph)
    return {
        "compute_data_type": "float",
        "inputs": inputs,
        "name": name,
        "outputs": {"out_0_tensor_uid": out},
        "type": "PointwiseAttributes",
    }


def template(topo):
    u = UIDS[topo]
    fused = topo != "conv"
    y_virtual = ph("attributes.conv_y_virtual") if fused else False
    nodes = [_conv_node(u)]
    inter = {}  # uid -> name for virtual intermediates
    if topo == "act":
        nodes.append(
            _pw(
                "Pointwise_1", "relu_fwd", u["y"], None, u["out"], act_placeholders=True
            )
        )
    elif topo == "bias_act":
        nodes.append(
            _pw(
                "Pointwise_1",
                "add",
                u["y"],
                u["bias"],
                u["t1"],
                in0_ph="bias_in_0",
                in1_ph="bias_in_1",
            )
        )
        nodes.append(
            _pw(
                "Pointwise_2",
                "relu_fwd",
                u["t1"],
                None,
                u["out"],
                act_placeholders=True,
            )
        )
        inter = {u["t1"]: "Pointwise_1::OUT_0"}
    elif topo == "scale_bias_act":
        nodes.append(_pw("Pointwise_1", "mul", u["y"], u["scale"], u["t1"]))
        nodes.append(_pw("Pointwise_2", "add", u["t1"], u["bias"], u["t2"]))
        nodes.append(_pw("Pointwise_3", "relu_fwd", u["t2"], None, u["out"]))
        inter = {u["t1"]: "Pointwise_1::OUT_0", u["t2"]: "Pointwise_2::OUT_0"}
    elif topo == "add_relu":
        nodes.append(_pw("Pointwise_1", "add", u["y"], u["res"], u["t1"]))
        nodes.append(_pw("Pointwise_2", "relu_fwd", u["t1"], None, u["out"]))
        inter = {u["t1"]: "Pointwise_1::OUT_0"}
    elif topo == "bias_add_relu":
        nodes.append(_pw("Pointwise_1", "add", u["y"], u["bias"], u["t1"]))
        nodes.append(_pw("Pointwise_2", "add", u["t1"], u["res"], u["t2"]))
        nodes.append(_pw("Pointwise_3", "relu_fwd", u["t2"], None, u["out"]))
        inter = {u["t1"]: "Pointwise_1::OUT_0", u["t2"]: "Pointwise_2::OUT_0"}
    elif topo == "act_add":
        nodes.append(_pw("Pointwise_1", "relu_fwd", u["y"], None, u["t1"]))
        nodes.append(_pw("Pointwise_2", "add", u["t1"], u["res"], u["out"]))
        inter = {u["t1"]: "Pointwise_1::OUT_0"}

    tensors = []
    for role, uid in u.items():
        if role == "y":
            tensors.append(_tensor(uid, NAMES["y"], y_virtual))
        elif role == "out":
            tensors.append(_tensor(uid, f"Pointwise_{len(nodes) - 1}::OUT_0", False))
        elif role in ("t1", "t2"):
            tensors.append(_tensor(uid, inter[uid], True))
        else:
            tensors.append(_tensor(uid, NAMES[role], False))
    tensors.sort(key=lambda t: t["uid"])
    return {
        "compute_data_type": "float",
        "intermediate_data_type": ph("intermediate_data_type"),
        "io_data_type": ph("io_data_type"),
        "is_override_shape_enabled": ph("override_shape"),
        "name": GRAPH_NAME.get(topo, "ConvFwdFusionTest"),
        "nodes": nodes,
        "tensors": tensors,
    }


def placeholders(node):
    """Case-wide placeholder paths (not the per-tensor dims/strides/data_type)."""
    found = set()

    def walk(v, in_tensor):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, in_tensor)
        elif isinstance(v, list):
            for x in v:
                walk(x, in_tensor)
        elif isinstance(v, str):
            m = re.fullmatch(r"\$\{case\.(.+)\}", v)
            if m and m.group(1) not in ("dims", "strides", "data_type"):
                found.add(m.group(1))

    walk(node, False)
    return found


# --------------------------------------------------------------------------
# Case -> sweep entry
# --------------------------------------------------------------------------


def _seed(case_id, uid):
    return (
        int(hashlib.sha1(f"{case_id}/{uid}".encode()).hexdigest()[:8], 16) % (2**31 - 1)
        + 1
    )


def _recipe(spec, seed):
    if spec[0] == "fixed":
        return {"kind": "fixed", "value": spec[1], "seed": seed}
    return {"kind": "free", "lo": spec[1], "hi": spec[2], "seed": seed}


def _channel_dims(c):
    return [1, c["k"]] + [1] * c["rank"]


def tensor_specs(c):
    """uid -> (dims, strides, dtype) for every tensor of the case."""
    u = UIDS[c["topo"]]
    r = c["rank"]
    xd, wd, yd = P.x_dims(c), P.w_dims(c), P.y_dims(c)
    y_strides = P.strides(yd, c["yl"], c["ye"])
    packed_y = P.strides(yd, c["yl"])
    spec = {
        u["x"]: (xd, P.strides(xd, c["xl"], c["xe"]), c["xt"]),
        u["w"]: (wd, P.strides(wd, c["wl"], c["we"]), c["wt"]),
    }
    if c["topo"] == "conv":
        spec[u["y"]] = (yd, y_strides, c["yt"])
        return spec
    spec[u["y"]] = (yd, packed_y, c["inter_t"])
    spec[u["out"]] = (yd, y_strides, c["yt"])
    for role in ("t1", "t2"):
        if role in u:
            spec[u[role]] = (yd, packed_y, c["inter_t"])
    if "bias" in u:
        dims = {"channel": _channel_dims(c), "full": yd, "scalar": [1] * (r + 2)}[
            c["bias"]
        ]
        spec[u["bias"]] = (dims, P.strides(dims, c["yl"]), c["bias_t"])
    if "scale" in u:
        dims = _channel_dims(c) if c["scale"] == "channel" else [1] * (r + 2)
        spec[u["scale"]] = (dims, P.strides(dims, c["yl"]), c["xt"])
    if "res" in u:
        spec[u["res"]] = (yd, packed_y, c["yt"])
    return spec


def input_recipes(c, case_id):
    """Fill recipes (metadata.inputs) for every device input of the case."""
    u = UIDS[c["topo"]]
    rec = P.value_recipes(c)
    if rec is None:  # fp8, int8, double: only a plain range
        a = 2.0 if c["xt"] == "int8" else 1.0
        rec = dict(x=("free", -a, a), w=("free", -a, a))
    inputs = {
        u["x"]: _recipe(rec["x"], _seed(case_id, u["x"])),
        u["w"]: _recipe(rec["w"], _seed(case_id, u["w"])),
    }
    if "bias" in u:
        inputs[u["bias"]] = _recipe(("free", -0.5, 0.5), _seed(case_id, u["bias"]))
    if "scale" in u and c["scale"] != "const":
        inputs[u["scale"]] = _recipe(("free", 0.5, 1.5), _seed(case_id, u["scale"]))
    if "res" in u:
        inputs[u["res"]] = _recipe(("free", -1.0, 1.0), _seed(case_id, u["res"]))
    return {str(k): v for k, v in sorted(inputs.items())}


def case_values(c):
    """The ``values`` block and tensor patches of a sweep case."""
    topo = c["topo"]
    u = UIDS[topo]
    specs = tensor_specs(c)
    attrs = {
        "conv_compute": c["comp"],
        "conv_mode": c["conv_mode"],
        "dilation": list(c["d"]),
        "post_padding": list(c["post"]),
        "pre_padding": list(c["pre"]),
        "stride": list(c["s"]),
    }
    if topo != "conv":
        attrs["conv_y_virtual"] = c["conv_y_virtual"]
    if topo in P.HAS_ACT_PLACEHOLDERS:
        op, lo, slope, hi, alpha, sbeta, pbeta = P.ACTS[c["act"]]
        attrs.update(
            act_operation=op,
            act_relu_lower_clip=lo,
            act_relu_lower_clip_slope=slope,
            act_relu_upper_clip=hi,
            act_elu_alpha=alpha,
            act_swish_beta=sbeta,
            act_softplus_beta=pbeta,
        )
    if topo == "bias_act":
        y, b = u["y"], u["bias"]
        attrs.update(
            bias_in_0=b if c["bias_first"] else y, bias_in_1=y if c["bias_first"] else b
        )
    values = {
        "io_data_type": c["xt"],
        "intermediate_data_type": c["inter_t"],
        "override_shape": c["override"],
        "attributes": attrs,
        "tensors": [
            {"uid": uid, "dims": d, "strides": s, "data_type": t}
            for uid, (d, s, t) in sorted(specs.items())
        ],
    }
    patches = []
    if topo in P.HAS_SCALE:
        if c["scale"] == "const":
            patches.append(
                {
                    "uid": u["scale"],
                    "set": {"value": SCALE_CONST, "value_type": VALUE_TYPE[c["xt"]]},
                }
            )
        elif c["scale"] == "runtime":
            patches.append(
                {"uid": u["scale"], "set": {"is_runtime_pass_by_value": True}}
            )
    return values, patches


def _tok(v):
    return "x".join(str(i) for i in v)


def case_id(c):
    r = c["rank"]
    parts = ["_".join(c["ids"])] if c["ids"] else []
    parts.append(f"{r}d")
    parts.append("i" + _tok(c["ins"]))
    parts.append(f"n{c['n']}c{c['c']}k{c['k']}" + (f"g{c['g']}" if c["g"] > 1 else ""))
    parts.append("f" + _tok(c["f"]))
    dt = P.SHORT[c["xt"]]
    if (c["wt"], c["yt"]) != (c["xt"], c["xt"]):
        dt += f"_w{P.SHORT[c['wt']]}_y{P.SHORT[c['yt']]}"
    parts.append(dt)
    if c["xl"] == c["wl"] == c["yl"]:
        parts.append(c["xl"])
    else:
        parts.append(f"x{c['xl']}w{c['wl']}y{c['yl']}")
    if c["topo"] in P.HAS_ACT_PLACEHOLDERS:
        parts.append(c["act"])
    return "_".join(p for p in parts if p)


def sweep_case(c, cid):
    values, patches = case_values(c)
    probe = P.probe_reasons(c)
    meta = {
        "format_version": 1,
        "generator": GENERATOR,
        "generator_version": "1",
        "reference_source": "hipDNN reference executors, auto verification (GPU reference, then CPU)",
        "operation": "ConvolutionFwd"
        + ("" if c["topo"] == "conv" else f"+{c['topo']}"),
        "plan_items": list(c["ids"]),
        "purpose": "probe" if probe else "claim",
        "size_class": c["size"],
        "seed": _seed(cid, "case"),
        "inputs": input_recipes(c, cid),
    }
    if probe:
        meta["notes"] = "probe: " + "; ".join(probe)
    total = sum(P.numel(d) * P.BYTES.get(t, 1) for d, _, t in tensor_specs(c).values())
    if total > (1 << 30):
        meta["minimum_vram_mb"] = int(total * 2 / (1 << 20)) + 1
    entry = {"id": cid, "values": values, "metadata": meta}
    if patches:
        entry["tensor_patches"] = patches
    return entry


# --------------------------------------------------------------------------
# Case builders and variants
# --------------------------------------------------------------------------


def with_(c, add_ids=(), **kw):
    d = dict(c)
    d.update(kw)
    d["ids"] = tuple(c["ids"]) + tuple(add_ids)
    return P.finalize(d)


def fuse(c, topo, **kw):
    return with_(c, topo=topo, inter_t="float", **kw)


def v_convmode(c):
    return with_(c, ("F13",), conv_mode="CONVOLUTION")


def v_asym(c):
    if c["pre"] != c["post"]:
        return None
    return with_(c, ("F11",), post=tuple(p + 1 for p in c["pre"]))


def v_layouts(c, combo):
    return with_(c, ("F15",), xl=combo[0], wl=combo[1], yl=combo[2])


MIXES = [
    ("cl", "cf", "cf"),
    ("cf", "cl", "cf"),
    ("cf", "cf", "cl"),
    ("cl", "cl", "cf"),
    ("cl", "cf", "cl"),
    ("cf", "cl", "cl"),
]


def v_nonpacked(c, kind):
    r = c["rank"]

    def extra(layout, tensor_rank):
        e = [0] * (tensor_rank + 2)
        if kind == "fast":
            e[P.fastest_dim(layout, tensor_rank)] = 4
        elif kind == "chan":
            e[1] = 8
        else:  # plane: pad the second-fastest dimension
            e[r + 1 if layout == "cl" else (r if r >= 2 else 1)] = 3
        return tuple(e)

    return with_(c, (f"F16{kind}",), xe=extra(c["xl"], r), ye=extra(c["yl"], r))


def v_type(c, name):
    xt, wt, yt, comp = P.EXTRA_TYPES.get(name) or P.MIXED_TYPES[name]
    tag = "F19" if name in P.EXTRA_TYPES else "F20"
    return with_(c, (tag,), xt=xt, wt=wt, yt=yt, comp=comp, inter_t=yt, bias_t=xt)


def v_vals(c, cls):
    if c["xt"] not in P.DTYPE_MAX:
        return None
    return with_(c, ("F21" + cls,), vals=cls)


def v_override(c):
    return with_(c, ("F35",), override=True)


def v_act(c, name):
    return with_(c, (), act=name)


def v_bias(c, kind):
    if kind == "first":
        return with_(c, ("F25first",), bias_first=True)
    if kind == "type":
        other = "float" if c["xt"] != "float" else "half"
        return with_(c, ("F25type",), bias_t=other)
    return with_(c, (f"F25{kind}",), bias=kind)


def v_scale(c, kind):
    return with_(c, (f"F27{kind}",), scale=kind)


def v_materialised(c):
    return with_(c, ("F30",), conv_y_virtual=False)


def v_inter_io(c):
    return with_(c, ("F31",), inter_t=c["yt"])


# --------------------------------------------------------------------------
# Level rules
# --------------------------------------------------------------------------


class Builder:
    def __init__(self, level_index):
        self.lv = level_index
        self.level = P.LEVELS[level_index]
        self.cases = []

    def add(self, c):
        if c is not None:
            self.cases.append(c)

    # ---- shape constructors -------------------------------------------------
    def S(self, sid, r, pair, **kw):
        kwargs = P.shape_S(sid, r, self.level)
        if kwargs is None:
            return None
        return self._make(kwargs, pair, (sid,), "S", **kw)

    def Ls(self, sid, r, pair, **kw):
        kwargs = P.shape_L(sid, r, "comprehensive")
        if kwargs is None:
            return None
        return self._make(kwargs, pair, ("L" + sid[1:],), "L", **kw)

    def M(self, mid, pair, **kw):
        return self._make(dict(P.M_SHAPES[mid]), pair, (mid,), "M", **kw)

    def H(self, hid, r, pair, **kw):
        kwargs = P.shape_H(hid, r, self.level)
        if kwargs is None:
            return None
        return self._make(kwargs, pair, (hid,), "H", **kw)

    def X(self, xid, pair, **kw):
        return self._make(dict(P.X_SHAPES[xid]), pair, (xid,), "X", **kw)

    def A(self, aid, pair, **kw):
        return self._make(dict(P.ACCUM_SHAPES[aid]), pair, (aid,), "A", **kw)

    def _make(self, kwargs, pair, ids, size, **kw):
        lay, dt = pair
        return P.new(**kwargs, xl=lay, wl=lay, yl=lay, xt=dt, ids=ids, size=size, **kw)

    def sids(self, r, candidates=P.S_IDS):
        return [s for s in candidates if P.shape_S(s, r, self.level) is not None]

    def lids(self, r):
        return [s for s in P.S_IDS if P.shape_L(s, r) is not None]


def rules(lv):
    b = Builder(lv)
    add = b.add
    P1 = P.PAIRS[0]  # (cf, fp32)
    acts = list(P.ACTS)

    if lv == 0:  # quick: what must we test to catch gross breakage fast?
        for r in P.RANKS:
            for i, s in enumerate(b.sids(r, P.S_QUICK)):
                add(b.S(s, r, P.PAIRS[(i + r - 1) % 6]))
        add(v_convmode(b.S("S3", 2, P1)))
        add(b.S("S18", 2, P1))
        add(v_type(b.S("S3", 2, P1), "fp8_e4m3"))
        add(v_type(b.S("S3", 2, P1), "double"))
        for ti, topo in enumerate(P.FUSED):
            for r in P.RANKS:
                sids = b.sids(r, P.S_QUICK)
                for j, pair in enumerate(P.PAIRS):
                    k = j + ti + r
                    base = b.S(sids[k % len(sids)], r, pair)
                    act = (
                        P.RELU_FAMILY[k % 5]
                        if topo in P.HAS_ACT_PLACEHOLDERS
                        else "relu"
                    )
                    add(fuse(base, topo, act=act))
        add(fuse(b.S("S3", 2, P1), "act", act="sigmoid"))

    elif lv == 1:  # PR: what must we test on every PR?
        for r in (1, 2, 3, 4):
            for i, s in enumerate(b.sids(r)):
                add(b.S(s, r, P.PAIRS[(i + r + 3) % 6]))
        S3 = lambda pair=P1, r=2: b.S("S3", r, pair)
        for r in P.RANKS:
            add(v_convmode(S3(P1, r)))
        add(v_layouts(S3(P.PAIRS[3]), MIXES[0]))
        add(v_nonpacked(S3(), "fast"))
        for name in list(P.EXTRA_TYPES) + list(P.MIXED_TYPES):
            add(v_type(S3(), name))
        add(v_vals(S3(), "zero"))
        add(v_vals(S3(), "const"))
        add(v_vals(S3(P.PAIRS[2]), "large"))
        add(v_override(S3()))
        for i, h in enumerate(P.H_SHAPES):
            add(b.H(h, 2, P.PAIRS[i % 6]))
        for i, m in enumerate(P.M_PR):
            add(b.M(m, P.PAIRS[(2 * i + 2) % 6]))
        for ti, topo in enumerate(P.FUSED):
            rounds = 2 if topo in P.HAS_ACT_PLACEHOLDERS else 1
            for rd in range(rounds):
                for r in P.RANKS:
                    sids = b.sids(r)
                    for j, pair in enumerate(P.PAIRS):
                        k = j + 3 * rd + 2 * ti + r + 4
                        act = (
                            acts[k % len(acts)]
                            if topo in P.HAS_ACT_PLACEHOLDERS
                            else "relu"
                        )
                        add(fuse(b.S(sids[k % len(sids)], r, pair), topo, act=act))
        base = S3()
        for name in acts:
            add(fuse(base, "act", act=name))
        bb = fuse(S3(P.PAIRS[2]), "bias_act", act="relu")
        for kind in ("full", "scalar", "type", "first"):
            add(v_bias(bb, kind))
        sc = fuse(base, "scale_bias_act")
        for kind in ("const", "runtime", "channel"):
            add(v_scale(sc, kind))
        add(v_materialised(fuse(base, "act", act="relu")))
        add(v_inter_io(fuse(S3(P.PAIRS[2]), "act", act="relu")))

    elif lv == 2:  # nightly: what should we test each day, wider per engine?
        for r in (1, 2, 3, 4):
            for s in b.sids(r):
                for pair in P.PAIRS:
                    add(b.S(s, r, pair))
        for r in P.RANKS:
            for i, s in enumerate(b.lids(r)):
                for pair in (P.PAIRS[(2 * i) % 6], P.PAIRS[(2 * i + 1) % 6]):
                    add(b.Ls(s, r, pair))
        for m in P.M_SHAPES:
            for pair in P.PAIRS:
                add(b.M(m, pair))
        for r in P.RANKS:
            for pair in P.PAIRS:
                add(v_convmode(b.S("S3", r, pair)))
            for di, dt in enumerate(P.CORE_DTYPES):
                for j in range(2):
                    add(
                        v_layouts(b.S("S3", r, ("cf", dt)), MIXES[(2 * di + j + r) % 6])
                    )
                for ki, kind in enumerate(("fast", "chan", "plane")):
                    add(v_nonpacked(b.S("S3", r, (P.LAYOUTS[(ki + di) % 2], dt)), kind))
                for cls in ("zero", "const", "large", "tiny"):
                    add(v_vals(b.S("S3", r, ("cf", dt)), cls))
                add(v_override(b.S("S3", r, ("cf", dt))))
            for pair in P.PAIRS:
                for name in list(P.EXTRA_TYPES) + list(P.MIXED_TYPES):
                    add(v_type(b.S("S3", r, pair), name))
        for aid in P.ACCUM_SHAPES:
            for pair in P.PAIRS:
                add(b.A(aid, pair))
        for h in P.H_SHAPES:
            for pair in P.PAIRS:
                add(b.H(h, 2, pair))
            for r in (1, 3):
                for pair in (P.PAIRS[0], P.PAIRS[3]):
                    add(b.H(h, r, pair))
        # fused
        n = 0
        for topo in ("act", "bias_act"):
            for r in (1, 2, 3, 4):
                for s in b.sids(r):
                    for pair in P.PAIRS:
                        add(fuse(b.S(s, r, pair), topo, act=acts[n % len(acts)]))
                        n += 1
            for r in P.RANKS:
                for i, s in enumerate(b.lids(r)):
                    add(
                        fuse(
                            b.Ls(s, r, P.PAIRS[(i + r) % 6]),
                            topo,
                            act=acts[n % len(acts)],
                        )
                    )
                    n += 1
            for m in ("M2", "M3", "M6", "M9"):
                for pair in P.PAIRS:
                    add(fuse(b.M(m, pair), topo, act=acts[n % len(acts)]))
                    n += 1
        for ti, topo in enumerate(
            ("scale_bias_act", "add_relu", "bias_add_relu", "act_add")
        ):
            for r in P.RANKS:
                sids = b.sids(r)
                for j, pair in enumerate(P.PAIRS):
                    add(
                        fuse(b.S(sids[(j * 3 + ti + r + 7) % len(sids)], r, pair), topo)
                    )
        for r in P.RANKS:
            pr = P.PAIRS[3]
            bb = fuse(b.S("S3", r, pr), "bias_act", act="relu")
            for kind in ("full", "scalar", "type", "first"):
                add(v_bias(bb, kind))
            sc = fuse(b.S("S3", r, pr), "scale_bias_act")
            for kind in ("const", "runtime", "channel"):
                add(v_scale(sc, kind))
            add(v_materialised(fuse(b.S("S3", r, pr), "act", act="relu")))
            add(v_inter_io(fuse(b.S("S3", r, pr), "act", act="relu")))

    else:  # full: which wider sweeps must exist in the full suite?
        for r in P.RANKS:
            for i, s in enumerate(b.lids(r)):
                for pair in P.PAIRS:
                    add(b.Ls(s, r, pair))
        for xid in ("X1", "X2", "X4"):
            for pair in P.PAIRS:
                add(b.X(xid, pair))
        add(b.X("X3", ("cf", "half")))
        add(b.X("X3", ("cl", "bfloat16")))
        # each feature paired with every S shape once, values rotated
        extra = list(P.EXTRA_TYPES)
        mixed = list(P.MIXED_TYPES)
        for r in (1, 2, 3, 4):
            for i, s in enumerate(b.sids(r)):

                def base(off=0):
                    return b.S(s, r, P.PAIRS[(i + off) % 6])

                add(v_convmode(base(0)))
                add(v_asym(base(1)))
                add(v_type(base(2), extra[i % len(extra)]))
                add(v_type(base(3), mixed[i % len(mixed)]))
                add(v_layouts(base(4), MIXES[i % 6]))
                add(v_nonpacked(base(5), ("fast", "chan", "plane")[i % 3]))
                add(v_vals(base(0), ("zero", "const", "large", "tiny")[i % 4]))
                add(v_override(base(1)))
        # core x feature: values and activations, complete
        for r in P.RANKS:
            for pair in P.PAIRS:
                for cls in ("zero", "const", "large", "tiny"):
                    add(v_vals(b.S("S2", r, pair), cls))
                for topo in ("act", "bias_act"):
                    for name in acts:
                        add(fuse(b.S("S3", r, pair), topo, act=name))
        # every activation x every S / L shape, act topology
        for ai, name in enumerate(acts):
            for r in P.RANKS:
                for i, s in enumerate(b.sids(r)):
                    add(fuse(b.S(s, r, P.PAIRS[(i + ai) % 6]), "act", act=name))
                for i, s in enumerate(b.lids(r)):
                    add(fuse(b.Ls(s, r, P.PAIRS[(i + ai) % 6]), "act", act=name))
        # the other topologies, each over every S shape once
        for ti, topo in enumerate(
            ("scale_bias_act", "add_relu", "bias_add_relu", "act_add")
        ):
            for r in P.RANKS:
                for i, s in enumerate(b.sids(r)):
                    add(fuse(b.S(s, r, P.PAIRS[(i + ti) % 6]), topo))
    return b.cases


def build():
    """All levels, cumulative-deduplicated: a graph appears in the lowest level
    that holds it. Returns {level: [(case, id)]} and the dropped counts."""
    seen = set()
    result = {}
    dropped = Counter()
    for lv, level in enumerate(P.LEVELS):
        out, ids = [], Counter()
        for c in rules(lv):
            reason = P.legal(c)
            if reason:
                dropped[f"{level}: {reason}"] += 1
                continue
            exempt = (
                c["size"] in ("X",)
                or any(i in P.M_PR for i in c["ids"])
                and level == "standard"
            )
            if not exempt and not P.fits(c, level) and c["size"] != "L":
                dropped[f"{level}: over ceiling"] += 1
                continue
            key = P.canon(c)
            if key in seen:
                continue
            seen.add(key)
            cid = case_id(c)
            ids[cid] += 1
            out.append([c, cid])
        # an id that collides inside one sweep gets a hash suffix
        counts = Counter()
        for c, cid in out:
            counts[(c["topo"], cid)] += 1
        for item in out:
            c, cid = item
            if counts[(c["topo"], cid)] > 1:
                item[1] = f"{cid}_{P.short_hash(c)}"
        result[level] = out
    return result, dropped


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def dump(obj):
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def support_json(path, case_ids):
    """The sidecar of a sweep. Claims are harvested, never written here: an
    existing sidecar keeps the claims of cases that still exist, so regenerating
    the bundles does not erase a harvest. A sweep with no sidecar gets an empty one."""
    claims = (
        {c for c in support_sidecar.load(path) if c.case in case_ids}
        if path.exists()
        else set()
    )
    return (
        support_sidecar.render(path, claims)
        if claims
        else json.dumps({"claims": {}, "version": 1}, indent=2, sort_keys=True) + "\n"
    )


def write_tree(result, bundle_dir, dry_run):
    files = {}
    for level, items in result.items():
        by_topo = defaultdict(list)
        for c, cid in items:
            by_topo[c["topo"]].append((c, cid))
        for topo, cs in by_topo.items():
            op, name = P.BUNDLE_DIR[topo]
            rel = Path(level) / op / name
            tpl = template(topo)
            need = placeholders(tpl)
            entries = []
            for c, cid in cs:
                e = sweep_case(c, cid)
                have = set()

                def flat(prefix, v):
                    if isinstance(v, dict):
                        for k, x in v.items():
                            flat(prefix + k + ".", x)
                    else:
                        have.add(prefix[:-1])

                for k, v in e["values"].items():
                    if k != "tensors":
                        flat(k + ".", v)
                if have != need:
                    raise SystemExit(
                        f"{rel}/{cid}: value keys {sorted(have ^ need)} do not match the template placeholders"
                    )
                entries.append(e)
            files[rel / "graph.template.json"] = dump(tpl)
            files[rel / "sweep.json"] = dump({"version": 1, "cases": entries})
            files[rel / "support.json"] = support_json(
                bundle_dir / rel / "support.json", {cid for _, cid in cs}
            )
    if dry_run:
        return files
    for level in P.LEVELS:
        for op in ("ConvolutionFwd", "ConvolutionFwdPointwise"):
            target = bundle_dir / level / op
            if target.exists():
                shutil.rmtree(target)
    for rel, text in files.items():
        path = bundle_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return files


def report(result, dropped):
    print("\nCases by level and topology (own level; cumulative in the last column)")
    total = 0
    hdr = ["level"] + list(P.TOPOLOGIES) + ["own", "cumulative"]
    print("  ".join(f"{h:>14}" for h in hdr))
    for level, items in result.items():
        cnt = Counter(c["topo"] for c, _ in items)
        total += len(items)
        print(
            "  ".join(
                f"{v:>14}"
                for v in [level] + [cnt[t] for t in P.TOPOLOGIES] + [len(items), total]
            )
        )
    print("\nBy purpose and size class")
    for level, items in result.items():
        pur = Counter(P.purpose(c) for c, _ in items)
        sz = Counter(c["size"] for c, _ in items)
        print(f"  {level:<14} {dict(pur)}  {dict(sorted(sz.items()))}")
    geo = set()
    for items in result.values():
        for c, _ in items:
            geo.add(
                (
                    c["rank"],
                    c["n"],
                    c["c"],
                    c["k"],
                    c["g"],
                    c["ins"],
                    c["f"],
                    c["s"],
                    c["d"],
                    c["pre"],
                    c["post"],
                )
            )
    print(f"\nDistinct conv geometries: {len(geo)}")
    print("\nFLOP per case (fp32 ceiling check) max by level")
    for level, items in result.items():
        if items:
            print(
                f"  {level:<14} max {max(P.flops(c) for c, _ in items) / 1e9:.3f} GFLOP, "
                f"max {max(P.fp32_bytes(c) for c, _ in items) / (1 << 20):.1f} MiB"
            )
    if dropped:
        print("\nDropped:")
        for k, v in sorted(dropped.items()):
            print(f"  {k}: {v}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    result, dropped = build()
    write_tree(result, args.bundle_dir, args.dry_run)
    if args.report or args.dry_run:
        report(result, dropped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
