#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the convolution backward-data (dgrad) bundles from the test plan.

Reads the declarations in ``conv_bwd_plan.py`` and the level rules below, and
writes, for each level (quick, standard, comprehensive, full), one template
sweep per topology under ``integration-test-bundles/<level>/``:

    ConvolutionBwd/Default                  dgrad
    ConvolutionBwdPointwise/Default         dgrad -> act_bwd(., x)
    ConvolutionBwdPointwise/Variant2        act_bwd(dy, y) -> dgrad
    ConvolutionBwdPointwise/Variant3        dgrad -> add(dx_residual)
    ConvolutionBwdPointwise/Variant4        dgrad -> add(dx_residual) -> relu_bwd(., x)

Each directory gets ``graph.template.json``, ``sweep.json`` and a
``support.json`` (claims are harvested with ``--write-support-claims``, never
written here; an existing sidecar keeps the claims of cases that still exist).
The generator is deterministic. Every case carries a ``purpose`` (claim |
probe) and the plan items it implements (``plan_items``). The level rules build
the complete products; the trims declared in ``conv_bwd_plan.TRIMS`` are
removed after case ids are assigned, so a trim never renames a surviving case.
See ``docs/test-plans/convolution-bwd.md``.

Usage:
    generate_conv_bwd_bundles.py --bundle-dir <integration-test-bundles> [--report] [--dry-run]
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

import conv_bwd_plan as P

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import support_sidecar  # noqa: E402

GENERATOR = "generate_conv_bwd_bundles.py"
BUNDLE_OPS = ("ConvolutionBwd", "ConvolutionBwdPointwise")


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


def ph(name):
    return "${case." + name + "}"


# role -> uid, per topology. dx is the dgrad output; in the fused epilogues it is
# virtual and ``out`` is the graph output. In pre_dact the virtual tensor is the
# activation gradient ``dyv`` that feeds the dgrad.
UIDS = {
    "dgrad": dict(dx=0, dy=1, w=2),
    "dact": dict(out=0, dx=1, dy=2, w=3, x=4),
    "pre_dact": dict(dx=0, dyv=1, dy=2, y=3, w=4),
    "add": dict(out=0, dx=1, res=2, dy=3, w=4),
    "add_drelu": dict(out=0, dx=1, t1=2, res=3, x=4, dy=5, w=6),
}
NAMES = dict(
    dx="ConvolutionDgrad_0::DX", dy="dy", w="w", x="x", y="y", res="dx_residual"
)
GRAPH_NAME = {"dgrad": "ConvolutionBackwardDataTest"}


def _tensor(uid, name, virtual):
    return {
        "data_type": ph("data_type"),
        "dims": ph("dims"),
        "name": name,
        "strides": ph("strides"),
        "uid": uid,
        "virtual": virtual,
    }


def _dgrad_node(dy_uid, w_uid, dx_uid):
    return {
        "compute_data_type": ph("attributes.conv_compute"),
        "inputs": {"dy_tensor_uid": dy_uid, "w_tensor_uid": w_uid},
        "name": "ConvolutionDgrad_0",
        "outputs": {"dx_tensor_uid": dx_uid},
        "parameters": {
            "conv_mode": ph("attributes.conv_mode"),
            "dilation": ph("attributes.dilation"),
            "post_padding": ph("attributes.post_padding"),
            "pre_padding": ph("attributes.pre_padding"),
            "stride": ph("attributes.stride"),
        },
        "type": "ConvolutionBwdAttributes",
    }


def _pw(name, op, in0, in1, out, act_placeholders=False):
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
    return {
        "compute_data_type": "float",
        "inputs": inputs,
        "name": name,
        "outputs": {"out_0_tensor_uid": out},
        "type": "PointwiseAttributes",
    }


def template(topo):
    u = UIDS[topo]
    inter_virtual = ph("attributes.inter_virtual")
    names = dict(NAMES)
    virtual = {}
    if topo == "dgrad":
        nodes = [_dgrad_node(u["dy"], u["w"], u["dx"])]
    elif topo == "dact":
        nodes = [
            _dgrad_node(u["dy"], u["w"], u["dx"]),
            _pw("Pointwise_1", "relu_bwd", u["dx"], u["x"], u["out"], True),
        ]
        virtual[u["dx"]] = inter_virtual
    elif topo == "pre_dact":
        nodes = [
            _pw("Pointwise_1", "relu_bwd", u["dy"], u["y"], u["dyv"], True),
            _dgrad_node(u["dyv"], u["w"], u["dx"]),
        ]
        names["dyv"] = "Pointwise_1::OUT_0"
        virtual[u["dyv"]] = inter_virtual
    elif topo == "add":
        nodes = [
            _dgrad_node(u["dy"], u["w"], u["dx"]),
            _pw("Pointwise_1", "add", u["dx"], u["res"], u["out"]),
        ]
        virtual[u["dx"]] = inter_virtual
    elif topo == "add_drelu":
        nodes = [
            _dgrad_node(u["dy"], u["w"], u["dx"]),
            _pw("Pointwise_1", "add", u["dx"], u["res"], u["t1"]),
            _pw("Pointwise_2", "relu_bwd", u["t1"], u["x"], u["out"]),
        ]
        names["t1"] = "Pointwise_1::OUT_0"
        virtual[u["dx"]] = inter_virtual
        virtual[u["t1"]] = True
    else:
        raise KeyError(topo)
    if "out" in u:
        names["out"] = f"Pointwise_{len(nodes) - 1}::OUT_0"
    tensors = [
        _tensor(uid, names[role], virtual.get(uid, False)) for role, uid in u.items()
    ]
    tensors.sort(key=lambda t: t["uid"])
    return {
        "compute_data_type": "float",
        "intermediate_data_type": ph("intermediate_data_type"),
        "io_data_type": ph("io_data_type"),
        "is_override_shape_enabled": ph("override_shape"),
        "name": GRAPH_NAME.get(topo, "ConvBwdDataFusionTest"),
        "nodes": nodes,
        "tensors": tensors,
    }


def placeholders(node):
    """Case-wide placeholder paths (not the per-tensor dims/strides/data_type)."""
    found = set()

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str):
            m = re.fullmatch(r"\$\{case\.(.+)\}", v)
            if m and m.group(1) not in ("dims", "strides", "data_type"):
                found.add(m.group(1))

    walk(node)
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


def tensor_specs(c):
    """uid -> (dims, strides, dtype) for every tensor of the case."""
    u = UIDS[c["topo"]]
    dxd, wd, dyd = P.dx_dims(c), P.w_dims(c), P.dy_dims(c)
    spec = {u["w"]: (wd, P.strides(wd, c["wl"], c["we"]), c["wt"])}
    topo = c["topo"]
    if topo == "dgrad":
        spec[u["dy"]] = (dyd, P.strides(dyd, c["dyl"], c["dye"]), c["dyt"])
        spec[u["dx"]] = (dxd, P.strides(dxd, c["dxl"], c["dxe"]), c["dxt"])
        return spec
    if topo == "pre_dact":
        spec[u["dy"]] = (dyd, P.strides(dyd, c["dyl"], c["dye"]), c["dyt"])
        spec[u["y"]] = (dyd, P.strides(dyd, c["dyl"]), c["dyt"])
        spec[u["dyv"]] = (dyd, P.strides(dyd, c["dyl"]), c["inter_t"])
        spec[u["dx"]] = (dxd, P.strides(dxd, c["dxl"], c["dxe"]), c["dxt"])
        return spec
    packed_dx = P.strides(dxd, c["dxl"])
    spec[u["dy"]] = (dyd, P.strides(dyd, c["dyl"], c["dye"]), c["dyt"])
    spec[u["dx"]] = (dxd, packed_dx, c["inter_t"])
    spec[u["out"]] = (dxd, P.strides(dxd, c["dxl"], c["dxe"]), c["dxt"])
    if "t1" in u:
        spec[u["t1"]] = (dxd, packed_dx, c["inter_t"])
    if "res" in u:
        spec[u["res"]] = (dxd, packed_dx, c["dxt"])
    if "x" in u:
        spec[u["x"]] = (dxd, packed_dx, c["dyt"])
    return spec


def input_recipes(c, case_id):
    """Fill recipes (metadata.inputs) for every device input of the case."""
    u = UIDS[c["topo"]]
    rec = P.value_recipes(c)
    if rec is None:  # fp8, int8, double: only a plain range
        a = 2.0 if c["dyt"] == "int8" else 1.0
        rec = dict(dy=("free", -a, a), w=("free", -a, a))
    inputs = {
        u["dy"]: _recipe(rec["dy"], _seed(case_id, u["dy"])),
        u["w"]: _recipe(rec["w"], _seed(case_id, u["w"])),
    }
    for role in ("x", "y", "res"):
        if role in u:
            inputs[u[role]] = _recipe(("free", -1.0, 1.0), _seed(case_id, u[role]))
    return {str(k): v for k, v in sorted(inputs.items())}


def case_values(c):
    """The ``values`` block of a sweep case."""
    topo = c["topo"]
    attrs = {
        "conv_compute": c["comp"],
        "conv_mode": c["conv_mode"],
        "dilation": list(c["d"]),
        "post_padding": list(c["post"]),
        "pre_padding": list(c["pre"]),
        "stride": list(c["s"]),
    }
    if topo != "dgrad":
        attrs["inter_virtual"] = c["inter_virtual"]
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
    return {
        "io_data_type": c["dyt"],
        "intermediate_data_type": c["inter_t"],
        "override_shape": c["override"],
        "attributes": attrs,
        "tensors": [
            {"uid": uid, "dims": d, "strides": s, "data_type": t}
            for uid, (d, s, t) in sorted(tensor_specs(c).items())
        ],
    }


def _tok(v):
    return "x".join(str(i) for i in v)


def case_id(c):
    r = c["rank"]
    parts = ["_".join(c["ids"])] if c["ids"] else []
    parts.append(f"{r}d")
    parts.append("i" + _tok(c["ins"]))
    parts.append(f"n{c['n']}c{c['c']}k{c['k']}" + (f"g{c['g']}" if c["g"] > 1 else ""))
    parts.append("f" + _tok(c["f"]))
    dt = P.SHORT[c["dyt"]]
    if (c["wt"], c["dxt"]) != (c["dyt"], c["dyt"]):
        dt += f"_w{P.SHORT[c['wt']]}_dx{P.SHORT[c['dxt']]}"
    parts.append(dt)
    if c["dyl"] == c["wl"] == c["dxl"]:
        parts.append(c["dxl"])
    else:
        parts.append(f"dy{c['dyl']}w{c['wl']}dx{c['dxl']}")
    if c["topo"] in P.HAS_ACT_PLACEHOLDERS:
        parts.append(c["act"])
    return "_".join(p for p in parts if p)


def sweep_case(c, cid):
    probe = P.probe_reasons(c)
    meta = {
        "format_version": 1,
        "generator": GENERATOR,
        "generator_version": "1",
        "reference_source": "hipDNN reference executors, auto verification (GPU reference, then CPU)",
        "operation": "ConvolutionBwd"
        + ("" if c["topo"] == "dgrad" else f"+{c['topo']}"),
        "plan_items": list(c["ids"]),
        "purpose": "probe" if probe else "claim",
        "size_class": c["size"],
        "seed": _seed(cid, "case"),
        "inputs": input_recipes(c, cid),
    }
    notes = []
    if probe:
        notes.append("probe: " + "; ".join(probe))
    if c.get("src"):
        notes.append(c["src"])
    if notes:
        meta["notes"] = "; ".join(notes)
    total = sum(P.numel(d) * P.BYTES.get(t, 1) for d, _, t in tensor_specs(c).values())
    if total > (1 << 30):
        meta["minimum_vram_mb"] = int(total * 2 / (1 << 20)) + 1
    return {"id": cid, "values": case_values(c), "metadata": meta}


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
    return with_(c, ("F15",), dyl=combo[0], wl=combo[1], dxl=combo[2])


# (dy, w, dx) layouts: the six non-uniform combinations
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

    def extra(layout):
        e = [0] * (r + 2)
        if kind == "fast":
            e[P.fastest_dim(layout, r)] = 4
        elif kind == "chan":
            e[1] = 8
        else:  # plane: pad the second-fastest dimension
            e[r + 1 if layout == "cl" else (r if r >= 2 else 1)] = 3
        return tuple(e)

    return with_(c, (f"F16{kind}",), dye=extra(c["dyl"]), dxe=extra(c["dxl"]))


def v_type(c, name):
    dyt, wt, dxt, comp = P.EXTRA_TYPES.get(name) or P.MIXED_TYPES[name]
    tag = "F19" if name in P.EXTRA_TYPES else "F20"
    return with_(c, (tag,), dyt=dyt, wt=wt, dxt=dxt, comp=comp, inter_t=dxt)


def v_vals(c, cls):
    if c["dyt"] not in P.DTYPE_MAX:
        return None
    return with_(c, ("F21" + cls,), vals=cls)


def v_override(c):
    return with_(c, ("F35",), override=True)


def v_materialised(c):
    return with_(c, ("F30",), inter_virtual=False)


def v_inter_io(c):
    return with_(c, ("F31",), inter_t=c["dxt"])


# --------------------------------------------------------------------------
# Level rules
# --------------------------------------------------------------------------


class Builder:
    def __init__(self, level_index):
        self.lv = level_index
        self.level = P.LEVELS[level_index]

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

    def R(self, rid):
        return P.new(**P.shape_R(rid), ids=(rid,), size="R", src=P.r_source(rid))

    def _make(self, kwargs, pair, ids, size, **kw):
        lay, dt = pair
        return P.new(
            **kwargs, dyl=lay, wl=lay, dxl=lay, dyt=dt, ids=ids, size=size, **kw
        )

    def sids(self, r, candidates=P.S_IDS):
        return [s for s in candidates if P.shape_S(s, r, self.level) is not None]

    def lids(self, r, candidates=P.S_IDS):
        return [s for s in candidates if P.shape_L(s, r) is not None]


def rules(lv):
    b = Builder(lv)
    cases = []
    add = cases.append
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
                    act = (
                        P.RELU_FAMILY[k % len(P.RELU_FAMILY)]
                        if topo in P.HAS_ACT_PLACEHOLDERS
                        else "relu"
                    )
                    add(fuse(b.S(sids[k % len(sids)], r, pair), topo, act=act))
        add(fuse(b.S("S3", 2, P1), "dact", act="sigmoid"))

    elif lv == 1:  # PR: what must we test on every PR?
        for r in (1, 2, 3, 4):
            for i, s in enumerate(b.sids(r)):
                add(b.S(s, r, P.PAIRS[(i + r + 3) % 6]))

        def S3(pair=P1, r=2):
            return b.S("S3", r, pair)

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
                    sids = b.sids(r, P.S_CROSSED)
                    for j, pair in enumerate(P.PAIRS):
                        k = j + 3 * rd + 2 * ti + r + 4
                        act = (
                            acts[k % len(acts)]
                            if topo in P.HAS_ACT_PLACEHOLDERS
                            else "relu"
                        )
                        add(fuse(b.S(sids[k % len(sids)], r, pair), topo, act=act))
        for topo in P.HAS_ACT_PLACEHOLDERS:
            for name in acts:
                add(fuse(S3(), topo, act=name))
        add(v_materialised(fuse(S3(), "dact", act="relu")))
        add(v_inter_io(fuse(S3(P.PAIRS[2]), "dact", act="relu")))
        add(v_materialised(fuse(S3(), "pre_dact", act="relu")))
        add(v_inter_io(fuse(S3(P.PAIRS[2]), "pre_dact", act="relu")))
        for rid, e in P.R_SHAPES.items():
            if e["level"] == "standard":
                add(b.R(rid))

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
        n = 0
        for topo in P.HAS_ACT_PLACEHOLDERS:
            for r in (1, 2, 3, 4):
                for s in b.sids(r, P.S_CROSSED):
                    for pair in P.PAIRS:
                        add(fuse(b.S(s, r, pair), topo, act=acts[n % len(acts)]))
                        n += 1
            for r in P.RANKS:
                for i, s in enumerate(b.lids(r, P.S_CROSSED)):
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
        for ti, topo in enumerate(("add", "add_drelu")):
            for r in P.RANKS:
                sids = b.sids(r, P.S_CROSSED)
                for j, pair in enumerate(P.PAIRS):
                    add(
                        fuse(b.S(sids[(j * 3 + ti + r + 7) % len(sids)], r, pair), topo)
                    )
        for r in P.RANKS:
            pr = P.PAIRS[3]
            for topo in P.HAS_ACT_PLACEHOLDERS:
                add(v_materialised(fuse(b.S("S3", r, pr), topo, act="relu")))
                add(v_inter_io(fuse(b.S("S3", r, pr), topo, act="relu")))
        for rid, e in P.R_SHAPES.items():
            if e["level"] == "comprehensive":
                add(b.R(rid))

    else:  # full: which wider sweeps must exist in the full suite?
        for r in P.RANKS:
            for s in b.lids(r):
                for pair in P.PAIRS:
                    add(b.Ls(s, r, pair))
        for xid in ("X1", "X2", "X4"):
            for pair in P.PAIRS:
                add(b.X(xid, pair))
        add(b.X("X3", ("cf", "half")))
        add(b.X("X3", ("cl", "bfloat16")))
        extra = list(P.EXTRA_TYPES)
        mixed = list(P.MIXED_TYPES)
        for r in (1, 2, 3, 4):
            for i, s in enumerate(b.sids(r, P.S_CROSSED)):

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
        for r in P.RANKS:
            for pair in P.PAIRS:
                for cls in ("zero", "const", "large", "tiny"):
                    add(v_vals(b.S("S2", r, pair), cls))
                for topo in P.HAS_ACT_PLACEHOLDERS:
                    for name in acts:
                        add(fuse(b.S("S3", r, pair), topo, act=name))
        for ai, name in enumerate(acts):
            for r in P.RANKS:
                for i, s in enumerate(b.sids(r, P.S_CROSSED)):
                    add(fuse(b.S(s, r, P.PAIRS[(i + ai) % 6]), "dact", act=name))
                for i, s in enumerate(b.lids(r, P.S_CROSSED)):
                    add(fuse(b.Ls(s, r, P.PAIRS[(i + ai) % 6]), "dact", act=name))
        for ti, topo in enumerate(("add", "add_drelu")):
            for r in P.RANKS:
                for i, s in enumerate(b.sids(r, P.S_CROSSED)):
                    add(fuse(b.S(s, r, P.PAIRS[(i + ti) % 6]), topo))
        for rid, e in P.R_SHAPES.items():
            if e["level"] == "full":
                add(b.R(rid))
    return [c for c in cases if c is not None]


def build():
    """All levels, cumulative-deduplicated: a graph appears in the lowest level
    that holds it. Returns {level: [(case, id)]} and the dropped counts."""
    seen = set()
    result = {}
    dropped = Counter()
    for lv, level in enumerate(P.LEVELS):
        out = []
        for c in rules(lv):
            reason = P.legal(c)
            if reason:
                dropped[f"{level}: {reason}"] += 1
                continue
            exempt = c["size"] == "X" or (
                any(i in P.M_PR for i in c["ids"]) and level == "standard"
            )
            if not exempt and not P.fits(c, level) and c["size"] != "L":
                dropped[f"{level}: over ceiling"] += 1
                continue
            key = P.canon(c)
            if key in seen:
                continue
            seen.add(key)
            out.append([c, case_id(c)])
        counts = Counter((c["topo"], cid) for c, cid in out)
        for item in out:
            c, cid = item
            if counts[(c["topo"], cid)] > 1:
                item[1] = f"{cid}_{P.short_hash(c)}"
        result[level] = out
    dropped.update(apply_trims(result))
    return result, dropped


def apply_trims(result):
    """Remove the products of ``P.TRIMS``. Runs after the ids are assigned, so the
    cases that remain keep their ids."""
    removed = Counter()
    for level, items in result.items():
        keep = []
        for item in items:
            t = P.trim(item[0], level)
            if t:
                removed[f"{level}: trimmed {t}"] += 1
            else:
                keep.append(item)
        result[level] = keep
    return removed


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def dump(obj):
    return json.dumps(obj, indent=2, sort_keys=False) + "\n"


def support_json(path, case_ids):
    """The sidecar of a sweep. Claims are harvested, never written here: an
    existing sidecar keeps the claims of cases that still exist."""
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
        for op in BUNDLE_OPS:
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
    print("  ".join(f"{h:>13}" for h in hdr))
    for level, items in result.items():
        cnt = Counter(c["topo"] for c, _ in items)
        total += len(items)
        print(
            "  ".join(
                f"{v:>13}"
                for v in [level] + [cnt[t] for t in P.TOPOLOGIES] + [len(items), total]
            )
        )
    print("\nBy purpose and size class")
    for level, items in result.items():
        pur = Counter(P.purpose(c) for c, _ in items)
        sz = Counter(c["size"] for c, _ in items)
        print(f"  {level:<14} {dict(pur)}  {dict(sorted(sz.items()))}")
    geo = {
        tuple(
            c[k]
            for k in ("rank", "n", "c", "k", "g", "ins", "f", "s", "d", "pre", "post")
        )
        for items in result.values()
        for c, _ in items
    }
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
