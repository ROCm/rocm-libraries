#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate the SDPA backward bundles from the test plan.

Reads the declarations in ``sdpa_bwd_plan.py`` and the level rules below, and
writes, for each level (quick, standard, comprehensive, full), one template
sweep per topology under ``integration-test-bundles/<level>/SdpaBackward/``:

    Default             Q, K, V, O, dO, Stats -> dQ, dK, dV
    AdditiveMask        + Bias
    AdditiveMaskDBias   + Bias, dBias
    DBiasOnly           + dBias
    ScaleTensor         + Scale (a compile-time constant 0.5; runtime cases patch it)
    DropoutScaleInv     + DropoutScaleInv
    SeqLenQOnly         + SeqLenQ
    DropoutMask         + DropoutMask, DropoutScale, DropoutScaleInv
    Philox              + Seed, Offset
    Padding             + SeqLenQ, SeqLenKv
    Ragged              + ragged offsets of Q, K, V, O
    FwdBwdChained       SdpaFwd -> SdpaBackward (O and Stats virtual)

A directory is written only for a topology with a runnable case. Each gets
``graph.template.json``, ``sweep.json`` and ``support.json`` (claims are
harvested with ``--write-support-claims``; the generator only drops the claims
of case ids that no longer exist). A case whose oracle is not ready (level
``notrun``) stays declared here and is not written.

Every case carries ``metadata.purpose`` (claim | probe | regression),
``metadata.plan_items`` (the plan rows, features and shapes it covers), its
size class and its tolerance class. The case id ends with the class token;
``--toml`` writes the matching ``[[tolerance_overrides]]`` rows into an engine
TOML between the generator's markers.

The tolerance classes rest on the magnitude estimates in
``sdpa_bwd_calibration.json``. A run fails if a case's estimate is missing or
stale; ``--calibrate`` (NumPy) computes it and rewrites the file.

The generator is deterministic: the same plan gives the same bytes. See
``docs/test-plans/sdpa-bwd.md``.

Usage:
    generate_sdpa_bwd_bundles.py --bundle-dir <integration-test-bundles> [--report] [--dry-run]
        [--toml <ENGINE>.toml] [--spec <cases.json>] [--calibrate]
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sdpa_bwd_plan as P  # noqa: E402

GENERATOR = "generate_sdpa_bwd_bundles.py"
GENERATOR_VERSION = "1"
REFERENCE_SOURCE = (
    "hipDNN reference executors, auto verification (GPU reference, then CPU); "
    "O and Stats derived from the forward by the fill (#14312)"
)
OP = "SdpaBackward"
ID_CAP = 96  # characters
ISQ = P.ISQ


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


def ph(name):
    return "${case." + name + "}"


def attr(name):
    return ph("attributes." + name)


# fmt: off
TENSOR_NAME = {
    "K": "K", "O": "O", "Q": "Q", "dK": "SdpaBwd_0::dK", "dQ": "SdpaBwd_0::dQ", "dV": "SdpaBwd_0::dV",
    "V": "V", "dO": "dO", "Stats": "stats", "Bias": "attn_mask", "SeqLenQ": "seq_len_q",
    "SeqLenKv": "seq_len_kv", "DropoutMask": "dropout_mask", "DropoutScale": "dropout_scale",
    "DropoutScaleInv": "dropout_scale_inv", "dBias": "SdpaBwd_0::dBias", "Scale": "attn_scale",
    "Seed": "seed", "Offset": "offset", "RaggedOffsetQ": "ragged_offset_q", "RaggedOffsetK": "ragged_offset_k",
    "RaggedOffsetV": "ragged_offset_v", "RaggedOffsetO": "ragged_offset_o",
}
# fmt: on
RAGGED_OF = {
    "Q": "Q",
    "dQ": "Q",
    "K": "K",
    "dK": "K",
    "V": "V",
    "dV": "V",
    "O": "O",
    "dO": "O",
}
GRAPH_NAME = {"Default": "SdpaBackwardTest"}  # others: SdpaBackward<Topology>Test
SCALE_CONST = 0.5  # the ScaleTensor template's compile-time constant
BAND_PARAMETERS = (
    "alibi_mask",
    "attn_scale_value",
    "causal_mask",
    "causal_mask_bottom_right",
    "diagonal_alignment",
    "dropout_probability",
    "left_bound",
    "right_bound",
)
FWD_INPUTS = (
    "attn_mask block_mask descale_k descale_q descale_s descale_v dropout_mask dropout_scale offset "
    "page_table_k page_table_v scale scale_o scale_s seed seq_len_kv seq_len_q sink_token"
).split()


def template_tensors(topo):
    return ["Q", "K", "V", "O", "dO", "Stats", "dQ", "dK", "dV"] + list(
        P.TOPOLOGIES[topo]
    )


def _bwd_node(names, topo):
    def uid(n):
        return P.UID[n] if n in names else None

    params = {k: attr("parameters__" + k) for k in BAND_PARAMETERS}
    params["padding_mask"] = (
        attr("parameters__padding_mask") if topo == "Padding" else False
    )
    inputs = {
        "attn_mask_tensor_uid": uid("Bias"),
        "do_tensor_uid": P.UID["dO"],
        "dropout_mask_tensor_uid": uid("DropoutMask"),
        "dropout_scale_inv_tensor_uid": uid("DropoutScaleInv"),
        "dropout_scale_tensor_uid": uid("DropoutScale"),
        "k_tensor_uid": P.UID["K"],
        "o_tensor_uid": P.UID["O"],
        "offset_tensor_uid": uid("Offset"),
        "q_tensor_uid": P.UID["Q"],
        "scale_tensor_uid": uid("Scale"),
        "seed_tensor_uid": uid("Seed"),
        "seq_len_kv_tensor_uid": uid("SeqLenKv"),
        "seq_len_q_tensor_uid": uid("SeqLenQ"),
        "stats_tensor_uid": P.UID["Stats"],
        "v_tensor_uid": P.UID["V"],
    }
    outputs = {
        "dbias_tensor_uid": uid("dBias"),
        "dk_tensor_uid": P.UID["dK"],
        "dq_tensor_uid": P.UID["dQ"],
        "dv_tensor_uid": P.UID["dV"],
    }
    return {
        "compute_data_type": attr("compute_data_type"),
        "inputs": inputs,
        "name": "SdpaBwd_0",
        "outputs": outputs,
        "parameters": dict(sorted(params.items())),
        "type": "SdpaBackwardAttributes",
    }


def _fwd_node():
    """The forward of FwdBwdChained: the backward's band and scale; it writes
    O and the Stats the backward reads."""
    inputs = {k + "_tensor_uid": None for k in FWD_INPUTS}
    inputs.update(
        q_tensor_uid=P.UID["Q"], k_tensor_uid=P.UID["K"], v_tensor_uid=P.UID["V"]
    )
    outputs = {
        k + "_tensor_uid": None
        for k in ("amax_o", "amax_s", "max", "rng_dump", "sum_exp")
    }
    outputs.update(o_tensor_uid=P.UID["O"], stats_tensor_uid=P.UID["Stats"])
    attributes = {k: attr("parameters__" + k) for k in BAND_PARAMETERS}
    attributes.update(
        generate_stats=True,
        implementation="AUTO",
        max_seq_len_kv=None,
        mma_core_mode="unset",
        padding_mask=False,
    )
    return {
        "attributes": dict(sorted(attributes.items())),
        "compute_data_type": attr("compute_data_type"),
        "inputs": dict(sorted(inputs.items())),
        "name": "SdpaFwd_0",
        "outputs": dict(sorted(outputs.items())),
        "type": "SdpaAttributes",
    }


def template(topo):
    """graph.template.json of a topology: every tensor's dims, strides,
    data_type and alignment, the node compute type, the band, the scale and
    the flags are placeholders."""
    names = template_tensors(topo)
    chained = topo == "FwdBwdChained"
    tensors = []
    for n in names:
        virtual = chained and n in ("O", "Stats")
        t = {
            "alignment": ph("alignment"),
            "data_type": ph("data_type"),
            "dims": ph("dims"),
            "is_runtime_pass_by_value": False,
            "name": "SdpaFwd_0::" + n if virtual else TENSOR_NAME[n],
            "strides": ph("strides"),
            "uid": P.UID[n],
            "virtual": virtual,
        }
        if n == "Scale":  # compile-time constant; a runtime case patches it
            t.update(value=SCALE_CONST, value_type="Float32Value")
        if topo == "Ragged" and n in RAGGED_OF:
            t["ragged_offset_tensor_uid"] = P.UID["RaggedOffset" + RAGGED_OF[n]]
        tensors.append(dict(sorted(t.items())))
    tensors.sort(key=lambda t: t["uid"])
    nodes = ([_fwd_node()] if chained else []) + [_bwd_node(names, topo)]
    return {
        "compute_data_type": "float",
        "intermediate_data_type": "float",
        "io_data_type": ph("io_data_type"),
        "is_override_shape_enabled": False,
        "min_required_engine_api_version": {"major": 1, "minor": 0, "patch": 0},
        "name": GRAPH_NAME.get(topo, f"SdpaBackward{topo}Test"),
        "nodes": nodes,
        "tensors": tensors,
    }


def placeholders(tpl):
    """Case-wide placeholder paths (not the per-tensor ones)."""
    found = set()

    def walk(v, in_tensor):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(x, in_tensor or k == "tensors")
        elif isinstance(v, list):
            for x in v:
                walk(x, in_tensor)
        elif isinstance(v, str) and not in_tensor:
            m = re.fullmatch(r"\$\{case\.(.+)\}", v)
            if m:
                found.add(m.group(1))

    walk(tpl, False)
    return found


# --------------------------------------------------------------------------
# Case -> sweep entry
# --------------------------------------------------------------------------


def _recipe(spec):
    """The fill recipe: free (uniform) or fixed. A per-batch list with one
    value is a fixed fill; other lists need HG4."""
    if isinstance(spec, list):
        return {"kind": "free", "lo": spec[0], "hi": spec[1]}
    if isinstance(spec, dict) and spec.get("kind") == "fixed":
        return {"kind": "fixed", "value": spec["value"]}
    if (
        isinstance(spec, dict)
        and spec.get("kind") == "list"
        and len(set(spec["values"])) == 1
    ):
        return {"kind": "fixed", "value": spec["values"][0]}
    raise ValueError(f"no fill recipe for {spec!r}")


def case_values(c):
    """The `values` block and the tensor patches of a sweep case."""
    rec = c["record"]
    lb, rb = P.bounds_written(c)
    at = c.get("attrs") or {}
    s = c.get("scale", ISQ)
    params = {
        "alibi_mask": bool(at.get("alibi_mask")),
        "attn_scale_value": (
            None if (s == "unset" or isinstance(s, dict)) else P.scale_value(c)
        ),
        "causal_mask": bool(at.get("causal_mask")),
        "causal_mask_bottom_right": bool(at.get("causal_mask_bottom_right")),
        "diagonal_alignment": rec["diagonal_alignment"],
        "dropout_probability": at.get("dropout_probability"),
        "left_bound": lb,
        "right_bound": rb,
    }
    if c["topology"] == "Padding":
        params["padding_mask"] = (c.get("padding") or {}).get("padding_mask", True)
    attributes = {"compute_data_type": "float"}
    attributes.update({"parameters__" + k: v for k, v in sorted(params.items())})
    keys = ("uid", "dims", "strides", "data_type", "alignment")
    tensors = [
        {k: t[k] for k in keys}
        for t in sorted(rec["strides"].values(), key=lambda t: t["uid"])
    ]
    values = {
        "io_data_type": rec["strides"]["Q"]["data_type"],
        "attributes": attributes,
        "tensors": tensors,
    }
    patches = []
    if isinstance(s, dict) and s["tensor"] == "runtime":
        patch = {
            "uid": P.UID["Scale"],
            "set": {"is_runtime_pass_by_value": True},
            "remove": ["value", "value_type"],
        }
        patches.append(patch)
    elif isinstance(s, dict) and s["value"] != SCALE_CONST:
        raise ValueError(
            f"{c['case_id']}: a constant scale tensor must be the template literal {SCALE_CONST}"
        )
    return values, patches


def input_recipes(c):
    """metadata.inputs: a fill recipe for every input, keyed by uid."""
    out = {
        str(P.UID[name]): _recipe(spec) for name, spec in c["record"]["inputs"].items()
    }
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def notes(c):
    parts = []
    if c["purpose_final"] == "probe":
        if c.get("rejection"):
            parts.append("rejection probe (layer U): " + c["legality"]["why"])
        elif c.get("defect"):
            parts.append(f"open defect {c['defect']}")
        else:
            parts.append("probe: " + "; ".join(P.probe_reasons(c, c["dtype"])))
    if c.get("regression_of"):
        parts.append("regression of " + c["regression_of"])
    if c.get("expected"):
        parts.append("expected: " + c["expected"])
    if c.get("repeat"):
        parts.append(f"repeat {c['repeat']} needs I5 (HG5); one run until then")
    return "; ".join(parts)


def sweep_case(c):
    values, patches = case_values(c)
    meta = {
        "format_version": 1,
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "reference_source": REFERENCE_SOURCE,
        "operation": OP + ("" if c["topology"] == "Default" else "+" + c["topology"]),
        "plan_items": c["record"]["plan_items"],
        "purpose": c["purpose_final"],
        "size_class": c["set"],
        "seed": c.get("seed", 0),
        "tol_class": c["tol_class"],
        "inputs": input_recipes(c),
    }
    text = notes(c)
    if text:
        meta["notes"] = text
    if c.get("repeat"):
        meta["repeat"] = c["repeat"]
    if c.get("rejection") and P.WRITE_EXPECTED_REJECTION:
        meta["expected_rejection"] = True
    total = sum(
        math.prod(t["dims"]) * P.BYTES[t["data_type"]]
        for t in c["record"]["strides"].values()
    )
    if total > (1 << 30):
        meta["minimum_vram_mb"] = int(total * 2 / (1 << 20)) + 1
    entry = {"id": c["case_id"], "values": values, "metadata": meta}
    if patches:
        entry["tensor_patches"] = patches
    if c.get("golden") and P.WRITE_GOLDEN_POINTERS:
        entry["golden"] = {"path": f"golden/{c['case_id']}/tensors.dvc"}
    return entry


# --------------------------------------------------------------------------
# Case ids: <item>_b<B>h<Hq>k<Hk>v<Hv>_<Sq>x<Skv>_d<Dqk>[v<Dv>]_<dtype>_<band>[_<extras>]_tol<class>
# (golden: <item>_<dtype>_<band>_tol<class>). The class token is last so the
# TOML rows can anchor on it; the dtype token tells bfp16 from fp16.
# --------------------------------------------------------------------------


def num_token(x):
    return ("%g" % x).replace("-", "m").replace(".", "p")


def band_token(c):
    W, R, al = c.get("W"), c.get("R"), c.get("align", "TL").lower()
    at = c.get("attrs") or {}
    pre = ("cm" if at.get("causal_mask") else "") + (
        "cmbr" if at.get("causal_mask_bottom_right") else ""
    )
    body = {
        "none": al + "none" if al == "br" else "none",
        "causal": f"{al}causal",
        "window": f"{al}w{W}",
        "band": f"{al}w{W}r{R}",
        "left_only": f"{al}w{W}rinf",
        "right_only": f"{al}r{R}",
    }[P.mask_kind(W, R)]
    return (pre + "_" if pre else "") + body


def extras(c):
    """Id tokens of everything that differs from the plain op."""
    ex = []
    if c.get("layout", "bhsd") != "bhsd":
        ex.append(c["layout"].replace("_", ""))
    ex += [
        f"{t.lower()}{k.replace('_', '')}"
        for t, k in (c.get("tensor_layout") or {}).items()
    ]
    if c.get("stats", "packed") != "packed":
        ex.append("stats" + c["stats"])
    s = c.get("scale", ISQ)
    if s == "unset":
        ex.append("sunset")
    elif isinstance(s, dict):
        ex.append(
            ("srt" if s["tensor"] == "runtime" else "sconst") + num_token(s["value"])
        )
    elif s != ISQ:
        ex.append("s" + num_token(float(s)))
    ins = c.get("inputs") or {}
    for k in ("q", "k", "v", "do", "bias"):
        v = ins.get(k)
        if isinstance(v, dict):
            ex.append(f"{k}fix{num_token(v['value'])}")
        elif isinstance(v, list):
            ex.append(f"{k}{num_token(v[0])}to{num_token(v[1])}")
        elif v is not None:
            ex.append(f"{k}inj")
    b = c.get("bias")
    if b:
        ex.append(
            "bias"
            + (
                b["dims"]
                if isinstance(b["dims"], str)
                else "x".join(map(str, b["dims"]))
            )
            + b["dtype"]
        )
    if c.get("dbias"):
        ex.append("dbias")
    at = c.get("attrs") or {}
    if at.get("alibi_mask"):
        ex.append("alibi")
    if "dropout_probability" in at:
        ex.append("p" + num_token(at["dropout_probability"]))
    if "right_bound_raw" in at:
        ex.append("rbraw" + num_token(at["right_bound_raw"]))
    if at.get("explicit_unbounded"):
        ex.append("lbm1rbm1")
    dov = c.get("dtype_over") or {}
    if dov and set(dov) >= set(P.IO) and len(set(dov.values())) == 1:
        ex.append("io" + next(iter(dov.values())).replace("_", ""))
    else:
        ex += [f"{t.lower()}{d}" for t, d in dov.items()]
    if c.get("alignment", 16) != 16:
        ex.append(f"align{c['alignment']}")
    if c.get("dropout"):
        ex.append("dropout" + num_token(c["dropout"]["p"]))
    if c.get("padding"):
        ex.append(
            "seqlen" + ("" if c["padding"].get("padding_mask", True) else "noflag")
        )
    if c.get("ragged"):
        ex.append("ragged")
    if c.get("topology") in (
        "DropoutScaleInv",
        "Philox",
        "FwdBwdChained",
        "DBiasOnly",
        "SeqLenQOnly",
    ):
        ex.append(c["topology"].lower())
    if (b or {}).get("stride") not in (None, "packed"):
        ex.append("bias" + b["stride"])
    if c.get("dbias_stride") not in (None, "packed"):
        ex.append("dbias" + c["dbias_stride"])
    ex += [
        f"{t.lower()}{how.replace('_', '')}"
        for t, how in (c.get("dims_over") or {}).items()
    ]
    return ex


def case_id(c, dt, cls):
    hdr = c["id"].lower().replace("-", "").replace(".", "")
    dtok, band, tok = P.DTYPE_TOKEN[dt], band_token(c), P.TOKEN[cls]
    if c.get("golden"):
        return "_".join([hdr, dtok, band, tok])
    geo = f"b{c['B']}h{c['Hq']}k{c['Hk']}v{c['Hv']}_{c['Sq']}x{c['Skv']}_d{c['Dqk']}"
    geo += f"v{c['Dv']}" if c["Dv"] != c["Dqk"] else ""
    ex = extras(c)
    cid = "_".join([hdr, geo, dtok, band] + ex + [tok])
    if len(cid) > ID_CAP:  # keep the item, geometry, dtype, band and class tokens
        cid = "_".join(
            [
                hdr,
                geo,
                dtok,
                band,
                "x" + hashlib.sha1("_".join(ex).encode()).hexdigest()[:6],
                tok,
            ]
        )
    return cid


# --------------------------------------------------------------------------
# Level rules
# --------------------------------------------------------------------------

CORE_BASE = dict(B=2, Hq=3, Hk=3, Hv=3, Dqk=128, Dv=128)
CORE_SEQ = {"lt": (40, 72), "eq": (64, 64), "gt": (72, 40)}
CORE_BAND = {"none": (None, None), "causal": (None, 0), "window": (16, 0)}
# Core cells (§4): Sq:Skv relation x mask class x alignment where it differs.
CORE_CELLS = [
    (seq, mask, al)
    for seq in ("lt", "eq", "gt")
    for mask in ("none", "causal", "window")
    for al in (("TL",) if (mask == "none" or seq == "eq") else ("TL", "BR"))
]
CAUSAL_FLAG = {"causal_mask": True}
BOTH_FLAGS = {"causal_mask": True, "causal_mask_bottom_right": True}


def g(B, Hq, Hk, Hv, Sq, Skv, Dqk=128, Dv=128):
    return dict(B=B, Hq=Hq, Hk=Hk, Hv=Hv, Sq=Sq, Skv=Skv, Dqk=Dqk, Dv=Dv)


def item(id_, level, purpose, items, geom, dt="bf16", **kw):
    """A §4 plan item (set H unless given): items and dt are space-separated;
    geom is (B, Hq, Hk, Hv, Sq, Skv[, Dqk, Dv])."""
    kw.setdefault("set", "H")
    return P.new(
        id=id_,
        level=level,
        purpose=purpose,
        items=items.split(),
        dtypes=dt.split(),
        **g(*geom),
        **kw,
    )


def core_cell(cell):
    seq, mask, al = cell
    Sq, Skv = CORE_SEQ[seq]
    W, R = CORE_BAND[mask]
    return dict(Sq=Sq, Skv=Skv, W=W, R=R, align=al, core=cell)


def derived_features(c):
    """Feature ids a case exercises, from its values (only what differs from
    the core base)."""
    W, R, al = P.effective_band(c)
    mk = P.mask_kind(W, R)
    f = ["DT1", "G7"]
    if mk != "none":
        f.append("A3" if R is not None else "A4")
        if W is not None:
            f.append("A4")
        if mk in ("band", "left_only", "right_only"):
            f.append("A5")
        if c["Sq"] != c["Skv"]:
            f.append("A2")
    if not (c["Hq"] == c["Hk"] == c["Hv"]):
        f.append("G3")
        if c["Hk"] != c["Hv"]:
            f.append("G4")
    if (c["Dqk"], c["Dv"]) != (128, 128):
        f.append("G8" if c["Dqk"] == c["Dv"] else "G9")
    lay = c.get("layout", "bhsd")
    if lay != "bhsd" or c.get("tensor_layout"):
        f.append("T18")
        if lay in ("do_bshd", "grads_bshd") or c.get("tensor_layout"):
            f.append("AW12")
        if lay == "grads_bshd":
            f.append("G10")
    s = c.get("scale", ISQ)
    if s != ISQ:
        f += ["A1"] + (["T12", "T22"] if isinstance(s, dict) else [])
    ins = c.get("inputs") or {}
    if any(
        not (k == "do" and isinstance(v, list) and v[0] == -v[1])
        for k, v in ins.items()
    ):
        f.append("V1")
    b = c.get("bias")
    if b:
        f += ["T10", "G11"] + (["DT4"] if b["dtype"] != "fp32" else [])
    if c["Sq"] > c["Skv"] and (al == "BR" and R is not None or W is not None):
        f.append("AW2")
    if c["Sq"] < c["Skv"] and al == "TL" and R is not None:
        f.append("AW3")
    return sorted(set(f))


def core_items():
    """Quick: every core cell in both dtypes (§4.1)."""
    out = []
    for n, cell in enumerate(CORE_CELLS, 1):
        seq, mask, al = cell
        feats = ["DT1", "G1", "G2", "G5", "G6", "G7", "T18", "V1", "V2"]
        if mask != "none":
            feats += ["A2", "A3"] + (["A4"] if mask == "window" else [])
        if seq == "gt" and (al == "BR" or mask == "window"):
            feats.append("AW2")
        if seq == "lt" and al == "TL" and mask != "none":
            feats.append("AW3")
        c = dict(CORE_BASE, **core_cell(cell))
        out.append(
            P.new(
                id=f"C{n:02d}",
                level="quick",
                purpose="claim",
                items=feats,
                dtypes=("bf16", "fp16"),
                inputs=P.domain("v01"),
                **c,
            )
        )
    return out


def quick_variant_items():
    """S features on the core base (quick, PR), and head dim 192 on three
    core cells (targeted product TP-E, PR)."""
    q = dict(CORE_BASE, Sq=40, Skv=72)
    out = [
        P.new(
            id="S40.q",
            level="quick",
            purpose="claim",
            items=["T18", "S40"],
            layout="bshd",
            **q,
        ),
        P.new(
            id="S42.q",
            level="quick",
            purpose="claim",
            items=["AW12", "G10", "T18", "S42"],
            layout="grads_bshd",
            **q,
        ),
        P.new(
            id="S17.q",
            level="quick",
            purpose="claim",
            items=["G8", "S17"],
            **dict(q, Dqk=192, Dv=192),
        ),
        P.new(
            id="S29.q",
            level="quick",
            purpose="claim",
            items=["A5", "A3", "A4", "S29"],
            dtypes=("fp16",),
            W=9,
            R=8,
            **q,
        ),
        P.new(
            id="S16.c",
            level="pr",
            purpose="probe",
            items=["G8", "S16"],
            **dict(q, Dqk=256, Dv=256),
        ),
    ]
    cells = (
        (("lt", "none", "TL"), "fp16"),
        (("eq", "causal", "TL"), "bf16"),
        (("gt", "causal", "BR"), "fp16"),
    )
    for k, (cell, dt) in enumerate(cells, 1):
        seq, mask, al = cell
        feats = ["G8", "G7", "TP-E"] + (["A2", "A3"] if mask != "none" else [])
        feats += ["AW2"] if seq == "gt" and al == "BR" else []
        c = dict(CORE_BASE, Dqk=192, Dv=192, **core_cell(cell))
        out.append(
            P.new(
                id=f"TP-E{k:02d}",
                level="pr",
                purpose="claim",
                items=feats,
                dtypes=(dt,),
                inputs=P.domain("v01"),
                **c,
            )
        )
    return out


# fmt: off
# Catalog placement (§3, §4.1): level -> shapes. A shape carried by another
# plan item is in MERGED instead (the carrier lists the shape id).
PLACE = {
    "quick":   "H08 H11 H50 H65 H90",
    "pr":      "S00 S01 S02 S04 S13 S15 S21 S23 S25 S26 S27 S28 S29 S30 S31 S32 S42 S43 S57 "
               "H02 H03 H04 H07 H09 H10 H12 H17 H18 H19 H20 H21 H22 H24 H25 H27 H28 H31 H33 H34 H35 H36 H37 "
               "H38 H39 H43 H44 H45 H46 H49 H58 H61 H66 H68 H69 H72 H73 H74 H75 H76 H83 H84 H86 H88 H89 H92",
    "nightly": "S05 S11 S14 S16 S17 S18 S19 S20 S22 S37 S38 S40 S41 S44 "
               "H05 H06 H23 H26 H29 H30 H32 H40 H41 H42 H47 H60 H67 H71 H77 H78 H85",
    "full":    "S03 S24 H94 H95 X05 X06 X07 X08 X09 X11 X13",
    "notrun":  "S46 S47 H62 H87 H91 H93 X01 X02 X03 X04 X10 X12 X14",
}
PLACE_DTYPES = {"S13": "fp16", "S25": "bf16", "S26": "fp16", "S27": "bf16", "S28": "fp16"}  # one dtype of two
OUT_OF_LEVEL = "X01 X02 X03 X04 X12".split()  # a timing run with the GPU reference re-sizes them
MERGED = {  # carrier: the shapes it carries
    "L08": "S06", "PRB-H08": "S07", "DEF-04b": "S08", "PRB-H03": "S09", "PRB-H01": "S10", "PRB-H06": "S12",
    "REG-01": "S33 S35", "REG-02b": "S34", "REG-04": "S36", "CLM-S1": "S39", "PRB-L03": "S45", "PRB-B01": "S48",
    "REG-07a": "S49", "PRB-B12": "S50", "PRB-M04": "S51", "PRB-P02": "S52", "PRB-R04": "S53", "PRB-R03": "S54",
    "PRB-P06": "S55", "REG-08": "S56", "GL-01": "H01", "PRB-G01": "H13", "DEF-03a": "H14 H15", "DEF-03d": "H16",
    "REG-05c": "H48", "DEF-01a": "H51", "DEF-01b": "H52", "DEF-01i": "H53", "DEF-02a": "H54", "DEF-02c": "H55",
    "DEF-02b": "H56", "DEF-02d": "H57", "DEF-05": "H59", "REG-13c": "H63", "REG-13a": "H64", "REG-13d": "H70",
    "REG-14": "H79", "PRB-B11": "H80", "DEF-06a": "H81", "DEF-06c": "H82",
}
EXTRA_ITEMS = {  # plan items a placed shape also carries (the §4 probes it reproduces)
    "S01": "PRB-T01", "S15": "PRB-D01", "S16": "PRB-D06", "S18": "PRB-D08", "S20": "PRB-D04", "S21": "PRB-D02",
    "S22": "PRB-D03", "S24": "PRB-D07", "S42": "PRB-L02", "H24": "PRB-E01", "H27": "PRB-M07", "H31": "PRB-E06",
    "H35": "PRB-D05", "H50": "REG-05d", "H58": "PRB-E05", "H72": "PRB-E02", "H75": "PRB-B09", "H76": "PRB-B05",
    "H77": "PRB-B08", "H88": "PRB-E04", "H89": "PRB-V01", "H90": "V9", "X11": "A3 G7",
}

# Model layers (§3.3; registers R-M2..R-M4 in §4.3): variants dropped for the nightly
# budget (each model stays covered), and edits.
M_DROP = ("M05.n M15.n M30.n M32.n M33.n M34.n M37.n M24.n M27.n M35.n M08.n M16.n M31.n M03.pr M36.pr "
          "M39.f").split()  # M39.f: replaced in full by X11 and M41.fc
M_EDIT = {
    "M25.n": {"Hq": 8},          # heads 16/1/1 -> 8/1/1 for the nightly budget
    "M44.n": {"level": "full"},  # the model layer, in full for the nightly budget
    "M40.n": {"Hq": 2},          # inside the nightly ceiling once its gate opens
    "M26.f": {"B": 32},          # 127 s instead of 254 s; the real batch is register R-M4
    # The trim rule (plan §4.4, register R-M6): probes move up a level, largest first,
    # until PR and nightly are inside their 5 % reserve.
    "M09.pr": {"level": "nightly"},
    "M28.pr": {"level": "nightly"},
    "M31.pr": {"level": "nightly"},
    "M03.n": {"level": "full"},
    "M12.n": {"level": "full"},
    "M20.n": {"level": "full"},
    "M36.n": {"level": "full"},
    "M43.n": {"level": "full"},
}

# Not-run items: the level each enters when its gate opens (default: pr; M
# and X: full)
TARGET_LEVEL = {
    "pr": "PRB-B11 PRB-B12 PRB-B13 PRB-B14 PRB-B16 PRB-M04 PRB-P01 PRB-P02 PRB-P03 PRB-P04 PRB-R04 REG-08 REG-14 "
          "S46 S47 H93 H91 M40.pr",
    "nightly": "PRB-P06 M40.n",
    "none": "PRB-R03 H62",
    "out-of-level": "X03",
}
# fmt: on
TARGET_OF = {i: lv for lv, ids in TARGET_LEVEL.items() for i in ids.split()}


def catalog_items():
    """The placed S, H and X shapes."""
    out = []
    for lv, ids in PLACE.items():
        for sid in ids.split():
            it = P.catalog_shape(sid)
            it["level"] = lv
            if sid in PLACE_DTYPES:
                it["dtypes"] = [PLACE_DTYPES[sid]]
            if sid in OUT_OF_LEVEL:
                it["out_of_level"] = True
            it["items"] += EXTRA_ITEMS.get(sid, "").split()
            out.append(it)
    return out


def model_items():
    """Model layers and their variants as the corpus stores them: BSHD IO
    tensors; causal TL written as causal_mask=true."""
    out = []
    for r in P.model_variants():
        mid, sfx = r["id"].split(".")
        if r["id"] in M_DROP:
            continue
        geo = {k: r[k] for k in P.GEOMETRY + ("W", "R", "align", "scale")}
        level = {"pr": "pr", "n": "nightly", "fc": "full", "f": "full"}[sfx]
        it = P.new(
            id=r["id"],
            set="M",
            level=level,
            items=r["features"] + [mid, "T18"],
            layout="bshd",
            **geo,
        )
        if r["oracle"] != "ready":
            it["oracle"] = r["oracle"]
        if r.get("seqlens"):
            it["padding"] = {
                "seq_len_q": list(r["seqlens"]),
                "seq_len_kv": list(r["seqlens"]),
                "padding_mask": True,
            }
            it["topology"] = "Padding"
            it["oracle"] = "14358+derivation+HG4"
        if mid in P.M_CORPUS_ENCODED and it["align"] == "TL" and it["R"] == 0:
            it["attrs"] = dict(CAUSAL_FLAG)
            it["R"] = None
            it["items"] += ["A6", "AW7"] if it["W"] else ["A6"]
        if r["id"] == "M09.pr":
            it["items"].append("M10")
        it.update(M_EDIT.get(r["id"], {}))
        if sfx == "f" and r["oracle"] != "ready":
            it["level"] = P.NOTRUN
            it["out_of_level"] = True
        if r["oracle"] == "14358" or it.get("padding"):
            it["level"] = P.NOTRUN
        out.append(it)
    return out


def l_items():
    return [dict(P.l_shape(lid), level="nightly") for lid in P.L_SHAPES]


# Defect entry gates (D7, resolved: companion engine pull requests). A DEF
# case is written once its gate is recorded here.
_DECLINE = "companion engine PR: ASM_SDPA_ENGINE declines "
DEFECT_GATE = {
    "DEF-01": _DECLINE + "a head-dim stride != 1 (asm-sdpa-bwd-decline-strides)",
    "DEF-02": _DECLINE + "non-packed Stats (asm-sdpa-bwd-decline-strides)",
    "DEF-04": _DECLINE + "V, dK, dV dims != K's (asm-sdpa-bwd-decline-strides)",
    "DEF-05": _DECLINE + "odd-element sequence strides (asm-sdpa-bwd-decline-strides)",
    "DEF-06": _DECLINE + "scale <= 0 (in diagnosis)",
    "DEF-03": "kernel precision (AITER code object): ASM_SDPA_ENGINE.toml [[test_skips]] row (tracking: to file)",
}

# Expected outcomes and the verdict of a rejection probe on the baseline
# (before #14346 adds expected_rejection).
E_ANY = "typed rejection, or correct accept"
E_OR = "correct accept, or typed rejection"
E_GROUPS = "dK/dV reduced over each head group, or typed rejection"
E_REJ = "typed rejection"
NOV = "none: the fill declines after an accept, so the case reports SKIP whatever the engine does"
FOV = "FAIL on a wrong accept (the reference throws or declines, or the output keeps the NaN sentinel)"


def regression_and_defect_items():
    """Escaped hipDNN bugs (REG, fixed) and open engine defects (DEF, probes
    until the engine declines them)."""
    bias = dict(
        topology="AdditiveMask", inputs=P.domain("bias_neg"), regression_of="#13474"
    )
    # fmt: off
    out = [
        item("REG-01", "quick", "regression", "A6 AW7 S33 S35", (1, 1, 1, 1, 128, 128), W=33, attrs=CAUSAL_FLAG,
             regression_of="#13065", expected="band W=33, R=0 TL (flag composes with the left bound)"),
        item("REG-02a", "quick", "regression", "A6 A2 AW7", (1, 1, 1, 1, 64, 128), align="BR", attrs=CAUSAL_FLAG,
             regression_of="#13065", expected="BR causal (flag keeps the alignment)"),
        item("REG-02b", "quick", "regression", "A7 S34", (1, 1, 1, 1, 64, 128),
             attrs={"causal_mask_bottom_right": True}, regression_of="#13065", expected="BR causal"),
        item("REG-03", "quick", "regression", "A6 A3 AW7", (1, 1, 1, 1, 64, 96), R=5, attrs=CAUSAL_FLAG,
             regression_of="#13065", expected="band (unbounded, R=5)"),
        item("REG-04", "quick", "regression", "A1 S36", (1, 1, 1, 1, 64, 96), scale="unset", regression_of="#12926",
             expected="scale 1.0"),
        item("REG-05a", "quick", "regression", "G3 G4", (1, 4, 1, 1, 64, 64), R=0, regression_of="#11543",
             expected=E_GROUPS),
        item("REG-05b", "quick", "regression", "G3 G4", (1, 4, 2, 2, 64, 64), R=0, regression_of="#11543",
             expected=E_GROUPS),
        item("REG-05c", "quick", "regression", "G3 G4 H48", (1, 4, 2, 1, 64, 64), R=0, regression_of="#11543",
             expected=E_GROUPS),
        item("REG-07a", "quick", "regression", "T10 DT4 S49", (1, 2, 2, 2, 64, 96), bias=P.bias_spec("11", "io"),
             expected="bias applied in its declared dtype", **bias),
        item("REG-07c", "pr", "regression", "T10 DT4", (1, 2, 2, 2, 64, 96), "fp16", bias=P.bias_spec("11", "io"),
             expected="bias applied in its declared dtype", **bias),
        item("REG-07b", "quick", "regression", "DT4 AW13", (1, 2, 2, 2, 64, 96), bias=P.bias_spec("1h", "fp16"),
             expected="HALF bias read as HALF with BF16 IO", **bias),
        item("REG-08", "notrun", "probe", "T6 S56", (1, 1, 1, 1, 128, 128), stats="rank3", oracle="14358",
             regression_of="#12933", expected="plan decision (rank-3 Stats in or out of the well-formed set)"),
        item("REG-13a", "quick", "regression", "A4 H64", (1, 1, 1, 1, 64, 128), W=2, R=0, regression_of="SEM-001",
             expected="exactly 2 keys per row"),
        item("REG-13c", "quick", "regression", "A4 AW4 H63", (1, 1, 1, 1, 64, 64), W=1, R=0, regression_of="SEM-001",
             expected="dQ = dK = 0 within the class tolerance; dV = dO"),
        item("REG-13d", "quick", "regression", "A4 AW10 H70", (1, 1, 1, 1, 65, 65), W=64, R=0,
             regression_of="#14483", expected="W=64 drops key 0 in row 64"),
        item("REG-13e", "quick", "regression", "A4 AW10", (1, 1, 1, 1, 65, 65), W=65, R=0, regression_of="#14483",
             expected="W=65 equals causal"),
        item("REG-14", "notrun", "probe", "V3 H79", (1, 2, 2, 2, 64, 96), bias=P.bias_spec("11"),
             inputs={"bias": "row 7 and column 5 all -inf"}, topology="AdditiveMask", oracle="14358+I4",
             regression_of="#14358 6be2472f007",
             expected="dead row: dQ row 0 exactly; dead key: dK = dV = 0; all finite"),
        # open engine defects (gfx942 pilots); a defect does not depend on the shape (§4.2 shrink rule)
        *[item(f"DEF-01{sfx}", "pr", "probe", "T19 IVP12 " + extra, (1, 1, 1, 1, 64, 64),
               tensor_layout={t: "dstride2"}, defect="DEF-01", expected=E_ANY)
          for t, sfx, extra in (("Q", "a", "H51"), ("dQ", "b", "H52"), ("K", "c", ""), ("V", "d", ""), ("O", "e", ""),
                                ("dO", "f", ""), ("dK", "g", ""), ("dV", "h", ""))],
        item("DEF-01i", "pr", "probe", "T19 H53", (1, 2, 2, 2, 72, 120), "fp16", tensor_layout={"Q": "sd_transposed"},
             defect="DEF-01", expected=E_ANY),
        item("DEF-02a", "pr", "probe", "T20 H54", (1, 2, 2, 2, 128, 128), stats="headpad64", defect="DEF-02",
             repeat=3, expected=E_ANY),
        item("DEF-02b", "pr", "probe", "T20 H56", (1, 2, 2, 2, 72, 120), stats="bsh", defect="DEF-02", expected=E_ANY),
        item("DEF-02c", "pr", "probe", "T20 H55", (1, 2, 2, 2, 72, 120), stats="seqstride2", defect="DEF-02",
             expected=E_ANY),
        item("DEF-02d", "pr", "probe", "T20 H57", (2, 2, 2, 2, 72, 120), stats="batchpad", defect="DEF-02",
             expected=E_ANY),
        item("DEF-03a", "pr", "probe", "AW9 V6 H14 H15", (1, 1, 1, 1, 1024, 1), "fp16", defect="DEF-03",
             expected="dV within class K1, or typed rejection"),
        item("DEF-03d", "pr", "probe", "AW9 H16", (2, 3, 3, 3, 512, 2), "fp16", defect="DEF-03",
             expected="dV within class K1, or typed rejection"),
        item("REG-18c", "quick", "claim", "AW9 V6", (1, 1, 1, 1, 1024, 1),
             expected="dV within class K1 (control for DEF-03)"),
        item("DEF-04a", "pr", "probe", "G4", (1, 4, 4, 2, 64, 64), defect="DEF-04",
             expected="dV reduced over each V-head group, or typed rejection"),
        item("DEF-04b", "pr", "probe", "G4 S08", (1, 4, 4, 1, 64, 64), defect="DEF-04",
             expected="dV reduced over all query heads, or typed rejection"),
        item("DEF-05", "pr", "probe", "T23 AW12 H59", (2, 2, 2, 2, 64, 64), "fp16", tensor_layout={"dO": "rowpad1"},
             defect="DEF-05", expected=E_ANY),
        item("DEF-06a", "pr", "probe", "A1 H81", (1, 2, 2, 2, 64, 64), R=0, scale=0.0, defect="DEF-06",
             expected="dQ = dK = 0 exactly (scale 0); dV finite"),
        item("DEF-06b", "pr", "probe", "A1", (1, 2, 2, 2, 128, 128), R=0, scale=-1.2, defect="DEF-06",
             expected="finite gradients within the class"),
        item("DEF-06c", "pr", "probe", "A1 H82", (1, 2, 2, 2, 72, 120), "fp16", scale=-1.2, defect="DEF-06",
             expected="finite gradients within the class"),
    ]
    # fmt: on
    return out


def probe_items():
    """Probes (PRB), rejection probes, scale-tensor claims (CLM), seed variants
    (SD), hazard additions and the 32-head model layer."""
    rj = dict(rejection=True, expected=E_REJ)
    nb = (1, 1, 1, 1, 64, 96)  # the rejection probes' base
    pad = dict(topology="Padding", oracle="14358+derivation+HG4")
    bias = dict(topology="AdditiveMask", inputs=P.domain("bias_neg"))
    scale = dict(set="H", topology="ScaleTensor", expected="scale 0.5 applied")
    m41 = dict(set="M", layout="bshd", attrs=CAUSAL_FLAG, expected="correct accept")
    fa = P.domain("fa_overflow")
    # fmt: off
    out = [
        item("PRB-E03a", "pr", "probe", "AW2 G8", (1, 2, 2, 2, 80, 48, 64, 64), "fp16", align="BR", R=0,
             expected="dead rows exactly 0, or typed rejection"),
        item("PRB-E03b", "pr", "probe", "AW2 G8", (1, 2, 2, 2, 192, 64, 160, 160), "fp16", align="BR", R=0,
             expected="dead rows exactly 0, or typed rejection"),
        item("PRB-E04t", "pr", "claim", "V1 PRB-E04", (2, 5, 5, 5, 17, 17), "fp16", R=0, inputs=fa,
             expected="finite; class holds"),
        item("PRB-E04Lt", "full", "claim", "V1 PRB-E04", (2, 5, 5, 5, 129, 129), "fp16", R=0, inputs=fa,
             expected="finite; class holds"),
        item("PRB-E05t", "pr", "claim", "AW12 PRB-E05", (2, 2, 2, 2, 97, 97), R=0, tensor_layout={"dO": "fa_transpose"},
             expected="correct accept"),
        item("PRB-E06a", "nightly", "claim", "G7 PRB-E06", (2, 4, 4, 4, 200, 200), R=0, expected="correct accept"),
        # GQA and MQA at model ratios
        item("PRB-H01", "pr", "probe", "G3", (1, 8, 2, 2, 64, 96), align="BR", R=0, expected=E_GROUPS),
        item("PRB-H02", "pr", "probe", "G3", (1, 10, 2, 2, 64, 64), R=0, expected=E_GROUPS),
        item("PRB-H03", "pr", "probe", "G3", (1, 6, 3, 3, 64, 64), R=0, expected=E_GROUPS),
        item("PRB-H04", "pr", "probe", "G3", (1, 14, 2, 2, 64, 64), R=0, expected=E_GROUPS),
        item("PRB-H05", "pr", "probe", "G3", (1, 12, 2, 2, 64, 64), R=0, expected=E_GROUPS),
        item("PRB-H06", "pr", "probe", "G3", (1, 8, 1, 1, 64, 64), R=0, expected=E_GROUPS),
        item("PRB-H07", "pr", "probe", "G3", (1, 16, 1, 1, 64, 64), R=0, expected=E_GROUPS),
        item("PRB-H08", "pr", "probe", "G4", (1, 4, 1, 2, 64, 64), expected=E_GROUPS),
        # bias broadcast classes (right-aligned broadcast, rule R8)
        *[item(f"PRB-{k}", "nightly", "probe", "T10 G11" + (" S48" if k == "B01" else ""), (*bb, 64, 96),
               bias=P.bias_spec(dims), expected=E_OR, **bias)
          for k, dims, bb in (("B01", "11", (1, 2, 2, 2)), ("B02", "1h", (1, 2, 2, 2)), ("B03", "b1", (2, 1, 1, 1)),
                              ("B04", "full", (2, 2, 2, 2)), ("B06", "r2", (1, 2, 2, 2)), ("B07", "r3", (1, 2, 2, 2)))],
        # rejection probes: frontend-legal, outside WF and EXT (layer U)
        item("PRB-M01", "quick", "probe", "A7 IVP6", nb, attrs=BOTH_FLAGS, verdict=FOV, **rj),
        item("PRB-R02", "pr", "probe", "A10 IVP7", nb, attrs={"dropout_probability": 0.1}, verdict=NOV, **rj),
        item("PRB-T04", "pr", "probe", "T4 DT2 IVP14", nb, dtype_over={"O": "fp16"}, verdict=NOV, **rj),
        item("PRB-R05", "pr", "probe", "T16 IVP8", nb, topology="DropoutScaleInv", verdict=NOV, **rj),
        item("PRB-M02", "pr", "probe", "A3 IV14", nb, attrs={"right_bound_raw": -2}, verdict=FOV, **rj),
        item("PRB-T05", "pr", "probe", "DT3 IVP2", nb, dtype_over={"Stats": "bf16"}, verdict=NOV, **rj),
        item("PRB-B10", "pr", "probe", "DT4 IVP15", (1, 2, 2, 2, 64, 96), topology="AdditiveMask",
             bias=P.bias_spec("11", "bool"), verdict=NOV, **rj),
        item("PRB-B15", "pr", "probe", "T11 IVP4", (1, 2, 2, 2, 64, 96), topology="DBiasOnly", dbias=True, verdict=FOV,
             **rj),
        item("PRB-M05", "pr", "probe", "A8 IVP10", (1, 4, 4, 4, 64, 64), attrs={"alibi_mask": True}, verdict=NOV, **rj),
        item("PRB-M06", "pr", "probe", "A8 IVP10", (1, 4, 4, 4, 64, 128), align="BR", R=0, attrs={"alibi_mask": True},
             verdict=NOV, **rj),
        item("PRB-M01n", "nightly", "probe", "A7 IVP6", (1, 2, 2, 2, 64, 96), "fp16", align="BR", attrs=BOTH_FLAGS,
             verdict=FOV, **rj),
        item("PRB-T04n", "nightly", "probe", "T4 DT2 IVP14", (1, 2, 2, 2, 64, 96), "fp16", align="BR",
             dtype_over={"O": "bf16"}, verdict=NOV, **rj),
        item("PRB-B15n", "nightly", "probe", "T11 IVP4", (1, 2, 2, 2, 64, 96), "fp16", align="BR",
             topology="DBiasOnly", dbias=True, verdict=FOV, **rj),
        item("PRB-M02f", "full", "probe", "A3 IV14", (1, 1, 1, 1, 1, 65537), attrs={"right_bound_raw": -2},
             verdict=FOV, **rj),
        item("PRB-R02f", "full", "probe", "A10 IVP7", (1, 1, 1, 1, 1, 65537), attrs={"dropout_probability": 0.5},
             verdict=NOV, **rj),
        item("PRB-T02", "pr", "probe", "DT2 IVP14", nb, dtype_over={"dQ": "fp32", "dK": "fp32", "dV": "fp32"},
             verdict=FOV, **rj),
        item("PRB-T03", "pr", "probe", "DT2 IVP14", nb, dtype_over={"dO": "fp16"}, verdict=FOV, **rj),
        item("PRB-U01", "pr", "probe", "T4 IVP1", nb, dims_over={"O": "dv_half", "dO": "dv_half"},
             verdict=FOV + " (gfx942, 2026-10-28: ASM_SDPA_ENGINE accepts it; the CPU reference throws on the O shape)", **rj),
        item("PRB-U03", "pr", "probe", "G10 IVP3", nb, dims_over={"dQ": "sq_half"},
             verdict=FOV + " (gfx942, 2026-10-28: ASM_SDPA_ENGINE accepts it; the CPU reference throws on the dQ shape)", **rj),
        item("PRB-U05", "pr", "probe", "G11 IVP5", (2, 1, 1, 1, 64, 96), bias=P.bias_spec([3, 1, 64, 96]),
             verdict="none: the reference reads bias[b] for b < B, so an accept that does the same passes",
             **rj, **bias),
        item("PRB-U09", "pr", "probe", "T13 IVP9", nb, topology="SeqLenQOnly",
             padding={"seq_len_q": [64], "seq_len_kv": None, "padding_mask": False}, verdict=NOV, **rj),
        item("PRB-U11", "pr", "probe", "DT1 IVP11", nb, dtype_over={t: "fp8_e4m3" for t in P.IO},
             verdict="FAIL (no oracle) on a wrong accept, if the harness fills FP8; otherwise SKIP", **rj),
        item("PRB-U13", "pr", "probe", "T7 IVP13", nb, tensor_layout={"dQ": "overlap_rows"},
             verdict=FOV + " (gfx942, 2026-10-28: ASM_SDPA_ENGINE accepts it and the aliased dQ fails the comparison)", **rj),
        # oracle not ready on the baseline (declared, not written)
        item("PRB-M04", "notrun", "probe", "A8 S51", (1, 4, 4, 4, 64, 64), R=0, attrs={"alibi_mask": True},
             oracle="14358+derivation", expected=E_OR),
        *[item(f"PRB-{k}", "notrun", "probe", "T11 G11 " + extra, (*bb, 64, 96), bias=P.bias_spec(dims), dbias=True,
               inputs=P.domain("bias_neg"), topology="AdditiveMaskDBias", oracle="14358",
               expected="dBias correct, or typed rejection")
          for k, dims, bb, extra in (("B11", "11", (1, 2, 2, 2), "H80"), ("B12", "1h", (1, 2, 2, 2), "S50"),
                                     ("B13", "b1", (2, 1, 1, 1), ""), ("B14", "full", (2, 2, 2, 2), ""))],
        item("PRB-B16", "notrun", "probe", "T11 G11", (1, 2, 2, 2, 64, 96),
             bias=dict(P.bias_spec("1h"), stride="rowpad8"), dbias=True, dbias_stride="rowpad8",
             inputs=P.domain("bias_neg"), topology="AdditiveMaskDBias", oracle="14358",
             expected="dBias correct at its strides, or typed rejection"),
        item("PRB-R01", "quick", "probe", "A10", nb, attrs={"dropout_probability": 0.0},
             expected="same as no dropout, or typed rejection"),
        item("PRB-R03", "notrun", "probe", "A10 T14 S54 DT6", (1, 1, 1, 1, 128, 128),
             attrs={"dropout_probability": 0.1}, topology="Philox", oracle="none",
             expected="typed rejection; an accept is unverifiable"),
        item("PRB-R04", "notrun", "probe", "T15 T16 G12 S53 DT6", (1, 2, 2, 2, 64, 96),
             dropout={"p": 0.2, "scale": 1.25, "scale_inv": 0.8, "mask_dims": "1h"}, inputs=P.domain("u01"),
             topology="DropoutMask", oracle="14358+derivation+HG4", expected=E_OR),
        item("PRB-P01", "notrun", "probe", "A9 T13 G13 DT6", (2, 2, 2, 2, 64, 96),
             padding={"seq_len_q": [64, 64], "seq_len_kv": [96, 40], "padding_mask": True}, expected=E_OR, **pad),
        item("PRB-P02", "notrun", "probe", "A9 T13 G13 S52", (3, 2, 2, 2, 64, 96),
             padding={"seq_len_q": [17, 64, 5], "seq_len_kv": [96, 40, 33], "padding_mask": True},
             expected="padded rows exactly 0", **pad),
        item("PRB-P03", "notrun", "probe", "G13", (3, 2, 2, 2, 64, 96),
             padding={"seq_len_q": [64, 0, 32], "seq_len_kv": [96, 40, 0], "padding_mask": True},
             expected="no fault; zero gradients", **pad),
        item("PRB-P04", "notrun", "probe", "AW6", (2, 2, 2, 2, 64, 96),
             padding={"seq_len_q": [64, 64], "seq_len_kv": [96, 40], "padding_mask": False},
             expected="padding semantics, or typed rejection", **pad),
        item("PRB-P06", "notrun", "probe", "T17 G14 S55", (2, 2, 2, 2, 64, 96), ragged={"q": [64, 40], "kv": [96, 50]},
             topology="Ragged", oracle="14358+derivation+HG20", expected=E_OR),
        # graph options and layouts
        item("PRB-X01", "nightly", "probe", "GO2", (1, 2, 2, 2, 256, 256), topology="FwdBwdChained", oracle="ready [I]",
             expected=E_OR),
        item("PRB-L01", "pr", "claim", "T18", (1, 2, 2, 2, 128, 128),
             tensor_layout={"Q": "qkv3", "K": "qkv3", "V": "qkv3"}, expected="correct accept"),
        item("PRB-L03", "pr", "probe", "T23 S45", (1, 1, 1, 1, 128, 128), alignment=8, expected=E_OR),
        item("PRB-G01", "quick", "claim", "AW8 H13 G5 G6", (1, 2, 2, 2, 1, 1), "bf16 fp16",
             expected="dQ = dK = 0; dV = dO"),
        # scale tensor claims
        item("CLM-S1", "quick", "claim", "T12 T22 A1 S39", nb, scale=P.RUNTIME_SCALE, **scale),
        item("CLM-S2", "pr", "claim", "T12 A2", (1, 1, 1, 1, 128, 256), align="BR", R=0, scale=P.RUNTIME_SCALE,
             **scale),
        item("CLM-S3", "pr", "claim", "T12 T22", (1, 1, 1, 1, 128, 128), scale={"tensor": "const", "value": 0.5},
             **scale),
        item("PRB-A01", "quick", "claim", "AW1 A2 A3", (1, 2, 2, 2, 64, 64), align="BR", R=0, inputs=P.domain("v01"),
             expected="same result as TOP_LEFT causal (C07)"),
        # hazard additions (§3.4)
        item("H97", "pr", "claim", "G5 G6 G7 A3", (2, 3, 3, 3, 128, 128), "fp16", R=0, expected="correct accept"),
        item("H98", "nightly", "claim", "A4", (1, 2, 2, 2, 256, 256), W=63, R=0, expected="correct accept"),
        item("H99", "nightly", "claim", "G8 G6 A2 A3", (1, 2, 2, 2, 64, 320, 192, 192), align="BR", R=0,
             expected="correct accept"),
        item("H100", "pr", "claim", "G5 A2 A3", (1, 2, 2, 2, 16, 256), align="BR", R=0, expected="correct accept"),
        item("H101", "nightly", "probe", "T10 G11", (1, 2, 2, 2, 72, 120), "fp16",
             bias=dict(P.bias_spec("11"), stride="rowpad8"), expected=E_OR, **bias),
        item("H102", "nightly", "probe", "T10 G11", (1, 2, 2, 2, 72, 120),
             bias=dict(P.bias_spec("11"), stride="transposed"),
             expected=E_OR, **bias),
        item("H96", "nightly", "claim", "V9", (1, 2, 2, 2, 72, 120), "fp16", inputs={"do": [-1e-4, 1e-4]},
             no_scale=True,
             expected="gradients within the dtype rounding"),
        # the 32-head model layer (M41) in the corpus encoding
        item("M41.n", "nightly", "claim", "M41 G2", (1, 32, 32, 32, 128, 128), **m41),
        item("M41.fc", "full", "claim", "M41 G2 V6", (1, 32, 32, 32, 512, 512), **m41),
    ]
    # fmt: on
    lt, gt = ("lt", "causal", "TL"), ("gt", "causal", "BR")
    for n, (cell, seed, dt) in enumerate(
        ((lt, 1, "bf16"), (lt, 2, "fp16"), (gt, 1, "fp16"), (gt, 2, "bf16")), 1
    ):
        # tolerance stability over the generator seed
        c = dict(CORE_BASE, **core_cell(cell))
        out.append(
            P.new(
                id=f"SD-{n:02d}",
                level="nightly",
                purpose="claim",
                items=["V10", "A3", "A2"],
                dtypes=(dt,),
                inputs=P.domain("v01"),
                seed=seed,
                expected="correct accept",
                **c,
            )
        )
    return out


def golden_items():
    return [P.golden_case(gid) for gid in P.GOLDEN]


def targeted_items():
    """Targeted products (§4): TP-A head dim 192 x core cells; TP-B layouts x
    core cells; TP-C GQA/MQA x core cells (nightly); TP-D the complete
    {bhsd, bshd} x {128, 192} x {bf16, fp16} x core cells not placed yet (full)."""
    out = []
    base = dict(CORE_BASE, inputs=P.domain("v01"))
    for k, cell in enumerate(CORE_CELLS, 1):
        tags = ["TP-A"] + (["PRB-D09"] if cell[1] == "window" else [])
        c = dict(base, Dqk=192, Dv=192, **core_cell(cell))
        out.append(
            P.new(
                id=f"TP-A{k:02d}",
                level="nightly",
                dtypes=(["bf16", "fp16"][k % 2],),
                items=tags,
                **c,
            )
        )
    for lay in ("bshd", "grads_bshd"):
        for k, cell in enumerate(CORE_CELLS, 1):
            dt = ["bf16", "fp16"][(k + (lay == "grads_bshd")) % 2]
            tid = f"TP-B{'s' if lay == 'bshd' else 'g'}{k:02d}"
            c = dict(base, **core_cell(cell))
            out.append(
                P.new(
                    id=tid,
                    level="nightly",
                    dtypes=(dt,),
                    items=["TP-B"],
                    layout=lay,
                    **c,
                )
            )
    for (hq, hk, hv), tag in (((4, 2, 2), "g"), ((4, 1, 1), "m")):
        for k, cell in enumerate(CORE_CELLS, 1):
            c = dict(base, Hq=hq, Hk=hk, Hv=hv, **core_cell(cell))
            out.append(
                P.new(id=f"TP-C{tag}{k:02d}", level="nightly", items=["TP-C"], **c)
            )
    placed = {("bhsd", 128, dt, cell) for cell in CORE_CELLS for dt in ("bf16", "fp16")}
    placed |= {
        ("bshd", 128, it["dtypes"][0], it["core"])
        for it in out
        if it["id"].startswith("TP-Bs")
    }
    placed |= {
        ("bhsd", 192, it["dtypes"][0], it["core"])
        for it in out
        if it["id"].startswith("TP-A")
    }
    k = 0
    for lay in ("bhsd", "bshd"):
        for D in (128, 192):
            for dt in ("bf16", "fp16"):
                for cell in CORE_CELLS:
                    if (lay, D, dt, cell) not in placed:
                        k += 1
                        c = dict(base, Dqk=D, Dv=D, **core_cell(cell))
                        out.append(
                            P.new(
                                id=f"TP-D{k:02d}",
                                level="full",
                                dtypes=(dt,),
                                items=["TP-D"],
                                layout=lay,
                                **c,
                            )
                        )
    for it in out:
        tags = list(it["items"])
        aw11 = ["AW11"] if it["id"].startswith("TP-C") and it["R"] is not None else []
        it["items"] = derived_features(it) + tags + aw11
    return out


def ap_item(n, av):
    """An all-pairs case: the axis values on the core base (B 2)."""
    W, R = P.AP_MASK[av["mask"]]
    Hq, Hk, Hv = P.AP_HEADS[av["heads"]]
    Dqk, Dv = P.AP_HDIM[av["hdim"]]
    Sq, Skv = P.AP_SEQ[av["seq"]]
    bias = P.AP_BIAS[av["bias"]]
    it = P.new(
        id=f"AP{n:03d}",
        level="nightly",
        items=["AP"],
        dtypes=(av["dtype"],),
        **g(2, Hq, Hk, Hv, Sq, Skv, Dqk, Dv),
        align=av["align"],
        W=W,
        R=R,
        layout=av["layout"],
        scale=P.AP_SCALE[av["scale"]],
        inputs=P.domain(av["vdom"]) if av["vdom"] != "default" else None,
        bias=bias,
        topology="AdditiveMask" if bias else None,
    )
    it["items"] += derived_features(it)
    return it


def plan_items():
    """Every plan item, with the merged shapes recorded on their carriers and
    the gate level of the not-run items."""
    items = (
        core_items()
        + quick_variant_items()
        + catalog_items()
        + model_items()
        + l_items()
        + regression_and_defect_items()
        + probe_items()
        + golden_items()
        + targeted_items()
    )
    dup = [k for k, v in Counter(it["id"] for it in items).items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate plan items: {dup}")
    by_id = {it["id"]: it for it in items}
    for carrier, shapes in MERGED.items():
        by_id[carrier]["items"] += [
            s for s in shapes.split() if s not in by_id[carrier]["items"]
        ]
    for it in items:
        if it["level"] == P.NOTRUN:
            default = "full" if it["set"] in ("M", "X") else "pr"
            it["target_level"] = (
                "out-of-level"
                if it.get("out_of_level")
                else TARGET_OF.get(it["id"], default)
            )
    return items


# --------------------------------------------------------------------------
# All-pairs generation (§4), seeded and greedy: stage 1 covers every pair of
# two claim values with claim-only cases; stage 2 every remaining required
# pair. The runnable cases up to nightly are the seeds, so only the pairs they
# miss get a case.
# --------------------------------------------------------------------------

AP_SEED = 20261026
AP_ROUNDS = 4
AP_CANDIDATES = 40


def _ap_candidate(rng, uncovered, stage):
    pool = [p for p in uncovered if stage == 2 or all(P.is_claim(n, v) for n, v in p)]
    if not pool:
        return None
    (a, va), (b, vb) = rng.choice(sorted(pool))
    case = {a: va, b: vb}
    order = [n for n in P.axis_names() if n not in case]
    rng.shuffle(order)
    for n in order:
        choices = [v for v in P.axis_values(n) if stage == 2 or P.is_claim(n, v)]
        rng.shuffle(choices)
        best, best_gain = None, -1
        for v in choices:
            trial = dict(case, **{n: v})
            if P.forbidden(trial) or (stage == 1 and P.probe_combination(trial)):
                continue
            gain = sum(1 for p in P.pairs_of(P.normalize(trial)) if p in uncovered)
            if gain > best_gain:
                best, best_gain = v, gain
        if best is None:
            return None
        case[n] = best
    case = P.normalize(case)
    if P.forbidden(case) or (stage == 1 and P.probe_combination(case)):
        return None
    return case


def allpairs_generate(seeds, seed):
    """seeds: (axis values, purpose). Returns the axis values of new cases."""
    rng = random.Random(seed)
    req = P.required_pairs()
    claim_req = P.claim_pairs(req)
    covered, covered_claim = set(), set()
    for s, pur in seeds:
        covered |= P.pairs_of(s) & req
        if P.claim_only(s, pur):
            covered_claim |= P.pairs_of(s) & claim_req
    out = []
    for stage in (1, 2):
        while True:
            uncovered = (claim_req - covered_claim) if stage == 1 else (req - covered)
            if not uncovered:
                break
            best, best_gain = None, 0
            for _ in range(AP_CANDIDATES):
                c = _ap_candidate(rng, uncovered, stage)
                if c is not None and len(P.pairs_of(c) & uncovered) > best_gain:
                    best, best_gain = c, len(P.pairs_of(c) & uncovered)
            if best is None:
                raise SystemExit(
                    f"all-pairs stage {stage}: cannot cover {sorted(uncovered)[:5]}"
                )
            out.append(best)
            covered |= P.pairs_of(best) & req
            if P.claim_only(best):
                covered_claim |= P.pairs_of(best) & claim_req
    return out


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


class Calibration:
    """The recorded magnitude estimates (sdpa_bwd_calibration.json). With
    `compute`, a missing or stale estimate is computed (NumPy)."""

    def __init__(self, path, compute=False):
        self.path = path
        self.entries = P.load_calibration(path)
        self.compute = compute
        self.used = set()
        self.computed = 0

    def get(self, c, dt, km):
        key = f"{c['id']}/{dt}"
        params = P.estimate_params(c, dt, km)
        fp = P.fingerprint(params)
        self.used.add(key)
        e = self.entries.get(key)
        if e is None or e["fp"] != fp:
            if not self.compute:
                raise SystemExit(
                    f"{key}: magnitude estimate missing or stale; rerun with --calibrate (needs numpy)"
                )
            e = dict(fp=fp, **P.estimate(params))
            self.entries[key] = e
            self.computed += 1
        return e

    def save(self):
        P.save_calibration(
            {k: v for k, v in self.entries.items() if k in self.used}, self.path
        )


def finalize(c, calib):
    """Resolve a case: topology, oracle, level, cost, purpose, tolerance
    class, id, axis values and its record."""
    dt = c["dtype"]
    c["topology"] = P.topology_of(c)
    c["oracle_status"] = P.oracle_of(c)
    c["runnable_now"] = c["oracle_status"] in P.READY and c["level"] in P.LEVELS
    if not c["runnable_now"] and c["level"] in P.LEVELS:
        c["target_level"], c["level"] = (
            c["level"],
            P.NOTRUN,
        )  # a gated case has no level
    cost = P.case_cost(c)
    c["n_pairs"], c["n_live"], c["flop"] = cost["n_pairs"], cost["n_live"], cost["flop"]
    c["est_s"], c["est_s_alt"], c["est_s_gpuref"] = P.seconds(c, dt)
    c["purpose_final"] = P.purpose(c, dt)
    if c.get("golden"):
        cls, info = c["tol_class"], {"source": "golden set"}
    elif c.get("rejection"):
        cls, info = P.provisional_class(dt), {"rule": "rejection probe: never compared"}
    elif P.estimable(c):
        km = P.kernel_model(c, dt, c["runnable_now"])
        cls, info, f = P.assign_class(c, dt, calib.get(c, dt, km))
        if f != 1.0:  # scale dO by 2^k: the gradients are linear in dO
            c["inputs"] = dict(c.get("inputs") or {}, do=_scaled(c.get("inputs"), f))
        if info.get("unassigned"):
            c["runnable_now"] = False
            c["oracle_status"] = P.UNASSIGNED_GATE
            if c["level"] in P.LEVELS:
                c["target_level"], c["level"] = c["level"], P.NOTRUN
    else:
        cls, info = P.provisional_class(dt), {"rule": "provisional (not modelled)"}
    c["tol_class"], c["tol_info"] = cls, info
    c["case_id"] = case_id(c, dt, cls)
    keep = c["runnable_now"] and not c.get("defect") and not c.get("rejection")
    c["axis_values"] = P.axis_projection(c, dt) if keep else None
    c["record"] = record(c)
    c["legality"] = c["record"]["legality"]


def _scaled(inputs, f):
    d0 = (inputs or {}).get("do", [-1.0, 1.0])
    if isinstance(d0, dict):
        return {"kind": "fixed", "value": d0["value"] * f}
    return [d0[0] * f, d0[1] * f]


def expand(it, ap=False):
    """One case per dtype; each also carries the plain op's items."""
    out = []
    for dt in it["dtypes"]:
        c = dict(it, dtype=dt, items=list(it["items"]))
        extra = list(P.BASE_ITEMS)
        stats_dtype = (c.get("dtype_over") or {}).get("Stats")
        packed_stats = c.get("stats", "packed") == "packed" and not stats_dtype
        if ap or (not c.get("golden") and packed_stats):
            extra += ["T6", "DT3"]
        if not c.get("golden"):
            extra.append("V2")
        c["items"] += extra
        out.append(c)
    return out


def build(calib):
    """Every case, runnable and not run, all-pairs cases included."""
    cases = [c for it in plan_items() for c in expand(it)]
    for c in cases:
        finalize(c, calib)
    n_ap = 0
    for rnd in range(AP_ROUNDS):
        seeds = [
            (c["axis_values"], c["purpose_final"])
            for c in cases
            if c["level"] != "full" and c["axis_values"]
        ]
        ap = allpairs_generate(seeds, AP_SEED + rnd)
        if not ap:
            break
        for av in ap:
            n_ap += 1
            for c in expand(ap_item(n_ap, av), ap=True):
                finalize(c, calib)
                cases.append(c)
    dup = [k for k, v in Counter(c["case_id"] for c in cases).items() if v > 1]
    if dup:
        raise SystemExit(f"duplicate case ids: {dup[:10]}")
    return cases


# --------------------------------------------------------------------------
# Case records (the plan's case table)
# --------------------------------------------------------------------------


def spec_inputs(c):
    ins = OrderedDict([("Q", [-1, 1]), ("K", [-1, 1]), ("V", [-1, 1]), ("dO", [-1, 1])])
    for k, v in (c.get("inputs") or {}).items():
        ins[P.INPUT_NAME[k]] = v
    if c.get("bias") and "Bias" not in ins:
        ins["Bias"] = [-1, 1]
    s = c.get("scale")
    if isinstance(s, dict) and s["tensor"] == "runtime":
        ins["Scale"] = {"kind": "fixed", "value": s["value"]}
    if c.get("dropout"):
        ins["DropoutScale"] = {"kind": "fixed", "value": c["dropout"]["scale"]}
        ins["DropoutScaleInv"] = {"kind": "fixed", "value": c["dropout"]["scale_inv"]}
        ins["DropoutMask"] = {
            "kind": "bernoulli",
            "keep": 1 - c["dropout"]["p"],
            "needs": "HG4",
        }
    if c.get("topology") == "DropoutScaleInv":
        ins["DropoutScaleInv"] = {"kind": "fixed", "value": 0.8}
    pad = c.get("padding")
    if pad:
        ins["SeqLenQ"] = {"kind": "list", "values": pad["seq_len_q"], "needs": "HG4"}
        if pad.get("seq_len_kv") is not None:
            ins["SeqLenKv"] = {
                "kind": "list",
                "values": pad["seq_len_kv"],
                "needs": "HG4",
            }
    return ins


def record(c):
    """The case as a plan record: what the plan tables, the legality layers
    and the sweep writer read."""
    dt = c["dtype"]
    tensors = P.tensor_table(c, dt)
    s = c.get("scale", ISQ)
    d = OrderedDict(id=c["case_id"], plan_item=c["id"], level=c["level"])
    if c.get("target_level"):
        d["target_level"] = c["target_level"]
    d.update(
        topology=c["topology"],
        purpose=c["purpose_final"],
        plan_items=sorted(set([c["id"]] + c["items"])),
    )
    d.update(
        set=c["set"],
        dtype=dt,
        **{k: c[k] for k in P.GEOMETRY},
    )
    d["diagonal_alignment"] = {"TL": "TOP_LEFT", "BR": "BOTTOM_RIGHT"}[
        c.get("align", "TL")
    ]
    d.update(W=c.get("W"), right_bound=c.get("R"))
    d["left_bound_written"] = P.left_bound_written(c)
    d["right_bound_written"] = P.bounds_written(c)[1]
    d["strides"] = {k: dict(v) for k, v in tensors.items()}
    d.update(layout=c.get("layout", "bhsd"), alignment=c.get("alignment", 16))
    if s == ISQ:
        d["scale"] = {"mode": "attn_scale_value", "value": 1 / math.sqrt(c["Dqk"])}
    elif s == "unset":
        d["scale"] = {"mode": "unset", "value": None}
    elif isinstance(s, dict):
        d["scale"] = {"mode": "tensor_" + s["tensor"] + "_pbv", "value": s["value"]}
    else:
        d["scale"] = {"mode": "attn_scale_value", "value": s}
    skip = ("right_bound_raw", "explicit_unbounded")
    d["attributes"] = {k: v for k, v in (c.get("attrs") or {}).items() if k not in skip}
    d["inputs"] = spec_inputs(c)
    if isinstance(s, dict) and s["tensor"] == "runtime":
        d["tensor_patches"] = [
            {
                "uid": P.UID["Scale"],
                "set": {"is_runtime_pass_by_value": True},
                "remove": ["value_type", "value"],
            }
        ]
    for k in (
        "bias",
        "dbias",
        "dbias_stride",
        "dropout",
        "padding",
        "ragged",
        "dtype_over",
        "dims_over",
    ):
        if c.get(k):
            d[k] = (
                {"p": c[k]["p"], "mask_dims": c[k]["mask_dims"]}
                if k == "dropout"
                else c[k]
            )
    d.update(
        tol_class=c["tol_class"],
        oracle_status=c["oracle_status"],
        runnable_now=c["runnable_now"],
    )
    d["expected_rejection"] = bool(c.get("rejection"))
    if c.get("rejection"):
        d["verdict_before_14346"] = c.get("verdict")
    d["golden"] = bool(c.get("golden"))
    if c.get("golden"):
        d.update(
            golden_id=c["golden"],
            golden_pointer_gate=c.get("golden_pointer_gate", "GT1"),
        )
    if c.get("defect"):
        d["defect"] = c["defect"]
        if not (c["defect"] == "DEF-03" and dt == "bf16"):
            d["entry_gate"] = DEFECT_GATE[c["defect"]]
    if c.get("regression_of"):
        d["regression_of"] = c["regression_of"]
    if c.get("repeat"):
        d["repeat"] = c["repeat"]
    d["seed"] = c.get("seed", 0)
    d["expected"] = c.get("expected") or (
        "correct accept" if c["purpose_final"] == "claim" else E_ANY
    )
    d.update(
        {
            k: c[k]
            for k in ("est_s", "est_s_alt", "est_s_gpuref", "flop", "n_pairs", "n_live")
        }
    )
    d["bytes"] = P.fp32_bytes(tensors)
    d["axis_values"] = c["axis_values"]
    layers = {sem: P.layer(d, sem) for sem in P.SEMANTICS}
    d["legality"] = {sem: layers[sem][0] for sem in P.SEMANTICS}
    d["legality"]["why"] = layers["develop"][1]
    return d


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def dump(obj):
    return json.dumps(obj, indent=2) + "\n"


def support_json(old, ids):
    """An empty sidecar, or the existing one without the claims of removed
    cases (in the canonical form verify_support_claims.py checks)."""
    doc = json.loads(old) if old is not None else {"claims": {}, "version": 1}
    for engine, groups in list(doc.get("claims", {}).items()):
        kept = []
        for grp in groups:
            grp["cases"] = sorted(i for i in grp["cases"] if i in ids)
            if grp["cases"]:
                kept.append(grp)
        if kept:
            doc["claims"][engine] = sorted(kept, key=lambda grp: grp["cases"][0])
        else:
            del doc["claims"][engine]
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def sweeps(cases):
    """{directory: (topology, cases)} of the written cases, heaviest first
    within a sweep (GTest shards by index, so heavy cases alternate)."""
    out = defaultdict(list)
    for c in cases:
        if c["runnable_now"]:
            out[(c["level"], c["topology"])].append(c)
    keys = sorted(
        out, key=lambda k: (P.LEVELS.index(k[0]), list(P.TOPOLOGIES).index(k[1]))
    )
    return OrderedDict(
        (
            Path(P.LEVEL_DIR[lv]) / OP / topo,
            (topo, sorted(out[(lv, topo)], key=lambda c: -c["est_s"])),
        )
        for lv, topo in keys
    )


def write_tree(cases, bundle_dir, dry_run):
    """The generator owns <level>/SdpaBackward/<topology>/{graph.template,
    sweep,support}.json; other files (golden data) are left alone."""
    files = {}
    for rel, (topo, cs) in sweeps(cases).items():
        tpl = template(topo)
        need = placeholders(tpl)
        names = set(template_tensors(topo))
        entries = []
        for c in cs:
            if set(c["record"]["strides"]) != names:
                raise SystemExit(
                    f"{rel}/{c['case_id']}: tensors {sorted(c['record']['strides'])} != {sorted(names)}"
                )
            e = sweep_case(c)
            have = {"io_data_type"} | {
                "attributes." + k for k in e["values"]["attributes"]
            }
            if have != need:
                raise SystemExit(
                    f"{rel}/{c['case_id']}: values {sorted(have ^ need)} do not match the template"
                )
            entries.append(e)
        files[rel / "graph.template.json"] = dump(tpl)
        files[rel / "sweep.json"] = dump({"version": 1, "cases": entries})
        sup = bundle_dir / rel / "support.json"
        ids = {c["case_id"] for c in cs}
        files[rel / "support.json"] = support_json(
            sup.read_text() if sup.exists() else None, ids
        )
    if dry_run:
        return files
    written = {f.parent for f in files}
    for lv in P.LEVELS:
        for topo in P.TOPOLOGIES:
            rel = Path(P.LEVEL_DIR[lv]) / OP / topo
            d = bundle_dir / rel
            if rel not in written and d.exists():
                for name in ("graph.template.json", "sweep.json", "support.json"):
                    (d / name).unlink(missing_ok=True)
                if not any(d.iterdir()):
                    d.rmdir()
    for rel, text in files.items():
        path = bundle_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return files


TOML_BEGIN = "# BEGIN SdpaBackward tolerance classes, written by generate_sdpa_bwd_bundles.py --toml"
TOML_END = "# END SdpaBackward tolerance classes"


def toml_rows():
    """(class, dtype, filter, atol, rtol, description): one row per class and
    dtype. The filter anchors on the dtype token and on the class token that
    ends every SdpaBackward case id."""
    return [
        (cls, dt, f"*_SdpaBackward_*.*_{P.DTYPE_TOKEN[dt]}_*_{tok}", atol, rtol, desc)
        for cls, (tok, tols, desc) in P.CLASSES.items()
        for dt, (atol, rtol) in tols.items()
    ]


def toml_block():
    lines = [
        TOML_BEGIN,
        "# Plan sdpa-bwd.md §6.1. Each SdpaBackward case id ends with its tolerance class token",
        "# after its dtype token, so every id matches exactly one row. The class follows from the",
        "# case's shape, band, scale and value domain and a NumPy estimate of max|ref| per",
        "# gradient: atol <= 0.1 x the smallest non-zero max|ref| (the guard), and a",
        "# same-precision kernel model passes with margin >= 2.5 against an FP32-accumulating",
        "# reference. rtol is two ulps of the dtype. Do not edit: rerun the generator.",
    ]
    for cls, dt, filt, atol, rtol, desc in toml_rows():
        lines += [
            "",
            f"# class {cls}, {dt}: {desc}",
            "[[tolerance_overrides]]",
            f'filters = ["{filt}"]',
        ]
        lines += [f"atol = {atol!r}", f"rtol = {rtol!r}"]
    return "\n".join(lines + [TOML_END]) + "\n"


def update_toml(path, dry_run):
    """Rewrite the generated block of an engine TOML. True if it changed."""
    text = path.read_text()
    i, j = text.find(TOML_BEGIN), text.find(TOML_END)
    if i < 0 or j < i:
        raise SystemExit(f"{path}: no '{TOML_BEGIN}' ... '{TOML_END}' block")
    new = text[:i] + toml_block() + text[j + len(TOML_END) + 1 :]
    if not dry_run and new != text:
        path.write_text(new)
    return new != text


def toml_matches(name, rows):
    return [k for k, row in enumerate(rows) if fnmatch.fnmatchcase(name, row[2])]


def gtest_name(c):
    return (
        f"{P.LEVEL_DIR.get(c['level'], c['level'])}_{OP}_{c['topology']}.{c['case_id']}"
    )


def write_spec(cases, path):
    spec = OrderedDict(
        format_version=2,
        op=OP,
        generator=GENERATOR,
        plan="dnn-providers/integration-tests/docs/test-plans/sdpa-bwd.md",
        window_semantics=P.WINDOW_SEMANTICS,
        cases=[c["record"] for c in cases],
    )
    Path(path).write_text(json.dumps(spec, indent=1) + "\n")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def report(cases):
    run = [c for c in cases if c["runnable_now"]]
    topos = [t for t in P.TOPOLOGIES if any(c["topology"] == t for c in cases)]
    print("\nCases by level and topology (notrun: declared, not written)")
    print(f"  {'level':<8}" + "".join(f"{t[:15]:>16}" for t in topos) + f"{'cases':>8}")
    for lv in P.LEVELS + (P.NOTRUN,):
        cnt = Counter(c["topology"] for c in cases if c["level"] == lv)
        print(
            f"  {lv:<8}"
            + "".join(f"{cnt[t]:>16}" for t in topos)
            + f"{sum(cnt.values()):>8}"
        )
    print("\nBy purpose and size class")
    for lv in P.LEVELS:
        cs = [c for c in run if c["level"] == lv]
        pur = dict(sorted(Counter(c["purpose_final"] for c in cs).items()))
        rej = sum(1 for c in cs if c.get("rejection"))
        print(
            f"  {lv:<8} {pur}  rejection probes {rej}  {dict(sorted(Counter(c['set'] for c in cs).items()))}"
        )

    def rule(c):
        return P.budget_seconds(c["est_s"], c["est_s_alt"])

    print("\nPer-case ceilings (rule: max(pilot, alt), alt x 1.1 for fp16)")
    over = [
        (c["case_id"], c["level"], rule(c))
        for c in run
        if rule(c) > P.CEIL_S[c["level"]]
        or c["record"]["bytes"] > (P.CEIL_BYTES[c["level"]] or float("inf"))
    ]
    print(f"  cases over their level ceiling: {len(over)} {over[:5]}")
    print(
        "\nCumulative budgets, s, every written case accepted, CPU reference. pilot: the binding"
    )
    print(
        "model; rule: the per-case rule above; +gates: rule plus the not-run cases whose gate is"
    )
    print(
        "a harness item (no GPU reference), at their target level. The 5 % reserve applies to rule."
    )
    print(
        f"  {'level':<8}{'pilot':>9}{'rule':>9}{'+gates':>9}{'budget':>9}{'reserve':>10}"
    )
    cum = cum_r = cum_g = 0.0
    for lv in P.LEVELS:
        cs = [c for c in run if c["level"] == lv]
        gated = [
            x for x in cases if not x["runnable_now"] and x.get("target_level") == lv
        ]
        gated = [x for x in gated if "14358" not in x["oracle_status"]]
        cum += sum(c["est_s"] for c in cs)
        cum_r += sum(rule(c) for c in cs)
        cum_g += sum(rule(c) for c in cs + gated)
        lim = P.CUM_S[lv] * (1 - P.RESERVE)
        verdict = (
            "OK"
            if cum_r <= lim
            else ("over the reserve" if cum_r <= P.CUM_S[lv] else "OVER")
        )
        print(
            f"  {lv:<8}{cum:9.1f}{cum_r:9.1f}{cum_g:9.1f}{P.CUM_S[lv]:9.0f}{lim:10.1f}  {verdict}"
        )

    print("\nAll-pairs proof (quick..nightly, runnable cases with axis values)")
    r = P.pairwise_check(
        [
            (c["axis_values"], c["purpose_final"])
            for c in run
            if c["level"] != "full" and c["axis_values"]
        ]
    )
    print(
        f"  required {r['required']}, covered {r['covered']}, uncovered {len(r['uncovered'])}; claim pairs "
        f"{r['claim_required']}, covered by claim-only cases {r['claim_covered']}; forbidden {len(r['forbidden'])}"
    )

    print("\nLegality under develop before SEM-001, after #14483 (SEM-001, R13), after #14358")
    for sem in P.SEMANTICS:
        print(f"  {sem:<12} {dict(Counter(c['legality'][sem] for c in cases))}")
    viol = [
        (c["case_id"], v) for c in cases for v in P.legality_violations(c["record"])
    ]
    print(f"  violations: {len(viol)} {viol[:5]}")
    later = sorted(
        {
            c["id"]
            for c in cases
            for s in ("after_14287", "after_14358")
            if c["legality"][s] == "illegal"
        }
    )
    print(f"  illegal under a later frontend (leave the bundles then): {later}")

    print("\nTolerance classes (written cases)")
    print(
        f"  {dict(sorted(Counter((c['dtype'], c['tol_class']) for c in run).items()))}"
    )
    print(
        f"  escalated: {sum(1 for c in cases if c['tol_info'].get('escalated_from'))}; "
        f"dO scaled: {sum(1 for c in cases if c['tol_info'].get('dO_scale'))}; "
        f"no class (not run until HG6): {[(c['id'], c['dtype']) for c in cases if c['tol_info'].get('unassigned')]}"
    )
    rows = toml_rows()
    multi = [c["case_id"] for c in cases if len(toml_matches(gtest_name(c), rows)) != 1]
    print(
        f"  TOML rows {len(rows)}; case ids matching other than one row: {len(multi)} {multi[:3]}"
    )

    print("\nNot run (declared; written when the gate opens)")
    by_gate = defaultdict(list)
    for c in cases:
        if not c["runnable_now"]:
            by_gate[c["oracle_status"]].append(c["id"])
    for gate, ids in sorted(by_gate.items()):
        what, track = P.GATES[gate]
        ids = sorted(set(ids))
        print(
            f"  {len(ids):3d}  {what}  [tracking: {track}]  {ids[:6]}{' ...' if len(ids) > 6 else ''}"
        )

    print("\nTraceability")
    covered = {i for c in cases for i in c["record"]["plan_items"]}
    missing = [
        f for f in P.FEATURES if f not in covered and f not in P.FEATURE_DISPOSITION
    ]
    print(f"  features without a case or a disposition: {missing}")
    print(
        f"  longest case id: {max(len(c['case_id']) for c in cases)} characters (cap {ID_CAP})"
    )
    print(
        f"\n  written: {len(run)} cases in {len(sweeps(cases))} sweeps; declared, not run: {len(cases) - len(run)}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--toml",
        type=Path,
        action="append",
        default=[],
        help="engine TOML whose generated block to rewrite",
    )
    ap.add_argument(
        "--spec",
        type=Path,
        help="write the case records (the plan's case table) as JSON",
    )
    ap.add_argument(
        "--calibrate",
        action="store_true",
        help="compute missing or stale magnitude estimates (numpy)",
    )
    ap.add_argument("--calibration", type=Path, default=P.CALIBRATION_FILE)
    args = ap.parse_args()
    calib = Calibration(args.calibration, compute=args.calibrate)
    cases = build(calib)
    if args.calibrate and not args.dry_run:
        calib.save()
        print(
            f"calibration: {calib.computed} estimates computed, {len(calib.used)} recorded"
        )
    write_tree(cases, args.bundle_dir, args.dry_run)
    for path in args.toml:
        update_toml(path, args.dry_run)
    if args.spec:
        write_spec(cases, args.spec)
    if args.report or args.dry_run:
        report(cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
