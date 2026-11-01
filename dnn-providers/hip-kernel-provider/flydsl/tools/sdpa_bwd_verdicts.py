# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Expected verdicts of the gfx11-generic SDPA backward split engine, derived without the engine.

    python tools/sdpa_bwd_verdicts.py            # rewrite tests/data/sdpa_bwd_gfx11_verdicts.csv
    python tools/sdpa_bwd_verdicts.py --check    # regenerate and diff against the committed CSV
    python tools/sdpa_bwd_verdicts.py --import-bundle BUNDLE_DIR
                                                 # refresh tests/data/sdpa_bwd_gfx11_selection.csv from the bundle

One row per graph spec (columns in ``COLUMNS``); the verdict is ``served`` with the 12-hex spec hash of each slot
object, or ``declined`` with the rule ID of the matcher (``no_row`` when no matcher rule names the decline: the graph
matcher accepts it and no gfx11 object serves it). Every expectation comes from one of:

* the RDNA bundle's routing: ``tests/data/sdpa_bwd_gfx11_selection.csv`` is the ``handover/selection.json`` of the
  RDNA kernel bundle the gfx11-generic objects were taken from, one row per entry with the hash of its ``exact: false`` row and of its default row;
  ``--import-bundle`` rewrites it from the bundle and cross-checks the vendored ``check_request`` below against the
  bundle's ``tools/selection.py``;
* the bundle's refusals: ``to_rung`` and ``check_request`` are the bundle's ``source/tools/selection.py`` verbatim
  (minus the CSV table helpers), the vendored minimum;
* the matcher's rule documentation (the doc comments of ``FlydslSdpaBwdCommon.hpp`` and
  ``FlydslSdpaBwdSplitNative.cpp``), restated as the predicates and ``RULE_CAUSE`` below. No engine code runs here.

Graph model of a row: Q/K have row stride ``stride_qk`` (head stride ``seq * stride_qk``, batch stride
``heads * head stride``), V/O/dO/dV have ``stride_v``; ``bias`` names the dims the bias is broadcast over
(``none``, ``full`` = no broadcast, ``b``, ``h``, ``bh``, ``q``); ``mask`` + windows follow cuDNN (left/right bound,
``-1`` = unbounded, a causal flag implies right bound 0).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import importlib.util
import json
import sys
from pathlib import Path

FLYDSL = Path(__file__).resolve().parents[1]
DATA = FLYDSL / "tests" / "data"
VERDICTS_CSV = DATA / "sdpa_bwd_gfx11_verdicts.csv"
SELECTION_CSV = DATA / "sdpa_bwd_gfx11_selection.csv"
CONTENT = (
    FLYDSL.parent
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / "FlyDSL"
)

COLUMNS = (
    "id", "source", "dtype", "dqk", "dv", "mask", "window_left", "window_right", "dropout", "bias", "dbias",
    "varlen", "hq", "hk", "b", "sq", "sk", "stride_qk", "stride_v", "seed_form",
    "verdict", "prep", "dkdv", "dq", "reduce", "rule", "basis",
)  # fmt: skip
SELECTION_COLUMNS = (
    "kernel",
    "dtype",
    "tile",
    "class",
    "dropout",
    "store_db",
    "exact_false_hash",
    "default_hash",
)
SLOT_COLUMNS = ("prep", "dkdv", "dq", "reduce")
INT_COLUMNS = (
    "dqk",
    "dv",
    "window_left",
    "window_right",
    "dropout",
    "dbias",
    "hq",
    "hk",
    "b",
    "sq",
    "sk",
    "stride_qk",
    "stride_v",
)

# ---------------------------------------------------------------------------
# Bundle routing: the bundle's `source/tools/selection.py`, vendored verbatim.
# ---------------------------------------------------------------------------

GENERIC_LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)
SHIPPED_DTYPES = ("f16", "bf16", "fp32")
REASONS = (
    "hd_lt_1",
    "hd_gt_256",
    "hdim_gt_tile",
    "dtype_not_shipped",
    "tile_not_rung",
    "bias_varlen",
    "bias_q_bcast_dbias",
    "bias_mask",
)


class RoutingRefusal(ValueError):
    def __init__(self, code, detail):
        assert code in REASONS, code
        super().__init__(f"{code}: {detail}")
        self.code = code


def to_rung(hd, tile=None, dtype=None):
    if dtype is not None and dtype not in SHIPPED_DTYPES:
        raise RoutingRefusal(
            "dtype_not_shipped", f"dtype {dtype!r} is not one of {SHIPPED_DTYPES}"
        )
    if hd < 1:
        raise RoutingRefusal("hd_lt_1", f"head dim {hd} < 1")
    if hd > GENERIC_LADDER[-1]:
        raise RoutingRefusal(
            "hd_gt_256",
            f"head dim {hd} > {GENERIC_LADDER[-1]}: the ladder ends at {GENERIC_LADDER[-1]}",
        )
    if tile is None:
        rung = next(r for r in GENERIC_LADDER if r >= hd)
    else:
        if tile not in GENERIC_LADDER:
            raise RoutingRefusal(
                "tile_not_rung", f"tile {tile} is not a ladder rung {GENERIC_LADDER}"
            )
        if hd > tile:
            raise RoutingRefusal("hdim_gt_tile", f"head dim {hd} > tile {tile}")
        rung = tile
    return rung, hd != rung


def check_request(
    *,
    dtype,
    hd_qk,
    hd_v=None,
    tile=None,
    bias=False,
    varlen=False,
    causal=False,
    window=False,
    bias_bcast="",
    dbias=False,
):
    hd_v = hd_qk if hd_v is None else hd_v
    rung, _ = to_rung(max(hd_qk, hd_v), tile, dtype)
    to_rung(min(hd_qk, hd_v))
    if bias:
        if varlen:
            raise RoutingRefusal(
                "bias_varlen",
                "bias with varlen (cu_seqlens) is not served; use dense layouts for a biased backward",
            )
        if causal or window:
            raise RoutingRefusal(
                "bias_mask",
                "bias with a causal or window mask is not served (AOTriton disables it; bias is its own class here)",
            )
        if dbias and "q" in bias_bcast:
            raise RoutingRefusal(
                "bias_q_bcast_dbias",
                "a q-broadcast bias cannot return dBias (a zero dB q stride means 'not requested'); pass a full-shape bias",
            )
    return rung, (hd_qk, hd_v) != (rung, rung)


# ---------------------------------------------------------------------------
# Matcher rules: rule ID -> the `[cause]` the engine logs ("declined the graph [<cause>]").
# Each ID is defined by the doc comment named in the comment; the cause word is the `Decline{cause, ...}` of
# that rule in FlydslSdpaBwdCommon.hpp.
# ---------------------------------------------------------------------------

RULE_CAUSE = {
    "SH-1": "shape",  # checkShapes doc: "every extent is positive"
    "DT-4": "head_size",  # checkHeads doc: max(Dqk, Dv) above MAX_HEAD_DIM (512) declines [head_size]
    "DT-6": "gqa",  # checkHeads doc: a head count that is not a multiple of the key/value heads declines [gqa]
    "DT-8": "layout",  # checkTensorLayout doc: head dim not a multiple of 8 (4 for FLOAT) needs smallest stride >= ceil8(D)
    "DT-9": "layout",  # checkTensorLayout / slabFits doc: the per-head slab fits 32-bit byte offsets
    "LY-3": "layout",  # checkTensorLayout doc: every stride of an axis of more than one element is a multiple of 8 (4 for FLOAT)
    "LY-6": "layout",  # checkTensorLayout doc: a multi-row sequence axis has a non-zero stride
    "MW-3": "mask",  # checkMask doc: a bound below -1 declines
    "MW-9b": "mask",  # checkMask doc: left_bound 0 with right bound 0 declines
    "MW-9c": "mask",  # checkMask doc: left_bound 0 with an unbounded right bound declines
    "MW-10": "mask",  # checkBias doc: bias with a mask declines
    "MW-13": "mask",  # checkMask doc: a window with bottom-right alignment under padding_mask declines
    "PV-3": "varlen",  # checkVarlen doc: a ragged offset on any operand declines (packed varlen)
    "PV-10": "varlen",  # checkBias doc: bias with padding_mask declines
    "BI-4": "dbias",  # checkBias doc: dBias with a row-broadcast bias declines
    "BI-6": "dbias",  # checkBias doc: dBias without bias declines
}

# `rule` token of a decline no matcher rule names: the graph matcher accepts the graph and no gfx11 row serves it.
NO_ROW = "no_row"

# Rules the matcher documents as accepting (served rows cite them in `basis`):
#   LY-4 packed Q/K/V views and shared dQ/dK/dV buffers; PV-1 padding_mask with both seq_len tensors; DR-4/DR-6 INT32/INT64
#   device or pass-by-value seed and offset; BI-3 bias broadcast over batch or head with dBias; MW-5..8, MW-9a windows;
#   MW-11 dropout with a bottom-right causal mask; DT-5 unequal Dqk and Dv; DT-6 any group size; DT-8 odd D at stride ceil8(D).


def bundle_rule(code, row):
    """The matcher rule ID of a bundle refusal; NO_ROW when the matcher accepts the graph and no object serves it.

    hd_lt_1: the matcher's SH-1 (non-positive extent). hd_gt_256: the matcher's MAX_HEAD_DIM is 512 (the gfx950 ladder), so only
    DT-4 (above 512) names the decline; 257..512 passes the matcher and no gfx11 object serves it (NO_ROW).
    bias_varlen: PV-10 (bias with padding_mask). bias_mask: MW-10. bias_q_bcast_dbias: BI-4 (q-broadcast = row-broadcast bias).
    hdim_gt_tile, tile_not_rung, dtype_not_shipped: a request parameter or an FP8/FLOAT64-class dtype, never a row of this table.
    """
    if code == "hd_lt_1":
        return "SH-1"
    if code == "hd_gt_256":
        return "DT-4" if max(row["dqk"], row["dv"]) > 512 else NO_ROW
    return {"bias_varlen": "PV-10", "bias_mask": "MW-10", "bias_q_bcast_dbias": "BI-4"}[
        code
    ]


# Refusal codes with no graph row, and why.
NOT_A_ROW = {
    "hdim_gt_tile": "the tile is a request parameter; a graph carries no tile (the engine routes the tile itself, DT-5)",
    "tile_not_rung": "the tile is a request parameter; a graph carries no tile",
    "dtype_not_shipped": "the table's dtype column is f16/bf16/fp32 only; a non-shipped dtype (FP8 etc.) declines under DT-1/DT-3",
}


def vec(dtype):
    return 4 if dtype == "fp32" else 8


def ceil_to(x, v):
    return -(-x // v) * v


INT32_LIMIT = 2147483647  # the 32-bit byte offset bound of slabFits (DT-9)


def layout_rule(dtype, extents, stride, dim):
    """First layout rule of one operand [b, heads, seq, dim] at row stride `stride`, in the matcher's order (LY-3, LY-6, DT-8, DT-9)."""
    v = vec(dtype)
    b, heads, seq = extents
    strides = (heads * seq * stride, seq * stride, stride)  # batch, head, seq
    smallest = None
    for axis, (extent, s) in enumerate(zip((b, heads, seq), strides)):
        if extent == 1:
            continue
        if s % v:
            return "LY-3"
        if axis == 2 and s == 0:
            return "LY-6"
        if s > 0:
            smallest = s if smallest is None else min(smallest, s)
    padded = ceil_to(dim, v)
    if dim % v and (smallest is None or smallest < padded):
        return "DT-8"
    elem = 4 if dtype == "fp32" else 2
    reach = (seq - 1) * stride
    if reach > INT32_LIMIT or (reach + padded) * elem > INT32_LIMIT:
        return "DT-9"
    return ""


def matcher_rule(r):
    """The first rule of the matcher documentation that declines the graph, '' if none."""
    if min(r["dqk"], r["dv"], r["hq"], r["hk"], r["b"], r["sq"], r["sk"]) <= 0:
        return "SH-1"
    if max(r["dqk"], r["dv"]) > 512:
        return "DT-4"
    if r["hq"] % r["hk"]:
        return "DT-6"
    d = r["dtype"]
    for extents, stride, dim in (
        ((r["b"], r["hq"], r["sq"]), r["stride_qk"], r["dqk"]),
        ((r["b"], r["hk"], r["sk"]), r["stride_qk"], r["dqk"]),
        ((r["b"], r["hk"], r["sk"]), r["stride_v"], r["dv"]),
        ((r["b"], r["hq"], r["sq"]), r["stride_v"], r["dv"]),
    ):
        if rule := layout_rule(d, extents, stride, dim):
            return rule
    left, right = r["window_left"], r["window_right"]
    if r["mask"] != "none" and right == -1:
        right = 0
    if left < -1 or right < -1:
        return "MW-3"
    if left == 0 and right == 0:
        return "MW-9b"
    if left == 0 and right == -1:
        return "MW-9c"
    if (
        r["varlen"] == "padded"
        and r["mask"] == "causal_br"
        and (left != -1 or right > 0)
    ):
        return "MW-13"
    if r["varlen"] == "packed":
        return "PV-3"
    if r["dbias"] and r["bias"] == "none":
        return "BI-6"
    if r["bias"] != "none":
        if r["varlen"] != "none":
            return "PV-10"
        if left != -1 or right != -1:
            return "MW-10"
        if r["dbias"] and r["bias"] == "q":
            return "BI-4"
    return ""


# ---------------------------------------------------------------------------
# Routing table (vendored selection.json) and in-tree hashes
# ---------------------------------------------------------------------------


def load_selection(path=SELECTION_CSV):
    """-> {(kernel, dtype, tile, class, dropout, store_db): (exact_false_hash, default_hash)}; tile 0 and store_db '' where absent."""
    table = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == SELECTION_COLUMNS, reader.fieldnames
        for e in reader:
            key = (
                e["kernel"],
                e["dtype"],
                int(e["tile"] or 0),
                e["class"],
                int(e["dropout"]),
                e["store_db"],
            )
            assert key not in table, key
            table[key] = (e["exact_false_hash"], e["default_hash"])
    return table


def in_tree_hashes(content=CONTENT):
    """{slot: set of 12-hex hashes} of the installed gfx11-generic content: manifest `spec_hash`es, reduce from the dq companions."""
    have = {}
    for op, slot in (
        ("sdpa_bwd_prep", "prep"),
        ("sdpa_bwd_dkdv", "dkdv"),
        ("sdpa_bwd_dq", "dq"),
    ):
        manifest = json.loads(
            (content / op / "gfx11-generic" / "manifest.json").read_text()
        )
        have[slot] = {i["spec_hash"] for i in manifest["instances"]}
        if slot == "dq":
            have["reduce"] = {c["hash"] for c in manifest["companions"]}
    return have


def row_class(r):
    if r["bias"] != "none":
        return "bias"
    masked = r["mask"] != "none" or r["window_left"] != -1 or r["window_right"] != -1
    return "causal" if masked else "plain"


def needs_reduce(r):
    """The user's dBias is a broadcast of the full-shape workspace over batch and/or head with that extent above 1."""
    return bool(r["dbias"]) and (
        (r["bias"] in ("b", "bh") and r["b"] > 1)
        or (r["bias"] in ("h", "bh") and r["hq"] > 1)
    )


def expect(r, sel):
    """-> (verdict, prep, dkdv, dq, reduce, rule, basis) of one graph spec."""
    try:
        check_request(
            dtype=r["dtype"], hd_qk=r["dqk"], hd_v=r["dv"], bias=r["bias"] != "none", varlen=r["varlen"] != "none",
            causal=r["mask"] != "none", window=r["window_left"] != -1 or r["window_right"] != -1,
            bias_bcast="" if r["bias"] == "full" else r["bias"].replace("none", ""), dbias=bool(r["dbias"]),
        )  # fmt: skip
        refusal = None
    except RoutingRefusal as e:
        refusal = e.code
    rule = matcher_rule(r)
    if refusal is not None:
        expected = bundle_rule(refusal, r)
        assert (rule or NO_ROW) == expected, (r["id"], refusal, rule, expected)
        basis = f"check_request:{refusal}" + (f";contract:{rule}" if rule else "")
        return ("declined", "", "", "", "", expected, basis)
    if rule:
        return ("declined", "", "", "", "", rule, f"contract:{rule}")
    tile, _ = to_rung(max(r["dqk"], r["dv"]))
    cls, drop = row_class(r), r["dropout"]
    store_db = ("requested" if r["dbias"] else "null") if cls == "bias" else ""
    # selection.json problem key: `tile` is the rung >= max(Dqk, Dv); for `prep` the rung >= Dv.
    # Each routing entry is keyed (kernel, dtype, tile, class, dropout, store_db).
    prep_tile, _ = to_rung(r["dv"])
    tiles = {"prep": prep_tile, "dkdv": tile, "dq": tile}
    hashes = [
        sel[(k, r["dtype"], tiles[k], cls, drop, store_db if k == "dq" else "")][1]
        for k in ("prep", "dkdv", "dq")
    ]
    reduce = (
        sel[("dbias_reduce", r["dtype"], 0, "bias", drop, "")][1]
        if needs_reduce(r)
        else ""
    )
    basis = (
        f"selection:{r['dtype']}/{cls}/{drop}/{store_db or '-'}/{tile}"
        + (f";prep_tile={prep_tile}(rung>=Dv)" if prep_tile != tile else "")
        + "".join(f";{c}" for c in contract_clauses(r))
    )
    return ("served", *hashes, reduce, "", basis)


def contract_clauses(r):
    """The matcher / contract rules a served graph exercises beyond the routing."""
    clauses = []
    if r["dqk"] != r["dv"]:
        clauses.append("DT-5")
    if r["dqk"] % vec(r["dtype"]) or r["dv"] % vec(r["dtype"]):
        clauses.append("DT-8")
    if r["hq"] != r["hk"]:
        clauses.append("DT-6")
    if r["varlen"] == "padded":
        clauses.append("PV-1")
    if r["dropout"]:
        clauses.append("DR-6" if r["seed_form"] else "DR-1")
    if r["mask"] == "causal_br" and r["dropout"]:
        clauses.append("MW-11")
    if r["window_left"] > 0 or r["window_right"] > 0:
        clauses.append("MW-5")
    if r["bias"] in ("b", "h", "bh") and r["dbias"]:
        clauses.append("BI-3")
    if (
        r["stride_qk"] > ceil_to(r["dqk"], 8)
        and r["stride_qk"] % 8 == 0
        and r["stride_qk"] >= 3 * r["dqk"]
    ):
        clauses.append("LY-4")
    return clauses


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    dtype="f16", dqk=64, dv=None, mask="none", window_left=-1, window_right=-1, dropout=0, bias="none", dbias=0,
    varlen="none", hq=4, hk=4, b=2, sq=128, sk=128, stride_qk=None, stride_v=None, seed_form="none",
)  # fmt: skip


def make(rid, source, **kw):
    r = dict(DEFAULTS, id=rid, source=source, **kw)
    if r["dv"] is None:
        r["dv"] = r["dqk"]
    if r["stride_qk"] is None:
        r["stride_qk"] = ceil_to(max(r["dqk"], 1), 8)
    if r["stride_v"] is None:
        r["stride_v"] = ceil_to(max(r["dv"], 1), 8)
    if r["dropout"] and r["seed_form"] == "none":
        r["seed_form"] = "host"
    return r


def build_rows():
    rows = []

    def add(rid, source, **kw):
        rows.append(make(rid, source, **kw))

    # (a) selection: every routing entry realised as an exact and a forced-padded graph.
    for dtype in SHIPPED_DTYPES:
        for tile in GENERIC_LADDER:
            for drop in (0, 1):
                for exact in (True, False):
                    d = (
                        tile if exact else tile - 4
                    )  # padded: Dqk = Dv = tile - 4 (a multiple of 4, not of 8): pad8 stride
                    form = "exact" if exact else "pad"
                    seed = "host" if exact else "device"
                    base = dict(
                        dtype=dtype,
                        dqk=d,
                        dropout=drop,
                        sq=256,
                        sk=256,
                        seed_form=seed if drop else "none",
                    )
                    stem = f"sel-{dtype}-t{tile}-d{drop}-{form}"
                    add(f"{stem}-plain", "selection", **base)
                    add(
                        f"{stem}-causal",
                        "selection",
                        mask="causal_tl" if exact else "causal_br",
                        **base,
                    )
                    add(
                        f"{stem}-bias-dbreq",
                        "selection",
                        bias="bh" if exact else "b",
                        dbias=1,
                        **base,
                    )
                    add(
                        f"{stem}-bias-dbnull",
                        "selection",
                        bias="full" if exact else "h",
                        dbias=0,
                        **base,
                    )

    # (b) boundaries
    for dtype in SHIPPED_DTYPES:
        for rung in GENERIC_LADDER:
            add(f"bnd-{dtype}-d{rung}", "boundary", dtype=dtype, dqk=rung)
            add(f"bnd-{dtype}-d{rung + 1}", "boundary", dtype=dtype, dqk=rung + 1)
        for d in (1, 255):
            add(f"bnd-{dtype}-d{d}", "boundary", dtype=dtype, dqk=d)
    for dtype, (dqk, dv) in [(t, p) for t in ("f16", "fp32") for p in
                             ((64, 128), (128, 64), (16, 256), (256, 16), (80, 96), (33, 64), (100, 48), (192, 224), (64, 257), (300, 64))]:  # fmt: skip
        add(f"bnd-{dtype}-qk{dqk}-v{dv}", "boundary", dtype=dtype, dqk=dqk, dv=dv)
    add("bnd-f16-qk96-v32-causal", "boundary", dqk=96, dv=32, mask="causal_tl")
    add(
        "bnd-bf16-qk40-v72-bias-dbreq",
        "boundary",
        dtype="bf16",
        dqk=40,
        dv=72,
        bias="bh",
        dbias=1,
    )
    add("bnd-f16-gqa8x2", "boundary", hq=8, hk=2)
    add("bnd-f16-mqa8x1", "boundary", hq=8, hk=1)
    add("bnd-f16-sq37-sk300", "boundary", sq=37, sk=300)
    add("bnd-f16-sq300-sk37-causal-br", "boundary", sq=300, sk=37, mask="causal_br")
    add("bnd-f16-b1-h1", "boundary", b=1, hq=1, hk=1)
    add("bnd-f16-varlen-padded", "boundary", varlen="padded")
    add("bnd-f16-win64-64", "boundary", window_left=64, window_right=64)
    add("bnd-f16-causal-tl-win64", "boundary", mask="causal_tl", window_left=64)
    add("bnd-f16-causal-br-win32", "boundary", mask="causal_br", window_left=32)
    add("bnd-f16-causal-br-dropout", "boundary", mask="causal_br", dropout=1)
    add("bnd-f16-dropout-host", "boundary", dropout=1, seed_form="host")
    add("bnd-f16-dropout-device", "boundary", dropout=1, seed_form="device")
    add(
        "bnd-bf16-dropout-device-causal",
        "boundary",
        dtype="bf16",
        dropout=1,
        seed_form="device",
        mask="causal_tl",
    )

    # (c) refusals (every check_request code a graph can express)
    add("ref-hd-lt-1-qk0", "refusal", dqk=0, dv=64)
    add("ref-hd-lt-1-v0", "refusal", dqk=64, dv=0)
    add("ref-hd-gt-256-bf16-384", "refusal", dtype="bf16", dqk=384)
    add("ref-hd-gt-256-fp32-512", "refusal", dtype="fp32", dqk=512)
    add("ref-hd-gt-256-f16-513", "refusal", dqk=513)
    add("ref-hd-gt-256-f16-1024", "refusal", dqk=1024)
    add("ref-hd-gt-256-v400", "refusal", dqk=64, dv=400)
    for dtype in SHIPPED_DTYPES:
        add(
            f"ref-bias-varlen-{dtype}",
            "refusal",
            dtype=dtype,
            bias="full",
            varlen="padded",
        )
        add(
            f"ref-bias-mask-causal-tl-{dtype}",
            "refusal",
            dtype=dtype,
            bias="full",
            mask="causal_tl",
        )
        add(f"ref-bias-q-dbias-{dtype}", "refusal", dtype=dtype, bias="q", dbias=1)
    add("ref-bias-mask-causal-br", "refusal", bias="full", mask="causal_br")
    add("ref-bias-mask-window", "refusal", bias="b", window_left=64, window_right=64)
    add("ref-bias-mask-dbias", "refusal", bias="full", dbias=1, mask="causal_tl")

    # (d) layout
    add("lay-f16-d20-pad8", "layout", dqk=20, stride_qk=24, stride_v=24)
    add("lay-f16-d251-pad8", "layout", dqk=251)
    add("lay-bf16-d100-pad8", "layout", dtype="bf16", dqk=100)
    add("lay-fp32-d18-pad4", "layout", dtype="fp32", dqk=18, stride_qk=20, stride_v=20)
    add("lay-fp32-d17-pad4", "layout", dtype="fp32", dqk=17, stride_qk=20, stride_v=20)
    add("lay-f16-d20-tight", "layout", dqk=20, stride_qk=20, stride_v=20)
    add("lay-f16-d20-tight-v", "layout", dqk=20, stride_qk=24, stride_v=20)
    add("lay-f16-d20-tight-qk", "layout", dqk=20, stride_qk=20, stride_v=24)
    add("lay-fp32-d18-tight", "layout", dtype="fp32", dqk=18, stride_qk=18, stride_v=18)
    add("lay-f16-d20-narrow", "layout", dqk=20, stride_qk=16, stride_v=24)
    add(
        "lay-fp32-d18-narrow", "layout", dtype="fp32", dqk=18, stride_qk=16, stride_v=20
    )
    add("lay-f16-d64-stride60", "layout", stride_qk=60)
    add("lay-f16-d64-stride72", "layout", stride_qk=72)
    add("lay-f16-d64-stride0", "layout", stride_qk=0)
    add("lay-f16-d64-v-stride0", "layout", stride_v=0)
    add("lay-f16-d64-packed-qkv", "layout", stride_qk=192)
    add("lay-f16-d64-packed-qkv-v", "layout", stride_qk=192, stride_v=192)
    add("lay-fp32-d64-stride68", "layout", dtype="fp32", stride_qk=68)
    add("lay-f16-d64-stride68", "layout", stride_qk=68)
    add(
        "lay-f16-d64-slab-over-int32",
        "layout",
        sq=1048577,
        b=1,
        hq=1,
        hk=1,
        stride_qk=1024,
    )

    # (e) bias / dBias broadcast forms
    for dtype in SHIPPED_DTYPES:
        for form in ("full", "b", "h", "bh", "q"):
            for dbias in (0, 1):
                if form == "q" and dbias:
                    continue  # the refusal rows (ref-bias-q-dbias-*) are exactly these graphs
                add(
                    f"bias-{dtype}-{form}-db{dbias}",
                    "bias",
                    dtype=dtype,
                    bias=form,
                    dbias=dbias,
                )
    add("bias-f16-b-extent1-db1", "bias", bias="b", dbias=1, b=1)
    add("bias-f16-h-extent1-db1", "bias", bias="h", dbias=1, hq=1, hk=1)
    add("bias-f16-bh-b1h1-db1", "bias", bias="bh", dbias=1, b=1, hq=1, hk=1)
    add("bias-f16-bh-b1-db1", "bias", bias="bh", dbias=1, b=1)
    add("bias-f16-gqa-bh-db1", "bias", bias="bh", dbias=1, hq=8, hk=2)
    add("bias-bf16-bh-dropout-db1", "bias", dtype="bf16", bias="bh", dbias=1, dropout=1)
    add("bias-f16-q-sq-ne-sk", "bias", bias="q", sq=128, sk=256)
    add("bias-f16-full-sq-ne-sk-db1", "bias", bias="full", dbias=1, sq=128, sk=256)
    add(
        "bias-fp32-bh-pad4-db1",
        "bias",
        dtype="fp32",
        dqk=18,
        stride_qk=20,
        stride_v=20,
        bias="bh",
        dbias=1,
    )

    # (f) unmodelled by the bundle: the matcher's named decline
    add("unm-f16-varlen-packed", "unmodelled", varlen="packed")
    add("unm-f16-varlen-packed-causal", "unmodelled", varlen="packed", mask="causal_tl")
    add("unm-f16-dbias-without-bias", "unmodelled", dbias=1)
    add("unm-f16-gqa-6x4", "unmodelled", hq=6, hk=4)
    add("unm-f16-win-left0-right0", "unmodelled", window_left=0, window_right=0)
    add("unm-f16-win-left0-unbounded", "unmodelled", window_left=0)
    add("unm-f16-causal-win-left0", "unmodelled", mask="causal_tl", window_left=0)
    add("unm-f16-win-left-below-1", "unmodelled", window_left=-2, window_right=8)
    add(
        "unm-f16-varlen-padded-br-window",
        "unmodelled",
        varlen="padded",
        mask="causal_br",
        window_left=32,
    )
    return rows


def dedupe_check(rows):
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    specs = [tuple(r[c] for c in COLUMNS[2:20]) for r in rows]
    dups = {s for s in specs if specs.count(s) > 1}
    assert not dups, f"duplicate graph specs: {sorted(dups)}"


def generate(selection=SELECTION_CSV, content=CONTENT):
    sel = load_selection(selection)
    rows = build_rows()
    dedupe_check(rows)
    have = in_tree_hashes(content)
    out = []
    for r in rows:
        verdict, prep, dkdv, dq, reduce, rule, basis = expect(r, sel)
        r.update(
            verdict=verdict,
            prep=prep,
            dkdv=dkdv,
            dq=dq,
            reduce=reduce,
            rule=rule,
            basis=basis,
        )
        for slot in SLOT_COLUMNS:
            if r[slot] and r[slot] not in have[slot]:
                raise SystemExit(
                    f"STOP: {r['id']}: {slot} hash {r[slot]} is not an in-tree gfx11-generic object"
                )
        out.append(r)
    for (kernel, *_), hashes in sel.items():
        for h in hashes:
            if h not in have["reduce" if kernel == "dbias_reduce" else kernel]:
                raise SystemExit(
                    f"STOP: selection hash {h} ({kernel}) is not an in-tree gfx11-generic object"
                )
    out.sort(key=lambda r: r["id"])
    return out


def render(rows):
    import io

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([r[c] for c in COLUMNS])
    return buf.getvalue()


def read_csv(path=VERDICTS_CSV):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == COLUMNS, reader.fieldnames
        rows = list(reader)
    for r in rows:
        for c in INT_COLUMNS:
            r[c] = int(r[c])
    return rows


# ---------------------------------------------------------------------------
# Bundle import
# ---------------------------------------------------------------------------


def import_bundle(bundle):
    """Rewrite the vendored routing CSV from the bundle's selection.json, after checking the vendored check_request against the bundle's."""
    bundle = Path(bundle)
    spec = importlib.util.spec_from_file_location(
        "bundle_selection", bundle / "source" / "tools" / "selection.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.GENERIC_LADDER, mod.SHIPPED_DTYPES, mod.REASONS) == (
        GENERIC_LADDER,
        SHIPPED_DTYPES,
        REASONS,
    )
    cases = [
        dict(
            dtype=d,
            hd_qk=a,
            hd_v=b,
            bias=bias,
            varlen=vl,
            causal=c,
            window=w,
            bias_bcast=bc,
            dbias=db,
        )
        for d in ("f16", "bf16", "fp32", "fp8")
        for a in (0, 1, 64, 256, 257)
        for b in (None, 0, 32, 300)
        for bias in (False, True)
        for vl in (False, True)
        for c in (False, True)
        for w in (False, True)
        for bc in ("", "b", "q", "bh")
        for db in (False, True)
    ]
    for kw in cases:
        outs = []
        for fn in (check_request, mod.check_request):
            try:
                outs.append(fn(**kw))
            except ValueError as e:
                outs.append(e.code)
        assert outs[0] == outs[1], (kw, outs)
    data = json.loads((bundle / "handover" / "selection.json").read_text())
    rows = []
    for e in data["entries"]:
        p = e["problem"]
        rows_by_when = {
            json.dumps(r["when"], sort_keys=True): r["hash"] for r in e["rows"]
        }
        assert set(rows_by_when) == {'{"exact": false}', "{}"}, rows_by_when
        rows.append(
            (p["kernel"], p["dtype"], p.get("tile", 0), p["class"], p["dropout"], p.get("store_db", ""),
             rows_by_when['{"exact": false}'], rows_by_when["{}"])
        )  # fmt: skip
    rows.sort()
    with open(SELECTION_CSV, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(SELECTION_COLUMNS)
        w.writerows(rows)
    print(f"wrote {len(rows)} routing entries to {SELECTION_CSV}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="regenerate and diff against the committed CSV",
    )
    ap.add_argument(
        "--import-bundle",
        metavar="BUNDLE_DIR",
        help="refresh the vendored routing table from the RDNA bundle",
    )
    ap.add_argument("--out", type=Path, default=VERDICTS_CSV)
    args = ap.parse_args(argv)
    if args.import_bundle:
        import_bundle(args.import_bundle)
        return 0
    text = render(generate())
    if args.check:
        have = args.out.read_text()
        if have != text:
            sys.stdout.writelines(
                difflib.unified_diff(
                    have.splitlines(True),
                    text.splitlines(True),
                    "committed",
                    "generated",
                    n=1,
                )
            )
            print(
                "sdpa_bwd_verdicts: the committed CSV differs from the regeneration",
                file=sys.stderr,
            )
            return 1
        print(
            f"sdpa_bwd_verdicts: {args.out.name} is up to date ({text.count(chr(10)) - 1} rows)"
        )
        return 0
    args.out.write_text(text)
    print(f"wrote {text.count(chr(10)) - 1} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
