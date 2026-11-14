# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Replay corpus of the native engine ``hipkernel:AttentionGeneralBwd``.

Each entry is a hipDNN ``SdpaBackward`` graph (flatbuffer JSON, as the hipDNN JSON
helpers read it) with the outcome rocKE's Python reference gives it on one arch and
device compute-unit count: the decline code, or for an accepted graph the whole plan
(graph-match tokens, workspace layout, and every launch with its kernel symbol, grid,
block and kernargs in ABI order). The engine's C++ graph match and ``prepare()`` must
reproduce every entry. The reference is ``dispatch/attention_bwd/hipdnn.py``:
``from_hipdnn_graph`` (``LOWERING`` on a graph it cannot read), ``engine_verdict`` and
``plan_record``.

Seeds are the predicate's own test rows (one trigger per reason code, the accept
and narrow-access rows), boundary rows, the backward case table, the representative
requests of every tuned-table key, the catalog representatives and the hipDNN
``SdpaBackward`` integration bundles. Mutations act on the graph JSON (one to three
per entry), so they exercise the C++ lowering as well as the predicate, and keep
every graph expressible by hipDNN (valid enum names, float32 scalars, the keys the
JSON readers require).

Usage::

    general_bwd_corpus.py --arch gfx942 --mode subset --out corpus_gfx942.jsonl [--check]
    general_bwd_corpus.py --arch gfx950 --mode full --count 100000 --out <file>

``subset`` is the committed CI fixture (at most ``--max-bytes``); ``full`` writes
every seed and ``--count`` mutated entries and lives outside the repository. The
output is deterministic (seed 20261105 plus the arch index). The first line is a
``meta`` record.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import json
import math
import os
import random
import struct
import sys
from collections import Counter
from pathlib import Path

_HKP = Path(__file__).resolve().parents[2]
for _sub in ("rocke/platform/python", "rocke/library"):
    _p = str(_HKP / _sub)
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
os.environ.setdefault("ROCKE_BACKEND", "python")

from dispatch.attention_bwd import hipdnn as H  # noqa: E402
from dispatch.attention_bwd import hipdnn_catalog as C  # noqa: E402
from kernels.common.attention_bwd_plan import REASON_CODES, AttnBwdRequest  # noqa: E402

BUNDLES = _HKP.parent / "integration-tests" / "integration-test-bundles"
GENERATOR = "general_bwd_corpus.py"
BASE_SEED = 20261105
FULL_CUS = {"gfx942": 304, "gfx950": 256}
CU_COUNTS = (304, 256, 80, 38, 32)
DEFAULT_MAX_BYTES = 6 * 1024 * 1024
DETAIL_CHARS = 160
# mutated candidates the subset is selected from
SUBSET_POOL = 6000
# bytes of the size cap kept for the meta line
META_RESERVE = 4096

DATA = ("q", "k", "v", "o", "do", "dq", "dk", "dv")
Q_SIDE = ("q", "o", "do", "dq")
KV_SIDE = ("k", "v", "dk", "dv")
OUTPUTS = ("dq", "dk", "dv")
ROLE_KEYS = {
    "q": ("inputs", "q_tensor_uid"),
    "k": ("inputs", "k_tensor_uid"),
    "v": ("inputs", "v_tensor_uid"),
    "o": ("inputs", "o_tensor_uid"),
    "do": ("inputs", "do_tensor_uid"),
    "lse": ("inputs", "stats_tensor_uid"),
    "dq": ("outputs", "dq_tensor_uid"),
    "dk": ("outputs", "dk_tensor_uid"),
    "dv": ("outputs", "dv_tensor_uid"),
}
# hipDNN DataType JSON names (flatbuffers_sdk utilities/json/Common.hpp).
DTYPE_NAMES = (
    "float",
    "half",
    "bfloat16",
    "double",
    "uint8",
    "int32",
    "int8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_e8m0",
    "fp4_e2m1",
    "int4",
    "fp6_e2m3",
    "fp6_e3m2",
    "int64",
    "boolean",
    "fp8_e4m3_fnuz",
    "fp8_e5m2_fnuz",
)
# Tensor keys whose absence the hipDNN JSON reader and the lowering both read as
# this value; dropped to keep the fixture small.
_TENSOR_DEFAULTS = {
    "alignment": 16,
    "ragged_offset_multiplier": 1,
    "is_runtime_pass_by_value": False,
}


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def node_of(g: dict) -> dict:
    return g["nodes"][0]


def tensors(g: dict) -> dict:
    return {t["uid"]: t for t in g["tensors"]}


def uid_of(g: dict, role: str):
    group, key = ROLE_KEYS[role]
    return node_of(g)[group].get(key)


def tensor_of(g: dict, role: str):
    uid = uid_of(g, role)
    return tensors(g).get(uid) if uid is not None else None


def new_uid(g: dict) -> int:
    return max([t["uid"] for t in g["tensors"]] + [-1]) + 1


def add_tensor(g: dict, name: str, dtype: str, dims, strides, **extra) -> int:
    uid = new_uid(g)
    t = {
        "uid": uid,
        "name": name,
        "data_type": dtype,
        "dims": [int(x) for x in dims],
        "strides": [int(x) for x in strides],
        "virtual": False,
    }
    t.update(extra)
    g["tensors"].append(t)
    return uid


def packed(dims) -> list[int]:
    out, acc = [], 1
    for n in reversed(dims):
        out.append(acc)
        acc *= max(int(n), 1)
    return list(reversed(out))


def normalize(g: dict, io_dtype: str | None = None) -> dict:
    """The canonical, hipDNN-readable form of a graph: the root data types the
    reader requires, float32 scalars, no null optional keys, default-valued
    tensor keys dropped."""
    g = copy.deepcopy(g)
    q = tensor_of(g, "q") if g.get("nodes") else None
    io = io_dtype or (q["data_type"] if q else "half")
    out = {
        "name": g.get("name", "SdpaBackward"),
        "compute_data_type": g.get("compute_data_type", "float"),
        "intermediate_data_type": g.get("intermediate_data_type", "float"),
        "io_data_type": g.get("io_data_type", io),
        "nodes": g["nodes"],
        "tensors": [],
    }
    params = node_of(out)["parameters"]
    for key in ("attn_scale_value", "dropout_probability"):
        if isinstance(params.get(key), (int, float)) and not isinstance(
            params[key], bool
        ):
            params[key] = f32(params[key])
    for t in g["tensors"]:
        t = dict(t)
        for key, default in _TENSOR_DEFAULTS.items():
            if key in t and type(t[key]) is type(default) and t[key] == default:
                del t[key]
        if t.get("ragged_offset_tensor_uid", 0) is None:
            del t["ragged_offset_tensor_uid"]
        if isinstance(t.get("value"), float):
            t["value"] = f32(t["value"])
        out["tensors"].append(t)
    out["tensors"].sort(key=lambda t: t["uid"])
    return out


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def shape_of(g: dict) -> dict:
    q, k, v = (tensor_of(g, r)["dims"] for r in ("q", "k", "v"))
    return {
        "b": q[0],
        "h_q": q[1],
        "h_k": k[1],
        "h_v": v[1],
        "s_q": q[2],
        "s_kv": k[2],
        "d_qk": q[3],
        "d_v": v[3],
    }


def _role_dims(s: dict, role: str) -> list[int]:
    h = {"q": "h_q", "o": "h_q", "do": "h_q", "dq": "h_q", "k": "h_k", "dk": "h_k"}
    h = s[h.get(role, "h_v")]
    seq = s["s_q"] if role in Q_SIDE else s["s_kv"]
    d = s["d_qk"] if role in ("q", "k", "dq", "dk") else s["d_v"]
    return [s["b"], h, seq, d]


def _layout(t: dict) -> str:
    if t.get("ragged_offset_tensor_uid") is not None:
        return "thd"
    _b, h, s, d = t["dims"]
    if list(t["strides"]) == [s * h * d, d, h * d, 1]:
        return "bshd"
    return "bhsd"


def _strides(layout: str, dims) -> list[int]:
    _b, h, s, d = dims
    if layout in ("bshd", "thd"):
        return [s * h * d, d, h * d, 1]
    return [h * s * d, s * d, d, 1]


def reshape(g: dict, **change) -> None:
    """Set the problem shape, keeping each operand's layout (packed strides) and
    each ragged tensor's offsets in its own units."""
    old = shape_of(g)
    new = dict(old, **change)
    for role in DATA:
        t = tensor_of(g, role)
        lay = _layout(t)
        dims = _role_dims(new, role)
        old_dims = list(t["dims"])
        if lay == "thd":
            m = t.get("ragged_offset_multiplier", 1)
            if m == old_dims[1] * old_dims[3]:
                t["ragged_offset_multiplier"] = dims[1] * dims[3]
        t["dims"], t["strides"] = dims, _strides(lay, dims)
    st = tensor_of(g, "lse")
    b, hq, sq = new["b"], new["h_q"], new["s_q"]
    st["dims"] = [b, hq, sq, 1]
    if st.get("ragged_offset_tensor_uid") is not None:
        if st.get("ragged_offset_multiplier", 1) == old["h_q"]:
            st["ragged_offset_multiplier"] = hq
        st["strides"] = [sq * hq, 1, hq, 1]
    else:
        st["strides"] = [hq * sq, sq, 1, 1]
    table = tensors(g)
    offsets = {
        t.get("ragged_offset_tensor_uid")
        for t in g["tensors"]
        if t.get("ragged_offset_tensor_uid") is not None
    }
    for uid in offsets:
        if uid in table and table[uid]["dims"] == [old["b"] + 1, 1, 1, 1]:
            table[uid]["dims"] = [b + 1, 1, 1, 1]
    inputs, outputs = node_of(g)["inputs"], node_of(g)["outputs"]
    for key in ("seq_len_q_tensor_uid", "seq_len_kv_tensor_uid"):
        t = table.get(inputs.get(key))
        if t is not None and t["dims"] == [old["b"], 1, 1, 1]:
            t["dims"] = [b, 1, 1, 1]
    for uid in (inputs.get("attn_mask_tensor_uid"), outputs.get("dbias_tensor_uid")):
        t = table.get(uid)
        if t is not None:
            t["dims"] = [b, hq, sq, new["s_kv"]]
            t["strides"] = packed(t["dims"])


_I64 = (-(2**63), 2**63 - 1)
_BOOL_PARAMS = ("alibi_mask", "padding_mask", "causal_mask", "causal_mask_bottom_right")
_INPUT_KEYS = (
    "q_tensor_uid",
    "k_tensor_uid",
    "v_tensor_uid",
    "o_tensor_uid",
    "do_tensor_uid",
    "stats_tensor_uid",
    "scale_tensor_uid",
    "attn_mask_tensor_uid",
    "seq_len_q_tensor_uid",
    "seq_len_kv_tensor_uid",
    "seed_tensor_uid",
    "offset_tensor_uid",
    "dropout_mask_tensor_uid",
    "dropout_scale_tensor_uid",
    "dropout_scale_inv_tensor_uid",
)
_OUTPUT_KEYS = ("dq_tensor_uid", "dk_tensor_uid", "dv_tensor_uid", "dbias_tensor_uid")


def _i64(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and _I64[0] <= x <= _I64[1]


def _opt_i64(x) -> bool:
    return x is None or _i64(x)


def _f32_exact(x) -> bool:
    return isinstance(x, float) and f32(x) == x


def expressible(g: dict) -> bool:
    """Whether the hipDNN JSON reader (flatbuffers_sdk ``utilities/json``) reads the
    graph back exactly: every key it requires, enum names it knows, int64 integers
    and float32 scalars."""
    if not all(k in g for k in ("name", "nodes", "tensors")):
        return False
    if not all(
        g.get(k) in DTYPE_NAMES
        for k in ("compute_data_type", "intermediate_data_type", "io_data_type")
    ):
        return False
    node = node_of(g)
    if node.get("compute_data_type") not in DTYPE_NAMES or not isinstance(
        node.get("name"), str
    ):
        return False
    inputs, outputs, p = node.get("inputs"), node.get("outputs"), node.get("parameters")
    if not all(isinstance(x, dict) for x in (inputs, outputs, p)):
        return False
    for group, keys in ((inputs, _INPUT_KEYS), (outputs, _OUTPUT_KEYS)):
        for k in keys:
            if k not in group or not _opt_i64(group[k]):
                return False
        required = keys[:6] if group is inputs else keys[:3]
        if any(group[k] is None for k in required):
            return False
    if not all(isinstance(p.get(k), bool) for k in _BOOL_PARAMS):
        return False
    for k in ("dropout_probability", "attn_scale_value"):
        if k not in p or not (p[k] is None or _f32_exact(p[k])):
            return False
    if not all(k in p and _opt_i64(p[k]) for k in ("left_bound", "right_bound")):
        return False
    if p.get("diagonal_alignment") not in ("TOP_LEFT", "BOTTOM_RIGHT"):
        return False
    for t in g["tensors"]:
        if not (_i64(t.get("uid")) and isinstance(t.get("name"), str)):
            return False
        if t.get("data_type") not in DTYPE_NAMES or not isinstance(
            t.get("virtual"), bool
        ):
            return False
        for k in ("dims", "strides"):
            if not (isinstance(t.get(k), list) and all(_i64(x) for x in t[k])):
                return False
        if not all(
            _i64(t.get(k, 1)) for k in ("alignment", "ragged_offset_multiplier")
        ):
            return False
        if not isinstance(t.get("is_runtime_pass_by_value", False), bool):
            return False
        if "ragged_offset_tensor_uid" in t and not _i64(t["ragged_offset_tensor_uid"]):
            return False
        if "value" in t and not (
            t.get("value_type") == "Float32Value" and _f32_exact(t["value"])
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# expectation
# ---------------------------------------------------------------------------


def outcome(g: dict, arch: str, num_cus: int):
    """``(expect, request)`` of a normalized graph."""
    try:
        req = H.from_hipdnn_graph(g, arch, num_cus=num_cus)
    except H.HipdnnLoweringError as exc:
        return {"code": H.LOWERING, "detail": str(exc)[:DETAIL_CHARS]}, None
    code, detail, plan = H.engine_verdict(req, arch)
    if plan is None:
        return {"code": code, "detail": detail[:DETAIL_CHARS]}, req
    return H.plan_record(plan, req, arch), req


def entry(eid: str, arch: str, g: dict, num_cus: int, with_request: bool) -> dict:
    expect, req = outcome(g, arch, num_cus)
    out = {"id": eid, "arch": arch, "num_cus": num_cus, "graph": g, "expect": expect}
    if with_request and req is not None:
        out["request"] = req.normalized()
    return out


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


def _req(**kw) -> AttnBwdRequest:
    base = {"b": 2, "h_q": 8, "h_k": 8, "h_v": 8, "s_q": 128, "s_kv": 128}
    base.update(d_qk=128, d_v=128)
    base.update(kw)
    return AttnBwdRequest(**base)


def _boundary_rows() -> dict:
    """Requests on both sides of the predicate's arithmetic limits."""
    one = dict(b=1, h_q=1, h_k=1, h_v=1, s_q=1, s_kv=16, d_qk=128, d_v=128)
    grid = dict(h_q=1, h_k=1, h_v=1, s_q=16, s_kv=16, d_qk=32, d_v=32)
    big_kv = dict(b=2**15, s_q=1, s_kv=2**16, d_qk=32, d_v=32, h_q=2, h_k=2)
    h, s, d = 8, 128, 128
    return {
        "q_stride_in_range": _req(**one, strides={"q": (2**31 - 129,) * 3 + (1,)}),
        "q_stride_out_of_range": _req(**one, strides={"q": (2**31 - 128,) * 3 + (1,)}),
        "b_65535": _req(b=65535, **grid),
        "b_65536": _req(b=65536, **grid),
        "rows_kv_2_31_atomic": _req(**big_kv, h_v=1),
        "rows_kv_2_31_direct": _req(**big_kv, h_v=2),
        "s_2_24": _req(b=1, s_q=2**24, s_kv=16, h_q=1, h_k=1, h_v=1),
        "heads_70000": _req(h_q=70000, h_k=70000, h_v=70000, s_q=16, s_kv=16, b=1),
        "prep_y": _req(h_q=40000, h_k=20000, h_v=10000, s_q=16, s_kv=16, b=1),
        "dq_s_stride_short": _req(strides={"dq": (h * s * d, s * d, d - 1, 1)}),
        "dq_h_stride_short": _req(strides={"dq": (h * s * d, s * d - 1, d, 1)}),
        "dk_b_stride_short": _req(strides={"dk": (h * s * d - 1, s * d, d, 1)}),
        "dkv_interleaved": _req(
            strides={
                "dk": (s * 2 * h * d, d, 2 * h * d, 1),
                "dv": (s * 2 * h * d, d, 2 * h * d, 1),
            }
        ),
        "dv_interleaved_short": _req(strides={"dv": (s * 2 * h * d, d, h * d - 1, 1)}),
        "window_left_1": _req(left_bound=0, right_bound=0),
        "window_far": _req(left_bound=2**30 - 1, right_bound=-1),
        "s_q_1_long_kv": _req(s_q=1, s_kv=4097, h_k=1, h_v=1),
        "s_kv_1": _req(s_q=33, s_kv=1),
    }


def _seed_requests() -> list[tuple[str, AttnBwdRequest]]:
    from tests.sdpa.bwd_cases import backward_cases
    from tests.sdpa.bwd_kernel_cases import request_from_case
    from tests.test_attention_bwd_plan import _NARROW_ROWS, _TRIGGERS, _accept_rows

    out = [(f"trigger:{c}", r) for c, (r, _a, _p) in sorted(_TRIGGERS.items())]
    out += [(f"accept:{n}", r) for n, r in sorted(_accept_rows().items())]
    out += [(f"narrow:{n}", _req(**kw)) for n, kw in sorted(_NARROW_ROWS.items())]
    out += [(f"boundary:{n}", r) for n, r in sorted(_boundary_rows().items())]
    out += [(f"case:{c.id}", request_from_case(c)) for c in backward_cases()]
    return out


def _expand(node, values, uid=None):
    """Resolve a bundle template's ``${case.<path>}`` placeholders (the integration
    harness's expansion: tensor fields from the case's tensor of the same uid)."""
    if isinstance(node, str):
        if node.startswith("${case.") and node.endswith("}"):
            path = node[len("${case.") : -1]
            if uid is not None:
                for tv in values.get("tensors", []):
                    if tv.get("uid") == uid and path in tv:
                        return tv[path]
            cur = values
            for tok in path.split("."):
                if not (isinstance(cur, dict) and tok in cur):
                    raise KeyError(f"unresolved {node}")
                cur = cur[tok]
            return cur
        return node
    if isinstance(node, list):
        return [_expand(x, values, uid) for x in node]
    if isinstance(node, dict):
        nt = node.get("uid") if isinstance(node.get("uid"), int) else uid
        return {k: _expand(v, values, nt) for k, v in node.items()}
    return node


def _single_backward(g) -> bool:
    nodes = g.get("nodes") if isinstance(g, dict) else None
    return (
        isinstance(nodes, list)
        and len(nodes) == 1
        and nodes[0].get("type") == H.NODE_TYPE
    )


def bundle_graphs(skipped: Counter) -> list[tuple[str, dict]]:
    out = []
    for tmpl_path in sorted(
        glob.glob(
            str(BUNDLES / "*" / "SdpaBackward" / "**" / "graph.template.json"),
            recursive=True,
        )
    ):
        d = Path(tmpl_path).parent
        sweep = d / "sweep.json"
        if not sweep.is_file():
            skipped["bundle_no_sweep"] += 1
            continue
        tmpl = json.loads(Path(tmpl_path).read_text(encoding="utf-8"))
        rel = d.relative_to(BUNDLES).as_posix()
        for case in json.loads(sweep.read_text(encoding="utf-8"))["cases"]:
            g = _expand(tmpl, case["values"])
            if not _single_backward(g):
                skipped["bundle_not_single_backward"] += 1
                continue
            out.append((f"bundle:{rel}/{case['id']}", g))
    for path in sorted(
        glob.glob(str(BUNDLES / "*" / "SdpaBackward" / "**" / "*.json"), recursive=True)
    ):
        p = Path(path)
        if p.name in (
            "graph.template.json",
            "sweep.json",
            "support.json",
        ) or p.name.endswith(".meta.json"):
            continue
        g = json.loads(p.read_text(encoding="utf-8"))
        if not _single_backward(g):
            skipped["bundle_not_single_backward"] += 1
            continue
        out.append((f"bundle:{p.relative_to(BUNDLES).as_posix()}", g))
    if not out:
        raise AssertionError(f"no SdpaBackward graph under {BUNDLES}")
    return out


@dataclasses.dataclass
class Seed:
    sid: str
    graph: dict
    num_cus: int


def seeds(arch: str, skipped: Counter) -> list[Seed]:
    full = FULL_CUS[arch]
    out: list[Seed] = []

    def add_req(sid, req):
        try:
            g = H.to_hipdnn_graph(req)
        except ValueError:
            skipped[sid.split(":", 1)[0]] += 1
            return
        out.append(Seed(sid, normalize(g), req.num_cus or full))

    for sid, req in _seed_requests():
        add_req(sid, req)
    for i, (key, req) in enumerate(C.tuned_requests(arch)):
        add_req(f"tuned:{H.tuning_key_id(arch, key)}/{i}", req)
    for i, row in enumerate(C.catalog_rows(arch)):
        rep = row.representative
        add_req(f"row:{row.pack}/{i}", dataclasses.replace(rep, num_cus=full))
        if rep.num_cus not in (None, full):
            # a tuned row whose key needs the representative's own grid fill
            add_req(f"row:{row.pack}/{i}@{rep.num_cus}", rep)
    for sid, g in bundle_graphs(skipped):
        out.append(Seed(sid, normalize(g), full))
    return out


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------


def _pick_role(rng, roles=DATA) -> str:
    return rng.choice(roles)


def m_dtype_one(g, rng):
    role = _pick_role(rng, DATA + ("lse",))
    name = rng.choice(DTYPE_NAMES)
    tensor_of(g, role)["data_type"] = name
    return f"dtype:{role}={name}"


def m_dtype_all(g, rng):
    name = rng.choice(("half", "bfloat16", "float"))
    for role in DATA:
        tensor_of(g, role)["data_type"] = name
    g["io_data_type"] = name
    return f"dtype_all={name}"


def m_compute(g, rng):
    name = rng.choice(("half", "bfloat16", "float", "double"))
    node_of(g)["compute_data_type"] = name
    return f"compute={name}"


def m_stride_d(g, rng):
    role = _pick_role(rng)
    v = rng.choice((0, 2, 3))
    tensor_of(g, role)["strides"][3] = v
    return f"stride_d:{role}={v}"


def m_stride_negative(g, rng):
    role = _pick_role(rng)
    i = rng.randrange(3)
    st = tensor_of(g, role)["strides"]
    st[i] = -abs(st[i]) - rng.choice((0, 1))
    return f"stride_neg:{role}[{i}]"


def m_stride_overlap(g, rng):
    role = _pick_role(rng, OUTPUTS)
    t = tensor_of(g, role)
    i = rng.choice((0, 1, 2))
    st = t["strides"]
    st[i] = rng.choice((0, max(1, st[i] // 2), st[i] - 1))
    return f"stride_overlap:{role}[{i}]={st[i]}"


def m_stride_odd(g, rng):
    role = _pick_role(rng)
    i = rng.randrange(3)
    add = rng.choice((1, 2, 3, 4, 8, 64))
    tensor_of(g, role)["strides"][i] += add
    return f"stride_odd:{role}[{i}]+{add}"


def m_alignment(g, rng):
    # Non-powers of two guarantee only their largest power-of-two divisor.
    v = rng.choice((1, 2, 3, 4, 6, 8, 16, 24, 32, 48))
    which = rng.choice(("one", "all", "lse"))
    if which == "one":
        role = _pick_role(rng)
        tensor_of(g, role)["alignment"] = v
        return f"align:{role}={v}"
    if which == "lse":
        tensor_of(g, "lse")["alignment"] = v
        return f"align:lse={v}"
    for role in DATA:
        tensor_of(g, role)["alignment"] = v
    return f"align:all={v}"


def m_heads(g, rng):
    s = shape_of(g)
    hq = s["h_q"]
    choice = rng.choice(("mqa", "gqa", "hk_ne_hv", "bad_ratio", "double_q", "mha"))
    if choice == "mqa":
        ch = dict(h_k=1, h_v=1)
    elif choice == "gqa":
        divs = [x for x in range(1, hq + 1) if hq % x == 0]
        x = rng.choice(divs)
        ch = dict(h_k=x, h_v=x)
    elif choice == "hk_ne_hv":
        ch = dict(h_q=8, h_k=rng.choice((4, 2)), h_v=rng.choice((2, 1)))
    elif choice == "bad_ratio":
        ch = dict(h_q=rng.choice((8, 6, 7)), h_k=3, h_v=rng.choice((3, 1)))
    elif choice == "double_q":
        ch = dict(h_q=2 * hq)
    else:
        ch = dict(h_k=hq, h_v=hq)
    reshape(g, **ch)
    return f"heads:{choice}"


def m_head_dim(g, rng):
    d = rng.choice((16, 32, 64, 96, 128, 192, 256))
    if rng.random() < 0.25:
        reshape(g, d_v=d)
        return f"d_v={d}"
    reshape(g, d_qk=d, d_v=d)
    return f"d={d}"


def m_lengths(g, rng):
    field = rng.choice(("s_q", "s_kv", "b", "s_q", "s_kv"))
    if rng.random() < 0.15:
        v = rng.choice(
            {
                "s_q": (2**24, 2**30, 2**21),
                "s_kv": (2**24, 2**30, 2**21),
                "b": (65535, 65536, 70000, 2**15),
            }[field]
        )
    else:
        v = rng.choice((0, 1, 2, 3, 17, 64, 128, 300, 1025, 4097))
    reshape(g, **{field: v})
    return f"{field}={v}"


def m_mask(g, rng):
    p = node_of(g)["parameters"]
    choice = rng.choice(
        ("causal", "causal_br", "both", "left", "right", "diag", "clear", "window")
    )
    if choice == "causal":
        p["causal_mask"] = True
    elif choice == "causal_br":
        p["causal_mask_bottom_right"] = True
    elif choice == "both":
        p["causal_mask"] = p["causal_mask_bottom_right"] = True
    elif choice == "left":
        p["left_bound"] = rng.choice((None, -2, -1, 0, 1, 16, 2**30 - 1, 2**30, 2**31))
    elif choice == "right":
        p["right_bound"] = rng.choice((None, -2, -1, 0, 1, 7, 2**30 - 1, 2**30))
    elif choice == "diag":
        p["diagonal_alignment"] = rng.choice(("TOP_LEFT", "BOTTOM_RIGHT"))
    elif choice == "window":
        p["left_bound"], p["right_bound"] = rng.choice(((16, 0), (64, 3), (1, 0)))
    else:
        p.update(causal_mask=False, causal_mask_bottom_right=False)
        p.update(left_bound=None, right_bound=None)
    return f"mask:{choice}"


def m_features(g, rng):
    node = node_of(g)
    p, inputs, outputs = node["parameters"], node["inputs"], node["outputs"]
    s = shape_of(g)
    q_dtype = tensor_of(g, "q")["data_type"]
    scalar = ([1, 1, 1, 1], [1, 1, 1, 1])
    choice = rng.choice(
        (
            "dropout_p",
            "dropout_tensor",
            "attn_mask",
            "dbias",
            "alibi",
            "scale_pbv",
            "scale_value",
            "scale_novalue",
            "scale_attr",
            "padding",
            "padding_only",
            "seq_len_only",
        )
    )
    if choice == "dropout_p":
        p["dropout_probability"] = rng.choice((0.1, 0.5, 0.0))
    elif choice == "dropout_tensor":
        key = rng.choice(
            ("seed_tensor_uid", "offset_tensor_uid", "dropout_mask_tensor_uid")
        )
        inputs[key] = add_tensor(g, key[:-11], "int64", *scalar)
    elif choice in ("attn_mask", "dbias"):
        dims = [s["b"], s["h_q"], s["s_q"], s["s_kv"]]
        uid = add_tensor(g, choice, q_dtype, dims, packed(dims))
        if choice == "attn_mask":
            inputs["attn_mask_tensor_uid"] = uid
        else:
            outputs["dbias_tensor_uid"] = uid
    elif choice == "alibi":
        p["alibi_mask"] = True
    elif choice == "scale_pbv":
        inputs["scale_tensor_uid"] = add_tensor(
            g, "attn_scale", "float", *scalar, is_runtime_pass_by_value=True
        )
    elif choice == "scale_value":
        inputs["scale_tensor_uid"] = add_tensor(
            g,
            "attn_scale",
            "float",
            *scalar,
            value_type="Float32Value",
            value=f32(rng.choice((0.125, 0.3, 1.0, 0.0883883461356163))),
        )
    elif choice == "scale_novalue":
        inputs["scale_tensor_uid"] = add_tensor(g, "attn_scale", "float", *scalar)
    elif choice == "scale_attr":
        p["attn_scale_value"] = f32(
            rng.choice((0.125, 0.3, 2.0, 1.0 / math.sqrt(s["d_qk"] or 1)))
        )
    else:
        if choice != "seq_len_only":
            p["padding_mask"] = True
        if choice != "padding_only":
            sides = ("q", "kv") if choice == "padding" else (rng.choice(("q", "kv")),)
            for side in sides:
                inputs[f"seq_len_{side}_tensor_uid"] = add_tensor(
                    g,
                    f"seq_len_{side}",
                    rng.choice(("int32", "int32", "int64")),
                    [s["b"], 1, 1, 1],
                    [1, 1, 1, 1],
                )
    return f"feature:{choice}"


def _offset_uid(t):
    return t.get("ragged_offset_tensor_uid")


def make_thd(g, rng=None, offsets_dtype="int32") -> str:
    """Make every operand ragged: packed token-major tensors with element
    multipliers per side (hipDNN's H * D) and the stats ragged on Q's table."""
    s = shape_of(g)
    b = s["b"]
    oq = add_tensor(g, "offsets_q", offsets_dtype, [b + 1, 1, 1, 1], [1, 1, 1, 1])
    okv = add_tensor(g, "offsets_kv", offsets_dtype, [b + 1, 1, 1, 1], [1, 1, 1, 1])
    for role in DATA:
        t = tensor_of(g, role)
        t["strides"] = _strides("thd", t["dims"])
        t["ragged_offset_tensor_uid"] = oq if role in Q_SIDE else okv
        t["ragged_offset_multiplier"] = t["dims"][1] * t["dims"][3]
    st = tensor_of(g, "lse")
    hq, sq = s["h_q"], s["s_q"]
    st["strides"] = [sq * hq, 1, hq, 1]
    st["ragged_offset_tensor_uid"] = oq
    st["ragged_offset_multiplier"] = hq
    return "thd"


def m_ragged(g, rng):
    ragged = [r for r in DATA if _offset_uid(tensor_of(g, r)) is not None]
    if not ragged:
        if rng.random() < 0.7:
            return make_thd(g, rng, rng.choice(("int32", "int32", "int64")))
        role = _pick_role(rng)
        b = shape_of(g)["b"]
        oq = add_tensor(g, "offsets", "int32", [b + 1, 1, 1, 1], [1, 1, 1, 1])
        tensor_of(g, role)["ragged_offset_tensor_uid"] = oq
        return f"ragged_partial:{role}"
    table = tensors(g)
    oq = _offset_uid(tensor_of(g, "q"))
    okv = _offset_uid(tensor_of(g, "k"))
    choice = rng.choice(
        (
            "drop_one",
            "multiplier",
            "offsets_dtype",
            "offsets_dims",
            "offsets_virtual",
            "split_table",
            "stats_dense",
            "stats_kv_table",
            "stats_packed_mult",
            "offsets_ragged",
            "mixed_dtype",
            "offsets_stride",
        )
    )
    if choice == "drop_one":
        role = _pick_role(rng, DATA + ("lse",))
        tensor_of(g, role).pop("ragged_offset_tensor_uid", None)
    elif choice == "multiplier":
        role = _pick_role(rng, DATA + ("lse",))
        t = tensor_of(g, role)
        m = t.get("ragged_offset_multiplier", 1)
        t["ragged_offset_multiplier"] = rng.choice(
            (m + 1, max(1, m - 1), 2 * m, 1, 0, -m, 2**31)
        )
    elif choice in ("offsets_dtype", "mixed_dtype"):
        uid = oq if choice == "offsets_dtype" else okv
        if uid in table:
            table[uid]["data_type"] = rng.choice(
                ("int64", "int8", "uint8", "float", "half", "int32")
            )
    elif choice == "offsets_dims":
        uid = rng.choice((oq, okv))
        if uid in table:
            b = shape_of(g)["b"]
            table[uid]["dims"] = rng.choice(
                ([b, 1, 1, 1], [b + 2, 1, 1, 1], [b + 1, 2, 1, 1], [1, b + 1, 1, 1])
            )
    elif choice == "offsets_virtual":
        uid = rng.choice((oq, okv))
        if uid in table:
            table[uid][rng.choice(("virtual", "is_runtime_pass_by_value"))] = True
    elif choice == "split_table":
        role = _pick_role(rng, ("o", "do", "dq", "v", "dk", "dv"))
        t = tensor_of(g, role)
        src = table.get(_offset_uid(t))
        if src is not None:
            t["ragged_offset_tensor_uid"] = add_tensor(
                g, "offsets_2", src["data_type"], src["dims"], src["strides"]
            )
    elif choice == "stats_dense":
        st = tensor_of(g, "lse")
        st.pop("ragged_offset_tensor_uid", None)
        st.pop("ragged_offset_multiplier", None)
        hq, sq = st["dims"][1], st["dims"][2]
        st["strides"] = [hq * sq, sq, 1, 1]
    elif choice == "stats_kv_table":
        if okv is not None:
            tensor_of(g, "lse")["ragged_offset_tensor_uid"] = okv
    elif choice == "stats_packed_mult":
        st = tensor_of(g, "lse")
        st["ragged_offset_multiplier"] = rng.choice(
            (1, st["dims"][1], 2 * st["dims"][1])
        )
    elif choice == "offsets_ragged":
        uid = rng.choice((oq, okv))
        if uid in table:
            table[uid]["ragged_offset_tensor_uid"] = uid
    else:
        uid = rng.choice((oq, okv))
        if uid in table:
            table[uid]["strides"] = [2, 1, 1, 1]
    return f"ragged:{choice}"


def m_flags(g, rng):
    role = _pick_role(rng, DATA + ("lse",))
    flag = rng.choice(("virtual", "is_runtime_pass_by_value"))
    tensor_of(g, role)[flag] = True
    return f"flag:{role}.{flag}"


def m_dangling_uid(g, rng):
    # a required uid cannot be null in hipDNN JSON; it can name no tensor
    role = _pick_role(rng, DATA + ("lse",))
    group, key = ROLE_KEYS[role]
    node_of(g)[group][key] = new_uid(g) + 1000
    return f"uid:{role}"


def m_dims(g, rng):
    role = _pick_role(rng, DATA + ("lse",))
    t = tensor_of(g, role)
    i = rng.randrange(4)
    t["dims"][i] = rng.choice((t["dims"][i] + 1, max(0, t["dims"][i] - 1), -1, 2))
    return f"dims:{role}[{i}]"


def m_rank(g, rng):
    role = _pick_role(rng, DATA + ("lse",))
    t = tensor_of(g, role)
    t["dims"] = t["dims"][:3]
    t["strides"] = t["strides"][:3]
    return f"rank:{role}"


def m_stats(g, rng):
    st = tensor_of(g, "lse")
    choice = rng.choice(("dtype", "strides", "dims", "alignment"))
    if choice == "dtype":
        st["data_type"] = rng.choice(("half", "bfloat16", "double"))
    elif choice == "strides":
        st["strides"] = [rng.choice((x, x + 1, -1, 0)) for x in st["strides"]]
    elif choice == "dims":
        st["dims"][3] = rng.choice((2, 0))
    else:
        st["alignment"] = rng.choice((1, 2, 4, 6, 8, 12))
    return f"stats:{choice}"


def m_extra_ragged(g, rng):
    s = shape_of(g)
    dims = [s["b"], s["h_q"], s["s_q"], s["s_kv"]]
    offs = add_tensor(g, "offsets_x", "int32", [s["b"] + 1, 1, 1, 1], [1, 1, 1, 1])
    uid = add_tensor(
        g,
        "bias",
        tensor_of(g, "q")["data_type"],
        dims,
        packed(dims),
        ragged_offset_tensor_uid=offs,
    )
    if rng.random() < 0.5:
        node_of(g)["inputs"]["attn_mask_tensor_uid"] = uid
    return "extra_ragged"


def m_num_cus(c, rng):
    c["num_cus"] = rng.choice(CU_COUNTS)
    return f"cus={c['num_cus']}"


# (operator, weight); an operator takes the graph, except num_cus (the candidate).
OPERATORS = (
    (m_dtype_one, 3),
    (m_dtype_all, 3),
    (m_compute, 1),
    (m_stride_d, 1),
    (m_stride_negative, 1),
    (m_stride_overlap, 2),
    (m_stride_odd, 3),
    (m_alignment, 4),
    (m_heads, 5),
    (m_head_dim, 4),
    (m_lengths, 5),
    (m_mask, 5),
    (m_features, 4),
    (m_ragged, 6),
    (m_flags, 1),
    (m_dangling_uid, 1),
    (m_dims, 1),
    (m_rank, 1),
    (m_stats, 2),
    (m_extra_ragged, 1),
    (m_num_cus, 6),
)


def mutate(seed: Seed, rng: random.Random, k: int) -> tuple[str, dict, int]:
    c = {"graph": copy.deepcopy(seed.graph), "num_cus": seed.num_cus}
    ops, weights = zip(*OPERATORS)
    labels = []
    for _ in range(k):
        op = rng.choices(ops, weights)[0]
        before = copy.deepcopy(c)
        try:
            label = op(c, rng) if op is m_num_cus else op(c["graph"], rng)
        except (KeyError, TypeError, IndexError, ValueError, AttributeError):
            # an earlier fault removed what this operator edits
            c, label = before, None
        if label:
            labels.append(label)
    eid = seed.sid + "".join(f"+m:{x}" for x in labels)
    return eid, normalize(c["graph"]), c["num_cus"]


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


def _size(e: dict) -> int:
    return len(canonical(e)) + 1


def _subset_seeds(seed_list: list[Seed], rng: random.Random) -> list[Seed]:
    """Seeds kept unmutated in the subset: every predicate row and catalog row, and a
    sample of the large seed families."""
    keep_all = ("trigger", "accept", "narrow", "boundary", "row")
    caps = {"case": 60, "tuned": 60, "bundle": 60}
    groups: dict[str, list[Seed]] = {}
    for s in seed_list:
        groups.setdefault(s.sid.split(":", 1)[0], []).append(s)
    out = []
    for kind in sorted(groups):
        items = groups[kind]
        if kind in keep_all or len(items) <= caps.get(kind, 0):
            out += items
        else:
            idx = sorted(rng.sample(range(len(items)), caps[kind]))
            out += [items[i] for i in idx]
    return out


def build(arch: str, mode: str, count: int, max_bytes: int, with_request: bool):
    index = list(H.ENGINE_ARCHS).index(arch)
    seed_value = BASE_SEED + index
    rng = random.Random(seed_value)
    skipped: Counter = Counter()
    seed_list = seeds(arch, skipped)
    entries: list[dict] = []
    seen: set[str] = set()

    def push(e: dict) -> bool:
        key = canonical([e["graph"], e["num_cus"]])
        if key in seen:
            return False
        seen.add(key)
        entries.append(e)
        return True

    for s in seed_list:
        if not expressible(s.graph):
            raise AssertionError(f"seed {s.sid} is not a hipDNN-readable graph")
    base = seed_list if mode == "full" else _subset_seeds(seed_list, rng)
    for s in base:
        push(entry(s.sid, arch, s.graph, s.num_cus, with_request))
    if mode == "full":
        target = len(entries) + count
        while len(entries) < target:
            seed = rng.choice(seed_list)
            eid, g, nc = mutate(seed, rng, rng.choice((1, 2, 3)))
            if expressible(g):
                push(entry(eid, arch, g, nc, with_request))
            else:
                skipped["mutation_not_expressible"] += 1
        return _finish(arch, mode, seed_value, entries, skipped, len(seed_list))

    # subset: mutated entries, first lifting every observed code to 10 entries,
    # then up to the size cap
    used = META_RESERVE + sum(_size(e) for e in entries)
    codes = Counter(e["expect"]["code"] for e in entries)
    pool, distinct = [], {}
    tries = 0

    def short() -> bool:
        return any(n + codes[c] < 10 for c, n in Counter(distinct.values()).items())

    while tries < SUBSET_POOL or (short() and tries < 5 * SUBSET_POOL):
        tries += 1
        seed = rng.choice(seed_list)
        eid, g, nc = mutate(seed, rng, rng.choice((1, 2, 3)))
        if not expressible(g):
            skipped["mutation_not_expressible"] += 1
            continue
        e = entry(eid, arch, g, nc, with_request)
        pool.append(e)
        distinct.setdefault(canonical([g, nc]), e["expect"]["code"])
    rest = []
    for e in pool:
        if codes[e["expect"]["code"]] < 10 and used + _size(e) <= max_bytes and push(e):
            codes[e["expect"]["code"]] += 1
            used += _size(e)
        else:
            rest.append(e)
    for e in rest:
        if len(entries) >= count:
            break
        if used + _size(e) > max_bytes:
            continue
        if push(e):
            used += _size(e)
    return _finish(arch, mode, seed_value, entries, skipped, len(seed_list))


def _finish(arch, mode, seed_value, entries, skipped, n_seeds):
    codes = Counter(e["expect"]["code"] for e in entries)
    reachable = [c for c in (*REASON_CODES, H.LOWERING) if codes[c]]
    meta = {
        "generator": GENERATOR,
        "arch": arch,
        "mode": mode,
        "seed": seed_value,
        "rocke_reason_codes": len(REASON_CODES),
        "entries": len(entries),
        "seeds": n_seeds,
        "skipped_seeds": dict(sorted(skipped.items())),
        "codes": {c: codes[c] for c in (*REASON_CODES, H.LOWERING, "OK") if codes[c]},
        "unreachable": [c for c in (*REASON_CODES, H.LOWERING) if c not in reachable],
    }
    return [{"meta": meta}] + entries


def render(records: list[dict]) -> str:
    return "".join(canonical(r) + "\n" for r in records)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", required=True, choices=H.ENGINE_ARCHS)
    ap.add_argument("--mode", choices=("subset", "full"), default="subset")
    ap.add_argument(
        "--count",
        type=int,
        default=None,
        help="subset: entry cap (default 5000); full: mutated entries (default 50000)",
    )
    ap.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="subset size cap"
    )
    ap.add_argument(
        "--with-request",
        action="store_true",
        help="add each request (full mode default)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="fail when the file differs")
    args = ap.parse_args(argv)
    count = (
        args.count
        if args.count is not None
        else (5000 if args.mode == "subset" else 50000)
    )
    with_request = args.with_request or args.mode == "full"
    text = render(build(args.arch, args.mode, count, args.max_bytes, with_request))
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != text:
            print(f"stale corpus {args.out} (rerun {GENERATOR})", file=sys.stderr)
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
