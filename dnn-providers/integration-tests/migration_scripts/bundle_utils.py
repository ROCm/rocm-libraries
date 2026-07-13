#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Shared utilities for the C++ graph -> bundle migration pipeline.

Extracted from place_bundles.py so that verify_migration.py, import_graph.py,
and test_migration.py can reuse the same canonicalization, skeleton hashing,
template expansion, and case-id derivation logic.
"""

import copy
import hashlib
import json
from collections import defaultdict

# --------------------------------------------------------------------------
# Field policy constants
# --------------------------------------------------------------------------

TENSOR_ALWAYS = ("dims", "strides", "data_type")
TENSOR_IF_VARIES = ("value", "value_type")
TOP_LEVEL_IF_VARIES = ("io_data_type", "intermediate_data_type", "compute_data_type")
TENSOR_STRUCTURAL = ("uid", "virtual", "name")
NODE_STRUCTURAL = ("type", "name")

TIER_MAP = {
    "Smoke": "quick",
    "Full": "full",
    "Standard": "standard",
    "Comprehensive": "comprehensive",
}

DTYPE_TOKEN = {
    "float": "fp32",
    "half": "fp16",
    "bfloat16": "bfp16",
    "float32": "fp32",
    "float16": "fp16",
}

LAYOUT_BY_RANK = {4: ("nchw", "nhwc"), 5: ("ncdhw", "ndhwc"), 3: ("ncl", "nlc")}


# --------------------------------------------------------------------------
# UID canonicalization + remapping
# --------------------------------------------------------------------------


def uid_keys(obj: dict) -> list:
    """Return sorted (key, uid-or-list) pairs for every *_tensor_uid key."""
    out = []
    for k, v in obj.items():
        if k.endswith("_tensor_uid"):
            out.append((k, v))
    return sorted(out)


def canonical_uid_map(graph: dict) -> dict:
    """Deterministic old_uid -> canonical_uid mapping by first-seen order."""
    canon: dict = {}

    def visit(u):
        if isinstance(u, list):
            for x in u:
                visit(x)
            return
        if u is None:
            return
        if u not in canon:
            canon[u] = len(canon)

    for node in graph.get("nodes", []):
        for section in ("inputs", "outputs"):
            sec = node.get(section, {})
            if isinstance(sec, dict):
                for _, v in uid_keys(sec):
                    visit(v)
        for _, v in uid_keys(node):
            visit(v)
    for t in graph.get("tensors", []):
        if "uid" in t:
            visit(t["uid"])
    return canon


def _remap_uid(u, m):
    if isinstance(u, list):
        return [_remap_uid(x, m) for x in u]
    if u is None:
        return None
    return m.get(u, u)


def remap_graph(graph: dict, m: dict) -> dict:
    """Deep copy of graph with all uids remapped and tensors sorted by uid."""
    g = copy.deepcopy(graph)
    for node in g.get("nodes", []):
        for section in ("inputs", "outputs"):
            sec = node.get(section)
            if isinstance(sec, dict):
                for k in list(sec):
                    if k.endswith("_tensor_uid"):
                        sec[k] = _remap_uid(sec[k], m)
        for k in list(node):
            if k.endswith("_tensor_uid"):
                node[k] = _remap_uid(node[k], m)
    for t in g.get("tensors", []):
        if "uid" in t:
            t["uid"] = _remap_uid(t["uid"], m)
    g["tensors"] = sorted(g.get("tensors", []), key=lambda t: t.get("uid", 0))
    return g


# --------------------------------------------------------------------------
# Skeleton hash
# --------------------------------------------------------------------------


def skeleton_hash(graph: dict) -> str:
    """Structural fingerprint: node types + wiring + tensor set."""
    canon: dict = {}

    def canon_uid(u):
        if isinstance(u, list):
            return [canon_uid(x) for x in u]
        if u is None:
            return None
        if u not in canon:
            canon[u] = len(canon)
        return canon[u]

    parts = []
    for node in graph.get("nodes", []):
        wiring = []
        for section in ("inputs", "outputs"):
            sec = node.get(section, {})
            if isinstance(sec, dict):
                for k, v in uid_keys(sec):
                    wiring.append((section, k, canon_uid(v)))
        for k, v in uid_keys(node):
            wiring.append(("flat", k, canon_uid(v)))
        parts.append({"type": node.get("type", ""), "wiring": sorted(wiring)})

    tensor_set = sorted(
        (canon_uid(t["uid"]), bool(t.get("virtual", False)))
        for t in graph.get("tensors", [])
        if "uid" in t
    )

    skeleton = {"nodes": parts, "tensors": tensor_set}
    blob = json.dumps(skeleton, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Operation name derivation
# --------------------------------------------------------------------------


def derive_operation(graph: dict) -> str:
    """Derive a PascalCase Operation name from the node-type sequence."""
    types = [n.get("type", "") for n in graph.get("nodes", [])]
    names = []
    for t in types:
        base = t[: -len("Attributes")] if t.endswith("Attributes") else t
        if base and base not in names:
            names.append(base)
    if names:
        return "".join(names)
    gname = graph.get("name", "") or "Unknown"
    return gname[:-4] if gname.endswith("Test") else gname


# --------------------------------------------------------------------------
# Tensor helpers
# --------------------------------------------------------------------------


def tensors_by_uid(graph: dict) -> dict:
    return {t["uid"]: t for t in graph.get("tensors", []) if "uid" in t}


# --------------------------------------------------------------------------
# Template expansion (mirrors C++ expandTemplateNode)
# --------------------------------------------------------------------------


def expand(node, case_values, tensor_uid=None):
    """Resolve ${case.<path>} placeholders against case_values."""
    if isinstance(node, str):
        if node.startswith("${case.") and node.endswith("}"):
            path = node[len("${case.") : -1]
            return _resolve(path, case_values, tensor_uid)
        return node
    if isinstance(node, list):
        return [expand(x, case_values, tensor_uid) for x in node]
    if isinstance(node, dict):
        uid = node.get("uid")
        nt = uid if isinstance(uid, int) else tensor_uid
        return {k: expand(v, case_values, nt) for k, v in node.items()}
    return node


def _resolve(path, case_values, tensor_uid):
    if tensor_uid is not None and path in TENSOR_ALWAYS + TENSOR_IF_VARIES:
        for tv in case_values.get("tensors", []):
            if tv.get("uid") == tensor_uid and path in tv:
                return tv[path]
    cur = case_values
    for tok in path.split("."):
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return f"${{UNRESOLVED:{path}}}"
    return cur


def canon(obj):
    """Canonical JSON string for comparison."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# Layout inference + sanitize
# --------------------------------------------------------------------------


def infer_layout(dims, strides):
    """Infer a layout token from dims-vs-strides ordering, best-effort."""
    if not dims or not strides or len(dims) != len(strides):
        return None
    rank = len(dims)
    names = LAYOUT_BY_RANK.get(rank)
    if not names:
        return None
    descending = all(strides[i] >= strides[i + 1] for i in range(rank - 1))
    return names[0] if descending else names[1]


def sanitize(s: str) -> str:
    out = []
    for c in str(s).lower():
        out.append(c if (c.isalnum() or c == "_") else "_")
    return "".join(out)


# --------------------------------------------------------------------------
# Case-ID derivation
# --------------------------------------------------------------------------


def _scalar_attr_token(v) -> str:
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, (int, float, str)):
        return sanitize(str(v))
    return ""


def derive_case_id(
    entry: dict,
    dtype_varies: bool,
    layout_varies: bool,
    shape_varies: bool,
    feature_keys: list,
) -> str:
    """Build a lowercase_snake_case case id with discriminator tokens."""
    values = entry["values"]
    tensors = values.get("tensors", [])
    tokens = []

    if shape_varies and tensors:
        dims = tensors[0].get("dims") or []
        if dims:
            tokens.append("_".join(str(d) for d in dims))

    if dtype_varies:
        dt = values.get("io_data_type")
        if dt is None and tensors:
            dt = tensors[0].get("data_type")
        if dt is not None:
            tokens.append(DTYPE_TOKEN.get(str(dt).lower(), str(dt).lower()))

    if layout_varies and tensors:
        lay = infer_layout(tensors[0].get("dims"), tensors[0].get("strides"))
        if lay:
            tokens.append(lay)

    attrs = values.get("attributes", {})
    for k in feature_keys:
        if k in attrs:
            tok = _scalar_attr_token(attrs[k])
            if tok:
                tokens.append(tok)

    base = "_".join(tokens) if tokens else "case"
    return sanitize(base)


def assign_case_ids(sweep_cases: list):
    """Fill in unique lowercase_snake_case ids with discriminator tokens."""
    dtypes, layouts, shapes = set(), set(), set()
    for e in sweep_cases:
        t = e["values"].get("tensors", [])
        if t:
            dt = e["values"].get("io_data_type") or t[0].get("data_type")
            dtypes.add(json.dumps(dt))
            shapes.add(json.dumps(t[0].get("dims")))
            layouts.add(json.dumps([t[0].get("dims"), t[0].get("strides")]))
    dtype_varies = len(dtypes) > 1
    layout_varies = len(layouts) > 1
    shape_varies = len(shapes) > 1

    feature_vals = defaultdict(set)
    for e in sweep_cases:
        for k, v in e["values"].get("attributes", {}).items():
            if isinstance(v, (bool, int, float, str)):
                feature_vals[k].add(json.dumps(v))
    feature_keys = sorted(k for k, vs in feature_vals.items() if len(vs) > 1)

    seen = {}
    for e in sweep_cases:
        cid = derive_case_id(e, dtype_varies, layout_varies, shape_varies, feature_keys)
        if cid in seen:
            seen[cid] += 1
            cid = f"{cid}_{seen[cid]}"
        else:
            seen[cid] = 1
        e["id"] = cid


# --------------------------------------------------------------------------
# Node attribute helpers (for template construction)
# --------------------------------------------------------------------------


def raw_node_attrs(node: dict):
    """Yield (base_key, value) for knob-eligible node attributes."""
    for k, v in node.items():
        if k in NODE_STRUCTURAL or k in ("inputs", "outputs", "parameters"):
            continue
        yield (k, v)
    params = node.get("parameters")
    if isinstance(params, dict):
        for k, v in params.items():
            yield (f"parameters__{k}", v)
    for section in ("inputs", "outputs"):
        sec = node.get(section, {})
        if isinstance(sec, dict):
            for k, v in sec.items():
                if not k.endswith("_tensor_uid"):
                    yield (k, v)


def ambiguous_attr_keys(graph: dict) -> set:
    """Base attribute keys that appear on more than one node."""
    counts = defaultdict(int)
    for node in graph.get("nodes", []):
        for k, _ in raw_node_attrs(node):
            counts[k] += 1
    return {k for k, n in counts.items() if n > 1}


def node_attr_items(node: dict, node_index: int, ambiguous: set):
    """Yield (namespaced_key, value) for node attributes."""
    for k, v in raw_node_attrs(node):
        yield (f"n{node_index}__{k}" if k in ambiguous else k, v)
