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


def name_by_uid(graph: dict) -> dict:
    """uid -> tensor name, for the graph's declared tensors."""
    return {t["uid"]: t.get("name", "") for t in graph.get("tensors", []) if "uid" in t}


def canonical_uid_map_by_name(graph: dict) -> dict:
    """Deterministic old_uid -> canonical_uid keyed on tensor NAME.

    The C++ graph builder auto-assigns UIDs to operation-created tensors (Y,
    MEAN, epsilon, momentum, ...) by iterating an ``unordered_set``
    (GraphTensorIds.hpp), so the same logical tensor gets a different UID in each
    captured case even when the topology is identical. That non-determinism
    breaks sweep compression, which requires the same role to carry the same UID
    across every case sharing a template.

    Tensor *names* are stable across cases (operation outputs are named
    ``<op>::<ROLE>``; inputs are named by the test), so we assign canonical UIDs
    by sorted name. Every case of a topology then maps a given named tensor to
    the same UID, independent of the builder's arbitrary assignment order. Ties
    on name fall back to the original UID so the map is still total and stable
    within a single graph.
    """
    named = sorted(
        (t.get("name", ""), t["uid"]) for t in graph.get("tensors", []) if "uid" in t
    )
    return {uid: i for i, (_name, uid) in enumerate(named)}


def remap_meta_inputs(meta: dict, m: dict) -> dict:
    """Deep copy of metadata with ``inputs`` fill-spec keys remapped via ``m``.

    ``inputs`` is keyed by stringified tensor UID, so it must be remapped in
    lockstep with the graph whenever UIDs are canonicalized — otherwise a
    tensor's fill spec (range/kind/seed) would end up attached to the wrong
    tensor after remapping.
    """
    out = copy.deepcopy(meta)
    inputs = out.get("inputs")
    if isinstance(inputs, dict):
        out["inputs"] = {str(m.get(int(k), int(k))): v for k, v in inputs.items()}
    return out


def inputs_by_name(meta: dict, graph: dict) -> dict:
    """Metadata ``inputs`` re-keyed by tensor name for UID-agnostic comparison.

    Two graphs describing the same test may label a tensor with different UIDs
    (see :func:`canonical_uid_map_by_name`), so comparing fill specs by UID is
    fragile. Re-keying by the stable tensor name lets a captured graph and its
    placed bundle be compared regardless of how their UIDs were assigned.
    """
    nm = name_by_uid(graph)
    out = {}
    for k, v in meta.get("inputs", {}).items():
        try:
            uid = int(k)
        except (TypeError, ValueError):
            out[k] = v
            continue
        out[nm.get(uid, k)] = v
    return out


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


# Short, readable prefixes for common node-attribute keys so ids like
# "prepad1x1_stride2x2" stay legible instead of dumping the raw key. Keys are
# matched after stripping the "parameters__" / "nN__" namespacing.
_ATTR_KEY_ABBREV = {
    "pre_padding": "prepad",
    "post_padding": "postpad",
    "padding": "pad",
    "stride": "stride",
    "dilation": "dil",
    "epsilon": "eps",
    "relu_lower_clip": "rlo",
    "relu_upper_clip": "rhi",
    "relu_lower_clip_slope": "rslope",
    "compute_data_type": "cdt",
    "mode": "mode",
}


def _attr_base_key(key: str) -> str:
    """Strip nN__ and parameters__ namespacing to the bare attribute name."""
    k = key
    if "__" in k:
        k = k.split("__")[-1]
    return k


def _num(v) -> str:
    """Render a number compactly: drop trailing zeros, keep '.' as 'p'.

    0.009999999776482582 -> 0p01, 2.0 -> 2, 1e-5 -> 0p00001. Keeps float-valued
    attribute tokens short instead of dumping full IEEE precision into the id.
    """
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f == int(f):
        return str(int(f))
    s = f"{f:.4g}"  # cap precision so ids stay short
    return s.replace("-", "neg").replace(".", "p")


def _attr_token(key: str, v) -> str:
    """A short discriminator token for a varying node attribute value.

    Scalars render compactly; lists render x-joined (e.g. [1, 1] -> 1x1), which
    is what distinguishes conv cases that share shape/dtype/layout but differ by
    padding, stride, or dilation. Prefixed with a short key abbreviation so
    several attrs in one id stay distinguishable. Full uniqueness is guaranteed
    by the hash suffix in assign_case_ids, so this only needs to be a readable
    filter hint, not a lossless encoding.
    """
    base = _attr_base_key(key)
    prefix = _ATTR_KEY_ABBREV.get(base, sanitize(base))
    if v is None:
        return f"no{prefix}"
    if isinstance(v, bool):
        return prefix if v else f"no{prefix}"
    if isinstance(v, (int, float)):
        return f"{prefix}{_num(v)}"
    if isinstance(v, str):
        # Collapse known dtype strings (bfloat16 -> bf16) to keep tokens short.
        val = DTYPE_TOKEN.get(v.lower(), sanitize(v))
        return f"{prefix}{val}"
    if (
        isinstance(v, (list, tuple))
        and v
        and all(isinstance(x, (int, float)) for x in v)
    ):
        return prefix + "x".join(_num(x) for x in v)
    return ""


def _layout_rep_tensor(tensors: list) -> dict:
    """Pick the tensor that best carries the layout/shape signal.

    A batchnorm scale/bias tensor like [1, C, 1] has ambiguous strides (NCL and
    NLC are indistinguishable), so it must not drive the layout token. Prefer
    the highest-rank tensor whose dims have at least two distinct non-unit
    extents (so NCHW vs NHWC is actually observable); fall back to the first
    tensor if none qualify.
    """
    if not tensors:
        return {}
    best = None
    best_key = (-1, -1)
    for t in tensors:
        dims = t.get("dims") or []
        rank = len(dims)
        non_unit = sum(1 for d in dims if d != 1)
        key = (rank, non_unit)
        if key > best_key:
            best_key = key
            best = t
    return best or tensors[0]


def derive_case_id(
    entry: dict,
    dtype_varies: bool,
    layout_varies: bool,
    shape_varies: bool,
    feature_keys: list,
) -> str:
    """Build a lowercase_snake_case case id with discriminator tokens.

    Tokens are appended in a stable order: varying shape, dtype, layout, then
    each varying node attribute (padding/stride/dilation/etc.). Only fields
    that actually vary within the sweep contribute, so ids stay as short as the
    sweep allows while remaining a stable filter surface (RFC 0011 4.1).
    """
    values = entry["values"]
    tensors = values.get("tensors", [])
    tokens = []

    # Layout/shape must be read from a tensor that actually carries the layout
    # signal. tensors[0] is often a scale/bias vector ([1, C, 1]) whose strides
    # are ambiguous (NCL and NLC look identical); reading it would tag distinct
    # layouts with the same token and force meaningless _N suffixes. Prefer the
    # highest-rank tensor with a non-degenerate stride ordering.
    rep = _layout_rep_tensor(tensors)

    if shape_varies and rep:
        dims = rep.get("dims") or []
        if dims:
            tokens.append("_".join(str(d) for d in dims))

    if dtype_varies:
        dt = values.get("io_data_type")
        if dt is None and rep:
            dt = rep.get("data_type")
        if dt is not None:
            tokens.append(DTYPE_TOKEN.get(str(dt).lower(), str(dt).lower()))

    if layout_varies and rep:
        lay = infer_layout(rep.get("dims"), rep.get("strides"))
        if lay:
            tokens.append(lay)

    # Cap the number of attribute tokens so ids stay legible as gtest names.
    # A case that differs only in a lower-priority attribute still gets a unique
    # id via the hash suffix appended in assign_case_ids.
    attrs = values.get("attributes", {})
    attr_tokens = []
    for k in feature_keys[:_MAX_ATTR_TOKENS]:
        if k in attrs:
            tok = _attr_token(k, attrs[k])
            if tok:
                attr_tokens.append(tok)
    tokens.extend(attr_tokens)

    base = "_".join(tokens) if tokens else "case"
    return sanitize(base)


# Ceiling on attribute tokens baked into a case id; beyond this the id stops
# being a readable filter hint. Remaining distinctions ride the hash suffix.
_MAX_ATTR_TOKENS = 3


def _case_identity(entry: dict) -> dict:
    """The full set of fields that make a case distinct from its siblings.

    A case is defined not just by its graph values but by how its inputs are
    synthesized: two cases with an identical graph but different per-tensor
    ranges/seeds (e.g. a bias filled from [-0.5, 0.5] vs [-1, 1], or a DERIVED
    tensor) are genuinely different tests. The readable id tokens cannot carry
    this high-dimensional per-tensor data, so it must ride the hash instead —
    otherwise such cases would silently collide.
    """
    meta = entry.get("metadata", {})
    return {
        "values": entry.get("values", {}),
        "inputs": meta.get("inputs", {}),
        "seed": meta.get("seed"),
    }


def _case_hash(entry: dict) -> str:
    """Short stable content hash over a case's full identity, for disambiguation."""
    blob = json.dumps(canon(_case_identity(entry)), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:6]


def assign_case_ids(sweep_cases: list):
    """Fill in unique lowercase_snake_case ids with discriminator tokens.

    Each id is human-readable tokens (shape, dtype, layout, top attrs) plus a
    short content-hash suffix. The tokens make ids filterable (RFC 0011 4.1);
    the hash guarantees uniqueness without unbounded token dumps or opaque _N
    counters that shuffle when cases are added or reordered.
    """
    dtypes, layouts, shapes = set(), set(), set()
    for e in sweep_cases:
        t = e["values"].get("tensors", [])
        if t:
            rep = _layout_rep_tensor(t)
            dt = e["values"].get("io_data_type") or rep.get("data_type")
            dtypes.add(json.dumps(dt))
            shapes.add(json.dumps(rep.get("dims")))
            layouts.add(json.dumps([rep.get("dims"), rep.get("strides")]))
    dtype_varies = len(dtypes) > 1
    layout_varies = len(layouts) > 1
    shape_varies = len(shapes) > 1

    # A node attribute is a discriminator when it takes more than one value
    # across the sweep. Include list-valued attrs (conv pad/stride/dilation) —
    # they are exactly what distinguishes cases sharing shape/dtype/layout.
    feature_vals = defaultdict(set)
    for e in sweep_cases:
        for k, v in e["values"].get("attributes", {}).items():
            if v is None or isinstance(v, (bool, int, float, str, list)):
                feature_vals[k].add(json.dumps(v))
    feature_keys = sorted(k for k, vs in feature_vals.items() if len(vs) > 1)

    # First pass: readable base id for each case.
    bases = [
        derive_case_id(e, dtype_varies, layout_varies, shape_varies, feature_keys)
        for e in sweep_cases
    ]
    base_counts = defaultdict(int)
    for b in bases:
        base_counts[b] += 1

    # Second pass: keep the clean base when it is already unique; otherwise
    # disambiguate with a short content hash over the full case identity
    # (values + input FillSpecs + seed). The hash is derived from the case, not
    # its position, so ids are stable across additions/reordering — unlike an
    # _N counter — and two cases differing only in input range get distinct ids.
    final = {}
    for e, base in zip(sweep_cases, bases):
        if base_counts[base] == 1:
            cid = base
        else:
            cid = f"{base}_{_case_hash(e)}"
            # Defensive: if two truly-identical cases share a hash, fall back to
            # a counter so ids stay unique (an exact-duplicate case is rare but
            # must not silently overwrite).
            if cid in final:
                n = 2
                while f"{cid}_{n}" in final:
                    n += 1
                cid = f"{cid}_{n}"
        final[cid] = True
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
