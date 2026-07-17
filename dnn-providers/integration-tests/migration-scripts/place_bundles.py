#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Place captured graph bundles into the template+sweep format (ALMIOPEN-2221, AC #8).

Hop B of the migration pipeline:
    C++ graph test --(Hop A: --capture-bundles)--> standalone bundle
                   --(Hop B: this script)--------> template+sweep

Reads the flat per-case output of ``--capture-bundles`` and groups graphs by
STRUCTURE (topology), collapsing cases that share the same graph skeleton and
differ only in per-case data (dims/strides/dtype/inline values/node attributes)
into one ``graph.template.json`` + ``sweep.json``.

Conforms to the Compressed Template Sweeps spec:
    integration_test_bundles/{Tier}/{Operation}/{TopologyName}/
        graph.template.json
        sweep.json

Design principle -- derive, don't classify. "Structural" is defined narrowly and
mechanically (node types + wiring + tensor set); everything else is a per-case
knob. Round-trip verification is the correctness proof: any bad merge fails
loudly and falls back to a standalone single-graph bundle. Nothing is dropped.

Usage::

    place_bundles.py --capture-dir <path> --output-dir <path> [--dry-run] [--no-verify]
"""

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Field policy (the single shared list, used two ways: masked for the skeleton
# hash, substituted for the template). See the plan for the full rationale.
# --------------------------------------------------------------------------

# Tensor fields ALWAYS templatized per-tensor (C++ requiresPerTensorValue).
_TENSOR_ALWAYS = ("dims", "strides", "data_type")
# Tensor fields templatized only if they vary across the pile (inline constants).
_TENSOR_IF_VARIES = ("value", "value_type")
# Top-level graph fields templatized only if they vary.
_TOP_LEVEL_IF_VARIES = ("io_data_type", "intermediate_data_type", "compute_data_type")

# Structural tensor fields: identity, never templatized.
_TENSOR_STRUCTURAL = ("uid", "virtual", "name")
# Structural node keys: never templatized (name is cosmetic-constant).
_NODE_STRUCTURAL = ("type", "name")

# Capture tier -> spec tier folder.
_TIER_MAP = {
    "Smoke": "quick",
    "Full": "full",
    "Standard": "standard",
    "Comprehensive": "comprehensive",
}


@dataclass
class CapturedCase:
    suite: str
    case_name: str
    graph_path: Path
    meta_path: Path
    graph: dict
    meta: dict
    tier: str = "quick"
    original_graph: dict = None


@dataclass
class Bucket:
    """A set of captured cases sharing one graph skeleton."""

    skeleton_hash: str
    tier: str
    operation: str
    cases: list = field(default_factory=list)
    topology_name: str = ""


@dataclass
class Stats:
    cases_found: int = 0
    buckets: int = 0
    sweeps_written: int = 0
    sweep_cases: int = 0
    standalone_written: int = 0
    verify_pass: int = 0
    verify_fail: int = 0
    errors: list = field(default_factory=list)


# --------------------------------------------------------------------------
# 1. Discovery
# --------------------------------------------------------------------------


def discover_captures(capture_dir: Path) -> list[CapturedCase]:
    """Walk capture directory and load all captured cases.

    C++ capture writes <capture-dir>/<suiteName>/<safeCaseName>/<safeCaseName>.json
    where suiteName may contain '/' (e.g. 'Smoke/IntegrationGpuConvFp32'). We find
    cases by locating .json files (excluding .meta.json) whose stem matches the
    parent directory name.
    """
    cases = []
    if not capture_dir.is_dir():
        print(
            f"place_bundles: capture dir does not exist: {capture_dir}", file=sys.stderr
        )
        return cases

    for graph_path in sorted(capture_dir.rglob("*.json")):
        if graph_path.name.endswith(".meta.json"):
            continue

        case_dir = graph_path.parent
        case_name = case_dir.name
        if graph_path.stem != case_name:
            continue

        suite_rel = case_dir.parent.relative_to(capture_dir)
        suite_name = str(suite_rel)

        try:
            with open(graph_path) as f:
                graph = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: bad graph {graph_path}: {e}", file=sys.stderr)
            continue

        meta_path = case_dir / f"{case_name}.meta.json"
        meta = {"format_version": 1}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARN: bad meta {meta_path}: {e}", file=sys.stderr)

        tier = (
            _TIER_MAP.get(suite_rel.parts[0], "quick") if suite_rel.parts else "quick"
        )

        cases.append(
            CapturedCase(
                suite=suite_name,
                case_name=case_name,
                graph_path=graph_path,
                meta_path=meta_path,
                graph=graph,
                meta=meta,
                tier=tier,
            )
        )
    return cases


# --------------------------------------------------------------------------
# 2. Canonicalization + skeleton hash (grouping key)
# --------------------------------------------------------------------------


def _uid_keys(obj: dict) -> list:
    """Return (key, uid-or-list) pairs for every *_tensor_uid key in a dict."""
    out = []
    for k, v in obj.items():
        if k.endswith("_tensor_uid"):
            out.append((k, v))
    return sorted(out)


def canonical_uid_map(graph: dict) -> dict:
    """Deterministic old_uid -> canonical_uid mapping.

    uids are arbitrary internal labels; the same logical tensor gets different
    numbers across suites (1d/2d/3d, layouts). Renumbering by first-seen order in
    the topo-sorted node walk (inputs then outputs, keys sorted) makes isomorphic
    graphs identical, so a template's wiring matches every case in its bucket.
    Order here MUST match skeleton_hash's walk.
    """
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
                for _, v in _uid_keys(sec):
                    visit(v)
        for _, v in _uid_keys(node):
            visit(v)
    # any tensor uids not referenced by nodes get numbered last
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
    """Return a deep copy of graph with all uids remapped via m and tensors
    sorted by canonical uid (so array order matches across a bucket)."""
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


def skeleton_hash(graph: dict) -> str:
    """Compute a structural fingerprint: node types + wiring + tensor set.

    Canonical uid renumbering (first-seen order across the topo-sorted node walk)
    ensures graphs that differ only in uid labels hash identically. These
    canonical uids are used ONLY for hashing; the emitted template/sweep/verify
    use original uids.
    """
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
                for k, v in _uid_keys(sec):
                    wiring.append((section, k, canon_uid(v)))
        # flat *_tensor_uid keys directly on the node (e.g. Reduction)
        for k, v in _uid_keys(node):
            wiring.append(("flat", k, canon_uid(v)))
        parts.append({"type": node.get("type", ""), "wiring": sorted(wiring)})

    # tensor set: canonical uid + virtual flag, count implicit in the list length
    tensor_set = sorted(
        (canon_uid(t["uid"]), bool(t.get("virtual", False)))
        for t in graph.get("tensors", [])
        if "uid" in t
    )

    skeleton = {"nodes": parts, "tensors": tensor_set}
    blob = json.dumps(skeleton, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# 3. Operation name (semantic, derived from node-type sequence)
# --------------------------------------------------------------------------


def derive_operation(graph: dict) -> str:
    """Derive a PascalCase Operation name from the node-type sequence.

    Strips the 'Attributes' suffix hipdnn appends to node types. For multi-node
    graphs, joins node op names. Falls back to the graph 'name' field cleaned up.
    """
    types = [n.get("type", "") for n in graph.get("nodes", [])]
    names = []
    for t in types:
        base = t[: -len("Attributes")] if t.endswith("Attributes") else t
        if base and base not in names:
            names.append(base)
    if names:
        return "".join(names)
    # fallback: graph name minus a trailing 'Test'
    gname = graph.get("name", "") or "Unknown"
    return gname[:-4] if gname.endswith("Test") else gname


# --------------------------------------------------------------------------
# 4. Knob detection + template / sweep construction
# --------------------------------------------------------------------------


def _tensors_by_uid(graph: dict) -> dict:
    return {t["uid"]: t for t in graph.get("tensors", []) if "uid" in t}


def _raw_node_attrs(node: dict):
    """Yield (base_key, value) for knob-eligible node attributes of one node.

    Sources: flat node scalars, nested parameters{} (prefixed 'parameters__'),
    and non-uid keys in inputs/outputs. Base keys never contain '.', since the
    resolver dot-splits placeholder paths.
    """
    for k, v in node.items():
        if k in _NODE_STRUCTURAL or k in ("inputs", "outputs", "parameters"):
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


def _ambiguous_attr_keys(graph: dict) -> set:
    """Base attribute keys that appear on more than one node — these must be
    node-scoped (n<idx>__key) so the flat values.attributes namespace has no
    collisions when several nodes expose the same attribute."""
    counts = defaultdict(int)
    for node in graph.get("nodes", []):
        for k, _ in _raw_node_attrs(node):
            counts[k] += 1
    return {k for k, n in counts.items() if n > 1}


def _node_attr_items(node: dict, node_index: int, ambiguous: set):
    """Yield (namespaced_key, value). Keys shared across nodes get an
    'n<idx>__' prefix; unique keys stay bare (clean names for single-node ops)."""
    for k, v in _raw_node_attrs(node):
        yield (f"n{node_index}__{k}" if k in ambiguous else k, v)


def detect_and_build(bucket: Bucket):
    """Analyze a bucket, returning (template, sweep_cases, error). error is
    reserved for future unmergeable cases; node-scoped attribute keys make flat-
    namespace collisions impossible, so it is currently always None.
    """
    cases = bucket.cases
    rep = cases[0].graph  # representative for structure walk
    ambiguous = _ambiguous_attr_keys(rep)

    # ---- top-level dtype knobs ----
    top_varies = set()
    for fld in _TOP_LEVEL_IF_VARIES:
        vals = {json.dumps(c.graph.get(fld)) for c in cases}
        if len(vals) > 1:
            top_varies.add(fld)

    # ---- tensor knobs ----
    rep_tensors = _tensors_by_uid(rep)
    tensor_value_varies = {}  # uid -> set of _TENSOR_IF_VARIES fields that vary
    for uid in rep_tensors:
        varies = set()
        for fld in _TENSOR_IF_VARIES:
            vals = {
                json.dumps(_tensors_by_uid(c.graph).get(uid, {}).get(fld))
                for c in cases
            }
            if len(vals) > 1:
                varies.add(fld)
        tensor_value_varies[uid] = varies

    # ---- node attribute knobs (node-scoped flat namespace) ----
    # Keys shared across nodes are scoped as n<idx>__key, so there is never a
    # flat-namespace collision. A key is templatized iff its value varies.
    attr_varies = set()  # namespaced attr keys that vary across the pile
    for ni in range(len(rep.get("nodes", []))):
        for attr_key, _ in _node_attr_items(rep["nodes"][ni], ni, ambiguous):
            vals = set()
            for c in cases:
                nodes = c.graph.get("nodes", [])
                if ni < len(nodes):
                    d = dict(_node_attr_items(nodes[ni], ni, ambiguous))
                    vals.add(json.dumps(d.get(attr_key)))
            if len(vals) > 1:
                attr_varies.add(attr_key)

    # ---- build template (deep copy of representative, knobs -> placeholders) ----
    template = copy.deepcopy(rep)
    for fld in top_varies:
        template[fld] = f"${{case.{fld}}}"
    for t in template.get("tensors", []):
        uid = t.get("uid")
        for fld in _TENSOR_ALWAYS:
            if fld in t:
                t[fld] = f"${{case.{fld}}}"
        for fld in tensor_value_varies.get(uid, set()):
            if fld in t:
                t[fld] = f"${{case.{fld}}}"
    for ni, node in enumerate(template.get("nodes", [])):
        _apply_attr_placeholders(node, ni, ambiguous, attr_varies)

    # ---- build sweep cases ----
    sweep_cases = []
    for c in cases:
        values = {}
        for fld in top_varies:
            values[fld] = c.graph.get(fld)
        # tensors
        tv = []
        ctensors = _tensors_by_uid(c.graph)
        for uid in sorted(rep_tensors):
            entry = {"uid": uid}
            src = ctensors.get(uid, {})
            for fld in _TENSOR_ALWAYS:
                if fld in src:
                    entry[fld] = src[fld]
            for fld in tensor_value_varies.get(uid, set()):
                if fld in src:
                    entry[fld] = src[fld]
            tv.append(entry)
        values["tensors"] = tv
        # attributes (node-scoped flat namespace)
        attrs = {}
        for ni, node in enumerate(c.graph.get("nodes", [])):
            for attr_key, v in _node_attr_items(node, ni, ambiguous):
                if attr_key in attr_varies:
                    attrs[attr_key] = v
        if attrs:
            values["attributes"] = attrs

        meta = dict(c.meta)
        meta.setdefault("format_version", 1)
        meta["ported_from"] = f"c++ integration suite: {c.suite}.{c.case_name}"

        sweep_cases.append(
            {"id": None, "values": values, "metadata": meta, "_origin": c}
        )  # id filled after naming; _origin for verify
    return template, sweep_cases, None


def _apply_attr_placeholders(
    node: dict, node_index: int, ambiguous: set, attr_varies: set
):
    """Replace varying node attributes with ${case.attributes.<nskey>}
    placeholders, where <nskey> matches the node-scoped key from _node_attr_items."""

    def nskey(base):
        return f"n{node_index}__{base}" if base in ambiguous else base

    for k in list(node.keys()):
        if k in _NODE_STRUCTURAL or k in ("inputs", "outputs", "parameters"):
            continue
        if nskey(k) in attr_varies:
            node[k] = f"${{case.attributes.{nskey(k)}}}"
    params = node.get("parameters")
    if isinstance(params, dict):
        for k in list(params.keys()):
            if nskey(f"parameters__{k}") in attr_varies:
                params[k] = f"${{case.attributes.{nskey(f'parameters__{k}')}}}"
    for section in ("inputs", "outputs"):
        sec = node.get(section, {})
        if isinstance(sec, dict):
            for k in list(sec.keys()):
                if not k.endswith("_tensor_uid") and nskey(k) in attr_varies:
                    sec[k] = f"${{case.attributes.{nskey(k)}}}"


# --------------------------------------------------------------------------
# 5. CaseId derivation (discriminator tokens from what varies)
# --------------------------------------------------------------------------

_DTYPE_TOKEN = {
    "float": "fp32",
    "half": "fp16",
    "bfloat16": "bfp16",
    "float32": "fp32",
    "float16": "fp16",
}
_LAYOUT_BY_RANK = {4: ("nchw", "nhwc"), 5: ("ncdhw", "ndhwc"), 3: ("ncl", "nlc")}


def _infer_layout(dims, strides):
    """Infer a layout token from dims-vs-strides ordering, best-effort."""
    if not dims or not strides or len(dims) != len(strides):
        return None
    rank = len(dims)
    names = _LAYOUT_BY_RANK.get(rank)
    if not names:
        return None
    # channels-last iff the channel dim (index 1) has stride 1 among spatial
    # heuristic: contiguous NCHW has descending strides; NHWC has stride[1]==1-ish
    descending = all(strides[i] >= strides[i + 1] for i in range(rank - 1))
    return names[0] if descending else names[1]


def _scalar_attr_token(v) -> str:
    """A short id token for a scalar attribute value (e.g. reduction mode 'add')."""
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, (int, float, str)):
        return _sanitize(str(v))
    return ""  # skip lists/objects (padding/stride) — too long for an id token


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

    # shape token from the first (input) tensor
    if shape_varies and tensors:
        dims = tensors[0].get("dims") or []
        if dims:
            tokens.append("_".join(str(d) for d in dims))

    # dtype token
    if dtype_varies:
        dt = values.get("io_data_type")
        if dt is None and tensors:
            dt = tensors[0].get("data_type")
        if dt is not None:
            tokens.append(_DTYPE_TOKEN.get(str(dt).lower(), str(dt).lower()))

    # layout token
    if layout_varies and tensors:
        lay = _infer_layout(tensors[0].get("dims"), tensors[0].get("strides"))
        if lay:
            tokens.append(lay)

    # feature tokens: scalar attributes that vary across the pile (e.g. mode)
    attrs = values.get("attributes", {})
    for k in feature_keys:
        if k in attrs:
            tok = _scalar_attr_token(attrs[k])
            if tok:
                tokens.append(tok)

    base = "_".join(tokens) if tokens else "case"
    return _sanitize(base)


def _sanitize(s: str) -> str:
    out = []
    for c in str(s).lower():
        out.append(c if (c.isalnum() or c == "_") else "_")
    return "".join(out)


def assign_case_ids(sweep_cases: list):
    """Fill in unique lowercase_snake_case ids with discriminator tokens."""
    # determine which axes vary across the pile
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

    # feature keys: scalar attributes that take >1 distinct value across the pile
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
# 6. Verify (round-trip: expand template, compare to original)
# --------------------------------------------------------------------------


def _expand(node, case_values, tensor_uid=None):
    """Mirror C++ expandTemplateNode: resolve ${case.<path>} placeholders."""
    if isinstance(node, str):
        if node.startswith("${case.") and node.endswith("}"):
            path = node[len("${case.") : -1]
            return _resolve(path, case_values, tensor_uid)
        return node
    if isinstance(node, list):
        return [_expand(x, case_values, tensor_uid) for x in node]
    if isinstance(node, dict):
        uid = node.get("uid")
        nt = uid if isinstance(uid, int) else tensor_uid
        return {k: _expand(v, case_values, nt) for k, v in node.items()}
    return node


def _resolve(path, case_values, tensor_uid):
    # per-tensor resolution first
    if tensor_uid is not None and path in _TENSOR_ALWAYS + _TENSOR_IF_VARIES:
        for tv in case_values.get("tensors", []):
            if tv.get("uid") == tensor_uid and path in tv:
                return tv[path]
    # dotted path (attributes.foo, attributes.parameters.bar) or top-level scalar
    cur = case_values
    for tok in path.split("."):
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return f"${{UNRESOLVED:{path}}}"
    return cur


def verify_case(template: dict, entry: dict) -> bool:
    expanded = _expand(template, entry["values"])
    original = entry["_origin"].graph
    return _canon(expanded) == _canon(original)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# 7. Writers
# --------------------------------------------------------------------------


def write_sweep(
    target: Path, bucket: Bucket, template: dict, sweep_cases: list, dry_run: bool
):
    out_dir = target / bucket.tier / bucket.operation / bucket.topology_name
    cases_out = []
    for e in sweep_cases:
        cases_out.append(
            {"id": e["id"], "values": e["values"], "metadata": e["metadata"]}
        )
    sweep = {"version": 1, "cases": cases_out}
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "graph.template.json", "w") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
        with open(out_dir / "sweep.json", "w") as f:
            json.dump(sweep, f, indent=2)
            f.write("\n")
    return out_dir


def write_standalone(target: Path, case: CapturedCase, reason: str, dry_run: bool):
    # standalone bundles keep the ORIGINAL (un-canonicalized) graph verbatim
    graph = case.original_graph if case.original_graph is not None else case.graph
    op = derive_operation(graph)
    tensors = _tensors_by_uid(graph)
    first = tensors[min(tensors)] if tensors else {}
    dt = graph.get("io_data_type") or first.get("data_type") or "unknown"
    dtok = _DTYPE_TOKEN.get(str(dt).lower(), str(dt).lower())
    layout = _infer_layout(first.get("dims"), first.get("strides")) or "any"
    bundle_name = _sanitize(case.case_name)
    out_dir = target / case.tier / op / layout / dtok / bundle_name
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{bundle_name}.json", "w") as f:
            json.dump(graph, f, indent=2)
            f.write("\n")
        meta = dict(case.meta)
        meta.setdefault("format_version", 1)
        meta["ported_from"] = f"c++ integration suite: {case.suite}.{case.case_name}"
        meta["standalone_reason"] = reason
        with open(out_dir / f"{bundle_name}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
            f.write("\n")
    return out_dir


# --------------------------------------------------------------------------
# 8. Main
# --------------------------------------------------------------------------


def assign_topology_names(buckets: list):
    """Provisional TopologyName: Default when an operation has one topology,
    Default/Variant2/... (hash-ordered) when several. Humans curate later."""
    by_op = defaultdict(list)
    for b in buckets:
        by_op[(b.tier, b.operation)].append(b)
    for _, group in by_op.items():
        group.sort(key=lambda b: b.skeleton_hash)
        for i, b in enumerate(group):
            b.topology_name = "Default" if i == 0 else f"Variant{i + 1}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--capture-dir",
        type=Path,
        required=True,
        help="root of --capture-bundles output",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="root of output tree (e.g. integration_test_bundles/)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written without writing",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip round-trip verification (NOT recommended)",
    )
    args = ap.parse_args()

    stats = Stats()

    cases = discover_captures(args.capture_dir)
    stats.cases_found = len(cases)
    if not cases:
        print("place_bundles: no captured cases found", file=sys.stderr)
        return 1

    # group by (tier, skeleton hash), and canonicalize each graph into a
    # consistent uid space so a bucket's template wiring matches every case.
    # (uids are arbitrary internal labels that differ across suites.) Standalone
    # fallback keeps the ORIGINAL graph, so preserve it separately.
    grouped: dict = defaultdict(list)
    for c in cases:
        h = skeleton_hash(c.graph)
        c.original_graph = c.graph
        c.graph = remap_graph(c.graph, canonical_uid_map(c.graph))
        grouped[(c.tier, h)].append(c)

    buckets = []
    for (tier, h), group in grouped.items():
        buckets.append(
            Bucket(
                skeleton_hash=h,
                tier=tier,
                operation=derive_operation(group[0].graph),
                cases=group,
            )
        )
    stats.buckets = len(buckets)
    assign_topology_names(buckets)

    topology_map = []

    for bucket in sorted(buckets, key=lambda b: (b.tier, b.operation, b.topology_name)):
        # size-1 bucket -> standalone
        if len(bucket.cases) == 1:
            reason = "single-case topology (no sweep benefit)"
            out = write_standalone(
                args.output_dir, bucket.cases[0], reason, args.dry_run
            )
            stats.standalone_written += 1
            print(f"  standalone: {out}  ({reason})", file=sys.stderr)
            continue

        template, sweep_cases, err = detect_and_build(bucket)
        if err is not None:
            for c in bucket.cases:
                write_standalone(args.output_dir, c, err, args.dry_run)
                stats.standalone_written += 1
            print(
                f"  SKIP->standalone {bucket.operation}/{bucket.topology_name} "
                f"({len(bucket.cases)} cases): {err}",
                file=sys.stderr,
            )
            continue

        assign_case_ids(sweep_cases)

        # verify
        if not args.no_verify:
            failed = [e for e in sweep_cases if not verify_case(template, e)]
            if failed:
                # fall back the whole bucket to standalone; nothing dropped
                for c in bucket.cases:
                    write_standalone(
                        args.output_dir, c, "round-trip verify failed", args.dry_run
                    )
                    stats.standalone_written += 1
                stats.verify_fail += len(failed)
                print(
                    f"  VERIFY FAIL {bucket.operation}/{bucket.topology_name}: "
                    f"{len(failed)}/{len(sweep_cases)} cases -> standalone",
                    file=sys.stderr,
                )
                continue
            stats.verify_pass += len(sweep_cases)

        out = write_sweep(args.output_dir, bucket, template, sweep_cases, args.dry_run)
        stats.sweeps_written += 1
        stats.sweep_cases += len(sweep_cases)
        topology_map.append(
            {
                "skeleton_hash": bucket.skeleton_hash,
                "tier": bucket.tier,
                "operation": bucket.operation,
                "topology_name": bucket.topology_name,
                "path": str(out),
                "case_count": len(sweep_cases),
            }
        )
        print(
            f"  sweep: {bucket.tier}/{bucket.operation}/{bucket.topology_name}  "
            f"({len(sweep_cases)} graphs -> 1 sweep)",
            file=sys.stderr,
        )

    # The topology map is a migration artifact for human curation, NOT test
    # data. Write it OUTSIDE the bundle tree so bundle discovery (which scans
    # every *.json recursively) does not try to load it as a graph.
    if not args.dry_run and topology_map:
        report_dir = args.output_dir.parent / ".migration_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        map_path = report_dir / "topology_map.json"
        with open(map_path, "w") as f:
            json.dump({"version": 1, "topologies": topology_map}, f, indent=2)
            f.write("\n")
        print(f"  topology map:      {map_path}", file=sys.stderr)

    # summary
    print("== place_bundles ==", file=sys.stderr)
    print(f"  capture dir:       {args.capture_dir}", file=sys.stderr)
    print(f"  cases found:       {stats.cases_found}", file=sys.stderr)
    print(f"  skeletons:         {stats.buckets}", file=sys.stderr)
    print(
        f"  sweeps written:    {stats.sweeps_written} ({stats.sweep_cases} cases)",
        file=sys.stderr,
    )
    print(f"  standalone:        {stats.standalone_written}", file=sys.stderr)
    if not args.no_verify:
        print(f"  verify pass:       {stats.verify_pass}", file=sys.stderr)
        print(f"  verify fail:       {stats.verify_fail}", file=sys.stderr)
    total_out = stats.sweep_cases + stats.standalone_written
    print(f"  accounted graphs:  {total_out} / {stats.cases_found}", file=sys.stderr)

    if total_out != stats.cases_found:
        print(
            f"  ERROR: {stats.cases_found - total_out} graphs unaccounted for!",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
