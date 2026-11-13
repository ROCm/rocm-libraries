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
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath

# --------------------------------------------------------------------------
# Field policy constants
# --------------------------------------------------------------------------

TENSOR_ALWAYS = ("dims", "strides", "data_type")
# Per-case fields of a tensor with a ragged offset link. A token-major ragged tensor's offset
# multiplier is its sequence stride, so it changes with the dims of each case.
TENSOR_RAGGED = ("ragged_offset_multiplier",)
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
# Capture layout (Hop A output)
# --------------------------------------------------------------------------


def capture_suite_case(graph_path, capture_dir=None, suite=None):
    """``(suite, case)`` for a graph in the capture layout, or None if it is not.

    ``--capture-bundles`` writes ``<capture_dir>/<suite>/<case>/<case>.json``,
    where the suite may contain ``/`` (``Smoke/IntegrationGpuConvFp32``) and is
    also recorded as the sidecar's ``operation``. With ``capture_dir``, the
    suite is the path between it and the case directory. Without one, nothing
    says where the suite starts, so ``suite`` (the sidecar's ``operation``) is
    accepted only if the path ends in it. The suite is always ``/``-separated,
    so the reference_source built from it is the same on every platform.
    """
    if capture_dir is None:
        graph_path = graph_path.resolve()
    case_dir = graph_path.parent
    case = case_dir.name
    if graph_path.name.endswith(".meta.json") or graph_path.stem != case:
        return None
    if capture_dir is not None:
        return case_dir.parent.relative_to(capture_dir).as_posix(), case
    if not isinstance(suite, str) or not suite:
        return None
    suite_parts = PurePosixPath(suite).parts
    if not suite_parts or case_dir.parent.parts[-len(suite_parts) :] != suite_parts:
        return None
    return suite, case


def tier_for_suite(suite: str) -> str:
    """The tier folder for a capture suite: ``TIER_MAP`` of its first component."""
    parts = PurePosixPath(suite).parts
    return TIER_MAP.get(parts[0], "quick") if parts else "quick"


def capture_reference_source(suite: str, case: str) -> str:
    """The ``reference_source`` that joins a bundle case to its C++ test."""
    return f"c++ integration suite: {suite}.{case}"


def inputs_error(inputs):
    """Why ``inputs`` can't be remapped (a UID-keyed object), or None if it can."""
    if not isinstance(inputs, dict):
        return f"inputs: expected a JSON object, got {type(inputs).__name__}"
    for k in inputs:
        try:
            int(k)
        except ValueError:
            return f"inputs: key {k!r} is not a tensor UID"
    return None


def read_capture_sidecar(graph_path):
    """Load the ``<stem>.meta.json`` that ``--capture-bundles`` writes beside a graph.

    Returns ``(path, meta, error)``. ``meta`` is None when the sidecar is absent
    (a hand-authored graph) or unusable, and ``error`` is set only in the
    unusable case, saying why. An ``inputs`` that :func:`remap_meta_inputs`
    cannot remap counts as unusable.
    """
    path = graph_path.with_suffix(".meta.json")
    if not path.exists():
        return path, None, None
    try:
        with open(path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return path, None, str(e)
    if not isinstance(meta, dict):
        return path, None, f"expected a JSON object, got {type(meta).__name__}"
    if "inputs" in meta:
        err = inputs_error(meta["inputs"])
        if err is not None:
            return path, None, err
    return path, meta, None


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
    for t in graph.get("tensors", []):
        for _, v in uid_keys(t):
            visit(v)
    return canon


def _remap_uid(u, m):
    if isinstance(u, list):
        return [_remap_uid(x, m) for x in u]
    if u is None:
        return None
    if u not in m:
        raise ValueError(f"uid {u!r} has no entry in the uid map")
    return m[u]


def remap_graph(graph: dict, m: dict) -> dict:
    """Deep copy of graph with all uids remapped and tensors sorted by uid.

    The remapped uids are the node ports, each tensor ``uid`` and each
    tensor-level ``*_tensor_uid`` link (such as ``ragged_offset_tensor_uid``).
    Raises ValueError for a uid that has no entry in ``m``.
    """
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
        for k in list(t):
            if k.endswith("_tensor_uid"):
                t[k] = _remap_uid(t[k], m)
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
    """Deep copy of metadata with every tensor UID remapped via ``m``.

    ``inputs`` is keyed by stringified tensor UID, ``placement_uid`` names the
    tensor a misaligned placement targets, and each ``aliases`` entry
    (``uid``, ``of``), ``start_offsets`` entry (``uid``) and ``live_rows`` entry
    (``uid``, ``lengths``, ``like``) names tensors by UID, so they must be
    remapped in lockstep with the graph whenever UIDs are canonicalized —
    otherwise a tensor's fill spec, misaligned pointer, alias, start offset or
    live rows would end up attached to the wrong tensor after remapping. Entries
    that are not integer UIDs are left as they are; the harness rejects them at
    load.
    """
    out = copy.deepcopy(meta)
    inputs = out.get("inputs")
    if isinstance(inputs, dict):
        out["inputs"] = {str(m.get(int(k), int(k))): v for k, v in inputs.items()}
    uid = out.get("placement_uid")
    if isinstance(uid, int) and not isinstance(uid, bool):
        out["placement_uid"] = m.get(uid, uid)
    for section, keys in (
        ("aliases", ("uid", "of")),
        ("start_offsets", ("uid",)),
        ("live_rows", ("uid", "lengths", "like")),
    ):
        entries = out.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for key in keys:
                if isinstance(entry.get(key), int) and not isinstance(entry[key], bool):
                    entry[key] = m.get(entry[key], entry[key])
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


def aliases_by_name(meta: dict, graph: dict) -> list:
    """Metadata ``aliases`` with tensor names in place of UIDs, sorted by view name.

    Each entry becomes ``{"view": <name>, "of": <name>, "byte_offset": <bytes>}``,
    so the aliases of a captured graph and of its placed bundle can be compared
    however their UIDs were assigned (see :func:`inputs_by_name`). A UID that
    names no tensor of ``graph`` stays as it is.
    """
    nm = name_by_uid(graph)
    out = [
        {
            "view": nm.get(a.get("uid"), a.get("uid")),
            "of": nm.get(a.get("of"), a.get("of")),
            "byte_offset": a.get("byte_offset"),
        }
        for a in meta.get("aliases", [])
        if isinstance(a, dict)
    ]
    return sorted(out, key=lambda a: str(a["view"]))


# --------------------------------------------------------------------------
# Skeleton hash
# --------------------------------------------------------------------------


def skeleton_hash(graph: dict, value_form: bool = False) -> str:
    """Structural fingerprint: node types + wiring + tensor set + tensor links.

    The tensor links are the tensor-level ``*_tensor_uid`` keys, such as
    ``ragged_offset_tensor_uid``.

    With ``value_form``, the fingerprint also records the value form of each
    tensor: whether it has a baked ``value``, its ``value_type`` and its
    ``is_runtime_pass_by_value`` flag.
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
                for k, v in uid_keys(sec):
                    wiring.append((section, k, canon_uid(v)))
        for k, v in uid_keys(node):
            wiring.append(("flat", k, canon_uid(v)))
        parts.append({"type": node.get("type", ""), "wiring": sorted(wiring)})

    # Number each linked tensor in the canonical order of the tensors that link
    # to it. Then the fingerprint does not depend on the uid values.
    for t in sorted(
        (t for t in graph.get("tensors", []) if t.get("uid") in canon),
        key=lambda t: canon[t["uid"]],
    ):
        for _, v in uid_keys(t):
            canon_uid(v)

    tensor_set = sorted(
        (canon_uid(t["uid"]), bool(t.get("virtual", False)))
        for t in graph.get("tensors", [])
        if "uid" in t
    )

    skeleton = {"nodes": parts, "tensors": tensor_set}
    tensor_links = sorted(
        [canon_uid(t["uid"]), k, canon_uid(v)]
        for t in graph.get("tensors", [])
        if "uid" in t
        for k, v in uid_keys(t)
        if v is not None
    )
    if tensor_links:
        skeleton["tensor_links"] = tensor_links
    if value_form:
        skeleton["value_forms"] = sorted(
            [
                canon_uid(t["uid"]),
                "value" in t,
                t.get("value_type"),
                bool(t.get("is_runtime_pass_by_value", False)),
            ]
            for t in graph.get("tensors", [])
            if "uid" in t
        )
    blob = json.dumps(skeleton, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Grouped import (opt-in templating used by import_graph.py)
# --------------------------------------------------------------------------

# Top-level sweep.json key that records the grouping options of a sweep.
IMPORT_MODE_KEY = "import_mode"

# import_mode key that records the file layout of a grouped sweep.
SWEEP_FORMAT_KEY = "format"

# "pretty" is json.dumps(indent=2). "compact" writes one line per member of
# each case and of its values, and one line per tensor in values.tensors.
SWEEP_FORMATS = ("pretty", "compact")


# Top-level values key for templated tensor alignment, keyed by tensor uid.
ALIGNMENT_VALUES_KEY = "alignment"

# Top-level graph fields that every template of a grouped sweep takes from the
# case values. hipDNN derives them from the tensors, so they can differ
# between cases that share a skeleton.
GROUPED_TOP_LEVEL = ("is_override_shape_enabled", "min_required_engine_api_version")


def import_mode_marker(grouping: list, sweep_format: str = None) -> dict:
    """The ``import_mode`` object for a sweep made with these grouping options.

    A ``sweep_format`` other than None is recorded as the ``format`` key.
    """
    marker = {"grouping": sorted(grouping)}
    if sweep_format is not None:
        marker[SWEEP_FORMAT_KEY] = sweep_format
    return marker


def sweep_import_mode(sweep: dict):
    """The ``import_mode`` object of a loaded sweep.json, or None if unmarked."""
    marker = sweep.get(IMPORT_MODE_KEY) if isinstance(sweep, dict) else None
    return marker if isinstance(marker, dict) else None


def import_mode_grouping(marker):
    """The part of an ``import_mode`` object that controls sweep matching.

    This is the object without its ``format`` key, or None for no object.
    The file layout of a sweep never decides which cases share it.
    """
    if marker is None:
        return None
    return {k: v for k, v in marker.items() if k != SWEEP_FORMAT_KEY}


# --------------------------------------------------------------------------
# sweep.json writer
# --------------------------------------------------------------------------


def sweep_format(sweep: dict) -> str:
    """The file layout recorded in a loaded sweep.json.

    An unmarked sweep, or a marker with no ``format`` key, is "pretty".
    """
    marker = sweep_import_mode(sweep)
    if marker is None or SWEEP_FORMAT_KEY not in marker:
        return "pretty"
    return marker[SWEEP_FORMAT_KEY]


def _compact(value) -> str:
    return json.dumps(value, separators=(", ", ": "))


def _key(k) -> str:
    """An object key as json.dumps writes it, for example 1 as "1"."""
    return json.dumps({k: 0})[1:-4]


def _layout_members(obj: dict, indent: str, compact_one) -> list:
    """One line per member of ``obj``; ``compact_one(key, value, comma)`` may expand one."""
    lines = []
    items = list(obj.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        expanded = compact_one(k, v, comma)
        if expanded is None:
            lines.append(f"{indent}{_key(k)}: {_compact(v)}{comma}")
        else:
            lines.extend(expanded)
    return lines


def _layout_values(k, v, comma):
    if k != "tensors" or not isinstance(v, list) or not v:
        return None
    lines = ['        "tensors": [']
    lines += [
        " " * 10 + _compact(t) + ("," if j < len(v) - 1 else "")
        for j, t in enumerate(v)
    ]
    lines.append("        ]" + comma)
    return lines


def _layout_case_member(k, v, comma):
    if k != "values" or not isinstance(v, dict) or not v:
        return None
    lines = ['      "values": {']
    lines += _layout_members(v, " " * 8, _layout_values)
    lines.append("      }" + comma)
    return lines


def _layout_case(case, comma: str) -> list:
    if not isinstance(case, dict) or not case:
        return ["    " + _compact(case) + comma]
    lines = ["    {"]
    lines += _layout_members(case, " " * 6, _layout_case_member)
    lines.append("    }" + comma)
    return lines


def _compact_sweep(sweep: dict) -> str:
    lines = ["{"]
    items = list(sweep.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if k == "cases" and isinstance(v, list) and v:
            lines.append(f"  {_key(k)}: [")
            for j, case in enumerate(v):
                lines += _layout_case(case, "," if j < len(v) - 1 else "")
            lines.append("  ]" + comma)
        else:
            body = json.dumps(v, indent=2).replace("\n", "\n  ")
            lines.append(f"  {_key(k)}: {body}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def dump_sweep(sweep: dict, fmt: str) -> str:
    """The text of a sweep.json in layout ``fmt``, with a final newline.

    Both layouts keep every key, its order and its nesting, so the text loads
    to an object equal to ``sweep``. Raises ValueError for an unknown layout.
    """
    if fmt not in SWEEP_FORMATS:
        raise ValueError(
            f"unknown sweep.json format {fmt!r}; expected one of {', '.join(SWEEP_FORMATS)}"
        )
    if fmt == "pretty" or not isinstance(sweep, dict) or not sweep:
        return json.dumps(sweep, indent=2) + "\n"
    return _compact_sweep(sweep)


def _node_attribute_leaves(node: dict, path: tuple = ()):
    """Yield (path, value) for every non-object attribute value of a node.

    Skips the structural node keys and every ``*_tensor_uid`` key.
    """
    for k, v in node.items():
        if (not path and k in NODE_STRUCTURAL) or k.endswith("_tensor_uid"):
            continue
        if isinstance(v, dict):
            yield from _node_attribute_leaves(v, path + (k,))
        else:
            yield path + (k,), v


def template_node_attributes(template: dict):
    """Replace every non-object node attribute value with a placeholder.

    The placeholder is ``${case.attributes.<key>}``. The key is the attribute
    path joined with ``__``. A key that occurs on more than one node gets an
    ``n<index>__`` prefix.
    """
    nodes = template.get("nodes", [])
    counts = defaultdict(int)
    for node in nodes:
        for path, _ in _node_attribute_leaves(node):
            counts["__".join(path)] += 1
    for ni, node in enumerate(nodes):
        for path, _ in list(_node_attribute_leaves(node)):
            key = "__".join(path)
            if counts[key] > 1:
                key = f"n{ni}__{key}"
            parent = node
            for tok in path[:-1]:
                parent = parent[tok]
            parent[path[-1]] = f"${{case.attributes.{key}}}"


def template_tensor_alignment(template: dict):
    """Replace each tensor ``alignment`` with ``${case.alignment.<uid>}``.

    The placeholder resolves from the top-level ``alignment`` object of the
    case values, so :func:`expand` and the C++ loader both resolve it.
    """
    for t in template.get("tensors", []):
        if "alignment" in t and isinstance(t.get("uid"), int):
            t["alignment"] = f"${{case.{ALIGNMENT_VALUES_KEY}.{t['uid']}}}"


def template_top_level_fields(template: dict):
    """Replace each ``GROUPED_TOP_LEVEL`` field with ``${case.<field>}``.

    The placeholder resolves to the whole value of the case, so an object
    field resolves in one piece in :func:`expand` and in the C++ loader.
    """
    for fld in GROUPED_TOP_LEVEL:
        if fld in template:
            template[fld] = f"${{case.{fld}}}"


def template_tensor_values(template: dict):
    """Replace each baked tensor ``value`` with the ``${case.value}`` placeholder."""
    for t in template.get("tensors", []):
        if "value" in t:
            t["value"] = "${case.value}"


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


def per_case_tensor_fields(tensor: dict) -> tuple:
    """The fields of a tensor that a template takes from each case."""
    if tensor.get("ragged_offset_tensor_uid") is None:
        return TENSOR_ALWAYS
    return TENSOR_ALWAYS + TENSOR_RAGGED


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


UNRESOLVED = object()


def walk_dotted_path(obj, path):
    """Walk a '.'-separated path through nested dicts, mirroring the C++ loader's
    lookupJsonPath(). Returns ``UNRESOLVED`` if any segment fails to resolve."""
    cur = obj
    for tok in path.split("."):
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return UNRESOLVED
    return cur


def _resolve(path, case_values, tensor_uid):
    if tensor_uid is not None:
        for tv in case_values.get("tensors", []):
            if tv.get("uid") != tensor_uid:
                continue
            resolved = walk_dotted_path(tv, path)
            if resolved is not UNRESOLVED:
                return resolved
    resolved = walk_dotted_path(case_values, path)
    if resolved is UNRESOLVED:
        return f"${{UNRESOLVED:{path}}}"
    return resolved


def canon(obj):
    """Canonical JSON string for comparison."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------
# Sweep cases: tensor patches and absent tensors (mirror the C++ sweep loader)
# --------------------------------------------------------------------------

# Sweep case key: the template tensor uids the case's graph does not have.
ABSENT_TENSORS_KEY = "absent_tensors"

# Sweep case key: per-tensor field edits applied after placeholder expansion.
TENSOR_PATCHES_KEY = "tensor_patches"

# Sweep version that a sweep with a case with absent tensors must declare.
ABSENT_TENSORS_SWEEP_VERSION = 2

# Case metadata format_version that a case with absent tensors must declare.
ABSENT_TENSORS_FORMAT_VERSION = 2

_INT64_BOUND = 2**63


def is_node_port_key(key) -> bool:
    """True for a key that ends in ``_tensor_uid``: a node port under inputs or outputs."""
    return isinstance(key, str) and key.endswith("_tensor_uid")


def uid_of(value):
    """The uid a JSON value holds: an int, or a float with an integral value; else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if (
        isinstance(value, float)
        and math.isfinite(value)
        and value == math.trunc(value)
        and -_INT64_BOUND <= value < _INT64_BOUND
    ):
        return int(value)
    return None


def prune_absent_tensors(graph: dict, uids) -> dict:
    """Copy of ``graph`` without the tensors in ``uids``; node ports that hold one become null.

    Nothing else changes and no uid is renumbered (``pruneAbsentTensors`` in the
    C++ loader).
    """
    absent = set(uids)
    out = copy.deepcopy(graph)
    if isinstance(out.get("tensors"), list):
        out["tensors"] = [
            t
            for t in out["tensors"]
            if not (isinstance(t, dict) and "uid" in t and uid_of(t["uid"]) in absent)
        ]
    if isinstance(out.get("nodes"), list):
        for node in out["nodes"]:
            if not isinstance(node, dict):
                continue
            for side in ("inputs", "outputs"):
                ports = node.get(side)
                if not isinstance(ports, dict):
                    continue
                for key, value in ports.items():
                    if is_node_port_key(key) and uid_of(value) in absent:
                        ports[key] = None
    return out


def _uid_references(value, uid, under_uid_key, path, found):
    """Append the path of each value below ``value`` that holds ``uid`` under a ``*uid*`` key."""
    if isinstance(value, dict):
        for k, v in value.items():
            _uid_references(v, uid, "uid" in k, f"{path}.{k}", found)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _uid_references(v, uid, under_uid_key, f"{path}[{i}]", found)
    elif under_uid_key and uid_of(value) == uid:
        found.append(path)


def _node_uid_use(template: dict, uid):
    """(indices of nodes that hold ``uid`` in a node port, paths of every other node use)."""
    port_nodes, other = set(), []
    nodes = template.get("nodes")
    if not isinstance(nodes, list):
        return port_nodes, other
    for index, node in enumerate(nodes):
        node_path = f"nodes[{index}]"
        if not isinstance(node, dict):
            _uid_references(node, uid, False, node_path, other)
            continue
        for field, value in node.items():
            field_path = f"{node_path}.{field}"
            if field not in ("inputs", "outputs") or not isinstance(value, dict):
                _uid_references(value, uid, "uid" in field, field_path, other)
                continue
            for port, held in value.items():
                if is_node_port_key(port) and uid_of(held) == uid:
                    port_nodes.add(index)
                    continue
                _uid_references(held, uid, "uid" in port, f"{field_path}.{port}", other)
    return port_nodes, other


def _other_uid_references(template: dict, uid) -> list:
    """Paths outside the nodes that refer to ``uid``, other than a tensor's own ``uid`` key."""
    found = []
    for field, value in template.items():
        if field == "nodes":
            continue
        if field != "tensors" or not isinstance(value, list):
            _uid_references(value, uid, "uid" in field, field, found)
            continue
        for index, tensor in enumerate(value):
            path = f"tensors[{index}]"
            if not isinstance(tensor, dict):
                _uid_references(tensor, uid, False, path, found)
                continue
            for key, held in tensor.items():
                if key != "uid":
                    _uid_references(held, uid, "uid" in key, f"{path}.{key}", found)
    return found


def absent_tensor_problems(template: dict, uids) -> list:
    """Why the C++ loader refuses these template uids as absent tensors; empty when it accepts.

    These are the template rules of ``prepareAbsentTensors``: each uid is a tensor
    of the template and not virtual; every node port of the template holds a uid
    or null; a node port of one node at most holds the uid, and nothing else
    refers to it.
    """
    uids = list(uids)
    problems = []
    tensors = (
        template.get("tensors") if isinstance(template.get("tensors"), list) else []
    )
    nodes = template.get("nodes") if isinstance(template.get("nodes"), list) else []
    by_uid = {
        uid_of(t.get("uid")): t
        for t in tensors
        if isinstance(t, dict) and uid_of(t.get("uid")) is not None
    }
    if len(set(uids)) != len(uids):
        problems.append("a uid is named twice")
    for uid in uids:
        tensor = by_uid.get(uid)
        if tensor is None:
            problems.append(f"{uid} is not a tensor of the template")
        elif "virtual" in tensor and tensor["virtual"] is not False:
            problems.append(f"{uid} is virtual or has a virtual flag that is not false")
    for index, node in enumerate(nodes):
        for side in ("inputs", "outputs"):
            ports = node.get(side) if isinstance(node, dict) else None
            if not isinstance(ports, dict):
                continue
            for key, value in ports.items():
                if (
                    is_node_port_key(key)
                    and value is not None
                    and uid_of(value) is None
                ):
                    problems.append(
                        f"nodes[{index}].{side}.{key} holds {json.dumps(value)},"
                        " not a uid or null"
                    )
    for uid in uids:
        port_nodes, other = _node_uid_use(template, uid)
        if len(port_nodes) > 1:
            problems.append(f"{uid} is held by node ports of more than one node")
        if other:
            problems.append(
                f"{uid} is held by a node outside a node port, at {other[0]}"
            )
        refs = _other_uid_references(template, uid)
        if refs:
            problems.append(f"{uid} is referenced at {refs[0]}")
    return problems


def apply_tensor_patches(graph: dict, case: dict, *, error_cls=ValueError):
    """Apply a sweep case's ``tensor_patches`` to an expanded graph, as the C++ loader does.

    ``set`` upserts fields, then ``remove`` erases fields, of the tensor with the
    patch uid. Raises ``error_cls`` (``ValueError`` by default) for a patch
    without an integer uid or with a uid that no tensor holds; a caller that
    wants its own exception type for this, such as a subclass it already
    catches elsewhere, may pass it.
    """
    patches = case.get(TENSOR_PATCHES_KEY)
    tensors = graph.get("tensors")
    if not isinstance(patches, list) or not isinstance(tensors, list):
        return
    for patch in patches:
        uid = patch.get("uid") if isinstance(patch, dict) else None
        if isinstance(uid, bool) or not isinstance(uid, int):
            raise error_cls("tensor_patch entry is missing an integer uid")
        target = next(
            (t for t in tensors if isinstance(t, dict) and t.get("uid") == uid), None
        )
        if target is None:
            raise error_cls(f"tensor_patch uid {uid} not found in expanded graph")
        if isinstance(patch.get("set"), dict):
            target.update(copy.deepcopy(patch["set"]))
        if isinstance(patch.get("remove"), list):
            for key in patch["remove"]:
                if isinstance(key, str):
                    target.pop(key, None)


def expand_case(template: dict, case: dict) -> dict:
    """The graph a sweep case describes, built in the C++ loader's order.

    The absent tensors leave a copy of the template first, then the placeholders
    expand, then the tensor patches apply.
    """
    absent = case.get(ABSENT_TENSORS_KEY)
    base = prune_absent_tensors(template, absent) if absent else template
    graph = expand(base, case.get("values", {}))
    apply_tensor_patches(graph, case)
    return graph


def uid_map_onto(graph: dict, template: dict):
    """{graph uid: template uid} matched by tensor name, or None.

    None when a tensor name of either side is not unique, or a tensor of the graph
    has no tensor of the same name in the template.
    """
    g_names = [t.get("name") for t in graph.get("tensors", []) if "uid" in t]
    t_names = [t.get("name") for t in template.get("tensors", []) if "uid" in t]
    if len(set(g_names)) != len(g_names) or len(set(t_names)) != len(t_names):
        return None
    t_uid = {t.get("name"): t["uid"] for t in template.get("tensors", []) if "uid" in t}
    out = {}
    for t in graph.get("tensors", []):
        if "uid" not in t:
            continue
        if t.get("name") not in t_uid:
            return None
        out[t["uid"]] = t_uid[t.get("name")]
    return out


def absent_by_name(case: dict, template: dict) -> list:
    """Sorted names of the template tensors that a sweep case marks absent."""
    absent = set(case.get(ABSENT_TENSORS_KEY) or [])
    return sorted(
        t.get("name", "")
        for t in template.get("tensors", [])
        if "uid" in t and t["uid"] in absent
    )


def absent_case_problems(sweep: dict, template: dict, case: dict) -> list:
    """Why the C++ loader refuses a sweep case with absent tensors; empty when it accepts.

    These are the absent-tensor rules of ``prepareAbsentTensors`` and
    ``checkCaseDoesNotNameAbsentTensors`` (format_version, sweep version, the list,
    the names of absent uids, golden data) and the template rules of
    :func:`absent_tensor_problems`. The metadata parse itself is not checked. A
    case without ``absent_tensors`` has none.
    """
    if ABSENT_TENSORS_KEY not in case:
        return []
    absent = case[ABSENT_TENSORS_KEY]
    meta = case.get("metadata")
    if not isinstance(meta, dict):
        return ["absent_tensors needs case metadata with format_version 2"]
    problems = []
    fv = meta.get("format_version", 1)
    if (
        isinstance(fv, bool)
        or not isinstance(fv, int)
        or fv < ABSENT_TENSORS_FORMAT_VERSION
    ):
        problems.append("absent_tensors needs format_version 2")
    version = sweep.get("version", 1)
    if type(version) is not int or version != ABSENT_TENSORS_SWEEP_VERSION:
        problems.append(
            f"absent_tensors needs sweep version 2, not {sweep.get('version', 1)}"
        )
    if (
        not isinstance(absent, list)
        or not absent
        or any(isinstance(u, bool) or not isinstance(u, int) for u in absent)
    ):
        return problems + ["absent_tensors must be a non-empty array of integer uids"]
    problems += absent_tensor_problems(template, absent)
    absent_set = set(absent)
    values = case.get("values", {})
    if isinstance(values, dict):
        tensors = values.get("tensors", [])
        if not isinstance(tensors, list):
            problems.append("values.tensors must be an array")
            tensors = []
        for entry in tensors:
            if isinstance(entry, dict) and uid_of(entry.get("uid")) in absent_set:
                problems.append(f"{entry['uid']} has a values.tensors entry")
        for field, value in values.items():
            if field != "tensors" and isinstance(value, dict):
                for uid in absent:
                    if str(uid) in value:
                        problems.append(
                            f"{uid} is a key of the case value values.{field}"
                        )
    for patch in case.get(TENSOR_PATCHES_KEY) or []:
        if isinstance(patch, dict) and uid_of(patch.get("uid")) in absent_set:
            problems.append(f"{patch['uid']} has a tensor_patches entry")
    for key in meta.get("inputs") or {}:
        try:
            uid = int(key)
        except (TypeError, ValueError):
            continue
        if uid in absent_set:
            problems.append(f"{key} has a metadata inputs entry")
    for key, fields in (
        ("aliases", ("uid", "of")),
        ("start_offsets", ("uid",)),
        ("live_rows", ("uid", "lengths", "like")),
    ):
        for entry in meta.get(key) or []:
            for f in fields:
                if isinstance(entry, dict) and uid_of(entry.get(f)) in absent_set:
                    problems.append(f"{entry[f]} is named by a metadata {key} entry")
    if case.get("golden") is not None:
        problems.append("absent_tensors and golden data do not mix")
    return problems


def metadata_uid_names(meta: dict, graph: dict) -> dict:
    """Every tensor uid that case metadata names, given as the tensor name.

    Covers the ``inputs`` keys and the uid fields of ``aliases`` (``uid``, ``of``),
    ``start_offsets`` (``uid``) and ``live_rows`` (``uid``, ``lengths``, ``like``).
    A uid that names no tensor of ``graph`` is given as ``<uid N>``.
    """
    names = name_by_uid(graph)

    def name(u):
        uid = uid_of(u)
        if uid is None:
            try:
                uid = int(u)
            except (TypeError, ValueError):
                return f"<not a uid {u!r}>"
        return names.get(uid, f"<uid {uid}>")

    out = {}
    if isinstance(meta.get("inputs"), dict):
        out["inputs"] = sorted(name(k) for k in meta["inputs"])
    for key, fields in (
        ("aliases", ("uid", "of")),
        ("start_offsets", ("uid",)),
        ("live_rows", ("uid", "lengths", "like")),
    ):
        entries = meta.get(key)
        if not isinstance(entries, list):
            continue
        out[key] = [
            {f: name(e[f]) for f in fields if f in e}
            for e in entries
            if isinstance(e, dict)
        ]
    return out


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
# Topology directory naming
# --------------------------------------------------------------------------


_SWEEP_MARKER_FILES = ("graph.template.json", "sweep.json", "support.json")


def _topology_dir_occupied(op_dir: Path, name: str) -> bool:
    """A topology folder is occupied if it already holds a bundle marker file."""
    d = op_dir / name
    return any((d / marker).exists() for marker in _SWEEP_MARKER_FILES)


def next_free_topology_dir(op_dir: Path, preferred: str = "Default") -> str:
    """An unoccupied topology folder name under an operation directory.

    Mirrors place_bundles.py's Default/Variant2/Variant3/... convention for
    multiple topologies of one op: ``preferred`` (the layout-derived name, or
    "Default") is used if free; otherwise the first free ``VariantN`` (N >= 2)
    wins. A folder counts as occupied only once it holds a template, sweep, or
    support marker file, so an empty or unrelated directory is still free.
    """
    if not _topology_dir_occupied(op_dir, preferred):
        return preferred
    n = 2
    while _topology_dir_occupied(op_dir, f"Variant{n}"):
        n += 1
    return f"Variant{n}"


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
    identity = {
        "values": entry.get("values", {}),
        "inputs": meta.get("inputs", {}),
        "seed": meta.get("seed"),
    }
    # A declared alignment (an alignment entry of tensor_patches) and a placement make cases
    # with one graph distinct tests. They ride the hash only when present. The other
    # tensor_patches shipped sweeps carry (is_runtime_pass_by_value, remove value) stay out
    # of it, so no existing case id changes (test_every_shipped_case_id_is_unchanged).
    aligned = [
        {"uid": p.get("uid"), "alignment": p["set"]["alignment"]}
        for p in entry.get("tensor_patches") or []
        if isinstance(p, dict)
        and isinstance(p.get("set"), dict)
        and "alignment" in p["set"]
    ]
    if aligned:
        identity["alignment"] = aligned
    # add_tensors and node_patches are new fields no shipped case has. A case that
    # carries them is a different graph from the case without, so they ride the hash, again
    # only when present.
    for key in ("add_tensors", "node_patches"):
        if entry.get(key):
            identity[key] = entry[key]
    for key in ("placement", "placement_uid"):
        if key in meta:
            identity[key] = meta[key]
    return identity


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
    # A lone case has nothing to vary against, but it is still fully identified
    # by its own shape/dtype/layout/attrs -- treat every field as "varying" so
    # a brand-new single-case sweep gets a descriptive id instead of the bare
    # "case" fallback in derive_case_id.
    solo = len(sweep_cases) == 1
    dtype_varies = solo or len(dtypes) > 1
    layout_varies = solo or len(layouts) > 1
    shape_varies = solo or len(shapes) > 1

    # A node attribute is a discriminator when it takes more than one value
    # across the sweep (or the sweep is the solo case above). Include
    # list-valued attrs (conv pad/stride/dilation) — they are exactly what
    # distinguishes cases sharing shape/dtype/layout.
    feature_vals = defaultdict(set)
    for e in sweep_cases:
        for k, v in e["values"].get("attributes", {}).items():
            if v is None or isinstance(v, (bool, int, float, str, list)):
                feature_vals[k].add(json.dumps(v))
    feature_keys = sorted(k for k, vs in feature_vals.items() if len(vs) > 1 or solo)

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
