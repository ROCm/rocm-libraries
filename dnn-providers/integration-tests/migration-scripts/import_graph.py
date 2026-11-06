#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Import a single graph JSON into the bundle tree with duplicate detection.

The incremental, idempotent counterpart to the batch ``place_bundles.py``:
safe to run repeatedly, never creates duplicate cases.

Duplicate detection (two levels):
  * Exact dup — expand each existing case; if any expands byte-identical
    to the input graph AND same seed/inputs -> DUPLICATE (skip by default).
  * Structural dup, new knobs — same skeleton hash, different dims/strides/
    dtype/attrs -> legitimate new case: append to existing sweep.json.
  * No structural match — create a new single-case template+sweep.

Dup policy: default is skip-and-report (idempotent, safe to re-run).
  --force   appends even if an exact dup exists.
  --strict  turns an exact dup into a non-zero exit (for CI gates).

Metadata: the ``<stem>.meta.json`` sidecar that ``--capture-bundles`` writes
beside each graph is the metadata base, exactly as in ``place_bundles.py``, so
the captured seed and input fill specs carry over with no flags. ``--meta`` and
``--seed`` override it, with a warning naming both values whenever an override
contradicts the sidecar or the ``reference_source`` derived from it; ``--strict``
turns that warning into a non-zero exit. ``--meta`` values for the non-string
keys (``seed``, ``format_version``, ``minimum_vram_mb``, ``inputs``) parse as
JSON, so ``seed=42`` is the number 42; every other key is stored as the string
given. A sidecar that exists but is unusable (unreadable, not an object, an
``inputs`` key that is not a UID, an ``operation`` that is not a non-empty
string) is refused with a non-zero exit, with or without ``--strict``: without
it the case would land in ``quick/`` with no ``reference_source``.

Tier: unless ``--tier`` is given, the tier comes from the sidecar's suite
prefix via ``TIER_MAP`` (``Full/...`` -> ``full/``), as in ``place_bundles.py``;
with no sidecar, or one with no ``operation``, it defaults to ``quick``. A
``--tier`` that contradicts the sidecar warns like any other override.

Grouped import (opt-in): ``--template-attributes``, ``--template-alignment``
and ``--skeleton-value-form`` let one sweep hold generated cases with different
node attributes, alignments or baked values. ``--op-dir``, ``--topology``,
``--only-op-dir``, ``--require-marked-sweep`` and ``--case-id`` control where a
case goes and its id. Each of these flags writes an ``import_mode`` object into
a new ``sweep.json``, and the new template takes ``is_override_shape_enabled``
and ``min_required_engine_api_version`` from the case values. Default mode
never appends into a sweep with this object.
Default mode still reports an exact duplicate of a case in such a sweep.
``--sweep-format`` sets the file layout of a new grouped sweep (default
``compact``) and records it in the object. An append keeps the layout of the sweep.
See README.md.

Usage::

    import_graph.py --graph case.json --bundle-dir integration-test-bundles/ \\
        [--tier TIER] [--meta reference_source="..."] [--dry-run] [--strict] [--force]
        [--op-dir NAME] [--topology NAME] [--template-attributes]
        [--template-alignment] [--skeleton-value-form] [--only-op-dir]
        [--require-marked-sweep] [--case-id ID] [--sweep-format {pretty,compact}]
"""

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path

from bundle_utils import (
    ABSENT_TENSORS_FORMAT_VERSION,
    ABSENT_TENSORS_KEY,
    ABSENT_TENSORS_SWEEP_VERSION,
    GROUPED_TOP_LEVEL,
    TENSOR_PATCHES_KEY,
    TENSOR_ALWAYS,
    TENSOR_RAGGED,
    TENSOR_IF_VARIES,
    IMPORT_MODE_KEY,
    TOP_LEVEL_IF_VARIES,
    SWEEP_FORMATS,
    _case_hash,
    absent_case_problems,
    absent_tensor_problems,
    dump_sweep,
    import_mode_grouping,
    import_mode_marker,
    assign_case_ids,
    canon,
    canonical_uid_map_by_name,
    capture_reference_source,
    capture_suite_case,
    derive_operation,
    expand,
    expand_case,
    infer_layout,
    inputs_error,
    metadata_uid_names,
    next_free_topology_dir,
    prune_absent_tensors,
    read_capture_sidecar,
    remap_graph,
    remap_meta_inputs,
    sanitize,
    skeleton_hash,
    sweep_format,
    sweep_import_mode,
    template_node_attributes,
    template_tensor_alignment,
    template_tensor_values,
    template_top_level_fields,
    per_case_tensor_fields,
    tensors_by_uid,
    tier_for_suite,
    uid_map_onto,
)


# --------------------------------------------------------------------------
# Scan existing bundles
# --------------------------------------------------------------------------


class ImportCache:
    """Parsed sweeps and expanded cases, shared by a series of :func:`main` calls.

    Use one cache for imports into one bundle tree that no other writer changes
    while the cache is in use. An append changes the sweep in memory and marks it
    dirty. :meth:`flush` writes each dirty sweep.json. The cache reads a sweep
    that is not dirty again when its files change on disk.
    """

    def __init__(self):
        self._entries = {}
        self._dirty = set()

    @staticmethod
    def _stamp(sweep_path: Path, template_path: Path) -> tuple:
        out = ()
        for path in (sweep_path, template_path):
            st = path.stat()
            out += (st.st_ino, st.st_size, st.st_mtime_ns)
        return out

    def load(self, sweep_path: Path, template_path: Path):
        """(template, sweep) of a sweep folder, or None when it cannot read a file."""
        entry = self._entries.get(sweep_path)
        if sweep_path in self._dirty:
            return entry["template"], entry["sweep"]
        try:
            stamp = self._stamp(sweep_path, template_path)
            if entry is not None and entry["stamp"] == stamp:
                return entry["template"], entry["sweep"]
            with open(template_path) as f:
                template = json.load(f)
            with open(sweep_path) as f:
                sweep = json.load(f)
        except (json.JSONDecodeError, OSError):
            self._entries.pop(sweep_path, None)
            return None
        self._entries[sweep_path] = {
            "template": template,
            "sweep": sweep,
            "stamp": stamp,
            "canons": [],
        }
        return template, sweep

    def case_canons(self, sweep_path: Path, template: dict, sweep: dict):
        """Yield canon(expand_case(template, case)) per case.

        Yield None for a case that does not expand.
        """
        entry = self._entries.get(sweep_path)
        known = entry["canons"] if entry is not None and entry["sweep"] is sweep else []
        for i, case in enumerate(sweep.get("cases", [])):
            if i < len(known) and known[i][0] is case:
                yield known[i][1]
                continue
            del known[i:]
            try:
                text = canon(expand_case(template, case))
            except ValueError:
                text = None
            known.append((case, text))
            yield text

    def memo(self, sweep_path: Path, template: dict):
        """A function (key, compute) -> value that keeps compute() per key.

        Use it only for values that depend on nothing but this template and the
        objects that the key names.
        """
        entry = self._entries.get(sweep_path)
        if entry is None or entry["template"] is not template:
            return lambda key, compute: compute()
        values = entry.setdefault("memo", {})

        def lookup(key, compute):
            if key not in values:
                values[key] = compute()
            return values[key]

        return lookup

    def mark_dirty(self, sweep_path: Path):
        self._dirty.add(sweep_path)

    def flush(self):
        """Write each dirty sweep.json."""
        for sweep_path in sorted(self._dirty):
            sweep = self._entries[sweep_path]["sweep"]
            with open(sweep_path, "w", newline="\n") as f:
                f.write(dump_sweep(sweep, sweep_format(sweep)))
            self._dirty.discard(sweep_path)
            self._entries[sweep_path]["stamp"] = self._stamp(
                sweep_path, sweep_path.parent / "graph.template.json"
            )


def _existing_sweeps(bundle_dir: Path, cache: ImportCache = None) -> list:
    """[(template, sweep, sweep_path)] for each readable sweep with a template and a case."""
    sweeps = []
    if not bundle_dir.is_dir():
        return sweeps
    for sweep_path in sorted(bundle_dir.rglob("sweep.json")):
        template_path = sweep_path.parent / "graph.template.json"
        if not template_path.exists():
            continue
        if cache is not None:
            loaded = cache.load(sweep_path, template_path)
            if loaded is None:
                continue
            template, sweep = loaded
        else:
            try:
                with open(template_path) as f:
                    template = json.load(f)
                with open(sweep_path) as f:
                    sweep = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
        if not sweep.get("cases", []):
            continue
        sweeps.append((template, sweep, sweep_path))
    return sweeps


def _index_existing(
    bundle_dir: Path, value_form: bool = False, sweeps=None, cache: ImportCache = None
) -> dict:
    """Build {skeleton_hash: [(template, sweep, sweep_path)]} from the tree.

    A sweep is hashed by the skeleton of its whole template: the graph of its
    first case without absent tensors, built as the C++ loader builds it
    (:func:`expand_case`), or, when every case has absent tensors, the template
    expanded with the first case's values and nothing pruned. A sweep whose case
    does not build is left out.
    """
    index = {}
    if sweeps is None:
        sweeps = _existing_sweeps(bundle_dir)
    for template, sweep, sweep_path in sweeps:
        cases = sweep["cases"]
        rep = next((c for c in cases if not c.get(ABSENT_TENSORS_KEY)), None)

        def rep_hash():
            try:
                if rep is not None:
                    expanded_rep = expand_case(template, rep)
                else:
                    expanded_rep = expand(template, cases[0].get("values", {}))
            except ValueError:
                return None
            return skeleton_hash(expanded_rep, value_form=value_form)

        if cache is not None:
            obj = rep if rep is not None else cases[0]
            key = ("index", value_form, rep is None, id(obj))
            held, h = cache.memo(sweep_path, template)(key, lambda: (obj, rep_hash()))
            if held is not obj:
                h = rep_hash()
        else:
            h = rep_hash()
        if h is None:
            continue
        index.setdefault(h, []).append((template, sweep, sweep_path))
    return index


def _superset_join(template: dict, graph: dict, value_form: bool, memo=None):
    """(uid map onto the template, absent template uids) when the graph is the template
    without some of its tensors; else None.

    Tensors match by name. The absent uids must satisfy the C++ loader's template
    rules (:func:`absent_tensor_problems`), and the template without them must have
    the graph's skeleton. ``memo``, from :meth:`ImportCache.memo`, keeps the
    template-side result per set of absent uids.
    """
    uid_map = uid_map_onto(graph, template)
    if uid_map is None:
        return None
    present = set(uid_map.values())
    absent = sorted(
        t["uid"]
        for t in template.get("tensors", [])
        if "uid" in t and t["uid"] not in present
    )
    if not absent:
        return None

    def pruned_hash():
        if absent_tensor_problems(template, absent):
            return None
        return skeleton_hash(
            prune_absent_tensors(template, absent), value_form=value_form
        )

    if memo is not None:
        want = memo(("superset", value_form, tuple(absent)), pruned_hash)
    else:
        want = pruned_hash()
    if want is None or want != skeleton_hash(
        remap_graph(graph, uid_map), value_form=value_form
    ):
        return None
    return uid_map, absent


def _tensor_names(template: dict) -> list:
    return sorted(t.get("name", "") for t in template.get("tensors", []) if "uid" in t)


# --------------------------------------------------------------------------
# Build single-case values from a graph (inverse of template expansion)
# --------------------------------------------------------------------------


def _extract_placeholders(node, prefix=""):
    """Walk a JSON tree and yield (dotted_path, placeholder) for every ${case.*}."""
    if isinstance(node, str) and node.startswith("${case.") and node.endswith("}"):
        yield (prefix, node[len("${case.") : -1])
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _extract_placeholders(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _extract_placeholders(v, f"{prefix}[{i}]")


_MISSING = object()


def _deep_get(obj, dotted_path, default=_MISSING):
    """Resolve a dotted path like 'attributes.padding' against a dict tree."""
    cur = obj
    for tok in dotted_path.split("."):
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            return default
    return cur


def _deep_set(obj, dotted_path, value):
    """Set a value at a dotted path, creating intermediate dicts as needed."""
    parts = dotted_path.split(".")
    cur = obj
    for tok in parts[:-1]:
        cur = cur.setdefault(tok, {})
    cur[parts[-1]] = value


def _first_mismatch(expected, actual, path=""):
    """First differing location between two JSON trees.

    Walks the sorted union of dict keys, then list indices, then scalars.
    Returns (path, expected_value, actual_value), or None when nothing compares
    unequal; a side that lacks the key or index is ``_MISSING``. Used only to
    describe a failed canonical comparison, never to decide it.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            found = _first_mismatch(
                expected.get(key, _MISSING),
                actual.get(key, _MISSING),
                f"{path}.{key}" if path else key,
            )
            if found:
                return found
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        for i in range(max(len(expected), len(actual))):
            found = _first_mismatch(
                expected[i] if i < len(expected) else _MISSING,
                actual[i] if i < len(actual) else _MISSING,
                f"{path}[{i}]",
            )
            if found:
                return found
        return None
    if expected is _MISSING or actual is _MISSING or expected != actual:
        return (path or "<root>", expected, actual)
    return None


def _brief(value, limit=200):
    """Render a JSON value (or ``<absent>`` for ``_MISSING``) in ``limit`` chars."""
    text = "<absent>" if value is _MISSING else json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _report_roundtrip_failure(where, expanded, graph, selected=None):
    """Name the first location a failed round-trip lost, with expected vs actual.

    `selected`, when given, describes the sweep the graph was matched to and is
    printed directly after the ERROR line.
    """
    print(f"  ERROR: round-trip verify failed {where}", file=sys.stderr)
    if selected is not None:
        print(f"  selected sweep: {selected}", file=sys.stderr)
    mismatch = _first_mismatch(graph, expanded)
    if mismatch is None:
        want, got = canon(graph), canon(expanded)
        at = next(
            (i for i, (a, b) in enumerate(zip(want, got)) if a != b),
            min(len(want), len(got)),
        )
        print(
            "    mismatch:  no field compares unequal, so this is a JSON rendering"
            " difference, such as int 1 against float 1.0",
            file=sys.stderr,
        )
        print(f"    at offset: {at}", file=sys.stderr)
        print(f"    expected:  {want[at : at + 60]}  (input graph)", file=sys.stderr)
        print(
            f"    actual:    {got[at : at + 60]}  (template + values)", file=sys.stderr
        )
        return
    field, expected, actual = mismatch
    print(f"    field:     {field}", file=sys.stderr)
    if field.startswith("tensors["):
        index = int(field[len("tensors[") : field.index("]")])
        tensors = graph.get("tensors", [])
        tensor = tensors[index] if index < len(tensors) else {}
        name = tensor.get("name", _MISSING) if isinstance(tensor, dict) else _MISSING
        print(
            f"    tensor:    name={_brief(name)}"
            "  (tensors listed in canonical-uid order)",
            file=sys.stderr,
        )
    print(f"    expected:  {_brief(expected)}  (input graph)", file=sys.stderr)
    print(f"    actual:    {_brief(actual)}  (template + values)", file=sys.stderr)


def _remove_patches(graph: dict, template: dict) -> list:
    """``tensor_patches`` entries that remove the fields a graph's tensor lacks.

    One entry per template tensor that has a field the graph's tensor of the same
    uid lacks. A field that holds a ``${case.*}`` placeholder is never removed:
    the C++ loader resolves placeholders before it applies patches.
    """
    tgraph = tensors_by_uid(graph)
    patches = []
    for tt in template.get("tensors", []):
        src = tgraph.get(tt.get("uid"))
        if src is None:
            continue
        missing = sorted(
            fld
            for fld, tval in tt.items()
            if fld not in src and next(_extract_placeholders(tval), None) is None
        )
        if missing:
            patches.append({"uid": tt["uid"], "remove": missing})
    return patches


def _sweep_case(case_id, values, patches, meta, absent=None) -> dict:
    """A sweep case: id, absent tensors (if any), values, tensor patches (if any), metadata."""
    case = {"id": case_id}
    if absent:
        case[ABSENT_TENSORS_KEY] = absent
    case["values"] = values
    if patches:
        case[TENSOR_PATCHES_KEY] = patches
    case["metadata"] = meta
    return case


def _extract_values(graph: dict, template: dict) -> dict:
    """Extract per-case values from a concrete graph given its template."""
    values = {}
    for fld in TOP_LEVEL_IF_VARIES + GROUPED_TOP_LEVEL:
        if isinstance(template.get(fld), str) and template[fld].startswith("${case."):
            values[fld] = graph.get(fld)

    tgraph = tensors_by_uid(graph)
    tv = []
    for tt in template.get("tensors", []):
        uid = tt.get("uid")
        src = tgraph.get(uid, {})
        entry = {"uid": uid}
        for fld in TENSOR_ALWAYS + TENSOR_IF_VARIES + TENSOR_RAGGED:
            if isinstance(tt.get(fld), str) and tt[fld].startswith("${case."):
                if fld in src:
                    entry[fld] = src[fld]
        tv.append(entry)
    values["tensors"] = tv

    # Whole-field tensor placeholders that resolve from the top-level values,
    # such as ${case.alignment.<uid>}.
    for tt in template.get("tensors", []):
        src = tgraph.get(tt.get("uid"), {})
        for fld, tval in tt.items():
            if (
                fld in TENSOR_ALWAYS + TENSOR_IF_VARIES + TENSOR_RAGGED
                or fld not in src
            ):
                continue
            for loc, case_path in _extract_placeholders(tval):
                if loc == "":
                    _deep_set(values, case_path, src[fld])

    for t_node, g_node in zip(template.get("nodes", []), graph.get("nodes", [])):
        for loc, case_path in _extract_placeholders(t_node):
            if case_path.startswith("tensors[") or case_path in TOP_LEVEL_IF_VARIES:
                continue
            src_val = _deep_get(g_node, loc)
            if src_val is not _MISSING:
                _deep_set(values, case_path, src_val)

    return values


# --------------------------------------------------------------------------
# Capture sidecar
# --------------------------------------------------------------------------


def _read_sidecar(graph_path: Path):
    """:func:`read_capture_sidecar`, plus the check that its suite is usable.

    import_graph is handed one graph with no capture root, so the sidecar's
    ``operation`` is the only record of the suite, and with it the tier. An
    ``operation`` that is present but not a non-empty string makes the sidecar
    unusable. An absent one is a hand-written sidecar with no suite.
    """
    path, meta, err = read_capture_sidecar(graph_path)
    if meta is not None and "operation" in meta:
        suite = meta["operation"]
        if not isinstance(suite, str) or not suite:
            return path, None, f"operation: expected a non-empty string, got {suite!r}"
    return path, meta, err


# --meta keys whose values are not strings in BundleMetadata. Only these parse
# as JSON; every other key is a string field and is stored as given.
_TYPED_META_KEYS = ("seed", "format_version", "minimum_vram_mb", "inputs")


def _reject_constant(name):
    raise ValueError(f"{name} is not allowed")


def _finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is out of range")
    return value


def _parse_meta_value(k: str, v: str):
    """A ``--meta`` value: JSON for the typed keys, else the string as given.

    A typed key parses so ``--meta seed=42`` stores the number 42, not ``"42"``
    (a false conflict with a captured 42, and a seed that never dedups). Any
    other key stays a string, so ``notes=123`` is ``"123"``, which the bundle
    loader reads. NaN and Infinity raise ValueError: json.dump would write them
    bare, and nlohmann would then reject the whole sweep.json.
    """
    if k not in _TYPED_META_KEYS:
        return v
    return json.loads(v, parse_constant=_reject_constant, parse_float=_finite_float)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

_SWEEP_SUPPORT_FILE = "support.json"


def _canonical_support_json() -> str:
    """Canonical bytes for an empty support-claim sidecar (LF line endings)."""
    return json.dumps({"claims": {}, "version": 1}, indent=2, sort_keys=True) + "\n"


def _ensure_support_json(sweep_dir: Path, dry_run: bool) -> bool:
    """Create an empty canonical support.json next to a sweep if one is missing.

    Never touches an existing support.json. Returns whether one was (or,
    under --dry-run, would be) created.
    """
    support_path = sweep_dir / _SWEEP_SUPPORT_FILE
    if support_path.exists():
        return False
    if not dry_run:
        with open(support_path, "w", newline="\n") as f:
            f.write(_canonical_support_json())
    return True


def _sanitize_gtest(s: str) -> str:
    """Mirror BundleDiscovery.hpp's sanitizeForGtest: non [A-Za-z0-9_] -> '_'."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def _gtest_name(sweep_dir: Path, bundle_dir: Path, case_id: str) -> str:
    """The full GTest name the harness registers for a case in this sweep dir."""
    suite = "_".join(
        _sanitize_gtest(part) for part in sweep_dir.relative_to(bundle_dir).parts
    )
    return f"{suite}.{_sanitize_gtest(case_id)}"


# Opt-in flags that change how a new sweep is templated or matched. Each one
# that is in effect is recorded in the sweep's import_mode object.
_GROUPING_FLAGS = (
    "template_attributes",
    "template_alignment",
    "skeleton_value_form",
    "superset_templates",
)


def _is_folder_name(name: str) -> bool:
    """True if ``name`` is one plain directory name (no separator, not . or ..)."""
    return bool(name) and name not in (".", "..") and Path(name).name == name


def _grouping_options(args):
    """The grouping options in effect, as their command-line spellings."""
    return [f.replace("_", "-") for f in _GROUPING_FLAGS if getattr(args, f)]


def _is_grouped(args) -> bool:
    """True if any opt-in grouped-import flag is given."""
    return bool(
        _grouping_options(args)
        or args.op_dir is not None
        or args.topology is not None
        or args.only_op_dir
        or args.require_marked_sweep
        or args.case_id is not None
    )


def _expanded_canons(template, sweep, sweep_path, cache):
    """Yield canon(expand_case(template, case)) per case; None if it does not expand."""
    if cache is not None:
        yield from cache.case_canons(sweep_path, template, sweep)
        return
    for case in sweep.get("cases", []):
        try:
            yield canon(expand_case(template, case))
        except ValueError:
            yield None


def _report_duplicate(args, candidates, graph_json: str, meta: dict, cache=None):
    """The exit code when a case of ``candidates`` duplicates the graph; else None.

    ``candidates`` holds (template, sweep, sweep_path, ...) entries whose uids
    match those of ``graph_json``.
    """
    for template, sweep, sweep_path, *_ in candidates:
        texts = _expanded_canons(template, sweep, sweep_path, cache)
        for case, text in zip(sweep.get("cases", []), texts):
            if text != graph_json:
                continue
            existing_meta = case.get("metadata", {})
            # An unknown seed is not a wildcard: the case would run with the
            # default seed, which differs from an existing case's explicit
            # one. Two absent seeds do match (both fall back to the default).
            seed_match = existing_meta.get("seed") == meta.get("seed")
            inputs_match = canon(meta.get("inputs", {})) == canon(
                existing_meta.get("inputs", {})
            )
            if seed_match and inputs_match:
                print(f"  DUPLICATE of {sweep_path}:{case['id']}", file=sys.stderr)
                if args.strict:
                    return 1
                if not args.force:
                    print("  skipped (use --force to append)", file=sys.stderr)
                    return 0
    return None


def _append_case(
    args, template, sweep, sweep_path, new_case, graph, selected, cache=None
) -> int:
    """Give ``new_case`` an id, check that it re-expands to ``graph``, and write the sweep.

    With ``cache``, the sweep changes in memory and the cache marks it dirty.
    """
    existing_cases = sweep.get("cases", [])
    saved_ids = [c["id"] for c in existing_cases]
    taken = set(saved_ids)
    if args.case_id is not None:
        if args.case_id in taken:
            print(
                f"  ERROR: --case-id {args.case_id} already exists in {sweep_path}",
                file=sys.stderr,
            )
            return 1
        new_case["id"] = args.case_id
    else:
        all_cases = existing_cases + [new_case]
        assign_case_ids(all_cases)
        for c, orig_id in zip(existing_cases, saved_ids):
            c["id"] = orig_id
        if new_case["id"] in taken:
            new_case["id"] = f"{new_case['id']}_{_case_hash(new_case)}"

    expanded = expand_case(template, new_case)
    if canon(expanded) != canon(graph):
        _report_roundtrip_failure(
            "after extraction", expanded, graph, selected=selected
        )
        return 1
    if new_case.get(ABSENT_TENSORS_KEY):
        problems = absent_case_problems(
            {**sweep, "version": ABSENT_TENSORS_SWEEP_VERSION}, template, new_case
        )
        if problems:
            print(
                f"  ERROR: the C++ loader would refuse the case: {problems[0]}",
                file=sys.stderr,
            )
            return 1

    fmt = sweep_format(sweep)
    if fmt not in SWEEP_FORMATS:
        print(
            f"  ERROR: {sweep_path}: unknown sweep.json format {fmt!r}",
            file=sys.stderr,
        )
        return 1
    support_created = _ensure_support_json(sweep_path.parent, args.dry_run)
    if not args.dry_run:
        # Only the cases and, for a case with absent tensors, the sweep version change.
        sweep["cases"] = existing_cases + [new_case]
        if new_case.get(ABSENT_TENSORS_KEY):
            sweep["version"] = ABSENT_TENSORS_SWEEP_VERSION
        if cache is not None:
            cache.mark_dirty(sweep_path)
        else:
            with open(sweep_path, "w", newline="\n") as f:
                f.write(dump_sweep(sweep, fmt))
    print(f"  appended case '{new_case['id']}' to {sweep_path}", file=sys.stderr)
    print(
        "  gtest name: "
        f"{_gtest_name(sweep_path.parent, args.bundle_dir, new_case['id'])}",
        file=sys.stderr,
    )
    if support_created:
        print(
            f"  created empty support.json in {sweep_path.parent}",
            file=sys.stderr,
        )
    return 0


def _named_tensors(named: dict) -> set:
    """The tensor names in a :func:`metadata_uid_names` result."""
    out = set()
    for entries in named.values():
        for entry in entries:
            out.update(entry.values() if isinstance(entry, dict) else [entry])
    return out


def _append_superset_case(args, join, n_joins, graph, meta, cache=None) -> int:
    """Append ``graph`` to the sweep of ``join`` as a case with absent tensors.

    The graph and the metadata ``inputs`` take the template's uids, matched by
    tensor name. The metadata gets format_version 2.
    """
    template, sweep, sweep_path, uid_map, absent = join
    graph_t = remap_graph(graph, uid_map)
    meta_t = remap_meta_inputs(meta, uid_map)
    named = metadata_uid_names(meta, graph)
    if metadata_uid_names(meta_t, graph_t) != named:
        print(
            "  ERROR: the uid remap onto the template moves a metadata uid to"
            f" another tensor: {named} became {metadata_uid_names(meta_t, graph_t)}",
            file=sys.stderr,
        )
        return 1
    absent_names = {
        t.get("name", "") for t in template.get("tensors", []) if t.get("uid") in absent
    }
    named_absent = sorted(
        _named_tensors(metadata_uid_names(meta_t, template)) & absent_names
    )
    if named_absent:
        print(
            f"  ERROR: the metadata names absent tensors {named_absent}",
            file=sys.stderr,
        )
        return 1
    fv = meta_t.get("format_version")
    if (
        isinstance(fv, bool)
        or not isinstance(fv, int)
        or fv < ABSENT_TENSORS_FORMAT_VERSION
    ):
        meta_t["format_version"] = ABSENT_TENSORS_FORMAT_VERSION

    dup = _report_duplicate(args, [join], canon(graph_t), meta_t, cache)
    if dup is not None:
        return dup

    pruned = prune_absent_tensors(template, absent)
    values = _extract_values(graph_t, pruned)
    new_case = _sweep_case(
        None, values, _remove_patches(graph_t, pruned), meta_t, absent=absent
    )
    print(
        f"  absent tensors: {absent} "
        f"({', '.join(sorted(absent_names))}) of the template in {sweep_path.parent}",
        file=sys.stderr,
    )
    return _append_case(
        args,
        template,
        sweep,
        sweep_path,
        new_case,
        graph_t,
        f"{sweep_path} (of {n_joins} superset template(s))",
        cache,
    )


def _missed_merges(sweeps, graph: dict, value_form: bool) -> list:
    """Paths of the sweeps whose template is ``graph`` without some of its tensors."""
    return [
        sweep_path
        for template, _, sweep_path in sweeps
        if _superset_join(graph, template, value_form) is not None
    ]


def main(argv=None, cache: ImportCache = None) -> int:
    """Run one import with the command-line arguments ``argv`` (default: sys.argv).

    With ``cache``, appends stay in memory until ``cache.flush()``.
    """
    ap = argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--graph", type=Path, required=True, help="path to the graph JSON to import"
    )
    ap.add_argument(
        "--bundle-dir", type=Path, required=True, help="root of the bundle tree"
    )
    ap.add_argument(
        "--tier",
        default=None,
        help="tier folder (default: from the sidecar's suite prefix, else quick)."
        " Contradicting the sidecar warns; errors under --strict",
    )
    ap.add_argument(
        "--meta",
        action="append",
        default=[],
        help="key=value metadata pairs (repeatable). For graphs without a sidecar,"
        " or a deliberate override of it (warns; errors under --strict)."
        " seed, format_version, minimum_vram_mb and inputs parse as JSON"
        " (seed=42 is a number); other keys are stored as strings",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="global seed for metadata. For graphs without a sidecar, or a"
        " deliberate override of its seed (warns; errors under --strict)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument(
        "--force", action="store_true", help="append even if exact dup exists"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on exact dup or sidecar override (CI mode)",
    )
    grouped = ap.add_argument_group(
        "grouped import (opt-in)",
        "Any of these flags writes an import_mode object into a new sweep.json.",
    )
    grouped.add_argument(
        "--op-dir",
        default=None,
        metavar="NAME",
        help="operation folder for a new sweep (default: from the node types)",
    )
    grouped.add_argument(
        "--topology",
        default=None,
        metavar="NAME",
        help="sweep folder for a new sweep. An append must go to the sweep in"
        " this folder, else the import fails. A folder that holds a different"
        " sweep is never written",
    )
    grouped.add_argument(
        "--template-attributes",
        action="store_true",
        help="a new template replaces every non-object node attribute value"
        " with ${case.attributes.<key>}",
    )
    grouped.add_argument(
        "--template-alignment",
        action="store_true",
        help="a new template replaces each tensor alignment with"
        " ${case.alignment.<uid>}",
    )
    grouped.add_argument(
        "--skeleton-value-form",
        action="store_true",
        help="the skeleton hash records each tensor's value form (baked value,"
        " value_type, is_runtime_pass_by_value), and a new template replaces"
        " each baked value with ${case.value}",
    )
    grouped.add_argument(
        "--superset-templates",
        action="store_true",
        help="a graph that is the template of a sweep without some of its"
        " tensors joins that sweep: the case lists the missing template uids in"
        " absent_tensors, its metadata gets format_version 2 and the sweep gets"
        " version 2. The smallest such template wins; with --topology, only the"
        " sweep in that folder. Without --topology, a new template that holds the"
        " template of another sweep leaves that sweep unchanged, with a note",
    )
    grouped.add_argument(
        "--only-op-dir",
        action="store_true",
        help="match only sweeps under <bundle-dir>/<tier>/<op-dir>/",
    )
    grouped.add_argument(
        "--require-marked-sweep",
        action="store_true",
        help="match only sweeps whose import_mode equals the import_mode of this"
        " run. The format key of import_mode is not compared",
    )
    grouped.add_argument(
        "--case-id",
        default=None,
        metavar="ID",
        help="case id to use (lowercase_snake_case). The import fails if the"
        " sweep already has this id",
    )
    grouped.add_argument(
        "--sweep-format",
        choices=SWEEP_FORMATS,
        default=None,
        help="file layout of a new sweep.json (default: compact). Needs another"
        " grouped-import flag. An append keeps the layout of the sweep",
    )
    args = ap.parse_args(argv)

    for flag, name in (("--op-dir", args.op_dir), ("--topology", args.topology)):
        if name is not None and not _is_folder_name(name):
            print(f"import_graph: {flag}: not a folder name: {name!r}", file=sys.stderr)
            return 1
    if args.case_id is not None and not re.fullmatch(r"[a-z0-9_]+", args.case_id):
        print(
            f"import_graph: --case-id: not lowercase_snake_case: {args.case_id!r}",
            file=sys.stderr,
        )
        return 1
    grouped_mode = _is_grouped(args)
    if args.sweep_format is not None and not grouped_mode:
        print(
            "import_graph: --sweep-format needs a grouped-import flag",
            file=sys.stderr,
        )
        return 1
    marker = (
        import_mode_marker(_grouping_options(args), args.sweep_format or "compact")
        if grouped_mode
        else None
    )

    try:
        with open(args.graph) as f:
            graph = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"import_graph: cannot read {args.graph}: {e}", file=sys.stderr)
        return 1

    # The capture sidecar is the metadata base, built the way place_bundles
    # builds it, so both hops record the same seed and inputs for one capture.
    sidecar_path, sidecar, sidecar_err = _read_sidecar(args.graph)
    if sidecar_err is not None:
        # Refused, not skipped: without the sidecar there is no suite, so the
        # case would land in quick/ with no reference_source.
        print(f"  ERROR: bad sidecar {sidecar_path}: {sidecar_err}", file=sys.stderr)
        return 1
    # What the capture recorded, including the reference_source derived from
    # it: overrides are checked against this, not just the raw sidecar keys.
    captured = dict(sidecar) if sidecar is not None else {}
    suite = captured.get("operation")
    found = capture_suite_case(args.graph, suite=suite)
    if found is not None:
        captured["reference_source"] = capture_reference_source(*found)
    meta = dict(captured)
    meta.setdefault("format_version", 1)
    captured_tier = tier_for_suite(suite) if suite is not None else None
    if args.tier is None:
        args.tier = captured_tier or "quick"

    # Explicit CLI values win. Overriding a value the capture recorded is
    # allowed but never silent: the sidecar is what the C++ test actually ran.
    overrides = []
    for kv in args.meta:
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                v = _parse_meta_value(k, v)
            except ValueError as e:
                print(f"import_graph: --meta {k}: {e}", file=sys.stderr)
                return 1
            if k == "inputs":
                err = inputs_error(v)
                if err is not None:
                    print(f"import_graph: --meta {err}", file=sys.stderr)
                    return 1
            overrides.append((f"--meta {k}", k, v))
            meta[k] = v
    if args.seed is not None:
        overrides.append(("--seed", "seed", args.seed))
        meta["seed"] = args.seed
    meta.setdefault("generator", "import_graph.py")
    if not str(meta.get("reference_source", "")).strip():
        print(
            "import_graph: warning: no reference_source in metadata;"
            " pass --meta reference_source=... to record where this graph came from",
            file=sys.stderr,
        )

    conflicts = [
        (flag, captured[k], v)
        for flag, k, v in overrides
        if k in captured and canon(captured[k]) != canon(v)
    ]
    # The tier is not metadata, but it picks the CI lane: --tier quick on a
    # Full/ capture would silently move the case to the smoke lane.
    if captured_tier is not None and args.tier != captured_tier:
        conflicts.append(("--tier", captured_tier, args.tier))
    for flag, was, supplied in conflicts:
        level = "ERROR" if args.strict else "WARN"
        print(
            f"  {level}: {flag} {canon(supplied)} overrides captured"
            f" {canon(was)} from {sidecar_path}",
            file=sys.stderr,
        )
    if conflicts and args.strict:
        return 1

    # Canonicalize UIDs by tensor name so an imported graph lines up with sweeps
    # built by place_bundles (which does the same). The C++ builder auto-assigns
    # UIDs non-deterministically, so the graph and its inputs fill-spec map must
    # be remapped in lockstep before hashing/matching (canonical_uid_map_by_name).
    uid_map = canonical_uid_map_by_name(graph)
    named_meta = metadata_uid_names(meta, graph)
    try:
        graph = remap_graph(graph, uid_map)
    except ValueError as e:
        print(
            f"  ERROR: the graph refers to a uid it does not declare: {e}",
            file=sys.stderr,
        )
        return 1
    meta = remap_meta_inputs(meta, uid_map)
    if metadata_uid_names(meta, graph) != named_meta:
        print(
            "  ERROR: the uid remap moves a metadata uid to another tensor:"
            f" {named_meta} became {metadata_uid_names(meta, graph)}",
            file=sys.stderr,
        )
        return 1

    h = skeleton_hash(graph, value_form=args.skeleton_value_form)
    op = args.op_dir if args.op_dir is not None else derive_operation(graph)
    graph_json = canon(graph)

    print(f"  skeleton hash: {h}", file=sys.stderr)
    print(f"  operation:     {op}", file=sys.stderr)

    sweeps = _existing_sweeps(args.bundle_dir, cache)
    if args.only_op_dir:
        op_prefix = str(args.bundle_dir / args.tier / op) + os.sep
        sweeps = [m for m in sweeps if str(m[2]).startswith(op_prefix)]
    if args.require_marked_sweep:
        grouping = import_mode_grouping(marker)
        sweeps = [
            m
            for m in sweeps
            if import_mode_grouping(sweep_import_mode(m[1])) == grouping
        ]
    index = _index_existing(
        args.bundle_dir, value_form=args.skeleton_value_form, sweeps=sweeps, cache=cache
    )
    matches = index.get(h, [])

    # --- Check for exact duplicates ---
    dup = _report_duplicate(args, matches, graph_json, meta, cache)
    if dup is not None:
        return dup

    # Default mode never appends into a sweep that grouped mode wrote. It
    # still reports an exact duplicate of a case there (checked above).
    if not grouped_mode:
        matches = [m for m in matches if sweep_import_mode(m[1]) is None]

    topology_dir = (
        args.bundle_dir / args.tier / op / args.topology
        if args.topology is not None
        else None
    )
    matched_elsewhere = []
    if topology_dir is not None and matches:
        in_topology = [m for m in matches if m[2].parent == topology_dir]
        if not in_topology:
            matched_elsewhere = matches
        matches = in_topology

    # --- Structural match: append to existing sweep ---
    if matches:
        # Prefer a sweep in the requested tier so Smoke cases land in quick/.
        tier_prefix = str(args.bundle_dir / args.tier) + os.sep
        tier_matches = [m for m in matches if str(m[2]).startswith(tier_prefix)]
        template, sweep, sweep_path = (tier_matches or matches)[0]
        values = _extract_values(graph, template)
        new_case = _sweep_case(None, values, _remove_patches(graph, template), meta)
        return _append_case(
            args,
            template,
            sweep,
            sweep_path,
            new_case,
            graph,
            f"{sweep_path}"
            f" (of {len(matches)} structural match(es),"
            f" {len(tier_matches)} in tier '{args.tier}')",
            cache,
        )

    # --- Superset match: append with absent tensors ---
    if args.superset_templates:
        joins = []
        for template, sweep, sweep_path in sweeps:
            memo = cache.memo(sweep_path, template) if cache is not None else None
            found = _superset_join(template, graph, args.skeleton_value_form, memo)
            if found is not None:
                joins.append((template, sweep, sweep_path) + found)
        if topology_dir is not None:
            others = [j for j in joins if j[2].parent != topology_dir]
            joins = [j for j in joins if j[2].parent == topology_dir]
            if others and not joins:
                print(
                    f"  NOTE: the graph also fits the templates of"
                    f" {', '.join(str(j[2].parent) for j in others)};"
                    f" --topology {args.topology} keeps it out of them",
                    file=sys.stderr,
                )
        if joins:
            tier_prefix = str(args.bundle_dir / args.tier) + os.sep
            joins.sort(
                key=lambda j: (
                    not str(j[2]).startswith(tier_prefix),
                    len(_tensor_names(j[0])),
                    _tensor_names(j[0]),
                    str(j[2]),
                )
            )
            return _append_superset_case(args, joins[0], len(joins), graph, meta, cache)

    if matched_elsewhere:
        print(
            f"  ERROR: --topology {args.topology}: the graph matches"
            f" {', '.join(str(m[2].parent) for m in matched_elsewhere)},"
            f" not {topology_dir}",
            file=sys.stderr,
        )
        return 1

    # --- No structural match: create new template+sweep ---
    template = copy.deepcopy(graph)
    for fld in TOP_LEVEL_IF_VARIES:
        if fld in template:
            template[fld] = f"${{case.{fld}}}"
    for t in template.get("tensors", []):
        for fld in per_case_tensor_fields(t):
            if fld in t:
                t[fld] = f"${{case.{fld}}}"
    if args.template_attributes:
        template_node_attributes(template)
    if args.template_alignment:
        template_tensor_alignment(template)
    if args.skeleton_value_form:
        template_tensor_values(template)
    if grouped_mode:
        template_top_level_fields(template)

    values = _extract_values(graph, template)

    expanded = expand(template, values)
    if canon(expanded) != canon(graph):
        _report_roundtrip_failure("for new template", expanded, graph)
        return 1

    if args.superset_templates and topology_dir is None:
        for sweep_path in _missed_merges(sweeps, graph, args.skeleton_value_form):
            print(
                f"  NOTE: missed merge: the new template holds the template of"
                f" {sweep_path}, which stays unchanged",
                file=sys.stderr,
            )

    tmap = tensors_by_uid(graph)
    first = tmap[min(tmap)] if tmap else {}
    layout = infer_layout(first.get("dims"), first.get("strides")) or "Default"
    preferred_topo_name = sanitize(layout).replace(" ", "_").title() or "Default"

    op_dir = args.bundle_dir / args.tier / op
    if topology_dir is not None:
        if next_free_topology_dir(op_dir, args.topology) != args.topology:
            print(
                f"  ERROR: --topology {args.topology}: {topology_dir} holds"
                " another sweep",
                file=sys.stderr,
            )
            return 1
        topo_name = args.topology
    else:
        topo_name = next_free_topology_dir(op_dir, preferred_topo_name)
    out_dir = op_dir / topo_name

    case_entry = {"id": None, "values": values, "metadata": meta}
    if args.case_id is not None:
        case_entry["id"] = args.case_id
    else:
        assign_case_ids([case_entry])
    sweep_out = {"version": 1, "cases": [case_entry]}
    if marker is not None:
        sweep_out = {"version": 1, IMPORT_MODE_KEY: marker, "cases": [case_entry]}

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    support_created = _ensure_support_json(out_dir, args.dry_run)
    if not args.dry_run:
        with open(out_dir / "graph.template.json", "w", newline="\n") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
        with open(out_dir / "sweep.json", "w", newline="\n") as f:
            f.write(dump_sweep(sweep_out, sweep_format(sweep_out)))
    print(
        f"  created new bundle: {out_dir} (case '{case_entry['id']}')",
        file=sys.stderr,
    )
    print(
        f"  gtest name: {_gtest_name(out_dir, args.bundle_dir, case_entry['id'])}",
        file=sys.stderr,
    )
    if support_created:
        print(f"  created empty support.json in {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
