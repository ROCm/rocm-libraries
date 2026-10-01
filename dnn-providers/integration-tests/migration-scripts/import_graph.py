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
contradicts the sidecar or the ``reference_source`` derived from it. ``--meta``
values parse as JSON when they can, so ``seed=42`` is the number 42. An
unreadable or malformed sidecar (e.g. an ``inputs`` key that is not a UID) warns
and the import proceeds without it. ``--strict`` turns both warnings into a
non-zero exit.

Tier: unless ``--tier`` is given, the tier comes from the sidecar's suite
prefix via ``TIER_MAP`` (``Full/...`` -> ``full/``), as in ``place_bundles.py``;
with no sidecar it defaults to ``quick``. A ``--tier`` that contradicts the
sidecar warns like any other override.

Usage::

    import_graph.py --graph case.json --bundle-dir integration-test-bundles/ \\
        [--tier TIER] [--meta reference_source="..."] [--dry-run] [--strict] [--force]
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from bundle_utils import (
    TENSOR_ALWAYS,
    TENSOR_IF_VARIES,
    TIER_MAP,
    TOP_LEVEL_IF_VARIES,
    _case_hash,
    assign_case_ids,
    canon,
    canonical_uid_map_by_name,
    derive_operation,
    expand,
    infer_layout,
    remap_graph,
    remap_meta_inputs,
    sanitize,
    skeleton_hash,
    tensors_by_uid,
)


# --------------------------------------------------------------------------
# Scan existing bundles
# --------------------------------------------------------------------------


def _index_existing(bundle_dir: Path) -> dict:
    """Build {skeleton_hash: [(template, sweep, sweep_path)]} from the tree."""
    index = {}
    if not bundle_dir.is_dir():
        return index
    for sweep_path in sorted(bundle_dir.rglob("sweep.json")):
        template_path = sweep_path.parent / "graph.template.json"
        if not template_path.exists():
            continue
        try:
            with open(template_path) as f:
                template = json.load(f)
            with open(sweep_path) as f:
                sweep = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cases = sweep.get("cases", [])
        if not cases:
            continue
        expanded_rep = expand(template, cases[0].get("values", {}))
        h = skeleton_hash(expanded_rep)
        index.setdefault(h, []).append((template, sweep, sweep_path))
    return index


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


def _extract_values(graph: dict, template: dict) -> dict:
    """Extract per-case values from a concrete graph given its template."""
    values = {}
    for fld in TOP_LEVEL_IF_VARIES:
        if isinstance(template.get(fld), str) and template[fld].startswith("${case."):
            values[fld] = graph.get(fld)

    tgraph = tensors_by_uid(graph)
    tv = []
    for tt in template.get("tensors", []):
        uid = tt.get("uid")
        src = tgraph.get(uid, {})
        entry = {"uid": uid}
        for fld in TENSOR_ALWAYS + TENSOR_IF_VARIES:
            if isinstance(tt.get(fld), str) and tt[fld].startswith("${case."):
                if fld in src:
                    entry[fld] = src[fld]
        tv.append(entry)
    values["tensors"] = tv

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
    """Load the ``<stem>.meta.json`` that ``--capture-bundles`` writes beside a graph.

    Returns ``(path, meta, error)``. ``meta`` is None when the sidecar is absent
    (a hand-authored graph) or unusable, and ``error`` is set only in the
    unusable case, saying why.
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
        err = _inputs_error(meta["inputs"])
        if err is not None:
            return path, None, err
    return path, meta, None


def _inputs_error(inputs):
    """Why ``inputs`` can't be remapped (a UID-keyed object), or None if it can."""
    if not isinstance(inputs, dict):
        return f"inputs: expected a JSON object, got {type(inputs).__name__}"
    for k in inputs:
        try:
            int(k)
        except ValueError:
            return f"inputs: key {k!r} is not a tensor UID"
    return None


def _parse_meta_value(v: str):
    """A ``--meta`` value as JSON when it parses (``42``, ``true``), else the string.

    Without this, ``--meta seed=42`` would store ``"42"``: a false conflict with
    a captured ``42`` and a seed that never dedups against an existing case.
    """
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _capture_reference_source(graph_path: Path, sidecar: dict):
    """The ``reference_source`` place_bundles records for this capture, or None.

    Capture writes ``<dir>/<suite>/<case>/<case>.json`` and stores ``<suite>`` as
    the sidecar's ``operation``; place_bundles names the case from both. When
    the graph is not at that path, there is nothing to derive it from.
    """
    suite = sidecar.get("operation")
    graph_path = graph_path.resolve()
    case_dir = graph_path.parent
    if not isinstance(suite, str) or not suite or graph_path.stem != case_dir.name:
        return None
    suite_parts = Path(suite).parts
    if case_dir.parent.parts[-len(suite_parts) :] != suite_parts:
        return None
    return f"c++ integration suite: {suite}.{case_dir.name}"


def _capture_tier(sidecar):
    """The tier place_bundles would pick: TIER_MAP of the suite's prefix.

    None when there is no sidecar suite to derive it from.
    """
    suite = sidecar.get("operation") if sidecar is not None else None
    if not isinstance(suite, str) or not suite:
        return None
    return TIER_MAP.get(Path(suite).parts[0], "quick")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        help="tier folder (default: from the sidecar's suite prefix, else quick;"
        " contradicting the sidecar warns, or errors under --strict)",
    )
    ap.add_argument(
        "--meta",
        action="append",
        default=[],
        help="key=value metadata pairs (repeatable); override the sidecar."
        " Values parse as JSON when they can (seed=42 is a number)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="global seed for metadata; overrides the sidecar's",
    )
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument(
        "--force", action="store_true", help="append even if exact dup exists"
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on exact dup, bad sidecar, or sidecar override (CI mode)",
    )
    args = ap.parse_args()

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
        level = "ERROR" if args.strict else "WARN"
        print(f"  {level}: bad sidecar {sidecar_path}: {sidecar_err}", file=sys.stderr)
        if args.strict:
            return 1
    # What the capture recorded, including the reference_source derived from
    # it: overrides are checked against this, not just the raw sidecar keys.
    captured = dict(sidecar) if sidecar is not None else {}
    if sidecar is not None:
        ref = _capture_reference_source(args.graph, sidecar)
        if ref is not None:
            captured["reference_source"] = ref
    meta = dict(captured)
    meta.setdefault("format_version", 1)
    captured_tier = _capture_tier(sidecar)
    if args.tier is None:
        args.tier = captured_tier or "quick"

    # Explicit CLI values win. Overriding a value the capture recorded is
    # allowed but never silent: the sidecar is what the C++ test actually ran.
    overrides = []
    for kv in args.meta:
        if "=" in kv:
            k, v = kv.split("=", 1)
            v = _parse_meta_value(v)
            if k == "inputs":
                err = _inputs_error(v)
                if err is not None:
                    print(f"import_graph: --meta {err}", file=sys.stderr)
                    return 1
            overrides.append((f"--meta {k}", k, v))
            meta[k] = v
    if args.seed is not None:
        overrides.append(("--seed", "seed", args.seed))
        meta["seed"] = args.seed

    conflicts = [
        (flag, captured[k], v)
        for flag, k, v in overrides
        if k in captured and canon(captured[k]) != canon(v)
    ]
    # The tier is not metadata, but it picks the CI lane: --tier quick on a
    # Full/ capture would silently move the case to the smoke lane.
    if captured_tier is not None and args.tier != captured_tier:
        conflicts.append(("--tier", captured_tier, args.tier))
    for flag, captured, supplied in conflicts:
        level = "ERROR" if args.strict else "WARN"
        print(
            f"  {level}: {flag} {canon(supplied)} overrides captured"
            f" {canon(captured)} from {sidecar_path}",
            file=sys.stderr,
        )
    if conflicts and args.strict:
        return 1

    # Canonicalize UIDs by tensor name so an imported graph lines up with sweeps
    # built by place_bundles (which does the same). The C++ builder auto-assigns
    # UIDs non-deterministically, so the graph and its inputs fill-spec map must
    # be remapped in lockstep before hashing/matching (canonical_uid_map_by_name).
    uid_map = canonical_uid_map_by_name(graph)
    graph = remap_graph(graph, uid_map)
    meta = remap_meta_inputs(meta, uid_map)

    h = skeleton_hash(graph)
    op = derive_operation(graph)
    graph_json = canon(graph)

    print(f"  skeleton hash: {h}", file=sys.stderr)
    print(f"  operation:     {op}", file=sys.stderr)

    index = _index_existing(args.bundle_dir)
    matches = index.get(h, [])

    # --- Check for exact duplicates ---
    for template, sweep, sweep_path in matches:
        for case in sweep.get("cases", []):
            expanded = expand(template, case.get("values", {}))
            if canon(expanded) == graph_json:
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

    # --- Structural match: append to existing sweep ---
    if matches:
        # Prefer a sweep in the requested tier so Smoke cases land in quick/.
        tier_prefix = str(args.bundle_dir / args.tier) + os.sep
        tier_matches = [m for m in matches if str(m[2]).startswith(tier_prefix)]
        template, sweep, sweep_path = (tier_matches or matches)[0]
        values = _extract_values(graph, template)
        new_case = {"id": None, "values": values, "metadata": meta}
        existing_cases = sweep.get("cases", [])
        saved_ids = [c["id"] for c in existing_cases]
        all_cases = existing_cases + [new_case]
        assign_case_ids(all_cases)
        for c, orig_id in zip(existing_cases, saved_ids):
            c["id"] = orig_id
        taken = set(saved_ids)
        if new_case["id"] in taken:
            new_case["id"] = f"{new_case['id']}_{_case_hash(new_case)}"

        expanded = expand(template, values)
        if canon(expanded) != canon(graph):
            _report_roundtrip_failure(
                "after extraction",
                expanded,
                graph,
                selected=f"{sweep_path}"
                f" (of {len(matches)} structural match(es),"
                f" {len(tier_matches)} in tier '{args.tier}')",
            )
            return 1

        if not args.dry_run:
            sweep["cases"] = [
                {"id": c["id"], "values": c["values"], "metadata": c["metadata"]}
                for c in all_cases
            ]
            with open(sweep_path, "w") as f:
                json.dump(sweep, f, indent=2)
                f.write("\n")
        print(f"  appended case '{new_case['id']}' to {sweep_path}", file=sys.stderr)
        return 0

    # --- No structural match: create new template+sweep ---
    template = copy.deepcopy(graph)
    for fld in TOP_LEVEL_IF_VARIES:
        if fld in template:
            template[fld] = f"${{case.{fld}}}"
    for t in template.get("tensors", []):
        for fld in TENSOR_ALWAYS:
            if fld in t:
                t[fld] = f"${{case.{fld}}}"

    values = _extract_values(graph, template)

    expanded = expand(template, values)
    if canon(expanded) != canon(graph):
        _report_roundtrip_failure("for new template", expanded, graph)
        return 1

    tmap = tensors_by_uid(graph)
    first = tmap[min(tmap)] if tmap else {}
    layout = infer_layout(first.get("dims"), first.get("strides")) or "Default"
    topo_name = sanitize(layout).replace(" ", "_").title() or "Default"

    out_dir = args.bundle_dir / args.tier / op / topo_name
    case_entry = {"id": "case", "values": values, "metadata": meta}
    sweep_out = {"version": 1, "cases": [case_entry]}

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "graph.template.json", "w") as f:
            json.dump(template, f, indent=2)
            f.write("\n")
        with open(out_dir / "sweep.json", "w") as f:
            json.dump(sweep_out, f, indent=2)
            f.write("\n")
    print(f"  created new bundle: {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
