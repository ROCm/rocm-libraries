# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Compare a kernel owner's benchmark numbers with what hipDNN serves.

``ingest`` converts the owner's rows (CSV or JSON list) into request fields of the
manifest's request class; ``graphs`` writes one hipDNN graph per row from the
manifest's ``ground_truth`` template; ``compare`` joins those graphs to a
``tools/sweep.py`` output by name and reports hipDNN / owner against a margin.

Exit 0 on success (``compare`` whatever the comparison finds), 2 on unusable input
(including a ground-truth row or graph that no longer matches what ``graphs``
recorded). ``compare --strict`` exits 1 unless every row is within the margin, was
not declined in any timing round, and has a passing correctness verdict (outside, missing,
declined, failing, not run and no-reference rows all refuse).
"""

from __future__ import annotations

import argparse
import csv
import enum
import hashlib
import json
import math
import re
import statistics
import sys
import typing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from device_probe import is_arch_token  # noqa: E402
from registry_enumerator import (  # noqa: E402
    EnumerationError,
    Registry,
    bind_provider,
    cell_groups,
    check_value_map_cells,
    load_manifest,
)
from sweep import semantic_identity  # noqa: E402
from variant_reduce import ReduceError, load_phase_records  # noqa: E402

SOURCE = "ground_truth"
_PLACEHOLDER = re.compile(r"\$\{([^{}]*)\}")
_INTEGER = re.compile(r"[+-]?\d+")
_TIME_SCALE = {"ms": 1.0, "us": 1e-3}
_GT_KEYS = {"request", "time_ms", "accuracy", "device", "_provenance"}


class GroundTruthError(RuntimeError):
    """The input cannot be used as given; exit 2."""


def _pairs(values: list, flag: str) -> list:
    out = []
    for value in values or []:
        key, sep, column = value.partition("=")
        if not sep or not key or not column:
            raise GroundTruthError(f"{flag} {value!r} must be KEY=COLUMN")
        out.append((key, column))
    return out


def _read_json(path, what: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GroundTruthError(f"{path} is not a readable {what}: {exc}") from exc


def read_rows(path) -> list:
    if Path(path).suffix.lower() != ".json":
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise GroundTruthError(f"cannot read {path}: {exc}") from exc
        return list(csv.DictReader(text.splitlines()))
    rows = _read_json(path, "JSON file")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise GroundTruthError(f"{path} must be a JSON list of objects")
    return rows


def convert(registry: Registry, field: str, raw):
    """``raw`` as request field ``field``'s type; ValueError when it is not one."""
    hint = registry.hints.get(field)
    args = typing.get_args(hint) or (hint,)
    if isinstance(raw, str):
        raw = raw.strip()
        if raw == "":
            raise ValueError("empty")
    is_integer = isinstance(raw, str) and _INTEGER.fullmatch(raw)
    enums = [a for a in args if isinstance(a, type) and issubclass(a, enum.Enum)]
    if enums:
        for candidate in [raw, int(raw)] if is_integer else [raw]:
            if not isinstance(candidate, bool):
                value = registry.coerce(field, candidate)
                if isinstance(value, enum.Enum):
                    return value
        members = [m.name for e in enums for m in e]
        raise ValueError(f"not a member name or value of {members}")
    if bool in args:
        lowered = str(raw).lower()
        if lowered not in ("true", "1", "false", "0"):
            raise ValueError("not a boolean")
        return lowered in ("true", "1")
    if int in args:
        if is_integer or (isinstance(raw, int) and not isinstance(raw, bool)):
            return int(raw)
        raise ValueError("not an integer")
    if float in args:
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError("not a number") from None
    if str in args:
        if isinstance(raw, str):
            return raw
        raise ValueError("not a string")
    raise ValueError(f"request field type {hint!r} is not mappable from a column")


def _time(raw, scale: float) -> float:
    try:
        value = float(raw.strip() if isinstance(raw, str) else raw)
    except (TypeError, ValueError):
        raise ValueError(f"time {raw!r} is not a number") from None
    if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"time {raw!r} is not a positive finite number")
    return value * scale


def _accuracy(raw):
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            return float(raw)
        except ValueError:
            return raw
    return raw


def ingest(rows: list, registry: Registry, args) -> list:
    """GT rows from the owner's rows, per the ``ingest`` command-line ``args``."""
    mapping, filters = _pairs(args.map, "--map"), _pairs(args.filter, "--filter")
    if not is_arch_token(args.device):
        raise GroundTruthError(f"--device {args.device!r} is not an exact gfx arch")
    fields = [field for field, _ in mapping]
    unknown = [f for f in fields if f not in registry.request_fields]
    if unknown:
        raise GroundTruthError(
            f"--map names {unknown}, which {registry.request_cls.__name__} does not "
            f"define. Request fields: {sorted(registry.request_fields)}"
        )
    repeated = sorted({f for f in fields if fields.count(f) > 1})
    if repeated:
        raise GroundTruthError(f"--map sets request field(s) {repeated} twice")
    columns = {c for _, c in mapping} | {c for c, _ in filters} | {args.time_col}
    columns |= set(args.accuracy_col)
    out, failures = [], []
    for number, row in enumerate(rows, start=1):
        absent = sorted(c for c in columns if c not in row)
        if absent:
            failures.append(f"row {number}: no column(s) {absent}")
            continue
        if any(str(row[c]).strip() != v for c, v in filters):
            continue
        request, problems = {}, []
        for field, column in mapping:
            try:
                value = convert(registry, field, row[column])
                request[field] = value.name if isinstance(value, enum.Enum) else value
            except ValueError as exc:
                problems.append(f"{field} <- {column}={row[column]!r}: {exc}")
        try:
            time_ms = _time(row[args.time_col], _TIME_SCALE[args.time_unit])
        except ValueError as exc:
            problems.append(f"{args.time_col}: {exc}")
        if problems:
            failures.append(f"row {number}: " + "; ".join(problems))
            continue
        accuracy = {
            c: _accuracy(row[c])
            for c in args.accuracy_col
            if row[c] is not None and str(row[c]).strip() != ""
        }
        provenance = {"source": SOURCE, "row": number}
        out.append(dict(request=request, time_ms=time_ms, accuracy=accuracy))
        out[-1].update(device=args.device, _provenance=provenance)
    if failures:
        raise GroundTruthError("unmappable row(s):\n  " + "\n  ".join(failures))
    if not out:
        raise GroundTruthError("no row is left after --filter")
    return out


def graph_name(row: dict) -> str:
    return f"{row['_provenance']['source']}_r{int(row['_provenance']['row']):04d}"


def row_digest(row: dict) -> str:
    """sha256 of one ground-truth row, which `graphs` records and `compare` rechecks.
    A row's accuracy cell may be non-finite, which `ingest` accepts and a JSON file
    round-trips, so the dump allows it."""
    text = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode()).hexdigest()


def graphs_map_path(gt_path) -> Path:
    gt_path = Path(gt_path)
    return gt_path.with_name(gt_path.name.removesuffix(".json") + ".graphs.json")


def load_gt(path) -> list:
    rows = _read_json(path, "ground-truth file")
    if not isinstance(rows, list) or any(
        not isinstance(r, dict) or set(r) != _GT_KEYS for r in rows
    ):
        raise GroundTruthError(f"{path} must be a JSON list of `ingest` rows")
    for index, row in enumerate(rows):
        time_ms = row["time_ms"]
        valid = type(time_ms) in (int, float) and math.isfinite(time_ms)
        if not valid or time_ms <= 0:
            raise GroundTruthError(
                f"{path}[{index}]: time_ms {time_ms!r} is not a positive finite number"
            )
    return rows


def ground_truth_block(manifest: dict, manifest_path) -> tuple:
    """``(template graph, value_map)`` from the manifest's ``ground_truth`` block."""
    block = manifest.get("ground_truth")
    if not isinstance(block, dict) or "graph_template" not in block:
        raise GroundTruthError(f"{manifest_path} has no ground_truth.graph_template")
    template_path = Path(manifest_path).resolve().parent / block["graph_template"]
    template = _read_json(template_path, "graph template")
    value_map = block.get("value_map") or {}
    if not isinstance(value_map, dict) or any(
        not isinstance(v, dict) for v in value_map.values()
    ):
        raise GroundTruthError("ground_truth.value_map must map field -> {value: x}")
    check_value_map_cells(cell_groups(manifest["enumerate"].get("cells")), value_map)
    return template, value_map


def _placeholders(node, out: set) -> set:
    if isinstance(node, (dict, list)):
        for value in node.values() if isinstance(node, dict) else node:
            _placeholders(value, out)
    elif isinstance(node, str):
        match = _PLACEHOLDER.fullmatch(node)
        if match:
            out.add(match.group(1))
        elif "${" in node:
            raise GroundTruthError(f"template string {node!r} is not one placeholder")
    return out


def _substitute(node, values: dict):
    if isinstance(node, dict):
        return {k: _substitute(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, values) for v in node]
    match = isinstance(node, str) and _PLACEHOLDER.fullmatch(node)
    return values[match.group(1)] if match else node


def _pointer_parent(graph, pointer: str):
    """``(container, key)`` of an existing location. This is RFC 6901 plus one
    extension: a segment under a list may be the ``name`` of exactly one dict element.
    Naming a node survives the template gaining or reordering nodes; an index still
    works, so an existing manifest keeps its meaning."""
    if not pointer.startswith("/"):
        raise GroundTruthError(f"value_map patch {pointer!r} is not a JSON pointer")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")]
    node = graph
    for index, part in enumerate(parts):
        if isinstance(node, list):
            named = [
                i
                for i, item in enumerate(node)
                if isinstance(item, dict) and item.get("name") == part
            ]
            if len(named) == 1:
                key = named[0]
            elif not named and part.isdigit() and int(part) < len(node):
                key = int(part)
            else:
                break
        elif isinstance(node, dict) and part in node:
            key = part
        else:
            break
        if index == len(parts) - 1:
            return node, key
        node = node[key]
    raise GroundTruthError(f"value_map patch {pointer!r} names no template location")


def packed_strides(dims: list, order: list | None = None) -> list:
    """Packed strides with the axes laid out outermost to innermost as ``order``
    lists them (default: ``dims`` order)."""
    strides, step = [0] * len(dims), 1
    for axis in reversed(order or range(len(dims))):
        strides[axis] = step
        step *= dims[axis]
    return strides


def instantiate(template: dict, value_map: dict, row: dict) -> dict:
    request = row["request"]
    where = f"row {row['_provenance']['row']}"
    placeholders = _placeholders(template, set())
    unbound = sorted(placeholders - set(request))
    if unbound:
        raise GroundTruthError(f"{where}: placeholder(s) {unbound} are not in the row")
    values, patches = {}, {}
    for field, value in request.items():
        if field in value_map:
            table = value_map[field]
            if value not in table:
                raise GroundTruthError(
                    f"{where}: ground_truth.value_map.{field} lists no {value!r}"
                )
            if isinstance(table[value], dict):
                patches.update(table[value])
                continue
            value = table[value]
        if field not in placeholders:
            raise GroundTruthError(
                f"{where}: {field} is consumed by no template placeholder or patch"
            )
        values[field] = value
    patched = sorted(placeholders - set(values))
    if patched:
        raise GroundTruthError(f"{where}: placeholder(s) {patched} map to patches")
    graph = _substitute(template, values)
    for pointer, value in patches.items():
        container, key = _pointer_parent(graph, pointer)
        container[key] = value
    for tensor in graph.get("tensors", []):
        order = tensor.pop("stride_order", None)
        if tensor.get("strides") == "packed":
            dims = tensor.get("dims")
            if not isinstance(dims, list) or any(
                type(d) is not int or d <= 0 for d in dims
            ):
                raise GroundTruthError(
                    f"{where}: tensor {tensor.get('name')!r} dims {dims!r} are not "
                    "positive integers"
                )
            if order is not None and sorted(order) != list(range(len(dims))):
                raise GroundTruthError(
                    f"template tensor {tensor.get('name')!r} stride_order {order!r} "
                    f"is not a permutation of its {len(dims)} axes"
                )
            tensor["strides"] = packed_strides(dims, order)
    graph["name"] = graph_name(row)
    graph["_provenance"] = dict(row["_provenance"])
    return graph


def write_graphs(gt_path, manifest_path, out_dir) -> list:
    rows = load_gt(gt_path)
    manifest = load_manifest(str(manifest_path))
    template, value_map = ground_truth_block(manifest, manifest_path)
    graphs = [instantiate(template, value_map, row) for row in rows]
    names = [g["name"] for g in graphs]
    if len(set(names)) != len(names):
        raise GroundTruthError(f"{gt_path}: two rows share a provenance row number")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    foreign = sorted(p.name for p in out_dir.glob("*.json") if p.stem not in names)
    if foreign:
        raise GroundTruthError(f"{out_dir} already holds other graphs: {foreign}")
    for graph in graphs:
        text = json.dumps(graph, indent=4) + "\n"
        (out_dir / f"{graph['name']}.json").write_text(text, encoding="utf-8")
    mapping = [
        {
            "row": row["_provenance"]["row"],
            "graph_name": graph["name"],
            "semantic_identity": semantic_identity(graph),
            "gt_row_sha256": row_digest(row),
        }
        for row, graph in zip(rows, graphs)
    ]
    text = json.dumps(mapping, indent=1) + "\n"
    graphs_map_path(gt_path).write_text(text, encoding="utf-8")
    return names


def _passed(correctness) -> bool:
    keys = ("passed", "execution_success", "tolerance_match")
    return isinstance(correctness, dict) and all(
        correctness.get(k) is True for k in keys
    )


def _entries(records: list, name: str):
    for record in records:
        yield from (
            e for e in record.get("ledger") or [] if e.get("graph_name") == name
        )


def compare(gt_path, sweep_dir, arm, margin: float) -> dict:
    """Each row's median served time over the arm's successful timing rounds, its
    ratio to the owner's time, and the arm's correctness verdict."""
    if not math.isfinite(margin) or margin < 0:
        raise GroundTruthError(f"--margin {margin} must be a non-negative fraction")
    rows = load_gt(gt_path)
    map_path = graphs_map_path(gt_path)
    try:
        mapped = {m["row"]: m for m in _read_json(map_path, "row map")}
        for entry in mapped.values():
            entry["graph_name"], entry["semantic_identity"], entry["gt_row_sha256"]
    except (KeyError, TypeError) as exc:
        raise GroundTruthError(
            f"{map_path} is not a `graphs` row map with identities; rerun `graphs`"
        ) from exc
    identities = Path(sweep_dir) / "input-identities.json"
    try:
        arch = _read_json(identities, "sweep identity")["device"]["arch"]
    except (KeyError, TypeError) as exc:
        raise GroundTruthError(f"{identities} names no device.arch") from exc
    devices = sorted({r["device"] for r in rows})
    if devices != [arch]:
        raise GroundTruthError(f"ground truth is from {devices}, the sweep {arch!r}")
    try:
        records = load_phase_records(sweep_dir)
    except ReduceError as exc:
        raise GroundTruthError(str(exc)) from exc
    arms = sorted(
        {r["phase"]["arm"] for r in records if r["phase"]["kind"] == "timing"}
    )
    if arm is None and len(arms) == 1:
        arm = arms[0]
    elif arm not in arms:
        raise GroundTruthError(f"name one timing arm with --arm; arms: {arms}")
    ours = [r for r in records if r["phase"]["arm"] == arm]
    timing = [
        r
        for r in ours
        if r["phase"]["kind"] == "timing" and r.get("status") == "success"
    ]
    checks = [r for r in ours if r["phase"]["kind"] == "correctness"]
    ours_timing = [r for r in ours if r["phase"]["kind"] == "timing"]
    results = []
    for row in rows:
        number = row["_provenance"]["row"]
        if number not in mapped:
            raise GroundTruthError(f"{map_path} maps no graph for row {number}")
        recorded = mapped[number]
        name = recorded["graph_name"]
        if row_digest(row) != recorded["gt_row_sha256"]:
            raise GroundTruthError(
                f"row {number} of {gt_path} differs from the row `graphs` wrote "
                f"{name} from; rerun `graphs` and the sweep"
            )
        for entry in (*_entries(ours_timing, name), *_entries(checks, name)):
            if entry.get("semantic_identity") != recorded["semantic_identity"]:
                raise GroundTruthError(
                    f"row {number}: the sweep ledger's graph {name} has semantic "
                    f"identity {entry.get('semantic_identity')!r}, not the "
                    f"{recorded['semantic_identity']!r} `graphs` recorded: the sweep "
                    f"timed a different graph under this name"
                )
        timed = list(_entries(timing, name))
        times = [e["mean_ms"] for e in timed if e.get("outcome") == "served"]
        outcomes = {
            f"{e.get('outcome')}: {e.get('reason')}"
            for e in timed
            if e.get("outcome") != "served"
        }
        # `--strict` also refuses a graph any timing round declined, and one whose
        # correctness phase failed a gate, whatever its own verdict.
        declined = sum(
            e.get("outcome") == "declined" for e in _entries(ours_timing, name)
        )
        phase_failed = any(
            r.get("status") != "success" for r in checks if any(_entries([r], name))
        )
        # A graph declined in the correctness phase has no verdict.
        verdicts = [e for e in _entries(checks, name) if e.get("outcome") == "served"]
        compared = [e for e in verdicts if (e.get("reference") or {}).get("usable")]
        if not verdicts:
            correctness = "not run"
        elif not compared:
            correctness = "no reference"
        elif all(_passed(e.get("correctness")) for e in compared):
            correctness = "pass"
        else:
            correctness = "FAIL"
        hipdnn_ms = statistics.median(times) if times else None
        ratio = hipdnn_ms / row["time_ms"] if times else None
        missing = "; ".join(sorted(outcomes)) or "in no successful timing phase"
        results.append(
            row
            | dict(row=number, graph_name=name, rounds=len(times), hipdnn_ms=hipdnn_ms)
            | dict(ratio=ratio, within_margin=ratio <= 1 + margin if times else None)
            | dict(correctness=correctness, missing_reason=None if times else missing)
            | {"correctness_detail": [e.get("correctness") for e in verdicts]}
            | dict(declined_rounds=declined, correctness_phase_failed=phase_failed)
        )
    rounds = sorted({r["phase"]["round"] for r in timing})
    return dict(
        arm=arm, arch=arch, margin=margin, timing_rounds=rounds, results=results
    )


def _fmt(value, digits=4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "-" if value is None else str(value)


def _summary(results: list) -> str:
    return (
        f"{len(results)} rows, "
        f"{sum(r['within_margin'] is True for r in results)} within margin, "
        f"{sum(r['within_margin'] is False for r in results)} outside, "
        f"{sum(r['correctness'] == 'FAIL' for r in results)} failing correctness, "
        f"{sum(r['hipdnn_ms'] is None for r in results)} missing"
    )


def render_report(comparison: dict, gt_path, sweep_dir) -> str:
    results = comparison["results"]
    lines = [
        "# Ground truth comparison",
        "",
        f"`{gt_path}` vs `{sweep_dir}` (arm `{comparison['arm']}`, arch "
        f"`{comparison['arch']}`, rounds {comparison['timing_rounds']}, margin "
        f"{comparison['margin']:g}): {_summary(results)}.",
    ]
    sections = (
        ("Rows outside the margin", "within_margin", False, "ratio"),
        ("Rows failing correctness", "correctness", "FAIL", "correctness_detail"),
        ("Rows with no correctness verdict", "correctness", "not run", "correctness"),
        ("Rows with no usable reference", "correctness", "no reference", "correctness"),
        ("Rows missing from the sweep", "hipdnn_ms", None, "missing_reason"),
    )
    for heading, field, value, detail in sections:
        found = [
            f"- row {r['row']} `{r['graph_name']}`: {_fmt(r[detail], 3)}"
            for r in results
            if r[field] == value
        ]
        lines += ["", f"## {heading}", "", *(found or ["- none"])]
    lines += [
        "",
        "## All rows",
        "",
        "| row | graph | request | owner ms | hipDNN ms (median) | rounds | ratio | "
        "margin | correctness | owner accuracy |",
        "|---" * 10 + "|",
    ]
    verdicts = {None: "missing", True: "within", False: "OUTSIDE"}
    for r in results:
        request = ", ".join(f"{k}={v}" for k, v in r["request"].items())
        accuracy = ", ".join(f"{k}={v}" for k, v in r["accuracy"].items()) or "-"
        lines.append(
            f"| {r['row']} | `{r['graph_name']}` | {request} | {_fmt(r['time_ms'])} | "
            f"{_fmt(r['hipdnn_ms'])} | {r['rounds']} | {_fmt(r['ratio'], 3)} | "
            f"{verdicts[r['within_margin']]} | {r['correctness']} | {accuracy} |"
        )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="owner rows -> GT.json")
    p.add_argument("--input", required=True, help="CSV or JSON list of rows")
    p.add_argument("--manifest", required=True)
    p.add_argument("--map", action="append", required=True, metavar="FIELD=COLUMN")
    p.add_argument("--time-col", required=True)
    p.add_argument("--time-unit", required=True, choices=sorted(_TIME_SCALE))
    p.add_argument("--accuracy-col", action="append", default=[])
    p.add_argument("--filter", action="append", default=[], metavar="COL=VALUE")
    p.add_argument("--device", required=True, help="gfx arch the owner measured on")
    p.add_argument("--out", required=True)
    p = sub.add_parser("graphs", help="GT.json -> one hipDNN graph per row")
    p.add_argument("--gt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare", help="GT.json vs a tools/sweep.py output")
    p.add_argument("--gt", required=True)
    p.add_argument("--sweep-output", required=True)
    p.add_argument("--arm", help="default: the only timing arm")
    p.add_argument("--margin", required=True, type=float, help="0.05 = 5%% slower")
    p.add_argument("--report", required=True)
    p.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 unless every row is within the margin with a passing correctness "
        "verdict",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "ingest":
            manifest = load_manifest(str(args.manifest))
            bind_provider(manifest.get("provider_root"))
            rows = ingest(read_rows(args.input), Registry(manifest), args)
            Path(args.out).write_text(json.dumps(rows, indent=1) + "\n", "utf-8")
            print(f"{len(rows)} rows -> {args.out}")
        elif args.command == "graphs":
            names = write_graphs(args.gt, args.manifest, args.out)
            print(f"{len(names)} graphs -> {args.out}")
        else:
            comparison = compare(args.gt, args.sweep_output, args.arm, args.margin)
            report = render_report(comparison, args.gt, args.sweep_output)
            Path(args.report).write_text(report, encoding="utf-8")
            print(f"report {args.report}: {_summary(comparison['results'])}")
            refused = [
                r["row"]
                for r in comparison["results"]
                if r["within_margin"] is not True
                or r["correctness"] != "pass"
                or r["declined_rounds"]
                or r["correctness_phase_failed"]
            ]
            if args.strict and refused:
                print(f"FAIL: --strict: rows {refused} are refused", file=sys.stderr)
                return 1
    except (GroundTruthError, EnumerationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
