# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Catalog check of a strided sweep: per cell, best shipped kernel against best enumerated.

``report`` joins the winner rankings of a ``tools/sweep.py`` output (the arm that timed the
enumeration) to the graph map ``ground_truth.py layouts`` wrote and to the catalogs: a kernel
is its catalog entry without the shape fields, so one binary shipped for several head-count
cells is one kernel, and "shipped" is a set recomputed from any KDP. Per graph: best shipped
time / best enumerated time. Per cell (dtype, head size, mask, layout): the geomean of those
ratios against the pre-registered band; shapes above the band are listed as WATCH. A kernel
the engine cannot launch is never in the arm; ``--full-kdp`` lists them apart.
``screen`` picks the candidates of a one-round screening sweep worth a stage-2 measurement:
the minimum of many noisy timings is biased fast, so a screen never gives a verdict.

Exit 0 (``report --strict``: 1 unless every cell passes), 2 on unusable input.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_diff import binary_identity  # noqa: E402
from variant_reduce import (  # noqa: E402
    Ranking,
    ReduceError,
    _phase_attempts,
    _shards,
    load_kdp,
    noise_floor,
    read_lines,
)

# Pre-registered before any data: a cell passes when geomean(best shipped / best enumerated)
# <= 1 + B, B = max(BAND_FLOOR, p95 |A'/A - 1| of the same run's A/A' control, DRIFT_MULTIPLIER
# x the cell's round-to-round drift). Do not tune these after seeing results.
BAND_FLOOR = 0.03
DRIFT_MULTIPLIER = 2
SCREEN_TOP_K = 12
SCREEN_MARGIN = 2 * BAND_FLOOR
SHAPE_FIELDS = frozenset(
    {"num_query_heads", "num_kv_heads", "batch", "seqlen_q", "seqlen_kv"}
)
_ID_FIELDS = ("pack_id", "dispatch_id", "kernel_id")


def _specs(entry: dict) -> list:
    sources = [entry["kernel_source"]] if "kernel_source" in entry else []
    return [s.get("spec") or {} for s in sources or entry["sources"].values()]


def kernel_identity(entry: dict, defaults: dict | None = None) -> str:
    """The binary an entry builds: its kernel source without the shape fields (head
    counts, batch, sequence lengths), so one binary shipped for several head-count
    cells is one kernel. ``defaults`` (the spec dataclass defaults) complete the spec
    fields an entry omits: a shipped entry omits the knobs it holds at default, a
    full-space entry carries them, and the two are the same binary."""
    defaults = {k: v for k, v in (defaults or {}).items() if k not in SHAPE_FIELDS}
    keep = frozenset(k for spec in _specs(entry) for k in spec) - SHAPE_FIELDS
    keep |= frozenset(defaults)

    def complete(source: dict) -> dict:
        return {**source, "spec": {**defaults, **(source.get("spec") or {})}}

    done = dict(entry)
    if "kernel_source" in entry:
        done["kernel_source"] = complete(entry["kernel_source"])
    else:
        done["sources"] = {n: complete(v) for n, v in entry["sources"].items()}
    return json.dumps(binary_identity(done, keep), sort_keys=True)


def spec_defaults(manifest_path) -> dict:
    """The spec dataclass defaults of the manifest's registry (rocKE must import)."""
    import dataclasses

    import registry_enumerator as enumerator

    manifest = enumerator.load_manifest(str(manifest_path))
    enumerator.bind_provider(manifest.get("provider_root"))
    spec = enumerator.enumerate_space(manifest).rows[0][1]
    return {
        f.name: f.default
        for f in dataclasses.fields(spec)
        if f.default is not dataclasses.MISSING
    }


class Catalog:
    """Kernel identities of one or more KDPs, and the ids winner lines use for them."""

    def __init__(self, kdp_paths, defaults: dict | None = None):
        self.by_id, self.names, self.persistent = {}, defaultdict(list), set()
        for path in kdp_paths:
            kdp = load_kdp(path)
            for entry in kdp["kernelDescriptors"]:
                key = (str(kdp["id"]).lower(), str(kdp["dispatch"]).lower())
                ident = kernel_identity(entry, defaults)
                self.by_id[(*key, str(entry["id"]).lower())] = ident
                self.names[ident].append(entry["name"])
                if any(spec.get("persistent") for spec in _specs(entry)):
                    self.persistent.add(ident)

    @property
    def identities(self) -> set:
        return set(self.names)

    def of(self, winner_entry: dict):
        return self.by_id.get(tuple(str(winner_entry[f]).lower() for f in _ID_FIELDS))


def graph_name_of(graph_identity: str):
    """The ``name`` of a winner line's Graph flatbuffer (graph.fbs field 0), or None.
    The winner cache keys on graph content, so this is the only way back to the graph.
    """
    try:
        buf = base64.b64decode(graph_identity)
        table = struct.unpack_from("<I", buf, 0)[0]
        vtable = table - struct.unpack_from("<i", buf, table)[0]
        if struct.unpack_from("<H", buf, vtable)[0] < 6:
            return None
        offset = struct.unpack_from("<H", buf, vtable + 4)[0]
        if not offset:
            return None
        start = table + offset + struct.unpack_from("<I", buf, table + offset)[0]
        size = struct.unpack_from("<I", buf, start)[0]
        return buf[start + 4 : start + 4 + size].decode()
    except (ValueError, struct.error):
        return None


def _timing(sweep_dir, arm):
    for attempt, record in _phase_attempts(sweep_dir, arm):
        if record["phase"]["kind"] == "timing" and record.get("status") == "success":
            yield attempt, record


def winner_rounds(sweep_dir, arm, catalog, graph_names) -> tuple:
    """``({round: {graph: {identity: time_ms}}}, problems)`` of the arm's successful
    timing phases. ``problems`` counts winner records of no mapped graph and winner
    entries of no catalog kernel (a different install than the KDPs given)."""
    rounds, problems = defaultdict(dict), Counter()
    for attempt, record in _timing(sweep_dir, arm):
        for shard in _shards(attempt):
            lines = {x.key: x for x in read_lines(shard) if x.key is not None}
            for line in lines.values():
                name, times = graph_name_of(line.graph), {}
                if name not in graph_names:
                    problems["unmapped_records"] += 1
                    continue
                for entry in line.entries:
                    ident = catalog.of(entry)
                    if ident is None:
                        problems["unknown_kernel_entries"] += 1
                    else:
                        times[ident] = min(entry["time_ms"], times.get(ident, math.inf))
                rounds[record["phase"]["round"]][name] = times
    return dict(rounds), problems


def served_times(sweep_dir, arm) -> dict:
    """``{round: {graph: mean_ms}}``: the benchmark's own mean for the kernel the engine
    served, from the ledgers of the arm's successful timing phases."""
    out = defaultdict(dict)
    for _, record in _timing(sweep_dir, arm):
        for e in record.get("ledger") or []:
            if e.get("outcome") == "served" and e.get("mean_ms"):
                out[record["phase"]["round"]][e["graph_name"]] = e["mean_ms"]
    return dict(out)


def median_over_rounds(by_round: dict) -> dict:
    """``{graph: median of its per-round values}``."""
    values = defaultdict(list)
    for by_graph in by_round.values():
        for graph, value in by_graph.items():
            values[graph].append(value)
    return {g: statistics.median(v) for g, v in values.items()}


def geomean(values) -> float:
    return math.exp(statistics.fmean(math.log(v) for v in values))


def p95(values) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def control_band(sweep_dir, arm, control):
    """p95 of ``|A'/A - 1|`` over the graphs both arms served (medians over rounds), or
    None when they share none. A and A' are one install tree listed under two names."""
    a = median_over_rounds(served_times(sweep_dir, arm))
    b = median_over_rounds(served_times(sweep_dir, control))
    shared = [abs(b[g] / a[g] - 1) for g in a.keys() & b.keys()]
    return p95(shared) if shared else None


def cell_drift(rounds: dict, graphs: set):
    """Median over graphs and kernels of (max - min) / median of ``t / t_best`` across
    rounds (``variant_reduce.noise_floor``'s drift) for ``graphs``, or None with fewer
    than two rounds."""
    ranked = {
        r: {
            g: Ranking(r, "", g, sorted(t.items(), key=lambda e: (e[1], e[0])), ())
            for g, t in by_graph.items()
            if g in graphs and t
        }
        for r, by_graph in rounds.items()
    }
    ranked = {r: k for r, k in ranked.items() if k}
    try:
        return noise_floor(ranked, [], 0.0, 1).drift if len(ranked) > 1 else None
    except ReduceError:
        return None


def _passed(entry: dict) -> bool:
    c = entry.get("correctness")
    keys = ("passed", "execution_success", "tolerance_match")
    return isinstance(c, dict) and all(c.get(k) is True for k in keys)


def numeric_checks(sweep_dir, arms, catalog, graph_names) -> dict:
    """``{graph: {identity: passed}}``: the kernel each correctness phase of ``arms``
    served for a graph (the best of its forced ranking) and whether it passed."""
    out = defaultdict(dict)
    for arm in arms:
        for attempt, record in _phase_attempts(sweep_dir, arm):
            if record["phase"]["kind"] != "correctness":
                continue
            ledger = {
                e.get("graph_name"): _passed(e)
                for e in record.get("ledger") or []
                if e.get("outcome") == "served"
                and e.get("correctness") is not None
                and (e.get("reference") or {}).get("usable") is not False
            }
            for shard in _shards(attempt):
                for line in read_lines(shard):
                    name = graph_name_of(line.graph) if line.key else None
                    served = [(e["time_ms"], catalog.of(e)) for e in line.entries]
                    if name in graph_names and name in ledger and served:
                        out[name][min(served, key=lambda s: (s[0], str(s[1])))[1]] = (
                            ledger[name]
                        )
    return out


def _numeric(checks: dict, ident) -> str:
    verdict = checks.get(ident)
    return "not checked" if verdict is None else "pass" if verdict else "FAIL"


def build_report(mapping, rounds, shipped, catalog, band=None, checks=None) -> dict:
    """The per-graph and per-cell verdicts. ``mapping``: the `layouts` map entries;
    ``rounds``: `winner_rounds`; ``shipped``: kernel identities; ``band``: the A/A'
    control's p95 (None: no control, the floor and the cell drift only)."""
    checks = checks or {}
    times = defaultdict(lambda: defaultdict(list))
    for by_graph in rounds.values():
        for graph, kernels in by_graph.items():
            for ident, t in kernels.items():
                times[graph][ident].append(t)
    graphs, cells = [], defaultdict(list)
    for m in mapping:
        med = {
            i: statistics.median(v) for i, v in times.get(m["graph_name"], {}).items()
        }
        best = min(med.items(), key=lambda e: (e[1], e[0]), default=None)
        mine = [e for e in med.items() if e[0] in shipped]
        ship = min(mine, key=lambda e: (e[1], e[0]), default=None)
        fam = m["family"]
        cell = (fam["dtype"], fam["head_size"], fam["mask_type"], m["layout"])
        repeats = m["layout"] != "bshd" and m.get("same_as_bshd", False)
        got = {"best_shipped": ship, "best_enumerated": best}
        row = dict(graph=m["graph_name"], cell=list(cell), shape=m["shape"])
        row |= dict(candidates=len(med), shipped_candidates=len(mine))
        row |= dict(repeats_twin=repeats)
        for key, found in got.items():
            checked = checks.get(m["graph_name"], {})
            row[key] = (
                None
                if found is None
                else dict(
                    kernel=catalog.names[found[0]][0],
                    time_ms=found[1],
                    numeric=_numeric(checked, found[0]),
                )
            )
        row["ratio"] = ship[1] / best[1] if ship and best else None
        graphs.append(row)
        if not repeats:
            cells[cell].append(row)
    out_cells = []
    for cell, members in cells.items():
        drift = cell_drift(rounds, {r["graph"] for r in members})
        bound = max(BAND_FLOOR, band or 0.0, DRIFT_MULTIPLIER * (drift or 0.0))
        ratios = [r["ratio"] for r in members if r["ratio"]]
        gm = geomean(ratios) if ratios else None
        verdict = (
            "INCOMPLETE"
            if len(ratios) < len(members) or not ratios
            else ("PASS" if gm <= 1 + bound else "FAIL")
        )
        out_cells.append(
            dict(
                cell=list(cell),
                graphs=len(members),
                geomean_ratio=gm,
                band=bound,
                drift=drift,
                verdict=verdict,
                watch=[
                    r["shape"] for r in members if r["ratio"] and r["ratio"] > 1 + bound
                ],
                missing=[r["graph"] for r in members if not r["ratio"]],
            )
        )
    return dict(
        band_basis=dict(
            floor=BAND_FLOOR, control_p95=band, drift_multiplier=DRIFT_MULTIPLIER
        ),
        cells=out_cells,
        graphs=graphs,
    )


def not_launchable(arm: Catalog, full: Catalog) -> dict:
    """Kernels of the full enumeration the arm does not hold: the engine cannot launch
    them (a persistent body on gfx950), so they never count as "best enumerated"."""
    absent = full.identities - arm.identities
    return dict(
        specs=len(absent),
        persistent=len(absent & full.persistent),
        examples=sorted(full.names[i][0] for i in absent)[:5],
    )


def screen_candidates(rounds, shipped, top_k=SCREEN_TOP_K, margin=SCREEN_MARGIN) -> set:
    """Kernel identities to re-measure: the shipped ones, and per graph the top ``top_k``
    and every kernel within ``margin`` of its best (median over the screen's rounds)."""
    times = defaultdict(lambda: defaultdict(list))
    for by_graph in rounds.values():
        for graph, kernels in by_graph.items():
            for ident, t in kernels.items():
                times[graph][ident].append(t)
    keep = set(shipped)
    for per_kernel in times.values():
        med = sorted((statistics.median(v), i) for i, v in per_kernel.items())
        if med:
            keep |= {i for _, i in med[:top_k]}
            keep |= {i for t, i in med if t <= med[0][0] * (1 + margin)}
    return keep


def _load_yaml(path):
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReduceError(f"{path}: not a readable YAML file ({exc})") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "screen"):
        p = sub.add_parser(name)
        p.add_argument("--sweep-output", required=True)
        p.add_argument(
            "--arm", required=True, help="the arm that timed the enumeration"
        )
        p.add_argument("--arm-kdp", action="append", required=True)
        p.add_argument("--shipped-kdp", action="append", required=True)
        p.add_argument("--map", required=True, help="`ground_truth.py layouts` map")
        p.add_argument(
            "--manifest", help="engine manifest: completes omitted spec fields"
        )
        p.add_argument("--out", required=True, help="YAML")
    p = sub.choices["report"]
    p.add_argument(
        "--control-arm", help="the A' arm (the same install under a 2nd name)"
    )
    p.add_argument("--check-arm", action="append", help="arms whose correctness counts")
    p.add_argument("--full-kdp", action="append", help="the whole enumeration")
    p.add_argument("--strict", action="store_true")
    p = sub.choices["screen"]
    p.add_argument(
        "--out-names", required=True, help="descriptor names to keep, one per line"
    )
    args = parser.parse_args(argv)
    try:
        mapping = _load_yaml(args.map)
        names = {m["graph_name"] for m in mapping}
        defaults = spec_defaults(args.manifest) if args.manifest else None
        arm_catalog = Catalog(args.arm_kdp, defaults)
        shipped = Catalog(args.shipped_kdp, defaults).identities
        if not shipped & arm_catalog.identities:
            raise ReduceError(
                "no shipped kernel is in the arm: the catalogs may spell the same "
                "binary with different spec fields; pass --manifest to complete them"
            )
        rounds, problems = winner_rounds(
            args.sweep_output, args.arm, arm_catalog, names
        )
        if not rounds:
            raise ReduceError(
                f"{args.sweep_output}: no timing winners for arm {args.arm!r}"
            )
        if args.command == "screen":
            keep = screen_candidates(rounds, shipped & arm_catalog.identities)
            kept = sorted(n for i in keep for n in arm_catalog.names[i])
            Path(args.out_names).write_text("".join(f"{n}\n" for n in kept), "utf-8")
            result = dict(
                kernels=len(keep), descriptors=len(kept), problems=dict(problems)
            )
            code = 0
        else:
            band = None
            if args.control_arm:
                band = control_band(args.sweep_output, args.arm, args.control_arm)
            checks = numeric_checks(
                args.sweep_output, args.check_arm or [args.arm], arm_catalog, names
            )
            result = build_report(mapping, rounds, shipped, arm_catalog, band, checks)
            result["problems"] = dict(problems)
            if args.full_kdp:
                result["not_launchable"] = not_launchable(
                    arm_catalog, Catalog(args.full_kdp, defaults)
                )
            verdicts = Counter(c["verdict"] for c in result["cells"])
            print(f"cells: {dict(verdicts)}; problems: {dict(problems)}")
            code = 1 if args.strict and set(verdicts) != {"PASS"} else 0
    except (ReduceError, KeyError, TypeError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    import yaml

    Path(args.out).write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
