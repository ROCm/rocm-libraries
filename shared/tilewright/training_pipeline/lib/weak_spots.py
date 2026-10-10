# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Weak spots of the bundle a run ships, and the pool gaps of its kernels.

Each grid cell gets the class of what limits it, which names the lever
that addresses it:

* `rule`: the serving rules leave out the winners: the best kernel the
  rules admit is below the target (the audit's pool_limited class);
* `identity`: the kernels a signature groups are served by a slower member
  (the audit's identity_limited class);
* `model`: a better pick exists within what the rules and the identity
  serve (the audit's model_limited class);
* `pool`: the kernels themselves: the true best of the cell's GEMMs comes
  on average to less than `report.pool_gap_fraction` of their roofline.

The first three come from the bundle's audit evaluation per group (under
v2 where the group has it); `pool` from the exhaustively benched GEMMs of
the run: the audit set when it is complete for the bundle's pool (held
out: where the kernels are weak, never which shapes), else round 0's
exhaustive seed bench. A cell is `passing` with none of them; one with
several names them all (`model+pool`).

Roofline of one GEMM: its attainable time is the larger of the benches'
roofline (`lib.bench_yaml.roofline_us`: its matrix work, 2*M*N*K*batch
FLOPs, at the peak matrix rate, and its compulsory traffic, A and B read
and D written once, at the memory bandwidth) and the device's latency
floor, the best time of the fastest exclusive GEMM of the set (no launch
finishes sooner, so that a launch-bound shape is no gap). The rates are
the run's `bench.adaptive_long` ones when it gives them, else the device's
(`lib.hardware.roofline_rates`), each raised to the best any GEMM of the set
reached when that is higher or the rate is unknown (the source says so).
The roofline fraction of a GEMM is attainable / best measured time; of a
cell, the geomean over its GEMMs. Exclusive GEMMs only: under a context a
co-tenant holds an unknown share of the device.

Pool gaps: the exclusive GEMMs whose best measured kernel reaches less than
`report.pool_gap_fraction` of the roofline, with that kernel's signature,
as input for kernel generation (`write_exports`: pool_gaps.json and
pool_gaps.csv, and weak_spots.json with the cells). They come from the
training side only, round 0's exhaustive seed bench, and never name a shape
of the audit set: kernels generated for the held-out shapes would fit the
shipped pool to the GEMMs the audit and the per-cell gate judge it on.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .features import SIGNATURE_FIELDS as SIGNATURE
from .kernel_identity import signature_label

POOL_GAP_FRACTION = 0.3
SCHEMA = "pool_gaps_v2"
WEAK_SCHEMA = "weak_spots_v2"
TRAINING_SOURCE = "round 0's exhaustive seed bench"
HELD_OUT_SOURCE = "the audit set"
POOL_GAPS_JSON = "pool_gaps.json"
POOL_GAPS_CSV = "pool_gaps.csv"
WEAK_SPOTS_JSON = "weak_spots.json"
EXCLUSIVE = "exclusive"
# The audit's class of a cell below the target -> its weak-spot class.
AUDIT_CLASS = {
    "pool_limited": "rule",
    "identity_limited": "identity",
    "model_limited": "model",
}
CLASSES = ("rule", "identity", "model", "pool")
LEVERS = {
    "rule": "serving rules: refine the rule that leaves out the winners",
    "identity": "kernel identity: member order or launch keys",
    "model": "ranking: retrain the cell (--mode enhance --cells)",
    "pool": "kernels: generate kernels for its pool gaps",
}
CSV_FIELDS = (
    "m",
    "n",
    "k",
    "batch",
    "cell",
    "best_us",
    "attainable_us",
    "fraction",
    "bound",
    "compute_us",
    "memory_us",
    "floor_us",
    "best_signature",
    *SIGNATURE,
)


def _geo(values: Iterable[float]) -> Optional[float]:
    logs = [math.log(v) for v in values if v is not None and v > 0]
    return math.exp(sum(logs) / len(logs)) if logs else None


def best_kernels(
    gemms: Iterable[Mapping[str, Any]], kernels: Iterable[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Per GEMM dict (with candidates, as lib/run_analysis.py `gemm_dicts`
    gives them) its true best: the fastest tested candidate of the pool
    `kernels`, with its signature fields; a GEMM without one is left out."""
    by_sid = {int(k["sol_idx_global"]): k for k in kernels}
    out = []
    for g in gemms:
        tested = [
            c
            for c in g.get("candidates") or []
            if not c.get("is_skip") and int(c["sol_idx_global"]) in by_sid
        ]
        if not tested:
            continue
        best = min(tested, key=lambda c: float(c["us"]))
        if not float(best["us"]) > 0:
            continue
        kernel = by_sid[int(best["sol_idx_global"])]
        out.append(
            {
                "m": int(g["m"]),
                "n": int(g["n"]),
                "k": int(g["k"]),
                "batch": int(g["batch_count"]),
                "cu_budget": int(g.get("cu_budget") or 0),
                "schedule": str(g.get("schedule") or "default"),
                "types": {
                    t: str(g.get(f"{t}_type") or "")
                    for t in ("a", "b", "c", "d", "compute")
                },
                "transA": str(g.get("transA") or ""),
                "transB": str(g.get("transB") or ""),
                "best_us": float(best["us"]),
                "kernel": {f: int(kernel[f]) for f in SIGNATURE},
            }
        )
    return out


def unit_times(row: Mapping[str, Any]) -> Tuple[float, float]:
    """(compute, memory) roofline time in us of one GEMM of `best_kernels` at
    a unit matrix rate and a unit bandwidth, in the units of
    `lib.bench_yaml.roofline_us`: a term is its unit time over the rate, and
    the unit time over the best kernel's time is the rate that kernel
    reached."""
    from .bench_yaml import roofline_us

    shape = (row["m"], row["n"], row["k"], row["batch"])
    hl = {f"{t}_type": v for t, v in row["types"].items()}
    return roofline_us(shape, hl, 1.0, None), roofline_us(shape, hl, None, 1.0)


def is_exclusive(row: Mapping[str, Any]) -> bool:
    return int(row.get("cu_budget") or 0) == 0 and str(
        row.get("schedule") or "default"
    ) in ("default", "")


def roofline(rows: Sequence[Mapping[str, Any]], rates: Any) -> Dict[str, Any]:
    """The roofline of every exclusive row of `best_kernels` (see the module
    docstring) under `rates` (lib.hardware.RooflineRates): the rates used
    with their sources, and per row its compute, memory and floor times,
    the attainable time, the term that sets it (`bound`) and the fraction
    attainable / best."""
    from .grid import cell_key

    ex = [dict(r) for r in rows if is_exclusive(r)]
    if not ex:
        return {"peaks": None, "rows": []}
    units = [unit_times(r) for r in ex]
    best_tflops = max(c / r["best_us"] for (c, _), r in zip(units, ex))
    best_gbs = max(m / r["best_us"] for (_, m), r in zip(units, ex))

    def peak(value: Optional[float], source: str, best: float) -> Tuple[float, str]:
        if value is None:
            return best, f"the best any GEMM reached ({source})"
        if best > value:
            return best, f"the best any GEMM reached, above the rate of {source}"
        return value, source

    tflops, t_src = peak(rates.peak_tflops, rates.peak_note, best_tflops)
    gbs, b_src = peak(rates.bandwidth_gbs, rates.bandwidth_note, best_gbs)
    floor = min(r["best_us"] for r in ex)
    for (unit_c, unit_m), r in zip(units, ex):
        compute = unit_c / tflops
        memory = unit_m / gbs
        attainable = max(compute, memory, floor)
        r.update(
            cell=cell_key(r["m"], r["n"], r["k"], r["batch"]),
            compute_us=compute,
            memory_us=memory,
            floor_us=floor,
            attainable_us=attainable,
            bound=(
                "compute"
                if attainable == compute
                else "memory" if attainable == memory else "latency"
            ),
            fraction=min(1.0, attainable / r["best_us"]),
            best_signature=signature_label(r["kernel"]),
        )
    return {
        "peaks": {
            "tflops": tflops,
            "tflops_source": t_src,
            "bandwidth_gbs": gbs,
            "bandwidth_source": b_src,
            "latency_floor_us": floor,
            "latency_floor_source": "the best time of the fastest exclusive GEMM",
        },
        "rows": ex,
    }


def cell_rooflines(
    rows: Sequence[Mapping[str, Any]], threshold: float = POOL_GAP_FRACTION
) -> Dict[str, Dict[str, Any]]:
    """Per cell of `roofline` rows: the geomean roofline fraction of its
    true bests, its GEMMs, how many are pool gaps (fraction below
    `threshold`) and the terms that bound them."""
    by_cell: Dict[str, List[Mapping[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault(str(r["cell"]), []).append(r)
    out = {}
    for cell, rs in sorted(by_cell.items()):
        bounds: Dict[str, int] = {}
        for r in rs:
            bounds[r["bound"]] = bounds.get(r["bound"], 0) + 1
        out[cell] = {
            "fraction": _geo(r["fraction"] for r in rs),
            "n": len(rs),
            "n_gaps": sum(1 for r in rs if r["fraction"] < threshold),
            "bounds": dict(sorted(bounds.items())),
        }
    return out


def pool_gaps(
    rows: Sequence[Mapping[str, Any]], threshold: float = POOL_GAP_FRACTION
) -> List[Dict[str, Any]]:
    """The `roofline` rows below `threshold`, farthest from their roofline
    first, each with its best kernel's signature."""
    gaps = [r for r in rows if r["fraction"] < threshold]
    gaps.sort(key=lambda r: (r["fraction"], r["m"], r["n"], r["k"], r["batch"]))
    return [
        {
            "m": r["m"],
            "n": r["n"],
            "k": r["k"],
            "batch": r["batch"],
            "cell": r["cell"],
            "best_us": r["best_us"],
            "attainable_us": r["attainable_us"],
            "fraction": r["fraction"],
            "bound": r["bound"],
            "compute_us": r["compute_us"],
            "memory_us": r["memory_us"],
            "floor_us": r["floor_us"],
            "best_signature": r["best_signature"],
            **r["kernel"],
        }
        for r in gaps
    ]


def pool_digest(digests: Mapping[int, str]) -> str:
    """Digest of a kernel pool ({solution: kernel digest}, `lib.audit
    .pool_digests`), whatever indices it serves its kernels under."""
    body = "\n".join(sorted(str(d) for d in digests.values()))
    return hashlib.sha1(body.encode()).hexdigest()[:16]


def exhaustive_best(
    run_dir: Path, audit_dir: Optional[Path] = None, jobs: int = 1
) -> Dict[str, Any]:
    """The true best kernel of every exhaustively benched GEMM of the run, at
    the indices of the pool of the bundle the run ships, from two sources
    (`lib.run_analysis.exhaustive_benches`): {training: round 0's exhaustive
    seed bench without any shape of the audit set `audit_dir`, held_out: the
    audit set when it is complete for that pool}, each {source, why, rows
    (`best_kernels`)}, the training one also `n_dropped_audit`; with the
    arch, the device its library was read with, the library's digest
    (lib/kernel_identity.py) and the pool's (`pool_digest`). Rows are empty,
    with why, where there are none."""
    from . import audit as al
    from . import evaluate as ev
    from . import run_analysis as ra
    from .bank_pool import config_pool

    def empty(why: Optional[str]) -> Dict[str, Any]:
        return {"source": None, "why": why, "rows": []}

    out: Dict[str, Any] = {
        "arch": None,
        "device": None,
        "library_digest": None,
        "pool_digest": None,
        "training": dict(empty(None), n_dropped_audit=0),
        "held_out": empty(None),
    }

    def missing(why: str) -> Dict[str, Any]:
        out["training"]["why"] = out["held_out"]["why"] = why
        return out

    model = ra.final_model(Path(run_dir), want_bytes=False, staged=True)
    if model is None or model.library is None:
        return missing("no bundle of the run names its kernel library")
    out.update(arch=model.arch, device=model.library_device)
    try:
        kernels = config_pool(
            model.library_cfg, *model.library, model.library_device
        ).kernels
    except (OSError, ValueError, RuntimeError) as e:
        return missing(f"the kernel library is unreadable ({type(e).__name__}: {e})")
    digests = al.pool_digests(kernels)
    identity = ra.library_identity(model)
    out.update(
        library_digest=identity.digest if identity is not None else None,
        pool_digest=pool_digest(digests),
    )
    pool = ev.pool_from_kernels(kernels)

    def rows(dirs: Sequence[Any]) -> List[Dict[str, Any]]:
        gemms, _, _ = ev.restrict_to_pool(
            ra.gemm_dicts(dirs, jobs=jobs, identity=identity), pool
        )
        return best_kernels(gemms, kernels)

    name, dirs, _ = ra.exhaustive_benches(Path(run_dir), None, digests)
    if not dirs:
        none = f"the run has no exhaustive training bench ({TRAINING_SOURCE})"
        out["training"]["why"] = none
    else:
        held = al.read_shapes(al.shapes_yaml(audit_dir)) if audit_dir else set()
        mine = rows(dirs)
        kept = [r for r in mine if (r["m"], r["n"], r["k"], r["batch"]) not in held]
        out["training"].update(
            source=name, rows=kept, n_dropped_audit=len(mine) - len(kept)
        )
    if audit_dir is None:
        out["held_out"]["why"] = "no audit set is configured (audit.dir)"
        return out
    name, dirs, why = ra.exhaustive_benches(Path(run_dir), audit_dir, digests)
    if name != HELD_OUT_SOURCE:
        out["held_out"]["why"] = why or "the audit set cannot be read"
        return out
    out["held_out"].update(source=name, rows=rows(dirs))
    return out


def analyze(
    run_dir: Path,
    cfg: Optional[Mapping[str, Any]],
    *,
    audit_dir: Optional[Path] = None,
    jobs: int = 1,
    threshold: float = POOL_GAP_FRACTION,
    peak_tflops: Optional[float] = None,
    peak_bandwidth_gbs: Optional[float] = None,
) -> Dict[str, Any]:
    """The roofline (`roofline`) of the run's exhaustively benched exclusive
    GEMMs (`exhaustive_best`, both sources under one set of peaks): per cell
    (`cell_rooflines`) over the held-out audit set when it is complete, else
    over the training side (`cells_source`, `cells_held_out`), and the pool
    gaps below `threshold` (`pool_gaps`) of the training side only (`source`,
    `why`, `n_gemms`). The rates are `peak_tflops` and `peak_bandwidth_gbs`
    when given, else the config `cfg`'s `bench.adaptive_long.peak_tflops`
    and `bandwidth_gbs`, else those of the device the bundle's library was
    read with (`lib.hardware.roofline_rates`)."""
    from . import hardware as hw

    cfg = dict(cfg or {})
    best = exhaustive_best(Path(run_dir), audit_dir, jobs)
    train, held = best["training"], best["held_out"]
    out: Dict[str, Any] = {
        "source": train["source"],
        "why": train["why"],
        "held_out": False,
        "n_dropped_audit": train.get("n_dropped_audit", 0),
        "cells_source": None,
        "cells_held_out": None,
        "cells_why": held["why"],
        "library_digest": best.get("library_digest"),
        "pool_digest": best.get("pool_digest"),
        "threshold": float(threshold),
        "peaks": None,
        "layout": None,
        "n_gemms": 0,
        "n_cell_gemms": 0,
        "cells": {},
        "gaps": [],
    }
    rows = [dict(r, source="training") for r in train["rows"] if is_exclusive(r)]
    rows += [dict(r, source="held_out") for r in held["rows"] if is_exclusive(r)]
    if not rows:
        return out
    first = rows[0]
    hl = {**(cfg.get("hipblaslt") or {})}
    hl.update({f"{t}_type": v for t, v in first["types"].items() if v})
    arch = str(best.get("arch") or cfg.get("arch") or "")
    out["layout"] = {
        "arch": arch,
        "transA": first["transA"],
        "transB": first["transB"],
        **{f"{t}_type": v for t, v in first["types"].items()},
    }
    device = hw.roofline_rates(
        dict(cfg, arch=arch, hipblaslt=hl), int(best.get("device") or 0)
    )
    given = (cfg.get("bench") or {}).get("adaptive_long") or {}

    def rate(
        option: Optional[float], key: str, flag: str, value: Optional[float], note: str
    ) -> Tuple[Optional[float], str]:
        if option is not None:
            return float(option), flag
        if given.get(key) is not None:
            return float(given[key]), f"bench.adaptive_long.{key}"
        if value is not None:
            return float(value), f"the device: {note}"
        return None, f"unknown: {note}"

    tflops, t_note = rate(
        peak_tflops,
        "peak_tflops",
        "--peak-tflops",
        device.peak_tflops,
        device.peak_note,
    )
    gbs, b_note = rate(
        peak_bandwidth_gbs,
        "bandwidth_gbs",
        "--peak-bandwidth-gbs",
        device.bandwidth_gbs,
        device.bandwidth_note,
    )
    roof = roofline(rows, hw.RooflineRates(tflops, t_note, gbs, b_note))
    mine = [r for r in roof["rows"] if r["source"] == "training"]
    theirs = [r for r in roof["rows"] if r["source"] == "held_out"]
    by_cell = theirs or mine
    out.update(
        peaks=roof["peaks"],
        n_gemms=len(mine),
        n_cell_gemms=len(by_cell),
        cells=cell_rooflines(by_cell, threshold),
        cells_source=held["source"] if theirs else train["source"],
        cells_held_out=bool(theirs),
        gaps=pool_gaps(mine, threshold),
    )
    return out


# ── cells ────────────────────────────────────────────────────────────────────


def origami_cells(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per cell, the model's and Origami's geomean selection efficiency on
    the problems where both have one (stage 7's `per_problem` rows of the
    audit set: m, n, k, batch, model_sel_eff, origami_sel_eff)."""
    from .grid import cell_key

    by_cell: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        a, b = r.get("model_sel_eff"), r.get("origami_sel_eff")
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            continue
        if a <= 0 or b <= 0:
            continue
        cell = cell_key(int(r["m"]), int(r["n"]), int(r["k"]), int(r["batch"]))
        by_cell.setdefault(cell, []).append((float(a), float(b)))
    return {
        c: {
            "model": _geo(a for a, _ in v),
            "origami": _geo(b for _, b in v),
            "n": len(v),
        }
        for c, v in sorted(by_cell.items())
    }


def classes_of(
    audit_class: Optional[str], roof: Optional[Mapping[str, Any]], threshold: float
) -> List[str]:
    """The weak-spot classes of a cell: its audit class below the target
    (`AUDIT_CLASS`), then `pool` when its roofline fraction is below
    `threshold`."""
    out = []
    if audit_class in AUDIT_CLASS:
        out.append(AUDIT_CLASS[audit_class])
    frac = (roof or {}).get("fraction")
    if frac is not None and frac < threshold:
        out.append("pool")
    return out


def _group_of(rec: Mapping[str, Any], label: str) -> Tuple[Optional[dict], str]:
    """(group, protocol) of `label` as the report shows it: v2 when the
    record's v2 evaluated the group, else v1."""
    from . import audit as al

    g2 = al.evaluated_group(al.protocol_record(rec, "v2"), label)
    if g2 is not None:
        return g2, "v2"
    return al.evaluated_group(rec, label), "v1"


def weak_cells(
    rec: Optional[Mapping[str, Any]],
    previous: Optional[Mapping[str, Any]] = None,
    *,
    roof_cells: Optional[Mapping[str, Mapping[str, Any]]] = None,
    origami: Optional[Mapping[str, Mapping[str, Any]]] = None,
    threshold: float = POOL_GAP_FRACTION,
    settings: Optional[Mapping[str, Any]] = None,
    bootstrap: int = 2000,
    previous_why: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """One row per (group, cell) of the audit record `rec` (under v2 for a
    group its v2 evaluated): the cell's classes (`classes_of`, the roofline
    of its exclusive cell from `roof_cells`), its selection efficiency and
    ceilings, its paired change from `previous` (`lib.audit.paired_change`,
    interval within the cell) with the family-wise p of the per-cell gate's
    cell test (`lib.audit.cell_gate` under the gate `settings`), `moved`
    `regressed` when that test flags the cell, Origami's numbers on the same
    problems (`origami`, exclusive only), its picks below the gate's tail
    floor and cell floor, and `weak`: a class, a regression or a pick below
    the tail floor. A count the record cannot give (a group without
    per-GEMM results, as in records written before them) is None, and a
    change that could not be paired or judged says why (`previous_why`; the
    argument says why there is no `previous`). Without `rec`, the exclusive
    cells of `roof_cells` with their roofline alone."""
    from . import audit as al

    st = {**al.CELL_GATE_DEFAULTS, **dict(settings or {})}
    cell_floor, tail_floor = float(st["cell_floor"]), float(st["tail_floor"])
    roof_cells = roof_cells or {}
    origami = origami or {}
    rows: List[Dict[str, Any]] = []
    labels: List[str] = []
    gate: Dict[str, Any] = {}
    untested: Optional[str] = None
    if rec is not None:
        labels = list(
            dict.fromkeys(
                [
                    *(rec.get("groups") or {}),
                    *(al.protocol_record(rec, "v2").get("groups") or {}),
                ]
            )
        )
        if previous is not None:
            judged_all = al.cell_gate(previous, rec, st, bootstrap)
            gate = judged_all.get("groups") or {}
            if judged_all.get("cell_rules") != "on":
                untested = f"the cell test was {judged_all.get('cell_rules')}"
    for label in labels:
        g, proto = _group_of(rec, label)
        if g is None:
            continue
        change, why = None, previous_why or "no previous candidate to pair with"
        if previous is not None:
            change, why, _ = al.paired_change(previous, rec, label, bootstrap)
        judged = gate.get(label) or {}
        p_of = judged.get("p") or {}
        flagged = {c["cell"] for c in judged.get("regressed") or []}
        has_gemms = "per_gemm" in g
        per: Dict[str, List[float]] = {}
        for r in g.get("per_gemm") or []:
            if r[4] is not None:
                per.setdefault(_cell_of(r), []).append(float(r[4]))
        for cell, c in sorted((g.get("per_cell") or {}).items()):
            roof = roof_cells.get(cell) if label == EXCLUSIVE else None
            classes = classes_of(c.get("class"), roof, threshold)
            moved = ((change or {}).get("per_cell") or {}).get(cell)
            prev = None
            if moved:
                prev = {k: moved.get(k) for k in ("ratio", "ratio_ci", "n", "before")}
                prev["p"] = p_of.get(cell)
                prev["moved"] = "regressed" if cell in flagged else None
            values = per.get(cell, [])
            row = {
                "group": label,
                "cell": cell,
                "protocol": proto,
                "classes": classes,
                "class": "+".join(classes) or "passing",
                "n": c.get("n"),
                "sel_eff": c.get("sel_eff"),
                "sel_eff_ci": c.get("sel_eff_ci"),
                "ceiling_rules": c.get("ceiling_rules"),
                "ceiling_served": c.get("ceiling_served"),
                "previous": prev,
                "previous_protocol": (change or {}).get("protocol"),
                "previous_why": untested if change is not None else why,
                "origami": origami.get(cell) if label == EXCLUSIVE else None,
                "n_tail": (
                    sum(1 for v in values if v < tail_floor) if has_gemms else None
                ),
                "n_below_floor": (
                    sum(1 for v in values if v < cell_floor) if has_gemms else None
                ),
                "roofline": (roof or {}).get("fraction"),
                "n_gaps": (roof or {}).get("n_gaps"),
                "levers": [LEVERS[k] for k in classes],
            }
            row["weak"] = bool(
                classes or row["n_tail"] or (prev or {}).get("moved") == "regressed"
            )
            rows.append(row)
    if rec is None:
        for cell, roof in sorted(roof_cells.items()):
            classes = classes_of(None, roof, threshold)
            rows.append(
                {
                    "group": EXCLUSIVE,
                    "cell": cell,
                    "protocol": None,
                    "classes": classes,
                    "class": "+".join(classes) or "passing",
                    "n": roof.get("n"),
                    "sel_eff": None,
                    "sel_eff_ci": None,
                    "ceiling_rules": None,
                    "ceiling_served": None,
                    "previous": None,
                    "previous_protocol": None,
                    "previous_why": "no audit evaluation of the bundle",
                    "origami": origami.get(cell),
                    "n_tail": None,
                    "n_below_floor": None,
                    "roofline": roof.get("fraction"),
                    "n_gaps": roof.get("n_gaps"),
                    "levers": [LEVERS[k] for k in classes],
                    "weak": bool(classes),
                }
            )
    return rows


def _cell_of(r: Sequence[Any]) -> str:
    from .grid import cell_key

    return cell_key(int(r[0]), int(r[1]), int(r[2]), int(r[3]))


def summary(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Per group of `weak_cells` rows: its cells, how many have each class
    and none, how many regressed against the previous bundle (None, with
    `regressed_why`, when the group was not paired with it or the cell test
    could not judge it), and its picks below the tail floor and below the
    cell floor (None when the record holds no per-GEMM results for the
    group)."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        s = out.setdefault(
            r["group"],
            {
                "group": r["group"],
                "n_cells": 0,
                "passing": 0,
                **{k: 0 for k in CLASSES},
                "regressed": 0,
                "regressed_why": None,
                "n_tail": 0,
                "n_below_floor": 0,
            },
        )
        s["n_cells"] += 1
        s["passing"] += 0 if r["classes"] else 1
        for k in r["classes"]:
            s[k] += 1
        if r.get("previous_why"):
            s["regressed"], s["regressed_why"] = None, r["previous_why"]
        elif s["regressed"] is not None:
            s["regressed"] += int((r.get("previous") or {}).get("moved") == "regressed")
        for k in ("n_tail", "n_below_floor"):
            if r.get(k) is None:
                s[k] = None
            elif s[k] is not None:
                s[k] += int(r[k])
    order = sorted(out, key=lambda g: (g != EXCLUSIVE, g))
    return [out[g] for g in order]


# ── exports ──────────────────────────────────────────────────────────────────


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def write_exports(
    out_dir: Path,
    analysis: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]] = (),
    *,
    run: str = "",
    bundle: Optional[str] = None,
) -> List[Path]:
    """pool_gaps.json (where its gaps come from, never the audit set; the
    digests of the library and the pool they were measured on; the layout,
    peaks, threshold and every pool gap), pool_gaps.csv (one row per pool
    gap) and weak_spots.json (the cells of `weak_cells`, without Origami's
    numbers, and their `summary`, with where the cells' roofline comes from)
    in `out_dir`, each written atomically; returns the paths."""
    from .fs import atomic_write_text

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gaps = list(analysis.get("gaps") or [])
    doc = {
        "schema": SCHEMA,
        "run": run,
        "bundle": bundle,
        "source": analysis.get("source"),
        "held_out": False,
        "why": analysis.get("why"),
        "n_dropped_audit": analysis.get("n_dropped_audit"),
        "library_digest": analysis.get("library_digest"),
        "pool_digest": analysis.get("pool_digest"),
        "layout": analysis.get("layout"),
        "threshold": analysis.get("threshold"),
        "peaks": analysis.get("peaks"),
        "n_gemms": analysis.get("n_gemms"),
        "n_gaps": len(gaps),
        "gaps": gaps,
    }
    paths = [
        out_dir / POOL_GAPS_JSON,
        out_dir / POOL_GAPS_CSV,
        out_dir / WEAK_SPOTS_JSON,
    ]
    atomic_write_text(paths[0], json.dumps(_json_ready(doc), indent=2) + "\n")
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
    w.writeheader()
    for g in gaps:
        w.writerow(
            {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in g.items()}
        )
    atomic_write_text(paths[1], buf.getvalue())
    weak = {
        "schema": WEAK_SCHEMA,
        "run": run,
        "bundle": bundle,
        "threshold": analysis.get("threshold"),
        "roofline_source": analysis.get("cells_source"),
        "roofline_held_out": analysis.get("cells_held_out"),
        "summary": summary(cells),
        "cells": [{k: v for k, v in c.items() if k != "origami"} for c in cells],
    }
    atomic_write_text(paths[2], json.dumps(_json_ready(weak), indent=2) + "\n")
    return paths
