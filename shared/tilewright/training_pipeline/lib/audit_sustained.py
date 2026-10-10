# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The measurement behind the audit's sustained composite: the power-capped
exclusive GEMMs of the audit set timed again in long self-soaked windows,
in `<audit>/v2/sustained/`.

A long or compute-bound GEMM run back to back holds the device at its power
limit, and the limit slows some kernels more than others (lib/sustained.py
says the same of the training labels). Protocol v2's windows of a few
milliseconds end before the clock settles, so for such a GEMM they compare
kernels at clocks it does not run at for long. The sustained composite
(lib/audit.py) scores a power-capped exclusive GEMM by this measurement,
in which every kernel runs alone until it holds the limit before it is
timed, and every other GEMM by protocol v2.

Strata, by a GEMM's v2 winner (the smallest v2 median of a kernel of the
pool) and its rate (2 M N K batch over the winner's latency):
  compute_bound  at least `compute_bound_min_us`, at a rate of at least
                 `compute_bound_tflops`
  very_long      not compute-bound, at least `very_long_us`
  long           neither, at least `min_us`
These are the power-capped GEMMs. `compute_bound_tflops`, unless given, is
`compute_bound_fraction` of the bench device's peak dense matrix rate for
the config's data types (lib/hardware.py `roofline_rates`), read when the
measurement is first planned; without a peak no GEMM is compute-bound.

Sets. The set of a power-capped GEMM: the kernels whose v2 median is within
`set_tolerance` of its winner's (a compute-bound GEMM's within
`compute_bound_set_tolerance`), fastest first, at most `set_max`
(`compute_bound_set_max`), then the pick of every model registered with the
v2 measurement that the exhaustive pass benched on it.

Bench. Protocol v2's machinery (lib/audit_v2.py `Bench`) with this
measurement's settings: parts of at most `part_gemms` GEMMs (a seeded
shuffle dealt round robin), benched one after another so that the device
stays heat-soaked, each `repetitions` times in separate stage03 processes,
the GEMM order and each set shuffled per repetition; each GEMM's line is
its v2 line timed adaptively as the config times it, with the windows
`warmup_time`, `measure_time` and `max_measure_time` (ms): each kernel runs
alone long enough to reach the power limit, then is timed. The v2 canary
is read before and after, medians take `min_readings` readings, a wide
spread is benched again, and a line that hangs is benched once more and
then dropped, quarantined under this measurement's own protocol
(audit_sustained), as in protocol v2.

Increments. A plan after the first plans the GEMMs that the current v2
measurement makes power-capped and no generation has planned, and the
planned ones whose set lacks a registered pick, each with its whole set:
its previous set, its near-winners now and every pick, so that a winner
and the picks are always measured together. A GEMM once planned stays
power-capped (its stratum as planned when the v2 winner no longer makes it
one).

Layout: protocol v2's, in `<audit>/v2/sustained/`: its manifest names
`regime` sustained and records the settings (with the compute-bound rate
resolved) and where that rate came from (`compute_bound`); each plan entry
records its stratum and v2 winner.
"""
from __future__ import annotations

import math
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from . import audit_v2 as v2
from . import kernel_identity as ki
from . import sustained as sus
from .bench_yaml import adaptive_line
from .fs import write_json
from .planner import stable_rng

DIRNAME = "sustained"
REGIME = "sustained"
EXCLUSIVE = "exclusive"
DEFAULTS: Dict[str, Any] = {
    **v2.DEFAULTS,
    "repetitions": 3,
    "min_readings": 2,
    "sacrificial_kernel": False,
    "set_tolerance": 0.08,
    "set_max": 10,
    "compute_bound_set_tolerance": 0.15,
    "compute_bound_set_max": 12,
    "min_us": 100.0,
    "very_long_us": 1000.0,
    "compute_bound_min_us": 100.0,
    "compute_bound_fraction": 0.3,
    "compute_bound_tflops": None,
    "warmup_time": 250.0,
    "measure_time": 300.0,
    "max_measure_time": 400.0,
    "part_gemms": 24,
}
STRATA = ("compute_bound", "very_long", "long")
Shape = Tuple[int, int, int, int]


def sustained_dir(audit_dir: Path) -> Path:
    return v2.v2_dir(audit_dir) / DIRNAME


def compute_bound_rate(
    st: Mapping[str, Any], peak_tflops: Optional[float], peak_note: str = ""
) -> Tuple[Optional[float], str]:
    """(the rate in TFLOP/s from which a GEMM is compute-bound, where it
    comes from): `compute_bound_tflops` when given, else
    `compute_bound_fraction` of the device's peak dense matrix rate
    `peak_tflops`; None without either."""
    given = st.get("compute_bound_tflops")
    if given is not None:
        return float(given), "given (compute_bound_tflops)"
    if peak_tflops:
        frac = float(st["compute_bound_fraction"])
        return frac * float(peak_tflops), (
            f"{frac:g} of the bench device's peak dense matrix rate ({peak_note})"
        )
    return None, (
        f"none: the bench device's peak dense matrix rate is unknown ({peak_note}), "
        f"so no GEMM is compute-bound"
    )


def stratum(shape: Shape, winner_us: float, st: Mapping[str, Any]) -> str:
    """The stratum of an exclusive GEMM whose v2 winner takes `winner_us`
    (see the module docstring), "" when it is not power-capped."""
    rate = st.get("compute_bound_tflops")
    if (
        rate is not None
        and winner_us >= float(st["compute_bound_min_us"])
        and sus.tflops(shape, winner_us) >= float(rate)
    ):
        return "compute_bound"
    if winner_us < float(st["min_us"]):
        return ""
    return "very_long" if winner_us >= float(st["very_long_us"]) else "long"


def near_winners(
    median: Mapping[int, float], stratum_: str, st: Mapping[str, Any]
) -> List[int]:
    """The kernels of a power-capped GEMM's medians within its stratum's
    tolerance of the fastest, fastest first, at most its stratum's count."""
    cb = stratum_ == "compute_bound"
    tol = float(st["compute_bound_set_tolerance" if cb else "set_tolerance"])
    cap = int(st["compute_bound_set_max" if cb else "set_max"])
    fastest = min(median.values())
    rows = sorted((v, int(k)) for k, v in median.items() if v <= fastest * (1 + tol))
    return [k for _, k in rows[: max(1, cap)]]


def current_entries(d: Path) -> Dict[Shape, Dict[str, Any]]:
    """{shape: plan entry} of the latest generation planning each exclusive
    GEMM of the measurement in `d`."""
    out: Dict[Shape, Dict[str, Any]] = {}
    for p in v2.plans(d):
        for part in (p.get("groups") or {}).get(EXCLUSIVE) or []:
            for e in part:
                out[tuple(e["shape"])] = e
    return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def plan(
    d: Path,
    *,
    v2d: Path,
    gemms: Mapping[Shape, Mapping[str, Any]],
    measured: Mapping[Shape, Any],
    config: Mapping[str, Any],
    settings: Optional[Mapping[str, Any]] = None,
    peak_tflops: Optional[float] = None,
    peak_note: str = "",
    identity: Optional[ki.LibraryIdentity] = None,
) -> Optional[int]:
    """Plan the generation of the sustained measurement in `d` that the v2
    measurement in `v2d` asks for now (see the module docstring). `gemms`
    holds the exhaustive exclusive GEMMs ({shape: GEMM}), `measured` their
    v2 measurements over the pool's kernels ({shape: lib/audit_v2.py
    Measurement}, lib/audit.py `pool_medians`) and `config` the run config,
    whose adaptive timing the lines take. The first plan writes the manifest
    with `settings` over DEFAULTS, the compute-bound rate resolved
    (`compute_bound_rate`) and the v2 measurement's canary and bench setup;
    with `identity`, the library the indices name, kernel ids are recorded
    and earlier records must name the same kernels. Returns the new
    generation's index, or None when nothing needs a bench (the manifest is
    written either way: a measurement that plans no GEMM says that none is
    power-capped)."""
    d = Path(d)
    m2 = v2.manifest(v2d)
    if m2 is None:
        raise v2.V2Error(f"no v2 measurement in {v2d} to plan a sustained one from")
    if identity is not None:
        v2.require_same_build(d, identity)
    m = v2.manifest(d)
    if m is None:
        st = {**DEFAULTS, **dict(settings or {})}
        rate, note = compute_bound_rate(st, peak_tflops, peak_note)
        st["compute_bound_tflops"] = rate
        m = {
            "schema": v2.SCHEMA,
            "regime": REGIME,
            "created": _now(),
            "settings": st,
            "compute_bound": note,
            "canary": dict(m2["canary"]),
            "bench": dict(m2.get("bench") or {}),
            "generations": [],
        }
    else:
        m["bench"] = {**dict(m.get("bench") or {}), **dict(m2.get("bench") or {})}
    st = v2.settings_of(m)
    picks: Dict[Shape, Dict[str, int]] = {}
    for name, rec in sorted(v2.models(v2d).items()):
        for (label, s), sid in v2.picks_of(rec).items():
            if label == EXCLUSIVE:
                picks.setdefault(s, {})[name] = int(sid)
    lines = {s: e["line"] for s, e in current_entries(v2d).items()}
    planned = current_entries(d)
    entries: List[Dict[str, Any]] = []
    for s in sorted(measured):
        mm, g, prev = measured[s], gemms.get(s), planned.get(s)
        if g is None or mm is None or mm.unplaced or not mm.median or s not in lines:
            continue
        win = min(mm.median, key=lambda k: (mm.median[k], k))
        cls = stratum(s, mm.median[win], st) or str((prev or {}).get("stratum") or "")
        if not cls:
            continue
        rows = {int(c["sol_idx_global"]) for c in g["candidates"]}
        mine = {n: k for n, k in (picks.get(s) or {}).items() if k in rows}
        if prev is not None and set(mine.values()) <= {int(k) for k in prev["kernels"]}:
            continue
        near = near_winners(mm.median, cls, st)
        kernels = list(
            dict.fromkeys(
                [*(int(k) for k in (prev or {}).get("kernels") or []), *near]
                + list(mine.values())
            )
        )
        line = sus.sustained_line(adaptive_line(lines[s], config), kernels, st)
        entries.append(
            {
                "shape": list(s),
                "stratum": cls,
                "v2_winner": [int(win), float(mm.median[win])],
                "v2_generation": int(mm.generation),
                "kernels": kernels,
                "candidates": near,
                "picks": dict(sorted(mine.items())),
                "line": line,
            }
        )
    d.mkdir(parents=True, exist_ok=True)
    if not entries:
        write_json(d / v2.MANIFEST, m)
        return None
    index = len(m["generations"])
    gd = v2.gen_dir(d, index)
    if gd.exists():
        shutil.rmtree(gd)
    order = list(entries)
    stable_rng(st["seed"], "sustained", index).shuffle(order)
    n = math.ceil(len(order) / max(1, int(st["part_gemms"])))
    doc: Dict[str, Any] = {
        "index": index,
        "regime": REGIME,
        "models": sorted(v2.models(v2d)),
        "groups": {EXCLUSIVE: [order[i::n] for i in range(n)]},
    }
    if identity is not None:
        named = {int(m["canary"]["kernel"])}
        named.update(int(k) for e in entries for k in e["kernels"])
        doc[ki.RECORD_KEY] = ki.record(identity, named)
    gd.mkdir(parents=True)
    write_json(gd / "plan.json", doc)
    m["generations"].append(
        {
            "index": index,
            "created": _now(),
            "n_gemms": len(entries),
            "n_parts": n,
            "strata": dict(Counter(e["stratum"] for e in entries)),
        }
    )
    write_json(d / v2.MANIFEST, m)
    return index
