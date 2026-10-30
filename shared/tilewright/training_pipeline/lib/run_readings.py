# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Report-only readings of a bundle on held-out GEMMs, without and with the
alignment veto of the run's config (deploy.alignment_veto): stage04b's on
the GEMMs a round holds out (`readings.json` in its out dir) and stage05's
on its validation fold (`readings.json`, summarized in metrics.json
`readings`), the final retrain's included. No decision reads them; they are
reported as measured and are not validated against another record.

Per GEMM a row (`gemm_row`): the shape, the execution context, the winner's
and the pick's latency, the selection efficiency and whether the pick is
skip-screened (its only reading is the bench's warm-up run). Per set of rows
(`summarize`):

  geomean       of the selection efficiency over the GEMMs with a pick;
  rw            the runtime-weighted efficiency: summed winner latency over
                summed pick latency;
  long_decile   the geomean over the tenth of those GEMMs with the longest
                winner latency (at least one GEMM);
  below         the picks below each of TAIL_FLOORS, timed picks and
                skip-screened picks apart.

`bundle_readings` evaluates a bundle (lib/evaluate.py, exactly as deployed)
without the veto and, when the config sets one and the bundle is MLREC_v3,
with it, and names the GEMMs whose pick the veto changes.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import evaluate as ev
from . import feasibility as feas

READINGS_FILE = "readings.json"
SCHEMA = "run_readings_v1"
TAIL_FLOORS = (0.8, 0.5)
# The long decile: the 1 / LONG_PARTS of the GEMMs with the longest winners.
LONG_PARTS = 10
STATUS = "reported, unvalidated"


def gemm_row(result: Any, gemm: Mapping[str, Any]) -> Dict[str, Any]:
    """The reading of one GEMM's deployed result (`ev.GemmResult`) and its
    GEMM (the enriched candidates, for the skip-screened flag)."""
    ctx = ev.context_of(gemm)
    se = result.sel_eff if result is not None else None
    pick = result.pick_sol_idx if result is not None else None
    timed, screened = ev.measurements(gemm)
    return {
        "m": int(result.m) if result is not None else None,
        "n": int(result.n) if result is not None else None,
        "k": int(result.k) if result is not None else None,
        "batch": int(result.batch) if result is not None else None,
        "cu_budget": getattr(ctx, "cu_budget", None),
        "schedule": getattr(ctx, "schedule", None),
        "cell": result.model_cell if result is not None else None,
        "winner_us": float(gemm.get("winner_us", 0) or 0),
        "pick_sol_idx": pick,
        "pick_us": result.pick_us if result is not None else None,
        "sel_eff": se,
        "screened": bool(pick is not None and pick in screened and pick not in timed),
        "served_by": result.served_by if result is not None else None,
        "miss": result.miss if result is not None else None,
    }


def _geomean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The readings of `rows` (`gemm_row`); a GEMM without a pick, or
    without a positive winner or pick latency, counts in `n_gemms` only."""
    ok = [
        r
        for r in rows
        if r.get("sel_eff") is not None
        and float(r.get("winner_us") or 0) > 0
        and float(r.get("pick_us") or 0) > 0
    ]
    effs = [float(r["sel_eff"]) for r in ok]
    win = sum(float(r["winner_us"]) for r in ok)
    pick = sum(float(r["pick_us"]) for r in ok)
    longest = sorted(ok, key=lambda r: float(r["winner_us"]), reverse=True)
    n_long = max(1, -(-len(ok) // LONG_PARTS)) if ok else 0
    long_rows = longest[:n_long]
    below = {}
    for floor in TAIL_FLOORS:
        low = [r for r in ok if float(r["sel_eff"]) < floor]
        below[f"{floor:g}"] = {
            "timed": sum(1 for r in low if not r.get("screened")),
            "screened": sum(1 for r in low if r.get("screened")),
        }
    return {
        "n_gemms": len(rows),
        "n_evaluated": len(ok),
        "geomean": _geomean(effs),
        "rw": (win / pick) if pick > 0 else None,
        "long_decile": {
            "n": len(long_rows),
            "winner_us_min": (float(long_rows[-1]["winner_us"]) if long_rows else None),
            "geomean": _geomean([float(r["sel_eff"]) for r in long_rows]),
        },
        "below": below,
    }


def bundle_readings(
    bundle: Mapping[str, Any],
    gemms: Sequence[Mapping[str, Any]],
    cfg: Optional[Mapping[str, Any]],
    evaluate: Callable[[Mapping[str, Any]], Sequence[Any]],
    plain: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """{status, without_veto, with_veto, veto, n_changed, rows} of `bundle`
    on `gemms`: `evaluate(bundle)` gives the deployed results in the order
    of `gemms` (`plain`, when given, are those of the bundle as it is, which
    are then not evaluated again); with the alignment veto of `cfg`
    (lib/feasibility.py `deployed_alignment_veto`) and an MLREC_v3 bundle,
    the bundle is evaluated with it too. Each row holds the GEMM's reading
    without the veto and, beside it, its pick and selection efficiency with
    it."""
    serving = dict(bundle.get("serving") or {})
    serving.pop("alignment_veto", None)
    base = dict(bundle, serving=serving)
    results = list(plain) if plain is not None else list(evaluate(base))
    rows = [gemm_row(r, g) for r, g in zip(results, gemms)]
    veto = feas.deployed_alignment_veto(cfg)
    if str(bundle.get("catalog") or "v2") != "v3":
        veto = None
    out: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "veto": veto,
        "without_veto": summarize(rows),
        "with_veto": None,
        "n_changed": None,
    }
    if veto is not None:
        vetoed = list(evaluate(dict(base, serving=dict(serving, alignment_veto=veto))))
        vrows = [gemm_row(r, g) for r, g in zip(vetoed, gemms)]
        out["with_veto"] = summarize(vrows)
        out["n_changed"] = sum(
            1 for a, b in zip(rows, vrows) if a["pick_sol_idx"] != b["pick_sol_idx"]
        )
        for a, b in zip(rows, vrows):
            a["veto_pick_sol_idx"] = b["pick_sol_idx"]
            a["veto_pick_us"] = b["pick_us"]
            a["veto_sel_eff"] = b["sel_eff"]
            a["veto_screened"] = b["screened"]
    out["rows"] = rows
    return out


def brief(readings: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """`readings` without its per-GEMM rows (for metrics.json)."""
    if readings is None:
        return None
    return {k: v for k, v in readings.items() if k != "rows"}


def failed(why: str) -> Dict[str, Any]:
    """The record of readings that could not be made (never fatal)."""
    return {"schema": SCHEMA, "status": STATUS, "error": why}


__all__: List[str] = [
    "LONG_PARTS",
    "READINGS_FILE",
    "SCHEMA",
    "STATUS",
    "TAIL_FLOORS",
    "brief",
    "bundle_readings",
    "failed",
    "gemm_row",
    "summarize",
]
