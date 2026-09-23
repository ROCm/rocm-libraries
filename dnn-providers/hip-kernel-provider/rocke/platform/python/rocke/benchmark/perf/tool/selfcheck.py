# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Self-check - advisory improve/regress verdict between two runs.

The decision layer on top of `report.diff`: it takes a previous and a current
record of the same kernel and says *regressed / improved / within_noise* - but only
flags a change that clears the **noise floor**, `max(threshold, noise_k * spread)`.
Both metrics are lower-is-better (`busy_cycles`, or wall `ms_median` as fallback),
so a positive percent change is slower.

Why a noise floor and not a bare threshold: one run wobbles even on the cycle
metric, so `aggregate` records a spread; gating on `noise_k * spread` stops the tool
crying wolf when a "slowdown" is smaller than the run-to-run wobble. The library
only returns a verdict and never raises on a regression; the shipped CLI
(`profile`/`compare`) turns a `regressed` verdict into exit status 1, so other
consumers can adopt their own policy.

Defaults: threshold 5% (fraction 0.05), noise_k 3. Stdlib only.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from rocke.benchmark.perf import report as _report
from . import store as _store

DEFAULT_THRESHOLD = 0.05  # 5% relative change
DEFAULT_NOISE_K = 3.0  # flag only past k * observed spread


def compare(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    noise_k: float = DEFAULT_NOISE_K,
) -> dict:
    """Advisory verdict comparing `current` against `previous`.

    Returns a dict with `verdict` in {regressed, improved, within_noise, unknown},
    the metric used, the percent change, the noise floor that gated it, and the full
    `report.diff` (panel included) so a caller can show *why*. The noise floor uses
    the larger spread from the baseline and current records.
    """
    d = _report.diff(previous, current)
    pct = d.get("pct_change")
    spread = d.get("spread_pct") or 0.0
    floor_pct = max(threshold * 100.0, noise_k * spread)

    if d.get("metric_mismatch") or pct is None:
        verdict = "unknown"
    elif abs(pct) <= floor_pct:
        verdict = "within_noise"
    elif pct > 0:
        verdict = "regressed"
    else:
        verdict = "improved"

    return {
        "verdict": verdict,
        "metric": d.get("metric"),
        "pct_change": pct,
        "threshold_pct": threshold * 100.0,
        "spread_pct": d.get("spread_pct"),
        "floor_pct": floor_pct,
        "diff": d,
    }


def check_history(
    records: Sequence[Mapping[str, Any]],
    identity: tuple,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    noise_k: float = DEFAULT_NOISE_K,
) -> dict:
    """Compare the two most recent runs of `identity` in a record history.

    `records` is what `store.load()` returns (append order). Returns a
    `no_baseline` verdict if fewer than two runs exist for that identity.
    """
    seq = _store.records_for(records, identity)
    ident = {
        "arch": identity[0],
        "op": identity[1],
        "kernel_name": identity[2],
        "shape": identity[3],
    }
    if len(seq) < 2:
        return {"verdict": "no_baseline", "n_runs": len(seq), "identity": ident}
    return compare(seq[-2], seq[-1], threshold=threshold, noise_k=noise_k)


_TAG = {
    "regressed": "REGRESSED",
    "improved": "improved",
    "within_noise": "within noise",
    "unknown": "unknown",
    "no_baseline": "no baseline",
}


def format_result(result: Mapping[str, Any]) -> str:
    """Human-readable one-line-plus-panel view of a `compare`/`check_history` result."""
    verdict = result.get("verdict", "unknown")
    tag = _TAG.get(verdict, verdict)
    if verdict == "no_baseline":
        ident = result.get("identity", {})
        return (
            f"[{tag}] {ident.get('arch','')}  {ident.get('op','')}  "
            f"{ident.get('kernel_name','')}  "
            f"{ident.get('shape','') or '(no shape)'}  "
            f"({result.get('n_runs', 0)} run(s), need 2)"
        )

    d = result.get("diff", {})
    ident = d.get("identity", {})
    lines = [
        f"[{tag}] {ident.get('arch','')}  {ident.get('op','')}  "
        f"{ident.get('kernel_name','')}  "
        f"{ident.get('shape','') or '(no shape)'}"
    ]
    metric = result.get("metric")
    pct = result.get("pct_change")
    if pct is not None:
        floor = result.get("floor_pct")
        lines.append(
            f"  {metric}: {d.get('baseline'):g} -> {d.get('current'):g}  "
            f"({pct:+.1f}%, floor {floor:.1f}%)"
        )
    else:
        lines.append(
            f"  {metric}: not directly comparable "
            f"(baseline={d.get('baseline')}, current={d.get('current')})"
        )
    for k, e in (d.get("panel") or {}).items():
        if "delta" in e:
            lines.append(
                f"    {k}: {e['baseline']:g} -> {e['current']:g} ({e['delta']:+g})"
            )
    return "\n".join(lines)
