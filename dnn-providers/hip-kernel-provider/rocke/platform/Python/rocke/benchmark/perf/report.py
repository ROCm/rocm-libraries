# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reporting primitive - serialize a record and format record/diff views (pure).

The consume-side counterpart to the harness: it turns a measurement record into
(a) round-trippable JSON and (b) human/agent-readable views - the diagnostic
**panel** for one record, and a **diff** of two records (metric delta + per-panel
change). It makes no judgement call: it computes and formats deltas but does not
decide "regression vs noise" - that gate lives in the user tool's `selfcheck`
(which layers a threshold + spread on top of this). Keeping the verdict out of here
means the same formatter serves the local tool and an external framework alike.

Pure, stdlib-only (`json`): reads records, writes nothing.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from . import schema as _schema

# Where each panel field lives in a record (searched in this order).
_PANEL_SECTIONS = ("derived", "counters", "resources")


def to_json(record: Mapping[str, Any], *, indent: Optional[int] = 2) -> str:
    """Serialize a record to JSON (round-trips via `from_json`)."""
    return json.dumps(record, indent=indent, sort_keys=True)


def from_json(text: str) -> dict:
    """Parse a record back from JSON produced by `to_json`."""
    return json.loads(text)


def panel(record: Mapping[str, Any]) -> dict:
    """Flatten the diagnostic panel (`schema.PANEL_KEYS`) into name -> value.

    Only keys that are actually present are returned, so a Tier-W or RDNA record
    (partial coverage) yields a smaller panel rather than zeros it never measured.
    """
    out: dict = {}
    for key in _schema.PANEL_KEYS:
        for section in _PANEL_SECTIONS:
            sec = record.get(section) or {}
            if key in sec and sec[key] is not None:
                out[key] = sec[key]
                break
    return out


def _spread_pct(record: Mapping[str, Any], which: str) -> Optional[float]:
    spread = record.get("spread") or {}
    if which == _schema.PRIMARY_METRIC:
        return spread.get("busy_cycles_pct")
    if which == _schema.FALLBACK_METRIC:
        # aggregate stores this both places; prefer the wall field.
        return (record.get("wall") or {}).get("ms_spread_pct", spread.get("ms_pct"))
    return None


def format_record(record: Mapping[str, Any]) -> str:
    """Human-readable one-record view: identity, primary metric, panel."""
    arch, kernel, sig = _schema.identity(record)
    val, which = _schema.metric(record)
    lines = [f"{arch}  {kernel}  {sig or '(no shape)'}"]
    if val is not None:
        sp = _spread_pct(record, which)
        tail = f"  (spread {sp:.1f}%)" if sp is not None else ""
        n = record.get("n_samples")
        tail += f"  [n={n}]" if n else ""
        lines.append(f"  metric: {which} = {val:g}{tail}")
    else:
        lines.append("  metric: (none captured)")
    p = panel(record)
    if p:
        lines.append("  panel:")
        for k, v in p.items():
            lines.append(f"    {k}: {v:g}" if isinstance(v, (int, float)) else f"    {k}: {v}")
    cap = record.get("captured_counters") or []
    if cap:
        lines.append(f"  captured: {', '.join(cap)}")
    return "\n".join(lines)


def diff(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> dict:
    """Structured comparison of two records (advisory - no pass/fail).

    Both metrics here are lower-is-better, so `slower` = current metric exceeds
    baseline. `pct_change` is relative to baseline. `metric_mismatch` is set when
    the two records' primary metric sources differ (e.g. one has counters, the
    other only wall) - a pct is still omitted in that case to avoid a bogus number.
    """
    b_val, b_which = _schema.metric(baseline)
    c_val, c_which = _schema.metric(current)
    out: dict = {
        "identity": {"arch": _schema.identity(current)[0],
                     "kernel_name": _schema.identity(current)[1],
                     "shape": _schema.identity(current)[2]},
        "metric": c_which,
        "baseline": b_val,
        "current": c_val,
        "metric_mismatch": bool(b_which and c_which and b_which != c_which),
        "spread_pct": _spread_pct(current, c_which),
    }
    if b_val is not None and c_val is not None and not out["metric_mismatch"]:
        out["abs_delta"] = c_val - b_val
        out["pct_change"] = ((c_val - b_val) / b_val * 100.0) if b_val else None
        out["slower"] = c_val > b_val
    # per-panel deltas for the fields present in both
    pb, pc = panel(baseline), panel(current)
    pan: dict = {}
    for k in sorted(set(pb) & set(pc)):
        entry = {"baseline": pb[k], "current": pc[k]}
        if isinstance(pb[k], (int, float)) and isinstance(pc[k], (int, float)):
            entry["delta"] = pc[k] - pb[k]
        pan[k] = entry
    out["panel"] = pan
    return out


def format_diff(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> str:
    """Human-readable diff view built from `diff`."""
    d = diff(baseline, current)
    ident = d["identity"]
    lines = [f"{ident['arch']}  {ident['kernel_name']}  {ident['shape'] or '(no shape)'}"]
    if d.get("metric_mismatch"):
        lines.append(f"  metric mismatch: baseline vs current use different metrics "
                     f"(current={d['metric']}); values not directly comparable")
        lines.append(f"    baseline={d['baseline']}  current={d['current']}")
    elif d.get("pct_change") is not None:
        arrow = "SLOWER" if d["slower"] else "faster"
        sp = d.get("spread_pct")
        sp_txt = f"  (noise ~{sp:.1f}%)" if sp is not None else ""
        lines.append(f"  {d['metric']}: {d['baseline']:g} -> {d['current']:g}  "
                     f"({d['pct_change']:+.1f}%, {arrow}){sp_txt}")
    else:
        lines.append(f"  {d['metric']}: {d['baseline']} -> {d['current']}")
    if d["panel"]:
        lines.append("  panel change:")
        for k, e in d["panel"].items():
            if "delta" in e:
                lines.append(f"    {k}: {e['baseline']:g} -> {e['current']:g} ({e['delta']:+g})")
            else:
                lines.append(f"    {k}: {e['baseline']} -> {e['current']}")
    return "\n".join(lines)
