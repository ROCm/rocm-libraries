# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Adaptive cell splitting for the active-learning loop.

INVARIANT: a leaf of the cumulative split tree is served by its own trained
model or, while it has none, by its nearest trained ancestor; stage05 keeps a
split parent's model in the bundle until every leaf under it has its own.
The 96 base grid cells (lib/grid.py) are the roots; binary splits add
children. A sub-cell label is its parent's label plus one
`#<axis><op><threshold>` segment per split, so it carries its ancestry:

    Large|Large|LargeK|Bany                          base cell
    Large|Large|LargeK|Bany#M<=4096                  split once on M
    Large|Large|LargeK|Bany#M>4096
    Large|Large|LargeK|Bany#M>4096#K<=2048           split of a split
    Large|Large|LargeK|Bany#M>4096#K>2048

`<round>/stage04b/splits.json` records the splits decided in a round; the
cumulative tree is the union over rounds (`load_cumulative_split_tree`). A
deployed model carries its own tree, rebuilt from its cell labels
(`split_tree_from_labels`). Routing (`assign_subcell` + `resolve_model_cell`)
must match the engine: walk from the base cell to the leaf (`value <=
threshold` goes to the lo child), then up to the nearest trained ancestor.

SPLIT DECISION (`choose_split_for_cell`):
  1. The caller supplies per-GEMM log ratios log(pick_us / winner_us).
  2. For each eligible axis (the widest tiers of the base cell: Large M/N,
     LargeK K), the threshold is the log-median of the axis values and the
     gain is the CART variance reduction of the log ratios.
  3. Axes spanning less than the depth-aware log range, or leaving a child
     with fewer than `min_split_count` GEMMs, are rejected.
  4. The largest gain wins; ties prefer an axis not yet in the cell's split
     path, then M before N before K.
  5. Cells at `max_split_depth` are never split.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Container, Dict, Iterable, List, Optional, Tuple

from . import grid as _grid

# B (Bnone vs Bany) is binary and never split.
SPLITTABLE_AXES: Tuple[str, ...] = ("M", "N", "K")
ROUTING_AXES: Tuple[str, ...] = ("M", "N", "K", "B")


@dataclass(frozen=True)
class SplitRule:
    """One node of the split tree: `cell` (the parent) sends shapes with
    `axis <= threshold` to `lo_label` and the rest to `hi_label`."""

    cell: str
    axis: str
    threshold: int
    lo_label: str
    hi_label: str
    reason: str = ""
    prior_sel_eff: Optional[float] = None
    this_sel_eff: Optional[float] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "cell": self.cell,
            "axis": self.axis,
            "threshold": int(self.threshold),
            "lo_label": self.lo_label,
            "hi_label": self.hi_label,
            "reason": self.reason,
            "prior_sel_eff": self.prior_sel_eff,
            "this_sel_eff": self.this_sel_eff,
        }

    @classmethod
    def from_json(cls, d: Dict[str, object]) -> "SplitRule":
        return cls(
            cell=str(d["cell"]),
            axis=str(d["axis"]),
            threshold=int(d["threshold"]),  # type: ignore[arg-type]
            lo_label=str(d["lo_label"]),
            hi_label=str(d["hi_label"]),
            reason=str(d.get("reason") or ""),
            prior_sel_eff=(
                float(d["prior_sel_eff"])  # type: ignore[arg-type]
                if d.get("prior_sel_eff") is not None
                else None
            ),
            this_sel_eff=(
                float(d["this_sel_eff"])  # type: ignore[arg-type]
                if d.get("this_sel_eff") is not None
                else None
            ),
        )


# ── label helpers ────────────────────────────────────────────────────────────


def make_subcell_label(parent: str, axis: str, op: str, threshold: int) -> str:
    """Append one split segment; `op` is '<=' or '>'."""
    return f"{parent}#{axis}{op}{int(threshold)}"


def parent_of(label: str) -> Optional[str]:
    """The immediate parent label, or None for a base cell."""
    i = label.rfind("#")
    return label[:i] if i > 0 else None


def is_subcell(label: str) -> bool:
    return "#" in label


def split_depth(label: str) -> int:
    return label.count("#")


def base_cell(label: str) -> str:
    """The base 96-cell label a sub-cell descends from."""
    i = label.find("#")
    return label[:i] if i > 0 else label


_SEG_RE = re.compile(r"^([MNKB])(<=|>)(-?\d+)$")


def split_tree_from_labels(labels: Iterable[str]) -> Dict[str, SplitRule]:
    """Rebuild the split rules implied by a set of cell labels: every segment
    `<axis><op><t>` of a label means its prefix splits on `axis` at `t`.

    This is the tree a deployed model carries, so validation that routes with
    it routes exactly like the runtime. Raises ValueError on a malformed
    segment or on two labels that disagree about a node's split."""
    rules: Dict[str, SplitRule] = {}
    for lbl in labels:
        parent = base_cell(lbl)
        for seg in lbl[len(parent) :].split("#")[1:]:
            m = _SEG_RE.match(seg)
            if not m:
                raise ValueError(f"malformed split segment {seg!r} in label {lbl!r}")
            axis, thr = m.group(1), int(m.group(3))
            ex = rules.get(parent)
            if ex is None:
                rules[parent] = SplitRule(
                    cell=parent,
                    axis=axis,
                    threshold=thr,
                    lo_label=make_subcell_label(parent, axis, "<=", thr),
                    hi_label=make_subcell_label(parent, axis, ">", thr),
                )
            elif ex.axis != axis or int(ex.threshold) != thr:
                raise ValueError(
                    f"inconsistent split for {parent}: "
                    f"{ex.axis}{ex.threshold} vs {axis}{thr}"
                )
            parent = f"{parent}#{seg}"
    return rules


# ── splits.json IO ───────────────────────────────────────────────────────────

SCHEMA_VERSION = "pipeline-v3-splits-2026-05-27"


def write_splits_json(path: Path, round_idx: int, rules: Iterable[SplitRule]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "round": int(round_idx),
        "splits": {r.cell: r.to_json() for r in rules},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _load_one_splits_file(path: Path) -> Dict[str, SplitRule]:
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    out: Dict[str, SplitRule] = {}
    for cell, rule_d in (data.get("splits") or {}).items():
        out[cell] = SplitRule.from_json(rule_d)
    return out


def load_cumulative_split_tree(round_dirs: Iterable[Path]) -> Dict[str, SplitRule]:
    """Merge every `<round>/stage04b/splits.json` of `round_dirs` (in order;
    a later round's rule for the same parent wins) into one tree mapping a
    parent label to its rule."""
    tree: Dict[str, SplitRule] = {}
    for rd in round_dirs:
        tree.update(_load_one_splits_file(Path(rd) / "stage04b" / "splits.json"))
    return tree


# ── routing ──────────────────────────────────────────────────────────────────


def axis_value(axis: str, m: int, n: int, k: int, batch: int) -> int:
    if axis == "M":
        return int(m)
    if axis == "N":
        return int(n)
    if axis == "K":
        return int(k)
    if axis == "B":
        return int(batch)
    raise ValueError(f"unknown split axis: {axis}")


def assign_subcell(
    base_label: str, m: int, n: int, k: int, batch: int, tree: Dict[str, SplitRule]
) -> str:
    """Walk the split tree from `base_label` down to the leaf of this shape."""
    label = base_label
    seen: set = set()
    while label in tree and label not in seen:
        seen.add(label)
        rule = tree[label]
        v = axis_value(rule.axis, m, n, k, batch)
        label = rule.lo_label if v <= rule.threshold else rule.hi_label
    return label


def resolve_model_cell(leaf: str, models: Container[str]) -> Optional[str]:
    """Nearest label from `leaf` up to its base cell that is in `models`, or
    None: the cell the engine scores a problem routed to `leaf` with."""
    cur = leaf
    while cur not in models:
        j = cur.rfind("#")
        if j < 0:
            return None
        cur = cur[:j]
    return cur


def route(
    m: int,
    n: int,
    k: int,
    batch: int,
    tree: Dict[str, SplitRule],
    models: Optional[Container[str]] = None,
) -> Tuple[str, Optional[str]]:
    """(leaf label, serving model cell) of one shape; the model cell is None
    when `models` is None or no ancestor is trained."""
    leaf = assign_subcell(_grid.cell_key(m, n, k, batch), m, n, k, batch, tree)
    return leaf, (None if models is None else resolve_model_cell(leaf, models))


# ── split-decision heuristic ────────────────────────────────────────────────


def min_log_range_for_depth(depth: int, base: float = 1.0) -> float:
    """Octaves an axis must span before it is considered for a split,
    tapering with depth (1.0, 0.7, 0.49, ... floored at 0.25) so deep,
    already narrow sub-cells can still be refined."""
    return max(0.25, base * (0.7**depth))


def _log_median(values: List[int]) -> int:
    """Median in log space, rounded to the nearest int."""
    if not values:
        return 0
    log_vals = sorted(math.log(max(int(v), 1)) for v in values)
    n = len(log_vals)
    if n % 2 == 1:
        med = log_vals[n // 2]
    else:
        med = (log_vals[n // 2 - 1] + log_vals[n // 2]) / 2.0
    return max(1, int(round(math.exp(med))))


def _path_axes_of(cell_label: str) -> set:
    """Axes already used by the `#` segments of `cell_label`."""
    out: set = set()
    for seg in cell_label.split("#")[1:]:
        for ax in SPLITTABLE_AXES:
            if seg.startswith(ax):
                out.add(ax)
                break
    return out


def _large_axes_in(cell_label: str) -> List[str]:
    """Axes whose base-cell tier is the unbounded widest tier ('Large' for
    M / N, 'LargeK' for K). Bounded tiers cover a finite range that more data
    improves; only the widest tiers span enough octaves to carve. Sub-cells
    inherit their base cell's tiers."""
    parts = base_cell(cell_label).split("|")
    if len(parts) < 4:
        return []
    m_t, n_t, k_t, _b_t = parts
    out: List[str] = []
    if m_t == "Large":
        out.append("M")
    if n_t == "Large":
        out.append("N")
    if k_t == "LargeK":
        out.append("K")
    return out


def split_eligible_axes(cell_label: str, max_split_depth: int) -> List[str]:
    """Axes `choose_split_for_cell` may split `cell_label` on (none at the
    depth cap)."""
    if split_depth(cell_label) >= max_split_depth:
        return []
    return _large_axes_in(cell_label)


def choose_split_for_cell(
    cell_label: str,
    gemms: List[Dict[str, Any]],
    log_ratios: Optional[List[float]] = None,
    *,
    min_split_count: int = 20,
    min_gain: float = 0.0,
    max_split_depth: int = 6,
    tie_eps: float = 1e-3,
) -> Optional[Tuple[str, int]]:
    """(axis, threshold) to split `cell_label` on, or None when no eligible
    axis gives two children of at least `min_split_count` GEMMs.

    `gemms` are the cell's GEMM dicts (m, n, k, batch_count); `log_ratios`
    is the parallel list of log(pick_us / winner_us). Without log ratios the
    widest axis (largest log range) wins."""
    if not gemms:
        return None
    depth = split_depth(cell_label)
    eligible_axes = split_eligible_axes(cell_label, max_split_depth)
    if not eligible_axes:
        return None
    have_ratios = log_ratios is not None and len(log_ratios) == len(gemms)
    if have_ratios:
        assert log_ratios is not None
        parent_mean = sum(log_ratios) / len(log_ratios)
        parent_var = sum((r - parent_mean) ** 2 for r in log_ratios) / len(log_ratios)
    else:
        parent_var = 0.0

    floor = min_log_range_for_depth(depth)
    path_axes = _path_axes_of(cell_label)

    candidates: List[Tuple[float, str, int, float]] = []
    for axis in eligible_axes:
        axis_vals: List[int] = []
        for g in gemms:
            v = axis_value(
                axis, int(g["m"]), int(g["n"]), int(g["k"]), int(g["batch_count"])
            )
            if v > 0:
                axis_vals.append(v)
        if len(axis_vals) < 2:
            continue
        lo, hi = min(axis_vals), max(axis_vals)
        log_range = math.log2(max(hi, 1)) - math.log2(max(lo, 1))
        if log_range < floor:
            continue
        threshold = _log_median(axis_vals)
        if threshold <= lo:
            threshold = lo + 1
        if threshold >= hi:
            threshold = hi - 1
        if threshold <= lo or threshold >= hi:
            continue
        lo_idx = [i for i, v in enumerate(axis_vals) if v <= threshold]
        hi_idx = [i for i, v in enumerate(axis_vals) if v > threshold]
        if len(lo_idx) < min_split_count or len(hi_idx) < min_split_count:
            continue
        if have_ratios:
            assert log_ratios is not None
            lo_r = [log_ratios[i] for i in lo_idx]
            hi_r = [log_ratios[i] for i in hi_idx]
            lo_mean = sum(lo_r) / len(lo_r)
            hi_mean = sum(hi_r) / len(hi_r)
            lo_var = sum((r - lo_mean) ** 2 for r in lo_r) / len(lo_r)
            hi_var = sum((r - hi_mean) ** 2 for r in hi_r) / len(hi_r)
            w_lo = len(lo_r) / len(gemms)
            w_hi = len(hi_r) / len(gemms)
            gain = parent_var - (w_lo * lo_var + w_hi * hi_var)
        else:
            gain = log_range
        candidates.append((gain, axis, threshold, log_range))

    if not candidates:
        return None
    if have_ratios:
        candidates = [c for c in candidates if c[0] >= min_gain]
        if not candidates:
            return None

    axis_order = {a: i for i, a in enumerate(SPLITTABLE_AXES)}

    def _key(c: Tuple[float, str, int, float]):
        gain, axis, _t, _lr = c
        axis_unused = 0 if axis in path_axes else 1
        return (-gain, -axis_unused, axis_order[axis])

    candidates.sort(key=_key)
    best_gain = candidates[0][0]
    tied = [c for c in candidates if abs(c[0] - best_gain) <= tie_eps]
    tied.sort(key=lambda c: (0 if c[1] not in path_axes else 1, axis_order[c[1]]))
    _, axis, threshold, _ = tied[0]
    return axis, threshold


def build_split_rule(
    cell_label: str,
    axis: str,
    threshold: int,
    reason: str = "",
    prior_sel_eff: Optional[float] = None,
    this_sel_eff: Optional[float] = None,
) -> SplitRule:
    return SplitRule(
        cell=cell_label,
        axis=axis,
        threshold=int(threshold),
        lo_label=make_subcell_label(cell_label, axis, "<=", threshold),
        hi_label=make_subcell_label(cell_label, axis, ">", threshold),
        reason=reason,
        prior_sel_eff=prior_sel_eff,
        this_sel_eff=this_sel_eff,
    )


# ── tree helpers ─────────────────────────────────────────────────────────────


def leaves_under(label: str, tree: Dict[str, SplitRule]) -> List[str]:
    """Every leaf reachable from `label` (a leaf is not a key of `tree`)."""
    if label not in tree:
        return [label]
    rule = tree[label]
    return leaves_under(rule.lo_label, tree) + leaves_under(rule.hi_label, tree)


def format_split_tree(
    tree: Dict[str, SplitRule], roots: Optional[Iterable[str]] = None
) -> str:
    """ASCII outline of the tree, one line per node."""
    if not tree:
        return "(empty -- no splits yet)"
    if roots is None:
        roots = sorted(set(base_cell(c) for c in tree))
    lines: List[str] = []

    def _rec(label: str, prefix: str, last: bool) -> None:
        if label in tree:
            rule = tree[label]
            tag = (
                f"  -> split on {rule.axis} at {rule.threshold}  "
                f"(reason={rule.reason or '-'})"
            )
        else:
            tag = ""
        connector = "└── " if last else "├── "
        if prefix == "":
            lines.append(f"{label}{tag}")
        else:
            lines.append(f"{prefix}{connector}#{label.rsplit('#', 1)[-1]}" f"{tag}")
        if label in tree:
            rule = tree[label]
            child_prefix = prefix + ("    " if last else "│   ")
            _rec(rule.lo_label, child_prefix, False)
            _rec(rule.hi_label, child_prefix, True)

    for r in roots:
        _rec(r, "", True)
    return "\n".join(lines)


def restrict_axis_bounds(
    label: str, axis: str, base_lo: int, base_hi: int
) -> Tuple[int, int]:
    """Tighten `[base_lo, base_hi]` on `axis` with the split segments of
    `label` (inclusive bounds; lo > hi means empty)."""
    lo, hi = int(base_lo), int(base_hi)
    for seg in label.split("#")[1:]:
        if not seg.startswith(axis):
            continue
        rest = seg[len(axis) :]
        if rest.startswith("<="):
            hi = min(hi, int(rest[2:]))
        elif rest.startswith(">"):
            lo = max(lo, int(rest[1:]) + 1)
    return lo, hi


def cell_axis_octaves(cell_label: str) -> Dict[str, float]:
    """Octaves (log2 span) of each widest-tier axis of `cell_label`, from the
    tier's sampling bounds tightened by the label's splits. Drives the
    structural width-split trigger: a leaf that serves a range this wide is
    carved regardless of its measured sel_eff."""
    parts = base_cell(cell_label).split("|")
    if len(parts) < 4:
        return {}
    m_t, n_t, k_t, _b_t = parts
    extents = {
        "M": _grid.TIER_RANGES_MN.get(m_t),
        "N": _grid.TIER_RANGES_MN.get(n_t),
        "K": _grid.TIER_RANGES_K.get(k_t),
    }
    out: Dict[str, float] = {}
    for ax in _large_axes_in(cell_label):
        rng = extents.get(ax)
        if not rng:
            continue
        lo, hi = restrict_axis_bounds(cell_label, ax, int(rng[0]), int(rng[1]))
        if hi > lo > 0:
            out[ax] = math.log2(hi / lo)
    return out
