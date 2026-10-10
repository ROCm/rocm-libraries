# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Training-side evidence of how well each leaf cell of a run is served, and
the weak-cell selection of the orchestrator's enhance mode.

The evidence of a leaf describes the model that serves it now: the model
trained in its latest training round T (its own, or its nearest trained
ancestor's when it has none), and is the first of

  held_out    a `round_R/stage04b/decisions.json` row of a round R > T: the
              model of round R-1 (the current one) on round R's new shapes;
  validation  the leaf's `round_T/stage05/cells.json` entry, `deployed_val`:
              its stage05 validation GEMMs, a model-selection score;
  in_sample   the same entry's `deployed`: its own training GEMMs;
  held_out    for a leaf without a model of its own, its latest stage04b row
              (served by an ancestor, or not by a model at all).

Every value is relative to the fastest measured kernel of each GEMM and
every threshold is absolute; no analytical baseline enters. Only training
artifacts are read: validation datasets (validate/stage08) never steer
sampling, training or the cells chosen for training.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import grid
from . import subcells as sc

SOURCES = ("held_out", "validation", "in_sample")
RANK_KEYS = ("sel_eff", "gain")
DEFAULT_TARGET = 0.95


def absolute_target(threshold: Any) -> float:
    """`validate.sel_eff_threshold` as an absolute target; a relative
    setting ("origami") falls back to DEFAULT_TARGET."""
    try:
        v = float(threshold)
    except (TypeError, ValueError):
        return DEFAULT_TARGET
    return v if 0.0 < v <= 1.0 else DEFAULT_TARGET


@dataclass
class CellEvidence:
    cell: str
    sel_eff: Optional[float] = None
    source: Optional[str] = None
    round: Optional[str] = None
    n_eval: int = 0
    in_sample: Optional[float] = None
    n_gemms: int = 0
    share: float = 0.0
    has_model: bool = False
    trained_round: Optional[str] = None
    decision: Optional[str] = None
    retrain_attempts: int = 0

    def gain(self, target: float) -> float:
        """GEMM share times the gap to `target`: roughly the points the run's
        GEMM-weighted geomean gains when the cell reaches the target."""
        if self.sel_eff is None:
            return 0.0
        return self.share * max(0.0, float(target) - self.sel_eff)

    def to_json(self, target: float) -> Dict[str, Any]:
        d = asdict(self)
        d["gain"] = self.gain(target)
        return d


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and v > 0 else None


def _read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


_ROUND_RE = re.compile(r"^round_(\d+)$")


def round_dirs(run_dir: Path) -> List[Path]:
    out = []
    for p in Path(run_dir).glob("round_*"):
        m = _ROUND_RE.match(p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    return [p for _i, p in sorted(out)]


def _round_no(name: Optional[str]) -> int:
    m = _ROUND_RE.match(name or "")
    return int(m.group(1)) if m else -1


def model_labels(run_dir: Path, rounds: Optional[Sequence[Path]] = None) -> List[str]:
    """Cells of the latest round's model bundle (`model_labels` of its
    stage05/cells.json, trained or carried forward)."""
    for rd in reversed(list(rounds) if rounds is not None else round_dirs(run_dir)):
        cj = _read(rd / "stage05" / "cells.json")
        if cj is None:
            continue
        labels = cj.get("model_labels")
        if isinstance(labels, list):
            return [str(x) for x in labels]
        return [str(c["label"]) for c in cj.get("cells") or [] if c.get("label")]
    return []


def trained_cells(rd: Path) -> List[Mapping[str, Any]]:
    """`cells.json` entries of the cells round `rd`'s stage05 trained; none
    when the round carried the previous bundle forward."""
    if (rd / "stage05" / "carried_forward.json").is_file():
        return []
    cj = _read(rd / "stage05" / "cells.json") or {}
    return [c for c in cj.get("cells") or [] if isinstance(c, dict) and c.get("label")]


def split_tree(run_dir: Path) -> Dict[str, sc.SplitRule]:
    return sc.load_cumulative_split_tree(round_dirs(run_dir))


def collect(
    run_dir: Path,
    gemm_counts: Optional[Mapping[str, int]] = None,
    rounds: Optional[Sequence[Path]] = None,
) -> Dict[str, CellEvidence]:
    """CellEvidence of every leaf of the run's split tree that a training
    artifact of `rounds` (default: every round of the run) mentions, and of
    every leaf of `gemm_counts` (training GEMMs per leaf, counted from the
    training benches), whose counts then replace the artifacts' for the
    leaves it holds. Shares are over all the leaves returned; a ranking by
    gain does not depend on them."""
    rounds = list(rounds) if rounds is not None else round_dirs(run_dir)
    tree = sc.load_cumulative_split_tree(rounds)
    entries: Dict[str, Tuple[str, Mapping[str, Any]]] = {}
    held: Dict[str, List[Tuple[str, Mapping[str, Any]]]] = {}
    counts: Dict[str, Tuple[int, int]] = {}
    decision: Dict[str, str] = {}
    attempts: Dict[str, int] = {}
    for rd in rounds:
        rno = _round_no(rd.name)
        for c in trained_cells(rd):
            entries[str(c["label"])] = (rd.name, c)
            counts[str(c["label"])] = (rno, int(c.get("n_gemms") or 0))
        dec = _read(rd / "stage04b" / "decisions.json") or {}
        for r in dec.get("per_cell") or []:
            if not isinstance(r, dict) or not r.get("cell"):
                continue
            label = str(r["cell"])
            if r.get("n_training_gemms") is not None:
                counts[label] = (rno, int(r["n_training_gemms"]))
            d = r.get("decision")
            if d and d != "no_change":
                decision[label] = str(d)
            if d == "retrain":
                attempts[label] = attempts.get(label, 0) + 1
            if int(r.get("n_eval") or 0) > 0 and _num(r.get("sel_eff_new")):
                held.setdefault(label, []).append((rd.name, r))

    labels = set(model_labels(run_dir, rounds))
    seen = set(entries) | set(held) | set(counts) | labels | set(gemm_counts or {})
    leaves = {lf for c in seen for lf in sc.leaves_under(c, tree)} - set(tree)
    out: Dict[str, CellEvidence] = {}
    for leaf in sorted(leaves):
        ev = CellEvidence(cell=leaf, has_model=leaf in labels)
        serving = sc.resolve_model_cell(leaf, labels) if labels else None
        trained = entries.get(leaf) if ev.has_model else None
        if trained is None and serving is not None:
            trained = entries.get(serving)
        ev.trained_round = trained[0] if trained else None
        t_no = _round_no(ev.trained_round)
        dep = (trained[1].get("deployed") or {}) if trained else {}
        ev.in_sample = _num(dep.get("sel_eff"))
        newer = [h for h in held.get(leaf, []) if _round_no(h[0]) > t_no]
        own = leaf in entries and ev.has_model
        if newer:
            _use_held(ev, newer[-1])
        elif own:
            val = entries[leaf][1].get("deployed_val") or {}
            if _num(val.get("sel_eff")) and int(val.get("n_eval") or 0) > 0:
                ev.sel_eff, ev.source = _num(val.get("sel_eff")), "validation"
                ev.n_eval = int(val.get("n_eval") or 0)
            elif ev.in_sample is not None:
                ev.sel_eff, ev.source = ev.in_sample, "in_sample"
                ev.n_eval = int(dep.get("n_eval") or 0)
            ev.round = entries[leaf][0]
        elif held.get(leaf):
            _use_held(ev, held[leaf][-1])
        if gemm_counts is not None and leaf in gemm_counts:
            ev.n_gemms = int(gemm_counts[leaf])
        else:
            ev.n_gemms = counts.get(leaf, (0, 0))[1]
        ev.decision = decision.get(leaf)
        ev.retrain_attempts = attempts.get(leaf, 0)
        out[leaf] = ev
    total = sum(e.n_gemms for e in out.values())
    for e in out.values():
        e.share = e.n_gemms / total if total else 0.0
    return out


def _use_held(ev: CellEvidence, item: Tuple[str, Mapping[str, Any]]) -> None:
    rnd, r = item
    ev.sel_eff, ev.source, ev.round = _num(r.get("sel_eff_new")), "held_out", rnd
    ev.n_eval = int(r.get("n_eval") or 0)


def select_weak_cells(
    evidence: Mapping[str, CellEvidence],
    *,
    weakest: Optional[int] = None,
    below: Optional[float] = None,
    rank_by: str = "sel_eff",
    target: float = 0.95,
) -> List[CellEvidence]:
    """Leaves with evidence below `below` (all when None), ranked by
    selection efficiency (lowest first) or by `gain(target)` (largest first),
    the first `weakest` of them (all when None)."""
    if rank_by not in RANK_KEYS:
        raise ValueError(f"rank_by must be one of {RANK_KEYS}, got {rank_by!r}")
    pool = [e for e in evidence.values() if e.sel_eff is not None]
    if below is not None:
        pool = [e for e in pool if e.sel_eff < float(below)]
    if rank_by == "gain":
        pool.sort(key=lambda e: (-e.gain(target), e.sel_eff, e.cell))
    else:
        pool.sort(key=lambda e: (e.sel_eff, -e.gain(target), e.cell))
    return pool[: int(weakest)] if weakest is not None else pool


def known_label(label: str, tree: Mapping[str, sc.SplitRule]) -> bool:
    """A base cell of the grid, a split parent of `tree`, or a child of one."""
    if label in tree or label in set(grid.all_cell_labels()):
        return True
    return any(label in (r.lo_label, r.hi_label) for r in tree.values())


def requested_leaves(
    labels: Iterable[str], tree: Mapping[str, sc.SplitRule]
) -> Tuple[List[str], List[str]]:
    """(the leaves of `tree` under `labels` in order without repeats, the
    labels that are not cells of the grid or the tree)."""
    leaves: List[str] = []
    unknown: List[str] = []
    for label in labels:
        if not known_label(label, tree):
            unknown.append(label)
            continue
        for leaf in sc.leaves_under(label, dict(tree)):
            if leaf not in leaves:
                leaves.append(leaf)
    return leaves, unknown


def under_any(label: str, roots: Sequence[str]) -> bool:
    """`label` is one of `roots` or a descendant of one."""
    return any(label == r or label.startswith(r + "#") for r in roots)
