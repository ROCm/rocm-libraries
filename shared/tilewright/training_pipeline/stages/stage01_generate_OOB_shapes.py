# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""stage01_generate_OOB_shapes -- shape generation for one round.

Modes:

  --mode initial   Round 0: the same budget (`--target-per-cell`) for every
                   cell of the 96-cell grid.

  --mode active    Later rounds: every leaf of the cumulative split tree that
                   a model of the latest prior round's bundle serves (its own
                   or its nearest trained ancestor's; the labels are
                   `model_labels` of that round's stage05/cells.json) gets
                   `--shapes-per-cell`; base cells without a model get none.
                   The prior per-cell sel_eff is carried as information only
                   (stage04b decides from the sel_eff measured on the new
                   shapes). `--only-cells` limits the round to the leaves
                   under the labels given, with or without a model.
                   `--base-bundle-dir` names the bundle the round builds on
                   when it is none of the prior rounds (a final retrain):
                   its stage05/cells.json gives the labels, and the splits
                   its repair revoked (stage05/revoked_splits.json,
                   lib/subcells.py) leave the tree before the round samples.

Both modes:

  - Shapes are drawn per cell by `lib.shapes.synthesize_cell_shapes` and must
    pass `lib.shapes.ShapeFilter` built from the config (device-memory budget,
    int32 indexing, cost caps, `seed.exclude` rules).
  - Randomness is derived from (`--seed`, `--round-index`, cell label), so a
    round is reproducible and different rounds draw different shapes.
  - History dedup: every (M, N, K, B) found in the prior rounds
    (`<round>/stage01/shapes.yaml` and `<round>/stage04/chunk_*.csv` of each
    `--prior-round-dir`, plus `--prior-shapes-yaml` / `--prior-enriched-csv-dir`)
    is excluded, so a round's shapes are new.
  - `--min-cell-gemms` tops a cell up to that many shapes when not enough new
    ones exist, re-using history shapes as a last resort. Those are
    re-benched training data, not held-out data, and are listed in
    `reused_shapes.yaml`.
  - Held-out lists (`--exclude-shapes-yaml`, repeatable; any format
    `lib.held_out` reads, read whole or refused): no GEMM of theirs is drawn,
    by exact (M, N, K, B), nor re-used by the top-up; a cell takes its next
    draw instead. categorization.json records each list (file, sha256, its
    GEMMs' digest, the draws it removed, and how many of its GEMMs the
    history holds: GEMMs earlier rounds drew before the list held them out).

Targeted shapes (active mode, `--targeted-per-gemm` N > 0): the previous
round's stage04b lists, per cell, the held-out GEMMs whose pick fell below
its tail floor (decisions.json `tail_gemms`, read from `--targeted-from`,
by default the last --prior-round-dir's stage04b). Each such GEMM, routed
to the leaf of this round's tree that holds it, gets N more new shapes
near it (`lib.shapes.synthesize_near`: each axis within an octave, in its
alignment class, its batch class kept, inside the leaf's box), worst GEMM
first and at most `--targeted-max-per-leaf` per leaf, under the same
filter, history dedup and exclusions as every other shape, so never an
audit shape; a leaf this round samples nothing in gets none. They are new
held-out shapes like the others, whatever the cell's mean.

Seeded targets (active mode, `--targeted-seed`): registered anchor GEMMs,
the rows of a CSV file (`read_seeded_targets`: m, n, k, batch, and the
contexts their near shapes are benched in, `bench_contexts`; with
`--targeted-seed-library`, the rows of that `library` only), join the
targeted draws after the tail GEMMs: each anchor whose leaf this round
samples gets `--targeted-seed-per-target` new shapes near it by the same
`synthesize_near` (each axis within an octave, in its leaf, alignment class
and batch class), under the same filter, history dedup and exclusions, from
an RNG stream of its own (round seed, leaf, the anchor and its contexts), and
not counted against `--targeted-max-per-leaf`. No anchor is ever drawn: every
anchor joins the shapes the draws avoid before any is drawn (they are held
out, `--exclude-shapes-yaml`, as well). targeted_shapes.json lists each
anchor (`anchor`, with its contexts) and its shapes beside the tails, so the
contexts it names bench them (lib/contexts.py `targeted_for`); the file and
categorization.json record the list (path, sha256, library, per target) and
the counts. Without the option the round draws exactly as before.

Weak-cell budget (active mode without `--only-cells`; `need_budget`): the
previous round's stage04b decisions (`--decisions-from`, by default the
last --prior-round-dir's stage04b/decisions.json) set each leaf's target.
A leaf that round retrained for its selection efficiency (decision
`retrain`) or for a held-out collapse (a pass with `collapse_retrain`) is
weak and gets `--weak-cell-factor` times `--shapes-per-cell`. A leaf that
passed with a held-out selection efficiency at least
`--strong-cell-margin` above the threshold that stage04b applied is strong
and gets `--strong-cell-factor` times it, but the strong leaves give up at
most the shapes the weak leaves gain: the round's total target never
falls below `--shapes-per-cell` per leaf and never exceeds that plus the
weak leaves' extra. A leaf's status reads its own row and the rows of the
leaves under it; the children of a split that round made read none (the
split already gave their region a leaf each). Without decisions every leaf
keeps `--shapes-per-cell`; with both factors at 1 none are read.

Very-large-K stratum (same rounds; `large_k_shapes`): a leaf whose new
shapes include some with K above LARGE_K_ABOVE draws `--large-k-factor`
minus one times that many more there, inside its box, under the same
filter, history dedup and exclusions, from a stream of its own, so its
other draws stay as they are.

Outputs in `--out-dir`:

  shapes.yaml          hipblaslt-bench yaml, shuffled so that stage03's
                       contiguous blocks are random samples of the round
  reused_shapes.yaml   shapes re-used from earlier rounds (see
                       `lib.shapes.read_reused_shapes`)
  categorization.json  per-cell targets and counts, seeds, filter settings
  per_cell_plan.json   active mode only: per-cell plan, prior sel_eff and
                       the round's budget (`budget`; per leaf `need`)
  targeted_shapes.json active mode with targeted shapes: per leaf, each
                       tail GEMM or seeded anchor and the shapes drawn
                       near it
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import held_out as hol  # noqa: E402
from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import parse_shapes_yaml, write_bench_yaml  # noqa: E402
from lib.fs import read_json, write_json  # noqa: E402
from lib.grid import TIER_RANGES_K, TIER_RANGES_MN, all_cell_labels  # noqa: E402
from lib.grid import cell_key  # noqa: E402
from lib.shapes import (  # noqa: E402
    MEMORY_SAFE_FRACTION,
    REUSED_SHAPES_FILE,
    SEEDED_PER_TARGET,
    Shape,
    ShapeFilter,
    ShapeTuple,
    device_memory_budget,
    read_seeded_targets,
    synthesize_cell_shapes,
    synthesize_near,
    write_reused_shapes,
)

# Fresh-shape pools drawn per cell before falling back to history re-use.
_FRESH_POOLS = 3
TARGETED_FILE = "targeted_shapes.json"
# Targeted shapes per tail GEMM and per leaf (`--targeted-per-gemm`,
# `--targeted-max-per-leaf`).
TARGETED_PER_GEMM = 4
TARGETED_MAX_PER_LEAF = 32
# The very-large-K stratum: shapes with K above this (`--large-k-factor`).
LARGE_K_ABOVE = 32768
# How far above stage04b's threshold a pass must read for its leaf to be
# strong (`--strong-cell-margin`).
STRONG_CELL_MARGIN = 0.03


def derive_seed(*parts: object) -> int:
    """Deterministic 64-bit seed from `parts`, stable across processes and
    Python versions (unlike `hash()`)."""
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ── budgets ──────────────────────────────────────────────────────────────────


def _build_initial_budget(target_per_cell: int) -> Dict[str, int]:
    return {lbl: target_per_cell for lbl in all_cell_labels()}


def _build_active_plan(
    prior_cells_json: Path, shapes_per_cell: int, split_tree: Dict[str, "sc.SplitRule"]
) -> Dict[str, Dict[str, Any]]:
    """plan[leaf] = {has_prior_model, sel_eff_prev_round, n_target, parent_cell}.

    `prior_cells_json` is the latest prior round's stage05/cells.json. Its
    `model_labels` (every cell of that round's models.pt, trained or carried
    forward; the `cells` entries when absent) are expanded through the
    cumulative `split_tree`, and every resulting leaf gets `shapes_per_cell`;
    `parent_cell` names the ancestor whose model serves a leaf without its
    own. Base cells without a prior model get 0."""
    cells_json = read_json(prior_cells_json)
    prior_by_cell: Dict[str, float] = {}
    for c in cells_json.get("cells", []):
        prior_by_cell[c["label"]] = float(c.get("best_sel_eff", float("nan")))
    labels = set(cells_json.get("model_labels", prior_by_cell))

    plan: Dict[str, Dict[str, Any]] = {}
    for lbl in all_cell_labels():
        if lbl not in labels and lbl not in split_tree:
            plan[lbl] = {
                "sel_eff_prev_round": None,
                "n_target": 0,
                "has_prior_model": False,
            }
    leaves = {leaf for lbl in labels for leaf in sc.leaves_under(lbl, split_tree)}
    for leaf in sorted(leaves):
        serving = sc.resolve_model_cell(leaf, labels)
        plan[leaf] = {
            "sel_eff_prev_round": prior_by_cell.get(leaf) if serving == leaf else None,
            "n_target": shapes_per_cell,
            "has_prior_model": serving is not None,
            "parent_cell": serving if serving != leaf else None,
        }
    return plan


def restrict_plan(
    plan: Dict[str, Dict[str, Any]],
    only_cells: List[str],
    shapes_per_cell: int,
    split_tree: Dict[str, "sc.SplitRule"],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """(`plan` limited to the leaves under `only_cells`, the labels that are
    not cells of the grid or the split tree). Those leaves get
    `shapes_per_cell`, also without a prior model, and are marked
    `requested`; every other entry gets 0."""
    from lib.cell_evidence import requested_leaves

    leaves, unknown = requested_leaves(only_cells, split_tree)
    out: Dict[str, Dict[str, Any]] = {}
    for cell, info in plan.items():
        out[cell] = dict(info, n_target=0, requested=False)
    for leaf in leaves:
        info = out.get(leaf) or {
            "sel_eff_prev_round": None,
            "has_prior_model": False,
        }
        out[leaf] = dict(info, n_target=shapes_per_cell, requested=True)
    return out, unknown


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def read_decisions(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The stage04b decisions.json at `path`, None without the file;
    ValueError for a file that holds no per-cell decisions."""
    if path is None or not Path(path).is_file():
        return None
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ValueError(f"cannot read the stage04b decisions {path}: {e}") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("per_cell"), list):
        raise ValueError(f"{path} holds no per_cell list of stage04b decisions")
    return doc


def _reading(row: Mapping[str, Any]) -> Optional[float]:
    """A decision row's held-out selection efficiency, when it measured one."""
    se = row.get("sel_eff_new")
    if isinstance(se, bool) or not isinstance(se, (int, float)):
        return None
    if not math.isfinite(se) or not int(row.get("n_eval") or 0):
        return None
    return float(se)


def _need_of(rows: List[Mapping[str, Any]], floor: Optional[float]) -> Optional[str]:
    """`weak` when stage04b retrained a leaf of `rows` (decision retrain, or
    a pass with collapse_retrain), `strong` when every one passed on
    held-out GEMMs at `floor` or above, else None."""
    if not rows:
        return None
    if any(r.get("decision") == "retrain" or r.get("collapse_retrain") for r in rows):
        return "weak"
    if floor is None:
        return None
    for r in rows:
        se = _reading(r)
        if r.get("decision") != "pass" or se is None or se < floor:
            return None
    return "strong"


def need_budget(
    plan: Dict[str, Dict[str, Any]],
    decisions: Optional[Mapping[str, Any]],
    shapes_per_cell: int,
    *,
    weak_factor: float = 1.0,
    strong_factor: float = 1.0,
    strong_margin: float = STRONG_CELL_MARGIN,
    min_per_cell: int = 0,
) -> Dict[str, Any]:
    """Set the target of every leaf `plan` samples (n_target > 0, each
    `shapes_per_cell`) from the previous round's stage04b `decisions`, as
    the module docstring's weak-cell budget says, mark it (`need`: weak,
    strong or None) and return the round's budget record. A strong leaf's
    target stays within [`min_per_cell`, `shapes_per_cell`] (the top-up
    lifts it to `min_per_cell` anyway); the strong leaves' cut, at most the
    weak leaves' extra, is shared equally (each would give up as many), and
    what does not divide evenly is a shape more each for the leaves with
    the highest readings (a leaf's lowest row)."""
    spc = int(shapes_per_cell)
    by_cell: Dict[str, List[Mapping[str, Any]]] = {}
    threshold: Optional[float] = None
    if decisions is not None:
        for row in decisions.get("per_cell") or []:
            if isinstance(row, Mapping) and isinstance(row.get("cell"), str):
                by_cell.setdefault(row["cell"], []).append(row)
        t = decisions.get("sel_eff_threshold")
        if isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t):
            threshold = float(t)
    floor = None if threshold is None else threshold + float(strong_margin)

    def rows_of(leaf: str) -> List[Mapping[str, Any]]:
        under = leaf + "#"
        return [
            r
            for c, rs in by_cell.items()
            if c == leaf or c.startswith(under)
            for r in rs
        ]

    leaves = sorted(c for c, v in plan.items() if int(v.get("n_target") or 0) > 0)
    rows = {leaf: rows_of(leaf) for leaf in leaves}
    need = {leaf: _need_of(rows[leaf], floor) for leaf in leaves}
    weak = [leaf for leaf in leaves if need[leaf] == "weak"]
    strong = [leaf for leaf in leaves if need[leaf] == "strong"]
    weak_target = max(spc, _half_up(weak_factor * spc))
    strong_target = min(spc, max(int(min_per_cell), 1, _half_up(strong_factor * spc)))
    gain = (weak_target - spc) * len(weak)
    want = (spc - strong_target) * len(strong)
    give = min(gain, want)
    cut = {leaf: 0 for leaf in strong}
    if give > 0:
        lowest = {leaf: min(_reading(r) for r in rows[leaf]) for leaf in strong}
        order = sorted(strong, key=lambda leaf: (-lowest[leaf], leaf))
        share, rest = divmod(give, len(strong))
        for i, leaf in enumerate(order):
            cut[leaf] = share + (1 if i < rest else 0)
    for cell, info in plan.items():
        info["need"] = need.get(cell)
        if need.get(cell) == "weak":
            info["n_target"] = weak_target
        elif need.get(cell) == "strong":
            info["n_target"] = spc - cut[cell]
    return {
        "applied": decisions is not None,
        "weak_cell_factor": float(weak_factor),
        "strong_cell_factor": float(strong_factor),
        "strong_cell_margin": float(strong_margin),
        "sel_eff_threshold": threshold,
        "n_leaves": len(leaves),
        "n_weak": len(weak),
        "n_strong": len(strong),
        "uniform_total": spc * len(leaves),
        "weak_extra": gain,
        "strong_cut_wanted": want,
        "strong_cut": give,
        "total_target": sum(int(plan[leaf]["n_target"]) for leaf in leaves),
    }


# ── history ──────────────────────────────────────────────────────────────────


def _harvest_enriched_dir(path: Path) -> Set[ShapeTuple]:
    seen: Set[ShapeTuple] = set()
    if not path.exists():
        return seen
    for fn in sorted(os.listdir(path)):
        if not (fn.startswith("chunk_") and fn.endswith(".csv")):
            continue
        with (path / fn).open() as f:
            for row in csv.DictReader(f):
                try:
                    seen.add(
                        (
                            int(row["m"]),
                            int(row["n"]),
                            int(row["k"]),
                            int(row["batch_count"]),
                        )
                    )
                except (KeyError, ValueError, TypeError):
                    continue
    return seen


_ROUND_RE = re.compile(r"/(round_\d+)/")


def _round_tag(p: Path) -> str:
    m = _ROUND_RE.search(str(p.resolve()))
    return m.group(1) if m else "?"


def collect_history(
    prior_shapes_yamls: List[Path], prior_enriched_dirs: List[Path], quiet: bool
) -> Set[ShapeTuple]:
    history: Set[ShapeTuple] = set()
    sources = [(p, "yaml", set(parse_shapes_yaml(p))) for p in prior_shapes_yamls]
    sources += [(d, "enriched", _harvest_enriched_dir(d)) for d in prior_enriched_dirs]
    for path, kind, shapes in sources:
        n0 = len(history)
        history |= shapes
        if not quiet:
            ui.info(
                "dedup",
                f"prior {_round_tag(path)} {kind} {path.name}: "
                f"+{len(history) - n0} shapes (cumulative {len(history):,})",
            )
    return history


# ── per-cell sampling ────────────────────────────────────────────────────────


@dataclass
class SampleResult:
    shapes: List[ShapeTuple] = field(default_factory=list)
    reused: List[ShapeTuple] = field(default_factory=list)
    actual_per_cell: Dict[str, int] = field(default_factory=dict)
    reused_per_cell: Dict[str, int] = field(default_factory=dict)
    fresh_per_cell: Dict[str, List[ShapeTuple]] = field(default_factory=dict)
    n_history_dropped: int = 0
    n_excluded: int = 0
    excluded_hits: Set[ShapeTuple] = field(default_factory=set)


def _axis_bounds(cell: str, base: str) -> Optional[Dict[str, Tuple[int, int]]]:
    if cell == base:
        return None
    mt, nt, kt, _bt = base.split("|")
    extents = {
        "M": TIER_RANGES_MN[mt],
        "N": TIER_RANGES_MN[nt],
        "K": TIER_RANGES_K[kt],
    }
    return {
        ax: sc.restrict_axis_bounds(cell, ax, lo, hi)
        for ax, (lo, hi) in extents.items()
    }


def sample_shapes(
    per_cell_target: Dict[str, int],
    round_seed: int,
    history: Set[ShapeTuple],
    shape_filter: ShapeFilter,
    *,
    n_strata: int = 1,
    min_per_cell: int = 0,
    excluded: Optional[Set[ShapeTuple]] = None,
) -> SampleResult:
    """Up to `per_cell_target[cell]` new shapes per cell (base label or
    sub-cell label), none of them in `history`. A cell left below
    `min_per_cell` is topped up, with new shapes first and history shapes
    after; the history shapes used are reported in `reused`. Each cell draws
    from its own RNG stream, so its shapes do not depend on the other cells'
    targets or their order. A shape in `excluded` is never drawn, not even by
    the top-up."""
    out = SampleResult()
    excluded = excluded or set()
    chosen: Set[ShapeTuple] = set()
    for cell in sorted(per_cell_target):
        target = int(per_cell_target[cell])
        if target <= 0:
            out.actual_per_cell[cell] = 0
            continue
        rng = random.Random(derive_seed(round_seed, "cell", cell))
        base = sc.base_cell(cell)
        bounds = _axis_bounds(cell, base)

        excluded_hits: Set[ShapeTuple] = set()

        def _pool(size: int) -> List[ShapeTuple]:
            drawn = synthesize_cell_shapes(
                base, size, rng, shape_filter, axis_bounds=bounds, n_strata=n_strata
            )
            excluded_hits.update(s for s in drawn if s in excluded)
            return [s for s in drawn if s not in excluded]

        fresh: List[ShapeTuple] = []
        history_hits: Set[ShapeTuple] = set()
        for _ in range(_FRESH_POOLS):
            pool = _pool(target * 3)
            for s in pool:
                if len(fresh) >= target:
                    break
                if s in chosen:
                    continue
                if s in history:
                    history_hits.add(s)
                    continue
                chosen.add(s)
                fresh.append(s)
            if len(fresh) >= target or not pool:
                break
        reused: List[ShapeTuple] = []
        if len(fresh) < min_per_cell:
            pool = [
                s for s in _pool(max(min_per_cell * 6, target * 3)) if s not in chosen
            ]
            for s in [s for s in pool if s not in history] + [
                s for s in pool if s in history
            ]:
                if len(fresh) + len(reused) >= min_per_cell:
                    break
                chosen.add(s)
                if s in history:
                    reused.append(s)
                else:
                    fresh.append(s)
            if len(fresh) + len(reused) < min_per_cell:
                ui.warn(
                    "coverage",
                    f"leaf {cell}: only {len(fresh) + len(reused)} distinct shapes "
                    f"(< min {min_per_cell}); the region is too small to cover "
                    f"and this leaf may stay undertrained.",
                )
        out.shapes.extend(fresh + reused)
        out.reused.extend(reused)
        out.fresh_per_cell[cell] = list(fresh)
        out.actual_per_cell[cell] = len(fresh) + len(reused)
        out.reused_per_cell[cell] = len(reused)
        out.n_history_dropped += len(history_hits)
        out.n_excluded += len(excluded_hits)
        out.excluded_hits |= excluded_hits
    return out


# ── targeted shapes ──────────────────────────────────────────────────────────


def leaf_box(leaf: str) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """((M lo, hi), (N lo, hi), (K lo, hi)) of a leaf: its base cell's tier
    ranges narrowed by the splits above it."""
    base = sc.base_cell(leaf)
    mt, nt, kt, _bt = base.split("|")
    box = [TIER_RANGES_MN[mt], TIER_RANGES_MN[nt], TIER_RANGES_K[kt]]
    bounds = _axis_bounds(leaf, base) or {}
    for i, ax in enumerate("MNK"):
        if ax in bounds:
            lo, hi = bounds[ax]
            box[i] = (max(box[i][0], lo), min(box[i][1], hi))
    return box[0], box[1], box[2]


def large_k_shapes(
    fresh: Mapping[str, List[ShapeTuple]],
    factor: float,
    round_seed: int,
    shape_filter: ShapeFilter,
    *,
    n_strata: int,
    taken: Set[ShapeTuple],
) -> Dict[str, List[ShapeTuple]]:
    """{leaf: shapes}: for each leaf of `fresh` (its new shapes of the
    round) with n of them above LARGE_K_ABOVE in K, (`factor` - 1) * n more
    (rounded half up) in the part of its box above LARGE_K_ABOVE, none of
    them in `taken`, which they join; fewer when the box holds no more.
    Each leaf draws from its own stream."""
    out: Dict[str, List[ShapeTuple]] = {}
    for leaf in sorted(fresh):
        want = _half_up(
            (factor - 1.0) * sum(1 for s in fresh[leaf] if s[2] > LARGE_K_ABOVE)
        )
        (mlo, mhi), (nlo, nhi), (klo, khi) = leaf_box(leaf)
        if want <= 0 or khi <= LARGE_K_ABOVE:
            continue
        bounds = {
            "M": (mlo, mhi),
            "N": (nlo, nhi),
            "K": (max(klo, LARGE_K_ABOVE + 1), khi),
        }
        rng = random.Random(derive_seed(round_seed, "large_k", leaf))
        drawn = synthesize_cell_shapes(
            sc.base_cell(leaf),
            want * 3,
            rng,
            shape_filter,
            axis_bounds=bounds,
            n_strata=n_strata,
        )
        new: List[ShapeTuple] = []
        for s in drawn:
            if len(new) >= want:
                break
            if s not in taken:
                taken.add(s)
                new.append(s)
        if new:
            out[leaf] = new
    return out


def read_tails(path: Optional[Path]) -> List[Dict[str, Any]]:
    """The tail GEMMs of every cell of a stage04b decisions.json
    (`tail_gemms`), worst first; none without the file."""
    doc = read_json(path) if path is not None and Path(path).is_file() else None
    out: List[Dict[str, Any]] = []
    for row in (doc or {}).get("per_cell") or []:
        for t in row.get("tail_gemms") or []:
            out.append(dict(t, cell=row.get("cell")))
    out.sort(key=lambda t: (float(t["sel_eff"]), t["m"], t["n"], t["k"], t["batch"]))
    return out


def targeted_shapes(
    tails: List[Dict[str, Any]],
    leaves: Set[str],
    split_tree: Dict[str, "sc.SplitRule"],
    round_seed: int,
    shape_filter: ShapeFilter,
    *,
    per_gemm: int,
    max_per_leaf: int,
    taken: Set[ShapeTuple],
) -> Dict[str, List[Dict[str, Any]]]:
    """{leaf: [{tail GEMM, the shapes drawn near it}]}: up to `per_gemm`
    shapes near each tail GEMM of `tails` (`synthesize_near`) in the leaf
    of `split_tree` that holds it, for the leaves of `leaves` only, worst
    GEMM first and at most `max_per_leaf` per leaf, none of them in
    `taken` (the history, the exclusions and the round's other shapes),
    which the drawn shapes join. Each GEMM draws from its own RNG stream."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    count: Dict[str, int] = {}
    for t in tails:
        m, n, k, b = int(t["m"]), int(t["n"]), int(t["k"]), int(t["batch"])
        leaf = sc.assign_subcell(cell_key(m, n, k, b), m, n, k, b, split_tree)
        room = max_per_leaf - count.get(leaf, 0)
        if leaf not in leaves or room <= 0:
            continue
        rng = random.Random(
            derive_seed(
                round_seed, "targeted", leaf, m, n, k, b, t.get("cu_budget", 0),
                t.get("schedule", "default"),
            )
        )  # fmt: skip
        want = min(per_gemm, room)
        drawn = synthesize_near(
            (m, n, k, b), want * 3, rng, shape_filter, leaf_box(leaf)
        )
        new = []
        for s in drawn:
            if len(new) >= want:
                break
            if s not in taken:
                taken.add(s)
                new.append(s)
        if not new:
            continue
        out.setdefault(leaf, []).append(
            {"tail": dict(t), "shapes": [list(x) for x in new]}
        )
        count[leaf] = count.get(leaf, 0) + len(new)
    return out


def seeded_shapes(
    anchors: List[Dict[str, Any]],
    leaves: Set[str],
    split_tree: Dict[str, "sc.SplitRule"],
    round_seed: int,
    shape_filter: ShapeFilter,
    *,
    per_target: int,
    taken: Set[ShapeTuple],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """({leaf: [{anchor, the shapes drawn near it}]}, [the anchors that got
    none, with why]): up to `per_target` shapes near each anchor of
    `anchors` (`synthesize_near`) in the leaf of `split_tree` that holds it,
    for the leaves of `leaves` only, in file order, none of them an anchor or
    in `taken` (the history, the exclusions and the round's other shapes),
    which the drawn shapes join. Each anchor draws from its own RNG stream,
    seeded with the round seed, its leaf, its shape and its contexts."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    skipped: List[Dict[str, Any]] = []
    taken.update(
        (int(a["m"]), int(a["n"]), int(a["k"]), int(a["batch"])) for a in anchors
    )
    for a in anchors:
        m, n, k, b = int(a["m"]), int(a["n"]), int(a["k"]), int(a["batch"])
        leaf = sc.assign_subcell(cell_key(m, n, k, b), m, n, k, b, split_tree)
        if leaf not in leaves:
            skipped.append(
                dict(a, leaf=leaf, why="the round samples no shape in its leaf")
            )
            continue
        rng = random.Random(
            derive_seed(
                round_seed, "seeded", leaf, m, n, k, b, ",".join(a["bench_contexts"])
            )
        )
        drawn = synthesize_near(
            (m, n, k, b), per_target * 3, rng, shape_filter, leaf_box(leaf)
        )
        new = []
        for s in drawn:
            if len(new) >= per_target:
                break
            if s not in taken:
                taken.add(s)
                new.append(s)
        if not new:
            skipped.append(dict(a, leaf=leaf, why="no new shape near it"))
            continue
        out.setdefault(leaf, []).append(
            {"anchor": dict(a), "shapes": [list(x) for x in new]}
        )
    return out, skipped


# ── main ─────────────────────────────────────────────────────────────────────


def _resolve_round_index(args: argparse.Namespace) -> Optional[int]:
    if args.round_index is not None:
        return int(args.round_index)
    if args.mode == "initial":
        return 0
    if args.prior_round_dir:
        return len(args.prior_round_dir)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="stage01 -- grid shape generator for one round."
    )
    ap.add_argument("--mode", choices=("initial", "active"), required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--config-yaml",
        type=Path,
        required=True,
        help="run config (hipblaslt dtype/layout, bench.*, seed.exclude).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--round-index",
        type=int,
        default=None,
        help="index of the round being generated; mixed into the RNG seed so "
        "every round draws different shapes. Default: 0 in initial mode, the "
        "number of --prior-round-dir entries in active mode.",
    )
    ap.add_argument(
        "--target-per-cell",
        type=int,
        default=1600,
        help="(initial mode) shapes per cell.",
    )
    ap.add_argument(
        "--prior-round-dir",
        type=Path,
        action="append",
        default=None,
        help="(active mode) repeatable; a `<run>/round_<i>/` directory. Its "
        "stage01/shapes.yaml and stage04/chunk_*.csv feed history dedup; the "
        "last one with stage05/cells.json provides the per-cell plan (the "
        "cells of its models bundle).",
    )
    ap.add_argument(
        "--prior-cells",
        type=Path,
        default=None,
        help="(active mode) explicit prior cells.json; overrides the one "
        "found under --prior-round-dir or --base-bundle-dir.",
    )
    ap.add_argument(
        "--base-bundle-dir",
        type=Path,
        default=None,
        help="(active mode) the bundle directory the round builds on when it "
        "is none of the --prior-round-dir (a final retrain): its "
        "stage05/cells.json provides the per-cell plan, and the splits it "
        "revoked leave the cumulative split tree, after the prior rounds' "
        "splits, before the round samples.",
    )
    ap.add_argument(
        "--shapes-per-cell",
        type=int,
        default=150,
        help="(active mode) shapes per leaf with a prior model.",
    )
    ap.add_argument(
        "--only-cells",
        default=None,
        help="(active mode) comma list of cell labels: only the leaves under "
        "them get --shapes-per-cell (also leaves without a prior model); every "
        "other leaf gets none.",
    )
    ap.add_argument(
        "--min-cell-gemms",
        type=int,
        default=0,
        help="top every cell up to at least this many shapes so stage05 can "
        "train it, re-using history shapes when no new ones are left "
        "(recorded in reused_shapes.yaml). 0 = off.",
    )
    ap.add_argument(
        "--n-strata",
        type=int,
        default=1,
        help="per-axis log-equal strata for within-cell coverage; 1 samples "
        "the whole cell box log-uniformly.",
    )
    ap.add_argument(
        "--prior-shapes-yaml",
        type=Path,
        action="append",
        default=None,
        help="repeatable; extra history shape yamls.",
    )
    ap.add_argument(
        "--prior-enriched-csv-dir",
        type=Path,
        action="append",
        default=None,
        help="repeatable; extra history chunk_*.csv directories.",
    )
    ap.add_argument(
        "--exclude-shapes-yaml",
        type=Path,
        action="append",
        default=None,
        help="repeatable; a held-out list (the audit set, a fresh audit, an OOD "
        "list) none of whose GEMMs this round may draw or re-use: " + hol.FORMATS + ".",
    )
    ap.add_argument(
        "--targeted-per-gemm",
        type=int,
        default=TARGETED_PER_GEMM,
        help="(active mode) new shapes near each tail GEMM the previous "
        "round's stage04b lists (0: none)",
    )
    ap.add_argument(
        "--targeted-max-per-leaf",
        type=int,
        default=TARGETED_MAX_PER_LEAF,
        help="(active mode) targeted shapes a leaf gets at most",
    )
    ap.add_argument(
        "--targeted-from",
        type=Path,
        default=None,
        help="(active mode) the stage04b decisions.json whose tail GEMMs get "
        "targeted shapes (default: the last --prior-round-dir's)",
    )
    ap.add_argument(
        "--targeted-seed",
        type=Path,
        default=None,
        help="(active mode) a CSV of registered anchor GEMMs (m, n, k, batch, "
        "bench_contexts): each gets --targeted-seed-per-target new shapes near "
        "it, beside the tail GEMMs' targeted shapes",
    )
    ap.add_argument(
        "--targeted-seed-per-target",
        type=int,
        default=SEEDED_PER_TARGET,
        help="(with --targeted-seed) new shapes near each anchor (>= 1)",
    )
    ap.add_argument(
        "--targeted-seed-library",
        default=None,
        help="(with --targeted-seed) the value of the file's library column "
        "whose rows are this run's anchors",
    )
    ap.add_argument(
        "--weak-cell-factor",
        type=float,
        default=1.0,
        help="(active mode) new shapes of a leaf the previous round's stage04b "
        "retrained (decision retrain, or a pass with collapse_retrain), as a "
        "multiple of --shapes-per-cell (>= 1; 1: --shapes-per-cell)",
    )
    ap.add_argument(
        "--strong-cell-factor",
        type=float,
        default=1.0,
        help="(active mode) the same for a leaf that passed at least "
        "--strong-cell-margin above stage04b's threshold (0 < f <= 1); the "
        "strong leaves give up at most what the weak leaves gain",
    )
    ap.add_argument(
        "--strong-cell-margin",
        type=float,
        default=STRONG_CELL_MARGIN,
        help="(active mode) how far above stage04b's threshold a pass must read "
        "for its leaf to be strong (>= 0)",
    )
    ap.add_argument(
        "--large-k-factor",
        type=float,
        default=1.0,
        help=f"(active mode) a leaf's new shapes with K above {LARGE_K_ABOVE}, as "
        "a multiple of the number its draw holds (>= 1; 1: none more)",
    )
    ap.add_argument(
        "--decisions-from",
        type=Path,
        default=None,
        help="(active mode) the stage04b decisions.json the weak-cell budget "
        "reads (default: the last --prior-round-dir's; without one every "
        "leaf gets --shapes-per-cell)",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.targeted_per_gemm < 0 or args.targeted_max_per_leaf < 0:
        ui.err("err", "--targeted-per-gemm and --targeted-max-per-leaf are >= 0")
        return 1
    anchors: List[Dict[str, Any]] = []
    seeded_sha256: Optional[str] = None
    if args.targeted_seed is not None:
        if args.targeted_seed_per_target < 1:
            ui.err("err", "--targeted-seed-per-target must be at least 1")
            return 1
        try:
            anchors = read_seeded_targets(
                args.targeted_seed, args.targeted_seed_library
            )
            seeded_sha256 = hashlib.sha256(args.targeted_seed.read_bytes()).hexdigest()
        except (OSError, ValueError) as e:
            ui.err("seeded", str(e))
            return 1
    bad = False
    for flag, value, ok, rule in (
        ("--weak-cell-factor", args.weak_cell_factor, lambda v: v >= 1, ">= 1"),
        (
            "--strong-cell-factor",
            args.strong_cell_factor,
            lambda v: 0 < v <= 1,
            "in (0, 1]",
        ),
        ("--strong-cell-margin", args.strong_cell_margin, lambda v: v >= 0, ">= 0"),
        ("--large-k-factor", args.large_k_factor, lambda v: v >= 1, ">= 1"),
    ):
        if not (math.isfinite(value) and ok(value)):
            ui.err("err", f"{flag} must be finite and {rule}, got {value!r}")
            bad = True
    if bad:
        return 1
    round_index = _resolve_round_index(args)
    if round_index is None:
        ui.err(
            "err",
            "--mode active without --prior-round-dir needs an explicit "
            "--round-index",
        )
        return 1
    round_seed = derive_seed(args.seed, round_index)

    with args.config_yaml.open() as f:
        cfg = yaml.safe_load(f) or {}
    try:
        budget = device_memory_budget(cfg)
        shape_filter = ShapeFilter.from_config(cfg, budget=budget)
    except ValueError as exc:
        ui.err("cfg", str(exc))
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        ui.banner(f"stage01 shape generator  (mode={args.mode})")
        ui.info("cfg", f"out_dir        : {args.out_dir}")
        ui.info("cfg", f"config_yaml    : {args.config_yaml}")
        ui.info("cfg", f"seed / round   : {args.seed} / {round_index}")
        ui.banner("Stage 1/3  Per-cell budget", ui.C.BLUE)

    per_cell: Dict[str, int]
    plan: Optional[Dict[str, Dict[str, Any]]] = None
    cells_json_path: Optional[Path] = None
    split_tree: Dict[str, "sc.SplitRule"] = {}
    revoked: List[str] = []
    tails_from: Optional[Path] = None
    plan_budget: Optional[Dict[str, Any]] = None
    if args.mode == "initial":
        per_cell = _build_initial_budget(args.target_per_cell)
        if not args.quiet:
            ui.ok(
                "budget",
                f"target_per_cell={args.target_per_cell}  "
                f"total_target={sum(per_cell.values()):,}  cells={len(per_cell)}",
            )
    else:
        prior_round_dirs = list(args.prior_round_dir or [])
        auto_shapes: List[Path] = []
        auto_enriched: List[Path] = []
        auto_cells_json: Optional[Path] = None
        for rd in prior_round_dirs:
            sy = rd / "stage01" / "shapes.yaml"
            if sy.exists():
                auto_shapes.append(sy)
            en = rd / "stage04"
            if en.exists() and any(
                p.name.startswith("chunk_") for p in en.iterdir() if p.is_file()
            ):
                auto_enriched.append(en)
            cj = rd / "stage05" / "cells.json"
            if cj.exists():
                auto_cells_json = cj
        tree_dirs = list(prior_round_dirs)
        if args.base_bundle_dir is not None:
            auto_cells_json = args.base_bundle_dir / "stage05" / "cells.json"
            tree_dirs.append(args.base_bundle_dir)
        cells_json_path = args.prior_cells or auto_cells_json
        if cells_json_path is None or not cells_json_path.exists():
            ui.err(
                "err",
                "--mode active requires --prior-cells, or a --base-bundle-dir or "
                "--prior-round-dir with stage05/cells.json; got prior_cells="
                f"{args.prior_cells} base_bundle_dir={args.base_bundle_dir} "
                f"prior_round_dirs={prior_round_dirs}",
            )
            return 1
        try:
            split_tree = sc.load_cumulative_split_tree(tree_dirs)
            revoked = (
                sc.revoked_parents(args.base_bundle_dir)
                if args.base_bundle_dir is not None
                else []
            )
        except ValueError as e:
            ui.err("err", str(e))
            return 1
        tails_from = args.targeted_from or (
            prior_round_dirs[-1] / "stage04b" / "decisions.json"
            if prior_round_dirs
            else None
        )
        plan = _build_active_plan(cells_json_path, args.shapes_per_cell, split_tree)
        if args.only_cells is not None:
            only = [c.strip() for c in args.only_cells.split(",") if c.strip()]
            plan, unknown = restrict_plan(plan, only, args.shapes_per_cell, split_tree)
            if not only:
                ui.err("err", "--only-cells names no cell")
                return 1
            if unknown:
                ui.err("err", f"--only-cells: unknown cell label(s) {unknown}")
                return 1
            plan_budget = need_budget(plan, None, args.shapes_per_cell)
            plan_budget.update(decisions=None, not_applied="--only-cells")
        else:
            decisions_from = args.decisions_from or (
                prior_round_dirs[-1] / "stage04b" / "decisions.json"
                if prior_round_dirs
                else None
            )
            if args.decisions_from is not None and not args.decisions_from.is_file():
                ui.err("err", f"--decisions-from {args.decisions_from}: no such file")
                return 1
            neutral = args.weak_cell_factor == 1 and args.strong_cell_factor == 1
            decisions = None
            if not neutral:
                try:
                    decisions = read_decisions(decisions_from)
                except ValueError as e:
                    ui.err("err", str(e))
                    return 1
            plan_budget = need_budget(
                plan,
                decisions,
                args.shapes_per_cell,
                weak_factor=args.weak_cell_factor,
                strong_factor=args.strong_cell_factor,
                strong_margin=args.strong_cell_margin,
                min_per_cell=int(args.min_cell_gemms or 0),
            )
            plan_budget.update(
                decisions=str(decisions_from) if decisions is not None else None,
                not_applied=(
                    None
                    if decisions is not None
                    else "every factor at 1" if neutral else "no stage04b decisions"
                ),
            )
        per_cell = {cell: int(info["n_target"]) for cell, info in plan.items()}
        args.prior_shapes_yaml = list(args.prior_shapes_yaml or []) + auto_shapes
        args.prior_enriched_csv_dir = (
            list(args.prior_enriched_csv_dir or []) + auto_enriched
        )
        if not args.quiet:
            ui.ok(
                "auto",
                f"from --prior-round-dir: {len(auto_shapes)} shapes.yaml, "
                f"{len(auto_enriched)} enriched dirs, cells.json={cells_json_path}",
            )
            if split_tree:
                n_extra = sum(
                    len(sc.leaves_under(c, split_tree)) - 1 for c in split_tree
                )
                ui.ok(
                    "splits",
                    f"cumulative split tree: {len(split_tree)} split cells "
                    f"-> {n_extra} extra leaves in the plan",
                )
            if revoked:
                ui.ok(
                    "splits",
                    f"left out, as {args.base_bundle_dir.name} revoked them: the "
                    f"splits of {', '.join(revoked)}",
                )
            n_with = sum(1 for p in plan.values() if p["has_prior_model"])
            ui.ok(
                "plan",
                f"leaves with a prior model={n_with}  without={len(plan) - n_with}  "
                f"shapes_per_cell={args.shapes_per_cell}  "
                f"total_target={sum(per_cell.values()):,}",
            )
            if plan_budget["applied"]:
                ui.ok(
                    "budget",
                    f"weak={plan_budget['n_weak']} (x{plan_budget['weak_cell_factor']:g}, "
                    f"+{plan_budget['weak_extra']:,})  strong={plan_budget['n_strong']} "
                    f"(x{plan_budget['strong_cell_factor']:g}, -{plan_budget['strong_cut']:,} of "
                    f"{plan_budget['strong_cut_wanted']:,})  "
                    f"total_target={plan_budget['total_target']:,} (uniform "
                    f"{plan_budget['uniform_total']:,}) from {plan_budget['decisions']}",
                )
            else:
                ui.info(
                    "budget",
                    f"{args.shapes_per_cell} per leaf: {plan_budget['not_applied']}",
                )

    if not args.quiet:
        ui.banner("Stage 2/3  Synthesize + filter + dedup", ui.C.BLUE)
    history = collect_history(
        args.prior_shapes_yaml or [], args.prior_enriched_csv_dir or [], args.quiet
    )
    if not args.quiet:
        if history:
            ui.ok("dedup", f"history shapes: {len(history):,}")
        ui.info(
            "filter",
            f"device memory {budget.device_bytes / 1024**3:.1f} GiB "
            f"({budget.source}); footprint budget "
            f"{budget.budget_bytes / 1024**3:.1f} GiB "
            f"({MEMORY_SAFE_FRACTION:.0%})  exclude rules: {len(shape_filter.exclude)}",
        )
        if budget.source != "config":
            ui.warn(
                "filter",
                "bench.device_memory_gib is not set; set it to make shape "
                "generation independent of the host",
            )
    held, errors = hol.read_all(args.exclude_shapes_yaml or [])
    for e in errors:
        ui.err("err", f"--exclude-shapes-yaml {e}")
    if errors:
        return 1
    excluded: Set[ShapeTuple] = set()
    for lst in held:
        excluded |= lst.gemms
    if excluded and not args.quiet:
        ui.ok(
            "exclude",
            f"excluded shapes: {len(excluded):,} from {len(held)} held-out list(s)",
        )
    t0 = time.time()
    result = sample_shapes(
        per_cell,
        round_seed,
        history,
        shape_filter,
        n_strata=args.n_strata,
        min_per_cell=int(args.min_cell_gemms or 0),
        excluded=excluded,
    )
    leaked = set(result.shapes) & excluded
    if leaked:
        ui.err(
            "exclude",
            f"{len(leaked)} drawn shapes are held out (e.g. {sorted(leaked)[0]}); "
            f"nothing written",
        )
        return 1
    sources = [
        lst.record(
            n_dropped=len(lst.gemms & result.excluded_hits),
            n_in_history=len(lst.gemms & history),
        )
        for lst in held
    ]
    for src in sources:
        if src["n_in_history"]:
            ui.warn(
                "exclude",
                f"{src['entry']}: {src['n_in_history']} of its {src['n_gemms']} "
                f"GEMMs are in the history: rounds drew them before the list "
                f"held them out, so they are training data",
            )
    if not args.quiet:
        ui.ok(
            "sample",
            f"sampled={len(result.shapes):,}  reused_history={len(result.reused):,}  "
            f"history_dedup_dropped={result.n_history_dropped:,}  "
            f"held_out_dropped={result.n_excluded:,}  "
            f"({time.time() - t0:.1f}s)",
        )
        n_populated = sum(1 for v in result.actual_per_cell.values() if v > 0)
        ui.ok("sample", f"populated cells: {n_populated}/{len(per_cell)}")
        gaps = [
            (c, t, result.actual_per_cell.get(c, 0))
            for c, t in per_cell.items()
            if t > 0 and result.actual_per_cell.get(c, 0) < t
        ]
        if gaps:
            zeros = sum(1 for _, _, a in gaps if a == 0)
            ui.warn(
                "sample",
                f"{len(gaps)} cells under target ({zeros} with no shape the "
                f"filter accepts)",
            )

    k_factor = (
        args.large_k_factor
        if args.mode == "active" and args.only_cells is None
        else 1.0
    )
    large_k: Dict[str, List[ShapeTuple]] = {}
    if k_factor > 1:
        large_k = large_k_shapes(
            result.fresh_per_cell,
            k_factor,
            round_seed,
            shape_filter,
            n_strata=args.n_strata,
            taken=set(history) | excluded | set(result.shapes),
        )
    stratum = [s for leaf in sorted(large_k) for s in large_k[leaf]]
    leaked = set(stratum) & excluded
    if leaked:
        ui.err(
            "exclude",
            f"{len(leaked)} very-large-K shapes are held out "
            f"(e.g. {sorted(leaked)[0]}); nothing written",
        )
        return 1
    if not args.quiet and stratum:
        ui.ok(
            "large_k",
            f"{len(stratum):,} more shapes with K above {LARGE_K_ABOVE} in "
            f"{len(large_k)} leaves (x{k_factor:g})",
        )

    targeted: Dict[str, List[Dict[str, Any]]] = {}
    tails: List[Dict[str, Any]] = []
    if args.mode == "active" and args.targeted_per_gemm > 0:
        tails = read_tails(tails_from)
        if tails:
            targeted = targeted_shapes(
                tails,
                {c for c, v in per_cell.items() if v > 0},
                split_tree,
                round_seed,
                shape_filter,
                per_gemm=args.targeted_per_gemm,
                max_per_leaf=args.targeted_max_per_leaf,
                taken=set(history) | excluded | set(result.shapes) | set(stratum),
            )
    n_tail_shapes = sum(len(e["shapes"]) for es in targeted.values() for e in es)
    seeded: Dict[str, List[Dict[str, Any]]] = {}
    seeded_skipped: List[Dict[str, Any]] = []
    if args.mode == "active" and anchors:
        seeded, seeded_skipped = seeded_shapes(
            anchors,
            {c for c, v in per_cell.items() if v > 0},
            split_tree,
            round_seed,
            shape_filter,
            per_target=args.targeted_seed_per_target,
            taken=set(history)
            | excluded
            | set(result.shapes)
            | set(stratum)
            | {tuple(s) for es in targeted.values() for e in es for s in e["shapes"]},
        )
        for leaf, entries in seeded.items():
            targeted.setdefault(leaf, []).extend(entries)
    seeded_per_cell = {
        leaf: sum(len(e["shapes"]) for e in entries) for leaf, entries in seeded.items()
    }
    n_seeded_shapes = sum(seeded_per_cell.values())
    targeted_per_cell = {
        leaf: sum(len(e["shapes"]) for e in entries)
        for leaf, entries in targeted.items()
    }
    extra = [tuple(s) for es in targeted.values() for e in es for s in e["shapes"]]
    leaked = set(extra) & excluded
    if leaked:
        ui.err(
            "exclude",
            f"{len(leaked)} targeted shapes are held out (e.g. {sorted(leaked)[0]}); "
            f"nothing written",
        )
        return 1
    if not args.quiet and args.mode == "active" and args.targeted_per_gemm > 0:
        n_tails = sum(1 for es in targeted.values() for e in es if "tail" in e)
        ui.ok(
            "targeted",
            f"{n_tail_shapes:,} shapes near {n_tails} "
            f"of {len(tails)} tail GEMMs "
            f"({tails_from if tails else 'no tail GEMMs'})",
        )
    if not args.quiet and args.mode == "active" and anchors:
        ui.ok(
            "seeded",
            f"{n_seeded_shapes:,} shapes near "
            f"{sum(len(v) for v in seeded.values())} of {len(anchors)} anchors "
            f"in {len(seeded)} leaves ({args.targeted_seed}"
            + (f", {len(seeded_skipped)} without shapes" if seeded_skipped else "")
            + ")",
        )

    seeded_record = (
        {
            "list": str(args.targeted_seed),
            "sha256": seeded_sha256,
            "library": args.targeted_seed_library,
            "per_target": args.targeted_seed_per_target,
            "n_anchors": len(anchors),
            "n_anchors_drawn": sum(len(v) for v in seeded.values()),
            "n_shapes": n_seeded_shapes,
            "per_cell": seeded_per_cell,
            "skipped": [
                {k: a.get(k) for k in ("row", "m", "n", "k", "batch", "leaf", "why")}
                for a in seeded_skipped
            ],
        }
        if args.targeted_seed is not None
        else None
    )
    if not args.quiet:
        ui.banner("Stage 3/3  Write outputs", ui.C.BLUE)
    shape_objs = [
        Shape(m=s[0], n=s[1], k=s[2], batch=s[3])
        for s in result.shapes + stratum + extra
    ]
    # stage03 benches contiguous blocks of this file, one process each, and
    # finishes with its slowest block; a shuffled file makes every block a
    # random sample of the round instead of a run of similar shapes.
    random.Random(derive_seed(round_seed, "shuffle")).shuffle(shape_objs)
    out_yaml = args.out_dir / "shapes.yaml"
    n_written = write_bench_yaml(shape_objs, cfg, out_yaml)
    write_reused_shapes(args.out_dir / REUSED_SHAPES_FILE, result.reused, round_index)
    if not args.quiet:
        ui.ok("write", f"shapes.yaml: {n_written:,} lines -> {out_yaml}")
        ui.ok("write", f"{REUSED_SHAPES_FILE}: {len(result.reused):,} shapes")

    write_json(
        args.out_dir / "categorization.json",
        {
            "mode": args.mode,
            "seed": args.seed,
            "round_index": round_index,
            "round_seed": round_seed,
            "config_yaml": str(args.config_yaml),
            "n_final": len(result.shapes) + len(stratum) + len(extra),
            "n_reused": len(result.reused),
            "n_history_dropped": result.n_history_dropped,
            "exclude_shapes_yaml": [str(p) for p in (args.exclude_shapes_yaml or [])],
            "n_excluded_shapes": len(excluded),
            "n_excluded_dropped": result.n_excluded,
            "exclusion_sources": sources,
            "memory_budget": {
                "source": budget.source,
                "device_bytes": budget.device_bytes,
                "budget_bytes": budget.budget_bytes,
            },
            "exclude": [dict(rule) for rule in shape_filter.exclude],
            "n_strata": args.n_strata,
            "min_cell_gemms": int(args.min_cell_gemms or 0),
            "target_per_cell_default": (
                args.target_per_cell if args.mode == "initial" else args.shapes_per_cell
            ),
            "target_per_cell": per_cell,
            "actual_per_cell": result.actual_per_cell,
            "reused_per_cell": result.reused_per_cell,
            "targeted": {
                "per_gemm": args.targeted_per_gemm,
                "max_per_leaf": args.targeted_max_per_leaf,
                "source": str(tails_from) if tails else None,
                "n_tail_gemms": len(tails),
                "n_shapes": len(extra),
                "per_cell": targeted_per_cell,
                **({"seeded": seeded_record} if seeded_record is not None else {}),
            },
            "large_k": {
                "factor": k_factor,
                "above": LARGE_K_ABOVE,
                "n_shapes": len(stratum),
                "per_cell": {leaf: len(v) for leaf, v in sorted(large_k.items())},
            },
        },
    )
    targeted_path = args.out_dir / TARGETED_FILE
    if targeted:
        write_json(
            targeted_path,
            {
                "source": str(tails_from),
                "per_gemm": args.targeted_per_gemm,
                "max_per_leaf": args.targeted_max_per_leaf,
                "n_shapes": len(extra),
                **({"seeded": seeded_record} if seeded_record is not None else {}),
                "leaves": targeted,
            },
        )
    else:
        targeted_path.unlink(missing_ok=True)
    if not args.quiet:
        ui.ok("write", "categorization.json")

    if args.mode == "active" and plan is not None:
        write_json(
            args.out_dir / "per_cell_plan.json",
            {
                "shapes_per_cell": args.shapes_per_cell,
                "only_cells": args.only_cells,
                "prior_cells_json": str(cells_json_path),
                "prior_round_dirs": [str(p) for p in (args.prior_round_dir or [])],
                "base_bundle_dir": (
                    str(args.base_bundle_dir) if args.base_bundle_dir else None
                ),
                "revoked_by_base_bundle": revoked,
                "prior_shapes_yaml": [str(p) for p in (args.prior_shapes_yaml or [])],
                "prior_enriched_csv_dir": [
                    str(p) for p in (args.prior_enriched_csv_dir or [])
                ],
                "n_history_shapes": len(history),
                "budget": plan_budget,
                "cells": {
                    cell: {
                        **info,
                        "n_actual": result.actual_per_cell.get(cell, 0),
                        "n_reused": result.reused_per_cell.get(cell, 0),
                        "n_targeted": targeted_per_cell.get(cell, 0),
                        "n_seeded": seeded_per_cell.get(cell, 0),
                        "n_large_k": len(large_k.get(cell, ())),
                    }
                    for cell, info in plan.items()
                },
            },
        )
        if not args.quiet:
            ui.ok("write", "per_cell_plan.json")

    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.ok(
            "done",
            f"total shapes  : {len(result.shapes) + len(stratum) + len(extra):,}",
        )
        ui.ok("done", f"output dir    : {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
