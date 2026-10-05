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
                   shapes).

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

Outputs in `--out-dir`:

  shapes.yaml          hipblaslt-bench yaml, shuffled so that stage03's
                       contiguous blocks are random samples of the round
  reused_shapes.yaml   shapes re-used from earlier rounds (see
                       `lib.shapes.read_reused_shapes`)
  categorization.json  per-cell targets and counts, seeds, filter settings
  per_cell_plan.json   active mode only: per-cell plan and prior sel_eff
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

THIS_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = THIS_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import yaml  # noqa: E402

from lib import subcells as sc  # noqa: E402
from lib import ui  # noqa: E402
from lib.bench_yaml import parse_shapes_yaml, write_bench_yaml  # noqa: E402
from lib.fs import read_json, write_json  # noqa: E402
from lib.grid import TIER_RANGES_K, TIER_RANGES_MN, all_cell_labels  # noqa: E402
from lib.shapes import (  # noqa: E402
    MEMORY_SAFE_FRACTION,
    REUSED_SHAPES_FILE,
    Shape,
    ShapeFilter,
    ShapeTuple,
    device_memory_budget,
    synthesize_cell_shapes,
    write_reused_shapes,
)

# Fresh-shape pools drawn per cell before falling back to history re-use.
_FRESH_POOLS = 3


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
    n_history_dropped: int = 0


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
) -> SampleResult:
    """Up to `per_cell_target[cell]` new shapes per cell (base label or
    sub-cell label), none of them in `history`. A cell left below
    `min_per_cell` is topped up, with new shapes first and history shapes
    after; the history shapes used are reported in `reused`. Each cell draws
    from its own RNG stream, so its shapes do not depend on the other cells'
    targets or their order."""
    out = SampleResult()
    chosen: Set[ShapeTuple] = set()
    for cell in sorted(per_cell_target):
        target = int(per_cell_target[cell])
        if target <= 0:
            out.actual_per_cell[cell] = 0
            continue
        rng = random.Random(derive_seed(round_seed, "cell", cell))
        base = sc.base_cell(cell)
        bounds = _axis_bounds(cell, base)

        def _pool(size: int) -> List[ShapeTuple]:
            return synthesize_cell_shapes(
                base, size, rng, shape_filter, axis_bounds=bounds, n_strata=n_strata
            )

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
        out.actual_per_cell[cell] = len(fresh) + len(reused)
        out.reused_per_cell[cell] = len(reused)
        out.n_history_dropped += len(history_hits)
    return out


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
        "found under --prior-round-dir.",
    )
    ap.add_argument(
        "--shapes-per-cell",
        type=int,
        default=150,
        help="(active mode) shapes per leaf with a prior model.",
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
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

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
        cells_json_path = args.prior_cells or auto_cells_json
        if cells_json_path is None or not cells_json_path.exists():
            ui.err(
                "err",
                "--mode active requires --prior-cells or a --prior-round-dir "
                "with stage05/cells.json; got prior_cells="
                f"{args.prior_cells} prior_round_dirs={prior_round_dirs}",
            )
            return 1
        split_tree = sc.load_cumulative_split_tree(prior_round_dirs)
        plan = _build_active_plan(cells_json_path, args.shapes_per_cell, split_tree)
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
            n_with = sum(1 for p in plan.values() if p["has_prior_model"])
            ui.ok(
                "plan",
                f"leaves with a prior model={n_with}  without={len(plan) - n_with}  "
                f"shapes_per_cell={args.shapes_per_cell}  "
                f"total_target={sum(per_cell.values()):,}",
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
    t0 = time.time()
    result = sample_shapes(
        per_cell,
        round_seed,
        history,
        shape_filter,
        n_strata=args.n_strata,
        min_per_cell=int(args.min_cell_gemms or 0),
    )
    if not args.quiet:
        ui.ok(
            "sample",
            f"sampled={len(result.shapes):,}  reused_history={len(result.reused):,}  "
            f"history_dedup_dropped={result.n_history_dropped:,}  "
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

    if not args.quiet:
        ui.banner("Stage 3/3  Write outputs", ui.C.BLUE)
    shape_objs = [Shape(m=s[0], n=s[1], k=s[2], batch=s[3]) for s in result.shapes]
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
            "n_final": len(result.shapes),
            "n_reused": len(result.reused),
            "n_history_dropped": result.n_history_dropped,
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
        },
    )
    if not args.quiet:
        ui.ok("write", "categorization.json")

    if args.mode == "active" and plan is not None:
        write_json(
            args.out_dir / "per_cell_plan.json",
            {
                "shapes_per_cell": args.shapes_per_cell,
                "prior_cells_json": str(cells_json_path),
                "prior_round_dirs": [str(p) for p in (args.prior_round_dir or [])],
                "prior_shapes_yaml": [str(p) for p in (args.prior_shapes_yaml or [])],
                "prior_enriched_csv_dir": [
                    str(p) for p in (args.prior_enriched_csv_dir or [])
                ],
                "n_history_shapes": len(history),
                "cells": {
                    cell: {
                        **info,
                        "n_actual": result.actual_per_cell.get(cell, 0),
                        "n_reused": result.reused_per_cell.get(cell, 0),
                    }
                    for cell, info in plan.items()
                },
            },
        )
        if not args.quiet:
            ui.ok("write", "per_cell_plan.json")

    if not args.quiet:
        ui.banner("Done", ui.C.GREEN)
        ui.ok("done", f"total shapes  : {len(result.shapes):,}")
        ui.ok("done", f"output dir    : {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
