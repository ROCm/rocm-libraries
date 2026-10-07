# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The training / validation split of a round's stage05, recovered from its
outputs.

stage05 splits every trained cell's GEMMs with `split_validation` (a hash
of `--seed` and the GEMM identity against `--validation-frac`; a cell whose
split leaves one side empty trains on all of its GEMMs). Its metrics.json
records the arguments (fraction, seed, data folders, round dirs of the
split tree) and its cells.json the trained cells with their training and
validation counts. `recover` routes bench GEMMs (`run_analysis.Gemm`) to
leaves with stage05's own routing and split tree, applies stage05's own
`split_validation` per trained cell and calls the split exact only when
every trained cell's counts and the data folders agree with the outputs.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import subcells as sc

STAGES_DIR = Path(__file__).resolve().parents[1] / "stages"


def stage05_module() -> Any:
    if str(STAGES_DIR) not in sys.path:
        sys.path.insert(0, str(STAGES_DIR))
    import stage05_train

    return stage05_train


@dataclass
class Split:
    """`train` / `validation`: the GEMMs of trained cells on each side;
    `untrained`: GEMMs of cells stage05 did not train. `exact` is False
    with the first disagreement in `reason`."""

    train: List[Any] = field(default_factory=list)
    validation: List[Any] = field(default_factory=list)
    untrained: List[Any] = field(default_factory=list)
    exact: bool = False
    reason: Optional[str] = None
    validation_frac: Optional[float] = None
    seed: Optional[int] = None
    n_cells: int = 0
    mismatched_cells: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "exact": self.exact,
            "reason": self.reason,
            "validation_frac": self.validation_frac,
            "seed": self.seed,
            "n_cells": self.n_cells,
            "n_train": len(self.train),
            "n_validation": len(self.validation),
            "n_untrained_cells_gemms": len(self.untrained),
            "mismatched_cells": self.mismatched_cells[:20],
        }


def _same_dirs(a: Sequence[Any], b: Sequence[Any]) -> bool:
    return sorted(os.path.realpath(str(p)) for p in a) == sorted(
        os.path.realpath(str(p)) for p in b
    )


def recover(
    stage05_dir: Path,
    gemms: Sequence[Any],
    *,
    data_dirs: Optional[Sequence[Path]] = None,
    s5: Any = None,
) -> Split:
    """Split `gemms` (the training-side bench GEMMs, read from `data_dirs`)
    as `stage05_dir`'s stage05 did."""
    out = Split()
    try:
        met = json.loads((Path(stage05_dir) / "metrics.json").read_text())
        cells = json.loads((Path(stage05_dir) / "cells.json").read_text())
    except (OSError, ValueError) as e:
        out.reason = f"stage05 outputs unreadable ({e})"
        return out
    args = met.get("args") or {}
    try:
        out.validation_frac = float(args["validation_frac"])
        out.seed = int(args["seed"])
    except (KeyError, TypeError, ValueError):
        out.reason = "metrics.json records no validation_frac / seed"
        return out
    if out.validation_frac <= 0:
        out.reason = "stage05 kept no validation GEMMs (validation_frac 0)"
        return out
    trained = {c["label"]: c for c in cells.get("cells") or [] if "label" in c}
    out.n_cells = len(trained)
    if not trained:
        out.reason = "cells.json lists no trained cell"
        return out
    recorded = args.get("enriched_csv_dir") or []
    if data_dirs is not None and not _same_dirs(recorded, data_dirs):
        out.reason = "stage05 read other data folders than the training-side bench"
        return out

    s5 = s5 or stage05_module()
    tree = sc.load_cumulative_split_tree(args.get("prior_round_dir") or [])
    if args.get("current_round_dir"):
        tree.update(sc.load_cumulative_split_tree([args["current_round_dir"]]))
    by_leaf: Dict[str, List[Any]] = {}
    for g in gemms:
        m, n, k, b = (int(v) for v in g.key[:4])
        by_leaf.setdefault(s5._leaf_of(m, n, k, b, tree), []).append(g)

    for leaf, group in by_leaf.items():
        entry = trained.get(leaf)
        if entry is None:
            out.untrained.extend(group)
            continue
        rows = [g.row for g in group]
        tr, va = s5.split_validation(rows, out.validation_frac, out.seed)
        val_ids = {id(r) for r in va}
        for g, r in zip(group, rows):
            (out.validation if id(r) in val_ids else out.train).append(g)
        if (len(tr), len(va)) != (
            int(entry.get("n_train_gemms", -1)),
            int(entry.get("n_val_gemms", -1)),
        ):
            out.mismatched_cells.append(leaf)
    missing = sorted(set(trained) - set(by_leaf))
    out.mismatched_cells += missing
    if out.mismatched_cells:
        out.reason = (
            f"{len(out.mismatched_cells)} of {len(trained)} trained cells have "
            f"other training / validation counts than cells.json"
        )
        return out
    out.exact = True
    return out
