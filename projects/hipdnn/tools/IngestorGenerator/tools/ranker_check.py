# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Ranker check of a strided sweep: does a shipped ranker pick the catalog's best
kernel on strided graphs as well as it does on the BSHD twin?

One ``tools/sweep.py`` run holds two arms over the same catalog: ``--ranker-arm`` with
benchmarking off (the engine serves the ranker's top-1) and ``--best-arm`` with forced
benchmarking (it serves the catalog's best). Per graph, regret = mean_ms(ranker) /
mean_ms(best) - 1, each the benchmark's own mean for the kernel it served, medians over
rounds. A cell (dtype, head size, mask, strided layout) is MIS-RANKED when its geomean
regret exceeds its BSHD twin's by more than REGRET_DELTA_LIMIT, or when any of its
graphs has regret above REGRET_GRAPH_LIMIT. The twin is the BSHD graphs of the same
shapes as the cell's own graphs, so a shape the layout repeats (``same_as_bshd``) is in
neither side. The twin's own regret is what the ranker already costs; only the
difference is blamed on strides. A layout whose every graph repeats its twin is no
cell: it is listed under ``repeats_twin``.

Exit 0 (``--strict``: 1 when a cell is MIS-RANKED or incomplete), 2 on unusable input.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_sweep_report import (  # noqa: E402
    geomean,
    load_map,
    median_over_rounds,
    served_times,
)
from variant_reduce import ReduceError  # noqa: E402

# Pre-registered before any data (D62); do not tune after seeing results.
REGRET_DELTA_LIMIT = 0.03
REGRET_GRAPH_LIMIT = 0.10


def regrets(ranker: dict, best: dict, mapping: list) -> dict:
    """``{graph: ratio}`` of ranker time to best time for graphs both arms served."""
    shared = {m["graph_name"] for m in mapping} & ranker.keys() & best.keys()
    return {g: ranker[g] / best[g] for g in shared}


def check(ratios: dict, mapping: list) -> list:
    """One verdict per strided cell, against the BSHD twin of the same family and
    shapes. A cell whose graphs all repeat the twin is left out (see `repeats_twin`)."""
    cells = defaultdict(list)
    for m in mapping:
        f = m["family"]
        key = (f["dtype"], f["head_size"], f["mask_type"])
        cells[key, m["layout"]].append(m)
    out = []
    for (family, layout), members in cells.items():
        if layout == "bshd":
            continue
        members = [m for m in members if not m.get("same_as_bshd")]
        if not members:
            continue
        twins = {m["shape"]: m for m in cells.get((family, "bshd"), [])}
        pairs = [(m, twins.get(m["shape"])) for m in members]
        have = [(m, t) for m, t in pairs if m["graph_name"] in ratios]
        row = dict(cell=[*family, layout], graphs=len(members), measured=len(have))
        if len(have) < len(members) or any(
            t is None or t["graph_name"] not in ratios for _, t in pairs
        ):
            out.append(row | dict(verdict="INCOMPLETE"))
            continue
        mine = [ratios[m["graph_name"]] for m, _ in have]
        twin = [ratios[t["graph_name"]] for _, t in have]
        delta = geomean(mine) - geomean(twin)
        worst = [
            m["graph_name"]
            for m in members
            if ratios[m["graph_name"]] > 1 + REGRET_GRAPH_LIMIT
        ]
        bad = delta > REGRET_DELTA_LIMIT or bool(worst)
        out.append(
            row
            | dict(
                regret=geomean(mine) - 1,
                twin_regret=geomean(twin) - 1,
                delta=delta,
                graphs_over_limit=worst,
                verdict="MIS-RANKED" if bad else "OK",
            )
        )
    return out


def repeats_twin(mapping: list) -> list:
    """The graphs of a strided layout that repeat their BSHD twin's measurement."""
    return sorted(
        m["graph_name"]
        for m in mapping
        if m["layout"] != "bshd" and m.get("same_as_bshd", False)
    )


def main(argv: list[str] | None = None) -> int:
    """Compare the two arms of a sweep; the exit code is in the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--ranker-arm", required=True)
    parser.add_argument("--best-arm", required=True)
    parser.add_argument("--map", required=True, help="`ground_truth.py layouts` map")
    parser.add_argument("--out", required=True, help="YAML")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    try:
        mapping = load_map(args.map)
        arms = (args.ranker_arm, args.best_arm)
        ranker, best = (
            median_over_rounds(served_times(args.sweep_output, a)) for a in arms
        )
        ratios = regrets(ranker, best, mapping)
        if not ratios:
            raise ReduceError("the two arms served no mapped graph in common")
        cells = check(ratios, mapping)
    except (ReduceError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    import yaml

    result = dict(
        limits=dict(delta=REGRET_DELTA_LIMIT, graph=REGRET_GRAPH_LIMIT),
        cells=cells,
        repeats_twin=repeats_twin(mapping),
        graph_ratios=dict(sorted(ratios.items())),
    )
    Path(args.out).write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    verdicts = {c["verdict"] for c in cells}
    print(f"cells: {sorted(verdicts)}")
    return 1 if args.strict and verdicts != {"OK"} else 0


if __name__ == "__main__":
    sys.exit(main())
