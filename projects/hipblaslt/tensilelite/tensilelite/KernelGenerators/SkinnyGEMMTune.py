# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Re-tunes the routing of the skinny MXFP8 GEMM kernels: benchmark, cells, rows, logic file.

Run from projects/hipblaslt/tensilelite. All commands take a results directory that holds
the run log (results.jsonl), the raw bench outputs and the reports, so every step can be
interrupted and resumed, and a re-run only measures what is missing.

  tune      Benchmarks the grid (data/SkinnyGEMMGrid.yaml) with hipblaslt-bench on the given GPUs,
            builds the routing cells, samples and benchmarks held-out points inside them,
            validates the cells offline against the held-out points, rebuilds the cells from
            all measurements, writes the rows file (data/SkinnyGEMMRows.yaml) and regenerates the
            logic file through SkinnyGEMMLogic. Needs a hipBLASLt build with the skinny pool;
            the benches run with TENSILE_PREDICTION_LIB=1, which skips Equality and Range rows,
            so a library that already has routing rows can be re-benchmarked.

      python -m tensilelite.KernelGenerators.SkinnyGEMMTune tune --bench PATH --results DIR \\
          [--gpus 0-7] [--grid GRID.yaml] [--margin 1.25] [--tolerance 1.10] [--regression 1.05] \\
          [--rows ROWS.yaml] [--output LOGIC.yaml]

  cells     The same, from the results already in the directory and without a GPU. Use it to try
            other thresholds, or to rebuild the rows file and the logic from a campaign.

      python -m tensilelite.KernelGenerators.SkinnyGEMMTune cells --results DIR [thresholds] [--rows ...] [--output ...]

  validate  Run after rebuilding hipBLASLt with the new logic. Benchmarks the library's own default
            pick (without TENSILE_PREDICTION_LIB) on every held-out point, up to 200 routed and 50
            non-routed lattice points, and compares it with the old default timed in the same
            process. Exits with status 1 on a regression or an unexpected pick; add such points
            to extra_points of the grid and run tune again.

      python -m tensilelite.KernelGenerators.SkinnyGEMMTune validate --bench PATH --results DIR [--gpus 0-7]

Thresholds: a pool entry serves a cell only where it is at least --margin faster than Origami's
default at every corner of the cell, within --tolerance of the best generated kernel there, and
never --regression slower than Origami at any other measured point inside the cell.

Reports written to the results directory: cells.md, points.md/csv and summary.md (tune, cells),
validate.md/csv (validate). After tune or cells, rebuild hipBLASLt (see README.md here).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from . import SkinnyGEMMLogic
from .SkinnyGEMMBench import (PHASES, Grid, Point, ResultsDir, Runner, lattice_points, load_grid, order_jobs,
                              parse_gpus)
from .SkinnyGEMMCells import (Cell, CellBuilder, Measurements, Route, Thresholds, aggregate, assert_no_regression,
                              make_routes, refinement_points, sample_held_out, validate_held_out)
from .SkinnyGEMMReport import (analyze_validation, cell_table, csv_text, describe_rows, held_out_lines, markdown_table,
                               point_stats, point_table, read_routes, render_rows, validation_points, validation_refinement,
                               validation_summary, validation_table)

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_GRID = DATA_DIR / "SkinnyGEMMGrid.yaml"
INTERRUPTED = 130

REBUILD_STEPS = """Remaining manual step: rebuild hipBLASLt with the new logic, then run validate.
  cd projects/hipblaslt
  rm -f build/Tensile/library/gfx950/*_Generated* build/install/lib/hipblaslt/library/gfx950/*_Generated*
  cmake --build build -j 200 && cmake --install build
(the library build does not remove stale generated code objects, hence the rm)"""


class Campaign:
    """Runs the phases of a campaign on the GPUs and re-reads the results between them."""

    def __init__(self, bench: str, grid: Grid, gpus: Sequence[int], results: ResultsDir):
        self.grid = grid
        self.results = results
        self.runner = Runner([bench], grid.bench, gpus, results)

    def measurements(self) -> Measurements:
        return aggregate(self.results.records(), self.results.table)

    def run_phase(self, phase: str, points: Sequence[Point], refine: Callable[[], List[Point]]) -> None:
        """Runs the base repeats of the points, then the extra repeats of the ones the noise leaves undecided."""
        params = self.grid.bench
        seed = PHASES.index(phase)
        start = time.monotonic()
        failed = self.runner.run(order_jobs(phase, points, range(params.repeats), seed))
        undecided = refine()
        print(f"{phase}: refining {len(undecided)} of {len(points)} points")
        failed += self.runner.run(order_jobs(phase, undecided, range(params.repeats, params.refine_repeats), seed))
        print(f"{phase}: {time.monotonic() - start:.0f} s, {failed} failed jobs")


def _grid_cells(grid: Grid, measurements: Measurements, limits: Thresholds, pool: SkinnyGEMMLogic.Pool) -> List[Cell]:
    return CellBuilder(grid, measurements, limits, pool, include_held_out=False).build()


def _final_routes(grid: Grid, measurements: Measurements, limits: Thresholds, pool: SkinnyGEMMLogic.Pool) -> List[Route]:
    cells = CellBuilder(grid, measurements, limits, pool, include_held_out=True).build()
    routes = make_routes(cells)
    assert_no_regression(routes, measurements, limits)
    return routes


def _run_counts(results: ResultsDir) -> List[str]:
    lines = []
    for phase in PHASES:
        runs = [run for run in results.records() if run.phase == phase]
        failed = sum(run.return_code != 0 for run in runs)
        lines.append(f"- {phase}: {len(runs)} runs ({failed} failed), {sum(run.wall_seconds for run in runs):.0f} run-seconds")
    return lines


def _summary(grid: Grid, grid_name: str, limits: Thresholds, results: ResultsDir, routes: Sequence[Route],
             held_out_lines_text: Sequence[str], points) -> str:
    lines = ["# Skinny GEMM routing summary", "", "## Grid", f"- file: {grid_name}",
             f"- orientations {list(grid.orientations)}, skinny {list(grid.skinny)}, wide {list(grid.wide)}, k {list(grid.k)}",
             f"- extra points: {len(grid.extra_points)}, held-out: {grid.held_out_count} (seed {grid.held_out_seed})",
             f"- bench: {grid.bench}", "", "## Thresholds", f"- {limits}", "", "## Runs", *_run_counts(results), "",
             "## Result", f"- rows: {len(routes)}", *point_stats(points, "grid"), *point_stats(points, "held_out"),
             "", "## Held-out validation", *held_out_lines_text,
             "- the final cells have no regression on any measured point (asserted)"]
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def finish(args: argparse.Namespace, grid: Grid, pool: SkinnyGEMMLogic.Pool, results: ResultsDir) -> List[Route]:
    """Cells, held-out validation, rows file, logic file and reports from the results in the directory."""
    limits = Thresholds(args.margin, args.tolerance, args.regression)
    measurements = aggregate(results.records(), results.table)
    grid_cells = _grid_cells(grid, measurements, limits, pool)
    held_out = validate_held_out(make_routes(grid_cells), measurements, limits, sample_held_out(grid_cells, grid))
    routes = _final_routes(grid, measurements, limits, pool)
    grid_name = Path(args.grid).name
    _write(args.rows, render_rows(routes, describe_rows(grid_name, limits)))
    SkinnyGEMMLogic.main(["--rows", str(args.rows), "--output", str(args.output)])
    points = point_table(routes, measurements)
    _write(results.path / "cells.md", markdown_table(cell_table(routes, points)))
    _write(results.path / "points.md", markdown_table(points))
    _write(results.path / "points.csv", csv_text(points))
    _write(results.path / "summary.md", _summary(grid, grid_name, limits, results, routes, held_out_lines(held_out), points))
    print(f"{len(routes)} rows written to {args.rows}, logic written to {args.output}, reports in {results.path}")
    return routes


def _open_results(args: argparse.Namespace) -> Tuple[Grid, SkinnyGEMMLogic.Pool, ResultsDir]:
    pool = SkinnyGEMMLogic.buildPool()
    return load_grid(args.grid), pool, ResultsDir(args.results, pool)


def tune(args: argparse.Namespace) -> int:
    grid, pool, results = _open_results(args)
    limits = Thresholds(args.margin, args.tolerance, args.regression)
    campaign = Campaign(args.bench, grid, parse_gpus(args.gpus), results)
    campaign.run_phase("grid", lattice_points(grid) + list(grid.extra_points),
                      lambda: refinement_points(campaign.measurements(), "grid"))
    grid_cells = _grid_cells(grid, campaign.measurements(), limits, pool)
    grid_routes = make_routes(grid_cells)
    campaign.run_phase("held_out", sample_held_out(grid_cells, grid),
                      lambda: refinement_points(campaign.measurements(), "held_out", grid_routes))
    finish(args, grid, pool, results)
    print(REBUILD_STEPS)
    return 0


def cells(args: argparse.Namespace) -> int:
    grid, pool, results = _open_results(args)
    finish(args, grid, pool, results)
    print(REBUILD_STEPS)
    return 0


def validate(args: argparse.Namespace) -> int:
    grid, pool, results = _open_results(args)
    limits = Thresholds(regression=args.regression)
    routes = read_routes(args.rows, pool)
    campaign = Campaign(args.bench, grid, parse_gpus(args.gpus), results)
    measurements = campaign.measurements()

    def analyze():
        return analyze_validation(results.records(), measurements, routes, results.table)

    campaign.run_phase("validate", validation_points(measurements, routes, grid.held_out_seed),
                      lambda: validation_refinement(analyze()))
    rows = analyze()
    table = validation_table(rows, limits)
    _write(results.path / "validate.md", markdown_table(table))
    _write(results.path / "validate.csv", csv_text(table))
    lines = validation_summary(rows, limits)
    _write(results.path / "validate_summary.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if any(row.regression(limits) or not row.pick_ok for row in rows) else 0


def _add_common(parser: argparse.ArgumentParser, *, build: bool, run: bool) -> None:
    parser.add_argument("--results", type=Path, required=True, help="results directory")
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID, help="grid file")
    parser.add_argument("--rows", type=Path, default=SkinnyGEMMLogic.DEFAULT_ROWS, help="routing rows file")
    parser.add_argument("--regression", type=float, default=Thresholds.regression,
                        help="slowdown against the old default that vetoes an entry")
    if run:
        parser.add_argument("--bench", required=True, help="hipblaslt-bench executable")
        parser.add_argument("--gpus", default="0-7", help="GPUs to use, such as 0-7 or 0,2,5")
    if build:
        parser.add_argument("--output", type=Path, default=SkinnyGEMMLogic.OUTPUT_LOGIC, help="logic file to write")
        parser.add_argument("--margin", type=float, default=Thresholds.margin,
                            help="Origami's default must be this much slower than the entry")
        parser.add_argument("--tolerance", type=float, default=Thresholds.tolerance,
                            help="an entry may be this much slower than the best generated kernel at a corner")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-tune the skinny MXFP8 GEMM routing.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function, build, run in (("tune", tune, True, True), ("cells", cells, True, False),
                                       ("validate", validate, False, True)):
        sub = commands.add_parser(name)
        _add_common(sub, build=build, run=run)
        sub.set_defaults(function=function)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        return args.function(args)
    except KeyboardInterrupt:
        print("interrupted: the completed runs are kept, run the same command again to resume", file=sys.stderr)
        return INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
