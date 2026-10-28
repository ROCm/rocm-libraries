# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the skinny GEMM routing tools (KernelGenerators.SkinnyGEMMBench, Cells, Report, Tune); no GPU."""

import json
import math
import stat
import sys
import textwrap

import pytest
import yaml

pytestmark = pytest.mark.unit

from Tensile.KernelGenerators import (SkinnyGEMM, SkinnyGEMMBench, SkinnyGEMMCells, SkinnyGEMMLogic,
                                      SkinnyGEMMReport, SkinnyGEMMTune)
from Tensile.KernelGenerators.SkinnyGEMMBench import Job, Point, RunRecord

RING = (1, 1, 16, 1, 1, 0, 1, 1, 1, True, False)
RING_B = (1, 1, 16, 2, 1, 0, 1, 1, 1, True, False)
UP_FRONT = (1, 1, 64, 4, 2, 2, 0, 1, 1, True, False)
LIMITS = SkinnyGEMMCells.Thresholds()
GRID_DATA = {"orientations": ["n", "m"], "skinny": [1, 6, 4], "wide": [4096, 1024], "k": [1024, 4100],
             "extra_points": [], "held_out": {"count": 20, "seed": 3},
             "bench": {"warmup": 2, "iterations": 3, "rotating": 512, "repeats": 2, "refine_repeats": 4}}


def generated_name(knobs):
    bm, tn, lanesK, waves, prefetch, upIters, partialSteps, group, numXcds, grid3d, useBeta = knobs
    return (f"gemm_BM{bm}_TN{tn}_LANES_K{lanesK}_WAVES{waves}_UP_ITERS{upIters}_PARTIAL_STEPS{partialSteps}"
            f"_PF{prefetch}_GROUP{group}_NUM_XCDS{numXcds}_GRID_3D{int(grid3d)}_USE_BETA{int(useBeta)}")


def block_text(index, us, kernel):
    return (f"[0]:transA,transB,grouped_gemm,m,n,k,us\n    T,N,0,5,1,8,{us}\n"
            f"    --Solution index: {index}\n    --Solution name:  {kernel}\n    --kernel name:    {kernel}\n")


CANNED = ("Is supported 3 / Total solutions: 3\n" + block_text(7, 20.5, "Cijk_Origami7") + block_text(9, 3.25, generated_name(RING))
          + block_text(8, 11.0, "Cijk_Origami8") + block_text(9, 3.0, generated_name(RING)))


@pytest.fixture(scope="module")
def pool():
    return SkinnyGEMMLogic.buildPool()


def make_grid(**changes):
    return SkinnyGEMMBench.parse_grid({**GRID_DATA, **changes})


def make_table(tmp_path, pool):
    table = SkinnyGEMMBench.SolutionTable(tmp_path / "solutions.json", pool)
    table.names = {0: "Cijk_Origami", 1: generated_name(RING), 2: generated_name(RING_B), 3: generated_name(UP_FRONT)}
    return table


def record(point, times, phase="grid", repeat=0, first=0, prediction_lib=True, code=0):
    return RunRecord(phase, point, repeat, 0, 1.0, code, prediction_lib, first, dict(times))


class TestParser:
    def test_normal_and_duplicate(self):
        first, blocks = SkinnyGEMMBench.parse_output(CANNED)
        assert first == 7
        assert blocks == [(7, 20.5, "Cijk_Origami7"), (9, 3.25, generated_name(RING)), (8, 11.0, "Cijk_Origami8"),
                          (9, 3.0, generated_name(RING))]
        assert SkinnyGEMMBench.fastest_per_index(blocks) == {7: 20.5, 9: 3.0, 8: 11.0}

    @pytest.mark.parametrize("text", [
        "no blocks here\n",
        block_text(1, 2.0, "k").replace("--Solution index: 1", "--Solution index: x"),
        block_text(1, 2.0, "k").replace("N,0,5,1,8,2.0", "N,0,5,1,8,fast"),
        "[0]:transA,transB\n    T,N,1.0\n    --Solution index: 3\n",
    ])
    def test_malformed(self, text):
        with pytest.raises(ValueError):
            SkinnyGEMMBench.parse_output(text)

    def test_generated_name_to_knobs(self, pool):
        assert SkinnyGEMMBench.generated_knobs(generated_name(UP_FRONT), pool) == UP_FRONT
        assert SkinnyGEMMBench.generated_knobs("Cijk_Alik_Bljk", pool) is None

    def test_use_beta_is_parsed_from_the_name(self, pool):
        withBeta = SkinnyGEMMLogic.withBeta(RING, True)
        assert generated_name(withBeta).endswith("_USE_BETA1")
        assert SkinnyGEMMBench.generated_knobs(generated_name(withBeta), pool) == withBeta
        assert SkinnyGEMMBench.generated_knobs(generated_name(RING), pool) == RING

    def test_name_without_suffix_is_use_beta_false(self, pool):
        assert SkinnyGEMMBench.generated_knobs(generated_name(RING).removesuffix("_USE_BETA0"), pool) == RING

    def test_unknown_knobs_are_errors(self, pool):
        with pytest.raises(ValueError, match="not in the pool"):
            SkinnyGEMMBench.generated_knobs(generated_name((1, 1, 4, 1, 1, 0, 1, 1, 1, True, False)), pool)
        with pytest.raises(ValueError, match="unparsable"):
            SkinnyGEMMBench.generated_knobs("gemm_BM1_TN1", pool)

    def test_names_match_the_emitter(self, pool):
        codegen = pytest.importorskip("tensilelite_codegen")
        for knobs in pool:
            assert codegen.skinnyGEMMKernelName(codegen.SkinnyGEMMConfig(**SkinnyGEMMLogic.asParams(knobs))) == generated_name(knobs)

    def test_table_detects_a_changed_library(self, tmp_path, pool):
        table = make_table(tmp_path, pool)
        with pytest.raises(ValueError, match="library changed"):
            table.register([(0, 1.0, "Cijk_Other")])


class TestGrid:
    def test_loading_sorts_and_validates(self):
        grid = make_grid()
        assert grid.skinny == (1, 4, 6) and grid.wide == (1024, 4096)
        assert grid.bench.refine_repeats == 4

    @pytest.mark.parametrize("change", [{"skinny": [32]}, {"wide": [8]}, {"orientations": ["x"]},
                                        {"extra_points": [[1, 2, 3]]}, {"unknown": 1}])
    def test_invalid(self, change):
        with pytest.raises(ValueError):
            SkinnyGEMMBench.parse_grid({**GRID_DATA, **change})

    def test_lattice_and_ld_padding(self):
        points = SkinnyGEMMBench.lattice_points(make_grid())
        assert len(points) == 2 * 3 * 2 * 2 and len(set(points)) == len(points)
        assert Point(1024, 1, 1024) in points and Point(1, 1024, 1024, 4) in points
        assert Point(6, 4096, 4100, 8) in points
        assert all(p.ld is None for p in points if p.m > 16)

    def test_extra_points_are_padded(self):
        grid = make_grid(extra_points=[[3, 2000, 777], [2000, 3, 777]])
        assert grid.extra_points == (Point(3, 2000, 777, 4), Point(2000, 3, 777))

    def test_command(self):
        grid = make_grid()
        command = SkinnyGEMMBench.bench_command(["bench"], Point(2, 1024, 1024, 4), grid.bench)
        assert command[:7] == ["bench", "-m", "2", "-n", "1024", "-k", "1024"]
        assert command[-4:] == ["--ldc", "4", "--ldd", "4"]
        assert "--ldc" not in SkinnyGEMMBench.bench_command(["bench"], Point(1024, 2, 1024), grid.bench)

    def test_gpus(self):
        assert SkinnyGEMMBench.parse_gpus("0-3,6") == [0, 1, 2, 3, 6]
        assert SkinnyGEMMBench.parse_gpus("5,2,5") == [5, 2]

    def test_shipped_grid_loads(self):
        grid = SkinnyGEMMBench.load_grid(SkinnyGEMMTune.DEFAULT_GRID)
        assert grid.held_out_count > 0 and len(SkinnyGEMMBench.lattice_points(grid)) > 0


FAKE_BENCH = textwrap.dedent('''\
    import json, os, sys
    from pathlib import Path
    args = sys.argv[1:]
    log = Path(os.environ["FAKE_DIR"])
    key = "x".join(args[args.index(flag) + 1] for flag in ("-m", "-n", "-k"))
    entry = {"args": args, "gpu": os.environ.get("HIP_VISIBLE_DEVICES"), "pred": os.environ.get("TENSILE_PREDICTION_LIB")}
    with open(log / "calls.jsonl", "a") as stream:
        stream.write(json.dumps(entry) + "\\n")
    once = log / ("once_" + key)
    if once.exists():
        once.unlink()
        sys.exit(3)
    if (log / ("always_" + key)).exists():
        sys.exit(4)
    sys.stdout.write(Path(os.environ["FAKE_OUTPUT"]).read_text())
    ''')


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_bench.py"
    script.write_text(FAKE_BENCH, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    (tmp_path / "out.txt").write_text(CANNED, encoding="utf-8")
    monkeypatch.setenv("FAKE_DIR", str(tmp_path))
    monkeypatch.setenv("FAKE_OUTPUT", str(tmp_path / "out.txt"))
    return tmp_path, [sys.executable, str(script)]


def calls(directory):
    path = directory / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def make_runner(fake, pool, gpus=(0,), name="results"):
    directory, bench = fake
    results = SkinnyGEMMBench.ResultsDir(directory / name, pool)
    return SkinnyGEMMBench.Runner(bench, make_grid().bench, list(gpus), results), results


POINTS = [Point(1024, 1, 1024), Point(1024, 2, 2048), Point(4096, 4, 4096)]


class TestRunner:
    def test_order_is_repeat_major_and_seeded(self):
        jobs = SkinnyGEMMBench.order_jobs("grid", POINTS, range(2), 0)
        assert [job.repeat for job in jobs] == [0, 0, 0, 1, 1, 1]
        assert sorted(job.point for job in jobs[:3]) == sorted(POINTS)
        assert jobs == SkinnyGEMMBench.order_jobs("grid", POINTS, range(2), 0)

    def test_records_environment_and_raw_output(self, fake, pool):
        runner, results = make_runner(fake, pool, gpus=(0, 1))
        jobs = SkinnyGEMMBench.order_jobs("grid", POINTS, range(2), 0)
        assert runner.run(jobs, progress=lambda _: None) == 0
        records = results.records()
        assert len(records) == 6 and all(r.return_code == 0 and r.prediction_lib for r in records)
        assert all(r.first_index == 7 and r.times == {7: 20.5, 9: 3.0, 8: 11.0} for r in records)
        assert {c["gpu"] for c in calls(fake[0])} <= {"0", "1"}
        assert all(c["pred"] == "1" for c in calls(fake[0]))
        assert results.raw_path(jobs[0]).exists()
        assert "Cijk_Origami7" in __import__("gzip").open(results.raw_path(jobs[0]), "rt").read()
        assert results.table.names[9] == generated_name(RING)

    def test_validate_phase_runs_without_the_env_var(self, fake, pool, monkeypatch):
        monkeypatch.setenv("TENSILE_PREDICTION_LIB", "1")
        runner, results = make_runner(fake, pool)
        runner.run([Job("validate", POINTS[0], 0)], progress=lambda _: None)
        assert calls(fake[0])[0]["pred"] is None
        assert results.records()[0].prediction_lib is False

    def test_resume_skips_done_jobs(self, fake, pool):
        runner, results = make_runner(fake, pool)
        jobs = SkinnyGEMMBench.order_jobs("grid", POINTS, range(2), 0)
        runner.run(jobs[:4], progress=lambda _: None)
        assert len(calls(fake[0])) == 4
        again, _ = make_runner(fake, pool)
        again.run(jobs, progress=lambda _: None)
        assert len(calls(fake[0])) == 6
        assert len(results.records()) == 6
        again.run(jobs, progress=lambda _: None)
        assert len(calls(fake[0])) == 6

    def test_single_gpu_runs_in_job_order(self, fake, pool):
        runner, _ = make_runner(fake, pool)
        jobs = SkinnyGEMMBench.order_jobs("grid", POINTS, range(2), 0)
        runner.run(jobs, progress=lambda _: None)
        assert [c["args"][1:6:2] for c in calls(fake[0])] == [[str(job.point.m), str(job.point.n), str(job.point.k)]
                                                               for job in jobs]

    def test_failure_is_retried_once(self, fake, pool):
        directory, _ = fake
        (directory / "once_1024x1x1024").write_text("")
        (directory / "always_1024x2x2048").write_text("")
        runner, results = make_runner(fake, pool)
        failed = runner.run([Job("grid", p, 0) for p in POINTS], progress=lambda _: None)
        assert failed == 1
        codes = {r.point: r.return_code for r in results.records()}
        assert codes == {POINTS[0]: 0, POINTS[1]: 4, POINTS[2]: 0}
        assert len(calls(directory)) == 3 + 1 + 1
        assert Job("grid", POINTS[1], 0) not in results.done_keys() and ("grid", POINTS[0], 0) in results.done_keys()

    def test_unparsable_output_is_a_failure(self, fake, pool):
        directory, _ = fake
        (directory / "out.txt").write_text("nothing useful\n")
        runner, results = make_runner(fake, pool)
        assert runner.run([Job("grid", POINTS[0], 0)], progress=lambda _: None) == 1
        assert results.records()[0].return_code == SkinnyGEMMBench.PARSE_ERROR


class TestAggregation:
    def test_minimum_over_repeats_and_old_default(self, tmp_path, pool):
        table = make_table(tmp_path, pool)
        point = Point(1024, 1, 1024)
        runs = [record(point, {0: 100.0, 1: 40.0, 2: 55.0}, repeat=0), record(point, {0: 80.0, 1: 60.0, 2: 50.0}, repeat=1),
                record(point, {0: 90.0, 1: 70.0}, repeat=2, code=1)]
        measurement = SkinnyGEMMCells.aggregate(runs, table)[("grid", point)]
        assert measurement.old_index == 0 and measurement.old_time == 80.0
        assert measurement.gen == {RING: 40.0, RING_B: 50.0}
        assert measurement.best_gen == 40.0

    def test_use_beta_results_are_ignored(self, tmp_path, pool):
        table = make_table(tmp_path, pool)
        table.names[4] = generated_name(SkinnyGEMMLogic.withBeta(RING, True))
        point = Point(1024, 1, 1024)
        measurement = SkinnyGEMMCells.aggregate([record(point, {0: 100.0, 1: 40.0, 4: 10.0})], table)[("grid", point)]
        assert measurement.gen == {RING: 40.0}
        assert measurement.best_gen == 40.0

    def test_generated_first_pick_means_no_old_default(self, tmp_path, pool):
        table = make_table(tmp_path, pool)
        point = Point(1024, 1, 4100)
        measurement = SkinnyGEMMCells.aggregate([record(point, {1: 5.0}, first=1)], table)[("grid", point)]
        assert measurement.old_index is None and math.isinf(measurement.old_time)

    def test_prediction_less_runs_are_ignored(self, tmp_path, pool):
        table = make_table(tmp_path, pool)
        runs = [record(Point(1024, 1, 1024), {0: 1.0}, prediction_lib=False, phase="validate")]
        assert SkinnyGEMMCells.aggregate(runs, table) == {}

    def test_acceptance_and_regression_rules(self):
        measurement = SkinnyGEMMCells.Measurement("grid", Point(1024, 1, 1024), 0, 100.0, {RING: 80.0, RING_B: 100.0})
        assert SkinnyGEMMCells.accepts(measurement, RING, LIMITS)
        assert not SkinnyGEMMCells.accepts(measurement, RING_B, LIMITS)
        assert not SkinnyGEMMCells.accepts(measurement, UP_FRONT, LIMITS)
        assert SkinnyGEMMCells.no_regression(measurement, RING_B, LIMITS)
        slow = SkinnyGEMMCells.Measurement("grid", Point(1024, 1, 1024), 0, 100.0, {RING: 105.1})
        assert not SkinnyGEMMCells.no_regression(slow, RING, LIMITS)
        unsupported = SkinnyGEMMCells.Measurement("grid", Point(1024, 1, 1024), None, math.inf, {RING: 1e9})
        assert SkinnyGEMMCells.no_regression(unsupported, RING, LIMITS)
        assert SkinnyGEMMCells.accepts(unsupported, RING, LIMITS)

    def test_tolerance_is_relative_to_the_best_kernel(self):
        measurement = SkinnyGEMMCells.Measurement("grid", Point(1024, 1, 1024), 0, 100.0, {RING: 50.0, RING_B: 56.0})
        assert not SkinnyGEMMCells.accepts(measurement, RING_B, LIMITS)
        assert SkinnyGEMMCells.accepts(measurement, RING_B, SkinnyGEMMCells.Thresholds(tolerance=1.2))


class TestKSupports:
    def test_ring_covers_every_k(self):
        assert SkinnyGEMMCells.k_supports(RING, 1, 1 << 20) and SkinnyGEMMCells.k_supports(RING, 4100, 4100)
        assert not SkinnyGEMMCells.k_supports(RING, 1, (1 << 20) + 1)

    def test_up_front_window_and_whole_steps(self):
        supports = SkinnyGEMMCells.k_supports
        assert supports(UP_FRONT, 8192, 8192) and supports(UP_FRONT, 4096 + 4096, 8192)
        assert not supports(UP_FRONT, 4096, 4096) and not supports(UP_FRONT, 8192, 8193)
        assert not supports(UP_FRONT, 5000, 5000)
        assert not supports(UP_FRONT, 4097, 8191)

    def test_partial_step_free_entries_need_a_point_interval(self):
        twin = (1, 1, 16, 1, 1, 0, 0, 1, 1, True, False)
        assert SkinnyGEMMCells.k_supports(twin, 2048, 2048) and not SkinnyGEMMCells.k_supports(twin, 1025, 2047)
        assert not SkinnyGEMMCells.k_supports(twin, 1500, 1500)


class TestPartition:
    def test_disjoint_cover(self):
        intervals = SkinnyGEMMCells.partition([8, 1, 3, 4, 8])
        assert [(i.lo, i.hi, i.corners) for i in intervals] == [
            (1, 1, (1,)), (2, 2, (1, 3)), (3, 3, (3,)), (4, 4, (4,)), (5, 7, (4, 8)), (8, 8, (8,))]
        covered = [v for i in intervals for v in range(i.lo, i.hi + 1)]
        assert covered == list(range(1, 9))


def synthetic(grid, tmp_path, pool, times, extra=()):
    """Measurements of every lattice point; times(point) -> {index: us} with index 0 the old default."""
    table = make_table(tmp_path, pool)
    runs = [record(p, times(p)) for p in SkinnyGEMMBench.lattice_points(grid)]
    runs += [record(p, times(p), phase="grid") for p in extra]
    return SkinnyGEMMCells.aggregate(runs, table)


def build_routes(grid, measurements, pool, held=False):
    cells = SkinnyGEMMCells.CellBuilder(grid, measurements, LIMITS, pool, held).build()
    return cells, SkinnyGEMMCells.make_routes(cells)


def good(point):
    return {0: 100.0, 1: 50.0, 2: 90.0}


def keys(routes):
    return [route.key for route in routes]


class TestCells:
    def grid(self, **changes):
        return SkinnyGEMMBench.parse_grid({**GRID_DATA, "orientations": ["n"], "held_out": {"count": 0, "seed": 0}, **changes})

    def test_everything_good_merges_into_one_cell(self, tmp_path, pool):
        grid = self.grid(skinny=[1, 4], wide=[1024, 4096], k=[1024, 4096])
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, good), pool)
        assert keys(routes) == [(1024, 4096, 1, 4, 1, 1, 1024, 4096)]
        assert routes[0].knobs == RING and routes[0].corners == 8

    def test_merging_along_k_stops_at_a_bad_point(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024], k=[1024, 2048, 4096])
        bad = lambda p: {0: 100.0, 1: 100.0 if p.k == 2048 else 50.0}
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, bad), pool)
        assert keys(routes) == [(1024, 1024, 4, 4, 1, 1, 1024, 1024), (1024, 1024, 4, 4, 1, 1, 4096, 4096)]

    def test_merging_along_wide(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024, 2048, 4096], k=[1024])
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, good), pool)
        assert keys(routes) == [(1024, 4096, 4, 4, 1, 1, 1024, 1024)]

    def test_merging_along_skinny(self, tmp_path, pool):
        grid = self.grid(skinny=[1, 2, 4], wide=[1024], k=[1024])
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, good), pool)
        assert keys(routes) == [(1024, 1024, 1, 4, 1, 1, 1024, 1024)]

    def test_both_orientations(self, tmp_path, pool):
        grid = self.grid(orientations=["n", "m"], skinny=[1, 4], wide=[1024], k=[1024])
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, good), pool)
        assert keys(routes) == [(1, 4, 1024, 1024, 1, 1, 1024, 1024), (1024, 1024, 1, 4, 1, 1, 1024, 1024)]
        assert [route.cell_id[0] for route in routes] == ["m", "n"]

    def test_different_entries_do_not_merge(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024], k=[1024, 4096])
        split = lambda p: {0: 100.0, 1: 50.0 if p.k == 1024 else 90.0, 2: 90.0 if p.k == 1024 else 50.0}
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, split), pool)
        assert [(r.key[6:], r.knobs) for r in routes] == [((1024, 1024), RING), ((4096, 4096), RING_B)]

    def test_missing_corner_gives_no_cell(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024, 4096], k=[1024])
        measurements = synthetic(grid, tmp_path, pool, good)
        del measurements[("grid", Point(4096, 4, 1024))]
        _, routes = build_routes(grid, measurements, pool)
        assert keys(routes) == [(1024, 1024, 4, 4, 1, 1, 1024, 1024)]

    def test_interior_point_vetoes_its_box(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024, 4096], k=[1024], extra_points=[[2000, 4, 1024]])
        regress = lambda p: {0: 100.0, 1: 200.0 if p == Point(2000, 4, 1024) else 50.0}
        measurements = synthetic(grid, tmp_path, pool, regress, extra=grid.extra_points)
        _, routes = build_routes(grid, measurements, pool)
        assert keys(routes) == [(1024, 1024, 4, 4, 1, 1, 1024, 1024), (4096, 4096, 4, 4, 1, 1, 1024, 1024)]
        SkinnyGEMMCells.assert_no_regression(routes, measurements, LIMITS)

    def test_entry_choice_prefers_the_closest_to_the_best(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024], k=[1024])
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, lambda p: {0: 100.0, 1: 50.0, 2: 48.0}), pool)
        assert routes[0].knobs == RING_B

    def test_entry_choice_ties_go_to_the_lowest_pool_index(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024], k=[1024])
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, lambda p: {0: 100.0, 1: 50.0, 2: 50.0}), pool)
        assert routes[0].knobs == min(RING, RING_B)

    def test_up_front_entry_needs_a_matching_k_window(self, tmp_path, pool):
        grid = self.grid(skinny=[4], wide=[1024], k=[8192, 7168])
        fast = lambda p: {0: 100.0, 3: 20.0}
        _, routes = build_routes(grid, synthetic(grid, tmp_path, pool, fast), pool)
        # 7168 is not a multiple of the 4096 step and the gap between the K values is not a point interval.
        assert keys(routes) == [(1024, 1024, 4, 4, 1, 1, 8192, 8192)]
        assert routes[0].knobs == UP_FRONT

    def test_routes_are_disjoint_and_deterministic(self, tmp_path, pool):
        grid = self.grid(orientations=["n", "m"], skinny=[1, 4, 8], wide=[1024, 2048, 4096], k=[1024, 2048, 4100])
        mixed = lambda p: {0: 100.0, 1: 50.0 if (p.m + p.n + p.k) % 3 else 130.0, 2: 55.0}
        first = build_routes(grid, synthetic(grid, tmp_path, pool, mixed), pool)[1]
        second = build_routes(grid, synthetic(grid, tmp_path, pool, mixed), pool)[1]
        assert first == second
        SkinnyGEMMCells.assert_disjoint(first)

    def test_overlapping_routes_are_rejected(self):
        overlapping = [SkinnyGEMMCells.Route("a", (1, 5, 1, 4, 1, 1, 1, 9), RING),
                       SkinnyGEMMCells.Route("b", (5, 9, 2, 3, 1, 1, 9, 9), RING)]
        with pytest.raises(AssertionError, match="overlap"):
            SkinnyGEMMCells.assert_disjoint(overlapping)


class TestHeldOut:
    def cells(self):
        return [SkinnyGEMMCells.Cell("n", (1, 6), (1024, 4096), (1024, 4100), frozenset([RING]), frozenset(), RING),
                SkinnyGEMMCells.Cell("m", (2, 2), (5000, 6000), (2048, 2048), frozenset([RING]), frozenset(), RING)]

    def test_deterministic_and_contained(self):
        grid = make_grid()
        sample = SkinnyGEMMCells.sample_held_out(self.cells(), grid)
        assert sample == SkinnyGEMMCells.sample_held_out(self.cells(), grid)
        assert len(sample) == grid.held_out_count and len(set(sample)) == len(sample)
        routes = SkinnyGEMMCells.make_routes(self.cells())
        assert all(SkinnyGEMMCells.find_route(routes, point) for point in sample)
        assert not set(sample) & set(SkinnyGEMMBench.lattice_points(grid))
        assert SkinnyGEMMCells.sample_held_out(self.cells(), make_grid(held_out={"count": 20, "seed": 4})) != sample

    def test_shapes_follow_the_kernel_contracts(self):
        for point in SkinnyGEMMCells.sample_held_out(self.cells(), make_grid(held_out={"count": 60, "seed": 1})):
            orientation, skinny, wide, k = SkinnyGEMMCells.split_point(point)
            assert wide % 4 == 0
            assert (point.ld == SkinnyGEMMBench.padded_ld(point.m)) == (orientation == "m")
            if orientation == "m":
                assert point.k == 2048 and point.m == 2
        assert any(p.k % 32 for p in SkinnyGEMMCells.sample_held_out(self.cells(), make_grid(held_out={"count": 60, "seed": 1})))

    def test_empty_cells_or_zero_count(self):
        assert SkinnyGEMMCells.sample_held_out([], make_grid()) == []
        assert SkinnyGEMMCells.sample_held_out(self.cells(), make_grid(held_out={"count": 0, "seed": 0})) == []

    def test_offline_validation_detects_a_planted_regression(self):
        routes = SkinnyGEMMCells.make_routes(self.cells())
        fine, slow, missing = Point(2048, 2, 2048), Point(2052, 3, 2048), Point(2056, 4, 2048)
        measurements = {
            ("held_out", fine): SkinnyGEMMCells.Measurement("held_out", fine, 0, 100.0, {RING: 50.0}),
            ("held_out", slow): SkinnyGEMMCells.Measurement("held_out", slow, 0, 100.0, {RING: 106.0}),
            ("held_out", missing): SkinnyGEMMCells.Measurement("held_out", missing, 0, 100.0, {RING_B: 1.0}),
        }
        results = SkinnyGEMMCells.validate_held_out(routes, measurements, LIMITS, [fine, slow, missing, Point(1, 1, 1)])
        assert [(r.point, r.regression) for r in results] == [(fine, False), (slow, True), (missing, True)]
        assert results[0].speedup == 2.0 and results[2].speedup is None


class TestRefinement:
    def test_grid_band_uses_the_best_generated_kernel(self):
        near, far, none = Point(1024, 1, 1024), Point(1024, 2, 1024), Point(1024, 3, 1024)
        measurements = {
            ("grid", near): SkinnyGEMMCells.Measurement("grid", near, 0, 130.0, {RING: 100.0}),
            ("grid", far): SkinnyGEMMCells.Measurement("grid", far, 0, 400.0, {RING: 100.0}),
            ("grid", none): SkinnyGEMMCells.Measurement("grid", none, None, math.inf, {RING: 100.0}),
        }
        assert SkinnyGEMMCells.refinement_points(measurements, "grid") == [near]

    def test_held_out_band_uses_the_routed_entry(self):
        cell = SkinnyGEMMCells.Cell("n", (1, 6), (1024, 4096), (1024, 4100), frozenset([RING]), frozenset(), RING)
        routes = SkinnyGEMMCells.make_routes([cell])
        inside, outside = Point(2048, 2, 2048), Point(8192, 2, 2048)
        measurements = {("held_out", p): SkinnyGEMMCells.Measurement("held_out", p, 0, 120.0, {RING: 100.0, RING_B: 10.0})
                        for p in (inside, outside)}
        assert SkinnyGEMMCells.refinement_points(measurements, "held_out", routes) == [inside]


class TestRows:
    def routes(self):
        return SkinnyGEMMCells.routes_from([((1024, 4096, 1, 4, 1, 1, 1024, 4096), RING, 8),
                                           ((1, 4, 1024, 4096, 1, 1, 1024, 4096), UP_FRONT, 4)])

    def test_round_trip_through_the_logic_generator(self, tmp_path, pool):
        routes = self.routes()
        path = tmp_path / "rows.yaml"
        path.write_text(SkinnyGEMMReport.render_rows(routes, SkinnyGEMMReport.describe_rows("grid.yaml", LIMITS)))
        text = path.read_text()
        assert text.startswith("# Copyright") and "do not edit by hand" in text and "margin 1.25" in text
        read = SkinnyGEMMReport.read_routes(path, pool)
        assert [(r.key, r.knobs) for r in read] == [(r.key, r.knobs) for r in routes]
        index_of = {knobs: index for index, knobs in enumerate(pool)}
        assert SkinnyGEMMLogic.readRoutingRows(path, pool) == [(list(r.key), index_of[r.knobs]) for r in read]
        logic = yaml.safe_load(SkinnyGEMMLogic.renderLogic(
            SkinnyGEMMLogic.headerLines(SkinnyGEMMLogic.ORIGAMI_LOGIC), pool, SkinnyGEMMLogic.readRoutingRows(path, pool)))
        assert [key for key, _ in logic["ExactLogic"][:2]] == [list(r.key) for r in read]

    def test_rendering_is_deterministic(self):
        description = SkinnyGEMMReport.describe_rows("grid.yaml", LIMITS)
        assert SkinnyGEMMReport.render_rows(self.routes(), description) == SkinnyGEMMReport.render_rows(self.routes(), description)

    def test_tables(self):
        routes = self.routes()
        measurements = {("grid", Point(1024, 1, 1024)): SkinnyGEMMCells.Measurement("grid", Point(1024, 1, 1024), 0, 100.0, {RING: 50.0}),
                        ("grid", Point(9999, 1, 1024)): SkinnyGEMMCells.Measurement("grid", Point(9999, 1, 1024), 0, 70.0, {RING: 50.0})}
        points = SkinnyGEMMReport.point_table(routes, measurements)
        assert [row[-2] for row in points[1]] == [2.0, 1.0]
        assert [row[-4] for row in points[1]] == [SkinnyGEMMReport.knobs_label(RING), "Origami"]
        cells = SkinnyGEMMReport.cell_table(routes, points)
        assert [row[0] for row in cells[1]] == ["m000", "n001"]
        assert cells[1][1][-5:] == [8, 1, 2.0, 2.0, 2.0]
        assert cells[1][0][5:7] == [4, 0] and math.isnan(cells[1][0][-1])
        assert "|---|" in SkinnyGEMMReport.markdown_table(points)
        assert SkinnyGEMMReport.csv_text(points).splitlines()[0].startswith("phase,m,n,k")


def write_campaign(directory, pool, grid, times):
    """A results directory with one run of every lattice point (and the given extra points)."""
    results = SkinnyGEMMBench.ResultsDir(directory, pool)
    results.table.names = {0: "Cijk_Origami", 1: generated_name(RING), 2: generated_name(RING_B)}
    results.table.save()
    with open(results.log, "w", encoding="utf-8") as stream:
        for point in SkinnyGEMMBench.lattice_points(grid):
            stream.write(record(point, times(point)).to_json() + "\n")
    return results


class TestTuneCommands:
    def arguments(self, tmp_path, name):
        grid = tmp_path / "grid.yaml"
        data = {**GRID_DATA, "skinny": [1, 4], "wide": [1024, 4096], "k": [1024, 2048], "held_out": {"count": 0, "seed": 0}}
        grid.write_text(yaml.safe_dump(data))
        return ["cells", "--results", str(tmp_path / "campaign"), "--grid", str(grid),
                "--rows", str(tmp_path / f"{name}.rows.yaml"), "--output", str(tmp_path / f"{name}.logic.yaml")]

    def test_cells_command_end_to_end_is_reproducible(self, tmp_path, pool, capsys):
        grid = SkinnyGEMMBench.parse_grid({**GRID_DATA, "skinny": [1, 4], "wide": [1024, 4096], "k": [1024, 2048],
                                          "held_out": {"count": 0, "seed": 0}})
        write_campaign(tmp_path / "campaign", pool, grid, good)
        for name in ("a", "b"):
            assert SkinnyGEMMTune.main(self.arguments(tmp_path, name)) == 0
        rows = (tmp_path / "a.rows.yaml").read_text()
        assert rows == (tmp_path / "b.rows.yaml").read_text()
        assert (tmp_path / "a.logic.yaml").read_bytes() == (tmp_path / "b.logic.yaml").read_bytes()
        params = SkinnyGEMMLogic.asParams(RING)
        del params["useBeta"]
        assert yaml.safe_load(rows) == [{"range": [1, 4, 1024, 4096, 1, 1, 1024, 2048], "params": params},
                                        {"range": [1024, 4096, 1, 4, 1, 1, 1024, 2048], "params": params}]
        for report in ("cells.md", "points.md", "points.csv", "summary.md"):
            assert (tmp_path / "campaign" / report).exists()
        assert "rows: 2" in (tmp_path / "campaign" / "summary.md").read_text()
        assert "rebuild hipBLASLt" in capsys.readouterr().out

    def test_logic_reflects_the_rows(self, tmp_path, pool):
        grid = SkinnyGEMMBench.parse_grid({**GRID_DATA, "skinny": [1, 4], "wide": [1024, 4096], "k": [1024, 2048],
                                          "held_out": {"count": 0, "seed": 0}})
        write_campaign(tmp_path / "campaign", pool, grid, good)
        SkinnyGEMMTune.main(self.arguments(tmp_path, "a"))
        logic = yaml.safe_load((tmp_path / "a.logic.yaml").read_text())
        assert len(logic["ExactLogic"]) == 2 + len(pool)


class TestValidation:
    def setup(self, tmp_path, pool):
        table = make_table(tmp_path, pool)
        routes = SkinnyGEMMCells.routes_from([((1024, 4096, 1, 4, 1, 1, 1024, 4096), RING, 8)])
        routed, other, slow = Point(2048, 2, 2048), Point(9000, 2, 2048), Point(3072, 3, 2048)
        measurements = {("grid", p): SkinnyGEMMCells.Measurement("grid", p, 0, 100.0, {RING: 50.0}) for p in (routed, other, slow)}
        return table, routes, measurements, routed, other, slow

    def test_paired_comparison_and_flags(self, tmp_path, pool):
        table, routes, measurements, routed, other, slow = self.setup(tmp_path, pool)
        runs = [record(routed, {0: 100.0, 1: 40.0}, phase="validate", first=1, prediction_lib=False),
                record(routed, {0: 90.0, 1: 45.0}, phase="validate", first=1, prediction_lib=False, repeat=1),
                record(other, {0: 60.0, 1: 40.0}, phase="validate", first=0, prediction_lib=False),
                record(slow, {0: 50.0, 1: 70.0}, phase="validate", first=1, prediction_lib=False)]
        rows = SkinnyGEMMReport.analyze_validation(runs, measurements, routes, table)
        by_point = {row.point: row for row in rows}
        assert by_point[routed].old_time == 90.0 and by_point[routed].new_time == 40.0 and by_point[routed].pick_ok
        assert by_point[other].speedup == 1.0 and by_point[other].pick_ok and not by_point[other].regression(LIMITS)
        assert by_point[slow].regression(LIMITS)
        # Only a routed point whose old to new ratio is inside the noise band gets more repeats.
        assert SkinnyGEMMReport.validation_refinement(rows) == []
        close = record(routed, {0: 52.0, 1: 40.0}, phase="validate", first=1, prediction_lib=False)
        assert SkinnyGEMMReport.validation_refinement(
            SkinnyGEMMReport.analyze_validation([close], measurements, routes, table)) == [routed]

    def test_unexpected_pick(self, tmp_path, pool):
        table, routes, measurements, routed, other, _ = self.setup(tmp_path, pool)
        runs = [record(routed, {0: 100.0, 1: 40.0}, phase="validate", first=0, prediction_lib=False),
                record(other, {0: 100.0, 1: 40.0}, phase="validate", first=1, prediction_lib=False)]
        rows = SkinnyGEMMReport.analyze_validation(runs, measurements, routes, table)
        assert [row.pick_ok for row in rows] == [False, False]
        lines = "\n".join(SkinnyGEMMReport.validation_summary(rows, LIMITS))
        assert "unexpected picks: 2" in lines

    def test_point_selection_is_capped_and_deterministic(self, tmp_path, pool):
        routes = SkinnyGEMMCells.routes_from([((1, 100000, 1, 16, 1, 1, 1, 100000), RING, 0)])
        measurements = {("grid", Point(1000 + i, 1, 1024)): None for i in range(300)}
        measurements.update({("grid", Point(1000 + i, 20, 1024)): None for i in range(80)})
        measurements[("held_out", Point(1234, 2, 4100))] = None
        points = SkinnyGEMMReport.validation_points(measurements, routes, 0)
        assert points == SkinnyGEMMReport.validation_points(measurements, routes, 0)
        assert len(points) == 1 + SkinnyGEMMReport.VALIDATE_ROUTED_CAP + SkinnyGEMMReport.VALIDATE_UNROUTED
