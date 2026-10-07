# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Near-tie re-bench: tie sets, bench lines, the median merge and the stage
end to end on the fake bench."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import bench_fakes as bf
import fake_hipblaslt as fh
from lib import rebench as rb
from lib.bench_log import iter_problems
from lib.bench_yaml import line_solution_list, read_bench_lines

PIPELINE_DIR = Path(__file__).resolve().parent.parent
G = bf.Gemm(256, 512, 1024)


def problem_lines(g, tested, skipped=(), winner=True):
    """Log lines of one problem: `tested` (rank, sol, us) rows and `skipped`
    (rank, sol, warm-up us) lines, then the Winner block."""
    n = len(tested) + len(skipped)
    out = bf.preamble(n)
    rows = sorted(
        [("t", r, s, u) for r, s, u in tested]
        + [("s", r, s, u) for r, s, u in skipped],
        key=lambda x: x[1],
    )
    for kind, rank, sol, us in rows:
        if kind == "t":
            out += bf.measured_lines(rank, g, sol, us)
        else:
            out.append(bf.skip(rank, sol, f"{us:.2f}"))
    if winner:
        rank, sol, us = min(tested, key=lambda x: x[2])
        out += bf.winner(rank, g, sol, us)
    return out


def as_log(*problems):
    lines = list(bf.BANNER)
    for p in problems:
        lines += p
    return rb.read_log_lines("\n".join(lines) + "\n")


def only(lines):
    (prob,) = list(iter_problems(lines))
    return prob


def test_near_ties_take_rows_within_the_tolerance_in_rank_order():
    lines = as_log(
        problem_lines(
            G,
            [(0, 100, 10.25), (1, 101, 10.0), (2, 102, 11.0), (4, 104, 10.2)],
            [(3, 103, 10.1), (5, 105, 10.4)],
        )
    )
    ties = rb.near_ties(only(lines), tolerance=0.03)
    assert ties.shape == (G.m, G.n, G.k, G.batch)
    assert ties.winner_sol == 101 and ties.winner_us == 10.0
    assert ties.sol_indices == [100, 101, 103, 104]
    assert [c.skipped for c in ties.candidates] == [False, False, True, False]
    assert rb.near_ties(only(lines), tolerance=0.03, max_kernels=2).sol_indices == [
        101,
        103,
    ]
    assert rb.near_ties(only(lines), tolerance=0.001) is None
    clear = as_log(problem_lines(G, [(0, 100, 10.0), (1, 101, 12.0)]))
    assert rb.near_ties(only(clear)) is None
    cut = problem_lines(G, [(0, 100, 10.0), (1, 101, 10.1)], winner=False)
    cut = as_log(bf.preamble(3) + cut[len(bf.preamble(2)) :])
    assert not only(cut).complete and rb.near_ties(only(cut)) is None


def test_rebench_line_times_exactly_the_listed_kernels():
    line = bf.bench_line(
        G, skip_slow_solution_ratio=0.3, algo_method=1, skip_screen_prepass="true"
    )
    out = rb.rebench_line(line, [7, 3, 9], iters_factor=2.5, rotating_mib=64)
    assert line_solution_list(out) == [7, 3, 9]
    assert "skip_slow_solution_ratio: 0," in out or out.rstrip().endswith(
        "skip_slow_solution_ratio: 0}"
    )
    assert "algo_method" not in out and "skip_screen_prepass" not in out
    assert re.search(r"(?<![\w])iters: 25\b", out)
    assert re.search(r"cold_iters: 25\b", out)
    assert re.search(r"rotating: 64\b", out)
    kept = rb.rebench_line(bf.bench_line(G, rotating=16), [1, 2], rotating_mib=64)
    assert re.search(r"rotating: 16\b", kept)
    adaptive = bf.bench_line(G, adaptive="true", measure_time=2)
    adaptive = adaptive.replace("iters: 10, cold_iters: 10, ", "")
    out = rb.rebench_line(adaptive, [1, 2], iters_factor=3)
    assert "iters" not in out and "measure_time: 2" in out
    assert rb.rotate([1, 2, 3], 0) == [1, 2, 3]
    assert rb.rotate([1, 2, 3], 4) == [2, 3, 1]


def run_log(g, rows):
    """A re-bench run's log: `rows` (rank, sol, us) of one problem."""
    return as_log(problem_lines(g, rows))


def test_merge_replaces_near_ties_by_their_median_run_row():
    original = as_log(
        problem_lines(
            G, [(0, 100, 10.0), (1, 101, 10.2), (3, 103, 13.0)], [(2, 102, 10.25)]
        ),
        problem_lines(bf.Gemm(64, 64, 64), [(0, 200, 5.0), (1, 201, 9.0)]),
    )
    ties, n_complete = rb.tie_sets_of_logs([original], tolerance=0.03)
    assert n_complete == 2 and list(ties) == [(G.m, G.n, G.k, G.batch)]
    samples = rb.Samples()
    for r, (a, b, c) in enumerate(
        [(10.5, 10.1, 9.9), (10.4, 10.3, 10.0), (10.6, 10.2, 10.1)]
    ):
        samples.add_log(run_log(G, [(0, 100, a), (1, 101, b), (2, 102, c)]), r)
    assert samples.median((G.m, G.n, G.k, G.batch), 101).us == 10.2
    merged, changes = rb.merge_log(original, ties, samples)
    (change,) = changes
    assert change.winner_before == 100 and change.winner_after == 102
    assert change.replaced == {100: (10.0, 10.5), 101: (10.2, 10.2), 102: (10.25, 10.0)}
    probs = list(iter_problems(merged))
    assert all(p.complete for p in probs)
    p = probs[0]
    assert not p.skipped
    rows = {t.sol_index: (t.rank, float(t.fields["us"])) for t in p.tested}
    assert rows == {100: (0, 10.5), 101: (1, 10.2), 102: (2, 10.0), 103: (3, 13.0)}
    assert p.winner.sol_index == 102 and p.winner.rank == 2
    assert float(p.winner.fields["us"]) == 10.0
    n_tail = len(problem_lines(bf.Gemm(64, 64, 64), [(0, 200, 5.0), (1, 201, 9.0)]))
    assert merged[-n_tail:] == original[-n_tail:]
    assert rb.merge_log(original, ties, rb.Samples()) == (original, [])


def make_stage03(tmp_path, root, shapes):
    """stage03 outputs of `shapes` on the fake bench, with the second
    fastest kernel of every problem logged 0.5% faster than the winner."""
    in_dir = tmp_path / "stage02"
    in_dir.mkdir()
    (in_dir / "shapes_probed.yaml").write_text(
        "".join(
            bf.bench_line(bf.Gemm(*s), algo_method=1, skip_slow_solution_ratio=0)
            for s in shapes
        )
    )
    out = tmp_path / "stage03"
    env = bench_env(root)
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03_load_balance_offline_tuning.py"),
            "--in-dir",
            str(in_dir),
            "--out-dir",
            str(out),
            "--bench-binary",
            str(root / "clients" / "hipblaslt-bench"),
            "--devices",
            "0",
            "--no-warmup-pass",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    flipped = {}
    for log in sorted((out / "logs").glob("block_*.log")):
        lines = rb.read_log_lines(log.read_text())
        for prob in iter_problems(lines):
            lat = rb.tested_latencies(prob)
            first, second = sorted(lat, key=lat.get)[:2]
            flipped[prob.shape()] = (first, second)
            t = next(t for t in prob.tested if t.sol_index == second)
            v = t.line_no
            fields = lines[v].rstrip("\n").split(",")
            fields[-1] = f"{lat[first] * 0.995:g}"
            lines[v] = ",".join(fields) + "\n"
        log.write_text("".join(lines))
    return in_dir, out, flipped


def bench_env(root):
    return dict(
        os.environ,
        FAKE_LIBRARY_DIR=str(root / "Tensile" / "library" / "gfx950"),
        FAKE_LIBRARY_STEM="TensileLibrary_test_gfx950",
    )


def test_stage_rebenches_near_ties_and_merges_medians(tmp_path):
    root = fh.write_fake_build(
        tmp_path / "build", "gfx950", "TensileLibrary_test_gfx950"
    )
    shapes = [(512, 512, 256, 1), (1000, 300, 128, 1), (64, 2048, 512, 2)]
    in_dir, s03, flipped = make_stage03(tmp_path, root, shapes)
    out = tmp_path / "stage03b"
    cfg = tmp_path / "config.yaml"
    cfg.write_text("bench:\n  rebench:\n    runs: 3\n    tolerance: 0.2\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03b_rebench_near_ties.py"),
            "--in-dir",
            str(in_dir),
            "--stage03-dir",
            str(s03),
            "--out-dir",
            str(out),
            "--bench-binary",
            str(root / "clients" / "hipblaslt-bench"),
            "--config-yaml",
            str(cfg),
            "--iters-factor",
            "2",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        env=bench_env(root),
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((out / "rebench_summary.json").read_text())
    assert summary["options"]["runs"] == 3 and summary["options"]["tolerance"] == 0.2
    assert summary["n_problems"] == 3 and summary["n_gemms_merged"] == 3
    assert summary["n_winner_changed"] == 3
    assert [r["rc"] for r in summary["runs"]] == [0, 0, 0]
    per_gemm = json.loads((out / "rebench.json").read_text())["gemms"]
    for g in per_gemm:
        first, second = flipped[(g["m"], g["n"], g["k"], g["batch"])]
        assert g["winner_before"] == second and g["winner_after"] == first
        assert all(len(c["runs_us"]) == 3 for c in g["candidates"])
    run_lines = read_bench_lines(out / "runs" / "run_1" / "shapes.yaml")
    lists = [line_solution_list(ln) for ln in run_lines]
    run0 = [
        line_solution_list(ln)
        for ln in read_bench_lines(out / "runs" / "run_0" / "shapes.yaml")
    ]
    assert all(b == rb.rotate(a, 1) for a, b in zip(run0, lists))
    assert all("algo_method" not in ln and "iters: 20" in ln for ln in run_lines)
    merged = sorted((out / "logs").glob("block_*.log"))
    assert [p.name for p in merged] == [
        p.name for p in sorted((s03 / "logs").glob("block_*.log"))
    ]
    for p in merged:
        for prob in iter_problems(rb.read_log_lines(p.read_text())):
            lat = rb.tested_latencies(prob)
            first, _second = flipped[prob.shape()]
            assert min(lat, key=lat.get) == first == prob.winner.sol_index


def test_stage_refuses_a_bench_without_solution_lists(tmp_path):
    root = fh.write_fake_build(
        tmp_path / "build", "gfx950", "TensileLibrary_test_gfx950"
    )
    in_dir, s03, _ = make_stage03(tmp_path, root, [(512, 512, 256, 1)])
    (root / "clients" / "hipblaslt_common.yaml").write_text(
        "Arguments:\n  - iters: c_int32\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03b_rebench_near_ties.py"),
            "--in-dir",
            str(in_dir),
            "--stage03-dir",
            str(s03),
            "--out-dir",
            str(tmp_path / "out"),
            "--bench-binary",
            str(root / "clients" / "hipblaslt-bench"),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        env=bench_env(root),
        timeout=120,
    )
    assert proc.returncode == 1 and "solution_index_list" in proc.stdout + proc.stderr


@pytest.mark.parametrize(
    "cli, cfg, expect", [(None, {}, 3), (5, {"runs": 1}, 5), (None, {"runs": 1}, 1)]
)
def test_options_come_from_the_command_line_then_the_config(cli, cfg, expect):
    import argparse

    import stage03b_rebench_near_ties as s03b

    args = argparse.Namespace(
        tolerance=None, runs=cli, iters_factor=None, max_kernels=None
    )
    s03b.resolve_options(args, {"bench": {"rebench": cfg}})
    assert args.runs == expect and args.tolerance == rb.DEFAULT_TOLERANCE


# ── pick re-bench ────────────────────────────────────────────────────────────


def _gemm(shape, timed, screened=()):
    m, n, k, b = shape
    cands = [dict(sol_idx_global=s, us=u, is_skip=False) for s, u in timed]
    cands += [dict(sol_idx_global=s, us=u, is_skip=True) for s, u in screened]
    return dict(m=m, n=n, k=k, batch_count=b, candidates=cands)


def test_pick_kernel_list_puts_the_pick_and_the_winner_first():
    timed = {5: 3.0, 6: 1.0, 7: 2.0, 8: 4.0, 9: 1.5}
    assert rb.pick_kernel_list(40, timed) == [40, 6, 9, 7]
    assert rb.pick_kernel_list(40, timed, extra=0) == [40, 6]
    assert rb.pick_kernel_list(9, timed) == [9, 6, 7, 5]
    assert rb.pick_kernel_list(40, {}) == []


def test_pick_ratios_pair_medians_of_the_runs_that_timed_both():
    shape = (G.m, G.n, G.k, G.batch)
    other = (64, 64, 64, 1)
    gemms = [
        _gemm(shape, [(101, 10.0), (102, 11.0), (103, 12.0)], [(100, 13.0)]),
        _gemm(other, [(201, 5.0)]),
        _gemm((32, 32, 32, 1), [(301, 5.0), (302, 6.0)]),
    ]
    picks = {shape: 100, other: 200, (8, 8, 8, 1): 1}
    lines = {s: bf.bench_line(bf.Gemm(*s)) for s in (shape, other, (32, 32, 32, 1))}
    pick_us = {0: [20.0], 1: [24.0], 2: [None]}
    win_us = {0: 10.0, 1: 16.0, 2: 1.0}
    seen = []

    def run_bench(r, yaml_lines):
        seen.append([line_solution_list(ln) for ln in yaml_lines])
        tested = [(1, 101, win_us[r]), (2, 102, 30.0)]
        if pick_us[r][0] is not None:
            tested.append((0, 100, pick_us[r][0]))
        return [as_log(problem_lines(G, tested))]

    ratios, samples = rb.rebench_picks(gemms, picks, lines, run_bench, runs=3)
    assert seen[0] == [[100, 101, 102, 103], [200, 201]]
    assert seen[1] == [rb.rotate(x, 1) for x in seen[0]]
    assert set(ratios) == {shape}
    r = ratios[shape]
    assert (r.pick_sol, r.winner_sol, r.n_runs) == (100, 101, 2)
    assert (r.pick_us, r.winner_us) == (20.0, 10.0) and r.ratio == 0.5
    assert len(samples.by_shape[shape][101]) == 3
    assert rb.rebench_picks(gemms, {}, lines, run_bench) == ({}, rb.Samples())

    g = dict(gemms[0])
    assert rb.attach_rebench([g, gemms[1]], ratios) == 1
    assert g["rebench"] == {100: 0.5} and "rebench" not in gemms[1]


def test_pick_rebench_through_stage03_on_the_fake_bench(tmp_path, monkeypatch):
    root = fh.write_fake_build(
        tmp_path / "build", "gfx950", "TensileLibrary_test_gfx950"
    )
    grid = fh.kernels_grid()
    shapes = [(512, 512, 256, 1), (1000, 300, 128, 1)]

    def us(shape, sol):
        return fh.kernel_time_us(*shape, grid[sol]["sizeMapping"])

    gemms, picks = [], {}
    for shape in shapes:
        order = sorted(range(len(grid)), key=lambda s: us(shape, s))
        picks[shape] = order[-1]
        gemms.append(
            _gemm(
                shape,
                [(s, us(shape, s)) for s in order[:4]],
                [(order[-1], 1.0)],
            )
        )
    lines = {s: bf.bench_line(bf.Gemm(*s), algo_method=1) for s in shapes}
    for key in ("FAKE_LIBRARY_DIR", "FAKE_LIBRARY_STEM"):
        monkeypatch.setenv(key, bench_env(root)[key])
    runner = rb.stage03_runner(root / "clients" / "hipblaslt-bench", tmp_path / "work")
    ratios, _ = rb.rebench_picks(gemms, picks, lines, runner, runs=3)
    assert set(ratios) == set(shapes)
    for shape in shapes:
        r = ratios[shape]
        win = min(range(len(grid)), key=lambda s: us(shape, s))
        assert r.n_runs == 3 and r.winner_sol == win
        assert r.ratio == pytest.approx(us(shape, win) / us(shape, picks[shape]))
    run1 = read_bench_lines(tmp_path / "work" / "run_1" / "shapes.yaml")
    run0 = read_bench_lines(tmp_path / "work" / "run_0" / "shapes.yaml")
    assert [line_solution_list(b) for b in run1] == [
        rb.rotate(line_solution_list(a), 1) for a in run0
    ]
    assert all("algo_method" not in ln for ln in run0)
