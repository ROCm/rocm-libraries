# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Near-winner re-reads: tie sets, bench lines, the median merge and the
stage end to end on the fake bench."""

import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
from lib import rebench as rb
from lib.bench_log import iter_problems
from lib.bench_yaml import (
    ADAPTIVE_KEYS,
    adaptive_line,
    line_is_adaptive,
    line_solution_list,
    read_bench_lines,
    render_bench_line,
    set_line_knob,
    shape_of_line,
)
from lib.shapes import Shape

PIPELINE_DIR = Path(__file__).resolve().parent.parent
G = bf.Gemm(256, 512, 1024)
CFG_HIPBLASLT = {
    "transA": "T",
    "transB": "N",
    "a_type": "f8_r",
    "b_type": "f8_r",
    "c_type": "bf16_r",
    "d_type": "bf16_r",
    "scale_type": "f32_r",
    "compute_type": "f32_r",
}
TIMING_KEYS = ("adaptive", *ADAPTIVE_KEYS)


def _knob(line, key):
    m = re.search(rf"(?<![\w]){key}:\s*([^,}}]+)", line)
    return m.group(1).strip() if m else None


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
    cfg = {"hipblaslt": CFG_HIPBLASLT, "bench": {"skip_slow_solution_ratio": 0.3}}
    line = render_bench_line(Shape(G.m, G.n, G.k, G.batch), cfg)
    line = set_line_knob(line, "algo_method", 1)
    line = set_line_knob(line, "skip_screen_prepass", "true")
    out = rb.rebench_line(line, [7, 3, 9], rotating_mib=64)
    assert line_solution_list(out) == [7, 3, 9]
    assert "skip_slow_solution_ratio: 0," in out or out.rstrip().endswith(
        "skip_slow_solution_ratio: 0}"
    )
    assert "algo_method" not in out and "skip_screen_prepass" not in out
    assert re.search(r"requested_solution_num: -1\b", out)
    assert [_knob(out, k) for k in TIMING_KEYS] == [_knob(line, k) for k in TIMING_KEYS]
    assert line_is_adaptive(out) and not re.search(r"(?<![\w])(cold_)?iters:", out)
    assert _knob(out, "rotating") == _knob(line, "rotating") != "0"
    unrotated = rb.rebench_line(
        set_line_knob(line, "rotating", 0), [1], rotating_mib=64
    )
    assert _knob(unrotated, "rotating") == "64"
    assert rb.rotate([1, 2, 3], 0) == [1, 2, 3]
    assert rb.rotate([1, 2, 3], 4) == [2, 3, 1]


def test_blocks_per_gpu_follow_the_lines_per_gpu():
    assert rb.blocks_per_gpu(0, 3) == 1
    assert rb.blocks_per_gpu(138, 6) == 1
    assert rb.blocks_per_gpu(3 * rb.REBENCH_LINES_PER_BLOCK, 3) == 1
    assert rb.blocks_per_gpu(3 * rb.REBENCH_LINES_PER_BLOCK + 3, 3) == 2
    assert rb.blocks_per_gpu(10**9, 1) == rb.MAX_BLOCKS_PER_GPU
    assert rb.blocks_per_gpu(100, 2, lines_per_block=24) == 3
    assert rb.blocks_per_gpu(100, 0, lines_per_block=24, cap=2) == 2


def test_run_order_rotates_a_seeded_shuffle():
    from lib.planner import shuffled

    items = list(range(10, 20))
    assert rb.run_order(items, 2, None, ("x",)) == rb.rotate(items, 2)
    base = shuffled(items, 5, "near-tie", 1, 2, 3, 1)
    assert base != items
    for r in range(3):
        got = rb.run_order(items, r, 5, ("near-tie", 1, 2, 3, 1))
        assert got == rb.rotate(base, r)
    places = {r: rb.run_order(items, r, 5, ("k",)).index(10) for r in range(3)}
    assert len(set(places.values())) == 3


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


def test_merge_leaves_the_sacrificial_row_to_be_set_aside_again():
    original = as_log(
        problem_lines(G, [(0, 100, 10.1), (1, 100, 10.0), (2, 101, 10.2)])
    )
    prob = only(original)
    assert prob.sacrificial.rank == 0
    ties = {(G.m, G.n, G.k, G.batch): rb.near_ties(prob, tolerance=0.05)}
    assert [c.rank for c in ties[(G.m, G.n, G.k, G.batch)].candidates] == [1, 2]
    samples = rb.Samples()
    for r, (a, b) in enumerate([(10.4, 10.0), (10.6, 10.1), (10.5, 9.9)]):
        samples.add_log(run_log(G, [(0, 100, a), (1, 101, b)]), r)
    merged, (change,) = rb.merge_log(original, ties, samples)
    assert change.replaced == {100: (10.0, 10.5), 101: (10.2, 10.0)}
    p = only(merged)
    assert (p.sacrificial.rank, float(p.sacrificial.fields["us"])) == (0, 10.1)
    rows = {t.sol_index: (t.rank, float(t.fields["us"])) for t in p.tested}
    assert rows == {100: (1, 10.5), 101: (2, 10.0)}
    assert p.winner.sol_index == 101


def make_stage03(tmp_path, root, shapes, extra=()):
    """stage03 outputs of `shapes` on the fake bench (`extra` stage03
    options), with the second fastest kernel of every problem logged 0.5%
    faster than the winner."""
    in_dir = tmp_path / "stage01"
    in_dir.mkdir()
    (in_dir / "shapes.yaml").write_text(
        "".join(
            adaptive_line(
                bf.bench_line(bf.Gemm(*s), algo_method=1, skip_slow_solution_ratio=0),
                {},
            )
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
            *extra,
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


def test_the_predating_settings_rebench_every_gemm_alike(tmp_path):
    """PREDATING_REREADS gives the stage as it was before the bands: every
    GEMM read three times in the input order, no sacrificial copy, each
    candidate labelled by the lower median of its re-reads alone."""
    root = fh.write_fake_build(
        tmp_path / "build", "gfx950", "TensileLibrary_test_gfx950"
    )
    shapes = [(512, 512, 256, 1), (1000, 300, 128, 1), (64, 2048, 512, 2)]
    in_dir, s03, flipped = make_stage03(tmp_path, root, shapes)
    out = tmp_path / "stage03b"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {"bench": {"rebench": {**rb.PREDATING_REREADS, "tolerance": 0.2}}}
        )
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
            str(out),
            "--bench-binary",
            str(root / "clients" / "hipblaslt-bench"),
            "--config-yaml",
            str(cfg),
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
    assert [r["blocks_per_gpu"] for r in summary["runs"]] == [1, 1, 1]
    assert [r["gemms"] for r in summary["runs"]] == [3, 3, 3]
    for r in range(3):
        run = out / "runs" / f"run_{r}" / "stage03" / "stage03_summary.json"
        assert json.loads(run.read_text())["n_blocks"] == 1
    assert summary["n_problems"] == 3 and summary["n_gemms_merged"] == 3
    assert summary["n_winner_changed"] == 3
    assert [r["rc"] for r in summary["runs"]] == [0, 0, 0]
    per_gemm = json.loads((out / "rebench.json").read_text())["gemms"]
    for g in per_gemm:
        first, second = flipped[(g["m"], g["n"], g["k"], g["batch"])]
        assert g["winner_before"] == second and g["winner_after"] == first
        assert all(len(c["runs_us"]) == 3 for c in g["candidates"])
        for c in g["candidates"]:
            assert c["median_us"] == sorted(c["runs_us"])[1]
    run_lines = read_bench_lines(out / "runs" / "run_1" / "shapes.yaml")
    lists = [line_solution_list(ln) for ln in run_lines]
    run0 = [
        line_solution_list(ln)
        for ln in read_bench_lines(out / "runs" / "run_0" / "shapes.yaml")
    ]
    assert all(b == rb.rotate(a, 1) for a, b in zip(run0, lists))
    by_rank = {
        (g["m"], g["n"], g["k"], g["batch"]): [c["sol_index"] for c in g["candidates"]]
        for g in per_gemm
    }
    for ln in read_bench_lines(out / "runs" / "run_0" / "shapes.yaml"):
        shape = shape_of_line(ln)
        want = rb.run_order(by_rank[shape], 0, 42, ("near-tie", *shape))
        assert line_solution_list(ln) == want
    (gemm_line, *_) = read_bench_lines(in_dir / "shapes.yaml")
    for ln in run_lines:
        assert "algo_method" not in ln and line_is_adaptive(ln)
        assert [_knob(ln, k) for k in TIMING_KEYS] == [
            _knob(gemm_line, k) for k in TIMING_KEYS
        ]
    merged = sorted((out / "logs").glob("block_*.log"))
    assert [p.name for p in merged] == [
        p.name for p in sorted((s03 / "logs").glob("block_*.log"))
    ]
    for p in merged:
        for prob in iter_problems(rb.read_log_lines(p.read_text())):
            lat = rb.tested_latencies(prob)
            first, _second = flipped[prob.shape()]
            assert min(lat, key=lat.get) == first == prob.winner.sol_index


def test_stage_rereads_by_band_and_merges_with_the_original(tmp_path):
    """The default protocol: a launch-bound GEMM read three times, the
    others once, the slow one in windows of at least slow_min_iters
    iterations; every list led by a sacrificial copy, the GEMMs shuffled per
    run, each candidate labelled by the median of its original timing and
    its re-reads; the launch record of the benches kept for stage04."""
    from lib import launch_settings
    from lib.planner import stable_rng

    root = fh.write_fake_build(
        tmp_path / "build", "gfx950", "TensileLibrary_test_gfx950"
    )
    fast, mid, slow = (32, 32, 32, 1), (64, 2048, 512, 2), (4096, 4096, 4096, 1)
    in_dir, s03, flipped = make_stage03(tmp_path, root, [mid, slow, fast])
    original = {}
    for log in (s03 / "logs").glob("block_*.log"):
        for prob in iter_problems(rb.read_log_lines(log.read_text())):
            original[prob.shape()] = rb.tested_latencies(prob)
    winners = [min(original[s].values()) for s in (fast, mid, slow)]
    assert winners[0] < 10 <= winners[1] < 1000 <= winners[2]
    out = tmp_path / "stage03b"
    cfg = tmp_path / "config.yaml"
    cfg.write_text("bench:\n  rebench:\n    tolerance: 0.2\n")
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03b_rebench_near_ties.py"),
            "--in-dir", str(in_dir), "--stage03-dir", str(s03),
            "--out-dir", str(out),
            "--bench-binary", str(root / "clients" / "hipblaslt-bench"),
            "--config-yaml", str(cfg), "--quiet",
        ],
        capture_output=True, text=True, env=bench_env(root), timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((out / "rebench_summary.json").read_text())
    assert summary["options"] == {**rb.REREAD_DEFAULTS, "tolerance": 0.2}
    assert {b: c["runs"] for b, c in summary["bands"].items()} == {
        "fast": 3,
        "mid": 1,
        "slow": 1,
    }
    assert [b["gemms"] for b in summary["bands"].values()] == [1, 1, 1]
    assert [r["gemms"] for r in summary["runs"]] == [3, 1, 1]
    assert summary["launch"] == launch_settings.summarize([None] * 4)
    jobs = [shape_of_line(ln) for ln in read_bench_lines(in_dir / "shapes.yaml")]
    per_gemm = {
        (g["m"], g["n"], g["k"], g["batch"]): g
        for g in json.loads((out / "rebench.json").read_text())["gemms"]
    }
    for r in range(3):
        lines = read_bench_lines(out / "runs" / f"run_{r}" / "shapes.yaml")
        want = [s for s in jobs if r == 0 or s == fast]
        stable_rng(42, "near-tie", r).shuffle(want)
        assert [shape_of_line(ln) for ln in lines] == want
        for ln in lines:
            shape = shape_of_line(ln)
            sols = [c["sol_index"] for c in per_gemm[shape]["candidates"]]
            order = rb.run_order(sols, r, 42, ("near-tie", *shape))
            assert line_solution_list(ln) == [order[0], *order]
            assert _knob(ln, "min_iters") == ("10" if shape == slow else "3")
    merged = {}
    for log in (out / "logs").glob("block_*.log"):
        for prob in iter_problems(rb.read_log_lines(log.read_text())):
            merged[prob.shape()] = (rb.tested_latencies(prob), prob.winner.sol_index)
    for shape, g in per_gemm.items():
        assert g["band"] == {fast: "fast", mid: "mid", slow: "slow"}[shape]
        assert g["with_original"] is True
        for c in g["candidates"]:
            assert len(c["runs_us"]) == g["runs"]
            readings = c["runs_us"] + [original[shape][c["sol_index"]]]
            assert c["median_us"] == pytest.approx(statistics.median(readings))
            assert merged[shape][0][c["sol_index"]] == pytest.approx(
                c["median_us"], rel=1e-5
            )
        lat = merged[shape][0]
        assert merged[shape][1] == min(lat, key=lat.get) == g["winner_after"]
    first, second = flipped[fast]
    assert per_gemm[fast]["winner_after"] == first != per_gemm[fast]["winner_before"]


def test_the_stages_record_the_library_their_indices_name(tmp_path):
    """stage03 records the kernel identity of the library files its bench
    lines name; stage03b checks it against those files before re-benching
    and records it for stage04, with its digest in rebench.json."""
    from lib import kernel_identity as ki

    stem = "TensileLibrary_test_gfx950"
    root = fh.write_fake_build(tmp_path / "build", "gfx950", stem)
    libdir = root / "Tensile" / "library" / "gfx950"
    identity = ["--identity-dir", str(libdir), "--identity-stem", stem]
    in_dir, s03, _ = make_stage03(tmp_path, root, [(512, 512, 256, 1)], identity)
    rec = ki.read_record(json.loads((s03 / "stage03_summary.json").read_text()))
    now = ki.library_identity(libdir, [stem])
    assert rec == ki.record(now, ())

    def stage03b(out):
        return subprocess.run(
            [
                sys.executable,
                str(PIPELINE_DIR / "stages" / "stage03b_rebench_near_ties.py"),
                "--in-dir", str(in_dir), "--stage03-dir", str(s03),
                "--out-dir", str(out),
                "--bench-binary", str(root / "clients" / "hipblaslt-bench"),
                "--tolerance", "0.2", "--runs", "1", "--quiet", *identity,
            ],
            capture_output=True, text=True, env=bench_env(root), timeout=300,
        )  # fmt: skip

    proc = stage03b(tmp_path / "s03b")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads((tmp_path / "s03b" / "rebench.json").read_text())
    assert doc["library_digest"] == now.digest and doc["gemms"]
    for g in doc["gemms"]:
        for key in ("winner_before", "winner_after"):
            assert g[f"{key}_kernel_id"] == now.ids[g[key]]
            assert g[f"{key}_signature"] == ki.label_of_id(now.ids[g[key]])
        for c in g["candidates"]:
            assert c["kernel_id"] == now.ids[c["sol_index"]]
            assert c["signature"] == ki.label_of_id(c["kernel_id"])
    assert ki.bench_record(tmp_path / "s03b" / "logs") == rec
    renumbered = [dict(k, index=int(k["index"]) + 7) for k in fh.kernels_grid()]
    bf.write_logic(libdir / f"{stem}.dat", renumbered)
    proc = stage03b(tmp_path / "s03b_rebuilt")
    assert proc.returncode == 1
    assert "benched other library files" in proc.stdout + proc.stderr


def test_the_stage_leaves_the_configs_blocked_kernels_out_of_the_sets(tmp_path):
    """A kernel the config's blocklist names, found through the library the
    logs were benched with, is no near-winner and sets no winner, also when
    stage03 timed it fastest; the merged logs name the fastest kept kernel."""
    stem = "TensileLibrary_test_gfx950"
    root = fh.write_fake_build(tmp_path / "build", "gfx950", stem)
    libdir = root / "Tensile" / "library" / "gfx950"
    identity = ["--identity-dir", str(libdir), "--identity-stem", stem]
    shape = (512, 512, 256, 1)
    in_dir, s03, flipped = make_stage03(tmp_path, root, [shape], identity)
    logged = flipped[shape][1]
    names = {int(k["index"]): str(k["name"]) for k in fh.kernels_grid()}
    blocked = {sid for sid, n in names.items() if n == names[logged]}
    lst = tmp_path / "blocked.tsv"
    lst.write_text(f"C\t{names[logged]}\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "bench": {"rebench": {"tolerance": 0.2}},
                "correctness": {"lists": [str(lst)]},
            }
        )
    )
    out = tmp_path / "stage03b"
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03b_rebench_near_ties.py"),
            "--in-dir", str(in_dir), "--stage03-dir", str(s03),
            "--out-dir", str(out),
            "--bench-binary", str(root / "clients" / "hipblaslt-bench"),
            "--config-yaml", str(cfg), *identity,
        ],
        capture_output=True, text=True, env=bench_env(root), timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"{len(blocked)} kernel(s) of the library compute wrong" in proc.stdout
    (g,) = json.loads((out / "rebench.json").read_text())["gemms"]
    assert g["winner_before"] not in blocked and g["winner_after"] not in blocked
    assert g["candidates"] and not {c["sol_index"] for c in g["candidates"]} & blocked
    (log,) = sorted((out / "logs").glob("block_*.log"))
    (prob,) = iter_problems(rb.read_log_lines(log.read_text()))
    kept = rb.tested_latencies(prob, blocked)
    assert prob.winner.sol_index == min(kept, key=kept.get) == g["winner_after"]


def test_a_rebench_of_logs_stage03_recorded_no_library_for_records_none(tmp_path):
    """Logs nothing checked against a library stay without a record when
    stage03b merges its runs into them, so stage04 reads them on its
    checks for logs without one."""
    from lib import kernel_identity as ki

    stem = "TensileLibrary_test_gfx950"
    root = fh.write_fake_build(tmp_path / "build", "gfx950", stem)
    libdir = root / "Tensile" / "library" / "gfx950"
    in_dir, s03, _ = make_stage03(tmp_path, root, [(512, 512, 256, 1)])
    assert ki.bench_record(s03 / "logs") is None
    out = tmp_path / "s03b"
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03b_rebench_near_ties.py"),
            "--in-dir", str(in_dir), "--stage03-dir", str(s03),
            "--out-dir", str(out),
            "--bench-binary", str(root / "clients" / "hipblaslt-bench"),
            "--tolerance", "0.2", "--runs", "1", "--quiet",
            "--identity-dir", str(libdir), "--identity-stem", stem,
        ],
        capture_output=True, text=True, env=bench_env(root), timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "recorded no library files" in proc.stdout + proc.stderr
    assert ki.bench_record(out / "logs") is None
    assert ki.RECORD_KEY not in json.loads((out / "rebench_summary.json").read_text())
    doc = json.loads((out / "rebench.json").read_text())
    assert doc["library_digest"] is None and doc["gemms"]
    assert all(
        g["winner_before_kernel_id"] is None and c["signature"] is None
        for g in doc["gemms"]
        for c in g["candidates"]
    )


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
    "cli, cfg, expect", [(None, {}, 1), (5, {"runs": 2}, 5), (None, {"runs": 2}, 2)]
)
def test_options_come_from_the_command_line_then_the_config(cli, cfg, expect):
    import argparse

    import stage03b_rebench_near_ties as s03b

    args = argparse.Namespace(tolerance=None, runs=cli, max_kernels=None)
    s03b.resolve_options(args, {"bench": {"rebench": dict(cfg, slow_runs=0)}})
    assert args.runs == expect and args.tolerance == rb.DEFAULT_TOLERANCE
    assert args.slow_runs == 0 and args.fast_runs == rb.Bands.fast_runs
    assert {k: getattr(args, k) for k in s03b.CONFIG_KEYS}.keys() == set(
        rb.REREAD_DEFAULTS
    )
    assert s03b.option_errors(args) == []
    bad = argparse.Namespace(**dict(vars(args), fast_below_us=2e3, gemm_order="x"))
    assert len(s03b.option_errors(bad)) == 2


def test_bands_by_the_winner_latency():
    bands = rb.Bands()
    assert [bands.band(us) for us in (2.0, 9.99, 10.0, 999.0, 1000.0, 5e4)] == [
        "fast",
        "fast",
        "mid",
        "mid",
        "slow",
        "slow",
    ]
    assert [bands.runs_of(us) for us in (2.0, 50.0, 2e3)] == [3, 1, 1]
    assert [bands.min_iters_of(us) for us in (2.0, 50.0, 2e3)] == [0, 0, 10]
    old = rb.bands_of({**rb.REREAD_DEFAULTS, **rb.PREDATING_REREADS})
    assert [old.runs_of(us) for us in (2.0, 50.0, 2e3)] == [3, 3, 3]
    assert old.min_iters_of(2e3) == 0
    assert set(rb.PREDATING_REREADS) <= set(rb.REREAD_DEFAULTS)


def test_reread_line_leads_with_a_sacrificial_copy_and_long_windows():
    cfg = {"hipblaslt": CFG_HIPBLASLT, "bench": {"skip_slow_solution_ratio": 0.3}}
    line = render_bench_line(Shape(G.m, G.n, G.k, G.batch), cfg)
    assert rb.reread_line(line, [7, 3]) == rb.rebench_line(line, [7, 3])
    out = rb.reread_line(line, [7, 3, 9], sacrificial=True)
    assert line_solution_list(out) == [7, 7, 3, 9]
    assert _knob(out, "min_iters") == _knob(line, "min_iters") == "3"
    slow = rb.reread_line(line, [7, 3], min_iters=10)
    assert _knob(slow, "min_iters") == "10" and line_solution_list(slow) == [7, 3]
    more = set_line_knob(line, "min_iters", 20)
    assert _knob(rb.reread_line(more, [7], min_iters=10), "min_iters") == "20"
    from lib.planner import MAX_BENCH_LIST

    full = list(range(MAX_BENCH_LIST))
    assert line_solution_list(rb.reread_line(line, full, sacrificial=True)) == full


def test_with_latency_rewrites_the_us_column_only():
    block = bf.measured_lines(3, G, 101, 12.5, adaptive=True)
    out = rb.with_latency(block, 11.25)
    assert out[0] == block[0] and out[2:] == block[2:]
    (prob,) = list(iter_problems(as_log(bf.preamble(1, adaptive=True) + out)))
    (row,) = prob.tested
    old = only(as_log(bf.preamble(1, adaptive=True) + block)).tested[0]
    assert float(row.fields["us"]) == 11.25
    assert {k: v for k, v in row.fields.items() if k != "us"} == {
        k: v for k, v in old.fields.items() if k != "us"
    }
    assert rb.with_latency(["not a row\n"], 1.0) == ["not a row\n"]


def _three_reads():
    samples = rb.Samples()
    samples.add_log(run_log(G, [(0, 100, 10.6), (1, 101, 9.8), (2, 102, 10.4)]), 0)
    samples.add_log(run_log(G, [(0, 100, 10.4)]), 1)
    samples.add_log(run_log(G, [(0, 100, 10.5)]), 2)
    return samples


def test_merge_with_the_original_takes_the_median_of_every_reading():
    shape = (G.m, G.n, G.k, G.batch)
    original = as_log(
        problem_lines(
            G, [(0, 100, 10.0), (1, 101, 10.2), (2, 102, 10.25), (3, 103, 13.0)]
        )
    )
    ties, _ = rb.tie_sets_of_logs([original], tolerance=0.03)
    assert not ties[shape].has_skip_row
    samples = _three_reads()
    label = samples.label(shape, 101, (10.2, ["x"]))
    assert label[0] == pytest.approx(10.0)
    merged, (change,) = rb.merge_log(original, ties, samples, with_original=True)
    assert change.replaced[100] == (10.0, pytest.approx(10.45))
    assert change.replaced[101] == (10.2, pytest.approx(10.0))
    assert change.replaced[102] == (10.25, pytest.approx(10.325))
    assert change.winner_before == 100 and change.winner_after == 101
    p = only(merged)
    rows = {t.sol_index: (t.rank, float(t.fields["us"])) for t in p.tested}
    assert rows == {
        100: (0, pytest.approx(10.45)),
        101: (1, pytest.approx(10.0)),
        102: (2, pytest.approx(10.325)),
        103: (3, 13.0),
    }
    assert p.winner.sol_index == 101 and p.winner.rank == 1
    alone, (old,) = rb.merge_log(original, ties, samples)
    assert old.replaced[100] == (10.0, 10.5) and old.replaced[101] == (10.2, 9.8)
    assert only(alone).winner.sol_index == 101


def test_a_set_with_a_skip_row_labels_every_candidate_from_its_rereads():
    """A skip row's warm-up is no timing, so its candidate's label comes from
    the re-reads alone; the others of its set do too, so that an offset
    between stage03's process and the re-reads' moves none against it."""
    shape = (G.m, G.n, G.k, G.batch)
    original = as_log(
        problem_lines(
            G, [(0, 100, 10.0), (1, 101, 10.2), (3, 103, 13.0)], [(2, 102, 10.25)]
        )
    )
    ties, _ = rb.tie_sets_of_logs([original], tolerance=0.03)
    assert ties[shape].has_skip_row
    merged, (change,) = rb.merge_log(original, ties, _three_reads(), True)
    assert change.replaced == {
        100: (10.0, 10.5),
        101: (10.2, 9.8),
        102: (10.25, 10.4),
    }
    p = only(merged)
    assert not p.skipped and p.winner.sol_index == 101
    rows = {t.sol_index: float(t.fields["us"]) for t in p.tested}
    assert rows == {100: 10.5, 101: 9.8, 102: 10.4, 103: 13.0}


def test_blocked_kernels_take_no_part_in_a_set():
    """A blocked kernel is no near-winner, and the window counts from the
    fastest kernel the blocklist keeps."""
    from lib import correctness as cx

    shape = (G.m, G.n, G.k, G.batch)
    original = as_log(
        problem_lines(
            G,
            [(0, 100, 8.0), (1, 101, 10.0), (2, 102, 10.2), (4, 104, 12.0)],
            [(3, 103, 8.1)],
        )
    )
    free, _ = rb.tie_sets_of_logs([original], tolerance=0.03)
    assert free[shape].sol_indices == [100, 103]
    ties, _ = rb.tie_sets_of_logs([original], 0.03, blocked={100, 103})
    assert ties[shape].sol_indices == [101, 102] and ties[shape].winner_sol == 101
    samples = rb.Samples()
    samples.add_log(run_log(G, [(0, 101, 10.4), (1, 102, 10.0)]), 0)
    merged, (change,) = rb.merge_log(
        original, ties, samples, with_original=True, blocked={100, 103}
    )
    assert change.winner_before == 101 and change.winner_after == 102
    assert only(merged).winner.sol_index == 102
    # Through the library's kernel ids, or else the names the rows print.
    from lib.kernel_identity import name_hash

    bad = bf.kernel_name((64, 64, 32))
    blocklist = cx.Blocklist([cx.Entry(name=bad)])
    ids = {100: f"sig:{name_hash(bad)}", 101: f"sig:{name_hash('Cijk_good')}"}
    assert rb.blocked_solutions([original], blocklist, ids) == {100}
    assert rb.blocked_solutions([original], None, ids) == set()
    named = as_log(
        bf.preamble(2)
        + bf.measured_lines(0, G, 100, 8.0, mt=(64, 64, 32))
        + bf.measured_lines(1, G, 101, 10.0)
        + bf.winner(0, G, 100, 8.0, mt=(64, 64, 32))
    )
    assert rb.blocked_solutions([named], blocklist) == {100}


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
    seen.clear()
    shuffled, _ = rb.rebench_picks(gemms, picks, lines, run_bench, runs=3, order_seed=9)
    assert shuffled == ratios
    for r, lists in enumerate(seen):
        assert lists == [
            rb.run_order(x, r, 9, ("rebench", *s))
            for x, s in zip([[100, 101, 102, 103], [200, 201]], (shape, other))
        ]
    seen.clear()
    ratios, samples = rb.rebench_picks(gemms, picks, lines, run_bench, runs=3)
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
    for r in range(3):
        run = tmp_path / "work" / f"run_{r}" / "stage03" / "stage03_summary.json"
        assert json.loads(run.read_text())["n_blocks"] == 1
    assert [line_solution_list(b) for b in run1] == [
        rb.rotate(line_solution_list(a), 1) for a in run0
    ]
    assert all("algo_method" not in ln for ln in run0)
