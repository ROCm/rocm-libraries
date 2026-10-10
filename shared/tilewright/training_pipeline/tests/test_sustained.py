# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The sustained finalist re-measurement: power-capped GEMMs and their
finalists, the stage on the fake bench, stage04's flag, and the labels
lib/repeats.py makes of the sustained rows."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

import bench_fakes as bf
import fake_hipblaslt as fh
import stage05_train as s5
from gpu_fakes import wait_for
from lib import attempts
from lib import rebench as rb
from lib import repeats
from lib import sustained as sus
from lib.bench_log import iter_problems
from lib.bench_yaml import line_has_prescreen, line_solution_list, read_bench_lines
from lib.bench_yaml import shape_of_line
from lib.planner import stable_rng
from test_rebench import as_log, bench_env, only, problem_lines

PIPELINE_DIR = Path(__file__).resolve().parent.parent
STEM = "TensileLibrary_test_gfx950"
G = bf.Gemm(4096, 4096, 4096)


def settings(**kw):
    return dict(sus.DEFAULTS, **kw)


def test_power_capped_gemms_are_long_or_compute_bound():
    st = settings(min_us=1000.0, compute_bound_tflops=700.0, compute_bound_min_us=100.0)
    shape = (8192, 8192, 8192, 1)
    flops = 2.0 * 8192**3
    fast = flops / (800.0 * 1e6)
    assert sus.tflops(shape, fast) == pytest.approx(800.0)
    assert sus.power_capped(shape, fast, st) == "compute_bound"
    assert sus.power_capped(shape, 3000.0, st) == "long"
    assert sus.power_capped((64, 64, 64, 1), 50.0, st) == ""
    mid = (4096, 4096, 4096, 1)
    rate_us = 2.0 * 4096**3 / 1e6
    assert sus.power_capped(mid, rate_us / 600.0, st) == ""
    assert sus.power_capped(mid, rate_us / 800.0, st) == "compute_bound"
    small = (1024, 1024, 1024, 1)
    assert sus.power_capped(small, 2.0 * 1024**3 / (800.0 * 1e6), st) == ""


def test_finalists_are_the_fastest_and_the_near_winners():
    tested = [(0, 100, 1000.0), (1, 101, 1010.0), (2, 102, 1100.0)]
    tested += [(3, 103, 1200.0), (4, 104, 1300.0), (5, 105, 1040.0)]
    skipped = [(6, 106, 1050.0), (7, 107, 5000.0)]
    prob = only(as_log(problem_lines(G, tested, skipped)))
    st = settings(min_us=500.0, top_k=2, tolerance=0.06, max_kernels=12)
    f = sus.finalists(prob, st)
    assert f.reason == "long" and f.winner_us == 1000.0
    assert f.sol_indices == [100, 101, 105, 106]
    assert [c.skipped for c in f.candidates] == [False, False, False, True]
    st = settings(min_us=500.0, top_k=5, tolerance=0.0, max_kernels=3)
    assert sus.finalists(prob, st).sol_indices == [100, 101, 105]
    assert sus.finalists(prob, settings(min_us=2000.0)) is None
    alone = only(as_log(problem_lines(G, [(0, 100, 1000.0)])))
    assert sus.finalists(alone, settings(min_us=500.0)) is None
    # A blocked kernel is no finalist and sets no winner.
    st = settings(min_us=500.0, top_k=2, tolerance=0.06, max_kernels=12)
    f = sus.finalists(prob, st, blocked={100, 106})
    assert f.winner_us == 1010.0 and f.sol_indices == [101, 105]


def test_sustained_lines_time_exactly_the_finalists_in_long_windows():
    line = bf.bench_line(G, skip_slow_solution_ratio=0.5, skip_screen_prepass="true")
    out = sus.sustained_line(line, [5, 3, 9], settings(), rotating_mib=64)
    assert line_solution_list(out) == [5, 3, 9] and not line_has_prescreen(out)
    for key in sus.WINDOW_KEYS:
        assert f"{key}: {float(sus.DEFAULTS[key])}" in out
    assert "skip_slow_solution_ratio: 0" in out
    assert sus.config_errors({}) == []
    bad = {"bench": {"sustained": {"measure_time": 500.0}}}
    assert sus.config_errors(bad) == [
        "bench.sustained.measure_time exceeds max_measure_time"
    ]


def cand(sid, us, **kw):
    return dict({"sol_idx_global": sid, "us": us, "is_skip": False}, **kw)


def test_sustained_rows_label_the_finalists_and_scale_the_rest():
    rows = [
        cand(1, 100.0),
        cand(2, 104.0),
        cand(3, 120.0),
        cand(4, 300.0, is_skip=True),
        cand(1, 125.0, sustained=True),
        cand(1, 121.0, sustained=True),
        cand(1, 130.0, sustained=True),
        cand(2, 115.0, sustained=True),
        cand(2, 117.0, sustained=True),
    ]
    merged = repeats.merge_candidates(rows, "median")
    by = {c["sol_idx_global"]: c for c in merged}
    assert [c["sol_idx_global"] for c in merged] == [1, 2, 3, 4]
    assert (by[1]["us"], by[1]["us_short"], by[1]["n_repeats"]) == (125.0, 100.0, 3)
    assert by[1]["sustained"] and by[1]["short_timed"]
    assert by[2]["us"] == 116.0 and by[2]["us_short"] == 104.0
    ratio = (1.25 + 116.0 / 104.0) / 2
    assert repeats.sustained_ratio(merged) == pytest.approx(ratio)
    assert by[3]["us"] == pytest.approx(120.0 * ratio) and by[3]["us_short"] == 120.0
    assert by[4]["us"] == pytest.approx(300.0 * ratio) and by[4]["is_skip"]
    assert min(merged, key=lambda c: c["us"])["sol_idx_global"] == 2
    again = repeats.merge_candidates(merged, "median")
    assert [(c["sol_idx_global"], c["us"]) for c in again] == [
        (c["sol_idx_global"], c["us"]) for c in merged
    ]
    plain = repeats.merge_candidates(rows[:4], "median")
    assert repeats.sustained_ratio(plain) is None
    assert [c["us"] for c in plain] == [100.0, 104.0, 120.0, 300.0]


def test_a_finalist_screened_in_the_short_windows_does_not_set_the_scale():
    rows = [
        cand(1, 50.0, is_skip=True),
        cand(1, 40.0, sustained=True),
        cand(2, 30.0),
        cand(3, 31.0),
        cand(3, 33.0, sustained=True),
    ]
    merged = {c["sol_idx_global"]: c for c in repeats.merge_candidates(rows, "min")}
    assert merged[1]["short_timed"] is False and merged[3]["short_timed"] is True
    assert merged[2]["us"] == pytest.approx(30.0 * 33.0 / 31.0)


def make_stage03(tmp_path, root, shapes, extra=()):
    """stage03 outputs of `shapes` on the fake bench (every kernel of the
    library, no screen)."""
    from lib.bench_yaml import adaptive_line

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
    proc = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03_load_balance_offline_tuning.py"),
            "--in-dir", str(in_dir), "--out-dir", str(out),
            "--bench-binary", str(root / "clients" / "hipblaslt-bench"),
            "--devices", "0", "--no-warmup-pass", "--quiet", *extra,
        ],
        capture_output=True, text=True, env=bench_env(root), timeout=300,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return in_dir, out


def _run_stage03d(tmp_path, root, in_dir, s03, cfg, out, env):
    libdir = root / "Tensile" / "library" / "gfx950"
    return subprocess.run(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03d_sustained_finalists.py"),
            "--in-dir", str(in_dir), "--stage03-dir", str(s03),
            "--out-dir", str(out), "--config-yaml", str(cfg),
            "--bench-binary", str(root / "clients" / "hipblaslt-bench"),
            "--identity-dir", str(libdir), "--identity-stem", STEM, "--quiet",
        ],
        capture_output=True, text=True, env=env, timeout=300,
    )  # fmt: skip


def test_the_stage_remeasures_finalists_and_stage04_flags_their_rows(
    tmp_path, monkeypatch
):
    root = fh.write_fake_build(tmp_path / "build", "gfx950", STEM)
    libdir = root / "Tensile" / "library" / "gfx950"
    identity = ["--identity-dir", str(libdir), "--identity-stem", STEM]
    shapes = [(4096, 4096, 2048, 1), (2048, 4096, 4096, 2), (64, 64, 64, 1)]
    in_dir, s03 = make_stage03(tmp_path, root, shapes, identity)
    grid = fh.kernels_grid()

    def us(shape, sid):
        return fh.kernel_time_us(*shape, grid[sid]["sizeMapping"])

    fastest = {s: sorted(range(len(grid)), key=lambda i: us(s, i))[0] for s in shapes}
    long_shapes = [s for s in shapes if us(s, fastest[s]) >= 200.0]
    assert long_shapes and len(long_shapes) < len(shapes)
    soaked = {"default": 1.1, **{str(fastest[s]): 1.4 for s in shapes}}
    env = dict(bench_env(root), FAKE_SUSTAINED=json.dumps(soaked))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "bench:\n  sustained:\n    enabled: true\n    min_us: 200.0\n"
        "    top_k: 3\n    tolerance: 0.0\n    runs: 3\n"
    )
    out = tmp_path / "stage03d"
    proc = _run_stage03d(tmp_path, root, in_dir, s03, cfg, out, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((out / "sustained_summary.json").read_text())
    assert summary["n_gemms"] == len(long_shapes) and summary["n_finalists"] == 3 * len(
        long_shapes
    )
    assert [r["rc"] for r in summary["runs"]] == [0, 0, 0]
    assert summary["blocks_per_gpu"] == 1
    per_gemm = {
        (g["m"], g["n"], g["k"], g["batch"]): g
        for g in json.loads((out / "sustained.json").read_text())["gemms"]
    }
    assert set(per_gemm) == set(long_shapes)
    for r in range(3):
        lines = read_bench_lines(out / "runs" / f"run_{r}" / "shapes.yaml")
        order = [s for s in per_gemm]
        stable_rng(42, "sustained", r).shuffle(order)
        assert [shape_of_line(ln) for ln in lines] == order
        for ln in lines:
            shape = shape_of_line(ln)
            finals = [f["sol_index"] for f in per_gemm[shape]["finalists"]]
            want = rb.run_order(finals, r, 42, ("sustained", *shape))
            assert line_solution_list(ln) == want
            assert "warmup_time: 250.0" in ln and "max_measure_time: 400.0" in ln
    for shape, g in per_gemm.items():
        best = g["finalists"][0]
        assert best["sol_index"] == fastest[shape]
        assert best["median_us"] == pytest.approx(us(shape, fastest[shape]) * 1.4)
        assert len(best["runs_us"]) == 3

    import stage04_convert_to_enriched_dataset as s04

    logs = sorted((out / "runs").glob("run_*/stage03/logs"))
    argv = ["stage04", "--log-dir", str(s03 / "logs"), "--out-dir", str(tmp_path / "e")]
    argv += ["--dat-dir", str(libdir), "--library-stem", STEM, "--quiet"]
    for d in logs:
        argv += ["--sustained-log-dir", str(d)]
    monkeypatch.setattr(sys, "argv", argv)
    assert s04.main() == 0
    rows = []
    for p in sorted((tmp_path / "e").glob("chunk_*.csv")):
        with p.open() as f:
            rows += list(csv.DictReader(f))
    flagged = [r for r in rows if r["is_sustained"] == "True"]
    assert len(flagged) == 3 * 3 * len(long_shapes)
    assert {r["is_origami_pick"] for r in flagged} == {"0"}
    enrich = json.loads((tmp_path / "e" / "enrich_summary.json").read_text())
    assert enrich["gemms"] == len(shapes)
    assert enrich["counters"]["sustained_problems"] == 3 * len(long_shapes)

    cells, stats = s5.load_enriched_chunks(str(tmp_path / "e"), True)
    gemms = {
        (g["m"], g["n"], g["k"], g["batch_count"]): g
        for gs in cells.values()
        for g in gs
    }
    assert stats["gemms_sustained"] == len(long_shapes)
    for shape in shapes:
        g = gemms[shape]
        by = {c["sol_idx_global"]: c for c in g["candidates"]}
        if shape not in long_shapes:
            assert not any(c.get("sustained") for c in g["candidates"])
            continue
        finals = [f["sol_index"] for f in per_gemm[shape]["finalists"]]
        assert all(by[s]["sustained"] for s in finals)
        assert g["sustained_ratio"] == pytest.approx(1.1, rel=0.01)
        assert by[fastest[shape]]["us"] == pytest.approx(
            us(shape, fastest[shape]) * 1.4, rel=1e-3
        )
        winner = min(
            (c for c in g["candidates"] if not c["is_skip"]), key=lambda c: c["us"]
        )
        assert winner["sol_idx_global"] != fastest[shape]
        assert winner["sol_idx_global"] in finals
        others = [c for c in g["candidates"] if c["sol_idx_global"] not in finals]
        for c in others:
            assert c["us"] == pytest.approx(c["us_short"] * g["sustained_ratio"])

    from lib import run_analysis as ra

    report = {g.key[:4]: g for g in ra.load_bench([tmp_path / "e"]).gemms}
    for shape, g in gemms.items():
        got = dict(zip(report[shape].sids.tolist(), report[shape].us.tolist()))
        assert got == pytest.approx(
            {c["sol_idx_global"]: c["us"] for c in g["candidates"]}
        )


def test_the_stage_leaves_the_configs_blocked_kernels_out_of_the_finalists(
    tmp_path,
):
    """A kernel the config's blocklist names, found through the library the
    logs were benched with, is no finalist and sets no winner, also when
    stage03 timed it fastest."""
    root = fh.write_fake_build(tmp_path / "build", "gfx950", STEM)
    libdir = root / "Tensile" / "library" / "gfx950"
    identity = ["--identity-dir", str(libdir), "--identity-stem", STEM]
    shape = (4096, 4096, 2048, 1)
    in_dir, s03 = make_stage03(tmp_path, root, [shape], identity)
    grid = fh.kernels_grid()
    us = {i: fh.kernel_time_us(*shape, k["sizeMapping"]) for i, k in enumerate(grid)}
    fastest = min(us, key=us.get)
    blocked = {i for i, k in enumerate(grid) if k["name"] == grid[fastest]["name"]}
    kept = {i: t for i, t in us.items() if i not in blocked}
    assert min(kept.values()) >= 200.0
    lst = tmp_path / "blocked.tsv"
    lst.write_text(f"C\t{grid[fastest]['name']}\n")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "bench:\n  sustained:\n    enabled: true\n    min_us: 200.0\n"
        "    top_k: 3\n    tolerance: 0.0\n    runs: 1\n"
        f"correctness:\n  lists: [{str(lst)!r}]\n"
    )
    out = tmp_path / "stage03d"
    proc = _run_stage03d(tmp_path, root, in_dir, s03, cfg, out, bench_env(root))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (g,) = json.loads((out / "sustained.json").read_text())["gemms"]
    finals = {f["sol_index"] for f in g["finalists"]}
    assert len(finals) == 3 and not finals & blocked
    assert finals == set(sorted(kept, key=kept.get)[:3])


def test_the_stage_needs_power_capped_gemms_to_bench(tmp_path):
    root = fh.write_fake_build(tmp_path / "build", "gfx950", STEM)
    libdir = root / "Tensile" / "library" / "gfx950"
    identity = ["--identity-dir", str(libdir), "--identity-stem", STEM]
    in_dir, s03 = make_stage03(tmp_path, root, [(64, 64, 64, 1)], identity)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("bench:\n  sustained:\n    enabled: true\n")
    out = tmp_path / "stage03d"
    proc = _run_stage03d(tmp_path, root, in_dir, s03, cfg, out, bench_env(root))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((out / "sustained_summary.json").read_text())
    assert summary["n_gemms"] == 0 and summary["runs"] == []
    cfg.write_text(
        "bench:\n  sustained:\n    measure_time: 500.0\n    max_measure_time: 400.0\n"
    )
    proc = _run_stage03d(tmp_path, root, in_dir, s03, cfg, out, bench_env(root))
    assert proc.returncode == 1
    assert "measure_time exceeds max_measure_time" in proc.stdout + proc.stderr


def test_a_sustained_run_ends_with_the_stage(tmp_path):
    """stage03d runs each sustained run's stage03 in its own session; when
    stage03d is killed, that stage03 and its bench must not run on."""
    root = fh.write_fake_build(tmp_path / "build", "gfx950", STEM)
    libdir = root / "Tensile" / "library" / "gfx950"
    identity = ["--identity-dir", str(libdir), "--identity-stem", STEM]
    in_dir, s03 = make_stage03(tmp_path, root, [(4096, 4096, 2048, 1)], identity)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("bench:\n  sustained:\n    enabled: true\n    min_us: 1.0\n")
    started = tmp_path / "bench.pid"
    hanging = tmp_path / "hanging-bench"
    hanging.write_text(
        f"#!{sys.executable}\nimport os, time\n"
        f"open({str(started)!r}, 'w').write(str(os.getpid()))\ntime.sleep(600)\n"
    )
    hanging.chmod(0o755)
    token = attempts.new_token()
    proc = subprocess.Popen(
        [
            sys.executable,
            str(PIPELINE_DIR / "stages" / "stage03d_sustained_finalists.py"),
            "--in-dir", str(in_dir), "--stage03-dir", str(s03),
            "--out-dir", str(tmp_path / "stage03d"), "--config-yaml", str(cfg),
            "--bench-binary", str(hanging), *identity, "--quiet",
        ],
        env=attempts.environ_with(token, bench_env(root)),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # fmt: skip
    try:
        assert wait_for(lambda: started.is_file() or proc.poll() is not None, 120)
        assert proc.poll() is None and len(attempts.processes_of(token)) >= 3
        proc.kill()
        proc.wait()
        assert wait_for(lambda: not attempts.processes_of(token), 30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        attempts.stop(token, grace_s=5)
