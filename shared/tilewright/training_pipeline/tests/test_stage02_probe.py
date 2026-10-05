# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import sys

import yaml

import stage02_probe as s02
from bench_fakes import (
    BANNER,
    Gemm,
    bench_line,
    measured_lines,
    preamble,
    write_fake_bench,
)
from lib.bench_yaml import read_bench_lines, shape_of_line

PROBE = {
    "first_line_iters": 5,
    "first_line_cold_iters": 5,
    "other_lines_iters": 3,
    "other_lines_cold_iters": 3,
    "duration_us": 1000,
    "fast_us_threshold": 30.0,
    "ratio_light": 0.3,
    "ratio_heavy": 0.8,
}
GEMMS = [Gemm(64, 64, 64), Gemm(96, 32, 512), Gemm(128, 256, 64), Gemm(33, 77, 1024)]


def test_stamp_replaces_or_appends_knobs():
    line = "- {M: 1, iters: 50, cold_iters: 50, min_iters: 7, skip_slow_solution_ratio: 0.5}\n"
    out = s02._stamp_iters_in_line(line, 9, 8, skip_ratio=0.4567, rsn=-1)
    assert "iters: 9" in out and "cold_iters: 8" in out and "min_iters: 7" in out
    assert "skip_slow_solution_ratio: 0.4567" in out
    assert out.endswith(", requested_solution_num: -1}\n")
    bare = s02._stamp_iters_in_line("- {M: 1}", 2, 2, skip_ratio=0.3)
    assert bare == "- {M: 1, cold_iters: 2, iters: 2, skip_slow_solution_ratio: 0.3}\n"
    sci = "- {M: 1, iters: 1, cold_iters: 1, skip_slow_solution_ratio: 5e-01}"
    assert "skip_slow_solution_ratio: 0.8}" in s02._stamp_iters_in_line(
        sci, 1, 1, skip_ratio=0.8
    )


def test_round_robin_probe_split():
    lines = [bench_line(g) for g in GEMMS] + [bench_line(Gemm(5, 5, 5))]
    per_dev, idx = s02._build_probe_yamls(lines, [0, 1], 5, 6, 3, 4)
    assert idx == [[0, 2, 4], [1, 3]]
    assert "iters: 5" in per_dev[0][0] and "cold_iters: 6" in per_dev[0][0]
    assert "iters: 3" in per_dev[0][1] and "cold_iters: 4" in per_dev[0][1]
    assert all("requested_solution_num: 1" in ln for dev in per_dev for ln in dev)


def test_probe_values_map_by_problem_identity(tmp_path):
    log = tmp_path / "probe.log"
    lines = list(BANNER)
    lines += preamble(1) + measured_lines(0, GEMMS[0], 1, 11.0)
    lines += preamble(1)[:1] + ["error: NO solution found! at x"]
    lines += preamble(1) + measured_lines(0, GEMMS[2], 3, 33.0)
    log.write_text("\n".join(lines) + "\n")
    got = s02.probe_us_by_shape(log)
    assert got == {(64, 64, 64, 1): 11.0, (128, 256, 64, 1): 33.0}
    assert s02.probe_us_by_shape(tmp_path / "missing.log") == {}


def test_calibration_modes():
    fast = s02._calibrate(10.0, PROBE)
    assert (fast["iters"], fast["skip_slow_solution_ratio"], fast["bucket"]) == (
        100,
        0.3,
        "fast",
    )
    slow = s02._calibrate(200.0, PROBE)
    assert (slow["iters"], slow["bucket"]) == (5, "slow")
    fail = s02._calibrate(None, PROBE)
    assert (fail["iters"], fail["bucket"]) == (100, "fail")
    sig = dict(PROBE, skip_ratio_mode="sigmoid", skip_ratio_center_us=30.0)
    lo, mid, hi = (s02._calibrate(us, sig) for us in (1.0, 30.0, 1000.0))
    assert lo["skip_slow_solution_ratio"] < mid["skip_slow_solution_ratio"] == 0.5
    assert mid["skip_slow_solution_ratio"] < hi["skip_slow_solution_ratio"] < 0.95


def test_probe_end_to_end_with_fake_bench(tmp_path, monkeypatch):
    in_dir = tmp_path / "stage01"
    in_dir.mkdir()
    (in_dir / "shapes.yaml").write_text("".join(bench_line(g) for g in GEMMS))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"probe": PROBE}))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"no_solution": ["96x32x512x1"], "us": {"128x256x64x1": 40.0}})
    )
    monkeypatch.setenv("FAKE_BENCH_PLAN", str(plan))
    out = tmp_path / "stage02"
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "probe_0007_lines1-1_gpu7.log").write_text("stale\n")
    argv = ["stage02", "--in-dir", in_dir, "--out-dir", out, "--config-yaml", cfg]
    argv += ["--bench-binary", write_fake_bench(tmp_path), "--devices", "0,1"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv + ["--quiet"]])
    assert s02.main() == 0
    probed = read_bench_lines(out / "shapes_probed.yaml")
    assert [shape_of_line(ln) for ln in probed] == [
        (g.m, g.n, g.k, g.batch) for g in GEMMS
    ]
    assert all("requested_solution_num: -1" in ln for ln in probed)
    # One candidate under rsn 1, timed at 1.25 x the plan's base time.
    assert "iters: 80," in probed[0] and "skip_slow_solution_ratio: 0.3" in probed[0]
    assert "iters: 100," in probed[1]
    assert "iters: 20," in probed[2] and "skip_slow_solution_ratio: 0.8" in probed[2]
    assert "iters: 80," in probed[3]
    logs = sorted(p.name for p in (out / "logs").iterdir())
    assert logs == ["probe_0000_lines1-3_gpu0.log", "probe_0001_lines2-4_gpu1.log"]


def test_missing_probe_keys_fail(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"probe": {"duration_us": 1}}))
    argv = ["stage02", "--in-dir", tmp_path, "--out-dir", tmp_path / "o"]
    argv += ["--config-yaml", cfg, "--bench-binary", tmp_path, "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    assert s02.main() == 1
