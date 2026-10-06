# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import copy

import pytest

from lib import bench_yaml
from lib.shapes import Shape

CFG = {
    "hipblaslt": {
        "transA": "T",
        "transB": "N",
        "a_type": "f8_r",
        "b_type": "f8_r",
        "c_type": "bf16_r",
        "d_type": "bf16_r",
        "scale_type": "f32_r",
        "compute_type": "f32_r",
    },
    "bench": {},
}


def _cfg(**bench):
    cfg = copy.deepcopy(CFG)
    cfg["bench"].update(bench)
    return cfg


def _kv(line):
    body = line.strip()[3:-1]
    return dict(part.split(": ", 1) for part in body.split(", "))


def test_strides_for():
    assert bench_yaml.strides_for("N", "N", 10, 20, 30) == (10, 30, 10, 10)
    assert bench_yaml.strides_for("T", "T", 10, 20, 30) == (30, 20, 10, 10)


def test_render_line_fields_and_default_scales():
    kv = _kv(bench_yaml.render_bench_line(Shape(64, 32, 128, 2), _cfg()))
    assert kv["function"] == "matmul"
    assert (kv["M"], kv["N"], kv["K"], kv["batch_count"]) == ("64", "32", "128", "2")
    assert (kv["lda"], kv["ldb"], kv["ldc"]) == ("128", "128", "64")
    assert kv["scaleA"] == kv["scaleB"] == "1"
    assert kv["requested_solution_num"] == "-1"
    assert kv["iters"] == "50"
    cfg = _cfg()
    cfg["hipblaslt"]["scaleA"] = 3
    cfg["hipblaslt"]["scaleB"] = 3
    kv = _kv(bench_yaml.render_bench_line(Shape(64, 32, 128, 1), cfg))
    assert kv["scaleA"] == kv["scaleB"] == "3"


def test_config_skip_ratio_wins_over_knobs():
    line = bench_yaml.render_bench_line(
        Shape(64, 64, 64, 1),
        _cfg(skip_slow_solution_ratio=0.5),
        {"skip_slow_solution_ratio": 0.9},
    )
    assert _kv(line)["skip_slow_solution_ratio"] == "0.5"


def test_adaptive_drops_iteration_knobs():
    cfg = _cfg(adaptive={"enabled": True, "measure_time": 2.0, "bogus": 1})
    kv = _kv(bench_yaml.render_bench_line(Shape(64, 64, 64, 1), cfg, {"iters": 9}))
    assert "iters" not in kv and "cold_iters" not in kv
    assert kv["adaptive"] == "true"
    assert kv["measure_time"] == "2.0"
    assert "bogus" not in kv
    assert bench_yaml.adaptive_cfg(_cfg(adaptive={"enabled": False})) is None


def test_rotating_is_sized_to_the_problem():
    cfg = _cfg(rotating_mb=512, rotating_target_blocks=16)
    assert bench_yaml.rotating_mb_for(Shape(2, 3, 32, 1), cfg) == 1
    assert bench_yaml.rotating_mb_for(Shape(8192, 8192, 8192, 1), cfg) == 512
    mid = bench_yaml.rotating_mb_for(Shape(1024, 1024, 1024, 1), cfg)
    assert 1 < mid < 512
    assert bench_yaml.rotating_mb_for(Shape(64, 64, 64, 1), _cfg(rotating_mb=0)) == 0
    line = bench_yaml.render_bench_line(Shape(64, 64, 64, 1), cfg, {"rotating": 7})
    assert _kv(line)["rotating"] == "7"


def test_write_and_read_back(tmp_path):
    out = tmp_path / "shapes.yaml"
    shapes = [Shape(1, 2, 3, 1), Shape(40, 50, 60, 4)]
    assert bench_yaml.write_bench_yaml(shapes, _cfg(), out) == 2
    assert bench_yaml.parse_shapes_yaml(out) == [(1, 2, 3, 1), (40, 50, 60, 4)]
    assert len(bench_yaml.read_bench_lines(out)) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shapes.yaml"]


@pytest.mark.parametrize(
    "line, shape",
    [
        ("- {M: 5, N: 6, K: 7, batch_count: 3}", (5, 6, 7, 3)),
        ("- {K: 7, N: 6, M: 5}", (5, 6, 7, 1)),
        ("- {maxM: 9, M: 5, N: 6, K: 7}", (5, 6, 7, 1)),
        ("- {N: 6, K: 7}", None),
    ],
)
def test_shape_of_line(line, shape):
    assert bench_yaml.shape_of_line(line) == shape
