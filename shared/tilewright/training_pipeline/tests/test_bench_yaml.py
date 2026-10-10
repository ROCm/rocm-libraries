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


def test_a_zero_skip_ratio_leaves_the_line_without_one():
    shape = Shape(64, 64, 64, 1)
    unset = bench_yaml.render_bench_line(shape, _cfg())
    assert "skip_slow_solution_ratio" not in _kv(unset)
    for ratio in (0, 0.0):
        cfg = _cfg(skip_slow_solution_ratio=ratio)
        assert bench_yaml.render_bench_line(shape, cfg) == unset
        knobs = {"skip_slow_solution_ratio": 0.9}
        assert bench_yaml.render_bench_line(shape, cfg, knobs) == unset


def test_every_line_is_timed_adaptively_with_every_field():
    kv = _kv(bench_yaml.render_bench_line(Shape(64, 64, 64, 1), _cfg()))
    assert kv["adaptive"] == "true"
    assert "iters" not in kv and "cold_iters" not in kv
    assert {k: kv[k] for k in bench_yaml.ADAPTIVE_KEYS} == {
        k: str(v) for k, v in bench_yaml.ADAPTIVE_DEFAULTS.items()
    }
    assert "skip_slow_solution_ratio" not in kv
    cfg = _cfg(adaptive={"measure_time": 2.0, "bogus": 1})
    kv = _kv(
        bench_yaml.render_bench_line(
            Shape(64, 64, 64, 1), cfg, {"iters": 9, "cold_iters": 9}
        )
    )
    assert "iters" not in kv and "cold_iters" not in kv
    assert kv["measure_time"] == "2.0" and kv["noise_threshold"] == "0.05"
    assert "bogus" not in kv


def test_adaptive_line_times_fixed_iteration_lines_adaptively():
    old = (
        "- {function: matmul, M: 8, N: 8, K: 8, iters: 40, cold_iters: 40, "
        "min_iters: 5, rotating: 1, skip_slow_solution_ratio: 0.25, "
        "solution_index_list: [3, 1, 2], skip_screen_prepass: true}\n"
    )
    cfg = _cfg(adaptive={"measure_time": 2.0})
    out = bench_yaml.adaptive_line(old, cfg)
    assert bench_yaml.line_solution_list(out) == [3, 1, 2]
    kv = _kv(bench_yaml.drop_line_knob(out, "solution_index_list"))
    assert out.endswith("}\n") and "iters" not in kv and "cold_iters" not in kv
    assert kv["adaptive"] == "true" and bench_yaml.line_is_adaptive(out)
    assert {k: kv[k] for k in bench_yaml.ADAPTIVE_KEYS} == {
        k: str(v) for k, v in bench_yaml.adaptive_settings(cfg).items()
    }
    assert kv["skip_slow_solution_ratio"] == "0.25" and kv["min_iters"] == "3"
    assert kv["skip_screen_prepass"] == "true" and kv["rotating"] == "1"
    assert bench_yaml.adaptive_line(out, _cfg(adaptive={"measure_time": 9.0})) == out
    new = bench_yaml.render_bench_line(Shape(8, 8, 8, 1), cfg)
    assert bench_yaml.adaptive_line(new, _cfg()) == new
    assert not bench_yaml.line_is_adaptive(old)
    assert not bench_yaml.line_is_adaptive("- {M: 1, adaptive: false}")


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


@pytest.mark.parametrize(
    "changes, want",
    [
        ({}, ("f8_r", "f8_r", "bf16_r", "bf16_r", "f32_r", 1, 1)),
        (
            {"a_type": "bf16_r", "b_type": "bf16_r", "compute_type": "c_f32_r"},
            ("bf16_r", "bf16_r", "bf16_r", "bf16_r", "f32_r", 0, 0),
        ),
        (
            {"compute_type": "c_xf32_r", "scaleA": 3},
            (None, None, None, None, "xf32_r", 3, 1),
        ),
        (
            {"compute_type": "c_f32_fast_bf16_r"},
            (None, None, None, None, "f32_bf16_r", 1, 1),
        ),
    ],
)
def test_printed_types_are_what_a_bench_row_of_the_line_prints(changes, want):
    import bench_fakes as bf

    cfg = _cfg()
    cfg["hipblaslt"].update(changes)
    got = bench_yaml.printed_types(cfg["hipblaslt"])
    keys = bench_yaml.TYPE_COLUMNS + bench_yaml.SCALE_COLUMNS
    assert list(got) == list(keys)
    for key, value in zip(keys, want):
        assert value is None or got[key] == value
    line = bench_yaml.render_bench_line(Shape(64, 32, 128, 1), cfg)
    assert bf.line_types(line) == tuple(str(got[k]) for k in keys)


def test_a_compute_type_the_bench_prints_no_name_for_is_left_out():
    cfg = _cfg()
    cfg["hipblaslt"]["compute_type"] = "c_f32_pedantic_r"
    assert "compute_type" not in bench_yaml.printed_types(cfg["hipblaslt"])


@pytest.mark.parametrize(
    "text, code",
    [
        ("3", 3),
        (" 0 ", 0),
        ("Block_32_UE8M0", 3),
        ("none", 0),
        ("Scalar", 1),
        ("x", None),
    ],
)
def test_scale_mode_code(text, code):
    assert bench_yaml.scale_mode_code(text) == code


# ── long lines ───────────────────────────────────────────────────────────────

BF16 = {
    "transA": "T",
    "transB": "N",
    "a_type": "bf16_r",
    "b_type": "bf16_r",
    "c_type": "bf16_r",
    "d_type": "bf16_r",
    "scale_type": "f32_r",
    "compute_type": "c_f32_r",
}


def _long(**block):
    cfg = copy.deepcopy(CFG)
    cfg["hipblaslt"] = dict(BF16)
    cfg["bench"]["adaptive_long"] = {"enabled": True, **block}
    return cfg


def test_the_roofline_is_the_larger_of_the_compute_and_memory_terms():
    shape = (1000, 2000, 500, 3)
    flops_us = 2 * 1000 * 2000 * 500 * 3 / (10.0 * 1e6)
    bytes_us = 3 * 2 * (1000 * 500 + 2000 * 500 + 1000 * 2000) / (5.0 * 1e3)
    roof = bench_yaml.roofline_us
    assert roof(shape, BF16, 10.0, None) == pytest.approx(flops_us)
    assert roof(shape, BF16, None, 5.0) == pytest.approx(bytes_us)
    assert roof(shape, BF16, 10.0, 5.0) == pytest.approx(max(flops_us, bytes_us))
    assert roof(shape, BF16, None, None) is None
    f8 = dict(BF16, a_type="f8_r", b_type="f8_r")
    assert roof(shape, f8, None, 5.0) < roof(shape, BF16, None, 5.0)
    # A type of unknown size has no byte count a lower bound may assume.
    odd = dict(BF16, b_type="mx9_r")
    assert roof(shape, odd, None, 5.0) is None
    assert roof(shape, odd, 10.0, 5.0) == pytest.approx(flops_us)
    assert roof(shape, {}, 10.0, 5.0) == pytest.approx(flops_us)


def test_long_lines_get_the_long_knobs_and_short_lines_the_default():
    cfg = _long(peak_tflops=1.0)
    small, big = Shape(64, 64, 64, 1), Shape(1024, 1024, 1024, 1)
    assert not bench_yaml.is_long_line(small.as_tuple(), cfg)
    assert bench_yaml.is_long_line(big.as_tuple(), cfg)
    base = bench_yaml.adaptive_settings(cfg)
    assert base == bench_yaml.ADAPTIVE_DEFAULTS
    assert bench_yaml.adaptive_settings(cfg, small.as_tuple()) == base
    assert bench_yaml.adaptive_settings(cfg, big.as_tuple()) == dict(
        base, warmup_time=0.5, sample_time=0.25, max_measure_time=4.0
    )
    kv = _kv(bench_yaml.render_bench_line(big, cfg))
    assert (kv["warmup_time"], kv["sample_time"], kv["max_measure_time"]) == (
        "0.5",
        "0.25",
        "4.0",
    )
    assert kv["measure_time"] == "1.5" and kv["min_iters"] == "3"
    kv = _kv(bench_yaml.render_bench_line(small, cfg))
    assert kv["max_measure_time"] == "8.0"
    tuned = _long(peak_tflops=1.0, min_roofline_us=1e9, warmup_time=0.7)
    assert not bench_yaml.is_long_line(big.as_tuple(), tuned)
    tuned["bench"]["adaptive_long"]["min_roofline_us"] = 0.1
    assert bench_yaml.adaptive_settings(tuned, small.as_tuple())["warmup_time"] == 0.7
    off = _long(peak_tflops=1.0, enabled=False)
    assert bench_yaml.adaptive_settings(off, big.as_tuple()) == base


def test_long_lines_need_a_rate():
    with pytest.raises(ValueError, match="peak_tflops or bandwidth_gbs"):
        bench_yaml.render_bench_line(Shape(64, 64, 64, 1), _long())


def test_old_lines_become_adaptive_with_the_knobs_of_their_shape():
    cfg = _long(peak_tflops=1.0)
    old = (
        "- {function: matmul, M: 1024, N: 1024, K: 1024, batch_count: 1, "
        "iters: 10, cold_iters: 2}\n"
    )
    kv = _kv(bench_yaml.adaptive_line(old, cfg))
    assert kv["adaptive"] == "true" and "iters" not in kv
    assert (kv["warmup_time"], kv["max_measure_time"]) == ("0.5", "4.0")
    short = old.replace("M: 1024", "M: 8").replace("N: 1024", "N: 8")
    assert _kv(bench_yaml.adaptive_line(short, cfg))["max_measure_time"] == "8.0"
