# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math
import random

import pytest

from bench_fakes import write_kfd_tree
from lib import grid, shapes
from lib.shapes import ShapeFilter

GIB = 1024**3


def _filter(**kw):
    kw.setdefault("memory_budget_bytes", 64 * GIB)
    return ShapeFilter(**kw)


# ── footprint ────────────────────────────────────────────────────────────────


def test_bytes_per_element():
    assert shapes.bytes_per_element("f8_r") == 1
    assert shapes.bytes_per_element("bf16_r") == 2
    assert shapes.bytes_per_element("f4_r") == 0.5
    assert shapes.bytes_per_element("unknown_r") == 4
    assert shapes.bytes_per_element(None) == 4


def test_footprint_counts_each_operand_with_its_own_type():
    fp = shapes.footprint_bytes(
        64, 32, 128, 2, a_type="f8_r", b_type="f8_r", c_type="bf16_r", d_type="bf16_r"
    )
    assert fp == 2 * (64 * 128 + 32 * 128 + 2 * 64 * 32 + 2 * 64 * 32)


def test_footprint_adds_mx_block_scales_rounded_up():
    base = dict(a_type="f8_r", b_type="f8_r", c_type="bf16_r", d_type="bf16_r")
    plain = shapes.footprint_bytes(33, 17, 64, 1, **base)
    mx = shapes.footprint_bytes(33, 17, 64, 1, scale_a=3, scale_b=3, **base)
    assert mx - plain == math.ceil(33 * 64 / 32) + math.ceil(17 * 64 / 32)
    scalar = shapes.footprint_bytes(33, 17, 64, 1, scale_a=1, scale_b=1, **base)
    assert scalar == plain


# ── filter ───────────────────────────────────────────────────────────────────


def test_int32_flop_and_memory_caps():
    f = _filter()
    assert f.is_benchable(1024, 1024, 1024, 1)
    assert not f.is_benchable(0, 8, 8, 1)
    assert not f.is_benchable(70000, 70000, 16, 1)  # M*N beyond int32
    assert not f.is_benchable(30000, 30000, 30000, 1)  # FLOP cap
    tight = _filter(memory_budget_bytes=600 * 1024**2, a_type="f32_r")
    assert tight.is_benchable(128, 128, 128, 1)
    assert not tight.is_benchable(8192, 8192, 1024, 1)


def test_batched_caps():
    f = _filter()
    assert f.is_benchable(256, 256, 256, 64)
    assert not f.is_benchable(9000, 16, 16, 2)  # per-dimension cap
    assert not f.is_benchable(4096, 4096, 4096, 8)  # batched FLOP cap


def test_parse_and_apply_exclude_rules():
    rules = shapes.parse_exclude_rules(
        [{"max_mn": 8}, {"min_k": 65536, "max_batch": 1}]
    )
    assert shapes.is_excluded(2, 500, 64, 1, rules)
    assert shapes.is_excluded(500, 8, 64, 4, rules)
    assert shapes.is_excluded(64, 64, 65536, 1, rules)
    assert not shapes.is_excluded(64, 64, 65536, 2, rules)
    assert not shapes.is_excluded(9, 9, 65535, 1, rules)
    assert shapes.parse_exclude_rules(None) == ()


@pytest.mark.parametrize(
    "bad",
    [
        {"max_mn": 8},
        [{}],
        [{"max_q": 8}],
        [{"min_k": "65536"}],
        [{"min_k": True}],
        ["min_k"],
    ],
)
def test_bad_exclude_rules_raise(bad):
    with pytest.raises(ValueError):
        shapes.parse_exclude_rules(bad)


def test_filter_from_config():
    cfg = {
        "hipblaslt": {
            "a_type": "f8_r",
            "b_type": "f8_r",
            "c_type": "bf16_r",
            "d_type": "bf16_r",
            "scaleA": 3,
            "scaleB": 3,
        },
        "bench": {"device_memory_gib": 10, "rotating_mb": 64},
        "seed": {"exclude": [{"max_mn": 8}]},
    }
    f = ShapeFilter.from_config(cfg)
    assert f.memory_budget_bytes == int(10 * GIB * shapes.MEMORY_SAFE_FRACTION)
    assert f.rotating_bytes == 64 * 1024**2
    assert f.mx_block == 32
    assert f.c_type == "bf16_r"
    assert not f.accepts(8, 128, 128, 1)
    assert f.accepts(9, 128, 128, 1)


# ── memory budget ────────────────────────────────────────────────────────────


def test_kfd_reader_takes_the_smallest_gpu(tmp_path):
    root = write_kfd_tree(
        tmp_path / "nodes",
        [[(1, 40 * GIB)], [(1, 24 * GIB), (2, 8 * GIB)], [(0, 99 * GIB)]],
        gpu_local_mem=[0, 0, 16 * GIB],
    )
    assert shapes.read_kfd_device_memory_bytes(root) == 16 * GIB
    assert shapes.read_kfd_device_memory_bytes(tmp_path / "missing") is None
    cpu_only = write_kfd_tree(tmp_path / "cpu", [], cpu_nodes=2)
    assert shapes.read_kfd_device_memory_bytes(cpu_only) is None


def test_memory_budget_precedence(tmp_path):
    root = write_kfd_tree(tmp_path / "nodes", [[(1, 48 * GIB)]])
    from_cfg = shapes.device_memory_budget(
        {"bench": {"device_memory_gib": 12.5}}, nodes_dir=root
    )
    assert (from_cfg.source, from_cfg.device_bytes) == ("config", int(12.5 * GIB))
    from_kfd = shapes.device_memory_budget({}, nodes_dir=root)
    assert (from_kfd.source, from_kfd.device_bytes) == ("kfd", 48 * GIB)
    fallback = shapes.device_memory_budget(None, nodes_dir=tmp_path / "none")
    assert fallback.source == "fallback"
    assert fallback.device_bytes == shapes.FALLBACK_DEVICE_MEMORY_GIB * GIB
    assert fallback.budget_bytes == int(fallback.device_bytes * 0.70)


@pytest.mark.parametrize("bad", [0, -4, "64", True, float("nan")])
def test_bad_device_memory_raises(bad):
    with pytest.raises(ValueError):
        shapes.device_memory_budget({"bench": {"device_memory_gib": bad}})


# ── synthesis ────────────────────────────────────────────────────────────────


def _synth(cell, n, seed=7, **kw):
    f = kw.pop("shape_filter", None) or _filter()
    return shapes.synthesize_cell_shapes(cell, n, random.Random(seed), f, **kw)


@pytest.mark.parametrize(
    "cell",
    ["Tiny|Small|MidK|Bnone", "Large|Large|LargeK|Bnone", "Mid|Large|LargeK|Bany"],
)
def test_synthesis_is_deterministic_distinct_and_in_cell(cell):
    a = _synth(cell, 120, n_strata=2)
    assert a == _synth(cell, 120, n_strata=2)
    assert a != _synth(cell, 120, seed=8, n_strata=2)
    assert len(a) == len(set(a)) == 120
    f = _filter()
    for m, n, k, b in a:
        assert grid.cell_key(m, n, k, b) == cell
        assert f.accepts(m, n, k, b)
        assert max(m * n, m * k, n * k) <= 2**31 - 1
        if b > 1:
            assert b in grid.BANY_VALUES


def test_quota_of_infeasible_strata_is_redistributed():
    blocked = _filter(exclude=shapes.parse_exclude_rules([{"min_m": 8000}]))
    out = _synth("Large|Large|LargeK|Bnone", 200, n_strata=2, shape_filter=blocked)
    assert len(out) == 200
    assert all(m < 8000 for m, _, _, _ in out)


def test_small_boxes_return_fewer_distinct_shapes():
    out = _synth("Tiny|Tiny|TinyK|Bnone", 50, axis_bounds={"M": (1, 2), "N": (1, 2)})
    assert 0 < len(out) <= 2 * 2 * 32
    assert len(out) == len(set(out))
    assert all(m <= 2 and n <= 2 for m, n, _, _ in out)


def test_mx_configs_get_block_aligned_k():
    mx = _filter(a_type="f8_r", b_type="f8_r", scale_a=3, scale_b=3)
    out = _synth("Mid|Mid|LargeK|Bnone", 80, shape_filter=mx, n_strata=2)
    assert len(out) == 80
    assert all(k % 32 == 0 for _, _, k, _ in out)
    tiny = _synth("Tiny|Tiny|TinyK|Bnone", 40, shape_filter=mx, n_strata=2)
    assert tiny and all(k == 32 for _, _, k, _ in tiny)


def test_empty_requests_and_boxes():
    assert _synth("Mid|Mid|MidK|Bnone", 0) == []
    assert _synth("Mid|Mid|MidK|Bnone", 10, axis_bounds={"M": (600, 700)}) == []


# ── re-used shapes file ──────────────────────────────────────────────────────


def test_reused_shapes_round_trip(tmp_path):
    path = tmp_path / shapes.REUSED_SHAPES_FILE
    shapes.write_reused_shapes(path, [(4, 5, 6, 1), (1, 2, 3, 8), (4, 5, 6, 1)], 3)
    text = path.read_text()
    assert text.startswith("round_index: 3\nshapes:\n- [1, 2, 3, 8]\n")
    assert shapes.read_reused_shapes(path) == {(1, 2, 3, 8), (4, 5, 6, 1)}
    assert shapes.read_reused_shapes(tmp_path) == {(1, 2, 3, 8), (4, 5, 6, 1)}
    shapes.write_reused_shapes(path, [], 0)
    assert shapes.read_reused_shapes(path) == set()
    assert shapes.read_reused_shapes(tmp_path / "nope") == set()
