# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv

import pytest

from lib import planner as pl
from lib.bench_yaml import (
    drop_line_knob,
    format_solution_list,
    line_has_prescreen,
    line_solution_list,
    set_line_knob,
    shape_of_line,
)

PTYPE = pl.ProblemType("T", "N", "bf16_r", "bf16_r", "bf16_r", "bf16_r", "c_f32_r")
LDS = 65536


def kern(sid, mt=(64, 64, 64), mi=(16, 16, 32), cha=0, chb=0):
    return {
        "sol_idx_global": sid,
        "mt_m": mt[0],
        "mt_n": mt[1],
        "mt_k": mt[2],
        "mi_m": mi[0],
        "mi_n": mi[1],
        "mi_k": mi[2],
        "occupancy": 1,
        "cache_hints_a": cha,
        "cache_hints_b": chb,
        "grvw_a": 1,
        "grvw_b": 1,
        "gwvw_d": 1,
    }


def pool_kernels(n=40):
    """`n` kernels over distinct (mt, mi) families, plus kernel 999 whose
    macro tile does not fit the LDS."""
    tiles = [(16, 16, 64), (32, 32, 64), (64, 64, 64), (128, 64, 64), (64, 128, 64)]
    mis = [(16, 16, 32), (32, 32, 16), (16, 16, 16), (4, 4, 4), (32, 32, 8)]
    out = []
    for i in range(n):
        out.append(kern(100 + i, mt=tiles[i % 5], mi=mis[(i // 5) % 5], cha=0))
    out.append(kern(999, mt=(256, 256, 256)))
    return out


def make(pool=None, bank=()):
    cat = pl.Catalog.build(pool if pool is not None else pool_kernels(), bank)
    return cat, pl.Feasibility(cat, PTYPE, LDS)


PARAMS = pl.PlanParams(
    top_k=4,
    families_top_f=2,
    family_variants=0,
    explore_random_k=5,
    min_coverage=0,
    max_list=512,
    probe_kernels=3,
    seed=7,
)


class FixedRanker(pl.Ranker):
    def __init__(self, order):
        self.order = list(order)

    def rank(self, shape):
        return list(self.order)


# ── catalog / feasibility ────────────────────────────────────────────────────


def test_catalog_keeps_pool_first_and_groups_families():
    pool = [kern(1), kern(2, mt=(128, 64, 64))]
    bank = [kern(2), kern(3), kern(4, cha=4)]
    cat = pl.Catalog.build(pool, bank)
    assert cat.pool_ids == [1, 2]
    assert cat.bank_ids == [3, 4]
    assert cat.kernels[2]["mt_m"] == 128
    assert cat.families[pl.family_of(kern(1))] == [1, 3]
    assert cat.families[pl.family_of(kern(4, cha=4))] == [4]
    assert cat.nt_available == (False, False)


def test_feasibility_drops_kernels_over_the_lds_budget():
    cat, feas = make()
    shape = (1024, 1024, 1024, 1)
    ok = feas.feasible(shape, cat.pool_ids)
    assert 999 not in ok and len(ok) == len(cat.pool_ids) - 1
    assert not feas.ok(shape, 999)


def test_feasibility_applies_the_given_soft_rules():
    from lib.feasibility import soft_rules

    cat = pl.Catalog.build([kern(1, mt=(16, 16, 64)), kern(2, mt=(128, 128, 64))])
    small_batched = (100, 100, 256, 4)
    v2 = pl.Feasibility(cat, PTYPE, LDS)
    assert v2.feasible(small_batched, [1, 2]) == [2]
    v3 = pl.Feasibility(cat, PTYPE, LDS, rules=soft_rules({}, 3))
    assert v3.feasible(small_batched, [1, 2]) == [1, 2]
    assert v3.rules["small_batched_rule"] is False
    on = pl.Feasibility(cat, PTYPE, LDS, rules={"small_batched_rule": True})
    assert on.feasible(small_batched, [1, 2]) == [2]
    with pytest.raises(ValueError, match="unknown soft rules"):
        pl.Feasibility(cat, PTYPE, LDS, rules={"tiny_rule": True})


# ── seeds ────────────────────────────────────────────────────────────────────


def test_select_seeds_is_per_cell_deterministic_and_ordered():
    shapes = [(m, 64, 1024, 1) for m in range(600, 640)] + [
        (8, 8, 8, 1),
        (9, 9, 9, 1),
    ]
    a = pl.select_seeds(shapes, 3, seed=1)
    assert a == pl.select_seeds(list(shapes), 3, seed=1)
    cells = [pl.shape_cell(s) for s in a]
    assert cells.count(pl.shape_cell(shapes[0])) == 3
    assert cells.count(pl.shape_cell((8, 8, 8, 1))) == 2
    assert a == [s for s in shapes if s in set(a)]
    assert pl.select_seeds(shapes, 3, seed=2) != a


# ── per-shape lists ──────────────────────────────────────────────────────────


def test_plan_shape_order_sources_and_feasibility():
    bank_variant = kern(500, mt=(16, 16, 64), mi=(16, 16, 32))
    cat, feas = make(bank=[bank_variant])
    ranked = [999, 105, 100, 110, 115, 120]
    sp = pl.plan_shape((512, 512, 2048, 1), ranked, feas, PARAMS)
    assert sp.role == "planned"
    assert 999 not in sp.kernels
    assert sp.kernels[:4] == [105, 100, 110, 115]
    fam_100 = cat.families[pl.family_of(cat.kernels[100])]
    assert 500 in fam_100
    for sid in cat.families[pl.family_of(cat.kernels[105])] + fam_100:
        assert sid in sp.kernels
    assert sp.sources["top_k"] == 4
    assert sp.sources["random"] == 5
    assert len(sp.kernels) == len(set(sp.kernels))
    assert sp.probe[0] == 105 and len(sp.probe) == 3
    assert set(sp.probe) <= set(sp.kernels)


def test_plan_shape_is_deterministic_and_seed_dependent():
    _, feas = make()
    shape = (300, 200, 100, 1)
    a = pl.plan_shape(shape, [100, 101], feas, PARAMS)
    b = pl.plan_shape(shape, [100, 101], feas, PARAMS)
    assert a.kernels == b.kernels and a.probe == b.probe
    c = pl.plan_shape(
        shape, [100, 101], feas, pl.PlanParams(**{**PARAMS.as_dict(), "seed": 8})
    )
    assert c.kernels[:2] == a.kernels[:2]
    assert c.kernels != a.kernels


def test_plan_shape_truncates_to_max_list():
    _, feas = make()
    p = pl.PlanParams(**{**PARAMS.as_dict(), "max_list": 6, "explore_random_k": 20})
    sp = pl.plan_shape((300, 200, 100, 1), [100, 101, 102, 103], feas, p)
    assert len(sp.kernels) == 6
    assert sp.kernels[:4] == [100, 101, 102, 103]
    assert sp.n_truncated > 0


def test_family_variants_caps_each_family():
    pool = [kern(1)] + [kern(10 + i) for i in range(5)] + [kern(50, mt=(32, 32, 64))]
    _, feas = make(pool=pool)
    p = pl.PlanParams(
        top_k=1,
        families_top_f=1,
        family_variants=2,
        explore_random_k=0,
        min_coverage=0,
    )
    sp = pl.plan_shape((256, 256, 256, 1), [1], feas, p)
    assert sp.kernels == [1, 10]


# ── coverage ─────────────────────────────────────────────────────────────────


def test_coverage_quota_per_cell():
    _, feas = make()
    shapes = [(600 + 8 * i, 600, 1024, 1) for i in range(6)]
    p = pl.PlanParams(
        top_k=2, families_top_f=0, explore_random_k=0, min_coverage=3, seed=1
    )
    plans, added = pl.build_plan(shapes, feas, p, ranker=FixedRanker([100, 101]))
    cell = pl.shape_cell(shapes[0])
    assert added[cell] > 0
    for sid in feas.feasible(shapes[0], feas.catalog.pool_ids):
        n = sum(1 for sp in plans if sid in sp.kernels)
        assert n >= 3, sid
    again, _ = pl.build_plan(shapes, feas, p, ranker=FixedRanker([100, 101]))
    assert [sp.kernels for sp in again] == [sp.kernels for sp in plans]


def test_seed_shapes_count_toward_coverage():
    _, feas = make()
    shapes = [(600 + 8 * i, 600, 1024, 1) for i in range(4)]
    p = pl.PlanParams(top_k=1, families_top_f=0, explore_random_k=0, min_coverage=2)
    plans, added = pl.build_plan(
        shapes, feas, p, ranker=FixedRanker([100]), seeds=shapes[:2]
    )
    assert [sp.role for sp in plans] == ["seed", "seed", "planned", "planned"]
    assert plans[0].kernels is None
    assert added[pl.shape_cell(shapes[0])] == 0
    assert all(sp.kernels == [100] for sp in plans[2:])


def test_exhaustive_plan_and_no_ranker_bench_everything():
    _, feas = make()
    shapes = [(64, 64, 64, 1), (2048, 2048, 2048, 1)]
    for kw in ({"exhaustive": True, "ranker": FixedRanker([100])}, {"ranker": None}):
        plans, added = pl.build_plan(shapes, feas, PARAMS, **kw)
        assert added == {}
        assert all(sp.kernels is None and sp.role == "exhaustive" for sp in plans)
        assert all(len(sp.probe) == 3 for sp in plans)
        assert plans[1].n_feasible == len(feas.catalog.pool_ids) - 1


def test_plan_document_round_trip():
    _, feas = make()
    plans, _ = pl.build_plan(
        [(64, 64, 64, 1), (128, 64, 64, 2)], feas, PARAMS, ranker=FixedRanker([100])
    )
    doc = pl.plan_document(plans, {"mode": "planned"})
    look = pl.plan_lookup(doc)
    assert set(look) == {(64, 64, 64, 1), (128, 64, 64, 2)}
    assert look[(128, 64, 64, 2)]["kernels"] == plans[1].kernels
    assert doc["version"] == pl.PLAN_VERSION and doc["mode"] == "planned"


# ── rankers ──────────────────────────────────────────────────────────────────


def test_neighbors_ranker_prefers_nearest_seed_of_the_same_cell():
    near, far = (1024, 1024, 1024, 1), (4096, 4096, 4096, 1)
    other_cell = (8, 8, 8, 1)
    seeds = {
        near: {1: 10.0, 2: 20.0, 3: 40.0},
        far: {1: 30.0, 2: 10.0},
        other_cell: {9: 1.0},
    }
    r = pl.NeighborsRanker(seeds, k=1)
    assert r.rank((1100, 1000, 1000, 1)) == [1, 2, 3]
    assert r.rank((4000, 4000, 4000, 1)) == [2, 1]
    assert r.rank((9, 9, 9, 1)) == [9]
    two = pl.NeighborsRanker(seeds, k=2).rank((2048, 2048, 2048, 1))
    assert set(two) == {1, 2, 3} and two[-1] == 3


def test_seed_latencies_reads_tested_and_skip_rows(tmp_path):
    rows = [
        {"m": 4, "n": 8, "k": 16, "batch_count": 1, "sol_idx_global": 1, "us": "5.0"},
        {"m": 4, "n": 8, "k": 16, "batch_count": 1, "sol_idx_global": 1, "us": "4.0"},
        {"m": 4, "n": 8, "k": 16, "batch_count": 1, "sol_idx_global": 2, "us": "9"},
        {"m": 4, "n": 8, "k": 16, "batch_count": 1, "sol_idx_global": 3, "us": ""},
        {"m": 2, "n": 2, "k": 2, "batch_count": 3, "sol_idx_global": 1, "us": "1"},
    ]
    with (tmp_path / "chunk_0000.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lat = pl.seed_latencies(str(tmp_path))
    assert lat == {(4, 8, 16, 1): {1: 4.0, 2: 9.0}, (2, 2, 2, 3): {1: 1.0}}


def test_model_ranker_orders_by_engine_scores(tmp_path, make_bundle):
    tw = pytest.importorskip("tilewright")
    pytest.importorskip("torch")
    import stage01b_plan_benchmarks as s01b
    from lib.hardware import DeviceHardware

    cat, _ = make()
    label = "Large|Large|LargeK|Bnone"
    sigs = sorted({pl.family_of(k) for k in cat.kernels.values()})
    models = make_bundle(tmp_path, [label], signatures={label: [list(s) for s in sigs]})
    data = s01b.model_bytes(models, "gfx950", "fp32")
    hw = DeviceHardware(n_cu=256, lds_bytes=LDS, l2_bytes=4 << 20)
    r = pl.ModelRanker(data, cat, PTYPE, hw, tw=tw)
    order = r.rank((2048, 2048, 2048, 1))
    assert order and set(order) <= set(cat.kernels)
    assert len(order) == len(set(order))
    assert order == r.rank((2048, 2048, 2048, 1))
    from lib import mlrec
    from lib.feasibility import soft_rules

    header = mlrec.read_header(data)
    assert r.rules == soft_rules(header["serving"], header["version"])
    feas = pl.Feasibility(cat, PTYPE, LDS, rules=r.rules)
    assert set(order) == set(feas.feasible((2048, 2048, 2048, 1), cat.kernels))
    fallback = pl.ModelRanker(data, cat, PTYPE, hw, tw=tw, fallback=FixedRanker([7]))
    assert fallback.rank((8, 8, 8, 1)) in ([7], fallback.rank((8, 8, 8, 1)))


# ── bench line knobs ─────────────────────────────────────────────────────────

LINE = (
    "- {function: matmul, transA: T, transB: N, M: 4, N: 8, K: 16, iters: 10, "
    "cold_iters: 10, batch_count: 2, requested_solution_num: -1}\n"
)


def test_set_and_drop_knobs_with_list_values():
    a = set_line_knob(LINE, "solution_index_list", format_solution_list([3, 1]))
    assert line_solution_list(a) == [3, 1]
    a = set_line_knob(a, "skip_screen_prepass", "true")
    a = set_line_knob(a, "solution_index_list", "[9, 8, 7]")
    assert line_solution_list(a) == [9, 8, 7]
    assert line_has_prescreen(a)
    assert shape_of_line(a) == (4, 8, 16, 2)
    assert "cold_iters: 10" in set_line_knob(a, "iters", 5)
    assert "iters: 5," in set_line_knob(a, "iters", 5)
    b = drop_line_knob(a, "solution_index_list")
    assert line_solution_list(b) is None and b.endswith("}\n")
    assert not line_has_prescreen(LINE)
    assert drop_line_knob(LINE, "algo_method") == LINE
