# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import csv
from dataclasses import replace

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


def hinted(sid, tdm, tha=0, thb=0, mt=(128, 128, 32)):
    """The `kernel_dat_info` of a solution with TDMInst `tdm` and temporal
    hints `tha` / `thb`, spelled in its name."""
    import bench_fakes as bf
    from lib import dat

    sol = bf.solution(
        sid,
        sid,
        mt,
        mi=(16, 16, 32, 1),
        hasTemporalHint=True,
        temporalHintA=tha,
        temporalHintB=thb,
    )
    sol["name"] = (
        f"Cijk_Alik_Bljk_BBS_BH_MT{mt[0]}x{mt[1]}x{mt[2]}_MI16x16x1_TDMI{tdm}"
        f"_THA{tha}_THB{thb}_WGM8"
    )
    return dat.kernel_dat_info(sol)


def test_an_alias_group_is_planned_as_its_first_member_in_pool_order():
    """A TDM kernel's THA/THB twins share its family and key; only the
    first in pool order (the pool's before the bank's) is plannable. The
    same hints on kernels without TDM-loaded operands keep their families
    and are planned each (a mixed pool)."""
    from lib.feasibility import soft_rules

    twins = [hinted(20, 3), hinted(21, 3, tha=1, thb=1), hinted(22, 3, thb=2)]
    loose = [hinted(30, 0), hinted(31, 0, tha=1)]
    cat = pl.Catalog.build(pool_kernels(4) + twins + loose, [hinted(40, 3, tha=3)])
    assert cat.aliases == {21: 20, 22: 20, 40: 20}
    assert cat.pool_ids == [100, 101, 102, 103, 999, 20, 30, 31]
    assert cat.bank_ids == []
    assert 21 not in cat.kernels and 40 not in cat.kernels
    assert cat.families[pl.family_of(cat.kernels[20])] == [20, 30]
    assert cat.families[pl.family_of(cat.kernels[31])] == [31]
    feas = pl.Feasibility(cat, PTYPE, LDS, rules=soft_rules({}, 3))
    plan = pl.plan_shape((512, 512, 512, 1), [21, 40, 20, 31], feas, PARAMS)
    assert plan.kernels[:3] == [20, 31, 30] and not {21, 22, 40} & set(plan.kernels)
    bank_first = pl.Catalog.build(pool_kernels(4), [hinted(40, 3, tha=3)] + twins)
    assert bank_first.aliases == {20: 40, 21: 40, 22: 40}
    assert bank_first.bank_ids == [40]


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
    assert "probe" not in sp.to_json()


def test_plan_shape_is_deterministic_and_seed_dependent():
    _, feas = make()
    shape = (300, 200, 100, 1)
    a = pl.plan_shape(shape, [100, 101], feas, PARAMS)
    b = pl.plan_shape(shape, [100, 101], feas, PARAMS)
    assert a.kernels == b.kernels
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
        assert plans[1].n_feasible == len(feas.catalog.pool_ids) - 1


def test_shuffled_is_a_seeded_permutation_per_key():
    ids = list(range(100, 140))
    a = pl.shuffled(ids, 42, 64, 128, 256, 1)
    assert sorted(a) == ids and a != ids
    assert a == pl.shuffled(list(ids), 42, 64, 128, 256, 1)
    assert a != pl.shuffled(ids, 42, 64, 128, 256, 2)
    assert a != pl.shuffled(ids, 43, 64, 128, 256, 1)
    assert pl.shuffled([], 42, 1) == [] and pl.shuffled([7], 42, 1) == [7]


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


def test_neighbors_ranker_ranks_without_the_excluded_seeds():
    a, b, c = (1024, 1024, 1024, 1), (1100, 1024, 1024, 1), (4096, 1024, 1024, 1)
    seeds = {a: {1: 10.0, 2: 20.0}, b: {1: 20.0, 2: 10.0}, c: {3: 5.0}}
    r = pl.NeighborsRanker(seeds, k=1)
    assert r.rank(a) == [1, 2]
    assert r.neighbours(a, exclude=(a,)) == [b]
    assert r.rank(a, exclude=(a,)) == [2, 1]
    assert r.rank(a, exclude=(a, b)) == [3]
    assert r.rank((8, 8, 8, 1), exclude=(a, b, c)) == []


# ── seed recall ──────────────────────────────────────────────────────────────

ONE = pl.PlanParams(top_k=1, families_top_f=0, explore_random_k=0, min_coverage=0)
SCORES = ("n_seeds", "winner_recall", "near_recall", "ceiling", "n_unmeasured")


def test_seed_recall_scores_each_seed_from_the_other_seeds_of_its_cell():
    """With the nearest seed as the only neighbour and one kernel per list, a
    and b list their winner, d a kernel 3% slower than its winner and c one
    twice as slow. Kernel 999 is the fastest of a but fits no LDS: the plan
    cannot list it, so it is not a's winner."""
    _, feas = make()
    a, b = (1024, 1024, 1024, 1), (2048, 1024, 1024, 1)
    c, d = (8192, 1024, 1024, 1), (1024, 1024, 8192, 1)
    cell = pl.shape_cell(a)
    assert {pl.shape_cell(s) for s in (b, c, d)} == {cell}
    seeds = {
        a: {100: 10.0, 101: 10.4, 102: 30.0, 999: 1.0},
        b: {100: 10.0, 101: 12.0, 102: 30.0},
        c: {100: 20.0, 101: 10.0, 102: 30.0},
        d: {100: 10.3, 101: 10.0, 102: 30.0},
    }
    ranker = pl.NeighborsRanker(seeds, k=1)
    assert [ranker.neighbours(s, exclude=(s,)) for s in (a, b, c, d)] == [
        [b],
        [a],
        [b],
        [a],
    ]
    rec = pl.seed_recall(ranker, feas, ONE, top_ks=(2, 64))
    assert (rec["ranker"], rec["neighbors_k"], rec["top_k"]) == ("neighbors", 1, 1)
    assert (rec["n_seeds"], rec["n_cells"], rec["n_alone"]) == (4, 1, 0)
    assert (rec["winner_recall"], rec["near_recall"]) == (0.5, 0.75)
    assert rec["ceiling"] == pytest.approx((0.5 * 10.0 / 10.3) ** 0.25)
    assert rec["cells"] == {cell: {k: rec[k] for k in SCORES}}
    # 64 is beyond the 40 kernels a seed may list, which are scored in its
    # place; at 2 every list holds both kernels the neighbours rank first.
    assert [e["top_k"] for e in rec["by_top_k"]] == [1, 2, 40]
    assert rec["by_top_k"][1]["ceiling"] == 1.0 and rec["top_k_reaching"] == 2
    capped = pl.seed_recall(ranker, feas, replace(ONE, max_list=8), top_ks=(4, 16))
    assert [e["top_k"] for e in capped["by_top_k"]] == [1, 4, 8]
    assert pl.seed_recall(ranker, feas, ONE, min_ceiling=1.0)["top_k_reaching"] == 16
    wide = pl.seed_recall(ranker, feas, replace(ONE, top_k=2), top_ks=())
    assert [e["top_k"] for e in wide["by_top_k"]] == [2, 40] and wide["ceiling"] == 1


def test_seed_recall_counts_the_seeds_it_cannot_score():
    """A seed alone in its cell is left out; so is one whose bench measured
    no kernel the plan may list (f: only 999), whose ranking then leaves its
    cell's other seed g a list without a measured kernel: a miss that the
    ceiling, a geometric mean, leaves out. Only the cells asked for count."""
    _, feas = make()
    a, b = (1024, 1024, 1024, 1), (2048, 1024, 1024, 1)
    alone, f, g = (8, 8, 8, 1), (64, 64, 64, 1), (96, 64, 64, 1)
    assert pl.shape_cell(f) == pl.shape_cell(g) != pl.shape_cell(alone)
    assert feas.feasible(g, [100, 101]) == [100, 101]
    seeds = {
        a: {100: 1.0},
        b: {100: 1.0},
        alone: {100: 1.0},
        f: {999: 1.0},
        g: {100: 1.0, 101: 2.0},
    }
    ranker = pl.NeighborsRanker(seeds, k=4)
    rec = pl.seed_recall(ranker, feas, ONE)
    assert (rec["n_alone"], rec["n_unscored"]) == (1, 1)
    assert (rec["n_seeds"], rec["n_cells"], rec["n_unmeasured"]) == (3, 2, 1)
    assert rec["winner_recall"] == rec["near_recall"] == pytest.approx(2 / 3)
    assert rec["ceiling"] == 1.0
    assert rec["cells"][pl.shape_cell(g)]["ceiling"] is None
    only = pl.seed_recall(ranker, feas, ONE, cells={pl.shape_cell(g)})
    assert (only["n_seeds"], only["n_alone"], only["n_unscored"]) == (1, 0, 1)
    assert only["ceiling"] is None and only["top_k_reaching"] is None
    none = pl.seed_recall(ranker, feas, ONE, cells=set())
    assert none["n_seeds"] == 0 and none["winner_recall"] is None
    assert [e["top_k"] for e in none["by_top_k"]] == [1]


def test_stage02_warns_when_the_lists_the_seeds_rank_miss_the_ceiling(capsys):
    import stage02_plan_benchmarks as s02

    _, feas = make()
    a, b, c = (1024, 1024, 1024, 1), (2048, 1024, 1024, 1), (8192, 1024, 1024, 1)
    seeds = {
        a: {100: 10.0, 101: 12.0},
        b: {100: 10.0, 101: 12.0},
        c: {100: 20.0, 101: 10.0},
    }
    shapes = [*seeds, (4096, 1024, 1024, 1)]
    ranker = pl.NeighborsRanker(seeds, k=1)
    rec = s02.check_seed_recall(ranker, feas, ONE, shapes, list(seeds), 0.9, False)
    out = capsys.readouterr()
    assert rec["plan_ranker"] == "neighbors" and rec["top_k_reaching"] == 16
    assert (
        "3 seed(s) of 1 cell(s)" in out.out and "ceiling by top_k: 1 0.794" in out.out
    )
    assert "below planner.recall_min_ceiling 0.9" in out.err
    assert "top_k 16 and above reach it (planner.top_k_round0)" in out.err
    assert f"weakest cells: {pl.shape_cell(a)} 0.794" in out.err
    s02.check_seed_recall(ranker, feas, ONE, shapes, list(seeds), 0.7, True)
    assert capsys.readouterr() == ("", "")
    alone = s02.check_seed_recall(
        ranker, feas, ONE, list(seeds), list(seeds), 0.9, True
    )
    assert alone["n_seeds"] == 0 and capsys.readouterr().err == ""
    # A stage05 seed model trained on every seed: its fallback is scored.
    model = pl.ModelRanker.__new__(pl.ModelRanker)
    model.fallback = ranker
    rec = s02.check_seed_recall(model, feas, ONE, shapes, list(seeds), 0.5, False)
    assert (rec["ranker"], rec["plan_ranker"]) == ("neighbors", "stage05")
    assert "by the neighbors ranker, the stage05 model's fallback" in (
        capsys.readouterr().out
    )
    assert (
        s02.check_seed_recall(FixedRanker([100]), feas, ONE, shapes, [], 0.9, 1) is None
    )


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
    from lib.feasibility import soft_rules

    tw = pytest.importorskip("tilewright")
    pytest.importorskip("torch")
    import stage02_plan_benchmarks as s02
    from lib.hardware import DeviceHardware

    cat, _ = make()
    label = "Large|Large|LargeK|Bnone"
    sigs = sorted({pl.family_of(k) for k in cat.kernels.values()})
    models = make_bundle(tmp_path, [label], signatures={label: [list(s) for s in sigs]})
    data = s02.model_bytes(models, "gfx950", "fp32")
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
    with pytest.raises(ValueError, match="only v3 models"):
        pl.ModelRanker(data, cat, PTYPE, hw, tw=tw, rules=soft_rules({}, 3))


def test_v3_model_ranker_ranks_under_the_given_soft_rules():
    tw = pytest.importorskip("tilewright")
    from lib.feasibility import soft_rules
    from lib.hardware import DeviceHardware
    from test_v3 import LARGE, v3_bundle, write

    nt_b = [kern(200 + i, mt=(64, 64, 64), chb=4) for i in range(3)]
    cat, _ = make(pool=pool_kernels() + nt_b)
    data = write(v3_bundle(labels=[LARGE], serving={"non_temporal_rule": True}))
    hw = DeviceHardware(n_cu=256, lds_bytes=LDS, l2_bytes=4 << 20)
    shape = (2048, 2048, 2048, 1)
    own = pl.ModelRanker(data, cat, PTYPE, hw, tw=tw)
    assert own.rules["non_temporal_rule"]
    off = soft_rules({}, 3)
    other = pl.ModelRanker(data, cat, PTYPE, hw, tw=tw, rules=off)
    assert other.rules == off
    for r in (own, other):
        feas = pl.Feasibility(cat, PTYPE, LDS, rules=r.rules)
        assert set(r.rank(shape)) == set(feas.feasible(shape, cat.kernels))
    assert set(other.rank(shape)) - set(own.rank(shape)) == {200, 201, 202}


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


def test_planning_follows_the_family_wise_non_temporal_rule():
    tw = pytest.importorskip("tilewright")
    from lib.feasibility import soft_rules
    from lib.hardware import DeviceHardware
    from test_v3 import LARGE, v3_bundle, write

    hinted_only = [kern(300, mt=(8, 128, 32), mi=(4, 4, 4), cha=3, chb=1),
                   kern(301, mt=(8, 128, 32), mi=(4, 4, 4), cha=1, chb=1)]  # fmt: skip
    nt = [kern(310, cha=4), kern(311, chb=4)]
    cat, _ = make(pool=pool_kernels() + nt, bank=hinted_only)
    shape = (2, 1792, 4, 64)
    today = soft_rules({"non_temporal_rule": True}, 3)
    family = soft_rules(
        {"non_temporal_rule": True, "non_temporal_family_rule": True}, 3
    )
    plain = pl.Feasibility(cat, PTYPE, LDS, rules=today).feasible(shape, cat.kernels)
    kept = pl.Feasibility(cat, PTYPE, LDS, rules=family).feasible(shape, cat.kernels)
    assert set(kept) - set(plain) == {301}
    data = write(v3_bundle(labels=[LARGE], serving={"non_temporal_rule": True}))
    hw = DeviceHardware(n_cu=256, lds_bytes=LDS, l2_bytes=4 << 20)
    own = pl.ModelRanker(data, cat, PTYPE, hw, tw=tw)
    refined = pl.ModelRanker(data, cat, PTYPE, hw, tw=tw, rules=family)
    assert refined.rules == family and own.rules == today
    assert set(refined.rank(shape)) == set(kept)
    assert set(own.rank(shape)) == set(plain)
    flagged = write(v3_bundle(labels=[LARGE], serving={
        "non_temporal_rule": True, "non_temporal_family_rule": True}))  # fmt: skip
    assert pl.ModelRanker(flagged, cat, PTYPE, hw, tw=tw).rules == family
    back = pl.ModelRanker(flagged, cat, PTYPE, hw, tw=tw, rules=today)
    assert back.rules == today and set(back.rank(shape)) == set(plain)


def _family_catalog():
    """A pool with non-temporal variants of both operands and a bank family
    that exists only with cache hints, which the non-temporal rule rejects
    whole on the shape below (K off the DepthU alignment)."""
    hinted_only = [kern(300, mt=(8, 128, 32), mi=(4, 4, 4), cha=3, chb=1),
                   kern(301, mt=(8, 128, 32), mi=(4, 4, 4), cha=1, chb=1)]  # fmt: skip
    nt = [kern(310, cha=4), kern(311, chb=4)]
    return pl.Catalog.build(pool_kernels() + nt, hinted_only)


FAMILY_SHAPE = (2, 1792, 4, 64)


def test_family_rule_only_lists_the_members_the_family_rule_keeps():
    from lib.feasibility import soft_rules

    cat = _family_catalog()
    family = soft_rules(
        {"non_temporal_rule": True, "non_temporal_family_rule": True}, 3
    )
    feas = pl.Feasibility(cat, PTYPE, LDS, rules=family)
    assert feas.family_rule_only(FAMILY_SHAPE) == [301]
    # Outside the rule's region (not batched) it adds nothing.
    assert feas.family_rule_only(FAMILY_SHAPE[:3] + (1,)) == []
    today = soft_rules({"non_temporal_rule": True}, 3)
    assert (
        pl.Feasibility(cat, PTYPE, LDS, rules=today).family_rule_only(FAMILY_SHAPE)
        == []
    )


def test_planned_lists_take_the_family_rule_members_past_max_list(monkeypatch):
    from lib.feasibility import soft_rules

    cat = _family_catalog()
    family = soft_rules(
        {"non_temporal_rule": True, "non_temporal_family_rule": True}, 3
    )
    feas = pl.Feasibility(cat, PTYPE, LDS, rules=family)
    params = pl.PlanParams(
        top_k=3,
        families_top_f=0,
        explore_random_k=0,
        min_coverage=0,
        max_list=3,
        seed=7,
    )
    flat = (64, 1792, 4, 64)
    seed = (2, 2048, 4, 64)
    shapes = [FAMILY_SHAPE, flat, seed]
    ranker = FixedRanker(cat.pool_ids)
    plain, _ = pl.build_plan(shapes, feas, params, ranker=ranker, seeds=[seed])
    plans, _ = pl.build_plan(
        shapes, feas, params, ranker=ranker, seeds=[seed], family_coverage=feas
    )
    by_shape = {sp.shape: sp for sp in plans}
    before = {sp.shape: sp for sp in plain}
    assert 301 not in before[FAMILY_SHAPE].kernels
    assert by_shape[FAMILY_SHAPE].kernels == before[FAMILY_SHAPE].kernels + [301]
    assert by_shape[FAMILY_SHAPE].sources["family_rule"] == 1
    assert len(by_shape[FAMILY_SHAPE].kernels) == params.max_list + 1
    # Outside the region, and for a seed benched exhaustively, nothing changes.
    assert by_shape[flat].kernels == before[flat].kernels
    assert "family_rule" not in by_shape[flat].sources
    assert by_shape[seed].kernels is None
    # The bench's own limit still holds.
    monkeypatch.setattr(pl, "MAX_BENCH_LIST", params.max_list)
    capped, _ = pl.build_plan(
        [FAMILY_SHAPE], feas, params, ranker=ranker, family_coverage=feas
    )
    assert capped[0].kernels == before[FAMILY_SHAPE].kernels
    assert capped[0].n_family_over_limit == 1
    assert capped[0].n_truncated == before[FAMILY_SHAPE].n_truncated
    assert capped[0].to_json()["n_family_over_limit"] == 1
