# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from bench_fakes import LIBRARY_STEM, solution, write_logic
from lib import dat
from lib import evaluate as ev
from lib import features as fs
from lib import mlrec
from lib.hardware import DeviceHardware, arch_constants

from test_mlrec import random_bundle

TESTS_DIR = Path(__file__).resolve().parent

HW = DeviceHardware(n_cu=100, lds_bytes=65536, l2_bytes=4 << 20)
CELLS = ["Large|Large|LargeK|Bnone", "Mid|Large|MidK|Bnone", "Small|Small|MidK|Bnone"]


def kernel(sid, mt_m, mt_n, mt_k, mi=(16, 16, 32), chb=0):
    return dict(
        sol_idx_global=sid,
        mt_m=mt_m,
        mt_n=mt_n,
        mt_k=mt_k,
        mi_m=mi[0],
        mi_n=mi[1],
        mi_k=mi[2],
        occupancy=1,
        cache_hints_a=0,
        cache_hints_b=chb,
        grvw_a=8,
        grvw_b=8,
        gwvw_d=4,
    )


POOL = [
    kernel(10 + i, mt_m, mt_n, mt_k)
    for i, (mt_m, mt_n, mt_k) in enumerate(
        [
            (m, n, k)
            for m in (32, 64, 128, 256)
            for n in (64, 128, 256)
            for k in (32, 64)
        ]
    )
]
POOL.append(kernel(90, 64, 256, 64, chb=4))
# Every kernel of POOL followed by a twin with the same Config fields: another
# solution with its own index and attributes.
TWINS = POOL + [
    dict(k, sol_idx_global=200 + i, attributes={"workgroup_mapping": 4})
    for i, k in enumerate(POOL)
]


def gemm(m, n, k, cands, b=1):
    tested = [c["us"] for c in cands if not c.get("is_skip")]
    return dict(
        m=m,
        n=n,
        k=k,
        batch_count=b,
        transA="T",
        transB="N",
        a_type="bf16_r",
        b_type="bf16_r",
        c_type="bf16_r",
        d_type="bf16_r",
        compute_type="f32_r",
        candidates=cands,
        winner_us=min(tested) if tested else min(c["us"] for c in cands),
    )


def random_gemms(rng, n_gemms, sizes):
    out = []
    for _ in range(n_gemms):
        m, n, k = (rng.choice(s) for s in sizes)
        cands = []
        for rank, kk in enumerate(rng.sample(POOL, rng.randint(1, len(POOL)))):
            c = dict(kk, us=rng.uniform(1.0, 50.0), rank=rank)
            c["is_skip"] = rank > 0 and rng.random() < 0.2
            c["is_origami_pick"] = rank == 0
            cands.append(c)
        out.append(gemm(m, n, k, cands))
    return out


def model_bytes(labels, dtype="fp32", signatures=None, seed=0):
    sigs = signatures or {
        lbl: [
            [k["mt_m"], k["mt_n"], k["mt_k"], 16, 16, 32, 0, 0]
            for k in POOL[2 + i :: 5]
        ]
        for i, lbl in enumerate(labels)
    }
    bundle = random_bundle(labels, seed=seed, signatures=sigs)
    return bundle, mlrec.write_model(
        bundle, None, "gfx950", arch_constants("gfx950"), dtype
    )


def ranked_sids(tw, model, g, min_scored=1):
    cs = tw.CandidateSet(model, [ev.make_config(tw, k, i) for i, k in enumerate(POOL)])
    prob = ev.make_problem(tw, fs.problem_kwargs_from_row(g))
    out = []
    for r in cs.rank(prob, ev.make_hardware(tw, HW), min_scored):
        if not r.scored:
            break
        out.append(POOL[r.config_index]["sol_idx_global"])
    return out


def expected_pick(tw, model, g, min_scored=1, measurement="timed"):
    """(solution, served_by, cost, miss) the module docstring describes."""
    timed = {c["sol_idx_global"]: c["us"] for c in g["candidates"] if not c["is_skip"]}
    screened = {
        c["sol_idx_global"]: c["us"]
        for c in g["candidates"]
        if c["is_skip"] and c["sol_idx_global"] not in timed
    }
    worst = max(c["us"] for c in g["candidates"])
    listed = g.get("listed")
    for sid in ranked_sids(tw, model, g, min_scored):
        if sid in timed:
            return sid, "model", timed[sid], None
        if sid in screened:
            if measurement == "any":
                return sid, "model", screened[sid], None
            return sid, "model", worst, "screened"
        if measurement == "timed" and listed is not None and sid not in listed:
            return sid, "model", worst, "not_benched"
    oc = [c for c in g["candidates"] if c["is_origami_pick"]]
    if oc:
        return oc[0]["sol_idx_global"], "origami_fallback", oc[0]["us"], None
    return None, None, None, None


@pytest.mark.parametrize("measurement", ["timed", "any"])
def test_picks_follow_candidate_set_rank(measurement):
    tw = pytest.importorskip("tilewright")
    rng = random.Random(7)
    gemms = random_gemms(rng, 80, [(600, 900, 2000), (700, 4000), (600, 1024, 3000)])
    gemms += random_gemms(rng, 40, [(200, 300), (700, 2000), (64, 256)])
    _bundle, data = model_bytes(CELLS[:2])
    out = ev.evaluate_model(
        data,
        gemms,
        hardware=HW,
        pools={ev.pool_key(gemms[0]): POOL},
        pick_measurement=measurement,
    )
    model = tw.load_model_from_memory(data)
    served, misses = set(), 0
    for g, r in zip(gemms, out.results):
        sid, by, us, miss = expected_pick(tw, model, g, measurement=measurement)
        assert (r.pick_sol_idx, r.served_by, r.miss) == (sid, by, miss)
        if sid is not None:
            assert r.pick_us == us and r.sel_eff == pytest.approx(g["winner_us"] / us)
        served.add(by)
        misses += miss is not None
    assert "model" in served
    assert (misses > 0) == (measurement == "timed")
    summary = out.summary()
    assert summary["n_miss"] == summary["n_miss_by_reason"]["screened"] == misses
    assert summary["pick_measurement"] == measurement
    assert out.n_routing_mismatch == 0 and out.weight_dtype == "fp32"


def test_a_pick_off_the_plan_is_a_miss():
    tw = pytest.importorskip("tilewright")
    _bundle, data = model_bytes(CELLS[:1])
    model = tw.load_model_from_memory(data)
    probe = gemm(2048, 2048, 2048, [dict(POOL[0], us=1.0, rank=0)])
    ranked = ranked_sids(tw, model, probe)
    assert len(ranked) >= 3
    first, second, third = ranked[:3]
    cands = [
        dict(k, us=10.0 + i, rank=i, is_skip=False, is_origami_pick=(i == 0))
        for i, k in enumerate(POOL)
        if k["sol_idx_global"] not in (first, second)
    ]
    planned = gemm(2048, 2048, 2048, cands)
    planned["listed"] = frozenset(c["sol_idx_global"] for c in cands) | {second}
    exhaustive = gemm(2048, 2048, 2048, [dict(c) for c in cands])
    pools = {ev.pool_key(planned): POOL}
    worst = max(c["us"] for c in cands)
    third_us = next(c["us"] for c in cands if c["sol_idx_global"] == third)

    timed = ev.evaluate_model(
        data, [planned, exhaustive], hardware=HW, pools=pools, pick_measurement="timed"
    )
    a, b = timed.results
    assert (a.pick_sol_idx, a.miss, a.pick_us, a.pick_rank) == (
        first,
        "not_benched",
        worst,
        0,
    )
    assert (b.pick_sol_idx, b.miss, b.pick_us, b.pick_rank) == (
        third,
        None,
        third_us,
        2,
    )
    assert timed.summary()["n_miss_by_reason"] == {"screened": 0, "not_benched": 1}

    walked = ev.evaluate_model(
        data, [planned], hardware=HW, pools=pools, pick_measurement="any"
    )
    (c,) = walked.results
    assert (c.pick_sol_idx, c.miss, c.pick_rank) == (third, None, 2)
    with pytest.raises(ValueError):
        ev.evaluate_model(data, [planned], hardware=HW, pick_measurement="first")


def test_measurements_prefer_timed_rows():
    cands = [
        dict(POOL[0], us=5.0, is_skip=False),
        dict(POOL[0], us=3.0, is_skip=True),
        dict(POOL[1], us=7.0, is_skip=True),
        dict(POOL[1], us=6.0, is_skip=True),
        dict(POOL[2], us=0.0, is_skip=False),
        dict(POOL[3], us=float("nan"), is_skip=False),
        dict(POOL[4], us=2.0, is_skip=False, sol_idx_global=-1),
    ]
    timed, screened = ev.measurements(gemm(1, 1, 1, cands))
    assert timed == {10: 5.0} and screened == {11: 6.5}
    assert ev.measurements(gemm(1, 1, 1, cands), "min") == ({10: 5.0}, {11: 6.0})
    timed, screened = ev.measurements(gemm(1, 1, 1, cands), "min")
    assert ev.pick_cost(10, timed, screened, None, "timed") == (True, 5.0, None)
    assert ev.pick_cost(11, timed, screened, None, "timed") == (True, 6.0, "screened")
    assert ev.pick_cost(11, timed, screened, None) == (True, 6.0, None)
    assert ev.pick_cost(12, timed, screened, None, "timed") == (False, None, None)
    assert ev.pick_cost(12, timed, screened, {10}, "timed") == (
        True,
        6.0,
        "not_benched",
    )
    assert ev.pick_cost(12, timed, screened, {12}, "timed") == (False, None, None)
    assert ev.pick_cost(12, timed, screened, {10}) == (False, None, None)
    with pytest.raises(ValueError):
        ev.pick_cost(10, timed, screened, None, "fastest")


def test_warmup_counts_screened_picks_and_misses_off_list_ones():
    timed, screened = {10: 5.0, 13: 9.0}, {11: 6.0}
    assert ev.pick_cost(11, timed, screened, None, "warmup") == (True, 6.0, None)
    assert ev.pick_cost(11, timed, screened, None, "timed") == (True, 9.0, "screened")
    assert ev.pick_cost(12, timed, screened, {10, 11, 13}, "warmup") == (
        True,
        9.0,
        "not_benched",
    )
    assert ev.pick_cost(12, timed, screened, {10, 11, 13}, "any") == (False, None, None)
    assert ev.pick_cost(12, timed, screened, None, "warmup") == (False, None, None)


def test_off_list_sel_eff_charges_an_unmeasured_off_list_pick():
    timed, screened = {10: 4.0, 13: 9.0}, {11: 6.0}
    listed = {10, 11, 13}

    def cost(sid, rule="warmup", rb=None):
        return ev.pick_cost(sid, timed, screened, listed, rule, rb, 0.5)

    assert cost(12) == (True, 8.0, "not_benched")
    assert cost(12, "timed") == (True, 8.0, "not_benched")
    assert cost(12, "any") == (False, None, None)
    assert cost(12, "rebench", {12: 0.8}) == (True, 5.0, "not_benched")
    assert cost(11, "timed") == (True, 9.0, "screened")
    assert ev.pick_cost(12, {}, screened, listed, "warmup", None, 0.5) == (
        True,
        6.0,
        "not_benched",
    )


def test_rebench_charges_a_miss_by_its_measured_ratio():
    timed = {10: 4.0, 13: 5.0}
    screened = {11: 9.0}
    ratios = {11: 0.8, 12: 0.5, 10: 0.1}
    cost = lambda sid, listed=None, rule="rebench", rb=ratios: ev.pick_cost(  # noqa
        sid, timed, screened, listed, rule, rb
    )
    assert cost(11) == (True, 5.0, "screened")
    assert cost(12, {10, 13}) == (True, 8.0, "not_benched")
    assert cost(10) == (True, 4.0, None)
    assert cost(11, rb={11: 1.3}) == (True, 4.0, "screened")
    assert cost(11, rb={}) == (True, 9.0, "screened")
    assert ev.pick_cost(11, {10: 4.0, 13: 20.0}, screened, None, "rebench", {}) == (
        True,
        9.0,
        "screened",
    )
    assert cost(12) == (False, None, None)
    assert cost(11, rule="timed") == (True, 9.0, "screened")
    assert cost(11, rule="any") == (True, 9.0, None)
    assert ev.REBENCH_PICK_MEASUREMENT not in ev.PICK_MEASUREMENTS


def test_rebench_rule_reads_the_ratios_attached_to_the_gemms():
    tw = pytest.importorskip("tilewright")
    from lib import rebench as rb

    _bundle, data = model_bytes(CELLS[:1])
    model = tw.load_model_from_memory(data)
    first = ranked_sids(tw, model, gemm(2048, 2048, 2048, [dict(POOL[0], us=1.0)]))[0]
    cands = [
        dict(k, us=10.0 + i, rank=i, is_skip=k["sol_idx_global"] == first)
        for i, k in enumerate(POOL)
    ]
    for c in cands:
        c["is_origami_pick"] = c["rank"] == 0
    measured, plain = gemm(2048, 2048, 2048, cands), gemm(2048, 2048, 2048, cands)
    shape = (2048, 2048, 2048, 1)
    ratio = rb.PickRatio(shape, first, -1, 8.0, 6.0, 3)
    assert rb.attach_rebench([measured], {shape: ratio}) == 1
    assert measured["rebench"] == {first: 0.75} and "rebench" not in plain
    pools = {ev.pool_key(measured): POOL}
    out = ev.evaluate_model(
        data, [measured, plain], hardware=HW, pools=pools, pick_measurement="rebench"
    )
    a, b = out.results
    assert (a.pick_sol_idx, a.miss) == (b.pick_sol_idx, b.miss) == (first, "screened")
    assert a.sel_eff == pytest.approx(0.75)
    assert b.pick_us == next(c["us"] for c in cands if c["sol_idx_global"] == first)
    assert out.pick_measurement == "rebench"


def test_attach_plan_lists_the_planned_kernels():
    cands = [dict(POOL[0], us=1.0)]
    gemms = [gemm(64, 64, 64, cands), gemm(128, 64, 64, cands), gemm(9, 9, 9, cands)]
    plan = {
        "shapes": [
            {"m": 64, "n": 64, "k": 64, "batch": 1, "kernels": [3, 1]},
            {"m": 128, "n": 64, "k": 64, "batch": 1, "kernels": None},
        ]
    }
    assert ev.attach_plan(gemms, plan) == 1
    assert gemms[0]["listed"] == {1, 3}
    assert "listed" not in gemms[1] and "listed" not in gemms[2]


def test_attach_plans_unions_the_lists_and_takes_context_replans():
    cands = [dict(POOL[0], us=1.0)]
    gemms = [
        gemm(64, 64, 64, cands),
        gemm(64, 64, 64, cands),
        gemm(128, 64, 64, cands),
        gemm(9, 9, 9, cands),
    ]
    gemms[1].update(cu_budget=32, schedule="default")
    gemms[3]["listed"] = frozenset({1})

    def plan(*shapes):
        return {
            "shapes": [
                {"m": m, "n": 64, "k": 64, "batch": 1, "kernels": ks}
                for m, ks in shapes
            ]
        }

    rounds = [plan((64, [1, 3]), (128, [2])), plan((64, [5]), (128, None))]
    replan = {(32, "default"): [plan((64, [7]))]}
    assert ev.attach_plans(gemms, rounds, replan) == 2
    assert gemms[0]["listed"] == {1, 3, 5}
    assert gemms[1]["listed"] == {7}
    assert "listed" not in gemms[2] and "listed" not in gemms[3]


def test_load_timings_merges_files_and_keeps_exclusive_rows(tmp_path):
    header = "m,n,k,batch_count,transA,transB,sol_idx_global,us,is_skip,cu_budget\n"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "chunk_0000.csv").write_text(
        header
        + "64,64,64,1,T,N,1,5.0,False,\n"
        + "64,64,64,1,T,N,2,9.0,True,\n"
        + "64,64,64,1,T,N,3,1.0,False,32\n"
        + "8,8,8,1,T,N,1,1.0,False,\n"
        + "x,8,8,1,T,N,1,1.0,False,\n"
    )
    (tmp_path / "b" / "chunk_0000.csv").write_text(
        header + "64,64,64,1,T,N,1,4.0,False,0\n"
    )
    key = (64, 64, 64, 1, "T", "N")
    got = ev.load_timings([tmp_path / "a", tmp_path / "b"], wanted=[key])
    assert list(got) == [key] and got[key]["winner_us"] == 4.5
    assert ev.measurements(got[key]) == ({1: 4.5}, {2: 9.0})
    got = ev.load_timings([tmp_path / "a", tmp_path / "b"], [key], aggregation="min")
    assert got[key]["winner_us"] == 4.0
    assert set(ev.load_timings([tmp_path / "a"])) == {key, (8, 8, 8, 1, "T", "N")}


def test_bootstrap_ci():
    assert (
        ev.bootstrap_ci([1.0]) is None
        and ev.bootstrap_ci([1.0, 2.0], resamples=0) is None
    )
    values = [float(v) for v in range(50)]
    lo, hi = ev.bootstrap_ci(values, resamples=200, seed=3)
    assert lo <= sum(values) / len(values) <= hi
    assert ev.bootstrap_ci(values, resamples=200, seed=3) == [lo, hi]
    med_lo, med_hi = ev.bootstrap_ci(values, statistic=np.median, resamples=200)
    assert med_lo <= 24.5 <= med_hi


def test_fallback_and_not_evaluable():
    pytest.importorskip("tilewright")
    _bundle, data = model_bytes(CELLS[:1])
    cands = [
        dict(POOL[0], us=5.0, rank=0, is_skip=False, is_origami_pick=True),
        dict(POOL[3], us=4.0, rank=1, is_skip=False, is_origami_pick=False),
    ]
    no_model = gemm(64, 64, 256, cands)
    no_origami = gemm(64, 64, 256, [dict(c, is_origami_pick=False) for c in cands])
    out = ev.evaluate_model(data, [no_model, no_origami], hardware=HW)
    a, b = out.results
    assert a.model_cell is None and a.served_by == "origami_fallback"
    assert a.pick_sol_idx == POOL[0]["sol_idx_global"] and a.sel_eff == 4.0 / 5.0
    assert b.served_by is None and b.sel_eff is None
    s = out.summary()
    assert s["n_gemms"] == 2 and s["n_evaluated"] == 1 and s["n_not_evaluable"] == 1
    assert s["n_origami_fallback"] == 1 and s["paired"]["n"] == 1
    assert s["paired"]["same_kernel"] == 1 and s["paired"]["ties"] == 1


def test_origami_candidate_without_the_column():
    cands = [
        dict(POOL[0], us=9.0, rank=0, is_skip=True, is_origami_pick=None),
        dict(POOL[1], us=7.0, rank=2, is_skip=False, is_origami_pick=None),
        dict(POOL[2], us=8.0, rank=1, is_skip=False, is_origami_pick=None),
    ]
    assert ev.origami_candidate(gemm(1, 1, 1, cands))["sol_idx_global"] == 12
    flagged = [dict(c, is_origami_pick=(i == 1)) for i, c in enumerate(cands)]
    assert ev.origami_candidate(gemm(1, 1, 1, flagged))["sol_idx_global"] == 11
    assert (
        ev.origami_candidate(
            gemm(1, 1, 1, [dict(c, is_origami_pick=False) for c in cands])
        )
        is None
    )
    s = ev.origami_summary([gemm(1, 1, 1, cands)])
    assert s["n_eval"] == 1 and s["sel_eff"] == pytest.approx(7.0 / 8.0)


def test_weight_dtype_is_what_ranks():
    tw = pytest.importorskip("tilewright")
    rng = random.Random(11)
    gemms = random_gemms(rng, 60, [(600, 900, 2000), (700, 4000), (600, 1024, 3000)])
    bundle, _ = model_bytes(CELLS[:1], seed=3)
    pools = {ev.pool_key(gemms[0]): POOL}
    out = ev.evaluate_bundle(
        bundle,
        gemms,
        arch="gfx950",
        constants=arch_constants("gfx950"),
        hardware=HW,
        weight_dtype="int4",
        pools=pools,
    )
    assert out.weight_dtype == "int4"
    model = tw.load_model_from_memory(
        mlrec.write_model(bundle, None, "gfx950", arch_constants("gfx950"), "int4")
    )
    for g, r in zip(gemms, out.results):
        assert (r.pick_sol_idx, r.served_by) == expected_pick(tw, model, g)[:2]


def test_pools():
    rng = random.Random(1)
    gemms = random_gemms(rng, 10, [(64,), (64,), (64,)])
    gemms[0]["candidates"].append(dict(POOL[0], sol_idx_global=-3, us=1.0, rank=9))
    pools = ev.build_pools(gemms)
    (key,) = pools
    ids = [k["sol_idx_global"] for k in pools[key]]
    assert ids == sorted(ids) and -3 not in ids
    lib = ev.pool_from_kernels([POOL[5], POOL[2], dict(POOL[5], mt_m=1)])
    assert [k["sol_idx_global"] for k in lib] == [15, 12]
    assert lib[0]["mt_m"] == POOL[5]["mt_m"]
    assert ev.pools_for(gemms, lib) == {key: lib}
    kept, n_rows, n_gemms = ev.restrict_to_pool(gemms, lib)
    assert all(c["sol_idx_global"] in (12, 15) for g in kept for c in g["candidates"])
    assert n_gemms == len(gemms) - len(kept) and n_rows > 0
    for g in kept:
        tested = [c["us"] for c in g["candidates"] if not c["is_skip"]]
        assert g["winner_us"] == (
            min(tested) if tested else min(c["us"] for c in g["candidates"])
        )


def result(pick, origami, winner=1.0, same=False):
    return ev.GemmResult(
        index=0,
        m=1,
        n=1,
        k=1,
        batch=1,
        leaf="x",
        model_cell="x",
        served_by="model",
        winner_us=winner,
        pick_sol_idx=1,
        pick_us=pick,
        origami_sol_idx=1 if same else 2,
        origami_us=origami,
    )


def test_summarize_paired_statistics():
    rs = [
        result(1.0, 2.0),
        result(2.0, 1.0),
        result(1.0, 1.005),
        result(4.0, 4.0, same=True),
    ]
    s = ev.summarize(rs, tie_tolerance=0.01, bootstrap=200)
    p = s["paired"]
    assert (p["wins"], p["ties"], p["losses"]) == (1, 2, 1)
    assert p["same_kernel"] == 1 and p["n"] == 4
    assert s["sel_eff"] == pytest.approx(math.exp(-(math.log(2) + math.log(4)) / 4))
    assert p["speedup_geomean"] == pytest.approx(1.005**0.25)
    lo, hi = p["speedup_ci95"]
    assert lo <= p["speedup_geomean"] <= hi
    assert s["model_sel_eff_percentiles"]["p50"] == pytest.approx(0.75)
    assert ev.summarize([])["n_gemms"] == 0


ENV_PROBE = """
import json, os, sys
sys.path[:0] = [{root!r}, {tests!r}]
from lib import evaluate as ev, mlrec
from lib.hardware import arch_constants
from test_evaluate import CELLS, HW, POOL, gemm
from test_mlrec import random_bundle

labels = [CELLS[0], CELLS[2]]
data = mlrec.write_model(
    random_bundle(labels), None, "gfx950", arch_constants("gfx950"), "fp32"
)
cands = [
    dict(k, us=10.0 + i, is_skip=False, rank=i, is_origami_pick=(i == 0))
    for i, k in enumerate(POOL)
]
gemms = [gemm(4096, 4096, 4096, cands), gemm(100, 100, 100, cands)]
out = ev.evaluate_model(data, gemms, hardware=HW)
ev.tilewright_module()
left = sorted(k for k in ev.ENGINE_DEBUG_ENV if k in os.environ)
print(json.dumps({{"mismatch": out.n_routing_mismatch, "left": left,
                  "cells": [r.model_cell for r in out.results]}}))
"""


def test_engine_debug_variables_do_not_reach_the_evaluation():
    pytest.importorskip("tilewright")
    code = ENV_PROBE.format(root=str(TESTS_DIR.parent), tests=str(TESTS_DIR))
    env = dict(
        os.environ,
        TILEWRIGHT_FORCE_CELL="0",
        TILEWRIGHT_PICK_LOG="1",
        TILEWRIGHT_DIAG="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    assert got == {"mismatch": 0, "left": [], "cells": [CELLS[0], CELLS[2]]}
    assert "[TILEWRIGHT_PICK]" not in proc.stderr
    assert "[TILEWRIGHT_DIAG" not in proc.stderr
    removed = "removed TILEWRIGHT_FORCE_CELL, TILEWRIGHT_PICK_LOG, TILEWRIGHT_DIAG"
    assert proc.stderr.count(removed) == 1


def test_routing_mismatch_is_an_error():
    ok = ev.Evaluation(results=[result(1.0, 1.0)], weight_dtype="fp32", min_scored=1)
    assert ev.require_consistent_routing(ok, "held-out") is ok
    bad = ev.Evaluation(
        results=[result(1.0, 1.0)] * 3,
        weight_dtype="fp32",
        min_scored=1,
        n_routing_mismatch=2,
    )
    with pytest.raises(ev.RoutingMismatchError, match="held-out: .* 2 of 3 GEMMs"):
        ev.require_consistent_routing(bad, "held-out")


def test_top1_picks():
    tw = pytest.importorskip("tilewright")
    _bundle, data = model_bytes(CELLS[:2])
    model = tw.load_model_from_memory(data)
    problems = [
        dict(
            m=m,
            n=n,
            k=k,
            batch_count=1,
            transA="T",
            transB="N",
            a_type="bf16_r",
            b_type="bf16_r",
            c_type="bf16_r",
            d_type="bf16_r",
            compute_type="f32_r",
        )
        for m, n, k in [(1000, 1000, 1000), (300, 2000, 128), (16, 16, 16)]
    ]
    picks = ev.top1_picks(data, problems, POOL, hardware=HW)
    assert [p["reason"] for p in picks] == ["ok", "ok", "no_model"]
    cs = tw.CandidateSet(model, [ev.make_config(tw, k, i) for i, k in enumerate(POOL)])
    for p, prob in zip(picks[:2], problems):
        r = cs.rank(
            ev.make_problem(tw, fs.problem_kwargs_from_row(prob)),
            ev.make_hardware(tw, HW),
            1,
        )[0]
        top = POOL[r.config_index]
        assert p["top1_index"] == r.config_index
        assert (
            p["sol_idx_global"] == top["sol_idx_global"] and p["top1_score"] == r.score
        )
        assert p["n_configs"] == len(POOL) and p["cell"] in CELLS
    ranked = ev.top1_picks(
        data, problems, POOL, hardware=HW, min_scored=8, with_ranking=True
    )
    for p, prob in zip(ranked[:2], problems):
        want = cs.rank(
            ev.make_problem(tw, fs.problem_kwargs_from_row(prob)),
            ev.make_hardware(tw, HW),
            8,
        )
        assert p["ranked"] == [r.config_index for r in want if r.scored]
        assert len(p["ranked"]) == p["n_scored"] >= 8
    assert "ranked" not in picks[0]
    twin_picks = ev.top1_picks(data, problems, TWINS, hardware=HW)
    for p, q in zip(twin_picks[:2], picks):
        assert (p["top1_index"], p["top1_sig"]) == (q["top1_index"], q["top1_sig"])
        assert p["n_configs"] == len(TWINS)
    assert "top1_index" not in twin_picks[2]


def _twin_gemm(drop=()):
    cands = []
    for i, k in enumerate(TWINS):
        if k["sol_idx_global"] in drop:
            continue
        base_us = 10.0 + (i % len(POOL))
        us = base_us if i < len(POOL) else base_us / 2
        cands.append(dict(k, us=us, rank=i, is_skip=False, is_origami_pick=(i == 3)))
    return gemm(2048, 2048, 2048, cands)


def test_duplicate_solutions_are_separate_pool_positions():
    tw = pytest.importorskip("tilewright")
    _bundle, data = model_bytes(CELLS[:1])
    model = tw.load_model_from_memory(data)
    g = _twin_gemm()
    prob = ev.make_problem(tw, fs.problem_kwargs_from_row(g))
    ranked = ev.candidate_set(tw, model, TWINS).rank(prob, ev.make_hardware(tw, HW), 1)
    top, second = ranked[0], ranked[1]
    n = len(POOL)
    assert top.scored and top.config_index < n
    assert (second.config_index, second.score) == (top.config_index + n, top.score)

    base, twin = TWINS[top.config_index], TWINS[second.config_index]
    without_base = _twin_gemm(drop={base["sol_idx_global"]})
    out = ev.evaluate_model(
        data, [g, without_base], hardware=HW, pools={ev.pool_key(g): TWINS}
    )
    first, other = out.results
    assert first.served_by == "model" and first.pick_rank == 0
    assert (first.pick_index, first.pick_sol_idx) == (
        top.config_index,
        base["sol_idx_global"],
    )
    assert first.pick_us == 10.0 + top.config_index
    assert first.winner_us == 5.0
    assert first.sel_eff == pytest.approx(5.0 / (10.0 + top.config_index))
    assert (first.origami_index, first.origami_sol_idx) == (3, 13)
    assert (other.pick_index, other.pick_sol_idx) == (
        second.config_index,
        twin["sol_idx_global"],
    )
    assert other.pick_us == (10.0 + top.config_index) / 2 and other.pick_rank == 1
    assert out.n_pool_configs == {"|".join(ev.pool_key(g)): 2 * n}


def test_configs_carry_the_kernel_attributes():
    tw = pytest.importorskip("tilewright")
    if not ev.module_takes_attributes(tw):
        pytest.skip("the installed tilewright module predates kernel attributes")
    attrs = {"workgroup_mapping": -4, "wave_num": 4, "stream_k_atomic": 0}
    c = ev.make_config(tw, dict(POOL[1], attributes=attrs), 7)
    assert c.attributes == attrs and c.index == 7
    assert ev.make_config(tw, POOL[1], 0).attributes == {}
    _bundle, data = model_bytes(CELLS[:1])
    cs = ev.candidate_set(tw, tw.load_model_from_memory(data), TWINS)
    assert [c.attributes for c in cs.configs] == [
        k.get("attributes", {}) for k in TWINS
    ]
    assert [c.index for c in cs.configs] == list(range(len(TWINS)))


class _ConfigWithoutAttributes:
    def __init__(self, **kwargs):
        if "attributes" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument")
        self.__dict__.update(kwargs)


def test_a_module_without_attributes_gets_configs_without_them(capsys):
    old = types.SimpleNamespace(
        Config=_ConfigWithoutAttributes, Dim3=types.SimpleNamespace
    )
    kern = dict(POOL[0], attributes={"workgroup_mapping": 8})
    configs = [ev.make_config(old, kern, i) for i in range(3)]
    assert [c.index for c in configs] == [0, 1, 2]
    assert not hasattr(configs[0], "attributes") and configs[0].grvw_a == 8
    assert not ev.module_takes_attributes(old)
    assert capsys.readouterr().err.count("takes no kernel attributes") == 1


def test_candidates_follow_the_pool_order():
    cands = [dict(POOL[i], us=1.0 + i) for i in (0, 1, 2, 4)]
    cands.append(dict(POOL[5], sol_idx_global=-4, us=2.0))
    g = gemm(64, 64, 64, cands)
    other = gemm(64, 64, 64, [dict(POOL[2], us=1.0), dict(POOL[0], us=2.0)])
    other["transB"] = "T"
    pools = {ev.pool_key(g): [POOL[2], POOL[0], POOL[1]]}
    ev.order_candidates_by_pool([g, other], pools)
    assert [c["sol_idx_global"] for c in g["candidates"]] == [12, 10, 11, -4, 14]
    assert [c["sol_idx_global"] for c in other["candidates"]] == [10, 12]


def test_measured_pools_carry_the_attributes_the_data_lists():
    attrs = {"workgroup_mapping": 4}
    g = gemm(64, 64, 64, [dict(TWINS[-1], us=1.0), dict(POOL[0], us=2.0)])
    (pool,) = ev.build_pools([g]).values()
    assert [k["sol_idx_global"] for k in pool] == [10, 200 + len(POOL) - 1]
    assert "attributes" not in pool[0] and pool[1]["attributes"] == attrs


def test_library_pools_carry_the_kernel_attributes(tmp_path):
    sols = [
        solution(7, 0, (128, 128, 64), workGroupMapping=8),
        solution(3, 1, (128, 128, 64), workGroupMapping=4),
        solution(7, 2, (256, 256, 64)),
    ]
    write_logic(tmp_path / f"{LIBRARY_STEM}.dat", sols)
    pool = ev.library_pool(tmp_path, LIBRARY_STEM)
    assert [k["sol_idx_global"] for k in pool] == [7, 3]
    for k, sol in zip(pool, sols):
        info = dat.kernel_dat_info(sol)
        assert k["attributes"] == info["attributes"]
        assert {f: k[f] for f in ev.CONFIG_FIELDS} == {
            f: info[f] for f in ev.CONFIG_FIELDS
        }
    assert [k["attributes"]["workgroup_mapping"] for k in pool] == [8, 4]


GUARDED_LABEL = "Tiny|Large|LargeK|Bnone"
M_GUARD = [{"free": "M", "minimum": 4}]


def timed_gemm(m, origami=None):
    """A GEMM of the guard tests: every kernel of POOL timed, Origami's pick
    `origami` (default the first)."""
    first = POOL[0]["sol_idx_global"] if origami is None else origami
    cands = [dict(k, us=10.0 + i, rank=i, is_skip=False) for i, k in enumerate(POOL)]
    for c in cands:
        c["is_origami_pick"] = c["sol_idx_global"] == first
    return gemm(m, 1024, 2048, cands)


def guarding(pool, sid):
    """`pool` with kernel `sid` marked with a guard of M >= 4, as a pool built
    with a guarded blocklist carries it."""
    return [dict(k, guards=M_GUARD) if k["sol_idx_global"] == sid else k for k in pool]


def v3_model(guards, view_keys=None):
    """An MLREC_v3 model of GUARDED_LABEL, flagged as trained with kernel
    guards when `guards`; `view_keys` its cell's whitelist."""
    from test_v3 import v3_bundle, write

    bundle = v3_bundle(labels=[GUARDED_LABEL])
    if view_keys is not None:
        bundle["models"][GUARDED_LABEL]["view_key_hashes"] = list(view_keys)
    data = write(bundle)
    return mlrec.with_kernel_guards(data) if guards else data


def served_ranking(tw, model, g, min_scored=1, pool=POOL):
    """The solutions the engine scores for `g`, best first, over `pool` with
    the launch plans an evaluation hands it."""
    cs = ev.candidate_set(tw, model, pool, HW)
    prob = ev.make_problem(tw, fs.problem_kwargs_from_row(g))
    ctx = ev.make_context(tw, ev.context_of(g))
    ranked = cs.rank(prob, ev.make_hardware(tw, HW), min_scored, ctx)
    return [pool[r.config_index]["sol_idx_global"] for r in ranked if r.scored]


def test_a_pick_passes_over_a_guarded_kernel_below_its_guard():
    """For a model trained with kernel guards a pool kernel with guards is
    no pick of a GEMM its guards deny: the walk passes over it, asking for
    one more scored config, as TensileLite's ranker does. Above its guard
    the evaluation is the unguarded pool's."""
    tw = pytest.importorskip("tilewright")
    data = v3_model(guards=True)
    assert ev.serves_kernel_guards(mlrec.read_header(data))
    model = tw.load_model_from_memory(data)
    gemms = [timed_gemm(m) for m in (1, 2, 3, 4, 5, 8)]
    top = served_ranking(tw, model, gemms[1])[0]
    guarded = guarding(POOL, top)
    assert ev.guarded_positions(guarded) == {
        next(i for i, k in enumerate(POOL) if k["sol_idx_global"] == top): (4, 0)
    }
    key = ev.pool_key(gemms[0])
    out = ev.evaluate_model(
        data, gemms, hardware=HW, pools={key: guarded}, pick_measurement="timed"
    )
    plain = ev.evaluate_model(
        data, gemms, hardware=HW, pools={key: POOL}, pick_measurement="timed"
    )
    assert out.n_pool_configs == {"|".join(key): len(POOL)}
    for g, r, p in zip(gemms, out.results, plain.results):
        below = g["m"] < 4
        ranked = served_ranking(tw, model, g, 1 + below)
        assert r.pick_sol_idx == next(s for s in ranked if not (below and s == top))
        if below:
            assert r.pick_sol_idx != top and r.served_by == "model"
        else:
            assert (r.pick_sol_idx, r.pick_us, r.n_scored) == (
                p.pick_sol_idx,
                p.pick_us,
                p.n_scored,
            )
    assert any(r.pick_sol_idx == top for r in plain.results if r.m < 4)
    # top1_picks is the engine's own ranking, the guards play no part.
    picks = ev.top1_picks(data, gemms, guarded, hardware=HW)
    assert [p["sol_idx_global"] for p in picks] == [
        r.pick_sol_idx for r in plain.results
    ]


def test_a_guard_on_a_whitelist_of_one_kernel_reaches_the_second_tier():
    """The one more scored config a denied one asks for is what reaches the
    engine's second tier: when the cell's whitelist keeps only the guarded
    kernel, a top-1 request below its guard is served by the model's best
    other config, not by the Origami fallback."""
    tw = pytest.importorskip("tilewright")
    below, above = timed_gemm(2), timed_gemm(8)
    for k in POOL:
        view = fs.v3_view_key(k, k.get("attributes"), fs.DEFAULT_ATTRIBUTE_SPEC)
        data = v3_model(guards=True, view_keys=[fs.view_key_hash(view)])
        model = tw.load_model_from_memory(data)
        one = [k["sol_idx_global"]]
        if served_ranking(tw, model, below) == one == served_ranking(tw, model, above):
            break
    else:
        pytest.fail("no kernel of POOL makes a whitelist of one feasible kernel")
    sid = k["sol_idx_global"]
    tiers = served_ranking(tw, model, below, 2)
    assert tiers[0] == sid and len(tiers) > 1
    out = ev.evaluate_model(
        data,
        [below, above],
        hardware=HW,
        pools={ev.pool_key(below): guarding(POOL, sid)},
        pick_measurement="timed",
    )
    low, high = out.results
    assert (low.served_by, low.pick_sol_idx) == ("model", tiers[1])
    assert (high.served_by, high.pick_sol_idx) == ("model", sid)


@pytest.mark.parametrize("version", [2, 3])
def test_a_model_without_kernel_guards_is_served_without_the_guarded_kernel(
    version,
):
    """TensileLite reads the guards sidecar only for a model trained with
    kernel guards; the blocklist sidecar, which names a guarded kernel too,
    leaves it out of any other model's pools whole. Such a model evaluated
    with a pool that keeps the kernel, marked with its guards, picks what it
    picks with the pool and the data without the kernel, above the guard as
    below it, its top-1 picks and the Origami fallback included; Origami's
    own pick, which reads no sidecar, stays."""
    tw = pytest.importorskip("tilewright")
    if version == 2:
        data = model_bytes([GUARDED_LABEL])[1]
        unscored = {"m": 64, "n": 64, "k": 256}
    else:
        data = v3_model(guards=False)
        # A schedule the model does not list: a CU budget outside its range
        # is ranked at the nearest budget in it, so it never leaves a GEMM
        # unscored.
        unscored = {"schedule": "auto"}
    assert not ev.serves_kernel_guards(mlrec.read_header(data))
    model = tw.load_model_from_memory(data)
    top = served_ranking(tw, model, timed_gemm(8))[0]
    other = next(k["sol_idx_global"] for k in POOL if k["sol_idx_global"] != top)
    gemms = [timed_gemm(m) for m in (1, 2, 3, 4, 5, 8)]
    gemms.append(dict(timed_gemm(8, origami=top), **unscored))
    gemms.append(dict(timed_gemm(8, origami=other), **unscored))
    without = [k for k in POOL if k["sol_idx_global"] != top]
    rows_without = [
        dict(g, candidates=[c for c in g["candidates"] if c["sol_idx_global"] != top])
        for g in gemms
    ]
    key = ev.pool_key(gemms[0])

    def served(e):
        return [
            (r.pick_sol_idx, r.pick_us, r.pick_index, r.pick_rank, r.served_by)
            + (r.n_scored, r.miss, r.model_cell)
            for r in e.results
        ]

    def evaluated(d, rows, pool):
        return ev.evaluate_model(
            d, rows, hardware=HW, pools={key: pool}, pick_measurement="timed"
        )

    out = evaluated(data, gemms, guarding(POOL, top))
    alone = evaluated(data, rows_without, without)
    assert served(out) == served(alone)
    assert out.n_pool_configs == alone.n_pool_configs == {"|".join(key): len(without)}
    assert all(r.pick_sol_idx != top for r in out.results)
    assert top in [r.pick_sol_idx for r in evaluated(data, gemms, POOL).results]
    *_, origami_picks_it, origami_other = out.results
    assert origami_picks_it.origami_sol_idx == top
    assert origami_picks_it.served_by is None
    assert (origami_other.served_by, origami_other.pick_sol_idx) == (
        "origami_fallback",
        other,
    )
    assert ev.top1_picks(data, gemms, guarding(POOL, top), hardware=HW) == (
        ev.top1_picks(data, gemms, without, hardware=HW)
    )
    if version == 3:
        # The same model flagged serves the kernel above its guard, from its
        # own ranking and from the Origami fallback.
        res = evaluated(mlrec.with_kernel_guards(data), gemms, guarding(POOL, top))
        assert res.results[5].pick_sol_idx == top
        assert all(r.pick_sol_idx != top for r in res.results[:3])
        assert (res.results[6].served_by, res.results[6].pick_sol_idx) == (
            "origami_fallback",
            top,
        )


@pytest.mark.parametrize("budget", [1, 8, 16, 24])
def test_a_budget_below_the_cu_range_is_guarded_as_the_nearest_budget(budget):
    """A CU budget below a model's range is ranked at the nearest budget in
    it, and the guards read the problem's free sizes alone, never its
    context, as TensileLite's do: a model trained with kernel guards passes
    over a guarded kernel below its guard at such a budget exactly as at the
    nearest budget, and serves what the unguarded pool serves above it."""
    tw = pytest.importorskip("tilewright")
    data = v3_model(guards=True)
    model = tw.load_model_from_memory(data)
    nearest = 25  # v3_bundle's smallest share, a quarter of HW's 100 CUs
    top = served_ranking(tw, model, dict(timed_gemm(2), cu_budget=nearest))[0]
    ms = (1, 2, 3, 4, 5, 8)
    low = [dict(timed_gemm(m), cu_budget=budget) for m in ms]
    at = [dict(timed_gemm(m), cu_budget=nearest) for m in ms]
    for g, h in zip(low, at):
        assert served_ranking(tw, model, g, 3) == served_ranking(tw, model, h, 3)
    key = ev.pool_key(low[0])

    def served(rows, pool):
        e = ev.evaluate_model(
            data, rows, hardware=HW, pools={key: pool}, pick_measurement="timed"
        )
        return [
            (r.pick_sol_idx, r.pick_us, r.pick_index, r.pick_rank, r.served_by)
            + (r.n_scored,)
            for r in e.results
        ]

    guarded = served(low, guarding(POOL, top))
    assert guarded == served(at, guarding(POOL, top))
    plain = served(low, POOL)
    for m, g, p in zip(ms, guarded, plain):
        if m < 4:
            assert g[0] != top and g[4] == "model", m
        else:
            assert g[:2] == p[:2], m
    assert any(p[0] == top for m, p in zip(ms, plain) if m < 4)


def test_the_origami_fallback_of_a_budget_outside_the_range_obeys_the_guards():
    """A schedule the model does not list leaves a GEMM to the Origami
    fallback at any CU budget; for a model trained with kernel guards the
    fallback passes over a guarded kernel below its guard there too, as
    TensileLite's analytical ranking does, and serves it above."""
    tw = pytest.importorskip("tilewright")
    data = v3_model(guards=True)
    model = tw.load_model_from_memory(data)
    top = served_ranking(tw, model, timed_gemm(2))[0]
    gemms = [
        dict(timed_gemm(m, origami=top), schedule="auto", cu_budget=16)
        for m in (1, 2, 3, 4, 8)
    ]
    out = ev.evaluate_model(
        data,
        gemms,
        hardware=HW,
        pools={ev.pool_key(gemms[0]): guarding(POOL, top)},
        pick_measurement="timed",
    )
    for g, r in zip(gemms, out.results):
        if g["m"] < 4:
            assert r.pick_sol_idx != top, g["m"]
        else:
            assert (r.served_by, r.pick_sol_idx) == ("origami_fallback", top), g["m"]


def test_origami_candidate_passes_over_the_rows_it_is_told_to():
    """Rows of excluded solutions count as absent, as a blocked kernel's
    dropped rows do: a flagged pick of one leaves the GEMM without Origami's
    pick, and without the column the next tested row by rank stands in."""
    cands = [
        dict(POOL[0], us=5.0, rank=0, is_skip=False, is_origami_pick=True),
        dict(POOL[3], us=4.0, rank=1, is_skip=False, is_origami_pick=False),
    ]
    first, other = POOL[0]["sol_idx_global"], POOL[3]["sol_idx_global"]
    flagged = gemm(64, 64, 256, cands)
    assert ev.origami_candidate(flagged)["sol_idx_global"] == first
    assert ev.origami_candidate(flagged, {other})["sol_idx_global"] == first
    assert ev.origami_candidate(flagged, {first}) is None
    column_free = gemm(64, 64, 256, [dict(c, is_origami_pick=None) for c in cands])
    assert ev.origami_candidate(column_free)["sol_idx_global"] == first
    assert ev.origami_candidate(column_free, {first})["sol_idx_global"] == other
