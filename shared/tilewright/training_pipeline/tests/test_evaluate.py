# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

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


def expected_pick(tw, model, g, min_scored=1):
    cs = tw.CandidateSet(model, [ev.make_config(tw, k, i) for i, k in enumerate(POOL)])
    measured = {c["sol_idx_global"]: c["us"] for c in g["candidates"]}
    prob = ev.make_problem(tw, fs.problem_kwargs_from_row(g))
    for r in cs.rank(prob, ev.make_hardware(tw, HW), min_scored):
        if not r.scored:
            break
        sid = POOL[r.config_index]["sol_idx_global"]
        if sid in measured:
            return sid, "model"
    oc = [c for c in g["candidates"] if c["is_origami_pick"]]
    return (oc[0]["sol_idx_global"], "origami_fallback") if oc else (None, None)


def test_picks_follow_candidate_set_rank():
    tw = pytest.importorskip("tilewright")
    rng = random.Random(7)
    gemms = random_gemms(rng, 80, [(600, 900, 2000), (700, 4000), (600, 1024, 3000)])
    gemms += random_gemms(rng, 40, [(200, 300), (700, 2000), (64, 256)])
    _bundle, data = model_bytes(CELLS[:2])
    out = ev.evaluate_model(
        data, gemms, hardware=HW, pools={ev.pool_key(gemms[0]): POOL}
    )
    model = tw.load_model_from_memory(data)
    served = set()
    for g, r in zip(gemms, out.results):
        sid, by = expected_pick(tw, model, g)
        assert (r.pick_sol_idx, r.served_by) == (sid, by)
        if sid is not None:
            us = {c["sol_idx_global"]: c["us"] for c in g["candidates"]}[sid]
            assert r.pick_us == us and r.sel_eff == pytest.approx(g["winner_us"] / us)
        served.add(by)
    assert "model" in served
    assert out.n_routing_mismatch == 0 and out.weight_dtype == "fp32"


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
        assert (r.pick_sol_idx, r.served_by) == expected_pick(tw, model, g)


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
        assert (
            p["sol_idx_global"] == top["sol_idx_global"] and p["top1_score"] == r.score
        )
        assert p["n_configs"] == len(POOL) and p["cell"] in CELLS
