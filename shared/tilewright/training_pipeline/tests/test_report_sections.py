# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bench analysis (lib/run_analysis.py), cell evidence and the report
sections built on them, on synthetic runs."""

import csv
import json
import math
import random
import re
import types

import numpy as np
import pytest

import bench_fakes as bf
import make_report as mr
import report_sections as rs
import stage05_train as s5mod
import stage07_optional_validate_selection_time as s7
from lib import cell_evidence as ce
from lib import evaluate as ev
from lib import feasibility
from lib import features as fs
from lib import grid, planner
from lib import miss_rebench
from lib import run_analysis as ra
from test_stage05_train import (
    CELLS3,
    COLUMNS,
    HARDWARE,
    gemm_rows,
    have_engine,
    kernel_pool,
    shapes_for,
    train,
    write_round,
)

TINY = "Tiny|Tiny|TinyK|Bnone"
LARGE = "Large|Large|LargeK|Bnone"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def row(m, n, k, sid, us, skip=False, **cfg):
    base = dict(
        is_skip="True" if skip else "False",
        rank=0,
        sol_idx_global=sid,
        transA="T",
        transB="N",
        batch_count=1,
        m=m,
        n=n,
        k=k,
        a_type="bf16_r",
        b_type="bf16_r",
        c_type="bf16_r",
        d_type="bf16_r",
        compute_type="f32_r",
        us=us,
        mt_m=64,
        mt_n=64,
        mt_k=64,
        mi_m=16,
        mi_n=16,
        mi_k=32,
        occupancy=1,
        cache_hints_a=0,
        cache_hints_b=0,
        grvw_a=8,
        grvw_b=8,
        gwvw_d=4,
    )
    base.update(cfg)
    return base


# ── bench loading ───────────────────────────────────────────────────────────


def test_bench_rows_follow_the_training_rules(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_rows(
        a / "chunk_0000.csv",
        [
            row(64, 64, 64, 1, 5.0),
            row(64, 64, 64, 2, 3.0, skip=True),
            row(64, 64, 64, 3, 4.0),
            row(64, 64, 64, 4, "nan"),
            row(64, 64, 64, 5, 1.0, mt_m=0),
            row(2048, 2048, 2048, 1, 9.0, skip=True),
            row(2048, 2048, 2048, 2, 8.0, skip=True),
        ],
    )
    write_rows(b / "chunk_0000.csv", [row(64, 64, 64, 1, 3.5)])
    bench = ra.load_bench([a, b])
    by_m = {g.key[0]: g for g in bench.gemms}
    small = by_m[64]
    assert dict(zip(small.sids.tolist(), small.us.tolist())) == {
        1: 4.25,
        2: 3.0,
        3: 4.0,
    }
    assert small.winner_us == 4.0
    assert by_m[2048].winner_us == 8.0
    fastest = {g.key[0]: g for g in ra.load_bench([a, b], aggregation="min").gemms}
    assert dict(zip(fastest[64].sids.tolist(), fastest[64].us.tolist()))[1] == 3.5
    assert fastest[64].winner_us == 3.5
    assert bench.stats["bad_rows"] == 2
    assert bench.stats["gemms"] == 2
    kept = ra.load_bench([a], keep=lambda p: int(p["m"]) > 100)
    assert [g.key[0] for g in kept.gemms] == [2048]


# The rules of an MLREC_v2 bundle (every rule; the non-temporal rule
# switched, refined family-wise) and of MLREC_v3 bundles by their flags.
RULE_SETS = [
    ra.serving_rules(None),
    ra.serving_rules(ra.ModelInfo(labels=[], whitelists={}, non_temporal_rule=False)),
    ra.serving_rules(
        ra.ModelInfo(labels=[], whitelists={}, non_temporal_family_rule=True)
    ),
    feasibility.soft_rules({}, 3),
    feasibility.soft_rules({"small_batched_rule": True, "dot2_rule": True}, 3),
    feasibility.soft_rules(
        {"non_temporal_rule": True, "non_temporal_family_rule": True}, 3
    ),
]


def test_an_mlrec_v2_bundle_keeps_every_rule():
    """A bundle whose model file was not read serves with MLREC_v2's rules,
    as every bundle did in the analysis before it read the file's flags; a
    read file's flags decide (lib.feasibility.soft_rules)."""
    every = {
        "small_batched_rule": True,
        "dot2_rule": True,
        "leading_free_rule": True,
        "non_temporal_rule": True,
    }
    assert ra.serving_rules(None) == every
    info = ra.ModelInfo(labels=[], whitelists={}, non_temporal_rule=False)
    assert ra.serving_rules(info) == dict(every, non_temporal_rule=False)
    info.rules = feasibility.soft_rules({"dot2_rule": True}, 3)
    assert ra.serving_rules(info) == {
        "leading_free_rule": False,
        "small_batched_rule": False,
        "dot2_rule": True,
        "non_temporal_rule": False,
    }


def test_the_analysis_caption_names_the_rules_it_applies():
    v3 = mr.serving_rules_text({"version": 3, "rules": feasibility.soft_rules({}, 3)})
    assert "small-batched, Dot2 and non-temporal rules off" in v3
    assert "TensileLite's predicate" in v3
    v2 = mr.serving_rules_text({"version": None, "rules": ra.serving_rules(None)})
    assert "small-batched, Dot2 and non-temporal rules on" in v2
    assert "not read" in v2


def test_gate_masks_apply_the_bundles_own_rules():
    rng = random.Random(3)
    kernels = []
    for i, k in enumerate(kernel_pool()):
        for cha, chb in ((0, 0), (4, 0), (0, 4), (1, 0), (0, 3), (4, 4)):
            kernels.append(
                dict(
                    k,
                    cache_hints_a=cha,
                    cache_hints_b=chb,
                    grvw_a=rng.choice((1, 4, 8)),
                    grvw_b=rng.choice((1, 4, 8)),
                    sol_idx_global=1000 + len(kernels),
                )
            )
    kernels.append(dict(kernels[0], mi_m=1, mi_n=1, mi_k=64, sol_idx_global=1))
    for cha, chb in ((3, 1), (1, 1)):
        kernels.append(dict(kernels[0], mt_m=8, mt_n=512, mt_k=32, mi_m=4, mi_n=4,
                            mi_k=4, cache_hints_a=cha, cache_hints_b=chb,
                            sol_idx_global=10 + cha))  # fmt: skip
    kernels.append(dict(kernels[0], mt_m=16, mt_n=128, mt_k=64, cache_hints_a=4,
                        cache_hints_b=0, sol_idx_global=20))  # fmt: skip
    kernels.append(dict(kernels[-1], cache_hints_a=0, sol_idx_global=21))
    pool = ra.Pool(kernels, "test")
    nt_a, nt_b = feasibility.non_temporal_availability(kernels)
    families = feasibility.non_temporal_families(kernels)
    cfgs = [fs.config_kwargs_from_row({**k}) for k in kernels]
    shapes = [(1, 4096, 1024, 1), (2, 64, 64, 4), (64, 64, 512, 8)]
    shapes += [(2, 1792, 4, 4096), (16, 1792, 64, 8), (100, 20, 256, 2), (20, 20, 4, 2)]
    shapes += [(4096, 8, 2048, 1), (8192, 8192, 4096, 1), (3, 3, 4096, 1)]
    shapes += [
        (rng.randint(1, 9000), rng.randint(1, 9000), 32 * rng.randint(1, 300), 1)
        for _ in range(30)
    ]
    for m, n, k, b in shapes:
        for ta, tb in (("T", "N"), ("N", "T"), ("N", "N")):
            for dt in ("bf16_r", "f8_r", "f32_r"):
                prob = fs.problem_kwargs_from_row(
                    dict(
                        m=m,
                        n=n,
                        k=k,
                        batch_count=b,
                        transA=ta,
                        transB=tb,
                        a_type=dt,
                        b_type=dt,
                        c_type="bf16_r",
                        d_type="bf16_r",
                        compute_type="f32_r",
                    )
                )
                lf = ra.leading_free_mask(pool, prob)
                for rules in RULE_SETS:
                    hard, soft = ra.gate_masks(
                        pool, prob, lds_bytes=HARDWARE["lds_bytes"], rules=rules
                    )

                    def passes(**rule):
                        return [
                            feasibility.passes_gates(
                                prob,
                                c,
                                lds_bytes=HARDWARE["lds_bytes"],
                                nt_a_available=nt_a,
                                nt_b_available=nt_b,
                                nt_families=families,
                                **dict(rules, **rule),
                            )
                            for c in cfgs
                        ]

                    where = (m, n, k, b, ta, tb, dt, rules)
                    assert soft.tolist() == passes(), where
                    assert hard.tolist() == passes(non_temporal_rule=False), where
                    assert not (soft & ~hard).any()
                    if rules.get("leading_free_rule", True):
                        assert not (hard & ~lf).any()
                only_lf = dict(
                    non_temporal_rule=False, small_batched_rule=False, dot2_rule=False
                )
                assert lf.tolist() == [
                    feasibility.passes_gates(
                        prob,
                        c,
                        lds_bytes=1 << 40,
                        nt_a_available=nt_a,
                        nt_b_available=nt_b,
                        **only_lf,
                    )
                    for c in cfgs
                ]


# ── evaluation without the engine ───────────────────────────────────────────


def _model(labels, whitelists, nt=True):
    return ra.ModelInfo(
        labels=list(labels),
        whitelists={c: [tuple(s) for s in w] for c, w in whitelists.items()},
        non_temporal_rule=nt,
        hardware=ra.DeviceHardware(**HARDWARE),
    )


def test_levels_and_decomposition_without_the_engine(tmp_path):
    pool_k = kernel_pool()
    data = write_round(tmp_path / "s4", shapes_for(1, 12, CELLS3), pool=pool_k)
    bench = ra.load_bench([data])
    pool = ra.pool_for(bench, None)
    assert pool.source == "bench" and len(pool) == len(pool_k)
    sig = [tuple(pool_k[0][f] for f in ra.SIG_FIELDS)]
    model = _model(CELLS3[:2], {CELLS3[0]: sig, CELLS3[1]: []})
    e = ra.evaluate(bench, pool, model)
    assert not e.engine and not e.served.any()
    lv = e.levels[:, :4]
    finite = np.isfinite(lv).all(axis=1)
    assert (np.diff(lv[finite], axis=1) >= 0).all()
    assert np.allclose(lv[:, 0], [g.winner_us for g in e.gemms])
    for g, cell, blk in zip(e.gemms, e.model_cell, e.blocked):
        if g.leaf == CELLS3[2]:
            assert cell is None and blk == "tier"
    d = ra.decomposition(e)
    assert d["*"]["n_gemms"] == len(e.gemms) and d["*"]["n_served"] == 0
    assert d["*"]["steps"]["ranking"] is None
    assert d[CELLS3[1]]["steps"]["whitelist"] == pytest.approx(1.0)
    assert d[CELLS3[0]]["steps"]["whitelist"] < 1.0
    assert d[CELLS3[2]]["blocked"] == {"tier": d[CELLS3[2]]["n_gemms"]}


def test_a_carried_bundle_is_read_with_the_files_of_the_round_it_came_from(
    tmp_path,
):
    """Rounds carried before options were copied hold only the model and
    cells.json, and no carried round holds metrics.json: the shipped
    bundle's options and metrics are those of the round it was trained in,
    through every carry."""
    import choose_shipped_subset as css

    run = tmp_path / "run"
    cells = json.dumps({"cells": [{"label": TINY}], "model_labels": [TINY]})
    met = {
        "weight_dtype": "int4",
        "hardware": HARDWARE,
        "args": {"arch": "gfx950", "launch_keys": True, "pick_measurement": "any"},
    }
    spec = [{"name": "param_a", "transform": "log2p1"}]
    src = run / "round_0" / "stage05"
    src.mkdir(parents=True)
    (src / "cells.json").write_text(cells)
    (src / "metrics.json").write_text(json.dumps(met))
    (src / "model_options.json").write_text(json.dumps({"attribute_spec": spec}))
    for rnd in ("round_1", "round_2"):
        dst = run / rnd / "stage05"
        dst.mkdir(parents=True)
        (dst / "cells.json").write_text(cells)
        (dst / "carried_forward.json").write_text(json.dumps({"source": str(src)}))
        src = dst
    (run / "round_2" / "stage06").mkdir()
    record = {"train_dir": str(src), "deployed": True}
    (run / "round_2" / "stage06" / "deploy_record.json").write_text(json.dumps(record))
    info = ra.final_model(run, want_bytes=False)
    assert (info.round, info.labels) == ("round_2", [TINY])
    assert (info.weight_dtype, info.arch, info.launch_keys) == ("int4", "gfx950", True)
    assert info.hardware == ra.DeviceHardware(**HARDWARE)
    assert info.attribute_spec == fs.parse_attribute_spec(spec)
    assert css.bundle_pick_measurement(run, info) == "any"


def test_kernels_group_by_signature(tmp_path):
    shapes = [((64, 64, 64), 1), ((128, 128, 128), 2), ((256, 256, 256), 3)]
    rows = []
    for (m, n, k), win in shapes:
        for sid, extra in ((1, {"grvw_a": 8}), (2, {"grvw_a": 4}), (3, {"mt_m": 128})):
            rows.append(row(m, n, k, sid, 1.0 if sid == win else 2.0 + sid, **extra))
    write_rows(tmp_path / "s4" / "chunk_0000.csv", rows)
    bench = ra.load_bench([tmp_path / "s4"])
    pool = ra.pool_for(bench, None)
    assert len(pool) == 3 and len(pool.signatures) == 2
    e = ra.evaluate(bench, pool, None)
    stats = ra.kernel_stats(e)
    by_sig = {r["sig"]: r for r in stats["kernels"]}
    shared = by_sig["MT64x64x64 MI16x16x32 NTA0 NTB0"]
    assert (shared["n_members"], shared["wins"], shared["tested"]) == (2, 2, 3)
    assert {m["sid"]: m["wins"] for m in shared["members"]} == {1: 1, 2: 1}
    assert by_sig["MT128x64x64 MI16x16x32 NTA0 NTB0"]["wins"] == 1
    # dropping a signature drops all of its members
    assert shared["drop_loss"] == pytest.approx(1.0 - (5.0 ** (-2 / 3)))
    assert stats["n_solutions"] == 3

    params = ra.parameter_stats(e)
    assert set(params["params"]) == set(ra.COMMON_PARAMS)
    assert params["diagnostics"] is None
    assert {
        r["value"]: r["kernels"] for r in params["params"]["macro tile (MT_M x MT_N)"]
    } == {
        "64x64": 1,
        "128x64": 1,
    }
    diag = ra.parameter_stats(e, diagnostics=True)["diagnostics"]
    grvw = {r["value"]: (r["kernels"], r["wins"]) for r in diag["params"]["GRVW (A,B)"]}
    assert grvw == {"8,8": (2, 2), "4,8": (1, 1)}

    a = ra.summarize_evaluation(e, bench)
    assert a["pool"] == {"source": "bench", "size": 3, "signatures": 2}
    html = rs.kernels_html(a)
    assert '<td class="c">MT64x64x64 MI16x16x32 NTA0 NTB0</td>' in html
    assert "<details><summary>2</summary>" in html and "wins per member: 1, 1" in html
    assert rs.diagnostics_html(a) == ""
    a = ra.summarize_evaluation(e, bench, diagnostics=True)
    html = rs.diagnostics_html(a)
    assert "Diagnostics: framework-specific kernel parameters" in html
    assert html.index("<details>") < html.index('id="diag_sel"')


def test_contexts_are_separate_gemms(tmp_path):
    cols = COLUMNS + ["cu_budget", "schedule"]
    rows = [
        dict(row(64, 64, 64, 1, 2.0), cu_budget="", schedule=""),
        dict(row(64, 64, 64, 2, 3.0), cu_budget="", schedule=""),
        dict(row(64, 64, 64, 1, 4.0), cu_budget="128", schedule="dynamic"),
        dict(row(64, 64, 64, 2, 1.0), cu_budget="128", schedule="dynamic"),
    ]
    path = tmp_path / "s4" / "chunk_0000.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    bench = ra.load_bench([tmp_path / "s4"])
    assert len(bench.gemms) == 2
    ctx = fs.ExecutionContext(cu_budget=128, schedule="dynamic")
    by_ctx = {g.context: g for g in bench.gemms}
    assert set(by_ctx) == {fs.EXCLUSIVE, ctx}
    assert (by_ctx[fs.EXCLUSIVE].winner_us, by_ctx[ctx].winner_us) == (2.0, 1.0)
    assert (
        by_ctx[ctx].row["cu_budget"] == 128
        and "cu_budget" not in by_ctx[fs.EXCLUSIVE].row
    )
    e = ra.evaluate(bench, ra.pool_for(bench, None), None)
    label = ra.context_label(ctx)
    assert label != "exclusive"
    dec = ra.context_decomposition(e)
    assert sorted(dec) == sorted(["exclusive", label])
    assert dec[label]["*"]["n_gemms"] == dec["exclusive"]["*"]["n_gemms"] == 1


# ── picks as the runtime serves them ────────────────────────────────────────


class FakeEngine:
    """Enough of the tilewright module for `run_analysis.evaluate`: its model
    routes every problem to `cell` (or, given several, to its base cell
    among them) and scores the pool positions `order` gives the problem's
    (m, n, k, batch), best first, whatever the rules (the other kernels
    unscored)."""

    Dim3 = Problem = Hardware = Config = ExecutionContext = types.SimpleNamespace
    DataType = int
    Transpose = types.SimpleNamespace(T="T", N="N")
    Schedule = types.SimpleNamespace(Default=0, Dynamic=1, Auto=2)

    def __init__(self, cell, order):
        self.cells = [cell] if isinstance(cell, str) else list(cell)
        self.order = order

    def load_model_from_memory(self, data):
        return data

    def describe(self, model):
        return types.SimpleNamespace(format_version=2, arch="gfx950")

    def CandidateSet(self, model, configs):
        order = self.order

        class Ranking:
            def rank(self, problem, hw, min_scored, context):
                size = problem.size
                top = list(order.get((size.m, size.n, size.k, problem.batch), ()))
                rest = [i for i in range(len(configs)) if i not in top]
                return [
                    types.SimpleNamespace(config_index=i, scored=i in top)
                    for i in top + rest
                ]

        return Ranking()

    def route(self, model, problem):
        if len(self.cells) == 1:
            return 0
        size = problem.size
        return self.cells.index(grid.cell_key(size.m, size.n, size.k, problem.batch))

    def cell_label(self, model, index):
        return self.cells[index]


def _kernel(sid, **kw):
    k = dict(
        mt_m=64,
        mt_n=64,
        mt_k=64,
        mi_m=16,
        mi_n=16,
        mi_k=32,
        occupancy=1,
        cache_hints_a=0,
        cache_hints_b=0,
        grvw_a=8,
        grvw_b=8,
        gwvw_d=4,
        sol_idx_global=sid,
    )
    k.update(kw)
    return k


def _bench_dir(path, rows, contexts=False):
    cols = COLUMNS + (["cu_budget", "schedule"] if contexts else [])
    path.mkdir(parents=True, exist_ok=True)
    with (path / "chunk_0000.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    return path


def _served(
    tmp_path, shape, kernels, readings, order, rules=None, listed=None, pm="warmup"
):
    """`run_analysis.evaluate` of one GEMM of `shape` ((m, n, k, transA))
    with the rows `readings` ({solution: latency}) of `kernels`, the pool,
    under a model serving with `rules` (MLREC_v2's when None) whose engine
    scores the solutions `order` in that order; `listed` is the GEMM's
    planned list."""
    m, n, k, ta = shape
    by_sid = {kk["sol_idx_global"]: kk for kk in kernels}
    rows = [
        row(m, n, k, sid, us, transA=ta, **{f: by_sid[sid][f] for f in ra.ev.CONFIG_FIELDS})
        for sid, us in readings.items()
    ]  # fmt: skip
    d = _bench_dir(tmp_path / "s4", rows)
    plans = None if listed is None else [{(m, n, k, 1): frozenset(listed)}]
    bench = ra.load_bench([d], plans=plans)
    pool = ra.Pool(kernels, "library")
    cell = grid.cell_key(m, n, k, 1)
    model = ra.ModelInfo(
        labels=[cell],
        whitelists={cell: []},
        hardware=ra.DeviceHardware(**HARDWARE),
        data=b"model",
        rules=rules,
    )
    tw = FakeEngine(cell, {(m, n, k, 1): [pool.pos[s] for s in order]})
    return ra.evaluate(bench, pool, model, tw=tw, pick_measurement=pm)


V3_RULES = feasibility.soft_rules({}, 3)
SHAPE = (8, 1024, 256, "T")
DOT2 = dict(mi_m=1, mi_n=1, mi_k=64)


def test_a_dot2_pick_of_a_v3_bundle_is_charged_its_own_reading(tmp_path):
    """A bundle whose flags leave the Dot2 and small-batched rules off
    serves Dot2 kernels for M > 2: the analysis admits them and charges the
    pick its own reading. An MLREC_v2 bundle keeps every rule; a pick its
    rules leave out is inconsistent and is never charged more than its own
    reading, the levels before it capped there."""
    kernels = [_kernel(1, **DOT2), _kernel(2), _kernel(3, mt_m=128)]
    readings = {1: 2.0, 2: 3.0, 3: 4.0}
    e = _served(tmp_path / "v3", SHAPE, kernels, readings, [1, 2, 3], V3_RULES)
    assert e.levels[0].tolist() == [2.0] * 7 and e.n_inconsistent == 0
    assert ra.decomposition(e)["*"]["sel_eff"] == 1.0
    e = _served(tmp_path / "v2", SHAPE, kernels, readings, [1, 2, 3])
    assert e.levels[0].tolist() == [2.0] * 7 and e.n_inconsistent == 1
    # The same bundle consistent with its rules: the levels as before.
    e = _served(tmp_path / "v2c", SHAPE, kernels, readings, [2, 3])
    assert e.levels[0].tolist() == [2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    d = ra.decomposition(e)["*"]
    assert d["sel_eff"] == pytest.approx(2.0 / 3.0)
    assert d["steps"]["hard_rules"] == pytest.approx(2.0 / 3.0)
    assert e.n_inconsistent == 0 and d["sel_eff_range"] is None


def test_an_unrunnable_pick_falls_to_the_next_ranked_kernel(tmp_path):
    """The engine of an MLREC_v3 bundle can rank first a kernel the problem
    fails the leading-free rule of (M < grvw_a, A not transposed): no bench
    has a row of it, and the runtime serves the next-ranked kernel that
    runs."""
    shape = (4, 1024, 256, "N")
    kernels = [_kernel(1), _kernel(2, grvw_a=1), _kernel(3, grvw_a=2, mt_n=128)]
    readings = {2: 3.0, 3: 2.0}
    e = _served(tmp_path, shape, kernels, readings, [1, 2, 3], V3_RULES, [2, 3])
    assert e.pool.sids[e.pick_pos[0]] == 2 and e.miss == [None]
    assert e.levels[0][-1] == 3.0
    d = ra.decomposition(e)["*"]
    assert (d["n_miss"], d["n_unmeasured"], d["n_top_unrunnable"]) == (0, 0, 1)
    assert d["sel_eff"] == pytest.approx(2.0 / 3.0) and d["sel_eff_range"] is None


def test_a_pick_with_a_row_ran_whatever_the_pool_says(tmp_path):
    """A bench row means the kernel ran, even where the pool's vector widths
    fail it on the leading-free rule (the pool disagrees with the build that
    was benched): such a pick is served at its own reading, and when no
    scored kernel the rule passes has a reading, the scored level is the
    cheapest scored kernel with one."""
    shape = (4, 1024, 256, "N")
    kernels = [_kernel(1), _kernel(2, grvw_a=1, mt_n=128)]
    e = _served(tmp_path, shape, kernels, {1: 3.0, 2: 2.0}, [1], V3_RULES)
    prob = fs.problem_kwargs_from_row(e.gemms[0].row)
    assert ra.leading_free_mask(e.pool, prob).tolist() == [False, True]
    assert e.pool.sids[e.pick_pos[0]] == 1 and e.served.tolist() == [True]
    assert e.levels[0].tolist() == [2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0]
    d = ra.decomposition(e)["*"]
    assert (d["n_miss"], d["n_unmeasured"], d["n_top_unrunnable"]) == (0, 0, 0)
    assert d["steps"]["cascade"] == pytest.approx(2.0 / 3.0)
    assert d["sel_eff"] == pytest.approx(2.0 / 3.0)


@pytest.mark.parametrize("pm", ["warmup", "timed"])
def test_a_runnable_unmeasured_pick_is_a_range(tmp_path, pm):
    """A pick off the GEMM's planned list that passes the leading-free rule
    can run but was never measured: it is a miss whose level holds the
    next-ranked measured kernel, its efficiency bounded by the slowest
    reading of the GEMM and the winner's; never the slowest reading alone."""
    shape = (4, 1024, 256, "N")
    kernels = [
        _kernel(1),
        _kernel(2, grvw_a=2, mt_m=128),
        _kernel(3, grvw_a=1),
        _kernel(4, grvw_a=1, mt_n=128),
        _kernel(5, grvw_a=1, mt_k=128),
    ]
    readings = {3: 3.0, 4: 2.0, 5: 9.0}
    e = _served(tmp_path, shape, kernels, readings, [1, 2, 3], V3_RULES, [3, 4, 5], pm)
    assert e.pool.sids[e.pick_pos[0]] == 2 and e.miss == ["not_benched"]
    assert e.unmeasured.tolist() == [True] and e.levels[0][-1] == 3.0
    d = ra.decomposition(e)["*"]
    assert (d["n_miss"], d["n_unmeasured"], d["n_top_unrunnable"]) == (1, 1, 1)
    assert d["sel_eff"] == pytest.approx(2.0 / 3.0)
    assert d["sel_eff_range"] == pytest.approx([2.0 / 9.0, 1.0])
    assert d["deployed_sel_eff_range"] == pytest.approx([2.0 / 9.0, 1.0])
    # "any" passes over kernels off the list, as the stages do.
    e = _served(
        tmp_path / "any",
        shape,
        kernels,
        readings,
        [1, 2, 3],
        V3_RULES,
        [3, 4, 5],
        "any",
    )
    assert e.pool.sids[e.pick_pos[0]] == 3 and e.miss == [None]  # fmt: skip
    assert not e.unmeasured.any()
    # Without a later measured kernel the estimate is the slowest reading.
    e = _served(
        tmp_path / "last", shape, kernels, readings, [1, 2], V3_RULES, [3, 4, 5], pm
    )
    assert e.levels[0][-1] == 9.0  # fmt: skip


def test_the_decomposition_shows_unmeasured_picks_as_a_range():
    dec = {
        "*": {
            "n_gemms": 10,
            "n_served": 10,
            "n_top_unrunnable": 3,
            "n_miss": 2,
            "sel_eff": 0.98,
            "sel_eff_range": [0.9, 0.99],
            "deployed_sel_eff": 0.98,
            "steps": dict.fromkeys(ra.STEPS, 1.0),
        }
    }
    html = rs.decomposition_html(
        {"decomposition": dec, "engine": True, "pick_measurement": "warmup"}
    )
    assert "<b>98.00</b>" in html
    assert "90.00&ndash;99.00" in html
    assert "unrunnable kernel ranked above the pick" in html
    assert "slowest latency its GEMM measured" in html
    assert "is a <b>miss</b> at the slowest" not in html
    note = mr.bench_note(
        {
            "bench_unmeasured": 1234,
            "bench_range": [80.5, 99.25],
            "bench_top_unrunnable": 2345,
            "bench_miss": 1234,
        }
    )
    assert note.startswith("1,234 picks can run but were never benched")
    assert "80.50% if each ran as slow" in note and "2,345 GEMMs" in note
    assert mr.bench_note({}) == "every pick has a reading of its own"


def test_the_unrunnable_count_names_any_kernel_ranked_above_the_pick(tmp_path):
    """`n_top_unrunnable` counts the GEMMs with any kernel that cannot run
    ranked above the served pick, the top-ranked one or another: here the
    top-ranked kernel passes the leading-free rule (on the list, the bench
    did not run it) and the next one fails it. The loss decomposition's
    column and caption and the in-sample tile's note say exactly that."""
    shape = (4, 1024, 256, "N")
    kernels = [
        _kernel(1, grvw_a=1),
        _kernel(2),
        _kernel(3, grvw_a=2, mt_n=128),
        _kernel(4, grvw_a=1, mt_m=128),
    ]
    readings, listed = {3: 3.0, 4: 2.0}, [1, 3, 4]
    e = _served(tmp_path, shape, kernels, readings, [1, 2, 3], V3_RULES, listed)
    prob = fs.problem_kwargs_from_row(e.gemms[0].row)
    assert ra.leading_free_mask(e.pool, prob).tolist() == [True, False, True, True]
    assert e.pool.sids[e.pick_pos[0]] == 3 and e.n_unrunnable.tolist() == [1]
    dec = ra.decomposition(e)
    assert dec["*"]["n_top_unrunnable"] == 1
    html = rs.decomposition_html(
        {"decomposition": dec, "engine": True, "pick_measurement": "warmup"}
    )
    label = "unrunnable kernel ranked above the pick"
    head = html[html.index("<thead>") : html.index("</thead>")]
    assert re.findall(r"<th[^>]*>([^<]*)</th>", head)[:5] == [
        "cell",
        "GEMMs",
        "served",
        label,
        "misses",
    ]
    assert (
        f"(<b>{label}</b> counts the GEMMs where at least one such kernel ranks "
        "above the served pick)" in html
    )
    assert "top pick cannot run" not in html and "top-ranked kernel it is" not in html
    assert mr.bench_note({"bench_top_unrunnable": 2345}) == (
        "2,345 GEMMs have at least one kernel ranked above the served pick that "
        "cannot run (the leading-free rule), which the runtime passes over"
    )


def test_the_stages_charge_an_unmeasured_pick_their_own_way(tmp_path):
    """The deployed number counts a pick that can run but was never measured
    at its estimate, so it is not the stages' there: without a re-bench,
    stages 4.5, 6 and 8 charge an off-list pick the slowest latency its
    GEMM measured (the low end of the range), and stage05 the winner's
    latency over train.off_list_sel_eff, or the slowest when that gives no
    efficiency. They also charge a kernel off the list that cannot run as
    their pick, which the runtime passes over. Stage 7 costs the runtime's
    top-1 without the planned list and leaves the problem out of its
    selection efficiency when that kernel has no row, whichever of the two
    it is. The caption names each stage's charge."""
    shape = (4, 1024, 256, "N")
    kernels = [
        _kernel(1),
        _kernel(2, grvw_a=2, mt_m=128),
        _kernel(3, grvw_a=1),
        _kernel(4, grvw_a=1, mt_n=128),
        _kernel(5, grvw_a=1, mt_k=128),
    ]
    readings, listed = {3: 3.0, 4: 2.0, 5: 9.0}, [3, 4, 5]
    e = _served(tmp_path, shape, kernels, readings, [1, 2, 3], V3_RULES, listed)
    assert e.pool.sids[e.pick_pos[0]] == 2 and e.n_unrunnable.tolist() == [1]
    dec = ra.decomposition(e)
    d = dec["*"]
    assert d["deployed_sel_eff"] == pytest.approx(2.0 / 3.0)
    key = (4, 1024, 256, 1, "N", "N")
    gemm = {
        "candidates": [{"sol_idx_global": s, "us": u} for s, u in readings.items()],
        "winner_us": 2.0,
        "listed": listed,
    }
    # auto with no re-benched misses and no fallback gives no efficiency
    no_efficiency, _ = miss_rebench.off_list_charge(None, min_misses=30)
    for sid in (1, 2):
        stages = ev.pick_cost(sid, readings, {}, listed, "warmup")
        assert stages == (True, 9.0, "not_benched")
        stage05 = ev.pick_cost(
            sid, readings, {}, listed, "warmup", off_list_sel_eff=0.8
        )
        assert stage05 == (True, pytest.approx(2.5), "not_benched")
        charge = ev.off_list_charge_for(no_efficiency, gemm)
        assert charge is None
        assert ev.pick_cost(sid, readings, {}, listed, "warmup", None, charge) == (
            stages
        )
    assert d["deployed_sel_eff_range"][0] == pytest.approx(2.0 / 9.0)
    # Stage 7's top-1 at pool position 0 cannot run, at 1 was never measured
    # (both without a row); at 3 it has the winner's reading.
    problems = [dict(zip(("m", "n", "k", "batch_count", "transA", "transB"), key))]
    pool = [{"sol_idx_global": k["sol_idx_global"]} for k in kernels]
    for pm in ("warmup", "timed", "any"):
        for top1, evaluated in ((0, 0), (1, 0), (3, 1)):
            cpp = [{"key": list(key), "top1_index": top1, "cell": "c"}]
            sel = s7.runtime_selection(problems, cpp, pool, {key: gemm}, pm)
            assert (sel["n_evaluated"], sel["n_top1_without_row"]) == (
                evaluated,
                1 - evaluated,
            )
            assert sel["n_miss"] == 0
            if evaluated:
                assert sel["sel_eff"] == pytest.approx(1.0)
    html = rs.decomposition_html(
        {"decomposition": dec, "engine": True, "pick_measurement": "warmup"}
    )
    assert (
        "the slowest latency its GEMM measured (what stages 4.5, 6 and 8 charge it "
        "unless a re-bench measured it; stage05 charges the winner's latency over "
        "<code>train.off_list_sel_eff</code>, the slowest when that gives no "
        "efficiency; stage 7 leaves the GEMM out of its selection efficiency) and "
        "at the winner's." in html
    )
    doc = " ".join(ra.decomposition.__doc__.split())
    assert "6, 7 and 8" not in html and "6, 7 and 8" not in doc
    assert "(what the stages charge it)" not in html
    assert "the number the stages report" not in doc


def test_kernels_per_cell_count_picks_with_the_winners_signature(tmp_path):
    """The per-cell counts are Python ints, and a numpy number formats like
    one: the column "pick has the winner's signature" showed a dash in every
    row while it held numpy integers."""
    kernels = [_kernel(1), _kernel(2, grvw_a=4), _kernel(3, mt_m=128)]
    readings = {1: 2.0, 2: 3.0, 3: 4.0}
    e = _served(tmp_path, SHAPE, kernels, readings, [2, 1, 3], V3_RULES)
    (usage,) = ra.cell_kernel_usage(e).values()
    assert (usage["pick_has_winner_signature"], usage["pick_is_winner"]) == (1, 0)
    assert type(usage["pick_has_winner_signature"]) is int
    html = rs.cell_usage_html({"cell_usage": ra.cell_kernel_usage(e)})
    row = html.split("<tbody>")[1]
    assert re.findall(r"<td>([^<]*)</td>", row)[:4] == ["1", "1", "1", "0"]
    assert rs._ok(np.int64(3)) and rs.intf(np.int64(1234)) == "1,234"
    assert not rs._ok(np.bool_(True)) and not rs._ok(True)
    assert not rs._ok(np.float64("nan"))


def test_context_gemms_take_their_lists_from_their_own_replan(tmp_path):
    """A context's shape is benched from the context's re-plan when stage02
    re-planned it (stage03c): its off-list check uses that list, not the
    round's exclusive one."""
    rd = tmp_path / "run" / "round_1"
    m, n, k = 8, 1024, 256
    entry = {"m": m, "n": n, "k": k, "batch": 1}
    (rd / "stage02" / "contexts" / "cu128_x").mkdir(parents=True)
    (rd / "stage02" / "plan.json").write_text(
        json.dumps({"shapes": [dict(entry, kernels=[1, 2])]})
    )
    (rd / "stage02" / "contexts" / "cu128_x" / "plan.json").write_text(
        json.dumps(
            {
                "context": {"cu_budget": 128, "schedule": "default"},
                "shapes": [dict(entry, kernels=[2, 3])],
            }
        )
    )
    kernels = [_kernel(1), _kernel(2, mt_n=128), _kernel(3, mt_m=128)]
    ctx = dict(cu_budget="128", schedule="default")
    cfg = {kk["sol_idx_global"]: kk for kk in kernels}
    rows = [
        row(m, n, k, sid, us, **{f: cfg[sid][f] for f in ra.ev.CONFIG_FIELDS}, **c)
        for sid, us, c in ((1, 2.0, {}), (2, 3.0, {}), (2, 3.0, ctx), (3, 4.0, ctx))
    ]  # fmt: skip
    d = _bench_dir(rd / "stage04", rows, contexts=True)
    replans = ra.round_context_plans(rd)
    assert replans == {(128, "default"): {(m, n, k, 1): frozenset({2, 3})}}
    sets = ra.model_benches(tmp_path / "run", None)
    assert sets["training"].context_plans == [replans]
    bench = ra.load_bench([d], plans=[ra.round_plan(rd)], context_plans=[replans])
    listed = {g.context.cu_budget: g.listed for g in bench.gemms}
    assert listed == {0: frozenset({1, 2}), 128: frozenset({2, 3})}
    # The engine ranks kernel 1 first: on the context's own list it is a
    # kernel the context's bench never listed, so it may run unmeasured.
    pool = ra.Pool(kernels, "library")
    cell = grid.cell_key(m, n, k, 1)
    model = ra.ModelInfo(
        labels=[cell],
        whitelists={cell: []},
        hardware=ra.DeviceHardware(**HARDWARE),
        data=b"model",
        rules=V3_RULES,
    )
    tw = FakeEngine(cell, {(m, n, k, 1): [0, 1, 2]})
    e = ra.evaluate(bench, pool, model, tw=tw)
    by_ctx = {g.context.cu_budget: i for i, g in enumerate(e.gemms)}
    assert e.miss[by_ctx[0]] is None and e.miss[by_ctx[128]] == "not_benched"
    old = ra.load_bench([d], plans=[ra.round_plan(rd)])
    e = ra.evaluate(old, pool, model, tw=tw)
    assert not any(e.miss)


def test_weak_cells_and_contexts_section():
    settings = _settings(target=0.95, min_eval_gemms=10)

    def leaf(se, n, miss=0):
        return {
            "n_gemms": n,
            "n_served": n,
            "n_miss": miss,
            "sel_eff": se,
            "deployed_sel_eff": se,
            "dominant": "ranking",
        }

    dec = {
        "*": leaf(0.9, 100),
        "A": leaf(0.80, 20, 3),
        "B": leaf(0.88, 60),
        "C": leaf(0.50, 5),
        "D": leaf(0.97, 15),
    }
    weak = rs.weak_cells(dec, settings)
    assert [r["cell"] for r in weak] == ["B", "A"]
    assert weak[0]["gain"] == pytest.approx(0.6 * 0.07)
    assert weak[1]["n_miss"] == 3

    analysis = {
        "contexts": {"exclusive": {"*": leaf(0.96, 50)}},
        "held_out": {"contexts": {"cu128": dec}},
        "holdout_contexts": {
            "contexts": {"cu64_dynamic": {"*": leaf(0.7, 30), "E": leaf(0.7, 30)}}
        },
    }
    assert not rs.has_contexts({"contexts": {"exclusive": {}}}, [])
    assert rs.has_contexts(analysis, [])
    s4b = [
        {
            "round": "round_1",
            "kind": "held-out context",
            "context": "cu64_dynamic",
            "n": 30,
            "n_miss": 2,
            "sel_eff": 0.7,
        }
    ]
    audit = {
        "exclusive": {
            "sel_eff": 0.912,
            "n": 40,
            "protocol": "sustained",
            "round": "final_retrain",
            "group": "exclusive",
        }
    }
    html = rs.contexts_html(
        analysis, s4b, settings, {"cu128": "small partition"}, audit
    )
    assert "cu128 (small partition)" in html
    for title in (
        "held-out GEMMs, trained context",
        "held-out context",
        "Held-out measurement per round",
        "Re-ranking of the GEMMs the bundle never trained on",
        "In-sample re-ranking (training benches)",
    ):
        assert title in html, title
    # The in-sample table stands apart, with the audit's held-out number of
    # each context beside it.
    insample = html[html.index("In-sample re-ranking (training benches)") :]
    insample = insample[: insample.index("</table>")]
    assert "held-out audit %" in insample and "<b>91.20</b>" in insample
    assert "final_retrain, group exclusive, protocol sustained" in insample
    assert "cu128" not in insample and "Shipped bundle per context" not in html
    bare = rs.contexts_html(analysis, s4b, settings)
    assert "not measured on the audit set" in bare
    assert (
        "cu128 (small partition), held-out GEMMs, trained context: 2 weak cells" in html
    )
    assert "cu64_dynamic, held-out context: 1 weak cells" in html
    assert "origami" not in html.lower()


# ── evidence and actions ────────────────────────────────────────────────────


def _settings(**kw):
    s = {
        "target": 0.95,
        "min_cell_gemms": 30,
        "min_eval_gemms": 10,
        "split_after": 2,
        "max_split_depth": 6,
    }
    s.update(kw)
    return s


def _ev(**kw):
    base = dict(
        cell=LARGE,
        sel_eff=0.9,
        source="measured",
        n_eval=40,
        in_sample=0.91,
        n_gemms=200,
        share=0.5,
        has_model=True,
    )
    base.update(kw)
    return rs.HeldOut(**base)


def _steps(**kw):
    s = dict.fromkeys(ra.STEPS, 1.0)
    s.update(kw)
    return {"steps": s}


def _audit(cls, **kw):
    return dict(kw, **{"class": cls})


@pytest.mark.parametrize(
    "ev,action",
    [
        (_ev(sel_eff=0.96, decomp=_steps(ranking=0.9)), "none"),
        (_ev(has_model=False, n_gemms=10, sel_eff=None), "more data"),
        (_ev(has_model=False, n_gemms=80, sel_eff=0.7), "retrain"),
        (_ev(sel_eff=None, source=None), rs.NO_EVIDENCE),
        (_ev(n_eval=5, decomp=_steps(ranking=0.9)), "more data"),
        (_ev(decomp=_steps(whitelist=0.95, ranking=0.97)), "tier policy: whitelist"),
        (
            _ev(decomp=_steps(soft_rules=0.9, ranking=0.97)),
            "tier policy: non-temporal rule",
        ),
        (_ev(decomp=_steps(library=0.9, ranking=0.97)), "kernel pool"),
        (_ev(decomp=_steps(hard_rules=0.5, ranking=0.97)), "retrain"),
        (_ev(retrain_attempts=2, decomp=_steps(ranking=0.9)), "split"),
        (_ev(retrain_attempts=2, n_gemms=40, decomp=_steps(ranking=0.9)), "retrain"),
        # In-sample numbers and the training-bench decomposition are context.
        (_ev(in_sample=0.99, decomp=_steps(ranking=0.9)), "retrain"),
        (_ev(sel_eff=0.97, in_sample=0.5), "none"),
        (
            _ev(train_decomp=_steps(whitelist=0.5), decomp=_steps(ranking=0.9)),
            "retrain",
        ),
        # The audit set's class decides first.
        (
            _ev(source="audit", audit=_audit("pool_limited", ceiling_rules=0.92)),
            "kernel pool",
        ),
        (
            _ev(
                source="audit",
                audit=_audit("pool_limited", ceiling_rules=0.92, ceiling_table=0.97),
            ),
            "tier policy: serving rules",
        ),
        (
            _ev(
                source="audit",
                audit=_audit(
                    "identity_limited", ceiling_rules=0.97, ceiling_served=0.93
                ),
            ),
            "kernel identity",
        ),
        (
            _ev(
                source="audit",
                n_eval=4,
                audit=_audit("model_limited"),
                decomp=_steps(soft_rules=0.8, cascade=0.97),
            ),
            "tier policy: cascade",
        ),
        (_ev(source="audit", sel_eff=0.96, audit=_audit("passing")), "none"),
    ],
)
def test_recommended_actions(ev, action):
    got, why = rs.recommend(ev, _settings())
    assert got == action, why
    assert "origami" not in why.lower() and "in-sample" not in why


def test_only_a_cell_that_can_split_is_told_to_split():
    """Only a Large M or N or a LargeK axis splits (lib.subcells), and not
    at validate.max_split_depth: another cell below the target after its
    retrains is retrained, saying why, and with split_after_attempts 0 the
    reason does not count retrain attempts that never happened."""
    ranking = _steps(ranking=0.9)
    for cell in ("Mid|Mid|TinyK|Bany", "Mid|Mid|MidK|Bany", "Small|Mid|MidK|Bany"):
        got, why = rs.recommend(
            _ev(cell=cell, retrain_attempts=2, decomp=ranking), _settings()
        )
        assert got == "retrain" and "its base cell has none" in why, cell
    deep = LARGE + "#M>4096#N>4096"
    got, why = rs.recommend(
        _ev(cell=deep, retrain_attempts=2, decomp=ranking), _settings(max_split_depth=2)
    )
    assert got == "retrain" and "at validate.max_split_depth 2" in why
    got, why = rs.recommend(_ev(decomp=ranking), _settings(split_after=0))
    assert got == "split" and "without a retrain first" in why
    assert "0 retrain attempt" not in why
    got, why = rs.recommend(
        _ev(retrain_attempts=2, decomp=ranking), _settings(split_after=0)
    )
    assert got == "split" and "2 retrain attempt(s) did not reach" in why
    got, why = rs.recommend(_ev(retrain_attempts=3, decomp=ranking), _settings())
    assert got == "split" and "3 retrain attempt(s) did not reach" in why


def test_held_out_evidence_takes_the_audit_then_the_measurements():
    leaves = {
        LARGE + "#M<=4096": {"n_gemms": 30, "in_sample": 0.99},
        LARGE + "#M>4096": {"n_gemms": 10},
        TINY: {"n_gemms": 60},
    }
    held = {
        LARGE: {"sel_eff_new": 0.8, "n_eval": 25, "sources": ["stage04b round_2"]},
        LARGE + "#M>4096": {"sel_eff_new": 0.9, "n_eval": 12, "sources": ["x"]},
    }
    audit = {TINY: {"sel_eff": 0.7, "n": 16, "class": "model_limited"}}
    ev = rs.heldout_evidence(leaves, held, audit_cells=audit)
    lo, hi, tiny = ev[LARGE + "#M<=4096"], ev[LARGE + "#M>4096"], ev[TINY]
    assert (lo.source, lo.sel_eff, lo.n_eval) == ("measured", 0.8, 25)
    assert lo.detail == f"stage04b round_2 (of {LARGE})" and lo.in_sample == 0.99
    assert (hi.sel_eff, hi.detail) == (0.9, "x")
    assert (tiny.source, tiny.sel_eff, tiny.n_eval) == ("audit", 0.7, 16)
    assert lo.share == pytest.approx(0.3)


def test_improve_rows_rank_by_expected_gain():
    evidence = {
        "a": _ev(cell="a", sel_eff=0.80, share=0.05),
        "b": _ev(cell="b", sel_eff=0.90, share=0.60),
        "c": _ev(cell="c", sel_eff=0.99, share=0.30),
        "d": _ev(
            cell="d", sel_eff=None, source=None, share=0.05, has_model=False, n_gemms=3
        ),
    }
    rows = rs.improve_rows(evidence, _settings())
    assert [r["cell"] for r in rows] == ["b", "a", "d", "c"]
    assert rows[0]["gain"] == pytest.approx(0.6 * 0.05)
    html = rs.improve_html(rows, _settings(), [("--weakest 1", ["b"])])
    assert "--weakest 1" in html and "origami" not in html.lower()
    assert "3 measured" in html


def test_run_settings_read_absolute_thresholds():
    cfg = (
        "validate:\n  sel_eff_threshold: 0.9\n  min_cell_gemms: 12\n"
        "  split_after_attempts: 3\ntrain:\n  min_cell_gemms: 40\n"
    )
    assert rs.run_settings(cfg) == {
        "pick_measurement": "warmup",
        "target": 0.9,
        "min_cell_gemms": 40,
        "min_eval_gemms": 12,
        "split_after": 3,
        "max_split_depth": 6,
    }
    relative = rs.run_settings("validate:\n  sel_eff_threshold: origami\n")
    assert relative["target"] == ce.DEFAULT_TARGET
    anyrow = rs.run_settings("validate:\n  pick_measurement: any\n")
    assert anyrow["pick_measurement"] == "any"


def test_evidence_prefers_newer_held_out_rows(tmp_path):
    run = tmp_path / "run"
    s5 = run / "round_0" / "stage05"
    s5.mkdir(parents=True)
    cells = [
        {
            "label": c,
            "n_gemms": 40,
            "deployed": {"sel_eff": 0.97, "n_eval": 30},
            "deployed_val": {"sel_eff": 0.93, "n_eval": 10},
        }
        for c in (TINY, LARGE)
    ]
    (s5 / "cells.json").write_text(json.dumps({"cells": cells}))
    s4b = run / "round_1" / "stage04b"
    s4b.mkdir(parents=True)
    rows = [
        {"cell": LARGE, "sel_eff_new": 0.8, "n_eval": 25, "decision": "retrain"},
        {"cell": "Mid|Mid|MidK|Bnone", "n_training_gemms": 12, "decision": "no_model"},
    ]
    (s4b / "decisions.json").write_text(json.dumps({"per_cell": rows}))
    ev = ce.collect(run)
    assert (ev[TINY].source, ev[TINY].sel_eff) == ("validation", 0.93)
    assert (ev[LARGE].source, ev[LARGE].sel_eff) == ("held_out", 0.8)
    assert ev[LARGE].retrain_attempts == 1
    assert ev["Mid|Mid|MidK|Bnone"].sel_eff is None
    only0 = ce.collect(run, rounds=[run / "round_0"])
    assert only0[LARGE].source == "validation"
    counted = ce.collect(run, gemm_counts={TINY: 60, "Small|Small|MidK|Bnone": 20})
    assert counted[TINY].n_gemms == 60 and "Small|Small|MidK|Bnone" in counted
    assert sum(e.share for e in counted.values()) == pytest.approx(1.0)


# ── stage 7 per cell and the template ───────────────────────────────────────


def _log(path, calls):
    lines = list(bf.BANNER)
    for g, t in calls:
        lines.append(f"Solution selection time: {t} us")
        lines.append("Is supported 1 / Total solutions: 9")
        lines += bf.measured_lines(0, g, 7, 10.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def test_selection_time_per_cell(tmp_path):
    s7 = tmp_path / "stage07"
    warm, g1, g2 = bf.Gemm(9, 9, 9), bf.Gemm(64, 64, 64), bf.Gemm(4096, 4096, 4096)
    for rep, (a, b) in enumerate(((3.0, 2.0), (5.0, 4.0), (4.0, 3.0))):
        _log(
            s7 / "w" / "timing" / "rsn1" / f"rep{rep}_ml_on.log",
            [(warm, 100.0), (g1, a), (g1, a - 1.0), (g2, 20.0)],
        )
        _log(
            s7 / "w" / "timing" / "rsn1" / f"rep{rep}_ml_off.log",
            [(warm, 100.0), (g1, b), (g1, b - 1.0), (g2, 2.0)],
        )
    parity = {
        "per_problem": [
            dict(m=64, n=64, k=64, batch=1, transA="T", transB="N", cpp_cell="X|served")
        ]
    }
    (s7 / "w" / "parity").mkdir(parents=True)
    (s7 / "w" / "parity" / "parity.json").write_text(json.dumps(parity))
    summary = {"per_yaml": {"w": {}, "../escape": {}}}
    (t,) = rs.stage07_per_cell(s7, summary, labels=[LARGE])
    assert (t["yaml"], t["rsn"], t["reps"], t["routed"]) == ("w", "1", 3, 1)
    served = {k: sorted(v) for k, v in t["cells"]["X|served"].items()}
    assert served == {
        "on": [3.0, 4.0],
        "off": [2.0, 3.0],
        "d": [1.0, 1.0],
        "first": [4.0],
        "warm": [3.0],
    }
    assert t["cells"][LARGE]["on"] == t["cells"][LARGE]["first"] == [20.0]
    assert t["cells"][LARGE]["warm"] == []
    html = rs.selection_time_html([t], 10.0)
    assert "1 over budget" in html
    assert "ML-off" not in html
    assert "ML-off" in rs.selection_time_html([t], 10.0, origami_reference=True)


def test_selection_time_per_cell_of_a_run_that_resumed_after_a_bench_death(tmp_path):
    """A timing log that resumed after a bench death holds several bench
    processes: the first call of each, which loads the library, stays out
    of the pairing, and each process's first call into a cell is a first
    one."""
    s7 = tmp_path / "stage07"
    g = [bf.Gemm(m, 64, 64) for m in (1, 2, 3, 4, 5)]

    def problem(gemm, us):
        return [
            f"Solution selection time: {us} us",
            "Is supported 1 / Total solutions: 1",
        ] + bf.measured_lines(0, gemm, 7, 10.0)

    on = list(bf.BANNER) + problem(g[0], 100.0) + problem(g[1], 3.0)
    on += ["Solution selection time: 4.0 us", "Is supported 1 / Total solutions: 1"]
    on += ["*** LEFT OUT M=3 N=64 K=64 batch=1 transA=T transB=N (rc=1) ***"]
    on += list(bf.BANNER) + problem(g[3], 90.0) + problem(g[4], 5.0)
    off = list(bf.BANNER) + [line for gm in g for line in problem(gm, 2.0)]
    rsn1 = s7 / "w" / "timing" / "rsn1"
    rsn1.mkdir(parents=True)
    (rsn1 / "rep0_ml_on.log").write_text("\n".join(on) + "\n")
    (rsn1 / "rep0_ml_off.log").write_text("\n".join(off) + "\n")
    parity = {
        "per_problem": [
            dict(m=m, n=64, k=64, batch=1, transA="T", transB="N", cpp_cell=cell)
            for m, cell in zip((1, 2, 3, 4, 5), ("X|a", "X|b", "X|a", "X|a", "X|b"))
        ]
    }
    (s7 / "w" / "parity").mkdir(parents=True)
    (s7 / "w" / "parity" / "parity.json").write_text(json.dumps(parity))
    (t,) = rs.stage07_per_cell(s7, {"per_yaml": {"w": {}}}, labels=[LARGE])
    assert {c: {k: sorted(v) for k, v in d.items()} for c, d in t["cells"].items()} == {
        "X|b": {
            "on": [3.0, 5.0],
            "off": [2.0, 2.0],
            "d": [1.0, 3.0],
            "first": [3.0, 5.0],
            "warm": [],
        }
    }
    assert "0 over budget" in rs.selection_time_html([t], 10.0)


def test_template_slots_fill_in_one_pass():
    page = mr.fill("<p>__A__|__B__|__C__</p>", {"A": "__B__", "B": "x"})
    assert page == "<p>__B__|x|__C__</p>"


# ── with the engine: a trained run end to end ───────────────────────────────


def _planned_bench(out_dir, shapes):
    """Bench chunks where a third of the kernels of every shape have no row,
    and a plan that benched every other shape from a list of the kernels
    that have one (the rest exhaustively)."""
    out_dir.mkdir(parents=True)
    rows, entries = [], []
    for j, shape in enumerate(shapes):
        kept = [
            r
            for r in gemm_rows(shape, kernel_pool())
            if r["is_winner"] == "True" or (int(r["sol_idx_global"]) + j) % 3
        ]
        rows += kept
        m, n, k, b = shape
        listed = sorted(int(r["sol_idx_global"]) for r in kept) if j % 2 == 0 else None
        entries.append({"m": m, "n": n, "k": k, "batch": b, "kernels": listed})
    with (out_dir / "chunk_0000.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out_dir, {"shapes": entries}


@pytest.mark.skipif(not have_engine(), reason="needs the tilewright module")
def test_trained_run_analysis_and_report(tmp_path):
    run = tmp_path / "run"
    data = write_round(run / "round_0" / "stage04", shapes_for(0, 24, CELLS3))
    s5 = run / "round_0" / "stage05"
    train(tmp_path, [data], s5, "--weight-dtype", "int4")
    a = ra.analyze_run(run, jobs=1)
    assert a["engine"] and a["n_inconsistent"] == 0
    assert a["n_served"] == a["n_gemms"] > 0
    assert a["pick_measurement"] == "warmup"

    model = ra.final_model(run)
    gemms = [
        g
        for gs in s5mod.load_enriched_chunks([str(data)], True)[0].values()
        for g in gs
    ]
    for pm, analysis in (
        ("warmup", a),
        ("timed", ra.analyze_run(run, jobs=1, pick_measurement="timed")),
        ("any", ra.analyze_run(run, jobs=1, pick_measurement="any")),
    ):
        out = ev.evaluate_model(
            model.data,
            gemms,
            hardware=model.hardware,
            tw=ev.tilewright_module(),
            pick_measurement=pm,
        )
        served = [r for r in out.results if r.served_by == "model"]
        d = analysis["decomposition"]["*"]
        assert d["sel_eff"] == pytest.approx(
            ev.geomean(r.sel_eff for r in served), rel=1e-9
        )
        assert d["n_miss"] == sum(1 for r in served if r.miss)

    planned, plan = _planned_bench(tmp_path / "planned", shapes_for(7, 12, CELLS3))
    pgemms = [
        g
        for gs in s5mod.load_enriched_chunks([str(planned)], True)[0].values()
        for g in gs
    ]
    assert ev.attach_plan(pgemms, plan) > 0
    lookup = {
        k: (frozenset(v["kernels"]) if v["kernels"] is not None else None)
        for k, v in planner.plan_lookup(plan).items()
    }
    bench = ra.load_bench([planned], plans=[lookup])
    pool = ra.pool_for(bench, None)
    reasons = {}
    first_with_a_row = {
        (r.m, r.n, r.k, r.batch): r
        for r in ev.evaluate_model(
            model.data,
            pgemms,
            hardware=model.hardware,
            tw=ev.tilewright_module(),
            pick_measurement="any",
        ).results
    }
    for pm in ev.PICK_MEASUREMENTS:
        e = ra.evaluate(
            bench, pool, model, tw=ev.tilewright_module(), pick_measurement=pm
        )
        out = ev.evaluate_model(
            model.data,
            pgemms,
            hardware=model.hardware,
            tw=ev.tilewright_module(),
            pick_measurement=pm,
        )
        ref = {(r.m, r.n, r.k, r.batch): r for r in out.results}
        for i, g in enumerate(e.gemms):
            r = ref[tuple(g.key[:4])]
            assert e.served[i] == (r.served_by == "model")
            if not e.served[i]:
                continue
            assert e.miss[i] == r.miss
            if not e.unmeasured[i]:
                assert e.levels[i, -1] == pytest.approx(r.pick_us, rel=1e-12)
                continue
            # A pick off its list the stages charge the slowest latency: the
            # analysis keeps that as its worst case and counts it at the
            # next-ranked kernel with a reading.
            assert r.miss == "not_benched"
            assert e.worst_us[i] == pytest.approx(r.pick_us, rel=1e-12)
            if pm == "warmup":
                nxt = first_with_a_row[tuple(g.key[:4])].pick_us
                assert e.levels[i, -1] == pytest.approx(
                    max(g.winner_us, nxt), rel=1e-12
                )
        reasons[pm] = ra.decomposition(e)["*"]["n_miss_by_reason"]
    assert reasons["timed"].get("not_benched", 0) > 0
    assert reasons["timed"].get("screened", 0) > 0
    assert reasons["warmup"] == {"not_benched": reasons["timed"]["not_benched"]}
    assert not reasons["any"]

    # stage05 scores its in-sample cells with the measurement it was given
    pm5 = json.loads((s5 / "metrics.json").read_text())["args"]["pick_measurement"]
    assert pm5 == "warmup"
    a5 = ra.analyze_run(run, jobs=1, pick_measurement=pm5)
    cells = {
        c["label"]: c for c in json.loads((s5 / "cells.json").read_text())["cells"]
    }
    checked = 0
    for label, c in cells.items():
        d = a5["decomposition"][label]
        parts = [c["deployed"], c.get("deployed_val") or {}]
        n = sum(int(p.get("n_eval") or 0) for p in parts)
        if n != d["n_served"]:
            continue
        logs = sum(
            int(p["n_eval"]) * math.log(p["sel_eff"]) for p in parts if p.get("n_eval")
        )
        assert d["sel_eff"] == pytest.approx(math.exp(logs / n), rel=1e-6)
        prod = math.prod(v for v in d["steps"].values())
        assert prod == pytest.approx(d["sel_eff"], rel=1e-9)
        checked += 1
    assert checked == len(cells)

    rows = a["kernels"]["kernels"]
    assert sum(r["wins"] for r in rows) == a["n_gemms"]
    assert sum(r["picks"] for r in rows) == a["n_served"]
    assert a["pruning"]["never"]
    for r in a["pruning"]["never"]:
        assert r["wins"] == 0 and r["picks"] == 0
    for name, vals in a["parameters"]["params"].items():
        assert sum(v["wins"] for v in vals) == a["n_gemms"], name
        assert sum(v["picks"] for v in vals) == a["n_served"], name

    out = tmp_path / "report.html"
    mr.build(str(run), str(out))
    page = out.read_text()
    for title in (
        "Where to improve",
        "Loss decomposition",
        "Kernel usage",
        "Pool-drop candidates",
    ):
        assert title in page, title
    assert "origami" not in page.lower()
    assert page.count("</script>") == 1
    assert "pick measurement <code>warmup</code>" in page
    assert "Diagnostics" not in page and '"DPARAMS": {}' in page

    mr.build(str(run), str(out), diagnostics=True)
    page = out.read_text()
    assert "Diagnostics: framework-specific kernel parameters" in page
    assert '"GRVW (A,B)"' in page


def test_kernel_tables_name_kernels_by_signature_not_solution_index():
    def kernel(sig, members, wins=0, picks=0, tested=0, screened=0):
        return {
            "sig": ra.signature_label(sig),
            "signature": list(sig),
            "n_members": len(members),
            "members": [{"sid": s, "wins": w, "picks": p} for s, w, p in members],
            "wins": wins,
            "picks": picks,
            "tested": tested,
            "screened": screened,
            "near": [0, 0, 0],
            "best_ratio": 1.5,
            "drop_loss": 0.0,
            "whitelisted_cells": 0,
            "win_cells": [],
            "pick_cells": [],
        }

    shared = kernel(
        (256, 256, 64, 16, 16, 32, 0, 4),
        [(7771, 3, 2), (7772, 1, 0)],
        wins=4,
        picks=2,
        tested=5,
    )
    alone = kernel((64, 64, 32, 16, 16, 32, 0, 0), [(7773, 0, 0)], tested=2)
    lonely = kernel((32, 32, 16, 16, 16, 16, 4, 0), [(7774, 0, 0), (7775, 0, 0)])
    analysis = {
        "engine": True,
        "kernels": {
            "kernels": [shared, alone, lonely],
            "n_gemms": 5,
            "n_served": 5,
            "n_winner_outside_pool": 0,
        },
        "pool": {"signatures": 3, "size": 5, "source": "library"},
        "pruning": ra.pruning_candidates({"kernels": [shared, alone, lonely]}),
    }
    usage, pruning = rs.kernels_html(analysis), rs.pruning_html(analysis)
    page = usage + pruning
    fields = "MT_M MT_N DepthU MI_M MI_N MI_K NTA NTB".split()
    for table in (usage, pruning):
        heads = re.findall(r"<th[^>]*>([^<]*)</th>", table)
        assert heads[: len(fields) + 1] == ["kernel (signature)", *fields]
        for row in re.findall(r"<tr>(.*?)</tr>", table.split("<tbody>")[1]):
            assert len(re.findall(r"<td\b", row)) == len(heads)
    numbers = "".join(f"<td>{v}</td>" for v in shared["signature"])
    assert f'<td class="c">MT256x256x64 MI16x16x32 NTA0 NTB4</td>{numbers}' in usage
    numbers = "".join(f"<td>{v}</td>" for v in alone["signature"])
    assert f'<td class="c">MT64x64x32 MI16x16x32 NTA0 NTB0</td>{numbers}' in pruning
    assert "<details><summary>2</summary>" in page
    assert "wins/picks per member: 3/2, 1/0" in page
    assert "MT32x32x16 MI16x16x16 NTA4 NTB0 (2 members)" in page
    assert not re.search(r"777[1-5]", page)


def _put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def test_external_audit_results_merge_into_the_rounds(tmp_path):
    run, audit = tmp_path / "run", tmp_path / "audit"
    for r, sha in ((0, "a" * 64), (1, "b" * 64)):
        _put(run / f"round_{r}" / "stage06" / "deploy_record.json", {"sha256": sha})
    old = {"status": "incomplete", "model_sha256": "b" * 64, "groups": {}}
    _put(run / "round_1" / "stage06" / "audit.json", old)
    group = {"cell_mean": 0.9, "classes": {"pool_limited": 1}}
    v1 = {
        "status": "ok",
        "groups": {"exclusive": group},
        "audit_dir": str(audit),
        "pick_measurement": "warmup",
    }
    staged = tmp_path / "offline" / "staged_for_deploy" / "w" / "m.tilewright.bin"
    elsewhere = tmp_path / "elsewhere" / "models.pt"
    ext = [
        dict(v1, what="round_0", model_sha256="a" * 64),
        dict(v1, what="round_1", model_sha256="b" * 64, v2={"status": "ok"}),
        dict(v1, what="m.tilewright.bin", model_sha256="c" * 64, model=str(staged)),
        dict(
            v1,
            what="round_2",
            model_sha256="d" * 64,
            model=str(run / "round_2" / "stage05" / "models.pt"),
        ),
        dict(v1, what="round_3", model_sha256="e" * 64, model=str(elsewhere)),
    ]
    paths = []
    for i, rec in enumerate(ext):
        paths.append(tmp_path / f"r{i}.json")
        _put(paths[-1], {"models": [rec]} if i % 2 else rec)
    recs, notes = rs.read_audit_results(paths + [tmp_path / "absent.json"])
    assert len(recs) == 5 and len(notes) == 1 and "absent.json" in notes[0]
    got = rs.audit_records(run, recs, audit_dir=audit, pick_measurement="warmup")
    assert [(rnd, role, r["model_sha256"][0]) for rnd, role, r in got] == [
        ("round_0", "candidate", "a"),
        ("round_1", "candidate", "b"),
        ("round_2", "candidate", "d"),
        ("extra: offline", "extra", "c"),
        ("extra: elsewhere/models.pt", "extra", "e"),
    ]
    assert got[1][2]["source"] == str(paths[1]) and "v2" in got[1][2]
    assert {r["audit_set"] for _, role, r in got if role == "extra"} == {"audit"}
    assert {r["extra_reason"] for _, role, r in got if role == "extra"} == {
        "no round of this run"
    }
    assert rs.audit_records(run)[0][2]["status"] == "incomplete"


def test_only_records_of_the_runs_audit_set_and_pick_rule_stand_for_a_round(
    tmp_path,
):
    run, audit, other = tmp_path / "run", tmp_path / "audit", tmp_path / "other"
    _put(run / "round_0" / "stage06" / "deploy_record.json", {"sha256": "a" * 64})
    own = {"status": "incomplete", "model_sha256": "a" * 64, "audit_dir": str(audit)}
    _put(run / "round_0" / "stage06" / "audit.json", own)
    audit.mkdir()
    (tmp_path / "linked").symlink_to(audit)
    model = run / "round_0" / "stage06" / "staged_for_deploy" / "w" / "m.bin"
    done = {
        "status": "ok",
        "what": "round_0",
        "model": str(model),
        "model_sha256": "a" * 64,
        "groups": {"exclusive": {"cell_mean": 0.9}},
        "pick_measurement": "warmup",
    }
    ext = [
        dict(done, audit_dir=str(other)),
        dict(done, audit_dir=str(audit), pick_measurement="timed"),
        dict(done, audit_dir=str(tmp_path / "linked")),
    ]
    got = rs.audit_records(run, ext, audit_dir=audit, pick_measurement="warmup")
    assert [(rnd, role, r["status"]) for rnd, role, r in got] == [
        ("round_0", "candidate", "ok"),
        ("extra: round_0/stage06", "extra", "ok"),
        ("extra: round_0/stage06", "extra", "ok"),
    ]
    assert got[0][2]["audit_dir"] == str(tmp_path / "linked")
    assert [(r["audit_set"], r["extra_reason"]) for _, _, r in got[1:]] == [
        ("other", "round_0's bundle, evaluated on audit set other, not audit"),
        ("audit", "round_0's bundle, picks charged as timed, not warmup"),
    ]
    alone = rs.audit_records(run, ext[2:], audit_dir=None, pick_measurement="warmup")
    assert alone[0][2]["status"] == "incomplete"
    assert alone[1][2]["extra_reason"] == (
        "round_0's bundle, the run configures no audit set (audit.dir)"
    )
    changes = rs.audit_changes(got)
    assert [(c["after"], c["groups"]["exclusive"]["reason"]) for c in changes] == [
        ("extra: round_0/stage06", "evaluated on audit set other, not audit"),
        ("extra: round_0/stage06", "picks charged as timed, not warmup"),
    ]


def test_extra_models_show_once_per_model_file_and_audit_set(tmp_path):
    run, audit, other = tmp_path / "run", tmp_path / "audit", tmp_path / "other"
    staged = tmp_path / "final" / "staged_for_deploy" / "w" / "m.tilewright.bin"
    base = {
        "what": staged.name,
        "model": str(staged),
        "model_sha256": "c" * 64,
        "audit_dir": str(audit),
        "pick_measurement": "warmup",
        "groups": {"exclusive": {"cell_mean": 0.9}},
    }
    v1 = dict(base, status="ok")
    v2 = dict(v1, v2={"status": "ok"})
    failed = {"status": "error", "reason": "x", "model_sha256": "c" * 64}
    ext = [
        v1,
        dict(failed, audit_dir=str(audit)),
        dict(v2, pick_measurement="timed"),
        v2,
        dict(v1, audit_dir=str(other)),
        {"status": "error", "reason": "absent", "model": str(staged)},
        dict(failed, audit_dir=str(audit)),
    ]
    got = rs.audit_records(run, ext, audit_dir=audit, pick_measurement="warmup")
    assert [(rnd, r["audit_set"], r["status"]) for rnd, _, r in got] == [
        ("extra: final", "audit", "ok"),
        ("extra: final", "other", "ok"),
        ("extra: final", "not recorded", "error"),
    ]
    assert got[0][2]["v2"] == {"status": "ok"}
    assert got[0][2]["pick_measurement"] == "warmup"


def test_the_audit_caption_states_the_width_of_the_v2_sets():
    def rec(settings):
        return ("round_0", "candidate", {"v2": {"status": "ok", "settings": settings}})

    plain = rec({"set_tolerance": 0.1, "set_max": 20})
    assert rs.v2_set_text([plain]) == "within 10% of that pass's winner"
    floor = rec({"set_tolerance": 0.12, "set_tolerance_us": 2.5, "set_max": 24})
    assert rs.v2_set_text([floor, floor]) == (
        "within 12% of that pass's winner or 2.5 us of it, whichever is wider"
    )
    named = "within the set tolerance of that pass's winner"
    assert rs.v2_set_text([plain, floor]) == named
    assert rs.v2_set_text([("round_0", "candidate", {"status": "ok"})]) == named
    assert rs.v2_set_text([]) == named


def test_an_entering_model_that_is_a_candidates_is_not_repeated(tmp_path):
    run = tmp_path / "run"
    done = {"status": "ok", "groups": {"exclusive": {"cell_mean": 0.9}}}
    rows = {
        ("round_0", "stage06"): {"status": "incomplete", "model_sha256": "a" * 64},
        ("round_1", "stage04b"): dict(done, model_sha256="a" * 64),
        ("round_1", "stage06"): dict(done, model_sha256="b" * 64),
        ("round_2", "stage04b"): dict(done, model_sha256="b" * 64),
        ("round_2", "stage05"): dict(done, model_sha256="c" * 64),
        ("round_4", "stage04b"): dict(done, model_sha256="d" * 64),
    }
    for (rnd, stage), rec in rows.items():
        _put(run / rnd / stage / "audit.json", rec)
    got = rs.audit_records(run)
    assert [(rnd, role, r["model_sha256"][0]) for rnd, role, r in got] == [
        ("round_0", "candidate", "a"),
        ("round_1", "candidate", "b"),
        ("round_2", "candidate", "c"),
        ("round_4", "entering", "d"),
    ]
    assert [r["source"] for _, _, r in got] == [
        str(run / rnd / stage / "audit.json")
        for rnd, stage in (
            ("round_1", "stage04b"),
            ("round_1", "stage06"),
            ("round_2", "stage05"),
            ("round_4", "stage04b"),
        )
    ]
    assert got[0][2]["status"] == "ok"
    staged = run / "round_9" / "staged_for_deploy" / "m.bin"
    extra = dict(done, model_sha256="a" * 64, model=str(staged))
    assert rs.audit_records(run, [extra])[:4] == got


def test_class_counts_fill_or_mark_the_identity_class():
    old = {"classes": {"pool_limited": 1, "model_limited": 2, "passing": 1}}
    counts = rs.class_counts(old, 0.95)
    assert counts == {
        "pool_limited": 1,
        "identity_limited": None,
        "model_limited": 2,
        "passing": 1,
    }
    assert rs.classes_text(counts) == "1 / not recorded / 2 / 1"
    cells = {
        "a": {"sel_eff": 0.9, "ceiling_rules": 0.99, "ceiling_served": 0.93},
        "b": {"sel_eff": 0.9, "ceiling_rules": 0.99, "ceiling_served": 0.98},
        "c": {"sel_eff": 0.96, "ceiling_rules": 0.99, "ceiling_served": 0.98},
        "d": {"sel_eff": 0.9, "ceiling_rules": 0.94, "ceiling_served": 0.93},
    }
    served = {"classes": {"pool_limited": 1, "model_limited": 2, "passing": 1}}
    assert rs.class_counts(dict(served, per_cell=cells), 0.95) == {
        "pool_limited": 1,
        "identity_limited": 1,
        "model_limited": 1,
        "passing": 1,
    }


# The bench analysis of an MLREC_v2 bundle on the seeded bench below, frozen
# from the analysis as it was before it read a bundle's own rules (every v2
# rule, the same levels): per pick measurement the selection efficiency, the
# deployed one, the step factors and the column sums of the levels.
GOLDEN_V2 = {
    "warmup": (
        0.6610792175617249,
        0.6577140866048069,
        {
            "library": 1.0,
            "hard_rules": 0.9174322361433133,
            "soft_rules": 0.8275055656723502,
            "whitelist": 0.8773399230553047,
            "cascade": 1.0,
            "ranking": 0.9925232585152992,
        },
        [812.14012, 812.14012, 873.35258, 900.495, 1000.8637, 1000.8637, 1011.165],
    ),
    "timed": (
        0.6259760692790383,
        0.6263393244019984,
        {
            "library": 1.0,
            "hard_rules": 0.9111908518406994,
            "soft_rules": 0.8132494529666849,
            "whitelist": 0.8760455743279784,
            "cascade": 1.0,
            "ranking": 0.9642683229155088,
        },
        [
            812.14012,
            812.14012,
            880.75212,
            923.51762,
            1024.75832,
            1024.75832,
            1069.90996,
        ],
    ),
}
GOLDEN_V2["any"] = GOLDEN_V2["warmup"]


def _golden_v2_bench(tmp_path):
    """A seeded bench of an MLREC_v2 bundle (every v2 rule) whose engine
    ranks only the kernels of tier 1 under those rules, as the analysis
    models it, with every kernel that runs measured (no plan)."""
    kernels = [
        _kernel(1),
        _kernel(2, mt_m=128, mt_n=128, mi_m=32, mi_n=32, mi_k=16),
        _kernel(3, mt_m=256, mt_n=128, mi_m=32, mi_n=32, mi_k=16, cache_hints_b=4),
        _kernel(4, mt_m=32, mt_n=256, cache_hints_a=4),
        _kernel(5, mt_m=8, mt_k=512, **DOT2),
        _kernel(6, mt_m=16, mt_n=512, grvw_a=2),
        _kernel(7, mt_m=128, mt_k=128, mi_m=32, mi_n=32, mi_k=16, grvw_a=4),
        _kernel(8, mt_n=256, mt_k=32, cache_hints_a=1),
        _kernel(9, mt_m=256, mt_n=256, mi_m=32, mi_n=32, mi_k=16, grvw_b=4),
        _kernel(10, mt_m=32, mt_n=32, grvw_a=1, grvw_b=1),
    ]
    rng = random.Random(11)
    shapes, seen = [], set()
    while len(shapes) < 48:
        m = rng.choice([1, 2, 3, 6, 24, 100, 256, 700, 2048])
        n = rng.choice([2, 48, 256, 640, 4096])
        k = rng.choice([32, 256, 512, 1024, 4096])
        b = rng.choice([1, 1, 4])
        ta, tb = rng.choice([("T", "N"), ("N", "N"), ("N", "T")])
        if (m, n, k, b) not in seen:
            seen.add((m, n, k, b))
            shapes.append((m, n, k, b, ta, tb))
    cells = sorted({grid.cell_key(m, n, k, b) for m, n, k, b, _, _ in shapes})
    sigs = [ra.signature_of(kk) for kk in kernels]
    whitelists = {c: sigs[i % 2 :: 2] + sigs[:1] for i, c in enumerate(cells)}
    nt_a, nt_b = feasibility.non_temporal_availability(kernels)
    families = feasibility.non_temporal_families(kernels)
    v2 = ra.serving_rules(None)
    rows, order = [], {}
    for m, n, k, b, ta, tb in shapes:
        prob = fs.problem_kwargs_from_row(
            dict(row(m, n, k, 0, 0.0), batch_count=b, transA=ta, transB=tb)
        )

        def passes(kk, **rules):
            return feasibility.passes_gates(
                prob,
                fs.config_kwargs_from_row(kk),
                nt_a_available=nt_a,
                nt_b_available=nt_b,
                nt_families=families,
                **rules,
            )

        lat = {kk["sol_idx_global"]: round(rng.uniform(5.0, 50.0), 4) for kk in kernels}
        admitted = [kk for kk in kernels if passes(kk, lds_bytes=65536, **v2)]
        wl = set(whitelists[grid.cell_key(m, n, k, b)])
        tier = [kk for kk in admitted if ra.signature_of(kk) in wl] or admitted
        ranked = sorted(
            tier, key=lambda kk: lat[kk["sol_idx_global"]] * rng.uniform(0.8, 1.3)
        )
        order[(m, n, k, b)] = [kernels.index(kk) for kk in ranked]
        lf_only = dict(
            non_temporal_rule=False, small_batched_rule=False, dot2_rule=False
        )
        runnable = [kk for kk in kernels if passes(kk, lds_bytes=1 << 40, **lf_only)]
        for j, kk in enumerate(runnable):
            skip = rng.random() < 0.15
            us = lat[kk["sol_idx_global"]] * (1.4 if skip else 1.0)
            cfg = {f: kk[f] for f in ra.ev.CONFIG_FIELDS}
            rows.append(
                row(m, n, k, kk["sol_idx_global"], us, skip, batch_count=b, transA=ta,
                    transB=tb, is_origami_pick=int(j == 0), **cfg)
            )  # fmt: skip
    bench = ra.load_bench([_bench_dir(tmp_path / "s4", rows)])
    model = ra.ModelInfo(
        labels=cells,
        whitelists=whitelists,
        hardware=ra.DeviceHardware(**HARDWARE),
        data=b"model",
    )
    return bench, ra.Pool(kernels, "library"), model, FakeEngine(cells, order)


def test_mlrec_v2_numbers_stay_where_its_rules_apply(tmp_path):
    """Golden: on a bench where its engine ranks what its rules admit and
    every pick has a reading, an MLREC_v2 bundle's analysis gives the
    numbers it gave before the analysis read a bundle's own rules."""
    bench, pool, model, tw = _golden_v2_bench(tmp_path)
    for pm, (sel, dep, steps, sums) in GOLDEN_V2.items():
        e = ra.evaluate(bench, pool, model, tw=tw, pick_measurement=pm)
        d = ra.decomposition(e)["*"]
        assert (d["n_gemms"], d["n_served"], d["blocked"]) == (48, 43, {"soft": 5})
        assert e.n_inconsistent == 0
        assert d["sel_eff"] == pytest.approx(sel, rel=1e-12)
        assert d["deployed_sel_eff"] == pytest.approx(dep, rel=1e-12)
        assert d["steps"] == pytest.approx(steps, rel=1e-12)
        assert np.nansum(e.levels, axis=0).tolist() == pytest.approx(sums, rel=1e-6)
        screened = {"screened": 6} if pm == "timed" else {}
        assert d["n_miss_by_reason"] == screened
