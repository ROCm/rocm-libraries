# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bench analysis (lib/run_analysis.py), cell evidence and the report
sections built on them, on synthetic runs."""

import csv
import json
import math
import random

import numpy as np
import pytest

import bench_fakes as bf
import make_report as mr
import report_sections as rs
import stage05_train as s5mod
from lib import cell_evidence as ce
from lib import evaluate as ev
from lib import feasibility
from lib import features as fs
from lib import planner
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
        1: 3.5,
        2: 3.0,
        3: 4.0,
    }
    assert small.winner_us == 3.5
    assert by_m[2048].winner_us == 8.0
    assert bench.stats["bad_rows"] == 2
    assert bench.stats["gemms"] == 2
    kept = ra.load_bench([a], keep=lambda p: int(p["m"]) > 100)
    assert [g.key[0] for g in kept.gemms] == [2048]


def test_gate_masks_match_the_feasibility_rules():
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
    pool = ra.Pool(kernels, "test")
    nt_a, nt_b = feasibility.non_temporal_availability(kernels)
    cfgs = [fs.config_kwargs_from_row({**k}) for k in kernels]
    shapes = [(1, 4096, 1024, 1), (2, 64, 64, 4), (64, 64, 512, 8)]
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
                for rule in (True, False):
                    hard, soft = ra.gate_masks(
                        pool,
                        prob,
                        lds_bytes=HARDWARE["lds_bytes"],
                        non_temporal_rule=rule,
                    )
                    want = [
                        feasibility.passes_gates(
                            prob,
                            c,
                            lds_bytes=HARDWARE["lds_bytes"],
                            nt_a_available=nt_a,
                            nt_b_available=nt_b,
                            non_temporal_rule=rule,
                        )
                        for c in cfgs
                    ]
                    assert soft.tolist() == want, (m, n, k, b, ta, tb, dt, rule)
                    assert not (soft & ~hard).any()


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
    shared = by_sig["MT64x64 DU64 MI16x16x32 CH0,0"]
    assert (shared["n_members"], shared["wins"], shared["tested"]) == (2, 2, 3)
    assert {m["sid"]: m["wins"] for m in shared["members"]} == {1: 1, 2: 1}
    assert by_sig["MT128x64 DU64 MI16x16x32 CH0,0"]["wins"] == 1
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
    assert '<td class="c">64x64</td><td>64</td>' in html
    assert "<details><summary>2</summary>" in html and "1 (1), 2 (1)" in html
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
    html = rs.contexts_html(analysis, s4b, settings, {"cu128": "small partition"})
    assert "cu128 (small partition)" in html
    for title in (
        "held-out GEMMs, trained context",
        "held-out context",
        "training GEMMs (in-sample)",
        "Held-out measurement per round",
    ):
        assert title in html, title
    assert (
        "cu128 (small partition), held-out GEMMs, trained context: 2 weak cells" in html
    )
    assert "cu64_dynamic, held-out context: 1 weak cells" in html
    assert "origami" not in html.lower()


# ── evidence and actions ────────────────────────────────────────────────────


def _settings(**kw):
    s = {"target": 0.95, "min_cell_gemms": 30, "min_eval_gemms": 10, "split_after": 2}
    s.update(kw)
    return s


def _ev(**kw):
    base = dict(
        cell=LARGE,
        sel_eff=0.9,
        source="held_out",
        n_eval=40,
        in_sample=0.91,
        n_gemms=200,
        share=0.5,
        has_model=True,
    )
    base.update(kw)
    return ce.CellEvidence(**base)


def _steps(**kw):
    s = dict.fromkeys(ra.STEPS, 1.0)
    s.update(kw)
    return {"steps": s}


@pytest.mark.parametrize(
    "ev,decomp,action",
    [
        (_ev(sel_eff=0.96), _steps(ranking=0.9), "none"),
        (_ev(has_model=False, n_gemms=10, sel_eff=None), None, "more data"),
        (_ev(has_model=False, n_gemms=80, sel_eff=0.7), None, "retrain"),
        (_ev(sel_eff=None), None, "more data"),
        (_ev(n_eval=5), _steps(ranking=0.9), "more data"),
        (_ev(), _steps(whitelist=0.95, ranking=0.97), "tier policy: whitelist"),
        (_ev(), _steps(soft_rules=0.9, ranking=0.97), "tier policy: non-temporal rule"),
        (_ev(), _steps(hard_rules=0.5, ranking=0.97), "retrain"),
        (_ev(in_sample=0.99), _steps(ranking=0.9), "more data"),
        (_ev(retrain_attempts=2), _steps(ranking=0.9), "split"),
        (_ev(retrain_attempts=2, n_gemms=40), _steps(ranking=0.9), "retrain"),
    ],
)
def test_recommended_actions(ev, decomp, action):
    got, why = rs.recommend(ev, decomp, _settings())
    assert got == action, why
    assert "origami" not in why.lower()


def test_improve_rows_rank_by_expected_gain():
    evidence = {
        "a": _ev(cell="a", sel_eff=0.80, share=0.05),
        "b": _ev(cell="b", sel_eff=0.90, share=0.60),
        "c": _ev(cell="c", sel_eff=0.99, share=0.30),
        "d": _ev(cell="d", sel_eff=None, share=0.05, has_model=False, n_gemms=3),
    }
    rows = rs.improve_rows(evidence, {}, _settings())
    assert [r["cell"] for r in rows] == ["b", "a", "d", "c"]
    assert rows[0]["gain"] == pytest.approx(0.6 * 0.05)
    html = rs.improve_html(rows, _settings(), [("--weakest 1", ["b"])])
    assert "--weakest 1" in html and "origami" not in html.lower()


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
            if e.served[i]:
                assert e.levels[i, -1] == pytest.approx(r.pick_us, rel=1e-12)
                assert e.miss[i] == r.miss
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
