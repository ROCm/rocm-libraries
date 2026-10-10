# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bench analysis (lib/run_analysis.py), cell evidence and the report
sections built on them, on synthetic runs."""

import csv
import json
import math
import random
import re

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
                for rule, family in ((True, False), (False, False), (True, True)):
                    hard, soft = ra.gate_masks(
                        pool,
                        prob,
                        lds_bytes=HARDWARE["lds_bytes"],
                        non_temporal_rule=rule,
                        non_temporal_family_rule=family,
                    )
                    want = [
                        feasibility.passes_gates(
                            prob,
                            c,
                            lds_bytes=HARDWARE["lds_bytes"],
                            nt_a_available=nt_a,
                            nt_b_available=nt_b,
                            non_temporal_rule=rule,
                            non_temporal_family_rule=family,
                            nt_families=families,
                        )
                        for c in cfgs
                    ]
                    assert soft.tolist() == want, (m, n, k, b, ta, tb, dt, rule, family)
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
