# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Weak spots: the classes of a bundle's audit cells with their levers, the
roofline of the true best of every exhaustively benched GEMM, and the pool
gaps exported for kernel generation (lib/weak_spots.py)."""

import csv
import json
import math

import pytest

from lib import hardware as hw
from lib import weak_spots as ws
from lib.grid import cell_key

TYPES = {"a": "bf16_r", "b": "bf16_r", "c": "bf16_r", "d": "bf16_r", "compute": "f32_r"}


def kernel(sid, mt_m=64, mt_n=64, du=32, nta=0, ntb=0):
    return {
        "sol_idx_global": sid,
        "mt_m": mt_m,
        "mt_n": mt_n,
        "mt_k": du,
        "mi_m": 16,
        "mi_n": 16,
        "mi_k": 32,
        "cache_hints_a": nta,
        "cache_hints_b": ntb,
        "occupancy": 1,
        "grvw_a": 8,
        "grvw_b": 8,
        "gwvw_d": 4,
    }


def gemm(m, n, k, batch, cands, **ctx):
    return {
        "m": m,
        "n": n,
        "k": k,
        "batch_count": batch,
        "transA": "T",
        "transB": "N",
        **{f"{t}_type": v for t, v in TYPES.items()},
        "candidates": [
            {"sol_idx_global": sid, "us": us, "is_skip": skip}
            for sid, us, skip in cands
        ],
        **ctx,
    }


def row(m, n, k, batch, best_us, **more):
    return {
        "m": m,
        "n": n,
        "k": k,
        "batch": batch,
        "cu_budget": 0,
        "schedule": "default",
        "types": dict(TYPES),
        "transA": "T",
        "transB": "N",
        "best_us": best_us,
        "kernel": {f: v for f, v in kernel(0).items() if f in ws.SIGNATURE},
        **more,
    }


PEAKS = hw.RooflineRates(100.0, "--peak-tflops", 1000.0, "--peak-bandwidth-gbs")
# the compute time of the cube and the memory time of the skinny GEMM below
# at the PEAKS rates (A, B and D of bf16)
BIG_COMPUTE_US = 2 * 4096**3 / 100e6
SKINNY_MEMORY_US = (4096 * 4096 + 4096 + 4096) * 2 / 1e6


def rows_abc():
    return [
        row(4096, 4096, 4096, 1, 2 * BIG_COMPUTE_US),
        row(4096, 1, 4096, 1, 4 * SKINNY_MEMORY_US),
        row(16, 16, 16, 1, 4.0),
        row(4096, 4096, 4096, 1, 1.0, cu_budget=64),
    ]


def test_best_kernels_take_the_fastest_tested_kernel_of_the_pool():
    pool = [kernel(1, 256, 16), kernel(2, 64, 64, ntb=4), kernel(3, 128, 128)]
    gemms = [
        gemm(3000, 5, 2, 1024, [(1, 9.0, False), (2, 5.0, True), (3, 7.0, False)]),
        gemm(64, 64, 64, 1, [(9, 1.0, False), (2, 3.0, False)]),
        gemm(32, 32, 32, 1, [(2, 1.0, True)]),
        gemm(128, 128, 128, 1, [(3, 2.0, False)], cu_budget=64, schedule="default"),
    ]
    out = ws.best_kernels(gemms, pool)
    assert [(r["m"], r["best_us"]) for r in out] == [(3000, 7.0), (64, 3.0), (128, 2.0)]
    assert out[0]["kernel"] == {f: kernel(3, 128, 128)[f] for f in ws.SIGNATURE}
    assert out[1]["kernel"]["cache_hints_b"] == 4
    assert out[0]["batch"] == 1024
    assert out[0]["types"]["a"] == "bf16_r" and out[0]["transA"] == "T"
    assert (out[2]["cu_budget"], ws.is_exclusive(out[2])) == (64, False)
    assert ws.signature_label(out[0]["kernel"]) == "MT128x128x32 MI16x16x32 NTA0 NTB0"


def test_the_roofline_takes_the_largest_term_of_each_gemm():
    roof = ws.roofline(rows_abc(), PEAKS)
    peaks = roof["peaks"]
    assert (peaks["tflops"], peaks["bandwidth_gbs"]) == (100.0, 1000.0)
    assert peaks["tflops_source"] == "--peak-tflops"
    assert peaks["latency_floor_us"] == 4.0
    big, skinny, tiny = roof["rows"]
    assert len(roof["rows"]) == 3
    assert big["compute_us"] == pytest.approx(BIG_COMPUTE_US)
    assert big["memory_us"] == pytest.approx(3 * 4096 * 4096 * 2 / 1e6)
    assert (big["bound"], big["fraction"]) == ("compute", pytest.approx(0.5))
    assert skinny["memory_us"] == pytest.approx(SKINNY_MEMORY_US)
    assert (skinny["bound"], skinny["fraction"]) == ("memory", pytest.approx(0.25))
    assert (tiny["bound"], tiny["fraction"], tiny["attainable_us"]) == (
        "latency",
        1.0,
        4.0,
    )
    assert big["cell"] == cell_key(4096, 4096, 4096, 1)
    assert big["best_signature"] == "MT64x64x32 MI16x16x32 NTA0 NTB0"
    assert ws.roofline([rows_abc()[3]], PEAKS) == {"peaks": None, "rows": []}
    # the larger of its terms is the benches' roofline
    from lib.bench_yaml import roofline_us

    hl = {f"{t}_type": v for t, v in TYPES.items()}
    for r in (big, skinny):
        shape = (r["m"], r["n"], r["k"], r["batch"])
        want = roofline_us(shape, hl, 100.0, 1000.0)
        assert max(r["compute_us"], r["memory_us"]) == pytest.approx(want)


def test_unknown_or_lower_peaks_are_the_best_any_gemm_reached():
    unknown = hw.RooflineRates(None, "unknown: no device", None, "unknown: none")
    roof = ws.roofline(rows_abc()[:3], unknown)
    peaks = roof["peaks"]
    assert peaks["tflops"] == pytest.approx(50.0)
    assert peaks["tflops_source"] == "the best any GEMM reached (unknown: no device)"
    assert peaks["bandwidth_gbs"] == pytest.approx(250.0)
    assert roof["rows"][0]["fraction"] == pytest.approx(1.0)
    low = hw.RooflineRates(10.0, "--peak-tflops", 1000.0, "--peak-bandwidth-gbs")
    peaks = ws.roofline(rows_abc()[:3], low)["peaks"]
    assert peaks["tflops"] == pytest.approx(50.0)
    assert peaks["tflops_source"] == (
        "the best any GEMM reached, above the rate of --peak-tflops"
    )
    assert peaks["bandwidth_gbs"] == 1000.0


def test_pool_gaps_are_the_gemms_farthest_below_their_roofline():
    rows = ws.roofline(rows_abc()[:3], PEAKS)["rows"]
    gaps = ws.pool_gaps(rows, 0.3)
    assert [(g["m"], g["n"]) for g in gaps] == [(4096, 1)]
    (g,) = gaps
    assert g["fraction"] == pytest.approx(0.25) and g["bound"] == "memory"
    assert g["best_signature"] == "MT64x64x32 MI16x16x32 NTA0 NTB0"
    assert {f: g[f] for f in ws.SIGNATURE} == rows[1]["kernel"]
    assert [(g["m"], g["n"]) for g in ws.pool_gaps(rows, 0.6)] == [
        (4096, 1),
        (4096, 4096),
    ]
    assert ws.pool_gaps(rows, 0.2) == []


def test_cell_rooflines_are_geomeans_with_their_gaps_and_bounds():
    rows = [row(4096, 4096, 4096, 1, 1.0), row(4097, 4096, 4096, 1, 1.0)]
    for r, frac, bound in zip(rows, (0.2, 0.8), ("memory", "compute")):
        r.update(fraction=frac, bound=bound, cell=cell_key(4096, 4096, 4096, 1))
    cells = ws.cell_rooflines(rows, 0.3)
    (c,) = cells.values()
    assert c["fraction"] == pytest.approx(0.4)
    assert (c["n"], c["n_gaps"]) == (2, 1)
    assert c["bounds"] == {"compute": 1, "memory": 1}


A = cell_key(64, 64, 64, 1)
B = cell_key(4096, 4096, 4096, 1)
C = cell_key(3000, 5, 2, 1024)


def group(cells):
    """An evaluated audit group of {cell: (class, [per-GEMM efficiency])}."""
    shapes = {A: (64, 64, 64, 1), B: (4096, 4096, 4096, 1), C: (3000, 5, 2, 1024)}
    per_gemm, per_cell = [], {}
    for cell, (cls, vals) in cells.items():
        m, n, k, b = shapes[cell]
        per_gemm += [[m + i, n, k, b, v] for i, v in enumerate(vals)]
        per_cell[cell] = {
            "n": len(vals),
            "sel_eff": math.exp(sum(map(math.log, vals)) / len(vals)),
            "sel_eff_ci": [min(vals), max(vals)],
            "class": cls,
            "ceiling_rules": 0.9 if cls == "pool_limited" else 0.99,
            "ceiling_served": 0.97,
        }
    return {"per_cell": per_cell, "per_gemm": per_gemm}


def record(excl, ctx=None, v2=None):
    groups = {"exclusive": group(excl)}
    if ctx is not None:
        groups["cu64"] = group(ctx)
    rec = {"status": "ok", "what": "bundle", "groups": groups}
    if v2 is not None:
        rec["v2"] = {"status": "ok", "groups": {"exclusive": group(v2)}}
    return rec


def test_each_cell_gets_its_classes_and_levers():
    rec = record(
        {
            A: ("pool_limited", [0.9] * 4),
            B: ("model_limited", [0.9, 0.9, 0.4, 0.8]),
            C: ("passing", [0.99] * 4),
        },
        ctx={A: ("identity_limited", [0.9] * 4)},
    )
    roof = {C: {"fraction": 0.2, "n": 4, "n_gaps": 3}, B: {"fraction": 0.7}}
    rows = ws.weak_cells(rec, roof_cells=roof, threshold=0.3)
    by = {(r["group"], r["cell"]): r for r in rows}
    assert by[("exclusive", A)]["class"] == "rule"
    assert by[("exclusive", B)]["class"] == "model"
    assert by[("exclusive", C)]["class"] == "pool"
    assert by[("exclusive", C)]["roofline"] == 0.2
    assert by[("exclusive", C)]["n_gaps"] == 3
    assert by[("exclusive", C)]["levers"] == [ws.LEVERS["pool"]]
    assert by[("cu64", A)]["class"] == "identity"
    assert by[("cu64", A)]["roofline"] is None
    assert (by[("exclusive", B)]["n_tail"], by[("exclusive", B)]["n_below_floor"]) == (
        1,
        2,
    )
    assert all(r["weak"] for r in rows)
    assert by[("exclusive", A)]["protocol"] == "v1"
    both = ws.classes_of("model_limited", {"fraction": 0.1}, 0.3)
    assert both == ["model", "pool"]
    assert ws.classes_of("passing", {"fraction": 0.5}, 0.3) == []
    assert ws.classes_of(None, None, 0.3) == []


def test_cells_pair_with_the_previous_bundle_under_v2_where_it_has_the_group():
    before = record({A: ("passing", [0.99] * 8)}, v2={A: ("passing", [0.99] * 8)})
    after = record({A: ("passing", [0.99] * 8)}, v2={A: ("passing", [0.95] * 8)})
    (r,) = ws.weak_cells(after, before)
    assert r["protocol"] == "v2" and r["previous_protocol"] == "v2"
    assert r["previous"]["ratio"] == pytest.approx(0.95 / 0.99)
    assert r["previous"]["p"] < 0.05 and r["previous_why"] is None
    assert r["previous"]["moved"] == "regressed" and r["weak"]
    assert r["class"] == "passing" and r["levers"] == []
    (same,) = ws.weak_cells(after, after)
    assert same["previous"]["moved"] is None and not same["weak"]
    (alone,) = ws.weak_cells(after)
    assert alone["previous"] is None
    assert alone["previous_why"] == "no previous candidate to pair with"
    (told,) = ws.weak_cells(after, previous_why="it scores other GEMMs")
    assert told["previous_why"] == "it scores other GEMMs"


def test_a_regression_is_the_per_cell_gates_family_wise_finding():
    """Half the cells a little worse and half a little better, each by six
    of its eight GEMMs: every worse cell's own 95% interval excludes the
    tolerance, but across the twelve cells no decline is significant; a
    cell much worse on every GEMM is."""
    shapes = {}
    for c in range(12):
        m, n = (16, 100, 300, 2000)[c % 4], (16, 100, 300, 2000)[c // 4]
        shapes[cell_key(m, n, 64, 1)] = [(m + i, n, 64, 1) for i in range(8)]
    base = {s: 0.9 for ss in shapes.values() for s in ss}
    moved = {}
    for j, ss in enumerate(sorted(shapes.values())):
        sign = -1 if j % 2 == 0 else 1
        for i, s in enumerate(ss):
            moved[s] = 0.9 * math.exp(sign * (0.08 if i < 6 else -0.02))

    def rec(vals):
        g = {"per_gemm": [[*s, v] for s, v in vals.items()], "per_cell": {}}
        for cell, ss in shapes.items():
            g["per_cell"][cell] = {"n": len(ss), "class": "passing"}
        return {"status": "ok", "what": "x", "groups": {"exclusive": g}}

    rows = ws.weak_cells(rec(moved), rec(base), bootstrap=400)
    own = [r for r in rows if r["previous"]["ratio_ci"][1] < 0.99]
    assert len(own) == 6
    assert not [r for r in rows if r["previous"]["moved"] == "regressed"]
    assert not [r for r in rows if r["weak"]]
    worse = dict(moved)
    cell = sorted(shapes)[3]
    for s in shapes[cell]:
        worse[s] = 0.9 * 0.85
    rows = ws.weak_cells(rec(worse), rec(base), bootstrap=400)
    (hit,) = [r for r in rows if r["previous"]["moved"] == "regressed"]
    assert hit["cell"] == cell and hit["weak"] and hit["previous"]["p"] < 0.05
    loose = ws.weak_cells(
        rec(worse), rec(base), settings={"cell_tolerance": 0.2}, bootstrap=400
    )
    assert not [r for r in loose if r["previous"]["moved"] == "regressed"]
    # too few sign flips for the cell test: the regressions are unknown
    blind = ws.weak_cells(rec(worse), rec(base), bootstrap=10)
    assert not [r for r in blind if r["previous"]["moved"] == "regressed"]
    assert blind[0]["previous_why"].startswith("the cell test was off: 10 sign flips")
    (summ,) = ws.summary(blind)
    assert summ["regressed"] is None and summ["regressed_why"].startswith(
        "the cell test was off"
    )


def test_a_record_without_per_gemm_results_gives_no_counts():
    """Records written before per-GEMM results existed hold per-cell numbers
    only: the counts they cannot give are None, and the pairing says why it
    failed."""
    old = record({A: ("model_limited", [0.84] * 4), B: ("passing", [0.99] * 4)})
    for g in old["groups"].values():
        g.pop("per_gemm")
    rows = ws.weak_cells(old, old)
    by = {r["cell"]: r for r in rows}
    assert by[A]["class"] == "model" and by[A]["weak"]
    assert (by[A]["n_tail"], by[A]["n_below_floor"]) == (None, None)
    assert by[A]["previous"] is None
    assert by[A]["previous_why"] == "the before evaluation holds no per-GEMM results"
    (summ,) = ws.summary(rows)
    assert (summ["regressed"], summ["n_tail"], summ["n_below_floor"]) == (
        None,
        None,
        None,
    )
    assert summ["regressed_why"] == by[A]["previous_why"]
    assert (summ["model"], summ["passing"]) == (1, 1)
    # a record with them beside it: only the one without is n/a
    new = record({A: ("model_limited", [0.84] * 4)}, ctx={A: ("passing", [0.99] * 4)})
    new["groups"]["exclusive"].pop("per_gemm")
    summ = {s["group"]: s for s in ws.summary(ws.weak_cells(new))}
    assert summ["exclusive"]["n_tail"] is None and summ["cu64"]["n_tail"] == 0


def test_without_an_audit_record_the_roofline_alone_classifies():
    rows = ws.weak_cells(None, roof_cells={C: {"fraction": 0.1, "n": 3, "n_gaps": 3}})
    (r,) = rows
    assert (r["group"], r["cell"], r["class"], r["weak"]) == (
        "exclusive",
        C,
        "pool",
        True,
    )
    assert r["sel_eff"] is None and r["n"] == 3


def test_the_summary_counts_each_group():
    rec = record(
        {
            A: ("pool_limited", [0.9] * 4),
            B: ("model_limited", [0.9, 0.9, 0.4, 0.8]),
            C: ("passing", [0.99] * 4),
        },
        ctx={A: ("passing", [0.99] * 4)},
    )
    summ = ws.summary(ws.weak_cells(rec, roof_cells={C: {"fraction": 0.2}}))
    assert [s["group"] for s in summ] == ["exclusive", "cu64"]
    ex = summ[0]
    assert (ex["n_cells"], ex["passing"], ex["rule"], ex["model"], ex["pool"]) == (
        3,
        0,
        1,
        1,
        1,
    )
    assert (ex["n_tail"], ex["n_below_floor"]) == (1, 2)
    assert (summ[1]["passing"], summ[1]["n_cells"]) == (1, 1)


def test_origami_cells_pair_the_problems_with_both_picks():
    rows = [
        {
            "m": 64,
            "n": 64,
            "k": 64,
            "batch": 1,
            "model_sel_eff": 0.9,
            "origami_sel_eff": 0.8,
        },
        {
            "m": 65,
            "n": 64,
            "k": 64,
            "batch": 1,
            "model_sel_eff": 0.9,
            "origami_sel_eff": 0.8,
        },
        {
            "m": 66,
            "n": 64,
            "k": 64,
            "batch": 1,
            "model_sel_eff": 0.9,
            "origami_sel_eff": None,
        },
    ]
    assert ws.origami_cells(rows) == {
        A: {"model": pytest.approx(0.9), "origami": pytest.approx(0.8), "n": 2}
    }


def test_the_exports_hold_every_pool_gap_and_the_cells(tmp_path):
    roof = ws.roofline(rows_abc()[:3], PEAKS)
    analysis = {
        "source": ws.TRAINING_SOURCE,
        "why": None,
        "n_dropped_audit": 1,
        "library_digest": "lib0",
        "pool_digest": "pool0",
        "cells_source": ws.HELD_OUT_SOURCE,
        "cells_held_out": True,
        "threshold": 0.3,
        "peaks": roof["peaks"],
        "layout": {"arch": "gfx950", "a_type": "bf16_r"},
        "n_gemms": 3,
        "gaps": ws.pool_gaps(roof["rows"], 0.3),
    }
    cells = ws.weak_cells(
        record({A: ("pool_limited", [0.9] * 4)}),
        roof_cells={},
        threshold=0.3,
        origami={A: {"model": 0.9, "origami": 0.8, "n": 4}},
    )
    assert cells[0]["origami"] is not None
    paths = ws.write_exports(tmp_path / "w", analysis, cells, run="run", bundle="b")
    assert [p.name for p in paths] == [
        ws.POOL_GAPS_JSON,
        ws.POOL_GAPS_CSV,
        ws.WEAK_SPOTS_JSON,
    ]
    assert sorted(p.name for p in (tmp_path / "w").iterdir()) == sorted(
        p.name for p in paths
    )
    doc = json.loads(paths[0].read_text())
    assert doc["schema"] == ws.SCHEMA and doc["n_gaps"] == 1 and doc["n_gemms"] == 3
    assert (doc["source"], doc["held_out"], doc["n_dropped_audit"]) == (
        ws.TRAINING_SOURCE,
        False,
        1,
    )
    assert (doc["library_digest"], doc["pool_digest"]) == ("lib0", "pool0")
    assert doc["gaps"][0]["best_signature"] == "MT64x64x32 MI16x16x32 NTA0 NTB0"
    assert doc["peaks"]["tflops"] == 100.0 and doc["layout"]["arch"] == "gfx950"
    with paths[1].open() as f:
        got = list(csv.DictReader(f))
    assert list(got[0]) == list(ws.CSV_FIELDS)
    assert (got[0]["m"], got[0]["n"], got[0]["bound"]) == ("4096", "1", "memory")
    assert float(got[0]["fraction"]) == pytest.approx(0.25)
    weak = json.loads(paths[2].read_text())
    assert weak["schema"] == ws.WEAK_SCHEMA and weak["summary"][0]["rule"] == 1
    assert weak["cells"][0]["class"] == "rule"
    assert "origami" not in weak["cells"][0]
    assert (weak["roofline_source"], weak["roofline_held_out"]) == (
        ws.HELD_OUT_SOURCE,
        True,
    )
    empty = ws.write_exports(tmp_path / "e", {"threshold": 0.3, "why": "none"})
    assert json.loads(empty[0].read_text())["gaps"] == []
    assert empty[1].read_text().splitlines() == [",".join(ws.CSV_FIELDS)]


def test_the_pool_gaps_come_from_training_benches_without_an_audit_shape(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from lib import audit as al
    from lib import bank_pool
    from lib import evaluate as ev
    from lib import run_analysis as ra

    pool = [kernel(1), kernel(2, 128, 128)]
    seed = [
        gemm(3000, 5, 2, 1024, [(1, 9.0, False)]),
        gemm(64, 64, 64, 1, [(2, 3.0, False)]),
    ]
    held = [gemm(512, 512, 512, 1, [(1, 5.0, False), (2, 4.0, False)])]
    audit = tmp_path / "audit"
    (audit / "stage01").mkdir(parents=True)
    (audit / "stage01" / "shapes.yaml").write_text(
        "- {M: 64, N: 64, K: 64, batch_count: 1}\n"
    )
    model = SimpleNamespace(
        library=("libdir", "stem"), library_cfg={}, library_device=3, arch="gfx950"
    )
    seen = []

    def benches(run_dir, audit_dir, current):
        seen.append(audit_dir)
        if audit_dir is None:
            return ws.TRAINING_SOURCE, [("seed", None)], "no audit set"
        return ws.HELD_OUT_SOURCE, [("audit", "gmap")], None

    monkeypatch.setattr(ra, "final_model", lambda *a, **k: model)
    monkeypatch.setattr(ra, "library_identity", lambda m: SimpleNamespace(digest="L"))
    monkeypatch.setattr(ra, "exhaustive_benches", benches)
    monkeypatch.setattr(
        ra,
        "gemm_dicts",
        lambda dirs, jobs, identity: seed if dirs[0][0] == "seed" else held,
    )
    monkeypatch.setattr(ev, "restrict_to_pool", lambda g, p: (g, None, None))
    monkeypatch.setattr(ev, "pool_from_kernels", lambda k: None)
    monkeypatch.setattr(
        bank_pool, "config_pool", lambda *a: SimpleNamespace(kernels=pool)
    )
    out = ws.exhaustive_best(tmp_path, audit, jobs=2)
    assert seen == [None, audit]
    train = out["training"]
    assert train["source"] == ws.TRAINING_SOURCE and train["n_dropped_audit"] == 1
    assert [(r["m"], r["n"]) for r in train["rows"]] == [(3000, 5)]
    assert out["held_out"]["source"] == ws.HELD_OUT_SOURCE
    assert [r["m"] for r in out["held_out"]["rows"]] == [512]
    assert out["library_digest"] == "L"
    assert out["pool_digest"] == ws.pool_digest(al.pool_digests(pool))
    assert (out["arch"], out["device"]) == ("gfx950", 3)
    alone = ws.exhaustive_best(tmp_path, None)
    assert alone["held_out"]["rows"] == [] and "audit.dir" in alone["held_out"]["why"]
    assert [r["m"] for r in alone["training"]["rows"]] == [3000, 64]

    def nothing(run_dir, audit_dir, current):
        return None, [], "incomplete"

    monkeypatch.setattr(ra, "exhaustive_benches", nothing)
    none = ws.exhaustive_best(tmp_path, audit)
    assert none["training"]["rows"] == [] and none["held_out"]["why"] == "incomplete"
    assert "no exhaustive training bench" in none["training"]["why"]


def test_analyze_takes_the_rates_given_then_the_runs_then_the_devices(
    tmp_path, monkeypatch
):
    seen = {}
    held_rows = [row(16, 16, 16, 1, 4.0), row(4096, 4096, 4096, 1, 4 * BIG_COMPUTE_US)]

    def best(run_dir, audit_dir, jobs):
        seen.update(run_dir=run_dir, audit_dir=audit_dir, jobs=jobs)
        return {
            "arch": "gfx950",
            "device": 2,
            "library_digest": "L",
            "pool_digest": "P",
            "training": {
                "source": ws.TRAINING_SOURCE,
                "why": None,
                "rows": rows_abc(),
                "n_dropped_audit": 0,
            },
            "held_out": {"source": ws.HELD_OUT_SOURCE, "why": None, "rows": held_rows},
        }

    def rates(cfg, device):
        seen.update(device=device, arch=cfg["arch"], a_type=cfg["hipblaslt"]["a_type"])
        return hw.RooflineRates(None, "no clock", 1000.0, "a bus")

    monkeypatch.setattr(ws, "exhaustive_best", best)
    monkeypatch.setattr(hw, "roofline_rates", rates)
    cfg = {"arch": "gfx950", "bench": {"adaptive_long": {"peak_tflops": 100.0}}}
    out = ws.analyze(tmp_path, cfg, audit_dir=tmp_path / "audit", jobs=3)
    assert seen == {
        "run_dir": tmp_path,
        "audit_dir": tmp_path / "audit",
        "jobs": 3,
        "device": 2,
        "arch": "gfx950",
        "a_type": "bf16_r",
    }
    assert out["n_gemms"] == 3 and out["threshold"] == ws.POOL_GAP_FRACTION
    assert out["peaks"]["tflops"] == 100.0
    assert out["peaks"]["tflops_source"] == "bench.adaptive_long.peak_tflops"
    assert out["peaks"]["bandwidth_gbs"] == 1000.0
    assert out["peaks"]["bandwidth_source"] == "the device: a bus"
    # the gaps from the training side only, the cells from the audit set
    assert [g["m"] for g in out["gaps"]] == [4096]
    assert (out["source"], out["held_out"]) == (ws.TRAINING_SOURCE, False)
    assert (out["cells_source"], out["cells_held_out"]) == (ws.HELD_OUT_SOURCE, True)
    assert out["n_cell_gemms"] == 2
    assert set(out["cells"]) == {cell_key(16, 16, 16, 1), cell_key(4096, 4096, 4096, 1)}
    assert out["cells"][cell_key(4096, 4096, 4096, 1)]["fraction"] == pytest.approx(
        0.25
    )
    assert (out["library_digest"], out["pool_digest"]) == ("L", "P")
    assert out["layout"]["arch"] == "gfx950" and out["layout"]["a_type"] == "bf16_r"
    given = ws.analyze(tmp_path, cfg, peak_tflops=200.0, peak_bandwidth_gbs=500.0)
    assert (given["peaks"]["tflops_source"], given["peaks"]["bandwidth_source"]) == (
        "--peak-tflops",
        "--peak-bandwidth-gbs",
    )
    unknown = ws.analyze(tmp_path, {"arch": "gfx950"})
    assert unknown["peaks"]["tflops_source"] == (
        "the best any GEMM reached (unknown: no clock)"
    )

    def training_only(run_dir, audit_dir, jobs):
        out = best(run_dir, audit_dir, jobs)
        out["held_out"] = {"source": None, "why": "incomplete", "rows": []}
        return out

    monkeypatch.setattr(ws, "exhaustive_best", training_only)
    alone = ws.analyze(tmp_path, cfg)
    assert (alone["cells_source"], alone["cells_held_out"]) == (
        ws.TRAINING_SOURCE,
        False,
    )
    assert alone["cells_why"] == "incomplete" and len(alone["cells"]) == 3

    def none(run_dir, audit_dir, jobs):
        empty = {"source": None, "why": "no bundle", "rows": []}
        return {"training": dict(empty), "held_out": dict(empty)}

    monkeypatch.setattr(ws, "exhaustive_best", none)
    empty = ws.analyze(tmp_path, cfg)
    assert (empty["n_gemms"], empty["why"], empty["gaps"]) == (0, "no bundle", [])
