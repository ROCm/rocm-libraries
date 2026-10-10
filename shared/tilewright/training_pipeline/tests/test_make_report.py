# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import html
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import make_report as mr

EVIL = '</script><script>alert(1)</script><img src=x onerror="alert(2)">'
LARGE = "Large|Large|LargeK|Bnone"
TINY = "Tiny|Tiny|TinyK|Bnone"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


def files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def stage08_dataset(name, per_cell_json):
    """One summary.json dataset entry the way stage08 writes it: 10 wins, 85
    ties and 5 losses against Origami."""
    from lib import evaluate as ev

    results = []
    for i, (pick, origami) in enumerate([(1.0, 2.0)] * 10 + [(1.0, 1.0)] * 85):
        results.append(
            ev.GemmResult(
                index=i,
                m=64,
                n=64,
                k=64,
                batch=1,
                leaf="x",
                model_cell="x",
                served_by="model",
                winner_us=1.0,
                pick_sol_idx=1,
                pick_us=pick,
                origami_sol_idx=2,
                origami_us=origami,
            )
        )
    results += [
        ev.GemmResult(
            index=95 + i,
            m=64,
            n=64,
            k=64,
            batch=1,
            leaf="x",
            model_cell="x",
            served_by="model",
            winner_us=1.0,
            pick_sol_idx=3,
            pick_us=2.0,
            origami_sol_idx=2,
            origami_us=1.0,
        )
        for i in range(5)
    ]
    overall = ev.summarize(results)
    return json.loads(
        json.dumps(
            {
                "name": name,
                "path": "/datasets/x",
                "rc": 0,
                "per_cell_json": per_cell_json,
                "n_evaluated_cells": 1,
                "model_geomean_sel_eff": overall["sel_eff"],
                "origami_geomean_sel_eff": overall["origami_sel_eff"],
                "summary": {
                    "n_gemms_before_filter": 100,
                    "n_gemms_dropped_by_filter": 0,
                    "n_gemms_without_library_candidate": 0,
                    "n_out_of_library_rows": 0,
                    "workload_filter": {},
                    "library_stem": None,
                    "weight_dtype": "int8",
                    "n_cells_evaluated": 1,
                    "n_routing_mismatch": 0,
                    "model_geomean_sel_eff": overall["sel_eff"],
                    "origami_geomean_sel_eff": overall["origami_sel_eff"],
                    "model_geomean_gemm_weighted": overall["sel_eff"],
                    "origami_geomean_gemm_weighted": overall["origami_sel_eff"],
                    "overall": overall,
                },
            },
            allow_nan=True,
        )
    )


def test_stage08_rows_read_the_overall_block(tmp_path):
    run = tmp_path / "run"
    write(
        run / "validate" / "stage08" / "summary.json",
        {"datasets": [stage08_dataset("ds", "per_cell.json")]},
    )
    (row,) = mr.stage08(str(run))
    assert row["paired_n"] == 100
    assert (row["wins"], row["ties"], row["losses"]) == (10, 85, 5)
    assert row["speedup"] == pytest.approx(2.0**0.1 * 0.5**0.05)
    assert row["model"] == round(0.5**0.05 * 100, 2)
    assert row["origami"] == round(0.5**0.1 * 100, 2)


def test_a_rebenched_stage08_dataset_is_decomposed_under_its_configured_rule(
    tmp_path, monkeypatch
):
    from lib import run_analysis as ra

    rebenched = {
        "pick_measurement": "rebench",
        "pre_rebench": {"pick_measurement": "timed", "sel_eff": 0.9},
    }
    assert mr.stage08_pick_measurement(rebenched, "warmup") == "timed"
    assert mr.stage08_pick_measurement({"pick_measurement": "any"}, "warmup") == "any"
    assert mr.stage08_pick_measurement({"pick_measurement": "rebench"}, "x") == "x"
    assert mr.stage08_pick_measurement({}, "warmup") == "warmup"

    class Bench:
        notes = []
        gemms = [object()]

    class Evaluated:
        engine = False
        pick_measurement = "timed"

    seen = []
    model = ra.ModelInfo(labels=[], whitelists={})
    monkeypatch.setattr(ra, "final_model", lambda *a, **k: model)
    monkeypatch.setattr(ra, "stage08_enriched_dir", lambda d, ds: tmp_path)
    monkeypatch.setattr(ra, "load_bench", lambda *a, **k: Bench())
    monkeypatch.setattr(ra, "pool_for", lambda bench, lib: None)
    monkeypatch.setattr(ra, "decomposition", lambda e: {"*": {}})
    monkeypatch.setattr(ra, "evaluate", lambda *a, **k: seen.append(k) or Evaluated())
    notes = []
    rows = [{"name": "ds", "dataset": {"summary": rebenched}}]
    got = mr._stage08_decomposition(str(tmp_path), rows, 1, False, notes, "warmup")
    assert notes == [] and [k["pick_measurement"] for k in seen] == ["timed"]
    assert got == {
        "ds": {"engine": False, "pick_measurement": "timed", "decomposition": {"*": {}}}
    }


def make_run(root):
    good = "Large|Large|LargeK|Bnone"
    evil_cell = good + "#M<=4096" + EVIL
    write(
        root / "round_0" / "stage05" / "cells.json",
        {
            "cells": [
                {
                    "label": good,
                    "n_gemms": 40,
                    "best_sel_eff": 0.9,
                    "deployed": {"sel_eff": 0.88},
                    "origami": {"sel_eff": 0.8},
                    "smart_k_signatures": [[1] * 8],
                    "embed_dim": 4,
                    "hidden_dim": 6,
                    "inter_hidden": 5,
                },
                {"label": evil_cell, "n_gemms": 10, "best_sel_eff": 0.7},
            ]
        },
    )
    write(root / "round_0" / "stage05" / "metrics.json", "{not json")
    write(
        root / "round_1" / "stage05" / "metrics.json",
        {
            "global_sel_eff": 0.91,
            "global_deployed_sel_eff": 0.9,
            "global_origami_sel_eff": 0.85,
            "n_cells_trained": 2,
            "n_gemms_trained": 50,
        },
    )
    write(
        root / "round_1" / "stage04b" / "decisions.json",
        {
            "aggregate": {
                "global_sel_eff_new": 0.87,
                "global_origami_sel_eff_new": 0.83,
            },
            "per_cell": [
                {
                    "cell": good,
                    "sel_eff_new": 0.86,
                    "n_eval": 20,
                    "paired_sel_eff_new": 0.855,
                    "origami_sel_eff_new": 0.84,
                    "origami_n_eval": 20,
                    "winner_us_geomean": 10.0,
                    "pick_us_geomean": 11.0,
                    "origami_pick_us_geomean": 12.0,
                },
                {"cell": evil_cell, "sel_eff_new": 0.5, "n_eval": 5},
            ],
        },
    )
    write(root / "round_1" / "stage06" / "manifest.json", {"file": {"size": 2_500_000}})
    write(
        root / "validate" / "stage07" / "summary.json",
        {
            "ok": False,
            "failures": ["x"],
            "per_yaml": {
                EVIL: {
                    "n_kept": 3,
                    "n_entries": 4,
                    "parity": {
                        "n_agree": 2,
                        "n_problems": 3,
                        "match_rate": 2 / 3,
                        "counts": {"match": 2, EVIL: 1},
                        "runtime_selection": {
                            "pick_measurement": "timed",
                            "n_problems": 3,
                            "n_evaluated": 3,
                            "n_miss": 1,
                            "sel_eff": 0.9312,
                            "sel_eff_ci95": [0.9011, 0.9533],
                        },
                    },
                    "timing": {
                        "rsn1": {
                            "repetitions": 3,
                            "ml_on_us": {"n": 2, "p50": 23.0, "p50_ci95": [22.0, 24.0]},
                            "first_in_cell": {
                                "n_problems": 1,
                                "ml_on_us": {
                                    "n": 1,
                                    "p50": 31.5,
                                    "p50_ci95": [30.0, 33.0],
                                },
                            },
                            "warm": {
                                "n_problems": 1,
                                "ml_on_us": {
                                    "n": 1,
                                    "p50": 6.75,
                                    "p50_ci95": [6.5, 7.0],
                                },
                            },
                            "ml_off_us": {"p50": 20.0},
                            "delta_us": {
                                "n": 2,
                                "p10": 2.5,
                                "p50": 3.25,
                                "p90": 4.0,
                                "p99": 4.5,
                            },
                        }
                    },
                }
            },
        },
    )
    write(
        root / "validate" / "stage08" / "summary.json",
        {"datasets": [stage08_dataset(EVIL, "per_cell.json")]},
    )
    write(
        root / "validate" / "stage08" / "per_cell.json",
        [{"cell": evil_cell, "sel_eff": 0.9, "origami_sel_eff": 0.8}],
    )
    write(
        root / "timing" / "20261004_000000_full.json",
        {
            "total_s": 75,
            "rows": [{"round": 0, "stage": EVIL, "rc": 0, "elapsed_s": 75}],
        },
    )
    write(
        root / "configs" / "20261004_000000_full.source.yaml", "config_id: x  # " + EVIL
    )
    write(root / "held_out_sel_eff_breakdown.json", "[1, 2")


def test_report_escapes_and_writes_only_its_file(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    before = files(run)
    out = tmp_path / "report.html"
    mr.build(str(run), str(out))
    assert files(run) == before
    page = out.read_text()
    assert page.count("</script>") == 1
    assert "<script>alert" not in page
    assert '<img src=x onerror="alert(2)">' not in page
    assert "\\u003c/script" in page
    assert "onchange=" not in page and "onclick=" not in page
    assert "2 / 3 problems agree" in page
    assert "93.12%" in page and "[90.11, 95.33]" in page
    assert "31.50" in page and "6.75" in page
    assert "ML-off" not in page and "3.25" not in page
    assert "2.5 MB" in page or '"bin_mb": 2.5' in page
    assert "origami" not in page.lower()
    assert "Diagnostics" not in page

    mr.build(str(run), str(out), origami_reference=True)
    assert files(run) == before
    page = out.read_text()
    assert page.count("</script>") == 1
    assert "<script>alert" not in page
    ref = page[page.index("External reference: Origami") :]
    heading = page.index('<h2 id="origami">')
    assert page.lower().index("origami") == heading + len('<h2 id="')
    assert page.index("<details", heading) < page.index("Origami %")
    assert "84.00" in ref and "85.50" in ref
    assert "3.25" in page and "ML-off" in page


def test_rounds_that_went_on_without_some_cells_say_so(tmp_path):
    """train.on_cell_failure carry: the training table counts the cells a
    round went on without, the notes at the top name them."""
    run = tmp_path / "run"
    make_run(run)
    write(
        run / "round_1" / "stage05" / "cells_carried.json",
        {
            "policy": "carry",
            "n_cells_trained": 2,
            "cells": [
                {
                    "cell": EVIL,
                    "why": "not trained: too few gemms (2)",
                    "model": "the model of an earlier round",
                }
            ],
        },
    )
    rows = {r["round"]: r for r in mr.training_rounds(str(run))}
    assert rows["round_1"]["carried"][0]["cell"] == EVIL
    out = tmp_path / "report.html"
    mr.build(str(run), str(out))
    page = out.read_text()
    assert "<script>alert" not in page
    note = (
        "round_1 went on without training 1 cell(s) (train.on_cell_failure: "
        f"carry): {html.escape(EVIL)} (not trained: too few gemms (2); keeps the "
        "model of an earlier round)"
    )
    assert note in page and page.index(note) < page.index('<h2 id="audit">')
    table = page[page.index("<h2>Training (stage05, in-sample)</h2>") :]
    assert "<th>cells carried</th>" in table
    assert "<td>2</td><td>1</td><td>50</td>" in table


def test_report_on_an_empty_run(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    out = run / "report.html"
    proc = subprocess.run(
        [sys.executable, str(Path(mr.__file__)), str(run)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert files(run) == ["report.html"]
    assert "no stage 7 results" in out.read_text()


def test_report_of_a_run_started_with_the_probe(tmp_path):
    """stage02 outputs and a probe section in the config snapshot of a run
    written before the planner became stage02 are ignored; its timing rows
    show as the removed probe's column."""
    run = tmp_path / "old"
    make_run(run)
    rd = run / "round_1"
    write(rd / "stage02" / "shapes_probed.yaml", "- {M: 64, N: 64, K: 64, iters: 40}\n")
    write(rd / "stage02" / ".stage_complete.json", {"cmd": ["stage02_probe.py"]})
    write(rd / "stage02" / "logs" / "probe_0000_lines1-1_gpu0.log", "probe\n")
    write(rd / "stage02.log", "probe\n")
    write(
        run / "timing" / "20261004_000001_full.json",
        {
            "total_s": 30,
            "rows": [
                {"round": 1, "stage": "stage02", "rc": 0, "elapsed_s": 20},
                {"round": 1, "stage": "stage03", "rc": 0, "elapsed_s": 10},
            ],
        },
    )
    write(
        run / "configs" / "20261004_000001_full.yaml",
        "config_id: x\nprobe:\n  duration_us: 2000\nbench:\n  adaptive:\n"
        "    enabled: false\n",
    )
    before = files(run)
    out = tmp_path / "report.html"
    mr.build(str(run), str(out))
    assert files(run) == before
    page = out.read_text()
    assert "<th>stage03</th>" in page
    assert "<th>stage02 (removed probe)</th>" in page and page.count("stage02") == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_the_wall_clock_table_has_a_column_for_every_stage(legacy):
    """The planner's column is stage02, or stage01b in a run written before
    the planner became stage02, whose stage02 rows are the removed probe's."""
    row = {"rc": 0, "elapsed_s": 60}
    planner = ("stage01b",) if legacy else ("stage02",)
    stages = ("stage01", *planner, "stage03", "stage03b", "stage03c", "stage04")
    rounds = {
        0: {k: dict(row) for k in stages + ("stage05", "audit_v2", "a_new_tool")},
        1: {"stage04b": dict(row, rc=2)},
        None: {
            k: dict(row)
            for k in ("audit set", "final stage05", "audit_eval", "stage07", "stage08")
        },
    }
    if legacy:
        rounds[1]["stage02"] = dict(row)
    page = mr.timing_html(rounds, 600, legacy)
    assert re.findall(r"<th>([^<]*)</th>", page) == [
        "audit set build",
        *stages[:2],
        *(["stage02 (removed probe)"] if legacy else []),
        *stages[2:],
        "stage04b",
        "stage05",
        "audit v2",
        "final retrain stage05",
        "audit eval",
        "stage07",
        "stage08",
        "a_new_tool",
        "round total",
    ]
    assert "1m00s (rc=2)" in page and ("probe" in page) == legacy


def test_stage7_says_whether_it_served_the_bank(tmp_path):
    run = tmp_path / "run"
    summary = run / "validate" / "stage07" / "summary.json"
    for served, text in ((True, "served"), (False, "not served")):
        write(summary, {"ok": True, "failures": [], "bank_served": served})
        assert (
            f"; the bank of kernels was {text} (TENSILE_TILEWRIGHT_BANK="
            f"{int(served)} on its ML-on runs)."
            in mr.stage07_html(mr.stage07(str(run)))
        )
    write(summary, {"ok": True, "failures": []})
    assert "bank" not in mr.stage07_html(mr.stage07(str(run)))


def test_stage7_times_read_as_min_mean_p99_max_and_pair_in_five_points(tmp_path):
    run = tmp_path / "run"
    on = {"n": 3, "min": 11.0, "mean": 12.5, "p50": 12.0, "p99": 14.9, "max": 15.0}
    paired = {"n": 3, "model_sel_eff": 0.97, "origami_sel_eff": 0.9}
    paired.update(wins=2, ties=1, losses=0)
    write(
        run / "validate" / "stage07" / "summary.json",
        {
            "ok": True,
            "failures": [],
            "bank_served": True,
            "origami_reference": True,
            "per_yaml": {
                "stage07_audit_50": {
                    "held_out": True,
                    "n_kept": 3,
                    "n_entries": 3,
                    "origami_reference": {
                        "n_problems": 3,
                        "n_without_timings": 0,
                        "n_origami_without_row": 0,
                        "paired": paired,
                    },
                    "timing": {
                        "rsn1": {
                            "repetitions": 2,
                            "ml_on_us": dict(on, p50_ci95=[11.5, 12.5]),
                            "ml_off_us": {"p50": 30.0},
                            "delta_us": {"n": 3, "p50": -18.0},
                        }
                    },
                },
                "stage07_trained_6": {
                    "held_out": False,
                    "n_kept": 6,
                    "n_entries": 6,
                    "origami_reference": {"n_problems": 6, "paired": paired},
                },
            },
        },
    )
    s7 = mr.stage07(str(run))
    page = mr.stage07_html(s7)
    assert "11.00 / 12.50 / 14.90 / 15.00" in page
    assert "<th>ML-on min / mean / p99 / max</th>" in page
    assert "match the config; held out of training)" in page
    assert "Origami" not in page
    page = mr.stage07_html(s7, origami_reference=True)
    model, origami = (paired[f"{k}_sel_eff"] * 100 for k in ("model", "origami"))
    assert f"model <b>{model:.2f}%</b>, Origami {origami:.2f}% (2 / 1 / 0 wins" in page
    agg = {"paired_n": 40, "paired_model": 96.0, "paired_origami": 93.0}
    s8 = [
        {
            "name": "val_a",
            "paired_n": 20,
            "model": 97.0,
            "origami": 95.5,
            "wins": 3,
            "ties": 15,
            "losses": 2,
            "speedup": 1.02,
        }
    ]

    def cells(html):
        rows = re.findall(r"<tr>(.*?)</tr>", html)
        return [c for c in (re.findall(r"<td[^>]*>([^<]*)</td>", r) for r in rows) if c]

    trained = "held-out GEMMs of the training distribution"
    five = mr.five_point_html(agg, s8, s7)
    assert cells(five) == [
        [
            f"{trained}: the rounds' new GEMMs",
            "40",
            "96.00",
            "93.00",
            "+3.00",
        ],
        [
            f"{trained}: the audit set's stage07_audit_50 (stage 7, hipBLASLt's picks)",
            "3",
            "97.00",
            "90.00",
            "+7.00",
        ],
        ["stage 8 val_a", "20", "97.00", "95.50", "+1.50"],
        ["stage07_audit_50", "1", "3", "11.00", "12.50", "14.90", "15.00", "30.00"],
    ]
    assert five in mr.origami_reference_html(agg, [], [], [], s8, s7)
    assert "no paired numbers" not in five
    assert "no stage 7 selection times" in mr.five_point_html(agg, [], None)
    five = mr.five_point_html({}, s8, None, why="planned benches do not time it")
    assert cells(five)[0] == [trained, "&ndash;", "&ndash;", "&ndash;", "&ndash;"]
    assert f"{trained}: no paired numbers: planned benches do not time it." in five


def test_stage7_names_what_production_serves_and_where_origami_picked(tmp_path):
    from lib.kernel_identity import label_of_name

    run = tmp_path / "run"
    grid = "TensileLibrary_X_Bias_UA_Type_X_gfx950"
    sig = "MT128x128x256 MI16x16x128 NTA0 NTB0"
    served_by = [
        {
            "row": "GridBased",
            "library": grid,
            "kernel": "Cijk_K_GRID",
            "signature": sig,
            "n": 5,
        },
        {"row": "Equality", "library": "TensileLibrary_X_UA_Type_X_gfx950", "n": 1},
    ] + [
        {"row": "Prediction", "library": "host", "kernel": f"Cijk_K{i}", "n": 1}
        for i in range(5)
    ]
    unscored = {"row": "Prediction", "library": "host", "kernel": "Cijk_K"}
    paired = {"n": 3, "model_sel_eff": 0.97, "origami_sel_eff": 0.9}
    paired.update(wins=2, ties=1, losses=0)
    left_out = {
        "m": 4,
        "n": 4,
        "k": 4,
        "batch": 1,
        "row": "Equality",
        "library": "anchors",
        "kernel": "MT64x64x64 MI16x16x128 NTA0 NTB0",
        "classes": ["cluster_dim_y4"],
    }
    write(
        run / "validate" / "stage07" / "summary.json",
        {
            "ok": True,
            "failures": [],
            "origami_reference": True,
            "production_route": {
                "n_problems": 16,
                "n_tilewright": 4,
                "n_unscored": 1,
                "n_reached": 5,
                "tilewright_share": 4 / 16,
                "warned_yamls": ["held"],
            },
            "per_yaml": {
                "held": {
                    "held_out": True,
                    "n_kept": 11,
                    "n_entries": 11,
                    "production_route": {
                        "n_problems": 11,
                        "n_tilewright": 0,
                        "n_unscored": 0,
                        "tilewright_share": 0.0,
                        "n_ahead": 6,
                        "warned": True,
                        "model_loaded": False,
                        "served_by": served_by,
                    },
                    "origami_reference": {
                        "n_problems": 3,
                        "paired": paired,
                        "origami_rows": {"Prediction": 3},
                    },
                },
                "trained": {
                    "n_kept": 3,
                    "production_route": {
                        "n_problems": 3,
                        "n_tilewright": 2,
                        "n_unscored": 1,
                        "tilewright_share": 2 / 3,
                        "warned": False,
                        "model_loaded": True,
                        "served_by": [dict(unscored, unscored=True, n=1)],
                    },
                },
                "anchored": {
                    "n_kept": 3,
                    "production_route": {
                        "n_problems": 2,
                        "n_tilewright": 2,
                        "tilewright_share": 1.0,
                        "warned": False,
                        "model_loaded": True,
                        "served_by": [],
                    },
                    "skipped": {"production_route_problems": [left_out]},
                },
                "quarantined": {"skipped": {"production_route": "a held kernel"}},
            },
        },
    )
    s7 = mr.stage07(str(run))
    page = mr.stage07_html(s7, origami_reference=True)
    assert (
        "; without TENSILE_PREDICTION_LIB, as production runs, tilewright served "
        "<b>4 of 16</b> problems (25.00%) and "
        "11 did not reach the model: see the production route of held" in page
    )
    # Kernels by signature, else by what their names spell, never by name.
    assert (
        '<tr class="weak"><td class="c">production route</td>' in page
        and "tilewright served <b>0 / 11</b> problems (0.00%), rows ahead of the "
        "Prediction row answered 6; the others: "
        f"5 by the GridBased row of {grid} ({sig}); 1 by the Equality row of "
        "TensileLibrary_X_UA_Type_X_gfx950; 1 by the Prediction row of host "
        f"({label_of_name('Cijk_K0')})" in page
    )
    assert "Cijk_K" not in page
    assert "; 2 more kernel(s); the runtime never loaded the model" in page
    # Problems the model scored nothing for reached it: no highlight.
    assert (
        '<tr><td class="c">production route</td><td colspan="6" '
        'style="text-align:left">without TENSILE_PREDICTION_LIB, as production '
        "runs: tilewright served <b>2 / 3</b> problems (66.67%), "
        "scored nothing for 1, which its Prediction row served</td>" in page
    )
    assert "tilewright served <b>2 / 2</b> problems (100.00%); 1 problem(s) left " in (
        page
    )
    assert (
        "an Equality row answers their size with a kernel of a hazard class "
        "(ClusterDim with a second dimension of 4) the canary never ran alone "
        "(MT64x64x64 MI16x16x128 NTA0 NTB0 of anchors)</td>" in page
    )
    assert 'style="text-align:left">skipped: a held kernel</td>' in page
    assert (
        "the runtime with tilewright off, its pick from the Prediction row "
        "(TENSILE_PREDICTION_LIB=1), on the same 3 of 3 problems" in page
    )


def test_the_unpaired_training_distribution_says_why():
    planned = {"planner": {"enabled": True}}
    why = mr.trained_pairing_reason(planned, None)
    assert why.startswith("planned benches do not time Origami's pick")
    assert why.endswith("with stage07.parity_sample_audit and audit.dir")
    audit = dict(planned, stage07={"parity_sample_audit": 20}, audit={"dir": "/a"})
    assert mr.trained_pairing_reason(audit, None, "stage 7 failed").endswith(
        "; no stage 7 results (stage 7 failed)"
    )
    s7 = {"origami_reference": None, "yamls": [{"yaml": "w", "held_out": False}]}
    assert "and this run's did not" in mr.trained_pairing_reason(audit, s7)
    s7["origami_reference"] = True
    assert mr.trained_pairing_reason(audit, s7).endswith(
        "stage 7 had no sample of the audit set to time"
    )
    s7["yamls"].append({"yaml": "a", "held_out": True})
    assert mr.trained_pairing_reason({}, s7).startswith(
        "no round's new GEMMs pair the bundle"
    )


def test_script_json_escapes_markup():
    text = mr.script_json({"a": "</script>&<!--"})
    assert "<" not in text and ">" not in text and "&" not in text
    assert json.loads(text) == {"a": "</script>&<!--"}


def test_held_out_breakdown_rebuilds_without_parents(tmp_path):
    run = tmp_path / "run"
    parent = "Large|Large|LargeK|Bnone"
    write(
        run / "round_1" / "stage04b" / "decisions.json",
        {"per_cell": [{"cell": parent, "sel_eff_new": 0.5, "n_eval": 10}]},
    )
    write(run / "round_1" / "stage04b" / "splits.json", {"splits": {parent: {}}})
    write(
        run / "round_2" / "stage04b" / "decisions.json",
        {"per_cell": [{"cell": parent + "#K<=99", "sel_eff_new": 0.9, "n_eval": 10}]},
    )
    assert list(mr.held_out_breakdown(str(run))) == [parent + "#K<=99"]


def test_a_cell_split_later_keeps_its_measurement(tmp_path):
    run = tmp_path / "run"
    leaf = "Large|Large|LargeK|Bnone"
    write(
        run / "round_1" / "stage04b" / "decisions.json",
        {"per_cell": [{"cell": leaf, "sel_eff_new": 0.5, "n_eval": 10}]},
    )
    write(run / "round_1" / "stage04b" / "splits.json", {"splits": {leaf: {}}})
    assert list(mr.held_out_breakdown(str(run))) == [leaf]


def stage06_round(rd, gate, deployed=True, reason=None):
    import run_orchestrator as ro

    (rd / "stage05").mkdir(parents=True, exist_ok=True)
    (rd / "stage05" / "models.pt").write_bytes(rd.name.encode())
    rec = {"train_dir": str(rd / "stage05"), "sha256": "ab" * 32, "deployed": deployed}
    if not deployed:
        rec.update(not_deployed=reason, gate={"sel_eff": gate["sel_eff"]})
    write(rd / "stage06" / "deploy_record.json", rec)
    write(rd / "stage06" / "gate.json", gate)
    ro.mark_stage_complete(rd / "stage06", None)


def gate_json(sel_eff, n, n_miss, passed, base=None):
    cell = "Large|Large|LargeK|Bnone"
    return {
        "n_gemms": n,
        "n_evaluated": n,
        "n_miss": n_miss,
        "sel_eff": sel_eff,
        "pick_measurement": "timed",
        "target": 0.95,
        "passed": passed,
        "reason": "ok" if passed else "below the bundle it would replace",
        "baseline": base or {},
        "per_cell": {
            cell: {
                "sel_eff_new": sel_eff,
                "n_eval": n,
                "n_miss": n_miss,
                "paired_sel_eff_new": sel_eff,
                "origami_sel_eff_new": 0.5,
                "origami_n_eval": n,
            }
        },
    }


def test_headline_is_the_deployed_bundle_not_a_later_staged_one(tmp_path):
    run = tmp_path / "run"
    stage06_round(run / "round_0", gate_json(0.9312, 30, 2, True))
    stage06_round(
        run / "round_1",
        gate_json(
            0.8125,
            40,
            9,
            False,
            base={
                "train_dir": str(run / "round_0" / "stage05"),
                "n": 40,
                "sel_eff": 0.9001,
                "bundle_sel_eff": 0.8125,
            },
        ),
        deployed=False,
        reason="below the bundle it would replace",
    )
    sb = mr.shipped_bundle(str(run))
    assert sb["shipped"]["round"] == "round_0"
    assert (sb["source"], sb["n_gemms"], sb["n_miss"]) == ("deploy_gate", 30, 2)
    assert sb["gate"]["sel_eff"] == 0.9312
    assert sb["final_bundle"]["round"] == "round_1"

    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    shipped = page[page.index('<h2 id="shipped">') : page.index("<b>Cells</b>")]
    assert "<b>round_0</b>" in shipped and "93.12" in shipped
    assert "Not deployed:</b> round_1" in shipped and "81.25" in shipped
    assert "90.01" not in shipped
    data = json.loads(page[page.index("const D=") + 8 : page.index(";\nconst TRAJ")])
    s = data["S"]
    assert (s["shipped"]["round"], s["shipped"]["sel_eff"]) == ("round_0", 93.12)
    assert (s["shipped"]["n"], s["shipped"]["n_miss"]) == (30, 2)
    assert s["not_deployed"]["round"] == "round_1"
    assert s["pick_measurement"] == "warmup"
    assert "origami" not in page.lower()

    mr.build(str(run), str(out), bench=False, origami_reference=True)
    ref = out.read_text()
    ref = ref[ref.index('<h2 id="origami">') :]
    assert "50.00" in ref and "93.12" in ref


def test_stage04b_results_per_context(tmp_path):
    from lib import features as fs
    from lib.run_analysis import context_label

    run = tmp_path / "run"
    write(
        run / "round_1" / "stage04b" / "decisions.json",
        {
            "aggregate": {
                "global_sel_eff_new": 0.9,
                "by_context": {
                    "0/default": {"n_evaluated": 40, "n_miss": 1, "sel_eff": 0.93},
                    "128/default": {"n_evaluated": 20, "n_miss": 4, "sel_eff": 0.88},
                },
                "context_holdout": {
                    "64/dynamic": {"n_evaluated": 10, "n_miss": 0, "sel_eff": 0.71}
                },
            },
            "per_cell": [],
        },
    )
    rows = mr.stage04b_contexts(str(run))
    cu128 = context_label(fs.ExecutionContext(cu_budget=128))
    held = context_label(fs.ExecutionContext(cu_budget=64, schedule="dynamic"))
    assert [
        (r["kind"], r["context"], r["n"], r["n_miss"], r["sel_eff"]) for r in rows
    ] == [
        ("trained context", "exclusive", 40, 1, 0.93),
        ("trained context", cu128, 20, 4, 0.88),
        ("held-out context", held, 10, 0, 0.71),
    ]
    names = mr.context_names(
        {
            "contexts": {
                "list": [{"cu_budget": 128, "name": "half"}],
                "holdout": [{"cu_budget": 64, "schedule": "dynamic"}],
            }
        }
    )
    assert names[cu128] == "half"
    assert names[held].endswith("held out")


FRACTION_SOURCE = (
    "contexts:\n  list:\n    - {cu_budget: 0}\n"
    "    - {cu_fraction: 0.75, cotenant: true}\n"
    "  holdout:\n    - {cu_fraction: 0.875, cotenant: true}\n"
    "    - {cu_fraction: 0.8, cotenant: true}\n"
)
# The snapshot a run of FRACTION_SOURCE on a 256-CU device writes: the last
# held-out entry resolves to the trained entry's budget and is left out.
FRACTION_SNAPSHOT = (
    "contexts:\n  list:\n    - {cu_budget: 0}\n"
    "    - {cu_fraction: 0.75, cotenant: true, cu_budget: 192}\n"
    "  holdout:\n    - {cu_fraction: 0.875, cotenant: true, cu_budget: 224}\n"
)


def fraction_run(run, snapshot=FRACTION_SNAPSHOT):
    write(run / "configs" / "20261009_000000_full.source.yaml", FRACTION_SOURCE)
    write(run / "configs" / "20261009_000000_full.yaml", snapshot)
    return run


def test_context_names_of_fractions_come_from_the_resolved_snapshot(tmp_path):
    import yaml

    from lib import features as fs
    from lib.run_analysis import context_label

    run = str(fraction_run(tmp_path / "run"))
    assert mr.context_names(yaml.safe_load(FRACTION_SOURCE)) == {}
    names = mr.context_names(mr.run_config(run, mr.config_source(run)))
    assert names == {
        context_label(fs.ExecutionContext(cu_budget=192)): "cu192_cot",
        context_label(fs.ExecutionContext(cu_budget=224)): "cu224_cot held out",
    }


def stage04b_of_fractions(run):
    """round_1's stage04b of a fraction_run: the trained and the remaining
    held-out context."""
    write(
        run / "round_1" / "stage04b" / "decisions.json",
        {
            "aggregate": {
                "global_sel_eff_new": 0.9,
                "by_context": {
                    "0/default": {"n_evaluated": 40, "n_miss": 1, "sel_eff": 0.93},
                    "192/default": {"n_evaluated": 20, "n_miss": 4, "sel_eff": 0.88},
                },
                "context_holdout": {
                    "224/default": {"n_evaluated": 10, "n_miss": 0, "sel_eff": 0.71}
                },
            },
            "per_cell": [],
        },
    )


def test_the_report_names_the_contexts_a_fraction_run_resolved(tmp_path):
    run = fraction_run(tmp_path / "run")
    stage04b_of_fractions(run)
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    contexts = page[page.index('<h2 id="contexts">') : page.index('<h2 id="identity">')]
    assert "cu192 (cu192_cot)" in contexts
    assert "cu224 (cu224_cot held out)" in contexts


def test_the_report_says_which_contexts_the_device_left_out(tmp_path):
    def page(snapshot):
        run = fraction_run(tmp_path / "run", snapshot)
        stage04b_of_fractions(run)
        out = tmp_path / "report.html"
        mr.build(str(run), str(out), bench=False)
        text = out.read_text()
        notes = text[text.index('<p class="cap" id="cap">') : text.index('id="audit"')]
        start = text.index('<h2 id="contexts">')
        return notes, text[start : text.index('<h2 id="identity">')]

    notes, contexts = page(FRACTION_SNAPSHOT)
    assert (
        "Left out on this run's bench device:</b> contexts.holdout[1] "
        "(cu_fraction 0.8)." in contexts
    )
    assert "No held-out context remains" not in contexts
    assert "every held-out context" not in notes
    none_held = FRACTION_SNAPSHOT[: FRACTION_SNAPSHOT.index("  holdout:")]
    notes, contexts = page(none_held + "  holdout: []\n")
    assert "contexts.holdout[0] (cu_fraction 0.875), contexts.holdout[1]" in contexts
    assert "No held-out context remains: the run measured the models on none" in (
        contexts
    )
    assert "every held-out context of the config was left out" in notes
    notes, contexts = page(FRACTION_SOURCE)
    assert "Left out" not in contexts and "every held-out context" not in notes


def test_diagnostics_flag(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    proc = subprocess.run(
        [sys.executable, str(Path(mr.__file__)), str(run), "--diagnostics"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# ── held-out first ──────────────────────────────────────────────────────────

CONFIG = (
    "full:\n  total_rounds: 8\n"
    "stage07:\n  parity_sample_trained: 1000\n  bench_yamls: []\n"
    "stage08:\n  datasets: []\n"
    "deploy:\n  apply_patch: false\n  build_after_apply: false\n"
)
SHA = "cd" * 32


def staged_run(root, audit_dir=None):
    """A run whose round_0 bundle its deploy gate kept out (staged, not
    deployed) and round_1's stage04b measured again. LARGE is good in-sample
    and weak held out, TINY the other way round."""
    import run_orchestrator as ro

    s5 = root / "round_0" / "stage05"
    write(
        s5 / "cells.json",
        {
            "cells": [
                {
                    "label": LARGE,
                    "n_gemms": 100,
                    "best_sel_eff": 0.991,
                    "deployed": {"sel_eff": 0.991},
                },
                {
                    "label": TINY,
                    "n_gemms": 100,
                    "best_sel_eff": 0.702,
                    "deployed": {"sel_eff": 0.702},
                },
            ]
        },
    )
    (s5 / "models.pt").write_bytes(b"bundle 0")
    gate = {
        "n_gemms": 60,
        "n_evaluated": 60,
        "n_miss": 4,
        "sel_eff": 0.8808,
        "pick_measurement": "rebench",
        "target": 0.95,
        "passed": False,
        "reason": "88.08% on the gate GEMMs is below the target",
        "per_cell": {
            LARGE: {"sel_eff_new": 0.80, "n_eval": 30, "n_miss": 4},
            TINY: {"sel_eff_new": 0.97, "n_eval": 30, "n_miss": 0},
        },
    }
    write(root / "round_0" / "stage06" / "gate.json", gate)
    write(
        root / "round_0" / "stage06" / "deploy_record.json",
        {
            "train_dir": str(s5),
            "sha256": SHA,
            "deployed": False,
            "not_deployed": gate["reason"],
            "gate": {"sel_eff": gate["sel_eff"]},
        },
    )
    ro.mark_stage_complete(root / "round_0" / "stage06", None)
    write(root / "round_0" / ".round_complete.json", {})
    write(
        root / "round_1" / "stage04b" / "decisions.json",
        {
            "models_dir": str(s5),
            "aggregate": {
                "global_sel_eff_new": 0.8655,
                "n_gemms_evaluated": 100,
                "n_miss_total": 6,
                "pick_measurement": "rebench",
            },
            "per_cell": [
                {
                    "cell": LARGE,
                    "sel_eff_new": 0.82,
                    "n_eval": 50,
                    "n_miss": 6,
                    "decision": "retrain",
                },
                {"cell": TINY, "sel_eff_new": 0.96, "n_eval": 50, "decision": "pass"},
            ],
        },
    )
    ro.mark_stage_complete(root / "round_1" / "stage04b", None)
    cfg = CONFIG + (f"audit:\n  dir: {audit_dir}\n" if audit_dir else "")
    write(root / "configs" / "20261001_000000_full.source.yaml", cfg)
    write(root / "configs" / "20261001_000000_full.yaml", cfg)
    return root


def benching_audit(root):
    write(root / "audit_set.json", {"schema": "audit-v1", "generations": []})
    write(
        root / "progress.json",
        {
            "what": "generation 0",
            "updated": "2026-10-01T10:00:00",
            "shapes_done": 100,
            "shapes_total": 3456,
            "expected_finish": "2026-10-02T08:00",
            "units": {
                "exclusive": {"shapes_done": 100, "shapes_total": 1536},
                "cu64": {"shapes_done": 0, "shapes_total": 1920},
            },
        },
    )
    return root


def headline(page):
    return page[page.index('id="cap"') : page.index('<details class="insample">')]


def pooled(*pairs):
    return math.exp(sum(n * math.log(v) for v, n in pairs) / sum(n for _, n in pairs))


def improve_row(page, cell):
    m = re.search(
        r'<tr class="[^"]*"><td class="c">\d+</td><td class="c">'
        + re.escape(html.escape(cell))
        + r"</td>(.*?)</tr>",
        page,
    )
    assert m, cell
    return m.group(1)


def test_headline_without_a_deployed_bundle_is_held_out(tmp_path):
    run = staged_run(tmp_path / "run")
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    head = headline(page)
    assert "round_0: staged, not deployed" in head
    assert "deploy gate of round_0 (stage06)" in head and "88.08%" in head
    assert "round_1 stage04b" in head and "86.55%" in head
    assert "1 of 2" in head
    assert "in-sample" not in head.lower()
    # the in-sample mean ((99.1 + 70.2) / 2) only in the in-sample block
    insample = page[page.index('<details class="insample">') :]
    insample = insample[: insample.index("</details>")]
    assert "84.65" not in head and "84.65" in insample
    assert page.index('<h2 id="audit">') < page.index('<h2 id="shipped">')
    assert "staged bundle (round_0, not deployed)" in page
    data = json.loads(page[page.index("const D=") + 8 : page.index(";\nconst TRAJ")])
    assert data["S"]["shipped"]["round"] is None
    assert data["S"]["bundle"]["kind"] == "staged"
    assert data["S"]["bundle"]["sel_eff"] == 88.08
    # held-out per leaf: the gate and the next stage04b pooled
    cell = next(c for c in data["CELLS"] if c["cell"] == LARGE)
    assert cell["held"] == round(pooled((0.80, 30), (0.82, 50)) * 100, 2)
    assert "origami" not in page.lower()
    cellmap = page[page.index('<h2 id="cellmap">') :]
    cellmap = cellmap[: cellmap.index("<h2", 1)]
    assert "<svg" in cellmap and "data:image" not in page
    assert "2 leaf cells of the staged bundle (round_0, not deployed)" in cellmap
    assert f"<title>{LARGE}: held-out {cell['held']:.2f}%" in cellmap


def test_headline_starts_with_a_complete_audit(tmp_path):
    run = staged_run(tmp_path / "run")
    write(
        run / "round_1" / "stage04b" / "audit.json",
        {
            "status": "ok",
            "model_sha256": SHA,
            "what": "stage04b: the model entering the round",
            "n_gemms": 32,
            "groups": {
                "exclusive": {
                    "n_gemms": 32,
                    "n_cells": 2,
                    "cell_mean": 0.9012,
                    "cell_mean_ci": [0.8811, 0.9203],
                    "gemm_geomean": 0.9,
                    "min_cell": {"cell": LARGE, "sel_eff": 0.8377},
                    "n_cells_below_target": 1,
                    "classes": {"pool_limited": 1, "model_limited": 0, "passing": 1},
                    "per_cell": {
                        LARGE: {
                            "n": 16,
                            "sel_eff": 0.8377,
                            "class": "pool_limited",
                            "ceiling_rules": 0.9101,
                            "ceiling_table": 0.9202,
                            "confident": True,
                        },
                        TINY: {"n": 16, "sel_eff": 0.9705, "class": "passing"},
                    },
                }
            },
        },
    )
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    head = headline(page)
    first = head.index("audit set: mean of per-cell selection efficiency")
    assert first < head.index("bundle this report describes")
    assert "90.12% [88.11, 92.03]" in head
    assert "83.77%" in head and "1 of 2" in head
    row = improve_row(page, LARGE)
    assert "<b>kernel pool</b>" in row and "audit: base cell" in row
    assert "<b>none</b>" in improve_row(page, TINY)


def _audit_group(cells, start=0):
    """An evaluated group of {cell: [selection efficiency per GEMM]}."""
    per_gemm, per_cell = [], {}
    for j, (cell, vals) in enumerate(cells.items()):
        per_cell[cell] = {
            "n": len(vals),
            "sel_eff": math.exp(sum(map(math.log, vals)) / len(vals)),
            "class": "model_limited",
        }
        shape = {LARGE: (8192, 8192, 8192), TINY: (16, 16, 16)}[cell]
        per_gemm += [
            [shape[0] + i, shape[1], shape[2], 1, v] for i, v in enumerate(vals)
        ]
    mean = math.exp(
        sum(math.log(c["sel_eff"]) for c in per_cell.values()) / len(per_cell)
    )
    return {
        "n_gemms": len(per_gemm),
        "n_cells": len(per_cell),
        "cell_mean": mean,
        "cell_mean_ci": [mean - 0.01, mean + 0.01],
        "gemm_geomean": mean,
        "min_cell": {"cell": LARGE, "sel_eff": per_cell[LARGE]["sel_eff"]},
        "n_cells_below_target": 2,
        "classes": {"pool_limited": 0, "model_limited": 2, "passing": 0},
        "per_cell": per_cell,
        "per_gemm": per_gemm,
    }


def test_a_partial_audit_says_so_and_compares_only_shared_groups(tmp_path):
    run = staged_run(tmp_path / "run")
    excl0 = _audit_group({LARGE: [0.80] * 8, TINY: [0.90] * 8})
    excl1 = _audit_group({LARGE: [0.86, 0.87, 0.85, 0.86] * 2, TINY: [0.90] * 8})
    write(
        run / "round_0" / "stage06" / "audit.json",
        {
            "status": "ok",
            "status_text": "complete",
            "model_sha256": "ab" * 32,
            "groups": {"exclusive": excl0, "cu64": excl0},
            "group_status": {"exclusive": {"status": "ok"}, "cu64": {"status": "ok"}},
        },
    )
    text = "partial: exclusive complete, 1 context benching"
    write(
        run / "round_1" / "stage04b" / "audit.json",
        {
            "status": "partial",
            "status_text": text,
            "model_sha256": SHA,
            "n_gemms": 16,
            "groups": {"exclusive": excl1, "cu64": None},
            "group_status": {
                "exclusive": {"status": "ok"},
                "cu64": {"status": "benching", "reason": "1 of 3 parts benched"},
            },
        },
    )
    write(
        run / "round_1" / "stage06" / "audit.json",
        {
            "status": "partial",
            "status_text": text,
            "model_sha256": "ef" * 32,
            "groups": {"exclusive": excl1, "cu64": None},
            "group_status": {
                "exclusive": {"status": "ok"},
                "cu64": {"status": "benching"},
            },
        },
    )
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    head = headline(page)
    assert f"exclusive, 16 GEMMs over 2 cells, never trained on; {text}" in head
    assert "audit set: change since round_0's candidate" in head
    assert "1 cell(s) improved, 0 regressed beyond their intervals" in head
    section = page[page.index('<h2 id="audit">') :]
    section = section[: section.index("<h2", 1)]
    rows = re.findall(r"<tr><td>round_1</td><td>\w+</td><td>([^<]*)</td>", section)
    assert rows == [text, text] and "null: benching" in section
    assert "round_0 &rarr; round_1" in section
    pair = re.search(r"round_0 &rarr; round_1</td><td>cu64</td>(.*?)</tr>", section)
    assert pair and "not compared" in pair.group(1) and "partial" in pair.group(1)
    excl = re.search(
        r"round_0 &rarr; round_1</td><td>exclusive</td>(.*?)</tr>", section
    )
    assert excl and LARGE in html.unescape(excl.group(1))


def _weak_analysis():
    from lib import weak_spots as ws

    gap = {
        "m": 8192,
        "n": 8192,
        "k": 8192,
        "batch": 1,
        "cell": LARGE,
        "best_us": 400.0,
        "attainable_us": 40.0,
        "fraction": 0.1,
        "bound": "compute",
        "compute_us": 40.0,
        "memory_us": 1.0,
        "floor_us": 3.0,
        "best_signature": "MT256x256x64 MI16x16x32 NTA0 NTB0",
        **dict(zip(ws.SIGNATURE, (256, 256, 64, 16, 16, 32, 0, 0))),
    }
    return {
        "source": ws.TRAINING_SOURCE,
        "why": None,
        "held_out": False,
        "n_dropped_audit": 0,
        "cells_source": ws.HELD_OUT_SOURCE,
        "cells_held_out": True,
        "cells_why": None,
        "library_digest": "L0",
        "pool_digest": "P0",
        "n_cell_gemms": 16,
        "threshold": 0.3,
        "peaks": {
            "tflops": 100.0,
            "tflops_source": "--peak-tflops",
            "bandwidth_gbs": 1000.0,
            "bandwidth_source": "--peak-bandwidth-gbs",
            "latency_floor_us": 3.0,
            "latency_floor_source": "the best time of the fastest exclusive GEMM",
        },
        "layout": {"arch": "gfx950", "a_type": "bf16_r"},
        "n_gemms": 16,
        "cells": {
            LARGE: {"fraction": 0.2, "n": 8, "n_gaps": 8, "bounds": {"compute": 8}},
            TINY: {"fraction": 0.9, "n": 8, "n_gaps": 0, "bounds": {"latency": 8}},
        },
        "gaps": [gap],
    }


def test_weak_spots_name_each_weak_cell_its_lever_and_the_pool_gaps(
    tmp_path, monkeypatch
):
    from lib import weak_spots as ws

    run = staged_run(tmp_path / "run")
    before = _audit_group({LARGE: [0.80] * 8, TINY: [0.99] * 8})
    after = _audit_group({LARGE: [0.70] * 8, TINY: [0.99, 0.4] + [0.99] * 6})
    after["per_cell"][LARGE]["class"] = "pool_limited"
    after["per_cell"][TINY]["class"] = "passing"
    for path, sha, group in (
        (run / "round_0" / "stage06" / "audit.json", "ab" * 32, before),
        (run / "round_1" / "stage04b" / "audit.json", SHA, after),
    ):
        write(
            path,
            {
                "status": "ok",
                "status_text": "complete",
                "model_sha256": sha,
                "groups": {"exclusive": group},
            },
        )
    seen = {}

    def analyze(run_dir, cfg, **kw):
        seen.update(kw)
        return _weak_analysis()

    monkeypatch.setattr(ws, "analyze", analyze)
    out = tmp_path / "report.html"
    weak_dir = tmp_path / "weak"
    mr.build(str(run), str(out), engine=False, weak_spots_dir=str(weak_dir))
    assert seen["threshold"] == ws.POOL_GAP_FRACTION
    page = out.read_text()
    section = page[page.index('<h2 id="weak">Weak spots</h2>') :]
    section = section[: section.index("<h2", 1)]
    assert section.index("Weak cells (2 of 2)") < section.index("<h3>Pool gaps</h3>")
    large = re.search(
        r'<td class="c">' + re.escape(html.escape(LARGE)) + r"</td>(.*?)</tr>", section
    )
    assert large and "<b>rule+pool</b>" in large.group(1)
    assert "&times;0.8750" in large.group(1) and "regressed" in large.group(1)
    assert "20.00" in large.group(1) and ws.LEVERS["pool"] in large.group(1)
    tiny = re.search(
        r'<td class="c">' + re.escape(html.escape(TINY)) + r"</td>(.*?)</tr>", section
    )
    assert tiny and "<b>passing</b>" in tiny.group(1)
    assert "regression / tail" in tiny.group(1)
    assert "MT256x256x64 MI16x16x32 NTA0 NTB0" in section
    assert "origami" not in page.lower()
    assert "never from the audit set" in section
    assert "held out: it names the cells, never their shapes" in section
    doc = json.loads((weak_dir / ws.POOL_GAPS_JSON).read_text())
    assert doc["n_gaps"] == 1 and doc["gaps"][0]["cell"] == LARGE
    assert doc["bundle"] == "staged bundle (round_0, not deployed)"
    assert (doc["source"], doc["held_out"]) == (ws.TRAINING_SOURCE, False)
    assert (doc["library_digest"], doc["pool_digest"]) == ("L0", "P0")
    weak = json.loads((weak_dir / ws.WEAK_SPOTS_JSON).read_text())
    assert weak["summary"][0]["rule"] == 1 and weak["summary"][0]["pool"] == 1
    assert (weak_dir / ws.POOL_GAPS_CSV).read_text().count("\n") == 2

    plain = tmp_path / "plain"
    mr.build(str(run), str(out), bench=False, weak_spots_dir=str(plain))
    doc = json.loads((plain / ws.POOL_GAPS_JSON).read_text())
    assert doc["gaps"] == [] and "--no-bench" in doc["why"]
    page = out.read_text()
    assert "no roofline: --no-bench skips the exhaustive benches" in page


def test_weak_spots_say_n_a_where_an_old_record_holds_no_per_gemm_results():
    from lib import weak_spots as ws

    old = {"status": "ok", "what": "round_6", "groups": {}}
    old["groups"]["exclusive"] = {
        k: v for k, v in _audit_group({LARGE: [0.84] * 4}).items() if k != "per_gemm"
    }
    rows = ws.weak_cells(old, old)
    page = mr.rs.weak_spots_html(
        rows,
        None,
        {"threshold": 0.3, "cell_floor": 0.85, "tail_floor": 0.5},
        previous="round_6",
    )
    (summary,) = re.findall(r'<tr><td class="c">exclusive</td>(.*?)</tr>', page)
    assert summary.count("<td>n/a</td>") == 3
    (weak,) = re.findall(
        r'<td class="c">' + re.escape(html.escape(LARGE)) + r"</td>(.*?)</tr>", page
    )
    assert weak.count("<td>n/a</td>") == 2
    assert (
        "regressed vs round_6: n/a in exclusive: the before evaluation holds no "
        "per-GEMM results"
    ) in html.unescape(page)
    assert "picks below the floors: n/a in exclusive" in page


def test_the_deploy_gates_name_the_per_cell_findings(tmp_path):
    run = staged_run(tmp_path / "run")
    finding = {
        "cell": LARGE,
        "n": 8,
        "ratio": 0.9,
        "ratio_ci": [0.85, 0.95],
        "p": 0.012,
        "before": 0.9,
        "after": 0.81,
    }
    write(
        run / "round_0" / "stage06" / "cell_gate.json",
        {
            "passed": False,
            "record_only": False,
            "reason": "1 cell(s) significantly worse beyond 1.0%",
            "baseline": {"name": "round_9's bundle"},
            "groups": {
                "exclusive": {
                    "protocol": "v2",
                    "deciding": True,
                    "regressed": [finding],
                    "floor": [finding],
                    "tail": [
                        {
                            "shape": [4100, 4096, 4096, 1],
                            "cell": LARGE,
                            "before": 0.8,
                            "after": 0.2,
                            "confirmed": True,
                            "decides": True,
                        }
                    ],
                    "tail_lifted": [
                        {
                            "shape": [4101, 4096, 4096, 2],
                            "cell": LARGE,
                            "before": 0.3,
                            "after": 0.9,
                            "confirmed": True,
                        }
                    ],
                },
                "cu64": {"protocol": "v1", "deciding": False, "reason": "no group"},
            },
        },
    )
    rows = mr.deploy_gates(str(run))
    (r,) = rows
    assert r["cell_gate"] == "kept it out" and r["cell_against"] == "round_9's bundle"
    kinds = [f["kind"] for f in r["findings"]]
    assert kinds == ["regressed", "floor", "tail", "tail_lifted"]
    assert [f["deciding"] for f in r["findings"]] == [True, True, True, False]
    assert r["findings"][2]["gemm"] == "4100x4096x4096"
    assert r["findings"][3]["gemm"] == "4101x4096x4096 batch 2"
    assert r["findings"][2]["ratio"] == pytest.approx(0.25)
    page = mr.deploy_gates_html(rows)
    assert "kept it out (against round_9&#x27;s bundle)" in page
    assert "<h3>Per-cell gate findings</h3>" in page
    assert page.count('<tr class="lose">') == 3
    assert "&times;0.9000" in page and "20.00" in page and "0.012" in page
    assert "pick collapsed below the tail floor" in page
    assert "pick lifted above the tail floor" in page
    assert "a policy on measured GEMMs" in page
    assert mr._cell_gate_text(None) == "not run"
    assert mr._cell_gate_text({"passed": None}) == "not measured"
    off = {"passed": None, "cell_rules": "off: 0 sign flips"}
    assert mr._cell_gate_text(off) == "not decided (its cell rules were off)"
    assert mr._cell_gate_text({"passed": False, "record_only": True}).startswith(
        "would keep it out"
    )


def test_a_partial_audit_without_the_exclusive_group_is_not_measured(tmp_path):
    run = staged_run(tmp_path / "run")
    text = "partial: cu64 complete, exclusive benching"
    write(
        run / "round_1" / "stage04b" / "audit.json",
        {
            "status": "partial",
            "status_text": text,
            "model_sha256": SHA,
            "groups": {
                "exclusive": None,
                "cu64": _audit_group({LARGE: [0.8] * 2, TINY: [0.9] * 2}),
            },
            "group_status": {
                "exclusive": {"status": "benching"},
                "cu64": {"status": "ok"},
            },
        },
    )
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    head = headline(page)
    assert "not measured" in head
    assert f"{text}; the exclusive group is not evaluated yet" in head
    assert "change since" not in head
    section = page[page.index('<h2 id="audit">') :]
    assert "exclusive null: benching" in section


def _v2(groups, status="ok"):
    return {
        "protocol": "v2",
        "status": status,
        "status_text": "complete" if status == "ok" else "partial: cu64 benching",
        "groups": groups,
        "group_status": {
            k: {"status": "ok" if v else "benching"} for k, v in groups.items()
        },
        "n_gemms": 16,
    }


def test_the_report_shows_v2_when_it_is_complete_and_labels_each_protocol(tmp_path):
    run = staged_run(tmp_path / "run")
    v1_0 = _audit_group({LARGE: [0.80] * 8, TINY: [0.90] * 8})
    v1_1 = _audit_group({LARGE: [0.81] * 8, TINY: [0.90] * 8})
    v2_0 = _audit_group({LARGE: [0.85] * 8, TINY: [0.95] * 8})
    v2_1 = _audit_group({LARGE: [0.90, 0.91, 0.89, 0.90] * 2, TINY: [0.95] * 8})
    base = {"status": "ok", "status_text": "complete", "protocol": "v1", "n_gemms": 16}
    write(
        run / "round_0" / "stage06" / "audit.json",
        dict(
            base,
            model_sha256="ab" * 32,
            groups={"exclusive": v1_0},
            v2=_v2({"exclusive": v2_0}),
        ),
    )
    write(
        run / "round_1" / "stage04b" / "audit.json",
        dict(
            base,
            model_sha256=SHA,
            groups={"exclusive": v1_1},
            v2=_v2({"exclusive": v2_1}),
        ),
    )
    write(
        run / "round_1" / "stage06" / "audit.json",
        dict(
            base,
            model_sha256="ef" * 32,
            groups={"exclusive": v1_1, "cu64": v1_1},
            v2=_v2({"exclusive": v2_1, "cu64": None}, "partial"),
        ),
    )
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    head = headline(page)
    assert f"{v2_1['cell_mean'] * 100:.2f}%" in head
    assert "protocol v2 median of repeated joint benches" in head
    assert (
        "regressed beyond their intervals, over 16 GEMMs both evaluated; "
        "protocol v2 median of repeated joint benches" in head
    )
    section = page[page.index('<h2 id="audit">') :]
    section = section[: section.index("<h2", 1)]
    rows = re.findall(
        r"<tr><td>(round_\d)</td><td>\w+</td><td>[^<]*</td><td>([^<]*)", section
    )
    assert rows == [
        ("round_0", "v2 median of repeated joint benches"),
        ("round_1", "v2 median of repeated joint benches"),
        ("round_1", "v1 single reading "),
    ]
    assert "(v2 partial: cu64 benching)" in section
    pairs = re.findall(
        r"<td>([^<]*) &rarr; ([^<]*)</td><td>(\w+)</td><td>([^<]*)</td>", section
    )
    assert pairs == [
        ("round_0", "round_1", "exclusive", "v2 median of repeated joint benches"),
        ("round_0", "round_1", "cu64", "v1 single reading"),
    ]


def test_audit_results_put_v2_first_and_show_extra_models(tmp_path):
    audit = tmp_path / "audit"
    run = staged_run(tmp_path / "run", audit_dir=audit)
    v1 = _audit_group({LARGE: [0.80] * 8, TINY: [0.90] * 8})
    write(
        run / "round_0" / "stage06" / "audit.json",
        {"status": "ok", "status_text": "complete", "model_sha256": SHA,
         "audit_dir": str(audit), "groups": {"exclusive": v1}},
    )  # fmt: skip
    out = tmp_path / "before.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    head = headline(page)
    assert "identity-limited not recorded" in head and "None" not in head
    assert "protocol v1" in head
    section = page[page.index('<h2 id="audit">') :]
    assert "0 / not recorded / 2 / 0" in section[: section.index("<h2", 1)]

    served = dict(
        v1, classes={"pool_limited": 0, "identity_limited": 1, "model_limited": 1}
    )
    v2 = _audit_group({LARGE: [0.90] * 8, TINY: [0.96] * 8})
    offline = _audit_group({LARGE: [0.93] * 8, TINY: [0.97] * 8})
    elsewhere = _audit_group({LARGE: [0.99] * 8, TINY: [0.99] * 8})
    for g in (v2, offline, elsewhere):
        g["classes"] = dict(served["classes"], passing=0)
    staged = tmp_path / "final" / "staged_for_deploy" / "w" / "m.tilewright.bin"
    base = {
        "status": "ok",
        "status_text": "complete",
        "protocol": "v1",
        "audit_dir": str(audit),
        "pick_measurement": "warmup",
    }
    final = dict(
        base,
        what=staged.name,
        model_sha256="ee" * 32,
        model=str(staged),
        groups={"exclusive": offline},
    )
    results = tmp_path / "results.json"
    write(
        results,
        {
            "models": [
                dict(base, what="round_0", model_sha256=SHA,
                     groups={"exclusive": served}, v2=_v2({"exclusive": v2})),
                dict(final, v2=_v2({"exclusive": offline})),
            ]
        },
    )  # fmt: skip
    v1_only = tmp_path / "v1_only.json"
    write(v1_only, final)
    other = tmp_path / "other.json"
    write(
        other,
        dict(base, what="round_0", model_sha256=SHA, audit_dir=str(tmp_path / "other"),
             groups={"exclusive": elsewhere}, v2=_v2({"exclusive": elsewhere})),
    )  # fmt: skip
    out = tmp_path / "after.html"
    argv = [str(run), str(out), "--no-bench"]
    for path in (v1_only, results, other):
        argv += ["--audit-results", str(path)]
    assert mr.main(argv) == 0
    page = out.read_text()
    head = headline(page)
    assert f"{v2['cell_mean'] * 100:.2f}%" in head
    assert "[95% CI], protocol v2" in head and "protocol v1" not in head
    assert "audit set audit: extra model final (no round of this run)" in head
    assert "not measured" not in head
    assert f"{offline['cell_mean'] * 100:.2f}%" in head
    assert "paired against round_0&#x27;s candidate: \u00d7" in head
    assert "identity-limited not recorded" not in head
    assert (
        "audit set other: extra model round_0 (round_0&#x27;s bundle, evaluated on "
        "audit set other, not audit)" in head
    )
    assert (
        "not compared with round_0&#x27;s candidate: evaluated on audit set other, "
        "not audit" in head
    )
    data = json.loads(page[page.index("const D=") + 8 : page.index(";\nconst TRAJ")])
    bundle, extras = data["AUDIT"]["bundle"], data["AUDIT"]["extra"]
    assert (bundle["round"], bundle["protocol"]) == ("round_0", "v2")
    assert bundle["cell_mean"] == round(v2["cell_mean"] * 100, 2)
    extra, apart = extras
    assert (extra["model"], extra["protocol"]) == ("final", "v2")
    assert (extra["audit_set"], extra["source"]) == ("audit", str(results))
    assert extra["cell_mean"] == round(offline["cell_mean"] * 100, 2)
    assert extra["vs_bundle"]["protocol"] == "v2"
    assert extra["vs_bundle"]["cell_mean_ratio"] > 1
    assert (apart["model"], apart["audit_set"], apart["vs_bundle"]) == (
        "round_0",
        "other",
        None,
    )
    assert apart["extra_reason"] == (
        "round_0's bundle, evaluated on audit set other, not audit"
    )
    section = page[page.index('<h2 id="audit">') :]
    section = section[: section.index("<h2", 1)]
    assert re.findall(r"<tr><td>(round_\d|extra)</td>", section) == [
        "round_0",
        "extra",
        "extra",
    ]
    assert "<td>extra</td><td>final " in section and "results.json</td>" in section
    assert "(audit set audit; no round of this run; sha256 eeeeeeeeeeee)" in section
    assert "round_0 &rarr; extra: final" in section
    pair = re.search(r"round_0 &rarr; extra: round_0</td>(.*?)</tr>", section)
    assert pair and "not compared: evaluated on audit set other" in pair.group(1)


def test_extra_model_tiles_classify_at_the_records_target(tmp_path):
    audit = tmp_path / "audit"
    run = staged_run(tmp_path / "run", audit_dir=audit)
    group = _audit_group({LARGE: [0.92] * 8, TINY: [0.97] * 8})
    for c in group["per_cell"].values():
        c.update(ceiling_rules=0.99, ceiling_served=0.99)
    group["n_cells_below_target"] = 0
    staged = tmp_path / "final" / "staged_for_deploy" / "w" / "m.tilewright.bin"
    results = tmp_path / "results.json"
    write(
        results,
        {"status": "ok", "status_text": "complete", "target": 0.9,
         "audit_dir": str(audit), "model": str(staged), "model_sha256": "ee" * 32,
         "groups": {"exclusive": group}},
    )  # fmt: skip
    out = tmp_path / "report.html"
    argv = [str(run), str(out), "--no-bench", "--audit-results", str(results)]
    assert mr.main(argv) == 0
    page = out.read_text()
    head = headline(page)
    assert (
        "0 of 2 cells below 90% (pool-limited 0, identity-limited 0, "
        "model-limited 0, passing 2)" in head
    )
    data = json.loads(page[page.index("const D=") + 8 : page.index(";\nconst TRAJ")])
    (extra,) = data["AUDIT"]["extra"]
    assert extra["classes"] == {
        "pool_limited": 0,
        "identity_limited": 0,
        "model_limited": 0,
        "passing": 2,
    }


def test_audit_labels_rename_extra_models(tmp_path):
    audit = tmp_path / "audit"
    run = staged_run(tmp_path / "run", audit_dir=audit)
    base = {"status": "ok", "status_text": "complete", "audit_dir": str(audit)}
    own = _audit_group({LARGE: [0.80] * 8, TINY: [0.90] * 8})
    write(
        run / "round_0" / "stage06" / "audit.json",
        dict(base, model_sha256=SHA, groups={"exclusive": own}),
    )
    better = _audit_group({LARGE: [0.93] * 8, TINY: [0.97] * 8})
    models = [
        dict(
            base,
            model=str(tmp_path / d / "staged_for_deploy" / "m.tilewright.bin"),
            model_sha256=sha * 32,
            groups={"exclusive": better},
        )
        for d, sha in (("a", "ee"), ("b", "dd"))
    ]
    results = tmp_path / "results.json"
    write(results, {"models": models})
    out = tmp_path / "report.html"
    argv = [str(run), str(out), "--no-bench", "--audit-results", str(results)]
    argv += ["--audit-label", "a=offline A", "--audit-label", "DDDDDD=offline B"]
    assert mr.main(argv + ["--audit-label", "nothing=X"]) == 0
    page = out.read_text()
    head = headline(page)
    for name in ("offline A", "offline B"):
        assert f"audit set audit: extra model {name} (no round of this run)" in head
    data = json.loads(page[page.index("const D=") + 8 : page.index(";\nconst TRAJ")])
    assert [e["model"] for e in data["AUDIT"]["extra"]] == ["offline A", "offline B"]
    section = page[page.index('<h2 id="audit">') :]
    section = section[: section.index("<h2", 1)]
    assert "<td>extra</td><td>offline A " in section
    assert "round_0 &rarr; extra: offline B" in section
    assert "--audit-label nothing: no extra model has this label" in page
    for bad in ("offline", "=X", "a="):
        with pytest.raises(SystemExit):
            mr.main(argv[:2] + ["--audit-label", bad])


def test_an_audit_label_names_an_extra_model_by_label_or_sha256_prefix():
    rec = {"model_sha256": "ab" * 32}
    recs = [("round_0", "candidate", rec), ("extra: x", "extra", rec)]
    assert mr.label_extras(recs, {"ababa": "short", "round_0": "r"}) == (
        recs,
        ["ababa", "round_0"],
    )
    renamed, unused = mr.label_extras(recs, {"x": "first", "ABABAB": "second"})
    assert renamed == [recs[0], ("extra: first", "extra", rec)]
    assert unused == ["ABABAB"]


def test_where_to_improve_acts_on_held_out_evidence(tmp_path):
    run = staged_run(tmp_path / "run")
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    large, tiny = improve_row(page, LARGE), improve_row(page, TINY)
    # in-sample says the opposite of the held-out measurements
    assert "<b>retrain</b>" in large and "99.10" in large
    assert (
        f"held-out {pooled((0.80, 30), (0.82, 50)) * 100:.1f}% over 80 GEMMs" in large
    )
    assert "measured: deploy_gate round_0 + stage04b round_1" in large
    assert "retrain (round_1)" in large
    assert "<b>none</b>" in tiny and "70.20" in tiny
    assert "2 measured" in page and "never choose the action" in page


def test_missing_values_say_why(tmp_path):
    audit = benching_audit(tmp_path / "audit")
    run = staged_run(tmp_path / "run", audit_dir=audit)
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    assert (
        "no stage 7 results in this run: stages 7 and 8 run after the last planned "
        "round (1 of 8 planned rounds complete) and on every later --mode full until "
        "they pass, or with --mode validate; stage 7 validates a deployed bundle and "
        "none was deployed; without stage07.build_dir, or deploy.apply_patch and "
        "deploy.build_after_apply, it can only run after a manual deploy and model "
        "build." in page
    )
    assert "no stage 7 timing logs in this run: stages 7 and 8 run after" in page
    assert "no stage 8 results in this run: stage08.datasets is empty" in page
    assert "shipped_subset.json does not exist" in page
    assert "the identity numbers come from the bench analysis, which --no-bench" in page
    status = (
        "the audit set is still being benched: 100 of 3,456 GEMMs (exclusive 100 of "
        "1,536; 1 contexts 0 of 1,920), expected to finish 2026-10-02T08:00"
    )
    assert status in headline(page) and f"Audit set now:</b> {status}" in page
    gates = page[page.index("Deploy gate per round") :]
    assert "88.08% on the gate GEMMs is below the target" in gates
    # every value the headline cannot give comes with its reason
    tiles = re.findall(
        r'<div class="v">([^<]*)</div><div class="l">[^<]*</div>'
        r'(?:<div class="n">([^<]*)</div>)?',
        headline(page),
    )
    missing = [(v, note) for v, note in tiles if v in ("n/a", "none", "not measured")]
    assert missing and all(note.strip() for _, note in missing), missing
    shutil.rmtree(run / "round_1")
    mr.build(str(run), str(out), bench=False)
    page = out.read_text()
    assert "round_1 has not started (1 of 8 planned rounds complete)" in headline(page)


# ── sortable tables ─────────────────────────────────────────────────────────

FAKE_DOM = r"""
class El {
  constructor(tag, o = {}) {
    this.tagName = tag.toUpperCase(); this.children = []; this.parentElement = null;
    this.dataset = {}; this.attrs = {}; this.cls = o.cls || ""; this.text = o.text || "";
    this.colSpan = o.colSpan || 1; this.rowSpan = o.rowSpan || 1;
  }
  get classList() { const c = this.cls.split(/\s+/); return { contains: (x) => c.includes(x) }; }
  set className(v) { this.cls = v; }
  get textContent() { return this.text + this.children.map((c) => c.textContent).join(""); }
  set textContent(v) { this.text = v; this.children = []; }
  appendChild(n) {
    if (n.tagName === "#FRAG") { for (const c of [...n.children]) this.appendChild(c); return n; }
    if (n.parentElement) n.remove();
    n.parentElement = this; this.children.push(n); return n;
  }
  remove() {
    const p = this.parentElement;
    if (p) { p.children.splice(p.children.indexOf(this), 1); this.parentElement = null; }
  }
  closest(sel) {
    const want = sel.split(",").map((s) => s.trim().toUpperCase());
    for (let n = this; n; n = n.parentElement) if (want.includes(n.tagName)) return n;
    return null;
  }
  querySelector(sel) {
    for (const c of this.children) {
      if ((sel === ".sortind" && c.cls === "sortind") || c.tagName === sel.toUpperCase()) return c;
      const r = c.querySelector(sel); if (r) return r;
    }
    return null;
  }
  setAttribute(k, v) { this.attrs[k] = v; }
  removeAttribute(k) { delete this.attrs[k]; }
  get rows() {
    return this.tagName === "TABLE" ? this.children.flatMap((s) => s.rows)
      : this.children.filter((c) => c.tagName === "TR");
  }
  get cells() { return this.children.filter((c) => c.tagName === "TD" || c.tagName === "TH"); }
  get tHead() { return this.children.find((c) => c.tagName === "THEAD") || null; }
  get tBodies() { return this.children.filter((c) => c.tagName === "TBODY"); }
}
global.document = {
  addEventListener() {},
  createDocumentFragment: () => new El("#frag"),
  createElement: (t) => new El(t),
};
function mk(tag, o, kids) { const e = new El(tag, o); for (const k of kids || []) e.appendChild(k); return e; }
const tr = (cells, cls) => mk("tr", { cls }, cells);
const th = (text, o) => mk("th", Object.assign({ text }, o));
const td = (text, o) => mk("td", Object.assign({ text }, o));
const col = (t, i) => t.tBodies[0].rows.map((r) => r.cells[i].textContent);
"""

SORT_CHECKS = r"""
const out = {};
const heldTh = th("held-out %"), gemmsTh = th("GEMMs"), groupTh = th("values", { colSpan: 2 });
const cellTh = th("cell", { rowSpan: 2 });
const t1 = mk("table", {}, [
  mk("thead", {}, [tr([cellTh, groupTh]), tr([heldTh, gemmsTh])]),
  mk("tbody", {}, [
    tr([td("all"), td("90.00"), td("3,201")], "total"),
    tr([td("b"), td("93.2 [92.1, 94.0]"), td("1,189")]),
    tr([td("a"), td("\u2013"), td("12")]),
    tr([td("c"), td(PP_DELTA), td("n/a")]),
    tr([td("d"), td("88.0*"), td("2,000")]),
    tr([td("e"), td("88.0*"), td("7")]),
  ]),
]);
TWSORT.sortBy(heldTh); out.held_asc = col(t1, 0);
out.ind_asc = heldTh.textContent;
TWSORT.sortBy(heldTh); out.held_desc = col(t1, 0);
out.ind_desc = heldTh.textContent;
TWSORT.sortBy(groupTh); out.group = col(t1, 0);
TWSORT.sortBy(gemmsTh); out.gemms = col(t1, 0);
out.held_after = heldTh.textContent; out.gemms_ind = gemmsTh.textContent;
TWSORT.sortBy(cellTh); out.cells = col(t1, 0);
out.head = t1.tHead.rows.map((r) => r.cells.length);
const sizeTh = th("size"), p50Th = th("p50 us");
const t2 = mk("table", {}, [mk("tbody", {}, [
  tr([sizeTh, p50Th]),
  tr([td("64"), td("12.5")]), tr([td("all"), td("3.1")]), tr([td("8"), td("\u2014")]),
])]);
TWSORT.sortBy(p50Th); out.nohead_p50 = col(t2, 0);
TWSORT.sortBy(sizeTh); out.nohead_size = col(t2, 0);
const xTh = th("x");
const t3 = mk("table", {}, [mk("thead", {}, [tr([xTh, th("y"), th("z")])]),
  mk("tbody", {}, [tr([td("nothing to show", { colSpan: 3 })])])]);
TWSORT.sortBy(xTh); out.placeholder = col(t3, 0);
const keys = {};
for (const t of KEYS) { const k = TWSORT.key(t); keys[t] = k === null ? "missing" : k.n; }
out.keys = keys;
console.log(JSON.stringify(out));
"""


def _signed_pp(value):
    """A percentage-point delta as the report writes one."""
    return f"+{value} pp"


KEYS = {
    "93.2 [92.1, 94.0]": 93.2,
    "1,189": 1189,
    "12,345.5": 12345.5,
    _signed_pp("3.76"): 3.76,
    "-2.03": -2.03,
    "\u22121.5": -1.5,
    "20.9%": 20.9,
    "2.5 MB": 2.5,
    "7 us": 7,
    "6.75 \u00b5s": 6.75,
    "88.81*": 88.81,
    "3 / 96": 3,
    "1e-3": 0.001,
    "1h25m": 5100,
    "12m26s": 746,
    "8s": 8,
    "4m39s (rc=1)": 279,
    "\u2013": "missing",
    "\u2014": "missing",
    "-": "missing",
    "n/a": "missing",
    "N/A": "missing",
    "": "missing",
    "16x16": None,
    "4,0": None,
    "round_2": None,
    "MT64x64 DU64": None,
}


def sort_script(page):
    start = page.index("const TWSORT=")
    return page[start : page.index("\n})();", start) + 6]


def test_every_table_is_sortable(tmp_path):
    run = tmp_path / "run"
    make_run(run)
    out = tmp_path / "report.html"
    mr.build(str(run), str(out))
    page = out.read_text()
    script = sort_script(page)
    assert page.count('document.addEventListener("click"') == 1
    assert 'id="chead"' not in page
    # every table, in the page and in the script's templates, has a header
    # row the document-wide sorter recognizes
    follow = re.findall(r"<table\b[^>]*>(.{7})", page)
    assert len(follow) >= 5 and set(follow) == {"<thead>"}, Counter(follow)
    assert (
        '<tr class="total" style="font-weight:600"><td class="c">all invocations'
        in (page)
    )
    assert '<tr class="total"' in page[page.index("const TRAJ") :]
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs node to run the page's sorting script")
    js = tmp_path / "sort.js"
    js.write_text(
        FAKE_DOM
        + script
        + "\nconst KEYS = "
        + json.dumps(list(KEYS))
        + ";\nconst PP_DELTA = "
        + json.dumps(_signed_pp("3.5"))
        + ";\n"
        + SORT_CHECKS
    )
    proc = subprocess.run([node, str(js)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["held_asc"] == ["all", "c", "d", "e", "b", "a"]
    assert got["held_desc"] == ["all", "b", "d", "e", "c", "a"]
    assert got["ind_asc"].endswith("\u25b2") and got["ind_desc"].endswith("\u25bc")
    assert got["group"] == got["held_desc"]
    assert got["gemms"] == ["all", "e", "a", "b", "d", "c"]
    assert got["held_after"] == "held-out %" and got["gemms_ind"].endswith("\u25b2")
    assert got["cells"] == ["all", "a", "b", "c", "d", "e"]
    assert got["head"] == [2, 2]
    assert got["nohead_p50"] == ["size", "all", "64", "8"]
    assert got["nohead_size"] == ["size \u25b2", "8", "64", "all"]
    assert got["placeholder"] == ["nothing to show"]
    for text, want in KEYS.items():
        v = got["keys"][text]
        if isinstance(want, float):
            assert v == pytest.approx(want), text
        else:
            assert v == want, text


# ── kernel identity ─────────────────────────────────────────────────────────

IDENT_KERNELS = {
    0: ({}, {"tile_processing_strategy": 0, "work_assignment": 0, "work_group_x": 0}),
    1: ({}, {"tile_processing_strategy": 2, "work_assignment": 2, "work_group_x": 256}),
    2: (
        {"mt_n": 32},
        {"tile_processing_strategy": 0, "work_assignment": 0, "work_group_x": 0},
    ),
    3: (
        {"grvw_a": 4},
        {"tile_processing_strategy": 0, "work_assignment": 0, "work_group_x": 0},
    ),
}
SEED = {64: {0: 10.0, 1: 5.0, 2: 8.0, 3: 10.0}, 128: {1: 6.0, 2: 7.0, 3: 9.0}}


def bench_dir(path, latencies):
    from test_report_sections import row, write_rows

    rows = [
        row(m, 4096, 4096, sid, us, **IDENT_KERNELS[sid][0])
        for m, per in latencies.items()
        for sid, us in per.items()
    ]
    write_rows(path / "chunk_0000.csv", rows)
    names = ["tile_processing_strategy", "work_assignment", "work_group_x"]
    write(
        path / "kernels.json",
        {
            "attribute_names": names,
            "kernels": {
                str(s): [a[n] for n in names] for s, (_, a) in IDENT_KERNELS.items()
            },
        },
    )


@pytest.mark.parametrize("planner", ["stage02", "stage01b"])
def test_identity_numbers_of_the_run(tmp_path, planner):
    """Of round 0's seed bench in the planner's directory: stage02/, or
    stage01b/ in a run written before the planner became stage02."""
    from lib import evaluate as ev
    from lib import hipblaslt_launch as hl
    from lib import identity_ceiling as ic
    from lib import run_analysis as ra
    from lib import run_layout

    run = tmp_path / "run"
    if planner == "stage02":
        run_layout.settle(run)
    bench_dir(run / "round_0" / "stage04", SEED)
    bench_dir(run / "round_0" / planner / "seed" / "stage04", SEED)
    cell = "Small|Large|LargeK|Bnone"
    write(
        run / "round_0" / "stage05" / "cells.json",
        {"cells": [{"label": cell, "n_gemms": 2, "best_sel_eff": 0.9}]},
    )
    write(run / "round_0" / "stage05" / "metrics.json", {"args": {"arch": "gfx950"}})
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), engine=False)
    page = out.read_text()
    ident = page[page.index('<h2 id="identity">') : page.index('<h2 id="kernels">')]

    bench = ra.load_bench([run / "round_0" / planner / "seed" / "stage04"])
    pool = ra.pool_for(bench, None).kernels
    gemms = ra.gemm_dicts([(run / "round_0" / planner / "seed" / "stage04", None)])
    pools = ev.pools_for(gemms, ev.pool_from_kernels(pool))
    sig = ic.ceiling(gemms, ic.engine_grouping(), "s", pools)
    launch = ic.ceiling(
        gemms, ic.engine_grouping(launch=hl.HipblasltLaunch("gfx950")), "l", pools
    )
    members = Counter(
        tuple(k[f] for f in ("mt_m", "mt_n", "mt_k", "mi_m", "mi_n", "mi_k"))
        + (k["cache_hints_a"], k["cache_hints_b"])
        for k in pool
    )
    assert sorted(members.values()) == [1, 3]
    assert sig.geomean < launch.geomean == 1.0
    assert (
        "the 4 measured solutions (the library pool and bank could not be read) "
        "form 2 keys: a key holds up to 3 and on average 2.00 members, and 1 key "
        "(3 solutions) holds more than one." in ident
    )
    assert "TieBreak::PoolOrder" in ident and "launch_keys</code>, off" in ident
    assert "On the 2 GEMMs of round 0&#x27;s exhaustive seed bench" in ident
    assert f"<b>{sig.geomean * 100:.2f}%</b>" in ident
    assert f"{sig.n_losing} GEMM loses more than 0.1%" in ident
    assert (
        f"With launch keys it would reach <b>{launch.geomean * 100:.2f}%</b>" in ident
    )
    assert "Not the audit set: no audit set is configured (audit.dir)" in ident
    assert "<td>3</td>" in ident and "<td>2.00</td>" in ident
    pool = page[page.index('<h2 id="pool">') : page.index('<h2 id="identity">')]
    assert "the run&#x27;s bundle names no library" in pool


def test_kernel_pool_of_the_run(tmp_path):
    """What the pool the bundle ranks is made of, per kind, from the library,
    config and device the bundle's stage05 ranked with."""
    import msgpack
    import yaml

    from bench_fakes import solution, write_logic

    stem = (
        "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_"
        "Dijk_gfx950"
    )
    bank = stem.replace("_gfx950", "_CU256_gfx950")
    lib = tmp_path / "lib"
    lib.mkdir()
    write_logic(lib / f"{stem}.dat", [solution(i, i) for i in (3, 1, 2)], [2, 1])
    on_256 = {"type": "AMDGPU", "value": {"type": "CUCount", "value": 256}}
    sols = [
        dict(solution(i, i, mt=(64, 64, 64)), hardwarePredicate=on_256) for i in (5, 4)
    ]
    table = [{"key": [i], "index": i} for i in (5, 4)]
    row = {
        "predicate": {"type": "EqualityMatching"},
        "library": {"type": "Matching", "table": table},
    }
    doc = {"solutions": sols, "library": {"type": "Problem", "rows": [row]}}
    (lib / f"{bank}.dat").write_bytes(msgpack.packb(doc))
    hardware = {"n_cu": 256, "lds_bytes": 65536, "l2_bytes": 4 << 20}
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "arch": "gfx950",
                "hardware": dict(hardware, pci_chip_id=0x75A0),
                "library": {"kind": ["origami", "bok"]},
                "train": {"catalog": "v3"},
            }
        )
    )
    run = tmp_path / "run"
    write(
        run / "round_0" / "stage05" / "cells.json",
        {"cells": [{"label": "Small|Large|LargeK|Bnone", "n_gemms": 2}]},
    )
    args = {
        "arch": "gfx950",
        "library_dir": str(lib),
        "library_stem": stem,
        "config_yaml": str(cfg),
        "hardware_device": 0,
    }
    write(run / "round_0" / "stage05" / "metrics.json", {"args": args})
    out = tmp_path / "report.html"
    mr.build(str(run), str(out), engine=False, bench=False)
    page = out.read_text()
    pool = page[page.index('<h2 id="pool">') : page.index('<h2 id="identity">')]
    assert "library.kind [origami, bok] with bok rows freesize, equality" in pool
    assert "on the bench device (gfx950 with 256 CUs, chip 0x75a0)" in pool
    assert "<b>4</b> kernels in" in pool
    assert '<td class="c">origami</td>' in pool and '<td class="c">bok</td>' in pool
    assert "Prediction table 2" in pool and "equality 2" in pool
    assert f"{stem} (no rows of the bank&#x27;s kinds)" in pool
