# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import subprocess
import sys
from pathlib import Path

import pytest

import make_report as mr

EVIL = '</script><script>alert(1)</script><img src=x onerror="alert(2)">'


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
        "contexts:\n  list:\n    - {cu_budget: 128, name: half}\n"
        "  holdout:\n    - {cu_budget: 64, schedule: dynamic}\n"
    )
    assert names[cu128] == "half"
    assert names[held].endswith("held out")


def test_diagnostics_flag(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    proc = subprocess.run(
        [sys.executable, str(Path(mr.__file__)), str(run), "--diagnostics"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
