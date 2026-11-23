# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Repeated measurements of one (GEMM, kernel): the label rule, the loader
and the evaluation that use it, and its record among the model options."""

import csv
import json

import pytest

import run_orchestrator as ro
import stage05_train as s5
from lib import evaluate as ev
from lib import repeats
from test_stage05_train import COLUMNS, gemm_rows, kernel_pool


def row(sid, us, skip=False, origami=None, **kw):
    return dict(sol_idx_global=sid, us=us, is_skip=skip, is_origami_pick=origami, **kw)


@pytest.mark.parametrize(
    "aggregation, values, want",
    [
        ("median", [10.0], 10.0),
        ("median", [12.0, 10.0], 11.0),
        ("median", [12.0, 9.0, 10.0], 10.0),
        ("median", [9.0, 14.0, 10.0, 11.0], 10.5),
        ("median_low", [12.0, 10.0], 10.0),
        ("median_low", [9.0, 14.0, 10.0, 11.0], 10.0),
        ("min", [12.0, 9.0, 10.0], 9.0),
    ],
)
def test_aggregate(aggregation, values, want):
    assert repeats.aggregate(values, aggregation) == want


def test_unknown_rules_and_empty_readings_are_errors():
    with pytest.raises(ValueError, match="not one of"):
        repeats.aggregate([1.0], "mean")
    with pytest.raises(ValueError, match="no readings"):
        repeats.aggregate([], "median")
    with pytest.raises(ValueError, match="not one of"):
        repeats.config_aggregation({"train": {"repeat_aggregation": "max"}})
    assert repeats.config_aggregation({}) == repeats.DEFAULT_AGGREGATION == "median"
    assert repeats.config_aggregation({"train": {"repeat_aggregation": "min"}}) == "min"


def test_tested_rows_label_a_kernel_before_skip_rows():
    rows = [
        row(7, 30.0, skip=True),
        row(7, 12.0, rank=4, cv=0.03, status="noisy"),
        row(7, 10.0, rank=2, cv=0.01, status="converged"),
        row(7, 11.0, rank=3, cv=0.02, status="converged"),
    ]
    merged = repeats.merge_rows(rows, "median")
    assert merged["us"] == 11.0 and merged["n_repeats"] == 3
    assert (merged["rank"], merged["cv"], merged["is_skip"]) == (3, 0.02, False)
    low = repeats.merge_rows(rows[:3], "median")
    assert low["us"] == 11.0 and low["rank"] == 2
    assert repeats.merge_rows(rows[:3], "median_low")["us"] == 10.0
    fastest = repeats.merge_rows(rows, "min")
    assert (fastest["us"], fastest["status"]) == (10.0, "converged")
    screened = repeats.merge_rows([row(7, 30.0, skip=True), row(7, 20.0, True)], "min")
    assert (screened["us"], screened["is_skip"]) == (20.0, True)
    assert rows[1]["us"] == 12.0 and "n_repeats" not in rows[1]


def test_merged_candidates_keep_first_seen_order_and_the_origami_pick():
    single = row(5, 4.0, origami=False)
    cands = [row(9, 3.0, origami=True), single, row(9, 2.0, origami=False)]
    merged = repeats.merge_candidates(cands, "median")
    assert [c["sol_idx_global"] for c in merged] == [9, 5]
    assert merged[1] is single
    assert merged[0]["us"] == 2.5 and merged[0]["is_origami_pick"] is True
    none = repeats.merge_candidates([row(1, 2.0), row(1, 3.0)], "median")
    assert none[0]["is_origami_pick"] is None


def write_rows(path, rows):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "chunk_0000.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.mark.parametrize("aggregation", repeats.AGGREGATIONS)
def test_the_loader_merges_repeated_rows_by_the_rule(tmp_path, aggregation):
    """Two rounds benched one shape: a kernel measured in both gets the
    rule's label, the others keep their only row."""
    pool = kernel_pool()[:4]
    shape = (512, 512, 512, 1)
    first = gemm_rows(shape, pool)
    second = gemm_rows(shape, pool)
    sid = int(first[0]["sol_idx_global"])
    for r, us, cv, status in ((first[0], 10.0, 0.01, "converged"),):
        r.update(us=us, is_skip="False", cv=cv, status=status)
    for r in second:
        if int(r["sol_idx_global"]) == sid:
            r.update(us=14.0, is_skip="False", cv=0.04, status="noisy")
    a = write_rows(tmp_path / "a", first)
    b = write_rows(tmp_path / "b", second)
    cells, stats = s5.load_enriched_chunks([a, b], True, aggregation=aggregation)
    ((g,),) = cells.values()
    by = {c["sol_idx_global"]: c for c in g["candidates"]}
    assert len(by) == len(g["candidates"]) == len(pool)
    want = {"median": 12.0, "median_low": 10.0, "min": 10.0}[aggregation]
    assert by[sid]["us"] == want and by[sid]["n_repeats"] == 2
    assert by[sid]["cv"] == 0.01 and by[sid]["status"] == "converged"
    assert stats["rows_merged"] == len(pool) and stats["kernels_repeated"] >= 1
    tested = [c["us"] for c in g["candidates"] if not c["is_skip"]]
    assert g["winner_us"] == min(tested)
    with pytest.raises(ValueError, match="not one of"):
        s5.load_enriched_chunks([a], True, aggregation="mean")


def test_measurements_merge_repeated_rows():
    g = {
        "candidates": [
            row(1, 10.0),
            row(1, 14.0),
            row(1, 11.0),
            row(2, 9.0, skip=True),
            row(2, 8.0, skip=True),
            row(3, 7.0),
            row(3, 30.0, skip=True),
            row(4, -1.0),
        ]
    }
    timed, screened = ev.measurements(g)
    assert timed == {1: 11.0, 3: 7.0} and screened == {2: 8.5}
    timed, screened = ev.measurements(g, "min")
    assert timed == {1: 10.0, 3: 7.0} and screened == {2: 8.0}


def test_timings_take_the_rule_of_the_caller(tmp_path):
    pool = kernel_pool()[:3]
    shape = (256, 256, 256, 1)
    rows = gemm_rows(shape, pool)
    for r in rows:
        r["is_skip"] = "False"
    again = [dict(r, us=f"{float(r['us']) * 2:.4f}") for r in rows]
    d = write_rows(tmp_path / "t", rows + again)
    key = (*shape, "T", "N")
    lo = min(float(r["us"]) for r in rows)
    assert ev.load_timings([d], aggregation="min")[key]["winner_us"] == lo
    assert ev.load_timings([d])[key]["winner_us"] == pytest.approx(lo * 1.5)


def test_options_recorded_before_the_rule_was_one_were_trained_with_the_minimum(
    tmp_path,
):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("train:\n  repeat_aggregation: median_low\n")
    base = ["--enriched-csv-dir", "d", "--output-dir", "o", "--arch", "gfx950"]
    base += ["--config-yaml", str(cfg)]
    assert s5.command_model_options(base)["repeat_aggregation"] == "min"
    given = s5.command_model_options(base + ["--repeat-aggregation", "median"])
    assert given["repeat_aggregation"] == "median"
    args = s5.arg_parser().parse_args(base)
    s5.resolve_options(args, s5.run_config(cfg))
    assert args.repeat_aggregation == "median_low"
    current = dict(given)
    old = {k: v for k, v in current.items() if k != "repeat_aggregation"}
    assert s5.changed_model_options(old, current) == {
        "repeat_aggregation": ("min", "median")
    }
    assert s5.changed_model_options(current, current) == {}


def test_the_orchestrator_passes_the_configured_rule(tmp_path):
    from test_run_orchestrator import make_config

    cfg, raw = ro.load_config(make_config(tmp_path))
    assert ro.validate_config(cfg)[0] == []
    snapshot = ro.write_snapshot(tmp_path / "run", cfg, raw, "full")
    pipe = ro.Pipeline(
        cfg=cfg,
        snapshot=snapshot,
        run_dir=tmp_path / "run",
        paths=ro.derive_paths(cfg),
        stages_dir=ro.THIS_DIR / "stages",
        timing=ro.Timing(),
        force_deploy=False,
    )
    step = ro.RoundStep(0, "initial", [], None, False)
    args = pipe.args_stage05(tmp_path / "run" / "round_0", step, None)
    assert args[args.index("--repeat-aggregation") + 1] == "median"
    cfg["train"] = dict(cfg.get("train") or {}, repeat_aggregation="min")
    assert ro.validate_config(cfg)[0] == []
    pipe.cfg = cfg
    args = pipe.args_stage05(tmp_path / "run" / "round_0", step, None)
    assert args[args.index("--repeat-aggregation") + 1] == "min"
    cfg["train"]["repeat_aggregation"] = "mean"
    errors = ro.validate_config(cfg)[0]
    assert any("train.repeat_aggregation" in e for e in errors), json.dumps(errors)
