# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`reproduce/score_predictions.py` asks whether the engine predicted best ran best."""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

from uhd_gen.reproduce.score_predictions import main  # noqa: E402


def _measured(path, engine, metric, rows):
    """An L1 corpus.csv of (graph, tflops, avgTimeMs) rows collected under `metric`."""
    binding = json.dumps({"engine": engine, "metric": metric})
    pd.DataFrame(
        [
            {
                "engine_name": engine,
                "benchmark": f"id-{graph}",
                "binding": binding,
                "tflops": tflops,
                "avgTimeMs": ms,
            }
            for graph, tflops, ms in rows
        ]
    ).to_csv(path, index=False)
    return path


def _run(monkeypatch, tmp_path, predictions, measured, *extra):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "graphs": [
                    {"benchmark": f"id-{g}", "name": g, "regime": regime}
                    for g, regime in (("g1", "short"), ("g2", "long"), ("g3", "long"))
                ]
            }
        ),
        encoding="utf-8",
    )
    predicted = tmp_path / "predictions.json"
    predicted.write_text(json.dumps(predictions), encoding="utf-8")
    report = tmp_path / "report.json"
    arguments = ["score_predictions", "--manifest", str(manifest)]
    arguments += ["--predictions", str(predicted), "--report", str(report), *extra]
    for path in measured:
        arguments += ["--measured", str(path)]
    monkeypatch.setattr("sys.argv", arguments)
    code = main()
    rows = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
    return code, rows


def _predict(engine, metric, values):
    return [
        {"graph": graph, "engine": engine, "metric": metric, "value": value}
        for graph, value in values.items()
    ]


def test_a_wrong_pick_costs_the_shortfall_of_the_engine_it_chose(monkeypatch, tmp_path):
    measured = [
        _measured(tmp_path / "a.csv", "A", "tflops", [("g1", 100, 1.0), ("g2", 50, 2.0)]),
        _measured(tmp_path / "b.csv", "B", "tflops", [("g1", 80, 1.2), ("g2", 200, 0.5)]),
    ]
    predictions = _predict("A", "tflops", {"g1": 90, "g2": 90}) + _predict(
        "B", "tflops", {"g1": 70, "g2": 60}
    )
    code, rows = _run(monkeypatch, tmp_path, predictions, measured)
    assert code == 0
    by_graph = {row["graph"]: row for row in rows}
    assert by_graph["g1"]["agree"] and by_graph["g1"]["regret"] == 0
    assert by_graph["g2"]["predicted_winner"] == "A"
    assert by_graph["g2"]["measured_winner"] == "B"
    assert by_graph["g2"]["regret"] == pytest.approx(0.75), "50 TFLOPS where 200 was there"
    assert by_graph["g2"]["regime"] == "long"


def test_lower_wins_for_time_and_rows_count_only_under_their_own_metric(
    monkeypatch, tmp_path
):
    """A's `tflops` collection ran g1 in 0.1 ms; only its `time` collection counts for time."""
    measured = [
        _measured(tmp_path / "a1.csv", "A", "tflops", [("g1", 900, 0.1)]),
        _measured(tmp_path / "a2.csv", "A", "time", [("g1", 300, 2.0)]),
        _measured(tmp_path / "b2.csv", "B", "time", [("g1", 400, 1.0)]),
    ]
    predictions = _predict("A", "time", {"g1": 0.5}) + _predict("B", "time", {"g1": 1.5})
    code, rows = _run(monkeypatch, tmp_path, predictions, measured)
    assert code == 0 and len(rows) == 1
    assert rows[0]["measured_winner"] == "B" and rows[0]["predicted_winner"] == "A"
    assert rows[0]["regret"] == pytest.approx(1.0), "2.0 ms where 1.0 ms was there"


def test_nothing_contested_is_a_failure_not_an_empty_pass(monkeypatch, tmp_path):
    measured = [_measured(tmp_path / "a.csv", "A", "tflops", [("g1", 100, 1.0)])]
    predictions = _predict("A", "tflops", {"g1": 90}) + _predict("B", "tflops", {"g1": 95})
    code, rows = _run(monkeypatch, tmp_path, predictions, measured)
    assert code == 1 and rows is None


def test_a_collection_that_names_no_metric_is_refused(monkeypatch, tmp_path):
    unbound = tmp_path / "a.csv"
    pd.DataFrame([{"engine_name": "A", "benchmark": "id-g1", "tflops": 1.0}]).to_csv(
        unbound, index=False
    )
    with pytest.raises(SystemExit, match="no binding column"):
        _run(monkeypatch, tmp_path, _predict("A", "tflops", {"g1": 1}), [unbound])
