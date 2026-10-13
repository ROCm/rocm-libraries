# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`reproduce/compare_engines.py` compares engines in the metric each row was collected under."""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

from uhd_gen.reproduce.compare_engines import main  # noqa: E402


def _spec(path, engine, metric, rows, label=None):
    """`LABEL=corpus.csv` for rows of (graph, tflops, avgTimeMs) collected under `metric`."""
    binding = json.dumps(
        {
            "engine": engine,
            "metric": metric,
            "trained_against": {"selector_revision": "r1"},
        }
    )
    pd.DataFrame(
        [
            {"benchmark": graph, "binding": binding, "tflops": tflops, "avgTimeMs": ms}
            for graph, tflops, ms in rows
        ]
    ).to_csv(path, index=False)
    return f"{label or engine}={path}"


def _run(monkeypatch, tmp_path, specs):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "graphs": [
                    {"benchmark": graph, "name": graph, "regime": "conv"}
                    for graph in ("g1", "g2")
                ]
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    arguments = ["compare_engines", "--manifest", str(manifest)]
    arguments += ["--report", str(report)]
    for spec in specs:
        arguments += ["--engine", spec]
    monkeypatch.setattr("sys.argv", arguments)
    return main(), report


def test_each_metric_is_won_by_the_rows_collected_under_it(monkeypatch, tmp_path):
    """A's `time` pick runs g1 at 300 TFLOPS, but its `tflops` answer is 100."""
    specs = [
        _spec(tmp_path / "a1.csv", "A", "tflops", [("g1", 100, 1.0), ("g2", 50, 2.0)]),
        _spec(tmp_path / "a2.csv", "A", "time", [("g1", 300, 0.4), ("g2", 40, 2.5)]),
        _spec(tmp_path / "b1.csv", "B", "tflops", [("g1", 200, 0.5), ("g2", 60, 1.7)]),
        _spec(tmp_path / "b2.csv", "B", "time", [("g1", 210, 0.45), ("g2", 70, 1.5)]),
    ]
    code, report = _run(monkeypatch, tmp_path, specs)
    assert code == 0
    metrics = json.loads(report.read_text(encoding="utf-8"))["metrics"]
    assert metrics["tflops"]["per_graph"]["g1"] == {"A": 100.0, "B": 200.0}
    assert metrics["tflops"]["wins"] == {"B": 2}
    # Lower time wins: A's 0.4 ms on g1, B's 1.5 ms on g2.
    assert metrics["time"]["wins"] == {"A": 1, "B": 1}


def test_a_label_naming_two_engines_in_one_metric_is_refused(monkeypatch, tmp_path):
    specs = [
        _spec(tmp_path / "a.csv", "A", "tflops", [("g1", 100, 1.0)]),
        _spec(tmp_path / "b.csv", "B", "tflops", [("g2", 100, 1.0)], label="A"),
    ]
    with pytest.raises(SystemExit, match="mixes engines"):
        _run(monkeypatch, tmp_path, specs)
