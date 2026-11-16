# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`compare_engines` compares engines in the metric each row was collected under."""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

from uhd_gen.compare_engines import main  # noqa: E402


def _spec(path, engine, metric, rows, label=None):
    """`LABEL=corpus.csv` for rows of (graph, tflops, avgTimeMs[, device]) collected under
    `metric`."""
    binding = json.dumps(
        {
            "engine": engine,
            "metric": metric,
            "trained_against": {"selector_revision": "r1"},
        }
    )
    pd.DataFrame(
        [
            {
                "benchmark": graph,
                "device": device,
                "binding": binding,
                "tflops": tflops,
                "avgTimeMs": ms,
            }
            for graph, tflops, ms, *device in rows
            for device in (device[0] if device else "board",)
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


def test_an_engines_measurements_on_two_devices_are_not_merged(monkeypatch, tmp_path):
    """B is slower than A on each device, but its MI325 number beats A's MI300 one: merged
    across devices, B would win g1 on throughput."""
    specs = [
        _spec(
            tmp_path / "a.csv",
            "A",
            "tflops",
            [("g1", 100, 1.0, "mi300"), ("g1", 300, 0.3, "mi325")],
        ),
        _spec(
            tmp_path / "b.csv",
            "B",
            "tflops",
            [("g1", 90, 1.1, "mi300"), ("g1", 250, 0.4, "mi325")],
        ),
    ]
    code, report = _run(monkeypatch, tmp_path, specs)
    assert code == 0
    tflops = json.loads(report.read_text(encoding="utf-8"))["metrics"]["tflops"]
    assert tflops["per_graph"] == {
        "g1 @ mi300": {"A": 100.0, "B": 90.0},
        "g1 @ mi325": {"A": 300.0, "B": 250.0},
    }
    assert tflops["wins"] == {"A": 2}


def test_a_margin_is_relative_to_the_slower_engine(monkeypatch, tmp_path, capsys):
    """Time: 1.0 ms beats 1.5 ms by 0.5 ms, which is 33.3% of the slower engine's time."""
    specs = [
        _spec(tmp_path / "a.csv", "A", "time", [("g1", 1, 1.0)]),
        _spec(tmp_path / "b.csv", "B", "time", [("g1", 1, 1.5)]),
    ]
    code, _ = _run(monkeypatch, tmp_path, specs)
    assert code == 0
    margin = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if "A 1" in line and "B 1.5" in line
    )
    assert margin.split("%")[0].strip() == "33.3"
