# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for `uhd_gen size`: learning curve, ceiling, fit, and regime quotas."""
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
from uhd_gen.provenance import snapshot_provenance
from uhd_gen.sizing import (
    allocate,
    ceiling_from_repeats,
    fit_curve,
    miss_at,
    size_for,
    stratified_test_set,
)

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"
GRAPHS = 150


# --------------------------------------------------------------------------------------
# The pure pieces.
# --------------------------------------------------------------------------------------


def test_the_fit_recovers_a_planted_power_law():
    floor, a, b = 0.12, 3.0, 0.45
    points = [(n, floor + a * n**-b) for n in (50, 100, 200, 400, 800, 1600)]
    fit = fit_curve(points, floor)
    assert fit["floor_source"] == "measured"
    assert fit["a"] == pytest.approx(a, rel=1e-6) and fit["b"] == pytest.approx(
        b, rel=1e-6
    )
    assert size_for(fit, 1 - (floor + a * 3200**-b)) == pytest.approx(3200, rel=1e-6)


def test_an_unknown_ceiling_is_estimated_and_says_so():
    floor, a, b = 0.2, 2.0, 0.5
    points = [(n, floor + a * n**-b) for n in (50, 100, 200, 400, 800, 1600)]
    fit = fit_curve(points, None)
    assert fit["floor_source"] == "estimated"
    assert fit["floor"] == pytest.approx(floor, abs=0.02)


def test_a_target_at_or_past_the_ceiling_is_unreachable():
    fit = {"floor": 0.13, "a": 3.0, "b": 0.5}
    assert size_for(fit, 0.87) is None, "the ceiling itself is only approached"
    assert size_for(fit, 0.95) is None
    assert size_for(fit, 0.80) > 0


def test_two_points_are_not_a_curve():
    assert fit_curve([(100, 0.5), (200, 0.4)], 0.1) is None


def test_the_ceiling_scores_repeats_against_the_label_never_the_label_itself():
    label = {"benchmark": "g", "arch": "gfx", "tflops": 100.0}
    same_gpu_again = {"benchmark": "g", "arch": "gfx", "tflops": 104.0}
    other_gpu = {"benchmark": "g", "arch": "gfx", "tflops": 130.0}
    key = ("g", "gfx", "None", "None")
    labels = {key: ((2, "gpu0"), label)}
    measurements = [
        ((0, "gpu0"), same_gpu_again),
        ((1, "gpu1"), other_gpu),
        ((2, "gpu0"), label),
    ]
    ceiling = ceiling_from_repeats(labels, measurements, "tflops", 0.10, {"g"})
    assert ceiling["repeats"] == 2 and ceiling["repeat_within"] == 0.5
    # 104 vs 100: two errors of ~2.8%; 130 vs 100: two of ~21%. One of each within 10%.
    assert ceiling["within"] == 0.5
    wide = ceiling_from_repeats(
        labels, [((1, "gpu1"), dict(other_gpu, tflops=113.0))], "tflops", 0.10, {"g"}
    )
    assert (wide["repeat_within"], wide["within"]) == (
        0.0,
        1.0,
    ), "13% apart is two measurements ~9% off each: a perfect model is within 10%"
    assert (
        ceiling_from_repeats(labels, [((2, "gpu0"), label)], "tflops", 0.1, {"g"})[
            "within"
        ]
        is None
    )


def test_every_regime_reaches_the_floor_before_the_rest_follows_the_misses():
    regimes = {
        "common": {"train": 500, "test": 100, "misses": 40},
        "weak": {"train": 60, "test": 20, "misses": 16},
        "thin": {"train": 4, "test": 2, "misses": 1},
        "unmeasured": {"train": 0, "test": 0, "misses": 0},
    }
    quotas = allocate(300, regimes, floor=30)
    assert sum(quotas.values()) == 300
    assert (
        quotas["unmeasured"] == 30
    ), "a regime nothing has tested gets its floor and no more"
    assert quotas["thin"] >= 26
    # Per test shape, `weak` misses twice as often as `common`, so it gets more.
    assert quotas["weak"] / 20 > quotas["common"] / 100


def test_floors_alone_may_exceed_what_was_asked():
    quotas = allocate(
        10,
        {
            "a": {"train": 0, "test": 0, "misses": 0},
            "b": {"train": 5, "test": 1, "misses": 1},
        },
        floor=30,
    )
    assert quotas == {"a": 30, "b": 25}


def test_the_test_set_is_stratified_and_reproducible():
    shapes = {
        "big": [f"b{i}" for i in range(100)],
        "small": ["s0", "s1", "s2"],
        "one": ["o0"],
    }
    chosen = stratified_test_set(shapes, 0.2, seed=4)
    assert chosen == stratified_test_set(shapes, 0.2, seed=4)
    assert sum(s.startswith("b") for s in chosen) == 20
    assert sum(s.startswith("s") for s in chosen) == 1
    assert "o0" not in chosen, "a regime of one shape cannot give it up"


# --------------------------------------------------------------------------------------
# The run.
# --------------------------------------------------------------------------------------


def _noise(device: str, graph: str) -> float:
    """Up to +/-8% per (device, shape), a few shapes much worse."""
    digest = hashlib.sha256(f"{device}:{graph}".encode()).digest()
    spread = 0.6 if digest[1] < 20 else 0.16
    return 1.0 + (digest[0] / 255.0 - 0.5) * spread


@pytest.fixture
def store(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    tree = tmp_path / "descriptors"
    tree.mkdir()
    (tree / "engine.ued.json").write_text(
        json.dumps(
            {"version": "1.0", "id": UED, "name": "provider:engine7", "metadata": KMD}
        ),
        encoding="utf-8",
    )
    (tree / "metadata.kmd.json").write_text(
        json.dumps({"version": "1.0", "id": KMD}), encoding="utf-8"
    )
    provenance = snapshot_provenance(tree)
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    with (graphs / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "name", "regime", "phase", "source", "op"])
        for index in range(GRAPHS):
            (graphs / f"{index}.json").write_text(
                json.dumps({"id": f"graph-{index}", "size": index}), encoding="utf-8"
            )
            phase = "decode" if index % 3 == 0 else "append"
            writer.writerow(
                [f"graph-{index}", f"g{index}", phase, phase, "sweep", "toy_op"]
            )
    engine = {"device": "board-a"}

    def bench(command, environment, log_dir, ordinal, commands):
        commands.append({"argv": command})
        metric = command[command.index("--ranking-metric") + 1]
        graph = json.loads(
            Path(command[command.index("--graph") + 1]).read_text(encoding="utf-8")
        )
        average = (1.0 + math.sqrt(graph["size"])) * _noise(
            engine["device"], graph["id"]
        )
        return {
            "engine_id": 7,
            "engine_name": "provider:engine7",
            "graph_id": graph["id"],
            "device_id": engine["device"],
            "arch": "gfx942",
            "metric": metric,
            "binding": {
                "engine": "provider:engine7",
                "role": "predict_engine",
                "arch": "gfx942",
                "metric": metric,
                "selector_revision": "provider-1",
                "trained_against": {**provenance, "selector_revision": "provider-1"},
            },
            "features": {
                "graph.flops": 2e9 * (graph["size"] + 1),
                "device.cu_count": 120,
            },
            "avgTimeMs": average,
            "robustMeanMs": average * 0.9,
            "stddevMs": 0.01,
            "iters": 30,
            "is_valid": True,
            "selection_mode": "immediate",
            "timing_statistic": "robustMeanMs",
        }

    monkeypatch.setattr("uhd_gen.generate._run_json", bench)
    monkeypatch.setattr("uhd_gen.generate.shutil.which", lambda name: name)

    def collect(name, device, collected_at):
        from uhd_gen.__main__ import main

        engine["device"] = device
        output = tmp_path / name
        assert (
            main(
                [
                    "generate",
                    "--collect-only",
                    "--graphs",
                    str(graphs),
                    "--descriptor-tree",
                    str(tree),
                    "--engine-id",
                    "7",
                    "--role",
                    "predict_engine",
                    "--output-dir",
                    str(output),
                ]
            )
            == 0
        )
        manifest = json.loads(
            (output / "collection_manifest.json").read_text(encoding="utf-8")
        )
        manifest["collected_at"] = collected_at
        (output / "collection_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return output

    first = collect("col_a", "board-a", "2026-09-28T10:00:00+00:00")
    second = collect("col_b", "board-b", "2026-09-29T10:00:00+00:00")
    return {"tree": tree, "collections": [first, second], "root": tmp_path}


def _size(store, evaluator, name, *extra):
    from uhd_gen.__main__ import main

    output = store["root"] / name
    code = main(
        [
            "size",
            "--collection",
            *map(str, store["collections"]),
            "--test-set",
            str(store["root"] / "test_set.json"),
            "--descriptor-tree",
            str(store["tree"]),
            "--engine-id",
            "7",
            "--arch",
            "gfx942",
            "--features",
            "graph.flops",
            "--num-boost-round",
            "30",
            "--early-stopping",
            "5",
            "--eval-fraction",
            "0.1",
            "--sizes",
            "15",
            "30",
            "60",
            "120",
            "--draws",
            "2",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(output),
            *extra,
        ]
    )
    return code, output


def test_a_sizing_run_pins_its_test_set_and_never_trains_on_it(
    store, evaluator, monkeypatch
):
    import uhd_gen.sizing as sizing

    trained_on = []
    real = sizing._write_subset

    def spy(template, merged, metric, rows, destination):
        trained_on.append({row["benchmark"] for row in rows})
        return real(template, merged, metric, rows, destination)

    monkeypatch.setattr(sizing, "_write_subset", spy)
    code, output = _size(
        store, evaluator, "round1", "--create-test-set", "--target", "0.6"
    )
    assert code == 0
    test_set = json.loads((store["root"] / "test_set.json").read_text(encoding="utf-8"))
    pinned = set(test_set["shapes"])
    assert len(pinned) == 30, "20% of each regime: 10 of 50 decode, 20 of 100 append"
    assert trained_on and not any(subset & pinned for subset in trained_on)
    # Nested within a draw: every smaller subset is a prefix of the larger ones.
    assert trained_on[0] <= trained_on[1] <= trained_on[2]

    report = json.loads((output / "sizing_report.json").read_text(encoding="utf-8"))
    assert report["schema"] == "uhd_gen.sizing/1"
    assert report["training_pool"] == 120 and len(report["curve"]) == 7
    assert (
        report["ceiling"]["repeats"] == 30
    ), "each test shape measured again on the other GPU"
    assert report["ceiling"]["repeat_within"] < report["ceiling"]["within"] < 1
    within = {}
    for point in report["curve"]:
        within.setdefault(point["shapes"], []).append(point["within"])
    assert sum(within[120]) / len(within[120]) > sum(within[15]) / len(
        within[15]
    ), "more shapes, better model: a fixture that does not show this sizes nothing"
    assert report["fit"] is not None
    assert set(report["per_regime"]) == {"decode", "append"}
    quotas = json.loads((output / "regime_quotas.json").read_text(encoding="utf-8"))
    assert set(quotas) == {"toy_op"}
    assert sum(quotas["toy_op"].values()) >= report["recommendation"]["new_shapes"]

    # The second round reuses the pin, and checks the first round's prediction.
    code, second = _size(
        store,
        evaluator,
        "round2",
        "--create-test-set",
        "--previous",
        str(output / "sizing_report.json"),
    )
    assert code == 0
    assert (
        json.loads((store["root"] / "test_set.json").read_text(encoding="utf-8"))
        == test_set
    )
    checked = json.loads((second / "sizing_report.json").read_text(encoding="utf-8"))[
        "previous"
    ]
    assert checked["comparable"] is True
    assert checked["observed_within"] is not None


def test_a_target_past_the_measured_ceiling_asks_for_measurement_not_shapes(
    store, evaluator
):
    code, output = _size(
        store, evaluator, "past", "--create-test-set", "--target", "0.99"
    )
    assert code == 0
    recommendation = json.loads(
        (output / "sizing_report.json").read_text(encoding="utf-8")
    )["recommendation"]
    assert recommendation["reachable"] is False
    assert "ceiling" in recommendation["reason"]
    assert recommendation["basis"] == "next_doubling"


def test_without_a_pinned_test_set_nothing_is_sized(store, evaluator, caplog):
    code, output = _size(store, evaluator, "unpinned")
    assert code == 1 and not output.exists()
    assert "--create-test-set" in caplog.text
