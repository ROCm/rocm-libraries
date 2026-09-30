# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Collection as its own step: `generate --collect-only`, then `generate --collection`.

Measuring and training used to be one run, so every model re-measured its corpus and the
measurements lived only inside that model's output. These pin the split: a recorded
collection trains exactly what the one-shot run would have, training takes one measurement
of each configuration on each shape (the newest), and what cannot be one model is refused
at the merge rather than trained.
"""
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
from uhd_gen.provenance import snapshot_provenance

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"
GRAPHS = 16


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An L1 engine behind a fake bench whose device, revision and speed a test can set."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    tree = tmp_path / "descriptors"
    tree.mkdir()
    (tree / "engine.ued.json").write_text(json.dumps(
        {"version": "1.0", "id": UED, "name": "provider:engine7", "metadata": KMD}), encoding="utf-8")
    (tree / "metadata.kmd.json").write_text(json.dumps({"version": "1.0", "id": KMD}), encoding="utf-8")
    provenance = snapshot_provenance(tree)
    graphs = tmp_path / "graphs"
    graphs.mkdir()
    for index in range(GRAPHS):
        (graphs / f"{index}.json").write_text(json.dumps({"id": f"graph-{index}", "size": index}),
                                              encoding="utf-8")
    engine = {"device": "board", "revision": "provider-1", "scale": 1.0, "engine_name": "provider:engine7",
              "calls": 0}

    def bench(command, environment, log_dir, ordinal, commands):
        engine["calls"] += 1
        commands.append({"argv": command})
        metric = command[command.index("--ranking-metric") + 1]
        graph = json.loads(Path(command[command.index("--graph") + 1]).read_text(encoding="utf-8"))
        average = (1.0 + graph["size"] / 10) * engine["scale"]
        # `--device N` runs on board-N, as HIP_VISIBLE_DEVICES picks a GPU.
        visible = environment.get("HIP_VISIBLE_DEVICES")
        device = f"board-{visible}" if visible else engine["device"]
        engine["scale"] *= engine.get("drift", 1.0)
        return {"engine_id": 7, "engine_name": engine["engine_name"], "graph_id": graph["id"],
                "device_id": device, "arch": "gfx942", "metric": metric,
                "binding": {"engine": engine["engine_name"], "role": "predict_engine", "arch": "gfx942",
                            "selector_revision": engine["revision"], "trained_against": provenance},
                "features": {"graph.flops": 2e9 * (graph["size"] + 1), "device.cu_count": 120},
                "avgTimeMs": average, "robustMeanMs": average * 0.9, "stddevMs": 0.01, "iters": 30,
                "is_valid": True, "selection_mode": "immediate", "timing_statistic": "robustMeanMs"}

    monkeypatch.setattr("uhd_gen.generate._run_json", bench)
    monkeypatch.setattr("uhd_gen.generate.shutil.which", lambda name: name)
    return {"tree": tree, "graphs": graphs, "engine": engine, "root": tmp_path}


def _collect(world, name, collected_at=None, **engine):
    from uhd_gen.__main__ import main

    world["engine"].update(engine)
    output = world["root"] / name
    assert main(["generate", "--collect-only", "--graphs", str(world["graphs"]),
                 "--descriptor-tree", str(world["tree"]), "--engine-id", "7",
                 "--role", "predict_engine", "--output-dir", str(output)]) == 0
    if collected_at:
        # The merge orders by when a collection was taken; pin it rather than sleep.
        manifest = json.loads((output / "collection_manifest.json").read_text(encoding="utf-8"))
        manifest["collected_at"] = collected_at
        (output / "collection_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return output


def _train(world, name, evaluator, *, graphs=None, collections=(), extra=()):
    from uhd_gen.__main__ import main

    source = ["--graphs", str(graphs)] if graphs else ["--collection", *map(str, collections)]
    output = world["root"] / name
    code = main(["generate", *source, "--descriptor-tree", str(world["tree"]), "--engine-id", "7",
                 "--role", "predict_engine", "--features", "graph.flops", "--num-boost-round", "4",
                 "--early-stopping", "2", "--eval-fraction", "0.25", "--arch", "gfx942",
                 "--no-promote", "--feature-evaluator", evaluator, "--output-dir", str(output), *extra])
    return code, output


def _labels(output):
    return {row["benchmark"]: row["avgTimeMs"]
            for row in json.loads((output / "corpus.json").read_text(encoding="utf-8"))}


def test_collect_only_records_measurements_and_trains_nothing(world):
    collection = _collect(world, "col")
    manifest = json.loads((collection / "collection_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "uhd_gen.collection/1"
    assert (manifest["graph_count"], manifest["row_counts"]) == (GRAPHS, {"tflops": GRAPHS})
    assert (manifest["selector_revision"], manifest["devices"], manifest["arches"]) == (
        "provider-1", ["board"], ["gfx942"])
    assert len(json.loads((collection / "corpus.json").read_text(encoding="utf-8"))) == GRAPHS
    assert not (collection / "model").exists() and not (collection / "generation_manifest.json").exists()


def test_a_recorded_collection_trains_what_the_one_shot_run_would(world, evaluator):
    """Same measurements, same seed: the split, and so the model's training set, match."""
    one_shot = _train(world, "one_shot", evaluator, graphs=world["graphs"])
    calls = world["engine"]["calls"]
    collection = _collect(world, "col")
    world["engine"]["calls"] = 0
    from_collection = _train(world, "from_col", evaluator, collections=[collection])
    assert one_shot[0] == from_collection[0] == 0
    assert world["engine"]["calls"] == 0, "training from a collection measured something"
    assert calls == GRAPHS
    for name in ("train.json", "corpus.json"):
        a = json.loads((one_shot[1] / name).read_text(encoding="utf-8"))
        b = json.loads((from_collection[1] / name).read_text(encoding="utf-8"))
        assert sorted(r["benchmark"] for r in a) == sorted(r["benchmark"] for r in b), name
    manifest = json.loads((from_collection[1] / "generation_manifest.json").read_text(encoding="utf-8"))
    assert [c["path"] for c in manifest["collections"]] == [str(collection)]
    assert manifest["superseded_rows"] == 0


def test_the_newest_measurement_of_a_graph_on_a_device_is_the_label(world, evaluator):
    older = _collect(world, "older", collected_at="2026-09-28T10:00:00+00:00", scale=1.0)
    newer = _collect(world, "newer", collected_at="2026-09-30T10:00:00+00:00", scale=2.0)
    # Named newest-first on purpose: the order given does not decide, the collection time does.
    code, output = _train(world, "merged", evaluator, collections=[newer, older])
    assert code == 0
    labels, newest = _labels(output), _labels(newer)
    assert labels == newest, "an older measurement survived the merge"
    manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["superseded_rows"] == GRAPHS
    assert [c["path"] for c in manifest["collections"]] == [str(older), str(newer)]


def test_a_shape_measured_on_two_gpus_of_the_arch_is_trained_on_once(world, evaluator):
    """Another GPU of the same arch measures the same problem: one label, the newest."""
    older = _collect(world, "a", collected_at="2026-09-28T10:00:00+00:00", device="board-a", scale=1.0)
    newer = _collect(world, "b", collected_at="2026-09-29T10:00:00+00:00", device="board-b", scale=2.0)
    code, output = _train(world, "both", evaluator, collections=[older, newer])
    assert code == 0
    rows = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert len(rows) == GRAPHS and {r["device"] for r in rows} == {"board-b"}
    assert _labels(output) == _labels(newer)
    manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["superseded_rows"] == GRAPHS


def test_one_run_on_several_gpus_trains_on_each_shape_once(world, evaluator):
    """`--device` repeated measures every shape on each GPU; the last GPU's is the label."""
    world["engine"]["drift"] = 1.01  # every measurement differs, so the label is traceable
    code, output = _train(world, "two_gpus", evaluator, graphs=world["graphs"],
                          extra=["--device", "0", "--device", "1"])
    assert code == 0 and world["engine"]["calls"] == 2 * GRAPHS
    rows = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert len(rows) == GRAPHS and {r["device"] for r in rows} == {"board-1"}
    manifest = json.loads((output / "generation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["superseded_rows"] == GRAPHS


def test_distinct_configurations_on_one_shape_are_distinct_rows():
    """A catalog sweep times many configurations of a shape; only a repeat is a duplicate."""
    from uhd_gen.generate import one_measurement_per_shape

    def row(graph, kernel, knobs, arch="gfx942"):
        return {"benchmark": graph, "arch": arch, "kernel": kernel, "knob_settings": knobs}

    measured = [
        ("a", row("g", "k1", '{"tile": 64}')), ("a", row("g", "k1", '{"tile": 128}')),
        ("a", row("g", "k2", '{"tile": 64}')), ("a", row("g", "k1", '{"tile": 64}', "gfx950")),
        ("b", row("g", "k1", '{"tile": 64}')),  # a repeat, on another GPU: supersedes a's
        ("b", row("h", "k1", '{"tile": 64}')),
    ]
    kept, dropped = one_measurement_per_shape(measured)
    assert dropped == 1
    assert kept == [m[1] for m in measured[1:]]
    # An L1 row names no configuration (the engine chose): its shape is its identity.
    l1 = [("a", {"benchmark": "g", "arch": "gfx942"}), ("b", {"benchmark": "g", "arch": "gfx942"})]
    assert one_measurement_per_shape(l1) == ([l1[1][1]], 1)
    # A repeat under another binding is a revision change, never a silent supersede.
    with pytest.raises(ValueError, match="two engine bindings"):
        one_measurement_per_shape([("a", dict(l1[0][1], binding="r1")), ("b", dict(l1[1][1], binding="r2"))])


@pytest.mark.parametrize("change, message", [
    ({"revision": "provider-2"}, "collections disagree on selector revision"),
    ({"engine_name": "provider:engine8"}, "collections disagree on engine"),
])
def test_what_cannot_be_one_model_is_refused(world, evaluator, caplog, change, message):
    first = _collect(world, "first", device="board-a")
    second = _collect(world, "second", device="board-b", **change)
    code, output = _train(world, "refused", evaluator, collections=[first, second])
    assert code == 1 and not output.exists()
    assert message in caplog.text


def test_a_metric_the_collection_did_not_measure_is_refused(world, evaluator, caplog):
    collection = _collect(world, "col")
    code, _ = _train(world, "time", evaluator, collections=[collection], extra=["--metric", "time"])
    assert code == 1
    assert "did not measure ['time']" in caplog.text


def test_measurement_options_are_refused_without_a_measurement(world, evaluator, caplog):
    collection = _collect(world, "col")
    code, _ = _train(world, "knobbed", evaluator, collections=[collection], extra=["--device", "0"])
    assert code == 1
    assert "measures nothing" in caplog.text


def test_rows_carry_the_regime_their_corpus_labelled_them_with(world, evaluator):
    """Beside `graphs/`, hipdnn_corpus_gen's manifest.csv names each graph's population."""
    import csv

    with (world["graphs"] / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "name", "regime", "phase", "context", "source", "q.size"])
        for index in range(GRAPHS):
            phase = "decode" if index % 4 == 0 else "append"
            writer.writerow([f"graph-{index}", f"g{index}", f"{phase}_short", phase, "short",
                             "sweep", index])
    collection = _collect(world, "labelled")
    rows = json.loads((collection / "corpus.json").read_text(encoding="utf-8"))
    assert {r["benchmark"]: (r["regime"], r["regime.phase"], r["regime.context"]) for r in rows} == {
        f"graph-{i}": ("decode_short" if i % 4 == 0 else "append_short",
                       "decode" if i % 4 == 0 else "append", "short") for i in range(GRAPHS)}
    assert not any(name.startswith("regime") for name in json.loads(rows[0]["features"])), \
        "a label is an envelope column, never a model input"
    code, output = _train(world, "from_labelled", evaluator, collections=[collection])
    assert code == 0
    assert {r["regime"] for r in json.loads((output / "corpus.json").read_text(encoding="utf-8"))} == {
        "decode_short", "append_short"}


def test_graphs_without_a_corpus_manifest_carry_no_regime(world):
    collection = _collect(world, "unlabelled")
    rows = json.loads((collection / "corpus.json").read_text(encoding="utf-8"))
    assert not any("regime" in row for row in rows)
