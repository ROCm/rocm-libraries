# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for `generate --collect-only` followed by `generate --dataset`.

Merging keeps the best (lowest `avgTimeMs`) usable measurement of each configuration on each
shape, and refuses contributions that cannot form one model.
"""
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
from uhd_gen.provenance import snapshot_provenance
from uhd_gen.tests.bench_stub import find_any_bench

UED = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
KMD = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"
GRAPHS = 16


@pytest.fixture
def world(tmp_path, monkeypatch):
    """An L1 engine behind a fake bench with settable device, revision and speed."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("flatbuffers")
    # The fake bench reads HIP_VISIBLE_DEVICES as `--device`; one inherited from the
    # environment would move every run off the default board.
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
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
    for index in range(GRAPHS):
        (graphs / f"{index}.json").write_text(
            json.dumps({"id": f"graph-{index}", "size": index}), encoding="utf-8"
        )
    engine = {
        "device": "board",
        "revision": "provider-1",
        "scale": 1.0,
        "engine_name": "provider:engine7",
        "calls": 0,
    }

    def bench(command, environment, log_dir, ordinal, commands, timeout=None):
        engine["calls"] += 1
        engine.setdefault("argv", []).append(command)
        commands.append({"argv": command})
        metric = command[command.index("--ranking-metric") + 1]
        graph = json.loads(
            Path(command[command.index("--graph") + 1]).read_text(encoding="utf-8")
        )
        average = (1.0 + graph["size"] / 10) * engine["scale"]
        # `--device N` picks a GPU (HIP_VISIBLE_DEVICES). As the real bench does, it reports
        # the GPU's own UUID as `device_uuid` and its kind (the hash of its properties) as
        # `device_id`; `gpus` gives an ordinal its own kind.
        ordinal = environment.get("HIP_VISIBLE_DEVICES", "0")
        device = engine.get("gpus", {}).get(ordinal, engine["device"])
        engine["scale"] *= engine.get("drift", 1.0)
        return {
            "engine_id": engine.get("engine_id", 7),
            "engine_name": engine["engine_name"],
            "graph_id": graph["id"],
            "graph_key": graph["id"],
            "device_id": device,
            "device_uuid": f"uuid-of-gpu-{ordinal}",
            "arch": "gfx942",
            "metric": metric,
            "binding": {
                "engine": engine["engine_name"],
                "role": "predict_engine",
                "arch": "gfx942",
                "metric": metric,
                "selector_revision": engine["revision"],
                "trained_against": {
                    **provenance,
                    "selector_revision": engine["revision"],
                },
                **(
                    {"provider_build": engine["provider_build"]}
                    if "provider_build" in engine
                    else {}
                ),
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
    find_any_bench(monkeypatch)
    return {"tree": tree, "graphs": graphs, "engine": engine, "root": tmp_path}


def _collect_arguments(world, name, engine_id="7"):
    return [
        "generate",
        "--collect-only",
        "--graphs",
        str(world["graphs"]),
        "--descriptor-tree",
        str(world["tree"]),
        "--engine-id",
        engine_id,
        "--role",
        "predict_engine",
        "--output-dir",
        str(world["root"] / name),
    ]


def _collect(world, name, collected_at=None, engine_id="7", **engine):
    from uhd_gen.__main__ import main

    world["engine"].update(engine)
    output = world["root"] / name
    assert main(_collect_arguments(world, name, engine_id)) == 0
    if collected_at:
        # The merge orders by when a collection was taken; pin it rather than sleep.
        manifest = json.loads(
            (output / "collection_manifest.json").read_text(encoding="utf-8")
        )
        manifest["collected_at"] = collected_at
        (output / "collection_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
    return output


def _dataset(world, name, collections):
    """The collections converted into one dataset, as `python -m uhd_gen.dataset add` does."""
    from uhd_gen.dataset import store

    directory = world["root"] / f"{name}.dataset"
    for collection in collections:
        store.add(directory, *store.from_collection(collection))
    return directory


def _train(
    world, name, evaluator, *, graphs=None, collections=(), dataset=None, extra=()
):
    from uhd_gen.__main__ import main

    if graphs:
        source = ["--graphs", str(graphs)]
    else:
        source = ["--dataset", str(dataset or _dataset(world, name, collections))]
    output = world["root"] / name
    code = main(
        [
            "generate",
            *source,
            "--descriptor-tree",
            str(world["tree"]),
            "--engine-id",
            "7",
            "--role",
            "predict_engine",
            "--features",
            "graph.flops",
            "--num-boost-round",
            "4",
            "--early-stopping",
            "2",
            "--eval-fraction",
            "0.25",
            "--arch",
            "gfx942",
            "--no-promote",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(output),
            *extra,
        ]
    )
    return code, output


def _labels(output):
    return {
        row["benchmark"]: row["avgTimeMs"]
        for row in json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    }


def test_collect_only_records_measurements_and_trains_nothing(world):
    collection = _collect(world, "col")
    manifest = json.loads(
        (collection / "collection_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == "uhd_gen.collection/1"
    assert (manifest["graph_count"], manifest["row_counts"]) == (
        GRAPHS,
        {"tflops": GRAPHS},
    )
    assert (manifest["selector_revision"], manifest["devices"], manifest["arches"]) == (
        "provider-1",
        ["board"],
        ["gfx942"],
    )
    assert (
        len(json.loads((collection / "corpus.json").read_text(encoding="utf-8")))
        == GRAPHS
    )
    assert (
        not (collection / "model").exists()
        and not (collection / "generation_manifest.json").exists()
    )


@pytest.mark.parametrize(
    "spelling", ["-677973494996465522", "17768770578713086094", "0xF6975AB2C79B088E"]
)
def test_an_engine_id_past_two_to_the_sixty_third_is_one_id_however_written(
    world, spelling
):
    """hipdnn_list_engines prints ids as hex and bench reports them signed: every spelling
    reaches bench as the id it reports, so its rows are accepted as that engine's."""
    world["engine"]["engine_id"] = -677973494996465522
    collection = _collect(world, "col", engine_id=spelling)
    asked = {argv[argv.index("--engine-id") + 1] for argv in world["engine"]["argv"]}
    assert asked == {"-677973494996465522"}
    assert (
        len(json.loads((collection / "corpus.json").read_text(encoding="utf-8")))
        == GRAPHS
    )


@pytest.mark.parametrize(
    "spelling,reason",
    [
        ("ten", "not an engine id"),
        ("1_000", "not an engine id"),
        ("0x", "not an engine id"),
        ("18446744073709551616", "not a 64-bit engine id"),
    ],
)
def test_an_engine_id_that_is_not_one_is_refused(world, capsys, spelling, reason):
    from uhd_gen.__main__ import main

    with pytest.raises(SystemExit) as refused:
        main(_collect_arguments(world, "x", spelling))
    assert refused.value.code == 2, "argparse's usage error"
    assert f"{reason}: {spelling!r}" in capsys.readouterr().err
    assert world["engine"]["calls"] == 0


def test_a_model_trained_from_collections_takes_its_name_and_version(world, evaluator):
    """--descriptor-name and --model-version reach the model, as they do from a dataset."""
    collection = _collect(world, "col")
    code, output = _train(
        world,
        "named",
        evaluator,
        collections=[collection],
        extra=["--descriptor-name", "custom", "--model-version", "3.1"],
    )
    assert code == 0
    assert (output / "model" / "custom.uhd.json").is_file()
    manifest = json.loads(
        (output / "model" / "train_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_version"] == "3.1"


def test_a_recorded_collection_trains_what_the_one_shot_run_would(world, evaluator):
    """Same measurements and seed give the same split and training set."""
    one_shot = _train(world, "one_shot", evaluator, graphs=world["graphs"])
    calls = world["engine"]["calls"]
    collection = _collect(world, "col")
    world["engine"]["calls"] = 0
    from_collection = _train(world, "from_col", evaluator, collections=[collection])
    assert one_shot[0] == from_collection[0] == 0
    assert (
        world["engine"]["calls"] == 0
    ), "training from a collection measured something"
    assert calls == GRAPHS
    for name in ("train.json", "corpus.json"):
        a = json.loads((one_shot[1] / name).read_text(encoding="utf-8"))
        b = json.loads((from_collection[1] / name).read_text(encoding="utf-8"))
        assert sorted(r["benchmark"] for r in a) == sorted(
            r["benchmark"] for r in b
        ), name
    manifest = json.loads(
        (from_collection[1] / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert [c["path"] for c in manifest["collections"]] == [str(collection)]
    assert manifest["superseded_rows"] == 0


def test_a_one_shot_runs_measurements_convert_into_a_dataset_that_trains_the_same(
    world, evaluator
):
    """`dataset add --generation`: an old run's folder, not re-measured, is training data."""
    from uhd_gen.dataset import store

    code, one_shot = _train(world, "one_shot", evaluator, graphs=world["graphs"])
    assert code == 0
    entry, rows = store.from_generation(one_shot)
    assert (entry["role"], entry["selector_revision"], entry["graph_count"]) == (
        "predict_engine",
        "provider-1",
        GRAPHS,
    )
    assert entry["origin"] == {"kind": "generation", "path": str(one_shot.resolve())}
    assert entry["collected_at"], "the first bench call's time"
    dataset = world["root"] / "converted.dataset"
    assert store.add(dataset, entry, rows)["rows"] == GRAPHS
    calls = world["engine"]["calls"]
    code, retrained = _train(world, "retrained", evaluator, dataset=dataset)
    assert code == 0 and world["engine"]["calls"] == calls, "nothing was measured again"
    for name in ("train.json", "corpus.json"):
        a = json.loads((one_shot / name).read_text(encoding="utf-8"))
        b = json.loads((retrained / name).read_text(encoding="utf-8"))
        assert sorted(r["benchmark"] for r in a) == sorted(
            r["benchmark"] for r in b
        ), name


def test_a_generation_of_another_role_is_not_converted(world, evaluator, tmp_path):
    from uhd_gen.dataset import store

    code, one_shot = _train(world, "one_shot", evaluator, graphs=world["graphs"])
    manifest = one_shot / "generation_manifest.json"
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(
        json.dumps({**recorded, "promotion_role": "sort_kernel_catalog"}),
        encoding="utf-8",
    )
    with pytest.raises(store.DatasetError, match="only engine-immediate"):
        store.from_generation(one_shot)


def test_the_best_measurement_of_a_graph_is_the_label(world, evaluator):
    faster = _collect(
        world, "older", collected_at="2026-09-28T10:00:00+00:00", scale=1.0
    )
    slower = _collect(
        world, "newer", collected_at="2026-09-30T10:00:00+00:00", scale=2.0
    )
    code, output = _train(world, "merged", evaluator, collections=[slower, faster])
    assert code == 0
    assert _labels(output) == _labels(faster), "a slower repeat survived the merge"
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["superseded_rows"] == GRAPHS
    # Collection time, not argument order, orders them.
    assert [c["path"] for c in manifest["collections"]] == [str(faster), str(slower)]


def test_a_rebuilt_provider_measures_repeats_not_another_binding(world, evaluator):
    """`binding.provider_build` names the loaded binary, not the selector contract."""
    build = {"name": "provider", "version": "1.0+aaaaaaa", "api_version": "1"}
    older = _collect(
        world,
        "older",
        collected_at="2026-09-28T10:00:00+00:00",
        provider_build=build,
    )
    newer = _collect(
        world,
        "newer",
        collected_at="2026-09-30T10:00:00+00:00",
        provider_build={**build, "version": "1.0+bbbbbbb"},
        scale=0.5,
    )
    code, output = _train(world, "rebuilt", evaluator, collections=[older, newer])
    assert code == 0
    assert _labels(output) == _labels(newer), "the rebuilt provider's faster repeat"
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["superseded_rows"] == GRAPHS


def test_a_shape_measured_on_two_gpus_of_one_kind_is_trained_on_once(world, evaluator):
    """Another GPU of the same kind measures the same problem: one label, the best."""
    older = _collect(world, "a", collected_at="2026-09-28T10:00:00+00:00", scale=1.0)
    newer = _collect(world, "b", collected_at="2026-09-29T10:00:00+00:00", scale=0.5)
    code, output = _train(world, "both", evaluator, collections=[older, newer])
    assert code == 0
    rows = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert len(rows) == GRAPHS
    assert _labels(output) == _labels(newer)
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["superseded_rows"] == GRAPHS


def test_a_different_kind_of_card_measures_another_problem(world, evaluator):
    """RFC 0019.13 §4.1: a device is its properties. A card of another kind (another device
    identity) measures other problems, so neither measurement sets the other aside."""
    first = _collect(
        world, "a", collected_at="2026-09-28T10:00:00+00:00", device="kind-a"
    )
    second = _collect(
        world, "b", collected_at="2026-09-29T10:00:00+00:00", device="kind-b"
    )
    code, output = _train(world, "kinds", evaluator, collections=[first, second])
    assert code == 0
    rows = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert len(rows) == 2 * GRAPHS
    assert {r["device"] for r in rows} == {"kind-a", "kind-b"}
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["superseded_rows"] == 0


def test_one_run_on_several_gpus_trains_on_each_shape_once(world, evaluator):
    """Repeated `--device` measures each shape per GPU; the faster GPU's is the label, and
    measurements.json keeps both."""
    world["engine"][
        "drift"
    ] = 1.01  # every measurement differs, so the label is traceable
    code, output = _train(
        world,
        "two_gpus",
        evaluator,
        graphs=world["graphs"],
        extra=["--device", "0", "--device", "1"],
    )
    assert code == 0 and world["engine"]["calls"] == 2 * GRAPHS
    measured = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    assert len(measured) == 2 * GRAPHS
    rows = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert len(rows) == GRAPHS
    fastest = {}
    for row in measured:
        fastest[row["benchmark"]] = min(
            fastest.get(row["benchmark"], row["avgTimeMs"]), row["avgTimeMs"]
        )
    assert {r["benchmark"]: r["avgTimeMs"] for r in rows} == fastest
    assert {r["device"] for r in rows} == {
        "board"
    }, "two GPUs of one kind are one device"
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["superseded_rows"] == GRAPHS


def test_a_collection_on_two_gpus_of_one_kind_keeps_both_and_trains_once(
    world, evaluator
):
    """A collection records every repeat; the dataset trains on one row per shape."""
    from uhd_gen.__main__ import main

    world["engine"]["drift"] = 1.01
    name = "two_gpus_collected"
    assert (
        main(_collect_arguments(world, name) + ["--device", "0", "--device", "1"]) == 0
    )
    collection = world["root"] / name
    assert (collection / "collection_manifest.json").is_file()
    code, output = _train(world, "from_two_gpus", evaluator, collections=[collection])
    assert code == 0
    rows = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert len(rows) == GRAPHS


def test_one_run_on_two_kinds_of_gpu_measures_each_kind_separately(world, evaluator):
    """The second GPU is another kind of card: its measurements are other problems."""
    world["engine"]["gpus"] = {"1": "other-kind"}
    code, output = _train(
        world,
        "two_kinds",
        evaluator,
        graphs=world["graphs"],
        extra=["--device", "0", "--device", "1"],
    )
    assert code == 0
    rows = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert len(rows) == 2 * GRAPHS
    assert {r["device"] for r in rows} == {"board", "other-kind"}


def test_distinct_configurations_on_one_shape_are_distinct_rows():
    """Only a repeat of the same configuration on a shape is a duplicate."""
    from uhd_gen.generate import best_measurements

    def row(graph, kernel, knobs, arch="gfx942", time=1.0):
        return {
            "benchmark": graph,
            "arch": arch,
            "kernel": kernel,
            "knob_settings": knobs,
            "avgTimeMs": time,
        }

    measured = [
        ("a", row("g", "k1", '{"tile": 64}', time=2.0)),
        ("a", row("g", "k1", '{"tile": 128}')),
        ("a", row("g", "k2", '{"tile": 64}')),
        ("a", row("g", "k1", '{"tile": 64}', "gfx950")),
        # A repeat, on another GPU, and faster: it is kept over a's.
        ("b", row("g", "k1", '{"tile": 64}', time=1.5)),
        ("b", row("h", "k1", '{"tile": 64}')),
    ]
    kept, superseded, excluded = best_measurements(measured)
    assert (superseded, excluded) == (1, {})
    assert kept == measured[1:]
    # An L1 row names no configuration (the engine chose): its shape is its identity.
    l1 = [
        ("a", {"benchmark": "g", "arch": "gfx942", "avgTimeMs": 1.0}),
        ("b", {"benchmark": "g", "arch": "gfx942", "avgTimeMs": 1.0}),
    ]
    assert best_measurements(l1) == ([l1[1]], 1, {}), "a tie keeps the newer"

    # A repeat under another binding is a revision change, never a silent supersede.
    def bound(revision, build):
        binding = {"selector_revision": revision, "provider_build": build}
        return dict(l1[0][1], binding=json.dumps(binding))

    with pytest.raises(ValueError, match="two engine bindings"):
        best_measurements([("a", bound("r1", "x")), ("b", bound("r2", "x"))])
    rebuilt = [("a", bound("r1", "x")), ("b", bound("r1", "y"))]
    assert best_measurements(rebuilt) == ([rebuilt[1]], 1, {})


def test_the_best_measurement_of_a_repeat_is_kept_and_failures_are_removed():
    """Repeats of one configuration keep the lowest avgTimeMs; a row with no usable
    measurement is never a candidate, however fast it claims to be."""
    from uhd_gen.generate import best_measurements

    slow = {"benchmark": "key-1", "arch": "gfx942", "avgTimeMs": 1.1}
    fast = {"benchmark": "key-1", "arch": "gfx942", "avgTimeMs": 1.0}
    wrong = dict(fast, avgTimeMs=0.1, numerically_valid=False)
    invalid = dict(fast, avgTimeMs=0.1, is_valid=False)
    not_run = dict(fast, avgTimeMs=0.1, succeeded=False)
    errored = dict(fast, avgTimeMs=0.1, error="timeout")
    unmeasured = dict(fast, avgTimeMs=None)
    other = {"benchmark": "key-2", "arch": "gfx942", "avgTimeMs": 2.0}
    kept, superseded, excluded = best_measurements(
        [
            ("gpu0", fast),
            ("gpu0", other),
            ("gpu0", wrong),
            ("gpu1", invalid),
            ("gpu1", not_run),
            ("gpu1", errored),
            ("gpu1", unmeasured),
            ("gpu1", slow),
        ]
    )
    assert [row for _, row in kept] == [fast, other]
    assert superseded == 1
    assert excluded == {
        "numerically_wrong": 1,
        "invalid": 1,
        "not_run": 1,
        "errored": 1,
        "unmeasured": 1,
    }


def test_published_candidates_of_one_problem_stay_distinct_rows():
    """A published row names its candidate by `kernel.*` fields, not `kernel`."""
    from uhd_gen.generate import best_measurements

    rows = [
        {"benchmark": "p", "arch": "gfx942", "kernel.solver_id": 1, "avgTimeMs": 1.0},
        {"benchmark": "p", "arch": "gfx942", "kernel.solver_id": 2, "avgTimeMs": 1.0},
    ]
    measured = [("s", r) for r in rows]
    assert best_measurements(measured) == (measured, 0, {})


def test_a_bench_that_names_no_graph_key_cannot_collect(world, caplog, monkeypatch):
    """A row keyed by anything but content could enter a training set twice."""
    from uhd_gen import generate
    from uhd_gen.__main__ import main

    measure = generate._run_json

    def unkeyed(*args, **kwargs):
        response = measure(*args, **kwargs)
        response.pop("graph_key", None)
        return response

    monkeypatch.setattr(generate, "_run_json", unkeyed)
    assert main(_collect_arguments(world, "unkeyed")) == 1
    assert "reported no graph_key" in caplog.text
    assert not (world["root"] / "unkeyed" / "collection_manifest.json").exists()


@pytest.mark.parametrize(
    "change, message",
    [
        ({"revision": "provider-2"}, "on selector_revision"),
        ({"engine_name": "provider:engine8"}, "on engine_name"),
    ],
)
def test_what_cannot_be_one_model_is_refused(world, change, message):
    """A collection under another binding is refused when converted into the dataset."""
    from uhd_gen.dataset import store

    first = _collect(world, "first", device="board-a")
    second = _collect(world, "second", device="board-b", **change)
    dataset = world["root"] / "one.dataset"
    store.add(dataset, *store.from_collection(first))
    with pytest.raises(store.DatasetError, match=message):
        store.add(dataset, *store.from_collection(second))
    # Converting the same collection again is a no-op, not a second measurement.
    assert store.add(dataset, *store.from_collection(first))["added"] is False


def test_a_metric_the_collection_did_not_measure_is_refused(world, evaluator, caplog):
    collection = _collect(world, "col")
    code, _ = _train(
        world, "time", evaluator, collections=[collection], extra=["--metric", "time"]
    )
    assert code == 1
    assert "did not measure ['time']" in caplog.text


def test_measurement_options_are_refused_without_a_measurement(
    world, evaluator, caplog
):
    collection = _collect(world, "col")
    code, _ = _train(
        world, "knobbed", evaluator, collections=[collection], extra=["--device", "0"]
    )
    assert code == 1
    assert "measures nothing" in caplog.text


def test_rows_carry_the_regime_their_corpus_labelled_them_with(world, evaluator):
    """hipdnn_corpus_gen's manifest.csv names each graph's population."""
    import csv

    with (world["graphs"] / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["benchmark", "name", "regime", "phase", "context", "source", "q.size"]
        )
        for index in range(GRAPHS):
            phase = "decode" if index % 4 == 0 else "append"
            writer.writerow(
                [
                    f"graph-{index}",
                    f"g{index}",
                    f"{phase}_short",
                    phase,
                    "short",
                    "sweep",
                    index,
                ]
            )
    collection = _collect(world, "labelled")
    rows = json.loads((collection / "corpus.json").read_text(encoding="utf-8"))
    assert {
        r["benchmark"]: (r["regime"], r["regime.phase"], r["regime.context"])
        for r in rows
    } == {
        f"graph-{i}": (
            "decode_short" if i % 4 == 0 else "append_short",
            "decode" if i % 4 == 0 else "append",
            "short",
        )
        for i in range(GRAPHS)
    }
    assert not any(
        name.startswith("regime") for name in json.loads(rows[0]["features"])
    ), "a label is an envelope column, never a model input"
    code, output = _train(world, "from_labelled", evaluator, collections=[collection])
    assert code == 0
    assert {
        r["regime"]
        for r in json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    } == {"decode_short", "append_short"}


def test_graphs_without_a_corpus_manifest_carry_no_regime(world):
    collection = _collect(world, "unlabelled")
    rows = json.loads((collection / "corpus.json").read_text(encoding="utf-8"))
    assert not any("regime" in row for row in rows)


def test_shards_partition_a_corpus_into_collections_that_train_as_one(world, evaluator):
    """Shards measure every graph exactly once (e.g. one shard per GPU)."""
    from uhd_gen.__main__ import main

    shards = []
    for index in range(3):
        output = world["root"] / f"shard{index}"
        assert (
            main(
                [
                    "generate",
                    "--collect-only",
                    "--shard",
                    f"{index}/3",
                    "--graphs",
                    str(world["graphs"]),
                    "--descriptor-tree",
                    str(world["tree"]),
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
        shards.append(output)
    measured = [
        {
            r["benchmark"]
            for r in json.loads((s / "corpus.json").read_text(encoding="utf-8"))
        }
        for s in shards
    ]
    assert sum(len(m) for m in measured) == GRAPHS and set().union(*measured) == {
        f"graph-{i}" for i in range(GRAPHS)
    }, "a partition: nothing twice, nothing missed"
    assert [
        json.loads((s / "collection_manifest.json").read_text())["shard"]
        for s in shards
    ] == ["0/3", "1/3", "2/3"]
    code, output = _train(world, "sharded", evaluator, collections=shards)
    assert code == 0
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["superseded_rows"] == 0
    assert (
        len(json.loads((output / "corpus.json").read_text(encoding="utf-8"))) == GRAPHS
    )


@pytest.mark.parametrize(
    "shard, message", [("3/3", "0 <= K < N"), ("x", "K/N"), ("1/0", "0 <= K < N")]
)
def test_a_malformed_shard_is_refused(world, caplog, shard, message):
    from uhd_gen.__main__ import main

    assert (
        main(
            [
                "generate",
                "--collect-only",
                "--shard",
                shard,
                "--graphs",
                str(world["graphs"]),
                "--descriptor-tree",
                str(world["tree"]),
                "--engine-id",
                "7",
                "--role",
                "predict_engine",
                "--output-dir",
                str(world["root"] / "bad"),
            ]
        )
        == 1
    )
    assert message in caplog.text


def test_a_closed_shape_space_trains_on_every_shape_and_reports_recall(
    world, evaluator
):
    """`--recall` trains on every shape: a pack-bound engine sees no others."""
    collection = _collect(world, "col")
    held, _ = _train(world, "held", evaluator, collections=[collection])
    assert held == 0
    code, output = _train(
        world, "recall", evaluator, collections=[collection], extra=["--recall"]
    )
    assert code == 0
    trained = json.loads((output / "train.json").read_text(encoding="utf-8"))
    corpus = json.loads((output / "corpus.json").read_text(encoding="utf-8"))
    assert {r["benchmark"] for r in trained} == {r["benchmark"] for r in corpus}
    held_out = json.loads(
        (world["root"] / "held" / "train.json").read_text(encoding="utf-8")
    )
    assert len(held_out) < len(trained), "the default still holds a slice out"
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert (manifest["evaluation"], manifest["eval_fraction"]) == ("recall", None)
    report = json.loads(
        (output / "model" / "eval_report.json").read_text(encoding="utf-8")
    )
    assert report["metrics"]["problems_scored"] == GRAPHS, "recall scores every shape"
    # Scored on its own training shapes, so it never claims a held-out slice.
    assert report["holdout_integrity"]["status"] == "recall"


def test_a_named_evaluation_set_holds_out_exactly_its_graphs(world, evaluator):
    """`--eval-benchmarks`: the held-out graphs are the ones a corpus fixed when it was made,
    not a fraction drawn from whatever this run happens to train on -- so they stay held out,
    and keep scoring, however the collections grow."""
    first = _collect(world, "a", collected_at="2026-01-01T00:00:00+00:00")
    second = _collect(world, "b", collected_at="2026-01-02T00:00:00+00:00")
    named = ["graph-1", "graph-4", "graph-9", "graph-12"]
    eval_set = world["root"] / "eval_slice.json"
    eval_set.write_text(
        json.dumps({"benchmarks": [*named, "graph-never-measured"]}), encoding="utf-8"
    )
    code, output = _train(
        world,
        "sliced",
        evaluator,
        collections=[first, second],
        extra=["--eval-benchmarks", str(eval_set)],
    )
    assert code == 0
    trained = {
        r["benchmark"]
        for r in json.loads((output / "train.json").read_text(encoding="utf-8"))
    }
    assert not trained & set(named)
    assert trained == {f"graph-{i}" for i in range(GRAPHS)} - set(named)
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evaluation"] == "eval_set" and manifest["eval_fraction"] is None
    assert (
        manifest["eval_benchmarks"]["listed"],
        manifest["eval_benchmarks"]["measured"],
    ) == (5, 4)
    # Every named graph measured is scored -- once, on the best measurement of it, as
    # training takes one measurement per shape -- and none trains, whichever GPU took it.
    scored = manifest["models"][0]["eval_problem_keys"]
    assert sorted(scored) == sorted([graph, "board"] for graph in named)


@pytest.mark.parametrize(
    "contents, extra, message",
    [
        (["graph-1"], ["--recall"], "--recall trains on every one"),
        (["graph-absent"], [], "none of the 1 --eval-benchmarks graphs was measured"),
        ([], [], "names no graph"),
    ],
)
def test_an_evaluation_set_that_cannot_hold_anything_out_is_refused(
    world, evaluator, caplog, contents, extra, message
):
    collection = _collect(world, "col")
    eval_set = world["root"] / "eval.json"
    eval_set.write_text(json.dumps(contents), encoding="utf-8")
    code, _ = _train(
        world,
        "refused",
        evaluator,
        collections=[collection],
        extra=["--eval-benchmarks", str(eval_set), *extra],
    )
    assert code != 0
    assert message in caplog.text


def test_a_run_can_train_on_some_contributions_of_a_dataset(world, evaluator):
    from uhd_gen.__main__ import main

    first = _collect(world, "first", collected_at="2026-01-01T00:00:00+00:00")
    second = _collect(
        world, "second", collected_at="2026-01-02T00:00:00+00:00", scale=2.0
    )
    dataset = _dataset(world, "both", [first, second])
    common = [
        "generate",
        "--dataset",
        str(dataset),
        "--descriptor-tree",
        str(world["tree"]),
        "--engine-id",
        "7",
        "--role",
        "predict_engine",
        "--features",
        "graph.flops",
        "--num-boost-round",
        "4",
        "--early-stopping",
        "2",
        "--eval-fraction",
        "0.25",
        "--arch",
        "gfx942",
        "--no-promote",
        "--feature-evaluator",
        evaluator,
    ]
    output = world["root"] / "first-only"
    assert main([*common, "--contribution", "first", "--output-dir", str(output)]) == 0
    manifest = json.loads(
        (output / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert [c["label"] for c in manifest["collections"]] == ["first"]
    assert manifest["superseded_rows"] == 0, "the newer contribution was not read"
    refused = world["root"] / "absent"
    assert (
        main([*common, "--contribution", "absent", "--output-dir", str(refused)]) == 1
    )


def test_training_reads_the_stored_table_and_a_subset_computes_the_same(tmp_path):
    """load_dataset's rows from training.parquet equal those computed from the history."""
    from uhd_gen import generate
    from uhd_gen.dataset import store

    def collection(name, at, times):
        directory = tmp_path / name
        directory.mkdir()
        rows = [
            {
                "benchmark": f"g{index}",
                "device": "d0",
                "arch": "gfx942",
                "kernel": "k0",
                "knob_settings": "{}",
                "avgTimeMs": time,
            }
            for index, time in enumerate(times)
        ]
        (directory / "corpus.json").write_text(json.dumps(rows))
        (directory / store.COLLECTION_MANIFEST).write_text(
            json.dumps(
                {
                    "schema": store.COLLECTION_SCHEMA,
                    "collected_at": at,
                    "role": "sort_kernel_catalog",
                    "engine": "e",
                    "engine_id": 7,
                    "engine_name": None,
                    "sources": [None],
                    "trained_against": {"ued": {"id": "u"}},
                    "selector_revision": None,
                    "arches": ["gfx942"],
                    "devices": ["d0"],
                    "graph_count": len(rows),
                    "row_counts": {"None": len(rows)},
                    "published": [],
                    "kernel_fields": [],
                    "knob_encodings": {},
                    "shipping_knobs": [],
                    "collection_knobs": [],
                    "graphs": [],
                    "commands": [],
                    "failed_graphs": [],
                }
            )
        )
        return directory

    dataset = tmp_path / "ds"
    store.add(
        dataset, *store.from_collection(collection("a", "2026-10-01", [1.0, 3.0]))
    )
    store.add(
        dataset, *store.from_collection(collection("b", "2026-10-02", [2.0, 2.0]))
    )
    labels = [c["label"] for c in store.read_manifest(dataset)["contributions"]]

    stored = generate.load_dataset(dataset, role="sort_kernel_catalog", sources=[None])
    computed = generate.load_dataset(
        dataset, role="sort_kernel_catalog", sources=[None], only=labels
    )
    assert stored["rows"] == computed["rows"]
    assert [r["avgTimeMs"] for r in stored["rows"][None]] == [1.0, 2.0]
    assert stored["superseded_rows"] == computed["superseded_rows"] == 2


def test_a_training_table_the_manifest_does_not_describe_is_not_read(tmp_path):
    """A write interrupted after training.parquet but before the manifest leaves a table the
    manifest does not describe; training recomputes rather than trusting it."""
    import pandas as pd
    from uhd_gen import generate
    from uhd_gen.dataset import store

    directory = tmp_path / "a"
    directory.mkdir()
    rows = [
        {
            "benchmark": f"g{index}",
            "device": "d0",
            "arch": "gfx942",
            "kernel": "k0",
            "knob_settings": "{}",
            "avgTimeMs": 1.0 + index,
        }
        for index in range(3)
    ]
    (directory / "corpus.json").write_text(json.dumps(rows))
    (directory / store.COLLECTION_MANIFEST).write_text(
        json.dumps(
            {
                "schema": store.COLLECTION_SCHEMA,
                "collected_at": "2026-10-01",
                "role": "sort_kernel_catalog",
                "engine": "e",
                "engine_id": 7,
                "engine_name": None,
                "sources": [None],
                "trained_against": {"ued": {"id": "u"}},
                "selector_revision": None,
                "arches": ["gfx942"],
                "devices": ["d0"],
                "graph_count": 3,
                "row_counts": {"None": 3},
                "published": [],
                "kernel_fields": [],
                "knob_encodings": {},
                "shipping_knobs": [],
                "collection_knobs": [],
                "graphs": [],
                "commands": [],
                "failed_graphs": [],
            }
        )
    )
    dataset = tmp_path / "ds"
    store.add(dataset, *store.from_collection(directory))
    # What an interrupted later write leaves: a table holding a shape fewer.
    table = dataset / store.TRAINING_FILE
    pd.read_parquet(table).iloc[1:].to_parquet(table, index=False)

    loaded = generate.load_dataset(dataset, role="sort_kernel_catalog", sources=[None])
    assert sorted(r["benchmark"] for r in loaded["rows"][None]) == ["g0", "g1", "g2"]
