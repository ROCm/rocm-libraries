# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The stored form of training data: what conversion appends, and what it refuses."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from uhd_gen import addressing  # noqa: E402
from uhd_gen.dataset import store  # noqa: E402


def _collection(
    tmp_path, name, *, at="2026-10-01T00:00:00+00:00", rows=None, **manifest
):
    directory = tmp_path / name
    directory.mkdir()
    rows = (
        rows
        if rows is not None
        else [
            {
                "benchmark": f"{name}-g{i}",
                "device": "d0",
                "arch": "gfx942",
                "kernel": f"k{i}",
                "knob_settings": "{}",
                "avgTimeMs": 1.0 + i,
                "numerically_valid": None,
            }
            for i in range(3)
        ]
    )
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
                "published": ["q.M"],
                "kernel_fields": ["kernel.dtype"],
                "knob_encodings": {},
                "shipping_knobs": [],
                "collection_knobs": [],
                "graphs": [],
                "commands": [],
                "failed_graphs": [],
                **manifest,
            }
        )
    )
    return directory


def test_a_collection_reads_back_as_the_records_it_wrote(tmp_path):
    dataset = tmp_path / "ds"
    narrow = _collection(tmp_path, "a")
    wide = _collection(
        tmp_path,
        "b",
        rows=[
            {
                "benchmark": "b-g0",
                "device": "d0",
                "arch": "gfx942",
                "kernel": "k0",
                "knob_settings": "{}",
                "avgTimeMs": 2.0,
                "extra": "only here",
            }
        ],
    )
    results = store.add_many(
        dataset, [store.from_collection(narrow), store.from_collection(wide)]
    )
    assert [r["added"] for r in results] == [True, True]
    manifest = store.read_manifest(dataset)
    # Each contribution records the keys its rows carried: the table's union ("extra") is not
    # one of them for the narrow collection, so a reader can hand back exactly its records.
    assert "extra" not in manifest["contributions"][0]["columns"]["None"]
    assert "extra" in manifest["contributions"][1]["columns"]["None"]
    rows = store.read_rows(dataset)
    ids = {c["label"]: c["contribution_id"] for c in manifest["contributions"]}
    assert {r["contribution_id"] for r in rows} == {ids["a"], ids["b"]}
    assert all(r["purpose"] == "training" and r["source"] is None for r in rows)
    # Missing reads back as None, as the JSON held it, not NaN.
    assert (
        next(r for r in rows if r["contribution_id"] == ids["a"])["numerically_valid"]
        is None
    )


def test_converting_the_same_collection_twice_adds_nothing(tmp_path):
    dataset = tmp_path / "ds"
    collection = _collection(tmp_path, "a")
    assert store.add(dataset, *store.from_collection(collection))["added"] is True
    assert store.add(dataset, *store.from_collection(collection))["added"] is False
    assert len(store.read_rows(dataset)) == 3


def test_a_retry_after_a_write_interrupted_between_the_two_files_adds_no_rows_twice(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "ds"
    contribution = store.from_collection(_collection(tmp_path, "a"))
    write = store._write_atomically

    def crash_before_manifest(path, writer):
        if path.name == store.MANIFEST_FILE:
            raise OSError("killed")
        write(path, writer)

    monkeypatch.setattr(store, "_write_atomically", crash_before_manifest)
    with pytest.raises(OSError, match="killed"):
        store.add(dataset, *contribution)
    monkeypatch.undo()

    result = store.add(dataset, *contribution)
    assert result["added"] is True
    assert len(store.read_rows(dataset)) == 3
    recorded = store.read_manifest(dataset)["contributions"]
    assert [c["contribution_id"] for c in recorded] == [result["contribution_id"]]
    assert store.add(dataset, *contribution)["added"] is False


def test_a_dataset_holds_one_binding(tmp_path):
    dataset = tmp_path / "ds"
    store.add(dataset, *store.from_collection(_collection(tmp_path, "a")))
    other = _collection(tmp_path, "b", trained_against={"ued": {"id": "v"}})
    with pytest.raises(store.DatasetError, match="on trained_against"):
        store.add(dataset, *store.from_collection(other))


def test_knob_addressing_that_contradicts_the_dataset_is_refused_and_nothing_is_written(
    tmp_path,
):
    def table(pin, value):
        return addressing.as_manifest(
            addressing.observe(
                [
                    {
                        "knob_settings": {"dtype": pin},
                        "kernel_features": {"kernel.dtype": value},
                    }
                ]
            )
        )

    dataset = tmp_path / "ds"
    store.add(
        dataset,
        *store.from_collection(
            _collection(tmp_path, "a", knob_encodings=table(0, "BF16"))
        ),
    )
    before = (dataset / store.DATA_FILE).read_bytes()
    with pytest.raises(store.DatasetError, match="knob addressing"):
        store.add_many(
            dataset,
            [
                store.from_collection(_collection(tmp_path, "b")),
                store.from_collection(
                    _collection(tmp_path, "c", knob_encodings=table(0, "FP16"))
                ),
            ],
        )
    assert (
        dataset / store.DATA_FILE
    ).read_bytes() == before, "all of the call or none of it"
    assert [c["label"] for c in store.read_manifest(dataset)["contributions"]] == ["a"]


def test_a_purpose_outside_the_three_is_refused(tmp_path):
    with pytest.raises(store.DatasetError, match="purpose"):
        store.add(
            tmp_path / "ds",
            *store.from_collection(_collection(tmp_path, "a"), purpose="scratch"),
        )


def test_published_csv_converts_with_the_binding_the_caller_states(tmp_path):
    csv = tmp_path / "sweep.csv"
    pd.DataFrame(
        {
            "benchmark": ["p1", "p1", "p2", "p2"],
            "device": ["d"] * 4,
            "q.M": [64, 64, 128, 128],
            "q.flops": [1e9] * 4,
            "q.bytes": [1e6] * 4,
            "kernel.solver_id": [1, 2, 1, 2],
            "kernel.descriptor": ["8,2", "4,4,1", "8,2", "4,4,1"],
            "device.cu_count": [80] * 4,
            "minTimeMs": [1.0, 2.0, 1.5, 1.0],
            "avgTimeMs": [1.1, 2.1, 1.6, 1.1],
            "stddevMs": [0.0] * 4,
            "iters": [10] * 4,
        }
    ).to_csv(csv, index=False)
    entry, rows = store.from_csv(
        [csv],
        engine="MIOPEN",
        engine_id=11,
        role="sort_kernel_catalog",
        arch="gfx942",
        selector_revision="miopen/abc",
        collected_at="2026-10-02T00:00:00+00:00",
        label="perfdb-1",
        expand_descriptor=["kernel.descriptor"],
        scope_by="kernel.solver_id",
    )
    assert entry["trained_against"] == {"selector_revision": "miopen/abc"}
    assert entry["training_options"] == {
        "expand_descriptor": ["kernel.descriptor"],
        "scope_by": "kernel.solver_id",
    }
    record = rows[None][0]
    assert record["arch"] == "gfx942" and "tflops" in record
    assert any(
        key.startswith("kernel.descriptor.s1_f") for key in record
    ), "expanded per solver"
    assert store.add(tmp_path / "ds", entry, rows)["rows"] == 4


def test_a_csv_that_names_no_device_kind_is_refused_by_column(tmp_path):
    """`device.*` features describe a device without naming it; the store keys on `device`."""
    csv = tmp_path / "sweep.csv"
    pd.DataFrame(
        {
            "benchmark": ["p1", "p1"],
            "q.M": [64, 64],
            "q.flops": [1e9] * 2,
            "q.bytes": [1e6] * 2,
            "kernel.solver_id": [1, 2],
            "device.cu_count": [80] * 2,
            "minTimeMs": [1.0, 2.0],
            "avgTimeMs": [1.1, 2.1],
            "stddevMs": [0.0] * 2,
            "iters": [10] * 2,
        }
    ).to_csv(csv, index=False)
    with pytest.raises(store.DatasetError, match="'device'"):
        store.from_csv(
            [csv],
            engine="MIOPEN",
            engine_id=11,
            role="sort_kernel_catalog",
            arch="gfx942",
            selector_revision="miopen/abc",
            collected_at="2026-10-02T00:00:00+00:00",
        )


def test_a_hipdnn_bench_csv_converts_with_its_untimed_candidates_as_failures(tmp_path):
    """fixtures/hipdnn_bench.csv is hipdnn_bench's CSV (TestCsvOutput pins its header): a row
    keyed by graph and device kind, and a candidate that did not run, with empty timings.
    """
    fixture = pathlib.Path(__file__).parent / "fixtures" / "hipdnn_bench.csv"
    entry, rows = store.from_csv(
        [fixture],
        engine="hipkernel:Example",
        engine_id=7,
        role="sort_kernel_catalog",
        arch="gfx942",
        selector_revision="example/1",
        collected_at="2026-10-15T00:00:00+00:00",
    )
    assert entry["devices"] == ["8215566447713624081"] and entry["graph_count"] == 1
    timed, untimed = sorted(rows[None], key=lambda r: r["kernel.tile_m"])
    assert timed["minTimeMs"] == 0.5 and not timed["error"]
    assert untimed["minTimeMs"] is None
    assert untimed["error"].startswith("config_not_applicable")
    assert store.add(tmp_path / "ds", entry, rows)["rows"] == 2


def test_the_command_line_refuses_csv_without_a_binding(tmp_path, capsys):
    with pytest.raises(SystemExit):
        store.main(["--into", str(tmp_path / "ds"), "--csv", str(tmp_path / "x.csv")])
    assert "carry no binding" in capsys.readouterr().err


def test_export_hands_back_the_rows_of_the_contributions_named(tmp_path):
    dataset = tmp_path / "ds"
    store.add(dataset, *store.from_collection(_collection(tmp_path, "train")))
    store.add(
        dataset, *store.from_collection(_collection(tmp_path, "held"), purpose="eval")
    )
    held = store.export_rows(dataset, purposes=["eval"])
    assert {r["benchmark"] for r in held} == {f"held-g{i}" for i in range(3)}
    assert not any(key in row for row in held for key in store.BOOKKEEPING)
    assert (
        store.export_rows(dataset, contributions=["train"])[0]["benchmark"]
        == "train-g0"
    )
    with pytest.raises(store.DatasetError, match="no contribution"):
        store.export_rows(dataset, contributions=["absent"])


def test_export_keeps_one_metric_where_a_contribution_measured_several(tmp_path):
    directory = _collection(
        tmp_path, "two", sources=["tflops", "time"], row_counts={"tflops": 1, "time": 1}
    )
    row = json.loads((directory / "corpus.json").read_text())[0]
    (directory / "corpus.json").unlink()
    (directory / "corpus_tflops.json").write_text(json.dumps([{**row, "kernel": "t"}]))
    (directory / "corpus_time.json").write_text(json.dumps([{**row, "kernel": "m"}]))
    store.add(tmp_path / "ds", *store.from_collection(directory))
    store.add(tmp_path / "ds", *store.from_collection(_collection(tmp_path, "one")))
    rows = store.export_rows(tmp_path / "ds", purposes=["training"], source="time")
    assert sorted(r["kernel"] for r in rows) == ["k0", "k1", "k2", "m"]


#: projects/hipdnn/tools, so `python -m uhd_gen.dataset` runs from the checkout.
TOOLS = pathlib.Path(__file__).resolve().parents[3]


def _dataset_command(*arguments):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TOOLS), env.get("PYTHONPATH")])
    )
    return subprocess.run(
        [sys.executable, "-m", "uhd_gen.dataset", *map(str, arguments)],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.parametrize(
    ("arguments", "said"),
    [
        (["remove", "--from", "{missing}", "--contribution", "c"], "dataset remove:"),
        (["retag", "--in", "{missing}", "--contribution", "c"], "dataset retag:"),
        (
            ["export", "--from", "{missing}", "--out", "{out}", "--purpose", "eval"],
            "dataset export:",
        ),
    ],
    ids=lambda case: case[0] if isinstance(case, list) else "",
)
def test_each_subcommand_runs_its_own_handler_and_exits_with_its_status(
    tmp_path, arguments, said
):
    """A refused request exits 1 from the handler it named, not through another one."""
    filled = [
        a.format(missing=tmp_path / "missing", out=tmp_path / "out.json")
        for a in arguments
    ]
    result = _dataset_command(*filled)
    assert result.returncode == 1, result.stderr
    assert said in result.stderr


def test_without_a_subcommand_the_dataset_command_publishes(tmp_path):
    result = _dataset_command("--out", tmp_path / "out")
    assert result.returncode == 2
    assert "usage: python -m uhd_gen.dataset [" in result.stderr
    assert "--csv" in result.stderr


def test_add_and_export_round_trip_on_the_command_line(tmp_path):
    dataset = tmp_path / "ds"
    added = _dataset_command(
        "add",
        "--into",
        dataset,
        "--collection",
        _collection(tmp_path, "train"),
        _collection(tmp_path, "other"),
    )
    assert added.returncode == 0, added.stderr
    assert [c["added"] for c in json.loads(added.stdout)["contributions"]] == [
        True,
        True,
    ]
    held = _dataset_command(
        "add",
        "--into",
        dataset,
        "--collection",
        _collection(tmp_path, "held"),
        "--purpose",
        "eval",
    )
    assert held.returncode == 0, held.stderr

    out = tmp_path / "held.json"
    exported = _dataset_command(
        "export", "--from", dataset, "--out", out, "--purpose", "eval"
    )
    assert exported.returncode == 0, exported.stderr
    assert json.loads(exported.stdout)["rows"] == 3
    original = json.loads((tmp_path / "held" / "corpus.json").read_text())
    assert json.loads(out.read_text()) == original, "rows as the source wrote them"

    missing = _dataset_command(
        "export", "--from", dataset, "--out", out, "--contribution", "absent"
    )
    assert missing.returncode == 1 and "no contribution" in missing.stderr


def test_one_label_cannot_name_several_collections(tmp_path, capsys):
    with pytest.raises(SystemExit):
        store.main(
            [
                "--into",
                str(tmp_path / "ds"),
                "--label",
                "x",
                "--collection",
                str(_collection(tmp_path, "a")),
                str(_collection(tmp_path, "b")),
            ]
        )
    assert "give one --collection" in capsys.readouterr().err


def test_the_same_measurements_under_another_name_are_added_once(tmp_path):
    dataset = tmp_path / "ds"
    collection = _collection(tmp_path, "a")
    first = store.add(dataset, *store.from_collection(collection))
    again = store.add(dataset, *store.from_collection(collection, label="renamed"))
    assert again == {**first, "added": False, "rows": 0, "label": "renamed"}
    assert len(store.read_rows(dataset)) == 3


def test_new_measurements_under_a_reused_name_are_stored_not_dropped(tmp_path):
    dataset = tmp_path / "ds"
    store.add(dataset, *store.from_collection(_collection(tmp_path, "a")))
    (tmp_path / "retry").mkdir()
    later = _collection(tmp_path / "retry", "a", at="2026-10-02T00:00:00+00:00")
    rows = json.loads((later / "corpus.json").read_text())
    rows[0]["avgTimeMs"] = 9.0
    (later / "corpus.json").write_text(json.dumps(rows))
    assert store.add(dataset, *store.from_collection(later))["added"] is True
    assert [c["label"] for c in store.read_manifest(dataset)["contributions"]] == [
        "a",
        "a",
    ]
    assert len(store.read_rows(dataset)) == 6


def test_replace_swaps_a_contribution_in_one_write_and_records_it(tmp_path):
    dataset = tmp_path / "ds"
    old = store.add(dataset, *store.from_collection(_collection(tmp_path, "a")))
    (tmp_path / "fixed").mkdir()
    fixed = _collection(
        tmp_path / "fixed",
        "a",
        rows=[
            {
                "benchmark": "g",
                "device": "d0",
                "arch": "gfx942",
                "kernel": "k0",
                "knob_settings": "{}",
                "avgTimeMs": 4.0,
            }
        ],
    )
    new = store.add_many(dataset, [store.from_collection(fixed)], replace=["a"])[0]
    manifest = store.read_manifest(dataset)
    assert [c["contribution_id"] for c in manifest["contributions"]] == [
        new["contribution_id"]
    ]
    assert {r["contribution_id"] for r in store.read_rows(dataset)} == {
        new["contribution_id"]
    }
    (event,) = manifest["history"]
    assert event["action"] == "replace"
    assert event["contribution_id"] == old["contribution_id"]
    assert event["replaced_by"] == [new["contribution_id"]]


def test_remove_drops_rows_and_entry_and_an_unknown_name_writes_nothing(tmp_path):
    dataset = tmp_path / "ds"
    store.add(dataset, *store.from_collection(_collection(tmp_path, "a")))
    kept = store.add(dataset, *store.from_collection(_collection(tmp_path, "b")))
    before = (dataset / store.DATA_FILE).read_bytes()
    with pytest.raises(store.DatasetError, match="no contribution"):
        store.remove(dataset, ["a", "absent"])
    assert (dataset / store.DATA_FILE).read_bytes() == before

    (removed,) = store.remove(dataset, ["a"])
    assert removed["label"] == "a"
    manifest = store.read_manifest(dataset)
    assert [c["label"] for c in manifest["contributions"]] == ["b"]
    assert {r["contribution_id"] for r in store.read_rows(dataset)} == {
        kept["contribution_id"]
    }
    assert manifest["history"][0]["action"] == "remove"
    assert manifest["history"][0]["entry"]["label"] == "a"


def test_retag_changes_the_purpose_and_keeps_the_measurements(tmp_path):
    dataset = tmp_path / "ds"
    added = store.add(dataset, *store.from_collection(_collection(tmp_path, "a")))
    assert store.retag(dataset, ["a"], purpose="holdout") == [added["contribution_id"]]
    (entry,) = store.read_manifest(dataset)["contributions"]
    assert entry["purpose"] == "holdout"
    assert entry["contribution_id"] == added["contribution_id"]
    assert {r["purpose"] for r in store.read_rows(dataset)} == {"holdout"}
    assert store.export_rows(dataset, purposes=["training"]) == []
    assert len(store.export_rows(dataset, purposes=["holdout"])) == 3


def test_rows_that_disagree_on_a_column_type_are_refused_whole(tmp_path):
    dataset = tmp_path / "ds"
    store.add(dataset, *store.from_collection(_collection(tmp_path, "a")))
    before = (dataset / store.DATA_FILE).read_bytes()
    clash = _collection(
        tmp_path,
        "b",
        rows=[
            {
                "benchmark": "b-g0",
                "device": "d0",
                "arch": "gfx942",
                "kernel": "k0",
                "knob_settings": "{}",
                "avgTimeMs": {"not": "a number"},
            }
        ],
    )
    with pytest.raises(store.DatasetError, match="one Parquet table"):
        store.add(dataset, *store.from_collection(clash))
    assert (dataset / store.DATA_FILE).read_bytes() == before
    assert len(store.read_manifest(dataset)["contributions"]) == 1


def test_a_held_lock_times_out_naming_itself(tmp_path):
    (tmp_path / ".lock").mkdir()
    with pytest.raises(store.DatasetError, match="held for over"):
        with store._locked(tmp_path, timeout_s=0.0):
            pass
    assert (tmp_path / ".lock").is_dir(), "another writer's lock is left alone"


def _timed(name, times, **extra):
    """Rows of one configuration on graph g, one per time; `extra` marks them."""
    return [
        {
            "benchmark": "g",
            "device": "d0",
            "arch": "gfx942",
            "kernel": "k0",
            "knob_settings": "{}",
            "avgTimeMs": time,
            "note": name,
            **extra,
        }
        for time in times
    ]


def test_the_training_table_keeps_the_best_usable_measurement(tmp_path):
    """training.parquet: errors removed, repeats reduced to the lowest avgTimeMs."""
    dataset = tmp_path / "ds"
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    store.add(
        dataset,
        *store.from_collection(
            _collection(tmp_path / "x", "a", rows=_timed("a", [2.0]))
        ),
    )
    store.add(
        dataset,
        *store.from_collection(
            _collection(
                tmp_path / "y",
                "b",
                at="2026-10-02T00:00:00+00:00",
                rows=_timed("b", [1.5])
                + _timed("wrong", [0.1], numerically_valid=False)
                + _timed("failed", [None], is_valid=False),
            )
        ),
    )
    assert len(store.read_rows(dataset)) == 4, "the history keeps everything"
    (row,) = store.read_training_rows(dataset)
    assert (row["note"], row["avgTimeMs"]) == ("b", 1.5)
    summary = store.read_manifest(dataset)["training_view"]["sources"]["None"]
    assert summary == {
        "rows": 1,
        "superseded": 1,
        "excluded": {"numerically_wrong": 1, "invalid": 1},
    }


def test_the_training_table_follows_remove_and_retag(tmp_path):
    dataset = tmp_path / "ds"
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    store.add(
        dataset,
        *store.from_collection(
            _collection(tmp_path / "x", "a", rows=_timed("a", [1.0]))
        ),
    )
    store.add(
        dataset,
        *store.from_collection(
            _collection(tmp_path / "y", "b", rows=_timed("b", [2.0]))
        ),
    )
    assert [r["note"] for r in store.read_training_rows(dataset)] == ["a"]
    store.retag(dataset, ["a"], purpose="eval")
    assert [r["note"] for r in store.read_training_rows(dataset)] == ["b"]
    store.remove(dataset, ["b"])
    assert store.read_training_rows(dataset) == []
    assert store.read_manifest(dataset)["training_view"]["contributions"] == []


def test_exported_rows_added_again_keep_their_id(tmp_path):
    """A dataset rebuilt from exported rows names each contribution as the original did,
    though Parquet hands back an int column holding a gap as floats."""
    rows = [
        {
            "benchmark": "g0",
            "device": "d0",
            "arch": "gfx942",
            "kernel": "k0",
            "knob_settings": "{}",
            "avgTimeMs": 1.5,
            "iters": 10,
            "rank": None,
        },
        {
            "benchmark": "g1",
            "device": "d0",
            "arch": "gfx942",
            "kernel": "k0",
            "knob_settings": "{}",
            "avgTimeMs": 2.5,
            "iters": 20,
            "rank": 3,
        },
    ]
    original = tmp_path / "original"
    entry, _ = store.from_collection(_collection(tmp_path, "a", rows=rows))
    added = store.add(original, *store.from_collection(tmp_path / "a"))
    (stored,) = store.read_manifest(original)["contributions"]
    exported = store.export_rows(original, contributions=[added["contribution_id"]])
    again = store.add(tmp_path / "rebuilt", stored, {None: exported})
    assert again["contribution_id"] == added["contribution_id"]


def test_rows_with_different_keys_exported_and_added_again_keep_their_id(tmp_path):
    """An exported row carries every column of its contribution, as null where it had none;
    absent and null are one fact, so the measurements are not stored a second time."""
    rows = [
        {
            "benchmark": "g0",
            "device": "d0",
            "arch": "gfx942",
            "kernel": "k0",
            "knob_settings": "{}",
            "avgTimeMs": 1.5,
        },
        {
            "benchmark": "g1",
            "device": "d0",
            "arch": "gfx942",
            "kernel": "k0",
            "knob_settings": "{}",
            "avgTimeMs": 2.5,
            "graph.flops": 1e9,
        },
    ]
    original = tmp_path / "original"
    added = store.add(
        original, *store.from_collection(_collection(tmp_path, "a", rows=rows))
    )
    (stored,) = store.read_manifest(original)["contributions"]
    exported = store.export_rows(original, contributions=[added["contribution_id"]])
    assert "graph.flops" in exported[0], "the export fills the column the row lacked"
    again = store.add(original, stored, {None: exported})
    assert again["contribution_id"] == added["contribution_id"]
    assert not again["added"], "the same measurements are stored once"


def test_a_zero_time_is_no_measurement_and_never_replaces_a_real_one():
    """A 0 ms row is a timer that did not run, not the fastest kernel."""
    from uhd_gen.dataset.training_table import best_per_shape, measurement_error

    def row(time):
        return {
            "benchmark": "g",
            "arch": "gfx942",
            "device": "d",
            "kernel": "k",
            "knob_settings": "{}",
            "avgTimeMs": time,
        }

    assert measurement_error(row(0.0)) == "unmeasured"
    assert measurement_error(row(-1.0)) == "unmeasured"
    kept, superseded, excluded = best_per_shape([("a", row(1.0)), ("b", row(0.0))])
    assert kept == [0] and superseded == 0 and excluded == {"unmeasured": 1}
