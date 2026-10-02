# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The stored form of training data: what conversion appends, and what it refuses."""

from __future__ import annotations

import json

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
    assert {r["contribution_id"] for r in rows} == {"a", "b"}
    assert all(r["purpose"] == "training" and r["source"] is None for r in rows)
    # Missing reads back as None, as the JSON held it, not NaN.
    assert (
        next(r for r in rows if r["contribution_id"] == "a")["numerically_valid"]
        is None
    )


def test_converting_the_same_collection_twice_adds_nothing(tmp_path):
    dataset = tmp_path / "ds"
    collection = _collection(tmp_path, "a")
    assert store.add(dataset, *store.from_collection(collection))["added"] is True
    assert store.add(dataset, *store.from_collection(collection))["added"] is False
    assert len(store.read_rows(dataset)) == 3


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
    assert [
        c["contribution_id"] for c in store.read_manifest(dataset)["contributions"]
    ] == ["a"]


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
        contribution_id="perfdb-1",
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
