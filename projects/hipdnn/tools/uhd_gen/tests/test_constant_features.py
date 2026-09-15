# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A column that never varies is dropped and named; pruning is never knob removal."""
import json

import pytest

pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")
pytest.importorskip("flatbuffers")

from uhd_gen.__main__ import main
from uhd_gen.features import compute_features_hash

PROVENANCE = {
    "ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "1.0"},
    "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.0"},
    "umd": [],
}

# What `deviceFeatureValues` publishes for the two gfx942 boards a merged sweep spans.
# Five of the eight fields are identical between them; `total_global_mem`,
# `memory_clock_rate` and the bandwidth derived from the clock are not. Using the real
# pair is the point of the test: the rule has to be decided by the numbers, not by the
# `device.` prefix on the name.
MI300X = {"device.cu_count": 304, "device.multi_processor_count": 304, "device.warp_size": 64,
          "device.lds_size": 65536, "device.memory_bus_width": 8192,
          "device.total_global_mem": 206158430208, "device.memory_clock_rate": 2600000,
          "device.peak_memory_bandwidth": 5.324e12}
MI325X = {**MI300X, "device.total_global_mem": 274877906944, "device.memory_clock_rate": 2933000,
          "device.peak_memory_bandwidth": 6.005e12}

DEVICE_FIELDS = sorted(MI300X)


def _run(tmp_path, frame, features):
    corpus = tmp_path / "corpus.csv"
    frame.to_csv(corpus, index=False)
    snapshot = tmp_path / "provenance.json"
    snapshot.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    output = tmp_path / "model"
    code = main(["train", "--input", str(corpus), "--provenance", str(snapshot),
                 "--features", *features, "--target", "tflops", "--output-dir", str(output),
                 "--num-boost-round", "10", "--early-stopping", "5"])
    return code, output


def _frame(rows):
    block = [64 if row % 2 else 256 for row in range(rows)]
    return pd.DataFrame({"kernel.block_size": block, "kernel.tile_m": [128] * rows,
                         "device.cu_count": [304] * rows,
                         "tflops": [120 - 0.2 * value for value in block]})


def _boards(board, rows=80):
    """One board's published device row, repeated under a varying kernel knob."""
    block = [64 if row % 2 else 256 for row in range(rows)]
    return pd.DataFrame({"kernel.block_size": block,
                         **{field: [board[field]] * rows for field in DEVICE_FIELDS},
                         "tflops": [120 - 0.2 * value for value in block]})


def test_a_constant_column_is_dropped_and_named_with_its_value(tmp_path, evaluator):
    code, output = _run(tmp_path, _frame(80),
                        ["kernel.block_size", "kernel.tile_m", "device.cu_count"])
    assert code == 0
    descriptor = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    assert descriptor["features_signature"] == ["$kernel.block_size"]
    # The contract the runtime checks is the pruned one (RFC 0019 §6.3), not the request.
    assert descriptor["features_hash"] == compute_features_hash(["$kernel.block_size"],
                                                                executable=evaluator)
    # Named with its value, so the omission is visible rather than silent: only the author
    # can say whether a field was pinned by the kernels or missed by the sweep.
    assert manifest["dropped_constant_features"] == [{"column": "kernel.tile_m", "value": 128},
                                                     {"column": "device.cu_count", "value": 304}]
    assert manifest["requested_features"] == ["kernel.block_size", "kernel.tile_m", "device.cu_count"]
    assert manifest["trained_against"] == PROVENANCE


def test_an_all_constant_feature_set_is_an_error_not_an_empty_signature(tmp_path, evaluator):
    # Dropping everything would leave a model that scores every candidate identically, and
    # shipping one is worse than shipping none: the engine ranks by a model that cannot
    # discriminate instead of falling back to its declared order.
    code, output = _run(tmp_path, _frame(80), ["kernel.tile_m", "device.cu_count"])
    assert code == 1
    assert not output.exists()


def test_a_single_board_corpus_keeps_no_device_column(tmp_path, evaluator):
    code, output = _run(tmp_path, _boards(MI300X), ["kernel.block_size", *DEVICE_FIELDS])
    assert code == 0
    descriptor = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    assert descriptor["features_signature"] == ["$kernel.block_size"]
    assert [entry["column"] for entry in manifest["dropped_constant_features"]] == DEVICE_FIELDS


def test_a_corpus_merged_across_two_boards_keeps_what_genuinely_varies(tmp_path, evaluator):
    # RFC 0019.13 states a normative "single-arch runs MUST NOT pass device.* as
    # features". That rule is wrong for this codebase:
    # GenericPlanBuilder::candidateFeatures merges a sweep across several boards of one
    # arch, and gfx942 spans MI300X and MI325X. A name-based rule would throw away the
    # three fields that separate them.
    merged = pd.concat([_boards(MI300X, rows=40), _boards(MI325X, rows=40)], ignore_index=True)
    code, output = _run(tmp_path, merged, ["kernel.block_size", *DEVICE_FIELDS])
    assert code == 0
    descriptor = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    assert descriptor["features_signature"] == [
        "$kernel.block_size", "$device.memory_clock_rate",
        "$device.peak_memory_bandwidth", "$device.total_global_mem",
    ]
    assert [entry["column"] for entry in manifest["dropped_constant_features"]] == [
        "device.cu_count", "device.lds_size", "device.memory_bus_width",
        "device.multi_processor_count", "device.warp_size",
    ]
