#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the corpus-derived categorical encoding shipped in the descriptor.

RFC 0019 §4/§6.5.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip (not a collection error) when optional training deps are missing; must run
# before importing uhd_gen.__main__, which imports them.
pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")
pytest.importorskip("flatbuffers")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from uhd_gen.__main__ import main  # noqa: E402
from uhd_gen.evaluate import load_model  # noqa: E402
from uhd_gen.features import (  # noqa: E402
    build_features_signature,
    compute_features_hash,
    derive_categorical_encoding,
    encode_feature_value,
)

#: Enough rows for the 5-fold CV in train_model, with every value of every knob
#: present in every fold.
ROWS = 80


def _varying(low, high, period: int = 1) -> list:
    return [low if (row // period) % 2 == 0 else high for row in range(ROWS)]


def _corpus(path: Path, columns: dict[str, list]) -> Path:
    names = list(columns)
    lines = [",".join(names)]
    for row in range(ROWS):
        lines.append(",".join(str(columns[name][row]) for name in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _train(output_dir: Path, csv: Path, features: list[str], *extra: str) -> int:
    snapshot = output_dir.parent / "provenance.json"
    snapshot.write_text(
        json.dumps(
            {
                "ued": {
                    "id": "13ab344f-4818-4772-bb8e-8e1441fec82c",
                    "revision": "1.0",
                },
                "kmd": {
                    "id": "46d64d06-18eb-483d-9bb4-94472d32b78d",
                    "revision": "1.0",
                },
                "umd": [],
            }
        ),
        encoding="utf-8",
    )
    return main(
        [
            "train",
            "--provenance",
            str(snapshot),
            "--input",
            str(csv),
            "--features",
            *features,
            "--target",
            "tflops",
            "--output-dir",
            str(output_dir),
            "--num-boost-round",
            "10",
            "--early-stopping",
            "5",
            *extra,
        ]
    )


def _descriptor(output_dir: Path) -> dict:
    return json.loads((output_dir / "heuristic.uhd.json").read_text(encoding="utf-8"))


def _manifest(output_dir: Path) -> dict:
    return json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------
# Numeric-only corpora
# --------------------------------------------------------------------------------

NUMERIC_FEATURES = ["kernel.block_size", "device.cu_count"]


def _numeric_corpus(path: Path) -> Path:
    block = _varying(64, 256)
    cu = _varying(64, 304, period=2)
    return _corpus(
        path,
        {
            "kernel.block_size": block,
            "device.cu_count": cu,
            "tflops": [
                round(120.0 - 0.2 * block[row] + 0.05 * cu[row] + 0.01 * row, 4)
                for row in range(ROWS)
            ],
        },
    )


def test_numeric_only_corpus_derives_no_encoding(tmp_path):
    """Numeric columns are read straight through, so they get no map entry."""
    frame = pd.read_csv(_numeric_corpus(tmp_path / "bench.csv"))

    assert derive_categorical_encoding(frame, NUMERIC_FEATURES) == {}


# Tests that train or recompute a digest need `evaluator`: the hipdnn_uhd_features
# binary is the single definition of both (RFC 0019 §6.3).
def test_numeric_only_descriptor_gains_no_key_and_does_not_move_its_hash(
    tmp_path, evaluator
):
    """An empty encoding must not change the descriptor or the features_hash."""
    output_dir = tmp_path / "model"

    assert (
        _train(output_dir, _numeric_corpus(tmp_path / "bench.csv"), NUMERIC_FEATURES)
        == 0
    )

    descriptor = _descriptor(output_dir)
    assert "categorical_encoding" not in descriptor

    signature = build_features_signature(NUMERIC_FEATURES)
    assert descriptor["features_signature"] == signature
    assert descriptor["features_hash"] == compute_features_hash(
        signature, executable=evaluator
    )
    assert descriptor["features_hash"] == compute_features_hash(
        signature, {}, evaluator
    )


def test_manifest_records_the_empty_map(tmp_path, evaluator):
    """Unlike the descriptor, the manifest records an empty map explicitly."""
    output_dir = tmp_path / "model"

    assert (
        _train(output_dir, _numeric_corpus(tmp_path / "bench.csv"), NUMERIC_FEATURES)
        == 0
    )

    assert _manifest(output_dir)["categorical_encoding"] == {}


# --------------------------------------------------------------------------------
# Keyed by reference, not by field name
# --------------------------------------------------------------------------------


def test_two_columns_sharing_a_field_name_keep_separate_vocabularies():
    """Values differ only in case on purpose: keys must keep the corpus bytes, since
    the runtime looks them up verbatim."""
    frame = pd.DataFrame(
        {
            "kernel.dtype": ["BF16", "FP16", "BF16"],
            "q.attention_dense.dtype": ["bf16", "bf16", "fp8"],
        }
    )

    encoding = derive_categorical_encoding(
        frame, ["kernel.dtype", "q.attention_dense.dtype"]
    )

    assert encoding == {
        "$kernel.dtype": {"BF16": 0, "FP16": 1},
        "$q.attention_dense.dtype": {"bf16": 0, "fp8": 1},
    }


def test_keys_are_full_references_not_field_names():
    frame = pd.DataFrame({"kernel.dtype": ["bf16", "fp16"]})

    assert list(derive_categorical_encoding(frame, ["kernel.dtype"])) == [
        "$kernel.dtype"
    ]


def test_codes_are_deterministic_across_derivations():
    """Codes follow sorted order from 0; model.bin thresholds depend on them."""
    frame = pd.DataFrame(
        {"kernel.pipeline": ["pingpong", "intrawave", "v3", "intrawave"]}
    )

    first = derive_categorical_encoding(frame, ["kernel.pipeline"])
    second = derive_categorical_encoding(frame, ["kernel.pipeline"])

    assert first == second
    assert first == {"$kernel.pipeline": {"intrawave": 0, "pingpong": 1, "v3": 2}}


def test_mixed_column_is_rejected():
    """A raw number would silently collide with the string assigned that code."""
    frame = pd.DataFrame({"kernel.pipeline": ["intrawave", 3]})

    with pytest.raises(ValueError, match="mixes strings with"):
        derive_categorical_encoding(frame, ["kernel.pipeline"])


# --------------------------------------------------------------------------------
# The map that ships is the map that was fitted
# --------------------------------------------------------------------------------

STRING_FEATURES = ["kernel.block_size", "kernel.pipeline"]

#: Values of `kernel.pipeline` in the corpus below, in the order sorted() puts them.
PIPELINES = ["intrawave", "pingpong"]


def _string_corpus(path: Path) -> Path:
    """Corpus whose target depends strongly on a string column, forcing a split."""
    block = _varying(64, 256)
    pipeline = _varying(PIPELINES[1], PIPELINES[0], period=2)
    return _corpus(
        path,
        {
            "kernel.block_size": block,
            "kernel.pipeline": pipeline,
            "tflops": [
                round(
                    120.0
                    - 0.2 * block[row]
                    + (40.0 if pipeline[row] == "intrawave" else 0.0)
                    + 0.01 * row,
                    4,
                )
                for row in range(ROWS)
            ],
        },
    )


def test_a_string_has_no_number_without_a_derived_encoding():
    """Premise of the tests below: strings can only be encoded via a derived map."""
    with pytest.raises(ValueError, match="no categorical_encoding declares"):
        encode_feature_value("$kernel.pipeline", "intrawave")


def test_string_column_trains_and_ships_its_own_vocabulary(tmp_path, evaluator):
    """The shipped map is exactly the column's distinct values, as fitted."""
    output_dir = tmp_path / "model"
    csv = _string_corpus(tmp_path / "bench.csv")

    assert _train(output_dir, csv, STRING_FEATURES) == 0

    descriptor = _descriptor(output_dir)
    assert descriptor["categorical_encoding"] == {
        "$kernel.pipeline": {"intrawave": 0, "pingpong": 1}
    }
    frame = pd.read_csv(csv)
    assert set(descriptor["categorical_encoding"]["$kernel.pipeline"]) == set(
        frame["kernel.pipeline"].unique()
    )
    assert "$kernel.pipeline" in descriptor["features_signature"]

    assert (
        _manifest(output_dir)["categorical_encoding"]
        == descriptor["categorical_encoding"]
    )


def test_encoding_is_folded_into_the_features_hash(tmp_path, evaluator):
    """RFC 0019 §6.5: the map changes what the model reads but not the signature."""
    output_dir = tmp_path / "model"

    assert (
        _train(output_dir, _string_corpus(tmp_path / "bench.csv"), STRING_FEATURES) == 0
    )

    descriptor = _descriptor(output_dir)
    signature = descriptor["features_signature"]
    assert descriptor["features_hash"] == compute_features_hash(
        signature, descriptor["categorical_encoding"], evaluator
    )
    assert descriptor["features_hash"] != compute_features_hash(
        signature, executable=evaluator
    )


def test_evaluation_scores_through_the_shipped_encoding(tmp_path, evaluator):
    """Changing shipped categorical codes invalidates the features_hash."""
    output_dir = tmp_path / "model"
    csv = _string_corpus(tmp_path / "bench.csv")

    assert _train(output_dir, csv, STRING_FEATURES) == 0

    descriptor_path = output_dir / "heuristic.uhd.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["categorical_encoding"]["$kernel.pipeline"] = {
        "intrawave": 1,
        "pingpong": 0,
    }
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="features_hash"):
        load_model(output_dir)


def test_a_value_outside_the_shipped_map_is_refused(tmp_path, evaluator):
    """Matches the runtime, which throws rather than scoring an unmapped value."""
    output_dir = tmp_path / "model"
    csv = _string_corpus(tmp_path / "bench.csv")

    assert _train(output_dir, csv, STRING_FEATURES) == 0

    frame = pd.read_csv(csv)
    frame.loc[0, "kernel.pipeline"] = "v3"

    with pytest.raises(ValueError) as excinfo:
        load_model(output_dir).scorer(frame)

    assert "v3" in str(excinfo.value)
    assert "$kernel.pipeline" in str(excinfo.value)


def test_numeric_looking_json_categories_are_not_coerced_into_numbers(
    tmp_path, evaluator
):
    frame = pd.DataFrame(
        {
            "kernel.block_size": _varying(64, 256),
            "kernel.pipeline": _varying("00", "0", period=2),
            "tflops": _varying(20.0, 80.0, period=2),
        }
    )
    corpus = tmp_path / "bench.json"
    frame.to_json(corpus, orient="records")
    output_dir = tmp_path / "model"
    assert _train(output_dir, corpus, STRING_FEATURES) == 0
    assert _descriptor(output_dir)["categorical_encoding"] == {
        "$kernel.pipeline": {"0": 0, "00": 1}
    }
    scores = load_model(output_dir).scorer(frame)
    assert scores[2] > scores[0]
