#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The categorical encoding is derived from the corpus and ships with the model.

RFC 0019 §4/§6.5: the string-to-code map belongs to the descriptor, beside the
signature it encodes, not to a global table both languages have to be kept in step
with by hand. Which makes three things load-bearing, and this file pins each:

  - the map is keyed by the full `$reference`, because two columns can share a
    trailing field name and not a vocabulary;
  - the map the descriptor ships is the map the model was FITTED with, or the
    thresholds mean something the runtime will never reproduce;
  - a corpus with no string column produces exactly the descriptor and exactly the
    features_hash it produced before any of this existed, because an empty map is not
    a contract change.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Optional heavyweight training dependencies. Imported before uhd_gen.__main__, which
# pulls them in transitively, so a missing dep is a skip rather than a collection error.
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
    snapshot.write_text(json.dumps({
        "ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "1.0"},
        "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.0"},
        "umd": [],
    }), encoding="utf-8")
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
# Numeric-only: the shape every shipped model has today
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
    """A numeric column already IS a number. An entry for one would tell the runtime to
    encode something it must read straight through."""
    frame = pd.read_csv(_numeric_corpus(tmp_path / "bench.csv"))

    assert derive_categorical_encoding(frame, NUMERIC_FEATURES) == {}


def test_numeric_only_descriptor_gains_no_key_and_does_not_move_its_hash(tmp_path):
    """The acceptance case for every model already in the field: no string column, so
    the descriptor and the hash are byte for byte what they were before the map
    existed. compute_features_hash appends only a truthy encoding, and this is what
    makes that matter."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _numeric_corpus(tmp_path / "bench.csv"), NUMERIC_FEATURES) == 0

    descriptor = _descriptor(output_dir)
    assert "categorical_encoding" not in descriptor

    signature = build_features_signature(NUMERIC_FEATURES)
    assert descriptor["features_signature"] == signature
    # The hash as it was computed before the argument existed, character for character.
    assert descriptor["features_hash"] == compute_features_hash(signature)
    assert descriptor["features_hash"] == compute_features_hash(signature, {})


def test_manifest_records_the_empty_map(tmp_path):
    """Unlike the descriptor: the manifest is provenance, and "this corpus had no
    categorical column" is a fact about the run, not an absence to be inferred."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _numeric_corpus(tmp_path / "bench.csv"), NUMERIC_FEATURES) == 0

    assert _manifest(output_dir)["categorical_encoding"] == {}


# --------------------------------------------------------------------------------
# Keyed by reference, not by field name
# --------------------------------------------------------------------------------


def test_two_columns_sharing_a_field_name_keep_separate_vocabularies():
    """`kernel.dtype` and `q.attention_dense.dtype` share a trailing name and nothing
    else. Keying by the field name would merge two vocabularies into one map, and every
    row of one column would then be encoded through the other's numbering.

    The two spellings differ only in case on purpose: the map ships with the model, so
    the runtime looks up the bytes the corpus held. Folding here would emit a key no
    lookup ever hits."""
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

    assert list(derive_categorical_encoding(frame, ["kernel.dtype"])) == ["$kernel.dtype"]


def test_codes_are_deterministic_across_derivations():
    """Sorted order, from 0. A model.bin bakes these numbers into its split thresholds,
    so a map that renumbered between two runs over the same corpus would silently
    re-point every threshold."""
    frame = pd.DataFrame({"kernel.pipeline": ["pingpong", "intrawave", "v3", "intrawave"]})

    first = derive_categorical_encoding(frame, ["kernel.pipeline"])
    second = derive_categorical_encoding(frame, ["kernel.pipeline"])

    assert first == second
    assert first == {"$kernel.pipeline": {"intrawave": 0, "pingpong": 1, "v3": 2}}


def test_mixed_column_is_rejected():
    """Codes are positions, so a raw number in a categorical column collides with
    whichever value took that code -- and the collision is invisible: the model trains,
    saves, and ranks two different things as one."""
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
    """A corpus whose target depends strongly on a string column, so the model has to
    split on it."""
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
    """The premise of the two tests below, asserted rather than assumed.

    There is no process-wide table any more, so a string reaching the numeric path with
    nothing to consult cannot be encoded by anything: a run that reaches training at all
    reached it through the derived map. This replaces an assertion that the frozen global
    table happened to lack 'pipeline' -- the table is gone, and the guarantee is now
    structural rather than a gap in one particular table.
    """
    with pytest.raises(ValueError, match="no categorical_encoding declares"):
        encode_feature_value("$kernel.pipeline", "intrawave")


def test_string_column_trains_and_ships_its_own_vocabulary(tmp_path):
    """Training succeeds where the global table would have raised, and the descriptor
    carries exactly the distinct values the column held -- no more (a vocabulary wider
    than the corpus claims codes nothing was fitted on) and no fewer (the runtime now
    throws on a declared reference whose value is missing from its map)."""
    output_dir = tmp_path / "model"
    csv = _string_corpus(tmp_path / "bench.csv")

    assert _train(output_dir, csv, STRING_FEATURES) == 0

    descriptor = _descriptor(output_dir)
    assert descriptor["categorical_encoding"] == {
        "$kernel.pipeline": {"intrawave": 0, "pingpong": 1}
    }
    # Exactly the distinct values present, keyed by the reference in the signature.
    frame = pd.read_csv(csv)
    assert set(descriptor["categorical_encoding"]["$kernel.pipeline"]) == set(
        frame["kernel.pipeline"].unique()
    )
    assert "$kernel.pipeline" in descriptor["features_signature"]

    # And it is the map the fit used: the manifest records the same object.
    assert _manifest(output_dir)["categorical_encoding"] == descriptor["categorical_encoding"]


def test_encoding_is_folded_into_the_features_hash(tmp_path):
    """RFC 0019 §6.5: a changed map changes what the model reads while leaving the
    signature text identical, so the hash has to cover it."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _string_corpus(tmp_path / "bench.csv"), STRING_FEATURES) == 0

    descriptor = _descriptor(output_dir)
    signature = descriptor["features_signature"]
    assert descriptor["features_hash"] == compute_features_hash(
        signature, descriptor["categorical_encoding"]
    )
    assert descriptor["features_hash"] != compute_features_hash(signature)


def test_evaluation_scores_through_the_shipped_encoding(tmp_path):
    """Changing shipped categorical codes invalidates the artifact's feature contract."""
    output_dir = tmp_path / "model"
    csv = _string_corpus(tmp_path / "bench.csv")

    assert _train(output_dir, csv, STRING_FEATURES) == 0

    descriptor_path = output_dir / "heuristic.uhd.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    descriptor["categorical_encoding"]["$kernel.pipeline"] = {"intrawave": 1, "pingpong": 0}
    descriptor_path.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="features_hash"):
        load_model(output_dir)


def test_a_value_outside_the_shipped_map_is_refused(tmp_path):
    """The runtime throws when a declared reference carries a value its map lacks, so
    the training-side scorer must not quietly hand LightGBM a NaN and return an
    ordinary leaf for it."""
    output_dir = tmp_path / "model"
    csv = _string_corpus(tmp_path / "bench.csv")

    assert _train(output_dir, csv, STRING_FEATURES) == 0

    frame = pd.read_csv(csv)
    frame.loc[0, "kernel.pipeline"] = "v3"

    with pytest.raises(ValueError) as excinfo:
        load_model(output_dir).scorer(frame)

    assert "v3" in str(excinfo.value)
    assert "$kernel.pipeline" in str(excinfo.value)


def test_numeric_looking_json_categories_are_not_coerced_into_numbers(tmp_path):
    frame = pd.DataFrame({
        "kernel.block_size": _varying(64, 256),
        "kernel.pipeline": _varying("00", "0", period=2),
        "tflops": _varying(20.0, 80.0, period=2),
    })
    corpus = tmp_path / "bench.json"
    frame.to_json(corpus, orient="records")
    output_dir = tmp_path / "model"
    assert _train(output_dir, corpus, STRING_FEATURES) == 0
    assert _descriptor(output_dir)["categorical_encoding"] == {
        "$kernel.pipeline": {"0": 0, "00": 1}
    }
    scores = load_model(output_dir).scorer(frame)
    assert scores[2] > scores[0]
