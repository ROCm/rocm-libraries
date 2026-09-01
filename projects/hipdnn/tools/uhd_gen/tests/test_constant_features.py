#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Feature columns that never vary, and the two opposite reasons they are constant.

A column with one value across the corpus carries no ranking signal, bloats
features_signature, changes features_hash, and buys a feature extraction per candidate
score at runtime (RFC 0019 §7.2). Dropping it is the default -- rocKE's attention
kernels bake their geometry in, so 8 of 14 kernel fields are pinned by the matcher
before ranking begins and are constant *by construction*.

The opposite case looks identical in a CSV: a column that does vary in the world, only
ever sampled at one value, where dropping produces a model that cannot generalise and
the real fix is a wider corpus. So every test here pins one of the three things the
tool owes the caller when it cannot tell those apart -- say what it dropped, record it
in the manifest, and let the caller override -- plus the one case where neither answer
is acceptable and the run has to fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# Optional heavyweight training dependencies. Imported before uhd_gen.__main__, which
# pulls them in transitively, so a missing dep is a skip rather than a collection error.
pytest.importorskip("lightgbm")
pytest.importorskip("pandas")
pytest.importorskip("flatbuffers")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from uhd_gen.__main__ import CONSTANT_FEATURE_WARN_FRACTION, main  # noqa: E402
from uhd_gen.features import build_features_signature, compute_features_hash  # noqa: E402

#: Enough rows for the 5-fold CV in train_model, with both values of the varying knob
#: present in every fold.
ROWS = 80


def _varying(period: int, low: int, high: int) -> list[int]:
    return [low if (row // period) % 2 == 0 else high for row in range(ROWS)]


def _constant(value) -> list:
    return [value] * ROWS


def _target(block_sizes: list[int]) -> list[float]:
    """A target that genuinely depends on the varying knob, so the model has something
    to learn and these tests are not measuring noise."""
    return [round(120.0 - 0.2 * size + 0.01 * row, 4) for row, size in enumerate(block_sizes)]


def _corpus(path: Path, columns: dict[str, list]) -> Path:
    names = list(columns)
    lines = [",".join(names)]
    for row in range(ROWS):
        lines.append(",".join(str(columns[name][row]) for name in names))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _one_varying_two_constant(path: Path) -> Path:
    """The shape the acceptance criteria name: one real knob, two pinned ones."""
    block = _varying(1, 64, 256)
    return _corpus(
        path,
        {
            "kernel.block_size": block,
            "kernel.tile_m": _constant(128),
            "device.cu_count": _constant(304),
            "tflops": _target(block),
        },
    )


THREE_FEATURES = ["kernel.block_size", "kernel.tile_m", "device.cu_count"]


def _train(output_dir: Path, csv: Path, features: list[str], *extra: str) -> int:
    return main(
        [
            "train",
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
# Default: drop, loudly
# --------------------------------------------------------------------------------


def test_constant_column_is_absent_from_the_emitted_signature(tmp_path):
    """The descriptor is the artifact that reaches the runtime. A constant column left
    in it costs an extraction per candidate score forever, for a value the extractor
    already knows."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _one_varying_two_constant(tmp_path / "bench.csv"), THREE_FEATURES) == 0

    assert _descriptor(output_dir)["features_signature"] == ["$kernel.block_size"]


def test_manifest_records_what_was_dropped_and_what_was_asked_for(tmp_path):
    """Provenance: a tool that emits a different feature set than requested and does not
    say so is the same defect class as a silently-ignored model."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _one_varying_two_constant(tmp_path / "bench.csv"), THREE_FEATURES) == 0

    manifest = _manifest(output_dir)
    assert manifest["requested_features"] == THREE_FEATURES
    assert manifest["features"] == ["kernel.block_size"]
    assert manifest["dropped_constant_features"] == ["kernel.tile_m", "device.cu_count"]
    assert manifest["keep_constant_features"] is False
    # Values, not just names: "which column" without "stuck at what" does not tell the
    # reader whether the corpus or the matcher pinned it. Plain ints, because a boxed
    # numpy scalar would have raised TypeError at the manifest write.
    assert manifest["constant_features"] == [
        {"column": "kernel.tile_m", "value": 128},
        {"column": "device.cu_count", "value": 304},
    ]


def test_drop_is_reported_by_name_with_its_single_value(tmp_path, caplog):
    output_dir = tmp_path / "model"

    with caplog.at_level("WARNING"):
        assert (
            _train(output_dir, _one_varying_two_constant(tmp_path / "bench.csv"), THREE_FEATURES)
            == 0
        )

    assert "DROPPED constant feature column kernel.tile_m: every row is 128" in caplog.text
    assert "DROPPED constant feature column device.cu_count: every row is 304" in caplog.text
    # The reader has to be told the other reading exists, or a thin corpus gets silently
    # blessed as a small signature.
    assert "--keep-constant-features" in caplog.text


def test_features_hash_is_over_the_signature_actually_emitted(tmp_path):
    """The runtime recomputes the hash from features_signature and refuses the pair when
    the two disagree (RFC 0019 §7.3). A hash over the *requested* columns would make
    every dropped-column model fail to load."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _one_varying_two_constant(tmp_path / "bench.csv"), THREE_FEATURES) == 0

    descriptor = _descriptor(output_dir)
    assert descriptor["features_hash"] == compute_features_hash(descriptor["features_signature"])
    assert descriptor["features_hash"] != compute_features_hash(
        build_features_signature(THREE_FEATURES)
    )
    assert _manifest(output_dir)["features_hash"] == descriptor["features_hash"]


# --------------------------------------------------------------------------------
# The override: the corpus is thin, not the world
# --------------------------------------------------------------------------------


def test_keep_constant_features_keeps_every_requested_column(tmp_path):
    """When the caller knows the column varies in the world, the signature has to match
    the richer corpus they will retrain on -- a dropped column would change the hash and
    strand the descriptor against that future model."""
    output_dir = tmp_path / "model"

    assert (
        _train(
            output_dir,
            _one_varying_two_constant(tmp_path / "bench.csv"),
            THREE_FEATURES,
            "--keep-constant-features",
        )
        == 0
    )

    descriptor = _descriptor(output_dir)
    assert descriptor["features_signature"] == build_features_signature(THREE_FEATURES)
    assert descriptor["features_hash"] == compute_features_hash(descriptor["features_signature"])

    manifest = _manifest(output_dir)
    assert manifest["features"] == THREE_FEATURES
    assert manifest["dropped_constant_features"] == []
    assert manifest["keep_constant_features"] is True


def test_keeping_constants_is_still_reported(tmp_path, caplog):
    """Silence would hide that the signature contains columns this model cannot use."""
    output_dir = tmp_path / "model"

    with caplog.at_level("WARNING"):
        assert (
            _train(
                output_dir,
                _one_varying_two_constant(tmp_path / "bench.csv"),
                THREE_FEATURES,
                "--keep-constant-features",
            )
            == 0
        )

    assert "KEPT 2 constant feature column(s)" in caplog.text
    assert "kernel.tile_m=128" in caplog.text


# --------------------------------------------------------------------------------
# Nothing left to train on
# --------------------------------------------------------------------------------


def _all_constant(path: Path) -> Path:
    block = _varying(1, 64, 256)
    return _corpus(
        path,
        {
            "kernel.tile_m": _constant(128),
            "device.cu_count": _constant(304),
            "tflops": _target(block),
        },
    )


def test_all_constant_features_fail_and_name_every_column(tmp_path, caplog):
    """A model over zero varying features scores every candidate identically. Shipping
    one is worse than shipping none: the engine ranks by a model that cannot
    discriminate instead of falling back to its declared order."""
    output_dir = tmp_path / "model"

    with caplog.at_level("ERROR"):
        assert (
            _train(output_dir, _all_constant(tmp_path / "bench.csv"), ["kernel.tile_m", "device.cu_count"])
            == 1
        )

    assert "kernel.tile_m=128" in caplog.text
    assert "device.cu_count=304" in caplog.text
    assert not (output_dir / "heuristic.uhd.json").exists()
    assert not (output_dir / "model.bin").exists()


def test_all_constant_fails_even_with_the_override(tmp_path):
    """The flag chooses a signature; it cannot make a column vary. Honouring it here
    would emit exactly the constant model the failure exists to prevent."""
    output_dir = tmp_path / "model"

    assert (
        _train(
            output_dir,
            _all_constant(tmp_path / "bench.csv"),
            ["kernel.tile_m", "device.cu_count"],
            "--keep-constant-features",
        )
        == 1
    )
    assert not (output_dir / "heuristic.uhd.json").exists()


# --------------------------------------------------------------------------------
# High proportion: a smell, not a diagnosis
# --------------------------------------------------------------------------------


def _rocke_shaped(path: Path) -> tuple[Path, list[str]]:
    """The real case: 8 of 14 kernel fields pinned by the matcher, 6 free knobs.

    57% constant, which is the ORDINARY reading for kernels that bake their geometry
    in. The threshold exists to catch thin corpora, so it must stay quiet here or it
    becomes a warning people learn to ignore.
    """
    block = _varying(1, 64, 256)
    columns: dict[str, list] = {"kernel.block_size": block}
    for index in range(5):
        columns[f"kernel.knob_{index}"] = _varying(index + 1, 1, 2)
    for index in range(8):
        columns[f"kernel.pinned_{index}"] = _constant(index + 1)
    features = list(columns)
    columns["tflops"] = _target(block)
    return _corpus(path, columns), features


def test_high_proportion_warning_stays_quiet_on_the_rocke_shape(tmp_path, caplog):
    csv, features = _rocke_shaped(tmp_path / "bench.csv")
    output_dir = tmp_path / "model"

    with caplog.at_level("WARNING"):
        assert _train(output_dir, csv, features) == 0

    # The eight pinned fields are still reported individually...
    assert "DROPPED constant feature column kernel.pinned_0" in caplog.text
    # ...but 8/14 is 57%, below the threshold, so the corpus is not impugned.
    assert "thin-corpus threshold" not in caplog.text
    assert len(_descriptor(output_dir)["features_signature"]) == 6


def test_high_proportion_warning_fires_at_the_threshold(tmp_path, caplog):
    """2 of 3 is exactly CONSTANT_FEATURE_WARN_FRACTION, and the boundary is inclusive."""
    assert 2 / 3 == CONSTANT_FEATURE_WARN_FRACTION
    output_dir = tmp_path / "model"

    with caplog.at_level("WARNING"):
        assert (
            _train(output_dir, _one_varying_two_constant(tmp_path / "bench.csv"), THREE_FEATURES)
            == 0
        )

    assert "2 of 3 requested feature columns are constant (67%" in caplog.text
    assert "thin-corpus threshold" in caplog.text
    # The point of the message: at this proportion, dropping may be the wrong response,
    # and the reader is sent to the corpus rather than told the features are useless.
    assert "WRONG fix" in caplog.text
    assert str(tmp_path / "bench.csv") in caplog.text


def test_high_proportion_warning_is_silent_just_below_the_threshold(tmp_path, caplog):
    """1 of 2 constant is 50%: a pinned knob beside a real one, the expected case."""
    block = _varying(1, 64, 256)
    csv = _corpus(
        tmp_path / "bench.csv",
        {
            "kernel.block_size": block,
            "kernel.tile_m": _constant(128),
            "tflops": _target(block),
        },
    )
    output_dir = tmp_path / "model"

    with caplog.at_level("WARNING"):
        assert _train(output_dir, csv, ["kernel.block_size", "kernel.tile_m"]) == 0

    assert "DROPPED constant feature column kernel.tile_m" in caplog.text
    assert "thin-corpus threshold" not in caplog.text


# --------------------------------------------------------------------------------
# The corpus with nothing constant must be untouched
# --------------------------------------------------------------------------------


def test_a_corpus_with_no_constant_column_is_unchanged(tmp_path, caplog):
    block = _varying(1, 64, 256)
    csv = _corpus(
        tmp_path / "bench.csv",
        {
            "kernel.block_size": block,
            "kernel.split_k": _varying(2, 1, 4),
            "tflops": _target(block),
        },
    )
    output_dir = tmp_path / "model"
    features = ["kernel.block_size", "kernel.split_k"]

    with caplog.at_level("WARNING"):
        assert _train(output_dir, csv, features) == 0

    assert _descriptor(output_dir)["features_signature"] == build_features_signature(features)
    assert _manifest(output_dir)["dropped_constant_features"] == []
    assert "DROPPED" not in caplog.text
    assert "thin-corpus threshold" not in caplog.text
