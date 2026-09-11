# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Constant model inputs are legal; pruning is never public-knob removal."""
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


def _train(tmp_path, drop=False, constant=False):
    block = [64 if row % 2 else 256 for row in range(80)]
    frame = pd.DataFrame({"kernel.block_size": [64] * 80 if constant else block,
                          "kernel.tile_m": [128] * 80, "device.cu_count": [304] * 80,
                          "tflops": [120 - 0.2 * value for value in block]})
    corpus = tmp_path / "corpus.csv"
    frame.to_csv(corpus, index=False)
    snapshot = tmp_path / "provenance.json"
    snapshot.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    output = tmp_path / "model"
    args = ["train", "--input", str(corpus), "--provenance", str(snapshot), "--features",
            "kernel.block_size", "kernel.tile_m", "device.cu_count", "--target", "tflops",
            "--output-dir", str(output), "--num-boost-round", "10", "--early-stopping", "5"]
    if drop:
        args.append("--drop-constant-features")
    return main(args), output


@pytest.mark.parametrize("drop", [False, True])
def test_pruning_changes_only_the_trained_signature(tmp_path, drop):
    code, output = _train(tmp_path, drop)
    assert code == 0
    descriptor = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    expected = ["$kernel.block_size"] if drop else ["$kernel.block_size", "$kernel.tile_m", "$device.cu_count"]
    assert descriptor["features_signature"] == expected
    assert descriptor["features_hash"] == compute_features_hash(expected)
    assert descriptor["trained_against"] == manifest["trained_against"] == PROVENANCE
    assert manifest["device_coverage"]["fields"]["device.cu_count"]["varies"] is False
    assert manifest["dropped_constant_features"] == (["kernel.tile_m", "device.cu_count"] if drop else [])


@pytest.mark.parametrize("drop", [False, True])
def test_no_discriminating_feature_never_publishes_model(tmp_path, drop):
    code, output = _train(tmp_path, drop, constant=True)
    assert code == 1
    assert not output.exists()
