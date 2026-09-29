# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise canonical expression training with the actual shared feature runtime."""
import json

import pytest

pytest.importorskip("lightgbm")
pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("flatbuffers")

from uhd_gen.__main__ import main
from uhd_gen.evaluate import load_model
from uhd_gen.features import compute_features_hash, derive_categorical_encoding, evaluate_feature_rows

PROVENANCE = {"ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "2.3"},
              "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.4"}, "umd": []}


def test_inline_ast_and_categorical_leaves_match_runtime(evaluator):
    frame = pd.DataFrame({"attention.query.dims[2]": [129, 64], "kernel.tile_m": [32, 64],
                          "kernel.dtype": ["fp16", "bf16"]})
    signature = [{"ceil_div": ["$attention.query.dims[2]", "$kernel.tile_m"]},
                 {"==": ["$kernel.dtype", "fp16"]}, "$kernel.dtype"]
    encoding = {"$kernel.dtype": {"bf16": 0, "fp16": 1}}
    digest, values = evaluate_feature_rows(frame, signature, encoding, evaluator)
    assert values == [[5, 1, 1], [1, 0, 0]]
    assert digest == compute_features_hash(signature, encoding, evaluator)


DY = "graph.nodes[0].dy.dims[0]"


def test_an_unpublished_binding_is_null_for_the_evaluator_not_nan_or_an_error(evaluator):
    """A conv-fwd row publishes no `dy`; pandas fills the hole with NaN, which is not JSON,
    and a column no row publishes is not in the frame at all. Both reach the evaluator as
    absent, where `value_or_default` answers exactly as it does at runtime."""
    signature = [{"value_or_default": [f"${DY}", 0]}]
    mixed = pd.DataFrame([{"graph.flops": 1e12}, {"graph.flops": 1e12, DY: 4}])
    assert evaluate_feature_rows(mixed, signature, executable=evaluator)[1] == [[0], [4]]
    unpublished = pd.DataFrame([{"graph.flops": 1e12}])
    assert evaluate_feature_rows(unpublished, signature, executable=evaluator)[1] == [[0]]
    # A bare reference has no default: the runtime refuses the row, and so does this.
    with pytest.raises(ValueError, match="Undefined variable"):
        evaluate_feature_rows(mixed, [f"${DY}"], executable=evaluator)
    # An absent string binding is not a second type in the column, and has no code.
    layouts = pd.DataFrame([{"x.layout": "NHWC"}, {}])
    assert derive_categorical_encoding(layouts, ["x.layout", "dy.layout"]) == {"$x.layout": {"NHWC": 0}}


def test_computed_training_ships_provenance_and_scores_real_artifact(tmp_path, evaluator):
    frame = pd.DataFrame({"attention.query.dims[2]": [64 + 32 * (row % 8) for row in range(80)],
                          "kernel.tile_m": [32 if row % 2 else 64 for row in range(80)],
                          "kernel.dtype": ["bf16" if row % 3 else "fp16" for row in range(80)],
                          "tflops": [float(10 + row % 8) for row in range(80)]})
    corpus, recipe, snapshot = [tmp_path / name for name in ("corpus.json", "features.json", "snapshot.json")]
    frame.to_json(corpus, orient="records")
    signature = [{"ceil_div": ["$attention.query.dims[2]", "$kernel.tile_m"]}, "$kernel.dtype"]
    recipe.write_text(json.dumps(signature), encoding="utf-8")
    snapshot.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    output = tmp_path / "model"
    assert main(["train", "--input", str(corpus), "--feature-signature", str(recipe), "--provenance", str(snapshot),
                 "--feature-evaluator", evaluator, "--output-dir", str(output), "--num-boost-round", "10",
                 "--early-stopping", "5"]) == 0
    descriptor = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    assert descriptor["features_signature"] == signature
    assert descriptor["trained_against"] == PROVENANCE
    bundle = load_model(output, feature_evaluator=evaluator)
    assert bundle.source.endswith("model.bin")
    scores = bundle.scorer(frame)
    assert np.isfinite(scores).all()
    assert scores[7] > scores[0]


def test_unsafe_explicit_device_expression_fails_before_artifacts(tmp_path):
    corpus = tmp_path / "corpus.json"
    pd.DataFrame({"attention.size": list(range(10)), "device": ["a", "b"] * 5,
                  "device.cu_count": [120] * 10, "tflops": list(range(10))}).to_json(corpus, orient="records")
    recipe, snapshot = tmp_path / "features.json", tmp_path / "snapshot.json"
    recipe.write_text(json.dumps([{"/": ["$attention.size", "$device.cu_count"]}]), encoding="utf-8")
    snapshot.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    output = tmp_path / "model"
    assert main(["train", "--input", str(corpus), "--feature-signature", str(recipe), "--provenance", str(snapshot),
                 "--output-dir", str(output)]) == 1
    assert not output.exists()
