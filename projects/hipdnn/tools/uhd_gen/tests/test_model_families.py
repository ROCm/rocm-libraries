# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`--model-family`: non-LightGBM ensembles in the tree_data artifact the runtime reads.

Every exported tree must route each row to the leaf its source library picks, with the
same leaf value, using only TreeDataAdapter's comparison (`<` or `<=` on a float64).
"""
import json
import warnings

import pytest

flatbuffers = pytest.importorskip("flatbuffers")
pytest.importorskip("lightgbm")  # uhd_gen.__main__ imports the LightGBM converter
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModel  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from uhd_gen.__main__ import main  # noqa: E402
from uhd_gen.artifact import verify_tree_artifact  # noqa: E402
from uhd_gen.lgbm_to_flatbuffer import GBDT_MODEL_FILE_IDENTIFIER  # noqa: E402
from uhd_gen.model_families import (  # noqa: E402
    build_family_model,
    early_stopped_rounds,
    export_family,
    float32_less_than,
    model_summary,
    tree_equivalence,
)

#: float32 split values covering both parities of the low significand bit, zero, the
#: subnormal range, negatives and integers past float32's exact range.
THRESHOLDS = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.1,
    -0.1,
    3.0,
    127.5,
    1e-40,
    -1e-40,
    float(np.finfo(np.float32).tiny),
    16777217.0,
    5324800000000.0,
    -5324800000000.0,
    float(np.finfo(np.float32).max),
]


def _probes(threshold: float) -> np.ndarray:
    """float64 values on and around the float32 rounding boundary below `threshold`."""
    upper = np.float32(threshold)
    below = np.nextafter(upper, np.float32(-np.inf))
    midpoint = (float(below) + float(upper)) / 2.0
    values = [float(upper), float(below), midpoint, threshold]
    for anchor in (midpoint, float(upper), float(below)):
        values += [np.nextafter(anchor, -np.inf), np.nextafter(anchor, np.inf)]
    return np.asarray(values, dtype=np.float64)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_a_float32_split_is_routed_exactly_by_a_float64_comparison(threshold):
    """`x < t` or `x <= t` in float64 agrees with XGBoost's `float32(x) < threshold`."""
    split, lte = float32_less_than(threshold)
    probes = _probes(threshold)
    with np.errstate(over="ignore"):
        expected = probes.astype(np.float32) < np.float32(threshold)
    actual = probes <= split if lte else probes < split
    assert actual.tolist() == expected.tolist()


def test_a_split_without_a_finite_float32_below_it_is_refused():
    with pytest.raises(ValueError, match="no finite float32"):
        float32_less_than(-float(np.finfo(np.float32).max))
    with pytest.raises(ValueError, match="not finite"):
        float32_less_than(float("inf"))


def test_early_stopping_keeps_the_best_round_as_lightgbm_does():
    assert early_stopped_rounds(np.array([5.0, 4.0, 3.0, 3.5, 3.6, 3.7]), 2) == 3
    # A later improvement inside the patience window still counts.
    assert early_stopped_rounds(np.array([5.0, 4.0, 4.5, 3.0, 3.1]), 2) == 4
    # A tie is not an improvement.
    assert early_stopped_rounds(np.array([2.0, 2.0, 2.0]), 5) == 1


def _training_data(rows: int = 2000, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.integers(1, 9, rows) * 128.0,
            rng.choice([32.0, 64.0, 128.0, 256.0], rows),
            rng.normal(size=rows),
            rng.random(rows) * 5324800000000.0,
        ]
    )
    y = 0.001 * X[:, 0] + np.log(X[:, 1]) + np.sin(X[:, 2]) + X[:, 3] / 5e12
    return X, y + rng.normal(0, 0.05, rows)


def _with_missing(X: np.ndarray, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(rng.random(X.shape) < 0.1, np.nan, X)


def _probe_rows(X: np.ndarray, artifact: bytes) -> np.ndarray:
    """Training rows, NaN rows, and a row moved onto and next to each split value.

    Finite values only: a HistGradientBoosting NaN-only split (threshold +inf) is stored
    as DBL_MAX, which routes +inf the other way, and no runtime feature is infinite.
    """
    model = GbdtModel.GetRootAs(artifact, 0)
    rows = [X, _with_missing(X, seed=9)]
    for tree_index in range(model.TreesLength()):
        tree = model.Trees(tree_index)
        for node in range(tree.LeftChildrenLength()):
            if tree.LeftChildren(node) < 0:
                continue
            threshold = tree.Thresholds(node)
            # Above DBL_MAX is +inf, which the isfinite check drops.
            with np.errstate(over="ignore"):
                values = (
                    threshold,
                    np.nextafter(threshold, -np.inf),
                    np.nextafter(threshold, np.inf),
                )
            for value in values:
                if np.isfinite(value):
                    row = X[:1].copy()
                    row[0, tree.FeatureIndices(node)] = value
                    rows.append(row)
    return np.vstack(rows)


def _check_export(model, tmp_path, framework: bytes):
    path = tmp_path / "model.bin"
    digest = export_family(model, "sha256:test", path, num_training_samples=7)
    artifact = verify_tree_artifact(path, digest, feature_count=4)
    stored = GbdtModel.GetRootAs(artifact, 0)
    assert GbdtModel.GbdtModelBufferHasIdentifier(artifact, 0)
    assert artifact[4:8] == GBDT_MODEL_FILE_IDENTIFIER
    assert stored.Framework() == framework
    assert stored.LearningRate() == 1.0
    assert stored.NumFeatures() == 4
    assert stored.NumTrainingSamples() == 7
    return artifact


def test_histgradientboosting_trees_route_and_score_as_scikit_learn(tmp_path):
    X, y = _training_data()
    model = HistGradientBoostingRegressor(
        max_iter=40, max_leaf_nodes=31, early_stopping=False, random_state=0
    ).fit(_with_missing(X), y)
    artifact = _check_export(model, tmp_path, b"sklearn")
    probes = _probe_rows(X, artifact)
    result = tree_equivalence(model, artifact, probes)
    assert result["trees"] == model.n_iter_
    assert result["trees_bitwise_equal"], result["mismatched_trees"]
    # Only the order of a float64 sum differs from predict.
    assert result["max_abs_diff"] < 1e-12


def test_xgboost_trees_route_and_score_as_xgboost(tmp_path):
    xgb = pytest.importorskip("xgboost")
    X, y = _training_data()
    model = xgb.train(
        {"objective": "reg:squarederror", "max_leaves": 31, "grow_policy": "lossguide"},
        xgb.DMatrix(_with_missing(X), label=y),
        num_boost_round=40,
    )
    artifact = _check_export(model, tmp_path, b"xgboost")
    probes = _probe_rows(X, artifact)
    # Also every float32 split value XGBoost stores, which the export moved.
    document = json.loads(model.save_raw(raw_format="json"))
    for tree in document["learner"]["gradient_booster"]["model"]["trees"]:
        for node, left in enumerate(tree["left_children"]):
            if left >= 0:
                row = X[:1].copy()
                row[0, tree["split_indices"][node]] = tree["split_conditions"][node]
                probes = np.vstack([probes, row])
    result = tree_equivalence(model, artifact, probes)
    assert result["trees"] == model.num_boosted_rounds()
    assert result["leaves_equal"]
    assert result["trees_bitwise_equal"], result["mismatched_trees"]
    # XGBoost accumulates the margin in float32; the runtime sums in float64.
    assert result["max_abs_diff"] < 1e-4


def test_unexportable_models_are_refused():
    X, y = _training_data(rows=300)
    categorical = HistGradientBoostingRegressor(
        max_iter=3, categorical_features=[1], random_state=0
    ).fit(X, y)
    with pytest.raises(ValueError, match="categorical"):
        build_family_model(categorical, "sha256:test")
    xgb = pytest.importorskip("xgboost")
    labels = (y > np.median(y)).astype(float)
    logistic = xgb.train(
        {"objective": "binary:logistic"}, xgb.DMatrix(X, label=labels), 2
    )
    with pytest.raises(ValueError, match="identity-link"):
        build_family_model(logistic, "sha256:test")
    # XGBoost 2 names this booster `dart`; XGBoost 3 stores it as gbtree with tree weights.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # XGBoost 3 deprecates `dart`
        dart = xgb.train(
            {"objective": "reg:squarederror", "booster": "dart", "rate_drop": 0.5},
            xgb.DMatrix(X, label=y),
            4,
        )
    with pytest.raises(ValueError, match="gbtree|dropout"):
        build_family_model(dart, "sha256:test")


def test_the_manifest_summary_has_the_lightgbm_keys_for_every_family():
    X, y = _training_data(rows=300)
    model = HistGradientBoostingRegressor(max_iter=5, random_state=0).fit(X, y)
    summary = model_summary(model, ["a", "b", "c", "d"])
    assert summary["num_trees"] == 5
    assert set(summary["feature_importance"]) == {"a", "b", "c", "d"}
    assert sum(item["split"] for item in summary["feature_importance"].values()) > 0


def _catalog_corpus(path):
    rows = [
        {
            "benchmark": f"g{problem:02d}",
            "device": "board",
            "kernel": f"k{tile}",
            "q.seqlen": 128 * (problem + 1),
            "kernel.tile_m": tile,
            "tflops": 10.0 + problem + (tile / 64 if problem % 2 else -tile / 64),
        }
        for problem in range(24)
        for tile in (64, 128, 256)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


PROVENANCE = {
    "ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "1.0"},
    "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.0"},
    "umd": [],
}


def _train_arguments(corpus, output, evaluator, family):
    snapshot = corpus.parent / "provenance.json"
    snapshot.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    return [
        "train",
        "--input",
        str(corpus),
        "--provenance",
        str(snapshot),
        "--features",
        "q.seqlen",
        "kernel.tile_m",
        "--target",
        "tflops",
        "--num-boost-round",
        "20",
        "--early-stopping",
        "5",
        "--model-family",
        family,
        "--output-dir",
        str(output),
        "--feature-evaluator",
        evaluator,
    ]


@pytest.mark.parametrize("family", ["sklearn-hgb", "xgboost"])
def test_a_trained_family_is_a_tree_data_model_evaluate_scores(
    tmp_path, evaluator, family
):
    if family == "xgboost":
        pytest.importorskip("xgboost")
    corpus = _catalog_corpus(tmp_path / "corpus.csv")
    model_dir = tmp_path / "model"
    assert main(_train_arguments(corpus, model_dir, evaluator, family)) == 0
    descriptor = json.loads((model_dir / "heuristic.uhd.json").read_text("utf-8"))
    assert descriptor["adapter"] == "tree_data"
    assert not (model_dir / "model.lgbm").exists()
    verify_tree_artifact(
        model_dir / "model.bin", descriptor["tree_data"]["hash"], feature_count=2
    )
    manifest = json.loads((model_dir / "train_manifest.json").read_text("utf-8"))
    assert manifest["model_family"] == family
    assert manifest["training_options"]["model_family"] == family
    assert 1 <= manifest["num_trees"] <= 20
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "evaluate",
                "--input",
                str(corpus),
                "--model-dir",
                str(model_dir),
                "--output",
                str(report),
                "--feature-evaluator",
                evaluator,
            ]
        )
        == 0
    )
    report = json.loads(report.read_text("utf-8"))
    assert report["corpus"]["problems"] == 24
    # The family's model ranked every held-out problem; none fell back to static order.
    assert report["metrics"]["problems_scored"] == report["split"]["eval_problems"] > 0
    assert report["fallback"]["problems"] == 0


def _engine_corpus(path):
    """A `predict_engine` corpus: one immediate-mode measurement per graph."""
    binding = {
        "engine": "probe:A",
        "role": "predict_engine",
        "metric": "tflops",
        "arch": "gfx942",
        "selector_revision": "probe-1",
        "trained_against": {"selector_revision": "probe-1"},
    }
    rows = [
        {
            "engine_id": 1,
            "engine_name": "probe:A",
            "graph_id": f"g{index}",
            "graph_key": f"g{index}",
            "device_id": "board",
            "arch": "gfx942",
            "binding": binding,
            "metric": "tflops",
            "features": {
                "graph.flops": 1e12 + index * 1e10,
                "graph.nodes[0].x.dims[0]": index % 7 + 1,
            },
            "avgTimeMs": 1 + index * 0.1,
            "robustMeanMs": 1 + index * 0.1,
            "is_valid": True,
            "selection_mode": "immediate",
            "timing_statistic": "robustMeanMs",
        }
        for index in range(60)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("family", ["sklearn-hgb", "xgboost"])
def test_an_engine_estimate_trains_with_every_family(tmp_path, evaluator, family):
    """`predict_engine` (layer 1) takes the same family switch as the catalog ranker."""
    if family == "xgboost":
        pytest.importorskip("xgboost")
    model_dir = tmp_path / "model"
    arguments = [
        "train",
        "--role",
        "predict_engine",
        "--engine",
        "probe:A",
        "--metric",
        "tflops",
        "--input",
        str(_engine_corpus(tmp_path / "corpus.json")),
        "--output-dir",
        str(model_dir),
        "--features",
        "graph.flops",
        "graph.nodes[0].x.dims[0]",
        "--num-boost-round",
        "20",
        "--early-stopping",
        "5",
        "--model-family",
        family,
        "--feature-evaluator",
        evaluator,
    ]
    assert main(arguments) == 0
    descriptor = json.loads((model_dir / "heuristic.uhd.json").read_text("utf-8"))
    assert descriptor["score"]["metric"] == "tflops"
    assert descriptor["adapter"] == "tree_data"
    verify_tree_artifact(
        model_dir / "model.bin", descriptor["tree_data"]["hash"], feature_count=2
    )
    manifest = json.loads((model_dir / "train_manifest.json").read_text("utf-8"))
    assert manifest["model_family"] == family


def test_grouped_training_is_lightgbm_only(tmp_path, caplog):
    corpus = _catalog_corpus(tmp_path / "corpus.csv")
    arguments = _train_arguments(corpus, tmp_path / "model", "unused", "sklearn-hgb")
    assert main([*arguments, "--group-by-feature", "kernel.tile_m"]) == 1
    assert "LightGBM only" in caplog.text
    assert not (tmp_path / "model").exists()
