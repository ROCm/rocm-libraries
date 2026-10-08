#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for evaluating two-layer (grouped) artifacts and tree routing.

Layer 1 picks a group per problem and layer 2 ranks within it; each failure mode here
yields a plausible number rather than an error.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

flatbuffers = pytest.importorskip("flatbuffers")
lgb = pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from uhd_gen.evaluate import evaluate_corpus, load_model  # noqa: E402
from uhd_gen.lgbm_to_flatbuffer import convert  # noqa: E402

FEATURES = ["q.size", "kernel.group"]
GROUPS = (0.0, 1.0)


def _booster(rows: list[tuple[float, float, float]], num_trees: int = 8) -> lgb.Booster:
    """A booster over (q.size, kernel.group) -> target with enough signal to split."""
    frame = pd.DataFrame(rows, columns=["q.size", "kernel.group", "y"])
    data = lgb.Dataset(
        frame[["q.size", "kernel.group"]].to_numpy(), label=frame["y"].to_numpy()
    )
    return lgb.train(
        {
            "objective": "regression",
            "num_leaves": 4,
            "learning_rate": 0.3,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "verbose": -1,
        },
        data,
        num_boost_round=num_trees,
    )


def _write_model(directory: Path, *, grouped: bool, objective: str = "max") -> Path:
    """A trained pair on disk, as `train --output-dir` leaves it."""
    directory.mkdir(parents=True, exist_ok=True)

    # Layer 1 prefers group 1 on large sizes and group 0 on small ones.
    layer_one = _booster(
        [
            (size, g, (10.0 + size) if g == 1.0 else (20.0 - size))
            for size in range(1, 12)
            for g in GROUPS
        ]
    )
    lgbm_path = directory / "model.lgbm"
    layer_one.save_model(str(lgbm_path))

    group_models = None
    if grouped:
        # Layer 1 does not express this, so any in-group ordering comes from layer 2.
        group_models = [
            (
                float(g),
                _booster(
                    [
                        (size, g, float(size) * (2.0 if g == 1.0 else 1.0))
                        for size in range(1, 12)
                    ]
                ),
            )
            for g in GROUPS
        ]

    artifact = directory / "model.bin"
    convert(
        lgbm_path,
        "sha256:grouped_test",
        artifact,
        num_training_samples=22,
        group_by_feature_index=FEATURES.index("kernel.group") if grouped else -1,
        group_models=group_models,
    )

    manifest = {
        "features": FEATURES,
        "target": "tflops",
        "objective": objective,
        "num_samples": 22,
        "group_by_feature": "kernel.group" if grouped else None,
        "group_models": len(group_models or []),
    }
    (directory / "train_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (directory / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": objective,
                "tree_data": {"artifact": "model.bin"},
                "features_signature": [f"${name}" for name in FEATURES],
            }
        ),
        encoding="utf-8",
    )
    return directory


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [{"q.size": size, "kernel.group": g} for g in GROUPS for size in (2.0, 9.0)]
    )


def test_a_grouped_artifact_rejects_every_candidate_outside_the_chosen_group():
    """If nothing is -inf, only layer 1 (`trees`) was scored."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "grouped", grouped=True))
    scores = bundle.scorer(_candidates())

    assert np.isneginf(
        scores
    ).any(), "no candidate was rejected; layer 2 was not consulted"
    assert np.isfinite(
        scores
    ).any(), "every candidate was rejected; nothing could be picked"

    survived = _candidates().loc[np.isfinite(scores), "kernel.group"].unique()
    assert len(survived) == 1


def test_an_ungrouped_artifact_scores_every_candidate():
    """`group_by_feature_index = -1` must not engage the grouped path."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "flat", grouped=False))
    scores = bundle.scorer(_candidates())

    assert np.all(np.isfinite(scores))
    assert bundle.group_feature is None


def test_a_grouped_model_is_not_ranked_with_the_booster():
    """`model.lgbm` holds only layer 1, so grouped models must load `model.bin`."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    directory = _write_model(with_tmp / "both", grouped=True)
    assert (
        directory / "model.lgbm"
    ).exists(), "fixture must keep the booster to be meaningful"

    bundle = load_model(directory)
    assert bundle.source.endswith("model.bin")


def test_the_group_decision_is_made_per_problem():
    """Like `scoreBatch`, the group is chosen within one query, not across a corpus."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "grouped", grouped=True))

    for size in (2.0, 9.0):
        frame = pd.DataFrame([{"q.size": size, "kernel.group": g} for g in GROUPS])
        scores = bundle.scorer(frame)
        assert np.isfinite(scores).any(), f"every candidate rejected at q.size={size}"


@pytest.mark.parametrize("objective, expected", [("max", 1.0), ("min", 0.0)])
def test_layer_one_picks_the_group_in_the_objective_direction(objective, expected):
    """At q.size 9 layer 1 scores group 1 near 19 and group 0 near 11."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(
        _write_model(with_tmp / objective, grouped=True, objective=objective)
    )
    frame = pd.DataFrame([{"q.size": 9.0, "kernel.group": g} for g in GROUPS])
    scores = bundle.scorer(frame)
    assert frame.loc[np.isfinite(scores), "kernel.group"].tolist() == [expected]


def _corpus() -> pd.DataFrame:
    """Two problems, two groups each, with the best candidate in different groups."""
    rows = []
    for benchmark, best_group in (("aaaa", 1.0), ("bbbb", 0.0)):
        for group in GROUPS:
            for size in (2.0, 9.0):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "device": "d0",
                        "kernel": f"k{group}{size}",
                        "q.size": size,
                        "kernel.group": group,
                        "is_valid": "True",
                        "tflops": (100.0 if group == best_group else 40.0) + size,
                    }
                )
    return pd.DataFrame(rows)


def test_two_stage_regret_sums_to_the_total():
    """Both parts are measured against the same oracle, so they must add up."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "grouped", grouped=True))

    result = evaluate_corpus(
        _corpus(),
        bundle.scorer,
        target="tflops",
        objective="max",
        eval_fraction=1.0,
        group_column="kernel.group",
    )

    two_stage = result.report["metrics"]["two_stage"]
    assert two_stage is not None, "a grouped evaluation reported no decomposition"
    assert two_stage["group_column"] == "kernel.group"

    for problem in result.problems:
        assert problem.group_regret is not None
        assert problem.group_regret + problem.in_group_regret == pytest.approx(
            problem.regret
        )
        assert problem.group_regret >= 0.0


def test_no_decomposition_is_reported_without_a_group_column():
    """Absent means "not a two-layer model"; zeros would claim perfect group picks."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "flat", grouped=False))

    result = evaluate_corpus(
        _corpus(), bundle.scorer, target="tflops", objective="max", eval_fraction=1.0
    )
    assert result.report["metrics"]["two_stage"] is None
    assert all(problem.group_regret is None for problem in result.problems)


def _strict_less_than_model(directory: Path) -> Path:
    """An artifact whose one split uses `<` rather than `<=`.

    Hand-built because LightGBM never emits `<`, so no trained model can exercise it.
    """
    from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT

    tree = GbdtTreeT()
    # Split at 10 with leaves 1.0/9.0: `<`, `<=` and `>` route rows 5/10/15 differently.
    tree.featureIndices = [0, 0, 0]
    tree.thresholds = [10.0, 0.0, 0.0]
    tree.leftChildren = [1, -1, -1]
    tree.rightChildren = [2, -1, -1]
    tree.leafValues = [0.0, 1.0, 9.0]
    tree.defaultLeft = [True, True, True]
    tree.decisionLte = [False, False, False]
    return _single_tree_model(directory, tree)


def _single_tree_model(
    directory: Path, tree, *, score: dict | None = None, manifest: dict | None = None
) -> Path:
    """A hand-written one-tree, one-feature (`q.size`) artifact with its descriptor."""
    import flatbuffers

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT

    model = GbdtModelT()
    model.trees = [tree]
    model.baseScore = 0.0
    model.numFeatures = 1
    model.featuresHash = "sha256:strict_lt"
    model.groupByFeatureIndex = -1

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"HGBM")

    objective = "min" if score else "max"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(bytes(builder.Output()))
    (directory / "train_manifest.json").write_text(
        json.dumps(
            {"features": ["q.size"], "target": "tflops", "objective": objective}
            | (manifest or {})
        ),
        encoding="utf-8",
    )
    descriptor = {
        "objective": objective,
        "tree_data": {"artifact": "model.bin"},
        "features_signature": ["$q.size"],
    }
    if score is not None:
        descriptor["score"] = score
    (directory / "heuristic.uhd.json").write_text(
        json.dumps(descriptor), encoding="utf-8"
    )
    return directory


def test_a_strict_less_than_split_routes_the_way_the_runtime_routes_it():
    """`decision_lte` false means `<`, not the complement `>` of `<=`."""
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_strict_less_than_model(with_tmp / "strict_lt"))

    scores = bundle.scorer(
        pd.DataFrame([{"q.size": 5.0}, {"q.size": 10.0}, {"q.size": 15.0}])
    )

    # No transform is declared, so scores are the raw leaves.
    assert scores[0] < scores[1], "a value below the threshold took the wrong branch"
    assert scores[1] == pytest.approx(
        scores[2]
    ), "10 and 15 must share the right-hand leaf"
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(9.0)


def _routing_tree(nodes: int, default_left, decision_lte):
    """3 nodes: a stump at 0. 5 nodes: same root, right child splitting at 2."""
    from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT

    tree = GbdtTreeT()
    if nodes == 3:
        tree.featureIndices = [0, -1, -1]
        tree.thresholds = [0.0, 0.0, 0.0]
        tree.leftChildren = [1, -1, -1]
        tree.rightChildren = [2, -1, -1]
        tree.leafValues = [0.0, 1.0, 9.0]
    else:
        tree.featureIndices = [0, -1, 0, -1, -1]
        tree.thresholds = [0.0, 0.0, 2.0, 0.0, 0.0]
        tree.leftChildren = [1, -1, 3, -1, -1]
        tree.rightChildren = [2, -1, 4, -1, -1]
        tree.leafValues = [0.0, 1.0, 0.0, 3.0, 9.0]
    tree.defaultLeft = default_left
    tree.decisionLte = decision_lte
    return tree


@pytest.mark.parametrize(
    "nodes, default_left, decision_lte, expected",
    [
        # Both vectors omitted: `<=` everywhere.
        (3, None, None, [1.0, 1.0, 9.0]),
        # Shorter than the tree: past the given entries a nonempty `decision_lte` means
        # `<`, so 2 goes right at node 2 (threshold 2), not left to 3.
        (5, [True], [True], [1.0, 1.0, 9.0]),
    ],
)
def test_optional_routing_vectors_take_the_runtimes_defaults(
    tmp_path, nodes, default_left, decision_lte, expected
):
    """Missing or short routing vectors are filled in as TreeDataAdapter does."""
    bundle = load_model(
        _single_tree_model(
            tmp_path / "model", _routing_tree(nodes, default_left, decision_lte)
        )
    )
    scores = bundle.scorer(pd.DataFrame({"q.size": [-1.0, 0.0, 2.0]}))
    assert scores.tolist() == pytest.approx(expected)


def test_a_descriptor_declaring_no_transform_is_scored_as_identity(tmp_path):
    """The descriptor, not the training manifest's log1p, decides the transform."""
    bundle = load_model(
        _single_tree_model(
            tmp_path / "model",
            _routing_tree(3, [False] * 3, [True] * 3),
            score={"metric": "time", "calibrated": True},
            manifest={"score_transform": "log1p", "target": "avgTimeMs"},
        )
    )
    scores = bundle.scorer(pd.DataFrame({"q.size": [-1.0, 0.0, 2.0]}))
    assert scores.tolist() == pytest.approx([1.0, 1.0, 9.0])


def test_a_generated_encoding_survives_train_then_score(tmp_path, evaluator):
    """A string feature's derived encoding is shipped and read back by the scorer."""
    import subprocess
    import sys

    corpus = tmp_path / "corpus.csv"
    frame = pd.DataFrame(
        [
            {
                "q.size": size,
                "kernel.pipeline": pipeline,
                "tflops": size * (2.0 if pipeline == "interwave" else 1.0),
            }
            # Enough rows to split past LightGBM's default min_data_in_leaf of 20.
            for size in range(1, 41)
            for pipeline in ("interwave", "intrawave")
        ]
    )
    frame.to_csv(corpus, index=False)

    # `train` requires a provenance snapshot; its contents are irrelevant here.
    snapshot = tmp_path / "provenance.json"
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

    # Pass the fixture's evaluator explicitly so the subprocess uses the same binary.
    out = tmp_path / "model"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uhd_gen",
            "train",
            "--input",
            str(corpus),
            "--provenance",
            str(snapshot),
            "--features",
            "q.size",
            "kernel.pipeline",
            "--target",
            "tflops",
            "--group-by",
            "q.size",
            "--feature-evaluator",
            str(evaluator),
            "--output-dir",
            str(out),
            "--name",
            "encoding round trip",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr[-3000:]

    descriptor = json.loads(next(out.glob("*.uhd.json")).read_text())
    assert descriptor["categorical_encoding"] == {
        "$kernel.pipeline": {"interwave": 0, "intrawave": 1}
    }, "the tool did not ship the table it trained with"

    bundle = load_model(out)
    scores = bundle.scorer(
        pd.DataFrame(
            [
                {"q.size": 12.0, "kernel.pipeline": "interwave"},
                {"q.size": 12.0, "kernel.pipeline": "intrawave"},
            ]
        )
    )
    assert np.all(np.isfinite(scores))
    assert scores[0] > scores[1], "the faster pipeline did not score higher"


def _two_group_time_model(directory: Path, group_zero: float, group_one: float) -> Path:
    """A grouped `time` ranker with exact layer-1 scores per group.

    Layer 1 splits on `kernel.group` at 0.5; each group's layer 2 is a constant.
    """
    from uhd_gen.lgbm_to_flatbuffer import build_gbdt_model

    def constant(leaf: float) -> dict:
        return {
            "max_feature_idx": 1,
            "tree_info": [{"tree_structure": {"leaf_value": leaf}}],
        }

    layer_one = {
        "max_feature_idx": 1,
        "tree_info": [
            {
                "tree_structure": {
                    "split_feature": 1,
                    "threshold": 0.5,
                    "decision_type": "<=",
                    "default_left": True,
                    "left_child": {"leaf_value": group_zero},
                    "right_child": {"leaf_value": group_one},
                }
            }
        ],
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(
        build_gbdt_model(
            layer_one,
            "sha256:grouped_admission",
            group_by_feature_index=1,
            groups=[(0.0, constant(1.0)), (1.0, constant(1.0))],
        )
    )
    (directory / "train_manifest.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "target": "avgTimeMs",
                "objective": "min",
                "group_by_feature": "kernel.group",
            }
        ),
        encoding="utf-8",
    )
    (directory / "heuristic.uhd.json").write_text(
        json.dumps(
            {
                "objective": "min",
                "tree_data": {"artifact": "model.bin"},
                "score": {"metric": "time", "calibrated": True, "transform": "log1p"},
                "features_signature": [f"${name}" for name in FEATURES],
            }
        ),
        encoding="utf-8",
    )
    return directory


@pytest.mark.parametrize(
    "group_zero, group_one, rows, survivor",
    [
        # The runtime discards a negative recovered time, though it is best under min.
        (-0.5, 0.5, (0.0, 1.0), 1.0),
        # An exact tie goes to the smaller group value regardless of row order.
        (0.5, 0.5, (1.0, 0.0), 0.0),
        # Nothing admissible: no group is chosen.
        (-0.5, -0.5, (0.0, 1.0), None),
    ],
)
def test_layer_one_chooses_a_group_only_from_scores_the_runtime_admits(
    tmp_path, group_zero, group_one, rows, survivor
):
    """Mirrors TreeDataAdapter::scoreBatch: only finite (and, for a physical score,
    positive) layer-1 scores may choose a group."""
    bundle = load_model(
        _two_group_time_model(tmp_path / "model", group_zero, group_one)
    )
    frame = pd.DataFrame([{"q.size": 4.0, "kernel.group": group} for group in rows])
    scores = bundle.scorer(frame)
    if survivor is None:
        assert np.isneginf(scores).all()
    else:
        assert frame.loc[np.isfinite(scores), "kernel.group"].tolist() == [survivor]
