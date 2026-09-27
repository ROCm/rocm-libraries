#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Evaluating a two-layer artifact, and the ways it silently reports half a model.

A grouped model decides twice -- layer 1 picks the group, layer 2 orders within it -- and every
failure here produces a *plausible* number rather than an error:

- reading only `trees` scores layer 1 alone, which is exactly what a correct single-layer model
  produces, so nothing about the output looks wrong;
- `model.lgbm` cannot carry the per-group ensembles at all, so preferring it silently measures
  half the model on any directory where training kept it;
- the group decision must be made over ONE problem's candidates: taken across a whole corpus,
  one problem's winning group blanks out every other problem's candidates, and the regret that
  comes back is a number about nothing;
- a rejected group must stay unusable through the score transform -- `expm1(-inf)` is -1.0, a
  finite value that outranks a genuinely negative score.
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
    """A booster over (q.size, kernel.group) -> target, with enough signal to split on."""
    frame = pd.DataFrame(rows, columns=["q.size", "kernel.group", "y"])
    data = lgb.Dataset(frame[["q.size", "kernel.group"]].to_numpy(), label=frame["y"].to_numpy())
    return lgb.train(
        {"objective": "regression", "num_leaves": 4, "learning_rate": 0.3,
         "min_data_in_leaf": 1, "min_data_in_bin": 1, "verbose": -1},
        data,
        num_boost_round=num_trees,
    )


def _write_model(directory: Path, *, grouped: bool) -> Path:
    """A trained pair on disk, as `train --output-dir` leaves it."""
    directory.mkdir(parents=True, exist_ok=True)

    # Layer 1 prefers group 1 on large sizes and group 0 on small ones.
    layer_one = _booster([(size, g, (10.0 + size) if g == 1.0 else (20.0 - size))
                          for size in range(1, 12) for g in GROUPS])
    lgbm_path = directory / "model.lgbm"
    layer_one.save_model(str(lgbm_path))

    group_models = None
    if grouped:
        # Within a group, larger q.size is better -- a shape layer 1 does not express, so a
        # difference in the ranking can only have come from layer 2.
        group_models = [
            (float(g), _booster([(size, g, float(size) * (2.0 if g == 1.0 else 1.0))
                                 for size in range(1, 12)]))
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
        "objective": "max",
        "num_samples": 22,
        "group_by_feature": "kernel.group" if grouped else None,
        "group_models": len(group_models or []),
    }
    (directory / "train_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / "heuristic.uhd.json").write_text(
        json.dumps({"objective": "max", "tree_data": {"artifact": "model.bin"},
                    "features_signature": [f"${name}" for name in FEATURES]}),
        encoding="utf-8",
    )
    return directory


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [{"q.size": size, "kernel.group": g} for g in GROUPS for size in (2.0, 9.0)]
    )


def test_a_grouped_artifact_rejects_every_candidate_outside_the_chosen_group():
    """The property that distinguishes two layers from one.

    Scoring only `trees` returns a usable score for every row, which is indistinguishable from a
    correct single-layer model. If nothing here is -inf, the second layer was never consulted.
    """
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "grouped", grouped=True))
    scores = bundle.scorer(_candidates())

    assert np.isneginf(scores).any(), "no candidate was rejected; layer 2 was not consulted"
    assert np.isfinite(scores).any(), "every candidate was rejected; nothing could be picked"

    # Exactly one group survives, and it survives whole.
    survived = _candidates().loc[np.isfinite(scores), "kernel.group"].unique()
    assert len(survived) == 1


def test_an_ungrouped_artifact_scores_every_candidate():
    """The compatibility claim, and the one that protects every shipped single-layer model.

    The grouped path must not engage on an artifact that has no groups: a model written before
    the two-layer fields existed has `group_by_feature_index = -1`, and blanking candidates on
    it would break ranking for every UHD already in the field.
    """
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "flat", grouped=False))
    scores = bundle.scorer(_candidates())

    assert np.all(np.isfinite(scores))
    assert bundle.group_feature is None


def test_a_grouped_model_is_not_ranked_with_the_booster():
    """`model.lgbm` holds layer 1 alone; LightGBM cannot represent the per-group ensembles.

    `load_model` prefers the booster when it is present, which on a grouped directory would
    measure half the model and report a single-layer figure that looks entirely plausible.
    """
    with_tmp = Path(__import__("tempfile").mkdtemp())
    directory = _write_model(with_tmp / "both", grouped=True)
    assert (directory / "model.lgbm").exists(), "fixture must keep the booster to be meaningful"

    bundle = load_model(directory)
    assert bundle.source.endswith("model.bin")


def test_the_group_decision_is_made_per_problem():
    """Each problem chooses its own group.

    `scoreBatch` is handed one query's candidates. Choosing a single group across a corpus would
    let one problem's winner blank out another's, so a problem whose best group differs would
    lose every candidate and the regret would describe nothing.
    """
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "grouped", grouped=True))

    for size in (2.0, 9.0):
        frame = pd.DataFrame([{"q.size": size, "kernel.group": g} for g in GROUPS])
        scores = bundle.scorer(frame)
        assert np.isfinite(scores).any(), f"every candidate rejected at q.size={size}"


def _corpus() -> pd.DataFrame:
    """Two problems, two groups each, with the best candidate in different groups."""
    rows = []
    for benchmark, best_group in (("aaaa", 1.0), ("bbbb", 0.0)):
        for group in GROUPS:
            for size in (2.0, 9.0):
                rows.append({
                    "benchmark": benchmark,
                    "device": "d0",
                    "kernel": f"k{group}{size}",
                    "q.size": size,
                    "kernel.group": group,
                    "is_valid": "True",
                    "tflops": (100.0 if group == best_group else 40.0) + size,
                })
    return pd.DataFrame(rows)


def test_two_stage_regret_sums_to_the_total():
    """The decomposition is only useful if it accounts for the whole shortfall.

    Both parts are measured against the same oracle precisely so they add up; a part computed
    against the group's own best instead would still look reasonable and would not sum, leaving
    a total nobody can attribute.
    """
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
        assert problem.group_regret + problem.in_group_regret == pytest.approx(problem.regret)
        assert problem.group_regret >= 0.0


def test_no_decomposition_is_reported_without_a_group_column():
    """Absent must mean "not a two-layer model", not "the split came out zero".

    A section of zeros would read as a model that never loses anything to its group choice,
    which is the most flattering possible misreading.
    """
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_write_model(with_tmp / "flat", grouped=False))

    result = evaluate_corpus(
        _corpus(), bundle.scorer, target="tflops", objective="max", eval_fraction=1.0
    )
    assert result.report["metrics"]["two_stage"] is None
    assert all(problem.group_regret is None for problem in result.problems)


def _strict_less_than_model(directory: Path) -> Path:
    """An artifact whose one split uses `<` rather than `<=`.

    Built through the object API rather than by training: LightGBM emits only `<=` and `==`
    decision types, so `lgbm_to_flatbuffer` always writes `decision_lte=True` and no trained
    model can exercise the other branch. A hand-written or foreign artifact can -- which is
    exactly what `TestTreeDataAdapter` and the rocKE model generator produce.
    """
    import flatbuffers

    from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT
    from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT

    tree = GbdtTreeT()
    # Root splits slot 0 at 10. Left leaf 1.0, right leaf 9.0, with `<` semantics: a row at
    # exactly 10 belongs on the RIGHT. Under `<=` it would go left, and under the complement
    # (`>`) every row would swap sides -- so the three readings give three different answers.
    tree.featureIndices = [0, 0, 0]
    tree.thresholds = [10.0, 0.0, 0.0]
    tree.leftChildren = [1, -1, -1]
    tree.rightChildren = [2, -1, -1]
    tree.leafValues = [0.0, 1.0, 9.0]
    tree.defaultLeft = [True, True, True]
    tree.decisionLte = [False, False, False]

    model = GbdtModelT()
    model.trees = [tree]
    model.baseScore = 0.0
    model.numFeatures = 1
    model.featuresHash = "sha256:strict_lt"
    model.groupByFeatureIndex = -1

    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder))

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.bin").write_bytes(bytes(builder.Output()))
    (directory / "train_manifest.json").write_text(
        json.dumps({"features": ["q.size"], "target": "tflops", "objective": "max"}),
        encoding="utf-8",
    )
    (directory / "heuristic.uhd.json").write_text(
        json.dumps({"objective": "max", "tree_data": {"artifact": "model.bin"},
                    "features_signature": ["$q.size"]}),
        encoding="utf-8",
    )
    return directory


def test_a_strict_less_than_split_routes_the_way_the_runtime_routes_it():
    """`decision_lte` false means `<`, and the complement is not the same thing.

    Regression: this traversal used `x > threshold` for the false case, which is the complement
    of `<=` rather than `<`. Every row went down the opposite subtree, so the evaluator ranked
    such a model backwards while the C++ runtime scored it correctly -- a disagreement that
    would read as a bad model rather than a bad reader. The schema and TreeDataAdapter both say
    `<`; the test mirror in test_converter.py already had it right.
    """
    with_tmp = Path(__import__("tempfile").mkdtemp())
    bundle = load_model(_strict_less_than_model(with_tmp / "strict_lt"))

    scores = bundle.scorer(pd.DataFrame([{"q.size": 5.0}, {"q.size": 10.0}, {"q.size": 15.0}]))

    # Under `<`: 5 goes left (1.0), 10 and 15 go right (9.0). expm1 is monotonic, so the
    # comparison holds on the returned values.
    assert scores[0] < scores[1], "a value below the threshold took the wrong branch"
    assert scores[1] == pytest.approx(scores[2]), "10 and 15 must share the right-hand leaf"
    assert scores[0] == pytest.approx(np.expm1(1.0))
    assert scores[1] == pytest.approx(np.expm1(9.0))


def test_a_generated_encoding_survives_train_then_score(tmp_path, evaluator):
    """The round trip the whole mechanism rests on.

    A string field outside the fixed table can only be a feature if the tool observes its
    values, ships the map, and the scorer reads it back. Any break gives a model that trains
    and saves and then cannot be reproduced: before this, training encoded nothing and the
    runtime threw on the first such value.
    """
    import subprocess
    import sys

    corpus = tmp_path / "corpus.csv"
    frame = pd.DataFrame([
        {"q.size": size, "kernel.pipeline": pipeline,
         "tflops": size * (2.0 if pipeline == "interwave" else 1.0)}
        # Wide enough to split: LightGBM's default min_data_in_leaf is 20, so a corpus of a
        # couple of dozen rows trains to a single constant leaf and would fail the ordering
        # assertion below for a reason that has nothing to do with the encoding.
        for size in range(1, 41) for pipeline in ("interwave", "intrawave")
    ])
    frame.to_csv(corpus, index=False)

    # `train` refuses to record an unattributable model, so the round trip needs a
    # provenance snapshot. It is fixture scaffolding: this test is about the encoding
    # surviving the trip, not about what the snapshot says.
    snapshot = tmp_path / "provenance.json"
    snapshot.write_text(json.dumps({
        "ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "1.0"},
        "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.0"},
        "umd": [],
    }), encoding="utf-8")

    # `evaluator` because training stamps a features_hash, and that digest has one
    # definition -- the binary. Named on the command line rather than left to the child's
    # own lookup so the subprocess resolves what the fixture resolved: a checkout with
    # nothing built skips here instead of failing on a search that came up empty.
    out = tmp_path / "model"
    result = subprocess.run(
        [sys.executable, "-m", "uhd_gen", "train",
         "--input", str(corpus), "--provenance", str(snapshot),
         "--features", "q.size", "kernel.pipeline",
         "--target", "tflops", "--group-by", "q.size",
         "--feature-evaluator", str(evaluator),
         "--output-dir", str(out), "--name", "encoding round trip"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert result.returncode == 0, result.stderr[-3000:]

    descriptor = json.loads(next(out.glob("*.uhd.json")).read_text())
    assert descriptor["categorical_encoding"] == {
        "$kernel.pipeline": {"interwave": 0, "intrawave": 1}
    }, "the tool did not ship the table it trained with"

    # And the scorer reads it back: passing the raw strings must produce numbers, not an
    # "is a string and has no categorical encoding" refusal.
    bundle = load_model(out)
    scores = bundle.scorer(
        pd.DataFrame([{"q.size": 12.0, "kernel.pipeline": "interwave"},
                      {"q.size": 12.0, "kernel.pipeline": "intrawave"}])
    )
    assert np.all(np.isfinite(scores))
    assert scores[0] > scores[1], "the faster pipeline did not score higher"
