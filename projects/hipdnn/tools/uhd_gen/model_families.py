#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Non-LightGBM tree ensembles exported to the GbdtModel artifact TreeDataAdapter reads.

`train --model-family` picks the library that fits the model. Every family is written to
the same `tree_data` artifact (gbdt_model.fbs) the runtime already loads, so a new family
needs no runtime change and no new descriptor field. Each exporter reproduces the source
library's routing exactly, and `tree_equivalence` proves it per tree:

* scikit-learn `HistGradientBoostingRegressor`: `x <= threshold` goes left, in float64, as
  the runtime does. NaN follows `missing_go_to_left`. Leaves already include the
  learning rate and `_baseline_prediction` is the base score. A split that separates only
  NaN has threshold +inf, which the artifact cannot hold; it becomes DBL_MAX, so the one
  input routed differently is +inf itself (expression features and training refuse
  non-finite values).
* XGBoost (`gbtree`): `float32(x) < threshold` goes left. The runtime compares a float64
  feature, so each float32 threshold is moved to the float64 midpoint below it, with `<` or
  `<=` chosen by which side the round-to-nearest-even tie falls on. Every float64 input
  then takes the branch XGBoost takes. Leaves are float32 and include eta; the intercept
  (`base_score`) is the base score.

The scores can still differ from the library's own `predict` by summation rounding:
scikit-learn adds the baseline first, XGBoost accumulates in float32, and the runtime sums
the leaves in float64 and adds the base last. Leaf routing and leaf values are identical.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import flatbuffers
import numpy as np

from . import score_transform

# Importing the package puts `_generated/` on sys.path; see uhd_gen/__init__.py.
import uhd_gen  # noqa: F401
from hipdnn_flatbuffers_sdk.data_objects.GbdtModel import GbdtModelT
from hipdnn_flatbuffers_sdk.data_objects.GbdtTree import GbdtTreeT

from .lgbm_to_flatbuffer import GBDT_MODEL_FILE_IDENTIFIER, resolve_training_date

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

LIGHTGBM = "lightgbm"
SKLEARN_HGB = "sklearn-hgb"
XGBOOST = "xgboost"
#: `train --model-family` choices; LightGBM stays the default.
FAMILIES = (LIGHTGBM, SKLEARN_HGB, XGBOOST)

#: Matched to train_uhd._DEFAULT_PARAMS where the libraries share a knob: 127 leaves,
#: rate 0.05, 255 bins, 20 samples per leaf (LightGBM's min_data_in_leaf default).
#: HistGradientBoosting has no row or per-tree column subsampling.
HGB_PARAMS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_leaf_nodes": 127,
    "min_samples_leaf": 20,
    "max_bins": 255,
    "l2_regularization": 0.0,
}

#: Leaf-wise growth like LightGBM. min_child_weight is a hessian sum, which equals the
#: sample count under squared error, so 20 matches min_data_in_leaf.
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "grow_policy": "lossguide",
    "max_depth": 0,
    "max_leaves": 127,
    "eta": 0.05,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_bin": 255,
}

#: XGBoost objectives whose margin is the prediction (identity link), so the summed
#: trees plus the intercept are the regression output. Others would need a link the
#: runtime does not apply.
_XGB_IDENTITY_OBJECTIVES = (
    "reg:squarederror",
    "reg:absoluteerror",
    "reg:pseudohubererror",
)


def _cv_folds(
    df: pd.DataFrame, group_cols: list[str] | None, n_splits: int, rows: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The folds train_uhd.train_model hands lgb.cv: GroupKFold over shapes, or KFold."""
    from sklearn.model_selection import GroupKFold, KFold

    if group_cols:
        groups = df.groupby(group_cols[0]).ngroup().values
        logger.info("Using GroupKFold with %d shapes", len(np.unique(groups)))
        return list(GroupKFold(n_splits=n_splits).split(np.zeros(rows), groups=groups))
    # lgb.cv shuffles its nfold split with seed 0.
    logger.info("Using standard KFold with %d splits", n_splits)
    return list(
        KFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros(rows))
    )


def early_stopped_rounds(curve: np.ndarray, patience: int) -> int:
    """Rounds lgb.early_stopping keeps for a validation curve (lower is better).

    Training stops once `patience` rounds pass without a new best, and the model keeps the
    best round. A patience of 0 or less disables early stopping, as it does for LightGBM
    and XGBoost, and keeps every round.
    """
    if patience <= 0:
        return len(curve)
    best = 0
    for index in range(1, len(curve)):
        if curve[index] < curve[best]:
            best = index
        elif index - best >= patience:
            break
    return best + 1


def train_family(
    family: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str] | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    n_splits: int = 5,
    categorical_encoding: dict[str, dict[str, int]] | None = None,
    feature_matrix: np.ndarray | None = None,
    seed: int = 0,
):
    """Fit `family` on log(target), choosing the round count as train_model does.

    Cross-validation uses the same folds as LightGBM's (GroupKFold over shapes), keeps the
    round with the best mean validation RMSE under the same early-stopping patience, then
    refits on every row for that many rounds.
    """
    from .train_uhd import build_feature_matrix

    if family not in (SKLEARN_HGB, XGBOOST):
        raise ValueError(
            f"train_family fits {SKLEARN_HGB} or {XGBOOST}, not {family!r}"
        )
    X = (
        build_feature_matrix(df, feature_cols, categorical_encoding)
        if feature_matrix is None
        else feature_matrix
    )
    target = df[target_col].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target <= 0).any():
        # RFC 0019 §8.3: a non-positive measurement is no measurement.
        raise ValueError(
            f"target {target_col!r} must contain finite, strictly positive values"
        )
    y = score_transform.forward(target)
    folds = _cv_folds(df, group_cols, n_splits, len(y))

    if family == SKLEARN_HGB:
        return _train_hgb(X, y, folds, num_boost_round, early_stopping_rounds, seed)
    return _train_xgb(X, y, folds, num_boost_round, early_stopping_rounds, seed)


def _train_hgb(X, y, folds, num_boost_round, early_stopping_rounds, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor

    def make(rounds: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            **HGB_PARAMS,
            max_iter=rounds,
            early_stopping=False,
            categorical_features=None,
            random_state=seed,
        )

    curves = []
    for train_rows, valid_rows in folds:
        model = make(num_boost_round).fit(X[train_rows], y[train_rows])
        curve = [
            float(np.sqrt(np.mean((stage - y[valid_rows]) ** 2)))
            for stage in model.staged_predict(X[valid_rows])
        ]
        # A fold can finish early when no split improves; hold its last value.
        curve += [curve[-1]] * (num_boost_round - len(curve))
        curves.append(curve)
    mean_curve = np.mean(np.asarray(curves), axis=0)
    rounds = early_stopped_rounds(mean_curve, early_stopping_rounds)
    logger.info("Best iteration: %d, RMSE: %.4f", rounds, mean_curve[rounds - 1])
    model = make(rounds).fit(X, y)
    logger.info("Trained model with %d trees", model.n_iter_)
    return model


def _train_xgb(X, y, folds, num_boost_round, early_stopping_rounds, seed):
    import xgboost as xgb

    params = dict(XGB_PARAMS, seed=seed)
    data = xgb.DMatrix(X, label=y)
    history = xgb.cv(
        params,
        data,
        num_boost_round=num_boost_round,
        folds=folds,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
    )
    rounds = len(history)
    logger.info(
        "Best iteration: %d, RMSE: %.4f", rounds, history["test-rmse-mean"].iloc[-1]
    )
    model = xgb.train(params, data, num_boost_round=rounds)
    logger.info("Trained model with %d trees", model.num_boosted_rounds())
    return model


def family_of(model: Any) -> str:
    """The `--model-family` name of a fitted model object."""
    module = type(model).__module__
    if module.startswith("lightgbm"):
        return LIGHTGBM
    if module.startswith("xgboost"):
        return XGBOOST
    if type(model).__name__ == "HistGradientBoostingRegressor":
        return SKLEARN_HGB
    raise TypeError(f"no model family for {type(model).__name__}")


def model_summary(model: Any, names: list[str]) -> dict[str, Any]:
    """`num_trees` and per-feature `{gain, split}` importance for the train manifest.

    Gain is each library's own split gain summed per feature (LightGBM `gain`, XGBoost
    `total_gain`, the HistGradientBoosting node `gain`), so compare it within a family.
    """
    family = family_of(model)
    if family == LIGHTGBM:
        gains = model.feature_importance(importance_type="gain")
        splits = model.feature_importance(importance_type="split")
        num_trees = model.num_trees()
    elif family == XGBOOST:
        total_gain = model.get_score(importance_type="total_gain")
        weight = model.get_score(importance_type="weight")
        gains = [total_gain.get(f"f{index}", 0.0) for index in range(len(names))]
        splits = [weight.get(f"f{index}", 0) for index in range(len(names))]
        num_trees = model.num_boosted_rounds()
    else:
        gains = np.zeros(len(names))
        splits = np.zeros(len(names), dtype=np.int64)
        for (predictor,) in model._predictors:
            internal = predictor.nodes[predictor.nodes["is_leaf"] == 0]
            np.add.at(gains, internal["feature_idx"], internal["gain"])
            np.add.at(splits, internal["feature_idx"], 1)
        num_trees = model.n_iter_
    return {
        "num_trees": int(num_trees),
        "feature_importance": {
            name: {"gain": float(gain), "split": int(split)}
            for name, gain, split in zip(names, gains, splits)
        },
    }


def _tree(
    feature, threshold, left, right, leaf, default_left, decision_lte
) -> GbdtTreeT:
    tree = GbdtTreeT()
    tree.featureIndices = [int(value) for value in feature]
    tree.thresholds = [float(value) for value in threshold]
    tree.leftChildren = [int(value) for value in left]
    tree.rightChildren = [int(value) for value in right]
    tree.leafValues = [float(value) for value in leaf]
    tree.defaultLeft = [bool(value) for value in default_left]
    tree.decisionLte = [bool(value) for value in decision_lte]
    return tree


def hgb_trees(model) -> tuple[list[GbdtTreeT], float, int]:
    """(trees, base score, feature count) of a fitted HistGradientBoostingRegressor."""
    from sklearn._loss.link import IdentityLink

    if model._baseline_prediction.size != 1 or any(
        len(iteration) != 1 for iteration in model._predictors
    ):
        raise ValueError(
            "only single-output HistGradientBoostingRegressor models export"
        )
    if not isinstance(model._loss.link, IdentityLink):
        # poisson and gamma predict through an exp link the runtime does not apply, so the
        # raw baseline plus leaves would be the log-margin.
        raise ValueError(
            f"HistGradientBoostingRegressor(loss={model.loss!r}) applies an inverse link "
            "to its raw sum; only identity-link losses (squared_error, absolute_error, "
            "quantile) export"
        )
    trees = []
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        if nodes["is_categorical"].any():
            raise ValueError(
                "categorical HistGradientBoosting splits have no tree_data form; fit with "
                "categorical_features=None (uhd_gen encodes categories as codes)"
            )
        leaf = nodes["is_leaf"].astype(bool)
        threshold = nodes["num_threshold"].astype(np.float64)
        # A split that sends only NaN right has threshold +inf; DBL_MAX with `<=` sends
        # every finite value left the same way and keeps the artifact finite.
        threshold = np.where(
            np.isposinf(threshold) & ~leaf, np.finfo(np.float64).max, threshold
        )
        trees.append(
            _tree(
                np.where(leaf, -1, nodes["feature_idx"].astype(np.int64)),
                np.where(leaf, 0.0, threshold),
                np.where(leaf, -1, nodes["left"].astype(np.int64)),
                np.where(leaf, -1, nodes["right"].astype(np.int64)),
                np.where(leaf, nodes["value"], 0.0),
                nodes["missing_go_to_left"].astype(bool),
                np.ones(len(nodes), dtype=bool),
            )
        )
    return (
        trees,
        float(model._baseline_prediction.ravel()[0]),
        int(model.n_features_in_),
    )


def float32_less_than(threshold: float) -> tuple[float, bool]:
    """(t, lte) so that, for every float64 x, `x < t` (or `x <= t`) == `float32(x) < threshold`.

    float32(x) < threshold holds exactly when x rounds to `below`, the float32 just under
    `threshold`, or lower: when x is under the float64 midpoint of the two, plus the
    midpoint itself when round-to-nearest-even takes it down to `below`. The midpoint of
    two adjacent float32 values is exact in float64.
    """
    upper = np.float32(threshold)
    if not np.isfinite(upper):
        raise ValueError(f"split threshold {threshold!r} is not finite")
    with np.errstate(over="ignore"):
        below = np.nextafter(upper, np.float32(-np.inf))
    if not np.isfinite(below):
        raise ValueError(
            f"split threshold {threshold!r} has no finite float32 below it"
        )
    midpoint = (float(below) + float(upper)) / 2.0
    # Adjacent float32 values alternate the low bit of their encoding, which is the low
    # bit of the significand; the tie goes to the even one.
    tie_goes_down = (int(below.view(np.uint32)) & 1) == 0
    return midpoint, tie_goes_down


def _xgb_document(booster) -> dict[str, Any]:
    return json.loads(booster.save_raw(raw_format="json"))


def _xgb_base_score(learner: dict[str, Any]) -> float:
    # "5E-1" in XGBoost 1.x/2.x, "[5E-1]" (one intercept per target) in 3.x.
    raw = learner["learner_model_param"]["base_score"]
    values = json.loads(raw) if raw.strip().startswith("[") else [float(raw)]
    if len(values) != 1:
        raise ValueError("only single-target XGBoost models export")
    return float(np.float32(values[0]))


def xgb_trees(booster) -> tuple[list[GbdtTreeT], float, int]:
    """(trees, base score, feature count) of a trained `gbtree` xgboost.Booster."""
    learner = _xgb_document(booster)["learner"]
    objective = learner["objective"]["name"]
    if objective not in _XGB_IDENTITY_OBJECTIVES:
        raise ValueError(
            f"XGBoost objective {objective!r} is not exported: tree_data sums margins, so "
            f"only identity-link objectives ({', '.join(_XGB_IDENTITY_OBJECTIVES)}) give "
            "the library's prediction"
        )
    booster_json = learner["gradient_booster"]
    if booster_json["name"] != "gbtree":
        raise ValueError(
            f"only gbtree XGBoost models export, not {booster_json['name']!r}"
        )
    # XGBoost 3 folds DART into gbtree: each tree's output is scaled by its weight.
    weights = booster_json["model"].get("weight_drop") or []
    if any(float(weight) != 1.0 for weight in weights):
        raise ValueError(
            "XGBoost trees scaled by dropout weights (rate_drop) have no tree_data form"
        )
    trees = []
    for tree in booster_json["model"]["trees"]:
        if int(tree["tree_param"].get("size_leaf_vector", "1")) > 1:
            raise ValueError("multi-target XGBoost trees have no tree_data form")
        if any(split_type != 0 for split_type in tree["split_type"]):
            raise ValueError(
                "categorical XGBoost splits have no tree_data form; uhd_gen encodes "
                "categories as numeric codes"
            )
        left = np.asarray(tree["left_children"], dtype=np.int64)
        leaf = left == -1
        # Leaves keep their float32 value in `split_conditions`.
        conditions = np.asarray(tree["split_conditions"], dtype=np.float32)
        thresholds = np.zeros(len(left), dtype=np.float64)
        lte = np.ones(len(left), dtype=bool)
        for node in np.flatnonzero(~leaf):
            thresholds[node], lte[node] = float32_less_than(conditions[node])
        trees.append(
            _tree(
                np.where(leaf, -1, tree["split_indices"]),
                thresholds,
                left,
                np.where(leaf, -1, tree["right_children"]),
                np.where(leaf, conditions.astype(np.float64), 0.0),
                np.asarray(tree["default_left"], dtype=bool),
                lte,
            )
        )
    return trees, _xgb_base_score(learner), int(booster.num_features())


def _framework_trees(model) -> tuple[str, str, tuple[list[GbdtTreeT], float, int]]:
    family = family_of(model)
    if family == SKLEARN_HGB:
        return "sklearn", "regression", hgb_trees(model)
    if family == XGBOOST:
        objective = _xgb_document(model)["learner"]["objective"]["name"]
        return "xgboost", objective, xgb_trees(model)
    raise ValueError(f"{family} models are written by lgbm_to_flatbuffer.convert")


def build_family_model(
    model,
    features_hash: str,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
) -> bytes:
    """GbdtModel bytes for a fitted HistGradientBoosting or XGBoost model."""
    framework, objective, (trees, base_score, num_features) = _framework_trees(model)
    artifact = GbdtModelT()
    artifact.trees = trees
    artifact.numFeatures = num_features
    artifact.featuresHash = features_hash
    artifact.baseScore = base_score
    # Both libraries fold the rate into their leaves; the field is provenance only.
    artifact.learningRate = 1.0
    artifact.framework = framework
    artifact.trainingObjective = objective
    stamp = resolve_training_date(training_date)
    if stamp is not None:
        artifact.trainingDate = stamp
    if num_training_samples is not None:
        artifact.numTrainingSamples = num_training_samples
    if training_arches:
        artifact.trainingArches = list(training_arches)
    if model_version:
        artifact.modelVersion = model_version
    builder = flatbuffers.Builder(1024 * 1024)
    builder.Finish(artifact.Pack(builder), file_identifier=GBDT_MODEL_FILE_IDENTIFIER)
    return bytes(builder.Output())


def export_family(
    model,
    features_hash: str,
    output_path: str | Path,
    num_training_samples: int | None = None,
    training_arches: list[str] | None = None,
    model_version: str | None = None,
    training_date: str | None = None,
) -> str:
    """Write `model` as a GbdtModel artifact; returns the bare hex SHA-256 of its bytes."""
    buffer = build_family_model(
        model,
        features_hash,
        num_training_samples,
        training_arches=training_arches,
        model_version=model_version,
        training_date=training_date,
    )
    Path(output_path).write_bytes(buffer)
    digest = hashlib.sha256(buffer).hexdigest()
    logger.info(
        "Exported %s model to %s (%d bytes, sha256:%s)",
        family_of(model),
        output_path,
        len(buffer),
        digest,
    )
    return digest


def _walk(tree: GbdtTreeT, X: np.ndarray) -> np.ndarray:
    """Leaf node index per row, routed as TreeDataAdapter::scorePrepared routes."""
    feature = np.asarray(tree.featureIndices, dtype=np.int64)
    threshold = np.asarray(tree.thresholds, dtype=np.float64)
    left = np.asarray(tree.leftChildren, dtype=np.int64)
    right = np.asarray(tree.rightChildren, dtype=np.int64)
    default_left = np.asarray(tree.defaultLeft, dtype=bool)
    lte = np.asarray(tree.decisionLte, dtype=bool)
    node = np.zeros(len(X), dtype=np.int64)
    while True:
        at = np.flatnonzero(left[node] >= 0)
        if not len(at):
            return node
        here = node[at]
        x = X[at, feature[here]]
        go_left = np.where(lte[here], x <= threshold[here], x < threshold[here])
        go_left = np.where(np.isnan(x), default_left[here], go_left)
        node[at] = np.where(go_left, left[here], right[here])


def _library_tree_outputs(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """(rows, trees) per-tree output the source library computes for X, and for XGBoost
    the (rows, trees) node id each row lands on (pred_leaf), which the export keeps."""
    if family_of(model) == SKLEARN_HGB:
        from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

        # As HistGradientBoosting._raw_predict calls each tree's predictor.
        bitsets, feature_map = model._bin_mapper.make_known_categories_bitsets()
        threads = _openmp_effective_n_threads()
        outputs = np.column_stack(
            [
                predictor.predict(X, bitsets, feature_map, threads)
                for (predictor,) in model._predictors
            ]
        )
        return outputs, None
    import xgboost as xgb

    leaves = model.predict(xgb.DMatrix(X), pred_leaf=True).astype(np.int64)
    leaves = leaves.reshape(len(X), -1)
    values = [
        np.asarray(tree["split_conditions"], dtype=np.float32).astype(np.float64)
        for tree in _xgb_document(model)["learner"]["gradient_booster"]["model"][
            "trees"
        ]
    ]
    outputs = np.column_stack([values[t][leaves[:, t]] for t in range(len(values))])
    return outputs, leaves


def _library_predict(model, X: np.ndarray) -> np.ndarray:
    if family_of(model) == XGBOOST:
        import xgboost as xgb

        return model.predict(xgb.DMatrix(X)).astype(np.float64)
    return model.predict(X)


def _runtime_sum(base_score: float, tree_outputs: np.ndarray) -> np.ndarray:
    """base + the per-tree outputs, added tree by tree as TreeDataAdapter adds them."""
    total = np.zeros(len(tree_outputs), dtype=np.float64)
    for column in tree_outputs.T:
        total += column
    return base_score + total


def library_scores_in_runtime_order(
    model, artifact: bytes, X: np.ndarray
) -> np.ndarray:
    """The source library's per-tree outputs for X, summed in the runtime's order onto the
    artifact's base score: what TreeDataAdapter::score must return bit for bit."""
    stored = GbdtModelT.InitFromPackedBuf(bytearray(artifact), 0)
    library, _ = _library_tree_outputs(model, np.asarray(X, dtype=np.float64))
    return _runtime_sum(float(stored.baseScore), library)


def tree_equivalence(model, artifact: bytes, X: np.ndarray) -> dict[str, Any]:
    """Compare an exported artifact with the library that trained it, on rows X.

    `trees_bitwise_equal` is true when every tree's output, routed as the runtime routes,
    is bit-for-bit the library's own per-tree output; for XGBoost `leaves_equal` also
    compares the node each row lands on with `pred_leaf`. `max_abs_diff` is the largest gap
    between the runtime's score (leaves summed in float64, base added last) and the
    library's `predict`, which only summation order and precision can move.
    """
    stored = GbdtModelT.InitFromPackedBuf(bytearray(artifact), 0)
    X = np.asarray(X, dtype=np.float64)
    library, library_leaves = _library_tree_outputs(model, X)
    leaves = np.column_stack([_walk(tree, X) for tree in stored.trees])
    exported = np.column_stack(
        [
            np.asarray(tree.leafValues, dtype=np.float64)[leaves[:, index]]
            for index, tree in enumerate(stored.trees)
        ]
    )
    mismatched = np.flatnonzero((exported != library).any(axis=0))
    runtime = _runtime_sum(float(stored.baseScore), exported)
    gap = np.abs(runtime - _library_predict(model, X))
    return {
        "rows": int(len(X)),
        "trees": int(len(stored.trees)),
        "trees_bitwise_equal": not len(mismatched),
        "mismatched_trees": mismatched.tolist(),
        "leaves_equal": (
            None if library_leaves is None else bool((leaves == library_leaves).all())
        ),
        "max_abs_diff": float(gap.max()) if len(X) else 0.0,
    }
