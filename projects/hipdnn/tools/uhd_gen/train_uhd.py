#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Train LightGBM model for UHD heuristics.

Adapts the training pipeline from CK dispatcher heuristics (train.py) for
hipDNN's UHD system. Key differences:
- Output is FlatBuffer GbdtModel (not .lgbm file)
- Features come from input data columns (not hardcoded per-op)
- Uses log(target) for scale-invariant training (see score_transform.py)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from . import score_transform
from .features import (
    derive_categorical_encoding,
    encode_feature_value,
    evaluate_feature_rows,
    feature_reference,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Shared by training and by regret evaluation. Held in one place because a regret
# figure measured under different hyperparameters than the shipped model describes a
# model nobody has.
_DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_encoding: dict[str, dict[str, int]] | None = None,
    *,
    signature: list | None = None,
    feature_evaluator: str | None = None,
) -> np.ndarray:
    """Project raw columns or batch inline expressions through the shared runtime."""
    if signature is not None:
        if any(isinstance(entry, dict) for entry in signature):
            _, values = evaluate_feature_rows(df, signature, categorical_encoding, feature_evaluator)
            return np.asarray(values, dtype=np.float64)
        feature_cols = [entry[1:] for entry in signature]
    if not feature_cols:
        raise ValueError("no feature columns; there is nothing to train on")

    columns = []
    for name in feature_cols:
        series = df[name]
        # numpy kinds b/i/u/f are the numeric ones. Object, string and pandas
        # `category` all report something else, and `category` in particular is what
        # pandas would quietly hand LightGBM as an integer code of its own choosing --
        # first-seen order, per DataFrame, which is precisely the encoding RFC 0019
        # §6.5 rules out.
        if getattr(series.dtype, "kind", "O") in "biuf":
            columns.append(series.to_numpy(dtype=np.float64))
            if not np.isfinite(columns[-1]).all():
                raise ValueError(f"feature column {name!r} contains non-finite values")
            continue

        reference = feature_reference(name)
        codes = categorical_encoding.get(reference) if categorical_encoding else None
        encoded = np.empty(len(series), dtype=np.float64)
        for row, raw in enumerate(series):
            try:
                if codes is not None and isinstance(raw, str):
                    code = codes.get(raw)
                    if code is None:
                        raise ValueError(
                            f"categorical value {raw!r} is not in the encoding for "
                            f"{reference}, whose vocabulary is {sorted(codes)}. The "
                            "encoding is derived from the training corpus and ships "
                            "with the model, so a value outside it has no number here "
                            "and none at inference either."
                        )
                    encoded[row] = float(code)
                else:
                    encoded[row] = encode_feature_value(reference, raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"feature column {name!r}, row {row}: {error}") from error
        columns.append(encoded)

    return np.column_stack(columns)


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str] | None = None,
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    n_splits: int = 5,
    categorical_encoding: dict[str, dict[str, int]] | None = None,
    feature_matrix: np.ndarray | None = None,
) -> lgb.Booster:
    """Train LightGBM regressor on log(target) -- score_transform.TRAINED.

    Uses GroupKFold cross-validation when group_cols is provided to prevent
    problem leakage (same problem appearing in both train and validation).

    Args:
        df: DataFrame with feature columns and target column, holding raw logged
            values -- string-valued columns are encoded here, not by the caller.
        feature_cols: List of column names to use as features.
        target_col: Name of target column (e.g., "tflops").
        group_cols: Optional columns for GroupKFold grouping.
        params: LightGBM parameters. Defaults to regression with RMSE.
        num_boost_round: Maximum number of boosting rounds.
        early_stopping_rounds: Early stopping patience.
        n_splits: Number of cross-validation folds.
        categorical_encoding: The `$reference` -> value -> code map to encode string
            columns with, as derived from this corpus by
            features.derive_categorical_encoding. This is the map the model is fitted
            with, so it is the map the descriptor must ship.

    Returns:
        Trained LightGBM Booster.
    """
    X = build_feature_matrix(df, feature_cols, categorical_encoding) if feature_matrix is None else feature_matrix
    target = df[target_col].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target <= 0).any():
        # RFC 0019 §8.3: a zero or negative measurement is no measurement, and the runtime
        # refuses a score that recovers to one. A label the model could only learn to
        # reproduce as an unusable score is an error in the corpus, not a data point.
        raise ValueError(f"target {target_col!r} must contain finite, strictly positive values")
    y = score_transform.forward(target)

    if params is None:
        params = dict(_DEFAULT_PARAMS)

    train_data = lgb.Dataset(X, label=y, feature_name=[f"f{index}" for index in range(X.shape[1])])

    # `folds` takes precomputed splits; a plain split count goes in `nfold`. Passing
    # the integer as `folds` raises AttributeError inside lgb.cv, which made the
    # no-group-columns path — the default — fail outright.
    if group_cols:
        groups = df.groupby(group_cols).ngroup().values
        cv_kwargs = {"folds": list(GroupKFold(n_splits=n_splits).split(X, y, groups))}
        logger.info("Using GroupKFold with %d groups", len(np.unique(groups)))
    else:
        # stratified defaults to True, which routes through StratifiedKFold and
        # rejects a continuous target ("Supported target types are: binary,
        # multiclass"). This is a regressor, so plain KFold is what we want.
        cv_kwargs = {"nfold": n_splits, "stratified": False}
        logger.info("Using standard KFold with %d splits", n_splits)

    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(early_stopping_rounds)],
        **cv_kwargs,
    )

    # Get best iteration from CV
    metric_key = "valid rmse-mean"
    if metric_key not in cv_results:
        metric_key = list(cv_results.keys())[0]
    best_iter = len(cv_results[metric_key])
    best_rmse = cv_results[metric_key][-1]
    logger.info("Best iteration: %d, RMSE: %.4f", best_iter, best_rmse)

    model = lgb.train(params, train_data, num_boost_round=best_iter)
    logger.info("Trained model with %d trees", model.num_trees())
    return model


def _problem_groups(df: pd.DataFrame, problem_cols: list[str]) -> np.ndarray:
    """Integer id per distinct problem, so a problem's variants stay together."""
    return df.groupby(problem_cols).ngroup().values


def induced_ranking_regret(
    measured: np.ndarray, predicted: np.ndarray
) -> tuple[float, bool]:
    """Relative regret of the pick this model induces for one problem.

    The model is a regressor over TFLOPS; the ranking is whatever sorting its
    prediction produces. So the quantity that matters is not how close the predicted
    number is, but how much throughput is lost by taking the configuration it ranks
    first:

        regret = (best_measured - measured[argmax predicted]) / best_measured

    Zero when the model picks a true best. Bounded above by 1. A model can lower its
    RMSE while raising this -- being uniformly closer in value does not imply picking
    better -- which is why prediction error alone cannot answer whether the heuristic
    works.

    Returns the regret and whether the pick was a true best (top-1 hit).
    """
    best = float(np.max(measured))
    if best <= 0.0:
        # Nothing measured a positive rate; a ratio here would be noise or a division
        # by zero, and reporting 0 regret would read as a perfect pick.
        return float("nan"), False

    chosen = float(measured[int(np.argmax(predicted))])
    return (best - chosen) / best, bool(np.isclose(chosen, best))


def evaluate_regret(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    problem_cols: list[str],
    params: dict | None = None,
    num_boost_round: int = 500,
    n_splits: int = 5,
    categorical_encoding: dict[str, dict[str, int]] | None = None,
    feature_matrix: np.ndarray | None = None,
    objective: str = "max",
) -> dict:
    """Out-of-fold top-1 regret of the ranking this regressor induces.

    RFC 0019.13 §11 asks whether the heuristic picks well, which CV RMSE cannot
    answer. Measured out of fold and grouped by problem, so every problem being
    scored was unseen when its scorer was fitted -- an in-sample regret is close to
    meaningless, since the model has already been shown the winner.

    Problems with a single measured configuration are excluded and counted: their
    regret is zero by construction and including them dilutes the metric toward
    whatever fraction of the corpus happens to be single-variant.

    Returns a dict of metrics, including the exclusions, so a number can never be
    read without the population it was computed over.
    """
    if params is None:
        params = dict(_DEFAULT_PARAMS)

    # Encoded, as train_model fits it. Raw values measure a different model, and for a
    # string feature measure nothing: LightGBM rejects the column. Regret is measured
    # over the corpus it is fitted on and publishes no descriptor, so when the caller
    # passes no encoding the corpus's own map is exactly the one training would ship.
    if feature_matrix is not None:
        features = feature_matrix
    else:
        if categorical_encoding is None:
            categorical_encoding = derive_categorical_encoding(df, feature_cols)
        features = build_feature_matrix(df, feature_cols, categorical_encoding)
    measured = df[target_col].values
    groups = _problem_groups(df, problem_cols)

    distinct = len(np.unique(groups))
    if distinct < n_splits:
        raise ValueError(
            f"{distinct} distinct problems is fewer than {n_splits} folds; "
            "regret cannot be measured out of fold on this corpus"
        )

    # Grouped by problem so a problem's other configurations are never in the fold
    # that scores it. Splitting rows at random would leak the answer: the model would
    # have seen the same problem's winner under a different knob setting.
    out_of_fold = np.full(len(df), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
        features, measured, groups
    ):
        booster = lgb.train(
            params,
            lgb.Dataset(
                features[train_idx],
                label=score_transform.forward(measured[train_idx]),
                feature_name=[f"f{index}" for index in range(features.shape[1])],
            ),
            num_boost_round=num_boost_round,
        )
        out_of_fold[test_idx] = booster.predict(features[test_idx])

    regrets: list[float] = []
    hits = 0
    single_variant = 0
    unusable = 0
    for group in np.unique(groups):
        rows = groups == group
        if int(np.count_nonzero(rows)) < 2:
            single_variant += 1
            continue
        if objective == "min":
            values = measured[rows]
            chosen = values[int(np.argmin(out_of_fold[rows]))]
            best = float(np.min(values))
            regret = (chosen - best) / best if best > 0 else float("nan")
            hit = bool(np.isclose(chosen, best))
        else:
            regret, hit = induced_ranking_regret(measured[rows], out_of_fold[rows])
        if np.isnan(regret):
            unusable += 1
            continue
        regrets.append(regret)
        hits += int(hit)

    if not regrets:
        raise ValueError(
            "no problem had two or more measured configurations; regret is undefined"
        )

    ranked = np.asarray(regrets)
    return {
        "problems_scored": len(ranked),
        "problems_single_variant": single_variant,
        "problems_unusable": unusable,
        "top1_accuracy": hits / len(ranked),
        "mean_regret": float(np.mean(ranked)),
        "median_regret": float(np.median(ranked)),
        "p90_regret": float(np.percentile(ranked, 90)),
        "p99_regret": float(np.percentile(ranked, 99)),
        "max_regret": float(np.max(ranked)),
    }


def predict(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Predict using trained model, inverting the trained transform.

    Args:
        model: Trained LightGBM Booster.
        X: Feature array.

    Returns:
        Predictions in original scale (TFLOPS).
    """
    return score_transform.inverse(model.predict(X), score_transform.TRAINED)
