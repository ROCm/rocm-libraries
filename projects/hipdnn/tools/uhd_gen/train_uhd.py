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
    encode_feature_value,
    evaluate_feature_rows,
    feature_reference,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# Training's default LightGBM parameters.
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
            _, values = evaluate_feature_rows(
                df, signature, categorical_encoding, feature_evaluator
            )
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
                raise ValueError(
                    f"feature column {name!r}, row {row}: {error}"
                ) from error
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
    X = (
        build_feature_matrix(df, feature_cols, categorical_encoding)
        if feature_matrix is None
        else feature_matrix
    )
    target = df[target_col].to_numpy(dtype=np.float64)
    if not np.isfinite(target).all() or (target <= 0).any():
        # RFC 0019 §8.3: a zero or negative measurement is no measurement, and the runtime
        # refuses a score that recovers to one. A label the model could only learn to
        # reproduce as an unusable score is an error in the corpus, not a data point.
        raise ValueError(
            f"target {target_col!r} must contain finite, strictly positive values"
        )
    y = score_transform.forward(target)

    if params is None:
        params = dict(_DEFAULT_PARAMS)

    train_data = lgb.Dataset(
        X, label=y, feature_name=[f"f{index}" for index in range(X.shape[1])]
    )

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


def predict(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """Predict using trained model, inverting the trained transform.

    Args:
        model: Trained LightGBM Booster.
        X: Feature array.

    Returns:
        Predictions in original scale (TFLOPS).
    """
    return score_transform.inverse(model.predict(X), score_transform.TRAINED)
