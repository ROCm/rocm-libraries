#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Train LightGBM model for UHD heuristics.

Adapts the training pipeline from CK dispatcher heuristics (train.py) for
hipDNN's UHD system. Key differences:
- Output is FlatBuffer GbdtModel (not .lgbm file)
- Features come from input data columns (not hardcoded per-op)
- Uses log1p(target) for scale-invariant training
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_cols: list[str] | None = None,
    params: dict | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    n_splits: int = 5,
) -> lgb.Booster:
    """Train LightGBM regressor on log1p(target).

    Uses GroupKFold cross-validation when group_cols is provided to prevent
    problem leakage (same problem appearing in both train and validation).

    Args:
        df: DataFrame with feature columns and target column.
        feature_cols: List of column names to use as features.
        target_col: Name of target column (e.g., "tflops").
        group_cols: Optional columns for GroupKFold grouping.
        params: LightGBM parameters. Defaults to regression with RMSE.
        num_boost_round: Maximum number of boosting rounds.
        early_stopping_rounds: Early stopping patience.
        n_splits: Number of cross-validation folds.

    Returns:
        Trained LightGBM Booster.
    """
    X = df[feature_cols].values
    y = np.log1p(df[target_col].values)

    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 127,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }

    train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)

    if group_cols:
        groups = df.groupby(group_cols).ngroup().values
        cv_folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
        logger.info("Using GroupKFold with %d groups", len(np.unique(groups)))
    else:
        cv_folds = n_splits
        logger.info("Using standard KFold with %d splits", n_splits)

    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=num_boost_round,
        folds=cv_folds,
        callbacks=[lgb.early_stopping(early_stopping_rounds)],
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
    """Predict using trained model, inverting log1p transform.

    Args:
        model: Trained LightGBM Booster.
        X: Feature array.

    Returns:
        Predictions in original scale (TFLOPS).
    """
    log_pred = model.predict(X)
    return np.expm1(log_pred)
