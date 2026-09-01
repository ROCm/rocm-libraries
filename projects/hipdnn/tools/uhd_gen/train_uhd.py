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
import math
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold

from .features import encode_feature_value

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def build_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Turn the raw benchmark-log columns into the float matrix LightGBM trains on.

    The log carries raw values: a string field such as `kernel.dtype` arrives as the
    string `"fp16"`, not a number. This is the training side of RFC 0019 §6.5 -- every
    string-valued column is encoded through the fixed table in features.py, which is
    the same table the C++ runtime reads, so a split threshold learned here means the
    same data type when the runtime recomputes the feature.

    Applied unconditionally, with nothing for the caller to select. The mapping is
    global and fixed, so a switch could only ever produce a model whose numbers
    disagree with what the runtime will compute.

    An unencodable value raises. Dropping the column, coercing it, or handing the
    string to LightGBM as a pandas `category` would each yield a model that trains and
    saves and then cannot be reproduced at inference -- the runtime throws on exactly
    this string, so training has to as well.
    """
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
            continue

        reference = f"${name}"
        encoded = np.empty(len(series), dtype=np.float64)
        for row, raw in enumerate(series):
            try:
                encoded[row] = encode_feature_value(reference, raw)
            except (TypeError, ValueError) as error:
                raise ValueError(f"feature column {name!r}, row {row}: {error}") from error
        columns.append(encoded)

    return np.column_stack(columns)


def _python_scalar(value):
    """A pandas/numpy cell as a plain Python scalar.

    The values reported here are also written into train_manifest.json. `np.int64` and
    friends are not JSON-serialisable, so leaving them boxed would raise TypeError at
    the manifest write -- after training has already spent its minutes -- and lose the
    run. Non-finite floats become null for the same reason: `json.dump` would emit the
    Python-only `NaN` literal, which no other JSON reader accepts.
    """
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, float) and not math.isfinite(item):
        return None
    return item


def find_constant_feature_columns(
    df: pd.DataFrame, feature_cols: list[str]
) -> list[tuple[str, object]]:
    """The requested feature columns that never vary, paired with their one value.

    A column with a single value across the corpus cannot separate one candidate from
    another: every tree split on it would be degenerate. Carrying it anyway bloats
    features_signature, changes features_hash, and buys a feature extraction per
    candidate score at runtime (RFC 0019 §7.2) for no ranking signal at all.

    Detection only. What to do about it is a policy question this cannot answer: a
    column is constant either *by construction* -- the kernel matcher pinned it, as
    rocKE's attention kernels pin 8 of their 14 fields, and it can never vary among the
    candidates the model will rank -- or because the *corpus* is thin, in which case the
    column does vary in the world and dropping it yields a model that cannot generalise.
    A CSV cannot tell the two apart, so the caller decides (see --keep-constant-features
    in __main__.py) and this function refuses to guess.

    Returned in the caller's requested order so messages and the manifest read the way
    the --features list was typed.
    """
    if df.empty:
        # No rows observed, so no column has been seen to vary or not. Reporting all of
        # them constant here would turn an empty corpus into a misleading "your features
        # are useless" report instead of the empty-corpus failure it actually is.
        return []

    constants = []
    for name in feature_cols:
        series = df[name]
        # dropna=False: a column of all-NaN is constant too, and a column of one value
        # plus NaN genuinely varies -- the runtime would see two different cells.
        if series.nunique(dropna=False) <= 1:
            constants.append((name, _python_scalar(series.iloc[0])))
    return constants


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
        df: DataFrame with feature columns and target column, holding raw logged
            values -- string-valued columns are encoded here, not by the caller.
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
    X = build_feature_matrix(df, feature_cols)
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
    """Predict using trained model, inverting log1p transform.

    Args:
        model: Trained LightGBM Booster.
        X: Feature array.

    Returns:
        Predictions in original scale (TFLOPS).
    """
    log_pred = model.predict(X)
    return np.expm1(log_pred)
