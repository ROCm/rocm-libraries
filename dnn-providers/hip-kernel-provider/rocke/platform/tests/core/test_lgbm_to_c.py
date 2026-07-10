"""Correctness test for the LightGBM->C emitter (lgbm_to_c).

Trains a tiny regression booster, emits C, compiles it, and asserts the
generated predictor matches ``booster.predict`` element-for-element on random
feature vectors (the whole point: the runtime C must score identically to the
trained model). Skipped if lightgbm or a C compiler is unavailable.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

# Make the heuristics package importable (lgbm_to_c lives alongside train.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_HEUR = os.path.normpath(os.path.join(_HERE, "..", "..", "python", "rocke", "heuristics"))
if _HEUR not in sys.path:
    sys.path.insert(0, _HEUR)

import lgbm_to_c  # noqa: E402


_CC = shutil.which("cc") or shutil.which("gcc")
requires_cc = pytest.mark.skipif(_CC is None, reason="no C compiler")


def _train_tiny_booster(n_features=8, n_rows=400, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n_rows, n_features).astype(np.float64)
    # A nonlinear target so the trees actually branch.
    y = (X[:, 0] * 3.0 + np.sin(X[:, 1] * 6) + (X[:, 2] > 0.5) * 2.0
         + rng.rand(n_rows) * 0.1)
    ds = lgb.Dataset(X, label=y)
    params = {"objective": "regression", "num_leaves": 15, "min_data_in_leaf": 5,
              "verbose": -1}
    booster = lgb.train(params, ds, num_boost_round=40)
    return booster, X


def _compile_and_load(c_src, func_name):
    tmp = tempfile.mkdtemp(prefix="lgbmc_")
    c_path = os.path.join(tmp, "pred.c")
    so_path = os.path.join(tmp, "pred.so")
    with open(c_path, "w") as f:
        f.write(c_src)
    subprocess.run([_CC, "-O2", "-shared", "-fPIC", "-o", so_path, c_path],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, func_name)
    fn.restype = ctypes.c_double
    fn.argtypes = [ctypes.POINTER(ctypes.c_double)]
    return fn


@requires_cc
def test_generated_c_matches_booster():
    booster, X = _train_tiny_booster()
    n_features = X.shape[1]
    func = "rocke_test_score"
    c_src, h_src = lgbm_to_c.emit_c_predictor(
        booster, func, num_features=n_features, source_note="unit test")
    assert f"double {func}(" in c_src
    assert f"double {func}(" in h_src
    assert f"{func.upper()}_NUM_FEATURES {n_features}" in c_src

    fn = _compile_and_load(c_src, func)

    # Booster raw prediction (no transform) must equal the C sum.
    preds = booster.predict(X)
    for i in range(X.shape[0]):
        row = (ctypes.c_double * n_features)(*X[i].tolist())
        got = fn(row)
        assert abs(got - preds[i]) < 1e-9, (i, got, preds[i])


@requires_cc
def test_argmax_ranking_preserved():
    # The dispatcher only needs the argmax to match; verify ordering is identical
    # on a batch (this is what the tie-break relies on).
    booster, X = _train_tiny_booster(seed=3)
    n = X.shape[1]
    fn = _compile_and_load(
        lgbm_to_c.emit_c_predictor(booster, "rocke_test_score", num_features=n)[0],
        "rocke_test_score")
    c_scores = np.array([fn((ctypes.c_double * n)(*X[i].tolist())) for i in range(len(X))])
    assert int(np.argmax(c_scores)) == int(np.argmax(booster.predict(X)))


def test_rejects_non_regression():
    # A classifier dump must be refused, not silently miscompiled.
    dumped = {"objective": "binary sigmoid:1", "tree_info": []}
    with pytest.raises(lgbm_to_c.UnsupportedModelError):
        lgbm_to_c.booster_to_c(dumped, "f", num_features=4)


def test_rejects_categorical_split():
    dumped = {
        "objective": "regression",
        "tree_info": [{
            "tree_index": 0,
            "tree_structure": {
                "split_feature": 0, "decision_type": "==", "threshold": "1",
                "default_left": True,
                "left_child": {"leaf_value": 1.0},
                "right_child": {"leaf_value": 2.0},
            },
        }],
    }
    with pytest.raises(lgbm_to_c.UnsupportedModelError):
        lgbm_to_c.booster_to_c(dumped, "f", num_features=4)
