#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Tests for evaluate.py.

Covers: shape family classification, K-depth regime classification,
and an end-to-end run of evaluate_model.

The evaluate_model test is deliberately routed through the public entry point
rather than through compute_tflops_efficiency directly. The defect it guards
was in the CALLER -- evaluate.py passed the pred_col value where the operation
is expected -- so a test of the helper in isolation passes while evaluate.py
raises ValueError on every invocation. Only exercising the caller catches it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import classify_shape_family, classify_k_regime, evaluate_model


class TestClassifyShapeFamily:
    def test_tiny_m(self):
        assert classify_shape_family(1, 4096, 4096) == "tiny_m"
        assert classify_shape_family(16, 1536, 7168) == "tiny_m"

    def test_small_m(self):
        assert classify_shape_family(32, 1536, 7168) == "small_m"
        assert classify_shape_family(128, 4096, 4096) == "small_m"

    def test_medium_m(self):
        assert classify_shape_family(256, 1024, 1024) == "medium_m"
        assert classify_shape_family(2048, 2048, 2048) == "medium_m"

    def test_large_m(self):
        assert classify_shape_family(4096, 4096, 4096) == "large_m"
        assert classify_shape_family(20480, 7168, 256) == "large_m"


class TestClassifyKRegime:
    def test_shallow(self):
        assert classify_k_regime(256) == "shallow_k"
        assert classify_k_regime(32) == "shallow_k"

    def test_medium(self):
        assert classify_k_regime(1024) == "medium_k"
        assert classify_k_regime(2048) == "medium_k"

    def test_deep(self):
        assert classify_k_regime(4096) == "deep_k"
        assert classify_k_regime(7168) == "deep_k"


class _StubModel:
    """Returns the prediction column planted in the frame, in row order."""

    def __init__(self, preds):
        self._preds = np.asarray(preds, dtype=float)

    def predict(self, X):
        assert len(X) == len(self._preds)
        return self._preds


class _StubPredictor:
    def __init__(self, preds, log_targets=()):
        self._model = _StubModel(preds)
        self._log_targets = log_targets

    def _load_model(self, target):
        assert target == "tflops"
        return self._model


class _StubFeatureEngine:
    """evaluate_model only needs one row of features per input row."""

    def extract_batch(self, df):
        return np.zeros((len(df), 1), dtype=float)


def _frame():
    """Two shapes, two candidate kernels each.

    Shape A: the model ranks the oracle first  -> efficiency 1.0
    Shape B: the model ranks the oracle second -> efficiency 0.25
    """
    return pd.DataFrame(
        {
            "m": [128, 128, 256, 256],
            "n": [128, 128, 256, 256],
            "k": [128, 128, 256, 256],
            "measured_tflops": [100.0, 50.0, 100.0, 25.0],
            "is_valid": [True, True, True, True],
            "pipeline": ["compv3", "compv3", "compv3", "compv3"],
        }
    )


class TestEvaluateModel:
    def test_runs_end_to_end(self):
        """Regression: evaluate.py used to raise ValueError on every call.

        compute_tflops_efficiency takes (df, operation, pred_col) and evaluate.py
        passed "pred_tflops" positionally as `operation`, which falls through to
        the unknown-operation branch. Both call sites were affected, so the
        failure was total rather than partial -- the tool could not run at all.
        """
        df = _frame()
        # predictions rank the oracle first on shape A, second on shape B
        preds = [100.0, 50.0, 25.0, 100.0]
        res = evaluate_model(_StubPredictor(preds), df, _StubFeatureEngine())

        gm = res["global_metrics"]
        assert gm["num_shapes"] == 2
        assert gm["num_valid_rows"] == 4
        # shape A efficiency 1.0, shape B efficiency 25/100
        assert gm["efficiency_mean"] == pytest.approx(0.625)
        assert gm["ndcg_at_1"] == pytest.approx(0.5)
        assert len(res["per_shape_efficiency"]) == 2

    def test_slice_metrics_populated(self):
        """The second call site lives in _slice_efficiency, reached only through
        the per-slice groupbys. Assert those produced numbers rather than the
        empty {"count": 0} that a silent failure would leave behind."""
        res = evaluate_model(
            _StubPredictor([100.0, 50.0, 25.0, 100.0]), _frame(), _StubFeatureEngine()
        )
        for key in ("shape_family_metrics", "k_regime_metrics", "pipeline_metrics"):
            assert res[key], f"{key} is empty"
            for name, metrics in res[key].items():
                assert metrics["count"] > 0, f"{key}[{name}] has no rows"
                assert 0.0 < metrics["mean"] <= 1.0


class TestLogTransform:
    """log1p(tflops) is the default training target, so the expm1 branch in
    evaluate_model is the production path. Covering only the pass-through
    branch leaves the one that actually runs untested."""

    #: _frame()'s measured_tflops, in row order.
    MEASURED = [100.0, 50.0, 100.0, 25.0]

    def test_log_space_predictions_are_inverted(self):
        """A perfect predictor in log space must score as a perfect predictor:
        r2 == 1 and efficiency == 1 only if expm1 was applied."""
        res = evaluate_model(
            _StubPredictor(np.log1p(self.MEASURED), log_targets=("tflops",)),
            _frame(),
            _StubFeatureEngine(),
        )
        gm = res["global_metrics"]
        assert gm["r2"] == pytest.approx(1.0, abs=1e-9)
        assert gm["efficiency_mean"] == pytest.approx(1.0)

    def test_without_the_inverse_transform_the_fit_collapses(self):
        """The same log-space values down the non-log path must NOT look like a
        good fit -- otherwise the test above would pass with the branch removed."""
        res = evaluate_model(
            _StubPredictor(np.log1p(self.MEASURED)),  # log_targets=() by default
            _frame(),
            _StubFeatureEngine(),
        )
        assert res["global_metrics"]["r2"] < 0.5


class TestArgParser:
    def test_the_supported_operation_parses(self):
        from evaluate import _OPERATION, build_arg_parser

        args = build_arg_parser().parse_args(
            ["--model_dir", "m", "--data_dir", "d", "--op", _OPERATION]
        )
        assert args.op == _OPERATION

    def test_it_defaults_to_the_supported_operation(self):
        from evaluate import _OPERATION, build_arg_parser

        parsed = build_arg_parser().parse_args(["--model_dir", "m", "--data_dir", "d"])
        assert parsed.op == _OPERATION

    @pytest.mark.parametrize("op", ["grouped_conv", "fmha", "gemm_universal_vec"])
    def test_an_unsupported_operation_is_rejected(self, op):
        from evaluate import build_arg_parser

        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(
                ["--model_dir", "m", "--data_dir", "d", "--op", op]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
