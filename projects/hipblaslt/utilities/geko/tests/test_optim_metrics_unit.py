# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from geko.optim import _metrics as metrics


def _gemm_row(**overrides):
    row = {
        "transA": "N",
        "transB": "N",
        "batch_count": 1,
        "m": 16,
        "n": 16,
        "k": 16,
        "a_type": "f16_r",
        "b_type": "f16_r",
        "c_type": "f16_r",
        "d_type": "f16_r",
        "compute_type": "f32_r",
    }
    row.update(overrides)
    return row


def test_load_and_normalize_weights_missing_file_returns_none(tmp_path: Path) -> None:
    out = metrics._load_and_normalize_weights(tmp_path / "missing.csv")
    assert out is None


def test_load_and_normalize_weights_missing_required_columns_raises(tmp_path: Path) -> None:
    p = tmp_path / "summary.csv"
    pd.DataFrame([{"transA": "N", "% of total": 100.0}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing GEMM keys"):
        metrics._load_and_normalize_weights(p)


def test_load_and_normalize_weights_requires_call_count_or_pct(tmp_path: Path) -> None:
    p = tmp_path / "summary.csv"
    pd.DataFrame([_gemm_row()]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing both '% of total' and 'call_count'"):
        metrics._load_and_normalize_weights(p)


def test_load_and_normalize_weights_infers_pct_from_us_reference(tmp_path: Path) -> None:
    p = tmp_path / "summary.csv"
    pd.DataFrame([
        _gemm_row(call_count=2),
        _gemm_row(m=32, call_count=3),
    ]).to_csv(p, index=False)

    df_ref = pd.DataFrame([
        _gemm_row(us_reference=10.0),
        _gemm_row(m=32, us_reference=20.0),
    ])
    out = metrics._load_and_normalize_weights(p, df_for_reference_us=df_ref)
    assert len(out) == 2
    assert pytest.approx(out["pct_of_total"].sum(), abs=1e-9) == 100.0


def test_enrich_adds_shapes_and_weighted_columns(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    pd.DataFrame([_gemm_row(call_count=5, us=2.0)]).to_csv(summary, index=False)

    df = pd.DataFrame(
        [
            _gemm_row(
                ratio=1.25,
                us_reference=20.0,
                us_tuned=16.0,
                kernel_reference="A_MT16x16x4_MI16x16x1",
                kernel_tuned="B_MT32x32x4_MI16x16x1",
                solutionIdx_reference=1,
                solutionIdx_tuned=2,
            )
        ]
    )
    out = metrics.enrich(df, summary_csv=summary)
    assert out.iloc[0]["uplift_pct"] == pytest.approx(25.0)
    assert out.iloc[0]["us_saved"] == pytest.approx(4.0)
    assert bool(out.iloc[0]["kernel_changed"]) is True
    assert bool(out.iloc[0]["solution_changed"]) is True
    assert out.iloc[0]["MT_tuned"] == "32x32x4"
    assert out.iloc[0]["MI_reference"] == "16x16x1"
    assert out.iloc[0]["call_count"] == 5
    assert out.iloc[0]["weighted_us_saved"] == pytest.approx(20.0)


def test_summarize_verify_false_and_e2e_from_prefilled_weights() -> None:
    raw = pd.DataFrame(
        [
            _gemm_row(ratio=1.10, us_reference=10.0, us_tuned=9.0, kernel_changed=True),
            _gemm_row(m=32, ratio=0.90, us_reference=20.0, us_tuned=22.0, kernel_changed=False),
            _gemm_row(m=64, ratio=1.0, us_reference=30.0, us_tuned=30.0, kernel_changed=False),
        ]
    )
    final = pd.DataFrame(
        [
            _gemm_row(
                ratio=1.10,
                us_reference=10.0,
                us_tuned=9.0,
                kernel_tuned="K_MT16x16x4_MI16x16x1",
                a_type="f16_r",
                transA="N",
                transB="N",
                call_count=10,
                pct_of_total=60.0,
            )
        ]
    )
    report = metrics.summarize(raw, final, summary_csv="dummy.csv", verify=False)
    assert report["counts"]["n_attempted"] == 3
    assert "n_valid" not in report["counts"]
    assert report["counts"]["n_improved"] == 1
    assert report["counts"]["n_regressed"] == 1
    assert report["counts"]["n_neutral"] == 1
    assert report["e2e"]["coverage_pct"] == pytest.approx(60.0)
    assert "f16_r_NN" in report["per_dtype_layout"]


def test_summarize_verify_true_uses_error_tuned_mask() -> None:
    raw = pd.DataFrame(
        [
            _gemm_row(ratio=2.0, error_tuned=0.5),
            _gemm_row(m=32, ratio=0.5, error_tuned=0.001),
        ]
    )
    final = raw.iloc[[1]].copy()
    report = metrics.summarize(raw, final, verify=True, error_thr=0.01)
    assert report["counts"]["n_valid"] == 1
    assert report["counts"]["n_invalid"] == 1
    assert report["counts"]["n_regressed"] == 1


def test_write_metrics_json_converts_numpy_and_inf(tmp_path: Path) -> None:
    out_file = tmp_path / "metrics.json"
    payload = {
        "i": np.int64(7),
        "f": np.float64(1.5),
        "bad": float("inf"),
        "nested": [np.float64(2.5), float("-inf")],
    }
    metrics.write_metrics_json(payload, out_file)

    loaded = json.loads(out_file.read_text())
    assert loaded["i"] == 7
    assert loaded["f"] == 1.5
    assert loaded["bad"] is None
    assert loaded["nested"][1] is None
