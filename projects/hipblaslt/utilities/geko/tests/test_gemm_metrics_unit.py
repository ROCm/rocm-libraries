# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pandas as pd

from geko import gemm_metrics as gm


def test_dtype_bytes_scalar_series_array() -> None:
    assert gm.dtype_bytes("f16_r") == 2.0
    assert np.isnan(gm.dtype_bytes("unknown"))

    s = pd.Series(["f32_r", "bad"])
    out_s = gm.dtype_bytes(s)
    assert out_s.iloc[0] == 4.0
    assert np.isnan(out_s.iloc[1])

    arr = gm.dtype_bytes(["f16_r", "bf16_r"])
    assert arr.tolist() == [2.0, 2.0]


def test_flop_bytes_ai() -> None:
    flops = gm.flop_count(16, 16, 16, 2)
    assert flops == 16384

    moved = gm.bytes_moved(16, 16, 16, 2, "f16_r", "f16_r", "f16_r")
    assert moved > 0

    ai = gm.arithmetic_intensity(16, 16, 16, 2, "f16_r", "f16_r", "f16_r")
    assert ai > 0


def test_categorize_scalar_branches() -> None:
    assert gm.categorize(16, 16, 16, 2) == "Small batch"
    assert gm.categorize(1, 32, 32, 1) == "M=1"
    assert gm.categorize(32, 1, 32, 1) == "N=1"
    assert gm.categorize(32, 32, 1, 1) == "K=1"
    assert gm.categorize(9000, 9000, 9000, 1) == "Very Large GEMMs"
    assert gm.categorize(9000, 9000, 128, 1) == "Large M and N"
    assert gm.categorize(9000, 64, 64, 1) == "Large M, very small N and K"
    assert gm.categorize(64, 9000, 64, 1) == "Large N, very small M and K"
    assert gm.categorize(64, 64, 9000, 1) == "Large K, very small M and N"
    assert gm.categorize(512, 512, 512, 1) == "Small GEMMs"
    assert gm.categorize(5000, 5000, 5000, 1) == "Large GEMMs"
    assert gm.categorize(2048, 1024, 1024, 1) == "Medium GEMMs"


def test_categorize_array_and_series() -> None:
    m = pd.Series([16, 9000])
    n = pd.Series([16, 64])
    k = pd.Series([16, 64])
    out = gm.categorize(m, n, k, batch=1)
    assert list(out) == ["Small GEMMs", "Large M, very small N and K"]

    arr = gm.categorize(np.array([16, 16]), np.array([16, 16]), np.array([16, 9000]), batch=[1, 1])
    assert arr.tolist() == ["Small GEMMs", "Large K, very small M and N"]


def test_categorize_2_scalar_and_vectorized() -> None:
    assert gm.categorize_2(64, 40000, 64, 1) == "1.1"
    assert gm.categorize_2(64, 40000, 65, 1) == "1.2"
    assert gm.categorize_2(512, 512, 4096, 1) == "2"
    assert gm.categorize_2(16, 16, 16, 2) == "3.1"
    assert gm.categorize_2(16, 16, 64, 1) == "3.2"
    assert gm.categorize_2(600, 600, 1024, 1) == "4"
    assert gm.categorize_2(300, 300, 32, 1) == "3.2"

    s = gm.categorize_2(pd.Series([16, 700]), pd.Series([16, 700]), pd.Series([16, 700]), batch=1)
    assert len(s) == 2
