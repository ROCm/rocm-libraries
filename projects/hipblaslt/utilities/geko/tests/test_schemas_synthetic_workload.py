################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Tensile and hipBLASLt workload rows; compatibility with bench.log.parse."""

from pathlib import Path

import pytest
import yaml

from geko.bench.log import parse
from geko.constants import GEMM_LOG_FIELDS
from geko.schemas import GemmConfig, GemmType


@pytest.mark.parametrize(
    "dt,dd,cd",
    [
        ("B", "B", "S"),
        ("H", "H", "S"),
        ("X", "S", "S"),
    ],
)
def test_tensile_roundtrip_through_hipblaslt(dt, dd, cd):
    a_t, b_t, c_t, comp = GemmType._tensile_triple_to_hipblaslt(dt, dd, cd)
    gt = GemmType.from_hipblaslt("N", "T", a_t, b_t, c_t, comp)
    assert (gt.data_type, gt.dest_data_type, gt.compute_data_type) == (dt, dd, cd)


def test_tensile_triple_mixed_bh():
    a_t, b_t, c_t, comp = GemmType._tensile_triple_to_hipblaslt("BH", "B", "S")
    assert a_t == "bf16_r" and b_t == "f16_r" and c_t == "bf16_r" and comp == "f32_r"


def test_workload_log_rows_keys_and_sample_values():
    gt = GemmType.from_tensile("N", "T", "B", "B", "S")
    row = GemmConfig(gt, [[1024, 1024, 1, 1024]]).workload_log_rows()[0]
    assert set(row) == set(GEMM_LOG_FIELDS)
    assert row["M"] == 1024 and row["transB"] == "T"


def test_single_gemm_workload_parseable(tmp_path: Path):
    gt = GemmType.from_tensile("N", "T", "B", "B", "S")
    rows = GemmConfig(gt, [[128, 256, 2, 512]]).workload_log_rows()
    assert len(rows) == 1
    ypath = tmp_path / "w.yaml"
    with ypath.open("w") as f:
        yaml.safe_dump(rows, f, default_flow_style=None, sort_keys=False, width=5000)
    df = parse(ypath, as_df=True)
    assert len(df) == 1
    assert df["M"].iloc[0] == 128 and df["batch_count"].iloc[0] == 2


def test_workload_log_rows_tensile_only():
    gt = GemmType.from_tensile("N", "N", "B", "B", "S")
    rows = GemmConfig(gt, [[64, 64, 1, 64]]).workload_log_rows()
    assert len(rows) == 1
    assert set(rows[0]) == set(GEMM_LOG_FIELDS)


def test_workload_log_rows_with_logical():
    gt = GemmType.from_hipblaslt("T", "N", "bf16_r", "bf16_r", "bf16_r", "f32_r")
    rows = GemmConfig(gt, [[32, 32, 1, 32]]).workload_log_rows()
    assert rows[0]["a_type"] == "bf16_r" and rows[0]["transA"] == "T"


def test_workload_log_rows_multi_sizes():
    gt = GemmType.from_tensile("N", "N", "B", "B", "S")
    rows = GemmConfig(gt, [[64, 64, 1, 64], [128, 128, 1, 128]]).workload_log_rows()
    assert len(rows) == 2
    assert rows[0]["M"] == 64 and rows[1]["M"] == 128


def test_workload_log_rows_concat_multiple_configs():
    g1 = GemmConfig(GemmType.from_tensile("N", "N", "B", "B", "S"), [[8, 8, 1, 8]])
    g2 = GemmConfig(GemmType.from_tensile("N", "T", "H", "H", "S"), [[16, 16, 1, 16]])
    rows: list[dict] = []
    for gc in (g1, g2):
        rows.extend(gc.workload_log_rows())
    assert len(rows) == 2
    assert rows[0]["a_type"] == "bf16_r" and rows[1]["a_type"] == "f16_r"
