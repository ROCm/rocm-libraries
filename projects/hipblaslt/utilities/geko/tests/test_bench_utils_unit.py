# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.bench import utils as butils


def _mock_bench_out() -> str:
    return """[0]:transA,transB,batch_count,m,n,k,a_type,b_type,c_type,d_type,compute_type,hipblaslt-Gflops,hipblaslt-GB/s,us
N,N,1,16,16,16,f16_r,f16_r,f16_r,f16_r,c_f32_r,1.0,1.0,1.0
--kernel name: k0
--Solution name: s0
--Solution index: 7
"""


def test_parse_benchmark_output_happy_path(tmp_path: Path) -> None:
    f = tmp_path / "a.out"
    f.write_text(_mock_bench_out())
    df = butils.parse_benchmark_output(f)
    assert len(df) == 1
    assert df.iloc[0]["kernel"] == "k0"
    assert str(df.iloc[0]["solutionIdx"]) == "7"


def test_parse_benchmark_output_invalid_format_raises(tmp_path: Path) -> None:
    f = tmp_path / "a.out"
    f.write_text("invalid\n")
    with pytest.raises(ValueError, match="correct format"):
        butils.parse_benchmark_output(f)


def test_parse_benchmark_output_dir_aggregates_and_skips_yaml(tmp_path: Path) -> None:
    (tmp_path / "one.out").write_text(_mock_bench_out())
    (tmp_path / "two.out").write_text(_mock_bench_out())
    yaml.safe_dump([{"dummy": 1}], (tmp_path / "skip.yaml").open("w"), sort_keys=False)

    df = butils.parse_benchmark_output_dir(tmp_path)
    assert len(df) == 2


def test_parse_benchmark_output_dir_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No latency logs"):
        butils.parse_benchmark_output_dir(tmp_path)


def test_as_dashboard_format_drops_and_renames_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "kernel": "k",
                "solutionIdx": 1,
                "lib": "x",
                "ratio": 1.1,
                "error_tuned": 0.0,
                "us_tuned": 1.0,
                "us_reference": 2.0,
            }
        ]
    )
    out = butils.as_dashboard_format(df)
    assert "solution_name" in out.columns
    assert "solution_index" in out.columns
    assert "us" in out.columns
    assert "us_reference" not in out.columns
    assert "lib" not in out.columns


def test_update_lib_source_handles_missing_match_table(tmp_path: Path) -> None:
    df = pd.DataFrame([{"solutionIdx": 0}])
    out = butils.update_lib_source(df, tmp_path / "missing.yaml")
    assert out.iloc[0]["lib_source"] == "NotFound"
