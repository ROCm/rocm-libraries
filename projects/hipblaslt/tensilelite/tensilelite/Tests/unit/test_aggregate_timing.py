# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for scripts/aggregate_timing.py
"""

import gzip
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(scripts_dir))

import aggregate_timing  # noqa: E402
from analyze_timing import analyze_timing_file, print_summary  # noqa: E402

RUN_A = """\
TIMING:library_loading:10.0
TIMING:hip_initialization:100.0
TIMING_CONTEXT:M=64,N=64,K=64,batch=1,typeA=f32,typeD=f32
TIMING:cpu_reference_gemm:20.0
TIMING:pre_problem:25.0
TIMING:python_client_execution:200.0
TIMING:python_benchmark_problems:210.0
"""

RUN_B = """\
TIMING:library_loading:30.0
TIMING:hip_initialization:300.0
TIMING_CONTEXT:M=128,N=128,K=128,batch=1,typeA=f32,typeD=f32
TIMING:cpu_reference_gemm:40.0
TIMING:cpu_reference_gemm:50.0
TIMING:pre_problem:95.0
TIMING:python_client_execution:600.0
TIMING:python_benchmark_problems:630.0
TIMING:some_new_category:7.0
"""


def _report(timings, problems, capsys):
    print_summary(timings, problems)
    return capsys.readouterr().out


def _aggregate_report(paths, capsys):
    total = aggregate_timing.empty_tree()
    problems = []
    for path in paths:
        timings, p = analyze_timing_file(str(path))
        aggregate_timing.add_trees(total, aggregate_timing.timings_to_tree(timings))
        problems.extend(p)
    return _report(aggregate_timing.tree_to_timings(total), problems, capsys)


def test_aggregate_equals_report_of_concatenated_logs(tmp_path, capsys):
    a = tmp_path / "a.log"
    b = tmp_path / "b.log.gz"
    a.write_text(RUN_A)
    with gzip.open(b, "wt") as f:
        f.write(RUN_B)
    both = tmp_path / "both.log"
    both.write_text(RUN_A + RUN_B)

    timings, problems = analyze_timing_file(str(both))
    expected = _report(timings, problems, capsys)

    assert _aggregate_report([a, b], capsys) == expected


def test_single_run_aggregate_equals_regular_report(tmp_path, capsys):
    a = tmp_path / "a.log"
    a.write_text(RUN_A)
    timings, problems = analyze_timing_file(str(a))

    assert _aggregate_report([a], capsys) == _report(timings, problems, capsys)


def test_add_trees_sums_count_and_total_per_node(tmp_path):
    a = tmp_path / "a.log"
    b = tmp_path / "b.log"
    a.write_text(RUN_A)
    b.write_text(RUN_B)
    total = aggregate_timing.empty_tree()
    for path in (a, b):
        timings, _ = analyze_timing_file(str(path))
        aggregate_timing.add_trees(total, aggregate_timing.timings_to_tree(timings))

    top = total["phases"]["python_benchmark_problems"]
    assert top["count"] == 2 and top["total_ms"] == 840.0
    client = top["children"]["python_client_execution"]
    assert client["count"] == 2 and client["total_ms"] == 800.0
    ref = client["children"]["pre_problem"]["children"]["cpu_reference_gemm"]
    assert ref["count"] == 3 and ref["total_ms"] == 110.0
    assert total["unknown_categories"] == {"some_new_category": {"count": 1, "total_ms": 7.0}}
