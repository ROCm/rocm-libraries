################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Regression guard for grouped GEMM FLOPs reporting paths."""

import re
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_TIMER_PATH = _REPO_ROOT / "client" / "src" / "BenchmarkTimer.cpp"
_PROGRESS_LISTENER_PATH = _REPO_ROOT / "client" / "src" / "ProgressListener.cpp"

_GROUPED_POST_SOLUTION_FLOPS_RE = re.compile(
    r"if\(auto problem = dynamic_cast<ContractionProblemGroupedGemm\*>\(m_problem\)\)\s*\{"
    r".*?flopCount = 0\.0;"
    r".*?for\(auto& gemm : problem->gemms\)"
    r".*?flopCount \+= gemm\.flopCount\(\);",
    re.DOTALL,
)

_GROUPED_TOTAL_FLOPS_REPORT_RE = re.compile(
    r"if\(auto groupedProblem = dynamic_cast<const ContractionProblemGroupedGemm\*>\(problem\)\)\s*\{"
    r".*?writeReport\(groupedProblem->gemms\[0\]\);"
    r".*?double totalFlops = 0\.0;"
    r".*?for\(auto& it : groupedProblem->gemms\)"
    r".*?totalFlops \+= it\.flopCount\(\);"
    r".*?m_reporter->report\(ResultKey::TotalFlops, totalFlops\);",
    re.DOTALL,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_grouped_gemm_benchmark_timer_sums_all_subgemm_flops():
    src = _read(_BENCHMARK_TIMER_PATH)
    assert _GROUPED_POST_SOLUTION_FLOPS_RE.search(src), (
        "Expected grouped GEMM postSolution FLOPs to sum all sub-gemms."
    )


def test_grouped_gemm_progress_listener_reports_total_flops_sum():
    src = _read(_PROGRESS_LISTENER_PATH)
    assert _GROUPED_TOTAL_FLOPS_REPORT_RE.search(src), (
        "Expected grouped GEMM ProgressListener TotalFlops to use summed sub-gemm FLOPs."
    )
