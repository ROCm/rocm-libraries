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
"""Guard common test YAMLs against accidental benchmark timing runs."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.unit

_TESTS_DIR = Path(__file__).resolve().parent.parent
_COMMON_DIR = _TESTS_DIR / "common"


@dataclass(frozen=True)
class SyncsPerBenchmarkOptOut:
    path: str
    reason: str

    def __post_init__(self):
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError(f"SyncsPerBenchmark opt-out for {self.path} requires a reason")


_SYNCSPERBENCHMARK_OPT_OUTS = {
    SyncsPerBenchmarkOptOut(
        "common/benchmark_runs/bf16_tn_gfx12.yaml",
        "benchmark timing coverage file under common/benchmark_runs",
    ),
    SyncsPerBenchmarkOptOut(
        "common/benchmark_runs/fp16_tn_gfx11.yaml",
        "benchmark timing coverage file under common/benchmark_runs",
    ),
    SyncsPerBenchmarkOptOut(
        "common/benchmark_runs/fp32_tn.yaml",
        "benchmark timing coverage file under common/benchmark_runs",
    ),
    SyncsPerBenchmarkOptOut(
        "common/client/rotate_mode0.yaml",
        "RotatingMode test requires the benchmark loop",
    ),
    SyncsPerBenchmarkOptOut(
        "common/client/rotate_mode0_gfx12.yaml",
        "RotatingMode test requires the benchmark loop",
    ),
    SyncsPerBenchmarkOptOut(
        "common/client/rotate_mode1.yaml",
        "RotatingMode test requires the benchmark loop",
    ),
    SyncsPerBenchmarkOptOut(
        "common/client/rotate_mode1_gfx12.yaml",
        "RotatingMode test requires the benchmark loop",
    ),
    SyncsPerBenchmarkOptOut(
        "common/gemm/fp8nfp16mix_hfp8ns.yaml",
        "LibraryLogic requires benchmark data to build the library table",
    ),
    SyncsPerBenchmarkOptOut(
        "common/gemm/icache_flush.yaml",
        "flush_icache is launched only inside the benchmark sync loop",
    ),
    SyncsPerBenchmarkOptOut(
        "common/gemm/xfp32.yaml",
        "LibraryLogic requires benchmark data to build the library table",
    ),
    SyncsPerBenchmarkOptOut(
        "common/groupedgemm/gfx11/grouped_gemm_gfx11.yaml",
        "LibraryLogic requires benchmark data to build the library table",
    ),
    SyncsPerBenchmarkOptOut(
        "common/groupedgemm/grouped_gemm.yaml",
        "LibraryLogic requires benchmark data to build the library table",
    ),
    SyncsPerBenchmarkOptOut(
        "common/groupedgemm/grouped_gemm_ck_gfx942.yaml",
        "LibraryLogic requires benchmark data to build the library table",
    ),
    SyncsPerBenchmarkOptOut(
        "common/sparse/gfx94x/libray_logic.yaml",
        "LibraryLogic requires benchmark data to build the library table",
    ),
}
_SYNCSPERBENCHMARK_OPT_OUT_PATHS = {
    opt_out.path for opt_out in _SYNCSPERBENCHMARK_OPT_OUTS
}


def _common_yaml_files():
    return sorted({
        path
        for pattern in ("*.yaml", "*.yml")
        for path in _COMMON_DIR.rglob(pattern)
    })


def _relative_to_tests(path):
    return path.relative_to(_TESTS_DIR).as_posix()


def _syncs_per_benchmark(path):
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return None

    global_parameters = data.get("GlobalParameters")
    if not isinstance(global_parameters, dict):
        return None

    return global_parameters.get("SyncsPerBenchmark")


def test_common_yaml_files_disable_benchmark_syncs():
    """Common YAMLs should opt out of benchmark timing unless explicitly listed."""
    stale_opt_outs = sorted(
        (
            opt_out
            for opt_out in _SYNCSPERBENCHMARK_OPT_OUTS
            if not (_TESTS_DIR / opt_out.path).is_file()
        ),
        key=lambda opt_out: opt_out.path,
    )
    failures = []

    for yaml_path in _common_yaml_files():
        rel_path = _relative_to_tests(yaml_path)
        if rel_path in _SYNCSPERBENCHMARK_OPT_OUT_PATHS:
            continue

        syncs_per_benchmark = _syncs_per_benchmark(yaml_path)
        if type(syncs_per_benchmark) is not int or syncs_per_benchmark != 0:
            failures.append(
                f"{rel_path} "
                f"(GlobalParameters.SyncsPerBenchmark={syncs_per_benchmark!r})"
            )

    if stale_opt_outs or failures:
        message_parts = []
        if stale_opt_outs:
            message_parts.append(
                "SyncsPerBenchmark opt-out entries do not exist:\n"
                + "\n".join(
                    f"  - {opt_out.path} ({opt_out.reason})"
                    for opt_out in stale_opt_outs
                )
            )
        if failures:
            message_parts.append(
                "Expected GlobalParameters.SyncsPerBenchmark == 0 for all "
                "common YAML files outside explicit opt-outs. Failing paths:\n"
                + "\n".join(f"  - {failure}" for failure in failures)
            )
        pytest.fail("\n\n".join(message_parts))
