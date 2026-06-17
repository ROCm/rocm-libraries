# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CI benchmark smoke gate.

Runs the committed sample graphs in-process and asserts the tool exercised
something: at least one graph/engine combination passed and none errored.
Skips when no GPU/plugin is present.
"""

from pathlib import Path

import pytest

from dnn_benchmarking.config.benchmark_config import SuiteConfig
from dnn_benchmarking.execution import suite_runner

pytestmark = [pytest.mark.gpu, pytest.mark.amd]

_TOOL_ROOT = Path(__file__).resolve().parents[2]


def _skip_if_no_rocm() -> None:
    try:
        import hipdnn_frontend as hipdnn
    except ImportError:
        pytest.fail(
            "hipdnn_frontend not installed — ensure the --tests slice was installed"
        )

    try:
        hipdnn.Handle()
    except Exception as e:
        pytest.skip(f"No AMD GPU available: {e}")


def test_benchmark_smoke_gate(plugin_path):
    _skip_if_no_rocm()

    graph_paths = sorted((_TOOL_ROOT / "graphs").glob("*.json"))
    assert graph_paths, "no sample graphs found under graphs/"

    config = SuiteConfig(warmup_iters=1, benchmark_iters=1, plugin_paths=[plugin_path])
    result = suite_runner.run(graph_paths, config)
    meta = result.metadata

    assert meta.error_combinations == 0, (
        f"benchmark smoke gate FAILED: error_combinations={meta.error_combinations} "
        "-- at least one graph/engine combination errored at runtime"
    )
    assert meta.pass_combinations > 0, (
        "benchmark smoke gate FAILED: pass_combinations=0 -- nothing ran "
        "successfully (all combinations skipped/unsupported; check the plugin "
        "path and that engines are installed)"
    )
