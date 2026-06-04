# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""CI benchmark smoke gate.

Runs ``python -m dnn_benchmarking`` over the committed sample graphs and asserts
the tool-health gate (``ci/check_results.py``): at least one graph/engine
combination ran successfully and none errored. Living in the suite keeps the CI
workflow a plain pytest invocation instead of hardcoding CLI args + a separate
gate script.

Like the other GPU integration tests this skips when no GPU/plugin is present,
so a local ``pytest -m "not gpu"`` never needs hardware.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.amd]

_TOOL_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "ci_check_results", _TOOL_ROOT / "ci" / "check_results.py"
)
check_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_results)


def _skip_if_no_rocm() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if not torch.cuda.is_available():
        pytest.skip("PyTorch GPU not available")

    if torch.version.hip is None:
        pytest.skip("CUDA build detected; skipping AMD-only test")

    try:
        import hipdnn_frontend as hipdnn

        hipdnn.Handle()
    except Exception as e:
        pytest.skip(f"hipdnn_frontend not available or no GPU: {e}")


def test_benchmark_smoke_gate(tmp_path, plugin_path):
    _skip_if_no_rocm()

    results = tmp_path / "results.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnn_benchmarking",
            "--graph",
            "graphs/*.json",
            "--warmup",
            "1",
            "--iters",
            "1",
            "--plugin-path",
            plugin_path,
            "--output",
            str(results),
        ],
        cwd=_TOOL_ROOT,
        capture_output=True,
        text=True,
    )

    ok, message = check_results.evaluate(results)
    assert ok, f"{message}\n--- benchmark stderr ---\n{proc.stderr}"
