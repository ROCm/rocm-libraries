# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for opt-in profiling sources.

Each test is double-gated: a pytest marker (``rocprofv3`` / ``perf``)
for selection, plus an inline binary/host probe that skips when the
precondition isn't met. This keeps unmarked runs green on the dev
host (no perf, no GPU) while still allowing ``pytest -m rocprofv3
tests/integration/`` to fire on a real GPU host.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _graphs_dir() -> Path:
    return Path(__file__).parent.parent.parent / "graphs"


def _require_gpu():
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch GPU not available")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def _require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        pytest.skip(f"{name} not found on PATH")
    return binary


def _require_perf_paranoid_low():
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as fh:
            paranoid = int(fh.read().strip())
    except (OSError, ValueError):
        pytest.skip("could not read perf_event_paranoid")
    if paranoid > 1:
        pytest.skip(f"perf_event_paranoid={paranoid} > 1 (kernel events blocked)")


def _conv_graph() -> Path:
    p = _graphs_dir() / "sample_conv_fwd.json"
    if not p.exists():
        pytest.skip(f"sample graph not found: {p}")
    return p


def _run_dnn_bench(extra_args, tmp_path) -> dict:
    out_path = tmp_path / "results.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnn_benchmarking",
            "--graph",
            str(_conv_graph()),
            "--warmup",
            "2",
            "--iters",
            "5",
            "--profiling-output-dir",
            str(tmp_path / "prof"),
            "-o",
            str(out_path),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode not in (0, 2):
        pytest.fail(
            f"dnn-benchmark failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    assert out_path.exists()
    return json.loads(out_path.read_text())


def _first_pe_extra(data: dict) -> dict:
    for graph in data.get("graphs", []):
        for pe in graph.get("results", []):
            if pe.get("status") == "success":
                return pe.get("extra_metrics") or {}
    pytest.fail("no successful provider/engine result in output JSON")


@pytest.mark.rocprofv3
def test_emit_trace_pftrace_records_artifact(tmp_path):
    _require_gpu()
    _require_binary("rocprofv3")
    data = _run_dnn_bench(["--emit-trace", "pftrace"], tmp_path)
    extra = _first_pe_extra(data)
    assert "trace" in extra
    trace = extra["trace"]
    assert trace["format"] == "pftrace"
    if "path" in trace:
        assert Path(trace["path"]).exists()


@pytest.mark.perf
def test_perf_records_user_cycles(tmp_path):
    _require_gpu()
    _require_binary("perf")
    _require_perf_paranoid_low()
    data = _run_dnn_bench(["--perf"], tmp_path)
    extra = _first_pe_extra(data)
    assert "perf" in extra
    perf = extra["perf"]
    if "skipped" not in perf:
        assert perf.get("cycles_user") is not None
        assert perf.get("ipc_user") is not None
