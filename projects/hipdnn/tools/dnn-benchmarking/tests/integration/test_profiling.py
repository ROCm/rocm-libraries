# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Integration tests for opt-in profiling sources.

Each non-strict test is double-gated: a pytest marker (``rocprofv3`` /
``perf`` / ``rocprof_compute``) for selection, plus an inline
binary/host probe that skips when the precondition isn't met. Strict
payload tests are additionally gated by the pytest ``--profiling-strict``
option so default ``pytest`` runs do not fail on profiler/runtime stacks
that can launch but cannot produce full artifacts.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from dnn_benchmarking.metrics._tool_resolver import resolve_rocm_tool


def _graphs_dir() -> Path:
    return Path(__file__).parent.parent.parent / "graphs"


def _require_gpu():
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch GPU not available")
    except ImportError as e:
        pytest.skip(f"PyTorch not available: {e}")


def _require_rocm_tool(name: str) -> str:
    """ROCm tool gate — mirrors production resolution.

    Production paths (``rocprof_pmc``, ``rocprof_trace``, ``roofline``)
    resolve ROCm tools via ``resolve_rocm_tool``, which prefers
    ``$ROCM_PATH/bin``. A bare ``shutil.which`` skip-gate would silently
    skip on hosts where ``/opt/rocm/bin`` isn't on PATH (Alola login
    nodes, sandboxed containers) even though production would run.
    """
    binary = resolve_rocm_tool(name)
    if binary is None:
        pytest.skip(f"{name} not found at $ROCM_PATH/bin or on PATH")
    return binary


def _require_binary(name: str) -> str:
    """Non-ROCm binary gate (perf, etc.) — PATH-only."""
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
def test_pmc_basic_populates_counters(tmp_path):
    _require_gpu()
    _require_rocm_tool("rocprofv3")
    data = _run_dnn_bench(["--pmc", "basic"], tmp_path)
    extra = _first_pe_extra(data)
    assert "pmc" in extra
    pmc = extra["pmc"]
    # Real counters, recorded failure tail, or an rc==0 warning (e.g.
    # the rocpd parser found no .db) — any of these means the slice
    # made the round trip; only a wholly-empty pmc dict is a regression.
    assert any(k in pmc for k in ("counters", "error_tail", "skipped", "warnings"))


@pytest.mark.rocprofv3
def test_emit_trace_pftrace_records_artifact(tmp_path):
    _require_gpu()
    _require_rocm_tool("rocprofv3")
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


@pytest.mark.rocprof_compute
def test_roofline_records_csv_artifacts(tmp_path):
    _require_gpu()
    _require_rocm_tool("rocprof-compute")
    data = _run_dnn_bench(["--roofline"], tmp_path)
    extra = _first_pe_extra(data)
    assert "roofline" in extra
    roof = extra["roofline"]
    if "roofline_csv" in roof:
        assert Path(roof["roofline_csv"]).exists()
        assert Path(roof["workload_path"]).is_dir()


@pytest.mark.rocprofv3
@pytest.mark.perf
@pytest.mark.rocprof_compute
def test_combined_pmc_perf_roofline_merge_into_one_extra_metrics(tmp_path):
    """All three opt-in sources requested simultaneously must each
    populate their own top-level key in `extra_metrics`. Regression
    guard for two failure modes:

    * The duplicate-orchestrator-call bug fixed in commit 196a0fb33ca —
      that bug ran each replay twice but the second overwrote the first
      via `result.extra_metrics = extra`. With three slices, an
      overwrite would have dropped two of them.
    * Future merge logic in `run_profiling_passes` that loses one source
      because of dict-update collisions.
    """
    _require_gpu()
    _require_rocm_tool("rocprofv3")
    _require_binary("perf")
    _require_rocm_tool("rocprof-compute")
    _require_perf_paranoid_low()

    data = _run_dnn_bench(["--pmc", "basic", "--perf", "--roofline"], tmp_path)
    extra = _first_pe_extra(data)
    for key in ("pmc", "perf", "roofline"):
        assert key in extra, f"{key} missing from extra_metrics: {sorted(extra)}"

    # At least one of (real-payload | tool-error) must be populated for
    # each slice — a slice that's wholly empty would indicate the
    # orchestrator skipped it without warning.
    assert any(
        k in extra["pmc"] for k in ("counters", "error_tail", "skipped", "warnings")
    )
    assert any(k in extra["perf"] for k in ("cycles_user", "error_tail", "skipped"))
    assert any(
        k in extra["roofline"]
        for k in ("roofline_csv", "error_tail", "skipped", "warnings")
    )


# --------------------------------------------------------------------------
# Strict tier — same workloads, harder assertions.
#
# The smoke tests above accept ``skipped`` / ``error_tail`` / ``warnings``
# as evidence the slice round-tripped, which keeps them green on hosts
# where the tool is installed but the workload (e.g. counter set, hw
# paranoid, replay budget) doesn't fully cooperate. That tolerance also
# means a regression that silently drops the real payload would slip
# through. The strict tier closes that gap: opt in with
# ``--profiling-strict -m profiling_strict`` on a known-good GPU host
# before merging changes to the profiling stack.
# --------------------------------------------------------------------------


@pytest.mark.rocprofv3
@pytest.mark.profiling_strict
def test_pmc_basic_strict_requires_db_and_counters(tmp_path):
    """Asserts the full PMC contract: rocpd db on disk + parsed counters
    in the JSON. A regression that produces no db (or fails to parse)
    surfaces here even though the smoke test would still pass on the
    ``warnings`` branch."""
    _require_gpu()
    _require_rocm_tool("rocprofv3")
    data = _run_dnn_bench(["--pmc", "basic"], tmp_path)
    extra = _first_pe_extra(data)
    pmc = extra.get("pmc") or {}
    assert "db_path" in pmc, f"db_path missing — slice keys: {sorted(pmc)}"
    assert Path(pmc["db_path"]).exists(), f"rocpd db not on disk: {pmc['db_path']}"
    counters = pmc.get("counters") or {}
    assert counters, f"no parsed counters — slice keys: {sorted(pmc)}"
    # GRBM_GUI_ACTIVE is in every arch's basic set; if it isn't here,
    # the rocpd schema walk is broken (or the arch table changed
    # without updating this test).
    assert (
        "GRBM_GUI_ACTIVE" in counters
    ), f"GRBM_GUI_ACTIVE absent from parsed counters: {sorted(counters)}"


@pytest.mark.rocprof_compute
@pytest.mark.profiling_strict
def test_roofline_strict_requires_csv_and_workload(tmp_path):
    """Asserts the roofline contract: roofline.csv emitted and the
    workload directory exists so ``rocprof-compute analyze --path`` can
    render the actual roofline post-hoc."""
    _require_gpu()
    _require_rocm_tool("rocprof-compute")
    data = _run_dnn_bench(["--roofline"], tmp_path)
    extra = _first_pe_extra(data)
    roof = extra.get("roofline") or {}
    assert "roofline_csv" in roof, f"roofline_csv missing — keys: {sorted(roof)}"
    assert Path(roof["roofline_csv"]).exists()
    assert "workload_path" in roof
    assert Path(roof["workload_path"]).is_dir()


@pytest.mark.rocprofv3
@pytest.mark.perf
@pytest.mark.rocprof_compute
@pytest.mark.profiling_strict
def test_combined_strict_includes_trace_and_real_payloads(tmp_path):
    """Combined four-source variant of the lenient combined test above.

    Adds ``--emit-trace pftrace`` to the four-source set (the lenient
    version only tests PMC+perf+roofline) and asserts each slice carries
    a real artifact, not a tool-error sentinel. This is the closest the
    test suite gets to a full integration cover.
    """
    _require_gpu()
    _require_rocm_tool("rocprofv3")
    _require_binary("perf")
    _require_rocm_tool("rocprof-compute")
    _require_perf_paranoid_low()

    data = _run_dnn_bench(
        ["--pmc", "basic", "--emit-trace", "pftrace", "--perf", "--roofline"], tmp_path
    )
    extra = _first_pe_extra(data)

    pmc = extra.get("pmc") or {}
    assert pmc.get("counters"), f"pmc.counters empty/missing — keys: {sorted(pmc)}"

    trace = extra.get("trace") or {}
    assert trace.get("path"), f"trace.path missing — keys: {sorted(trace)}"
    assert Path(trace["path"]).exists()

    perf = extra.get("perf") or {}
    assert perf.get("cycles_user") is not None, f"perf.cycles_user missing — {perf}"
    assert perf.get("ipc_user") is not None

    roof = extra.get("roofline") or {}
    assert roof.get("roofline_csv"), f"roofline_csv missing — keys: {sorted(roof)}"
