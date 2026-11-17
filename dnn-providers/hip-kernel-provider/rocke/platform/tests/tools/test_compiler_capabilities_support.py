# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused process and scheduling tests for COMGR capability probing."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

_SOURCE_TOOLS = Path(__file__).resolve().parents[2] / "tools"
_INSTALLED_TOOLS = Path(__file__).resolve().parent
_TOOLS = (
    _SOURCE_TOOLS
    if (_SOURCE_TOOLS / "compiler_capabilities").is_dir()
    else _INSTALLED_TOOLS
)
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from compiler_capabilities import _comgr_worker as worker
from compiler_capabilities import _probe_support as support


def _completed(document: dict[str, object]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        ["python"], 0, json.dumps(document).encode(), b""
    )


def test_raw_worker_bypasses_only_the_legacy_profile_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def build(ir_texts: tuple[str, ...], **kwargs: object):
        seen["ir_texts"] = ir_texts
        seen.update(kwargs)
        timings = types.SimpleNamespace(bc=1.0, relocatable=2.0, executable=3.0)
        return b"hsaco", timings

    fake = types.SimpleNamespace(ComgrError=RuntimeError, _build_hsaco=build)
    import rocke.runtime

    monkeypatch.setitem(sys.modules, "rocke.runtime.comgr", fake)
    monkeypatch.setattr(rocke.runtime, "comgr", fake, raising=False)

    result = worker._probe({"isa": "isa", "options": ["-O3"]}, "module")

    assert result["outcome"] == "accepted"
    assert seen == {
        "ir_texts": ("module",),
        "isa": "isa",
        "options": ["-O3"],
        "validate_profile": False,
    }


def test_worker_result_preserves_full_pipeline_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {
                "outcome": "accepted",
                "evidence": {
                    "hsaco_sha256": "a" * 64,
                    "hsaco_size": 32,
                    "completed_stages": [
                        "compile_source_to_bc",
                        "codegen_bc_to_relocatable",
                        "link_relocatable_to_executable",
                    ],
                },
            }
        ),
    )
    request = support.ProbeRequest("one", "module", "isa")

    result = support._probe_one(request, comgr_lib=None, timeout=10)

    assert result.outcome == "accepted"
    assert result.evidence["hsaco_size"] == 32


def test_worker_normal_rejection_is_distinct_from_process_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = support.ProbeRequest("one", "module", "isa")
    monkeypatch.setattr(
        support.subprocess,
        "run",
        lambda *args, **kwargs: _completed(
            {
                "outcome": "rejected",
                "failed_stage": "codegen_bc_to_relocatable",
                "diagnostic": "COMGR status=1",
            }
        ),
    )
    rejected = support._probe_one(request, comgr_lib=None, timeout=10)
    assert rejected == support.ProbeResult(
        "one",
        "rejected",
        "COMGR status=1",
        failed_stage="codegen_bc_to_relocatable",
    )

    monkeypatch.setattr(
        support.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            ["python"], -11, b"", b"LLVM ERROR: synthetic crash"
        ),
    )
    crashed = support._probe_one(request, comgr_lib=None, timeout=10)
    assert crashed.outcome == "crash"
    assert "LLVM ERROR" in crashed.diagnostic


def test_worker_timeout_is_bounded_and_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["python"], 5)

    monkeypatch.setattr(support.subprocess, "run", timeout)
    result = support._probe_one(
        support.ProbeRequest("one", "module", "isa"),
        comgr_lib=None,
        timeout=5,
    )
    assert result.outcome == "timeout"
    assert "timed out" in result.diagnostic


def test_probe_all_runs_every_unique_request_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    def run(
        request: support.ProbeRequest, *, comgr_lib: Path | None, timeout: int
    ) -> support.ProbeResult:
        del comgr_lib, timeout
        seen.append(request.key)
        return support.ProbeResult(request.key, "rejected", request.key)

    monkeypatch.setattr(support, "_probe_one", run)
    requests = [
        support.ProbeRequest("a", "a", "isa"),
        support.ProbeRequest("b", "b", "isa"),
        support.ProbeRequest("c", "c", "isa"),
    ]
    results = support.probe_all(requests, comgr_lib=None, jobs=1)
    assert list(results) == ["a", "b", "c"]
    assert seen == ["a", "b", "c"]


def test_default_jobs_uses_available_cpus_without_exceeding_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support.os, "sched_getaffinity", lambda _: set(range(6)), raising=False
    )
    assert support.default_jobs(2) == 2
    assert support.default_jobs(20) == 6
    assert support.default_jobs(0) == 1


def test_identity_hashes_the_library_that_the_loader_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libamd_comgr.so"
    library.write_bytes(b"comgr")
    monkeypatch.setattr(
        support,
        "_run_worker",
        lambda *args, **kwargs: {
            "llvm_version": [23, 0, 0],
            "version_source": "LLVMGetVersion",
            "requested_library": "alias",
            "loaded_library": str(library),
            "query_library": str(library),
        },
    )

    identity = support.identify_comgr(None)

    assert identity["library"] == str(library.resolve())
    assert identity["sha256"] == support._sha256(library)
    assert "loaded_library" not in identity
    assert "requested_library" not in identity


def test_identity_refuses_unhashable_loader_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        support,
        "_run_worker",
        lambda *args, **kwargs: {
            "llvm_version": [23, 0, 0],
            "version_source": "test",
            "requested_library": "libamd_comgr.so",
            "loaded_library": None,
            "query_library": None,
        },
    )
    with pytest.raises(support.ProbeError, match="not a file"):
        support.identify_comgr(None)


def test_worker_environment_pins_comgr_and_disables_cache(tmp_path: Path) -> None:
    library = tmp_path / "libamd_comgr.so"
    env = support._worker_environment(library)
    assert env["ROCKE_COMGR_LIB"] == str(library)
    assert env["AMD_COMGR_CACHE"] == "0"
    assert "PYTHONPATH" in env
