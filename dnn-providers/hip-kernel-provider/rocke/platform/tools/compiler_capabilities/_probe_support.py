#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Process isolation, COMGR provenance, caching, and parallel probe support."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from rocke.tests.subprocess_launcher import rocke_python_command, rocke_subprocess_env

_TARGET_TRIPLE = "amdgcn-amd-amdhsa"
_DEFAULT_PROCESSOR = "gfx950"
_DEFAULT_TIMEOUT_SECONDS = 60
_WORKER = Path(__file__).with_name("_comgr_worker.py")
_FATAL_PROCESS_MARKERS = (
    "LLVM ERROR:",
    "PLEASE submit a bug report",
    "Stack dump:",
    "Segmentation fault",
    "UNREACHABLE executed",
    "Assertion failed",
)


class ProbeError(RuntimeError):
    """The selected COMGR library could not produce reliable capability evidence."""


@dataclass(frozen=True)
class ProbeRequest:
    """One independently attributable COMGR compilation."""

    key: str
    ir_text: str
    isa: str
    options: tuple[str, ...] = ("-O3",)


@dataclass(frozen=True)
class ProbeResult:
    """Normalized outcome of one isolated worker process."""

    key: str
    outcome: str
    diagnostic: str | None = None
    evidence: dict[str, object] | None = None
    failed_stage: str | None = None

    def as_report_entry(self) -> dict[str, object]:
        entry: dict[str, object] = {"outcome": self.outcome}
        if self.diagnostic:
            entry["diagnostic"] = self.diagnostic
        if self.evidence:
            entry["evidence"] = self.evidence
        if self.failed_stage:
            entry["failed_stage"] = self.failed_stage
        return entry


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(text)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_cache(path: Path, fingerprint: str) -> dict[str, object] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(document, dict):
        return None
    probe = document.get("probe")
    if not isinstance(probe, dict) or probe.get("fingerprint") != fingerprint:
        return None
    return document


def _worker_environment(comgr_lib: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    if comgr_lib is not None:
        env["ROCKE_COMGR_LIB"] = str(comgr_lib)
    env["AMD_COMGR_CACHE"] = "0"
    return rocke_subprocess_env(env)


def _first_diagnostic(stderr: str, stdout: str, returncode: int | None) -> str:
    for line in (stderr + "\n" + stdout).splitlines():
        if line.strip():
            return line.strip()[:1000]
    if returncode is None:
        return "COMGR probe worker timed out"
    return f"COMGR probe worker exited with status {returncode}"


def _worker_crashed(returncode: int, stderr: str, stdout: str) -> bool:
    if returncode < 0:
        return True
    output = stderr + "\n" + stdout
    return any(marker in output for marker in _FATAL_PROCESS_MARKERS)


def _run_worker(
    header: dict[str, object],
    body: str,
    *,
    comgr_lib: Path | None,
    timeout: int,
) -> dict[str, object]:
    payload = json.dumps(header, sort_keys=True).encode() + b"\n" + body.encode()
    try:
        completed = subprocess.run(
            rocke_python_command(_WORKER),
            input=payload,
            capture_output=True,
            check=False,
            timeout=timeout,
            env=_worker_environment(comgr_lib),
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", "replace")
        stderr = (exc.stderr or b"").decode("utf-8", "replace")
        raise ProbeError(_first_diagnostic(stderr, stdout, None)) from exc
    stdout = completed.stdout.decode("utf-8", "replace")
    stderr = completed.stderr.decode("utf-8", "replace")
    if completed.returncode != 0:
        kind = (
            "crashed"
            if _worker_crashed(completed.returncode, stderr, stdout)
            else "failed"
        )
        raise ProbeError(
            f"COMGR probe worker {kind}: "
            + _first_diagnostic(stderr, stdout, completed.returncode)
        )
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError(
            "COMGR probe worker returned malformed JSON: "
            + _first_diagnostic(stderr, stdout, completed.returncode)
        ) from exc
    if not isinstance(result, dict):
        raise ProbeError("COMGR probe worker result must be an object")
    return result


def identify_comgr(
    comgr_lib: Path | None, *, timeout: int = _DEFAULT_TIMEOUT_SECONDS
) -> dict[str, object]:
    """Load the selected COMGR in isolation and return hashable provenance."""
    identity = _run_worker(
        {"command": "identity"}, "", comgr_lib=comgr_lib, timeout=timeout
    )
    raw_path = identity.get("loaded_library") or identity.get("requested_library")
    if not isinstance(raw_path, str) or not raw_path:
        raise ProbeError("loaded COMGR did not report its library path")
    path = Path(raw_path)
    if not path.is_file():
        raise ProbeError(f"loaded COMGR path is not a file: {raw_path!r}")
    identity["library"] = str(path.resolve())
    identity["sha256"] = _sha256(path.resolve())
    identity.pop("loaded_library", None)
    identity.pop("requested_library", None)
    return identity


def _probe_one(
    request: ProbeRequest,
    *,
    comgr_lib: Path | None,
    timeout: int,
) -> ProbeResult:
    try:
        result = _run_worker(
            {
                "command": "probe",
                "isa": request.isa,
                "options": list(request.options),
            },
            request.ir_text,
            comgr_lib=comgr_lib,
            timeout=timeout,
        )
    except ProbeError as exc:
        text = str(exc)
        if "timed out" in text:
            outcome = "timeout"
        elif "worker crashed" in text:
            outcome = "crash"
        else:
            outcome = "error"
        return ProbeResult(request.key, outcome, text)
    outcome = result.get("outcome")
    if outcome not in {"accepted", "rejected"}:
        return ProbeResult(
            request.key,
            "error",
            f"COMGR worker returned unknown outcome {outcome!r}",
        )
    evidence = result.get("evidence")
    diagnostic = result.get("diagnostic")
    failed_stage = result.get("failed_stage")
    return ProbeResult(
        request.key,
        str(outcome),
        str(diagnostic) if diagnostic else None,
        evidence if isinstance(evidence, dict) else None,
        str(failed_stage) if failed_stage else None,
    )


def default_jobs(request_count: int) -> int:
    """Choose enough isolated workers to minimize wall time without oversubscription."""
    if request_count <= 0:
        return 1
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    return max(1, min(request_count, available))


def probe_all(
    requests: Iterable[ProbeRequest],
    *,
    comgr_lib: Path | None,
    jobs: int = 0,
    timeout: int = _DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, ProbeResult]:
    """Run independent candidates concurrently and preserve input order by key."""
    work = list(requests)
    if len({request.key for request in work}) != len(work):
        raise ValueError("COMGR probe request keys must be unique")
    worker_count = jobs or default_jobs(len(work))
    if worker_count < 1:
        raise ValueError(f"COMGR probe jobs={jobs}, expected a non-negative integer")

    def run(request: ProbeRequest) -> ProbeResult:
        return _probe_one(request, comgr_lib=comgr_lib, timeout=timeout)

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        results = list(pool.map(run, work))
    return {result.key: result for result in results}
