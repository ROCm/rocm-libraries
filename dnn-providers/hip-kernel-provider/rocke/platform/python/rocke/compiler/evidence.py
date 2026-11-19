# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe the COMGR compiler contract that consumes rocKE's production LLVM IR.

Every observation traverses COMGR's LLVM-IR-to-HSACO pipeline in a disposable
process. Clang and opt are intentionally absent: their behavior does not decide
whether rocKE's production compiler accepts an emitted module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

from . import _support as support
from . import corpus, model
from .model import CAPABILITY_PROCESSOR_NAMES, TARGET_TRIPLE, EvidenceReport
from .policy import CompilerPolicy, resolve_policy

__all__ = [
    "ProbeError",
    "default_cache_dir",
    "load_or_probe",
    "query_compiler_policy",
    "validate_report",
    "validate_report_structure",
]

ProbeError = support.ProbeError


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(corpus.__file__).resolve(),
        Path(model.__file__).resolve(),
        Path(support.__file__).resolve(),
        Path(__file__).with_name("_worker.py"),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _schema_sha256() -> str:
    return support._sha256(
        Path(__file__).with_name("data") / "capabilities.schema.json"
    )


def _fingerprint(
    identity: dict[str, object],
    processor: str,
    implementation: str,
    protocol: str,
    schema: str,
) -> str:
    inputs = {
        "comgr": identity,
        "processor": processor,
        "target_triple": TARGET_TRIPLE,
        "implementation_sha256": implementation,
        "protocol_sha256": protocol,
        "schema_sha256": schema,
    }
    return hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _build_capabilities(
    requests: list[support.ProbeRequest],
    metadata: dict[str, tuple[str, str, str]],
    results: dict[str, support.ProbeResult],
) -> dict[str, object]:
    capabilities: dict[str, dict[str, object]] = {}
    for request in requests:
        capability, family, candidate = metadata[request.key]
        result = results[request.key]
        observation: dict[str, object] = {
            "family": family,
            "candidate": candidate,
            **result.as_report_entry(),
        }
        entry = capabilities.setdefault(
            capability,
            {
                "status": "known",
                "value": {
                    "route": "comgr_llvm_ir_to_hsaco",
                    "observations": [],
                },
            },
        )
        entry["value"]["observations"].append(observation)
    return capabilities


_Identify = Callable[[Path | None], dict[str, object]]
_ProbeAll = Callable[..., dict[str, support.ProbeResult]]


def default_cache_dir() -> Path:
    """Persistent per-user cache for exact COMGR evidence."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "rocke" / "compiler_capabilities"


def load_or_probe(
    *,
    comgr_lib: Path | None,
    processor: str,
    cache_dir: Path,
    refresh: bool = False,
    jobs: int = 0,
    timeout: int = 60,
    identify: _Identify = support.identify_comgr,
    run: _ProbeAll = support.probe_all,
) -> tuple[EvidenceReport, bool]:
    """Return COMGR capability evidence and whether it came from cache."""
    if re.fullmatch(r"gfx[0-9a-z-]+", processor) is None:
        raise ValueError(
            f"compiler capability probe failed - processor={processor!r}, "
            "expected a gfx processor identifier"
        )
    identity = identify(comgr_lib)
    selected_library = identity.get("library")
    if not isinstance(selected_library, str) or not selected_library:
        raise support.ProbeError("COMGR identity omitted the selected library")
    pinned_comgr = Path(selected_library)
    implementation = _implementation_sha256()
    protocol = corpus.protocol_sha256(processor)
    schema = _schema_sha256()
    fingerprint = _fingerprint(identity, processor, implementation, protocol, schema)
    cache_path = cache_dir / f"{fingerprint}.json"
    cacheable = identity.get("dependency_identity_complete") is True
    if not refresh and cacheable:
        cached = support._read_cache(cache_path, fingerprint)
        if cached is not None:
            validate_report_structure(cached)
            return cast(EvidenceReport, cached), True

    requests, metadata = corpus.requests_for_processor(processor)
    results = run(
        requests,
        comgr_lib=pinned_comgr,
        jobs=jobs,
        timeout=timeout,
    )

    # The common path speculatively probes every behavior with the requested
    # processor in one concurrent wave. If only its catalogued generic fallback
    # works, rerun the behavior requests with that ISA. This costs no serial
    # preflight on current compilers and asks no irrelevant duplicate questions
    # unless fallback is actually needed.
    effective_processor = processor
    requested_key = corpus.request_key(
        CAPABILITY_PROCESSOR_NAMES, processor, "requested"
    )
    generic = corpus.GENERIC_PROCESSORS.get(processor)
    generic_key = (
        corpus.request_key(CAPABILITY_PROCESSOR_NAMES, generic, "generic")
        if generic
        else None
    )
    if (
        results[requested_key].outcome != "accepted"
        and generic_key is not None
        and results[generic_key].outcome == "accepted"
    ):
        effective_processor = generic
        fallback_isa = f"{TARGET_TRIPLE}--{generic}"
        fallback_requests = [
            support.ProbeRequest(
                request.key,
                request.ir_text,
                fallback_isa,
                request.options,
            )
            for request in requests
            if metadata[request.key][0] != CAPABILITY_PROCESSOR_NAMES
        ]
        results.update(
            run(
                fallback_requests,
                comgr_lib=pinned_comgr,
                jobs=jobs,
                timeout=timeout,
            )
        )

    report = {
        "probe": {
            "name": "rocke-comgr-capabilities",
            "implementation_sha256": implementation,
            "protocol_sha256": protocol,
            "schema_sha256": schema,
            "target_triple": TARGET_TRIPLE,
            "processor": processor,
            "effective_processor": effective_processor,
            "fingerprint": fingerprint,
        },
        "compiler": {"route": "comgr", **identity},
        "capabilities": _build_capabilities(requests, metadata, results),
    }
    validate_report_structure(report)
    outcomes = [result.outcome for result in results.values()]
    if cacheable and all(outcome not in {"timeout", "error"} for outcome in outcomes):
        support._write_json(cache_path, report)
    return cast(EvidenceReport, report), False


def query_compiler_policy(
    *,
    processor: str,
    comgr_lib: Path | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
    jobs: int = 0,
    timeout: int = 60,
) -> tuple[CompilerPolicy, bool]:
    """Measure the selected COMGR and return its canonical lowering policy.

    The cache contains raw evidence. Policy resolution is deterministic and
    reruns on every call so a resolver change cannot reuse an old decision.
    """
    report, cache_hit = load_or_probe(
        comgr_lib=comgr_lib,
        processor=processor,
        cache_dir=cache_dir or default_cache_dir(),
        refresh=refresh,
        jobs=jobs,
        timeout=timeout,
    )
    return resolve_policy(report), cache_hit


def validate_report_structure(report: dict[str, object]) -> None:
    capabilities = report.get("capabilities")
    probe = report.get("probe")
    compiler = report.get("compiler")
    if (
        not isinstance(capabilities, dict)
        or not isinstance(probe, dict)
        or not isinstance(compiler, dict)
    ):
        raise support.ProbeError(
            "report must contain probe, compiler, and capabilities objects"
        )
    if probe.get("name") != "rocke-comgr-capabilities":
        raise support.ProbeError("report probe.name must identify the COMGR probe")
    if probe.get("target_triple") != TARGET_TRIPLE:
        raise support.ProbeError("report probe.target_triple is not AMDGPU")
    for field in (
        "implementation_sha256",
        "protocol_sha256",
        "schema_sha256",
        "fingerprint",
    ):
        value = probe.get(field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise support.ProbeError(f"report probe.{field} must be a SHA-256 digest")
    if compiler.get("route") != "comgr":
        raise support.ProbeError("report compiler.route must be comgr")
    library = compiler.get("library")
    if not isinstance(library, str) or not library:
        raise support.ProbeError("report compiler.library must be a non-empty string")
    compiler_sha = compiler.get("sha256")
    if (
        not isinstance(compiler_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", compiler_sha) is None
    ):
        raise support.ProbeError("report compiler.sha256 must be a SHA-256 digest")
    if not isinstance(compiler.get("version_source"), str) or not compiler.get(
        "version_source"
    ):
        raise support.ProbeError("report compiler.version_source must be non-empty")
    if not isinstance(compiler.get("dependency_identity_complete"), bool):
        raise support.ProbeError(
            "report compiler.dependency_identity_complete must be a boolean"
        )
    dependencies = compiler.get("dependencies")
    if not isinstance(dependencies, list):
        raise support.ProbeError("report compiler.dependencies must be a list")
    for dependency in dependencies:
        if (
            not isinstance(dependency, dict)
            or set(dependency) != {"library", "sha256"}
            or not isinstance(dependency["library"], str)
            or not dependency["library"]
            or not isinstance(dependency["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", dependency["sha256"]) is None
        ):
            raise support.ProbeError(
                "report compiler.dependencies contains malformed provenance"
            )
    llvm_version = compiler.get("llvm_version")
    if llvm_version is not None and (
        not isinstance(llvm_version, list)
        or len(llvm_version) != 3
        or not all(isinstance(part, int) and part >= 0 for part in llvm_version)
    ):
        raise support.ProbeError(
            "report compiler.llvm_version must be three non-negative integers or null"
        )
    processor = probe.get("processor")
    effective_processor = probe.get("effective_processor")
    if not isinstance(processor, str):
        raise support.ProbeError("report probe.processor must be a string")
    if not isinstance(effective_processor, str) or not effective_processor:
        raise support.ProbeError("report probe.effective_processor must be a string")
    allowed_effective = {processor}
    if generic := corpus.GENERIC_PROCESSORS.get(processor):
        allowed_effective.add(generic)
    if effective_processor not in allowed_effective:
        raise support.ProbeError(
            "report probe.effective_processor is not the requested processor "
            "or its explicit generic fallback"
        )
    expected = corpus.expected_observations(processor)
    if set(capabilities) != set(expected):
        raise support.ProbeError("report capability set differs from the COMGR corpus")
    for name, expected_corpus in expected.items():
        result = capabilities[name]
        if not isinstance(result, dict) or result.get("status") != "known":
            raise support.ProbeError(f"{name} must be a known observation set")
        value = result.get("value")
        if (
            not isinstance(value, dict)
            or value.get("route") != "comgr_llvm_ir_to_hsaco"
        ):
            raise support.ProbeError(f"{name} has no COMGR route")
        observations = value.get("observations")
        if not isinstance(observations, list):
            raise support.ProbeError(f"{name} observations must be a list")
        observed = [
            (entry.get("family"), entry.get("candidate")) for entry in observations
        ]
        if observed != expected_corpus:
            raise support.ProbeError(
                f"{name} did not report its complete ordered corpus"
            )
        for entry in observations:
            if entry.get("outcome") not in {
                "accepted",
                "rejected",
                "crash",
                "timeout",
                "error",
            }:
                raise support.ProbeError(f"{name} reported an unknown outcome")
            if entry.get("outcome") == "accepted" and not isinstance(
                entry.get("evidence"), dict
            ):
                raise support.ProbeError(f"{name} accepted candidate lacks evidence")
            if entry.get("outcome") != "accepted" and not entry.get("diagnostic"):
                raise support.ProbeError(
                    f"{name} non-accepted candidate lacks diagnostic"
                )


def validate_report(report: dict[str, object]) -> None:
    validate_report_structure(report)
    try:
        import jsonschema
    except ImportError as exc:
        raise support.ProbeError(
            "full schema validation requires jsonschema; the dependency-free "
            "report contract passed"
        ) from exc
    schema = json.loads(
        (Path(__file__).with_name("data") / "capabilities.schema.json").read_text(
            encoding="utf-8"
        )
    )
    try:
        jsonschema.validate(instance=report, schema=schema)
    except jsonschema.ValidationError as exc:
        raise support.ProbeError(
            f"generated report failed schema validation: {exc.message}"
        ) from exc
