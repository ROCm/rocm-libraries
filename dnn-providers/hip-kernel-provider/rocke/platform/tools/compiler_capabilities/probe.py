#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe the COMGR compiler contract that consumes rocKE's production LLVM IR.

Every observation traverses COMGR's LLVM-IR-to-HSACO pipeline in a disposable
process. Clang and opt are intentionally absent: their behavior does not decide
whether rocKE's production compiler accepts an emitted module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_PLATFORM = Path(__file__).resolve().parents[2]
_PYTHON = _PLATFORM / "python"
if _PYTHON.is_dir() and str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

if __package__:
    from . import _probe_support as support
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compiler_capabilities import _probe_support as support

_TARGET_TRIPLE = "amdgcn-amd-amdhsa"
_CAPABILITY_LAYOUT = "amdgpu.comgr.explicit_data_layout"
_CAPABILITY_BUFFER_RESOURCE = "amdgpu.comgr.intrinsic.make_buffer_resource.abis"
_CAPABILITY_FP8_PACKING = "amdgpu.comgr.intrinsic.mfma_fp8_bf8.packed_operand_type"
_CAPABILITY_NAMED_BARRIER = "amdgpu.comgr.intrinsic.named_barrier.pointer_address_space"
_CAPABILITY_SCALED_MFMA = "amdgpu.comgr.intrinsic.mfma_f8f6f4.call_abi"
_CAPABILITY_PROCESSOR_NAMES = "amdgpu.comgr.accepts_processor_name"

# L0, L1, and L3 are the complete layouts emitted by the current llvm20,
# llvm22, and llvm23/llvm24 profiles, respectively. L2 is the observed
# intermediate layout with ``m:e`` but without p10-p15; no current profile emits
# it. Keep these as probe-owned archaeology rather than importing the policy
# constants whose assumptions this evidence is intended to test.
_LAYOUT_CANDIDATES = {
    "L0": (
        "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
        "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32"
        "-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
        "-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
    ),
    "L1": (
        "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
        "-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64"
        "-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512"
        "-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
    ),
    "L2": (
        "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32"
        "-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32"
        "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256"
        "-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
    ),
    "L3": (
        "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32"
        "-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32"
        "-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32"
        "-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256"
        "-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
    ),
}

# LLVM generic processor names are not derivable from the concrete gfx digits.
# Keep only names whose spelling is part of rocKE's target catalog/history.
_GENERIC_PROCESSORS = {
    "gfx906": "gfx9-generic",
    "gfx908": "gfx9-generic",
    "gfx90a": "gfx9-generic",
    "gfx942": "gfx9-4-generic",
    "gfx950": "gfx9-4-generic",
    "gfx1100": "gfx11-generic",
    "gfx1151": "gfx11-generic",
    "gfx1201": "gfx12-generic",
}


@dataclass(frozen=True)
class _IntrinsicCandidate:
    """One exact intrinsic declaration and call submitted to COMGR."""

    candidate: str
    family: str
    intrinsic: str
    result_type: str
    declaration_args: tuple[str, ...]
    call_args: tuple[tuple[str, str], ...]

    def module(self, *, layout: str | None = None) -> str:
        arguments = ", ".join(f"{ty} {value}" for ty, value in self.call_args)
        declaration = ", ".join(self.declaration_args)
        call = f"call {self.result_type} @{self.intrinsic}({arguments})"
        if self.result_type != "void":
            call = "%result = " + call
        preamble = f'target triple = "{_TARGET_TRIPLE}"\n'
        if layout is not None:
            preamble = f'target datalayout = "{layout}"\n' + preamble
        return (
            preamble
            + "define amdgpu_kernel void @probe() {\n"
            + f"  {call}\n"
            + "  ret void\n"
            + "}\n"
            + f"declare {self.result_type} @{self.intrinsic}({declaration})\n"
        )


def _buffer_resource_candidates() -> tuple[_IntrinsicCandidate, ...]:
    result = []
    for spelling, intrinsic in (
        ("p1", "llvm.amdgcn.make.buffer.rsrc.p1"),
        ("p8_p1", "llvm.amdgcn.make.buffer.rsrc.p8.p1"),
    ):
        for count_type in ("i32", "i64"):
            result.append(
                _IntrinsicCandidate(
                    f"{spelling}_{count_type}",
                    "make_buffer_resource",
                    intrinsic,
                    "ptr addrspace(8)",
                    ("ptr addrspace(1)", "i16", count_type, "i32"),
                    (
                        ("ptr addrspace(1)", "null"),
                        ("i16", "0"),
                        (count_type, "4"),
                        ("i32", "0"),
                    ),
                )
            )
    return tuple(result)


def _fp8_packing_candidates() -> tuple[_IntrinsicCandidate, ...]:
    result = []
    for shape, result_type in (
        ("16x16x32", "<4 x float>"),
        ("32x32x16", "<16 x float>"),
    ):
        for lhs, rhs in (
            ("fp8", "fp8"),
            ("bf8", "bf8"),
            ("fp8", "bf8"),
            ("bf8", "fp8"),
        ):
            family = f"{shape}_{lhs}_{rhs}"
            intrinsic = f"llvm.amdgcn.mfma.f32.{shape}.{lhs}.{rhs}"
            for candidate, operand_type, value in (
                ("v2i32", "<2 x i32>", "zeroinitializer"),
                ("i64", "i64", "0"),
            ):
                result.append(
                    _IntrinsicCandidate(
                        candidate,
                        family,
                        intrinsic,
                        result_type,
                        (
                            operand_type,
                            operand_type,
                            result_type,
                            "i32 immarg",
                            "i32 immarg",
                            "i32 immarg",
                        ),
                        (
                            (operand_type, value),
                            (operand_type, value),
                            (result_type, "zeroinitializer"),
                            ("i32", "0"),
                            ("i32", "0"),
                            ("i32", "0"),
                        ),
                    )
                )
    return tuple(result)


def _named_barrier_candidates() -> tuple[_IntrinsicCandidate, ...]:
    # The current flavor profiles disagree on whether named-barrier pointers use
    # LDS address space 3 or target-extension address space 15. Probe every
    # pointer-taking operation through codegen rather than inferring one ABI from
    # the compiler major.
    result = []
    for operation, has_count in (
        ("s.barrier.init", True),
        ("s.barrier.signal.var", True),
        ("s.barrier.join", False),
        ("s.wakeup.barrier", False),
    ):
        for address_space in (3, 15):
            pointer = f"ptr addrspace({address_space})"
            result.append(
                _IntrinsicCandidate(
                    f"as{address_space}",
                    operation,
                    f"llvm.amdgcn.{operation}",
                    "void",
                    (pointer, "i32") if has_count else (pointer,),
                    (
                        ((pointer, "null"), ("i32", "1"))
                        if has_count
                        else ((pointer, "null"),)
                    ),
                )
            )
    return tuple(result)


def _scaled_mfma_candidates() -> tuple[_IntrinsicCandidate, ...]:
    # rocKE's scaled-MFMA emitter has used both the 9-argument form and the
    # earlier 11-argument form with two leading i1 operands. Keep both exact call
    # ABIs so COMGR, rather than the current lowering policy, supplies the answer.
    result = []
    for shape, result_type in (
        ("16x16x128", "<4 x float>"),
        ("32x32x64", "<16 x float>"),
    ):
        intrinsic = f"llvm.amdgcn.mfma.scale.f32.{shape}.f8f6f4.v8i32.v8i32"
        common_call = (
            ("<8 x i32>", "zeroinitializer"),
            ("<8 x i32>", "zeroinitializer"),
            (result_type, "zeroinitializer"),
            ("i32", "0"),
            ("i32", "0"),
        )
        for candidate, extra_decl, extra_call in (
            (
                "current_9",
                ("i32 immarg", "i32", "i32 immarg", "i32"),
                (("i32", "0"), ("i32", "0"), ("i32", "0"), ("i32", "0")),
            ),
            (
                "legacy_11",
                ("i1", "i1", "i32 immarg", "i32", "i32 immarg", "i32"),
                (
                    ("i1", "false"),
                    ("i1", "false"),
                    ("i32", "0"),
                    ("i32", "0"),
                    ("i32", "0"),
                    ("i32", "0"),
                ),
            ),
        ):
            result.append(
                _IntrinsicCandidate(
                    candidate,
                    shape,
                    intrinsic,
                    result_type,
                    ("<8 x i32>", "<8 x i32>", result_type, "i32 immarg", "i32 immarg")
                    + extra_decl,
                    common_call + extra_call,
                )
            )
    return tuple(result)


def _empty_module(layout: str | None = None) -> str:
    preamble = f'target triple = "{_TARGET_TRIPLE}"\n'
    if layout is not None:
        preamble = f'target datalayout = "{layout}"\n' + preamble
    return preamble + "define amdgpu_kernel void @probe() { ret void }\n"


def _candidate_matrix_module(candidates: tuple[_IntrinsicCandidate, ...]) -> str:
    """Compile a policy-wide ABI candidate as one all-or-nothing module."""
    definitions = []
    declarations = []
    for index, item in enumerate(candidates):
        arguments = ", ".join(f"{ty} {value}" for ty, value in item.call_args)
        declaration = ", ".join(item.declaration_args)
        call = f"call {item.result_type} @{item.intrinsic}({arguments})"
        if item.result_type != "void":
            call = f"%result{index} = " + call
        definitions.append(
            f"define amdgpu_kernel void @probe{index}() {{\n  {call}\n  ret void\n}}\n"
        )
        declarations.append(
            f"declare {item.result_type} @{item.intrinsic}({declaration})\n"
        )
    return (
        f'target triple = "{_TARGET_TRIPLE}"\n'
        + "".join(definitions)
        + "".join(declarations)
    )


def _request_key(capability: str, family: str, candidate: str) -> str:
    return f"{capability}\0{family}\0{candidate}"


def _requests(
    processor: str,
) -> tuple[list[support.ProbeRequest], dict[str, tuple[str, str, str]]]:
    isa = f"{_TARGET_TRIPLE}--{processor}"
    requests: list[support.ProbeRequest] = []
    metadata: dict[str, tuple[str, str, str]] = {}

    def add(
        capability: str,
        family: str,
        candidate: str,
        module: str,
        *,
        request_isa: str = isa,
        options: tuple[str, ...] = ("-O3",),
    ) -> None:
        key = _request_key(capability, family, candidate)
        requests.append(support.ProbeRequest(key, module, request_isa, options))
        metadata[key] = (capability, family, candidate)

    for name, layout in _LAYOUT_CANDIDATES.items():
        add(_CAPABILITY_LAYOUT, "module", name, _empty_module(layout))

    # Buffer-resource spelling and count width are independent policy axes, so
    # retain their four exact combinations.
    for item in _buffer_resource_candidates():
        add(_CAPABILITY_BUFFER_RESOURCE, item.family, item.candidate, item.module())

    # The remaining choices are global lowering policies. Compile every affected
    # intrinsic in one module per candidate: rocKE can select a candidate only
    # when the complete family lowers, and one result avoids repeating the same
    # policy question for every shape or operation.
    for capability, family, candidates in (
        (
            _CAPABILITY_FP8_PACKING,
            "all_mfma_fp8_bf8",
            _fp8_packing_candidates(),
        ),
        (
            _CAPABILITY_NAMED_BARRIER,
            "all_named_barriers",
            _named_barrier_candidates(),
        ),
        (
            _CAPABILITY_SCALED_MFMA,
            "all_mfma_f8f6f4",
            _scaled_mfma_candidates(),
        ),
    ):
        by_candidate: dict[str, list[_IntrinsicCandidate]] = {}
        for item in candidates:
            by_candidate.setdefault(item.candidate, []).append(item)
        for candidate, members in by_candidate.items():
            add(
                capability,
                family,
                candidate,
                _candidate_matrix_module(tuple(members)),
            )

    add(_CAPABILITY_PROCESSOR_NAMES, processor, "requested", _empty_module())
    generic = _GENERIC_PROCESSORS.get(processor)
    if generic:
        add(
            _CAPABILITY_PROCESSOR_NAMES,
            generic,
            "generic",
            _empty_module(),
            request_isa=f"{_TARGET_TRIPLE}--{generic}",
        )
    return requests, metadata


def _implementation_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(support.__file__).resolve(),
        Path(__file__).with_name("_comgr_worker.py"),
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _schema_sha256() -> str:
    return support._sha256(Path(__file__).with_name("capabilities.schema.json"))


def _fingerprint(
    identity: dict[str, object], processor: str, implementation: str, schema: str
) -> str:
    inputs = {
        "comgr": identity,
        "processor": processor,
        "target_triple": _TARGET_TRIPLE,
        "implementation_sha256": implementation,
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
) -> tuple[dict[str, object], bool]:
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
    schema = _schema_sha256()
    fingerprint = _fingerprint(identity, processor, implementation, schema)
    cache_path = cache_dir / f"{fingerprint}.json"
    if not refresh:
        cached = support._read_cache(cache_path, fingerprint)
        if cached is not None:
            _validate_report_completeness(cached)
            return cached, True

    requests, metadata = _requests(processor)
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
    requested_key = _request_key(_CAPABILITY_PROCESSOR_NAMES, processor, "requested")
    generic = _GENERIC_PROCESSORS.get(processor)
    generic_key = (
        _request_key(_CAPABILITY_PROCESSOR_NAMES, generic, "generic")
        if generic
        else None
    )
    if (
        results[requested_key].outcome != "accepted"
        and generic_key is not None
        and results[generic_key].outcome == "accepted"
    ):
        effective_processor = generic
        fallback_isa = f"{_TARGET_TRIPLE}--{generic}"
        fallback_requests = [
            support.ProbeRequest(
                request.key,
                request.ir_text,
                fallback_isa,
                request.options,
            )
            for request in requests
            if metadata[request.key][0] != _CAPABILITY_PROCESSOR_NAMES
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
            "schema_sha256": schema,
            "target_triple": _TARGET_TRIPLE,
            "processor": processor,
            "effective_processor": effective_processor,
            "fingerprint": fingerprint,
        },
        "compiler": {"route": "comgr", **identity},
        "capabilities": _build_capabilities(requests, metadata, results),
    }
    _validate_report_completeness(report)
    outcomes = [result.outcome for result in results.values()]
    if all(outcome not in {"timeout", "error"} for outcome in outcomes):
        support._write_json(cache_path, report)
    return report, False


def _expected_observations(processor: str) -> dict[str, list[tuple[str, str]]]:
    expected = {
        _CAPABILITY_LAYOUT: [("module", name) for name in _LAYOUT_CANDIDATES],
        _CAPABILITY_BUFFER_RESOURCE: [
            (item.family, item.candidate) for item in _buffer_resource_candidates()
        ],
        _CAPABILITY_FP8_PACKING: [
            ("all_mfma_fp8_bf8", candidate) for candidate in ("v2i32", "i64")
        ],
        _CAPABILITY_NAMED_BARRIER: [
            ("all_named_barriers", candidate) for candidate in ("as3", "as15")
        ],
        _CAPABILITY_SCALED_MFMA: [
            ("all_mfma_f8f6f4", candidate) for candidate in ("current_9", "legacy_11")
        ],
        _CAPABILITY_PROCESSOR_NAMES: [(processor, "requested")],
    }
    generic = _GENERIC_PROCESSORS.get(processor)
    if generic:
        expected[_CAPABILITY_PROCESSOR_NAMES].append((generic, "generic"))
    return expected


def _validate_report_completeness(report: dict[str, object]) -> None:
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
    if probe.get("target_triple") != _TARGET_TRIPLE:
        raise support.ProbeError("report probe.target_triple is not AMDGPU")
    for field in (
        "implementation_sha256",
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
    if generic := _GENERIC_PROCESSORS.get(processor):
        allowed_effective.add(generic)
    if effective_processor not in allowed_effective:
        raise support.ProbeError(
            "report probe.effective_processor is not the requested processor "
            "or its explicit generic fallback"
        )
    expected = _expected_observations(processor)
    if set(capabilities) != set(expected):
        raise support.ProbeError("report capability set differs from the COMGR corpus")
    for name, corpus in expected.items():
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
        if observed != corpus:
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


def _validate_report(report: dict[str, object]) -> None:
    _validate_report_completeness(report)
    try:
        import jsonschema
    except ImportError as exc:
        raise support.ProbeError(
            "full schema validation requires jsonschema; the dependency-free "
            "report contract passed"
        ) from exc
    schema = json.loads(
        Path(__file__).with_name("capabilities.schema.json").read_text(encoding="utf-8")
    )
    try:
        jsonschema.validate(instance=report, schema=schema)
    except jsonschema.ValidationError as exc:
        raise support.ProbeError(
            f"generated report failed schema validation: {exc.message}"
        ) from exc


def _print_summary(report: dict[str, object], cache_hit: bool, output: Path) -> None:
    compiler = report["compiler"]
    version = compiler.get("llvm_version")
    version_text = ".".join(str(part) for part in version) if version else "unknown"
    print(f"COMGR: {compiler['library']}")
    print(f"COMGR sha256: {compiler['sha256'][:12]}…")
    print(f"embedded LLVM: {version_text} via {compiler['version_source']}")
    print(f"cache: {'hit' if cache_hit else 'miss'}")
    for name, result in report["capabilities"].items():
        observations = result["value"]["observations"]
        counts = {
            outcome: sum(entry["outcome"] == outcome for entry in observations)
            for outcome in ("accepted", "rejected", "crash", "timeout", "error")
        }
        detail = ", ".join(f"{key}={value}" for key, value in counts.items() if value)
        print(f"{name}: {detail}")
    print(f"output: {output}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="probe the COMGR contract that compiles rocKE LLVM IR"
    )
    parser.add_argument(
        "--comgr-lib",
        type=Path,
        default=None,
        help="explicit libamd_comgr path (default: rocKE runtime discovery)",
    )
    parser.add_argument("--processor", default="gfx950")
    parser.add_argument("--output", type=Path, default=Path("capabilities.json"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "rocke_comgr_capabilities",
    )
    parser.add_argument(
        "--jobs", type=int, default=0, help="parallel workers; 0=CPU count"
    )
    parser.add_argument("--timeout", type=int, default=60, help="seconds per candidate")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    try:
        report, cache_hit = load_or_probe(
            comgr_lib=args.comgr_lib,
            processor=args.processor,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            jobs=args.jobs,
            timeout=args.timeout,
        )
        if args.validate:
            _validate_report(report)
        support._write_json(args.output, report)
    except (OSError, TypeError, ValueError, support.ProbeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        _print_summary(report, cache_hit, args.output)
        if args.validate:
            print("schema validation: passed")
        print(f"elapsed: {time.perf_counter() - started:.3f} s")
    outcomes = [
        observation["outcome"]
        for result in report["capabilities"].values()
        for observation in result["value"]["observations"]
    ]
    return 0 if all(outcome != "error" for outcome in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
