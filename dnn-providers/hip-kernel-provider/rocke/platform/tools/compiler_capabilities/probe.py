#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Probe structural AMD LLVM capabilities and write ``capabilities.json``.

The report compares focused module-policy and intrinsic-ABI observations across
compiler snapshots. It remains an evidence producer: rocKE lowering does not
consume this report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from . import _probe_support as support
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compiler_capabilities import _probe_support as support

_TARGET_TRIPLE = "amdgcn-amd-amdhsa"
_CAPABILITY_PREAMBLE = "amdgpu.ir.module_preamble"
_CAPABILITY_LAYOUT_POLICY = "amdgpu.ir.explicit_data_layout_policy"
_CAPABILITY_BUFFER_RESOURCE = "amdgpu.intrinsic.make_buffer_resource.abis"
_CAPABILITY_FP8_PACKING = "amdgpu.intrinsic.mfma_fp8_bf8.packed_operand_type"
_CAPABILITY_NAMED_BARRIER = "amdgpu.intrinsic.named_barrier.pointer_address_space"
_CAPABILITY_SCALED_MFMA = "amdgpu.intrinsic.mfma_f8f6f4.call_abi"
_CAPABILITY_FRONTEND_FP16 = "amdgpu.frontend.builtin.mfma_fp16.source_abi"
_CAPABILITY_PROCESSOR_NAMES = "amdgpu.compiler.accepts_processor_name"
_CAPABILITY_OPT_LEVELS = "amdgpu.pipeline.optimization_level_codegen"
_CAPABILITY_COMPILER_OPTIONS = "amdgpu.compiler_option.known_controls"
_CAPABILITY_FUNCTION_ATTRIBUTES = "amdgpu.function_attribute.known_controls"

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


@dataclass(frozen=True)
class _IntrinsicCandidate:
    """One exact intrinsic declaration and call submitted to ``opt``."""

    candidate: str
    family: str
    intrinsic: str
    result_type: str
    declaration_args: tuple[str, ...]
    call_args: tuple[tuple[str, str], ...]

    def module(self) -> str:
        """Return a minimal module that verifies the candidate call shape."""
        arguments = ", ".join(f"{ty} {value}" for ty, value in self.call_args)
        declaration = ", ".join(self.declaration_args)
        call = f"call {self.result_type} @{self.intrinsic}({arguments})"
        if self.result_type != "void":
            call = "%result = " + call
        return (
            f'target triple = "{_TARGET_TRIPLE}"\n'
            "define amdgpu_kernel void @probe() {\n"
            f"  {call}\n"
            "  ret void\n"
            "}\n"
            f"declare {self.result_type} @{self.intrinsic}({declaration})\n"
        )


def _canonical_declaration(text: str, intrinsic_stem: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("declare ") or f"@{intrinsic_stem}" not in stripped:
            continue
        previous = lines[index - 1].strip() if index else ""
        if previous == "; Unknown intrinsic" or re.search(r" #\d+$", stripped) is None:
            return None
        return re.sub(r" #\d+$", "", stripped)
    return None


def _probe_intrinsic_candidates(
    opt: Path,
    candidates: tuple[_IntrinsicCandidate, ...],
    run: support._RunCommand,
) -> dict[str, object]:
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    corpus = [f"{candidate.family}:{candidate.candidate}" for candidate in candidates]
    for candidate in candidates:
        result = run([str(opt), "-S", "-", "-o", "-"], candidate.module())
        if result.timed_out:
            return {
                "status": "error",
                "diagnostic": f"{candidate.candidate}: {support._first_diagnostic(result)}",
            }
        if support._process_crashed(result):
            return {
                "status": "error",
                "diagnostic": f"{candidate.candidate}: compiler process crashed",
            }
        if result.returncode != 0:
            rejected.append(
                {
                    "candidate": candidate.candidate,
                    "family": candidate.family,
                    "diagnostic": support._first_diagnostic(result),
                }
            )
            continue
        intrinsic_stem = candidate.intrinsic
        if intrinsic_stem.startswith("llvm.amdgcn.make.buffer.rsrc."):
            intrinsic_stem = "llvm.amdgcn.make.buffer.rsrc."
        declaration = _canonical_declaration(result.stdout, intrinsic_stem)
        if declaration is None:
            rejected.append(
                {
                    "candidate": candidate.candidate,
                    "family": candidate.family,
                    "diagnostic": "opt did not resolve the declaration as an intrinsic",
                }
            )
            continue
        accepted.append(
            {
                "candidate": candidate.candidate,
                "family": candidate.family,
                "canonical_declaration": declaration,
            }
        )
    return {
        "status": "known",
        "value": {"corpus": corpus, "accepted": accepted, "rejected": rejected},
    }


def _probe_module_preamble(
    clang: Path, processor: str, run: support._RunCommand
) -> dict[str, object]:
    result = run(
        [
            str(clang),
            "-target",
            _TARGET_TRIPLE,
            f"-mcpu={processor}",
            "-nogpulib",
            "-S",
            "-emit-llvm",
            "-x",
            "c",
            "-",
            "-o",
            "-",
        ],
        "__attribute__((amdgpu_kernel)) void probe(void) {}\n",
    )
    if result.returncode != 0:
        return {"status": "error", "diagnostic": support._first_diagnostic(result)}
    triple = re.search(r'^target triple = "([^"]+)"$', result.stdout, re.MULTILINE)
    calling_convention = re.search(
        r"^define\b.*\b(amdgpu_kernel)\b.*@probe\(", result.stdout, re.MULTILINE
    )
    if triple is None or calling_convention is None:
        return {
            "status": "error",
            "diagnostic": "compiler output omitted the target triple or kernel calling convention",
        }
    metadata = {
        match.group(1): match.group(2)
        for line in result.stdout.splitlines()
        if (match := re.match(r"^!(\d+) = (.+)$", line))
    }
    root = next(
        (
            line
            for line in result.stdout.splitlines()
            if line.startswith("!llvm.module.flags =")
        ),
        "",
    )
    flags = [
        metadata[index] for index in re.findall(r"!(\d+)", root) if index in metadata
    ]
    return {
        "status": "known",
        "value": {
            "target_triple": triple.group(1),
            "kernel_calling_convention": calling_convention.group(1),
            "module_flags": flags,
        },
    }


def _probe_layout_policy(
    clang: Path, processor: str, run: support._RunCommand
) -> dict[str, object]:
    observations: list[dict[str, object]] = []
    for name, layout in _LAYOUT_CANDIDATES.items():
        module = (
            f'target datalayout = "{layout}"\n'
            f'target triple = "{_TARGET_TRIPLE}"\n'
            "define amdgpu_kernel void @probe() { ret void }\n"
        )
        result = run(
            [
                str(clang),
                "-target",
                _TARGET_TRIPLE,
                f"-mcpu={processor}",
                "-nogpulib",
                "-S",
                "-emit-llvm",
                "-x",
                "ir",
                "-",
                "-o",
                "-",
            ],
            module,
        )
        if result.timed_out:
            return {
                "status": "error",
                "diagnostic": f"{name}: {support._first_diagnostic(result)}",
            }
        if support._process_crashed(result):
            return {
                "status": "error",
                "diagnostic": f"{name}: compiler process crashed",
            }
        if result.returncode != 0:
            observations.append(
                {
                    "candidate": name,
                    "outcome": "rejected",
                    "diagnostic": support._first_diagnostic(result),
                }
            )
            continue
        try:
            observed = support._data_layout_value(result.stdout)["text"]
        except support.ProbeError as exc:
            return {"status": "error", "diagnostic": f"{name}: {exc}"}
        observations.append(
            {
                "candidate": name,
                "outcome": "accepted" if observed == layout else "overwritten",
                "observed_layout": observed,
            }
        )
    return {"status": "known", "value": {"candidates": observations}}


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


def _probe_buffer_resource_matrix(
    opt: Path, run: support._RunCommand
) -> dict[str, object]:
    result = _probe_intrinsic_candidates(opt, _buffer_resource_candidates(), run)
    if result["status"] != "known":
        return result
    for accepted in result["value"]["accepted"]:
        declaration = accepted["canonical_declaration"]
        name = re.search(r"@(llvm\.amdgcn\.make\.buffer\.rsrc\.[^(]+)", declaration)
        if name is None:
            return {
                "status": "error",
                "diagnostic": f"unrecognized buffer-resource declaration: {declaration}",
            }
        accepted["canonical_intrinsic"] = name.group(1)
        try:
            accepted["canonical_abi"] = support._canonical_buffer_resource_abi(
                name.group(1), declaration
            )
        except support.ProbeError as exc:
            return {"status": "error", "diagnostic": str(exc)}
    return result


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
    result = []
    for operation, has_count in (
        ("s.barrier.init", True),
        ("s.barrier.signal.var", True),
        ("s.barrier.join", False),
        ("s.wakeup.barrier", False),
    ):
        for address_space in (3, 15):
            pointer = f"ptr addrspace({address_space})"
            declaration_args = (pointer, "i32") if has_count else (pointer,)
            call_args = (
                ((pointer, "null"), ("i32", "1")) if has_count else ((pointer, "null"),)
            )
            result.append(
                _IntrinsicCandidate(
                    f"as{address_space}",
                    operation,
                    f"llvm.amdgcn.{operation}",
                    "void",
                    declaration_args,
                    call_args,
                )
            )
    return tuple(result)


def _scaled_mfma_candidates() -> tuple[_IntrinsicCandidate, ...]:
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


def _observation(
    *,
    candidate: str,
    family: str,
    stage: str,
    result: support.CommandResult,
    evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    if result.timed_out:
        return {
            "candidate": candidate,
            "family": family,
            "stage": stage,
            "outcome": "timeout",
            "diagnostic": support._first_diagnostic(result),
        }
    if result.returncode != 0:
        return {
            "candidate": candidate,
            "family": family,
            "stage": stage,
            "outcome": "crash" if support._process_crashed(result) else "rejected",
            "diagnostic": support._first_diagnostic(result),
        }
    value: dict[str, object] = {
        "candidate": candidate,
        "family": family,
        "stage": stage,
        "outcome": "accepted",
    }
    if evidence:
        value["evidence"] = evidence
    return value


def _observations_result(observations: list[dict[str, object]]) -> dict[str, object]:
    return {"status": "known", "value": {"observations": observations}}


def _output_evidence(result: support.CommandResult) -> dict[str, object]:
    text = result.stdout
    return {
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_lines": len(text.splitlines()),
    }


def _probe_frontend_fp16_abi(
    clang: Path, processor: str, run: support._RunCommand
) -> dict[str, object]:
    builtin = "__builtin_amdgcn_mfma_f32_32x32x8f16"
    result = run(
        [
            str(clang),
            "-target",
            _TARGET_TRIPLE,
            f"-mcpu={processor}",
            "-nogpulib",
            "-fsyntax-only",
            "-ferror-limit=0",
            "-Xclang",
            "-ast-dump",
            "-x",
            "c",
            "-",
        ],
        f"void query(void) {{ {builtin}(); }}\n",
    )
    if result.timed_out or (support._process_crashed(result)):
        return _observations_result(
            [
                _observation(
                    candidate="canonical_prototype",
                    family=builtin,
                    stage="frontend",
                    result=result,
                )
            ]
        )
    match = re.search(
        rf"FunctionDecl[^\n]*\b{re.escape(builtin)} '(?P<prototype>.*)'",
        result.stdout,
    )
    if match is not None:
        return _observations_result(
            [
                {
                    "candidate": "canonical_prototype",
                    "family": builtin,
                    "stage": "frontend",
                    "outcome": "accepted",
                    "evidence": {"prototype": match.group("prototype")},
                }
            ]
        )
    return _observations_result(
        [
            _observation(
                candidate="canonical_prototype",
                family=builtin,
                stage="frontend",
                result=result,
            )
        ]
    )


def _generic_processor_name(processor: str) -> str | None:
    if re.fullmatch(r"gfx11\d+", processor):
        return "gfx11-generic"
    if re.fullmatch(r"gfx12\d+", processor):
        return "gfx12-generic"
    match = re.fullmatch(r"gfx(9|10)(\d)\d+", processor)
    if match is None:
        return None
    return f"gfx{match.group(1)}-{match.group(2)}-generic"


def _probe_processor_names(
    clang: Path, processor: str, run: support._RunCommand
) -> dict[str, object]:
    names = [("requested", processor)]
    generic = _generic_processor_name(processor)
    if generic is not None and generic != processor:
        names.append(("generic", generic))
    observations = []
    for candidate, name in names:
        result = run(
            [
                str(clang),
                "-target",
                _TARGET_TRIPLE,
                f"-mcpu={name}",
                "-nogpulib",
                "-S",
                "-emit-llvm",
                "-x",
                "c",
                "-",
                "-o",
                "-",
            ],
            "__attribute__((amdgpu_kernel)) void probe(void) {}\n",
        )
        observations.append(
            _observation(
                candidate=candidate,
                family=name,
                stage="frontend",
                result=result,
                evidence=_output_evidence(result) if result.returncode == 0 else None,
            )
        )
    return _observations_result(observations)


def _codegen_reproducer() -> str:
    return (
        f'target triple = "{_TARGET_TRIPLE}"\n'
        "@uniform = internal addrspace(4) constant <4 x half> zeroinitializer\n"
        "define amdgpu_kernel void @probe(ptr addrspace(1) %out) {\n"
        "  %v = load <4 x half>, ptr addrspace(4) @uniform, align 8\n"
        "  %x0 = extractelement <4 x half> %v, i32 0\n"
        "  %x1 = extractelement <4 x half> %v, i32 1\n"
        "  %f0 = fpext half %x0 to float\n"
        "  %f1 = fpext half %x1 to float\n"
        "  %sum = fadd float %f0, %f1\n"
        "  store float %sum, ptr addrspace(1) %out, align 4\n"
        "  ret void\n"
        "}\n"
    )


def _probe_optimization_levels(
    clang: Path, processor: str, run: support._RunCommand
) -> dict[str, object]:
    observations = []
    for level in ("O0", "O1", "O2", "O3"):
        result = run(
            [
                str(clang),
                "-target",
                _TARGET_TRIPLE,
                f"-mcpu={processor}",
                "-nogpulib",
                f"-{level}",
                "-S",
                "-x",
                "ir",
                "-",
                "-o",
                "-",
            ],
            _codegen_reproducer(),
        )
        observations.append(
            _observation(
                candidate=level,
                family="uniform_half_extract",
                stage="codegen",
                result=result,
                evidence=_output_evidence(result) if result.returncode == 0 else None,
            )
        )
    return _observations_result(observations)


def _probe_compiler_options(
    clang: Path, processor: str, run: support._RunCommand
) -> dict[str, object]:
    source = "__attribute__((amdgpu_kernel)) void probe(void) {}\n"
    candidates = (
        ("slp_off", ("-fno-slp-vectorize",)),
        ("mfma_vgpr_form", ("-mllvm", "-amdgpu-mfma-vgpr-form")),
        (
            "iterative_ilp_scheduler",
            ("-mllvm", "-amdgpu-sched-strategy=iterative-ilp"),
        ),
    )
    observations = []
    for candidate, options in candidates:
        result = run(
            [
                str(clang),
                "-target",
                _TARGET_TRIPLE,
                f"-mcpu={processor}",
                "-nogpulib",
                "-O3",
                *options,
                "-S",
                "-emit-llvm",
                "-x",
                "c",
                "-",
                "-o",
                "-",
            ],
            source,
        )
        observations.append(
            _observation(
                candidate=candidate,
                family="backend_option",
                stage="option_parser",
                result=result,
                evidence=_output_evidence(result) if result.returncode == 0 else None,
            )
        )
    return _observations_result(observations)


def _attribute_module(attribute: str) -> str:
    return (
        f'target triple = "{_TARGET_TRIPLE}"\n'
        "define amdgpu_kernel void @probe() #0 { ret void }\n"
        f'attributes #0 = {{ "{attribute}" }}\n'
    )


def _probe_function_attributes(
    clang: Path, processor: str, run: support._RunCommand
) -> dict[str, object]:
    candidates = (
        ("waves_per_eu_1", 'amdgpu-waves-per-eu"="1,1'),
        ("waves_per_eu_2", 'amdgpu-waves-per-eu"="2,2'),
        ("zero_agpr", 'amdgpu-agpr-alloc"="0'),
    )
    observations = []
    for candidate, attribute in candidates:
        result = run(
            [
                str(clang),
                "-target",
                _TARGET_TRIPLE,
                f"-mcpu={processor}",
                "-nogpulib",
                "-O3",
                "-S",
                "-x",
                "ir",
                "-",
                "-o",
                "-",
            ],
            _attribute_module(attribute),
        )
        observations.append(
            _observation(
                candidate=candidate,
                family="function_attribute",
                stage="codegen",
                result=result,
                evidence=_output_evidence(result) if result.returncode == 0 else None,
            )
        )
    return _observations_result(observations)


def _structural_report(
    llvm_bin: Path | None,
    processor: str,
    cache_dir: Path,
    refresh: bool,
    run: support._RunCommand,
) -> tuple[dict[str, object], bool]:
    base, _ = support.load_or_probe(
        llvm_bin=llvm_bin,
        processor=processor,
        cache_dir=cache_dir / "base",
        refresh=refresh,
        run=run,
    )
    clang = support._tool_path("clang", llvm_bin)
    opt = support._tool_path("opt", llvm_bin)
    implementation_sha256 = hashlib.sha256(
        Path(__file__).resolve().read_bytes()
        + Path(support.__file__).resolve().read_bytes()
    ).hexdigest()
    schema_sha256 = support._sha256(
        Path(__file__).with_name("capabilities.schema.json")
    )
    fingerprint = hashlib.sha256(
        f"structural:{implementation_sha256}:{schema_sha256}:"
        f"{base['probe']['fingerprint']}".encode()
    ).hexdigest()
    cache_path = cache_dir / f"{fingerprint}.json"
    if not refresh:
        cached = support._read_cache(cache_path, fingerprint)
        if cached is not None:
            _validate_report_completeness(cached)
            return cached, True
    capabilities = dict(base["capabilities"])
    capabilities[_CAPABILITY_PREAMBLE] = _probe_module_preamble(clang, processor, run)
    capabilities[_CAPABILITY_LAYOUT_POLICY] = _probe_layout_policy(
        clang, processor, run
    )
    capabilities[_CAPABILITY_BUFFER_RESOURCE] = _probe_buffer_resource_matrix(opt, run)
    capabilities[_CAPABILITY_FP8_PACKING] = _probe_intrinsic_candidates(
        opt, _fp8_packing_candidates(), run
    )
    capabilities[_CAPABILITY_NAMED_BARRIER] = _probe_intrinsic_candidates(
        opt, _named_barrier_candidates(), run
    )
    capabilities[_CAPABILITY_SCALED_MFMA] = _probe_intrinsic_candidates(
        opt, _scaled_mfma_candidates(), run
    )
    capabilities[_CAPABILITY_FRONTEND_FP16] = _probe_frontend_fp16_abi(
        clang, processor, run
    )
    capabilities[_CAPABILITY_PROCESSOR_NAMES] = _probe_processor_names(
        clang, processor, run
    )
    capabilities[_CAPABILITY_OPT_LEVELS] = _probe_optimization_levels(
        clang, processor, run
    )
    capabilities[_CAPABILITY_COMPILER_OPTIONS] = _probe_compiler_options(
        clang, processor, run
    )
    capabilities[_CAPABILITY_FUNCTION_ATTRIBUTES] = _probe_function_attributes(
        clang, processor, run
    )
    report = dict(base)
    probe = dict(base["probe"])
    probe.pop("implementation_sha256", None)
    report["probe"] = dict(
        probe,
        implementation_sha256=implementation_sha256,
        schema_sha256=schema_sha256,
        fingerprint=fingerprint,
    )
    report["capabilities"] = capabilities
    _validate_report_completeness(report)
    if all(result.get("status") == "known" for result in capabilities.values()):
        support._write_json(cache_path, report)
    return report, False


def load_or_probe(
    *,
    llvm_bin: Path | None,
    processor: str,
    cache_dir: Path,
    refresh: bool = False,
    run: support._RunCommand = support._run_command,
) -> tuple[dict[str, object], bool]:
    """Return a structural report and whether it came from the report cache."""
    return _structural_report(llvm_bin, processor, cache_dir, refresh, run)


def _llvm_bin_from_python_env(python_env: Path) -> Path:
    """Find the LLVM tool directory installed by ROCm Python packages."""
    executable = ".exe" if os.name == "nt" else ""
    candidates = [
        python_env / "lib" / "llvm" / "bin",
        python_env
        / "Lib"
        / "site-packages"
        / "_rocm_sdk_core"
        / "lib"
        / "llvm"
        / "bin",
    ]
    candidates.extend(
        python_env.glob("lib/python*/site-packages/_rocm_sdk_core/lib/llvm/bin")
    )
    candidates.extend(
        python_env.glob("lib/python*/site-packages/_rocm_sdk_devel/lib/llvm/bin")
    )
    usable: list[Path] = []
    for candidate in candidates:
        if (candidate / f"clang{executable}").is_file() and (
            candidate / f"opt{executable}"
        ).is_file():
            resolved = candidate.resolve()
            if resolved not in usable:
                usable.append(resolved)
    if not usable:
        raise support.ProbeError(
            "compiler capability probe failed - no clang/opt pair found in "
            f"python_env={str(python_env)!r}; install rocm-sdk-core or pass --llvm-bin"
        )
    return usable[0]


def _layout_name(value: dict[str, object]) -> str:
    key = (
        value.get("elf_mangling"),
        tuple(value.get("address_space_8") or ()),
        tuple(value.get("additional_pointer_address_spaces") or ()),
    )
    return {
        (None, (128, 128), ()): "L0",
        (None, (128, 128, 128, 48), ()): "L1",
        ("e", (128, 128, 128, 48), ()): "L2",
        ("e", (128, 128, 128, 48), (10, 11, 12, 13, 14, 15)): "L3",
    }.get(key, "custom")


def _accepted_candidates(result: dict[str, object]) -> list[dict[str, str]]:
    if result.get("status") != "known":
        return []
    value = result.get("value")
    if not isinstance(value, dict):
        return []
    accepted = value.get("accepted")
    return accepted if isinstance(accepted, list) else []


def _observation_summary(result: dict[str, object]) -> str:
    if result.get("status") != "known":
        return "error"
    value = result.get("value")
    if not isinstance(value, dict) or not isinstance(value.get("observations"), list):
        return "malformed"
    return ", ".join(
        f"{entry['candidate']}={entry['outcome']}" for entry in value["observations"]
    )


def _uniform_candidate(result: dict[str, object], family_count: int) -> str:
    accepted = _accepted_candidates(result)
    families = {entry["family"] for entry in accepted}
    candidates = {entry["candidate"] for entry in accepted}
    if len(families) != family_count or len(candidates) != 1:
        return "mixed or unavailable"
    return next(iter(candidates))


def _print_summary(report: dict[str, object], cache_hit: bool, output: Path) -> None:
    capabilities = report["capabilities"]
    layout = capabilities[support._CAPABILITY_DATA_LAYOUT]
    preamble = capabilities[_CAPABILITY_PREAMBLE]
    policy = capabilities[_CAPABILITY_LAYOUT_POLICY]
    print(f"probe implementation: {report['probe']['implementation_sha256'][:12]}…")
    print(f"schema: {report['probe']['schema_sha256'][:12]}…")
    print(
        f"compiler revision: {report['compiler'].get('source_revision') or 'unreported'}"
    )
    print(f"cache: {'hit' if cache_hit else 'miss'}")
    if layout.get("status") == "known":
        print(f"module layout: {_layout_name(layout['value'])}")
    if preamble.get("status") == "known":
        value = preamble["value"]
        print(
            "module preamble: "
            f"{value['target_triple']}, {value['kernel_calling_convention']}, "
            f"{len(value['module_flags'])} flags"
        )
    if policy.get("status") == "known":
        outcomes = ", ".join(
            f"{entry['candidate']}={entry['outcome']}"
            for entry in policy["value"]["candidates"]
        )
        print(f"explicit layouts: {outcomes}")
    buffer_abis = sorted(
        {
            entry["canonical_abi"]
            for entry in _accepted_candidates(capabilities[_CAPABILITY_BUFFER_RESOURCE])
        }
    )
    print(f"buffer-resource ABIs: {', '.join(buffer_abis) or 'none'}")
    print(
        "FP8/BF8 packed operand: "
        + _uniform_candidate(capabilities[_CAPABILITY_FP8_PACKING], 8)
    )
    print(
        "named-barrier pointer: "
        + _uniform_candidate(capabilities[_CAPABILITY_NAMED_BARRIER], 4)
    )
    print(
        "scaled-MFMA call ABI: "
        + _uniform_candidate(capabilities[_CAPABILITY_SCALED_MFMA], 2)
    )
    print(
        "frontend FP16 ABI: "
        + _observation_summary(capabilities[_CAPABILITY_FRONTEND_FP16])
    )
    print(
        "processor names: "
        + _observation_summary(capabilities[_CAPABILITY_PROCESSOR_NAMES])
    )
    print(
        "optimization levels: "
        + _observation_summary(capabilities[_CAPABILITY_OPT_LEVELS])
    )
    print(
        "compiler options: "
        + _observation_summary(capabilities[_CAPABILITY_COMPILER_OPTIONS])
    )
    print(
        "function attributes: "
        + _observation_summary(capabilities[_CAPABILITY_FUNCTION_ATTRIBUTES])
    )
    errors = [
        f"{name}: {result.get('diagnostic', 'unknown probe error')}"
        for name, result in capabilities.items()
        if result.get("status") != "known"
    ]
    for error in errors:
        print(f"ERROR: {error}")
    print(f"output: {output}")


def _candidate_id(entry: dict[str, object]) -> str:
    return f"{entry['family']}:{entry['candidate']}"


def _validate_report_completeness(report: dict[str, object]) -> None:
    capabilities = report["capabilities"]
    expected_corpora = {
        _CAPABILITY_BUFFER_RESOURCE: [
            f"{candidate.family}:{candidate.candidate}"
            for candidate in _buffer_resource_candidates()
        ],
        _CAPABILITY_FP8_PACKING: [
            f"{candidate.family}:{candidate.candidate}"
            for candidate in _fp8_packing_candidates()
        ],
        _CAPABILITY_NAMED_BARRIER: [
            f"{candidate.family}:{candidate.candidate}"
            for candidate in _named_barrier_candidates()
        ],
        _CAPABILITY_SCALED_MFMA: [
            f"{candidate.family}:{candidate.candidate}"
            for candidate in _scaled_mfma_candidates()
        ],
    }
    for name, expected in expected_corpora.items():
        result = capabilities[name]
        if result.get("status") != "known":
            continue
        value = result["value"]
        observed = [
            _candidate_id(entry) for entry in (*value["accepted"], *value["rejected"])
        ]
        if value["corpus"] != expected or sorted(observed) != sorted(expected):
            raise support.ProbeError(
                f"{name} did not report its complete candidate corpus"
            )
        if len(observed) != len(set(observed)):
            raise support.ProbeError(f"{name} reported a candidate more than once")

    layout = capabilities[_CAPABILITY_LAYOUT_POLICY]
    if layout.get("status") == "known":
        entries = layout["value"]["candidates"]
        if [entry["candidate"] for entry in entries] != list(_LAYOUT_CANDIDATES):
            raise support.ProbeError(
                f"{_CAPABILITY_LAYOUT_POLICY} did not report L0 through L3 exactly once"
            )
        for entry in entries:
            outcome = entry["outcome"]
            if outcome in {"accepted", "overwritten"} and not entry.get(
                "observed_layout"
            ):
                raise support.ProbeError(
                    f"{_CAPABILITY_LAYOUT_POLICY} {entry['candidate']} lacks observed_layout"
                )
            if outcome == "rejected" and not entry.get("diagnostic"):
                raise support.ProbeError(
                    f"{_CAPABILITY_LAYOUT_POLICY} {entry['candidate']} lacks diagnostic"
                )

    expected_observations = {
        _CAPABILITY_FRONTEND_FP16: ["canonical_prototype"],
        _CAPABILITY_OPT_LEVELS: ["O0", "O1", "O2", "O3"],
        _CAPABILITY_COMPILER_OPTIONS: [
            "slp_off",
            "mfma_vgpr_form",
            "iterative_ilp_scheduler",
        ],
        _CAPABILITY_FUNCTION_ATTRIBUTES: [
            "waves_per_eu_1",
            "waves_per_eu_2",
            "zero_agpr",
        ],
    }
    processor = report["probe"]["processor"]
    expected_observations[_CAPABILITY_PROCESSOR_NAMES] = ["requested"] + (
        ["generic"] if _generic_processor_name(processor) is not None else []
    )
    for name, expected in expected_observations.items():
        result = capabilities[name]
        if result.get("status") != "known":
            continue
        observed = [entry["candidate"] for entry in result["value"]["observations"]]
        if observed != expected:
            raise support.ProbeError(
                f"{name} did not report its complete observation corpus"
            )


def _validate_report(report: dict[str, object]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise support.ProbeError(
            "schema validation requires jsonschema; install it or omit --validate"
        ) from exc
    schema_path = Path(__file__).with_name("capabilities.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(instance=report, schema=schema)
    except jsonschema.ValidationError as exc:
        raise support.ProbeError(
            f"generated report failed schema validation: {exc.message}"
        ) from exc
    _validate_report_completeness(report)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="probe structural AMD LLVM behavior")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--llvm-bin", type=Path, default=None, help="directory containing clang and opt"
    )
    source.add_argument(
        "--python-env",
        type=Path,
        default=None,
        help="ROCm Python environment containing rocm-sdk-core",
    )
    parser.add_argument("--processor", default="gfx950")
    parser.add_argument("--output", type=Path, default=Path("capabilities.json"))
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "rocke_compiler_capabilities",
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--validate", action="store_true", help="validate the report against the schema"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="write JSON without a summary"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    try:
        llvm_bin = args.llvm_bin
        if llvm_bin is None and args.python_env is not None:
            llvm_bin = _llvm_bin_from_python_env(args.python_env)
        if llvm_bin is None:
            raw_env = os.environ.get("ROCKE_PYTHON_ENV")
            raw_bin = os.environ.get("ROCKE_LLVM_BIN")
            if raw_env:
                llvm_bin = _llvm_bin_from_python_env(Path(raw_env))
            elif raw_bin:
                llvm_bin = Path(raw_bin)
        report, cache_hit = load_or_probe(
            llvm_bin=llvm_bin,
            processor=args.processor,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
        )
        if args.validate:
            _validate_report(report)
        support._write_json(args.output, report)
    except (OSError, ValueError, support.ProbeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        _print_summary(report, cache_hit, args.output)
        if args.validate:
            print("schema validation: passed")
        print(f"elapsed: {time.perf_counter() - started:.3f} s")
    return (
        0
        if all(x.get("status") == "known" for x in report["capabilities"].values())
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
