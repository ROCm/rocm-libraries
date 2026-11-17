# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bounded LLVM module corpus for measuring COMGR behavior."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from . import _support as support
from .model import (
    CAPABILITY_BUFFER_RESOURCE,
    CAPABILITY_FP8_PACKING,
    CAPABILITY_LAYOUT,
    CAPABILITY_NAMED_BARRIER,
    CAPABILITY_PROCESSOR_NAMES,
    CAPABILITY_SCALED_MFMA,
    PROBE_PROTOCOL,
    TARGET_TRIPLE,
)

__all__ = [
    "GENERIC_PROCESSORS",
    "LAYOUT_CANDIDATES",
    "expected_observations",
    "protocol_sha256",
    "request_key",
    "requests_for_processor",
]

# L0, L1, and L3 are the complete layouts emitted by the current llvm20,
# llvm22, and llvm23/llvm24 profiles, respectively. L2 is the observed
# intermediate layout with ``m:e`` but without p10-p15; no current profile emits
# it. Keep these as probe-owned archaeology rather than importing the policy
# constants whose assumptions this evidence is intended to test.
LAYOUT_CANDIDATES = {
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
GENERIC_PROCESSORS = {
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
        preamble = f'target triple = "{TARGET_TRIPLE}"\n'
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
    preamble = f'target triple = "{TARGET_TRIPLE}"\n'
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
        f'target triple = "{TARGET_TRIPLE}"\n'
        + "".join(definitions)
        + "".join(declarations)
    )


def request_key(capability: str, family: str, candidate: str) -> str:
    return f"{capability}\0{family}\0{candidate}"


def requests_for_processor(
    processor: str,
) -> tuple[list[support.ProbeRequest], dict[str, tuple[str, str, str]]]:
    isa = f"{TARGET_TRIPLE}--{processor}"
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
        key = request_key(capability, family, candidate)
        requests.append(support.ProbeRequest(key, module, request_isa, options))
        metadata[key] = (capability, family, candidate)

    for name, layout in LAYOUT_CANDIDATES.items():
        add(CAPABILITY_LAYOUT, "module", name, _empty_module(layout))

    # Buffer-resource spelling and count width are independent policy axes, so
    # retain their four exact combinations.
    for item in _buffer_resource_candidates():
        add(CAPABILITY_BUFFER_RESOURCE, item.family, item.candidate, item.module())

    # The remaining choices are global lowering policies. Compile every affected
    # intrinsic in one module per candidate: rocKE can select a candidate only
    # when the complete family lowers, and one result avoids repeating the same
    # policy question for every shape or operation.
    for capability, family, candidates in (
        (
            CAPABILITY_FP8_PACKING,
            "all_mfma_fp8_bf8",
            _fp8_packing_candidates(),
        ),
        (
            CAPABILITY_NAMED_BARRIER,
            "all_named_barriers",
            _named_barrier_candidates(),
        ),
        (
            CAPABILITY_SCALED_MFMA,
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

    add(CAPABILITY_PROCESSOR_NAMES, processor, "requested", _empty_module())
    generic = GENERIC_PROCESSORS.get(processor)
    if generic:
        add(
            CAPABILITY_PROCESSOR_NAMES,
            generic,
            "generic",
            _empty_module(),
            request_isa=f"{TARGET_TRIPLE}--{generic}",
        )
    return requests, metadata


def protocol_sha256(processor: str) -> str:
    """Digest the exact compiler questions and stable verdict semantics."""
    requests, metadata = requests_for_processor(processor)
    document = {
        "protocol": PROBE_PROTOCOL,
        "processor": processor,
        "requests": [
            {
                "key": request.key,
                "ir_text": request.ir_text,
                "isa": request.isa,
                "options": request.options,
                "metadata": metadata[request.key],
            }
            for request in requests
        ],
    }
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def expected_observations(processor: str) -> dict[str, list[tuple[str, str]]]:
    expected = {
        CAPABILITY_LAYOUT: [("module", name) for name in LAYOUT_CANDIDATES],
        CAPABILITY_BUFFER_RESOURCE: [
            (item.family, item.candidate) for item in _buffer_resource_candidates()
        ],
        CAPABILITY_FP8_PACKING: [
            ("all_mfma_fp8_bf8", candidate) for candidate in ("v2i32", "i64")
        ],
        CAPABILITY_NAMED_BARRIER: [
            ("all_named_barriers", candidate) for candidate in ("as3", "as15")
        ],
        CAPABILITY_SCALED_MFMA: [
            ("all_mfma_f8f6f4", candidate) for candidate in ("current_9", "legacy_11")
        ],
        CAPABILITY_PROCESSOR_NAMES: [(processor, "requested")],
    }
    generic = GENERIC_PROCESSORS.get(processor)
    if generic:
        expected[CAPABILITY_PROCESSOR_NAMES].append((generic, "generic"))
    return expected
