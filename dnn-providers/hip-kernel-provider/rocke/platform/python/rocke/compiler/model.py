# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared names and typed report model for compiler evidence and policy."""

from __future__ import annotations

from typing import Literal, TypedDict

__all__ = [
    "CAPABILITY_BUFFER_RESOURCE",
    "CAPABILITY_FP8_PACKING",
    "CAPABILITY_LAYOUT",
    "CAPABILITY_NAMED_BARRIER",
    "CAPABILITY_PROCESSOR_NAMES",
    "CAPABILITY_SCALED_MFMA",
    "PROBE_PROTOCOL",
    "TARGET_TRIPLE",
    "CapabilityEntry",
    "CapabilityValue",
    "CompilerIdentity",
    "EvidenceReport",
    "Observation",
    "ProbeEnvelope",
]

TARGET_TRIPLE = "amdgcn-amd-amdhsa"
# Bump the protocol schema when outcome classification or required COMGR stages
# change. Corpus request bytes are added separately by corpus.protocol_sha256.
PROBE_PROTOCOL = {
    "schema": "rocke.comgr_probe_protocol/v1",
    "route": (
        "compile_source_to_bc",
        "codegen_bc_to_relocatable",
        "link_relocatable_to_executable",
    ),
    "outcomes": ("accepted", "rejected", "crash", "timeout", "error"),
    "normal_rejection": "comgr_error",
    "fatal_process": "crash",
    "infrastructure_failure": "error",
}
CAPABILITY_LAYOUT = "amdgpu.comgr.explicit_data_layout"
CAPABILITY_BUFFER_RESOURCE = "amdgpu.comgr.intrinsic.make_buffer_resource.abis"
CAPABILITY_FP8_PACKING = "amdgpu.comgr.intrinsic.mfma_fp8_bf8.packed_operand_type"
CAPABILITY_NAMED_BARRIER = "amdgpu.comgr.intrinsic.named_barrier.pointer_address_space"
CAPABILITY_SCALED_MFMA = "amdgpu.comgr.intrinsic.mfma_f8f6f4.call_abi"
CAPABILITY_PROCESSOR_NAMES = "amdgpu.comgr.accepts_processor_name"

Outcome = Literal["accepted", "rejected", "crash", "timeout", "error"]


class CompilerDependency(TypedDict):
    library: str
    sha256: str


class CompilerIdentity(TypedDict):
    route: Literal["comgr"]
    llvm_version: list[int] | None
    version_source: str
    query_library: str | None
    library: str
    sha256: str
    dependencies: list[CompilerDependency]
    dependency_identity_complete: bool


class ProbeEnvelope(TypedDict):
    name: Literal["rocke-comgr-capabilities"]
    implementation_sha256: str
    protocol_sha256: str
    schema_sha256: str
    target_triple: str
    processor: str
    effective_processor: str
    fingerprint: str


class _ObservationRequired(TypedDict):
    family: str
    candidate: str
    outcome: Outcome


class Observation(_ObservationRequired, total=False):
    diagnostic: str
    evidence: dict[str, object]
    failed_stage: str


class CapabilityValue(TypedDict):
    route: Literal["comgr_llvm_ir_to_hsaco"]
    observations: list[Observation]


class CapabilityEntry(TypedDict):
    status: Literal["known"]
    value: CapabilityValue


class EvidenceReport(TypedDict):
    probe: ProbeEnvelope
    compiler: CompilerIdentity
    capabilities: dict[str, CapabilityEntry]
