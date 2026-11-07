# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compiler, ISA, and resource gate for counted backward synchronization.

The gate compiles each row of the pinned representative counted ISA subset once
on the active supported LLVM/comgr toolchain. The subset spans every final-ISA
control-flow class while the structural gate owns the complete legal matrix.
Early and late counted issue modes share one canonical
full-barrier twin, so that twin is compiled only on the early row. CTest and
``library/tests/run_all.py`` invoke this module explicitly after a toolchain
preflight.
"""

from __future__ import annotations

import dataclasses

import pytest
from rocke.helpers.compile import compile_kernel

from kernels.common._attention_bwd_counted import COUNTED_ISSUES, CountedWaits
from kernels.common.attention_bwd import (
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    validate_attn_bwd_spec,
)

from tests._attention_bwd_counted_cases import (
    COUNTED_COMPILER_FLAVORS,
    MFMA_ARCHS,
    isa_rows,
    require_explicit_gate_entrypoint,
    row_digest,
    row_id,
    selected_isa_arch,
    spec_of,
)
from tests._attention_bwd_counted_model import counted_isa_problems
from tests._attention_bwd_resources import (
    disassemble,
    ResourceGateResult,
    evaluate_resource_gate,
)

require_explicit_gate_entrypoint()

_ALL_ROWS = isa_rows()
_SELECTED_ARCH = selected_isa_arch()
_SELECTED_ROWS = tuple(
    row for row in _ALL_ROWS if _SELECTED_ARCH is None or row[0] == _SELECTED_ARCH
)
_EXPECTED_ROWS = 224
_EXPECTED_ROW_DIGEST = (
    "f00533c3a779daa28346751c8edb2a1aab59066cba6e3a54a350e45932045c83"
)
_EXPECTED_FULL_TWINS = 112
_EXPECTED_FULL_TWIN_DIGEST = (
    "487b8cc810cb8836e6cc8b6aeec0bf5dedba2ecc060af526874bf9cc7763309a"
)


def test_gate_manifest_is_stable():
    """Pin the compiler matrix and its canonical full-twin deduplication."""
    assert len(_ALL_ROWS) == _EXPECTED_ROWS
    assert len(set(_ALL_ROWS)) == len(_ALL_ROWS)
    assert row_digest(_ALL_ROWS) == _EXPECTED_ROW_DIGEST
    full_rows = tuple(row for row in _ALL_ROWS if row[-1] == "early")
    assert row_digest(full_rows) == _EXPECTED_FULL_TWIN_DIGEST
    full = {
        (
            row[0],
            dataclasses.replace(spec_of(*row), loop_sync="full", counted_issue="early"),
        )
        for row in _ALL_ROWS
    }
    assert len(full) == _EXPECTED_FULL_TWINS
    assert _SELECTED_ROWS


def _compile(spec, arch):
    kernel = build_attn_bwd_main(spec, arch=arch)
    return compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")


def _assert_resource_gate(
    artifact, spec, arch: str, isa_text: str
) -> ResourceGateResult:
    resolved = validate_attn_bwd_spec(spec, arch)
    facts = attn_bwd_arch_facts(arch)
    result = evaluate_resource_gate(
        artifact.hsaco,
        arch=arch,
        lds_capacity_bytes=facts.lds_capacity_bytes,
        spec=resolved,
        isa_text=isa_text,
    )
    assert result.has_supported_flavor_and_identified_comgr(
        COUNTED_COMPILER_FLAVORS
    ), result.record()
    assert not result.failures and not result.warnings, result.record()
    return result


@pytest.mark.parametrize("row", _SELECTED_ROWS, ids=row_id)
def test_counted_isa_and_resources(row):
    """Check final waits and resources from one compile of each counted object."""
    arch, _, _, _, _, issue = row
    spec = spec_of(*row)
    artifact = _compile(spec, arch)
    isa_text = disassemble(artifact.hsaco, arch=arch)
    resource_result = _assert_resource_gate(artifact, spec, arch, isa_text)
    resources = resource_result.resources
    result = counted_isa_problems(
        isa_text,
        single_wave=resources.max_flat_workgroup_size <= resources.wavefront_size,
    )
    if issue == "early":
        full = dataclasses.replace(spec, loop_sync="full", counted_issue="early")
        full_artifact = _compile(full, arch)
        full_isa = disassemble(full_artifact.hsaco, arch=arch)
        _assert_resource_gate(full_artifact, full, arch, full_isa)

    assert not result["problems"], result
    if issue == "late":
        assert not any(record["vmcnt0"] for record in result["loops"]), result["loops"]


def _mutated(monkeypatch, mode: str) -> None:
    original = CountedWaits.barrier

    def barrier(self, builder):
        count = self._since_load
        self._since_load = count + 1 if mode == "plus1" else 0
        try:
            original(self, builder)
        finally:
            self._since_load = count

    monkeypatch.setattr(CountedWaits, "barrier", barrier)


@pytest.mark.parametrize("mode", ("plus1", "zero"))
@pytest.mark.parametrize("issue", COUNTED_ISSUES)
@pytest.mark.parametrize("arch", MFMA_ARCHS)
def test_isa_model_catches_wrong_counts(monkeypatch, arch, issue, mode):
    if _SELECTED_ARCH not in (None, arch):
        pytest.skip(f"ROCKE_BWD_ISA_ARCH selects {_SELECTED_ARCH}")
    _mutated(monkeypatch, mode)
    spec = spec_of(arch, "default", 64, "fp16", "band", issue)
    artifact = _compile(spec, arch)
    result = counted_isa_problems(disassemble(artifact.hsaco, arch=arch))
    expected = "in flight" if mode == "plus1" else "drains"
    assert any(expected in problem for problem in result["problems"]), result
