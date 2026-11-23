# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fast unit contracts for the backward kernel's counted synchronization.

Exhaustive IR/LDS validation and compiler/ISA/resource checks are explicit
non-discovered gates under ``tests/gates``. This module keeps the value model,
validator couplings, mutation controls, and representative structural checks
that belong in ordinary unit-test discovery.
"""

from __future__ import annotations

import dataclasses

import pytest

from kernels.common import _attention_bwd_counted as counted_mod
from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common._attention_bwd_counted import (
    COUNTED_ISSUES,
    COUNTED_VMCNT_MAX,
    CountedWaits,
)
from kernels.common.attention_bwd import (
    LOOP_SYNCS,
    AttnBwdSpec,
    attn_bwd_arch_facts,
    build_attn_bwd_main,
    check_counted_sync,
    register_estimates,
    validate_attn_bwd_spec,
)

from ._attention_bwd_counted_cases import (
    HEADS,
    MFMA_ARCHS,
    RDNA_ARCHS,
    spec_of,
)
from ._attention_bwd_counted_ir import assert_counted_ir_model, q_loops
from ._attention_bwd_counted_model import counted_ir_problems, counted_isa_problems


def test_values_defaults_and_couplings():
    assert LOOP_SYNCS == ("full", "lds_only", "counted")
    assert COUNTED_ISSUES == ("early", "late")
    base = AttnBwdSpec(head_size=64)
    assert (base.loop_sync, base.counted_issue, base.lb_mirror) == (
        "full",
        "early",
        False,
    )
    spelled = AttnBwdSpec(head_size=64, loop_sync="full", counted_issue="early")
    assert spelled.kernel_name("main") == base.kernel_name("main")
    with pytest.raises(ValueError, match="counted_issue"):
        AttnBwdSpec(head_size=64, counted_issue="late")
    with pytest.raises(ValueError, match="counted_issue"):
        AttnBwdSpec(head_size=64, loop_sync="counted", counted_issue="mid")
    for issue in COUNTED_ISSUES:
        spec = AttnBwdSpec(head_size=64, loop_sync="counted", counted_issue=issue)
        assert spec.kernel_name("main") != base.kernel_name("main")
        for arch in MFMA_ARCHS:
            validate_attn_bwd_spec(spec, arch)
        for arch in RDNA_ARCHS:
            with pytest.raises(ValueError, match="CDNA"):
                validate_attn_bwd_spec(spec, arch)
    counted = AttnBwdSpec(head_size=64, loop_sync="counted")
    for kwargs, match in (
        (dict(ring_depth=2), "ring_depth"),
        (dict(qdo_stage="loads_first"), "qdo_stage"),
        (dict(stats_load="step_top"), "stats_load"),
        (dict(sched="stage_table_atomics", counted_issue="late"), "early"),
        (dict(step_order="head_outer", edge_tiles=True), "grouped head-outer"),
    ):
        for arch in MFMA_ARCHS:
            with pytest.raises(ValueError, match=match):
                validate_attn_bwd_spec(dataclasses.replace(counted, **kwargs), arch)
    for kwargs in (
        dict(step_order="head_outer"),
        dict(edge_tiles=True),
        dict(sched="stage_table"),
        dict(sched="stage_table_atomics"),
    ):
        for arch in MFMA_ARCHS:
            validate_attn_bwd_spec(dataclasses.replace(counted, **kwargs), arch)
    with pytest.raises(ValueError, match="global_path"):
        AttnBwdSpec(head_size=64, loop_sync="counted", global_path="dma")


def test_counted_excludes_barrier_merge():
    merged = AttnBwdSpec(head_size=64, loop_sync="counted", barrier_merge="on")
    with pytest.raises(ValueError, match="barrier_merge"):
        check_counted_sync(merged.resolved("gfx942"), "gfx942")
    for arch in MFMA_ARCHS:
        with pytest.raises(ValueError, match="barrier_merge"):
            validate_attn_bwd_spec(merged, arch)
    off = AttnBwdSpec(head_size=64, loop_sync="counted", barrier_merge="off")
    check_counted_sync(off.resolved("gfx942"), "gfx942")
    full = AttnBwdSpec(head_size=64, barrier_merge="on").resolved("gfx942")
    check_counted_sync(full, "gfx942")


def test_counted_excludes_the_split_dq_main_kernel():
    for arch in MFMA_ARCHS:
        with pytest.raises(ValueError, match="dq_mode='atomic'"):
            validate_attn_bwd_spec(
                AttnBwdSpec(head_size=64, loop_sync="counted", dq_mode="split"),
                arch,
            )
    from dispatch.attention_bwd.tuning_common import backward_axis

    assert not backward_axis("loop_sync").applies("counted", {"dq_mode": "split"})


def test_registry_axes():
    from dispatch.attention_bwd.tuning_common import backward_axis

    loop_sync = backward_axis("loop_sync")
    for arch in MFMA_ARCHS:
        assert loop_sync.values(arch) == ("full", "lds_only", "counted")
    for arch in RDNA_ARCHS:
        assert loop_sync.values(arch) == ("full", "lds_only")
    assert loop_sync.applies("counted", {})
    assert not loop_sync.applies("counted", {"ring_depth": 2})
    assert not loop_sync.applies("counted", {"qdo_stage": "joint"})
    assert loop_sync.applies("counted", {"sched": "stage_table"})
    assert not loop_sync.applies("counted", {"barrier_merge": "on"})
    assert not loop_sync.applies(
        "counted", {"step_order": "head_outer", "edge_tiles": True}
    )
    assert loop_sync.applies("counted", {"step_order": "head_outer"})

    counted_issue = backward_axis("counted_issue")
    assert counted_issue.choices == COUNTED_ISSUES and counted_issue.family == 5
    assert counted_issue.values("gfx1151") == ("early",)
    assert counted_issue.applies("late", {"loop_sync": "counted"})
    stage_table = {"loop_sync": "counted", "sched": "stage_table_atomics"}
    assert counted_issue.applies("early", stage_table)
    assert not counted_issue.applies("late", stage_table)
    assert not counted_issue.applies("late", {"loop_sync": "full"})
    for name in ("qdo_stage", "stats_load"):
        axis = backward_axis(name)
        other = next(value for value in axis.choices if value != axis.default)
        assert axis.applies(other, {})
        assert not axis.applies(other, {"loop_sync": "counted"})


@pytest.mark.parametrize("arch", MFMA_ARCHS)
@pytest.mark.parametrize("head", HEADS)
def test_vmcnt_field_bounds_representative_geometry(arch, head):
    CountedWaits(COUNTED_VMCNT_MAX)
    with pytest.raises(ValueError, match="vmcnt"):
        CountedWaits(COUNTED_VMCNT_MAX + 1)
    spec = validate_attn_bwd_spec(
        AttnBwdSpec(head_size=head, loop_sync="counted"), arch
    )
    assert BwdTileGeometry.from_spec(spec, arch).counted_atomics <= COUNTED_VMCNT_MAX


def test_counted_estimate_carries_the_next_step():
    for arch in MFMA_ARCHS:
        for head in HEADS:
            full = validate_attn_bwd_spec(
                AttnBwdSpec(head_size=head, agpr_alloc=(0, 0)), arch
            )
            counted = dataclasses.replace(full, loop_sync="counted")
            wave = attn_bwd_arch_facts(arch).wave_size
            extra = (
                full.block_m * head // (wave * full.waves)
                + 2 * full.block_m * 16 // wave
            )
            assert (
                register_estimates(counted, arch)["arch_vgpr"]
                == register_estimates(full, arch)["arch_vgpr"] + extra
            )


@pytest.mark.parametrize("arch", MFMA_ARCHS)
@pytest.mark.parametrize("issue", COUNTED_ISSUES)
def test_counted_ir_model_representative(arch, issue):
    spec = spec_of(arch, "default", 64, "fp16", "band_edge", issue)
    kernel = build_attn_bwd_main(spec, arch=arch)
    assert_counted_ir_model(kernel, spec, arch, issue)


def _mutated(monkeypatch, mode):
    original = CountedWaits.barrier

    def barrier(self, builder):
        count = self._since_load
        if mode == "plus1":
            self._since_load = count + 1
        elif mode == "zero":
            self._since_load = 0
        try:
            original(self, builder)
        finally:
            self._since_load = count

    if mode == "lost_load":
        monkeypatch.setattr(CountedWaits, "loads", lambda self: None)
    else:
        monkeypatch.setattr(CountedWaits, "barrier", barrier)


@pytest.mark.parametrize("mode", ("plus1", "zero", "lost_load"))
@pytest.mark.parametrize("issue", COUNTED_ISSUES)
def test_ir_model_catches_wrong_counts(monkeypatch, mode, issue):
    _mutated(monkeypatch, mode)
    spec = spec_of("gfx942", "default", 64, "fp16", "band_edge", issue)
    kernel = build_attn_bwd_main(spec, arch="gfx942")
    found = [
        problem for loop in q_loops(kernel) for problem in counted_ir_problems(loop)[0]
    ]
    if mode == "lost_load" and issue == "late":
        assert any("barrier 2" in problem for problem in found), found
    else:
        assert found, (mode, issue)


def _isa_text(body: list[str], *, rotate: bool = False) -> str:
    """Build a tiny ``llvm-objdump`` style loop for the ISA-model unit test."""
    lines = ["0000000000001000 <k>:"]
    address = 0x1000
    output = []

    def emit(text):
        nonlocal address
        output.append((address, text))
        address += 4

    emit("s_nop 0")
    if rotate:
        emit("s_branch BRANCH_TO_HEADER")
        emit("s_endpgm")
    header = address
    for text in body:
        emit(text)
    emit("s_cbranch_scc1 BACK")
    emit("s_endpgm")
    for current, text in output:
        text = text.replace("BACK", f"0 <k+0x{header - 0x1000:x}>")
        text = text.replace("BRANCH_TO_HEADER", f"0 <k+0x{header - 0x1000:x}>")
        lines.append(f"\t{text} // {current:012X}: 00000000")
    return "\n".join(lines) + "\n"


_GOOD_ISA = [
    "s_waitcnt vmcnt(1) lgkmcnt(0)",
    "s_barrier",
    "global_load_dwordx4 v[0:3], v[4:5], off",
    "v_mfma_f32_16x16x16_f16 v[8:11], v[0:1], v[2:3], v[8:11]",
    "s_waitcnt vmcnt(0) lgkmcnt(0)",
    "s_barrier",
    "global_atomic_add_f32 v[4:5], v6, off",
    "s_waitcnt vmcnt(1) lgkmcnt(0)",
    "s_barrier",
]


@pytest.mark.parametrize("rotate", (False, True))
def test_isa_model_on_a_synthetic_loop(rotate):
    result = counted_isa_problems(_isa_text(_GOOD_ISA, rotate=rotate))
    assert not result["problems"], result
    assert [record["waits"] for record in result["loops"]] == [[1, 0, 1]]

    plus_one = list(_GOOD_ISA)
    plus_one[4] = "s_waitcnt vmcnt(1) lgkmcnt(0)"
    assert any(
        "in flight" in problem
        for problem in counted_isa_problems(_isa_text(plus_one))["problems"]
    )
    zero = list(_GOOD_ISA)
    zero[7] = "s_waitcnt vmcnt(0) lgkmcnt(0)"
    assert any(
        "drains" in problem
        for problem in counted_isa_problems(_isa_text(zero))["problems"]
    )
    early = _GOOD_ISA[:3] + ["s_waitcnt vmcnt(0)"] + _GOOD_ISA[3:]
    result = counted_isa_problems(_isa_text(early))
    assert any(
        "waits for a load of this trip" in problem for problem in result["problems"]
    ), result
    no_barrier = [text for index, text in enumerate(_GOOD_ISA) if index != 5]
    result = counted_isa_problems(_isa_text(no_barrier))
    assert any("barriers per trip" in problem for problem in result["problems"]), result
    dropped = [text for text in _GOOD_ISA if text != "s_barrier"]
    assert not counted_isa_problems(_isa_text(dropped), single_wave=True)["problems"]
    result = counted_isa_problems(_isa_text(dropped))
    assert any("barriers per trip" in problem for problem in result["problems"]), result


def test_model_module_constants():
    assert counted_mod.COUNTED_VMCNT_MAX == 63
