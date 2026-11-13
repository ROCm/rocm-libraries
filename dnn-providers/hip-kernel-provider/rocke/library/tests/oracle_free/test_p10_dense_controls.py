# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU negative controls of the coordinate-codeword placement check.

Each control launches a dense kernel with one deliberate defect and requires the
analyzer to reject its output through the channel the kill matrix names. An
emitted control is a private build whose lowered IR must differ from the shipped
kernel's; the row-bound control runs the shipped binary on one query row past a
whole tile and must change an exact number of guard elements. The shipped kernel
is launched first on the same case and must pass, so a rejection is the
control's defect and not the kernel's.

The only skip is a processor that is not the case's architecture, through the
shared ``rocke.tests.gpu.current_device_arch_in`` gate, so
``ROCKE_REQUIRE_<ARCH>=1`` turns that skip into a failure.
"""

from __future__ import annotations

import math

import pytest
from rocke.tests.gpu import current_device_arch_in

from ._analytical import UniformCase, encode_bits
from ._p10_analytical import GuardedOutput, StoreSafety, codeword_inputs
from ._p10_controls import (
    P10_BLIND_CONTROL_CASES,
    P10_EMITTED_CONTROL_CASES,
    P10_ROW_BOUND_CASES,
    UNDERRUN_ELEMENTS,
    lowered_ir,
    row_bound_bypass_spec,
    run_p10_control,
    run_row_bound_bypass,
)
from ._p10_policy import (
    P10_KILL_MATRIX,
    Channel,
    CodewordResult,
    P10Mutation,
    analyze_codeword,
    format_codeword_result,
    path_exact_admissible,
)
from ._p10_subject import (
    p10_kernel_key,
    p10_schedule,
    resolve_p10_subject,
    run_p10_case,
)
from .test_p10_dense import _case_label

pytestmark = pytest.mark.gpu


def _control_id(item: tuple) -> str:
    arch, case, *mutation = item
    return "-".join([arch, case.case_id, *(control.value for control in mutation)])


def _judge(case: UniformCase, output: GuardedOutput) -> CodewordResult:
    return analyze_codeword(case, output, path_exact_admissible(case))


def _launch_shipped_and_control(
    arch: str, case: UniformCase, mutation: P10Mutation
) -> tuple[str, CodewordResult]:
    """Launch the shipped kernel, which must pass, then the emitted control.

    The control's IR is compared before anything runs: a control that lowers to
    the shipped IR would launch the shipped kernel under another name."""
    label = f"{_case_label(arch, case)} control {mutation.value}"
    subject = resolve_p10_subject(arch, case)
    assert lowered_ir(arch, case, mutation) != lowered_ir(arch, case, private=False), (
        f"{label}: the control lowers to the shipped kernel's IR"
    )
    q, k, v = (encode_bits(values, case.dtype) for values in codeword_inputs(case))
    shipped = _judge(case, run_p10_case(arch, case, q, k, v, subject=subject))
    assert not shipped.violations, (
        f"{label}: the shipped kernel itself fails; {format_codeword_result(shipped)}"
    )
    control = _judge(case, run_p10_control(arch, case, mutation, q, k, v))
    print(f"{label}: {format_codeword_result(control)}")
    return label, control


@pytest.mark.parametrize("control", P10_EMITTED_CONTROL_CASES, ids=_control_id)
def test_emitted_defect_is_rejected_through_its_channel(
    control: tuple[str, UniformCase, P10Mutation],
) -> None:
    arch, case, mutation = control
    current_device_arch_in((arch,), subject=f"the {arch} dense codeword controls")
    label, result = _launch_shipped_and_control(arch, case, mutation)
    channel = P10_KILL_MATRIX[mutation].channel
    assert channel in result.channels, (
        f"{label}: not rejected through the {channel.value} channel; "
        f"{format_codeword_result(result)}"
    )

    # The two store defects change exact element counts, and nothing else.
    rows = case.batch * case.seqlen_q * case.nhead_q
    if mutation is P10Mutation.DROPPED_STORE:
        spec = resolve_p10_subject(arch, case).tuning_spec.kernel_spec
        dropped = 2 * int(spec.o_store_width) * rows
        assert result.safety == StoreSafety(0, 0, dropped, 0), label
        assert result.channels == frozenset({Channel.UNWRITTEN}), label
    if mutation is P10Mutation.BASE_UNDERRUN:
        assert result.safety == StoreSafety(
            UNDERRUN_ELEMENTS, 0, UNDERRUN_ELEMENTS, 0
        ), label


@pytest.mark.parametrize("control", P10_BLIND_CONTROL_CASES, ids=_control_id)
def test_head_swap_inside_a_kv_group_is_invisible(
    control: tuple[str, UniformCase, P10Mutation],
) -> None:
    """The probe's stated blind spot, on a real mutant: the query heads of one KV
    group store identical values, so swapping two of them passes the verdict."""
    arch, case, mutation = control
    current_device_arch_in((arch,), subject=f"the {arch} dense codeword controls")
    label, result = _launch_shipped_and_control(arch, case, mutation)
    assert not result.violations, f"{label}: {format_codeword_result(result)}"
    assert result.compared == math.prod(case.output_shape), label


@pytest.mark.parametrize("control", P10_ROW_BOUND_CASES, ids=_control_id)
def test_one_row_past_a_query_tile_overruns_by_the_rest_of_the_tile(
    control: tuple[str, UniformCase],
) -> None:
    """The kernel stores without a row bound. With one query row past a whole
    tile, every logical row is stored correctly and the other ``block_m - 1``
    rows of the last tile land after the tensor, inside the guard."""
    arch, case = control
    current_device_arch_in((arch,), subject=f"the {arch} dense codeword controls")
    label = f"{_case_label(arch, case)} control row_bound_bypass"
    subject = resolve_p10_subject(arch, case)
    q, k, v = (encode_bits(values, case.dtype) for values in codeword_inputs(case))
    shipped = _judge(case, run_p10_case(arch, case, q, k, v, subject=subject))
    assert not shipped.violations, (
        f"{label}: the shipped kernel itself fails; {format_codeword_result(shipped)}"
    )
    # The control then runs that same compiled kernel: its spec has the shipped
    # launcher-cache key.
    spec = row_bound_bypass_spec(arch, case)
    shipped_key = p10_kernel_key(arch, subject.tuning_spec.kernel_spec)
    assert p10_kernel_key(arch, spec) == shipped_key, label

    ragged, output = run_row_bound_bypass(arch, case)
    result = _judge(ragged, output)
    print(f"{label}: {format_codeword_result(result)}")
    block_m = p10_schedule(subject).block_m
    overrun = (block_m - 1) * case.nhead_q * case.head_dim
    assert result.safety == StoreSafety(0, overrun, 0, 0), (
        f"{label}: expected exactly {overrun} guard elements changed after the "
        f"tensor and nothing else; {format_codeword_result(result)}"
    )
    assert result.channels == frozenset({Channel.SUFFIX}), label
    assert result.compared == math.prod(ragged.output_shape), label
