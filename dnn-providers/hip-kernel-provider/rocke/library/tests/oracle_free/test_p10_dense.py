# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU raw-bit coordinate-codeword placement checks for the shipped dense kernels.

Zero Q, a finite K and the codeword V are encoded exactly into raw FP16/BF16
bits and launched through the architecture's public dense runner into a guarded
output allocation, which is read back whole as raw bits. Every stored output must
equal its correctly rounded target bit for bit, and the four store-safety counts
must be zero. Only numpy and the rocKE HIP runtime are needed, no tensor
framework.

The only skip is a processor that is not the case's architecture, through the
shared ``rocke.tests.gpu.current_device_arch_in`` gate, so
``ROCKE_REQUIRE_<ARCH>=1`` turns that skip into a failure. On the architecture,
a dispatch outcome outside the derivation's scope fails the test; tuning changes
inside the scope keep running.
"""

from __future__ import annotations

import math

import pytest
from rocke.tests.gpu import current_device_arch_in

from ._analytical import UniformCase, encode_bits
from ._p10_analytical import GuardedOutput, TileSchedule, codeword_inputs
from ._p10_policy import (
    P10_KILL_MATRIX,
    CodewordResult,
    P10Mutation,
    analyze_codeword,
    applicable_mutations,
    apply_mutation,
    format_codeword_result,
    path_exact_admissible,
)
from ._p10_subject import (
    P10_SUBJECT_CASES,
    p10_schedule,
    resolve_p10_subject,
    run_p10_case,
)

pytestmark = pytest.mark.gpu


def _case_label(arch: str, case: UniformCase) -> str:
    mask = f"window-{case.window}" if case.mask == "window" else case.mask
    return (
        f"{arch} codeword {case.dtype} {mask} B{case.batch} "
        f"Sq{case.seqlen_q} Sk{case.seqlen_k} "
        f"H{case.nhead_q}/{case.nhead_k} D{case.head_dim}"
    )


def _not_rejected(
    label: str, mutation: P10Mutation, mutated: CodewordResult, control: CodewordResult
) -> str:
    """Explain a mutation the analyzer did not reject through its channel.

    A placement mutation moves the kernel's own outputs. When the real output
    already fails the verdict, a placement bug in the kernel can cancel the
    mutation, so the control's violations are the diagnosis, not the analyzer."""
    channel = P10_KILL_MATRIX[mutation].channel.value
    if control.violations:
        return (
            f"{label}: {mutation.value} was not rejected through the {channel} "
            f"channel and the kernel output itself fails the verdict, so it may "
            f"cancel {mutation.value}; control: {format_codeword_result(control)}"
        )
    return (
        f"{label}: analyzer failed to reject {mutation.value} through the "
        f"{channel} channel; {format_codeword_result(mutated)}"
    )


def _judge_output(
    label: str, case: UniformCase, schedule: TileSchedule, output: GuardedOutput
) -> CodewordResult:
    """Require the exact verdict on one guarded output, and require every owned
    mutation of that same output to be rejected through its channel."""
    admissible = path_exact_admissible(case)
    control = analyze_codeword(case, output, admissible)
    for mutation in applicable_mutations(case):
        mutant = apply_mutation(case, schedule, mutation, output)
        mutated = analyze_codeword(case, mutant, admissible)
        assert P10_KILL_MATRIX[mutation].channel in mutated.channels, _not_rejected(
            label, mutation, mutated, control
        )

    assert not control.violations, f"{label}: {format_codeword_result(control)}"
    assert control.compared == math.prod(case.output_shape), (
        f"{label}: {control.compared} outputs compared, expected every logical "
        f"output; {format_codeword_result(control)}"
    )
    return control


def _subject_case_id(item: tuple[str, UniformCase]) -> str:
    arch, case = item
    return f"{arch}-{case.case_id}"


@pytest.mark.parametrize("subject_case", P10_SUBJECT_CASES, ids=_subject_case_id)
def test_codeword_placement_and_store_safety(
    subject_case: tuple[str, UniformCase],
) -> None:
    """Gate one dense case on exact stored targets and clean guards, and always
    self-test mutation sensitivity on the kernel's own output."""
    arch, case = subject_case
    current_device_arch_in((arch,), subject=f"the {arch} dense codeword gate")
    label = _case_label(arch, case)
    subject = resolve_p10_subject(arch, case)
    q, k, v = (encode_bits(values, case.dtype) for values in codeword_inputs(case))
    output = run_p10_case(arch, case, q, k, v, subject=subject)
    control = _judge_output(label, case, p10_schedule(subject), output)
    print(f"{label}: {format_codeword_result(control)}")
