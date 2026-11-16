# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU raw-bit uniform-probability checks for the shipped dense attention kernels.

Inputs are encoded exactly into raw FP16/BF16 bits, uploaded into HIP device
buffers and launched through the architecture's public dense runner; outputs
come back as raw bits. Only numpy and the rocKE HIP runtime are needed, no
tensor framework.

The only skip is a processor that is not the case's architecture, through the
shared ``rocke.tests.gpu.current_device_arch_in`` gate, so
``ROCKE_REQUIRE_<ARCH>=1`` turns that skip into a failure. On the architecture,
a dispatch outcome outside the proof's scope or an observation with no legal
coordinate fails the test; tuning changes inside the scope keep running.
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.tests.gpu import current_device_arch_in

from ._analytical import (
    UniformCase,
    UniformMutation,
    UniformObservation,
    UniformResult,
    analyze_uniform_policy,
    broadcast_query_head,
    encode_bits,
    format_uniform_result,
    uniform_basis_offsets,
    uniform_inputs,
    uniform_inputs_folded,
)
from ._dense_subject import (
    QUICK_SUBJECT_CASES,
    ResolvedDenseSubject,
    resolve_dense_subject,
    run_dense_case,
)

pytestmark = pytest.mark.gpu


def _run_exposures(
    case: UniformCase, subject: ResolvedDenseSubject
) -> UniformObservation:
    outputs: list[tuple[str, np.ndarray]] = []
    profiles = [
        (f"basis@{offset}", uniform_inputs(case, offset))
        for offset in uniform_basis_offsets(case)
    ]
    profiles.append(("folded", uniform_inputs_folded(case)))
    for name, (q_values, k_values, v_values) in profiles:
        out = run_dense_case(
            subject.arch,
            case,
            encode_bits(q_values, case.dtype),
            encode_bits(k_values, case.dtype),
            encode_bits(v_values, case.dtype),
            subject=subject,
        )
        outputs.append((name, out))
    return UniformObservation(tuple(outputs))


def _case_label(arch: str, case: UniformCase) -> str:
    mask = f"window-{case.window}" if case.mask == "window" else case.mask
    return (
        f"{arch} uniform {case.dtype} {mask} B{case.batch} "
        f"Sq{case.seqlen_q} Sk{case.seqlen_k} "
        f"H{case.nhead_q}/{case.nhead_k} D{case.head_dim}"
    )


def _applicable_mutations(case: UniformCase) -> tuple[UniformMutation, ...]:
    if case.mask == "causal":
        return (
            UniformMutation.INCLUDE_FUTURE_KEY,
            UniformMutation.EXCLUDE_LAST_KEY,
            UniformMutation.DENOMINATOR_PLUS_ONE,
        )
    return (
        UniformMutation.INCLUDE_OLD_WINDOW_KEY,
        UniformMutation.EXCLUDE_OLDEST_WINDOW_KEY,
        UniformMutation.DENOMINATOR_PLUS_ONE,
    )


def _not_rejected(
    label: str, what: str, mutated: UniformResult, control: UniformResult
) -> str:
    """Explain a mutation the checker did not reject, blaming the right side.

    When the real output already breaks the exact policy, a mask or denominator
    bug in the kernel can match the mutation's prediction exactly, so the
    control's violations are the diagnosis, not the checker."""
    if control.violations:
        return (
            f"{label}: {what} was not rejected and the kernel output "
            f"itself fails the exact policy, so it may match {what}; "
            f"control: {format_uniform_result(control)}"
        )
    return f"{label}: checker failed to reject {what}; {format_uniform_result(mutated)}"


def _subject_case_id(item: tuple[str, UniformCase]) -> str:
    arch, case = item
    return f"{arch}-{case.case_id}"


@pytest.mark.parametrize("subject_case", QUICK_SUBJECT_CASES, ids=_subject_case_id)
def test_uniform_masked_probabilities(subject_case: tuple[str, UniformCase]) -> None:
    """Gate one dense case at zero ULP and always self-test mutation sensitivity."""
    arch, case = subject_case
    current_device_arch_in((arch,), subject=f"the {arch} dense uniform gate")
    label = _case_label(arch, case)
    subject = resolve_dense_subject(arch, case)
    observation = _run_exposures(case, subject)
    control = analyze_uniform_policy(case, observation, budget=0)
    for mutation in _applicable_mutations(case):
        mutated = analyze_uniform_policy(case, observation, budget=0, mutation=mutation)
        assert mutated.violations, _not_rejected(
            label, mutation.value, mutated, control
        )
    # Head sensitivity on the same real output: head 0 written into every head
    # (one V head read for all, or one output head broadcast) must be rejected.
    broadcast = analyze_uniform_policy(
        case, broadcast_query_head(observation, 0), budget=0
    )
    assert broadcast.violations, _not_rejected(
        label, "a head-0 broadcast", broadcast, control
    )

    assert (
        control.compared_count > 0
    ), f"{label}: no legal coordinate compared; {format_uniform_result(control)}"
    assert not control.violations, f"{label}: {format_uniform_result(control)}"
