# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU raw-bit uniform-probability checks for the shipped gfx942 dense kernel.

Inputs are encoded exactly into raw FP16/BF16 bits, uploaded into HIP device
buffers and launched through the production pointer entry; outputs come back as
raw bits. Only numpy and the rocKE HIP runtime are needed, no tensor framework.

The only skip is a processor that is not gfx942. On gfx942 a missing calibration
entry, a calibration scope mismatch, or an observation with no legal coordinate
fails the test, so a dispatch change cannot silently remove the gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from rocke.runtime.hip_module import get_device_arch

from ._analytical import (
    UNIFORM_BASIS_OFFSETS,
    UNIFORM_QUICK_CASES,
    UniformCase,
    UniformMutation,
    UniformObservation,
    UniformResult,
    analyze_uniform_policy,
    broadcast_query_head,
    encode_bits,
    format_uniform_result,
    require_calibration,
    uniform_inputs,
    uniform_inputs_folded,
)
from ._gfx942_subject import (
    ResolvedDenseSubject,
    resolve_dense_subject,
    run_dense_case,
)

_ARCH = get_device_arch()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        _ARCH != "gfx942",
        reason=f"requires a HIP-visible gfx942 GPU; HIP reports {_ARCH!r}",
    ),
]


def _run_exposures(
    case: UniformCase, subject: ResolvedDenseSubject
) -> UniformObservation:
    outputs: list[tuple[str, np.ndarray]] = []
    profiles = [
        (f"basis@{offset}", uniform_inputs(case, offset))
        for offset in UNIFORM_BASIS_OFFSETS
    ]
    profiles.append(("folded", uniform_inputs_folded(case)))
    for name, (q_values, k_values, v_values) in profiles:
        out = run_dense_case(
            case,
            encode_bits(q_values, case.dtype),
            encode_bits(k_values, case.dtype),
            encode_bits(v_values, case.dtype),
            subject=subject,
        )
        outputs.append((name, out))
    return UniformObservation(tuple(outputs))


def _case_label(case: UniformCase) -> str:
    mask = f"window-{case.window}" if case.mask == "window" else case.mask
    return (
        f"uniform {case.dtype} {mask} B{case.batch} "
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


def _capture_calibration(
    case: UniformCase,
    subject: ResolvedDenseSubject,
    results: dict[str, UniformResult],
) -> None:
    output_text = os.environ.get("ROCKE_ORACLE_FREE_CALIBRATION_OUT")
    if not output_text:
        return
    output_dir = Path(output_text).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.case_id}.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for mutation, result in results.items():
            for exposure, groups in result.per_nq.items():
                for legal_count, stats in groups.items():
                    record = {
                        "record_type": "rocke_uniform_legal_count",
                        "arch": "gfx942",
                        "candidate_name": subject.candidate_name,
                        "case_id": case.case_id,
                        "dtype": case.dtype,
                        "mask": case.mask,
                        "mutation": mutation,
                        "exposure": exposure,
                        "legal_count": legal_count,
                        "rows": stats.rows,
                        "compared": stats.compared,
                        "exact": stats.exact,
                        "max_ulp": stats.max_ulp,
                        "max_stored_sum_residual": stats.max_stored_sum_residual,
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")


@pytest.mark.parametrize("case", UNIFORM_QUICK_CASES, ids=lambda case: case.case_id)
def test_uniform_masked_probabilities(case: UniformCase) -> None:
    """Gate a ratified dense case and always self-test mutation sensitivity.

    Mutation self-checks and calibration capture run before the gate, so an
    uncalibrated or rescoped case still records what re-ratification needs."""
    subject = resolve_dense_subject(case)
    observation = _run_exposures(case, subject)
    control = analyze_uniform_policy(case, observation, budget=0)
    results = {"none": control}
    for mutation in _applicable_mutations(case):
        mutated = analyze_uniform_policy(case, observation, budget=0, mutation=mutation)
        assert mutated.violations, (
            f"{_case_label(case)}: checker failed to reject {mutation.value}; "
            f"{format_uniform_result(mutated)}"
        )
        results[mutation.value] = mutated
    # Head sensitivity on the same real output: head 0 written into every head
    # (one V head read for all, or one output head broadcast) must be rejected.
    broadcast = analyze_uniform_policy(
        case, broadcast_query_head(observation, 0), budget=0
    )
    assert broadcast.violations, (
        f"{_case_label(case)}: checker failed to reject a head-0 broadcast; "
        f"{format_uniform_result(broadcast)}"
    )

    _capture_calibration(case, subject, results)

    require_calibration(
        _ARCH,
        subject.candidate_name,
        case,
        subject.fingerprint,
        detail=format_uniform_result(control),
    )
    assert control.compared_count > 0, (
        f"{_case_label(case)}: no legal coordinate compared; "
        f"{format_uniform_result(control)}"
    )
    assert (
        not control.violations
    ), f"{_case_label(case)}: {format_uniform_result(control)}"
