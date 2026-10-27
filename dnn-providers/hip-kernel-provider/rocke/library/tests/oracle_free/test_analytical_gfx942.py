# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU raw-bit P3 policy checks for the shipped gfx942 dense kernel."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from rocke.runtime.hip_module import get_device_arch

from ._analytical import (
    P3_BASIS_OFFSETS,
    P3_CALIBRATION,
    P3_CAUSAL_FP16,
    P3_WINDOW_BF16,
    P3Case,
    P3Mutation,
    P3Observation,
    P3Result,
    analyze_p3_policy,
    format_p3_result,
    p3_inputs,
    p3_inputs_folded,
)
from ._gfx942_subject import (
    ResolvedDenseSubject,
    resolve_dense_subject,
    run_dense_case,
)

torch = pytest.importorskip("torch", reason="ROCm torch is required for gfx942 P3")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_device_arch() != "gfx942", reason="requires a HIP-visible gfx942 GPU"
    ),
]


def _torch_dtype(case: P3Case):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]


def _raw_output_bits(out) -> np.ndarray:
    """Copy raw storage bits without normalizing signed zero or NaN payloads."""
    return out.view(torch.int16).cpu().numpy().view(np.uint16).copy()


def _run_exposures(case: P3Case, subject: ResolvedDenseSubject) -> P3Observation:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _torch_dtype(case)
    outputs: list[tuple[str, np.ndarray]] = []
    profiles = [
        (f"basis@{offset}", p3_inputs(case, offset)) for offset in P3_BASIS_OFFSETS
    ]
    profiles.append(("folded", p3_inputs_folded(case)))
    for name, (q_numpy, k_numpy, v_numpy) in profiles:
        q = torch.from_numpy(q_numpy).to(device=device, dtype=dtype)
        k = torch.from_numpy(k_numpy).to(device=device, dtype=dtype)
        v = torch.from_numpy(v_numpy).to(device=device, dtype=dtype)
        out = torch.full(case.output_shape, float("nan"), device=device, dtype=dtype)
        run_dense_case(case, q, k, v, out, subject=subject)
        outputs.append((name, _raw_output_bits(out)))
    return P3Observation(tuple(outputs))


def _case_label(case: P3Case) -> str:
    mask = f"window-{case.window}" if case.mask == "window" else case.mask
    return (
        f"P3 {case.dtype} {mask} B{case.batch} "
        f"Sq{case.seqlen_q} Sk{case.seqlen_k} "
        f"H{case.nhead_q}/{case.nhead_k} D{case.head_dim}"
    )


def _applicable_mutations(case: P3Case) -> tuple[P3Mutation, ...]:
    if case.mask == "causal":
        return (
            P3Mutation.INCLUDE_FUTURE_KEY,
            P3Mutation.EXCLUDE_LAST_KEY,
            P3Mutation.DENOMINATOR_PLUS_ONE,
        )
    return (
        P3Mutation.INCLUDE_OLD_WINDOW_KEY,
        P3Mutation.EXCLUDE_OLDEST_WINDOW_KEY,
        P3Mutation.DENOMINATOR_PLUS_ONE,
    )


def _capture_calibration(
    case: P3Case,
    subject: ResolvedDenseSubject,
    results: dict[str, P3Result],
) -> None:
    output_text = os.environ.get("ROCKE_P3_CALIBRATION_OUT")
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
                        "record_type": "rocke_p3_legal_count",
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


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(P3_CAUSAL_FP16, id="fp16-causal"),
        pytest.param(P3_WINDOW_BF16, id="bf16-window64"),
    ],
)
def test_p3_uniform_masked_probabilities(case: P3Case) -> None:
    """Gate a ratified dense case and always self-test mutation sensitivity."""
    subject = resolve_dense_subject(case)
    observation = _run_exposures(case, subject)
    control = analyze_p3_policy(case, observation, budget=0)
    results = {"none": control}
    for mutation in _applicable_mutations(case):
        mutated = analyze_p3_policy(case, observation, budget=0, mutation=mutation)
        assert mutated.violations, (
            f"{_case_label(case)}: checker failed to reject {mutation.value}; "
            f"{format_p3_result(mutated)}"
        )
        results[mutation.value] = mutated

    _capture_calibration(case, subject, results)

    arch = get_device_arch()
    key = (arch, subject.candidate_name, case.dtype)
    entry = P3_CALIBRATION.get(key)
    if entry is None:
        pytest.skip(f"P3 policy uncalibrated for {key}; {format_p3_result(control)}")
    if not entry.covers(case, subject.fingerprint):
        pytest.skip(
            f"P3 policy scope mismatch for {key}, case={case.case_id}; "
            f"{format_p3_result(control)}"
        )

    assert not control.violations, f"{_case_label(case)}: {format_p3_result(control)}"
