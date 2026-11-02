# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU raw-bit uniform-probability checks for the shipped gfx942 dense kernel."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from rocke.runtime.hip_module import get_device_arch

from ._analytical import (
    UNIFORM_BASIS_OFFSETS,
    UNIFORM_CALIBRATION,
    UNIFORM_CAUSAL_FP16,
    UNIFORM_WINDOW_BF16,
    UniformCase,
    UniformMutation,
    UniformObservation,
    UniformResult,
    analyze_uniform_policy,
    format_uniform_result,
    uniform_inputs,
    uniform_inputs_folded,
)
from ._gfx942_subject import (
    ResolvedDenseSubject,
    resolve_dense_subject,
    run_dense_case,
)

torch = pytest.importorskip(
    "torch", reason="ROCm torch is required for the gfx942 uniform-probability check"
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_device_arch() != "gfx942", reason="requires a HIP-visible gfx942 GPU"
    ),
]


def _torch_dtype(case: UniformCase):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]


def _raw_output_bits(out) -> np.ndarray:
    """Copy raw storage bits without normalizing signed zero or NaN payloads."""
    return out.view(torch.int16).cpu().numpy().view(np.uint16).copy()


def _run_exposures(
    case: UniformCase, subject: ResolvedDenseSubject
) -> UniformObservation:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _torch_dtype(case)
    outputs: list[tuple[str, np.ndarray]] = []
    profiles = [
        (f"basis@{offset}", uniform_inputs(case, offset))
        for offset in UNIFORM_BASIS_OFFSETS
    ]
    profiles.append(("folded", uniform_inputs_folded(case)))
    for name, (q_numpy, k_numpy, v_numpy) in profiles:
        q = torch.from_numpy(q_numpy).to(device=device, dtype=dtype)
        k = torch.from_numpy(k_numpy).to(device=device, dtype=dtype)
        v = torch.from_numpy(v_numpy).to(device=device, dtype=dtype)
        out = torch.full(case.output_shape, float("nan"), device=device, dtype=dtype)
        run_dense_case(case, q, k, v, out, subject=subject)
        outputs.append((name, _raw_output_bits(out)))
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


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(UNIFORM_CAUSAL_FP16, id="fp16-causal"),
        pytest.param(UNIFORM_WINDOW_BF16, id="bf16-window64"),
    ],
)
def test_uniform_masked_probabilities(case: UniformCase) -> None:
    """Gate a ratified dense case and always self-test mutation sensitivity."""
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

    _capture_calibration(case, subject, results)

    arch = get_device_arch()
    key = (arch, subject.candidate_name, case.dtype)
    entry = UNIFORM_CALIBRATION.get(key)
    if entry is None:
        pytest.skip(
            f"uniform-probability policy uncalibrated for {key}; "
            f"{format_uniform_result(control)}"
        )
    if not entry.covers(case, subject.fingerprint):
        pytest.skip(
            f"uniform-probability policy scope mismatch for {key}, "
            f"case={case.case_id}; {format_uniform_result(control)}"
        )

    assert (
        not control.violations
    ), f"{_case_label(case)}: {format_uniform_result(control)}"
