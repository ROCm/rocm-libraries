# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU P3 checks for the shipped gfx942 dense attention kernel."""

from __future__ import annotations

import numpy as np
import pytest

from rocke.runtime.hip_module import get_device_arch

from ._analytical import (
    P3_BASIS_OFFSETS,
    P3_CAUSAL_FP16,
    P3_WINDOW_BF16,
    P3Case,
    analyze_p3,
    format_measurement,
    p3_inputs,
    p3_tolerance,
)
from ._gfx942_subject import run_dense_case


torch = pytest.importorskip("torch", reason="ROCm torch is required for gfx942 P3")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        get_device_arch() != "gfx942", reason="requires a HIP-visible gfx942 GPU"
    ),
]


def _torch_dtype(case: P3Case):
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[case.dtype]


def _run_basis_windows(case: P3Case) -> dict[int, np.ndarray]:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _torch_dtype(case)
    outputs: dict[int, np.ndarray] = {}
    for offset in P3_BASIS_OFFSETS:
        q_numpy, k_numpy, v_numpy = p3_inputs(case, offset)
        q = torch.from_numpy(q_numpy).to(device=device, dtype=dtype)
        k = torch.from_numpy(k_numpy).to(device=device, dtype=dtype)
        v = torch.from_numpy(v_numpy).to(device=device, dtype=dtype)
        out = torch.full(case.output_shape, float("nan"), device=device, dtype=dtype)
        run_dense_case(case, q, k, v, out)
        outputs[offset] = out.float().cpu().numpy()
    return outputs


def _case_label(case: P3Case) -> str:
    mask = f"window-{case.window}" if case.mask == "window" else case.mask
    return (
        f"P3 {case.dtype} {mask} B{case.batch} "
        f"Sq{case.seqlen_q} Sk{case.seqlen_k} "
        f"H{case.nhead_q}/{case.nhead_k} D{case.head_dim}"
    )


def _focused_summary(errors) -> str:
    return ", ".join(f"q{query}={error:.6g}" for query, error in errors.items())


@pytest.mark.parametrize(
    "case,wrong_rule",
    [
        pytest.param(P3_CAUSAL_FP16, "causal_exclusive", id="fp16-causal"),
        pytest.param(
            P3_WINDOW_BF16,
            "window_inclusive_lower",
            id="bf16-window64",
        ),
    ],
)
def test_p3_uniform_masked_probabilities(case: P3Case, wrong_rule: str) -> None:
    """Expose all 256 probabilities and reject the paired off-by-one model."""
    outputs = _run_basis_windows(case)
    result = analyze_p3(case, outputs)
    label = _case_label(case)
    focus = _focused_summary(result.focused_row_errors)

    assert (
        result.invalid_count == 0
    ), f"{label}: invalid/unwritten={result.invalid_count}"
    assert result.zero_leakage_count == 0, (
        f"{label}: exact masked/basis-tail leakage count="
        f"{result.zero_leakage_count}; focused rows: {focus}"
    )
    element_ceiling = p3_tolerance("element", case.dtype)
    for offset, measurement in result.elements.items():
        assert measurement.maximum_error <= element_ceiling, (
            f"{label} basis_offset={offset}: {format_measurement(measurement)} "
            f"> atol={element_ceiling}; focused rows: {focus}"
        )
    mass_ceiling = p3_tolerance("row_mass", case.dtype)
    assert result.row_mass_error <= mass_ceiling, (
        f"{label}: row_mass_max_abs={result.row_mass_error:.6g} "
        f"> atol={mass_ceiling}; focused rows: {focus}"
    )

    wrong = analyze_p3(case, outputs, rule=wrong_rule)
    wrong_error = max(item.maximum_error for item in wrong.elements.values())
    assert wrong_error > element_ceiling, (
        f"{label}: checker failed to reject {wrong_rule!r}; "
        f"wrong_model_max_abs={wrong_error:.6g} <= atol={element_ceiling}"
    )
