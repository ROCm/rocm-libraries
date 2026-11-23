# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Torch-free GPU gradient checks with small live NumPy oracles.

These are source numerical tests, not pinned baseline-replay tests. Atomic
outputs are checked independently on each of three executions.
"""

import pytest

from conv_reference.contract import Case
from conv_reference.gradient_cases import DGRAD_CASES, WGRAD_CASES
from conv_reference.gradients import make_inputs, independent_reference, check_output
from conv_reference.gradient_runner import execute
from conv_reference.implementations import implicit_gemm_dgrad, implicit_gemm_wgrad


@pytest.fixture(scope="module")
def gradient_architecture(request):
    from reference_common.targets import get_device_processor

    architecture = get_device_processor()
    required = request.config.getoption("--rocke-reference-arch")
    if required and (architecture != required or required not in ("gfx942", "gfx950")):
        pytest.fail(
            f"gradient lane requires gfx942/gfx950 target {required}, found {architecture}"
        )
    if architecture not in ("gfx942", "gfx950"):
        pytest.skip("gradient numerics require gfx942 or gfx950")
    return architecture


@pytest.mark.parametrize(
    "direction,case",
    [("dgrad", c) for c in DGRAD_CASES] + [("wgrad", c) for c in WGRAD_CASES],
    ids=lambda v: v.id if isinstance(v, Case) else v,
)
def test_gradient_against_numpy(gradient_architecture, direction, case):
    implementation = (
        implicit_gemm_dgrad if direction == "dgrad" else implicit_gemm_wgrad
    )
    plan = implementation.prepare(case, gradient_architecture)
    inputs = make_inputs(case, direction)
    reference = independent_reference(case, direction, inputs)
    outputs = execute(
        plan, case, direction, gradient_architecture, inputs, repetitions=3
    )
    assert len(outputs) == 3
    for output in outputs:
        check_output(case, output, reference)
