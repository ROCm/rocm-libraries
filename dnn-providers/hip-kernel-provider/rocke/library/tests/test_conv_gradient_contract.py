# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Independent gradient identities and production execution-plan contracts."""

import numpy as np
import pytest

from conv_reference.contract import Case, independent_reference as forward_reference
from conv_reference.gradients import independent_reference, make_inputs, check_output
from conv_reference.gradient_cases import DGRAD_CASES, WGRAD_CASES
from conv_reference.implementations import implicit_gemm_dgrad, implicit_gemm_wgrad
from reference_common.numeric import encode, decode


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "geometry", [{}, {"sH": 2, "sW": 2}, {"dH": 2, "pH": 2}, {"groups": 2}]
)
def test_gradient_oracles_satisfy_forward_adjoint_identity(dtype, geometry):
    case = Case("adjoint", dtype, N=2, Hi=4, Wi=5, C=4, K=4, Y=2, X=3, **geometry)
    rng = np.random.default_rng(7)
    # Small integers keep forward storage rounding exact in both dtypes.
    x = encode(rng.integers(-1, 2, case.input_shapes["a"]), dtype)
    w = encode(rng.integers(-1, 2, case.input_shapes["b"]), dtype)
    dy = encode(rng.integers(-1, 2, case.output_shape), dtype)
    y = forward_reference(case, {"a": x, "b": w})
    dx = independent_reference(case, "dgrad", {"dy": dy, "w": w})
    dw = independent_reference(case, "wgrad", {"dy": dy, "x": x})
    expected = np.sum(y * decode(dy, dtype))
    assert np.sum(decode(x, dtype) * dx) == expected
    assert np.sum(decode(w, dtype) * dw) == expected


@pytest.mark.parametrize("direction", ["dgrad", "wgrad"])
def test_gradient_oracle_checks_operands_and_rejects_corrupt_output(direction):
    case = Case("tiny", "fp16", Hi=2, Wi=2, C=2, K=2, Y=1, X=1, pH=0, pW=0)
    inputs = make_inputs(case, direction)
    ref = independent_reference(case, direction, inputs)
    assert check_output(case, encode(ref, case.dtype), ref) < 0.05
    with pytest.raises(AssertionError):
        check_output(case, encode(ref + 100, case.dtype), ref)
    with pytest.raises(ValueError):
        independent_reference(case, direction, {})
    bad = dict(inputs)
    bad["dy"] = np.full(case.output_shape, np.nan, dtype=np.float16)
    with pytest.raises(ValueError, match="invalid gradient operand"):
        independent_reference(case, direction, bad)


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "direction,case",
    [("dgrad", c) for c in DGRAD_CASES] + [("wgrad", c) for c in WGRAD_CASES],
    ids=lambda v: v.id if isinstance(v, Case) else v,
)
def test_gradient_plan_builds_and_captures_every_abi_argument(
    architecture, direction, case
):
    impl = implicit_gemm_dgrad if direction == "dgrad" else implicit_gemm_wgrad
    plan = impl.prepare(case, architecture)
    production = (
        direction == "dgrad"
        and architecture == "gfx950"
        and case.name != "odd-channels"
    )
    expected = (
        "production-implicit-gemm"
        if production
        else (
            f"explicit-{architecture}-64x64x64"
            if direction == "dgrad"
            else f"explicit-{architecture}-wgrad"
        )
    )
    assert plan.selection == expected
    if production:
        assert plan.spec.identity.algorithm == "implicit_gemm"
        assert plan.steps[0].llvm_flavor == plan.spec.identity.llvm_flavor
    for step in plan.steps:
        assert set(step.scalars).isdisjoint(step.bindings)
        assert set(step.scalars) | set(step.bindings) == {
            a["name"] for a in step.signature
        }
        assert all(v > 0 for v in (*step.grid, *step.block))
    if direction == "dgrad":
        assert len(plan.steps) == 1
        assert plan.immutable["sub_gemms"].size > 0
        if case.name == "stride2":
            assert plan.steps[0].scalars["num_sub_gemms"] > 1
    else:
        assert plan.spec.split_k > 1
        if case.name in ("odd-channels", "depthwise"):
            assert plan.spec.two_stage
            assert len(plan.steps) == 2
            assert plan.scratch_bytes["workspace"] > 0
            assert plan.output_fill == 255
        if architecture == "gfx950" and case.name == "depthwise":
            assert plan.spec.group_merge > 1


def test_required_production_dgrad_does_not_fall_back(monkeypatch):
    from dispatch import grouped_convolution

    def refuse(request):
        raise ValueError("required instance missing")

    monkeypatch.setattr(grouped_convolution, "dispatch_conv_grouped", refuse)
    with pytest.raises(ValueError, match="required instance missing"):
        implicit_gemm_dgrad.prepare(DGRAD_CASES[0], "gfx950")
