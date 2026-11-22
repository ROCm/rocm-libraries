# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-GPU tests for ``rocke.tests.fp_policy``.

A lowered kernel carries no floating-point relaxation; each kind the numerics
policy forbids, edited into that IR, is reported with its kind, and a value
merely named like a flag is not."""

from __future__ import annotations

import pytest

from rocke.core.ir import F32, IRBuilder, PtrType
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.tests.fp_policy import fp_policy_violations


def _lowered_kernel() -> str:
    """``Y[i] = X[i] * X[i] / (X[i] + 1)``: an fmul, an fadd and an fdiv."""
    b = IRBuilder("fp_policy_probe")
    x = b.param("X", PtrType(F32, "global"))
    y = b.param("Y", PtrType(F32, "global"))
    lane = b.thread_id_x()
    value = b.global_load_f32(x, lane)
    ratio = b.fdiv(b.fmul(value, value), b.fadd(value, b.const_f32(1.0)))
    b.global_store(y, lane, ratio, align=4)
    b.ret()
    return lower_kernel_to_llvm(b.kernel, arch="gfx942")


def test_a_lowered_kernel_carries_no_relaxation():
    assert fp_policy_violations(_lowered_kernel()) == []


@pytest.mark.parametrize(
    ("old", "new", "kind"),
    [
        ("= fmul float", "= fmul contract float", "fast-math flags"),
        ("= fadd float", "= fadd nnan ninf float", "fast-math flags"),
        ("= fdiv float", "= fdiv fast float", "fast-math flags"),
        (
            '"uniform-work-group-size"="true"',
            '"uniform-work-group-size"="true" "denormal-fp-math-f32"="preserve-sign"',
            "FP mode attribute denormal-fp-math",
        ),
    ],
)
def test_each_relaxation_is_reported_with_its_kind(old, new, kind):
    ir = _lowered_kernel()
    assert old in ir
    violations = fp_policy_violations(ir.replace(old, new, 1))
    assert len(violations) == 1
    assert violations[0].startswith(f"{kind}: ")


def test_fpmath_metadata_is_reported():
    ir = _lowered_kernel()
    fdiv = next(line for line in ir.splitlines() if "= fdiv float" in line)
    violations = fp_policy_violations(ir.replace(fdiv, f"{fdiv}, !fpmath !0", 1))
    assert violations == [f"!fpmath metadata: {fdiv.strip()}, !fpmath !0"]


def test_a_value_named_like_a_flag_is_not_a_violation():
    ir = _lowered_kernel()
    fmul = next(line for line in ir.splitlines() if "= fmul float" in line)
    result = fmul.split("=")[0].strip()
    assert fp_policy_violations(ir.replace(result, "%fast")) == []
