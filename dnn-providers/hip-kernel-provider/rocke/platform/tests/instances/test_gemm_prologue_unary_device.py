# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the GEMM prologue's operand operators (``Prologue.unary_a`` /
``unary_b``) on gfx950: every hipTensor unary operator on f32 operands (the pair view with the
``x6`` split, operator before the split) and on bf16 operands (``cvt``: operator in f32, rounded
once to bf16).

``D = op_a(A) op_b(B)^T``. The reference applies the operator in f64 to the stored operand and
rounds the result once to f32 (to bf16 for the bf16 operands), then sums in f64; the bound is the
f32-class bound of the route plus the documented error of each operator (ulps of f32), on
inputs of each operator's domain plus a column of special values whose products must reach D
exactly as NaN / infinity where the reference has them.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import pytest

# The device through torch (the references are torch tensors), before rocKE's runtime starts: a
# HIP runtime started first by rocKE can leave torch's own without a visible device, while rocKE
# reuses the HIP library of an imported torch.
_ARCH = None
try:
    import torch as _torch

    if _torch.cuda.is_available():
        _ARCH = _torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
except Exception:  # noqa: BLE001 - no torch / no device
    _ARCH = None

pytestmark = pytest.mark.skipif(
    _ARCH != "gfx950", reason=f"needs a gfx950 device; device is {_ARCH}"
)

#: max error of the f32 operator in ulps of f32 (programmer's guide; the trigonometric and
#: hyperbolic operators, their inverses, ceil and floor are rounded once from f64)
ULPS = {
    "relu": 0.0,
    "abs": 0.0,
    "neg": 0.0,
    "ceil": 0.0,
    "floor": 0.0,
    "sqrt": 0.5,
    "rcp": 0.5,
    "exp": 1.0,
    "log": 2.25,
    "tanh": 2.0,
    "sigmoid": 3.0,
}
OPS = [
    "relu", "abs", "neg", "sqrt", "rcp", "exp", "log", "sigmoid", "tanh", "sin", "cos", "tan",
    "sinh", "cosh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "ceil", "floor",
]  # fmt: skip


def _domain(op, n, g, torch):
    u = torch.rand(n, generator=g, device="cuda", dtype=torch.float64)
    if op in ("sqrt", "log"):
        return torch.exp2(u * 20 - 10)
    if op == "rcp":
        return torch.where(u < 0.5, -1.0, 1.0) * torch.exp2(torch.rand_like(u) * 12 - 6)
    if op in ("asin", "acos", "atanh"):
        return u * 1.9 - 0.95
    if op == "acosh":
        return 1 + torch.exp2(u * 12 - 8)
    if op in ("exp", "sinh", "cosh"):
        return u * 16 - 8
    if op in ("sin", "cos"):
        return u * 200 - 100
    if op == "tan":
        return u * 3 - 1.5
    if op in ("ceil", "floor"):
        return u * 40 - 20
    return u * 8 - 4


def _ref(op, x, torch):
    f = {
        "relu": lambda t: torch.where(t > 0, t, torch.zeros_like(t)),
        "abs": torch.abs,
        "neg": torch.neg,
        "sqrt": torch.sqrt,
        "rcp": torch.reciprocal,
        "exp": torch.exp,
        "log": torch.log,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "ceil": torch.ceil,
        "floor": torch.floor,
    }.get(op) or getattr(torch, op)
    return f(x)


def _launcher(kd):
    from rocke.core.ir import PtrType
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.launcher import KernelLauncher

    sig = []
    for prm in kd.params:
        nm = getattr(prm.type, "name", "") or ""
        if isinstance(prm.type, PtrType):
            sig.append(
                {
                    "name": prm.name,
                    "type": nm if nm.startswith("ptr<") else f"ptr<{nm}>",
                }
            )
        else:
            sig.append({"name": prm.name, "type": str(nm)})
    art = compile_kernel(kd, arch="gfx950", capture_ir_text=False, backend="python")
    return KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name, signature=sig)


def _run(ua, ub, *, bf16=False, M=200, N=136, K=96, seed=3):
    """(bad outputs, worst bound ratio) of D = ua(A) ub(B)^T."""
    import torch
    from rocke.dispatch.gemm.cf32 import plan_f32_gemm
    from rocke.instances.common.gemm_prologue import attach_prologue, get_prologue
    from rocke.instances.common.gemm_universal import build_universal_gemm
    from rocke.runtime.launcher import LaunchConfig

    plan = plan_f32_gemm(M, N, K // 2 if bf16 else K, body="RIW")
    spec = copy.copy(plan.spec)
    pro = get_prologue(plan.spec)
    if bf16:
        pro = dataclasses.replace(pro, src="same", recipe="cvt", psub=False)
    pro = dataclasses.replace(pro, unary_a=ua, unary_b=ub, fast_exp=False)
    attach_prologue(spec, pro)
    kd = build_universal_gemm(spec, "gfx950")
    L = _launcher(kd)
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    A = _domain(ua or "x", M * K, g, torch).view(M, K)
    B = _domain(ub or "x", N * K, g, torch).view(N, K)
    st = torch.bfloat16 if bf16 else torch.float32
    As, Bs = A.to(st), B.to(st)
    D = torch.full((M, N), float("nan"), device="cuda")
    vals = {
        "A": As if bf16 else As.view(torch.bfloat16),
        "B": Bs if bf16 else Bs.view(torch.bfloat16),
        "C": D,
        **plan.dims,
    }
    L(
        {p.name: vals[p.name] for p in kd.params},
        config=LaunchConfig(grid=plan.grid, block=plan.block, fence=False),
    )
    torch.cuda.synchronize()

    def opd(op, x):
        y = x.double() if not op else _ref(op, x.double(), torch)
        y = y.float()  # the operator's f32 result
        return (y.to(torch.bfloat16) if bf16 else y).double()

    Ar, Br = opd(ua, As), opd(ub, Bs)
    ref = Ar @ Br.t()
    bnd = Ar.abs() @ Br.abs().t()
    op_err = (ULPS.get(ua, 0.501) if ua else 0) + (ULPS.get(ub, 0.501) if ub else 0)
    u_in = 2.0**-8 if bf16 else 0.0  # the bf16 rounding of an operator's f32 result
    tol = (
        2 * 2.0**-24 * ref.abs()
        + bnd * (2.0**-23 + 2 * K * 2.0**-23 + op_err * 2.0**-23 + 2 * u_in)
        + 2.0**-126
    )
    got = D.double()
    q = torch.nan_to_num((got - ref).abs() / tol, nan=float("inf"))
    return int((q > 1).sum()), float(q.max())


@pytest.mark.parametrize("op", OPS)
def test_operator_on_a_f32(op):
    bad, worst = _run(op, "")
    assert bad == 0, (op, worst)


@pytest.mark.parametrize(
    "ua,ub", [("sin", "cos"), ("atan", "exp"), ("acosh", "tanh"), ("floor", "asinh")]
)
def test_operators_on_both_f32(ua, ub):
    bad, worst = _run(ua, ub, seed=11)
    assert bad == 0, (ua, ub, worst)


@pytest.mark.parametrize("op", ["sin", "sigmoid", "acos", "ceil", "log"])
def test_operator_on_a_bf16(op):
    # bf16 pairs: K a multiple of the route's K tile in both views (no K padding)
    bad, worst = _run(op, "", bf16=True, K=128, seed=5)
    assert bad == 0, (op, worst)


def test_nan_results_reach_d():
    """NaN operator results (sqrt(-1), asin(2), acosh(1/2), log(-1)) reach exactly their row of
    D. (An infinite f32 operand becomes NaN in the x6 split, inf - inf, whatever its source: the
    route's ``cap`` knob, not the operators.)"""
    import torch
    from rocke.dispatch.gemm.cf32 import plan_f32_gemm
    from rocke.instances.common.gemm_prologue import attach_prologue, get_prologue
    from rocke.instances.common.gemm_universal import build_universal_gemm
    from rocke.runtime.launcher import LaunchConfig

    M, N, K = 64, 64, 32
    for op, x, want in (
        ("sqrt", -1.0, math.nan),
        ("asin", 2.0, math.nan),
        ("acosh", 0.5, math.nan),
        ("log", -1.0, math.nan),
        ("atanh", -3.0, math.nan),
    ):
        plan = plan_f32_gemm(M, N, K, body="RIW")
        spec = copy.copy(plan.spec)
        attach_prologue(spec, dataclasses.replace(get_prologue(plan.spec), unary_a=op))
        kd = build_universal_gemm(spec, "gfx950")
        A = torch.full((M, K), 0.5, device="cuda")
        if op == "acosh":
            A.fill_(2.0)
        A[3, 7] = x
        B = torch.full((N, K), 0.25, device="cuda")
        D = torch.zeros((M, N), device="cuda")
        vals = {
            "A": A.view(torch.bfloat16),
            "B": B.view(torch.bfloat16),
            "C": D,
            **plan.dims,
        }
        _launcher(kd)(
            {p.name: vals[p.name] for p in kd.params},
            config=LaunchConfig(grid=plan.grid, block=plan.block, fence=False),
        )
        torch.cuda.synchronize()
        row = D[3].double().cpu()
        if math.isnan(want):
            assert bool(torch.isnan(row).all()), (op, row[:4])
        else:
            assert bool((row == want).all()), (op, row[:4])
        others = torch.cat([D[:3], D[4:]]).double()
        assert bool(torch.isfinite(others).all()), op
