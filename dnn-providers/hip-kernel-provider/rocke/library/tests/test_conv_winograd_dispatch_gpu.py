# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numeric test of the opt-in Winograd forward route of the
grouped-convolution dispatcher: ``dispatch_conv_grouped`` (pinned
``algorithm="winograd"``) -> ``launch_plan_for`` -> compile every step ->
launch the steps in order -> compare y with the conv reference, using the
conv benchmarks' metric max|out - ref| / max(max|ref|, 1) < 5e-2.

The workspaces and y are poisoned (bytes 0x7F) first, so an element no kernel
writes shows up. Host copies go through pinned memory with the device drained
around each one.

Requires a gfx1250 GPU and torch.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
if _HAS_TORCH:
    import torch

    torch.cuda.is_available()  # let torch claim the HIP context first

from rocke.runtime.hip_module import get_device_arch  # noqa: E402

GPU_ARCH = get_device_arch(0)

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH or GPU_ARCH != "gfx1250",
    reason="needs torch and a gfx1250 GPU",
)

_TOL = 5e-2


def _u8(t):
    import ctypes

    return (ctypes.c_uint8 * t.nbytes).from_address(t.data_ptr())


def _run(req, spec):
    """Run ``spec`` (a Winograd plan selected for ``req``); ``(err, relL2)``."""
    from benchmarks.common.benchmark_winograd_conv import (
        make_inputs,
        reference,
        verify_error,
    )
    from dispatch.grouped_convolution import launch_plan_for
    from rocke import compile_kernel
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig

    problem, dtype = spec.problem, spec.dtype
    plan = launch_plan_for(req, spec)
    launchers = []
    for step in plan.steps:
        art = compile_kernel(step.kernel.build(GPU_ARCH), arch=GPU_ARCH)
        launchers.append(
            KernelLauncher(
                hsaco=art.hsaco, kernel_name=art.kernel_name, signature=step.signature
            )
        )
    x_t, w_t, y_t = make_inputs(problem, dtype, seed=11)
    ref = reference(x_t, w_t, problem, dtype)
    sizes = {"x": x_t.nbytes, "w": w_t.nbytes, "y": y_t.nbytes, **plan.workspaces}
    rt = Runtime()
    ptrs = {k: int(rt.alloc(max(int(v), 4))) for k, v in sizes.items()}
    try:
        rt.sync()
        for key, t in (("x", x_t), ("w", w_t)):
            h = t.contiguous().pin_memory()
            rt.memcpy_h2d(ptrs[key], _u8(h), h.nbytes)
            rt.sync()
            del h
        for key in (*plan.workspaces, "y"):
            rt.memset(ptrs[key], 0x7F, sizes[key])
        rt.sync()
        for step, lau in zip(plan.steps, launchers):
            lau(
                plan.values(step, ptrs, sizes),
                config=LaunchConfig(grid=step.grid, block=step.block, fence=True),
            )
        rt.sync()
        out = torch.empty_like(y_t, device="cpu").pin_memory()
        rt.memcpy_d2h(_u8(out), ptrs["y"], out.nbytes)
        rt.sync()
    finally:
        rt.sync()
        for p in ptrs.values():
            rt.free(p)
    return verify_error(out, ref)


def _request(**kw):
    from dispatch.grouped_convolution import ConvGroupedRequest

    return ConvGroupedRequest(arch=GPU_ARCH, algorithm="winograd", **kw)


def test_dispatch_winograd_3x3_fp16():
    from dispatch.grouped_convolution import dispatch_conv_grouped

    req = _request(N=2, C=64, K=64, Hi=16, Wi=16, Y=3, X=3, pad_h=1, pad_w=1)
    res = dispatch_conv_grouped(req)
    assert res.spec.plan.kind == "unfused2d"
    err, l2 = _run(req, res.spec)
    assert err < _TOL, f"{res.spec.name}: err={err:.3e} relL2={l2:.3e}"


def test_dispatch_winograd_1x11_fp16():
    from dispatch.grouped_convolution import conv_grouped_sweep_space

    req = _request(N=2, C=64, K=64, Hi=1, Wi=300, Y=1, X=11, pad_w=5)
    specs = conv_grouped_sweep_space(dataclasses.replace(req, algorithm="auto"))
    kinds = {}
    for s in specs:
        kinds.setdefault(s.plan.kind, s)
    assert set(kinds) == {"fused1d", "unfused_gen"}
    for spec in kinds.values():
        err, l2 = _run(req, spec)
        assert err < _TOL, f"{spec.name}: err={err:.3e} relL2={l2:.3e}"
