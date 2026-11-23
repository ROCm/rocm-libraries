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

Requires a gfx1250 GPU.
"""

from __future__ import annotations

import dataclasses

import pytest
from rocke.runtime.hip_module import get_device_arch
from rocke.tests.lanes import arch_skip_reason

from ._winograd_plan import run_winograd_plan

GPU_ARCH = get_device_arch(0)

pytestmark = pytest.mark.skipif(
    GPU_ARCH != "gfx1250",
    reason=arch_skip_reason(("gfx1250",), "needs a gfx1250 GPU"),
)

_TOL = 5e-2


def _run(req, spec):
    """Run ``spec`` (a Winograd plan selected for ``req``); ``(err, relL2)``."""
    from rocke import compile_kernel

    return run_winograd_plan(
        req,
        spec,
        compile_step=lambda step: compile_kernel(
            step.kernel.build(GPU_ARCH), arch=GPU_ARCH
        ),
        seed=11,
    )


def _request(**kw):
    from dispatch.grouped_convolution import ConvGroupedRequest
    from kernels.conv.kernel_identity import current_llvm_flavor

    # The flavor compile_kernel lowers at (the K3 GEMM needs llvm23).
    return ConvGroupedRequest(
        arch=GPU_ARCH, algorithm="winograd", llvm_flavor=current_llvm_flavor(), **kw
    )


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


@pytest.mark.parametrize(
    "geom, dtype",
    [
        (dict(Hi=1, Wi=157, Y=1, X=5, pad_w=2), "fp16"),
        (dict(Hi=61, Wi=3, Y=5, X=1, pad_h=4, dilation_h=2), "bf16"),
    ],
    ids=["1x5_fp16", "5x1_d2_bf16"],
)
def test_dispatch_winograd_f45(geom, dtype):
    """1x5 / 5x1 filters: the selector's only variant is F(4,5) (unfused)."""
    from dispatch.grouped_convolution import dispatch_conv_grouped

    req = _request(N=2, C=64, K=48, dtype=dtype, **geom)
    res = dispatch_conv_grouped(req)
    plan = res.spec.plan
    assert (plan.kind, plan.variant) == ("unfused_gen", "4")
    err, l2 = _run(req, res.spec)
    assert err < _TOL, f"{res.spec.name}: err={err:.3e} relL2={l2:.3e}"
