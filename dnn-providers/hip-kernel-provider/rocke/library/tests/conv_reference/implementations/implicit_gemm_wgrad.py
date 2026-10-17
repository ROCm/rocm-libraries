# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Production weight-gradient selection, including two-stage workspaces."""

from __future__ import annotations

from dataclasses import replace
import math

from ..architectures.common import convolution_request
from ..contract import Case
from ..gradients import operand_shapes, output_shape
from .gradient_plan import Plan, step


def prepare(case: Case, architecture: str) -> Plan:
    from dispatch.grouped_convolution import dispatch_conv_grouped, launch_values_for
    from kernels.common.conv_implicit_gemm import ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import build_implicit_gemm_conv_wgrad
    from kernels.common.conv_implicit_gemm_wgrad_two_stage import (
        wgrad_two_stage_workspace_nbytes,
    )
    from kernels.common.conv_wgrad_workspace_reduce import (
        WgradReduceSpec,
        build_conv_wgrad_workspace_reduce,
        wgrad_reduce_grid,
        wgrad_reduce_signature,
    )

    if architecture not in ("gfx942", "gfx950"):
        raise ValueError(f"unsupported gradient architecture: {architecture}")
    request = replace(convolution_request(case, architecture), direction="wgrad")
    selected = dispatch_conv_grouped(request)
    spec = selected.spec.to_wgrad_spec(ConvProblem(**case.problem_fields()))
    kernel = build_implicit_gemm_conv_wgrad(spec, arch=architecture)
    sizes = {
        name: 2 * math.prod(shape)
        for name, shape in operand_shapes(case, "wgrad").items()
    }
    out_bytes = 2 * math.prod(output_shape(case, "wgrad"))
    extra = {}
    bindings = {"dY": "dy", "X": "x", "dW": "out"}
    scratch = {}
    if spec.two_stage:
        scratch["workspace"] = wgrad_two_stage_workspace_nbytes(spec)
        extra = {"ws_ptr": 0, "ws_bytes": scratch["workspace"]}
        bindings["ws_ptr"] = "workspace"
    values = launch_values_for(
        request,
        selected.spec,
        A_ptr=0,
        B_ptr=0,
        D_ptr=0,
        A_bytes=sizes["dy"],
        B_bytes=sizes["x"],
        D_bytes=out_bytes,
        **extra,
    )
    steps = [
        step(
            kernel, selected.signature, selected.grid, selected.block, values, bindings
        )
    ]
    if spec.two_stage:
        reduction = WgradReduceSpec(
            problem=spec.problem,
            dtype_d=case.dtype,
            groups=case.groups,
            ws_replicas=spec.ws_replicas,
        )
        steps.append(
            step(
                build_conv_wgrad_workspace_reduce(reduction, arch=architecture),
                wgrad_reduce_signature(reduction),
                wgrad_reduce_grid(reduction),
                (reduction.block_size, 1, 1),
                {
                    "ws_ptr": 0,
                    "dw_ptr": 0,
                    "wg_M": spec.wg_M,
                    "wg_N": spec.wg_N,
                    "ws_bytes": scratch["workspace"],
                    "dw_bytes": out_bytes,
                    "groups": case.groups,
                },
                {"ws_ptr": "workspace", "dw_ptr": "out"},
            )
        )
    return Plan(
        steps=steps,
        selection="production",
        spec=spec,
        output_fill=0xFF if spec.two_stage else 0,
        scratch_bytes=scratch,
    )
