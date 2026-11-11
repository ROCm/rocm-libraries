# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Explicit weight-gradient coverage for atomic and two-stage workspaces."""

from __future__ import annotations

import math

from ..contract import Case
from ..gradients import operand_shapes, output_shape
from .gradient_plan import Plan, step


def prepare(case: Case, architecture: str) -> Plan:
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
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
    spec = kernel_spec(case, architecture)
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
    values = ConvArgs.from_problem(
        spec.problem,
        direction="wgrad",
        tile_m=spec.tile_m,
        tile_n=spec.tile_n,
        tile_k=spec.tile_k,
    ).to_launch_values(
        0, 0, 0, sizes["dy"], sizes["x"], out_bytes, split_k=spec.split_k, **extra
    )
    grid = (
        math.ceil(spec.wg_N / spec.tile_n),
        math.ceil(spec.wg_M / spec.tile_m),
        spec.grid_groups * spec.split_k,
    )
    steps = [
        step(
            kernel,
            conv_args_signature(
                case.dtype, direction="wgrad", two_stage=spec.two_stage
            ),
            grid,
            (spec.block_size, 1, 1),
            values,
            bindings,
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
        selection=f"explicit-{architecture}-wgrad",
        spec=spec,
        output_fill=0xFF if spec.two_stage else 0,
        scratch_bytes=scratch,
    )


def kernel_spec(case: Case, architecture: str):
    """Retain the six-lane cohort's explicit October 14 execution protocols.

    There is no production wgrad instance list. The split degrees and tile
    choices below retain the previously exercised atomic, two-stage, and merged
    paths without reintroducing a dispatcher policy into the test harness.
    """
    from kernels.common.conv_implicit_gemm import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_wgrad import WgradConvSpec

    if architecture not in ("gfx942", "gfx950"):
        raise ValueError(f"unsupported gradient architecture: {architecture}")
    split_degrees = {
        "padded": 64,
        "pointwise": 64,
        "stride2": 30,
        "dilation2": 64,
        "groups2": 64,
        "odd-channels": 51 if architecture == "gfx950" else 64,
        "depthwise": 64 if architecture == "gfx950" else 16,
    }
    if case.name not in split_degrees:
        raise ValueError(f"no explicit wgrad configuration for {case.id}")
    scalar_b = architecture == "gfx950" and case.name == "odd-channels"
    two_stage = case.name in ("odd-channels", "depthwise")
    atom_mn = 16 if architecture == "gfx942" or scalar_b else 32
    return WgradConvSpec(
        problem=ConvProblem(**case.problem_fields()),
        name="reference_explicit_wgrad",
        data=ConvDataSpec(dtype_a=case.dtype, dtype_b=case.dtype, dtype_d=case.dtype),
        tile_m=64,
        tile_n=32 if scalar_b else 64,
        tile_k=64,
        warp_m=4 if scalar_b else 2,
        warp_n=2,
        warp_tile_m=atom_mn,
        warp_tile_n=atom_mn,
        warp_tile_k=32 if scalar_b else 16,
        wave_size=64,
        pipeline="compv3" if scalar_b else "mem",
        epilogue="default" if two_stage else "cshuffle",
        lds_k_outer=architecture == "gfx950",
        split_k=split_degrees[case.name],
        two_stage=two_stage,
        ws_replicas=8,
        group_merge=4 if architecture == "gfx950" and case.name == "depthwise" else 1,
    )
