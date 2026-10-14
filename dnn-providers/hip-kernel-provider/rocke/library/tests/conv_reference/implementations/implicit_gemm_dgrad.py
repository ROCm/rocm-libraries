# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Production gfx950 and explicit gfx942 input-gradient preparation."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..architectures.common import convolution_request
from ..contract import Case
from ..gradients import operand_shapes, output_shape
from .gradient_plan import Plan, step


def prepare(case: Case, architecture: str) -> Plan:
    from dispatch.grouped_convolution import (
        ConvGroupedSpec,
        dispatch_conv_grouped,
        launch_values_for,
    )
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_implicit_gemm import ConvProblem
    from kernels.common.conv_implicit_gemm_dgrad import (
        build_implicit_gemm_conv_dgrad,
        pack_sub_gemm_buffer,
    )

    request = replace(convolution_request(case, architecture), direction="dgrad")
    if architecture == "gfx950":
        selected = dispatch_conv_grouped(request).spec
        selection = "production"
    elif architecture == "gfx942":
        # There is no gfx942 dgrad production candidate. Keep this choice explicit
        # instead of changing production dispatch solely to enable reference tests.
        selected = ConvGroupedSpec(
            direction="dgrad",
            tile_m=64,
            tile_n=64,
            tile_k=64,
            warp_m=2,
            warp_n=2,
            warp_tile_mn=16,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="default",
            dtype=case.dtype,
            arch=architecture,
            name="reference_explicit_dgrad",
        )
        selection = "explicit-gfx942-64x64x64"
    else:
        raise ValueError(f"unsupported gradient architecture: {architecture}")
    spec = selected.to_dgrad_spec(ConvProblem(**case.problem_fields()))
    kernel = build_implicit_gemm_conv_dgrad(spec, arch=architecture)
    records = spec.compute_sub_gemms()
    packed = np.asarray(
        pack_sub_gemm_buffer(records, spec.tile_m, spec.tile_n), dtype=np.int64
    ).astype(np.int32)
    sizes = {
        name: 2 * math.prod(shape)
        for name, shape in operand_shapes(case, "dgrad").items()
    }
    values = launch_values_for(
        request,
        selected,
        A_ptr=0,
        B_ptr=0,
        D_ptr=0,
        A_bytes=sizes["dy"],
        B_bytes=sizes["w"],
        D_bytes=2 * math.prod(output_shape(case, "dgrad")),
        sub_gemm_buf=0,
        num_sub_gemms=len(records),
    )
    return Plan(
        steps=[
            step(
                kernel,
                conv_args_signature(case.dtype, direction="dgrad"),
                (records[-1].block_end, case.groups, spec.split_k),
                (spec.launch_block_size, 1, 1),
                values,
                {"dY": "dy", "W": "w", "dX": "out", "sub_gemm_buf": "sub_gemms"},
            )
        ],
        selection=selection,
        spec=spec,
        immutable={"sub_gemms": packed},
    )
