# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Production gfx950 and explicitly enrolled input-gradient configurations."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from ..architectures.common import convolution_request
from ..contract import Case
from ..gradients import operand_shapes, output_shape
from .gradient_plan import Plan, step


def prepare(case: Case, architecture: str) -> Plan:
    """Use registered instances where enrolled, with no fallback on refusal."""
    from dispatch.grouped_convolution import dispatch_conv_grouped, launch_plan_for
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import ConvDataSpec, ConvProblem
    from kernels.common.conv_implicit_gemm_dgrad import (
        DgradConvSpec,
        build_implicit_gemm_conv_dgrad,
        pack_sub_gemm_buffer,
    )

    if architecture not in ("gfx942", "gfx950"):
        raise ValueError(f"unsupported gradient architecture: {architecture}")
    sizes = {
        name: 2 * math.prod(shape)
        for name, shape in operand_shapes(case, "dgrad").items()
    }
    sizes["dx"] = 2 * math.prod(output_shape(case, "dgrad"))
    if architecture == "gfx950" and case.name != "odd-channels":
        request = replace(convolution_request(case, architecture), direction="dgrad")
        selected = dispatch_conv_grouped(request)
        launch = launch_plan_for(request, selected.spec)
        if len(launch.steps) != 1 or launch.workspaces or not launch.sub_gemm_table:
            raise ValueError(
                "dgrad reference requires one implicit-GEMM kernel and its sub-GEMM table"
            )
        current = launch.steps[0]
        ptrs = dict.fromkeys((*sizes, "sub_gemms"), 0)
        values = launch.values(current, ptrs, sizes)
        return Plan(
            steps=[
                step(
                    selected.build(),
                    current.signature,
                    current.grid,
                    current.block,
                    values,
                    {"dY": "dy", "W": "w", "dX": "out", "sub_gemm_buf": "sub_gemms"},
                    llvm_flavor=selected.spec.identity.llvm_flavor,
                )
            ],
            selection="production-implicit-gemm",
            spec=selected.spec,
            immutable={"sub_gemms": np.asarray(launch.sub_gemm_table, dtype=np.int32)},
        )

    # gfx942 has no production list; gfx950's list does not admit K=33.
    # These named cases retain explicit kernel coverage, not dispatch coverage.
    problem = ConvProblem(**case.problem_fields())
    atom_mn = 16 if architecture == "gfx942" else 32
    spec = DgradConvSpec(
        problem=problem,
        name="reference_explicit_dgrad",
        data=ConvDataSpec(dtype_a=case.dtype, dtype_b=case.dtype, dtype_d=case.dtype),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=atom_mn,
        warp_tile_n=atom_mn,
        warp_tile_k=16,
        wave_size=64,
        pipeline="mem",
        epilogue="default",
        split_k=1,
    )
    records = spec.compute_sub_gemms()
    packed = np.asarray(
        pack_sub_gemm_buffer(records, spec.tile_m, spec.tile_n), dtype=np.int64
    ).astype(np.int32)
    values = ConvArgs.from_problem(
        problem, direction="dgrad", tile_m=spec.tile_m, tile_n=spec.tile_n
    ).to_launch_values(
        0,
        0,
        0,
        sizes["dy"],
        sizes["w"],
        sizes["dx"],
        sub_gemm_buf=0,
        num_sub_gemms=len(records),
    )
    return Plan(
        steps=[
            step(
                build_implicit_gemm_conv_dgrad(spec, arch=architecture),
                conv_args_signature(case.dtype, direction="dgrad"),
                (records[-1].block_end, case.groups, spec.split_k),
                (spec.launch_block_size, 1, 1),
                values,
                {"dY": "dy", "W": "w", "dX": "out", "sub_gemm_buf": "sub_gemms"},
            )
        ],
        selection=f"explicit-{architecture}-64x64x64",
        spec=spec,
        immutable={"sub_gemms": packed},
    )
