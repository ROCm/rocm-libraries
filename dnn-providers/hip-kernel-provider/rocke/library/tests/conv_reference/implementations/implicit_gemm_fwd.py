# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Forward implicit-GEMM build, launch, and export mechanics."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import TYPE_CHECKING

from ..contract import Case

if TYPE_CHECKING:
    from dispatch.grouped_convolution import ConvGroupedRequest
    from dispatch.grouped_convolution import DispatchResult
    from rocke.runtime import KernelLauncher


def prepare(
    case: Case, request: ConvGroupedRequest, selected: DispatchResult
) -> tuple[KernelLauncher, dict]:
    """Build the selected forward kernel and freeze its complete launch ABI."""
    from dispatch.grouped_convolution import launch_values_for
    from kernels.common.conv_implicit_gemm import ConvProblem, build_implicit_gemm_conv
    from rocke import compile_kernel
    from rocke.runtime import KernelLauncher

    spec = selected.spec.to_fwd_spec(ConvProblem(**case.problem_fields()))
    kernel = build_implicit_gemm_conv(spec, arch=request.arch)
    artifact = compile_kernel(kernel, arch=request.arch, backend="python")
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=selected.signature,
    )
    sizes = {name: 2 * math.prod(shape) for name, shape in case.input_shapes.items()}
    # Capture every scalar ABI argument once. Replay substitutes only pointers;
    # it never imports current convolution geometry or dispatch code.
    values = launch_values_for(
        request,
        selected.spec,
        A_ptr=0,
        B_ptr=0,
        D_ptr=0,
        A_bytes=sizes["a"],
        B_bytes=sizes["b"],
        D_bytes=2 * math.prod(case.output_shape),
    )
    for name in ("A", "B", "D"):
        del values[name]
    metadata = {
        "name": launcher.kernel_name,
        "signature": launcher.signature,
        "grid": selected.grid,
        "block": selected.block,
        "scalars": values,
        "bindings": {"A": "a", "B": "b", "D": "out"},
        "dispatch_spec": asdict(selected.spec),
    }
    return launcher, metadata


def launch(prepared: tuple[KernelLauncher, dict], buffers: dict) -> None:
    from rocke.runtime import LaunchConfig

    launcher, metadata = prepared
    values = dict(metadata["scalars"])
    values.update(
        {name: buffers[operand] for name, operand in metadata["bindings"].items()}
    )
    launcher(
        values,
        config=LaunchConfig(
            grid=tuple(metadata["grid"]), block=tuple(metadata["block"]), fence=True
        ),
    )


def exported_kernel(prepared: tuple[KernelLauncher, dict]) -> tuple[bytes, dict]:
    launcher, metadata = prepared
    return launcher._hsaco, metadata
