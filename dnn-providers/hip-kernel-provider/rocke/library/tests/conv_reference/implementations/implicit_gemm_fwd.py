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
    from dispatch.grouped_convolution import launch_plan_for
    from rocke.runtime import KernelLauncher
    from ..architectures.common import compile_reference_kernel

    identity = selected.spec.identity
    if identity.algorithm != "implicit_gemm" or identity.direction != "fwd":
        raise ValueError("forward reference requires an implicit-GEMM forward instance")
    plan = launch_plan_for(request, selected.spec)
    if len(plan.steps) != 1 or plan.workspaces:
        raise ValueError("forward reference requires one kernel without workspace")
    launch = plan.steps[0]
    artifact = compile_reference_kernel(
        selected.build(), request.arch, identity.llvm_flavor
    )
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=launch.signature,
    )
    sizes = {
        "x": 2 * math.prod(case.input_shapes["a"]),
        "w": 2 * math.prod(case.input_shapes["b"]),
        "y": 2 * math.prod(case.output_shape),
    }
    values = plan.values(launch, dict.fromkeys(sizes, 0), sizes)
    for name in ("A", "B", "D"):
        del values[name]
    metadata = {
        "name": launcher.kernel_name,
        "signature": launcher.signature,
        "grid": launch.grid,
        "block": launch.block,
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
