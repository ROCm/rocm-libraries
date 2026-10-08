# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 forward cohort using production grouped-convolution dispatch."""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import asdict
from typing import TYPE_CHECKING

from ...contract import Case

if TYPE_CHECKING:
    from rocke.runtime import KernelLauncher

NAME = "gfx942"
FAMILIES = ("gfx942", "gfx94x")
CASES = tuple(
    Case(name=name, dtype=dtype, **geometry)
    for name, geometry in (
        ("padded", {}),
        ("pointwise", {"Y": 1, "X": 1, "pH": 0, "pW": 0}),
        ("stride2", {"sH": 2, "sW": 2}),
        ("dilation2", {"dH": 2, "dW": 2, "pH": 2, "pW": 2}),
        ("groups2", {"groups": 2}),
        ("asymmetric", {"Hi": 7, "Wi": 11, "N": 2}),
    )
    for dtype in ("fp16", "bf16")
)
CASE_BY_ID = {case.id: case for case in CASES}


def prepare(case: Case, library_root: str) -> tuple[KernelLauncher, dict]:
    """Build the current production-selected Python kernel and its AOT ABI."""
    import kernels
    import dispatch
    from dispatch.grouped_convolution import (
        ConvGroupedRequest,
        dispatch_conv_grouped,
        launch_values_for,
    )
    from kernels.common.conv_implicit_gemm import ConvProblem, build_implicit_gemm_conv
    from rocke import compile_kernel
    from rocke.runtime import KernelLauncher

    root = Path(library_root).resolve()
    for package in (kernels, dispatch):
        if not Path(package.__file__).resolve().is_relative_to(root):
            raise RuntimeError(
                "worker imported production packages outside the selected library"
            )
    req = ConvGroupedRequest(
        N=case.N,
        C=case.C,
        K=case.K,
        Hi=case.Hi,
        Wi=case.Wi,
        Y=case.Y,
        X=case.X,
        G=case.groups,
        stride_h=case.sH,
        stride_w=case.sW,
        pad_h=case.pH,
        pad_w=case.pW,
        dilation_h=case.dH,
        dilation_w=case.dW,
        dtype=case.dtype,
        arch=NAME,
    )
    selected = dispatch_conv_grouped(req)
    spec = selected.spec.to_fwd_spec(ConvProblem(**case.problem_fields()))
    kernel = build_implicit_gemm_conv(spec, arch=NAME)
    artifact = compile_kernel(kernel, arch=NAME, backend="python")
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=selected.signature,
    )
    sizes = {name: 2 * math.prod(shape) for name, shape in case.input_shapes.items()}
    # Capture every scalar ABI argument once. Replay substitutes only pointers;
    # it never imports current convolution geometry or dispatch code.
    values = launch_values_for(
        req,
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
