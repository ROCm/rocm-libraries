# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 forward cohort: the implicit-GEMM conv kernel at its gfx942 configuration.

This cohort pins the gfx942 forward configuration the grouped-convolution
dispatcher used to select, before gfx942 fwd was removed from the dispatcher
(no gfx942 instance list is registered, pending a curated one): a 64x64x64
tile, 2x2 waves, the 16x16x16 MFMA
atom, the ``mem`` pipeline, and the cshuffle epilogue whenever the output store
vectorises. The dispatcher no longer registers gfx942 instances, so this cohort
only certifies ``build_implicit_gemm_conv`` at that fixed configuration -- it
does not certify anything the live dispatcher currently selects.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from reference_common.cases import index_cases

from ...contract import Case

if TYPE_CHECKING:
    from rocke.runtime import KernelLauncher

NAME = "gfx942"
CASES = tuple(
    Case(name=name, dtype=dtype, **geometry)
    for name, geometry in (
        ("padded", {}),
        ("pointwise", {"Y": 1, "X": 1, "pH": 0, "pW": 0}),
        ("stride2", {"sH": 2, "sW": 2}),
        ("dilation2", {"dH": 2, "dW": 2, "pH": 2, "pW": 2}),
        ("groups2", {"groups": 2}),
        ("asymmetric", {"Hi": 7, "Wi": 11, "N": 2}),
        ("multi-tile", {"K": 96}),
        ("odd-channels", {"K": 33}),
        ("odd-input-channels", {"C": 33}),
        ("stem", {"C": 3, "Hi": 16, "Wi": 16}),
        ("large", {"N": 2, "Hi": 14, "Wi": 14, "C": 64, "K": 128}),
        ("long-k", {"Y": 5, "X": 5, "pH": 2, "pW": 2, "sH": 2, "sW": 2}),
        ("grouped-odd", {"C": 66, "K": 66, "groups": 2}),
        ("depthwise", {"C": 64, "K": 64, "groups": 64}),
        ("anisotropic", {"Y": 1, "X": 7, "pH": 0, "pW": 3, "sW": 2}),
        ("out-1x1", {"Hi": 3, "Wi": 3, "pH": 0, "pW": 0}),
    )
    for dtype in ("fp16", "bf16")
)
CASE_BY_ID = index_cases(CASES, label="convolution")


# The gfx942 forward configuration the cohort was qualified on.
_KERNEL_NAME = "implicit_gemm_conv_gfx942"
_CONFIG = dict(
    tile_m=64,
    tile_n=64,
    tile_k=64,
    warp_m=2,
    warp_n=2,
    warp_tile_m=16,
    warp_tile_n=16,
    warp_tile_k=16,
    wave_size=64,
    pipeline="mem",
)


def kernel_spec(case: Case):
    """The forward spec this cohort builds for ``case``."""
    from kernels.common.conv_implicit_gemm import (
        ConvDataSpec,
        ConvProblem,
        ImplicitGemmConvSpec,
    )

    problem = ConvProblem(**case.problem_fields())
    _va, _vb, vec_c = ImplicitGemmConvSpec.default_vector_sizes(
        problem.C, problem.K, case.dtype
    )
    return ImplicitGemmConvSpec(
        problem=problem,
        name=_KERNEL_NAME,
        data=ConvDataSpec(dtype_a=case.dtype, dtype_b=case.dtype, dtype_d=case.dtype),
        epilogue="cshuffle" if vec_c > 1 else "default",
        groups=problem.groups,
        **_CONFIG,
    )


def prepare(case: Case, library_root: str) -> tuple[KernelLauncher, dict]:
    """Build the current Python kernel for ``case`` and its AOT ABI."""
    import kernels
    from kernels.common.conv_abi import conv_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_implicit_gemm import (
        build_implicit_gemm_conv,
        implicit_gemm_conv_grid,
    )
    from rocke import compile_kernel
    from rocke.runtime import KernelLauncher

    root = Path(library_root).resolve()
    if not Path(kernels.__file__).resolve().is_relative_to(root):
        raise RuntimeError(
            "worker imported production packages outside the selected library"
        )
    spec = kernel_spec(case)
    kernel = build_implicit_gemm_conv(spec, arch=NAME)
    artifact = compile_kernel(kernel, arch=NAME, backend="python")
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        signature=conv_args_signature(case.dtype, is_3d=spec.problem.is_3d),
    )
    sizes = {name: 2 * math.prod(shape) for name, shape in case.input_shapes.items()}
    # Capture every scalar ABI argument once. Replay substitutes only pointers;
    # it never imports current convolution geometry code.
    values = ConvArgs.from_problem(
        spec.problem, tile_m=spec.tile_m, tile_n=spec.tile_n
    ).to_launch_values(
        0, 0, 0, sizes["a"], sizes["b"], 2 * math.prod(case.output_shape)
    )
    for name in ("A", "B", "D"):
        del values[name]
    metadata = {
        "name": launcher.kernel_name,
        "signature": launcher.signature,
        "grid": implicit_gemm_conv_grid(spec),
        "block": (spec.block_size, 1, 1),
        "scalars": values,
        "bindings": {"A": "a", "B": "b", "D": "out"},
        "kernel_spec": dict(_CONFIG, epilogue=spec.epilogue, dtype=case.dtype),
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
