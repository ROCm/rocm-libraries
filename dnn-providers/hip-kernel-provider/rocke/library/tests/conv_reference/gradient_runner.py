# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute small gradient plans against a live NumPy oracle, without Torch."""

from __future__ import annotations

import ctypes
import numpy as np

from .contract import Case
from .gradients import output_shape
from .implementations.gradient_plan import Plan


def execute(
    plan: Plan,
    case: Case,
    direction: str,
    architecture: str,
    inputs: dict[str, np.ndarray],
    repetitions: int = 3,
) -> list[np.ndarray]:
    from .architectures.common import compile_reference_kernel
    from rocke.runtime import KernelLauncher, LaunchConfig, Runtime
    from reference_common.targets import get_device_processor

    if get_device_processor() != architecture:
        raise RuntimeError(f"gradient execution requires {architecture}")
    if not plan.steps or repetitions < 1:
        raise ValueError("gradient execution requires steps and repetitions")
    launchers = []
    for step in plan.steps:
        artifact = compile_reference_kernel(step.kernel, architecture, step.llvm_flavor)
        launchers.append(
            KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=step.signature,
            )
        )
    rt = Runtime()
    out = np.empty(
        output_shape(case, direction),
        dtype=np.float16 if case.dtype == "fp16" else np.uint16,
    )
    readonly = {**inputs, **plan.immutable}
    sizes = {name: array.nbytes for name, array in readonly.items()}
    sizes.update(plan.scratch_bytes)
    sizes["out"] = out.nbytes
    buffers = {}
    results = []
    try:
        for name, size in sizes.items():
            buffers[name] = rt.alloc(size)
        for name, array in readonly.items():
            rt.memcpy_h2d(
                buffers[name], ctypes.c_void_p(array.ctypes.data), array.nbytes
            )
        for _ in range(repetitions):
            rt.memset(buffers["out"], plan.output_fill, out.nbytes)
            # Atomic partials must not carry state between executions.
            for name, size in plan.scratch_bytes.items():
                rt.memset(buffers[name], 0, size)
            launches = 0
            for launcher, step in zip(launchers, plan.steps):
                values = dict(step.scalars)
                values.update(
                    {arg: int(buffers[name]) for arg, name in step.bindings.items()}
                )
                result = launcher(
                    values,
                    config=LaunchConfig(
                        grid=step.grid, block=step.block, stream=0, fence=False
                    ),
                )
                launches += result.launches
            rt.sync()
            if launches != len(plan.steps):
                raise AssertionError(
                    f"expected {len(plan.steps)} gradient launches, got {launches}"
                )
            rt.memcpy_d2h(ctypes.c_void_p(out.ctypes.data), buffers["out"], out.nbytes)
            results.append(out.copy())
            for name, original in readonly.items():
                observed = np.empty_like(original)
                rt.memcpy_d2h(
                    ctypes.c_void_p(observed.ctypes.data),
                    buffers[name],
                    observed.nbytes,
                )
                np.testing.assert_array_equal(
                    observed.view(np.uint8),
                    original.view(np.uint8),
                    err_msg=f"gradient mutated {name}",
                )
    finally:
        for ptr in buffers.values():
            rt.free(ptr)
    return results
