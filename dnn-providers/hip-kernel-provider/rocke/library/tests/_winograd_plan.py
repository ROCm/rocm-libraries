# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run a Winograd launch plan the grouped-convolution dispatcher selected, for
the gfx1250 Winograd route tests, without torch."""

from __future__ import annotations

from collections.abc import Callable

from rocke.numeric import storage
from rocke.runtime.hip_module import Runtime
from rocke.runtime.launcher import KernelLauncher, LaunchConfig

from benchmarks.common.benchmark_winograd_conv import (
    make_inputs,
    read_back_drained,
    reference,
    upload_drained,
    verify_error,
)
from dispatch.grouped_convolution import launch_plan_for

__all__ = ["run_winograd_plan"]


def run_winograd_plan(
    request, spec, *, compile_step: Callable, seed: int
) -> tuple[float, float]:
    """Run ``spec`` (a Winograd plan selected for ``request``) on inputs drawn
    from ``seed``; ``(rocke metric, relL2)`` against the forward reference.

    ``compile_step(step)`` returns the step's compiled artifact. The
    workspaces and y are poisoned (bytes 0x7F) first, so an element no kernel
    writes shows up, and every host copy is staged and drained
    (``upload_drained`` / ``read_back_drained``)."""
    problem, dtype = spec.problem, spec.dtype
    plan = launch_plan_for(request, spec)
    launchers = []
    for step in plan.steps:
        artifact = compile_step(step)
        launchers.append(
            KernelLauncher(
                hsaco=artifact.hsaco,
                kernel_name=artifact.kernel_name,
                signature=step.signature,
            )
        )
    x, w = make_inputs(problem, dtype, seed=seed)
    ref = reference(x, w, problem, dtype)
    y_shape = (problem.N, problem.Ho, problem.Wo, problem.K)
    y_bytes = ref.size * storage.host_dtype(dtype).itemsize
    sizes = {"x": x.nbytes, "w": w.nbytes, "y": y_bytes}
    sizes.update(plan.workspaces)
    runtime = Runtime()
    ptrs = {name: int(runtime.alloc(max(int(n), 4))) for name, n in sizes.items()}
    try:
        upload_drained(runtime, ptrs["x"], x)
        upload_drained(runtime, ptrs["w"], w)
        for name in (*plan.workspaces, "y"):
            runtime.memset(ptrs[name], 0x7F, sizes[name])
        runtime.sync()
        for step, launcher in zip(plan.steps, launchers):
            launcher(
                plan.values(step, ptrs, sizes),
                config=LaunchConfig(grid=step.grid, block=step.block, fence=True),
            )
        out = read_back_drained(runtime, ptrs["y"], y_shape, dtype)
    finally:
        runtime.sync()
        for ptr in ptrs.values():
            runtime.free(ptr)
    return verify_error(out, ref)
