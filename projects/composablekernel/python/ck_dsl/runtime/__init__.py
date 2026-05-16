# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HSACO compilation + HIP module load + kernel launch.

`ck_dsl.runtime` is the bottom of the DSL stack: it owns the
in-process pipeline that turns AMDGPU LLVM IR text into a running
kernel.

Layered modules (bottom-up):

  - ``comgr``       : ctypes wrapper over `libamd_comgr.so`. Implements
                      `LLVM IR (text) -> BC -> relocatable ELF -> HSA
                      executable`. Provides `build_hsaco_from_llvm_ir`.
                      Steady-state ~1.2 ms per kernel.

  - ``hip_module``  : ctypes wrapper over `libamdhip64.so`. Bare
                      minimum for `hipModuleLoadData`,
                      `hipModuleLaunchKernel`, `hipMalloc`,
                      `hipMemcpy{H2D,D2H}`, `hipFree`, `hipEvent*`.
                      Owns ``Runtime._pending_args``, the per-stream
                      keep-alive queue that fixes the
                      `HIP_LAUNCH_PARAM_BUFFER_POINTER` arg-buffer
                      lifetime race.

  - ``torch_module``: torch-tensor arg packing + `resolve_stream`
                      (which collapses ``stream=0`` to torch's current
                      stream so the caching allocator sees our
                      launches).

  - ``launcher``    : long-lived launch abstractions (CK Tile / FlyDSL /
                      Triton inspired). The recommended entry point
                      for any new op:

                        * :class:`KernelLauncher`  -- owns one compiled
                          HSACO + loaded HIP module + function entry.
                          Construct once per (problem-shape, dtype),
                          call many times.
                        * :class:`PipelineLauncher` -- N stages on one
                          stream (split-KV attention's seg+reduce,
                          grouped GEMM's k-fixup, ...).
                        * :class:`WorkspacePool` -- long-lived owner of
                          named torch workspace tensors keyed by
                          (shape, dtype, device); replaces ad-hoc
                          ``torch.empty(..., device=q.device)`` per
                          call.
                        * :class:`DeviceMem` -- RAII over
                          ``Runtime.alloc/free`` for numpy / manifest
                          flows that don't have torch.
                        * :func:`time_launches` -- event-based
                          benchmark loop. The only path that should
                          create HIP events; the launchers themselves
                          are event-free so production calls stay
                          overhead-free.

The previous monolithic ``launch_torch_kernel`` is now a thin
back-compat shim over :class:`KernelLauncher` + :func:`time_launches`;
prefer the launcher primitives directly in new code.

High-level helpers (`ck_dsl.helpers.compile_kernel`,
`ck_dsl.run_manifest`) layer on top of these.

When to drop to the lower-level APIs:
  - Drive an alternate codegen (emit MLIR -> LLVM IR yourself and
    hand the text to ``build_hsaco_from_llvm_ir``).
  - Implement a custom benchmarking harness that doesn't fit
    :func:`time_launches`.
"""

from __future__ import annotations

from .comgr import ComgrError, ComgrTimings, build_hsaco_from_llvm_ir
from .hip_module import HipError, Runtime
from .launcher import (
    DeviceMem,
    KernelLauncher,
    LaunchConfig,
    LaunchSummary,
    PipelineLauncher,
    WorkspacePool,
    no_fence,
    release_retained_for_stream,
    synchronize_and_release,
    time_launches,
    wait_stream_and_release,
)
from .torch_module import (
    TorchLaunchSummary,
    empty_workspace,
    launch_torch_kernel,
    pack_args,
    resolve_stream,
)

__all__ = [
    "ComgrError",
    "ComgrTimings",
    "DeviceMem",
    "HipError",
    "KernelLauncher",
    "LaunchConfig",
    "LaunchSummary",
    "PipelineLauncher",
    "Runtime",
    "TorchLaunchSummary",
    "WorkspacePool",
    "build_hsaco_from_llvm_ir",
    "empty_workspace",
    "launch_torch_kernel",
    "no_fence",
    "pack_args",
    "release_retained_for_stream",
    "resolve_stream",
    "synchronize_and_release",
    "time_launches",
    "wait_stream_and_release",
]
