# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""HSACO compilation + HIP module load + kernel launch.

`ck_dsl.runtime` is the bottom of the DSL stack: it owns the
in-process pipeline that turns AMDGPU LLVM IR text into a running
kernel.

Modules:

  - ``comgr``      : ctypes wrapper over `libamd_comgr.so`. Provides
                     `build_hsaco_from_llvm_ir(ll_text, isa) -> (hsaco_bytes, ComgrTimings)`.
                     Implements the chain `LLVM IR (text) -> BC ->
                     relocatable ELF -> HSA executable`. Steady-state
                     wall time is ~1.2 ms per kernel; that is the
                     `cold first call` -> `~2 ms` -> `~1.2 ms` curve
                     visible in `ck_dsl_current_results.md`.

  - ``hip_module`` : ctypes wrapper over `libamdhip64.so`. Provides
                     `Runtime`, `Module`, `Function`, `Event` — the
                     bare minimum for `hipModuleLoadData`,
                     `hipModuleGetFunction`, `hipModuleLaunchKernel`,
                     `hipMalloc`, `hipMemcpy{H2D,D2H}`, `hipFree`,
                     `hipEventRecord` + `hipEventElapsedTime`.

The high-level `ck_dsl.helpers.compile_kernel` wraps both modules into
a single `KernelArtifact`. The high-level `ck_dsl.run_manifest`
wraps the `Runtime` into a manifest-driven runner.

Use the lower-level interfaces directly only when you need to:
  - Drive an alternate codegen (e.g. emit MLIR -> LLVM IR yourself and
    hand the text to `build_hsaco_from_llvm_ir`).
  - Implement a custom benchmarking harness.
  - Hold a Runtime instance across many module loads (the default
    `compile_kernel` + `run_manifest` flow creates a fresh Runtime
    per call).
"""

from __future__ import annotations

from .comgr import ComgrError, ComgrTimings, build_hsaco_from_llvm_ir
from .hip_module import HipError, Runtime
from .torch_module import (
    TorchLaunchSummary,
    empty_workspace,
    launch_torch_kernel,
    pack_args,
)

__all__ = [
    "ComgrError",
    "ComgrTimings",
    "HipError",
    "Runtime",
    "TorchLaunchSummary",
    "build_hsaco_from_llvm_ir",
    "empty_workspace",
    "launch_torch_kernel",
    "pack_args",
]
