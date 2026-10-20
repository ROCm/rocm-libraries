# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: run a small kernel on the current GPU, only on arches that support it.

Each numeric test names the arches whose hardware supports its data types. On any other
arch, or with no GPU, the test skips with the reason. On a supported arch every failure is
loud: a lowering error or a compiler rejection fails the test, it is never turned into a
skip. The one exception is a host toolchain that cannot judge the kernel (no clang, a clang
that does not know the arch, or one that cannot run its linker), which is a missing
toolchain, not a kernel defect.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
import pytest

from rocke.core.ir import KernelDef
from rocke.tests.ir_validity import arbiter_verdict

__all__ = ["current_device_arch", "run_on_device", "supported_device_arch"]

# Host array dtype -> rocke signature dtype name.
_SIGNATURE_DTYPE = {
    np.float64: "f64",
    np.float32: "f32",
    np.float16: "f16",
    np.int32: "i32",
    np.int16: "i16",
    np.int8: "i8",
    np.uint8: "u8",
}


def current_device_arch() -> str:
    """The arch of the GPU this process launches on: HIP's current device, the one
    rocke's runtime keeps and launches kernels on. Skips when there is no GPU."""
    hip = pytest.importorskip("rocke.runtime.hip_module")
    try:
        arch = hip.get_device_arch(hip.current_device())
    except Exception as exc:  # noqa: BLE001 - any HIP failure means "no GPU here"
        pytest.skip(f"no GPU available: {exc}")
    if not arch:
        pytest.skip("no GPU available")
    return arch


def supported_device_arch(kernel: KernelDef, arches: Collection[str]) -> str:
    """The current GPU's arch when it is one of `arches` (the arches whose hardware
    supports `kernel`'s data types) and the IR the active engine lowers `kernel` to
    compiles and links for it; otherwise skip, naming why, or fail on a compiler
    rejection."""
    arch = current_device_arch()
    if arch not in arches:
        pytest.skip(
            f"{kernel.name} needs one of {sorted(arches)}; the current GPU is {arch}"
        )
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    text = lower_kernel_to_llvm(kernel, arch=arch)
    verdict, reason = arbiter_verdict(text, arch)
    if verdict == "cannot_judge":
        pytest.skip(reason)
    assert verdict == "accepted", f"LLVM rejected {kernel.name} for {arch}: {reason}"
    return arch


def run_on_device(
    kernel: KernelDef,
    buffers: Mapping[str, np.ndarray],
    threads: int,
    arches: Collection[str],
) -> None:
    """Launch `kernel` on the current GPU when it is one of `arches`, as one block of
    `threads` lanes with one pointer parameter per entry of `buffers` (in order), then copy
    every buffer back into its host array."""
    arch = supported_device_arch(kernel, arches)
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import sig_param
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    artifact = compile_kernel(kernel, arch=arch)
    signature = [
        sig_param(name, _SIGNATURE_DTYPE[host.dtype.type])
        for name, host in buffers.items()
    ]
    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=signature
    )
    rt = Runtime()
    device = {name: DeviceMem(host.nbytes) for name, host in buffers.items()}
    for name, host in buffers.items():
        rt.memcpy_h2d(device[name].ptr(), as_u8_buffer(host), host.nbytes)
    launcher(device, config=LaunchConfig(grid=(1, 1, 1), block=(threads, 1, 1)))
    synchronize_and_release()
    for name, host in buffers.items():
        rt.memcpy_d2h(as_u8_buffer(host), device[name].ptr(), host.nbytes)
