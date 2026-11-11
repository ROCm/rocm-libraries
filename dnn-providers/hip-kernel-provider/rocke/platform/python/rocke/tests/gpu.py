# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: run a small kernel on the current GPU, only on arches that support it.

Each numeric test names the arches whose hardware supports its data types. On any other
arch, or with no GPU, the test skips with the reason. On a supported arch every failure is
loud: a lowering error or a compiler rejection fails the test, it is never turned into a
skip. The one exception is a host toolchain that cannot judge the kernel (no clang, a clang
that does not know the arch, or one that cannot run its linker), which is a missing
toolchain, not a kernel defect.

A CI lane that must run a test on a given arch sets ``ROCKE_REQUIRE_<ARCH>=1`` (for
example ``ROCKE_REQUIRE_GFX942=1``). A test that names that arch then fails instead of
skipping when the current GPU is a different arch or there is no GPU, so a lane that
lost its GPU does not pass with no coverage.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from rocke.core.ir import KernelDef
from rocke.tests.ir_validity import arbiter_verdict

if TYPE_CHECKING:
    from rocke.helpers.compile import KernelArtifact

__all__ = [
    "current_device_arch",
    "current_device_arch_in",
    "launch_with_buffers",
    "probe_device_arch",
    "run_on_device",
    "supported_device_arch",
]

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


def probe_device_arch() -> tuple[str | None, str]:
    """``(arch, "")`` for HIP's current device, the one rocke's runtime keeps and
    launches kernels on, or ``(None, reason)`` when there is no GPU. Never skips;
    for callers that decide themselves what to do without a GPU."""
    try:
        from rocke.runtime import hip_module as hip

        arch = hip.get_device_arch(hip.current_device())
    except Exception as exc:  # noqa: BLE001 - any HIP failure means "no GPU here"
        return None, f"no GPU available: {exc}"
    if not arch:
        return None, "no GPU available"
    return arch, ""


def current_device_arch() -> str:
    """The arch of the GPU this process launches on: HIP's current device, the one
    rocke's runtime keeps and launches kernels on. Skips when there is no GPU."""
    arch, absence = probe_device_arch()
    if arch is None:
        pytest.skip(absence)
    return arch


def current_device_arch_in(arches: Collection[str], subject: str = "this test") -> str:
    """The current GPU's arch when it is one of `arches`; otherwise skip, naming `subject`
    and the GPU found. Fails instead when ``ROCKE_REQUIRE_<ARCH>=1`` is set for an arch
    in `arches` and the current GPU is not that arch (see the module docstring)."""
    arch, absence = probe_device_arch()
    found = arch or absence
    for required in arches:
        variable = f"ROCKE_REQUIRE_{required.upper()}"
        if os.environ.get(variable) == "1" and arch != required:
            pytest.fail(f"{variable}=1 needs a {required} GPU; found {found}")
    if arch not in arches:
        pytest.skip(f"{subject} needs one of {sorted(arches)}; found {found}")
    return arch


def supported_device_arch(kernel: KernelDef, arches: Collection[str]) -> str:
    """The current GPU's arch when it is one of `arches` (the arches whose hardware
    supports `kernel`'s data types) and the IR the active engine lowers `kernel` to
    compiles and links for it; otherwise skip, naming why, or fail on a compiler
    rejection or on a missing required GPU (see `current_device_arch_in`)."""
    arch = current_device_arch_in(arches, subject=kernel.name)
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

    artifact = compile_kernel(kernel, arch=arch)
    signature = [
        sig_param(name, _SIGNATURE_DTYPE[host.dtype.type])
        for name, host in buffers.items()
    ]
    launch_with_buffers(
        artifact, signature, buffers, grid=(1, 1, 1), block=(threads, 1, 1)
    )


def launch_with_buffers(
    artifact: KernelArtifact,
    signature: Sequence[Mapping[str, Any]],
    buffers: Mapping[str, np.ndarray],
    *,
    grid: tuple[int, int, int],
    block: tuple[int, int, int],
    scalars: Mapping[str, int | float] | None = None,
) -> None:
    """Launch `artifact`'s kernel on the current GPU and wait for it: copy every entry of
    `buffers` (a contiguous, writable host array per pointer parameter of `signature`) to
    the device, pass `scalars` for the scalar parameters, then copy every buffer back into
    its host array."""
    from rocke.runtime.hip_module import Runtime
    from rocke.runtime.host_buffers import as_u8_buffer
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        synchronize_and_release,
    )

    launcher = KernelLauncher(
        hsaco=artifact.hsaco, kernel_name=artifact.kernel_name, signature=signature
    )
    rt = Runtime()
    device = {name: DeviceMem(host.nbytes) for name, host in buffers.items()}
    for name, host in buffers.items():
        rt.memcpy_h2d(device[name].ptr(), as_u8_buffer(host), host.nbytes)
    launcher({**device, **(scalars or {})}, config=LaunchConfig(grid=grid, block=block))
    synchronize_and_release()
    for name, host in buffers.items():
        rt.memcpy_d2h(as_u8_buffer(host), device[name].ptr(), host.nbytes)
