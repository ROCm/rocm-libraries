# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helpers for the loaded COMGR: which compiler build it is, and whether it can build
for a test's arch.

A host test that compiles a kernel through COMGR calls :func:`require_comgr_target` first, or
compiles with :func:`compile_kernel_or_skip`, which does. A missing or unloadable COMGR, or one
that does not list the arch (an older ROCm lacks the newest targets), skips the test with the
reason, or fails it under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``
(:func:`~rocke.tests.compiler_tools.require_tools_or_skip`); after the check, any compile error is
the kernel's and fails the test.

:func:`loaded_compiler_build` names the loaded compiler the way the registry's records that
apply to some compilers only do (``rocke.tests.pending_lanes.Compilers``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..runtime import comgr
from .compiler_tools import require_tools_or_skip

if TYPE_CHECKING:
    from ..core.ir import KernelDef
    from ..helpers.compile import KernelArtifact

__all__ = [
    "CompilerBuild",
    "compile_kernel_or_skip",
    "loaded_compiler_build",
    "require_comgr_target",
]


@dataclass(frozen=True)
class CompilerBuild:
    """One compiler build: its LLVM major, and the llvm-project commit it was built from
    (40 hex digits); either is None when unknown."""

    llvm_major: int | None
    llvm_commit: str | None


def loaded_compiler_build() -> CompilerBuild:
    """The build of the compiler the loaded COMGR runs
    (``rocke.runtime.comgr.loaded_compiler_info`` and ``loaded_compiler_commit``); both
    parts None when no COMGR loads."""
    info = comgr.loaded_compiler_info()
    major = None if info is None or info.llvm_version is None else info.llvm_version[0]
    return CompilerBuild(major, comgr.loaded_compiler_commit())


def require_comgr_target(arch: str) -> None:
    """Skip unless the loaded COMGR can build code objects for ``arch`` (``"gfx942"``); fail
    instead under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``."""
    try:
        isas = comgr.supported_isas()
    except comgr.ComgrError as exc:
        require_tools_or_skip(f"comgr unavailable: {exc}")
    if arch not in {name.split("--")[-1].split(":")[0] for name in isas}:
        require_tools_or_skip(f"the loaded comgr cannot target {arch}")


def compile_kernel_or_skip(kernel: KernelDef, *, arch: str) -> KernelArtifact:
    """``kernel`` compiled through COMGR for ``arch`` (``"gfx942"``), without its IR text, once
    :func:`require_comgr_target` has passed: a host whose COMGR does not load, or cannot target
    ``arch``, skips the test (fails it under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``). An error the
    compile raises after that is the kernel's and propagates."""
    # Imported here: the registry (rocke.tests.pending_lanes) imports this
    # module, and the tools that read the registry need no compiler.
    from ..helpers.compile import compile_kernel

    require_comgr_target(arch)
    return compile_kernel(kernel, arch=arch, capture_ir_text=False)
