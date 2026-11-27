# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared GDN/KDA decode test helper: kernel -> HSACO -> code-object resources.

Compiles a kernel through comgr (no GPU) and reads its resource metadata
(scratch bytes and the like) with ``llvm-objdump`` / ``llvm-readelf``. When
the toolchain is missing or cannot read the code object it applies the
platform's compiler-tool policy (``rocke.tests.compiler_tools``): the test
skips, or fails under ``ROCKE_REQUIRE_COMPILER_TOOLS=1``. A compile error is
a defect and propagates.

Tool lookup is ``rocke.analysis.isa.analyze_hsaco``'s default: the
``llvm-objdump`` / ``llvm-readelf`` found on ``PATH``, nothing else. Put a
ROCm ``llvm/bin`` first on ``PATH`` to run the checks. A tool that is missing,
or a distro llvm-objdump that cannot read an AMDGPU code object, takes that
policy with the tool's own message. A code object whose scratch size cannot be
parsed skips.

A test that checks many kernels compiles them together (:func:`compile_each`):
the kernels are lowered and their code objects read here, and COMGR runs on the
lowered text in worker processes (``_parallel_compile.compile_in_workers``).
"""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rocke.tests.comgr import compile_kernel_or_skip, require_comgr_target
from rocke.tests.compiler_tools import require_tools_or_skip

from ._parallel_compile import compile_in_workers

ARCH = "gfx950"


def _toolchain():
    """``rocke.helpers.compile`` and ``rocke.analysis.isa``, imported when a
    check first compiles; the compiler-tool policy when they cannot be."""
    try:
        from rocke.analysis import isa
        from rocke.helpers import compile as compiling
    except Exception as e:  # pragma: no cover - env-dependent
        require_tools_or_skip(f"comgr toolchain unavailable: {e}")
    return compiling, isa


def compiled_resources(kernel, arch: str = ARCH):
    """Compile ``kernel`` through comgr and return its ``ResourceInfo``."""
    art = compile_kernel_or_skip(kernel, arch=arch)
    return _code_object_resources(bytes(art.hsaco))


def compiled_scratch_bytes(kernel, arch: str = ARCH) -> int:
    """Compile ``kernel`` and return the scratch bytes it spills to."""
    return _scratch_of(compiled_resources(kernel, arch))


@dataclass(frozen=True)
class CompileOutcome:
    """One kernel of a :func:`compile_each` batch: its code object, or the error
    lowering or compiling it raised. :meth:`scratch_bytes` raises that error, so
    a test asks it inside the kernel's own subtest and one failure leaves the
    other kernels' checks standing."""

    kernel_name: str
    hsaco: bytes | None
    error: BaseException | None

    def scratch_bytes(self) -> int:
        """:func:`compiled_scratch_bytes` of the kernel."""
        if self.error is not None:
            raise self.error
        return _scratch_of(_code_object_resources(self.hsaco))


def compile_each(kernels: Sequence, arch: str = ARCH) -> list[CompileOutcome]:
    """Every kernel of ``kernels`` compiled, in order, in worker processes (see
    the module docstring); a kernel's failure is its own outcome. A host whose
    COMGR does not load or cannot target ``arch`` takes the compiler-tool policy
    first (``require_comgr_target``)."""
    require_comgr_target(arch)
    compiling, _ = _toolchain()
    inputs, outcomes = [], []
    for kernel in kernels:
        try:
            inputs.append(compiling.lower_kernel_for_comgr(kernel, arch=arch))
        except Exception as error:  # noqa: BLE001 - reported on its kernel
            outcomes.append(CompileOutcome(kernel.name, None, error))
        else:
            outcomes.append(None)
    compiled = iter(zip(inputs, compile_in_workers(inputs)))
    merged = []
    for outcome in outcomes:
        if outcome is None:
            comgr_input, (hsaco, error) = next(compiled)
            outcome = CompileOutcome(comgr_input.kernel_name, hsaco, error)
        merged.append(outcome)
    return merged


def _code_object_resources(hsaco: bytes):
    """The ``ResourceInfo`` of a code object, read with the tools on ``PATH``."""
    _, isa = _toolchain()
    with tempfile.NamedTemporaryFile(suffix=".hsaco") as fh:
        fh.write(hsaco)
        fh.flush()
        try:
            return isa.analyze_hsaco(Path(fh.name)).resources
        except (FileNotFoundError, RuntimeError) as e:
            require_tools_or_skip(f"HSACO introspection unavailable: {e}")


def _scratch_of(resources) -> int:
    """The scratch bytes of a code object's resources."""
    if resources.scratch_bytes is None:  # pragma: no cover - metadata shape drift
        raise unittest.SkipTest("could not parse the scratch size from the HSACO")
    return resources.scratch_bytes
