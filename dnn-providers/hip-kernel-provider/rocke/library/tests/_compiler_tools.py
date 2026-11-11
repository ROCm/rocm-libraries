# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The skip-or-fail rule for resource guards that compile kernels and read
the code object.

One rule for every guard, so a lane that sets ``ROCKE_REQUIRE_COMPILER_TOOLS=1``
holds the same contract everywhere. A load or LLVM-version failure is told
apart by *when* it happens (before any compile); the one compile-time skip, a
forced IR flavor the loaded comgr rejects, is recognised by comgr's message:

* **Missing tools** -- comgr or the HSACO readers cannot be imported or loaded
  (:func:`load_comgr_or_skip` runs before any compile): skip, or fail under
  ``ROCKE_REQUIRE_COMPILER_TOOLS=1``. The lane that supplies the tools sets it,
  so a guard that skips there checks nothing.
* **Environment facts** -- the loaded comgr is older than the kernel needs, or
  rejects the forced IR flavor: always skip. The flag is set on every Linux
  entry that appends ``_rocke_compiler_requirement_environment``, whatever
  its ROCm version, so these are not missing tools.
* **Anything else** -- any other error once comgr has loaded: raise. It is a
  real compile failure.
"""

from __future__ import annotations

import os
import unittest
from typing import NoReturn

__all__ = [
    "REQUIRE_TOOLS_ENV",
    "compile_for_guard",
    "load_comgr_or_skip",
    "require_tools_or_skip",
]

REQUIRE_TOOLS_ENV = "ROCKE_REQUIRE_COMPILER_TOOLS"

# comgr's refusal of IR whose flavor does not match the loaded compiler
# (rocke.runtime.comgr).
_MISMATCH_MARKER = "loaded compiler mismatch"


def require_tools_or_skip(reason: str) -> NoReturn:
    """Skip for a missing tool, or fail when the run requires the tools."""
    if os.environ.get(REQUIRE_TOOLS_ENV) == "1":
        raise AssertionError(f"{REQUIRE_TOOLS_ENV}=1 but {reason}")
    raise unittest.SkipTest(reason)


def load_comgr_or_skip(min_llvm_major: int, why: str) -> int:
    """Load comgr and return its LLVM major, before anything is compiled.

    A comgr that cannot load is a missing tool. One older than
    ``min_llvm_major`` is an environment fact and skips: ``why`` says what
    needs the newer one.
    """
    try:
        from rocke.runtime.comgr import _require_compiler_info

        major = int(_require_compiler_info().llvm_version[0])
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover
        require_tools_or_skip(f"comgr unavailable: {exc}")
    if major < min_llvm_major:
        raise unittest.SkipTest(f"loaded comgr is LLVM {major}; {why}")
    return major


def compile_for_guard(kernel_def, *, arch: str):
    """``compile_kernel`` once :func:`load_comgr_or_skip` has passed.

    A forced IR flavor the loaded comgr cannot take skips; any other error is
    a real compile failure and raises.
    """
    try:
        from rocke.helpers.compile import compile_kernel
    except ImportError as exc:  # pragma: no cover - environment-dependent
        require_tools_or_skip(f"comgr toolchain unavailable: {exc}")
    try:
        return compile_kernel(kernel_def, arch=arch, capture_ir_text=False)
    except RuntimeError as exc:
        if _MISMATCH_MARKER in str(exc):
            raise unittest.SkipTest(f"comgr cannot take this IR flavor: {exc}") from exc
        raise
