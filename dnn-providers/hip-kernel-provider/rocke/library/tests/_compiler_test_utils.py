# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared skip policy for library tests that compile a kernel through COMGR.

Such a test depends on the host: COMGR and the code-object tools must be
installed, and the loaded compiler must accept every ``-mllvm`` option the
kernel needs. A test that uses this policy skips when one of those is missing,
because that is a fact about the host, not a defect in the kernel. A run that
sets ``ROCKE_REQUIRE_COMPILER_TOOLS=1`` (the CI lane that supplies the tools)
fails instead: a gate that skips everywhere looks green and checks nothing.

* :func:`require_tools_or_skip`: skip (or fail under the variable) for a
  stated reason.
* :func:`skip_where_compiler_refuses_option`: apply that policy when the body
  raises ``CompilerOptionNotAccepted``.

A skip is raised as ``unittest.SkipTest``, which pytest test functions and a
``unittest`` ``setUpClass`` both report as a skip.
"""

from __future__ import annotations

import os
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NoReturn

from rocke.runtime.compiler_options import CompilerOptionNotAccepted

__all__ = [
    "REQUIRE_COMPILER_TOOLS_ENV",
    "require_tools_or_skip",
    "skip_where_compiler_refuses_option",
]

# platform/CMakeLists.txt sets it to "1" only for the installed ctest entries
# whose tests consume it, where ROCM_PATH supplies the tools (Linux); see
# TESTING.md section 5.1.4.
REQUIRE_COMPILER_TOOLS_ENV = "ROCKE_REQUIRE_COMPILER_TOOLS"


def require_tools_or_skip(reason: str) -> NoReturn:
    """Skip the running test for ``reason``, or fail it with ``AssertionError``
    when ``ROCKE_REQUIRE_COMPILER_TOOLS=1`` says the tools must be there."""
    if os.environ.get(REQUIRE_COMPILER_TOOLS_ENV) == "1":
        raise AssertionError(f"{REQUIRE_COMPILER_TOOLS_ENV}=1 but {reason}")
    raise unittest.SkipTest(reason)


@contextmanager
def skip_where_compiler_refuses_option() -> Iterator[None]:
    """Apply :func:`require_tools_or_skip` if the body raises
    ``CompilerOptionNotAccepted``.

    That error means the loaded compiler does not accept a ``-mllvm`` option
    the kernel needs (see ``rocke.runtime.compiler_options``). Every other
    error propagates unchanged.
    """
    try:
        yield
    except CompilerOptionNotAccepted as exc:
        require_tools_or_skip(f"loaded compiler cannot build this kernel: {exc}")
