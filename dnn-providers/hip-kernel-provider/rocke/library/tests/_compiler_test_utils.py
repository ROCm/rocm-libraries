# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared skip policy for library tests that compile a kernel through COMGR.

Such a test depends on the host: COMGR and the code-object tools must be
installed, and the loaded compiler must accept every ``-mllvm`` option the
kernel needs. The policy itself, skip or fail under
``ROCKE_REQUIRE_COMPILER_TOOLS=1``, is the platform's
(:func:`rocke.tests.compiler_tools.require_tools_or_skip`); this module applies
it to the one compiler error the library's tests turn into a skip.

* :func:`skip_where_compiler_refuses_option`: apply that policy when the body
  raises ``CompilerOptionNotAccepted``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rocke.runtime import CompilerOptionNotAccepted
from rocke.tests.compiler_tools import require_tools_or_skip

__all__ = ["skip_where_compiler_refuses_option"]


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
