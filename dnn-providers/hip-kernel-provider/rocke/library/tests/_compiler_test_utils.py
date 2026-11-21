# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shared skip policy for library tests that compile a kernel through COMGR.

Such a test depends on the host: COMGR and the code-object tools must be
installed, and the loaded compiler must accept every ``-mllvm`` option the
kernel needs. The policy itself, skip or fail under
``ROCKE_REQUIRE_COMPILER_TOOLS=1``, is the platform's
(:func:`rocke.tests.compiler_tools.require_tools_or_skip`); this module applies
it to the compiler errors the library's tests turn into a skip.

* :func:`skip_where_compiler_refuses_option`: apply that policy when the body
  raises ``CompilerOptionNotAccepted``.
* :func:`skip_where_compiler_refuses_processor`: apply that policy when the body
  raises the ``ValueError`` that names a processor the loaded compiler does not
  accept (a ``gfx1250`` kernel on ``llvm20``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rocke.runtime import CompilerOptionNotAccepted
from rocke.tests.compiler_tools import require_tools_or_skip

__all__ = [
    "skip_where_compiler_refuses_option",
    "skip_where_compiler_refuses_processor",
]

# The two phrases of the ``ValueError`` that ``rocke.core.arch.target`` raises
# when the loaded compiler accepts neither the processor nor a generic target
# its ``arch_specs.json`` row names. Pinned by test_arch_target_identity.py.
_PROCESSOR_REFUSAL = (
    "compiler does not accept the processor",
    "points at no generic target",
)


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


@contextmanager
def skip_where_compiler_refuses_processor() -> Iterator[None]:
    """Apply :func:`require_tools_or_skip` if the body raises the ``ValueError``
    that says the loaded compiler does not accept the target processor.

    That is a fact about the host's LLVM (``llvm20`` has no ``gfx1250``). Every
    other ``ValueError``, and every other error, propagates unchanged.
    """
    try:
        yield
    except ValueError as exc:
        message = str(exc)
        if not all(phrase in message for phrase in _PROCESSOR_REFUSAL):
            raise
        require_tools_or_skip(
            f"loaded compiler cannot target this processor: {message}"
        )
