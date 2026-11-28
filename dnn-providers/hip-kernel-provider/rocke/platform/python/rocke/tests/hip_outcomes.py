# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What one engine's HIP lowering of a kernel came to: its source or its refusal.

The Python and C++ engines must lower a kernel to the same HIP source, or refuse
it with the same error type and message. An outcome is the ``str`` source or a
:class:`HipRefusal`, so two outcomes are equal exactly when the engines agree.
The C++ engine reports a refusal as a ``rocke_status_t`` code (``rocke/ir.h`` is
the source of truth); :data:`NATIVE_REFUSAL_STATUS` maps each refusal type to
its code.

This module imports no test framework, so a parity emitter can compute the
Python outcome of a kernel it has just built, in its own process
(``rocke.tests.hip_identity_cases``); ``rocke.tests.engines`` adds the C++
side.
"""

from __future__ import annotations

import dataclasses

from ..core.ir import KernelDef
from ..core.lower_hip import lower_kernel_to_hip

__all__ = [
    "NATIVE_REFUSAL_STATUS",
    "REFUSAL_ERROR_TYPES",
    "HipRefusal",
    "hip_refusal_of",
    "python_hip_outcome_of_kernel",
]

#: The ``rocke_status_t`` each refusal type maps to. ``hip_refusal_of`` names an
#: error by the first type here it is an instance of. ArchDomainError (status 6,
#: a NotImplementedError subclass) is left out: no HIP lowering raises it.
NATIVE_REFUSAL_STATUS: dict[type[Exception], int] = {
    ValueError: 1,
    TypeError: 2,
    KeyError: 3,
    NotImplementedError: 5,
}

#: The exception types a lowerer or builder raises to refuse a kernel: those a
#: C++ engine status code stands for. Any other exception is a failure.
REFUSAL_ERROR_TYPES: tuple[type[Exception], ...] = tuple(NATIVE_REFUSAL_STATUS)


@dataclasses.dataclass(frozen=True)
class HipRefusal:
    """One engine's refusal to lower a kernel to HIP source.

    ``error_type_name`` is the Python exception's type name -- for the C++
    engine, the name of the type its status code stands for, or
    ``"status <code>"`` for a code that stands for none.
    """

    error_type_name: str
    message: str


def hip_refusal_of(error: Exception) -> HipRefusal:
    """The Python engine's ``error`` as a refusal: named by the refusal type it
    is an instance of (so a subclass compares as its C++ status does), or by
    its own type when it is none of them, with its first argument as text."""
    error_type = next(
        (kind for kind in NATIVE_REFUSAL_STATUS if isinstance(error, kind)),
        type(error),
    )
    return HipRefusal(error_type.__name__, str(error.args[0]) if error.args else "")


def python_hip_outcome_of_kernel(kernel: KernelDef, arch: str) -> str | HipRefusal:
    """The Python engine's HIP source for ``kernel``, or its refusal. Any
    exception the lowering raises is its refusal, so a crash shows as a refusal
    that the C++ engine does not share."""
    try:
        return lower_kernel_to_hip(kernel, arch=arch)
    except Exception as error:  # noqa: BLE001 -- a refusal is an outcome
        return hip_refusal_of(error)
