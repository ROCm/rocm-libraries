# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: lower one kernel with both engines and require the same outcome.

rocKE's Python and C++ engines must emit the same LLVM IR for the same kernel,
and refuse the same kernel with the same message. ``ROCKE_BACKEND=both`` checks
the first half for every ``lower_kernel_to_llvm`` call, but it cannot check the
second: the Python engine raises before the C++ engine runs. These helpers drive
each engine directly, so a test states the parity it relies on whatever backend
the run selects.

- ``lower_with_both_engines`` returns the LLVM IR both engines emit, after
  asserting that the texts are identical.
- ``assert_both_engines_reject`` asserts that both engines refuse the kernel
  with exactly ``message``. The C++ engine reports through the ``rocke_engine``
  binding as a ``RuntimeError`` whose text ends with ``": " + message``.
- ``lower_hip_with_both_engines`` and ``assert_both_engines_reject_hip`` are
  the same two checks for the HIP-source lowering.
- ``hip_outcomes_of_both_engines`` returns each engine's HIP source, or its
  refusal as a ``HipRefusal`` (error type and message), for a test that
  accepts either as long as the two agree.

All five skip, naming the reason, when the ``rocke_engine`` binding is not built.
``available_engines`` names the engines a test that drives each engine itself can
use: the installed wheel has no ``rocke_engine`` binding, so such a test runs the
Python engine alone there and both engines wherever the binding is supplied.
``all_engines_backend`` is the ``backend`` argument that runs those same engines
through ``compile_kernel`` and similar entry points.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass

import pytest

from rocke.core.backend import _lower_via_cpp_engine
from rocke.core.ir import KernelDef
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

__all__ = [
    "all_engines_backend",
    "assert_both_engines_reject",
    "assert_both_engines_reject_hip",
    "HipRefusal",
    "available_engines",
    "hip_outcomes_of_both_engines",
    "lower_hip_with_both_engines",
    "lower_with_both_engines",
]

# rocke_status_t (cpp/include/rocke/ir.h): the code the C++ engine reports for
# each Python exception type the lowerers raise for a kernel they refuse.
_NATIVE_STATUS: dict[type[Exception], int] = {
    ValueError: 1,
    TypeError: 2,
    KeyError: 3,
    NotImplementedError: 5,
}

# The binding's refusal text: "<entry point>: lower failed for arch '<gfx>'
# (status <code>): <message>".
_NATIVE_REFUSAL = re.compile(r"\(status (\d+)\): (.*)", re.DOTALL)


def available_engines() -> tuple[str, ...]:
    """``("python", "cpp")`` when the ``rocke_engine`` binding is importable, else
    ``("python",)``."""
    if importlib.util.find_spec("rocke_engine") is None:
        return ("python",)
    return ("python", "cpp")


def all_engines_backend() -> str:
    """The ``backend`` that lowers with every engine in ``available_engines()``:
    ``"both"`` when the ``rocke_engine`` binding is importable, else ``"python"``."""
    return "both" if "cpp" in available_engines() else "python"


def lower_with_both_engines(
    kernel: KernelDef, arch: str, llvm_flavor: str | None = None
) -> str:
    """Lower ``kernel`` for ``arch`` with each engine; return the shared IR text."""
    pytest.importorskip("rocke_engine")
    python_ir = _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=llvm_flavor)
    native_ir = _lower_via_cpp_engine(serialize(kernel), arch, llvm_flavor)
    assert native_ir == python_ir, (
        f"the C++ engine lowered kernel '{kernel.name}' for {arch} differently "
        "from the Python engine"
    )
    return python_ir


def assert_both_engines_reject(
    kernel: KernelDef,
    arch: str,
    error_type: type[Exception],
    message: str,
    llvm_flavor: str | None = None,
) -> None:
    """Assert that both engines refuse ``kernel`` for ``arch`` with ``message``.

    The Python engine must raise ``error_type`` with exactly ``message`` (its
    first argument, so a ``KeyError`` compares unquoted); the C++ engine must
    fail with the same message after its binding prefix.
    """
    pytest.importorskip("rocke_engine")
    with pytest.raises(error_type) as python_error:
        _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=llvm_flavor)
    assert python_error.value.args[0] == message
    with pytest.raises(RuntimeError) as native_error:
        _lower_via_cpp_engine(serialize(kernel), arch, llvm_flavor)
    assert str(native_error.value).endswith(": " + message), str(native_error.value)


def lower_hip_with_both_engines(kernel: KernelDef, arch: str) -> str:
    """Lower ``kernel`` to HIP source for ``arch`` with each engine; return the shared text."""
    engine = pytest.importorskip("rocke_engine")
    python_hip = lower_kernel_to_hip(kernel, arch=arch)
    native_hip = engine.lower_serialized_ir_hip(serialize(kernel), arch)
    assert native_hip == python_hip, (
        f"the C++ engine lowered kernel '{kernel.name}' to HIP for {arch} "
        "differently from the Python engine"
    )
    return python_hip


def assert_both_engines_reject_hip(
    kernel: KernelDef,
    arch: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """Assert that both HIP lowerings refuse ``kernel`` for ``arch`` with ``message``.

    Same contract as :func:`assert_both_engines_reject`, for the HIP source.
    """
    engine = pytest.importorskip("rocke_engine")
    with pytest.raises(error_type) as python_error:
        lower_kernel_to_hip(kernel, arch=arch)
    assert python_error.value.args[0] == message
    with pytest.raises(RuntimeError) as native_error:
        engine.lower_serialized_ir_hip(serialize(kernel), arch)
    assert str(native_error.value).endswith(": " + message), str(native_error.value)


@dataclass(frozen=True)
class HipRefusal:
    """One engine's refusal to lower a kernel to HIP source.

    ``error_type_name`` is the Python exception's type name -- for the C++
    engine, the name of the type its status code stands for, or
    ``"status <code>"`` for a code that stands for none.
    """

    error_type_name: str
    message: str


def hip_outcomes_of_both_engines(
    ir_text: str, arch: str
) -> tuple[str | HipRefusal, str | HipRefusal]:
    """Each engine's HIP source for the serialized kernel ``ir_text``, or its refusal.

    The text is what the C++ engine reads; the Python engine lowers it parsed.
    Two outcomes are equal exactly when both engines emit the same source, or
    refuse with the same error type and message. Any exception the Python
    lowering raises is its refusal, so a crash shows as a refusal that the C++
    engine does not share.
    """
    engine = pytest.importorskip("rocke_engine")
    python_outcome: str | HipRefusal
    native_outcome: str | HipRefusal
    try:
        python_outcome = lower_kernel_to_hip(parse(ir_text), arch=arch)
    except Exception as error:  # noqa: BLE001 -- a refusal is an outcome
        error_type = next(
            (kind for kind in _NATIVE_STATUS if isinstance(error, kind)), type(error)
        )
        message = error.args[0] if error.args else ""
        python_outcome = HipRefusal(error_type.__name__, message)
    try:
        native_outcome = engine.lower_serialized_ir_hip(ir_text, arch)
    except RuntimeError as error:
        refusal = _NATIVE_REFUSAL.search(str(error))
        if refusal is None:
            raise
        status, message = int(refusal.group(1)), refusal.group(2)
        error_type = next(
            (kind for kind, code in _NATIVE_STATUS.items() if code == status), None
        )
        type_name = error_type.__name__ if error_type else f"status {status}"
        native_outcome = HipRefusal(type_name, message)
    return python_outcome, native_outcome
