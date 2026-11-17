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

Both skip, naming the reason, when the ``rocke_engine`` binding is not built.
"""

from __future__ import annotations

import pytest

from rocke.core.backend import _lower_via_cpp_engine
from rocke.core.ir import KernelDef
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

__all__ = [
    "assert_both_engines_reject",
    "assert_both_engines_reject_hip",
    "lower_hip_with_both_engines",
    "lower_with_both_engines",
]


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

    The Python engine must raise ``error_type`` with exactly ``message``; the
    C++ engine must fail with the same message after its binding prefix.
    """
    pytest.importorskip("rocke_engine")
    with pytest.raises(error_type) as python_error:
        _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=llvm_flavor)
    assert str(python_error.value) == message
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
    assert str(python_error.value) == message
    with pytest.raises(RuntimeError) as native_error:
        engine.lower_serialized_ir_hip(serialize(kernel), arch)
    assert str(native_error.value).endswith(": " + message), str(native_error.value)
