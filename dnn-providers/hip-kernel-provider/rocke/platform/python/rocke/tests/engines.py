# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: lower one kernel with each engine and require the same outcome.

rocKE's Python and C++ engines must emit the same LLVM IR for the same kernel,
and refuse the same kernel with the same message. ``ROCKE_BACKEND=both`` checks
the first half for every ``lower_kernel_to_llvm`` call, but it cannot check the
second: the Python engine raises before the C++ engine runs. These helpers drive
each engine directly, so a test states the parity it relies on whatever backend
the run selects.

Per engine (``engine`` is one of :data:`ENGINES`):

- ``lower_with_engine`` returns the LLVM IR ``engine`` emits. For ``"cpp"`` it
  also asserts that the text is identical to the Python engine's: IR parity is
  part of what the C++ engine owes. A refusal has no output to compare, so
  ``assert_engine_rejects`` checks the one engine named.
- ``assert_engine_rejects`` asserts that ``engine`` refuses the kernel with
  exactly ``message``. The Python engine raises ``error_type``; the C++ engine
  reports through the ``rocke_engine`` binding as a ``RuntimeError`` whose text
  ends with ``"(status N): " + message``, where ``N`` is the
  ``rocke_status_t`` that ``error_type`` maps to.
- ``lower_hip_with_engine`` and ``assert_engine_rejects_hip`` are the same two
  checks for the HIP-source lowering.

A test parametrized over :data:`ENGINES` checks the Python engine in every
lane, including the lanes that do not build the binding; only its ``"cpp"``
cases skip there, naming the reason.

Both engines at once: ``lower_with_both_engines``,
``assert_both_engines_reject``, ``lower_hip_with_both_engines`` and
``assert_both_engines_reject_hip`` run the same checks for the Python and the
C++ engine, and skip as a whole when the binding is not built. Their refusal
checks compare the C++ engine's message but not its status, which some older
refusals do not yet share with the Python engine's exception type.

A test that loops over the engines itself, rather than through ``pytest``
parameters, iterates ``available_engines()``: the part of :data:`ENGINES` this
run can drive. The installed wheel has no ``rocke_engine`` binding, so such a
test runs the Python engine alone there and both engines wherever the binding
is supplied. ``all_engines_backend`` is the ``backend`` argument that runs those
same engines through ``compile_kernel`` and similar entry points.

Every engine this run can drive, plus the verifier:
``assert_refused_everywhere`` and ``assert_accepted_everywhere`` run the Python
verifier and the LLVM and HIP lowerings of each engine in
``available_engines()``. Unlike the both-engine helpers they do not skip without
the binding, so a run without it (installed CI) still checks the Python engine.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import ModuleType
from typing import Literal

import pytest

from rocke.core.backend import _lower_via_cpp_engine
from rocke.core.ir import KernelDef
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify

__all__ = [
    "ENGINES",
    "Engine",
    "all_engines_backend",
    "assert_accepted_everywhere",
    "assert_both_engines_reject",
    "assert_both_engines_reject_hip",
    "assert_engine_rejects",
    "assert_engine_rejects_hip",
    "assert_refused_everywhere",
    "available_engines",
    "lower_hip_with_both_engines",
    "lower_hip_with_engine",
    "lower_with_both_engines",
    "lower_with_engine",
]

Engine = Literal["python", "cpp"]
ENGINES: tuple[Engine, ...] = ("python", "cpp")

# The rocke_status_t (rocke/ir.h is the source of truth) each Python exception
# type maps to. Matched by exact type: ArchDomainError (status 6) is left out, as
# no refusal checked here raises it.
_NATIVE_STATUS: dict[type[Exception], int] = {
    ValueError: 1,
    TypeError: 2,
    KeyError: 3,
    NotImplementedError: 5,
}


def available_engines() -> tuple[Engine, ...]:
    """:data:`ENGINES` when the ``rocke_engine`` binding is importable, else
    ``("python",)``.

    For a test that loops over the engines itself; a ``pytest`` test
    parametrizes over :data:`ENGINES` instead.
    """
    if importlib.util.find_spec("rocke_engine") is None:
        return ("python",)
    return ENGINES


def all_engines_backend() -> str:
    """The ``backend`` that lowers with every engine in ``available_engines()``:
    ``"both"`` when the ``rocke_engine`` binding is importable, else ``"python"``."""
    return "both" if "cpp" in available_engines() else "python"


def _require_native_engine() -> ModuleType:
    """The ``rocke_engine`` binding; skip the calling test when it is not built."""
    return pytest.importorskip("rocke_engine")


def _require_engine(engine: Engine) -> None:
    """Validate ``engine``; for ``"cpp"``, skip the calling test when the
    binding is not built, before any engine runs."""
    if engine not in ENGINES:
        raise ValueError(f"engine={engine!r}, expected one of {ENGINES}")
    if engine == "cpp":
        _require_native_engine()


def _native_status(error_type: type[Exception]) -> int:
    if error_type not in _NATIVE_STATUS:
        names = sorted(t.__name__ for t in _NATIVE_STATUS)
        raise ValueError(f"error_type={error_type.__name__}, expected one of {names}")
    return _NATIVE_STATUS[error_type]


def _assert_python_rejects(
    lower: Callable[[], str], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type) as python_error:
        lower()
    assert str(python_error.value) == message


def _assert_native_rejects(
    lower: Callable[[], str], message: str, status: int | None
) -> None:
    """Assert the binding refuses with ``message``, after ``(status N): `` when
    ``status`` is given."""
    with pytest.raises(RuntimeError) as native_error:
        lower()
    text = str(native_error.value)
    suffix = f"(status {status}): {message}" if status is not None else f": {message}"
    assert text.endswith(suffix), text


def _python_llvm(kernel: KernelDef, arch: str, llvm_flavor: str | None) -> str:
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=llvm_flavor)


def _native_llvm(kernel: KernelDef, arch: str, llvm_flavor: str | None) -> str:
    _require_native_engine()
    return _lower_via_cpp_engine(serialize(kernel), arch, llvm_flavor)


def _native_hip(kernel: KernelDef, arch: str) -> str:
    return _require_native_engine().lower_serialized_ir_hip(serialize(kernel), arch)


def lower_with_engine(
    kernel: KernelDef, arch: str, *, engine: Engine, llvm_flavor: str | None = None
) -> str:
    """Lower ``kernel`` for ``arch`` with ``engine``; return its LLVM IR text."""
    _require_engine(engine)
    python_ir = _python_llvm(kernel, arch, llvm_flavor)
    if engine == "python":
        return python_ir
    native_ir = _native_llvm(kernel, arch, llvm_flavor)
    assert native_ir == python_ir, (
        f"the C++ engine lowered kernel '{kernel.name}' for {arch} differently "
        "from the Python engine"
    )
    return native_ir


def assert_engine_rejects(
    kernel: KernelDef,
    arch: str,
    *,
    engine: Engine,
    error_type: type[Exception],
    message: str,
    llvm_flavor: str | None = None,
) -> None:
    """Assert that ``engine`` refuses ``kernel`` for ``arch`` with ``message``.

    The Python engine must raise ``error_type`` with exactly ``message``; the
    C++ engine must fail with ``error_type``'s status and the same message
    after its binding prefix.
    """
    _require_engine(engine)
    if engine == "python":
        _assert_python_rejects(
            lambda: _python_llvm(kernel, arch, llvm_flavor), error_type, message
        )
    else:
        _assert_native_rejects(
            lambda: _native_llvm(kernel, arch, llvm_flavor),
            message,
            _native_status(error_type),
        )


def lower_hip_with_engine(kernel: KernelDef, arch: str, *, engine: Engine) -> str:
    """Lower ``kernel`` to HIP source for ``arch`` with ``engine``; return the text.

    For ``"cpp"`` it also asserts that the text is identical to the Python
    engine's.
    """
    _require_engine(engine)
    python_hip = lower_kernel_to_hip(kernel, arch=arch)
    if engine == "python":
        return python_hip
    native_hip = _native_hip(kernel, arch)
    assert native_hip == python_hip, (
        f"the C++ engine lowered kernel '{kernel.name}' to HIP for {arch} "
        "differently from the Python engine"
    )
    return native_hip


def assert_engine_rejects_hip(
    kernel: KernelDef,
    arch: str,
    *,
    engine: Engine,
    error_type: type[Exception],
    message: str,
) -> None:
    """Assert that ``engine``'s HIP lowering refuses ``kernel`` with ``message``.

    Same contract as :func:`assert_engine_rejects`, for the HIP source.
    """
    _require_engine(engine)
    if engine == "python":
        _assert_python_rejects(
            lambda: lower_kernel_to_hip(kernel, arch=arch), error_type, message
        )
    else:
        _assert_native_rejects(
            lambda: _native_hip(kernel, arch), message, _native_status(error_type)
        )


def lower_with_both_engines(
    kernel: KernelDef, arch: str, llvm_flavor: str | None = None
) -> str:
    """Lower ``kernel`` for ``arch`` with each engine; return the shared IR text."""
    return lower_with_engine(kernel, arch, engine="cpp", llvm_flavor=llvm_flavor)


def assert_both_engines_reject(
    kernel: KernelDef,
    arch: str,
    error_type: type[Exception],
    message: str,
    llvm_flavor: str | None = None,
) -> None:
    """Assert that both engines refuse ``kernel`` for ``arch`` with ``message``.

    The C++ engine's status is not compared; see the module docstring.
    """
    _require_native_engine()
    _assert_python_rejects(
        lambda: _python_llvm(kernel, arch, llvm_flavor), error_type, message
    )
    _assert_native_rejects(
        lambda: _native_llvm(kernel, arch, llvm_flavor), message, None
    )


def lower_hip_with_both_engines(kernel: KernelDef, arch: str) -> str:
    """Lower ``kernel`` to HIP source for ``arch`` with each engine; return the
    shared text."""
    return lower_hip_with_engine(kernel, arch, engine="cpp")


def assert_both_engines_reject_hip(
    kernel: KernelDef,
    arch: str,
    error_type: type[Exception],
    message: str,
) -> None:
    """Assert that both HIP lowerings refuse ``kernel`` for ``arch`` with ``message``.

    The C++ engine's status is not compared; see the module docstring.
    """
    _require_native_engine()
    _assert_python_rejects(
        lambda: lower_kernel_to_hip(kernel, arch=arch), error_type, message
    )
    _assert_native_rejects(lambda: _native_hip(kernel, arch), message, None)


def assert_refused_everywhere(
    kernel: KernelDef,
    arch: str,
    error_type: type[Exception],
    message: str,
    llvm_flavor: str | None = None,
) -> None:
    """Assert that the Python verifier reports ``message`` for ``kernel`` and
    that the LLVM and HIP lowerings of every engine in ``available_engines()``
    refuse it as :func:`assert_engine_rejects` and
    :func:`assert_engine_rejects_hip` check. The C++ verifier is not reached
    through the binding; the native tests check it."""
    assert message in [diagnostic.message for diagnostic in verify(kernel)]
    for engine in available_engines():
        assert_engine_rejects(
            kernel,
            arch,
            engine=engine,
            error_type=error_type,
            message=message,
            llvm_flavor=llvm_flavor,
        )
        assert_engine_rejects_hip(
            kernel, arch, engine=engine, error_type=error_type, message=message
        )


def assert_accepted_everywhere(
    kernel: KernelDef, arch: str, llvm_flavor: str | None = None
) -> None:
    """Assert that the Python verifier reports nothing for ``kernel`` and that
    the LLVM and HIP lowerings of every engine in ``available_engines()`` accept
    it, the two engines emitting the same text where both run."""
    assert verify(kernel) == []
    # The last engine available lowers with the Python engine too, and for
    # "cpp" asserts the texts are identical.
    engine = available_engines()[-1]
    lower_with_engine(kernel, arch, engine=engine, llvm_flavor=llvm_flavor)
    lower_hip_with_engine(kernel, arch, engine=engine)
