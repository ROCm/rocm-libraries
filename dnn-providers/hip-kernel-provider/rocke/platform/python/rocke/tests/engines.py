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

``hip_outcomes_of_both_engines`` returns each engine's HIP source, or its
refusal as a ``HipRefusal`` (error type and message), for a test that accepts
either as long as the two agree. Where the binding is not built it runs the C++
HIP lowering through the ``rocke_ir_lower_cli`` program that
``ROCKE_IR_LOWER_CLI`` names (``tests/core/ir_lower_cli.cpp``), as the
installed test tree does. ``skip_without_native_hip_lowering`` skips a test or
module that has neither, and ``ir_lower_cli_hip_outcome`` runs the program
alone, for a test that holds it to the binding. ``REFUSAL_ERROR_TYPES`` are the
exception types that stand for a refusal in both engines.

A test that loops over the engines itself, rather than through ``pytest``
parameters, iterates ``available_engines()``: the part of :data:`ENGINES` this
run can drive. The installed wheel has no ``rocke_engine`` binding, so such a
test runs the Python engine alone there and both engines wherever the binding
is supplied. ``all_engines_backend`` is the ``backend`` argument that runs those
same engines through ``compile_kernel`` and similar entry points.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import pytest

from rocke.core.backend import _lower_via_cpp_engine
from rocke.core.ir import KernelDef
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

__all__ = [
    "ENGINES",
    "Engine",
    "HipRefusal",
    "IR_LOWER_CLI_ENV",
    "REFUSAL_ERROR_TYPES",
    "HipRefusal",
    "all_engines_backend",
    "assert_both_engines_reject",
    "assert_both_engines_reject_hip",
    "assert_engine_rejects",
    "assert_engine_rejects_hip",
    "available_engines",
    "hip_outcomes_of_both_engines",
    "ir_lower_cli_hip_outcome",
    "lower_hip_with_both_engines",
    "lower_hip_with_engine",
    "lower_with_both_engines",
    "lower_with_engine",
    "skip_without_native_hip_lowering",
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

#: The exception types a lowerer or builder raises to refuse a kernel: those a
#: C++ engine status code stands for. Any other exception is a failure.
REFUSAL_ERROR_TYPES: tuple[type[Exception], ...] = tuple(_NATIVE_STATUS)

# The C++ engine's refusal text, from the binding or rocke_ir_lower_cli:
# "<entry point>: lower failed for arch '<gfx>' (status <code>): <message>".
_NATIVE_REFUSAL = re.compile(r"\(status (\d+)\): (.*)", re.DOTALL)

#: The environment variable naming a built ``rocke_ir_lower_cli``, which
#: ``hip_outcomes_of_both_engines`` runs where the binding is not built.
IR_LOWER_CLI_ENV = "ROCKE_IR_LOWER_CLI"

# A generous bound on one kernel's C++ HIP lowering through the program; the
# slowest parity kernel takes well under a second.
_IR_LOWER_CLI_TIMEOUT_S = 60


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
    """Assert ``lower`` raises ``error_type`` with exactly ``message`` as its
    first argument, so a ``KeyError`` compares unquoted."""
    with pytest.raises(error_type) as python_error:
        lower()
    assert python_error.value.args[0] == message


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
    skip_without_native_hip_lowering()
    python_outcome: str | HipRefusal
    try:
        python_outcome = lower_kernel_to_hip(parse(ir_text), arch=arch)
    except Exception as error:  # noqa: BLE001 -- a refusal is an outcome
        error_type = next(
            (kind for kind in _NATIVE_STATUS if isinstance(error, kind)), type(error)
        )
        message = error.args[0] if error.args else ""
        python_outcome = HipRefusal(error_type.__name__, message)
    return python_outcome, _native_hip_outcome(ir_text, arch)


def _native_refusal(error_text: str) -> HipRefusal | None:
    """The C++ engine's refusal reported in ``error_text``, or None when the
    text is not a refusal (a failure of the binding or program itself)."""
    refusal = _NATIVE_REFUSAL.search(error_text)
    if refusal is None:
        return None
    status, message = int(refusal.group(1)), refusal.group(2)
    error_type = next(
        (kind for kind, code in _NATIVE_STATUS.items() if code == status), None
    )
    type_name = error_type.__name__ if error_type else f"status {status}"
    return HipRefusal(type_name, message)


def skip_without_native_hip_lowering() -> None:
    """Skip the calling test, or the module when called at import, unless the
    C++ HIP lowering can run: the ``rocke_engine`` binding is built, or
    ``ROCKE_IR_LOWER_CLI`` names the ``rocke_ir_lower_cli`` program."""
    if importlib.util.find_spec("rocke_engine") is None and not os.environ.get(
        IR_LOWER_CLI_ENV
    ):
        pytest.skip(
            f"needs the rocke_engine binding, or {IR_LOWER_CLI_ENV} naming a "
            "built rocke_ir_lower_cli",
            allow_module_level=True,
        )


def _native_hip_outcome(ir_text: str, arch: str) -> str | HipRefusal:
    """The C++ engine's HIP source for the serialized kernel ``ir_text``, or its
    refusal: through the ``rocke_engine`` binding when it is built, else through
    the ``rocke_ir_lower_cli`` program ``ROCKE_IR_LOWER_CLI`` names."""
    if importlib.util.find_spec("rocke_engine") is None:
        return ir_lower_cli_hip_outcome(ir_text, arch)
    engine = importlib.import_module("rocke_engine")
    try:
        return engine.lower_serialized_ir_hip(ir_text, arch)
    except RuntimeError as error:
        refusal = _native_refusal(str(error))
        if refusal is None:
            raise
        return refusal


def ir_lower_cli_hip_outcome(ir_text: str, arch: str) -> str | HipRefusal:
    """The C++ engine's HIP source for the serialized kernel ``ir_text``, or its
    refusal, through the ``rocke_ir_lower_cli`` program ``ROCKE_IR_LOWER_CLI``
    names; skips when the variable is unset. A name that is not a file is a
    broken setup and fails."""
    program = os.environ.get(IR_LOWER_CLI_ENV)
    if not program:
        pytest.skip(f"set {IR_LOWER_CLI_ENV} to a built rocke_ir_lower_cli")
    path = Path(program).resolve()
    assert path.is_file(), f"{IR_LOWER_CLI_ENV} names {path}, not a file"
    # Bytes both ways: the program writes the engine's text unchanged, so no
    # newline translation may touch it.
    run = subprocess.run(
        [str(path), "--hip", arch],
        input=ir_text.encode("utf-8"),
        capture_output=True,
        check=False,  # a non-zero exit carries the refusal
        timeout=_IR_LOWER_CLI_TIMEOUT_S,
    )
    if run.returncode == 0:
        return run.stdout.decode("utf-8")
    error_text = run.stderr.decode("utf-8").removesuffix("\n")
    refusal = _native_refusal(error_text)
    assert (
        refusal is not None
    ), f"{path.name} exited {run.returncode} without a refusal: {error_text}"
    return refusal
