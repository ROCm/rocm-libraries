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

All five skip, naming the reason, when the ``rocke_engine`` binding is not built,
except that ``hip_outcomes_of_both_engines`` then runs the C++ HIP lowering
through the ``rocke_ir_lower_cli`` program that ``ROCKE_IR_LOWER_CLI`` names
(``tests/core/ir_lower_cli.cpp``), as the installed test tree does.
``skip_without_native_hip_lowering`` skips a test or module that has neither.
``python_hip_outcome`` and ``native_hip_outcomes`` are each engine's half, the
second over many kernels in one call, and ``ir_lower_cli_hip_outcomes`` runs the
program alone, for a test that holds it to the binding. ``REFUSAL_ERROR_TYPES``
are the exception types that stand for a refusal in both engines.
``available_engines`` names the engines a test that drives each engine itself can
use: the installed wheel has no ``rocke_engine`` binding, so such a test runs the
Python engine alone there and both engines wherever the binding is supplied.
``all_engines_backend`` is the ``backend`` argument that runs those same engines
through ``compile_kernel`` and similar entry points.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from rocke.core.backend import _lower_via_cpp_engine
from rocke.core.ir import KernelDef
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

__all__ = [
    "IR_LOWER_CLI_ENV",
    "REFUSAL_ERROR_TYPES",
    "HipRefusal",
    "all_engines_backend",
    "assert_both_engines_reject",
    "assert_both_engines_reject_hip",
    "available_engines",
    "hip_outcomes_of_both_engines",
    "ir_lower_cli_hip_outcomes",
    "lower_hip_with_both_engines",
    "lower_with_both_engines",
    "native_hip_outcomes",
    "python_hip_outcome",
    "skip_without_native_hip_lowering",
]

# rocke_status_t (cpp/include/rocke/ir.h): the code the C++ engine reports for
# each Python exception type the lowerers raise for a kernel they refuse.
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

# A generous bound on one rocke_ir_lower_cli run, which lowers every kernel it
# is given: a parity family's kernels take a few seconds at most.
_IR_LOWER_CLI_TIMEOUT_S = 90


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
    refuse with the same error type and message.
    """
    skip_without_native_hip_lowering()
    (native_outcome,) = native_hip_outcomes([(ir_text, arch)])
    return python_hip_outcome(ir_text, arch), native_outcome


def python_hip_outcome(ir_text: str, arch: str) -> str | HipRefusal:
    """The Python engine's HIP source for the serialized kernel ``ir_text``, or
    its refusal. Any exception the lowering raises is its refusal, so a crash
    shows as a refusal that the C++ engine does not share."""
    try:
        return lower_kernel_to_hip(parse(ir_text), arch=arch)
    except Exception as error:  # noqa: BLE001 -- a refusal is an outcome
        error_type = next(
            (kind for kind in _NATIVE_STATUS if isinstance(error, kind)), type(error)
        )
        return HipRefusal(error_type.__name__, error.args[0] if error.args else "")


def _native_refusal(error_text: str) -> HipRefusal | None:
    """The C++ engine's refusal reported in the binding's ``error_text``, or
    None when the text is not a refusal (a failure of the binding itself)."""
    refusal = _NATIVE_REFUSAL.search(error_text)
    if refusal is None:
        return None
    return _native_status_refusal(int(refusal.group(1)), refusal.group(2))


def _native_status_refusal(status: int, message: str) -> HipRefusal:
    """The refusal the C++ engine's ``status`` code and ``message`` stand for."""
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


def native_hip_outcomes(
    kernels: Sequence[tuple[str, str]],
) -> list[str | HipRefusal]:
    """The C++ engine's HIP source, or its refusal, for each ``(serialized
    kernel, arch)`` of ``kernels``, in order: through the ``rocke_engine``
    binding when it is built, else in one run of the ``rocke_ir_lower_cli``
    program ``ROCKE_IR_LOWER_CLI`` names."""
    if importlib.util.find_spec("rocke_engine") is None:
        return ir_lower_cli_hip_outcomes(kernels)
    engine = importlib.import_module("rocke_engine")
    outcomes: list[str | HipRefusal] = []
    for ir_text, arch in kernels:
        try:
            outcomes.append(engine.lower_serialized_ir_hip(ir_text, arch))
        except RuntimeError as error:
            refusal = _native_refusal(str(error))
            if refusal is None:
                raise
            outcomes.append(refusal)
    return outcomes


def ir_lower_cli_hip_outcomes(
    kernels: Sequence[tuple[str, str]],
) -> list[str | HipRefusal]:
    """The C++ engine's HIP source, or its refusal, for each ``(serialized
    kernel, arch)`` of ``kernels``, in order, from one ``rocke_ir_lower_cli
    --hip-batch`` run of the program ``ROCKE_IR_LOWER_CLI`` names (its frame
    format is in ``tests/core/ir_lower_cli.cpp``); skips when the variable is
    unset. A name that is not a file is a broken setup and fails."""
    program = os.environ.get(IR_LOWER_CLI_ENV)
    if not program:
        pytest.skip(f"set {IR_LOWER_CLI_ENV} to a built rocke_ir_lower_cli")
    path = Path(program).resolve()
    assert path.is_file(), f"{IR_LOWER_CLI_ENV} names {path}, not a file"
    request = bytearray()
    for ir_text, arch in kernels:
        ir_bytes = ir_text.encode("utf-8")
        request += f"{arch} {len(ir_bytes)}\n".encode("ascii") + ir_bytes
    run = subprocess.run(
        [str(path), "--hip-batch"],
        input=bytes(request),
        capture_output=True,
        check=False,  # the assert below reports the program's stderr
        timeout=_IR_LOWER_CLI_TIMEOUT_S,
    )
    assert run.returncode == 0, (
        f"{path.name} --hip-batch exited {run.returncode}: "
        f"{run.stderr.decode('utf-8', 'replace')}"
    )
    outcomes: list[str | HipRefusal] = []
    reply = run.stdout
    position = 0
    while position < len(reply):
        header_end = reply.find(b"\n", position)
        header = reply[position:header_end].split()
        assert header_end >= 0 and len(header) == 2, (
            f"{path.name} --hip-batch reply frame {len(outcomes)} has a malformed "
            f"header: {reply[position:position + 40]!r}"
        )
        status, size = int(header[0]), int(header[1])
        body = reply[header_end + 1 : header_end + 1 + size]
        assert len(body) == size, (
            f"{path.name} --hip-batch reply frame {len(outcomes)} is "
            f"{len(body)} of its {size} bytes"
        )
        position = header_end + 1 + size
        text = body.decode("utf-8")
        outcomes.append(text if status == 0 else _native_status_refusal(status, text))
    assert len(outcomes) == len(
        kernels
    ), f"{path.name} --hip-batch answered {len(outcomes)} of {len(kernels)} kernels"
    return outcomes
