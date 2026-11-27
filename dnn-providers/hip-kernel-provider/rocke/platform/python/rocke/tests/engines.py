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
  reports as a ``RuntimeError`` whose text ends with ``"(status N): " +
  message``, where ``N`` is the ``rocke_status_t`` that ``error_type`` maps to.
- ``lower_hip_with_engine`` and ``assert_engine_rejects_hip`` are the same two
  checks for the HIP-source lowering, whose C++ side runs as described below:
  in process, or through the ``rocke_ir_lower_cli`` program.

The C++ engine here is the shared ``librocke``, reached through its C ABI by
``rocke.core.native_engine``: the same serialized-IR entries the
``rocke_engine`` binding wraps, without needing the binding. A test
parametrized over :data:`ENGINES` checks the Python engine in every lane,
including the lanes without the library; only its ``"cpp"`` cases skip there,
naming the reason (for the HIP checks, only where the program is not supplied
either).

Both engines at once: ``lower_with_both_engines``,
``assert_both_engines_reject``, ``lower_hip_with_both_engines`` and
``assert_both_engines_reject_hip`` run the same checks for the Python and the
C++ engine, and skip as a whole when the C++ lowering they need cannot run.
Their refusal checks compare the C++ engine's message but not its status,
which some older refusals do not yet share with the Python engine's exception
type.

The C++ HIP lowering runs in process when the shared library is found, and
otherwise through the ``rocke_ir_lower_cli`` program that
``ROCKE_IR_LOWER_CLI`` names (``tests/core/ir_lower_cli.cpp``); both call the
engine's one serialized-IR HIP entry. ``skip_without_native_hip_lowering``
skips a test or module that has neither. ``hip_outcomes_of_both_engines``
returns each engine's HIP source, or its refusal as a ``HipRefusal`` (error
type and message), for a test that accepts either as long as the two agree.
``python_hip_outcome`` and ``native_hip_outcomes`` are each engine's half, the
second over many kernels in one call, and ``ir_lower_cli_hip_outcomes`` runs
the program alone, for a test that holds it to the in-process entry.
``HipRefusal`` and the exception types that stand for a refusal in both
engines live in ``rocke.tests.hip_outcomes``, which imports no test framework.

A test that loops over the engines itself, rather than through ``pytest``
parameters, iterates ``available_engines()``: the part of :data:`ENGINES` this
run can drive. The installed wheel has no shared library, so such a test runs
the Python engine alone there and both engines wherever the library is
supplied. ``all_engines_backend`` is the ``backend`` argument that runs those
same engines through ``compile_kernel`` and similar entry points.

A kernel family's C++ builder is reached only through the C++ engine's Python
modules (``rocke_engine``, ``rocke_library_engine``), which
``engine_modules_available`` reports. ``lower_spec_with_all_engines`` lowers an
instance spec with both builders where the modules are built, each engine
building the kernel with its own builder (``instance_llvm_from_all_engines``
does the same through an instance's binding entry point, for an instance with
no ``rocke.core.backend.lower_*`` entry point).

The helpers above drive each engine directly. ``assert_backend_lane_rejects``
instead checks one refusal through the public ``lower_kernel_to_llvm``
dispatcher in one row of the source backend matrix (``TESTING.md`` section
5.1.1, one name per row in :data:`BACKEND_LANES`), asserting the exception that
row's caller observes.

Where ``ROCKE_REQUIRE_CPP_ENGINE`` requires the C++ engine (set by the
``rocke.tests.require_cpp_engine`` plugin, which the installed CTest entries
that count as covering it load), a missing shared library fails each helper's
``"cpp"`` case instead of skipping it, and ``available_engines()`` keeps
``"cpp"``: such a run cannot go green without the C++ engine. The C++
engine's Python modules stay optional there.

Every engine this run can drive, plus the verifier:
``assert_refused_everywhere`` and ``assert_accepted_everywhere`` run the Python
verifier and the LLVM and HIP lowerings of each engine in
``available_engines()``. Unlike the both-engine helpers they do not skip when
the library is absent.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal, NoReturn

import pytest

from rocke.core import native_engine
from rocke.core.backend import CPP_STRICT_ENV, _lower_via_cpp_engine, cpp_fallbacks
from rocke.core.ir import KernelDef
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python, lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.tests.hip_outcomes import (
    NATIVE_REFUSAL_STATUS,
    HipRefusal,
    hip_refusal_of,
    python_hip_outcome_of_kernel,
)
from rocke.tests.lanes import (
    BUILD_VARIANT,
    cpp_engine_required,
    import_or_skip_until,
    skip_until,
    skip_until_shared_engine,
)

__all__ = [
    "BACKEND_LANES",
    "ENGINES",
    "IR_LOWER_CLI_ENV",
    "Engine",
    "HipRefusal",
    "all_engines_backend",
    "assert_accepted_everywhere",
    "assert_backend_lane_rejects",
    "assert_both_engines_reject",
    "assert_both_engines_reject_hip",
    "assert_engine_rejects",
    "assert_engine_rejects_hip",
    "assert_refused_everywhere",
    "available_builder_engines",
    "available_engines",
    "engine_modules_available",
    "hide_shared_engine",
    "hip_outcomes_of_both_engines",
    "instance_llvm_from_all_engines",
    "ir_lower_cli_hip_outcomes",
    "lower_hip_with_both_engines",
    "lower_hip_with_engine",
    "lower_spec_with_all_engines",
    "lower_with_both_engines",
    "lower_with_engine",
    "native_engine_or_skip",
    "native_hip_outcomes",
    "python_hip_outcome",
    "skip_without_cpp_engine",
    "skip_without_native_hip_lowering",
]

Engine = Literal["python", "cpp"]
ENGINES: tuple[Engine, ...] = ("python", "cpp")

# The C++ engine's refusal text, in process or from rocke_ir_lower_cli:
# "<entry point>: lower failed for arch '<gfx>' (status <code>): <message>".
_NATIVE_REFUSAL = re.compile(r"\(status (\d+)\): (.*)", re.DOTALL)

#: The environment variable naming a built ``rocke_ir_lower_cli``, which runs
#: the C++ HIP lowering where the binding is not built.
IR_LOWER_CLI_ENV = "ROCKE_IR_LOWER_CLI"

# Bounds a hung rocke_ir_lower_cli run, which lowers every kernel it is given.
# It sits below the 120 s per-test limit of the rocke_hip_source_identity CTest,
# whose calls lower whole families, so a hang there is reported as this timeout
# rather than as a pytest-timeout kill; a lane with a lower per-test limit
# reports it as that limit's kill instead.
_IR_LOWER_CLI_TIMEOUT_S = 90


def available_engines() -> tuple[Engine, ...]:
    """:data:`ENGINES` when the C++ engine's shared library is found
    (``rocke.core.native_engine.native_lowering_available``) or
    ``ROCKE_REQUIRE_CPP_ENGINE`` requires it, else ``("python",)``: where it is
    required, a missing library fails the ``"cpp"`` lowering instead of
    dropping it.

    For a test that loops over the engines itself; a ``pytest`` test
    parametrizes over :data:`ENGINES` instead.
    """
    if cpp_engine_required() or native_engine.native_lowering_available():
        return ENGINES
    return ("python",)


def all_engines_backend() -> str:
    """The ``backend`` that lowers with every engine in ``available_engines()``:
    ``"both"`` when the C++ engine's shared library is found, else
    ``"python"``."""
    return "both" if "cpp" in available_engines() else "python"


def available_builder_engines() -> tuple[Engine, ...]:
    """The part of :data:`ENGINES` whose kernel-family builders this run can
    drive: :data:`ENGINES` when the C++ engine's Python modules are built
    (:func:`engine_modules_available`), else ``("python",)``.

    For a test that builds a kernel with each engine's own family builder; a
    test that lowers one Python-built kernel with each engine iterates
    :func:`available_engines`.
    """
    return ENGINES if engine_modules_available() else ("python",)


def engine_modules_available() -> bool:
    """Whether the C++ engine's Python modules are built: the only route to a
    kernel family's C++ builder (``rocke_engine.<family>_lower_llvm`` and the
    like). Family-agnostic lowering needs only the shared library; see
    :func:`available_engines`."""
    return importlib.util.find_spec("rocke_engine") is not None


def lower_spec_with_all_engines(
    spec: object,
    arch: str,
    *,
    build_kernel: Callable[..., KernelDef],
    lower_spec: Callable[..., object],
) -> str:
    """The LLVM IR of the instance ``spec`` describes, for ``arch``, from every
    engine in ``available_engines()``.

    With the C++ engine's Python modules (:func:`engine_modules_available`),
    each engine's builder builds the kernel from the spec
    (``lower_spec(spec, arch=arch, backend="both")``, an instance's
    ``rocke.core.backend.lower_*`` entry point), both kernels are lowered, and
    the two texts must be the same byte for byte. Without them,
    ``build_kernel(spec, arch=arch)``, the Python builder, lowered by the
    default lowerer; a ``lower_*`` entry point needs the modules even for the
    Python backend.
    """
    if engine_modules_available():
        return lower_spec(spec, arch=arch, backend="both").llvm_text
    return lower_kernel_to_llvm(build_kernel(spec, arch=arch), arch=arch)


def instance_llvm_from_all_engines(
    spec: object,
    arch: str,
    *,
    build_kernel: Callable[..., KernelDef],
    binding: str,
) -> str:
    """The LLVM IR of the instance ``spec`` describes, for ``arch``: the Python
    builder's kernel (``build_kernel(spec, arch=arch)``) lowered by the Python
    engine and, with the ``rocke_engine`` binding, the C++ builder's, through
    the binding's ``binding`` entry point (``<family>_lower_llvm``) fed the
    spec's fields that are not None; the two texts must be the same byte for
    byte.

    For an instance with no ``rocke.core.backend.lower_*`` entry point
    (:func:`lower_spec_with_all_engines` covers those).
    """
    python_ir = _python_llvm(build_kernel(spec, arch=arch), arch, None)
    if engine_modules_available():
        fields = {
            name: value
            for name, value in dataclasses.asdict(spec).items()
            if value is not None
        }
        native_ir = getattr(_require_native_engine(), binding)(fields, arch=arch)
        assert (
            native_ir == python_ir
        ), f"the C++ builder emitted different LLVM IR for {binding} on {arch}"
    return python_ir


def _require_native_engine() -> ModuleType:
    """The ``rocke_engine`` binding; skip the calling test, as waiting for the
    build-variant lane, when it is not built."""
    return import_or_skip_until("rocke_engine", BUILD_VARIANT)


def native_engine_or_skip() -> ModuleType:
    """:mod:`rocke.core.native_engine`, the C++ engine's serialized-IR
    lowering. When no shared library is found, skip the calling test (or
    module, at import) as waiting for the build-variant lane, or, where
    ``ROCKE_REQUIRE_CPP_ENGINE`` requires the C++ engine, fail it. A library
    that is found but does not load, or is from another build, fails the
    test."""
    __tracebackhide__ = True  # pytest reports the skip at the caller
    try:
        native_engine.require_native_lowering()
    except native_engine.SharedEngineNotFound as missing:
        skip_until_shared_engine(str(missing))
    return native_engine


def _validate_engine(engine: Engine) -> None:
    if engine not in ENGINES:
        raise ValueError(f"engine={engine!r}, expected one of {ENGINES}")


def _require_engine(engine: Engine) -> None:
    """Validate ``engine``; for ``"cpp"``, skip the calling test when the
    shared library is not found, before any engine runs."""
    _validate_engine(engine)
    if engine == "cpp":
        native_engine_or_skip()


def _require_hip_engine(engine: Engine) -> None:
    """Validate ``engine``; for ``"cpp"``, skip the calling test when the C++
    HIP lowering cannot run (see :func:`skip_without_native_hip_lowering`),
    before any engine runs."""
    _validate_engine(engine)
    if engine == "cpp":
        skip_without_native_hip_lowering()


def _native_status(error_type: type[Exception]) -> int:
    if error_type not in NATIVE_REFUSAL_STATUS:
        names = sorted(t.__name__ for t in NATIVE_REFUSAL_STATUS)
        raise ValueError(f"error_type={error_type.__name__}, expected one of {names}")
    return NATIVE_REFUSAL_STATUS[error_type]


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
    """Assert the C++ engine refuses with ``message``, after ``(status N): ``
    when ``status`` is given."""
    with pytest.raises(RuntimeError) as native_error:
        lower()
    text = str(native_error.value)
    assert text.endswith(_native_refusal_suffix(message, status)), text


def _native_refusal_suffix(message: str, status: int | None) -> str:
    """How the C++ engine's refusal text ends: ``"(status N): " + message``,
    or ``": " + message`` when ``status`` is not compared."""
    return f"(status {status}): {message}" if status is not None else f": {message}"


def _python_llvm(kernel: KernelDef, arch: str, llvm_flavor: str | None) -> str:
    return _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor=llvm_flavor)


def _native_llvm(kernel: KernelDef, arch: str, llvm_flavor: str | None) -> str:
    native_engine_or_skip()
    return _lower_via_cpp_engine(serialize(kernel), arch, llvm_flavor)


def _native_hip_outcome(kernel: KernelDef, arch: str) -> str | HipRefusal:
    (outcome,) = native_hip_outcomes([(serialize(kernel), arch)])
    return outcome


def _assert_native_hip_rejects(
    kernel: KernelDef, arch: str, message: str, status: int | None
) -> None:
    """Assert the C++ HIP lowering refuses ``kernel`` with exactly ``message``,
    and with ``status`` when it is given."""
    outcome = _native_hip_outcome(kernel, arch)
    assert isinstance(outcome, HipRefusal), (
        f"the C++ engine lowered kernel '{kernel.name}' to HIP for {arch}; "
        f"expected it to refuse with: {message}"
    )
    assert outcome.message == message, outcome
    if status is not None:
        assert outcome == _native_status_refusal(status, message), outcome


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
    after its entry-point prefix.
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
    _require_hip_engine(engine)
    python_hip = lower_kernel_to_hip(kernel, arch=arch)
    if engine == "python":
        return python_hip
    native_outcome = _native_hip_outcome(kernel, arch)
    refused = (
        f"; it refused: {native_outcome}"
        if isinstance(native_outcome, HipRefusal)
        else ""
    )
    assert native_outcome == python_hip, (
        f"the C++ engine lowered kernel '{kernel.name}' to HIP for {arch} "
        f"differently from the Python engine{refused}"
    )
    return python_hip


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
    _require_hip_engine(engine)
    if engine == "python":
        _assert_python_rejects(
            lambda: lower_kernel_to_hip(kernel, arch=arch), error_type, message
        )
    else:
        _assert_native_hip_rejects(kernel, arch, message, _native_status(error_type))


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
    native_engine_or_skip()
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
    skip_without_native_hip_lowering()
    _assert_python_rejects(
        lambda: lower_kernel_to_hip(kernel, arch=arch), error_type, message
    )
    _assert_native_hip_rejects(kernel, arch, message, None)


# The rows of the source backend matrix in TESTING.md section 5.1.1: each
# lane's ROCKE_BACKEND and ROCKE_CPP_STRICT (None leaves the variable unset).
_LANE_ENVIRONMENT: dict[str, tuple[str | None, str | None]] = {
    "installed-fallback": (None, None),
    "python": ("python", None),
    "cpp-permissive": ("cpp", "0"),
    "cpp-strict": ("cpp", "1"),
    "both": ("both", "1"),
}

BACKEND_LANES: tuple[str, ...] = tuple(_LANE_ENVIRONMENT)

# The lanes in which the dispatcher is expected to fall back to the Python
# engine: the C++ engine refuses (cpp-permissive) or is absent
# (installed-fallback).
_FALLBACK_LANES = frozenset({"cpp-permissive", "installed-fallback"})


def assert_backend_lane_rejects(
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    kernel: KernelDef,
    arch: str,
    *,
    error_type: type[Exception],
    message: str,
    llvm_flavor: str,
) -> None:
    """Assert that ``lower_kernel_to_llvm`` refuses ``kernel`` in backend ``lane``.

    ``lane`` is one of :data:`BACKEND_LANES`. The lane's ``ROCKE_BACKEND`` and
    ``ROCKE_CPP_STRICT`` are set through ``monkeypatch``, and an inherited value
    is cleared where the row leaves the variable unset. ``installed-fallback``
    also hides the C++ engine's shared library, as in the installed wheel; the
    ``cpp-*`` and ``both`` lanes skip, naming the reason, when it is not found.

    In the ``cpp-strict`` lane the caller sees the C++ engine's own refusal: a
    ``RuntimeError`` (not a subclass) whose text ends with
    ``"(status N): " + message``, ``N`` being the status ``error_type`` maps to,
    as in :func:`assert_engine_rejects`. In every other lane it sees the Python
    engine's: exactly ``error_type`` (not a subclass) with exactly ``message``.
    The two fallback lanes must also have fallen back for the expected reason:
    ``cpp-permissive`` after the C++ engine refused with ``message``,
    ``installed-fallback`` after the shared library was not found. Only these two
    lanes silence the fallback warning (``ROCKE_CPP_QUIET_FALLBACK``); every
    other lane clears it and must not fall back at all. ``error_type`` must be
    one the C++ status map covers, as for :func:`assert_engine_rejects`; that
    is checked in every lane. ``llvm_flavor`` is required because the
    ``cpp`` lanes cannot resolve a flavor without a compiler.
    """
    if lane not in _LANE_ENVIRONMENT:
        raise ValueError(f"lane={lane!r}, expected one of {BACKEND_LANES}")
    status = _native_status(error_type)
    if lane not in ("python", "installed-fallback"):
        # Whether a missing engine skips or fails is the run's setting
        # (ROCKE_REQUIRE_CPP_ENGINE), checked before the lane's own.
        native_engine_or_skip()
    backend, strict = _LANE_ENVIRONMENT[lane]
    for variable, value in (("ROCKE_BACKEND", backend), (CPP_STRICT_ENV, strict)):
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
    if lane in _FALLBACK_LANES:
        # The fallback is asserted below, so its stderr warning would be noise.
        monkeypatch.setenv("ROCKE_CPP_QUIET_FALLBACK", "1")
    else:
        # A fallback here is unexpected: keep its warning, even if inherited
        # from the caller's environment.
        monkeypatch.delenv("ROCKE_CPP_QUIET_FALLBACK", raising=False)
    if lane == "installed-fallback":
        hide_shared_engine(monkeypatch)
    fallbacks_before = len(cpp_fallbacks())
    with pytest.raises(Exception) as refusal:
        lower_kernel_to_llvm(kernel, arch=arch, llvm_flavor=llvm_flavor)
    fallbacks = [reason for _, reason in cpp_fallbacks()[fallbacks_before:]]
    if lane == "cpp-strict":
        assert type(refusal.value) is RuntimeError, repr(refusal.value)
        assert str(refusal.value).endswith(
            _native_refusal_suffix(message, status)
        ), str(refusal.value)
    else:
        assert type(refusal.value) is error_type, repr(refusal.value)
        assert str(refusal.value) == message, str(refusal.value)
    if lane == "cpp-permissive":
        assert len(fallbacks) == 1 and fallbacks[0].endswith(": " + message), fallbacks
    elif lane == "installed-fallback":
        assert len(fallbacks) == 1 and fallbacks[0].startswith(
            "BackendError: "
        ), fallbacks
    else:
        assert fallbacks == [], fallbacks


def hide_shared_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give this test no shared C++ engine, as in the installed wheel, which
    ships none, whatever this build has: ``ROCKE_ONLINE_LIB=none``
    (``native_engine.NO_SHARED_ENGINE``), set through ``monkeypatch``. The cpp
    backend then fails over with ``BackendError``."""
    monkeypatch.setenv(native_engine.SHARED_ENGINE_ENV, native_engine.NO_SHARED_ENGINE)


def assert_refused_everywhere(
    kernel: KernelDef,
    arch: str,
    error_type: type[Exception],
    message: str,
    llvm_flavor: str | None = None,
    *,
    engines: tuple[Engine, ...] | None = None,
) -> None:
    """Assert that the Python verifier reports ``message`` for ``kernel`` and
    that the LLVM and HIP lowerings of each of ``engines`` (default:
    ``available_engines()``) refuse it as :func:`assert_engine_rejects` and
    :func:`assert_engine_rejects_hip` check. ``engines=("python",)`` holds a
    kernel only the Python engine can receive (one serialized IR cannot
    spell) to the Python engine alone. The C++ verifier is not reached through
    the binding; the native tests check it."""
    assert message in [diagnostic.message for diagnostic in verify(kernel)]
    for engine in available_engines() if engines is None else engines:
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
    its refusal: ``ir_text`` parsed, then ``python_hip_outcome_of_kernel``. Any
    exception parsing or lowering raises is its refusal, so a crash shows as a
    refusal that the C++ engine does not share."""
    try:
        kernel = parse(ir_text)
    except Exception as error:  # noqa: BLE001 -- a refusal is an outcome
        return hip_refusal_of(error)
    return python_hip_outcome_of_kernel(kernel, arch)


def _native_refusal(error_text: str) -> HipRefusal | None:
    """The C++ engine's refusal reported in ``error_text``, or None when the
    text is not a refusal (a failure of the transport itself)."""
    refusal = _NATIVE_REFUSAL.search(error_text)
    if refusal is None:
        return None
    return _native_status_refusal(int(refusal.group(1)), refusal.group(2))


def _native_status_refusal(status: int, message: str) -> HipRefusal:
    """The refusal the C++ engine's ``status`` code and ``message`` stand for."""
    error_type = next(
        (kind for kind, code in NATIVE_REFUSAL_STATUS.items() if code == status), None
    )
    type_name = error_type.__name__ if error_type else f"status {status}"
    return HipRefusal(type_name, message)


def skip_without_cpp_engine(error: Exception, reason: str) -> NoReturn:
    """Skip the calling test, for ``reason``, after the C++ engine raised
    ``error`` (a ``rocke.core.backend.BackendError``): as waiting for the
    build-variant lane when no shared library was found (the error's cause is
    ``rocke.core.native_engine.SharedEngineNotFound``; failing instead where
    ``ROCKE_REQUIRE_CPP_ENGINE`` requires the C++ engine), as waiting for the
    build-variant lane when an engine module is not installed (the cause is the
    ``ModuleNotFoundError`` ``import_engine`` raises it from), else as a plain
    skip, so an engine that is built but fails to load is not booked as waiting
    for a lane."""
    __tracebackhide__ = True  # pytest reports the skip at the caller
    if isinstance(error.__cause__, native_engine.SharedEngineNotFound):
        skip_until_shared_engine(reason)
    if isinstance(error.__cause__, ModuleNotFoundError):
        skip_until(BUILD_VARIANT, reason=reason)
    pytest.skip(reason)


def skip_without_native_hip_lowering() -> None:
    """Skip the calling test, or the module when called at import, as waiting
    for the build-variant lane, unless the C++ HIP lowering can run: the
    shared library is found, or ``ROCKE_IR_LOWER_CLI`` names the
    ``rocke_ir_lower_cli`` program. Where ``ROCKE_REQUIRE_CPP_ENGINE`` requires
    the C++ engine, only the shared library counts, and its absence fails."""
    if native_engine.native_lowering_available():
        return
    if os.environ.get(IR_LOWER_CLI_ENV) and not cpp_engine_required():
        return
    skip_until_shared_engine(
        f"needs the shared librocke ({native_engine.SHARED_ENGINE_ENV}), or "
        f"{IR_LOWER_CLI_ENV} naming a built rocke_ir_lower_cli"
    )


def native_hip_outcomes(
    kernels: Sequence[tuple[str, str]],
) -> list[str | HipRefusal]:
    """The C++ engine's HIP source, or its refusal, for each ``(serialized
    kernel, arch)`` of ``kernels``, in order: in process when the shared
    library is found, else in one run of the ``rocke_ir_lower_cli`` program
    ``ROCKE_IR_LOWER_CLI`` names. Where ``ROCKE_REQUIRE_CPP_ENGINE`` requires
    the C++ engine, a missing library fails the caller instead."""
    if cpp_engine_required():
        native_engine_or_skip()
    if not native_engine.native_lowering_available():
        return ir_lower_cli_hip_outcomes(kernels)
    outcomes: list[str | HipRefusal] = []
    for ir_text, arch in kernels:
        try:
            outcomes.append(native_engine.lower_serialized_ir_to_hip(ir_text, arch))
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
        skip_until(
            BUILD_VARIANT,
            reason=f"set {IR_LOWER_CLI_ENV} to a built rocke_ir_lower_cli",
        )
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
