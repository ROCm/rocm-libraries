# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared helpers for the logicalIR adaptor shim.

This module provides factories used by every rocisa.* shim submodule to
produce "dummy" symbols whose only side-effect is printing their fully
qualified rocisa path when the symbol is used.

Design:
  - class shim : prints when instantiated (``__init__``)
  - function shim: prints when called (``__call__``)
  - enum shim  : a plain class with class-level attributes that mimics the
                 original nanobind ``nb::enum_`` + ``export_values()`` pair.

The intent at this stage is ONLY structural: every name the upstream
KernelWriter code imports must exist and must do *something* observable
when touched. Real dispatch into logicalIR is deferred to a later pass.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Type


def make_dummy_class(full_name: str) -> Type[Any]:
    """Create a dummy class whose ``__init__`` prints ``full_name``.

    Calls such as ``BufferLoadB128(dst=...)`` will print
    ``rocisa.instruction.BufferLoadB128`` and return a dummy instance.
    Arbitrary attribute access on the instance returns another dummy
    callable so that chained method calls (e.g. ``.add(...)``) never
    raise AttributeError during the structural-only phase.
    """

    short = full_name.rsplit(".", 1)[-1]

    class _DummyInstance:
        __slots__ = ("_full_name",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            object.__setattr__(self, "_full_name", full_name)
            print(full_name)

        def __getattr__(self, name: str) -> Any:
            def _noop(*args: Any, **kwargs: Any) -> None:
                return None

            return _noop

        def __setattr__(self, name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        def __repr__(self) -> str:
            return f"<DummyShim {full_name}>"

    _DummyInstance.__name__ = short
    _DummyInstance.__qualname__ = short
    _DummyInstance.__module__ = full_name.rsplit(".", 1)[0]
    return _DummyInstance


def make_dummy_func(full_name: str):
    """Create a dummy function that prints ``full_name`` when called."""

    short = full_name.rsplit(".", 1)[-1]

    def _dummy(*args: Any, **kwargs: Any) -> None:
        print(full_name)
        return None

    _dummy.__name__ = short
    _dummy.__qualname__ = short
    _dummy.__module__ = full_name.rsplit(".", 1)[0]
    return _dummy


def make_dummy_enum(full_name: str, values: Iterable[str]) -> Type[Any]:
    """Create a dummy enum-like class mirroring ``nb::enum_ + export_values``.

    Each value is an ``int`` member attached both to the class itself and
    returned to the caller so the caller can also re-export them at
    module scope (matching nanobind's ``export_values()`` behaviour).
    Instantiating the enum class prints the full qualified name.
    """

    short = full_name.rsplit(".", 1)[-1]
    values = list(values)

    class _DummyEnum:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            print(full_name)

        def __repr__(self) -> str:
            return f"<DummyEnum {full_name}>"

    _DummyEnum.__name__ = short
    _DummyEnum.__qualname__ = short
    _DummyEnum.__module__ = full_name.rsplit(".", 1)[0]

    for i, v in enumerate(values):
        setattr(_DummyEnum, v, i)

    return _DummyEnum


def export_enum_values(target_namespace: Mapping[str, Any], enum_cls: Type[Any],
                        values: Iterable[str]) -> None:
    """Replicate nanobind's ``.export_values()`` at Python module scope.

    Usage:
        _my = make_dummy_enum("rocisa.enum.SelectBit", ["SEL_NONE", "DWORD", ...])
        export_enum_values(globals(), _my, ["SEL_NONE", "DWORD", ...])
    """
    for v in values:
        target_namespace[v] = getattr(enum_cls, v)  # type: ignore[index]
