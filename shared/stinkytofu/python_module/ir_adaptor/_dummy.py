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

import enum as _stdenum
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
    """Create a real ``IntEnum`` mirroring ``nb::enum_`` + ``export_values``.

    Each member exposes ``.name`` and ``.value`` (matching nanobind), is an
    ``int`` itself (so ``DataTypeEnum.Float == 0`` keeps working), and the
    class is callable as ``DataTypeEnum(0)``. The numeric value of each
    member is its 0-based index in ``values`` — this matches the implicit
    ``nb::enum_<...>::value(...)`` ordering used by ``rocisa::enum.cpp``.

    Note (was originally a "structural-only" dummy):
        Tensile's import-time machinery in ``Tensile/Common/DataType.py``
        reads ``e['enum'].value`` and ``e['enum'].name`` while building the
        ``DataType`` lookup table, so a bare ``int`` placeholder is not
        enough to pass ``import Tensile``. ``IntEnum`` gives us both the
        attribute surface and the raw-int behaviour the rest of the code
        treats it as.
    """

    short = full_name.rsplit(".", 1)[-1]
    values = list(values)
    module = full_name.rsplit(".", 1)[0]

    cls = _stdenum.IntEnum(short, [(v, i) for i, v in enumerate(values)])
    cls.__module__ = module
    cls.__qualname__ = short
    return cls


def export_enum_values(target_namespace: Mapping[str, Any], enum_cls: Type[Any],
                        values: Iterable[str]) -> None:
    """Replicate nanobind's ``.export_values()`` at Python module scope.

    Usage:
        _my = make_dummy_enum("rocisa.enum.SelectBit", ["SEL_NONE", "DWORD", ...])
        export_enum_values(globals(), _my, ["SEL_NONE", "DWORD", ...])
    """
    for v in values:
        target_namespace[v] = getattr(enum_cls, v)  # type: ignore[index]
