################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Factory helpers for building rocisa-shaped dummies in the adaptor.

What this file is:
    Factories used by every ``rocisa_stinkytofu_adaptor.*`` shim to
    produce class / function / enum stand-ins matching the nanobind
    binding surface.

What it does (real):
    - ``make_dummy_class`` — class shim; metaclass handles class-level
      attribute access (e.g. ``nb::def_static`` methods).
    - ``make_dummy_func`` — function shim; prints when called.
    - ``make_dummy_enum`` — real ``IntEnum`` matching nanobind's
      ``nb::enum_`` + ``export_values()`` semantics; member ``.value`` /
      ``.name`` and ``cls(0)`` round-trip.
    - ``export_enum_values`` — replicates ``nb::export_values()`` at
      module scope.

Not yet done:
    - None. Every helper here is real.
"""

from __future__ import annotations

import enum as _stdenum
from typing import Any, Iterable, Mapping, Type


def make_dummy_class(full_name: str, *, base: Type[Any] = object) -> Type[Any]:
    """Create a dummy class whose ``__init__`` prints ``full_name``.

    Calls such as ``BufferLoadB128(dst=...)`` will print
    ``rocisa.instruction.BufferLoadB128`` and return a dummy instance.
    Arbitrary attribute access on the instance returns another dummy
    callable so that chained method calls (e.g. ``.add(...)``) never
    raise AttributeError during the structural-only phase.

    ``base`` lets the dummy participate in the real class hierarchy
    (rocisa C++ has every code-composition node inheriting from
    ``Item``). When set, ``isinstance(dummy_instance, base)`` returns
    ``True`` -- so
    e.g. ``Module.findIndexByType(Item)`` would also match dummy
    Label / ValueSet / Macro nodes that are still pending real
    implementations. The dummy ``__init__`` calls ``super().__init__()``
    with no args, so ``base`` must have an ``__init__`` whose required
    parameters are all defaultable; ``Item`` qualifies because
    ``Item.__init__(self, name="")`` defaults name.

    Methods defined on ``base`` (``toString`` / ``prettyPrint`` /
    ``countType`` / ``countExactType`` / capability proxies) take
    precedence over the dummy ``__getattr__`` no-op, so a dummy Label
    will inherit the real Item behaviour for those methods rather
    than silently returning ``None``.
    """

    short = full_name.rsplit(".", 1)[-1]

    # The custom metaclass must be a subclass of ``base``'s metaclass so
    # Python's "child metaclass derives from all parent metaclasses"
    # rule is satisfied. ``type(base)`` covers both ``type`` (the
    # default for regular classes like ``Item``) and any custom
    # metaclass downstream might introduce.
    _BaseMeta = type(base)

    class _DummyMeta(_BaseMeta):
        def __getattr__(cls, name: str) -> Any:
            def _classlevel_noop(*args: Any, **kwargs: Any) -> None:
                return 1
            return _classlevel_noop

    class _DummyInstance(base, metaclass=_DummyMeta):  # type: ignore[misc]
        __slots__ = ("_full_name",)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Initialise the ``base`` portion (e.g. Item.name / parent)
            # so downstream code that touches inherited slots through
            # ``base``'s methods (Item.prettyPrint reads self.name)
            # doesn't AttributeError on uninitialised slots.
            super().__init__()
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
