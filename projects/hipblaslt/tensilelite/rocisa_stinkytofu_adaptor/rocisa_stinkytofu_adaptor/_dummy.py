# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Factories for building rocisa-shaped class/function/enum stand-ins.

All helpers here are real implementations (IntEnum, metaclass shims).
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
    member is its 0-based index in ``values`` — this matches enums whose C++
    underlying values are sequential from zero (``RegisterType``,
    ``DataTypeEnum``, ``CacheScope``, ...).

    Note (was originally a "structural-only" dummy):
        Tensile's import-time machinery in ``Tensile/Common/DataType.py``
        reads ``e['enum'].value`` and ``e['enum'].name`` while building the
        ``DataType`` lookup table, so a bare ``int`` placeholder is not
        enough to pass ``import Tensile``. ``IntEnum`` gives us both the
        attribute surface and the raw-int behaviour the rest of the code
        treats it as.

    For enums with explicit non-sequential C++ values (``InstType``,
    ``TemporalHint``, ...), use ``make_bound_enum`` instead.
    """

    short = full_name.rsplit(".", 1)[-1]
    values = list(values)
    module = full_name.rsplit(".", 1)[0]

    cls = _stdenum.IntEnum(short, [(v, i) for i, v in enumerate(values)])
    cls.__module__ = module
    cls.__qualname__ = short
    return cls


def make_bound_enum(
    full_name: str,
    members: Iterable[tuple[str, int]],
) -> Type[Any]:
    """Create an ``IntEnum`` with explicit per-member C++ integral values.

    Mirrors ``nb::enum_<T>::value("NAME", T::NAME)`` bindings in
    ``rocisa::enum.cpp`` where the exposed ``.value`` is the C++ enumerator,
    not a 0-based Python index. Supports alias members that share a value
    (e.g. ``TH_WB`` and ``TH_LU`` both equal ``3``).
    """

    short = full_name.rsplit(".", 1)[-1]
    module = full_name.rsplit(".", 1)[0]
    member_list = list(members)

    cls = _stdenum.IntEnum(short, member_list)
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
