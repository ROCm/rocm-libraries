# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Lazy ctypes function binding shared by the HIP / comgr wrappers.

A tiny ergonomics helper: defer ``getattr`` on the underlying ``CDLL``
until the first call so the runtime wrappers (`hip_module`, `comgr`) can
be imported before their shared library is resolved -- which lets rocke
and torch be imported in any order and still bind the same loaded HIP /
comgr runtime instance (see :mod:`rocke.runtime.runtime_coexistence`).
"""

from __future__ import annotations

import ctypes
from typing import Any, Callable, List, Optional


class _LazyFn:
    """Lazy ctypes function wrapper.

    Defers ``getattr`` on the underlying CDLL until the first call so
    that rocke and torch can be imported in any order without ending
    up with two HIP runtimes (see
    ``runtime_coexistence._torch_bundled_lib``). Resolved on
    first use; subsequent calls dispatch directly through the cached
    function pointer.

    ``lib_resolver`` returns the shared ``ctypes.CDLL`` for this lib
    family (HIP runtime / comgr / ...). It is invoked exactly once per
    function on first call.
    """

    __slots__ = ("_name", "_argtypes", "_restype", "_lib_resolver", "_fn")

    def __init__(
        self,
        name: str,
        argtypes: List[Any],
        restype: Any,
        lib_resolver: "Callable[[], ctypes.CDLL]",
    ) -> None:
        self._name = name
        self._argtypes = argtypes
        self._restype = restype
        self._lib_resolver = lib_resolver
        self._fn: Optional[Any] = None

    def _resolve(self) -> Any:
        fn = getattr(self._lib_resolver(), self._name)
        fn.argtypes = self._argtypes
        fn.restype = self._restype
        self._fn = fn
        return fn

    def __call__(self, *args: Any) -> Any:
        fn = self._fn or self._resolve()
        return fn(*args)
