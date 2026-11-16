# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""pymosaic — host-side, JIT-backed mosaic layouts for Python.

A standalone, pip-installable slice of `mosaic` that exposes the
runtime-value / compile-time-structure ``Layout`` class. Each distinct
layout *structure* (nesting profile) is compiled once into a small C++
extension via ninja + a C++20 compiler; the shape/stride *values* are
supplied at call time, so one compiled module serves every layout that
shares a structure.

Quick start::

    from pymosaic import Layout

    L = Layout(shape=(4, 8))      # column-major by default (CuTe canonical)
    L[2, 3]                       # 2*1 + 3*4 = 14
    list(L)[:4]                   # [0, 1, 2, 3]  (flat iter is column-major)

    H = Layout.from_modes(
        Layout.row_major(2, 2),
        Layout.column_major(3, 3),
    )

Requirements at runtime: a C++20 compiler (``g++`` by default, override
with ``$CXX``), plus the ``ninja`` and ``nanobind`` Python packages
(installed automatically as dependencies).

Cache and build knobs are exposed under :mod:`pymosaic.jit`.
"""

from __future__ import annotations

from . import jit
from .layout import (
    Layout,
    append,
    blocked_product,
    coalesce,
    column_major,
    complement,
    composition,
    cosize,
    crd2idx,
    depth,
    dice,
    filter,
    flat_divide,
    flat_product,
    flatten,
    get,
    group,
    idx2crd,
    logical_divide,
    logical_product,
    prepend,
    rank,
    right_inverse,
    row_major,
    select,
    shape,
    shape_at,
    size,
    size_at,
    stride,
    stride_at,
    take,
    tiled_divide,
    tiled_product,
    zipped_divide,
    zipped_product,
)
from .pretty import pretty, pretty_print

__all__ = [
    "Layout",
    "row_major",
    "column_major",
    "idx2crd",
    "crd2idx",
    "pretty",
    "pretty_print",
    "jit",
    # structural accessors (CuTe-style free functions)
    "shape",
    "stride",
    "rank",
    "depth",
    "size",
    "cosize",
    "shape_at",
    "stride_at",
    "size_at",
    "get",
    "flatten",
    "take",
    "select",
    "group",
    "append",
    "prepend",
    # algebra (JIT-backed)
    "coalesce",
    # algebra (pure-Python)
    "blocked_product",
    "complement",
    "right_inverse",
    "filter",
    "composition",
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "flat_divide",
    "logical_product",
    "zipped_product",
    "tiled_product",
    "flat_product",
    # slicing
    "dice",
]

__version__ = "0.1.0"
