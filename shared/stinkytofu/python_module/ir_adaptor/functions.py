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
"""Shim for ``rocisa.functions``.

What this file is:
    Mirrors ``rocisa/rocisa/src/functions/`` — composite KernelWriter
    helpers that emit IR sequences (vector divide / multiply, branch
    helpers, magic-number division, argument loader, DS init).

What it does (real):
    - None.

Not yet done (dummy):
    - ``ArgumentLoader``
    - Branch helpers: ``BranchIfZero``, ``BranchIfNotZero``
    - Cast helper: ``VSaturateCastInt``
    - Math helpers: vector / scalar divide-and-remainder /
      ceil-divide / multiply / multiply-add / Bpe family,
      ``sMagicDiv`` / ``sMagicDiv2``
    - ``DSInit``

Note on overloads:
    nanobind exposes overload-resolved dispatch under one Python name,
    so the shim exports one symbol per *name*, not per overload.

logicalIR correspondence:
    None. These reduce to several primitive instructions which DO have
    logicalIR mappings, but the helpers themselves do not.
"""

from __future__ import annotations

from ._dummy import make_dummy_class, make_dummy_func

_P = "rocisa.functions"


ArgumentLoader = make_dummy_class(f"{_P}.ArgumentLoader")

BranchIfZero = make_dummy_func(f"{_P}.BranchIfZero")
BranchIfNotZero = make_dummy_func(f"{_P}.BranchIfNotZero")

VSaturateCastInt = make_dummy_func(f"{_P}.VSaturateCastInt")

vectorStaticDivideAndRemainder = make_dummy_func(f"{_P}.vectorStaticDivideAndRemainder")
vectorStaticDivide = make_dummy_func(f"{_P}.vectorStaticDivide")
vectorUInt32DivideAndRemainder = make_dummy_func(f"{_P}.vectorUInt32DivideAndRemainder")
vectorUInt32CeilDivideAndRemainder = make_dummy_func(f"{_P}.vectorUInt32CeilDivideAndRemainder")
vectorStaticRemainder = make_dummy_func(f"{_P}.vectorStaticRemainder")
scalarStaticDivideAndRemainder = make_dummy_func(f"{_P}.scalarStaticDivideAndRemainder")
scalarStaticCeilDivide = make_dummy_func(f"{_P}.scalarStaticCeilDivide")
scalarStaticRemainder = make_dummy_func(f"{_P}.scalarStaticRemainder")
scalarUInt24DivideAndRemainder = make_dummy_func(f"{_P}.scalarUInt24DivideAndRemainder")
scalarUInt32DivideAndRemainder = make_dummy_func(f"{_P}.scalarUInt32DivideAndRemainder")
sMagicDiv = make_dummy_func(f"{_P}.sMagicDiv")
sMagicDiv2 = make_dummy_func(f"{_P}.sMagicDiv2")
vectorStaticMultiply = make_dummy_func(f"{_P}.vectorStaticMultiply")
vectorStaticMultiplyAdd = make_dummy_func(f"{_P}.vectorStaticMultiplyAdd")
scalarStaticMultiply64 = make_dummy_func(f"{_P}.scalarStaticMultiply64")
vectorAddMultiplyBpe = make_dummy_func(f"{_P}.vectorAddMultiplyBpe")
vectorMultiplyBpe = make_dummy_func(f"{_P}.vectorMultiplyBpe")
vectorMultiply64Bpe = make_dummy_func(f"{_P}.vectorMultiply64Bpe")
scalarMultiplyBpe = make_dummy_func(f"{_P}.scalarMultiplyBpe")
scalarMultiply64Bpe = make_dummy_func(f"{_P}.scalarMultiply64Bpe")

DSInit = make_dummy_func(f"{_P}.DSInit")
