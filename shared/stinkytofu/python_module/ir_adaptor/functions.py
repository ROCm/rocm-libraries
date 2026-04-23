# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.functions``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/functions/``
- ``argument.cpp``   : ``ArgumentLoader``
- ``f_branch.cpp``   : ``BranchIfZero``, ``BranchIfNotZero``
- ``f_cast.cpp``     : ``VSaturateCastInt``
- ``f_math.cpp``     : static-divide / remainder / multiply helpers + ``sMagicDiv``
- ``functions.cpp``  : ``DSInit``

logicalIR correspondence: none - these are KernelWriter-level helper
function builders (they emit composite IR sequences), not individual
opcodes. They are always composites that eventually reduce to several
primitive instructions which DO have logicalIR mappings, but the helpers
themselves have no opcode-level counterpart. Dummies only.

Note on overloads: ``m.def("vgpr", ...)`` called multiple times in
nanobind produces a single Python name bound to overload-resolved
dispatch. From the Python caller's point of view there is only one
``vectorStaticDivide``, ``scalarStaticDivideAndRemainder``, ``sMagicDiv``
etc. - we therefore expose one dummy per *name*, not per overload.
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
