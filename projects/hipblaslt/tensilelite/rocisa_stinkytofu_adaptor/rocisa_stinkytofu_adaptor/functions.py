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
    - ``ArgumentLoader`` — kernel-argument offset accumulator.
      ``getOffset`` / ``setOffset`` / ``resetOffset`` are real;
      ``loadKernArg`` / ``loadAllKernArg`` advance ``kernArgOffset``
      byte-for-byte the same way ``functions/argument.cpp`` does, but
      return ``None`` instead of an emitted ``SLoadB*`` ``Item`` —
      real emission is unblocked when
      ``rocisa_stinkytofu_adaptor.instruction`` graduates from dummies.

Not yet done (dummy):
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

from ._dummy import make_dummy_func

_P = "rocisa.functions"


class ArgumentLoader:
    """Workaround port of ``rocisa::ArgumentLoader``.

    Mirrors the offset-bookkeeping half of ``rocisa::ArgumentLoader``
    (see ``rocisa/include/functions/argument.hpp``). Tensile reads
    ``getOffset()`` directly to compute SGPR-offset immediates (e.g.
    ``KernelWriterAssembly.py:2351,2452,2454``); a stale value here
    would silently produce wrong ``s_load_b*`` immediates.

    Instruction-emission half is stubbed: ``loadKernArg`` and
    ``loadAllKernArg`` advance ``kernArgOffset`` exactly like the C++
    but return ``None`` instead of a real ``SLoadB{32,64,128,256,512}``
    ``Item`` / ``Module``. Tensile feeds the return through
    ``module.add(...)`` which dummy-swallows ``None`` for now; emission
    becomes real once ``rocisa_stinkytofu_adaptor.instruction`` is real.

    Keep parity with ``functions/argument.hpp`` if you ever touch this.
    """

    __slots__ = ("_kernArgOffset",)

    def __init__(self) -> None:
        self._kernArgOffset: int = 0

    def resetOffset(self) -> None:
        self._kernArgOffset = 0

    def setOffset(self, offset: int) -> None:
        self._kernArgOffset = int(offset)

    def getOffset(self) -> int:
        return self._kernArgOffset

    def loadKernArg(self, dst, srcAddr, sgprOffset=None, dword: int = 1,
                    writeSgpr: bool = True):
        """Advance ``kernArgOffset`` and return ``None`` (stub emission).

        Mirrors ``functions/argument.hpp:60-121``. The C++ does
        ``kernArgOffset += sgprOffset ? 0 : dword * 4`` *outside* the
        ``if(writeSgpr)`` branch, so ``writeSgpr=False`` still advances
        the offset (used to skip unused parms). Real emission would
        produce an ``SLoadB{32,64,128,256,512}`` (writeSgpr=True) or a
        ``TextBlock("Move offset by N\\n")`` (writeSgpr=False); both
        deferred until ``rocisa_stinkytofu_adaptor.instruction`` is real.
        """
        if sgprOffset is None:
            self._kernArgOffset += int(dword) * 4
        return None

    def loadAllKernArg(self, sgprStartIndex: int, srcAddr, numSgprToLoad: int,
                       numSgprPreload: int = 0):
        """Advance ``kernArgOffset`` and return ``None`` (stub emission).

        Mirrors ``functions/argument.hpp:126-199``. The C++ greedily
        loads in chunks of ``i ∈ {16,8,4,2,1}`` SGPRs, advancing
        ``kernArgOffset`` by ``i * 4`` per chunk plus ``numSgprPreload
        * 4`` up front; the *total* advancement is exactly
        ``numSgprToLoad * 4`` regardless of how the chunks partition.
        We collapse the loop to a single byte-equivalent add since the
        emitted ``Module`` is unused while ``instruction.py`` is dummy.
        """
        self._kernArgOffset += int(numSgprToLoad) * 4
        return None


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
