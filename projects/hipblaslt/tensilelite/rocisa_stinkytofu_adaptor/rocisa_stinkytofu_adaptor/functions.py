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
    - ``ArgumentLoader`` — kernel-argument offset accumulator and
      ``SLoadB*`` emitter. ``loadKernArg`` / ``loadAllKernArg`` emit
      real ``SLoadB{32,64,128,256,512}`` instructions matching the C++
      logic in ``functions/argument.hpp``.

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

from typing import Any

from ._dummy import make_dummy_func
from .code import Module, TextBlock
from .container import sgpr
from .instruction import SLoadB32, SLoadB64, SLoadB128, SLoadB256, SLoadB512

_P = "rocisa.functions"

_SLOAD_BY_BITS = {
    32: SLoadB32,
    64: SLoadB64,
    128: SLoadB128,
    256: SLoadB256,
    512: SLoadB512,
}


class ArgumentLoader:
    """Port of ``rocisa::ArgumentLoader`` (``functions/argument.hpp``).

    Manages kernel-argument offset bookkeeping and emits ``SLoadB*``
    instructions matching the C++ logic exactly.
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
                    writeSgpr: bool = True) -> Any:
        """Emit an ``SLoadB*`` and advance ``kernArgOffset``.

        Mirrors ``functions/argument.hpp:60-121``.
        """
        size = int(dword) * 4

        if writeSgpr:
            if sgprOffset is not None:
                comment = (sgprOffset.toString()
                           if hasattr(sgprOffset, "toString")
                           else str(sgprOffset))
            else:
                comment = str(self._kernArgOffset)

            dst_sgpr = sgpr(dst, dword)
            src_sgpr = sgpr(srcAddr, 2)
            offset = sgprOffset if sgprOffset is not None else self._kernArgOffset

            bits = int(dword) * 32
            cls = _SLOAD_BY_BITS.get(bits)
            if cls is None:
                raise ValueError(f"Invalid dword size {dword}")
            item = cls(dst=dst_sgpr, base=src_sgpr, soffset=offset, comment=comment)
        else:
            item = TextBlock("Move offset by " + str(size) + "\n")

        if sgprOffset is None:
            self._kernArgOffset += size
        return item

    def loadAllKernArg(self, sgprStartIndex: int, srcAddr, numSgprToLoad: int,
                       numSgprPreload: int = 0) -> Module:
        """Emit a ``Module`` of greedy-packed ``SLoadB*`` instructions.

        Mirrors ``functions/argument.hpp:126-199``.
        """
        module = Module("LoadAllKernArg")
        actual_load = int(numSgprToLoad) - int(numSgprPreload)
        sgpr_idx = int(sgprStartIndex) + int(numSgprPreload)
        self._kernArgOffset += int(numSgprPreload) * 4

        while actual_load > 0:
            i = 16
            while i >= 1:
                is_aligned = False
                if i >= 4 and sgpr_idx % 4 == 0:
                    is_aligned = True
                elif i == 2 and sgpr_idx % 2 == 0:
                    is_aligned = True
                elif i == 1:
                    is_aligned = True

                if is_aligned and actual_load >= i:
                    actual_load -= i
                    dst_sgpr = sgpr(sgpr_idx, i)
                    src_sgpr = sgpr(srcAddr, 2)
                    comment = str(self._kernArgOffset)

                    bits = i * 32
                    cls = _SLOAD_BY_BITS.get(bits)
                    if cls is None:
                        raise ValueError(f"Invalid SGPR size {i}")
                    module.add(cls(dst=dst_sgpr, base=src_sgpr,
                                   soffset=self._kernArgOffset, comment=comment))
                    sgpr_idx += i
                    self._kernArgOffset += i * 4
                    break
                i //= 2

        return module


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
