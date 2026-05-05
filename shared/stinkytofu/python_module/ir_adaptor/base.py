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
"""Shim for ``rocisa.base``.

What this file is:
    Mirrors ``rocisa/rocisa/src/base.cpp`` — IR-tree root types and
    per-thread kernel state holders.

What it does (real):
    - ``KernelInfo`` — current kernel ISA + wavefrontSize; picklable
      for ``ParallelMap2`` workers.
    - ``OutputOptions`` — output toggles (just ``outputNoComment``
      today); picklable.

Not yet done (dummy):
    - ``IsaVersion`` — used as a marker only today.
    - ``Item``, ``DummyItem`` — IR-tree base nodes; arrive with the
      Module / Item phase.
"""

from __future__ import annotations

from ._dummy import make_dummy_class

_P = "rocisa.base"


IsaVersion = make_dummy_class(f"{_P}.IsaVersion")
Item = make_dummy_class(f"{_P}.Item")
DummyItem = make_dummy_class(f"{_P}.DummyItem")


class OutputOptions:
    """Mirror of ``rocisa::OutputOptions`` (mutable, picklable).

    The C++ struct only carries one bool today. We keep the Python shape
    identical so ``rocIsa.getInstance().getOutputOptions().outputNoComment =
    True`` and the subsequent ``setOutputOptions(opts)`` round-trip across
    multiprocessing pickles unchanged.
    """

    __slots__ = ("outputNoComment",)

    def __init__(self, outputNoComment: bool = False) -> None:
        self.outputNoComment = bool(outputNoComment)

    def __repr__(self) -> str:
        return f"OutputOptions(outputNoComment={self.outputNoComment})"

    # Pickle support — needed because Tensile passes this object across the
    # ParallelMap2 fork/spawn boundary.
    def __getstate__(self) -> tuple:
        return (self.outputNoComment,)

    def __setstate__(self, state: tuple) -> None:
        (self.outputNoComment,) = state


class KernelInfo:
    """Mirror of ``rocisa::KernelInfo`` (per-thread current kernel state).

    Only the attributes Tensile actually reads back are typed:
    ``isa`` (a 3-tuple) and ``wavefrontSize``.
    """

    __slots__ = ("isa", "wavefrontSize")

    def __init__(self, isa=None, wavefrontSize: int = 0) -> None:
        self.isa = isa
        self.wavefrontSize = int(wavefrontSize)

    def __repr__(self) -> str:
        return f"KernelInfo(isa={self.isa}, wavefrontSize={self.wavefrontSize})"

    def __getstate__(self) -> tuple:
        return (self.isa, self.wavefrontSize)

    def __setstate__(self, state: tuple) -> None:
        self.isa, self.wavefrontSize = state
