# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.base``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/base.cpp``
(the ``m.def_submodule("base", ...)`` block). These classes describe the
IR tree's root types (``Item``, ``KernelInfo`` etc.); they are state-holding
objects not individual opcodes, so they have no logicalIR counterpart and
remain pure dummies per the agreed dummy-stage plan.
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
