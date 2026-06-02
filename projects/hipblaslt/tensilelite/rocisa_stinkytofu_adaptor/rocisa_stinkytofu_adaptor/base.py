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
    - ``getOutputOptions`` / ``setOutputOptions`` / ``outputNoComment``
      — module-level accessors for the **process-wide** OutputOptions
      instance that holds the codegen comment-suppression flag. This
      module owns the state; ``rocIsa.getInstance().setOutputOptions(...)``
      in ``__init__.py`` is a thin forwarding shim that exists purely
      for rocisa API surface compatibility (Tensile / KernelWriter
      callers).

Not yet done (dummy):
    - ``IsaVersion`` — used as a marker only today.
    - ``Item``, ``DummyItem`` — IR-tree base nodes; arrive with the
      Module / Item phase.

Design note — singleton decomposition:
    Earlier iterations housed ``OutputOptions`` on the ``rocIsa``
    god-singleton in ``__init__.py``; ``base.outputNoComment()`` then
    reached back up through ``rocIsa.getInstance().getOutputOptions()``
    to read its own flag. That inverted the dependency direction
    (base depended on package facade) and forced a lazy-import +
    try/except dance to dodge the import cycle.
    Now the state lives here, in the module that actually owns the
    ``OutputOptions`` class definition, and ``rocIsa`` is downgraded
    to a forwarding shell. ``KernelInfo`` / ``IsaInfo`` / ``vgprIdx``
    are slated for the same treatment in follow-up commits — they
    currently still live on the ``rocIsa`` class.
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


# ---------------------------------------------------------------------------
# Process-wide OutputOptions state.
# ---------------------------------------------------------------------------
#
# ``base`` owns the live ``OutputOptions`` instance for the whole process.
# Everything else -- ``code.TextBlock.toString``, ``instruction._fmt_str``,
# the ``rocIsa.getInstance().setOutputOptions(...)`` API exposed for Tensile
# compatibility -- routes through the three accessors below.
#
# Why a module-level global instead of a class attribute on ``rocIsa``:
#   * Dependency direction stays one-way (``__init__`` depends on ``base``,
#     not vice versa). The previous reverse-import dance is gone.
#   * Pythonic process-singleton idiom: module globals ARE the per-process
#     singleton in Python, no Singleton class needed.
#   * Forwarding shim in ``rocIsa.getOutputOptions/setOutputOptions`` keeps
#     the rocisa API contract intact for KernelWriter callers; we lose
#     nothing externally, gain a clean internal design.
#
# Thread / process semantics:
#   * Process-wide: a single instance per Python interpreter. Workers
#     spawned by ``ParallelMap2`` get a fresh module-level default and
#     pick up Tensile's flag via ``rocIsa.getInstance().setOutputOptions(
#     pickled_opts)`` in the worker init path.
#   * NOT thread-local. ``OutputOptions`` is set once before codegen
#     starts and never mutated mid-flight; if that ever changes,
#     switch the global below to a ``threading.local()`` -- the three
#     accessors are the only places that need updating.
_current_output_options: "OutputOptions" = OutputOptions()


def getOutputOptions() -> OutputOptions:
    """Return the process-wide ``OutputOptions`` instance.

    The returned object IS the source of truth: mutating
    ``getOutputOptions().outputNoComment = True`` immediately affects
    every subsequent ``outputNoComment()`` call and every TextBlock
    rendered after this point. That mirrors rocisa C++ behaviour where
    ``rocIsa::getOutputOptions()`` returns a mutable reference.
    """
    return _current_output_options


def setOutputOptions(options: OutputOptions) -> None:
    """Replace the process-wide ``OutputOptions`` instance.

    Used by ``rocIsa.getInstance().setOutputOptions(...)`` -- which is
    what Tensile / ParallelMap2 workers call to ship a pickled
    ``OutputOptions`` from the parent into the worker process.
    """
    global _current_output_options
    _current_output_options = options


def outputNoComment() -> bool:
    """Return whether the codegen should suppress all comments / TextBlocks.

    Used by ``code.TextBlock.toString`` (gates the entire text payload,
    mirroring rocisa ``code.hpp:154-159``) and ``instruction._fmt_str``
    (drops the per-instruction ``// comment`` tail). One-line direct
    read of the module-level singleton: no lazy import, no try/except,
    no defensive ``getattr`` -- the state object is guaranteed to exist
    because we eagerly construct it at module load.
    """
    return _current_output_options.outputNoComment


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
