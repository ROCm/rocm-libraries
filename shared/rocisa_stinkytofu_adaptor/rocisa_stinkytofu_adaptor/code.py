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
"""Shim for ``rocisa.code``.

What this file is:
    Mirrors ``rocisa/rocisa/src/code.cpp`` (``init_code``) — IR-tree
    structural nodes (``Module``, ``KernelBody``, ``Label``, ``RegSet``,
    ``ValueSet``, ...).

What it does (real):
    - ``SrdUpperValue(isa)`` — gfx1250-only wrapper backed by the
      stinkytofu C++ free functions ``getSrdUpperValue125X`` /
      ``getSrdUpperDesc125X`` (declared in
      ``shared/stinkytofu/include/stinkytofu/ir/asm/StinkySignature.hpp``
      next to ``SrdUpperValue125X``, implemented via
      ``SrdUpperValue125X::staticInit()``). Returns a small wrapper
      exposing rocisa's ``.getValue() / .desc() / .toString()`` API.

      This is the first end-to-end "vertical slice" through
      KernelWriter → rocisa_stinkytofu_adaptor → ``_stinkytofu.so``
      (nanobind) → ``libstinkytofu.so`` (C++). Use the same recipe for
      future shim entries that need to delegate to logicalIR.

      Other ISAs are intentionally not supported today — the rocisa →
      stinkytofu adapter is gfx1250-only.

Not yet done (dummy):
    - Container nodes: ``Module``, ``KernelBody``, ``Label``,
      ``TextBlock``, ``Macro``, ``StructuredModule``,
      ``ValueIf`` / ``ValueElseIf`` / ``ValueEndif``, ``ValueSet``,
      ``RegSet``, ``BitfieldUnion``, ``SignatureCodeMeta``,
      ``SignatureBase``.

Future:
    When this shim grows beyond gfx1250, prefer adding sibling free
    functions in ``StinkySignature.hpp`` (``getSrdUpperValue12XX`` …)
    or surface a method on ``SignatureBase`` (already exported) rather
    than re-exporting the polymorphic ``BitfieldUnion`` base across
    DSO boundaries.

logicalIR correspondence:
    ``StinkyAsmModule`` is the closest analogue at the *module* level
    (different API: ``getName / emitAssembly / runOptimizationPipeline``).
    Sub-nodes have no 1:1 counterpart.
"""

from __future__ import annotations

from ._dummy import make_dummy_class

_P = "rocisa.code"


Label = make_dummy_class(f"{_P}.Label")
TextBlock = make_dummy_class(f"{_P}.TextBlock")
Module = make_dummy_class(f"{_P}.Module")
Macro = make_dummy_class(f"{_P}.Macro")
StructuredModule = make_dummy_class(f"{_P}.StructuredModule")
ValueEndif = make_dummy_class(f"{_P}.ValueEndif")
ValueIf = make_dummy_class(f"{_P}.ValueIf")
ValueElseIf = make_dummy_class(f"{_P}.ValueElseIf")
ValueSet = make_dummy_class(f"{_P}.ValueSet")
RegSet = make_dummy_class(f"{_P}.RegSet")
BitfieldUnion = make_dummy_class(f"{_P}.BitfieldUnion")
SignatureCodeMeta = make_dummy_class(f"{_P}.SignatureCodeMeta")
SignatureBase = make_dummy_class(f"{_P}.SignatureBase")
KernelBody = make_dummy_class(f"{_P}.KernelBody")


# ---------------------------------------------------------------------------
# logicalIR-backed: gfx1250 SRD upper accessor (the first end-to-end slice).
#
# Soft-import so this package itself stays importable even when
# ``_stinkytofu.so`` hasn't been built yet (the rocisa dispatcher silently
# falls back to native bindings on any adapter import failure; we don't
# want that fallback triggered just because logicalIR is missing).
# ``SrdUpperValue`` re-raises a clear actionable error on first use.
# ---------------------------------------------------------------------------
try:
    from stinkytofu import (  # type: ignore[import-not-found]
        getSrdUpperValue125X as _get_srd_upper_value_125x,
        getSrdUpperDesc125X as _get_srd_upper_desc_125x,
    )

    _STINKYTOFU_IMPORT_ERR: "ImportError | None" = None
except ImportError as _e:  # pragma: no cover - exercised only without a build
    _get_srd_upper_value_125x = None
    _get_srd_upper_desc_125x = None
    _STINKYTOFU_IMPORT_ERR = _e


class _Gfx1250SrdUpper:
    """rocisa-shaped wrapper around the two gfx1250 free functions.

    Tensile only reads ``.getValue() / .desc() / .toString()`` off
    ``SrdUpperValue(IsaVersion)`` (see ``KernelWriterAssembly.py:1497``);
    we expose exactly that surface and forward to logicalIR. Keeping
    the wrapper on the Python side lets logicalIR's public C++ ABI
    stay primitive-typed (no ``BitfieldUnion`` base crossing the DSO).
    """

    __slots__ = ()

    def getValue(self) -> int:  # noqa: N802 (matches rocisa public API)
        return int(_get_srd_upper_value_125x())

    def desc(self) -> str:
        return _get_srd_upper_desc_125x()

    def toString(self) -> str:  # noqa: N802 (matches rocisa public API)
        return f"0x{self.getValue():x}"


def SrdUpperValue(isa):  # noqa: N802 (matches rocisa public API)
    """Wrapper matching ``rocisa::SrdUpperValue(IsaVersion)`` for gfx1250.

    Accepts either a 3-tuple/list (``kernel["ISA"]``-style) or a struct
    with ``.major / .minor / .stepping`` (rocisa's ``IsaVersion``).
    Only ``(12, 5, *)`` is supported today; other ISAs raise
    ``NotImplementedError`` deliberately — extending coverage should add
    sibling free functions in ``StinkySignature.hpp`` rather than
    re-exporting the polymorphic ``BitfieldUnion`` base.
    """
    if _get_srd_upper_value_125x is None:
        raise ImportError(
            "rocisa_stinkytofu_adaptor.code.SrdUpperValue requires the "
            "stinkytofu Python binding (_stinkytofu.so). Build it via:\n"
            "  cmake --build <build_dir> --target stinkytofu_python\n"
            "and ensure <build_dir>/lib is on PYTHONPATH.\n"
            f"  Underlying error: {_STINKYTOFU_IMPORT_ERR}"
        )
    if hasattr(isa, "major"):
        major, minor = int(isa.major), int(isa.minor)
    else:
        major, minor = int(isa[0]), int(isa[1])
    if (major, minor) != (12, 5):
        raise NotImplementedError(
            f"rocisa_stinkytofu_adaptor.code.SrdUpperValue is gfx1250-only; "
            f"got ISA major={major}, minor={minor}. Extend coverage by "
            f"adding sibling free functions in "
            f"shared/stinkytofu/include/stinkytofu/ir/asm/StinkySignature.hpp."
        )
    return _Gfx1250SrdUpper()
