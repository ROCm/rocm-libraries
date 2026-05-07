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
    - ``SrdUpperValue(isa)`` — workaround port of
      ``rocisa::SrdUpperValue``. Returns a stub matching the
      ``BitfieldUnion`` interface (``.desc()`` / ``.getValue()``); for
      gfx1250 the SRD upper 32 bits are zero by ``staticInit`` (see
      ``rocisa/include/code.hpp:1174-1260``), so the stub returns 0
      byte-for-byte. Other ISAs fall through to the same stub today.
      TODO: dispatch on ``IsaVersion`` when extending beyond gfx1250.

Not yet done (dummy):
    - Container nodes: ``Module``, ``KernelBody``, ``Label``,
      ``TextBlock``, ``Macro``, ``StructuredModule``,
      ``ValueIf`` / ``ValueElseIf`` / ``ValueEndif``, ``ValueSet``,
      ``RegSet``, ``BitfieldUnion``, ``SignatureCodeMeta``,
      ``SignatureBase``.

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

class _Gfx1250SrdUpperStub:
    """Workaround port of ``rocisa::SrdUpperValue125X::staticInit()`` for
    gfx1250.

    The gfx1250 SRD upper 32 bits are zero (see
    ``rocisa/include/code.hpp:1174-1260``); the bitfield layout
    (``num_records_upper`` / ``stride`` / ``oob_select`` / ...) is
    gfx125X-specific and unrelated to gfx12XX. The stub returns 0 to
    match the C++ default and reproduces the C++ ``desc()`` text format
    so KernelWriter's ``addComment2`` prints byte-for-byte equivalent
    output.

    TODO: replace with a real ``BitfieldUnion`` family when extending
    beyond gfx1250 — at that point ``SrdUpperValue`` should dispatch on
    ``IsaVersion`` (major / minor) like ``rocisa::SrdUpperValue`` does
    in ``rocisa/src/code.cpp:56``.
    """

    def getValue(self) -> int:
        return 0

    def desc(self) -> str:
        return (
            "hex: 0\n"
            "num_records_upper (6b): 0\n"
            "reserved (6b): 0\n"
            "stride (14b): 0\n"
            "stride_scale (2b): 0\n"
            "swizzle_enable (1b): 0\n"
            "oob_select (1b): 0\n"
            "type (2b): 0"
        )


def SrdUpperValue(isa):  # noqa: N802 (matches rocisa public API)
    """Factory matching ``rocisa::SrdUpperValue(IsaVersion)``.

    Today every ISA gets the gfx1250 stub. Once we extend beyond
    gfx1250, branch on ``isa.major / isa.minor`` here just like
    ``rocisa/src/code.cpp:56-77``.
    """
    return _Gfx1250SrdUpperStub()
