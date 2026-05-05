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
    - None.

Not yet done (dummy):
    - Container nodes: ``Module``, ``KernelBody``, ``Label``,
      ``TextBlock``, ``Macro``, ``StructuredModule``,
      ``ValueIf`` / ``ValueElseIf`` / ``ValueEndif``, ``ValueSet``,
      ``RegSet``, ``BitfieldUnion``, ``SignatureCodeMeta``,
      ``SignatureBase``.
    - ``SrdUpperValue`` (function).

logicalIR correspondence:
    ``StinkyAsmModule`` is the closest analogue at the *module* level
    (different API: ``getName / emitAssembly / runOptimizationPipeline``).
    Sub-nodes have no 1:1 counterpart.
"""

from __future__ import annotations

from ._dummy import make_dummy_class, make_dummy_func

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

SrdUpperValue = make_dummy_func(f"{_P}.SrdUpperValue")
