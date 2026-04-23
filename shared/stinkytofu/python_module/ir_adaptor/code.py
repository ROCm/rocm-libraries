# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.code``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/code.cpp``
(``init_code``). These are the IR-tree structural nodes (``Module``,
``KernelBody``, ``Label``, ...).

logicalIR correspondence: ``StinkyAsmModule`` (see
``shared/stinkytofu/python_module/src/python_bindings.cpp``) is the
closest analogue at the module level but carries a different API
(``getName / emitAssembly / runOptimizationPipeline``). The sub-nodes
(``TextBlock``, ``ValueIf``, ``ValueSet``, ``RegSet``, ``SignatureBase``,
``KernelBody``) have no 1:1 logicalIR counterpart. Dummies only.
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
