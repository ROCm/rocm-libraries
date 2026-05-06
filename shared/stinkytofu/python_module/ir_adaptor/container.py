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
"""Shim for ``rocisa.container``.

What this file is:
    Mirrors ``rocisa/rocisa/src/container.cpp`` — register references
    (``RegisterContainer``, ``RegName``, ``Holder*``), factory helpers
    (``sgpr`` / ``vgpr`` / ``accvgpr`` / ``mgpr``, ``replaceHolder``),
    asm modifier descriptors (``DS/FLAT/SMEM/...Modifiers``), hardware
    aliases (``VCC``, ``EXEC``, ``HWRegContainer``), and small value
    objects (``ContinuousRegister``, ``MemTokenData``).

What it does (real):
    - None.

Not yet done (dummy):
    - ``RegName``, ``RegisterContainer``                      (Step 2)
    - ``sgpr``, ``vgpr``                                      (Step 3)
    - ``accvgpr``, ``mgpr``                                   (Step 4)
    - ``ContinuousRegister``                                  (Step 5)
    - ``VCC``, ``EXEC``, ``EXECLO``, ``EXECHI``               (Step 6)
    - ``HWRegContainer``                                      (Step 7)
    - ``Holder``, ``HolderContainer``, ``replaceHolder``      (Step 8)
    - ``MemTokenData``                                        (Step 10)
    - Modifier classes (``DS/FLAT/GLOBAL/MUBUF/SMEM/SDWA/DPP/
      VOP3P/True16Modifiers``) — deferred to instruction-emit phase.
    - ``Container`` base class.

logicalIR correspondence:
    ``StinkyRegister`` (``shared/stinkytofu/python_module/src/
    python_bindings.cpp``) covers the instance-level data carried by
    ``RegisterContainer``; modifier classes are encoded as intrinsic
    attributes on each operation, not as stand-alone objects.
"""

from __future__ import annotations

from ._dummy import make_dummy_class, make_dummy_func

_P = "rocisa.container"


replaceHolder = make_dummy_func(f"{_P}.replaceHolder")
vgpr = make_dummy_func(f"{_P}.vgpr")
sgpr = make_dummy_func(f"{_P}.sgpr")
accvgpr = make_dummy_func(f"{_P}.accvgpr")
mgpr = make_dummy_func(f"{_P}.mgpr")

Holder = make_dummy_class(f"{_P}.Holder")
Container = make_dummy_class(f"{_P}.Container")
DSModifiers = make_dummy_class(f"{_P}.DSModifiers")
FLATModifiers = make_dummy_class(f"{_P}.FLATModifiers")
GLOBALModifiers = make_dummy_class(f"{_P}.GLOBALModifiers")
MUBUFModifiers = make_dummy_class(f"{_P}.MUBUFModifiers")
SMEMModifiers = make_dummy_class(f"{_P}.SMEMModifiers")
SDWAModifiers = make_dummy_class(f"{_P}.SDWAModifiers")
DPPModifiers = make_dummy_class(f"{_P}.DPPModifiers")
VOP3PModifiers = make_dummy_class(f"{_P}.VOP3PModifiers")
True16Modifiers = make_dummy_class(f"{_P}.True16Modifiers")
EXEC = make_dummy_class(f"{_P}.EXEC")
EXECLO = make_dummy_class(f"{_P}.EXECLO")
EXECHI = make_dummy_class(f"{_P}.EXECHI")
VCC = make_dummy_class(f"{_P}.VCC")
HWRegContainer = make_dummy_class(f"{_P}.HWRegContainer")
RegName = make_dummy_class(f"{_P}.RegName")
RegisterContainer = make_dummy_class(f"{_P}.RegisterContainer")
HolderContainer = make_dummy_class(f"{_P}.HolderContainer")
MemTokenData = make_dummy_class(f"{_P}.MemTokenData")
ContinuousRegister = make_dummy_class(f"{_P}.ContinuousRegister")
