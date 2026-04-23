# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.container``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/container.cpp``
(``init_containers``). Contains modifier descriptors (DS/FLAT/SMEM/...) and
register wrappers (``RegisterContainer``, ``VCC``, ``EXEC``, ...).

logicalIR correspondence: partial. Logical-side has ``StinkyRegister`` +
``RegType`` (python_bindings.cpp) which subsumes what ``RegisterContainer``
carries at the instance level, but the modifier classes (DSModifiers etc.)
are asm-level concerns; logicalIR encodes those as intrinsic attributes on
each operation rather than as stand-alone container objects. Pure dummies
this pass.
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
VCC = make_dummy_class(f"{_P}.VCC")
HWRegContainer = make_dummy_class(f"{_P}.HWRegContainer")
RegName = make_dummy_class(f"{_P}.RegName")
RegisterContainer = make_dummy_class(f"{_P}.RegisterContainer")
HolderContainer = make_dummy_class(f"{_P}.HolderContainer")
MemTokenData = make_dummy_class(f"{_P}.MemTokenData")
ContinuousRegister = make_dummy_class(f"{_P}.ContinuousRegister")
