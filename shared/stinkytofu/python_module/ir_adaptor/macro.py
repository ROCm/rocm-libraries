# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.macro``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/macro.cpp``.
These are composite helper builders (magic-number division macros, PRNG
boilerplate); no opcode-level logicalIR counterpart. Dummies only.
"""

from __future__ import annotations

from ._dummy import make_dummy_func

_P = "rocisa.macro"


MacroVMagicDiv = make_dummy_func(f"{_P}.MacroVMagicDiv")
PseudoRandomGenerator = make_dummy_func(f"{_P}.PseudoRandomGenerator")
VMagicDiv = make_dummy_func(f"{_P}.VMagicDiv")
PseudoRandomGeneratorModule = make_dummy_func(f"{_P}.PseudoRandomGeneratorModule")
