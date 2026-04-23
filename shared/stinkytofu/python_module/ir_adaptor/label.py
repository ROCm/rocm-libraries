# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.label``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/label.cpp``.
No direct logicalIR counterpart (label management is an asm-level concern);
dummies only.
"""

from __future__ import annotations

from ._dummy import make_dummy_class, make_dummy_func

_P = "rocisa.label"


LabelManager = make_dummy_class(f"{_P}.LabelManager")
magicGenerator = make_dummy_func(f"{_P}.magicGenerator")
