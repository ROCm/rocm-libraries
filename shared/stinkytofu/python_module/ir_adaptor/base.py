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
KernelInfo = make_dummy_class(f"{_P}.KernelInfo")
OutputOptions = make_dummy_class(f"{_P}.OutputOptions")
Item = make_dummy_class(f"{_P}.Item")
DummyItem = make_dummy_class(f"{_P}.DummyItem")
