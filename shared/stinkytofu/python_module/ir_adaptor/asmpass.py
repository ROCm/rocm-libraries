# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shim for ``rocisa.asmpass``.

Source of truth: ``projects/hipblaslt/tensilelite/rocisa/rocisa/src/pass/pass.cpp``
(``init_pass``). The corresponding logicalIR optimization pipeline is
exposed through ``StinkyAsmModule::runOptimizationPipeline`` (see
``python_module/src/python_bindings.cpp``) but that is an entire-module
operation, not a named pass list like ``rocIsaPass``. No opcode-level
match; dummies only.
"""

from __future__ import annotations

from ._dummy import make_dummy_class, make_dummy_func

_P = "rocisa.asmpass"


getActFuncModuleName = make_dummy_func(f"{_P}.getActFuncModuleName")
getActFuncBranchModuleName = make_dummy_func(f"{_P}.getActFuncBranchModuleName")
rocIsaPass = make_dummy_func(f"{_P}.rocIsaPass")
rocIsaPassOption = make_dummy_class(f"{_P}.rocIsaPassOption")
rocIsaPassResult = make_dummy_class(f"{_P}.rocIsaPassResult")
