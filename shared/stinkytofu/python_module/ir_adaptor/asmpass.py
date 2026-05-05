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
"""Shim for ``rocisa.asmpass``.

What this file is:
    Mirrors ``rocisa/rocisa/src/pass/pass.cpp`` (``init_pass``) — the
    rocIsa optimisation pass list and per-pass options.

What it does (real):
    - None.

Not yet done (dummy):
    - ``getActFuncModuleName``, ``getActFuncBranchModuleName``
    - ``rocIsaPass``, ``rocIsaPassOption``, ``rocIsaPassResult``

logicalIR correspondence:
    Optimisation lives in ``StinkyAsmModule::runOptimizationPipeline`` —
    a single module-level call, not a named pass list.
"""

from __future__ import annotations

from ._dummy import make_dummy_class, make_dummy_func

_P = "rocisa.asmpass"


getActFuncModuleName = make_dummy_func(f"{_P}.getActFuncModuleName")
getActFuncBranchModuleName = make_dummy_func(f"{_P}.getActFuncBranchModuleName")
rocIsaPass = make_dummy_func(f"{_P}.rocIsaPass")
rocIsaPassOption = make_dummy_class(f"{_P}.rocIsaPassOption")
rocIsaPassResult = make_dummy_class(f"{_P}.rocIsaPassResult")
