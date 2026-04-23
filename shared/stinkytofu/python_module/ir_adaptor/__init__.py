# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""logicalIR adaptor package (Tensilelite ``rocisa`` shim).

This package mimics the public shape of
``projects/hipblaslt/tensilelite/rocisa/rocisa`` (the nanobind-based C++
bindings) so that legacy Python consumers - primarily
``KernelWriter.py`` / ``KernelWriterAssembly.py`` / ``KernelWriterModules.py``
in ``tensilelite/Tensile/`` - can continue using ``from rocisa import ...``
and ``from rocisa.<submodule> import ...`` unchanged while we are porting
them onto the higher-level logicalIR in ``shared/stinkytofu``.

Activation:
    The rewire happens only when ``ROCISA_BACKEND=logical`` (see
    ``projects/hipblaslt/tensilelite/rocisa/rocisa/__init__.py``).  For
    any other value (or when unset) the original nanobind ``rocisa``
    remains untouched; logicalIR is reserved for ISA gfx1250 right now
    but the env-var keeps the switch explicit and easy to flip by hand.

Phase (current):
    Pure structural dummies. Every class shim prints its fully qualified
    rocisa name when instantiated; every function shim prints when
    called. Real delegation to ``stinkytofu`` is a separate pass.

Top-level ``rocisa`` module surface (reproduced here):
    - Class ``rocIsa``, ``IsaInfo``
    - Functions ``isaToGfx``, ``getGlcBitName``, ``getSlcBitName``
    - Functions ``countType``, ``countInstruction``, ``countGlobalRead``,
      ``countSMemLoad``, ``countLocalRead``, ``countLocalWrite``,
      ``countWeightedLocalRead``, ``countWeightedLocalWrite``,
      ``countDSStoreB128``, ``countDSStoreB192``, ``countDSStoreB256``,
      ``countVMovB32``, ``getMFMAs``, ``findInstCount``
    - Interop helpers ``isSupportedByStinkyTofu``, ``StinkyAsmModule``,
      ``toStinkyTofuModule`` (contributed by ``init_stinkytofu`` in
      ``shared/stinkytofu/src/conversion/rocisa/ToStinkyTofuUtils.cpp``)
    - Submodules: base, enum, container, code, label, instruction,
      functions, asmpass, macro, register
"""

from __future__ import annotations

from ._dummy import make_dummy_class, make_dummy_func

# Make submodules importable as attributes (``rocisa.code`` etc.). The
# rocisa dispatcher in ``tensilelite/rocisa/rocisa/__init__.py`` is what
# ultimately installs them under the ``rocisa.*`` name in ``sys.modules``.
from . import asmpass as asmpass
from . import base as base
from . import code as code
from . import container as container
from . import enum as enum
from . import functions as functions
from . import instruction as instruction
from . import label as label
from . import macro as macro
from . import register as register

_P = "rocisa"


rocIsa = make_dummy_class(f"{_P}.rocIsa")
IsaInfo = make_dummy_class(f"{_P}.IsaInfo")

isaToGfx = make_dummy_func(f"{_P}.isaToGfx")
getGlcBitName = make_dummy_func(f"{_P}.getGlcBitName")
getSlcBitName = make_dummy_func(f"{_P}.getSlcBitName")

countType = make_dummy_func(f"{_P}.countType")
countInstruction = make_dummy_func(f"{_P}.countInstruction")
countGlobalRead = make_dummy_func(f"{_P}.countGlobalRead")
countSMemLoad = make_dummy_func(f"{_P}.countSMemLoad")
countLocalRead = make_dummy_func(f"{_P}.countLocalRead")
countLocalWrite = make_dummy_func(f"{_P}.countLocalWrite")
countWeightedLocalRead = make_dummy_func(f"{_P}.countWeightedLocalRead")
countWeightedLocalWrite = make_dummy_func(f"{_P}.countWeightedLocalWrite")
countDSStoreB128 = make_dummy_func(f"{_P}.countDSStoreB128")
countDSStoreB192 = make_dummy_func(f"{_P}.countDSStoreB192")
countDSStoreB256 = make_dummy_func(f"{_P}.countDSStoreB256")
countVMovB32 = make_dummy_func(f"{_P}.countVMovB32")
getMFMAs = make_dummy_func(f"{_P}.getMFMAs")
findInstCount = make_dummy_func(f"{_P}.findInstCount")

# rocisa <-> stinkytofu interop (see init_stinkytofu in
# shared/stinkytofu/src/conversion/rocisa/ToStinkyTofuUtils.cpp). These
# are real logicalIR bridges; once the dummy phase is over they should
# delegate into the compiled stinkytofu bindings rather than print.
isSupportedByStinkyTofu = make_dummy_func(f"{_P}.isSupportedByStinkyTofu")
StinkyAsmModule = make_dummy_class(f"{_P}.StinkyAsmModule")
toStinkyTofuModule = make_dummy_func(f"{_P}.toStinkyTofuModule")
