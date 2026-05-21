# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Provider-local Python package for the CK DSL hipDNN provider.

This package is the thin Python compile-service layer that the C++
provider invokes via the embedded CPython interpreter. The C++ side
walks the hipDNN graph and packs a typed payload; the Python side
hands that payload to ``ck_dsl`` and returns HSACO bytes + a launch
ABI dict.

For M1 step I-3 the package only exposes :func:`noop_smoke`, used to
prove the cross-package import path. The real ``compile()`` entry
point lands in step I-7.
"""

from __future__ import annotations

from .compile_service import noop_smoke

__version__ = "0.1.0"

__all__ = ["noop_smoke", "__version__"]
