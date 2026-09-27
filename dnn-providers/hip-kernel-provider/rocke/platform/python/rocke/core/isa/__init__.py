# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ISA backends — per-gfx LLVM codegen details behind a selection seam.

See :mod:`rocke.core.isa.backend` and
``dsl_docs/architecture/multi_arch_data_layout.md``.
"""

from .backend import (  # noqa: F401
    BACKEND_REGISTRY,
    Gfx9MfmaBackend,
    Gfx11RdnaBackend,
    Gfx12RdnaBackend,
    Gfx950Backend,
    ISABackend,
    backend_for,
)

__all__ = [
    "BACKEND_REGISTRY",
    "Gfx9MfmaBackend",
    "Gfx11RdnaBackend",
    "Gfx12RdnaBackend",
    "Gfx950Backend",
    "ISABackend",
    "backend_for",
]
