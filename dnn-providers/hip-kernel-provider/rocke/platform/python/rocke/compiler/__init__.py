# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compiler evidence and normalized policy for rocKE kernel authoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evidence import ProbeError, load_or_probe, query_compiler_policy
    from .policy import CompilerPolicy, resolve_policy

__all__ = [
    "CompilerPolicy",
    "ProbeError",
    "load_or_probe",
    "query_compiler_policy",
    "resolve_policy",
]


def __getattr__(name: str) -> object:
    """Load the authoring API without preloading the disposable worker."""
    if name in {"CompilerPolicy", "resolve_policy"}:
        from . import policy

        return getattr(policy, name)
    if name in {"ProbeError", "load_or_probe", "query_compiler_policy"}:
        from . import evidence

        return getattr(evidence, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
