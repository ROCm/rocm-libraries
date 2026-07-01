# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared audit dataclasses for HIP-lowering parity harnesses.

Both the platform HIP-lowering audit (hip_lowering_parity) and the library
attention HIP-lowering audit (builders.common.hip_lowering_attention_parity)
import ``Case`` and ``AuditResult`` from here.  This module is pure-platform:
it imports only from stdlib — never from ``kernels``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    build: Callable[[], object]


@dataclass
class AuditResult:
    name: str
    group: str
    llvm_ok: bool = False
    hip_ok: bool = False
    hip_compile_ok: Optional[bool] = None
    hip_chars: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        if not (self.llvm_ok and self.hip_ok):
            return False
        return self.hip_compile_ok is not False
