# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SdpaRequest — minimal sketch for the ST2/ST4 SDPA binding phase.

Section 6.1 of the hackathon plan defines the full contract.
ST9 (DV-01) owns the production version with all axes.
"""
from __future__ import annotations

from dataclasses import dataclass

_DTYPE_ALIASES = {"f16": "fp16", "fp16": "fp16", "bf16": "bf16"}


@dataclass(frozen=True)
class SdpaRequest:
    """Request passed to the SDPA dispatch path.

    Fields relevant to U-03 only; ST9 adds strides, diagonal, bias, etc.
    """

    arch: str
    dtype: str  # "fp16" | "bf16"  (or "f16" alias — normalized in __post_init__)
    head_dim: int  # unified across Q, K, V (square only)
    # Extend in ST9 (DV-01) — do not add fields here without ST0 sign-off.

    def __post_init__(self) -> None:
        canonical = _DTYPE_ALIASES.get(self.dtype.lower())
        if canonical is None:
            raise ValueError(
                f"unsupported dtype {self.dtype!r}; expected fp16, f16, or bf16"
            )
        object.__setattr__(self, "dtype", canonical)
