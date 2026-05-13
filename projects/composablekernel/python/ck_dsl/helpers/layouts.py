# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shared LDS layout descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class LdsLayout:
    """2D LDS layout for fp16 tiles.

    `logical_cols` is the number of useful columns. `k_pad` adds trailing
    padding to the physical row stride. `requires_packed_async` marks layouts
    that can be directly written by `raw_ptr_buffer_load_lds`.
    """

    logical_cols: int
    k_pad: int = 0
    swizzle: Optional[str] = None
    requires_packed_async: bool = False

    @classmethod
    def padded_k(cls, logical_cols: int, k_pad: int = 8) -> "LdsLayout":
        return cls(logical_cols=int(logical_cols), k_pad=int(k_pad), swizzle=None)

    @classmethod
    def packed_async(cls, logical_cols: int) -> "LdsLayout":
        return cls(
            logical_cols=int(logical_cols),
            k_pad=0,
            swizzle=None,
            requires_packed_async=True,
        )

    @property
    def row_stride(self) -> int:
        return self.logical_cols + self.k_pad

    def storage_shape(self, rows: int) -> Tuple[int, int]:
        return int(rows), self.row_stride

    def validate_for_async(self) -> None:
        if self.k_pad != 0:
            raise ValueError("async LDS layout must be packed: k_pad must be 0")
        if self.swizzle is not None:
            raise ValueError(
                "async LDS layout cannot use arbitrary per-lane swizzle; "
                "express swizzle in consumer read math instead"
            )

    def validate(self) -> None:
        if self.logical_cols <= 0:
            raise ValueError("logical_cols must be positive")
        if self.k_pad < 0:
            raise ValueError("k_pad must be >= 0")
        if self.swizzle not in (None, "xor", "cyclic"):
            raise ValueError(f"unsupported LDS swizzle {self.swizzle!r}")
