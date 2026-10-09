# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tensor storage, digests, and conservative error budgets shared by operations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


def encode(values: np.ndarray, dtype: str) -> np.ndarray:
    """Store fp16 values or round-to-nearest-even bf16 bits without Torch."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if dtype == "fp16":
        return values.astype("<f2")
    if dtype != "bf16":
        raise ValueError(f"unsupported GPU reference dtype: {dtype}")
    bits = values.view(np.uint32)
    bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return ((bits + bias) >> 16).astype("<u2")


def decode(values: np.ndarray, dtype: str) -> np.ndarray:
    """Decode tensor storage, checking its dtype rather than reinterpreting it."""
    expected = np.dtype("<f2" if dtype == "fp16" else "<u2")
    if dtype not in ("fp16", "bf16") or values.dtype != expected:
        raise ValueError(f"invalid {dtype} tensor storage: {values.dtype}")
    if dtype == "fp16":
        return values.astype(np.float32)
    return (values.astype(np.uint32) << 16).view(np.float32)


def max_abs_upper(left: np.ndarray, right: np.ndarray) -> float:
    """Bound max(abs(left-right)) for the represented finite floating values.

    Both operands are promoted before subtraction. Rounding the largest f64
    difference upward by one representable value encloses subtraction's
    round-to-nearest error; max/abs introduce no further rounding. An exact
    zero stays zero. No reductions, relative denominators, or squared norms
    are silently substituted for the existing absolute maximum norm.
    """
    if left.shape != right.shape or not left.size:
        raise ValueError(f"incompatible or empty outputs: {left.shape}, {right.shape}")
    left, right = left.astype(np.float64), right.astype(np.float64)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("GPU reference comparison contains non-finite values")
    with np.errstate(over="ignore", invalid="ignore"):
        distance = float(np.max(np.abs(left - right)))
    if not math.isfinite(distance):
        raise ValueError("GPU reference distance overflowed")
    return math.nextafter(distance, math.inf) if distance else 0.0


@dataclass(frozen=True)
class ErrorBudget:
    """A conservative triangle-inequality budget with strictly positive margin."""

    tolerance: float
    baseline_error_bound: float
    margin: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("non-finite GPU reference error budget")
        if self.baseline_error_bound < 0 or self.margin <= 0 or self.tolerance <= 0:
            raise ValueError("invalid GPU reference error budget")
        if self.remaining <= 0:
            raise ValueError("pinned GPU reference leaves no comparison budget")

    @property
    def remaining(self) -> Fraction:
        return (
            Fraction(self.tolerance)
            - Fraction(self.baseline_error_bound)
            - Fraction(self.margin)
        )

    @property
    def comparison_limit(self) -> float:
        # Fraction avoids cancellation/rounding while allocating the budget.
        # Round downward so a binary float threshold cannot overdraw it.
        return math.nextafter(float(self.remaining), -math.inf)

    def check(self, distance: float) -> None:
        """Reject a comparison that does not certify the original tolerance."""
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("invalid GPU reference comparison distance")
        if distance > self.comparison_limit:
            raise AssertionError(
                f"GPU reference max_abs={distance:.9g} exceeds remaining limit "
                f"{self.comparison_limit:.9g}; baseline_bound="
                f"{self.baseline_error_bound:.9g}, margin={self.margin:.9g}, "
                f"original_tolerance={self.tolerance:.9g}"
            )


def array_digest(array: np.ndarray) -> str:
    """Hash the logical tensor contract and contiguous little-endian bytes."""
    array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    header = json.dumps(
        {"shape": array.shape, "dtype": array.dtype.str}, sort_keys=True
    ).encode()
    return hashlib.sha256(header + b"\n" + array.tobytes()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digests(root: Path) -> dict[str, str]:
    """Describe a source/artifact tree without interpreter-generated caches."""
    return {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
