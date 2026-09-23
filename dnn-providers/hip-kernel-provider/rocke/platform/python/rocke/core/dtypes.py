# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Target-independent dtype names shared by specs and architecture catalogs.

These names describe logical formats. Recognition does not imply an IR scalar
representation, conversion support, or a matrix instruction on a given target.
Target-specific support belongs to the architecture catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Canonical dtype spellings used as catalog keys. Instance/spec dtype strings
# are normalised through this map so "f16"/"half" and "fp16" all resolve.
_DTYPE_ALIASES = {
    "f16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f32": "fp32",
    "float": "fp32",
    "fp32": "fp32",
    "fp8": "fp8e4m3",
    "fp8e4m3": "fp8e4m3",
    "bf8": "bf8e5m2",
    "bf8e5m2": "bf8e5m2",
    "fp6": "fp6e2m3",
    "fp6e2m3": "fp6e2m3",
    "bf6": "fp6e3m2",
    "fp6e3m2": "fp6e3m2",
    "fp4": "fp4e2m1",
    "fp4e2m1": "fp4e2m1",
    # Integer WMMA: "iu8"/"iu4" are the RDNA WMMA integer operand families
    # (signedness is an instruction operand, not the dtype); "i32" is the
    # integer accumulator. Scalar int spellings pass through for completeness.
    "iu8": "iu8",
    "iu4": "iu4",
    "i8": "i8",
    "int8": "i8",
    "i4": "i4",
    "int4": "i4",
    "i32": "i32",
    "int32": "i32",
}


def normalize_dtype(name: str) -> str:
    """Map a dtype spelling to its canonical catalog key."""
    key = name.strip().lower()
    return _DTYPE_ALIASES.get(key, key)


class DTypeCategory(str, Enum):
    FLOAT = "float"
    INTEGER = "integer"
    PREDICATE = "predicate"


@dataclass(frozen=True)
class DTypeInfo:
    """Logical encoding, independent of storage slots and instruction support."""

    name: str
    category: DTypeCategory
    encoded_bits: int


_DTYPE_INFO = {
    name: DTypeInfo(name, category, bits)
    for category, entries in (
        (
            DTypeCategory.FLOAT,
            (
                ("fp32", 32),
                ("fp16", 16),
                ("bf16", 16),
                ("fp8e4m3", 8),
                ("bf8e5m2", 8),
                ("fp6e2m3", 6),
                ("fp6e3m2", 6),
                ("fp4e2m1", 4),
                ("e8m0", 8),
                ("e4m3", 8),
                ("e5m3", 8),
            ),
        ),
        (
            DTypeCategory.INTEGER,
            (
                ("i4", 4),
                ("iu4", 4),
                ("i8", 8),
                ("iu8", 8),
                ("i16", 16),
                ("i32", 32),
                ("i64", 64),
            ),
        ),
        (DTypeCategory.PREDICATE, (("i1", 1),)),
    )
    for name, bits in entries
}


def dtype_info(name: str) -> DTypeInfo:
    """Resolve an encoding; recognition alone does not enable an operation.

    Scale names remain distinct keys. In particular e5m3 is not bf8e5m2.
    Operand-role validation belongs to the selected atom.
    """
    key = normalize_dtype(name)
    if key not in _DTYPE_INFO:
        raise ValueError(f"unknown dtype {name!r}")
    return _DTYPE_INFO[key]
