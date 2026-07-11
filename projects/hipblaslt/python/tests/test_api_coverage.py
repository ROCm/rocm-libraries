# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Header-derived enum coverage tests.

For each bound Python enum, verify that every C header member is either:
  (a) represented by a bound Python value (integer equality), OR
  (b) listed in ``_coverage.ALLOWED_MISSING`` with an explanatory comment.

These tests are GPU-free — they only parse the header and inspect Python
enum values via the reflection registry (``_core.enum_members``).
"""
import pytest
import hipblaslt
from hipblaslt import _coverage

c = hipblaslt._core

# Pairs of (Python bound enum name, C header typedef enum name).
ENUMS = [
    ("Epilogue", "hipblasLtEpilogue_t"),
    ("ScaleMode", "hipblasLtMatmulMatrixScale_t"),
    ("MatmulDescAttr", "hipblasLtMatmulDescAttributes_t"),
]


@pytest.mark.parametrize("bound_name,header_enum", ENUMS)
def test_every_header_value_is_bound_or_allowed(bound_name, header_enum):
    """All header enum values must be bound or explicitly allowed missing."""
    header = _coverage.find_header()
    header_values = set(_coverage.header_enum_values(header, header_enum).values())
    bound_values = set(c.enum_members(bound_name).values())
    missing = header_values - bound_values
    allowed_missing = _coverage.ALLOWED_MISSING.get(header_enum, set())
    unexpected = missing - allowed_missing
    assert not unexpected, (
        f"{header_enum}: header integer values {sorted(unexpected)} are not bound "
        f"and not in ALLOWED_MISSING.\n"
        f"Either add them to enums.cpp or add their values to "
        f"_coverage.ALLOWED_MISSING with a comment explaining why."
    )


@pytest.mark.parametrize("bound_name,header_enum", ENUMS)
def test_bound_values_match_header_integers(bound_name, header_enum):
    """Every bound Python enum value must match the header integer exactly."""
    header = _coverage.find_header()
    header_by_value = {
        v: k
        for k, v in _coverage.header_enum_values(header, header_enum).items()
    }
    bound_members = c.enum_members(bound_name)
    mismatches = []
    for py_name, py_value in bound_members.items():
        if py_value not in header_by_value:
            mismatches.append(
                f"  {py_name}={py_value} not found in header {header_enum}"
            )
    assert not mismatches, (
        f"{header_enum}: bound value(s) not present in header:\n"
        + "\n".join(mismatches)
    )
