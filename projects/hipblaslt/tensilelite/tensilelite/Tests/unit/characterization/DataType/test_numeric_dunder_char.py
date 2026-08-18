################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the numeric reads, the string/state forms, the
comparison dunders, and the module-level ``_populateLookupTable`` of
``Tensile.Common.DataType``.

* numeric: ``numRegisters`` / ``numBytes`` / ``MIOutputTypeNameAbbrev`` /
  ``flopsPerMac`` — table reads (and the real/complex branch of ``flopsPerMac``).
* state: ``state`` / ``__str__`` / ``__repr__`` / ``getAttributes``.
* dunders: ``__hash__`` / ``__eq__`` (DataType and non-DataType) / ``__lt__``
  (DataType and the ``NotImplemented`` -> ``TypeError`` non-DataType path).
* ``_populateLookupTable``: the happy rebuild plus its two guard raises
  (index-mismatch and duplicate-key) driven by tiny synthetic property lists.
"""

import pytest
from rocisa.enum import DataTypeEnum

from Tensile.Common.DataType import DataType, _populateLookupTable

pytestmark = pytest.mark.unit


_ENUMS = [p["enum"] for p in DataType.properties]


# ===========================================================================
# Numeric reads
# ===========================================================================

def test_num_registers(snapshot):
    assert {e.name: DataType(e).numRegisters() for e in _ENUMS} == snapshot


def test_num_bytes(snapshot):
    assert {e.name: DataType(e).numBytes() for e in _ENUMS} == snapshot


def test_mi_output_type_name_abbrev(snapshot):
    assert {e.name: DataType(e).MIOutputTypeNameAbbrev() for e in _ENUMS} == snapshot


def test_flops_per_mac(snapshot):
    # 2 for real, 8 for complex -> exercises both isReal() outcomes.
    assert {e.name: DataType(e).flopsPerMac() for e in _ENUMS} == snapshot


# ===========================================================================
# state / string forms
# ===========================================================================

def test_state(snapshot):
    assert {e.name: DataType(e).state() for e in _ENUMS} == snapshot


def test_str(snapshot):
    assert {e.name: str(DataType(e)) for e in _ENUMS} == snapshot


def test_repr(snapshot):
    assert {e.name: repr(DataType(e)) for e in _ENUMS} == snapshot


def test_get_attributes(snapshot):
    assert {e.name: DataType(e).getAttributes() for e in _ENUMS} == snapshot


# ===========================================================================
# __hash__ / __eq__ / __lt__
# ===========================================================================

def test_hash_consistency(snapshot):
    a = DataType(DataTypeEnum.Half)
    b = DataType(DataTypeEnum.Half)
    c = DataType(DataTypeEnum.Float)
    summary = {
        "equal_objects_equal_hash": hash(a) == hash(b),
        "hash_matches_attributes": hash(a) == hash(a.getAttributes()),
        "distinct_values_distinct_hash": hash(a) != hash(c),
    }
    assert summary == snapshot


def test_eq(snapshot):
    a = DataType(DataTypeEnum.Half)
    b = DataType(DataTypeEnum.Half)
    c = DataType(DataTypeEnum.Float)
    summary = {
        "same_value": a == b,
        "different_value": a == c,
        # __eq__ returns NotImplemented for non-DataType -> Python falls back
        # to identity, yielding False (and True for !=).
        "vs_str_eq": a == "H",
        "vs_int_eq": a == 4,
        "vs_none_ne": a != None,  # noqa: E711 - exercising __eq__/__ne__ fallback
    }
    assert summary == snapshot


def test_lt_ordering(snapshot):
    lo = DataType(DataTypeEnum.Float)   # value 0
    hi = DataType(DataTypeEnum.Double)  # value 1
    summary = {
        "lo_lt_hi": lo < hi,
        "hi_lt_lo": hi < lo,
        # total_ordering derives these from __lt__/__eq__.
        "lo_le_lo": lo <= DataType(DataTypeEnum.Float),
        "hi_gt_lo": hi > lo,
    }
    assert summary == snapshot


def test_lt_vs_non_datatype_raises():
    # __lt__ returns NotImplemented for a non-DataType; with no reflected
    # operation available Python raises TypeError.
    with pytest.raises(TypeError):
        _ = DataType(DataTypeEnum.Float) < "Z"


# ===========================================================================
# _populateLookupTable — happy rebuild + the two guard raises
# ===========================================================================

def test_populate_lookup_table_rebuild(snapshot):
    # Rebuild into a fresh dict from the real properties; snapshot the full
    # (sorted) str -> index map. Covers the loop, the index-match check (false
    # branch), and both key insertions per row.
    fresh = {}
    _populateLookupTable(DataType.properties, fresh)
    assert dict(sorted(fresh.items())) == snapshot


def test_populate_lookup_table_index_mismatch_raises(snapshot):
    # An entry whose position != its enum value trips the index guard.
    # Double.value == 1 placed at index 0 -> RuntimeError.
    bad = [{"enum": DataTypeEnum.Double, "char": "D"}]
    with pytest.raises(RuntimeError) as excinfo:
        _populateLookupTable(bad, {})
    assert str(excinfo.value) == snapshot


def test_populate_lookup_table_duplicate_key_raises(snapshot):
    # Two rows at their correct indices but sharing a 'char' -> duplicate-key
    # guard fires on the second row.
    dup = [
        {"enum": DataTypeEnum.Float, "char": "S"},   # index 0, value 0
        {"enum": DataTypeEnum.Double, "char": "S"},  # index 1, value 1, dup char
    ]
    with pytest.raises(RuntimeError) as excinfo:
        _populateLookupTable(dup, {})
    assert str(excinfo.value) == snapshot
