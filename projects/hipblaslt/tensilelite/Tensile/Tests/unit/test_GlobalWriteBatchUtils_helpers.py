# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the shared low-level helpers exported from GlobalWriteBatchUtils.

These helpers are consumed by both GlobalWriteBatch and StreamK to avoid
duplication (issue #7 fix).  The tests verify that:

* _extract_direct_vgpr_from_acc_read returns the physical VGPR index for
  VGPR-backed accumulator reads and None for AGPR-backed reads.

* _is_legal_valuC_offset correctly determines whether a ValuC staging slot
  is within the hardware VGPR limit.

* _has_any_vgpr_backed_accumulator correctly reflects the actual allocation
  recorded in tileInfo.vgprTiles, not just the kernel data types.
"""

import pytest
from unittest.mock import MagicMock

from Tensile.Components.Subtile.GlobalWriteBatchUtils import (
    _extract_direct_vgpr_from_acc_read,
    _has_any_vgpr_backed_accumulator,
    _is_legal_valuC_offset,
)


# ---------------------------------------------------------------------------
# Helpers for constructing mock acc-read instructions
# ---------------------------------------------------------------------------

def _mock_vgpr_src(reg_idx):
    """Simulate a plain arch-VGPR source operand (regType='v', numeric regIdx)."""
    src = MagicMock()
    src.regType = "v"
    src.regName = None
    src.regIdx = reg_idx
    return src


def _mock_agpr_src(reg_idx):
    """Simulate an accumulator register source (regType='a')."""
    src = MagicMock()
    src.regType = "a"
    src.regName = None
    src.regIdx = reg_idx
    return src


def _mock_named_vgpr_src(name):
    """Simulate a named VGPR operand (e.g. vgpr('ValuC+3'))."""
    src = MagicMock()
    src.regType = "v"
    src.regName = name
    src.regIdx = None
    return src


def _mock_inst(srcs):
    """Return a minimal mock instruction with the given source list."""
    inst = MagicMock()
    inst.srcs = srcs
    return inst


# ---------------------------------------------------------------------------
# Tests for _extract_direct_vgpr_from_acc_read
# ---------------------------------------------------------------------------

class TestExtractDirectVgprFromAccReadInst:
    """_extract_direct_vgpr_from_acc_read returns the physical VGPR index when
    the instruction is a VGPR→VGPR move (VGPR-first FP4 subtile accumulators),
    and None for all other cases.
    """

    def test_vgpr_backed_returns_index(self):
        """VGPR-backed accumulator: single plain-VGPR source → returns index."""
        inst = _mock_inst([_mock_vgpr_src(42)])
        assert _extract_direct_vgpr_from_acc_read(inst) == 42

    def test_vgpr_backed_zero_index(self):
        """Physical index 0 is a valid register and must be returned."""
        inst = _mock_inst([_mock_vgpr_src(0)])
        assert _extract_direct_vgpr_from_acc_read(inst) == 0

    def test_agpr_backed_returns_none(self):
        """AGPR source (regType='a'): this is a v_accvgpr_read → returns None."""
        inst = _mock_inst([_mock_agpr_src(10)])
        assert _extract_direct_vgpr_from_acc_read(inst) is None

    def test_named_vgpr_returns_none(self):
        """Named VGPR operand (regName is not None) → returns None."""
        inst = _mock_inst([_mock_named_vgpr_src("ValuC+3")])
        assert _extract_direct_vgpr_from_acc_read(inst) is None

    def test_multiple_srcs_returns_none(self):
        """Instructions with more than one source are not a simple move → None."""
        inst = _mock_inst([_mock_vgpr_src(5), _mock_vgpr_src(6)])
        assert _extract_direct_vgpr_from_acc_read(inst) is None

    def test_no_srcs_returns_none(self):
        """Empty source list → None."""
        inst = _mock_inst([])
        assert _extract_direct_vgpr_from_acc_read(inst) is None

    def test_non_integer_reg_idx_returns_none(self):
        """regIdx that is not an int (e.g. a string name) → None."""
        src = MagicMock()
        src.regType = "v"
        src.regName = None
        src.regIdx = "someSymbol"
        inst = _mock_inst([src])
        assert _extract_direct_vgpr_from_acc_read(inst) is None


# ---------------------------------------------------------------------------
# Tests for _is_legal_valuC_offset
# ---------------------------------------------------------------------------

class TestIsLegalValuCOffset:
    """_is_legal_valuC_offset(startVgprValu, maxVgpr, valuCOffset, width=1).

    Returns True when the ValuC staging slot at (startVgprValu + valuCOffset)
    fits entirely within [0, maxVgpr).
    """

    # Exact boundary cases
    def test_exactly_at_limit_is_legal(self):
        """startVgprValu + offset + 1 == maxVgpr is exactly on the boundary → legal."""
        assert _is_legal_valuC_offset(4, 256, 251, width=1) is True   # 4+251+1 = 256 ≤ 256

    def test_one_past_limit_is_illegal(self):
        """startVgprValu + offset + 1 > maxVgpr → illegal."""
        assert _is_legal_valuC_offset(4, 256, 252, width=1) is False  # 4+252+1 = 257 > 256

    def test_zero_offset_is_legal(self):
        """offset=0 with width=1 at startVgprValu=4 is always legal for MaxVgpr=256."""
        assert _is_legal_valuC_offset(4, 256, 0, width=1) is True

    # Width > 1
    def test_width_4_exact_boundary(self):
        """Multi-register store: last register at maxVgpr - 1 is still legal."""
        assert _is_legal_valuC_offset(4, 256, 248, width=4) is True   # 4+248+4 = 256

    def test_width_4_overflow(self):
        """Multi-register store spills past maxVgpr → illegal."""
        assert _is_legal_valuC_offset(4, 256, 249, width=4) is False  # 4+249+4 = 257

    def test_large_start_small_offset(self):
        """High startVgprValu reduces the legal range."""
        assert _is_legal_valuC_offset(200, 256, 55, width=1) is True   # 200+55+1 = 256
        assert _is_legal_valuC_offset(200, 256, 56, width=1) is False  # 200+56+1 = 257

    def test_start_zero(self):
        """startVgprValu=0 (unusual but must work)."""
        assert _is_legal_valuC_offset(0, 256, 255, width=1) is True
        assert _is_legal_valuC_offset(0, 256, 256, width=1) is False

    @pytest.mark.parametrize("offset,expected", [
        (0,   True),
        (100, True),
        (251, True),
        (252, False),
        (300, False),
    ])
    def test_parametrized_legal_checks(self, offset, expected):
        assert _is_legal_valuC_offset(4, 256, offset, width=1) is expected


# ---------------------------------------------------------------------------
# Tests for _has_any_vgpr_backed_accumulator
# ---------------------------------------------------------------------------

def _mock_vtile(is_vgpr: bool):
    """Return a minimal mock tile with a regList whose is_vgpr flag is set."""
    vtile = MagicMock()
    vtile.regList.is_vgpr = is_vgpr
    return vtile


def _mock_tile_info(vtile_flags):
    """Return a mock tileInfo whose vgprTiles list has the given is_vgpr flags."""
    ti = MagicMock()
    ti.vgprTiles = [_mock_vtile(f) for f in vtile_flags]
    return ti


class TestHasAnyVgprBackedAccumulator:
    """_has_any_vgpr_backed_accumulator(tileInfo) must reflect actual allocation.

    Unlike a data-type-only check, this function inspects the real register-
    allocation records, so it returns False when all D-tile registers fell back
    to AGPR (e.g. VGPR budget exhausted).
    """

    def test_none_tile_info_returns_false(self):
        """No tileInfo at all (non-subtile kernel) → False."""
        assert _has_any_vgpr_backed_accumulator(None) is False

    def test_all_agpr_returns_false(self):
        """All tiles allocated as AGPR → False, regardless of data type."""
        ti = _mock_tile_info([False, False, False, False])
        assert _has_any_vgpr_backed_accumulator(ti) is False

    def test_all_vgpr_returns_true(self):
        """All tiles allocated as VGPR → True."""
        ti = _mock_tile_info([True, True, True, True])
        assert _has_any_vgpr_backed_accumulator(ti) is True

    def test_mixed_vgpr_agpr_returns_true(self):
        """Some VGPR, some AGPR (partial VGPR-first) → True."""
        ti = _mock_tile_info([True, False, False, False])
        assert _has_any_vgpr_backed_accumulator(ti) is True

    def test_empty_tile_list_returns_false(self):
        """Empty vgprTiles (no D-tile registers at all) → False."""
        ti = _mock_tile_info([])
        assert _has_any_vgpr_backed_accumulator(ti) is False

    def test_single_vgpr_tile_returns_true(self):
        """Single VGPR-backed tile → True."""
        ti = _mock_tile_info([True])
        assert _has_any_vgpr_backed_accumulator(ti) is True

    def test_single_agpr_tile_returns_false(self):
        """Single AGPR-backed tile → False."""
        ti = _mock_tile_info([False])
        assert _has_any_vgpr_backed_accumulator(ti) is False
