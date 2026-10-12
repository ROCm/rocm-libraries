################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Common.RegisterPool``: the
``allocTmpGpr`` / ``allocTmpGprList`` context-manager allocators over a real
rocisa ``RegisterPool``."""

import pytest

from rocisa.register import RegisterPool as RocisaPool
from rocisa.enum import RegisterType

from Tensile.Common.RegisterPool import allocTmpGpr, allocTmpGprList

pytestmark = pytest.mark.unit


def _pool(n=64):
    p = RocisaPool(8, RegisterType.Sgpr, False)
    p.add(0, n, "init")
    return p


def test_alloc_tmp_gpr_basic():
    pool = _pool()
    with allocTmpGpr(pool, 2, upperLimit=64) as reg:
        assert reg.size == 2
        assert reg.idx >= 0


@pytest.mark.parametrize("num", [1, 4], ids=["num1_align1", "num4_align2"])
def test_alloc_tmp_gpr_default_alignment(num):
    pool = _pool()
    with allocTmpGpr(pool, num, upperLimit=64) as reg:
        assert reg.size == num


def test_alloc_tmp_gpr_explicit_tag_and_alignment():
    pool = _pool()
    with allocTmpGpr(pool, 2, upperLimit=64, alignment=2, tag="mine") as reg:
        assert reg.size == 2


def test_alloc_tmp_gpr_overflow_listener_receives_exception():
    # Overflow WITH a listener: the listener is invoked and the body still runs.
    pool = _pool()
    seen = []
    with allocTmpGpr(pool, 4, upperLimit=2, overflowListener=seen.append) as reg:
        assert reg.size == 4
    # CHARACTERIZED BUG: ResourceOverflowException is defined as a *function*
    # (`def ResourceOverflowException(Exception): pass`), so the constructed
    # "exception" is actually None — the listener receives None. See DECISIONS.
    assert seen == [None]


def test_alloc_tmp_gpr_overflow_no_listener_raises_typeerror():
    # Overflow WITHOUT a listener -> `raise None` -> TypeError (the same bug).
    pool = _pool()
    with pytest.raises(TypeError):
        with allocTmpGpr(pool, 4, upperLimit=2):
            pass


def test_alloc_tmp_gpr_list_default_alignments():
    pool = _pool()
    with allocTmpGprList(pool, [1, 2], upperLimit=64) as regs:
        assert [r.size for r in regs] == [1, 2]


def test_alloc_tmp_gpr_list_broadcast_alignment():
    pool = _pool()
    with allocTmpGprList(pool, [2, 4], upperLimit=64, alignments=[2]) as regs:
        assert [r.size for r in regs] == [2, 4]


def test_alloc_tmp_gpr_list_matching_alignments():
    pool = _pool()
    with allocTmpGprList(pool, [2, 4], upperLimit=64, alignments=[2, 2]) as regs:
        assert [r.size for r in regs] == [2, 4]


def test_alloc_tmp_gpr_list_bad_alignment_asserts():
    # num % alignment != 0 -> print + `assert 0` -> AssertionError.
    pool = _pool()
    with pytest.raises(AssertionError):
        with allocTmpGprList(pool, [3], upperLimit=64, alignments=[2]):
            pass


def test_alloc_tmp_gpr_list_matching_bad_alignment_asserts():
    # len(alignments) > 1 matching path with num % alignment != 0 -> assert 0.
    pool = _pool()
    with pytest.raises(AssertionError):
        with allocTmpGprList(pool, [3, 4], upperLimit=64, alignments=[2, 2]):
            pass


def test_alloc_tmp_gpr_list_overflow_no_listener_raises_typeerror():
    pool = _pool()
    with pytest.raises(TypeError):
        with allocTmpGprList(pool, [4], upperLimit=2):
            pass


def test_alloc_tmp_gpr_list_overflow_listener():
    pool = _pool()
    seen = []
    with allocTmpGprList(pool, [4], upperLimit=2, overflowListener=seen.append) as regs:
        assert [r.size for r in regs] == [4]
    assert seen == [None]  # same ResourceOverflowException-is-a-function bug
