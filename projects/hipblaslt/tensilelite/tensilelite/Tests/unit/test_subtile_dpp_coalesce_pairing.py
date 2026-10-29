# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""DPP store-coalesce pair-detection unit tests.

`_subtileCoalescePairing` is the single source of truth shared by the boundary store body and
the interior peel.  Its correctness decides which paired stores get coalesced,
so it is covered here in isolation -- without a GPU -- by driving it against a synthetic
`batchElements` list.  Each element is (blockIdxN, tt0, ...); the coalesce joins pair P-1
(even pairIdx, deferred) with pair P (odd pairIdx, emits the coalesced store of prev+this), and only
when both pairs exist on adjacent tt0 rows in the SAME N-group.
"""

import pytest

from Tensile.Components.GlobalWriteBatch import GlobalWriteBatchWriter

pytestmark = pytest.mark.unit


def _writer(batch_elements):
    w = object.__new__(GlobalWriteBatchWriter)
    # only .batchElements is read by _subtileCoalescePairing
    w.batchElements = batch_elements
    return w


# Four M-adjacent pairs (tt0 = 0,1,2,3) in one N-group: pairs (0,1) and (2,3).
_FOUR_ROWS_N0 = [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_first_pair_defers_second_pair_coalesces():
    w = _writer(_FOUR_ROWS_N0)
    # pair P-1: sba=1 element at elementIdx=1, tt0=1 -> pairIdx=0 (even) -> DEFER
    first, second = w._subtileCoalescePairing(elementIdx=1, tt0=1, blockIdxN=0, canCoalesce=True)
    assert (first, second) == (True, False)
    # pair P: sba=1 element at elementIdx=3, tt0=3 -> pairIdx=1 (odd) -> COALESCE prev+this
    first, second = w._subtileCoalescePairing(elementIdx=3, tt0=3, blockIdxN=0, canCoalesce=True)
    assert (first, second) == (False, True)


def test_cannot_coalesce_never_coalesces():
    w = _writer(_FOUR_ROWS_N0)
    assert w._subtileCoalescePairing(1, 1, 0, canCoalesce=False) == (False, False)
    assert w._subtileCoalescePairing(3, 3, 0, canCoalesce=False) == (False, False)


def test_cross_n_group_is_not_coalesced():
    # Same tt0 layout but the second pair sits in N-group 1: the coalesce spans M, never N,
    # so the deferred-first detection must fail and nothing coalesces.
    w = _writer([(0, 0), (0, 1), (1, 2), (1, 3)])
    # elementIdx=1 is pairIdx=0: would defer only if elements 2,3 are same N (they are N=1).
    first, second = w._subtileCoalescePairing(1, 1, blockIdxN=0, canCoalesce=True)
    assert first is False  # partner pair is in a different N-group -> no defer
    # elementIdx=3 is pairIdx=1: its batchA (elems 0,1) is N=0, but blockIdxN here is 1.
    first, second = w._subtileCoalescePairing(3, 3, blockIdxN=1, canCoalesce=True)
    assert second is False


def test_missing_partner_pair_does_not_coalesce():
    # Only two rows (one pair): the even pair has no following pair to coalesce with.
    w = _writer([(0, 0), (0, 1)])
    first, second = w._subtileCoalescePairing(1, 1, 0, canCoalesce=True)
    assert (first, second) == (False, False)


def test_out_of_range_indices_are_safe():
    # elementIdx near the end: _samePair must bounds-check (i1 < len) and not raise.
    w = _writer(_FOUR_ROWS_N0)
    # elementIdx=3, pairIdx=1 but look-back is in range; a look-forward from the last even
    # pair with no successor must simply not coalesce.
    w2 = _writer([(0, 0), (0, 1)])  # elementIdx=1 even, look-forward 2,3 out of range
    first, second = w2._subtileCoalescePairing(1, 1, 0, canCoalesce=True)
    assert (first, second) == (False, False)


def test_parity_alternates_defer_coalesce():
    # Eight rows -> four pairs -> defer, coalesce, defer, coalesce.
    rows = [(0, t) for t in range(8)]
    w = _writer(rows)
    results = []
    for eidx, tt0 in [(1, 1), (3, 3), (5, 5), (7, 7)]:
        results.append(w._subtileCoalescePairing(eidx, tt0, 0, canCoalesce=True))
    assert results == [(True, False), (False, True), (True, False), (False, True)]
