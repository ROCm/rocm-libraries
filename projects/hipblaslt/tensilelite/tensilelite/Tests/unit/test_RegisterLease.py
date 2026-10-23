# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for DeadRegisterLease (Common/RegisterLease.py).

Covers the two properties the subtile store relies on:
  * backward-compat with the raw ``[(base, size)]`` list the weave operand lend used to
    return -- consumers iterate / len() / bool() it unchanged; and
  * disjoint explicit reuse via ``lease`` -- two leases can never alias, over-request
    fails loud, and construction rejects overlapping dead windows up front.
"""
import pytest

from Tensile.Common.RegisterLease import DeadRegisterLease

pytestmark = pytest.mark.unit


# -- backward-compatible list view ---------------------------------------------------------

def test_iterates_as_base_size_windows():
    lease = DeadRegisterLease([(8, 4), (20, 2)])
    assert list(lease) == [(8, 4), (20, 2)]


def test_windows_are_sorted_by_base():
    lease = DeadRegisterLease([(20, 2), (8, 4)])
    assert list(lease) == [(8, 4), (20, 2)]


def test_len_and_bool_match_the_raw_list():
    assert len(DeadRegisterLease([(8, 4), (20, 2)])) == 2
    assert bool(DeadRegisterLease([(8, 4)])) is True


def test_empty_pool_is_falsy_and_zero_length():
    empty = DeadRegisterLease([])
    assert len(empty) == 0
    assert bool(empty) is False
    assert list(empty) == []


def test_total_counts_registers_not_windows():
    assert DeadRegisterLease([(8, 4), (20, 2)]).total() == 6


# -- construction validation (fail loud, not silent corruption) ----------------------------

def test_overlapping_windows_rejected_at_construction():
    with pytest.raises(AssertionError, match="overlapping"):
        DeadRegisterLease([(8, 4), (10, 4)])


def test_abutting_windows_are_allowed():
    # [8,12) then [12,14) touch but do not overlap -- legal.
    lease = DeadRegisterLease([(8, 4), (12, 2)])
    assert list(lease) == [(8, 4), (12, 2)]


def test_nonpositive_window_size_rejected():
    with pytest.raises(AssertionError):
        DeadRegisterLease([(8, 0), (20, 2)])


# -- explicit reuse: lease carves disjoint sub-windows -------------------------------------

def test_lease_returns_base_and_shrinks_pool():
    lease = DeadRegisterLease([(8, 4)])
    assert lease.lease(2) == 8
    # remaining [10,12) is still leasable
    assert lease.lease(2) == 10
    assert lease.total() == 0


def test_two_leases_never_alias():
    lease = DeadRegisterLease([(8, 4)])
    a = lease.lease(2)
    b = lease.lease(2)
    assert a + 2 <= b or b + 2 <= a  # disjoint by construction


def test_lease_respects_alignment():
    # window [9,17): a 2-aligned lease must start at 10, not 9.
    lease = DeadRegisterLease([(9, 8)])
    assert lease.lease(4, align=2) == 10
    # the pre-alignment gap [9,10) survives as a leftover window
    assert (9, 1) in list(lease)


def test_lease_over_request_fails_loud():
    lease = DeadRegisterLease([(8, 4)])
    with pytest.raises(AssertionError, match="no 8-reg window"):
        lease.lease(8)


def test_lease_nonpositive_size_fails_loud():
    lease = DeadRegisterLease([(8, 4)])
    with pytest.raises(AssertionError, match="positive"):
        lease.lease(0)


def test_valuc_slot_reuse_shape_two_disjoint_4reg_windows():
    # The DPP fold's ValuC reuse: two 2-aligned 4-reg leases out of batchA's dead
    # accumulator slots. This is the pattern that replaces F1's hand-written parity
    # + disjointness assert.
    lease = DeadRegisterLease([(16, 8)])
    pack = lease.lease(4, align=2)
    store = lease.lease(4, align=2)
    assert pack % 2 == 0 and store % 2 == 0
    assert abs(store - pack) >= 4  # the window F1 asserted, now true by construction
