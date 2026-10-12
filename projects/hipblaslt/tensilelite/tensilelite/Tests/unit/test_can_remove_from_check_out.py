# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""RegisterPool.canRemoveFromCheckOut agrees with removeFromCheckOut.

defineSgpr has a legitimate reason to skip a parked SGPR var: its registers may
not be reclaimable right now. It used to discover that by calling
removeFromCheckOut and matching the text of the RuntimeError that came back.
That is why these tests exist -- the match was substring-based, so any unrelated
pool failure whose message happened to contain the same phrase was swallowed
too, and a duplicate ScalarGlobalReadOffsetMXSA define went unnoticed because of
it.

The predicate replaces the match, so what is worth pinning is not the predicate's
answers on their own but that they are exactly the cases removeFromCheckOut
accepts. A predicate that drifted looser would start skipping registers that
should have been protected; one that drifted tighter would raise where the old
code correctly continued.
"""

import pytest

from rocisa.enum import RegisterType
from rocisa.register import RegisterPool

pytestmark = pytest.mark.unit


def pool(size=16):
    # preventOverflow=False: these tests care about status bookkeeping, not the
    # occupancy ceiling, and the pool grows rather than refusing a checkout.
    return RegisterPool(size, RegisterType.Sgpr, False)


def actuallyRemovable(p, start):
    """Whether removeFromCheckOut(start) really succeeds, by trying it."""
    try:
        p.removeFromCheckOut(start)
    except RuntimeError:
        return False
    return True


@pytest.mark.parametrize("state", ["in use", "parked", "restored", "borrowed", "unknown"])
def test_predicate_matches_the_operation(state):
    """The predicate answers True exactly when the operation succeeds."""
    p = pool()
    start = p.checkOut(2, "x")

    if state == "parked":
        p.addFromCheckOut(start)
    elif state == "restored":
        p.addFromCheckOut(start)
        p.removeFromCheckOut(start)
    elif state == "borrowed":
        # Parked, then the registers are handed to someone else -- the StreamK
        # SrdWS case, where a parked var is reused as a main-loop temp.
        p.addFromCheckOut(start)
        assert p.checkOut(2, "borrower") == start
    elif state == "unknown":
        start = start + 2 * 2  # an index that was never checked out

    predicted = p.canRemoveFromCheckOut(start)
    assert predicted is actuallyRemovable(p, start), \
        f"canRemoveFromCheckOut disagrees with removeFromCheckOut when {state}"


def test_only_a_parked_var_is_reclaimable():
    """Of those five states, exactly one is reclaimable."""
    p = pool()
    start = p.checkOut(2, "x")
    assert p.canRemoveFromCheckOut(start) is False      # held in use
    p.addFromCheckOut(start)
    assert p.canRemoveFromCheckOut(start) is True       # parked
    p.removeFromCheckOut(start)
    assert p.canRemoveFromCheckOut(start) is False      # restored


def test_the_predicate_does_not_mutate_the_pool():
    """Asking must be free of side effects, since defineSgpr asks every var."""
    p = pool()
    start = p.checkOut(2, "x")
    p.addFromCheckOut(start)
    before = p.state()
    for _ in range(3):
        assert p.canRemoveFromCheckOut(start) is True
    assert p.state() == before
    # and the operation it predicted still works afterwards
    p.removeFromCheckOut(start)
