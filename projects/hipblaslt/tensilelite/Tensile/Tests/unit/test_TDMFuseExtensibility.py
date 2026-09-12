# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adding a grouping row is three edits, in any order, and none of them silent.

Every test here fails against the code before `B_MX` was wired, and each one
names the failure mode it closes. The reason they exist as a file of their own
is that the defect class they guard is not "the table is wrong" -- the table was
right every time -- but "production never read the table". A test that asserts
`TDM_GROUPS["B_MX"]` exists passes just as well when nothing can select it.

The measured baseline these are written against: at commit a7328e108d, adding
`3` to `ValidParameters` and `3: "B_MX"` to `TDM_FUSE_GROUPING` produced a
kernel named `_TDMF3` that was byte-identical to the `f0` kernel, with no
error, no warning and no rejection -- and the two runs had the same digest, so
the mapping entry changed nothing at all.
"""

import pytest

from Tensile.Common.ValidParameters import validParameters
from Tensile.Components import TDMFuse as TF

pytestmark = pytest.mark.unit


def ks(fuse=3, **over):
    """A solution the resolver accepts as TDMFuse=3 / B_MX on four waves."""
    state = dict(TDMFuse=fuse, TDMInst=3, NumWaves=4, TDMSplit=0,
                 UseSubtileImpl=False, enableTDMA=True, enableTDMB=True,
                 ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    state.update(over)
    return state


# --------------------------------------------------------------------------
# The resolver fails closed on both halves of the lookup
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fuse", [-3, -1, 4, 5, 6, 7, 8])
def test_an_unmapped_integer_raises_instead_of_resolving(fuse):
    """Fails against the unfixed resolver, which returned the default row.

    `tdmGrouping` was `return TDM_GROUPS[TDM_FUSE_GROUPING[0]]`, so every
    integer outside {0,1,2} resolved to `MX_AB` while `Naming.py` still put the
    integer in the kernel name. Measured over -3..8: eleven of fourteen values
    silently produced an `f0` kernel under another name.
    """
    with pytest.raises(ValueError, match="names no grouping"):
        TF.tdmGrouping(ks(fuse=fuse))


def test_a_row_without_an_acceptance_entry_raises(monkeypatch):
    """A mapping entry alone must not make a row selectable.

    This is the fail-open half, and it is the one that made the wiring order
    matter: once `TDM_FUSE_GROUPING` named a row the resolver stopped raising,
    and if nothing had written the row's preconditions there was nothing left
    to enforce them. Raising here is what makes the three edits order-free.
    """
    monkeypatch.setitem(TF.TDM_FUSE_GROUPING, 6, "AB")
    with pytest.raises(ValueError, match="no _GROUPING_ACCEPTED entry"):
        TF.tdmGrouping(ks(fuse=6))


def test_every_mapped_row_has_an_acceptance_entry():
    """The invariant the test above protects, asserted over the shipped table."""
    for fuse, name in TF.TDM_FUSE_GROUPING.items():
        assert name in TF.TDM_GROUPS, (fuse, name)
        assert name in TF._GROUPING_ACCEPTED, (fuse, name)


def test_valid_parameters_and_the_mapping_name_the_same_integers():
    """Fails against the unfixed pair, where 3 was mappable but not valid.

    Either direction of drift is a defect with a different shape: listed but
    unmapped built a kernel named for a grouping it did not have, mapped but
    unlisted is merely unreachable. Pinning them together closes both.
    """
    assert sorted(validParameters["TDMFuse"]) == sorted(TF.TDM_FUSE_GROUPING)


def test_unwired_rows_are_unreachable_not_merely_unswept():
    """`AB` and `None` have no integer and no acceptance entry."""
    unreachable = set(TF.TDM_GROUPS) - set(TF.TDM_FUSE_GROUPING.values())
    assert unreachable == {"AB", "None"}
    assert not unreachable & set(TF._GROUPING_ACCEPTED)


# --------------------------------------------------------------------------
# B_MX resolves, and resolves as the mirror of A_MX
# --------------------------------------------------------------------------

def test_three_resolves_to_b_mx():
    """The assertion that a table-content test cannot make.

    Fails against the unfixed resolver with `B_MX` present in `TDM_GROUPS` and
    `3` present in `TDM_FUSE_GROUPING`: it still answered `MX_AB`.
    """
    assert TF.tdmGrouping(ks()).name == "B_MX"
    assert TF.tdmGrouping(ks()) is not TF.TDM_GROUPS[TF.TDM_GROUPING_DEFAULT]


def test_b_mx_is_the_wave_mirror_of_a_mx():
    """Same partition as TDMFuse=2 with A and B exchanged.

    Checked as a mirror rather than against literals so that a change to the
    2/1/1 policy cannot leave the two rows disagreeing.
    """
    swap = {"A": "B", "B": "A", "MXSA": "MXSA", "MXSB": "MXSB"}
    for tc, mirrored in swap.items():
        assert TF.tdmWavePartition(ks(fuse=2), tc) \
               == TF.tdmWavePartition(ks(fuse=3), mirrored), tc


def test_b_mx_separates_a_from_b_and_earns_the_token_gate():
    """The property the decoupled PGR thick gate turns on."""
    assert TF.tdmSeparateABDescriptors(ks()) is True
    assert TF.tdmGroupingSeparatesAB(ks()) is True


def test_b_mx_inherits_the_nothing_to_cross_rejection():
    """One partitioned group, so crossing is refused with no new branch."""
    assert TF.tdmCrossRejectReason(ks(TDMCross=1)) is not None
    assert "nothing to cross" in TF.tdmCrossRejectReason(ks(TDMCross=1))


def test_b_mx_inherits_the_pap_rejection():
    """A scale seated on a data tensor's set cannot take PAP's handoff."""
    reason = TF.tdmPapRejectReason(ks())
    assert reason is not None
    assert "B_MX" in reason and "{B,MXSA,MXSB}" in reason


# --------------------------------------------------------------------------
# NumWaves == 4 is pinned in both places, which is the cross-repo condition
# --------------------------------------------------------------------------

@pytest.mark.parametrize("numWaves", [1, 2, 3, 5, 8, 12, 16])
def test_b_mx_is_declined_away_from_four_waves(numWaves):
    """The guard the cross-repository claim on integer 3 depends on.

    The two trees agree on membership, member order and the unfused tensor, and
    their wave dispatch coincides ONLY at four waves: this side hands the
    `divmod` remainder to the leading members (3/3/2 at eight waves), the other
    side uses a fixed 2:1:1 (4/2/2). Pinning four waves makes the divergence
    unreachable, so it has to be asserted rather than assumed.

    Independently, 3/3/2 is not even emittable here -- the trailing two-wave
    share has no right-shift onto its components -- so this guard is also what
    keeps `tdmWaveComponents` from having to refuse.
    """
    assert TF.tdmGroupingAccepted(ks(NumWaves=numWaves)) is False
    assert TF.tdmGrouping(ks(NumWaves=numWaves)).name == TF.TDM_GROUPING_DEFAULT


def test_four_waves_is_accepted_and_shares_a_mx_entry():
    assert TF.tdmGroupingAccepted(ks()) is True
    assert TF._GROUPING_ACCEPTED["B_MX"] is TF._GROUPING_ACCEPTED["A_MX"]


@pytest.mark.parametrize("overrides", [
    {"TDMInst": 1}, {"TDMSplit": True}, {"UseSubtileImpl": True},
    {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0}},
    {"ProblemType": {"MXBlockA": 0, "MXBlockB": 32}},
])
def test_b_mx_inherits_every_a_mx_precondition(overrides):
    """Sharing the entry is the point: these are not written twice."""
    assert TF.tdmGroupingAccepted(ks(**overrides)) is False
    assert TF.tdmGroupingAccepted(ks(fuse=2, **overrides)) is False


# --------------------------------------------------------------------------
# Descriptor-set ownership is derived, not branched per value
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("MX_AB", {"A": "A", "B": "A", "MXSA": "MXSA", "MXSB": "MXSA"}),
    ("paired", {"A": "A", "MXSA": "A", "B": "B", "MXSB": "B"}),
    ("A_MX", {"A": "A", "MXSA": "A", "MXSB": "A", "B": "B"}),
    ("B_MX", {"B": "B", "MXSA": "B", "MXSB": "B", "A": "A"}),
    ("AB", {"A": "A", "B": "A", "MXSA": "MXSA", "MXSB": "MXSB"}),
    ("None", {"A": "A", "B": "B", "MXSA": "MXSA", "MXSB": "MXSB"}),
])
def test_set_owner_reproduces_every_rows_allocation(monkeypatch, name, expected):
    """`defineTdmSgprs` reads this instead of carrying an arm per TDMFuse value.

    The three wired rows' expectations here are the allocations the writer
    already emitted, which is why moving `defineTdmSgprs` onto the table left
    the corpus byte-identical. The unwired rows are included so the derivation
    is pinned before somebody points an integer at one.
    """
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS[name])
    for tc, owner in expected.items():
        assert TF.tdmSetOwner(ks(fuse=0), tc) == owner, tc


def test_the_shared_scale_set_is_recognised_by_structure():
    """A_MX and B_MX only; the paired row splits the scales between two sets."""
    assert TF.tdmSharedScaleSetOwner(ks(fuse=2)) == "A"
    assert TF.tdmSharedScaleSetOwner(ks(fuse=3)) == "B"
    assert TF.tdmSharedScaleSetOwner(ks(fuse=1)) is None
    assert TF.tdmSharedScaleSetOwner(ks(fuse=0)) is None


def test_shared_set_order_exchanges_the_pair_for_the_mirror():
    """What the writer asks instead of assuming the owner is A."""
    assert TF.tdmSharedSetOrder(ks(fuse=2), "A", "B") == ("A", "B")
    assert TF.tdmSharedSetOrder(ks(fuse=3), "A", "B") == ("B", "A")


def test_shared_scale_set_needs_wave_separation():
    """The precondition the table does not carry, kept out of the row."""
    assert TF.tdmSharedScaleSetActive(ks()) is True
    assert TF.tdmSharedScaleSetActive(ks(enableTDMA=False)) is False
    assert TF.tdmSharedScaleSetActive(ks(NumWaves=1)) is False
