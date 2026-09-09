# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for `decoupledThickGateRelaxation` -- the single owner of whether a
decoupled PGR pair gets a relaxed thick-tensor gate, and of the count it reads.

The invariant these pin, and the reason the function exists: the *presence* of a
relaxation follows from the resolved PGR pair alone and must be identical under
every descriptor grouping, because the thick/thin LDS-block asymmetry that makes
an early start legal is a property of the pair and the grouping cannot create or
remove it. What the grouping decides is the mechanism and the count, because the
legal count is how many independent tensor ops the grouping actually leaves
outstanding -- `s_wait_tensorcnt N` retires while N remain, so 0 is a full drain
and a larger N is a weaker gate. That is not a tunable: raising it past what the
grouping supports starts LDS reads before their data has landed.

Consequence worth stating because it has been misread: counting
`s_wait_tensorcnt 1` is a separate-descriptor-specific proxy. The shared
grouping's gate reads 2 -- the weaker of the two -- so a census of "wait1"
reports the shared grouping as unrelaxed when it is in fact relaxed further.

The cross-product runs over the rows of TDM_GROUPS rather than over the TDMFuse
integers. The integer is a stable-but-arbitrary index into that table, so a test
keyed on it cannot see the rows no integer selects yet.
"""

import pytest

from Tensile.Components import TDMFuse as TF
from Tensile.Components.DecouplePGR import decouplePGRBlocks
from Tensile.Components.TDMFuse import (
    DCP_THICK_GATE_SUPPORTED,
    DCP_THICK_GATE_TEXT,
    DCP_THICK_GATE_TOKENS,
    TDM_FUSE_GROUPING,
    TDM_GROUPS,
    _parseThickGateCountOverride,
    decoupledThickGateRelaxation,
    tdmGrouping,
    tdmSeparateABDescriptors,
)


def ks(fuse=0, pgrA=2, pgrB=1, **over):
    """A solution state that satisfies every guard the grouping consults.

    Deliberately complete: a state missing TDMInst or the MX scales would be
    declined by `_tdmFuseCanShareDescriptors` and fall back to the default
    grouping, which is the very case `test_declined_fusion_*` isolates on
    purpose. Starting from a passing state keeps those tests honest.
    """
    state = {
        "PrefetchGlobalRead": 1,
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "NumWaves": 4,
        "TDMCross": 0,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }
    if pgrA is None:
        del state["PrefetchGlobalReadA"]
    if pgrB is None:
        del state["PrefetchGlobalReadB"]
    state.update(over)
    return state


# Every pair shape derivation can hand the writer. None/None is the legacy
# scalar shape, which is also what an equal (k, k) pair is folded to by
# equalPairDegeneratesToScalar before it reaches the writer.
PAIRS = (
    (None, None, "scalar"),
    (0, 0, "equal"),
    (1, 1, "equal"),
    (2, 2, "equal"),
    (1, 0, "equal"),
    (2, 1, "divergent"),
    (1, 2, "divergent"),
    (2, 0, "divergent"),
    (0, 2, "divergent"),
)

# Expected answer per grouping row. Written out rather than derived from the row
# so the table and the predicate cannot agree by sharing a bug.
ROW_SEPARATES_AB = {
    "MX_AB": False,   # {A,B} + {MXSA,MXSB}
    "AB": False,      # {A,B} + {MXSA} + {MXSB}
    "paired": True,   # {A,MXSA} + {MXSB,B}
    "A_MX": True,     # {A,MXSA,MXSB} + {B}
    "B_MX": True,     # {B,MXSA,MXSB} + {A}
    "None": True,     # nothing fused, so every tensor is its own set
}


def shape_of(state):
    decoupled, blkA, blkB = decouplePGRBlocks(state)
    if not decoupled:
        return "scalar"
    return "divergent" if blkA != blkB else "equal"


def test_every_grouping_row_has_a_stated_expectation():
    assert set(ROW_SEPARATES_AB) == set(TDM_GROUPS), (
        "a grouping row was added without stating whether it separates A from B")


@pytest.mark.parametrize("name", sorted(TDM_GROUPS))
def test_separate_ab_descriptors_over_every_grouping_row(name, monkeypatch):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _s: TDM_GROUPS[name])
    assert tdmSeparateABDescriptors(ks()) is ROW_SEPARATES_AB[name]


@pytest.mark.parametrize("name", sorted(TDM_GROUPS))
@pytest.mark.parametrize("pgrA,pgrB,shape", PAIRS)
def test_owner_over_the_full_row_by_pair_cross_product(
        name, pgrA, pgrB, shape, monkeypatch):
    """Presence follows the pair; mechanism and count follow the row."""
    monkeypatch.setattr(TF, "tdmGrouping", lambda _s: TDM_GROUPS[name])
    gate = decoupledThickGateRelaxation(ks(pgrA=pgrA, pgrB=pgrB))
    if shape != "divergent":
        assert gate is None
        return
    assert gate is not None, (
        "a divergent pair earns a relaxation under every grouping; the %s row "
        "returned None, which is the fuse-gated-presence bug" % name)
    if ROW_SEPARATES_AB[name]:
        assert gate.mechanism == DCP_THICK_GATE_TOKENS
    else:
        assert gate.mechanism == DCP_THICK_GATE_TEXT
    assert gate.tensorcnt == DCP_THICK_GATE_SUPPORTED[gate.mechanism]


@pytest.mark.parametrize("pgrA,pgrB,shape", PAIRS)
def test_presence_is_identical_across_the_reachable_fuse_levels(pgrA, pgrB, shape):
    """No TDMFuse integer may add or remove a relaxation.

    This is the mutation guard. Re-gating *presence* on a fuse integer -- the
    defect the owner exists to prevent -- makes the divergent rows disagree here
    and fails loudly.
    """
    present = {
        fuse: decoupledThickGateRelaxation(ks(fuse=fuse, pgrA=pgrA, pgrB=pgrB)) is not None
        for fuse in sorted(TDM_FUSE_GROUPING)
    }
    assert len(set(present.values())) == 1, (
        "presence differs by TDMFuse: %s" % present)
    assert next(iter(present.values())) is (shape == "divergent")


@pytest.mark.parametrize("fuse,expected", sorted(TDM_FUSE_GROUPING.items()))
def test_reachable_fuse_integers_resolve_to_their_row(fuse, expected):
    """Pins the integer-to-row mapping in one place, so nothing else reads it."""
    assert tdmGrouping(ks(fuse=fuse)).name == expected


# --- the guard is now where the decision is --------------------------------
#
# `_tdmFuseCanShareDescriptors` can decline the grouping TDMFuse names and fall
# back to the default, where A and B share one descriptor set. When the emission
# sites tested `TDMFuse == 1` themselves, a declined solution was handed disjoint
# tokens for descriptors that are in fact shared: the guard was not where the
# decision was, and it failed open. These cases are unreachable in a surviving
# solution today -- Solution.py rejects each of them -- so they are an invariant
# check, and the one that catches the divergence if a future change reopens it.
DECLINES = (
    ("TDMSplit", {"TDMSplit": True}),
    ("UseSubtileImpl", {"UseSubtileImpl": True}),
    ("no MX scale on A", {"ProblemType": {"MXBlockA": 0, "MXBlockB": 32}}),
    ("no MX scale on B", {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0}}),
    ("NumWaves==1", {"NumWaves": 1}),
    ("TDM moves A only", {"TDMInst": 1}),
    ("TDM moves B only", {"TDMInst": 2}),
)


@pytest.mark.parametrize("why,override", DECLINES)
def test_declined_fusion_falls_back_to_the_shared_grouping(why, override):
    assert tdmGrouping(ks(fuse=1, **override)).name == TDM_FUSE_GROUPING[0], why


@pytest.mark.parametrize("why,override", DECLINES)
def test_declined_fusion_never_gets_disjoint_tokens(why, override):
    """The divergence catcher for the relocated guard."""
    gate = decoupledThickGateRelaxation(ks(fuse=1, **override))
    assert gate is not None, why
    assert gate.mechanism == DCP_THICK_GATE_TEXT, (
        "TDMFuse=1 declined on %s shares one descriptor set, so its tensor ops "
        "share a token and cannot be drained independently; handing it "
        "DCP_THICK_GATE_TOKENS would gate reads on data that has not landed" % why)


@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
@pytest.mark.parametrize("why,override", DECLINES + (("(none)", {}),))
def test_mechanism_agrees_with_the_grouping_by_construction(fuse, why, override):
    """The owner and the grouping cannot disagree, whatever the guards say."""
    state = ks(fuse=fuse, **override)
    gate = decoupledThickGateRelaxation(state)
    if gate is None:
        return
    separates = tdmSeparateABDescriptors(state)
    assert (gate.mechanism == DCP_THICK_GATE_TOKENS) is separates, why


def test_token_path_requires_a_second_token_stream():
    """A separate-descriptor pair the TDM does not move on both tensors, or whose
    thin side is not single-buffered, has no second stream to skip past."""
    assert decoupledThickGateRelaxation(ks(fuse=1, enableTDMA=False)) is None
    assert decoupledThickGateRelaxation(ks(fuse=1, enableTDMB=False)) is None
    assert decoupledThickGateRelaxation(ks(fuse=1, pgrA=3, pgrB=2)) is None


# --- gate-pricing hook: downward only --------------------------------------

def test_override_accepts_the_downward_sweep():
    assert _parseThickGateCountOverride("") == {}
    assert _parseThickGateCountOverride("text=2") == {DCP_THICK_GATE_TEXT: 2}
    assert _parseThickGateCountOverride("text=1") == {DCP_THICK_GATE_TEXT: 1}
    assert _parseThickGateCountOverride("text=0") == {DCP_THICK_GATE_TEXT: 0}
    assert _parseThickGateCountOverride("tokens=1") == {DCP_THICK_GATE_TOKENS: 1}
    assert _parseThickGateCountOverride("tokens=0") == {DCP_THICK_GATE_TOKENS: 0}
    assert _parseThickGateCountOverride("text=1,tokens=0") == {
        DCP_THICK_GATE_TEXT: 1, DCP_THICK_GATE_TOKENS: 0}


@pytest.mark.parametrize("spec", ["text=3", "tokens=2", "text=-1", "tokens=99"])
def test_override_refuses_to_sweep_upward(spec):
    with pytest.raises(ValueError, match="upward|outside"):
        _parseThickGateCountOverride(spec)


def test_override_refuses_an_unknown_mechanism():
    with pytest.raises(ValueError, match="unknown mechanism"):
        _parseThickGateCountOverride("tokns=1")


def test_supported_counts_match_the_documented_mechanisms():
    assert DCP_THICK_GATE_SUPPORTED == {DCP_THICK_GATE_TEXT: 2,
                                        DCP_THICK_GATE_TOKENS: 1}
