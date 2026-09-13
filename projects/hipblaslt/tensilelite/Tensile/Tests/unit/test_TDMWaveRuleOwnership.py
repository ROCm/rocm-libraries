# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wave rules and descriptor-set predicates have one owner, and the writer uses it.

Every test here drives a real `KernelWriterAssembly` method body, bound to a
stub instance, rather than comparing the partition model against itself. A model
test cannot see these defects: the model was already right, and the writer held
a second copy of each answer.  Where a row that no TDMFuse integer selects is
needed, `tdmGrouping` is patched to return it -- that makes the unwired row
reachable, it does not stand in for the writer.
"""

import pytest

import Tensile.KernelWriterAssembly as KWAMod
from Tensile.Components import TDMFuse as TF
from Tensile.KernelWriterAssembly import KernelWriterAssembly


def ks(**over):
    """A solution the writer accepts as TDMFuse=2 / A_MX on four waves."""
    state = dict(TDMFuse=2, TDMInst=3, NumWaves=4, TDMSplit=0, UseSubtileImpl=False,
                 enableTDMA=True, enableTDMB=True,
                 ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    state.update(over)
    return state


class Stub:
    """Carries real KernelWriterAssembly method bodies and nothing else."""

    def __init__(self, **binds):
        for name in ("isTdmWaveSeparated", "tdmFuseAMx", "tdmFusePaired",
                     "tdmSeparateABDescriptors", "tdmSetupIncrementWaveSeparated"):
                fn = getattr(KernelWriterAssembly, name, None)
                if fn is not None:
                    setattr(self, name, fn.__get__(self, Stub))
        for k, v in binds.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------
# One owner for "are A's and B's descriptors separate register sets?"
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TF.TDM_GROUPS))
def test_writer_predicate_matches_the_owner_on_every_row(monkeypatch, name):
    """The writer's answer must be the table's answer for every row, wired or not.

    Fails against the unfixed writer on `B_MX` and `None`: spelled
    `tdmFuseAMx or tdmFusePaired` it names two TDMFuse values, so it answers
    False for any row those values do not select -- including the two rows that
    do separate A from B. `B_MX` is the row being wired next.
    """
    row = TF.TDM_GROUPS[name]
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: row)
    # TDMFuse=0 so neither writer predicate fires off the integer; the row is
    # supplied by the table, which is where the writer should be reading it.
    state = ks(TDMFuse=0)
    assert Stub().tdmSeparateABDescriptors(state) is TF.tdmSeparateABDescriptors(state)


def test_owner_separates_ab_for_b_mx(monkeypatch):
    """B_MX separates A from B, so both owner and writer must say so."""
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS["B_MX"])
    state = ks(TDMFuse=0)
    assert TF.tdmSeparateABDescriptors(state) is True
    assert Stub().tdmSeparateABDescriptors(state) is True


def test_wave_separation_still_gates_the_writer():
    """Delegating must not drop the wave-separation precondition.

    TDMInst names both tensors, so the grouping resolves to A_MX, but the TDM is
    not moving A: the writer programs the default shared set and the thick gate
    must not hand out disjoint tokens for it.
    """
    assert TF.tdmSeparateABDescriptors(ks()) is True
    assert TF.tdmSeparateABDescriptors(ks(enableTDMA=False)) is False
    assert Stub().tdmSeparateABDescriptors(ks(enableTDMA=False)) is False


# --------------------------------------------------------------------------
# The TDMFuse=2 shared-set increment is a wave rule; it must read the partition
# --------------------------------------------------------------------------

def _incrementText(state):
    w = Stub(tdmFuseAMx=lambda _k: True, tdmFusePaired=lambda _k: False)
    mod = w.tdmSetupIncrementWaveSeparated(
        state, {"tensorChar": "A"}, {"tensorChar": "B"})
    return str(mod)


def test_shipped_increment_selects_waves_2_and_3():
    """The shipped 2/1/1 layout: MXSA on wave 2, MXSB on wave 3, A on waves 0-1."""
    text = _incrementText(ks())
    assert "wave 2 carries MXSA" in text
    assert "wave 3 carries MXSB" in text
    assert "waves 0-1 carry A" in text


def test_shipped_increment_literals_match_the_partition():
    """Pin the increment's literal wave rule to the table's answer.

    This site is deliberately NOT routed through the partition. A_MX has a single
    partitioned group and crossing reverses groups after the first, so crossing is
    already refused for the row and no reachable arrangement makes the literals
    wrong -- a structural finding, not a defect. Routing it would also make the
    dispatch read solution keys that stub-driven writer tests do not supply.

    So the duplication is made explicit and checked. If a row ever moves MXSA or
    MXSB, tdmSoleWave's answer changes, this test fails, and it names the literal
    that has to follow. That is the guard the duplication was missing, not a
    behaviour change: the emitted assembly is unchanged.
    """
    state = ks()
    assert TF.tdmSoleWave(state, "MXSA") == 2
    assert TF.tdmSoleWave(state, "MXSB") == 3
    assert TF.tdmWaveRangeText(state, "A") == "waves 0-1"
    text = _incrementText(state)
    assert "wave %d carries MXSA" % TF.tdmSoleWave(state, "MXSA") in text, text
    assert "wave %d carries MXSB" % TF.tdmSoleWave(state, "MXSB") in text, text
    assert "%s carry A" % TF.tdmWaveRangeText(state, "A") in text, text


def test_sole_wave_refuses_a_scale_spread_over_two_waves(monkeypatch):
    """One compare cannot select two waves, so the table's answer must refuse."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (2, (2, 3)))
    with pytest.raises(TF.TdmArrangementNotEmittable, match="one wave per member"):
        TF.tdmSoleWave(ks(), "MXSA")


# --------------------------------------------------------------------------
# tdmWaveComponents must refuse a share it cannot spell
# --------------------------------------------------------------------------

def test_wave_components_spells_the_three_shapes(monkeypatch):
    shapes = {(1, (2,)): None, (2, (0, 1)): 0, (2, (0, 2)): 1, (2, (1, 3)): 1}
    for part, shift in shapes.items():
        monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc, p=part: p)
        assert TF.tdmWaveComponents(ks(), "A") == (part[0], shift)


def test_wave_components_does_not_refuse_zero_components(monkeypatch):
    """A member on no wave at all keeps its shipped answer, and is not refused.

    At NumWaves=1 a multi-member group reports zero components for its later
    members: `_waveShares` computes `numWaves // numMembers`, so the member rides
    no wave and nothing reads its component id. Refusing here breaks every
    single-wave TDM kernel -- it is the boundary the refusal above must not cross,
    and getting it wrong failed nine codegen tests while leaving the feature
    corpus byte-identical, because that corpus has no single-wave TDM solution.
    """
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (0, (0,)))
    assert TF.tdmWaveComponents(ks(NumWaves=1), "A") == (0, 1)


def test_wave_components_refuses_a_trailing_two_wave_share(monkeypatch):
    """Waves (2,3) have no right-shift onto components 0..1.

    Fails against the unfixed function, which falls through to a shift of one and
    maps both waves onto component 1 -- two waves filling one LDS block, which
    the assembly gives no sign of. This is the shape the "remainder to the
    trailing members" reading of the comments would have produced.
    """
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (2, (2, 3)))
    with pytest.raises(TF.TdmArrangementNotEmittable, match=r"rides waves \(2, 3\)"):
        TF.tdmWaveComponents(ks(), "A")


# --------------------------------------------------------------------------
# The invariant that retired the de-aliased fill's hardcoded parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TF.TDM_GROUPS))
def test_a_set_that_b_owns_is_issued_by_every_wave(monkeypatch, name):
    """Whenever B owns a descriptor set, every wave issues that set.

    This invariant is what retired `_emitTdmDealiasedIssue`, a parity-gated
    fill carrying a literal even=A / odd=B wave rule. A set is filled once per
    wave and the per-wave descriptor programming picks the member, so a group
    is either one member -- issued by every wave -- or several, each wave
    carrying exactly one. Either way no row leaves a wave with nothing to
    issue, so there was never a row for the parity gate to fire on: it was
    unreachable, not merely unused.

    Asserted over every row including the unwired ones, because the row such a
    gate would fire on first is the next one somebody adds. `B_MX` is the
    concrete case -- it puts A in a one-member group, so the literal expected A
    on the even waves while A is on all four.

    This holds on the table before the change as well as after, and that is the
    point rather than a weakness: the invariant was always true, so the parity
    gate was always unreachable. What the change did was stop maintaining a
    wave rule that no row could ever select. Written with nothing but the
    published partition so it cannot pass merely because a new helper exists.
    """
    row = TF.TDM_GROUPS[name]
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: row)
    state = ks(TDMFuse=0)
    if not TF.tdmSeparateABDescriptors(state):
        pytest.skip("%s seats A and B on one set, so B issues nothing" % name)
    numWaves = state["NumWaves"]
    assignment = TF.tdmWaveAssignment(state)
    for tc in ("A", "B"):
        group = next(g for g in row.groups if tc in g)
        issuing = sorted(w for w, members in assignment.items()
                         if any(m in group for m in members))
        assert issuing == list(range(numWaves)), (
            "%s: every wave must carry some member of %s's set %s, got %s"
            % (name, tc, group, issuing))
