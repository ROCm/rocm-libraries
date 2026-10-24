# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The two stack-selection rules that decide a TLU=1 subtile's LDS layout.

Both are easy to replace with something that looks equivalent and is not, so
both are pinned here against an executable statement of the rejected form.

1. `_subtileStackForTile` -- how much padding is worth a taller stack.

   The rule in force is a strip bound: round up to the next power of two only
   while the rounded height still holds the whole tile in one strip::

       if roundedUp > exact and roundedUp >= mtTiles:

   The alternative form is a ratio cap, which rounds up while the padding stays
   within 4/3 of the tile and does not care whether the result spans one strip.

   The two agree on every power-of-two tile and on the small exact divisors,
   and disagree for 27 of the first 40 tile counts -- including mtTiles 20,
   which is MT320 at MatrixInstM 16.  Neither is obviously wrong; they are
   different policies, and the point of this file is that swapping one for the
   other is a deliberate act with a visible blast radius, not a merge artifact.

   This file was written when the ratio cap was the rule in force and the strip
   bound was the alternative.  Upstream swapped them -- which is exactly the
   event this pair of rules was pinned to make visible -- so the roles below are
   reversed from the original and the divergence set is unchanged.

2. `subtileTLU1StackReason` -- how wide a strip is.

   `stripBytes = stack * MatrixInstM * MatrixInstK * bpe`, where bpe is the
   operand's own dtype.  This started life with the fp4 literal 0.5 hardcoded,
   which is a 4x under-count for bf16.  At stack 2 / MatrixInstK 32 that
   under-count takes the slot count to zero and rejects the solution outright,
   so restoring the literal does not mis-lay-out MT320x320 -- it stops it being
   generated at all.
"""

import pytest

from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Validators.Subtile import (
    SUBTILE_STACK_SIZES_B16,
    _SUBTILE_STACK_SIZES,
    _subtileStackForTile,
    subtileStackForB16TLU1,
    subtileStackForTLU1,
    subtileTLU1StackReason,
)

MI_M = 16


# The ratio cap's bound, inlined: upstream dropped the two constants when it
# swapped this rule out, and the alternative has to stay executable to be
# compared against.  Exact integers on purpose -- see the coincidence test.
_PAD_NUM, _PAD_DEN = 4, 3


def _ratioCapRule(mtTiles):
    """The rejected alternative to the strip bound, kept executable.

    Rounds up while the padding stays within _PAD_NUM/_PAD_DEN of the tile,
    with no requirement that the result hold the tile in a single strip.
    """
    exact = next((s for s in _SUBTILE_STACK_SIZES if mtTiles % s == 0), 2)
    if mtTiles <= 1:
        return exact
    roundedUp = min(max(_SUBTILE_STACK_SIZES), 1 << (mtTiles - 1).bit_length())
    withinPadCap = roundedUp * _PAD_DEN <= mtTiles * _PAD_NUM
    return roundedUp if (roundedUp > exact and withinPadCap) else exact


# (mtTiles, stack under the ratio cap, stack under the fits-in-one-strip rule).
# Every entry is a case where the two rules disagree; mtTiles * 16 is the macro
# tile at MatrixInstM 16.  The third column is the one in force.
DIVERGENT = [
    (5, 2, 8),      # MT80
    (9, 2, 16),     # MT144
    (10, 2, 16),    # MT160
    (11, 2, 16),    # MT176
    (17, 16, 2),    # MT272
    (20, 16, 4),    # MT320  <-- the bf16 NN tile, see the test below
    (24, 16, 8),    # MT384
    (28, 16, 4),    # MT448
    (36, 16, 4),    # MT576
    (40, 16, 8),    # MT640
]


@pytest.mark.parametrize("mtTiles,ratioCap,fitsInStrip", DIVERGENT)
def test_strip_bound_is_the_rule_in_force(mtTiles, ratioCap, fitsInStrip):
    assert _subtileStackForTile(mtTiles) == fitsInStrip
    assert _ratioCapRule(mtTiles) == ratioCap
    assert ratioCap != fitsInStrip


def test_mt320_is_one_of_the_divergent_tiles():
    """Called out on its own because it is the tile the bf16 NN MT320x320
    kernel generates.

    20 tiles has no power-of-two divisor above 4, so the two rules land three
    rungs apart: the strip bound in force refuses to round because 16 cannot
    hold 20 tiles in one strip, while the ratio cap would pad 20 -> 32 ->
    capped at 16.
    """
    assert _subtileStackForTile(20) == 4
    assert _ratioCapRule(20) == 16


def test_the_two_rules_agree_on_powers_of_two():
    """Where they agree, so the divergence list above is known to be complete."""
    for mtTiles in (1, 2, 4, 8, 16, 32):
        assert _subtileStackForTile(mtTiles) == _ratioCapRule(mtTiles)


def test_divergence_set_is_exactly_as_recorded():
    """Anything that changes either rule moves this set, which is the signal."""
    diverging = [m for m in range(1, 41)
                 if _subtileStackForTile(m) != _ratioCapRule(m)]
    assert diverging == [5, 9, 10, 11,
                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                         31, 33, 34, 35, 36, 37, 38, 39, 40]
    assert set(m for m, _, _ in DIVERGENT) <= set(diverging)


def test_the_rules_coincide_at_the_ratio_equality_points():
    """3, 6 and 12 sit exactly on the ratio cap, and both rules admit them.

    The cap admits them by equality rather than by margin, which is why they
    are absent from the divergence set above and why the alternative has to
    stay an exact integer ratio: a float cap would decide them on binary
    rounding and the set would move for a reason that is not a policy change.
    """
    for mtTiles in (3, 6, 12):
        stack = _subtileStackForTile(mtTiles)
        assert stack == _ratioCapRule(mtTiles)
        assert stack * _PAD_DEN == mtTiles * _PAD_NUM


# --- strip width reads the operand's bpe -------------------------------------


def _state(mtTiles, waveGroup, dtype, instK):
    dt = DataType(dtype)
    return {
        "ProblemType": {"DataTypeA": dt, "DataTypeB": dt},
        "ISA": (9, 5, 0),
        "MIWaveGroup": list(waveGroup),
        "MIWaveTile": [mtTiles // waveGroup[0], mtTiles // waveGroup[1]],
        "MacroTile0": mtTiles * MI_M,
        "MacroTile1": mtTiles * MI_M,
        "MatrixInstM": MI_M,
        "MatrixInstK": instK,
        "WavefrontSize": 64,
        "DepthU": 256,
    }


# Shapes where a 4x strip-width difference decides accept vs reject. MatrixInstK
# is held at 32 for both dtypes so that bpe is the only thing varying -- at each
# dtype's natural instK (128 for fp4, 32 for bf16) the two cancel exactly and
# the difference is invisible.
BPE_SENSITIVE = [
    (16, (4, 1)),
    (16, (2, 2)),
    (16, (1, 4)),
    (8, (4, 1)),
    (32, (4, 1)),
]


@pytest.mark.parametrize("mtTiles,waveGroup", BPE_SENSITIVE)
def test_strip_width_scales_with_the_operand_dtype(mtTiles, waveGroup):
    """A bf16 strip is 4x an fp4 strip at the same stack and MatrixInstK.

    At stack 2 / instK 32 the fp4 strip is 512 B, which is less than one
    wave's 64 x 16 B fetch and leaves zero (block x K window) slots -- so fp4
    is rejected while bf16, at 2048 B and two slots, is not.  If the strip
    width were computed with the fp4 literal, bf16 would take the fp4 answer
    here and every stack-2 bf16 TLU=1 solution would vanish.
    """
    fp4 = subtileTLU1StackReason(_state(mtTiles, waveGroup, "float4", 32),
                                  "A", mtTiles, 2)
    bf16 = subtileTLU1StackReason(_state(mtTiles, waveGroup, "bfloat16", 32),
                                   "A", mtTiles, 2)
    assert fp4 is not None and "slots for a fetch group" in fp4
    assert bf16 is None, "bf16 has 4x the strip and must not inherit the fp4 verdict"


# --- both callers, pinned together -------------------------------------------

# The two entry points share subtileTLU1StackReason but nothing else, and the
# correctness of that sharing rests on the shared helper being dtype-general.
# So the fp4 answer and the bf16 answer are pinned in ONE table, at each dtype's
# natural MatrixInstK (128 for fp4, 32 for bf16) -- which is what a real
# solution carries.
#
# What this table does and does not guard: it pins the selection LADDERS -- the
# pad rule above and the bf16 ceiling -- from both sides, so a change that moves
# one dtype's ladder while fixing the other's fails here.  It is NOT sensitive
# to the bpe read; nothing could be, on the fp4 side.  See
# test_generalized_bpe_read_is_inert_for_fp4 for why that is a proof rather
# than a gap.
#
# Read the None entries carefully: they are NOT places where fp4 can lay a shape
# out and bf16 cannot.  On both such rows below, *neither* dtype has a layout --
# every rung is rejected.  They differ only in what they return when that
# happens, which is a deliberate contract difference between the two entry
# points, pinned by test_total_failure_contracts_differ.
#
# (mtTiles, MIWaveGroup, fp4 stack, bf16 stack).  A bf16 None means nothing laid
# out; the fp4 column on those rows is a fallback height, not a layout.
BOTH_DTYPES = [
    (4,  (1, 1),  4,  4),      # MT64  -- agree
    (8,  (4, 1),  8,  4),      # MT128 -- fp4 goes taller; bf16 caps at 4
    (10, (2, 2), 16,  None),   # MT160 at [2,2] -- 5 tiles/wave, neither lays
                               #   out; fp4's 16 is the fallback, not a layout
                               #   (10 rounds up to 16, which covers it)
    (16, (4, 1), 16,  4),      # MT256 -- the widest divergence, a full ladder
    (20, (4, 1),  4,  None),   # MT320 at [4,1] -- both straddle; 4 is again
                               #   fp4's fallback (20 rounds up past the 16-tile
                               #   line, so the exact divisor 4 stands instead).
                               #   See the contract test below
    (20, (2, 2),  2,  2),      # MT320 at [2,2] -- the bf16 NN shape, agreeing
    (24, (1, 1),  8,  4),      # MT384
    (24, (4, 1),  2,  2),      # MT384 -- agree, by backing off on both sides
    (32, (2, 2), 16,  4),      # MT512
]


@pytest.mark.parametrize("mtTiles,waveGroup,fp4Stack,b16Stack", BOTH_DTYPES)
def test_fp4_stack_ladder(mtTiles, waveGroup, fp4Stack, b16Stack):
    """The fp4 side: subtileStackForTLU1 keeps the stack it picks today.

    Asserting the returned stack rather than the shape of the return value is
    the point -- a test that only checks "None or a string" passes under every
    rewrite of the rule, including the ones this file exists to catch.
    """
    state = _state(mtTiles, waveGroup, "float4", 128)
    assert subtileStackForTLU1(state, "A", mtTiles) == fp4Stack


def test_generalized_bpe_read_is_inert_for_fp4():
    """Why no fp4 test above can be sensitive to the bpe read -- and why that
    is a proof rather than a hole.

    The literal this rule started with was 0.5, and 0.5 is exactly what the
    generalized read returns for fp4.  So for fp4 the generalization is not
    "tested and found harmless", it is arithmetically the same expression; the
    substitution cannot move an fp4 answer for any shape, DepthU or wave group.
    Asserting the constant keeps that argument standing: if fp4's numBytes ever
    stops being 0.5, every fp4 expectation in this file is suspect at once.
    """
    assert DataType("float4").numBytes() == 0.5
    # And the reason bf16 is a different story: 4x the bytes, same expression.
    assert DataType("bfloat16").numBytes() == 4 * DataType("float4").numBytes()
    assert DataType("half").numBytes() == DataType("bfloat16").numBytes()


@pytest.mark.parametrize("mtTiles,waveGroup,fp4Stack,b16Stack", BOTH_DTYPES)
def test_bf16_stack_ladder(mtTiles, waveGroup, fp4Stack, b16Stack):
    """The bf16 side: subtileStackForB16TLU1, over the same shapes as the fp4
    ladder above, so the two answers stay directly comparable."""
    state = _state(mtTiles, waveGroup, "bfloat16", 32)
    assert subtileStackForB16TLU1(state, "A", mtTiles) == b16Stack


@pytest.mark.parametrize("mtTiles,waveGroup,fp4Stack,b16Stack", BOTH_DTYPES)
def test_total_failure_contracts_differ(mtTiles, waveGroup, fp4Stack, b16Stack):
    """What the two entry points do when NO rung lays out -- the real asymmetry.

    It would be easy to read the None entries in BOTH_DTYPES as "fp4 supports a
    shape bf16 does not".  It is not that.  On every row here the two agree on
    which rungs are layoutable, because they consult the same
    subtileTLU1StackReason; they disagree only on the fallback:

      subtileStackForTLU1    exhausts its ladder and returns the *preferred*
                              height anyway, leaving the rejection to a later
                              validator.
      subtileStackForB16TLU1 returns None, and its caller rejects with the
                              preferred height's reason.

    A resolution that gives bf16 the fp4 fallback would turn a clean rejection
    into a kernel emitted against a stack that was already refused, so the
    difference is asserted rather than left as a property of two table rows.
    """
    fp4State = _state(mtTiles, waveGroup, "float4", 128)
    b16State = _state(mtTiles, waveGroup, "bfloat16", 32)

    # Asserted against the live return, not against b16Stack: the point is the
    # behaviour of the two functions, and reading the expected value back out
    # of the table would make this pass under the very swap it exists to catch.
    b16Got = subtileStackForB16TLU1(b16State, "A", mtTiles)
    fp4Got = subtileStackForTLU1(fp4State, "A", mtTiles)
    b16CanLayOut = any(
        subtileTLU1StackReason(b16State, "A", mtTiles, s) is None
        for s in SUBTILE_STACK_SIZES_B16
    )

    # fp4 always answers -- that is its half of the contract, and it is what
    # makes the bf16 None meaningful rather than incidental.
    assert fp4Got is not None
    if b16CanLayOut:
        assert b16Got is not None
        assert subtileTLU1StackReason(b16State, "A", mtTiles, b16Got) is None
    else:
        assert b16Got is None, (
            f"mtTiles={mtTiles} wg={waveGroup}: no bf16 stack lays out, so the "
            f"caller must get None to reject on, not {b16Got}"
        )
        # ...and on exactly these shapes fp4 is falling back, not laying out:
        # its answer is the preferred height, still carrying a reason.
        assert fp4Got == _subtileStackForTile(mtTiles)
        assert subtileTLU1StackReason(fp4State, "A", mtTiles, fp4Got) is not None


def test_the_two_dtypes_really_do_disagree():
    """Guard the table itself: if these columns ever coincide, the tests above
    stop distinguishing a dtype-general rule from a dtype-blind one."""
    disagree = [row for row in BOTH_DTYPES if row[2] != row[3]]
    assert len(disagree) >= 5
    # bf16 can never exceed its own ceiling, whatever fp4 does.
    assert all(b16 is None or b16 <= max(SUBTILE_STACK_SIZES_B16)
               for _, _, _, b16 in BOTH_DTYPES)


def test_reason_text_is_no_longer_dtype_specific():
    """The messages lost their 'fp4' wording when the rule stopped being fp4-only.

    A reason naming fp4 while rejecting a bf16 operand is what this catches.

    Swept rather than sampled, and deliberately so: most single (tile, wave
    group, stack) triples lay out cleanly and return None, and `reason is None
    or "fp4" not in reason` asserts precisely nothing on a None.  Picking one
    shape by hand is how this test spent its first draft passing without ever
    reading a rejection string.  The floor below is what makes it a test.
    """
    seen = 0
    for mtTiles in range(2, 41):
        for waveGroup in ((1, 1), (4, 1), (2, 2), (1, 4)):
            state = _state(mtTiles, waveGroup, "bfloat16", 32)
            for stack in _SUBTILE_STACK_SIZES:
                reason = subtileTLU1StackReason(state, "A", mtTiles, stack)
                if reason is None:
                    continue
                seen += 1
                assert "fp4" not in reason, (
                    f"bf16 operand rejected with fp4 wording at mtTiles={mtTiles}, "
                    f"MIWaveGroup={waveGroup}, stack={stack}: {reason}")
    assert seen > 400, (
        f"only {seen} rejection strings reached -- the sweep stopped exercising "
        "the reason text, so this test no longer checks anything")
