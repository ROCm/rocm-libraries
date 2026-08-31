# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for planGRCoopSpread, the TLU=1 cooperative global-read fetch planner.

The bulk of this file is an equivalence check against a verbatim copy of the
algorithm as it lived inline in Kernel.py before extraction. Behaviour is pinned
by comparison rather than by hand-written expectations, so the move can be shown
to change nothing across the input space.

Keep `_originalInline` frozen. If the planner is intentionally changed later,
this test is expected to fail; record which cases moved and why rather than
editing the reference to match.
"""

import itertools

import pytest

from Tensile.Components.Subtile.SubtileGeometry import GRCoopSpread, planGRCoopSpread


def _originalInline(wavesPerStrip, otherWaves, stripBytes, numWindows, bytesPerLoad):
    """Verbatim pre-extraction algorithm (Kernel.py, gfx950-transposed). Frozen."""
    coopWaves = wavesPerStrip
    winSplit = 1
    if otherWaves > 1:
        cand = wavesPerStrip * otherWaves
        if wavesPerStrip > 1:
            if cand > coopWaves and stripBytes % bytesPerLoad(cand) == 0:
                coopWaves = cand
        else:
            best, bestKey = (coopWaves, 1), (coopWaves, coopWaves)
            c = 1
            while c <= cand:
                if stripBytes % bytesPerLoad(c) == 0:
                    w = 1
                    while w <= numWindows and c * w <= cand:
                        if numWindows % w == 0 and (c * w, c) > bestKey:
                            bestKey, best = (c * w, c), (c, w)
                        w *= 2
                c *= 2
            coopWaves, winSplit = best
    return coopWaves, winSplit


def _loader(unitBytes):
    """bytesPerLoad is linear in the wave count: B * n (SubtileGeometry:331)."""
    return lambda n: unitBytes * n


# Ranges cover the shapes the subtile path produces. Instrumenting real kernel
# generation for the gfx950 TLU=1 suite shows every input is a power of two --
# strips 2-16 loads wide, 2 or 4 windows, 1-4 waves per strip. The odd values
# here are deliberately outside that: they are the cases where the doubling
# search silently declines to spread, and they are worth pinning in case a future
# tile shape reaches them.
_WAVES_PER_STRIP = [1, 2, 4, 8]
_OTHER_WAVES = [1, 2, 4, 8]
_UNIT_BYTES = [256, 512, 1024]
_STRIP_MULT = [1, 2, 3, 4, 6, 8, 12, 16]
_NUM_WINDOWS = [1, 2, 3, 4, 6, 8]


def _cases():
    for wps, ow, ub, mult, win in itertools.product(
        _WAVES_PER_STRIP, _OTHER_WAVES, _UNIT_BYTES, _STRIP_MULT, _NUM_WINDOWS
    ):
        yield wps, ow, ub * mult, win, ub


def test_matches_pre_extraction_behaviour():
    """Extraction must not change any output across the covered input space."""
    mismatches = []
    for wps, ow, stripBytes, win, ub in _cases():
        want = _originalInline(wps, ow, stripBytes, win, _loader(ub))
        got = planGRCoopSpread(wps, ow, stripBytes, win, _loader(ub))
        if tuple(got) != want:
            mismatches.append(
                f"wavesPerStrip={wps} otherWaves={ow} stripBytes={stripBytes} "
                f"numWindows={win} unit={ub}: want {want}, got {tuple(got)}"
            )
    assert not mismatches, "planGRCoopSpread diverged from the inline original:\n" + "\n".join(
        mismatches[:20]
    )


def test_returns_named_fields():
    spread = planGRCoopSpread(1, 4, 4096, 4, _loader(512))
    assert isinstance(spread, GRCoopSpread)
    assert spread.coopWaves >= 1
    assert spread.windowSplit >= 1


def test_no_other_axis_waves_means_no_spreading():
    """With nothing on the other axis there is no refetch to eliminate."""
    spread = planGRCoopSpread(4, 1, 8192, 4, _loader(512))
    assert spread == GRCoopSpread(4, 1)


def test_falls_back_when_no_wider_group_divides():
    """No usable widening must degrade to no spreading, not to a bad split.

    A one-load strip cannot be split further, so the group stays at
    wavesPerStrip however many waves are available to help.
    """
    spread = planGRCoopSpread(1, 8, 512, 1, _loader(512))
    assert spread == GRCoopSpread(1, 1)


def test_spread_never_exceeds_the_group():
    """coopWaves * windowSplit is capped by wavesPerStrip * otherWaves."""
    for wps, ow, stripBytes, win, ub in _cases():
        spread = planGRCoopSpread(wps, ow, stripBytes, win, _loader(ub))
        if ow > 1:
            assert spread.coopWaves * spread.windowSplit <= wps * max(1, ow)


def test_window_split_divides_the_window_count():
    """A split that does not divide the windows would leave a wave short."""
    for wps, ow, stripBytes, win, ub in _cases():
        spread = planGRCoopSpread(wps, ow, stripBytes, win, _loader(ub))
        assert win % spread.windowSplit == 0


def test_coop_waves_is_a_whole_number_of_strip_sharing_waves():
    """grKSplit is coopWaves // wavesPerStrip, so a non-multiple would truncate.

    A count below wavesPerStrip would also leave the strip under-fetched for the
    waves reading it back. Neither is a rejected solution, so the invariant is
    worth stating even though the current search cannot break it.
    """
    for wps, ow, stripBytes, win, ub in _cases():
        spread = planGRCoopSpread(wps, ow, stripBytes, win, _loader(ub))
        assert spread.coopWaves >= wps, (wps, ow, stripBytes, win, ub, spread)
        assert spread.coopWaves % wps == 0, (wps, ow, stripBytes, win, ub, spread)


def test_per_wave_share_divides_the_strip():
    """The whole point of the search: each wave's share must tile the strip."""
    for wps, ow, stripBytes, win, ub in _cases():
        spread = planGRCoopSpread(wps, ow, stripBytes, win, _loader(ub))
        if spread.coopWaves > wps:  # only the widened cases made a claim
            assert stripBytes % _loader(ub)(spread.coopWaves) == 0


# Every distinct input the generator produces, measured by instrumenting kernel
# generation over the gfx950 TLU=1 yaml (subtile_mxfp4_tlu1, the only one that
# reaches this code): 316 calls, 20 combinations, unit always 1024 bytes. The
# TLU=0 suites never call it.
#
# This is here so a shape landing outside the range is noticed as new rather than
# assumed covered. It is not a supported-values list: the function handles more
# than this, it just has never been asked to.
_MEASURED_DOMAIN = [
    (1, 1, 2, 2), (1, 1, 4, 2), (1, 1, 8, 2),
    (1, 2, 2, 2), (1, 2, 16, 2),
    (1, 4, 2, 2), (1, 4, 2, 4), (1, 4, 4, 2), (1, 4, 16, 2),
    (2, 1, 4, 2), (2, 1, 8, 2), (2, 1, 16, 2),
    (2, 2, 4, 2), (2, 2, 4, 4), (2, 2, 8, 2), (2, 2, 8, 4), (2, 2, 16, 2),
    (4, 1, 8, 2), (4, 1, 8, 4), (4, 1, 16, 2),
]
_MEASURED_UNIT = 1024


@pytest.mark.parametrize("wps,ow,stripLoads,win", _MEASURED_DOMAIN)
def test_every_generated_config_is_planned_without_guard_trips(wps, ow, stripLoads, win):
    """No shape the generator actually produces may hit a guard."""
    spread = planGRCoopSpread(wps, ow, stripLoads * _MEASURED_UNIT, win,
                              _loader(_MEASURED_UNIT))
    assert spread.coopWaves >= wps and spread.coopWaves % wps == 0
    assert win % spread.windowSplit == 0
    assert (stripLoads * _MEASURED_UNIT) % _loader(_MEASURED_UNIT)(spread.coopWaves) == 0


@pytest.mark.parametrize(
    "wps,ow,stripBytes,win",
    [
        (0, 4, 4096, 2),    # no waves on the strip
        (1, 0, 4096, 2),    # no waves on the other axis
        (1, 4, 0, 2),       # empty strip
        (1, 4, 4096, 0),    # no windows
    ],
)
def test_nonpositive_geometry_is_rejected(wps, ow, stripBytes, win):
    with pytest.raises(ValueError, match="positive geometry"):
        planGRCoopSpread(wps, ow, stripBytes, win, _loader(512))


def test_partial_trailing_load_is_rejected():
    """A strip that is not a whole number of loads has no valid split at all.

    Left unguarded the search finds nothing and quietly returns "no spreading",
    so the refetch this code exists to remove comes back with no indication why.
    """
    with pytest.raises(ValueError, match="whole number"):
        planGRCoopSpread(1, 4, 4096 + 128, 2, _loader(512))


def test_bad_load_size_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        planGRCoopSpread(1, 4, 4096, 2, _loader(0))
