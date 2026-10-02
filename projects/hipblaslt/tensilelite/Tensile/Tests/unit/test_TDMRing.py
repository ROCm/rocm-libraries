################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the deep TDM LDS ring predicate and the per-wave TDM in-flight bound."""
import pytest

from Tensile.Components.TDMRing import (
    TDM_INFLIGHT_PER_WAVE_LIMIT,
    tdmDeepRing,
    tdmInflightPerWaveBound,
    tdmInflightRejectReason,
    tdmWaveIssueMembers,
)

pytestmark = pytest.mark.unit

_MX = {"MXBlockA": 32, "MXBlockB": 32, "Sparse": 0}
_DENSE = {"MXBlockA": 0, "MXBlockB": 0, "Sparse": 0}


def _tdmState(**overrides):
    """Four-wave MX PGR2 TDM state with the keys the bound reads already settled."""
    state = {
        "TDMInst": 3,
        "enableTDMA": True,
        "enableTDMB": True,
        "enableTDMMetadata": False,
        "NumWaves": 4,
        "TDMFuse": 0,
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "ProblemType": dict(_MX),
        "PrefetchGlobalRead": 2,
        "NumLdsBlk": 2,
        "HalfPLR": 0,
        "SuppressNoLoadLoop": False,
        "NoTailLoop": False,
        "PrefetchAcrossPersistent": 0,
        "ReuseAcrossPersistent": 0,
    }
    state.update(overrides)
    return state


@pytest.mark.parametrize(
    "overrides, issues",
    [
        pytest.param({}, (("A", "MXSA"), ("B", "MXSB")) * 2, id="fuse0"),
        pytest.param({"TDMFuse": 1}, (("A", "MXSB"), ("MXSA", "B")) * 2, id="fuse1"),
        pytest.param({"TDMFuse": 2},
                     (("A", "B"), ("A", "B"), ("MXSA", "B"), ("MXSB", "B")), id="fuse2"),
        pytest.param({"TDMFuse": 3},
                     (("B", "A"), ("B", "A"), ("MXSA", "A"), ("MXSB", "A")), id="fuse3"),
        pytest.param({"ProblemType": dict(_DENSE)}, (("A",), ("B",)) * 2, id="dense"),
        pytest.param({"ProblemType": {**_DENSE, "Sparse": 1}, "enableTDMMetadata": True},
                     (("A", "Metadata"), ("B",)) * 2, id="sparse"),
        pytest.param({"TDMSplit": True}, (("A", "A", "MXSA"), ("B", "B", "MXSB")) * 2,
                     id="split"),
        pytest.param({"NumWaves": 1}, (("A", "MXSA", "MXSB", "B"),), id="one-wave"),
        pytest.param({"NumWaves": 1, "TDMSplit": True},
                     (("A", "A", "MXSA", "MXSB", "B", "B"),), id="one-wave-split"),
    ],
)
def test_wave_issue_members(overrides, issues):
    assert tdmWaveIssueMembers(_tdmState(**overrides)) == issues


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({}, 4, id="mx-fuse0"),
        pytest.param({"TDMFuse": 1}, 4, id="mx-fuse1"),
        pytest.param({"TDMFuse": 2}, 4, id="mx-fuse2"),
        pytest.param({"TDMFuse": 3}, 4, id="mx-fuse3"),
        pytest.param({"ProblemType": dict(_DENSE)}, 2, id="dense"),
        pytest.param({"ProblemType": {**_DENSE, "Sparse": 1}, "enableTDMMetadata": True}, 4,
                     id="sparse"),
        pytest.param({"HalfPLR": 1}, 5, id="mx-halfplr"),
        pytest.param({"PrefetchAcrossPersistent": 1}, 6, id="mx-pap"),
        pytest.param({"HalfPLR": 1, "PrefetchAcrossPersistent": 1}, 8, id="mx-halfplr-pap"),
        pytest.param({"PrefetchGlobalRead": 1, "NumLdsBlk": 1}, 2, id="one-lds-buffer"),
        pytest.param({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 1,
                      "PrefetchGlobalReadB": 2}, 4, id="decoupled-1-2-fuse0"),
        pytest.param({"PrefetchGlobalRead": 1, "PrefetchGlobalReadA": 1,
                      "PrefetchGlobalReadB": 2, "TDMFuse": 1}, 3, id="decoupled-1-2-fuse1"),
        pytest.param({"NumWaves": 1, "ProblemType": dict(_DENSE)}, 4, id="one-wave-dense"),
        pytest.param({"NumWaves": 1}, 8, id="one-wave-mx"),
        pytest.param({"NumWaves": 1, "HalfPLR": 1}, 11, id="one-wave-mx-halfplr"),
    ],
)
def test_bound_for_todays_configs(overrides, bound):
    state = _tdmState(**overrides)
    assert tdmInflightPerWaveBound(state) == bound
    assert tdmInflightRejectReason(state) is None


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, 6, id="ring3-mx"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4}, 8, id="ring4-mx"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "TDMFuse": 1}, 8,
                     id="ring4-mx-fuse1"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "ProblemType": dict(_DENSE)}, 4,
                     id="ring4-dense"),
        pytest.param({"PrefetchGlobalRead": 2, "PrefetchGlobalReadA": 2,
                      "PrefetchGlobalReadB": 4}, 8, id="ring-2-4-mx-fuse0"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "PrefetchAcrossPersistent": 1},
                     10, id="ring4-mx-pap"),
    ],
)
def test_deep_ring_within_limit(overrides, bound):
    state = _tdmState(**overrides)
    assert tdmInflightPerWaveBound(state) == bound
    assert tdmInflightRejectReason(state) is None


@pytest.mark.parametrize(
    "overrides, bound",
    [
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "HalfPLR": 1,
                      "PrefetchAcrossPersistent": 1}, 16, id="ring4-mx-halfplr-pap"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "NumWaves": 1}, 16,
                     id="ring4-one-wave-mx"),
        pytest.param({"TDMSplit": True, "NumWaves": 1}, 12, id="split-one-wave-mx"),
        pytest.param({"TDMSplit": True, "HalfPLR": 1, "PrefetchAcrossPersistent": 1}, 12,
                     id="split-mx-halfplr-pap"),
    ],
)
def test_bound_over_limit_is_rejected(overrides, bound):
    state = _tdmState(**overrides)
    assert tdmInflightPerWaveBound(state) == bound
    assert bound > TDM_INFLIGHT_PER_WAVE_LIMIT
    reason = tdmInflightRejectReason(state)
    assert reason is not None
    assert "up to %u TDM operations" % bound in reason
    assert "limit of %u" % TDM_INFLIGHT_PER_WAVE_LIMIT in reason


def test_tdm_plus_lds_buf_keeps_two_stages_in_flight():
    """Three LDS blocks, but PGR2 prefetches two stages, so the third block adds nothing."""
    state = _tdmState(TDMPlusLdsBuf=1, NumLdsBlk=3, NumWaves=1)
    assert tdmInflightPerWaveBound(state) == 8
    assert tdmInflightRejectReason(state) is None


def test_two_block_unsplit_path_is_not_rejected():
    """Over the limit by the bound, but on the two-block unsplit path existing solutions use."""
    state = _tdmState(NumWaves=1, HalfPLR=1, enableTDMMetadata=True,
                      ProblemType={**_MX, "Sparse": 1})
    assert tdmInflightPerWaveBound(state) == 14
    assert tdmInflightRejectReason(state) is None


def test_no_tail_loop_drops_the_tail_term():
    state = _tdmState(NumWaves=1, HalfPLR=1, NoTailLoop=True)
    assert tdmInflightPerWaveBound(state) == 8


def test_bound_is_zero_without_tdm():
    state = _tdmState(TDMInst=0, enableTDMA=False, enableTDMB=False)
    assert tdmInflightPerWaveBound(state) == 0
    assert tdmInflightRejectReason(state) is None


@pytest.mark.parametrize(
    "overrides, deep",
    [
        pytest.param({}, False, id="pgr2"),
        pytest.param({"PrefetchGlobalRead": 1, "NumLdsBlk": 1}, False, id="pgr1"),
        pytest.param({"TDMPlusLdsBuf": 1, "NumLdsBlk": 3}, False, id="tdmpluslds"),
        pytest.param({"PrefetchGlobalRead": 3, "NumLdsBlk": 3}, True, id="ring3"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4}, True, id="ring4"),
        pytest.param({"PrefetchGlobalRead": 4, "NumLdsBlk": 4, "enableTDMB": False}, False,
                     id="b-not-tdm"),
    ],
)
def test_deep_ring_is_independent_of_tdm_plus_lds_buf(overrides, deep):
    assert tdmDeepRing(_tdmState(**overrides)) is deep
