################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the TDMCross arrangement as KernelWriterAssembly emits it.

test_TDMCross.py owns the partition model and states its assertions in terms
of that model, which leaves the gap this file closes. The defect that motivated
it lived entirely in the writer: descriptor initialisation resolved which group
member rides which wave parity through the partition, while the increment
selection resolved the same question from the order its arguments happened to
arrive in. At TDMCross=1 those two answers differ, so the model was right and
the emitted code was wrong -- a test written against tdmWaveAssignment passes
either way and cannot see it.

Every assertion here therefore reads operands off instructions the writer
actually produced. The writer is driven through a stand-in rather than a real
KernelWriterAssembly instance: the functions under test reach only the parity
helpers and the GlobalReadIncs operand accessor, so binding the shipped methods
onto a minimal object runs the real code at unit-test cost.
"""
import re
import types

import pytest

from Tensile.Components.TDMFuse import (
    TDM_CROSS_CROSSED,
    TDM_CROSS_DEFAULT,
    tdmWavePartition,
)
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit


def _ks(fuse=0, cross=TDM_CROSS_DEFAULT, numWaves=4, mxBlockA=32, mxBlockB=32,
        **overrides):
    ks = {
        "TDMFuse": fuse,
        "TDMCross": cross,
        "TDMInst": 3,
        "TDMSplit": False,
        "NumWaves": numWaves,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 2,
        "PrefetchGlobalReadB": 2,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {"MXBlockA": mxBlockA, "MXBlockB": mxBlockB},
    }
    ks.update(overrides)
    return ks


class _Writer:
    """Runs the shipped writer methods against the smallest state they touch."""

    states = types.SimpleNamespace(unrollIdx=0)

    isTdmWaveSeparated = KernelWriterAssembly.isTdmWaveSeparated
    tdmFuseAMx = KernelWriterAssembly.tdmFuseAMx
    tdmFusePaired = KernelWriterAssembly.tdmFusePaired
    globalReadIncsOperand = KernelWriterAssembly.globalReadIncsOperand
    _tdmPairedParityOrder = KernelWriterAssembly._tdmPairedParityOrder
    _tdmSetMembersByParity = KernelWriterAssembly._tdmSetMembersByParity
    tdmSetupIncrementWaveSeparated = \
        KernelWriterAssembly.tdmSetupIncrementWaveSeparated

    def useConstSgprGlobalReadIncsForTc(self, tc):
        # Keep every increment a register, so a select's source names a tensor.
        return False


def _selectSources(kernel, tc1, tc2):
    """(src0, src1) of the select that fills this call's increment register.

    s_cselect_b32 is SCC ? src0 : src1, and the guard ahead of it is
    s_bitcmp1 WaveIdx, 0, which raises SCC on the odd waves. So src0 is the
    odd-wave increment and src1 the even-wave one.
    """
    mod = _Writer().tdmSetupIncrementWaveSeparated(
        kernel, {"tensorChar": tc1}, {"tensorChar": tc2})
    text = mod.toString() if hasattr(mod, "toString") else str(mod)
    match = re.search(
        r"s_cselect_b32 s\[sgprtdm%s%sIncs\], *([^,]+?), *([^,\s]+)" % (tc1, tc2),
        text)
    assert match is not None, "no increment select emitted:\n%s" % text
    return match.group(1).strip(), match.group(2).strip()


def _parityOf(kernel, tc):
    """Which wave parity this tensor is assigned, as "even" or "odd"."""
    waves = tdmWavePartition(kernel, tc)[1]
    parities = {wave % 2 for wave in waves}
    assert len(parities) == 1, "%s spans both parities: %s" % (tc, waves)
    return "even" if parities == {0} else "odd"


def _incs(tc):
    return "s[sgprGlobalReadIncs%s+0]" % tc


@pytest.mark.parametrize("group", (("A", "B"), ("MXSA", "MXSB")))
@pytest.mark.parametrize("cross", (TDM_CROSS_DEFAULT, TDM_CROSS_CROSSED))
def test_increment_select_agrees_with_the_partition(cross, group):
    """The guard: the wave a descriptor is built on must advance by its stride.

    Expressed against the partition rather than against a literal so that it
    keeps its meaning if the arrangement is ever redefined. Before the fix the
    crossed scale call initialised MXSB on the even waves and then advanced
    them by MXSA's increment, which is the whole defect in one line.
    """
    kernel = _ks(fuse=0, cross=cross)
    srcOdd, srcEven = _selectSources(kernel, *group)
    byParity = {_parityOf(kernel, tc): tc for tc in group}
    assert srcEven == _incs(byParity["even"])
    assert srcOdd == _incs(byParity["odd"])


def test_default_arrangement_pins_each_member_to_its_own_increment():
    """The shipped arrangement, pinned as a literal.

    test_TDMCross.py gates the default partition against a frozen copy of the
    pre-refactor formula for the same reason this is here: the silicon evidence
    chain was generated at TDMCross=0 and cannot be regenerated, so the default
    moving is something to notice deliberately rather than discover later.
    """
    assert _selectSources(_ks(fuse=0), "A", "B") == (_incs("B"), _incs("A"))
    assert _selectSources(_ks(fuse=0), "MXSA", "MXSB") == \
        (_incs("MXSB"), _incs("MXSA"))


def test_crossing_swaps_the_scale_sources_and_leaves_the_data_sources():
    """Crossing reverses groups after the first, so only the scales move."""
    default = _ks(fuse=0, cross=TDM_CROSS_DEFAULT)
    crossed = _ks(fuse=0, cross=TDM_CROSS_CROSSED)
    assert _selectSources(crossed, "A", "B") == _selectSources(default, "A", "B")
    assert _selectSources(crossed, "MXSA", "MXSB") == \
        tuple(reversed(_selectSources(default, "MXSA", "MXSB")))
