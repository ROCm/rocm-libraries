# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Characterization tests for a COMPANION's local-read lead under UseLoopModel.

A companion operand -- an MX scale or a sparse problem's metadata -- has no PrefetchLocalRead of
its own; it reads the one belonging to the tensor it rides.  PLR counts prefetch UNITS, and a
companion's unit is the wider one: a merged read lands a whole fold in one buffer, so one of its
units can already span every tile the parent's lead reaches.  Taking the parent's NUMBER then asks
for a read of a unit the loop has not arrived at.

The lead is therefore stated in the companion's own units, and a parent lead that does not fill
one of them is no read-ahead at all:

    lead  ==  (parent unit tiles * parent PLR) // companion unit tiles

pinned below for the three cases that differ -- the parent's lead inside one companion unit, on
its boundary, and past it.  A and B are pinned alongside because the derivation must not reach
them: their lead is the parameter itself.
"""

import pytest

from tensilelite.LoopModel import adapter
from tensilelite.LoopModel import traversal as geometry
from tensilelite.LoopModel.loopir import Axis
from tensilelite.LoopModel.traversal import TransferCoverage

#: MT64x64x256, MI 16x16x128, MXBlock 32 -- the mxf8 triage shape, whose scale read folds 2 tiles.
_KERNEL = {
    "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2], "DepthU": 256, "MatrixInstK": 128,
    "MIWaveTile": [2, 2], "MIWaveTileA": 2, "MIWaveTileB": 2,
    "ProblemType": {"DataType": "F8", "MXBlockA": 32, "MXBlockB": 32},
    "PrefetchGlobalRead": 1, "NumWaves": 4, "WavefrontSize": 32,
    "enableTDMA": 1, "enableTDMB": 1, "NumLdsBlk": 2, "LoopOrder": "KMN",
    "TDMSplitA": 1, "TDMSplitB": 1, "TDMFuse": 0,
    "VectorWidthA": 2, "VectorWidthB": 2, "LocalReadVectorWidthMXS": 8,
    "UnrollMajorLDSMXSA": 1, "UnrollMajorLDSMXSB": 1,
}


def _two_tile_fold():
    """The read coverage the target supplies: one instruction lands tiles `2c` and `2c+1`."""
    tile = Axis("tile", 0)
    return TransferCoverage(carrier=tile // 2, slot=tile % 2)


def _theta(plr):
    """The model for `_KERNEL` at this PrefetchLocalRead, registers derived as a Solution does."""
    from tensilelite.Components.LoopModel import Order, Registers
    kernel = dict(adapter.DEFAULTS)
    kernel.update(_KERNEL)
    kernel["PrefetchLocalRead"] = plr
    kernel["UseLoopModel"] = True
    kernel["LoopIters"] = kernel["DepthU"] // kernel["MatrixInstK"]
    mi, wave = kernel["MatrixInstruction"], kernel["WavefrontSize"]
    for tc, mn in (("A", mi[0]), ("B", mi[1])):
        per_thread = mn * mi[2] * mi[3] // wave
        kernel["MIInputPerThread" + tc] = per_thread
        kernel["MIInputPerThreadMXS" + tc] = (
            per_thread // kernel["ProblemType"]["MXBlock" + tc] * (32 // mn))
    Order.calculate(kernel)
    Registers.calculate(kernel)
    target = {"ReadVectorElems": {"MXSA": 8, "MXSB": 8},
              "ReadCoverage": {"MXSA": _two_tile_fold(), "MXSB": _two_tile_fold()}}
    return adapter.params_to_theta(adapter.kernel_to_params(kernel, target=target))


def _operand(theta, name):
    return next(op for op in theta.operands if op.name == name)


def _lead(theta, name):
    """The lead as the SCHEDULE sees it -- the offset placed on the operand's read hop."""
    return geometry._raw_read_ahead(theta, _operand(theta, name))


@pytest.mark.parametrize("plr,parent_tiles,companion_lead", [(1, 1, 0), (2, 2, 1), (3, 3, 1)])
def test_the_lead_is_the_parents_coverage_in_the_companions_units(plr, parent_tiles,
                                                                  companion_lead):
    """PLR 1 is half an MXSA unit, so MXSA does not lead at all; PLR 2 and 3 are one of them."""
    theta = _theta(plr)
    unit = adapter._unit_free_tiles(theta, _operand(theta, "MXSA"))
    assert adapter._unit_free_tiles(theta, _operand(theta, "A")) * plr == parent_tiles
    assert parent_tiles // unit == companion_lead
    assert _lead(theta, "MXSA") == companion_lead
    assert _lead(theta, "MXSB") == companion_lead


@pytest.mark.parametrize("plr", [1, 2, 3])
def test_a_data_tensors_lead_is_still_the_parameter_itself(plr):
    """The derivation is the companion's; A and B ask for exactly what was requested."""
    theta = _theta(plr)
    assert _lead(theta, "A") == plr
    assert _lead(theta, "B") == plr
    assert _operand(theta, "A").fragment.read_lead == -1


@pytest.mark.parametrize("plr", [1, 2, 3])
def test_the_ring_holds_the_tiles_its_parents_ring_holds(plr):
    """THE ONLY TEST THE DEPTH HAS TO MEET.  No pipeline term is added on top of coverage."""
    theta = _theta(plr)
    for name, side in (("MXSA", "A"), ("MXSB", "B")):
        operand, parent = _operand(theta, name), _operand(theta, side)
        unit = adapter._unit_free_tiles(theta, operand)
        held = max(operand.fragment.ring_depths.values()) * unit
        parent_held = (adapter._unit_free_tiles(theta, parent)
                       * max(parent.fragment.ring_depths.values()))
        assert held >= parent_held, (
            "%s holds %d tiles against %s's %d" % (name, held, side, parent_held))
        assert _lead(theta, name) <= max(operand.fragment.ring_depths.values())
