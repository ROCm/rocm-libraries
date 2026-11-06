# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Characterization tests for ``Components.TDMSplit.wave_region_span`` and the
``seg_interleave_row_shape`` it leans on.

``wave_region_span`` answers how many TDMSplit regions ONE WAVE's tiles reach,
which is not the split factor: the wave's tiles are distributed by
``VectorWidth`` and ``MIWaveGroup``, so whether its share crosses the region cut
depends on the arithmetic, not on the cut count. Every input it reads gets a
case here, for A and for B, because the two differ only in which index of
``MatrixInstruction`` / ``MIWaveGroup`` / ``MIWaveTile`` they read and a
transposed pair is the cheapest way to pin that.

The reference the cases are written against, from the function's own docstring:

    row(t)    = (t // VW) * (MI * MIBlock * MIWaveGroup * VW) + (t % VW)
    region(t) = row(t) // (MacroTile // nsplit)
    span      = |{ region(t) : t < MIWaveTile }|
"""

import pytest

from tensilelite.Components.TDMSplit import (
    TDM_SPLIT_AXIS,
    seg_interleave_row_shape,
    wave_region_span,
)


class _DataType:
    """Only ``numRegisters`` is consulted, by the VectorWidth AUTO replay."""

    def __init__(self, registers):
        self._registers = registers

    def numRegisters(self):
        return self._registers


def _kernel(*, split=(1, 1), mi=(16, 16, 128, 1, 1, 8, 10, 2, 2), vw=(2, 2),
            waveGroup=None, waveTile=None, lddsi=0, offsets=None, numWaves=4,
            macDataType=None):
    """A theta-shaped dict carrying exactly what ``wave_region_span`` reads."""
    kernel = {
        "TDMSplitA": split[0],
        "TDMSplitB": split[1],
        "MatrixInstruction": list(mi),
        "LDSSegmentInterleave": lddsi,
        "NumWaves": numWaves,
    }
    if vw[0] is not None:
        kernel["VectorWidthA"] = vw[0]
    if vw[1] is not None:
        kernel["VectorWidthB"] = vw[1]
    if waveGroup is not None:
        kernel["MIWaveGroup"] = list(waveGroup)
    if waveTile is not None:
        kernel["MIWaveTile"] = list(waveTile)
    if offsets is not None:
        kernel["LDSSegInterleaveOffsets"] = offsets
    if macDataType is not None:
        kernel["ProblemType"] = {"MacDataTypeA": macDataType, "MacDataTypeB": macDataType}
    return kernel


def _reference(wt, vw, miDim, miBlk, wg, nsplit, segShape=0):
    """The docstring's formula, independent of the implementation.

    Under the interleave the region extent shortens with the stride, so both terms
    carry `numVectorsPerTile` -- shortening only the stride is the bug the arm fixed.
    """
    group = miDim * max(1, miBlk) * wg * vw
    macroTile = wt * wg * miDim * max(1, miBlk)
    if segShape:
        rowsPerRegion = max(1, macroTile // nsplit // max(1, wt // vw))
    else:
        rowsPerRegion = max(1, macroTile // nsplit)
    rows = {((t // vw) * (segShape or group) + (t % vw)) // rowsPerRegion for t in range(wt)}
    return max(1, min(nsplit, len(rows)))


# --- the split factor gate ------------------------------------------------


def test_split_axis_table_is_the_three_tdmsplit_values():
    """0 unsplit, 1 cuts the macro tile, 2 cuts DepthU."""
    assert TDM_SPLIT_AXIS == {0: (1, 1), 1: (2, 1), 2: (1, 2)}


@pytest.mark.parametrize("tc", ["A", "B"])
@pytest.mark.parametrize("value, why", [(0, "unsplit"), (2, "a DU split")])
def test_span_is_one_when_the_macro_tile_is_not_cut(tc, value, why):
    """A DU split puts every region in front of every wave, so the span is 1."""
    split = (value, 0) if tc == "A" else (0, value)
    assert wave_region_span(_kernel(split=split), tc) == 1


@pytest.mark.parametrize("tc", ["A", "B"])
def test_span_is_one_when_the_operand_is_absent_from_the_kernel(tc):
    """A missing TDMSplit key reads as 0, not as a crash."""
    assert wave_region_span({"MatrixInstruction": [16, 16, 128, 1, 1, 8, 8, 1, 1]}, tc) == 1


# --- the defect this function exists for ----------------------------------


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_wave_group_one_lines_the_groups_up_with_the_region_cut(tc, t01):
    """At MIWaveGroup[1,1] the index form `t // tiles_per_region` happens to be right."""
    split = (1, 0) if tc == "A" else (0, 1)
    kernel = _kernel(split=split, mi=(16, 16, 128, 1, 1, 8, 8, 1, 1), vw=(2, 2))
    wt, vw, miDim, wg = 8, 2, 16, 1
    assert wave_region_span(kernel, tc) == _reference(wt, vw, miDim, 1, wg, 2)


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_wave_group_two_puts_a_whole_region_in_the_group_jump(tc, t01):
    """The jump is a whole region, so a wave's tiles sit adjacent and share region 0."""
    split = (1, 0) if tc == "A" else (0, 1)
    kernel = _kernel(split=split, mi=(16, 16, 128, 1, 1, 8, 8, 2, 2), vw=(2, 2))
    assert wave_region_span(kernel, tc) == _reference(8, 2, 16, 1, 2, 2)


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_an_odd_region_tile_count_makes_a_vector_pair_cross_the_cut(tc, t01):
    """MIWaveTile 10 over 2 regions is 5 tiles each; VW 2 pairs {4,5} across the cut."""
    split = (1, 0) if tc == "A" else (0, 1)
    mi = (16, 16, 128, 1, 1, 10, 10, 2, 2)
    kernel = _kernel(split=split, mi=mi, vw=(2, 2))
    assert wave_region_span(kernel, tc) == _reference(10, 2, 16, 1, 2, 2)


@pytest.mark.parametrize("tc", ["A", "B"])
@pytest.mark.parametrize("vw", [1, 2, 4])
def test_span_tracks_the_vector_width_for_both_operands(tc, vw):
    split = (1, 0) if tc == "A" else (0, 1)
    vws = (vw, 1) if tc == "A" else (1, vw)
    kernel = _kernel(split=split, mi=(16, 16, 128, 1, 1, 8, 8, 2, 2), vw=vws)
    assert wave_region_span(kernel, tc) == _reference(8, vw, 16, 1, 2, 2)


@pytest.mark.parametrize("tc", ["A", "B"])
def test_span_never_exceeds_the_split_factor(tc):
    """`min(mt, ...)`: a wave cannot reach more regions than the tile was cut into."""
    split = (1, 0) if tc == "A" else (0, 1)
    kernel = _kernel(split=split, mi=(16, 16, 128, 1, 1, 16, 16, 1, 1), vw=(1, 1))
    assert wave_region_span(kernel, tc) <= 2


# --- the explicit MIWaveGroup / MIWaveTile overrides ----------------------


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_explicit_wave_group_and_tile_override_the_matrix_instruction(tc, t01):
    """A hand-built theta carries MIWaveGroup/MIWaveTile instead of mi[5:9]."""
    split = (1, 0) if tc == "A" else (0, 1)
    mi = (16, 16, 128, 1, 1, 99, 99, 99, 99)      # mi[5:9] must be ignored
    kernel = _kernel(split=split, mi=mi, vw=(2, 2), waveGroup=(2, 2), waveTile=(8, 8))
    assert wave_region_span(kernel, tc) == _reference(8, 2, 16, 1, 2, 2)


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_the_matrix_instruction_block_count_widens_the_group(tc, t01):
    """mi[3+t01] is the block count; it multiplies both the group and the macro tile."""
    split = (1, 0) if tc == "A" else (0, 1)
    mi = (16, 16, 128, 2, 2, 8, 8, 2, 2)
    kernel = _kernel(split=split, mi=mi, vw=(2, 2))
    assert wave_region_span(kernel, tc) == _reference(8, 2, 16, 2, 2, 2)


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_each_operand_reads_its_own_index_of_every_pair(tc, t01):
    """A transposed pair: A and B must disagree exactly as their indices do."""
    mi = (16, 32, 128, 1, 1, 8, 4, 1, 2)
    kernel = _kernel(split=(1, 1), mi=mi, vw=(2, 2))
    expected = _reference(mi[5 + t01], 2, mi[t01], 1, mi[7 + t01], 2)
    assert wave_region_span(kernel, tc) == expected


# --- VectorWidth AUTO -----------------------------------------------------


@pytest.mark.parametrize("tc", ["A", "B"])
def test_auto_vector_width_falls_back_to_the_split_factor_without_a_data_type(tc):
    """Nothing to replay the rule from, so the span is the whole split."""
    split = (1, 0) if tc == "A" else (0, 1)
    kernel = _kernel(split=split, mi=(16, 16, 128, 1, 1, 8, 8, 2, 2), vw=(-1, -1))
    assert wave_region_span(kernel, tc) == 2


@pytest.mark.parametrize("tc", ["A", "B"])
@pytest.mark.parametrize("registers, vw", [(0.25, 8), (0.5, 8), (1.0, 4)])
def test_auto_vector_width_replays_the_registers_per_element_rule(tc, registers, vw):
    """`4 // numRegisters`, halved until it divides MIWaveTile.

    At 0.25 the quotient is 16, which MIWaveTile 8 rejects, so it settles on 8 --
    the same place 0.5 starts.
    """
    split = (1, 0) if tc == "A" else (0, 1)
    kernel = _kernel(split=split, mi=(16, 16, 128, 1, 1, 8, 8, 2, 2), vw=(-1, -1),
                     macDataType=_DataType(registers))
    assert wave_region_span(kernel, tc) == _reference(8, vw, 16, 1, 2, 2)


@pytest.mark.parametrize("tc", ["A", "B"])
def test_auto_vector_width_halves_until_it_divides_the_wave_tile(tc):
    """MIWaveTile 6 rejects 4, so the replay settles on 2."""
    split = (1, 0) if tc == "A" else (0, 1)
    kernel = _kernel(split=split, mi=(16, 16, 128, 1, 1, 6, 6, 2, 2), vw=(-1, -1),
                     macDataType=_DataType(1.0))
    assert wave_region_span(kernel, tc) == _reference(6, 2, 16, 1, 2, 2)


# --- seg_interleave_row_shape --------------------------------------------


@pytest.mark.parametrize("tc", ["A", "B"])
def test_row_shape_is_baseline_without_the_interleave(tc):
    assert seg_interleave_row_shape(_kernel(waveTile=(8, 8), waveGroup=(2, 2)),
                                    tc, 0 if tc == "A" else 1, 128) == 0


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_row_shape_is_baseline_when_one_vector_covers_the_tile(tc, t01):
    """`MIWaveTile // vw <= 1` leaves nothing for the interleave to stride over."""
    vws = (8, 1) if tc == "A" else (1, 8)
    kernel = _kernel(vw=vws, waveTile=(8, 8), waveGroup=(2, 2), lddsi=1,
                     offsets={"portSplitA": True, "portSplitB": True})
    assert seg_interleave_row_shape(kernel, tc, t01, 128) == 0


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_row_shape_is_baseline_for_an_auto_vector_width(tc, t01):
    vws = (-1, -1)
    kernel = _kernel(vw=vws, waveTile=(8, 8), waveGroup=(2, 2), lddsi=1,
                     offsets={"portSplitA": True, "portSplitB": True})
    assert seg_interleave_row_shape(kernel, tc, t01, 128) == 0


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_port_split_divides_the_row_by_the_wave_group(tc, t01):
    kernel = _kernel(vw=(2, 2), waveTile=(8, 8), waveGroup=(2, 2), lddsi=1,
                     offsets={"portSplit%s" % tc: True})
    assert seg_interleave_row_shape(kernel, tc, t01, 128) == 128 // 2


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_port_split_on_the_other_operand_leaves_this_one_baseline(tc, t01):
    other = "B" if tc == "A" else "A"
    kernel = _kernel(vw=(2, 2), waveTile=(8, 8), waveGroup=(2, 2), lddsi=1,
                     offsets={"portSplit%s" % other: True})
    assert seg_interleave_row_shape(kernel, tc, t01, 128) == 0


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_component_split_applies_only_to_the_active_operand(tc, t01):
    kernel = _kernel(vw=(2, 2), waveTile=(8, 8), waveGroup=(2, 2), lddsi=1, numWaves=4,
                     offsets={"componentSplit": True, "activeTC": tc})
    assert seg_interleave_row_shape(kernel, tc, t01, 128) == 128 // max(1, 2 // (4 // 2))
    inactive = _kernel(vw=(2, 2), waveTile=(8, 8), waveGroup=(2, 2), lddsi=1, numWaves=4,
                       offsets={"componentSplit": True, "activeTC": "B" if tc == "A" else "A"})
    assert seg_interleave_row_shape(inactive, tc, t01, 128) == 0


@pytest.mark.parametrize("tc, t01", [("A", 0), ("B", 1)])
def test_the_interleave_row_shape_reaches_wave_region_span(tc, t01):
    """The regression the arm was added for: shortening only the stride made an
    operand carrying portSplit look wave-relative when its read layout was not."""
    split = (1, 0) if tc == "A" else (0, 1)
    mi = (16, 16, 128, 1, 1, 8, 8, 2, 2)
    plain = _kernel(split=split, mi=mi, vw=(4, 4))
    interleaved = _kernel(split=split, mi=mi, vw=(4, 4), lddsi=1,
                          waveTile=(8, 8), waveGroup=(2, 2),
                          offsets={"portSplit%s" % tc: True})
    assert wave_region_span(plain, tc) == _reference(8, 4, 16, 1, 2, 2)
    assert wave_region_span(interleaved, tc) == _reference(8, 4, 16, 1, 2, 2, segShape=64)
