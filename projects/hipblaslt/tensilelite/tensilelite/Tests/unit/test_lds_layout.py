# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Check reported addresses, buffer copies and segment-boundary clipping."""

from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest

from tensilelite.Common.GlobalParameters import globalParameters, _assertGlobalParametersAreValid
from tensilelite.Common.TypeValidationErrors import ConfigTypeError
from tensilelite.SolutionStructs.lds_layout import format_lds_layout
from tensilelite.SolutionStructs.segment_interleave import evaluate, paired_buffer_offsets
from tensilelite.Tests.unit.test_segment_interleave import _vw8_state

pytestmark = pytest.mark.unit


def _ordinary(**overrides):
    state = dict(ISA=(12, 5, 0), MIWaveGroup=[2, 2], PrefetchGlobalRead=1,
                 NumLdsBlk=2, LdsOffsetA=0, LdsOffsetB=1024, LdsOffsetA_Blk=65536,
                 ldsNumBytesA=1024, ldsNumBytesB=2048, LdsNumBytes=68608)
    state.update(overrides)
    return state


def test_ordinary_addresses_and_segment_order():
    text = format_lds_layout(_ordinary(), 'example')
    assert text.startswith('[LDS layout] example\n')
    assert '[     0,   1024) A ping' in text
    assert '[ 65536,  66560) A pong' in text
    assert 'S0 [0, 65536): [A ping][B ping][gap 62464 B]' in text
    assert 'S1 [65536, 68608): [A pong][B pong]' in text
    segments = [line for line in text.splitlines() if line.startswith('    S')]
    assert len(segments) == 2
    assert segments[0].startswith('    S0 ') and segments[1].startswith('    S1 ')


@pytest.mark.parametrize('overrides', [{'1LDSBuffer': 1}, {'PrefetchGlobalRead': 0}])
def test_single_buffer_does_not_report_phantom_pong(overrides):
    text = format_lds_layout(_ordinary(**overrides), 'single')
    assert 'A buffer0' in text
    assert 'pong' not in text
    assert 'Buffer strides (bytes): A=0, B=0' in text


def test_regions_crossing_segments_appear_in_both():
    text = format_lds_layout(_ordinary(**{'1LDSBuffer': 1, 'ldsNumBytesA': 66048,
                                       'ldsNumBytesB': 0}), 'crossing')
    assert 'S0 [0, 65536): [A buffer0]' in text
    assert 'S1 [65536, 68608): [A buffer0][gap 2560 B]' in text


def test_three_buffers_and_non_gfx1250_have_no_invented_segment_geometry():
    text = format_lds_layout(_ordinary(ISA=(9, 4, 2), NumLdsBlk=3,
                                      LdsNumBytes=134144), 'triple')
    assert '[131072, 132096) A buffer2' in text
    assert '64 KiB segments' not in text


def test_decoupled_groups_use_their_own_buffer_counts_and_strides():
    text = format_lds_layout(_ordinary(PrefetchGlobalReadA=1, PrefetchGlobalReadB=2,
                                      LdsOffsetB=1024, LdsOffsetBlkA=1024,
                                      LdsOffsetBlkB=2048, LdsNumBytes=5120), 'decoupled')
    assert 'A buffer0' in text and 'A pong' not in text
    assert '[  3072,   5120) B pong' in text
    assert 'A=0, B=2048' in text


def _interleaved(**overrides):
    state = _vw8_state(PrefetchGlobalRead=2, LDSSegmentInterleave=1,
                       NumLdsBlk=2, MaxLDS=327680, LdsOffsetA_Blk=131072,
                       VectorWidthB=4, **overrides)
    state['LDSSegInterleaveOffsets'] = evaluate(state)['offsets']
    return state


def test_paired_physical_order_scales_and_no_state_mutation():
    state = _interleaved(ProblemType=dict(MXBlockA=32, MXBlockB=32),
                         LdsNumElementsAlignedMXSA=1024, LdsNumElementsAlignedMXSB=2048)
    offsets = paired_buffer_offsets(state)
    state.update(LDSSegInterleaveOffsets=offsets, LdsNumBytes=offsets['pairedUsedBytes'])
    before = deepcopy(state)
    text = format_lds_layout(state, 'paired')
    assert 'layout=paired' in text
    names = [line.split(') ')[1].split(' (')[0] for line in text.splitlines()
             if line.startswith('    [')]
    assert names == ['A0 ping', 'A0 pong', 'B0 ping', 'B0 pong',
                     'A1 ping', 'A1 pong', 'B1 ping', 'B1 pong',
                     'MXSA ping', 'MXSA pong', 'MXSB ping', 'MXSB pong']
    assert f"[{offsets['ldsBaseMXSA']:6d}, {offsets['ldsBaseMXSA'] + 1024:6d}) MXSA ping" in text
    assert text == format_lds_layout(state, 'paired')
    # Ignore identity-only fake data types; all scalar/layout fields stay unchanged.
    assert {k: v for k, v in state.items() if k != 'ProblemType'} == {
        k: v for k, v in before.items() if k != 'ProblemType'}


def test_aligned_and_tdmsplit_keep_whole_buffer_stride():
    state = _interleaved(DepthU=64, TDMSplit=1)
    state['LdsNumBytes'] = 229632
    text = format_lds_layout(state, 'tdmsplit')
    assert 'layout=interleaved' in text
    assert '[     0,  16512) A group0 ping' in text
    assert '[ 65536,  82048) A group1 ping' in text
    assert '[131072, 147584) A group0 pong' in text
    assert 'group0/1 name physical read groups' in text


def test_whole_operand_and_direct_scales_are_not_split_or_allocated():
    state = _interleaved(MIWaveTile=[8, 7], VectorWidthA=8,
                         DirectToVgprMXSA=True, LdsNumElementsAlignedMXSA=2048)
    # Force B's whole-operand fallback (the helper otherwise uses VWB=4).
    state['VectorWidthB'] = 1
    state['LDSSegInterleaveOffsets'] = evaluate(state)['offsets']
    offsets = paired_buffer_offsets(state)
    state.update(LDSSegInterleaveOffsets=offsets, LdsNumBytes=offsets['pairedUsedBytes'])
    text = format_lds_layout(state, 'whole')
    assert 'B ping' in text and 'B0 ping' not in text
    assert 'MXSA ping' not in text


def test_global_flag_is_boolean_and_off_by_default():
    assert globalParameters['PrintLDSLayout'] is False
    _assertGlobalParametersAreValid({'PrintLDSLayout': True}, [])
    with pytest.raises(ConfigTypeError):
        _assertGlobalParametersAreValid({'PrintLDSLayout': 1}, [])


@pytest.mark.parametrize('enabled,valid,expected', [(False, True, False),
                                                   (True, False, False),
                                                   (True, True, True)])
def test_solution_logs_only_valid_final_layout_when_enabled(monkeypatch, capsys,
                                                          enabled, valid, expected):
    module = importlib.import_module('tensilelite.SolutionStructs.Solution')
    final = _ordinary(Valid=valid)
    # Bypass hardware/toolchain derivation; exercise the real constructor and
    # log with a fully specified allocation, independent of local GPU support.
    monkeypatch.setattr(module.Solution, 'assignDerivedParameters',
                        staticmethod(lambda state, *args: state.update(final)))
    monkeypatch.setattr(module, 'getSolutionNameFull', lambda *args: 'test_kernel')
    monkeypatch.setitem(globalParameters, 'PrintLDSLayout', enabled)
    isa = (12, 5, 0)
    module.Solution({'ISA': isa, 'ProblemType': {'OperationType': 'GEMM', 'DataType': 'H'}},
                    False, False, False,
                    SimpleNamespace(code_object_version='default', rocm_version=None),
                    {isa: SimpleNamespace(asmCaps={'SupportedISA': True})})
    text = capsys.readouterr().out
    assert ('[LDS layout] test_kernel' in text) == expected
    if expected:
        assert '[ 65536,  66560) A pong' in text


def _asymmetric_layout(wg, variant):
    active = 'A' if wg == [4, 1] else 'B'
    tile = [4, 8] if active == 'A' else [8, 4]
    widths = [4, 8] if active == 'A' else [8, 4]
    split = variant in ('tdmsplit_full', 'tdmsplit_half')
    if variant == 'tdmsplit_half':
        widths[0 if active == 'A' else 1] //= 2
    state = _vw8_state(MIWaveGroup=wg, MIWaveTile=tile,
                       VectorWidthA=widths[0], VectorWidthB=widths[1],
                       DepthU=64 if variant == 'aligned' else 128,
                       TDMSplit=split, PrefetchGlobalRead=2,
                       LDSSegmentInterleave=1, NumLdsBlk=2,
                       LdsOffsetA_Blk=131072, MaxLDS=327680)
    result = evaluate(state)
    assert result['applicable'], result['reason']
    state['LDSSegInterleaveOffsets'] = result['offsets']
    if variant == 'paired':
        offsets = paired_buffer_offsets(state)
        assert offsets is not None
        state.update(LDSSegInterleaveOffsets=offsets, LdsNumBytes=offsets['pairedUsedBytes'])
    else:
        # The complete ping block fits a 128 KiB stride. Its last active group
        # ends at 82048 bytes (aligned) or 99072 bytes (TDMSplit).
        state['LdsNumBytes'] = 131072 + (82048 if variant == 'aligned' else 99072)
    return state


@pytest.mark.parametrize('wg,active,whole', [([4, 1], 'A', 'B'), ([1, 4], 'B', 'A')])
@pytest.mark.parametrize('variant', ['paired', 'aligned', 'tdmsplit_full', 'tdmsplit_half'])
def test_asymmetric_patterns_report_active_groups_and_whole_operand(wg, active, whole, variant):
    state = _asymmetric_layout(wg, variant)
    text = format_lds_layout(state, f'{wg}_{variant}')
    regions = [line for line in text.splitlines() if line.startswith('    [')]
    assert len(regions) == 6
    assert 'MIWaveGroup=' + str(wg) in text
    assert f'{whole} ping' in text and f'{whole} pong' in text
    assert f'{whole}0 ping' not in text and f'{whole} group0 ping' not in text
    if variant == 'paired':
        assert 'layout=paired' in text
        names = [line.split(') ')[1].split(' (')[0] for line in regions]
        assert names == [f'{active}0 ping', f'{active}0 pong', f'{whole} ping',
                         f'{whole} pong', f'{active}1 ping', f'{active}1 pong']
        assert f'[132096, 165120) {active}1 ping' in text
        assert f'[ 66048,  99072) {whole} ping' in text
    elif variant == 'aligned':
        assert f'[ 65536,  82048) {active}1 ping' in text
        assert f'[196608, 213120) {active}1 pong' in text
        assert f'[ 16512,  33024) {whole} ping' in text
    else:
        assert f'[ 66048,  99072) {active} group1 ping' in text
        assert f'[ 33024,  66048) {whole} ping' in text
        offsets = state['LDSSegInterleaveOffsets']
        key = 'componentSplit' if variant == 'tdmsplit_half' else 'portSplit' + active
        assert offsets[key]
