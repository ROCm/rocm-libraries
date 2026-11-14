#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for BufferLoadOffsetLimitCheck predicate in Contractions.py.

Covers Contractions.py:702-713 (ProblemPredicate.CompoundPredicates BufferLoad block).

The BufferLoadOffsetLimitCheck predicate computes host-side buffer-load offset-limit
gating values to prevent extremely large problems from launching with buffer_load
offsets exceeding 2^32.

Critical branches:
- MayShiftA = TLUA and AssertFree0ElementMultiple < GlobalReadVectorWidthA
- MayShiftB = TLUB and AssertFree1ElementMultiple < GlobalReadVectorWidthB
- ShiftPtrElemA = GRVWA if MayShiftA else 0
- ShiftPtrElemB = GRVWB if MayShiftB else 0
- DUorMT0 = DepthU if TLUA else MacroTile0
- DUorMT1 = DepthU if TLUB else MacroTile1
"""

import pytest

# Mark as unit test
pytestmark = pytest.mark.unit


def test_buffer_load_offset_limit_check_shift_conditions():
    """BufferLoadOffsetLimitCheck MayShift logic and value dict.

    Validates that ShiftPtrElemA/B are set to GlobalReadVectorWidth when
    shifting is possible (TLUA/B=True and AssertFree < GRVW), else 0.

    Validates that DUorMT0/1 use DepthU when TLU is true, else MacroTile.
    """
    from tensilelite.Contractions import ProblemPredicate
    from types import SimpleNamespace

    # Minimal problem type (need aType.isInt8x4() method)
    problem_type = SimpleNamespace(aType=SimpleNamespace(isInt8x4=lambda: False))

    # Minimal state dict for BufferLoadOffsetLimitCheck block
    def make_state(tlua, tlub, assert_free0, assert_free1, grvwa=4, grvwb=4,
                   depth_u=16, mt0=128, mt1=128):
        return {
            'BufferLoad': True,
            'BufferStore': False,
            'ProblemType': {
                'TLUA': tlua,
                'TLUB': tlub,
                'DataType': 0,
                'DestDataType': 0,
                'ComputeDataType': 0,
                'SwizzleTensorA': False,
                'SwizzleTensorB': False,
                'FusedGemmA2A': False,
            },
            'AssertFree0ElementMultiple': assert_free0,
            'AssertFree1ElementMultiple': assert_free1,
            'GlobalReadVectorWidthA': grvwa,
            'GlobalReadVectorWidthB': grvwb,
            'DepthU': depth_u,
            'MacroTile0': mt0,
            'MacroTile1': mt1,
            'InternalSupportParams': {'KernArgsVersion': 0},
            'GlobalSplitU': 1,
            'PackedC0IndicesX': [0],
        }

    # Case 1: TLUA=True, AssertFree0 < GRVWA → MayShiftA=True → ShiftPtrElemA=GRVWA
    state1 = make_state(tlua=True, tlub=True, assert_free0=2, assert_free1=2,
                        grvwa=4, grvwb=4, depth_u=16, mt0=128, mt1=128)
    preds1 = ProblemPredicate.CompoundPredicates(state1, problem_type)
    blolc1 = next(p for p in preds1 if p.tag == 'BufferLoadOffsetLimitCheck')
    assert blolc1.value['ShiftPtrElemA'] == 4, "TLUA=True, assert<GRVW → shift"
    assert blolc1.value['ShiftPtrElemB'] == 4, "TLUB=True, assert<GRVW → shift"
    assert blolc1.value['DUorMT0'] == 16, "TLUA=True → DepthU"
    assert blolc1.value['DUorMT1'] == 16, "TLUB=True → DepthU"

    # Case 2: TLUA=True, AssertFree0 == GRVWA → MayShiftA=False → ShiftPtrElemA=0
    state2 = make_state(tlua=True, tlub=True, assert_free0=4, assert_free1=4,
                        grvwa=4, grvwb=4, depth_u=16, mt0=128, mt1=128)
    preds2 = ProblemPredicate.CompoundPredicates(state2, problem_type)
    blolc2 = next(p for p in preds2 if p.tag == 'BufferLoadOffsetLimitCheck')
    assert blolc2.value['ShiftPtrElemA'] == 0, "assert==GRVW → no shift"
    assert blolc2.value['ShiftPtrElemB'] == 0, "assert==GRVW → no shift"
    assert blolc2.value['DUorMT0'] == 16, "TLUA=True → DepthU"
    assert blolc2.value['DUorMT1'] == 16, "TLUB=True → DepthU"

    # Case 3: TLUA=False, TLUB=False → no shift, use MacroTile
    state3 = make_state(tlua=False, tlub=False, assert_free0=2, assert_free1=2,
                        grvwa=4, grvwb=4, depth_u=16, mt0=128, mt1=256)
    preds3 = ProblemPredicate.CompoundPredicates(state3, problem_type)
    blolc3 = next(p for p in preds3 if p.tag == 'BufferLoadOffsetLimitCheck')
    assert blolc3.value['ShiftPtrElemA'] == 0, "TLUA=False → no shift"
    assert blolc3.value['ShiftPtrElemB'] == 0, "TLUB=False → no shift"
    assert blolc3.value['DUorMT0'] == 128, "TLUA=False → MacroTile0"
    assert blolc3.value['DUorMT1'] == 256, "TLUB=False → MacroTile1"

    # Case 4: Mixed - TLUA=True (shift), TLUB=False (no shift)
    state4 = make_state(tlua=True, tlub=False, assert_free0=1, assert_free1=1,
                        grvwa=4, grvwb=4, depth_u=32, mt0=64, mt1=64)
    preds4 = ProblemPredicate.CompoundPredicates(state4, problem_type)
    blolc4 = next(p for p in preds4 if p.tag == 'BufferLoadOffsetLimitCheck')
    assert blolc4.value['ShiftPtrElemA'] == 4, "TLUA=True, assert<GRVW → shift"
    assert blolc4.value['ShiftPtrElemB'] == 0, "TLUB=False → no shift"
    assert blolc4.value['DUorMT0'] == 32, "TLUA=True → DepthU"
    assert blolc4.value['DUorMT1'] == 64, "TLUB=False → MacroTile1"

    # Case 5: AssertFree0 > GRVWA (edge case: greater, not just equal) → no shift
    state5 = make_state(tlua=True, tlub=True, assert_free0=8, assert_free1=8,
                        grvwa=4, grvwb=4, depth_u=16, mt0=128, mt1=128)
    preds5 = ProblemPredicate.CompoundPredicates(state5, problem_type)
    blolc5 = next(p for p in preds5 if p.tag == 'BufferLoadOffsetLimitCheck')
    assert blolc5.value['ShiftPtrElemA'] == 0, "assert>GRVW → no shift"
    assert blolc5.value['ShiftPtrElemB'] == 0, "assert>GRVW → no shift"


def test_buffer_load_offset_limit_check_no_predicate_when_buffer_load_false():
    """BufferLoadOffsetLimitCheck only emitted when BufferLoad=True.

    Guards the 'BufferLoad' in state and state['BufferLoad'] == True condition.
    """
    from tensilelite.Contractions import ProblemPredicate
    from types import SimpleNamespace

    state = {
        'BufferLoad': False,
        'BufferStore': False,
        'ProblemType': {
            'TLUA': True,
            'TLUB': True,
            'DataType': 0,
            'DestDataType': 0,
            'ComputeDataType': 0,
            'SwizzleTensorA': False,
            'SwizzleTensorB': False,
            'FusedGemmA2A': False,
        },
        'AssertFree0ElementMultiple': 2,
        'AssertFree1ElementMultiple': 2,
        'GlobalReadVectorWidthA': 4,
        'GlobalReadVectorWidthB': 4,
        'DepthU': 16,
        'MacroTile0': 128,
        'MacroTile1': 128,
        'InternalSupportParams': {'KernArgsVersion': 0},
        'GlobalSplitU': 1,
        'PackedC0IndicesX': [0],
    }
    problem_type = SimpleNamespace(aType=SimpleNamespace(isInt8x4=lambda: False))
    preds = ProblemPredicate.CompoundPredicates(state, problem_type)
    blolc = [p for p in preds if p.tag == 'BufferLoadOffsetLimitCheck']
    assert blolc == [], "BufferLoad=False → no BufferLoadOffsetLimitCheck predicate"
