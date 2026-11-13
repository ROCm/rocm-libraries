# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest

from tensilelite.Common.DataType import DataType
from tensilelite.KernelWriterAssembly import _fp8D16UsesHighWord, _tailPacksTwoElementsPerLoad


def _kernel(asem=32, convertAfterDS=False, directToLds=False, nonDTLTailLoop=False, glTr=False):
    return {
        "ProblemType": {"DataTypeB": DataType("F8N"), "MacDataTypeB": DataType("H")},
        "ConvertAfterDS": convertAfterDS,
        "DirectToLdsB": directToLds,
        "NonDTLTailLoopB": nonDTLTailLoop,
        "enableGLTrB": glTr,
        "AssertSummationElementMultiple": asem,
    }


def _tP(glvw=2):
    return {"tensorChar": "B", "glvw": glvw}


def test_fp8_d16_main_loop_preserves_alternating_words():
    assert not _fp8D16UsesHighWord(False, False, True)
    assert _fp8D16UsesHighWord(True, False, True)


def test_fp8_d16_packed_tail_always_uses_low_word():
    assert not _fp8D16UsesHighWord(False, True, True)
    assert not _fp8D16UsesHighWord(True, True, True)


def test_fp8_d16_unpacked_tail_preserves_alternating_words():
    assert not _fp8D16UsesHighWord(False, True, False)
    assert _fp8D16UsesHighWord(True, True, False)


def test_fp8_half_glvw2_even_assert_packs_tail_reads():
    assert _tailPacksTwoElementsPerLoad(_kernel(asem=32), _tP(glvw=2))


@pytest.mark.parametrize(
    "kernel, tP",
    [
        (_kernel(asem=1), _tP(glvw=2)),
        (_kernel(asem=32), _tP(glvw=1)),
        (_kernel(asem=32), _tP(glvw=4)),
        (_kernel(asem=32, convertAfterDS=True), _tP(glvw=2)),
        (_kernel(asem=32, directToLds=True), _tP(glvw=2)),
        (_kernel(asem=32, glTr=True), _tP(glvw=2)),
    ],
    ids=["odd-assert", "glvw1", "glvw4", "convert-after-ds", "direct-to-lds", "gl-tr"],
)
def test_fp8_half_unpacked_tail_reads(kernel, tP):
    assert not _tailPacksTwoElementsPerLoad(kernel, tP)


def test_fp8_half_direct_to_lds_with_non_dtl_tail_packs_tail_reads():
    kernel = _kernel(asem=32, directToLds=True, nonDTLTailLoop=True)
    assert _tailPacksTwoElementsPerLoad(kernel, _tP(glvw=2))
