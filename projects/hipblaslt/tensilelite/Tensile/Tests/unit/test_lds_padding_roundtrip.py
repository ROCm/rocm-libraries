# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every LDS padding value the solver picks must survive a yaml round trip.

Tuning resolves LdsPad and LdsBlockSizePerPad from -1, writes the result into
a yaml, and reads it back. A value validParameters rejects breaks that loop,
and the failure shows up far from its cause: a shipped config stops loading.

Sparse metadata is out of scope; LdsPadMetadata keeps its own narrow list.
"""

import pytest

from Tensile.Common.ValidParameters import checkParametersAreValid, validParametersForArch
from Tensile.SolutionStructs.LdsPadding import (
    get_fp4_mt_config,
    get_fp8_mt_config,
    get_fp16_mt_config,
    get_fp32_mt_config,
    get_mxs_mt_config,
)

pytestmark = pytest.mark.unit

# The solver is gfx1250's, and so is the widened set a yaml may name.
gfx1250Parameters = validParametersForArch("gfx1250")

# MatrixInstK per read path. The tail loop advances mtBytes * MatrixInstK per
# pass, and that step decides which block sizes the search may offer.
_K_B64 = 128
_K_B128 = 32
_K_B32 = 4

_WAVE_GROUPS = (1, 2, 4)
_WAVE_TILES = (1, 2, 3, 4, 6, 8)


def _solverOutputs(mt, miWaveTile, miWaveGroup, usesTDM):
    """(parameter name, value) for every selector, at one shape."""
    yield "LdsPadA", get_fp4_mt_config(mt, "pad", miWaveTile, miWaveGroup, _K_B64, usesTDM)
    yield "LdsBlockSizePerPadA", get_fp4_mt_config(
        mt, "perBlock", miWaveTile, miWaveGroup, _K_B64, usesTDM)

    yield "LdsPadA", get_fp8_mt_config(mt, "pad", miWaveTile, miWaveGroup, _K_B64, usesTDM)
    yield "LdsBlockSizePerPadA", get_fp8_mt_config(
        mt, "perBlock", miWaveTile, miWaveGroup, _K_B64, usesTDM)

    yield "LdsPadA", get_fp16_mt_config(
        mt, "pad", miWaveGroup, 16, 8, miWaveTile, 1, _K_B128, usesTDM)
    yield "LdsBlockSizePerPadA", get_fp16_mt_config(
        mt, "perBlock", miWaveGroup, 16, 8, miWaveTile, 1, _K_B128, usesTDM)

    for vw in (1, 2, 4):
        if miWaveTile % vw:
            continue
        yield "LdsPadA", get_fp32_mt_config(
            mt, "pad", vw, 2, miWaveGroup, 2, miWaveTile, _K_B32, usesTDM)
        yield "LdsBlockSizePerPadA", get_fp32_mt_config(
            mt, "perBlock", vw, 2, miWaveGroup, 2, miWaveTile, _K_B32, usesTDM)



@pytest.mark.parametrize("usesTDM", [True, False])
def test_solver_values_are_settable_in_yaml(usesTDM):
    for miWaveGroup in _WAVE_GROUPS:
        for miWaveTile in _WAVE_TILES:
            mt = 16 * miWaveTile * miWaveGroup
            for name, value in _solverOutputs(mt, miWaveTile, miWaveGroup, usesTDM):
                # A yaml names A and B the same way, so checking A covers both.
                checkParametersAreValid((name, [value]), gfx1250Parameters)
                assert value in gfx1250Parameters[name], (
                    name, value, mt, miWaveTile, miWaveGroup, usesTDM)


def test_mx_scale_values_are_settable_in_yaml():
    for matrixInstK in (32, 64, 128):
        for mxBlock in (16, 32):
            for vw in (1, 2, 4, 8):
                for name, key in (("LdsPadMXSA", "pad"),
                                  ("LdsBlockSizePerPadMXSA", "perBlock")):
                    value = get_mxs_mt_config(matrixInstK, mxBlock, vw, key)
                    checkParametersAreValid((name, [value]), gfx1250Parameters)


def test_a_and_b_share_one_list():
    # A tuned config sets both, and a value legal for one has to be legal for
    # the other, so the two entries must be the same object.
    assert gfx1250Parameters["LdsPadA"] is gfx1250Parameters["LdsPadB"]
    assert gfx1250Parameters["LdsBlockSizePerPadA"] is gfx1250Parameters["LdsBlockSizePerPadB"]


def test_other_architectures_keep_their_list():
    # The widening is gfx1250's. A yaml for anything else names what it always
    # could, so nothing new reaches a path no one checks.
    other = validParametersForArch("gfx942")
    assert other["LdsPadA"] == [-1, 0, 1, 2, 3, 4, 8, 16, 32, 48, 64]
    for value in (6, 24, 96):
        with pytest.raises(Exception):
            checkParametersAreValid(("LdsPadA", [value]), other)


def test_odd_dword_pads_stay_rejected():
    # Widening the list to fit the solver must not open up pads the solver
    # would never pick. 5 and 7 elements are odd dwords for fp32.
    for value in (5, 7, 9):
        with pytest.raises(Exception):
            checkParametersAreValid(("LdsPadA", [value]), gfx1250Parameters)
