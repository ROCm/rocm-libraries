################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
import pytest

from Tensile.SolutionStructs.LdsPadding import (
    get_fp4_mt_config,
    get_fp8_mt_config,
    get_fp16_mt_config,
    get_fp32_mt_config,
    get_metadata_mt_config,
    get_mxs_mt_config,
)


# FP4
#   MT=32  from [16,16,128,1,1,1,1,2,2]
#   MT=64  from [16,16,128,1,1,2,2,2,2]
#   MT=128 from [16,16,128,1,1,4,4,2,2]
#   MT=256 from [16,16,128,1,1,8,8,2,2]
@pytest.mark.parametrize(
    "mt, miWaveTile, miWaveGroup, perBlock, pad, shift",
    [
        ( 32, 1, 2,  512, 16, 0),
        ( 64, 2, 2, 1024, 16, 4),
        (128, 4, 2,  128, 16, 0),
        (256, 8, 2,    0,  0, 0),
    ],
)
def test_fp4(mt, miWaveTile, miWaveGroup, perBlock, pad, shift):
    assert get_fp4_mt_config(mt, "perBlock", miWaveTile, miWaveGroup) == perBlock
    assert get_fp4_mt_config(mt, "pad",      miWaveTile, miWaveGroup) == pad
    assert get_fp4_mt_config(mt, "shift",    miWaveTile, miWaveGroup) == shift


# FP8
# Yaml shapes:
#   MT=32  from [16,16,128,1,1,1,1,2,2]
#   MT=64  from [16,16,128,1,1,2,2,2,2]
#   MT=128 from [16,16,128,1,1,4,4,2,2]
#   MT=256 from [16,16,128,1,1,8,8,2,2]
@pytest.mark.parametrize(
    "mt, miWaveTile, miWaveGroup, perBlock, pad, shift",
    [
        ( 32, 1, 2, 256,  8, 0),
        ( 64, 2, 2, 128, 16, 0),
        (128, 4, 2, 128,  8, 4),
        (256, 8, 2, 128,  8, 4),
    ],
)
def test_fp8(mt, miWaveTile, miWaveGroup, perBlock, pad, shift):
    assert get_fp8_mt_config(mt, "perBlock", miWaveTile, miWaveGroup) == perBlock
    assert get_fp8_mt_config(mt, "pad",      miWaveTile, miWaveGroup) == pad
    assert get_fp8_mt_config(mt, "shift",    miWaveTile, miWaveGroup) == shift


# FP16 / BF16
#   MT=16  from [16,16,32,1,1,1,1,1,1]
#   MT=32  from [16,16,32,1,1,1,1,2,2]
#   MT=64  from [16,16,32,1,1,2,2,2,2]
#   MT=256 from [16,16,32,1,1,8,8,2,2]
@pytest.mark.parametrize(
    "mt, miWaveGroup, miWaveTile, perBlock, pad",
    [
        ( 16, 1, 1,   0, 0),
        ( 32, 2, 1,  32, 8),
        ( 64, 2, 2,  64, 8),
        (256, 2, 8, 256, 8),
    ],
)
def test_fp16(mt, miWaveGroup, miWaveTile, perBlock, pad):
    assert get_fp16_mt_config(mt, "perBlock", miWaveGroup, 16, 8, miWaveTile, 1) == perBlock
    assert get_fp16_mt_config(mt, "pad",      miWaveGroup, 16, 8, miWaveTile, 1) == pad


# FP32
#   MT=32  from [16,16,4,1,1,1,1,2,2]
#   MT=64  from [16,16,4,1,1,2,2,2,2]
#   MT=128 from [16,16,4,1,1,4,4,2,2]
#   MT=256 from [16,16,4,1,1,8,8,2,2]
@pytest.mark.parametrize(
    "mt, vw, lrvw, miWaveGroup, miInputPerThread, miWaveTile, perBlock, pad",
    [
        ( 32, 1, 2, 2, 2, 1,  256, 16),
        ( 64, 2, 2, 2, 2, 2,  512, 32),
        (128, 4, 2, 2, 2, 4, 1024,  2),
        (256, 4, 2, 2, 2, 8,  128,  2),
    ],
)
def test_fp32(mt, vw, lrvw, miWaveGroup, miInputPerThread, miWaveTile,
              perBlock, pad):
    assert get_fp32_mt_config(mt, "perBlock", vw, lrvw, miWaveGroup,
                              miInputPerThread, miWaveTile,
                              xf32EmuPack=False) == perBlock
    assert get_fp32_mt_config(mt, "pad", vw, lrvw, miWaveGroup,
                              miInputPerThread, miWaveTile,
                              xf32EmuPack=False) == pad


# XF32 (xf32EmuPack=True)
#   MT=32  from [16,16,32,1,1,1,1,2,2]
#   MT=64  from [16,16,32,1,1,2,2,2,2]
#   MT=128 from [16,16,32,1,1,4,4,2,2]
#   MT=256 from [16,16,32,1,1,8,8,2,2]   (solver rejects: VGPR > 1024)
@pytest.mark.parametrize(
    "mt, vw, lrvw, miWaveGroup, miInputPerThread, miWaveTile, perBlock, pad",
    [
        ( 32, 1, 4, 2, 16, 1,  512, 16),
        ( 64, 2, 4, 2, 16, 2, 1024, 32),
        (128, 2, 4, 2, 16, 4, 1024, 16),
    ],
)
def test_xf32(mt, vw, lrvw, miWaveGroup, miInputPerThread, miWaveTile,
              perBlock, pad):
    assert get_fp32_mt_config(mt, "perBlock", vw, lrvw, miWaveGroup,
                              miInputPerThread, miWaveTile,
                              xf32EmuPack=True) == perBlock
    assert get_fp32_mt_config(mt, "pad", vw, lrvw, miWaveGroup,
                              miInputPerThread, miWaveTile,
                              xf32EmuPack=True) == pad


# MX
@pytest.mark.parametrize(
    "mxBlock, vw, perBlock, pad",
    [
        (32,  8, 256, 16),  # mxBlock=32: needs vw multiple of 8
        (16,  4, 256, 16),  # mxBlock=16: needs vw multiple of 4
        (32,  4,   0,  0),  # d/16 odd -> no padding
        (32,  1,   0,  0),  # vw < 4 -> no padding
        (16,  1,   0,  0),  # vw < 4 -> no padding
    ],
)
def test_mxs(mxBlock, vw, perBlock, pad):
    assert get_mxs_mt_config(128, mxBlock, vw, "perBlock") == perBlock
    assert get_mxs_mt_config(128, mxBlock, vw, "pad")      == pad


# Every public entry point must return a pad whose byte size is an even number
# of dwords. Odd-dword padding puts the gfx1250 LDS hardware in a state that is
# not understood, so it is never emitted.
_WAVE_GROUPS = (1, 2, 4)
_WAVE_TILES = tuple(range(1, 21))


def _pad_bytes_x2(pad, bpeDS):
    """Pad size in half-bytes, so FP4 (bpeDS 0.5) stays an exact integer."""
    return int(round(pad * bpeDS * 2))


def test_fp4_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            pad = get_fp4_mt_config(mt, "pad", wt, wg)
            assert _pad_bytes_x2(pad, 0.5) % 16 == 0, (mt, wt, wg, pad)


def test_fp8_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            pad = get_fp8_mt_config(mt, "pad", wt, wg)
            assert _pad_bytes_x2(pad, 1.0) % 16 == 0, (mt, wt, wg, pad)


def test_fp16_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            pad = get_fp16_mt_config(mt, "pad", wg, 16, 8, wt, 1)
            assert _pad_bytes_x2(pad, 2.0) % 16 == 0, (mt, wt, wg, pad)


def test_fp32_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for vw in (1, 2, 4):
                if wt % vw:
                    continue
                pad = get_fp32_mt_config(mt, "pad", vw, 2, wg, 2, wt)
                assert _pad_bytes_x2(pad, 4.0) % 16 == 0, (mt, wt, wg, vw, pad)


def test_metadata_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            pad = get_metadata_mt_config(mt, "pad", wt, wg, 16, 64)
            assert _pad_bytes_x2(pad, 1.0) % 16 == 0, (mt, wt, wg, pad)


def test_mxs_pad_is_even_dwords():
    for matrixInstK in (32, 64, 128):
        for mxBlock in (16, 32):
            for vw in (1, 2, 4, 8):
                pad = get_mxs_mt_config(matrixInstK, mxBlock, vw, "pad")
                assert _pad_bytes_x2(pad, 1.0) % 16 == 0, (matrixInstK, mxBlock, vw, pad)
