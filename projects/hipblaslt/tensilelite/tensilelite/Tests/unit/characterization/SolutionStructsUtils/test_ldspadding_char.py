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

"""Characterization tests for ``Tensile.SolutionStructs.LdsPadding`` — the pure
numeric LDS-padding solvers (``get_fp4/fp8/fp16/fp32/mxs_mt_config`` and their
``_compute_*`` / bank-conflict-check helpers). Each public selector is exercised
over a grid of (macro-tile, wave, vector-width) inputs and the full result
config (``perBlock`` / ``pad`` / ``shift``) is snapshotted, so the padding
tables are pinned and the tier-search branches are walked.
"""

import pytest

import Tensile.SolutionStructs.LdsPadding as L

pytestmark = pytest.mark.unit

_B64_KEYS = ("perBlock", "pad", "shift")
_B128_KEYS = ("perBlock", "pad")

_MTS = [64, 96, 128, 192, 256, 384, 512]
_WAVES = [(1, 1), (2, 1), (2, 2), (4, 2)]


# ===========================================================================
# FP4 / FP8 (ds_load_tr*_b64)
# ===========================================================================

@pytest.mark.parametrize("wt,wg", _WAVES, ids=[f"wt{a}_wg{b}" for a, b in _WAVES])
def test_get_fp4_mt_config(wt, wg, snapshot):
    result = {
        mt: {k: L.get_fp4_mt_config(mt, k, wt, wg) for k in _B64_KEYS}
        for mt in _MTS
    }
    assert result == snapshot


@pytest.mark.parametrize("wt,wg", _WAVES, ids=[f"wt{a}_wg{b}" for a, b in _WAVES])
def test_get_fp8_mt_config(wt, wg, snapshot):
    result = {
        mt: {k: L.get_fp8_mt_config(mt, k, wt, wg) for k in _B64_KEYS}
        for mt in _MTS
    }
    assert result == snapshot


# ===========================================================================
# FP16 (ds_load_tr16_b128)
# ===========================================================================

# (miWaveGroup, miInputPerThUnroll, lrvw, miWaveTile, vw)
_FP16_PARAMS = [
    (1, 16, 8, 1, 1),
    (2, 16, 8, 2, 1),
    (2, 32, 16, 4, 2),
    (4, 16, 8, 1, 1),
]


@pytest.mark.parametrize("params", _FP16_PARAMS, ids=[str(p) for p in _FP16_PARAMS])
def test_get_fp16_mt_config(params, snapshot):
    wg, ipt, lrvw, wt, vw = params
    result = {
        mt: {k: L.get_fp16_mt_config(mt, k, wg, ipt, lrvw, wt, vw) for k in _B128_KEYS}
        for mt in _MTS
    }
    assert result == snapshot


# ===========================================================================
# FP32 / XF32 (ds_load_b32)
# ===========================================================================

# (vw, lrvw, miWaveGroup, miInputPerThread, miWaveTile, xf32EmuPack)
_FP32_PARAMS = [
    (1, 1, 1, 1, 1, False),
    (2, 1, 2, 2, 2, False),
    (1, 1, 1, 2, 1, True),
    (4, 1, 2, 4, 4, False),
]


@pytest.mark.parametrize("params", _FP32_PARAMS, ids=[str(p) for p in _FP32_PARAMS])
def test_get_fp32_mt_config(params, snapshot):
    vw, lrvw, wg, ipt, wt, emu = params
    result = {
        mt: {k: L.get_fp32_mt_config(mt, k, vw, lrvw, wg, ipt, wt, emu) for k in _B128_KEYS}
        for mt in _MTS
    }
    assert result == snapshot


# ===========================================================================
# MXS (scale tensor)
# ===========================================================================

@pytest.mark.parametrize(
    "k,mxBlock,vw",
    [
        (256, 32, 4),   # d=32 -> even d//16 -> {256, 16}
        (128, 32, 4),   # d=16 -> d//16==1 (odd) -> {0, 0}
        (128, 32, 2),   # vw<4 -> {0, 0}
        (128, 0, 4),    # mxBlock<=0 -> {0, 0}
        (128, 32, 0),   # vw<=0 -> {0, 0}
        (100, 32, 4),   # d not multiple of 16 -> {0, 0}
    ],
    ids=["valid", "odd_half", "vw_lt4", "mxblock_zero", "vw_zero", "d_not_16x"],
)
def test_get_mxs_mt_config(k, mxBlock, vw, snapshot):
    result = {key: L.get_mxs_mt_config(k, mxBlock, vw, key) for key in _B128_KEYS}
    assert result == snapshot


# ===========================================================================
# Private bank-conflict helpers — reject branches that the public selectors
# never reach for the real b64/b128 access patterns (the closed-form / tiered
# search always picks a passing config). Exercised directly to pin the
# contract; see resistance.md.
# ===========================================================================

def test_fp16_odd_n_no_padding(snapshot):
    # n = mt//16 odd -> closed-form returns {0, 0} with no search (line 286).
    result = {k: L.get_fp16_mt_config(80, k, 1, 16, 8, 1, 1) for k in _B128_KEYS}
    assert result == snapshot


def test_b128_check_not_16_aligned_false():
    # A non-16-byte-aligned address -> reject (line 243).
    assert L._b128_check([4], 1024, 0, (0,), (0,)) is False


def test_b128_check_bank_conflict_false():
    # Two threads at the same base address collide in banks (line 250).
    assert L._b128_check([0, 0], 1024, 0, (0,), (0,)) is False


def test_b64_check_not_dword_aligned_false():
    # A per-thread address not 4-byte aligned -> reject (line 113).
    assert L._b64_check([2], [0], 1024, 0, (0,), (0,), 0) is False


def test_b64_compute_config_no_valid_block_returns_zero(snapshot):
    # minB larger than every valid block size -> empty candidate set -> no
    # config in any tier -> all-zero result (line 169).
    cfg = L._b64_compute_config(128, 0.5, L._b64_base_addrs_fp4, 2048, (0,), (0,))
    assert cfg == snapshot


def test_fp16_search_fallback_finds_config(monkeypatch, snapshot):
    # The closed-form check never fails for real inputs, so the search fallback
    # (lines 301-313) is unreachable via the public selector. Force it by
    # making the bank check reject the closed-form pick and accept only
    # (B=16, P=4); the search then settles on that.
    monkeypatch.setattr(L, "_b128_check", lambda half, B, P, wOffsets, instOffs=(0,): B == 16 and P == 4)
    cfg = L._compute_fp16_config(20000, 1, miInputPerThUnroll=16, lrvw=8, miWaveTile=1, vw=1)
    assert cfg == snapshot


def test_fp16_search_fallback_no_config(monkeypatch, snapshot):
    # Same forced search, but nothing passes -> the all-zero return (line 314).
    monkeypatch.setattr(L, "_b128_check", lambda half, B, P, wOffsets, instOffs=(0,): False)
    cfg = L._compute_fp16_config(20032, 1, miInputPerThUnroll=16, lrvw=8, miWaveTile=1, vw=1)
    assert cfg == snapshot
