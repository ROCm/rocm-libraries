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

# MatrixInstK the tile-major sweeps use, per read path. The tail loop advances
# mtBytes * MatrixInstK per pass, and that step decides which block sizes the
# search may offer.
_K_B64  = 128   # FP4 / FP8  ds_load_tr*_b64
_K_B128 = 32    # FP16       ds_load_tr16_b128
_K_B32  = 4     # FP32       ds_load_b32
_K_XF32 = 32    # XF32       ds_load_b32, MatrixInstruction [16,16,32,1]

# The sweeps are TDM, where the hardware writes LDS itself and no per-thread
# ds_write constrains the block from below.
_TDM = True


import Tensile.SolutionStructs.LdsPadding as _L
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
    "mt, miWaveTile, miWaveGroup, perBlock, pad",
    [
        ( 32, 1, 2,  512, 16),
        ( 64, 2, 2, 1024, 16),
        (128, 4, 2,  128, 16),
        (256, 8, 2,  512, 16),
    ],
)
def test_fp4(mt, miWaveTile, miWaveGroup, perBlock, pad):
    assert get_fp4_mt_config(mt, "perBlock", miWaveTile, miWaveGroup, _K_B64, _TDM) == perBlock
    assert get_fp4_mt_config(mt, "pad",      miWaveTile, miWaveGroup, _K_B64, _TDM) == pad


# FP8
# Yaml shapes:
#   MT=32  from [16,16,128,1,1,1,1,2,2]
#   MT=64  from [16,16,128,1,1,2,2,2,2]
#   MT=128 from [16,16,128,1,1,4,4,2,2]
#   MT=256 from [16,16,128,1,1,8,8,2,2]
@pytest.mark.parametrize(
    "mt, miWaveTile, miWaveGroup, perBlock, pad",
    [
        ( 32, 1, 2, 512, 16),
        ( 64, 2, 2, 128, 16),
        (128, 4, 2, 512, 16),
        (256, 8, 2, 512, 16),
    ],
)
def test_fp8(mt, miWaveTile, miWaveGroup, perBlock, pad):
    assert get_fp8_mt_config(mt, "perBlock", miWaveTile, miWaveGroup, _K_B64, _TDM) == perBlock
    assert get_fp8_mt_config(mt, "pad",      miWaveTile, miWaveGroup, _K_B64, _TDM) == pad


# FP16 / BF16
#   MT=16  from [16,16,32,1,1,1,1,1,1]
#   MT=32  from [16,16,32,1,1,1,1,2,2]
#   MT=64  from [16,16,32,1,1,2,2,2,2]
#   MT=256 from [16,16,32,1,1,8,8,2,2]
@pytest.mark.parametrize(
    "mt, miWaveGroup, miWaveTile, perBlock, pad",
    [
        ( 16, 1, 1,   0,  0),
        ( 32, 2, 1, 256, 16),
        ( 64, 2, 2, 256, 16),
        (256, 2, 8, 512, 16),
    ],
)
def test_fp16(mt, miWaveGroup, miWaveTile, perBlock, pad):
    assert get_fp16_mt_config(mt, "perBlock", miWaveGroup, 16, 8, miWaveTile, 1, _K_B128, _TDM) == perBlock
    assert get_fp16_mt_config(mt, "pad",      miWaveGroup, 16, 8, miWaveTile, 1, _K_B128, _TDM) == pad


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
                              matrixInstK=_K_B32,
                              usesTDM=_TDM,
                              xf32EmuPack=False) == perBlock
    assert get_fp32_mt_config(mt, "pad", vw, lrvw, miWaveGroup,
                              miInputPerThread, miWaveTile,
                              matrixInstK=_K_B32,
                              usesTDM=_TDM,
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
                              matrixInstK=_K_XF32,
                              usesTDM=_TDM,
                              xf32EmuPack=True) == perBlock
    assert get_fp32_mt_config(mt, "pad", vw, lrvw, miWaveGroup,
                              miInputPerThread, miWaveTile,
                              matrixInstK=_K_XF32,
                              usesTDM=_TDM,
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


@pytest.mark.parametrize("matrixInstK", [0, -1, -64])
def test_mxs_rejects_non_positive_matrix_inst_k(matrixInstK):
    assert get_mxs_mt_config(matrixInstK, 32, 8, "perBlock") == 0
    assert get_mxs_mt_config(matrixInstK, 32, 8, "pad")      == 0


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
            pad = get_fp4_mt_config(mt, "pad", wt, wg, _K_B64, _TDM)
            assert _pad_bytes_x2(pad, 0.5) % 16 == 0, (mt, wt, wg, pad)


def test_fp8_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            pad = get_fp8_mt_config(mt, "pad", wt, wg, _K_B64, _TDM)
            assert _pad_bytes_x2(pad, 1.0) % 16 == 0, (mt, wt, wg, pad)


def test_fp16_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            pad = get_fp16_mt_config(mt, "pad", wg, 16, 8, wt, 1, _K_B128, _TDM)
            assert _pad_bytes_x2(pad, 2.0) % 16 == 0, (mt, wt, wg, pad)


def test_fp32_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for vw in (1, 2, 4):
                if wt % vw:
                    continue
                pad = get_fp32_mt_config(mt, "pad", vw, 2, wg, 2, wt, _K_B32, _TDM)
                assert _pad_bytes_x2(pad, 4.0) % 16 == 0, (mt, wt, wg, vw, pad)


def test_metadata_pad_is_even_dwords():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            pad = get_metadata_mt_config(mt, "pad", wt, wg, 16, 64, _K_B64, _TDM)
            assert _pad_bytes_x2(pad, 1.0) % 16 == 0, (mt, wt, wg, pad)


def test_mxs_pad_is_even_dwords():
    for matrixInstK in (32, 64, 128):
        for mxBlock in (16, 32):
            for vw in (1, 2, 4, 8):
                pad = get_mxs_mt_config(matrixInstK, mxBlock, vw, "pad")
                assert _pad_bytes_x2(pad, 1.0) % 16 == 0, (matrixInstK, mxBlock, vw, pad)


def test_max_threads_per_bank_counts_each_bank_a_thread_touches():
    # Two threads, 2 banks each, starting at byte 0 and byte 4: banks
    # {0,1} and {1,2}. Bank 1 carries two threads.
    assert _L._max_threads_per_bank([0, 4], 2) == 2
    # Same two threads 8 bytes apart: banks {0,1} and {2,3}, no sharing.
    assert _L._max_threads_per_bank([0, 8], 2) == 1


def test_no_block_carry_detects_a_carrying_pair():
    # base 256 and instOffs 256, B=512: (256 % 512) + (256 % 512) == 512,
    # which is not less than B, so the pair carries and the base/instOffs
    # padded independently disagrees with padding the combined offset.
    assert _L._no_block_carry([256], [256], 512) is False
    # Same base and instOffs at B=1024: (256 % 1024) + (256 % 1024) == 512,
    # which is less than B, so the pair does not carry.
    assert _L._no_block_carry([256], [256], 1024) is True


def test_b64_wave_costs_counts_two_banks_per_thread():
    # 32 threads, 8 bytes apart, no padding: each thread owns 2 banks and all
    # 64 bank slots are distinct, so the cost is 1.
    addrs = [8 * i for i in range(32)]
    assert _L._b64_wave_costs(addrs, 0, 0, (0,), (0,)) == [1]
    # Every thread 256 bytes apart lands on the same pair of banks.
    addrs = [256 * i for i in range(32)]
    assert _L._b64_wave_costs(addrs, 0, 0, (0,), (0,)) == [32]


def test_b64_wave_costs_rejects_addresses_below_8_byte_alignment():
    # The cost model describes a wave the hardware takes in one batch, which
    # it only does when every address is 8-byte aligned.
    assert _L._b64_wave_costs([4] * 32, 256, 8, (0,), (0,)) is None


def test_b64_config_is_legal_over_the_reachable_shapes():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for getter, bpeDS in ((get_fp8_mt_config, 1.0), (get_fp4_mt_config, 0.5)):
                perBlock = getter(mt, "perBlock", wt, wg, _K_B64, _TDM)
                pad = getter(mt, "pad", wt, wg, _K_B64, _TDM)
                if perBlock == 0:
                    assert pad == 0, (mt, wt, wg, pad)
                    continue
                assert perBlock in _L._LDS_PAD_BLOCK_BYTES, (mt, wt, wg, perBlock)
                padBytes = int(round(pad * bpeDS))
                assert padBytes % 8 == 0, (mt, wt, wg, padBytes)
                assert 0 < padBytes <= _L._LDS_MAX_PAD_BYTES, (mt, wt, wg, padBytes)


def test_b128_wave_costs_pads_the_instruction_offset_on_its_own():
    # Instruction offset 16 is not a multiple of the block size, so padding it
    # on its own is not the same as padding it together with the base address.
    # Padded on its own it is added as a constant and every bank stays
    # distinct, giving cost 1. Folded into the base it would push two threads
    # onto one bank, giving cost 2, so this value pins the address expression.
    half = _L._b128_base_addrs_fp16(16)
    assert _L._b128_wave_costs(half, 256, 16, (0,), (16,)) == [1]


def test_b128_wave_costs_rejects_addresses_below_16_byte_alignment():
    half = [0, 16, 32, 48, 64, 80, 96, 112]
    # P = 8 breaks 16-byte alignment for every address past the first block.
    assert _L._b128_wave_costs(half, 16, 8, (0,), (0,)) is None


def test_fp16_config_is_legal_over_the_reachable_shapes():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for mipt, lrvw in ((16, 8), (16, 16), (32, 8)):
                perBlock = get_fp16_mt_config(mt, "perBlock", wg, mipt, lrvw, wt, 1, _K_B128, _TDM)
                pad = get_fp16_mt_config(mt, "pad", wg, mipt, lrvw, wt, 1, _K_B128, _TDM)
                if perBlock == 0:
                    assert pad == 0, (mt, wt, wg, pad)
                    continue
                assert perBlock in _L._LDS_PAD_BLOCK_BYTES, (mt, wt, wg, perBlock)
                padBytes = pad * 2
                assert padBytes % 8 == 0, (mt, wt, wg, padBytes)
                assert 0 < padBytes <= _L._LDS_MAX_PAD_BYTES, (mt, wt, wg, padBytes)


def test_b32_wave_costs_counts_one_bank_per_thread():
    # 32 threads, 4 bytes apart, no padding: banks 0..31, all distinct.
    raw = [4 * i for i in range(32)]
    assert _L._b32_wave_costs(raw, 0, 0, (0,), (0,)) == [1]
    # Every thread 256 bytes apart lands on bank 0.
    raw = [256 * i for i in range(32)]
    assert _L._b32_wave_costs(raw, 0, 0, (0,), (0,)) == [32]


def test_b32_wave_costs_rejects_addresses_below_dword_alignment():
    assert _L._b32_wave_costs([1] * 32, 0, 0, (0,), (0,)) is None


def test_fp32_config_is_legal_over_the_reachable_shapes():
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for vw in (1, 2, 4):
                if wt % vw:
                    continue
                perBlock = get_fp32_mt_config(mt, "perBlock", vw, 2, wg, 2, wt, _K_B32, _TDM)
                pad = get_fp32_mt_config(mt, "pad", vw, 2, wg, 2, wt, _K_B32, _TDM)
                if perBlock == 0:
                    assert pad == 0, (mt, wt, wg, vw, pad)
                    continue
                assert perBlock in _L._LDS_PAD_BLOCK_BYTES, (mt, wt, wg, vw, perBlock)
                padBytes = pad * 4
                assert padBytes % 8 == 0, (mt, wt, wg, vw, padBytes)
                assert 0 < padBytes <= _L._LDS_MAX_PAD_BYTES, (mt, wt, wg, vw, padBytes)


def test_pick_best_ranks_cost_over_overhead():
    # A candidate with lower cost but higher LDS overhead (P / B) must win
    # over one with higher cost but lower overhead. If the key tuple were
    # ever reordered so overhead came before cost, this would flip.
    high_cost_low_overhead = (64, 8)
    low_cost_high_overhead = (16, 8)
    candidates = [high_cost_low_overhead, low_cost_high_overhead]

    def costFn(cand):
        return [5] if cand == high_cost_low_overhead else [1]

    assert _L._pick_best(candidates, costFn) == low_cost_high_overhead


def test_valid_blocks_divide_the_tail_loop_step():
    # The tail loop advances the local read address by one compile-time
    # constant per pass. How much padding a step of incBytes skips is the
    # same from every starting offset only when the block divides incBytes,
    # so a block that does not is never offered.
    for incBytes in (256, 768, 1024, 2048, 192):
        blocks = _L._valid_blocks(incBytes, readBases=(0,), readOffs=(0,),
                                  writeMinBytes=0, writeRowBytes=0)
        for b in blocks:
            assert incBytes % b == 0, (incBytes, b)
        for b in _L._LDS_PAD_BLOCK_BYTES:
            if incBytes % b:
                assert b not in blocks, (incBytes, b)


def test_chosen_block_divides_the_tail_loop_step():
    # Same property, but on what the entry points actually return.
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for getter, bpeDS, k in ((get_fp8_mt_config, 1.0, _K_B64),
                                     (get_fp4_mt_config, 0.5, _K_B64)):
                perBlock = getter(mt, "perBlock", wt, wg, k, _TDM)
                if perBlock:
                    assert int(mt * bpeDS) * k % perBlock == 0, (mt, wt, wg, perBlock)
            perBlock = get_fp16_mt_config(mt, "perBlock", wg, 16, 8, wt, 1,
                                          _K_B128, _TDM)
            if perBlock:
                assert mt * 2 * _K_B128 % perBlock == 0, (mt, wt, wg, perBlock)
            for vw in (1, 2, 4):
                if wt % vw:
                    continue
                perBlock = get_fp32_mt_config(mt, "perBlock", vw, 2, wg, 2, wt, _K_B32, _TDM)
                if perBlock:
                    assert mt * 4 * _K_B32 % perBlock == 0, (mt, wt, wg, vw, perBlock)


def test_chosen_block_is_one_the_validator_accepts():
    # The search picks from _valid_blocks_for and Solution.py holds a
    # hand-written block to the same list. If the two ever read different
    # shapes, a config the solver produced would be rejected as unusable.
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for tdm in (True, False):
                for getter, blocksFn, args in (
                        (get_fp8_mt_config, _L.get_fp8_valid_blocks, (wt, wg, _K_B64, tdm)),
                        (get_fp4_mt_config, _L.get_fp4_valid_blocks, (wt, wg, _K_B64, tdm))):
                    perBlock = getter(mt, "perBlock", *args)
                    if perBlock:
                        assert perBlock in blocksFn(mt, *args), (mt, wt, wg, tdm, perBlock)

                perBlock = get_fp16_mt_config(mt, "perBlock", wg, 16, 8, wt, 1, _K_B128, tdm)
                if perBlock:
                    assert perBlock in _L.get_fp16_valid_blocks(
                        mt, wg, 16, 8, wt, 1, _K_B128, tdm), (mt, wt, wg, tdm, perBlock)

                for vw in (1, 2, 4):
                    if wt % vw:
                        continue
                    perBlock = get_fp32_mt_config(mt, "perBlock", vw, 2, wg, 2, wt, _K_B32, tdm)
                    if perBlock:
                        assert perBlock in _L.get_fp32_valid_blocks(
                            mt, vw, 2, wg, 2, wt, _K_B32, tdm), (mt, wt, wg, vw, tdm, perBlock)


def test_chosen_pad_and_block_are_a_legal_pair():
    # A pad shifts every address past the first block, so it has to carry the
    # load's own alignment: 16 bytes for ds_load_tr16_b128, 8 for the rest.
    # Checking pad and block apart misses a pair that breaks it.
    for wg in _WAVE_GROUPS:
        for wt in _WAVE_TILES:
            mt = 16 * wt * wg
            for tdm in (True, False):
                sh = _L._fp16_shape(mt, wg, 16, 8, wt, 1, _K_B128, tdm)
                block = get_fp16_mt_config(mt, "perBlock", wg, 16, 8, wt, 1, _K_B128, tdm)
                pad = get_fp16_mt_config(mt, "pad", wg, 16, 8, wt, 1, _K_B128, tdm) * 2
                assert _L._b128_wave_costs(sh.rawAddrs, block, pad,
                                           sh.wOffsets, sh.instOffs) is not None, \
                    (mt, wt, wg, tdm, block, pad)

                sh = _L._fp8_shape(mt, wt, wg, _K_B64, tdm)
                block = get_fp8_mt_config(mt, "perBlock", wt, wg, _K_B64, tdm)
                pad = get_fp8_mt_config(mt, "pad", wt, wg, _K_B64, tdm)
                assert _L._b64_wave_costs(sh.rawAddrs, block, pad,
                                          sh.instOffs, sh.wOffsets) is not None, \
                    (mt, wt, wg, tdm, block, pad)


def test_a_pad_below_the_load_alignment_is_illegal():
    # 8 bytes is an even number of dwords, so the even-dword rule lets it
    # through, but ds_load_tr16_b128 needs 16.
    sh = _L._fp16_shape(128, 2, 16, 8, 4, 1, _K_B128, True)
    assert _L._b128_wave_costs(sh.rawAddrs, 256, 8, sh.wOffsets, sh.instOffs) is None
    assert _L._b128_wave_costs(sh.rawAddrs, 256, 16, sh.wOffsets, sh.instOffs) is not None
