# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Field-level tests for the gfx1250 TDM descriptor builder.

A misplaced bit in the descriptor produces silent garbage or a hang, never a
compile error, so the packing is pinned here field by field against the layout
in CK's ``amd_tdm_descriptor.hpp``.
"""

from __future__ import annotations

import unittest

from rocke.core.tdm import (
    TDM_PAD_AMOUNT_MAX,
    TDM_PAD_INTERVAL_MAX,
    encode_tdm_padding,
    pack_tdm_group0,
    pack_tdm_group1_2d,
    tdm_data_size_code,
    tdm_padding_for_tile,
)


def _field(words, word, lsb, width):
    return (words[word] >> lsb) & ((1 << width) - 1)


class TestTdmPadding(unittest.TestCase):
    def test_interval_is_a_biased_log2_of_dwords(self):
        # pad_interval = log2(interval_bytes / 4) - 1
        for interval_bytes, expected in ((8, 0), (16, 1), (32, 2), (64, 3), (1024, 7)):
            with self.subTest(interval_bytes=interval_bytes):
                self.assertEqual(encode_tdm_padding(interval_bytes, 4)[0], expected)

    def test_amount_is_dwords_minus_one(self):
        for pad_bytes, expected in ((4, 0), (16, 3), (32, 7), (512, 127)):
            with self.subTest(pad_bytes=pad_bytes):
                self.assertEqual(encode_tdm_padding(64, pad_bytes)[1], expected)

    def test_bf16_block_k32_pad16_matches_hand_encoding(self):
        # One 32-element bf16 row is 64 B; a 16-element pad is 32 B.
        self.assertEqual(encode_tdm_padding(64, 32), (3, 7))

    def test_tile_helper_disables_padding_at_pad_zero(self):
        self.assertEqual(tdm_padding_for_tile(2, 32, 0), (0, 0, 0))
        self.assertEqual(tdm_padding_for_tile(2, 32, 16), (1, 3, 7))

    def test_rejects_inexpressible_padding(self):
        with self.assertRaises(ValueError):  # not a dword multiple
            encode_tdm_padding(64, 2)
        with self.assertRaises(ValueError):  # interval not a power of two
            encode_tdm_padding(12, 16)
        with self.assertRaises(ValueError):  # interval overflows 3 bits
            encode_tdm_padding(2048, 16)
        with self.assertRaises(ValueError):  # amount overflows 7 bits
            encode_tdm_padding(64, 1024)

    def test_field_widths_are_the_documented_maxima(self):
        self.assertEqual(encode_tdm_padding(1024, 512), (TDM_PAD_INTERVAL_MAX, TDM_PAD_AMOUNT_MAX))


class TestTdmGroup0(unittest.TestCase):
    def test_constant_fields(self):
        words = pack_tdm_group0(lds_addr=0, global_addr=0)
        self.assertEqual(_field(words, 0, 0, 2), 1)  # count
        self.assertEqual(_field(words, 3, 30, 2), 2)  # type ("2 for spg")
        self.assertEqual(_field(words, 0, 31, 1), 0)  # gather_mode

    def test_addresses_split_across_words(self):
        words = pack_tdm_group0(lds_addr=0x1234, global_addr=0x1F_0011_2233)
        self.assertEqual(words[1], 0x1234)
        self.assertEqual(words[2], 0x00112233)
        self.assertEqual(_field(words, 3, 0, 25), 0x1F)

    def test_type_and_global_addr_hi_share_word_three(self):
        words = pack_tdm_group0(lds_addr=0, global_addr=0xFF_0000_0000)
        self.assertEqual(_field(words, 3, 0, 25), 0xFF)
        self.assertEqual(_field(words, 3, 30, 2), 2)


class TestTdmGroup1(unittest.TestCase):
    def _pack(self, **kwargs):
        args = dict(
            elem_bytes=2,
            tensor_dim0=4096,
            tensor_dim1=1024,
            tile_dim0=32,
            tile_dim1=128,
            dim0_stride=4096,
            dim1_stride=1,
        )
        args.update(kwargs)
        return pack_tdm_group1_2d(**args)

    def test_data_size_code(self):
        for elem_bytes, code in ((1, 0), (2, 1), (4, 2), (8, 3)):
            with self.subTest(elem_bytes=elem_bytes):
                self.assertEqual(tdm_data_size_code(elem_bytes), code)
                self.assertEqual(_field(self._pack(elem_bytes=elem_bytes), 0, 16, 2), code)
        with self.assertRaises(ValueError):
            tdm_data_size_code(3)

    def test_extents_split_across_word_boundaries(self):
        words = self._pack(tensor_dim0=0x1234_5678, tensor_dim1=0x9ABC_DEF0)
        self.assertEqual(_field(words, 1, 16, 16), 0x5678)
        self.assertEqual(_field(words, 2, 0, 16), 0x1234)
        self.assertEqual(_field(words, 2, 16, 16), 0xDEF0)
        self.assertEqual(_field(words, 3, 0, 16), 0x9ABC)

    def test_tile_dims_and_row_pitch(self):
        words = self._pack(tile_dim0=64, tile_dim1=256, dim0_stride=8192)
        self.assertEqual(_field(words, 3, 16, 16), 64)
        self.assertEqual(_field(words, 4, 0, 16), 256)
        self.assertEqual(_field(words, 4, 16, 16), 0)  # tile_dim2 unused at rank 2
        # Stride slot 0 is the pitch from one dim0 row to the next; measured on
        # hardware, the observed row step equals this word exactly.
        self.assertEqual(words[5], 8192)

    def test_padding_fields(self):
        words = self._pack(pad_enable=1, pad_interval=3, pad_amount=7)
        self.assertEqual(_field(words, 0, 20, 1), 1)
        self.assertEqual(_field(words, 0, 22, 3), 3)
        self.assertEqual(_field(words, 0, 25, 7), 7)
        self.assertEqual(_field(self._pack(), 0, 20, 1), 0)

    def test_iterate_and_barrier_stay_off(self):
        words = self._pack()
        self.assertEqual(_field(words, 0, 18, 1), 0)  # atomic_barrier_enable
        self.assertEqual(_field(words, 0, 19, 1), 0)  # iterate_enable
        self.assertEqual(_field(words, 0, 21, 1), 0)  # early_timeout

    def test_rejects_oversized_fields(self):
        with self.assertRaises(ValueError):
            self._pack(tile_dim0=1 << 16)
        with self.assertRaises(ValueError):
            # dim1_stride lands in a 16-bit field that silently truncates.
            self._pack(dim1_stride=1 << 17)


if __name__ == "__main__":
    unittest.main()
