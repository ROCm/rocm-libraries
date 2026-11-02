# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the universal GEMM's TDM LDS-padding encode.

The TDM path reproduces the VGPR path's padded LDS row (``lds_k_pad``) by having
the mover insert a gap after every row. The descriptor carries that as two raw
fields, both in dwords: ``pad_interval`` (3 bits, a log2 biased by one) and
``pad_amount`` (7 bits, minus one). A wrong encode shifts every LDS row, which
reads as wrong results rather than an error, so it is pinned here. The C++
engine has its own copy of the encode; both must reject an inexpressible pad
for the same reason.
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

# pytest.ini uses --import-mode=importlib, which does not put this directory on
# sys.path: without the insert the sibling import below only resolves when an
# earlier-collected module happened to add it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_gemm_split_barrier import _spec  # noqa: E402

from rocke.instances.common.gemm_universal import (  # noqa: E402
    _TDM_PAD_AMOUNT_MAX,
    _TDM_PAD_INTERVAL_MAX,
    _encode_tdm_padding,
    _tdm_padding_for_tile,
)

ARCH = "gfx1250"


class TestTdmPaddingEncode(unittest.TestCase):
    def test_interval_is_a_biased_log2_of_dwords(self):
        # pad_interval = log2(interval_bytes / 4) - 1
        for interval_bytes, expected in ((8, 0), (16, 1), (32, 2), (64, 3), (1024, 7)):
            with self.subTest(interval_bytes=interval_bytes):
                self.assertEqual(_encode_tdm_padding(interval_bytes, 4)[0], expected)

    def test_amount_is_dwords_minus_one(self):
        for pad_bytes, expected in ((4, 0), (16, 3), (32, 7), (512, 127)):
            with self.subTest(pad_bytes=pad_bytes):
                self.assertEqual(_encode_tdm_padding(64, pad_bytes)[1], expected)

    def test_bf16_block_k32_pad16_matches_hand_encoding(self):
        # One 32-element bf16 row is 64 B; a 16-element pad is 32 B.
        self.assertEqual(_encode_tdm_padding(64, 32), (3, 7))

    def test_tile_helper_disables_padding_at_pad_zero(self):
        self.assertEqual(_tdm_padding_for_tile(2, 32, 0), (0, 0, 0))
        self.assertEqual(_tdm_padding_for_tile(2, 32, 16), (1, 3, 7))

    def test_rejects_inexpressible_padding(self):
        with self.assertRaises(ValueError):  # not a dword multiple
            _encode_tdm_padding(64, 2)
        with self.assertRaises(ValueError):  # interval not a power of two
            _encode_tdm_padding(12, 16)
        with self.assertRaises(ValueError):  # interval overflows 3 bits
            _encode_tdm_padding(2048, 16)
        with self.assertRaises(ValueError):  # amount overflows 7 bits
            _encode_tdm_padding(64, 1024)

    def test_field_widths_are_the_documented_maxima(self):
        self.assertEqual(
            _encode_tdm_padding(1024, 512), (_TDM_PAD_INTERVAL_MAX, _TDM_PAD_AMOUNT_MAX)
        )


class TestTdmPaddingCppEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import rocke_engine  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"rocke_engine C++ binding not importable: {e}")

    def test_both_engines_judge_lds_k_pad_alike(self):
        """Encodable pads pass in both engines; the others fail for the same reason."""
        import rocke_engine

        from rocke.core.backend import universal_gemm_spec_to_dict
        from rocke.instances.common.gemm_universal import is_valid_spec

        base = _spec()
        for pad in (0, 16, 48, 3, 600):
            with self.subTest(lds_k_pad=pad):
                spec = replace(base, trait=replace(base.trait, lds_k_pad=pad))
                py = is_valid_spec(spec, arch=ARCH)
                cpp = rocke_engine.gemm_is_valid(
                    universal_gemm_spec_to_dict(spec), ARCH
                )
                self.assertEqual(tuple(cpp), tuple(py))


if __name__ == "__main__":
    unittest.main()
