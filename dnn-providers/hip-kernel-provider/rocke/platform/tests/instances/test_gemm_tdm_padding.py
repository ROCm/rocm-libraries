# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the universal GEMM's TDM LDS padding.

The TDM path reproduces the VGPR path's padded LDS row (``lds_k_pad``) by having
the mover insert a gap after every row: ``_tdm_padding_for_tile`` turns one tile
row and the pad into the descriptor fields through the core
``encode_tdm_padding`` (tested in ``tests/core/test_gfx1250_tdm.py``). Both
engines must reject an inexpressible pad for the same reason.
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

from rocke.instances.common.gemm_universal import _tdm_padding_for_tile  # noqa: E402

ARCH = "gfx1250"


class TestTdmPaddingForTile(unittest.TestCase):
    def test_bf16_block_k32_pad16_matches_hand_encoding(self):
        # One 32-element bf16 row is 64 B; a 16-element pad is 32 B.
        self.assertEqual(_tdm_padding_for_tile(2, 32, 16), (1, 3, 7))

    def test_pad_zero_disables_padding(self):
        self.assertEqual(_tdm_padding_for_tile(2, 32, 0), (0, 0, 0))

    def test_rejects_inexpressible_padding(self):
        with self.assertRaisesRegex(ValueError, "pad_amount 255 exceeds 127"):
            _tdm_padding_for_tile(2, 32, 512)


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
