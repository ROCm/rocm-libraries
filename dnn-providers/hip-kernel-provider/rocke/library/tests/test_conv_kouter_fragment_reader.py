# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Capability gate of :class:`ConvKOuterFragmentReader`.

Host-only: construction resolves the arch and checks for the transpose-LDS
read that the reader's wave regime emits. No IR is built here; the emitted
fragment stream is pinned by the conv IR goldens.
"""

from __future__ import annotations

import unittest

from kernels.common._conv_layouts import ConvKOuterFragmentReader


class TestConvKOuterFragmentReaderGate(unittest.TestCase):
    def test_supported_targets_construct(self) -> None:
        # gfx950: wave64 ds_read_b64_tr_b16; gfx1250: wave32 ds_load_tr16_b128.
        ConvKOuterFragmentReader(wave_size=64, arch="gfx950")
        ConvKOuterFragmentReader(wave_size=32, arch="gfx1250")

    def test_wave64_without_transpose_read_is_refused(self) -> None:
        # gfx942 is wave64 but has no ds_read_b64_tr_b16; wave size alone
        # must not select the gfx950 recipe.
        with self.assertRaisesRegex(ValueError, "gfx942.*has_ds_read_tr"):
            ConvKOuterFragmentReader(wave_size=64, arch="gfx942")

    def test_wave32_without_wide_transpose_read_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "gfx1201.*ds_load_tr16_b128"):
            ConvKOuterFragmentReader(wave_size=32, arch="gfx1201")

    def test_wave_size_must_match_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match gfx950"):
            ConvKOuterFragmentReader(wave_size=32, arch="gfx950")


if __name__ == "__main__":
    unittest.main()
