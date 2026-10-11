# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The ISA check: packed-FP32 instructions beside MFMA are rejected (the gfx950
erratum pattern); either alone passes."""

import sys
import unittest
from pathlib import Path

TRITON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRITON_DIR))

from generators import _contract as C  # noqa: E402
from generators._codeobject import (  # noqa: E402
    GeneratorError,
    check_erratum_text,
    isa_counts,
)


class IsaTextCheckerTest(unittest.TestCase):
    def test_pk_f32_beside_mfma_rejected(self):
        text = "\tv_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], 0\n\tv_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7]\n"
        with self.assertRaises(GeneratorError) as raised:
            check_erratum_text(text, "text")
        self.assertTrue(str(raised.exception).startswith(C.ERR_ERRATUM))

    def test_each_alone_passes(self):
        check_erratum_text("\tv_pk_fma_f32 v[0:1], v[2:3], v[4:5], v[6:7]\n", "text")
        check_erratum_text(
            "\tv_mfma_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], 0\n", "text"
        )

    def test_packed_f32_variants_counted(self):
        for op in ("v_pk_fma_f32", "v_pk_mul_f32", "v_pk_add_f32"):
            self.assertEqual(isa_counts(f"v_mfma_x\n{op} v0\n"), (1, 1))
        self.assertEqual(isa_counts("v_mfma_x\nv_pk_mov_b32 v0\n"), (1, 0))

    def test_smfmac_counts_as_mfma(self):
        text = "\tv_smfmac_f32_16x16x32_bf16 v[0:3], v[4:7], v[8:11], v12\n"
        self.assertEqual(isa_counts(text), (1, 0))
        with self.assertRaises(GeneratorError):
            check_erratum_text(text + "\tv_pk_mul_f32 v[0:1], v[2:3], v[4:5]\n", "t")


if __name__ == "__main__":
    unittest.main()
