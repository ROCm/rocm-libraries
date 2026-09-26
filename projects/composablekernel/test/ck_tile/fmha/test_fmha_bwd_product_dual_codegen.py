#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression test for the gfx12 BF16 D=128 product-dual codegen path."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_FMHA_EXAMPLE = (_HERE / "../../../example/ck_tile/01_fmha").resolve()
_GENERATE = _FMHA_EXAMPLE / "generate.py"
_ELIGIBLE_GLOB = (
    "fmha_bwd_d128_bf16_batch_"
    "b32x32x128x32x128x32x16x128x128_*_"
    "maxq0_npad_nbias_ndbias_nmask_ndropout_"
    "ndeterministic_ntrload_gfx12.cpp"
)


class TestFmhaBwdProductDualCodegen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _GENERATE.is_file():
            raise unittest.SkipTest(f"generate.py not found at {_GENERATE}")

    def test_gfx12_bf16_d128_emits_product_dual_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    str(_GENERATE),
                    "--targets",
                    "gfx1201",
                    "--api",
                    "bwd",
                    "--receipt",
                    "2",
                    "--optdim",
                    "128",
                    "--output_dir",
                    str(output_dir),
                ],
                cwd=_FMHA_EXAMPLE,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=(
                    "gfx1201 backward codegen failed.\n"
                    f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                ),
            )

            eligible = list(output_dir.glob(_ELIGIBLE_GLOB))
            self.assertEqual(
                len(eligible),
                1,
                msg=f"expected one eligible product-dual blob, found {eligible}",
            )

            kernel_source = eligible[0].read_text()
            self.assertIn("FMHA_PRODUCT_DUAL_DISPATCH", kernel_source)
            self.assertIn("BlockFmhaBwdDQOnlyQMajor", kernel_source)
            self.assertIn("BlockFmhaBwdDQDKDVPipelineKRKTRVRIGLPDKDVOpt", kernel_source)
            self.assertIn("product_dual@", kernel_source)

            api_source = (output_dir / "fmha_bwd_api.cpp").read_text()
            self.assertIn("product_dual_dispatch_", api_source)
            self.assertIn(
                "std::conditional_t<product_dual_dispatch_, void, convert_dq_trait_>",
                api_source,
            )


if __name__ == "__main__":
    unittest.main()
