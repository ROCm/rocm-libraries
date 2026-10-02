#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the gfx12 BF16 D=64/128 product-dual codegen paths."""

import re
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
    dim = 128
    eligible_glob = _ELIGIBLE_GLOB

    @classmethod
    def setUpClass(cls):
        if not _GENERATE.is_file():
            raise unittest.SkipTest(f"generate.py not found at {_GENERATE}")

    def test_gfx12_bf16_emits_product_dual_dispatch(self):
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
                    str(self.dim),
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

            eligible = list(output_dir.glob(self.eligible_glob))
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
            self.assertIn("std::is_same_v<ck_tile::gfx12_t, ck_tile::gfx12_t>", kernel_source)

            # The host workspace helpers use the DKDV alias. Only that wrapper
            # receives the product predicate; dQ still writes its output directly.
            idx = re.search(r"using fmha_bwd_dkdv_kernel_(\d+) =", kernel_source).group(
                1
            )
            dkdv_alias = kernel_source.split(f"using fmha_bwd_dkdv_kernel_{idx} =", 1)[
                1
            ].split(";", 1)[0]
            self.assertIn(f"fmha_bwd_product_dual_{idx}>", dkdv_alias)
            dq_alias = kernel_source.split(f"using fmha_bwd_dq_kernel_{idx} =", 1)[
                1
            ].split(";", 1)[0]
            self.assertNotIn(f"fmha_bwd_product_dual_{idx}", dq_alias)
            self.assertIn(f"fmha_bwd_dkdv_kernel_{idx}::kNoDqWorkspace", kernel_source)
            self.assertIn(
                f"!fmha_bwd_dkdv_kernel_{idx}::NeedsZeroDqAcc()", kernel_source
            )

            # These specializations must retain a false workspace-skip predicate.
            # Check actual emitted guards, not just the generator template.
            for suffix, guard in (
                (
                    "fp16_batch_*_npad_nbias_ndbias_nmask_ndropout_ndeterministic_ntrload_gfx12.cpp",
                    "FmhaBwdBf16>",
                ),
                (
                    "bf16_batch_*_npad_nbias_ndbias_nmask_ndropout_deterministic_ntrload_gfx12.cpp",
                    "!(true)",
                ),
                (
                    "bf16_group_*_npad_nbias_ndbias_nmask_ndropout_ndeterministic_ntrload_gfx12.cpp",
                    "!(true)",
                ),
            ):
                with self.subTest(fallback=suffix):
                    candidates = list(
                        output_dir.glob(f"fmha_bwd_d{self.dim}_" + suffix)
                    )
                    self.assertTrue(candidates, suffix)
                    source = candidates[0].read_text()
                    predicate = source.split(
                        "static constexpr bool fmha_bwd_product_dual_", 1
                    )[1].split(";", 1)[0]
                    self.assertIn(guard, predicate)
                    if suffix.startswith("fp16"):
                        self.assertRegex(source, r"using fmha_dtype_\d+ = FmhaBwdFp16;")

            api_source = (output_dir / "fmha_bwd_api.cpp").read_text()
            self.assertIn("product_dual_dispatch_", api_source)
            self.assertIn(
                "std::conditional_t<product_dual_dispatch_, void, convert_dq_trait_>",
                api_source,
            )


class TestFmhaBwdD64ProductDualCodegen(TestFmhaBwdProductDualCodegen):
    dim = 64
    eligible_glob = (
        "fmha_bwd_d64_bf16_batch_"
        "b32x64x64x32x64x32x32x64x64_*_"
        "maxq0_npad_nbias_ndbias_nmask_ndropout_"
        "ndeterministic_ntrload_gfx12.cpp"
    )


if __name__ == "__main__":
    unittest.main()
