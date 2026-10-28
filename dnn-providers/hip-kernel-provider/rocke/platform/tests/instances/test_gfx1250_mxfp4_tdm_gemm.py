# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the configurable gfx1250 TDM/LDS MX FP4 GEMM."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm import run_case
from rocke.instances.gfx1250.mxfp4_tdm_gemm import (
    MxFp4TdmGemmSpec,
    build_mxfp4_tdm_gemm,
    is_valid_spec,
)
from rocke.instances.gfx1250.mxfp4_tdm_gemm_variants import (
    BASE_VARIANT,
    MXFP4_GEMM_VARIANTS,
    assert_variants_cumulative,
    variant_names,
    variant_spec,
)
from rocke.runtime.hip_module import get_device_arch


class TestGfx1250MxFp4TdmGemm(unittest.TestCase):
    def test_geometry_and_resource_validation(self):
        from rocke.core.arch import ArchTarget

        vgpr_cap = ArchTarget.from_gfx("gfx1250").limits.vgprs
        self.assertEqual(vgpr_cap, 1024)

        spec = MxFp4TdmGemmSpec(name="valid", M=128, N=64, K=256)
        self.assertEqual((spec.wave_repeat_m, spec.wave_repeat_n), (2, 2))
        self.assertEqual(spec.block_size, 128)
        self.assertEqual(spec.accumulator_values_per_lane, 64)
        self.assertEqual(spec.lds_bytes, 24576)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))

        # YAML MacroTile with SourceSwap: 512 FP32 acc / lane fits MaxVgpr=1024.
        yaml_ss = MxFp4TdmGemmSpec(
            name="yaml_ss",
            M=256,
            N=256,
            K=512,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        self.assertEqual(yaml_ss.accumulator_values_per_lane, 512)
        self.assertLessEqual(yaml_ss.lds_bytes, 327680)
        self.assertEqual(is_valid_spec(yaml_ss), (True, "ok"))

        # Accumulators alone past MaxVgpr (LDS kept under the 320 KiB cap).
        over_vgpr = MxFp4TdmGemmSpec(
            name="over_vgpr",
            M=512,
            N=512,
            K=128,
            tile_m=512,
            tile_n=512,
            tile_k=128,
        )
        ok, reason = is_valid_spec(over_vgpr)
        self.assertFalse(ok)
        self.assertEqual(over_vgpr.accumulator_values_per_lane, 2048)
        self.assertIn("2048 FP32 values per lane", reason)
        self.assertIn("1024 MaxVgpr", reason)

    def test_step5_accepted_geometries_pass_resource_checks(self):
        from rocke.core.arch import ArchTarget

        vgpr_cap = ArchTarget.from_gfx("gfx1250").limits.vgprs
        cases = (
            MxFp4TdmGemmSpec(
                name="grow_m",
                M=256,
                N=64,
                K=512,
                tile_m=256,
                tile_n=64,
                tile_k=512,
            ),
            MxFp4TdmGemmSpec(
                name="grow_n",
                M=128,
                N=128,
                K=512,
                tile_m=128,
                tile_n=128,
                tile_k=512,
            ),
            MxFp4TdmGemmSpec(
                name="waves_4x1",
                M=128,
                N=64,
                K=256,
                warp_m=4,
                warp_n=1,
            ),
            MxFp4TdmGemmSpec(
                name="waves_4x2",
                M=256,
                N=64,
                K=256,
                tile_m=256,
                tile_n=64,
                warp_m=4,
                warp_n=2,
            ),
            MxFp4TdmGemmSpec(
                name="yaml_mt256",
                M=256,
                N=256,
                K=512,
                tile_m=256,
                tile_n=256,
                tile_k=512,
            ),
        )
        for spec in cases:
            with self.subTest(name=spec.name):
                self.assertLessEqual(spec.accumulator_values_per_lane, vgpr_cap)
                self.assertEqual(is_valid_spec(spec), (True, "ok"))

    def test_invalid_geometry_is_rejected(self):
        cases = (
            (
                MxFp4TdmGemmSpec(name="bad", M=128, N=64, K=256, warp_k=2),
                "warp_k must be 1",
            ),
            (
                MxFp4TdmGemmSpec(name="bad", M=128, N=64, K=384, tile_k=256),
                "M/N/K must divide",
            ),
            (
                MxFp4TdmGemmSpec(
                    name="bad",
                    M=128,
                    N=64,
                    K=256,
                    tdm_prefetch_depth=3,
                ),
                "tdm_plus_lds_buf",
            ),
            (
                MxFp4TdmGemmSpec(
                    name="bad",
                    M=128,
                    N=64,
                    K=256,
                    lds_pad=True,
                ),
                "not implemented yet",
            ),
            (
                MxFp4TdmGemmSpec(
                    name="bad",
                    M=128,
                    N=64,
                    K=256,
                    scale_tdm=True,
                ),
                "scale_tdm requires scale_double_buffer",
            ),
        )
        for spec, message in cases:
            with self.subTest(message=message):
                ok, reason = is_valid_spec(spec)
                self.assertFalse(ok)
                self.assertIn(message, reason)

    def test_variant_registry_is_cumulative(self):
        assert_variants_cumulative()
        self.assertEqual(variant_names()[0], BASE_VARIANT)
        self.assertEqual(MXFP4_GEMM_VARIANTS[0][1], {})
        base = variant_spec(BASE_VARIANT, 128, 64, 256)
        self.assertEqual(base.name, BASE_VARIANT)
        self.assertFalse(base.local_read_prefetch)
        self.assertEqual(is_valid_spec(base), (True, "ok"))
        default = MxFp4TdmGemmSpec(name="mxfp4_tdm_gemm", M=128, N=64, K=256)
        base_llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(base, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        default_llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(default, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertEqual(
            base_llvm.replace(BASE_VARIANT, "mxfp4_tdm_gemm"),
            default_llvm,
        )

    def test_step6_1_local_read_prefetch_emits_and_validates(self):
        self.assertIn("step6_1_mxfp4_gemm", variant_names())
        spec = variant_spec(
            "step6_1_mxfp4_gemm",
            128,
            64,
            256,
            tile_k=256,
        )
        self.assertTrue(spec.local_read_prefetch)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn("load <4 x i32>, ptr addrspace(3)", llvm)
        self.assertIn("llvm.amdgcn.wmma.scale.f32.32x16x128.f4", llvm)

    def test_step6_2_prefetch_gl2_emits_intrinsic(self):
        spec = variant_spec(
            "step6_2_mxfp4_gemm",
            128,
            64,
            512,
            tile_k=128,
        )
        self.assertTrue(spec.prefetch_gl2)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn("llvm.amdgcn.global.prefetch", llvm)

    def test_step6_3_scale_double_buffer_validates(self):
        self.assertIn("step6_3_mxfp4_gemm", variant_names())
        spec = variant_spec(
            "step6_3_mxfp4_gemm",
            128,
            64,
            512,
            tile_k=512,
        )
        self.assertTrue(spec.scale_double_buffer)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn("llvm.amdgcn.wmma.scale.f32.32x16x128.f4", llvm)
        # Prefetched scale packs still originate as global i8 loads.
        self.assertIn("load <4 x i8>, ptr addrspace(1)", llvm)

    def test_step6_4_wave_separated_tdm_emits_and_validates(self):
        self.assertIn("step6_4_mxfp4_gemm", variant_names())
        spec = variant_spec(
            "step6_4_mxfp4_gemm",
            128,
            64,
            256,
        )
        self.assertTrue(spec.wave_separated_tdm)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        base = variant_spec("step6_3_mxfp4_gemm", 128, 64, 256)
        llvm_sep = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        llvm_base = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(base, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertGreaterEqual(llvm_sep.count("llvm.amdgcn.tensor.load.to.lds"), 4)
        # Wave-0-only path guards TDM; wave-separated issues without that gate.
        self.assertNotEqual(llvm_sep, llvm_base)

    def test_step6_5_scale_tdm_grows_lds_and_emits(self):
        self.assertIn("step6_5_mxfp4_gemm", variant_names())
        spec = variant_spec(
            "step6_5_mxfp4_gemm",
            128,
            64,
            512,
            tile_k=512,
        )
        self.assertTrue(spec.scale_tdm)
        self.assertTrue(spec.scale_double_buffer)
        prior = variant_spec(
            "step6_4_mxfp4_gemm",
            128,
            64,
            512,
            tile_k=512,
        )
        self.assertGreater(spec.lds_bytes, prior.lds_bytes)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        yaml = variant_spec(
            "step6_5_mxfp4_gemm",
            256,
            256,
            512,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        self.assertLessEqual(yaml.lds_bytes, 327680)
        self.assertEqual(is_valid_spec(yaml), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        # A/B data + A/B scale tiles: at least 8 TDM issues across prefill+loop.
        self.assertGreaterEqual(llvm.count("llvm.amdgcn.tensor.load.to.lds"), 8)
        self.assertIn("load <4 x i8>, ptr addrspace(3)", llvm)

    def test_llvm_contains_double_buffered_tdm_lds_wmma_and_fp8_epilogue(self):
        spec = MxFp4TdmGemmSpec(name="llvm", M=128, N=64, K=256)
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertGreaterEqual(llvm.count("llvm.amdgcn.tensor.load.to.lds"), 4)
        self.assertGreaterEqual(llvm.count("llvm.amdgcn.s.wait.tensorcnt"), 2)
        self.assertIn("llvm.amdgcn.wmma.scale.f32.32x16x128.f4", llvm)
        self.assertIn("load <4 x i32>, ptr addrspace(3)", llvm)
        self.assertIn("llvm.amdgcn.cvt.f32.fp8", llvm)
        self.assertIn("llvm.amdgcn.cvt.pk.fp8.f32", llvm)

    def test_fp8_result_matches_reference_on_gfx1250(self):
        if get_device_arch(0) != "gfx1250":
            self.skipTest("visible HIP device 0 is not gfx1250")
        run_case(
            MxFp4TdmGemmSpec(
                name="numeric",
                M=64,
                N=32,
                K=256,
                tile_m=64,
                tile_n=32,
                warp_m=1,
                warp_n=1,
            )
        )


if __name__ == "__main__":
    unittest.main()
