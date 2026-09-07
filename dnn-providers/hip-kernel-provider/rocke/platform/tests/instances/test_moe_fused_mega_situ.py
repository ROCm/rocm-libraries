# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Alternate gated-activation coverage for the FP8 fused MoE kernel."""

from __future__ import annotations

import unittest

from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.instances.common.moe_fused_mega_fp8 import (
    FusedMegaKernelSpecFp8,
    build_moe_fused_mega_gemm_fp8,
    moe_fused_mega_fp8_signature,
)


class TestMoeFusedMegaSitu(unittest.TestCase):
    def test_default_path_keeps_existing_name_and_has_no_tanh(self) -> None:
        spec = FusedMegaKernelSpecFp8(name="default_activation")
        self.assertEqual(
            spec.kernel_name(),
            "default_activation_moe_fused_mega_fp8_m16n256k32",
        )
        llvm = lower_kernel_to_llvm(
            build_moe_fused_mega_gemm_fp8(spec, arch="gfx950"),
            arch="gfx950",
        )
        self.assertNotIn("@llvm.tanh.f32", llvm)

    def test_situ_path_encodes_parameters_and_emits_both_clips(self) -> None:
        spec = FusedMegaKernelSpecFp8(
            name="alternate_activation",
            activation="situ",
            activation_beta=2.0,
            activation_linear_beta=3.0,
        )
        self.assertTrue(spec.kernel_name().endswith("_situ_b2_lb3"))
        kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
        self.assertEqual(verify(kernel), [])
        llvm = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertNotIn("@llvm.tanh.f32", llvm)
        self.assertGreaterEqual(llvm.count("call float @llvm.exp2.f32"), 3)

    def test_situ_without_linear_clip_emits_one_tanh(self) -> None:
        spec = FusedMegaKernelSpecFp8(
            name="gate_clip_only",
            activation="situ",
            activation_beta=2.0,
        )
        self.assertTrue(spec.kernel_name().endswith("_situ_b2_lbnone"))
        llvm = lower_kernel_to_llvm(
            build_moe_fused_mega_gemm_fp8(spec, arch="gfx950"),
            arch="gfx950",
        )
        self.assertNotIn("@llvm.tanh.f32", llvm)
        self.assertGreaterEqual(llvm.count("call float @llvm.exp2.f32"), 2)

    def test_invalid_activation_parameters_are_rejected(self) -> None:
        bad_specs = (
            FusedMegaKernelSpecFp8(name="bad", activation="other"),
            FusedMegaKernelSpecFp8(name="bad", activation="situ", activation_beta=0.0),
            FusedMegaKernelSpecFp8(
                name="bad",
                activation="situ",
                activation_beta=1.0,
                activation_linear_beta=-1.0,
            ),
        )
        for spec in bad_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")

    def test_mxfp4_python_path_builds_with_packed_weight_abi(self) -> None:
        spec = FusedMegaKernelSpecFp8(
            name="packed_weights",
            gate_up_k=32,
            down_k=32,
            use_dtla=False,
            activation="situ",
            activation_beta=2.0,
            activation_linear_beta=3.0,
            weight_dtype="mxfp4",
        )
        signature = {
            entry["name"]: entry["type"] for entry in moe_fused_mega_fp8_signature(spec)
        }
        for name in (
            "WGate",
            "WUp",
            "WDown",
            "WGateScale",
            "WUpScale",
            "WDownScale",
        ):
            self.assertEqual(signature[name], "ptr<i8, global>")
        kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
        self.assertEqual(verify(kernel), [])
        llvm = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("mfma.f32.16x16x32.fp8.fp8", llvm)
        self.assertIn("load <4 x i8>", llvm)

    def test_mxfp4_rejects_incompatible_wide_atom_path(self) -> None:
        spec = FusedMegaKernelSpecFp8(
            name="bad_packed_weights",
            activation="situ",
            weight_dtype="mxfp4",
        )
        with self.assertRaisesRegex(ValueError, "gate_up_k=32"):
            build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")

    def test_native_mxfp4_uses_scaled_a8w4_intrinsic(self) -> None:
        spec = FusedMegaKernelSpecFp8(
            name="native_packed_weights",
            gate_up_k=128,
            down_k=128,
            use_dtla=False,
            activation="situ",
            activation_beta=2.0,
            activation_linear_beta=3.0,
            weight_dtype="mxfp4",
            mxfp4_native=True,
        )
        signature = {
            entry["name"]: entry["type"] for entry in moe_fused_mega_fp8_signature(spec)
        }
        self.assertEqual(signature["MxScaleA"], "i32")
        self.assertTrue(spec.kernel_name().endswith("_mxfp4_native"))
        kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
        self.assertEqual(verify(kernel), [])
        llvm = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4", llvm)
        self.assertIn("i32 0, i32 4, i32 0", llvm)
        self.assertIn("load <16 x i8>", llvm)
        self.assertIn("load i8", llvm)

    def test_native_mxfp4_requires_packed_weights(self) -> None:
        spec = FusedMegaKernelSpecFp8(
            name="bad_native_weights",
            mxfp4_native=True,
        )
        with self.assertRaisesRegex(ValueError, "weight_dtype='mxfp4'"):
            build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")

    def test_native_mxfp4_optimized_layout_flags_build(self) -> None:
        spec = FusedMegaKernelSpecFp8(
            name="optimized_native_weights",
            gate_up_k=128,
            down_k=128,
            warp_n=8,
            use_dtla=False,
            activation="situ",
            weight_dtype="mxfp4",
            mxfp4_native=True,
            prefetch_routing_meta=True,
            pipeline_native_down=True,
            mxfp4_preshuffled=True,
            pipeline_native_gateup=True,
        )
        self.assertTrue(spec.kernel_name().endswith("_mxfp4_native_rmeta_dp2_ps_gp2"))
        kernel = build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
        self.assertEqual(verify(kernel), [])
        llvm = lower_kernel_to_llvm(kernel, arch="gfx950")
        self.assertIn("sdiv i32", llvm)
        self.assertIn("srem i32", llvm)

    def test_native_only_flags_reject_legacy_weight_path(self) -> None:
        for field in (
            "pipeline_native_down",
            "mxfp4_preshuffled",
            "pipeline_native_gateup",
        ):
            spec = FusedMegaKernelSpecFp8(
                name="bad_native_flag",
                weight_dtype="mxfp4",
                gate_up_k=32,
                down_k=32,
                use_dtla=False,
                **{field: True},
            )
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "requires mxfp4_native"
            ):
                build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")


if __name__ == "__main__":
    unittest.main()
