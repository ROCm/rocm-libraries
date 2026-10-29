# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Coverage for the configurable gfx1250 TDM/LDS MX FP4 GEMM."""

from __future__ import annotations

import dataclasses
import re
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
    MXFP4_GEMM_BLOCKED_VARIANTS,
    MXFP4_GEMM_VARIANTS,
    assert_variants_cumulative,
    variant_flags,
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
                    cluster_dim=(4, 4, 1),
                ),
                "must be divisible by cluster_dim",
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
            (
                MxFp4TdmGemmSpec(
                    name="bad",
                    M=128,
                    N=64,
                    K=256,
                    lds_segment_b_rotate=True,
                ),
                "lds_segment_b_rotate requires lds_segment_interleave",
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

    def test_step6_6_lds_pad_grows_stride_and_validates(self):
        self.assertIn("step6_6_mxfp4_gemm", variant_names())
        spec = variant_spec(
            "step6_6_mxfp4_gemm",
            128,
            64,
            512,
            tile_k=512,
        )
        self.assertTrue(spec.lds_pad)
        self.assertEqual(spec.a_lds_row_stride_bytes, spec.tile_k // 2 + 16)
        self.assertEqual(spec.b_lds_row_stride_bytes, spec.tile_k // 2 + 16)
        prior = variant_spec(
            "step6_5_mxfp4_gemm",
            128,
            64,
            512,
            tile_k=512,
        )
        self.assertGreater(spec.lds_bytes, prior.lds_bytes)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        yaml = variant_spec(
            "step6_6_mxfp4_gemm",
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
        self.assertIn("llvm.amdgcn.tensor.load.to.lds", llvm)

    def test_step6_7_bcontig_segment_layout_variant(self):
        self.assertIn("step6_7_mxfp4_gemm", variant_names())
        smoke = variant_spec("step6_7_mxfp4_gemm", 128, 64, 512, tile_k=512)
        self.assertTrue(smoke.lds_segment_interleave)
        self.assertFalse(smoke.lds_segment_b_rotate)
        layout = smoke.segment_interleave_layout
        self.assertEqual(layout.component_rows, 64)
        self.assertEqual(layout.a_component_stride_bytes, 65536)
        self.assertEqual(smoke.lds_bytes, 172032)
        self.assertEqual(is_valid_spec(smoke), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(smoke, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn("llvm.amdgcn.wmma.scale.f32.32x16x128.f4", llvm)
        self.assertIn("65536", llvm)
        yaml7 = variant_spec(
            "step6_7_mxfp4_gemm",
            256,
            256,
            512,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        layout = yaml7.segment_interleave_layout
        self.assertEqual(layout.component_rows, 128)
        self.assertEqual(layout.a_component_stride_bytes, 104448)
        self.assertEqual(layout.b_base_bytes, 34816)
        self.assertEqual(layout.a_scale_base_bytes, 139264)
        self.assertEqual(layout.b_scale_base_bytes, 143360)
        self.assertEqual(layout.buffer_bytes, 147456)
        self.assertEqual(yaml7.lds_bytes, 294912)
        self.assertEqual(is_valid_spec(yaml7), (True, "ok"))
        yaml_llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(yaml7, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        for value in ("104448", "139264", "143360", "147456"):
            self.assertIn(value, yaml_llvm)
        rotate = MxFp4TdmGemmSpec(
            name="rotate",
            M=128,
            N=64,
            K=512,
            tile_k=512,
            **{**variant_flags("step6_7_mxfp4_gemm"), "lds_segment_b_rotate": True},
        )
        ok, reason = is_valid_spec(rotate)
        self.assertFalse(ok)
        self.assertIn("bcontig", reason)

    def test_step6_8_b_rotate_is_dropped_from_the_chain(self):
        self.assertNotIn("step6_8_mxfp4_gemm", variant_names())
        self.assertIn(
            "step6_8_mxfp4_gemm",
            [name for name, _ in MXFP4_GEMM_BLOCKED_VARIANTS],
        )
        with self.assertRaisesRegex(ValueError, "blocked"):
            variant_flags("step6_8_mxfp4_gemm")
        for name, flags in MXFP4_GEMM_VARIANTS:
            with self.subTest(name=name):
                self.assertNotIn("lds_segment_b_rotate", flags)

    def test_step6_9_tdm_plus_lds_buf_depth(self):
        self.assertIn("step6_9_mxfp4_gemm", variant_names())
        # Cumulative flags include interleave; the smoke buffer fits depth 3.
        smoke = variant_spec("step6_9_mxfp4_gemm", 128, 64, 512, tile_k=512)
        self.assertTrue(smoke.tdm_plus_lds_buf)
        self.assertFalse(smoke.lds_segment_b_rotate)
        self.assertEqual(smoke.effective_prefetch_depth, 3)
        self.assertEqual(is_valid_spec(smoke), (True, "ok"))
        # Without interleave, depth 3 still requires LDS headroom.
        depth3 = MxFp4TdmGemmSpec(
            name="depth3",
            M=128,
            N=64,
            K=512,
            tile_k=512,
            tdm_plus_lds_buf=True,
        )
        self.assertEqual(depth3.effective_prefetch_depth, 3)
        self.assertEqual(is_valid_spec(depth3), (True, "ok"))
        # YAML MacroTile: three bcontig buffers exceed the LDS cap, so the
        # variant stays at depth 2 and remains valid.
        yaml = variant_spec(
            "step6_9_mxfp4_gemm",
            256,
            256,
            512,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        self.assertEqual(yaml.effective_prefetch_depth, 2)
        self.assertEqual(is_valid_spec(yaml), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(smoke, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn("db3", smoke.kernel_name())
        self.assertIn("llvm.amdgcn.tensor.load.to.lds", llvm)

    def test_step6_10_init_c_wmma_emits_undef_seed(self):
        self.assertIn("step6_10_mxfp4_gemm", variant_names())
        spec = variant_spec("step6_10_mxfp4_gemm", 128, 64, 512, tile_k=512)
        self.assertTrue(spec.init_c_wmma)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn("undef", llvm)
        self.assertIn("initc", spec.kernel_name())
        self.assertIn("llvm.amdgcn.wmma.scale.f32.32x16x128.f4", llvm)
        # The first K tile is peeled, so no accumulator select remains.
        self.assertIsNone(re.search(r"select i1 %\S+, <\d+ x float>", llvm))

    def test_step6_11_cluster_dim_requires_grid_and_emits(self):
        self.assertIn("step6_11_mxfp4_gemm", variant_names())
        # Default smoke geometry yields grid (1,1); cluster 4x4 needs M=N=512.
        too_small = variant_spec("step6_11_mxfp4_gemm", 128, 64, 256)
        ok, reason = is_valid_spec(too_small)
        self.assertFalse(ok)
        self.assertIn("cluster_dim", reason)
        spec = variant_spec(
            "step6_11_mxfp4_gemm",
            512,
            512,
            128,
            tile_m=128,
            tile_n=64,
            tile_k=128,
        )
        self.assertEqual(spec.cluster_dim, (4, 4, 1))
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn('amdgpu-cluster-dims"="4,4,1"', llvm)
        self.assertIn("clu", spec.kernel_name())

    def test_step6_12_b_scale_n_major_uses_word_scale_loads(self):
        self.assertIn("step6_12_mxfp4_gemm", variant_names())
        yaml = variant_spec(
            "step6_12_mxfp4_gemm",
            4096,
            4096,
            65536,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        self.assertTrue(yaml.b_scale_n_major)
        self.assertIn("bsn", yaml.kernel_name())
        self.assertEqual(is_valid_spec(yaml), (True, "ok"))
        # Same LDS footprint: the B-scale tile is only transposed.
        row_major = dataclasses.replace(yaml, b_scale_n_major=False)
        self.assertEqual(yaml.lds_bytes, row_major.lds_bytes)

        def lds_byte_loads(spec):
            llvm = _lower_kernel_to_llvm_python(
                build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
                arch="gfx1250",
                llvm_flavor="llvm23",
            )
            return llvm.count("load i8, ptr addrspace(3)"), llvm

        before, _ = lds_byte_loads(row_major)
        after, llvm = lds_byte_loads(yaml)
        self.assertGreater(before, 0)
        self.assertEqual(after, 0)
        self.assertIn("llvm.amdgcn.wmma.scale.f32.32x16x128.f4", llvm)
        # Without scale TDM the global B-scale path also uses 4-byte loads.
        global_scales = MxFp4TdmGemmSpec(
            name="bsn_global", M=128, N=64, K=256, b_scale_n_major=True
        )
        self.assertEqual(is_valid_spec(global_scales), (True, "ok"))

        def global_byte_loads(spec):
            llvm = _lower_kernel_to_llvm_python(
                build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
                arch="gfx1250",
                llvm_flavor="llvm23",
            )
            return llvm.count("load i8, ptr addrspace(1)")

        self.assertLess(
            global_byte_loads(global_scales),
            global_byte_loads(
                dataclasses.replace(global_scales, b_scale_n_major=False)
            ),
        )

    def test_step6_13_tdm_multicast_masks_and_cluster_barrier(self):
        self.assertIn("step6_13_mxfp4_gemm", variant_names())
        spec = variant_spec(
            "step6_13_mxfp4_gemm",
            4096,
            4096,
            65536,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        self.assertTrue(spec.tdm_multicast)
        self.assertIn("mcast", spec.kernel_name())
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        llvm = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )
        self.assertIn("llvm.amdgcn.cluster.workgroup.id.x", llvm)
        self.assertIn("llvm.amdgcn.cluster.workgroup.id.y", llvm)
        self.assertIn("llvm.amdgcn.s.cluster.barrier", llvm)
        # LDS is filled by TDM and ordered by s_wait_tensorcnt, so the
        # barrier carries no cluster-scope fences.
        self.assertNotIn('fence syncscope("cluster")', llvm)
        no_cluster = dataclasses.replace(spec, cluster_dim=None)
        ok, reason = is_valid_spec(no_cluster)
        self.assertFalse(ok)
        self.assertIn("tdm_multicast requires cluster_dim", reason)
        too_wide = dataclasses.replace(spec, cluster_dim=(8, 4, 1))
        ok, reason = is_valid_spec(too_wide)
        self.assertFalse(ok)
        self.assertIn("at most 16 workgroups", reason)

    def test_step6_14_cross_tile_prefetch_fences_the_tile_boundary(self):
        self.assertIn("step6_14_mxfp4_gemm", variant_names())
        spec = variant_spec(
            "step6_14_mxfp4_gemm",
            4096,
            4096,
            65536,
            tile_m=256,
            tile_n=256,
            tile_k=512,
        )
        self.assertTrue(spec.cross_tile_prefetch)
        self.assertIn("xtile", spec.kernel_name())
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        lines = _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        ).splitlines()
        clobbers = [i for i, line in enumerate(lines) if '"~{memory}"' in line]
        # One fenced boundary in the peeled first tile, one in the loop.
        self.assertEqual(len(clobbers), 4)
        bounds = [*clobbers[2::2], len(lines)]
        for start, end, stop in zip(clobbers[::2], clobbers[1::2], bounds):
            window = lines[start:end]
            self.assertTrue(any("s.cluster.barrier" in line for line in window))
            self.assertFalse(any("addrspace(3)" in line for line in window))
            # The last K-step's WMMAs follow the boundary.
            self.assertTrue(any("wmma" in line for line in lines[end:stop]))
        for flag in ("init_c_wmma", "local_read_prefetch"):
            ok, reason = is_valid_spec(dataclasses.replace(spec, **{flag: False}))
            self.assertFalse(ok)
            self.assertIn("cross_tile_prefetch requires", reason)

    def _yaml_llvm(self, variant, **overrides):
        spec = variant_spec(
            variant, 4096, 4096, 65536, tile_m=256, tile_n=256, tile_k=512
        )
        spec = dataclasses.replace(spec, **overrides)
        self.assertEqual(is_valid_spec(spec), (True, "ok"))
        return spec, _lower_kernel_to_llvm_python(
            build_mxfp4_tdm_gemm(spec, arch="gfx1250"),
            arch="gfx1250",
            llvm_flavor="llvm23",
        )

    def test_step6_15_cluster_gl2_prefetch_splits_the_cluster_block(self):
        spec, llvm = self._yaml_llvm("step6_15_mxfp4_gemm")
        self.assertTrue(spec.cluster_gl2_prefetch)
        self.assertIn("cgl2", spec.kernel_name())
        _, base = self._yaml_llvm("step6_14_mxfp4_gemm")
        # 4 A tiles + 4 B tiles of 256 rows, one 256 B chunk per row, over
        # 16 x 128 cluster threads: one prefetch per thread per call site.
        call = "call void @llvm.amdgcn.global.prefetch("
        self.assertEqual(llvm.count(call) * 2, base.count(call))
        self.assertIn("llvm.amdgcn.cluster.workgroup.id.x", llvm)
        for overrides, needle in (
            ({"prefetch_gl2": False}, "requires prefetch_gl2"),
            ({"cluster_dim": None, "tdm_multicast": False}, "requires prefetch_gl2"),
            ({"tile_k": 256}, "multiple of 256 B"),
        ):
            ok, reason = is_valid_spec(dataclasses.replace(spec, **overrides))
            self.assertFalse(ok)
            self.assertIn(needle, reason)

    def test_step6_17_pads_the_a_scale_tile(self):
        spec, llvm = self._yaml_llvm("step6_17_mxfp4_gemm")
        self.assertTrue(spec.a_scale_lds_pad)
        self.assertIn("aspad", spec.kernel_name())
        base_spec, base = self._yaml_llvm("step6_15_mxfp4_gemm")
        # 256 rows x 16 B of E8M0 plus 8 B after every 256 B.
        self.assertEqual(spec.a_scale_tile_bytes, 4096 + 16 * 8)
        self.assertEqual(base_spec.a_scale_tile_bytes, 4096)
        # TDM flags: pad enable | interval 256 B (5) | amount 8 B (1).
        a_scale_flags = 1 << 20 | 5 << 22 | 1 << 25
        ab_flags = 1 << 20 | 5 << 22 | 3 << 25
        self.assertIn(f"i32 {a_scale_flags}", llvm)
        self.assertNotIn(f"i32 {a_scale_flags}", base)
        self.assertIn(f"i32 {ab_flags}", llvm)
        # Each A-scale read adds 8 B per 256 B block: offset + (offset >> 8) * 8.
        block_shift = re.compile(r"lshr i32 [^,]+, 8\b")
        self.assertTrue(block_shift.search(llvm))
        self.assertIsNone(block_shift.search(base))
        for overrides, needle in (
            ({"scale_tdm": False, "lds_segment_interleave": False}, "requires scale_tdm"),
            ({"tile_k": 256 * 32 * 2}, "divide 256"),
        ):
            ok, reason = is_valid_spec(dataclasses.replace(spec, **overrides))
            self.assertFalse(ok)
            self.assertIn(needle, reason)

    def test_tdm_descriptor_accepts_dynamic_workgroup_mask(self):
        from rocke.core.ir import I8, IRBuilder, PtrType

        b = IRBuilder("tdm_mask")
        src = b.param("src", PtrType(I8, "global"))
        shared = b.smem_alloc(I8, [256], name_hint="lds")
        mask = b.shl(b.const_i32(0xF), b.thread_id_x())
        one = b.const_i32(1)
        _, group1, *_ = b.tdm_descriptor_2d(
            src,
            b.smem_addr_of(shared),
            elem_bytes=1,
            tensor_dim0=one,
            tensor_dim1=one,
            row_stride=one,
            tile_dim0=16,
            tile_dim1=16,
            workgroup_mask=mask,
        )
        self.assertIsNotNone(group1)
        with self.assertRaises(ValueError):
            b.tdm_descriptor_2d(
                src,
                b.smem_addr_of(shared),
                elem_bytes=1,
                tensor_dim0=one,
                tensor_dim1=one,
                row_stride=one,
                tile_dim0=16,
                tile_dim1=16,
                workgroup_mask=0x10000,
            )

    def test_kernel_inputs_transpose_b_scale_for_n_major(self):
        import numpy as np
        from rocke.examples.gfx1250.gemm.mxfp4_tdm_gemm import (
            kernel_inputs,
            make_inputs,
        )

        spec = MxFp4TdmGemmSpec(name="bsn", M=128, N=64, K=256, b_scale_n_major=True)
        inputs = make_inputs(spec)
        self.assertEqual(inputs[3].shape, (spec.K // 32, spec.N))
        device = kernel_inputs(spec, inputs)
        self.assertEqual(device[3].shape, (spec.N, spec.K // 32))
        self.assertTrue(device[3].flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(device[3], inputs[3].T)
        plain = kernel_inputs(
            MxFp4TdmGemmSpec(name="plain", M=128, N=64, K=256), inputs
        )
        self.assertIs(plain[3], inputs[3])

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
