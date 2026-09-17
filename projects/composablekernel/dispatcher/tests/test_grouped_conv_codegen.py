#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
TDD tests for codegen/unified_grouped_conv_codegen.py -- grouped convolution code generator.

These tests are written BEFORE the implementation exists.
Run: python3 -m pytest dispatcher/tests/test_grouped_conv_codegen.py -v
"""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent.resolve()
DISPATCHER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(DISPATCHER_DIR / "codegen"))
sys.path.insert(0, str(DISPATCHER_DIR / "python"))
sys.path.insert(0, str(DISPATCHER_DIR / "scripts"))

from codegen_common import TileConfig, TraitConfigBase  # noqa: E402
from registration_codegen import make_registration_block, parse_kernel_metadata  # noqa: E402

from unified_grouped_conv_codegen import (  # noqa: E402
    GroupedConvVariant,
    GroupedConvLayout,
    GroupedConvKernelConfig,
    GroupedConvTypeMappings,
    GroupedConvTraitConfig,
    CKTileGroupedConvKernelGenerator,
    UnifiedGroupedConvCodegen,
    get_default_configs,
)


# =============================================================================
# TestGroupedConvVariant
# =============================================================================


class TestGroupedConvVariant(unittest.TestCase):
    """Test GroupedConvVariant enum values."""

    def test_forward_value(self):
        self.assertEqual(GroupedConvVariant.FORWARD.value, "forward")

    def test_backward_data_value(self):
        self.assertEqual(GroupedConvVariant.BACKWARD_DATA.value, "bwd_data")

    def test_backward_weight_value(self):
        self.assertEqual(GroupedConvVariant.BACKWARD_WEIGHT.value, "bwd_weight")

    def test_all_variants_exist(self):
        self.assertIn(GroupedConvVariant.FORWARD, GroupedConvVariant)
        self.assertIn(GroupedConvVariant.BACKWARD_DATA, GroupedConvVariant)
        self.assertIn(GroupedConvVariant.BACKWARD_WEIGHT, GroupedConvVariant)


# =============================================================================
# TestGroupedConvLayout
# =============================================================================


class TestGroupedConvLayout(unittest.TestCase):
    """Test GroupedConvLayout enum for 1D/2D/3D layouts."""

    def test_nhwgc_value(self):
        self.assertEqual(GroupedConvLayout.NHWGC.value, "NHWGC")

    def test_gkyxc_value(self):
        self.assertEqual(GroupedConvLayout.GKYXC.value, "GKYXC")

    def test_nhwgk_value(self):
        self.assertEqual(GroupedConvLayout.NHWGK.value, "NHWGK")

    def test_1d_layouts_exist(self):
        """1D conv layouts (e.g., NWGC, GYXC, NWGK)."""
        layouts_1d = [
            lay
            for lay in GroupedConvLayout
            if "W" in lay.value and "H" not in lay.value
        ]
        self.assertGreater(len(layouts_1d), 0)

    def test_2d_layouts_exist(self):
        """2D conv layouts (e.g., NHWGC, GKYXC, NHWGK)."""
        layouts_2d = [lay for lay in GroupedConvLayout if "HW" in lay.value]
        self.assertGreater(len(layouts_2d), 0)

    def test_3d_layouts_exist(self):
        """3D conv layouts (e.g., NDHWGC, GDKYXC)."""
        layouts_3d = [
            lay for lay in GroupedConvLayout if "D" in lay.value or "DHW" in lay.value
        ]
        self.assertGreater(len(layouts_3d), 0)


# =============================================================================
# TestGroupedConvKernelConfig
# =============================================================================


class TestGroupedConvKernelConfig(unittest.TestCase):
    """Test GroupedConvKernelConfig dataclass."""

    def _make_tile(self):
        return TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)

    def _make_trait(self):
        return GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )

    def test_name_contains_grouped_conv_fwd(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        name = config.name("fp16")
        self.assertIn("grouped_conv_fwd", name)

    def test_name_backward_data_contains_bwd_data(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.BACKWARD_DATA,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        name = config.name("fp16")
        self.assertIn("bwd_data", name)

    def test_is_valid_for_arch_supported(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        self.assertTrue(config.is_valid_for_arch("gfx942"))

    def test_is_valid_for_arch_unsupported(self):
        config = GroupedConvKernelConfig(
            tile=self._make_tile(),
            trait=self._make_trait(),
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        self.assertFalse(config.is_valid_for_arch("gfx600"))


# =============================================================================
# TestGroupedConvTypeMappings
# =============================================================================


class TestGroupedConvTypeMappings(unittest.TestCase):
    """Test GroupedConvTypeMappings class."""

    def test_dtype_to_ck_fp16(self):
        self.assertEqual(GroupedConvTypeMappings.DTYPE_TO_CK["fp16"], "half_t")

    def test_dtype_to_ck_bf16(self):
        self.assertIn("bf16", GroupedConvTypeMappings.DTYPE_TO_CK)

    def test_dtype_to_ck_fp32(self):
        self.assertIn("fp32", GroupedConvTypeMappings.DTYPE_TO_CK)

    def test_get_layouts_2d_has_in_wei_out_keys(self):
        layouts = GroupedConvTypeMappings.get_layouts(2)
        self.assertIn("in", layouts)
        self.assertIn("wei", layouts)
        self.assertIn("out", layouts)

    def test_get_layouts_2d_returns_dict(self):
        layouts = GroupedConvTypeMappings.get_layouts(2)
        self.assertIsInstance(layouts, dict)

    def test_get_layouts_1d(self):
        layouts = GroupedConvTypeMappings.get_layouts(1)
        self.assertIn("in", layouts)
        self.assertIn("wei", layouts)
        self.assertIn("out", layouts)

    def test_get_layouts_3d(self):
        layouts = GroupedConvTypeMappings.get_layouts(3)
        self.assertIn("in", layouts)
        self.assertIn("wei", layouts)
        self.assertIn("out", layouts)


# =============================================================================
# TestCKTileGroupedConvKernelGenerator
# =============================================================================


class TestCKTileGroupedConvKernelGenerator(unittest.TestCase):
    """Test CKTileGroupedConvKernelGenerator.generate()."""

    def _make_config(self):
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )
        return GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )

    def test_generate_contains_pragma_once(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIn("#pragma once", result)

    def test_generate_contains_forward_kernel_include(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIn("grouped_convolution_forward_kernel.hpp", result)

    def test_generate_returns_non_empty_string(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 100)

    def test_generate_valid_cpp_structure(self):
        gen = CKTileGroupedConvKernelGenerator("fp16")
        config = self._make_config()
        result = gen.generate(config)
        self.assertIn("#include", result)
        self.assertIn("ck_tile", result)


# =============================================================================
# TestUnifiedGroupedConvCodegen
# =============================================================================


class TestUnifiedGroupedConvCodegen(unittest.TestCase):
    """Test UnifiedGroupedConvCodegen.generate_all()."""

    def test_generate_all_returns_dict_with_expected_keys(self):
        output_dir = DISPATCHER_DIR / "build" / "generated" / "grouped_conv"
        output_dir.mkdir(parents=True, exist_ok=True)
        codegen = UnifiedGroupedConvCodegen(
            output_dir=output_dir,
            datatype="fp16",
            ndim_spatial=2,
            gpu_target="gfx942",
        )
        with patch.object(
            codegen,
            "_get_configs",
            return_value=[],  # Mock empty config list for fast test
        ):
            results = codegen.generate_all(parallel=False)
        self.assertIn("kernels", results)
        self.assertIn("failed", results)
        self.assertIsInstance(results["kernels"], list)
        self.assertIsInstance(results["failed"], list)

    def test_generate_all_rejects_duplicate_kernel_identity(self):
        output_dir = DISPATCHER_DIR / "build" / "generated" / "duplicate_test"
        codegen = UnifiedGroupedConvCodegen(
            output_dir=output_dir,
            datatype="fp16",
            ndim_spatial=2,
            gpu_target="gfx942",
            enable_arch_filter=False,
        )
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig(
            "mem", "cshuffle", "intrawave", False, False, False
        )
        config = GroupedConvKernelConfig(tile=tile, trait=trait)
        results = codegen.generate_all(
            configs=[config, config], datatypes=["fp16"], parallel=False
        )
        self.assertEqual([], results["kernels"])
        self.assertEqual(1, len(results["failed"]))
        self.assertIn("Duplicate generated kernel identity", results["failed"][0])

    def test_generate_all_with_mock_config_produces_output(self):
        output_dir = DISPATCHER_DIR / "build" / "generated" / "grouped_conv_test"
        output_dir.mkdir(parents=True, exist_ok=True)
        codegen = UnifiedGroupedConvCodegen(
            output_dir=output_dir,
            datatype="fp16",
            ndim_spatial=2,
            gpu_target="gfx942",
        )
        # Use a real config - patch the config source to return one config
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )
        config = GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )

        with patch.object(codegen, "_get_configs", return_value=[config]):
            results = codegen.generate_all(parallel=False)
        self.assertIsInstance(results, dict)
        self.assertIn("kernels", results)


# =============================================================================
# TestSharedImports
# =============================================================================


class TestSharedImports(unittest.TestCase):
    """Verify TileConfig from codegen_common and GroupedConvTraitConfig extends TraitConfigBase."""

    def test_tile_config_has_expected_fields(self):
        """TileConfig from codegen_common has tile_m, tile_n, tile_k, etc."""
        tc = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertEqual(tc.tile_m, 128)
        self.assertEqual(tc.tile_n, 128)
        self.assertEqual(tc.tile_k, 32)
        self.assertEqual(tc.warp_m, 2)
        self.assertEqual(tc.warp_n, 2)
        self.assertEqual(tc.warp_k, 1)
        self.assertEqual(tc.warp_tile_m, 32)
        self.assertEqual(tc.warp_tile_n, 32)
        self.assertEqual(tc.warp_tile_k, 16)

    def test_tile_config_is_from_codegen_common(self):
        """TileConfig used by grouped conv is the same as codegen_common.TileConfig."""
        tc = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        self.assertTrue(tc.is_valid())

    def test_grouped_conv_trait_config_extends_trait_config_base(self):
        """GroupedConvTraitConfig extends TraitConfigBase."""
        self.assertTrue(issubclass(GroupedConvTraitConfig, TraitConfigBase))

    def test_grouped_conv_trait_config_has_double_smem_buffer(self):
        """GroupedConvTraitConfig has double_smem_buffer field."""
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=True,
            num_groups_to_merge=2,
        )
        self.assertTrue(trait.double_smem_buffer)
        self.assertEqual(trait.num_groups_to_merge, 2)

    def test_grouped_conv_trait_config_has_num_groups_to_merge(self):
        """GroupedConvTraitConfig has num_groups_to_merge field."""
        trait = GroupedConvTraitConfig(
            "mem",
            "cshuffle",
            "intrawave",
            False,
            False,
            False,
            double_smem_buffer=False,
            num_groups_to_merge=4,
        )
        self.assertEqual(trait.num_groups_to_merge, 4)

    def test_grouped_conv_trait_config_inherits_base_fields(self):
        """GroupedConvTraitConfig inherits pipeline, epilogue, scheduler from base."""
        trait = GroupedConvTraitConfig(
            "compv4",
            "cshuffle",
            "intrawave",
            True,
            True,
            True,
            double_smem_buffer=False,
            num_groups_to_merge=1,
        )
        self.assertEqual(trait.pipeline, "compv4")
        self.assertEqual(trait.epilogue, "cshuffle")
        self.assertEqual(trait.scheduler, "intrawave")
        self.assertTrue(trait.pad_m)
        self.assertTrue(trait.pad_n)
        self.assertTrue(trait.pad_k)


# =============================================================================
# TestTwoStageBwdWeightCodegen
# =============================================================================


def _make_two_stage_config():
    """Helper: create a two-stage bwd_weight config."""
    return GroupedConvKernelConfig(
        tile=TileConfig(16, 64, 64, 1, 4, 1, 16, 16, 32),
        trait=GroupedConvTraitConfig(
            pipeline="compv3",
            epilogue="cshuffle",
            scheduler="intrawave",
            pad_m=True,
            pad_n=True,
            pad_k=True,
            two_stage=True,
        ),
        variant=GroupedConvVariant.BACKWARD_WEIGHT,
        ndim_spatial=2,
        arch="gfx942",
    )


class TestTwoStageBwdWeightCodegen(unittest.TestCase):
    """Tests for two-stage backward weight kernel generation."""

    def test_kernel_name_contains_2stage(self):
        config = _make_two_stage_config()
        name = config.name("fp16")
        self.assertIn("_2stage", name)
        self.assertIn("bwd_weight", name)

    def test_single_stage_name_has_no_2stage(self):
        config = _make_two_stage_config()
        config.trait.two_stage = False
        name = config.name("fp16")
        self.assertNotIn("_2stage", name)

    def test_generate_contains_elementwise_include(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("elementwise.hpp", code)

    def test_generate_contains_workspace_type(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("WorkspaceDataType", code)

    def test_generate_contains_elementwise_kernel(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("ElementWiseKernel", code)

    def test_generate_contains_launch_kernel_time_mask(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("launch_kernel_time_mask", code)

    def test_two_stage_uses_fp32_workspace_vector_size_c(self):
        # Two-stage writes the GEMM result to an fp32 workspace, so it uses the
        # configured VectorSizeC directly instead of forcing it to 1.
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("WorkspaceDataType = float", code)
        self.assertIn("Config::VectorSizeC", code)
        self.assertNotIn("VectorSizeC_TwoStage", code)

    def test_generate_contains_workspace_memset(self):
        config = _make_two_stage_config()
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertIn("hipMemsetAsync", code)

    def test_single_stage_does_not_contain_workspace(self):
        config = _make_two_stage_config()
        config.trait.two_stage = False
        gen = CKTileGroupedConvKernelGenerator(
            "fp16", GroupedConvVariant.BACKWARD_WEIGHT
        )
        code = gen.generate(config)
        self.assertNotIn("WorkspaceDataType", code)
        self.assertNotIn("ElementWiseKernel", code)

    def test_default_configs_include_two_stage(self):
        from unified_grouped_conv_codegen import get_default_configs

        configs = get_default_configs(
            arch="gfx942",
            variants=[GroupedConvVariant.BACKWARD_WEIGHT],
            ndims=[2],
        )
        two_stage = [c for c in configs if c.trait.two_stage]
        single_stage = [c for c in configs if not c.trait.two_stage]
        self.assertGreater(len(two_stage), 0, "Should have two-stage configs")
        self.assertGreater(
            len(single_stage), 0, "Should still have single-stage configs"
        )


class TestProductionRegistrationMetadata(unittest.TestCase):
    def test_2d_uses_canonical_combined_layout(self):
        name = (
            "grouped_conv_fwd_fp16_nhwgc_2d_compv3_cshuffle_intrawave_"
            "32x64x32_2x4x1_16x16x16"
        )
        metadata = parse_kernel_metadata(name)
        self.assertEqual("nhwgc_gkyxc_nhwgk", metadata["layout"])
        block = "\n".join(
            make_registration_block(
                name, 0, "GroupedConvOp::Forward",
                "backends::make_conv_fwd_run_fn",
                "backends::make_conv_fwd_is_supported_fn",
            )
        )
        self.assertIn('key.layout       = "nhwgc_gkyxc_nhwgk";', block)
        self.assertIn("if(!registry.register_kernel(key, inst))", block)
        self.assertNotIn("registry.register_kernel(key, inst);", block)

    def test_3d_uses_canonical_combined_layout(self):
        name = (
            "grouped_conv_fwd_fp16_ndhwgc_3d_compv3_cshuffle_intrawave_"
            "32x64x32_2x4x1_16x16x16"
        )
        metadata = parse_kernel_metadata(name)
        self.assertEqual("ndhwgc_gkzyxc_ndhwgk", metadata["layout"])
        block = "\n".join(
            make_registration_block(
                name, 0, "GroupedConvOp::Forward",
                "backends::make_conv_fwd_run_fn",
                "backends::make_conv_fwd_is_supported_fn",
            )
        )
        self.assertIn('key.layout       = "ndhwgc_gkzyxc_ndhwgk";', block)
        self.assertIn("if(!registry.register_kernel(key, inst))", block)
        self.assertNotIn("registry.register_kernel(key, inst);", block)


class TestRdnaProductionCatalog(unittest.TestCase):
    WAVES = ((2, 4, 1), (1, 8, 1), (8, 1, 1), (4, 2, 1))
    CATALOG_NAMES = (
        "grouped_conv_fwd_bf16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "128x16x32_8x1x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_bf16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "16x128x32_1x8x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_bf16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "32x64x32_2x4x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_bf16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "64x32x32_4x2x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_bf16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "128x16x32_8x1x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_bf16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "16x128x32_1x8x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_bf16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "32x64x32_2x4x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_bf16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "64x32x32_4x2x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "128x16x32_8x1x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "16x128x32_1x8x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "32x64x32_2x4x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_ndhwgc_3d_compv3_cshuffle_intrawave_"
        "64x32x32_4x2x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "128x16x32_8x1x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "16x128x32_1x8x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "32x64x32_2x4x1_16x16x16_vec2_2_8",
        "grouped_conv_fwd_fp16_nhwgc_2d_compv3_cshuffle_intrawave_"
        "64x32x32_4x2x1_16x16x16_vec2_2_8",
    )

    def _configs(self, arch="gfx1100", ndim=None, datatypes=None):
        return get_default_configs(
            arch=arch,
            variants=[GroupedConvVariant.FORWARD],
            ndims=[ndim] if ndim is not None else [2, 3],
            datatypes=datatypes or ["fp16", "bf16"],
            rule_set="rdna",
        )

    def test_catalog_is_stable_and_covers_required_waves(self):
        configs = self._configs(ndim=2, datatypes=["fp16"])
        self.assertEqual(4, len(configs))
        self.assertEqual(
            list(self.WAVES),
            [(c.tile.warp_m, c.tile.warp_n, c.tile.warp_k) for c in configs],
        )
        self.assertEqual(
            sorted(c.name("fp16") for c in configs),
            sorted(c.name("fp16") for c in self._configs(ndim=2, datatypes=["fp16"])),
        )

    def test_catalog_has_16_forward_instances(self):
        configs = self._configs()
        self.assertEqual(16, len(configs))
        self.assertEqual(self.CATALOG_NAMES, tuple(sorted(c.name(c.datatype) for c in configs)))
        self.assertEqual(
            {"fp16", "bf16"},
            {config.datatype for config in configs},
        )
        self.assertEqual({2, 3}, {config.ndim_spatial for config in configs})

    def test_gfx1201_matches_gfx1100_instance_set(self):
        names = lambda arch: tuple(sorted(c.name(c.datatype) for c in self._configs(arch=arch)))
        self.assertEqual(self.CATALOG_NAMES, names("gfx1100"))
        self.assertEqual(names("gfx1100"), names("gfx1200"))
        self.assertEqual(names("gfx1100"), names("gfx1201"))

    def test_catalog_has_nonzero_2d_and_3d_fp16_forward(self):
        for ndim in (2, 3):
            configs = self._configs(ndim=ndim, datatypes=["fp16"])
            self.assertTrue(configs)
            for config in configs:
                self.assertEqual("fp16", config.datatype)
                self.assertEqual(GroupedConvVariant.FORWARD, config.variant)
                self.assertEqual("compv3", config.trait.pipeline)
                self.assertEqual((2, 2, 8), (
                    config.vector_size_a,
                    config.vector_size_b,
                    config.vector_size_c,
                ))
                block_size = 32 * config.tile.warp_m * config.tile.warp_n * config.tile.warp_k
                self.assertGreaterEqual(
                    config.tile.tile_m * config.tile.tile_k // (block_size * config.vector_size_a), 1
                )
                self.assertGreaterEqual(
                    config.tile.tile_n * config.tile.tile_k // (block_size * config.vector_size_b), 1
                )
                self.assertEqual((16, 16, 16), (
                    config.tile.warp_tile_m,
                    config.tile.warp_tile_n,
                    config.tile.warp_tile_k,
                ))
                self.assertEqual("ndhwgc" if ndim == 3 else "nhwgc", config._layout_str())

    def test_default_catalog_remains_empty_for_gfx1100(self):
        configs = get_default_configs(
            arch="gfx1100",
            variants=[GroupedConvVariant.FORWARD],
            ndims=[2, 3],
            datatypes=["fp16"],
            rule_set="default",
        )
        self.assertEqual([], configs)

    def test_production_filter_is_fail_closed(self):
        with patch("unified_grouped_conv_codegen.HAS_ARCH_FILTER", False):
            with self.assertRaisesRegex(RuntimeError, "required but unavailable"):
                UnifiedGroupedConvCodegen(
                    output_dir=DISPATCHER_DIR / "build" / "missing-filter",
                    gpu_target="gfx1100",
                    require_arch_filter=True,
                )

    def test_generate_all_emits_headers_not_dispatcher_wrappers(self):
        output_dir = DISPATCHER_DIR / "build" / "generated" / "no_wrappers"
        codegen = UnifiedGroupedConvCodegen(
            output_dir=output_dir,
            datatype="fp16",
            ndim_spatial=2,
            gpu_target="gfx942",
            enable_arch_filter=False,
        )
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig(
            "mem", "cshuffle", "intrawave", False, False, False
        )
        config = GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        results = codegen.generate_all(configs=[config], datatypes=["fp16"], parallel=False)
        self.assertNotIn("wrappers", results)
        self.assertEqual(1, len(results["kernels"]))
        self.assertTrue(results["kernels"][0].is_file())
        self.assertFalse((output_dir / "dispatcher_wrappers").exists())


class TestRdnaFailClosed(unittest.TestCase):
    def test_rdna_rejects_wave64_arch(self):
        with self.assertRaises(ValueError):
            get_default_configs(
                arch="gfx942",
                variants=[GroupedConvVariant.FORWARD],
                ndims=[2],
                datatypes=["fp16"],
                rule_set="rdna",
            )

    def test_rdna_rejects_backward_variant(self):
        with self.assertRaises(ValueError):
            get_default_configs(
                arch="gfx1100",
                variants=[GroupedConvVariant.BACKWARD_DATA],
                ndims=[2],
                datatypes=["fp16"],
                rule_set="rdna",
            )

    def test_rdna_rejects_mixed_forward_and_backward(self):
        with self.assertRaises(ValueError):
            get_default_configs(
                arch="gfx1100",
                variants=[GroupedConvVariant.FORWARD, GroupedConvVariant.BACKWARD_WEIGHT],
                ndims=[2],
                datatypes=["fp16"],
                rule_set="rdna",
            )

    def test_rdna_rejects_ndim_1(self):
        with self.assertRaises(ValueError):
            get_default_configs(
                arch="gfx1100",
                variants=[GroupedConvVariant.FORWARD],
                ndims=[1],
                datatypes=["fp16"],
                rule_set="rdna",
            )

    def test_rdna_rejects_fp32(self):
        with self.assertRaises(ValueError):
            get_default_configs(
                arch="gfx1100",
                variants=[GroupedConvVariant.FORWARD],
                ndims=[2],
                datatypes=["fp32"],
                rule_set="rdna",
            )

    def test_rdna_cli_gfx942_list_configs_is_nonzero(self):
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(DISPATCHER_DIR / "codegen") + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [
                sys.executable,
                str(DISPATCHER_DIR / "codegen" / "unified_grouped_conv_codegen.py"),
                "--rule-set", "rdna",
                "--arch", "gfx942",
                "--variant", "forward",
                "--list-configs",
            ],
            cwd=str(DISPATCHER_DIR / "codegen"),
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(0, result.returncode, msg=result.stdout + result.stderr)

    def test_generate_all_fails_when_all_supplied_configs_are_rejected(self):
        output_dir = DISPATCHER_DIR / "build" / "generated" / "all_rejected"
        codegen = UnifiedGroupedConvCodegen(
            output_dir=output_dir,
            datatype="fp16",
            ndim_spatial=2,
            gpu_target="gfx942",
            enable_arch_filter=False,
        )
        tile = TileConfig(128, 128, 32, 2, 2, 1, 32, 32, 16)
        trait = GroupedConvTraitConfig("mem", "cshuffle", "intrawave", False, False, False)
        config = GroupedConvKernelConfig(
            tile=tile,
            trait=trait,
            variant=GroupedConvVariant.FORWARD,
            ndim_spatial=2,
            arch="gfx942",
            layout=GroupedConvLayout.NHWGC,
            vector_sizes=(4, 4, 4),
        )
        with patch.object(codegen, "is_config_valid", return_value=False):
            results = codegen.generate_all(configs=[config], datatypes=["fp16"], parallel=False)
        self.assertEqual([], results["kernels"])
        self.assertTrue(results["failed"])


class TestPerArchRegistrationSymbols(unittest.TestCase):
    def test_chunk_and_register_all_symbols_include_arch(self):
        import tempfile
        from registration_codegen import generate_chunked_registration

        with tempfile.TemporaryDirectory() as tmp:
            header = Path(tmp) / "grouped_conv_fwd_fp16_nhwgc_2d_compv3_cshuffle_intrawave_32x64x32_2x4x1_16x16x16.hpp"
            header.write_text("// stub\n")
            files = generate_chunked_registration(
                [header],
                tmp,
                variant="fwd",
                op_enum="GroupedConvOp::Forward",
                run_fn_maker="backends::make_conv_fwd_run_fn",
                is_supported_fn_maker="backends::make_conv_fwd_is_supported_fn",
                register_fn_name="register_all_grouped_conv_fwd_kernels_gfx1100",
                arch="gfx1100",
            )
            texts = [Path(p).read_text() for p in files]
            joined = "\n".join(texts)
            self.assertIn("void register_fwd_gfx1100_chunk_0(", joined)
            self.assertIn("void register_all_grouped_conv_fwd_kernels_gfx1100(", joined)
            self.assertNotIn("void register_fwd_chunk_0(", joined)
            self.assertNotIn(
                "void register_all_grouped_conv_fwd_kernels(GroupedConvRegistry& registry, const std::string& arch)",
                joined,
            )


if __name__ == "__main__":
    unittest.main()
