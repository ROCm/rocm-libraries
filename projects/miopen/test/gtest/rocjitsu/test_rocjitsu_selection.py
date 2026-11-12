# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from fnmatch import fnmatchcase
import itertools
import json
from pathlib import Path
import tempfile
import unittest

from rocjitsu_selection import (
    build_filter,
    export_policy,
    listed_tests,
    reference_features,
)


def matches(name, test_filter):
    positive, _, negative = test_filter.partition("-")
    return any(fnmatchcase(name, p) for p in (positive or "*").split(":")) and not any(
        fnmatchcase(name, p) for p in negative.split(":")
    )


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "policy.json"
            export_policy(
                Path(__file__).parent.parent / "test_categories.yaml", destination
            )
            cls.category = json.loads(destination.read_text())
        cls.policy = cls.category["representative_cases"]
        cls.base = (
            ":".join(cls.category["test_patterns"])
            + "-"
            + ":".join(cls.category["exclude"])
        )

    def test_ordinary_quick_and_ffm_keep_their_independent_cost_boundaries(self):
        import yaml

        with open(Path(__file__).parent.parent / "test_categories.yaml") as source:
            categories = yaml.safe_load(source)["test_categories"]
        inventory = [
            "Smoke/GPU_UnitTestConvSolverConvHipConv_FP16.Run/0",
            "Smoke/GPU_Reduce_FP32.Run/0",
            "Smoke/GPU_Dropout_FP16.Run/0",
            "Smoke/GPU_UnitTestConvSolverDirectNaiveFwd_FP16.Run/0",
            "CPU_UnitTestConvSolverConvHipConv_NONE.IsApplicable",
        ]
        selections = {}
        for key in ("quick", "ffm-quick"):
            category = categories[key]
            base = (
                ":".join(category["test_patterns"])
                + "-"
                + ":".join(category["exclude"])
            )
            selections[key] = [name for name in inventory if matches(name, base)]
        self.assertEqual(selections["quick"], inventory)
        self.assertEqual(selections["ffm-quick"], [inventory[3]])
        _, emulated, _ = build_filter(self.base, inventory, self.policy)
        self.assertEqual(emulated, inventory[1:])

    def test_listing_parameter_comments_and_non_listing_output(self):
        listing = """Running main() from gtest_main.cc
Smoke/GPU_Example_FP32.  # TypeParam = float
  Numerical/shape_1  # GetParam() = 1
  DISABLED_Boundary/shape_2
[diagnostic] ignored
  not_a_test
CPU_Example_NONE.
  Host
"""
        self.assertEqual(
            listed_tests(listing),
            [
                "Smoke/GPU_Example_FP32.Numerical/shape_1",
                "Smoke/GPU_Example_FP32.DISABLED_Boundary/shape_2",
                "CPU_Example_NONE.Host",
            ],
        )

    def test_base_filter_and_existing_negatives_never_widen(self):
        names = [
            "Smoke/GPU_Keep_FP32.Run/0",
            "Smoke/GPU_Keep_FP32.Run/1",
            "CPU_Keep_NONE.Host",
            "Standard/GPU_Keep_FP32.Run/0",
        ]
        base = "Smoke/*:CPU_*-*.*/1:CPU_*"
        result, selected, exclusions = build_filter(base, names, self.policy)
        self.assertEqual(selected, [names[0]])
        self.assertEqual(result, base)
        self.assertEqual(exclusions, [])
        self.assertEqual([n for n in names if matches(n, result)], selected)

    def test_static_policy_boundaries(self):
        retained = [
            "CPU_UnitTestConvSolverConvHipConv_NONE.IsApplicable",
            "Smoke/GPU_UnitTestConvSolverImplicitGemm3DGroupBwdXdlops_FP16.Run/0",
            "Smoke/GPU_ConvGrpBiasActivInfer3D_FP32.Run/shape_G_1_N_1_C_3_K_4_D_4_H_4_W_4_pad_0",
            "Smoke/GPU_MultiMarginLoss_FP32.Run/5",
            "Smoke/GPU_BNFWDTrainSmall3D_FP32.Run/NCDHW_test_id_7",
            "Smoke/GPU_BNInferSmall3D_FP32.Run/NCDHW_test_id_24",
            "Smoke/GPU_Find2Conv_FP32.Run/direction_0_tune_0_preallocate_0_ws_limit_1_attach_binaries_0",
            "Smoke/GPU_Reduce_FP32.Run/0",
            "Smoke/GPU_reference_kernel_fwd_3d_NCDHW_BFP16_BFP16.Run/unknown",
        ]
        excluded = [
            "Smoke/GPU_UnitTestConvSolverConvHipConv_FP16.Run/0",
            "Smoke/GPU_UnitTestConvSolverImplicitGemm3DGroupFwdXdlops_FP16.Run/0",
            "Smoke/GPU_ConvGrpBiasActivInfer3D_FP32.Run/shape_G_1_N_32_C_64_K_128_D_28_H_28_W_28_pad_0",
            "Smoke/GPU_MultiMarginLoss_FP32.Run/6",
            "Smoke/GPU_BNFWDTrainSmall3D_FP32.Run/NCDHW_test_id_8",
            "Smoke/GPU_BNInferSmall3D_FP32.Run/NCDHW_test_id_23",
            "Smoke/GPU_Find2Conv_FP32.Run/direction_0_tune_1_preallocate_0_ws_limit_1_attach_binaries_0",
            "Smoke/GPU_Find2Conv_FP32.Run/direction_0_tune_0_preallocate_0_ws_limit_0_attach_binaries_0",
            "CPU_PerfDb_NONE.Run",
            "Smoke/GPU_MIOpenDriverConv_FP32.Run/0",
            "Standard/GPU_Reduce_FP32.Run/0",
        ]
        result, selected, _ = build_filter(self.base, retained + excluded, self.policy)
        self.assertEqual(selected, retained)
        self.assertEqual(
            [n for n in retained + excluded if matches(n, result)], retained
        )

    def test_reorder_retains_shape_classes_for_every_permutation_and_dtype(self):
        axes = ((1, 5, 11), (1, 7, 13), (1, 9, 17), (1, 15, 19))
        expected = {
            (1, 1, 1, 1),
            (5, 7, 9, 15),
            (11, 13, 17, 19),
            (1, 1, 17, 19),
            (11, 13, 1, 1),
            (11, 1, 1, 1),
            (1, 13, 1, 1),
            (1, 1, 17, 1),
            (1, 1, 1, 19),
        }
        names = []
        identities = {}
        for dtype, permutation, shape in itertools.product(
            ("FP32", "FP16"), ("0123", "3210"), itertools.product(*axes)
        ):
            a, b, c, d = shape
            name = (
                f"Smoke/GPU_TensorReorder_{dtype}.Run/perm_{permutation}"
                f"_dim0_{a}_dim1_{b}_dim2_{c}_dim3_{d}_case"
            )
            names.append(name)
            identities[name] = (dtype, permutation, shape)
        unknown = "Smoke/GPU_TensorReorder_FP32.Run/new_parameter_format"
        unrelated = "Standard/GPU_TensorReorder_FP32.Run/shape"
        result, selected, exclusions = build_filter(
            "Smoke/*", names + [unknown, unrelated], self.policy
        )
        for dtype, permutation in itertools.product(("FP32", "FP16"), ("0123", "3210")):
            shapes = {
                identities[n][2]
                for n in selected
                if n in identities and identities[n][:2] == (dtype, permutation)
            }
            self.assertEqual(shapes, expected)
        self.assertIn(unknown, selected)
        self.assertNotIn(unrelated, selected)
        self.assertTrue(exclusions)
        self.assertEqual(
            [n for n in names + [unknown, unrelated] if matches(n, result)], selected
        )

    def test_reference_semantics_and_work_preference_per_fixture(self):
        small = "n1c4k4wi4hi4fx1fy1g1"
        large = "n1c4k4wi40hi40fx1fy1g1"
        grouped = "n2c4k8wi6hi6fx3fy1g2sx2dy2px1"
        fewer_channels = "n1c8k4wi4hi4fx3fy3g1"
        geometries = [small, large, grouped, fewer_channels]
        families = [
            "Smoke/GPU_reference_kernel_fwd_2d_NCHW_FP32_FP32",
            "Smoke/GPU_reference_kernel_bwd_2d_NHWC_FP16_FP16",
            "Smoke/GPU_reference_kernel_wrw_3d_NCDHW_BFP16_BFP16",
        ]
        names = [
            f"{family}.Run/{g}" for family, g in itertools.product(families, geometries)
        ]
        protected = self.policy["reference_kernels"]["protected_patterns"]
        protected_names = [
            f"{family}.Run/{g}"
            for family, g in itertools.product(protected, geometries)
        ]
        unknown = f"{families[0]}.Run/new_geometry_format"
        result, selected, _ = build_filter(
            "Smoke/*", names + protected_names + [unknown], self.policy
        )
        for family in families:
            self.assertIn(f"{family}.Run/{small}", selected)
            self.assertIn(f"{family}.Run/{grouped}", selected)
            self.assertIn(f"{family}.Run/{fewer_channels}", selected)
            self.assertNotIn(f"{family}.Run/{large}", selected)
        self.assertTrue(set(protected_names) <= set(selected))
        self.assertIn(unknown, selected)
        self.assertEqual(
            [n for n in names + protected_names + [unknown] if matches(n, result)],
            selected,
        )
        features, work = reference_features(grouped)
        self.assertEqual(
            features,
            {
                ("grouped", True),
                ("batch", True),
                ("strided", True),
                ("dilated", True),
                ("padded", True),
                ("pointwise", False),
                ("anisotropic", True),
                ("channels", 1),
                ("group_stride", True, True),
                ("group_dilation", True, True),
                ("stride_dilation", True, True),
            },
        )
        self.assertEqual(work, 3456)

    def test_reference_retains_all_group_stride_dilation_pair_interactions(self):
        geometries = {}
        for grouped, strided, dilated in itertools.product((False, True), repeat=3):
            geometry = (
                f"n1c4k4wi4hi4di4fx1fy1fz1g{2 if grouped else 1}"
                f"sz{2 if strided else 1}dz{2 if dilated else 1}"
            )
            geometries[geometry] = (grouped, strided, dilated)
        families = [
            "Smoke/GPU_reference_kernel_fwd_3d_NDHWC_FP16_FP16",
            "Smoke/GPU_reference_kernel_bwd_3d_NCDHW_FP32_FP32",
        ]
        names = [
            f"{family}.Run/{geometry}"
            for family, geometry in itertools.product(families, geometries)
        ]
        _, selected, _ = build_filter("Smoke/*", names, self.policy)
        for family in families:
            classes = [
                geometries[name.rsplit("/", 1)[-1]]
                for name in selected
                if name.startswith(family + ".")
            ]
            for first, second in ((0, 1), (0, 2), (1, 2)):
                self.assertEqual(
                    {(values[first], values[second]) for values in classes},
                    set(itertools.product((False, True), repeat=2)),
                )

    def test_reference_policy_boundaries_and_inventory_order(self):
        geometries = ["n1c4k4wi4hi4fx1fy1g1", "n1c4k4wi40hi40fx1fy1g1"]
        family = "Smoke/GPU_reference_kernel_fwd_2d_NCHW_FP32_FP32"
        outside = "Smoke/GPU_reference_kernel_bwd_2d_NHWC_FP32_FP32"
        names = [
            f"{f}.Run/{g}" for f, g in itertools.product((family, outside), geometries)
        ]
        policy = {
            "reference_kernels": dict(
                self.policy["reference_kernels"], suite_pattern=family
            )
        }
        result, selected, _ = build_filter("Smoke/*", names, policy)
        self.assertIn(names[-1], selected)
        reversed_result, reversed_selected, _ = build_filter(
            "Smoke/*", list(reversed(names)), policy
        )
        self.assertEqual(result, reversed_result)
        self.assertEqual(set(selected), set(reversed_selected))
        self.assertEqual(reversed_selected, list(reversed(selected)))


if __name__ == "__main__":
    unittest.main()
