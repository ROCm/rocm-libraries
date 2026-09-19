# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _mx_generation_oracle_support import *  # noqa: F403


class MxGenerationOracleTests(unittest.TestCase):
    def assert_float_values_equal(self, observed, expected):
        observed = np.asarray(observed, dtype=np.float32)
        expected = np.asarray(expected, dtype=np.float32)
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
        not_nan = ~np.isnan(expected)
        np.testing.assert_array_equal(observed[not_nan], expected[not_nan])
        zeros = not_nan & (expected == 0.0)
        np.testing.assert_array_equal(
            np.signbit(observed[zeros]), np.signbit(expected[zeros])
        )

    def assert_matches_oracle(self, case):
        expected = expected_mx(case)
        observed = hv.generate_mx(*make_problem(case))
        rows = case.shape[0]
        leading_dimension = case.leading_dimension or rows
        blocked_extent = case.shape[case.block_axis]
        free_extent = case.shape[1 - case.block_axis]
        block_count = (blocked_extent + case.block_size - 1) // case.block_size
        expected_scale_shape = (
            (free_extent, block_count)
            if case.block_axis == 0
            else (block_count, free_extent)
        )

        self.assertEqual(observed.data.strides, [1, leading_dimension])
        self.assertEqual(observed.scales.shape, list(expected_scale_shape))
        self.assertEqual(observed.data.storage, expected.data_storage)
        self.assertEqual(observed.scales.storage, expected.scale_storage)
        self.assertEqual(observed.scale_indices.strides, [1, rows])
        self.assertEqual(observed.reference.strides, [1, rows])
        np.testing.assert_array_equal(
            hv.to_numpy(observed.scale_indices, np.uint32),
            expected.scale_indices,
        )
        self.assert_float_values_equal(
            hv.to_numpy(observed.reference, np.float32),
            expected.reference,
        )
        return observed, expected

    def test_every_mx_generation_mode_matches_exact_oracle(self):
        recipes = (
            Recipe(RecipeKind.BOUNDED, lower=-1.25, upper=0.875),
            Recipe(RecipeKind.BOUNDED_ALTERNATING_SIGN, maximum_magnitude=1.25),
            Recipe(RecipeKind.UNBOUNDED),
            Recipe(RecipeKind.IDENTITY),
            Recipe(RecipeKind.CONSTANT, value=1.0),
            Recipe(RecipeKind.CONSTANT, value=0.0),
            Recipe(RecipeKind.SEQUENTIAL),
            Recipe(RecipeKind.ROW_INDEX),
            Recipe(RecipeKind.COLUMN_INDEX),
            Recipe(RecipeKind.CHECKERBOARD),
            Recipe(RecipeKind.SCALED_DIAGONAL),
            Recipe(RecipeKind.CONSTANT, value=2.0),
            Recipe(RecipeKind.CONSTANT, value=-1.0),
            Recipe(RecipeKind.TYPE_MAXIMUM),
            Recipe(RecipeKind.TYPE_DENORMAL_MINIMUM),
            Recipe(RecipeKind.TYPE_DENORMAL_MAXIMUM),
            Recipe(RecipeKind.TYPE_NAN),
            Recipe(RecipeKind.TYPE_INFINITY),
            Recipe(RecipeKind.TRIGONOMETRIC),
            Recipe(RecipeKind.NORMAL, mean=0.25, standard_deviation=0.75),
            Recipe(RecipeKind.UNIFORM_INTEGER, integer_lower=-3, integer_upper=4),
        )

        for recipe in recipes:
            case = MxCase(
                label=recipe.kind.name,
                data_type=hv.ScalarType.Float8E5M2,
                scale_type=hv.ScalarType.E8M0,
                shape=(3, 4),
                leading_dimension=5,
                block_axis=1,
                block_size=3,
                data=recipe,
            )
            with self.subTest(mode=case.label):
                self.assert_matches_oracle(case)

    def test_all_supported_pairs_axes_partial_blocks_and_scale_selection(self):
        cases = (
            MxCase(
                "fp4_e8m0_axis0",
                hv.ScalarType.Float4E2M1,
                hv.ScalarType.E8M0,
                (5, 3),
                7,
                0,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp6_e2m3_e8m0_axis1",
                hv.ScalarType.Float6E2M3,
                hv.ScalarType.E8M0,
                (3, 5),
                5,
                1,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp6_e3m2_e8m0_axis0",
                hv.ScalarType.Float6E3M2,
                hv.ScalarType.E8M0,
                (5, 2),
                6,
                0,
                3,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp8_e4m3_e8m0_axis1",
                hv.ScalarType.Float8E4M3,
                hv.ScalarType.E8M0,
                (3, 5),
                4,
                1,
                2,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp8_e5m2_e8m0_axis0",
                hv.ScalarType.Float8E5M2,
                hv.ScalarType.E8M0,
                (5, 3),
                8,
                0,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp4_e4m3_axis1",
                hv.ScalarType.Float4E2M1,
                hv.ScalarType.E4M3,
                (3, 5),
                5,
                1,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
            MxCase(
                "fp4_e5m3_axis0",
                hv.ScalarType.Float4E2M1,
                hv.ScalarType.E5M3,
                (5, 3),
                6,
                0,
                4,
                Recipe(RecipeKind.BOUNDED, lower=-0.9, upper=1.1),
                seed=12345,
            ),
        )
        for case in cases:
            with self.subTest(case=case.label):
                _, expected = self.assert_matches_oracle(case)
                if case.scale_type in (hv.ScalarType.E4M3, hv.ScalarType.E5M3):
                    self.assertGreater(len(set(expected.scale_storage)), 1)

    def test_fp6_packing_tails_are_exact(self):
        for data_type in (hv.ScalarType.Float6E2M3, hv.ScalarType.Float6E3M2):
            for leading_dimension in (5, 6, 7):
                case = MxCase(
                    label=f"{data_type.name}_ld{leading_dimension}",
                    data_type=data_type,
                    scale_type=hv.ScalarType.E8M0,
                    shape=(5, 1),
                    leading_dimension=leading_dimension,
                    block_axis=0,
                    block_size=4,
                    data=Recipe(RecipeKind.UNBOUNDED),
                    seed=0xA5A5,
                )
                with self.subTest(
                    data_type=data_type.name, leading_dimension=leading_dimension
                ):
                    observed, _ = self.assert_matches_oracle(case)
                    self.assertEqual(
                        len(observed.data.storage),
                        (leading_dimension * 6 + 7) // 8,
                    )

    def test_independent_constant_scales_are_exact(self):
        constants = (
            (hv.ScalarType.E8M0, 0x80),
            (hv.ScalarType.E4M3, 0x40),
            (hv.ScalarType.E5M3, 0x80),
        )
        for scale_type, raw_two in constants:
            case = MxCase(
                label=f"constant_{scale_type.name}",
                data_type=hv.ScalarType.Float4E2M1,
                scale_type=scale_type,
                shape=(3, 3),
                leading_dimension=4,
                block_axis=0,
                block_size=2,
                data=Recipe(RecipeKind.SEQUENTIAL),
                scale=hv.MxScaleGenerationMode.Two,
            )
            with self.subTest(scale_type=scale_type.name):
                observed, _ = self.assert_matches_oracle(case)
                self.assertEqual(observed.scales.storage, bytes([raw_two]) * 6)

    def test_invalid_bounds_are_rejected(self):
        invalid_data_recipes = (
            (
                "bounded_equal",
                Recipe(RecipeKind.BOUNDED, lower=1.0, upper=1.0),
            ),
            (
                "bounded_reversed",
                Recipe(RecipeKind.BOUNDED, lower=2.0, upper=-1.0),
            ),
            (
                "bounded_nonfinite",
                Recipe(RecipeKind.BOUNDED, lower=-1.0, upper=math.inf),
            ),
            (
                "alternating_nonfinite",
                Recipe(
                    RecipeKind.BOUNDED_ALTERNATING_SIGN,
                    maximum_magnitude=math.inf,
                ),
            ),
            (
                "normal_negative_sigma",
                Recipe(RecipeKind.NORMAL, mean=0.0, standard_deviation=-0.1),
            ),
            (
                "normal_nan_mean",
                Recipe(
                    RecipeKind.NORMAL,
                    mean=math.nan,
                    standard_deviation=1.0,
                ),
            ),
            (
                "integer_reversed",
                Recipe(
                    RecipeKind.UNIFORM_INTEGER,
                    integer_lower=4,
                    integer_upper=-3,
                ),
            ),
        )
        for label, recipe in invalid_data_recipes:
            case = MxCase(
                label,
                hv.ScalarType.Float8E5M2,
                hv.ScalarType.E8M0,
                (3, 3),
                3,
                0,
                2,
                recipe,
            )
            with self.subTest(recipe=label):
                with self.assertRaises(ValueError):
                    hv.generate_mx(*make_problem(case))

        with self.assertRaises(TypeError):
            hv.UniformIntegerGenerationParameters(0, 1 << 40)

        no_infinity = MxCase(
            "fp4_infinity",
            hv.ScalarType.Float4E2M1,
            hv.ScalarType.E8M0,
            (3, 3),
            3,
            0,
            2,
            Recipe(RecipeKind.TYPE_INFINITY),
        )
        with self.assertRaises(ValueError):
            hv.generate_mx(*make_problem(no_infinity))

        impossible_interval = MxCase(
            "unrepresentable_bounded_interval",
            hv.ScalarType.Float4E2M1,
            hv.ScalarType.E8M0,
            (3, 1),
            3,
            0,
            2,
            Recipe(RecipeKind.BOUNDED, lower=0.1, upper=0.2),
            scale=hv.MxScaleGenerationMode.One,
        )
        with self.assertRaises(ValueError):
            hv.generate_mx(*make_problem(impossible_interval))

    def test_invalid_geometry_and_type_pairs_are_rejected(self):
        base = dict(
            label="invalid",
            data_type=hv.ScalarType.Float4E2M1,
            scale_type=hv.ScalarType.E8M0,
            shape=(3, 3),
            leading_dimension=3,
            block_axis=0,
            block_size=2,
            data=Recipe(RecipeKind.CONSTANT, value=1.0),
        )

        empty = hv.generate_mx(*make_problem(MxCase(**(base | {"shape": (0, 3)}))))
        self.assertEqual(empty.data.size, 0)
        self.assertEqual(empty.scales.size, 0)
        self.assertEqual(empty.scale_indices.size, 0)
        self.assertEqual(empty.reference.size, 0)

        invalid_cases = (
            ("rank", MxCase(**(base | {"shape": (3,)})), ValueError),
            (
                "short_leading_dimension",
                MxCase(**(base | {"leading_dimension": 2})),
                ValueError,
            ),
            ("block_axis", MxCase(**(base | {"block_axis": 2})), IndexError),
            ("zero_block", MxCase(**(base | {"block_size": 0})), ValueError),
            (
                "unsupported_data",
                MxCase(**(base | {"data_type": hv.ScalarType.Float16})),
                ValueError,
            ),
            (
                "unsupported_scale",
                MxCase(**(base | {"scale_type": hv.ScalarType.Float32})),
                ValueError,
            ),
            (
                "unsupported_pair",
                MxCase(
                    **(
                        base
                        | {
                            "data_type": hv.ScalarType.Float6E2M3,
                            "scale_type": hv.ScalarType.E4M3,
                        }
                    )
                ),
                ValueError,
            ),
        )
        for label, case, error in invalid_cases:
            with self.subTest(case=label):
                with self.assertRaises(error):
                    hv.generate_mx(*make_problem(case))
