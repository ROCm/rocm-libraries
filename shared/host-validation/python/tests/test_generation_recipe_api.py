# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

import roc_host_validation as hv


class GenerationRecipeApiTests(unittest.TestCase):
    def test_named_parameters_and_real_only_recipe(self):
        settings = hv.GenerationRecipeSettings(
            seed=37,
            index_order=hv.IndexOrder.LastDimensionFastest,
        )
        component = hv.GenerationRecipe.uniform_integer(
            hv.UniformIntegerGenerationParameters(lower=1, upper=1)
        )
        component = component.with_affine_value_mapping(
            hv.GenerationAffineValueParameters(scale=2.0, offset=-1.0)
        )
        component = component.with_alternating_sign(
            hv.AlternatingSignGenerationParameters(
                dimensions=[0, 1],
                negative_when_odd=True,
            )
        )
        recipe = hv.GenerationRecipe.real_only(component, settings)

        observed = hv.to_numpy(
            hv.generate_tensor(hv.ScalarType.Float32, [2, 3], recipe)
        )
        np.testing.assert_array_equal(
            observed,
            np.asarray([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=np.float32),
        )
        self.assertEqual(recipe.seed, 37)
        self.assertEqual(
            recipe.index_order,
            hv.IndexOrder.LastDimensionFastest,
        )

    def test_explicit_complex_policy_factories(self):
        real = hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=2.0))
        imaginary = hv.GenerationRecipe.constant(
            hv.ConstantGenerationParameters(value=-3.0)
        )

        real_only = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32,
            hv.Shape([2]),
            hv.GenerationRecipe.real_only(real),
        )
        replicated = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32,
            [2],
            hv.GenerationRecipe.replicated(real),
        )
        cartesian = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32,
            [2],
            hv.GenerationRecipe.cartesian(real, imaginary),
        )

        np.testing.assert_array_equal(
            hv.to_numpy(real_only),
            np.asarray([2.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(replicated),
            np.asarray([2.0 + 2.0j, 2.0 + 2.0j], dtype=np.complex64),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(cartesian),
            np.asarray([2.0 - 3.0j, 2.0 - 3.0j], dtype=np.complex64),
        )

    def test_layout_generation_and_generate_at_use_native_recipe_api(self):
        layout = hv.Layout(hv.Shape([2, 2]), [3, 1], 1)
        recipe = hv.GenerationRecipe.real_only(hv.GenerationRecipe.serial_index())
        tensor = hv.generate_tensor(hv.ScalarType.Float32, layout, recipe)

        np.testing.assert_array_equal(
            hv.to_numpy(tensor),
            np.asarray([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32),
        )
        storage = np.frombuffer(tensor.storage, dtype=np.float32)
        np.testing.assert_array_equal(
            storage,
            np.asarray([0.0, 0.0, 2.0, 0.0, 1.0, 3.0], dtype=np.float32),
        )

        replacement = hv.GenerationRecipe.real_only(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=9.0))
        )
        self.assertIs(hv.generate_at(tensor, 1, replacement), tensor)
        np.testing.assert_array_equal(
            hv.to_numpy(tensor),
            np.asarray([[0.0, 2.0], [9.0, 3.0]], dtype=np.float32),
        )

    def test_typed_factories_validate_named_parameters(self):
        with self.assertRaises(ValueError):
            hv.GenerationRecipe.candidate_set(
                hv.CandidateSetGenerationParameters(values=[])
            )
        with self.assertRaises(ValueError):
            hv.GenerationRecipe.uniform_real(
                hv.UniformRealGenerationParameters(lower=2.0, upper=-1.0)
            )
        with self.assertRaises(ValueError):
            hv.GenerationRecipe.replicated(
                hv.GenerationRecipe.raw_constant(
                    hv.RawConstantGenerationParameters(bits=1)
                )
            )

    def test_all_named_parameter_factories_are_bound(self):
        components = [
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=1.0)),
            hv.GenerationRecipe.candidate_set(
                hv.CandidateSetGenerationParameters(values=[-1.0, 1.0])
            ),
            hv.GenerationRecipe.uniform_integer(
                hv.UniformIntegerGenerationParameters(lower=-2, upper=2)
            ),
            hv.GenerationRecipe.absolute_uniform_integer(
                hv.UniformIntegerGenerationParameters(lower=-2, upper=2)
            ),
            hv.GenerationRecipe.uniform_real(
                hv.UniformRealGenerationParameters(lower=-1.0, upper=1.0)
            ),
            hv.GenerationRecipe.normal(
                hv.NormalGenerationParameters(
                    mean=0.25,
                    standard_deviation=0.5,
                )
            ),
            hv.GenerationRecipe.serial_dimension(
                hv.DimensionGenerationParameters(dimension=0)
            ),
            hv.GenerationRecipe.affine_index_remainder(
                hv.AffineIndexRemainderGenerationParameters(
                    dimension_coefficients=[2],
                    offset=1,
                    positive_divisor=3,
                )
            ),
            hv.GenerationRecipe.checkerboard_uniform_integer(
                hv.UniformIntegerGenerationParameters(lower=1, upper=2)
            ),
            hv.GenerationRecipe.random_encoded_exponent(
                hv.RandomEncodedExponentGenerationParameters(
                    lower_unbiased_exponent=-2,
                    upper_unbiased_exponent=2,
                    source_type=hv.ScalarType.Float32,
                )
            ),
            hv.GenerationRecipe.raw_constant(
                hv.RawConstantGenerationParameters(bits=1)
            ),
            hv.GenerationRecipe.uniform_raw_integer(
                hv.UniformIntegerGenerationParameters(lower=0, upper=3)
            ),
            hv.GenerationRecipe.raw_serial_dimension(
                hv.DimensionGenerationParameters(dimension=0)
            ),
        ]
        self.assertTrue(
            all(
                isinstance(component, hv.GenerationRecipe.Component)
                for component in components
            )
        )

    def test_legacy_generation_bags_are_not_exposed(self):
        for name in (
            "GenerationOptions",
            "GenerationPattern",
            "GenerationPatternSpec",
            "GenerationTransform",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(hv, name))
        self.assertNotIn("GenerationOptions", hv.generate_tensor.__doc__)
        self.assertNotIn("GenerationOptions", hv.generate_at.__doc__)

    def test_decoded_dtype_is_distinct_from_storage_encoding(self):
        self.assertEqual(
            hv.default_decoded_dtype(hv.ScalarType.Float4E2M1),
            np.dtype(np.float32),
        )
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.Float4E2M1,
            [2],
            bytes([0x21]),
        )
        self.assertEqual(len(tensor.storage), 1)
        self.assertEqual(hv.to_numpy(tensor).dtype, np.dtype(np.float32))

    def test_only_canonical_index_order_name_is_exposed(self):
        self.assertTrue(hasattr(hv, "IndexOrder"))
        self.assertFalse(hasattr(hv, "LogicalIndexOrder"))
        self.assertFalse(hasattr(hv, "ComparisonIndexOrder"))


if __name__ == "__main__":
    unittest.main()
