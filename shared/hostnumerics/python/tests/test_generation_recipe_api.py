# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

import sys

import hostnumerics as hv


class GenerationRecipeApiTests(unittest.TestCase):
    def test_selection_and_recipe_settings_require_explicit_choices(self):
        with self.assertRaises(TypeError):
            hv.GenerationRecipeSettings()
        with self.assertRaises(TypeError):
            hv.GenerationRecipeSettings(seed=37)
        with self.assertRaises(TypeError):
            hv.IndexRange(0, 2)

    def test_strict_and_named_recipe_orders(self):
        component = hv.GenerationRecipe.uniform_integer(
            hv.UniformIntegerGenerationParameters(lower=-7, upper=7)
        )
        scalar = hv.GenerationRecipe.real_only(
            component, hv.GenerationRecipeSettings(37, [])
        )
        self.assertEqual(scalar.index_order(0), [])
        with self.assertRaises(RuntimeError):
            hv.GenerationRecipe.real_only(
                component, hv.GenerationRecipeSettings(37, [0, 0])
            )
        with self.assertRaises(RuntimeError):
            hv.generate_tensor(hv.ScalarType.Float32, [2, 3], scalar)
        for factory, components in (
            (hv.GenerationRecipe.real_only_last_dimension_fastest, [component]),
            (hv.GenerationRecipe.replicated_last_dimension_fastest, [component]),
            (
                hv.GenerationRecipe.cartesian_last_dimension_fastest,
                [component, component],
            ),
        ):
            named = factory(*components, seed=37)
            explicit = named.with_index_order([0, 1])
            self.assertEqual(named.index_order(2), [0, 1])
            self.assertEqual(named.index_order(3), [0, 1, 2])
            self.assertEqual(named.seed, 37)
            for scalar_type in (hv.ScalarType.Float32, hv.ScalarType.ComplexFloat32):
                left = hv.generate_tensor(scalar_type, [2, 3], named)
                right = hv.generate_tensor(scalar_type, [2, 3], explicit)
                self.assertEqual(left.storage, right.storage)
            with self.assertRaises(RuntimeError):
                explicit.index_order(3)
            with self.assertRaises(RuntimeError):
                named.with_index_order([0, 0])

    def test_recipes_copy_order_sequences(self):
        for source in ([2, 1, 0], (2, 1, 0)):
            settings = hv.GenerationRecipeSettings(index_order=source, seed=0)
            base = hv.GenerationRecipe.real_only_last_dimension_fastest(
                hv.GenerationRecipe.serial_index(),
                seed=0,
            )
            recipe = base.with_index_order(source)
            if isinstance(source, list):
                source[:] = [0, 1, 2]
            self.assertEqual(settings.index_order, [2, 1, 0])
            self.assertEqual(recipe.index_order(3), [2, 1, 0])
            returned_order = recipe.index_order(3)
            returned_order[:] = [0, 1, 2]
            self.assertEqual(recipe.index_order(3), [2, 1, 0])
            tensor = hv.generate_tensor(hv.ScalarType.Float32, [2, 3, 4], recipe)
            np.testing.assert_array_equal(
                hv.to_numpy(tensor, dtype=None),
                np.arange(24, dtype=np.float32).reshape((2, 3, 4), order="F"),
            )

    def test_named_parameters_and_real_only_recipe(self):
        settings = hv.GenerationRecipeSettings(
            seed=37,
            index_order=[0, 1],
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
            hv.generate_tensor(hv.ScalarType.Float32, [2, 3], recipe), dtype=None
        )
        np.testing.assert_array_equal(
            observed,
            np.asarray([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0]], dtype=np.float32),
        )
        self.assertEqual(recipe.seed, 37)
        self.assertEqual(
            recipe.index_order(2),
            [0, 1],
        )

    def test_explicit_complex_policy_factories(self):
        real = hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=2.0))
        imaginary = hv.GenerationRecipe.constant(
            hv.ConstantGenerationParameters(value=-3.0)
        )

        real_only = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32,
            hv.Shape([2]),
            hv.GenerationRecipe.real_only_last_dimension_fastest(real, seed=0),
        )
        replicated = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32,
            [2],
            hv.GenerationRecipe.replicated_last_dimension_fastest(real, seed=0),
        )
        cartesian = hv.generate_tensor(
            hv.ScalarType.ComplexFloat32,
            [2],
            hv.GenerationRecipe.cartesian_last_dimension_fastest(
                real, imaginary, seed=0
            ),
        )

        np.testing.assert_array_equal(
            hv.to_numpy(real_only, dtype=None),
            np.asarray([2.0 + 0.0j, 2.0 + 0.0j], dtype=np.complex64),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(replicated, dtype=None),
            np.asarray([2.0 + 2.0j, 2.0 + 2.0j], dtype=np.complex64),
        )
        np.testing.assert_array_equal(
            hv.to_numpy(cartesian, dtype=None),
            np.asarray([2.0 - 3.0j, 2.0 - 3.0j], dtype=np.complex64),
        )

    def test_layout_generation_and_generate_at_use_native_recipe_api(self):
        layout = hv.Layout(hv.Shape([2, 2]), [3, 1], 1)
        recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.serial_index(),
            seed=0,
        )
        tensor = hv.generate_tensor(hv.ScalarType.Float32, layout, recipe)

        np.testing.assert_array_equal(
            hv.to_numpy(tensor, dtype=None),
            np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        )
        storage = np.frombuffer(tensor.storage, dtype=np.float32)
        np.testing.assert_array_equal(
            storage,
            np.asarray([0.0, 0.0, 1.0, 0.0, 2.0, 3.0], dtype=np.float32),
        )

        replacement = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=9.0)),
            seed=0,
        )
        self.assertIs(hv.generate_at(tensor, 1, replacement), tensor)
        np.testing.assert_array_equal(
            hv.to_numpy(tensor, dtype=None),
            np.asarray([[0.0, 9.0], [2.0, 3.0]], dtype=np.float32),
        )

    def test_typed_factories_validate_named_parameters(self):
        with self.assertRaises(RuntimeError):
            hv.GenerationRecipe.choice(hv.ChoiceGenerationParameters(values=[]))
        with self.assertRaises(RuntimeError):
            hv.GenerationRecipe.uniform_real(
                hv.UniformRealGenerationParameters(lower=2.0, upper=-1.0)
            )
        with self.assertRaises(RuntimeError):
            hv.GenerationRecipe.replicated_last_dimension_fastest(
                hv.GenerationRecipe.raw_constant(
                    hv.RawConstantGenerationParameters(bits=1)
                ),
                seed=0,
            )

    def test_all_named_parameter_factories_are_bound(self):
        components = [
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value=1.0)),
            hv.GenerationRecipe.choice(
                hv.ChoiceGenerationParameters(values=[-1.0, 1.0])
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

    def test_mutable_generation_bags_are_not_exposed(self):
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
            hv.ScalarType.Float4E2M1, [2], bytes([0x21]), strides=None, offset=0
        )
        self.assertEqual(len(tensor.storage), 1)
        self.assertEqual(hv.to_numpy(tensor, dtype=None).dtype, np.dtype(np.float32))

    def test_only_canonical_index_order_name_is_exposed(self):
        self.assertFalse(hasattr(hv, "IndexOrder"))
        self.assertFalse(hasattr(hv, "LogicalIndexOrder"))
        self.assertFalse(hasattr(hv, "ComparisonIndexOrder"))


if __name__ == "__main__":
    unittest.main()
