# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math
import unittest

import numpy as np
import hostnumerics as hv
import test_scalar_encoding_oracle as oracle


FORMATS = oracle.BINARY_FORMATS + (
    oracle.BinaryFormat(
        "fnuz_fp8_e4m3",
        hv.ScalarType.Float8E4M3Fnuz,
        8,
        4,
        3,
        8,
        0x7F,
        0x7F,
        None,
        0x80,
        signed_zero=False,
    ),
    oracle.BinaryFormat(
        "fnuz_fp8_e5m2",
        hv.ScalarType.Float8E5M2Fnuz,
        8,
        5,
        2,
        16,
        0x7F,
        0x7F,
        None,
        0x80,
        signed_zero=False,
    ),
)


def options():
    return hv.ScalarConversionOptions(
        hv.IntegerRounding.NearestEven,
        hv.IntegerOverflow.Saturate,
        floating_source_precision=hv.FloatingSourcePrecision.Direct,
        floating_overflow=hv.FloatingOverflow.SaturateFinite,
    )


def encode_values(kind, values, conversion):
    tensor = hv.Tensor(kind, hv.Shape([len(values)]))
    for i, value in enumerate(values):
        recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(value)), 0
        )
        hv.generate_at(tensor, i, recipe, conversion)
    return tensor


class GenerationPrecisionTests(unittest.TestCase):
    def test_direct_double_midpoints_for_each_encoding(self):
        for fmt in FORMATS:
            unity = fmt.exponent_bias << fmt.mantissa_bits
            minimum_normal = 1 << fmt.mantissa_bits
            if fmt.storage_bits <= 8:
                lower_codes = range(fmt.maximum_finite_raw)
            else:
                lower_codes = sorted(
                    set(range(32))
                    | set(range(unity - 16, unity + 32))
                    | set(range(minimum_normal - 8, minimum_normal + 8))
                    | set(range(fmt.maximum_finite_raw - 8, fmt.maximum_finite_raw))
                )
            values, expected = [], []
            for code in lower_codes:
                lo = oracle.finite_value(fmt, code)
                hi = oracle.finite_value(fmt, code + 1)
                midpoint = (lo + hi) / 2
                for value, result in (
                    (math.nextafter(midpoint, -math.inf), code),
                    (midpoint, code if code % 2 == 0 else code + 1),
                    (math.nextafter(midpoint, math.inf), code + 1),
                ):
                    values.append(value)
                    expected.append(result)
                    if fmt.signed:
                        values.append(-value)
                        expected.append(
                            result | fmt.sign_mask if result or fmt.signed_zero else 0
                        )
            with self.subTest(format=fmt.name):
                tensor = encode_values(fmt.scalar_type, values, options())
                self.assertEqual(
                    oracle.unpack_little_endian(
                        tensor.storage, len(values), fmt.storage_bits
                    ),
                    expected,
                )

    def test_explicit_float_intermediate_and_e8m0_cutoff(self):
        value = 1 + 2**-8 + 2**-30
        conversion = options()
        self.assertEqual(
            encode_values(hv.ScalarType.BFloat16, [value], conversion).storage,
            bytes.fromhex("813f"),
        )
        conversion.floating_source_precision = hv.FloatingSourcePrecision.Float32
        self.assertEqual(
            encode_values(hv.ScalarType.BFloat16, [value], conversion).storage,
            bytes.fromhex("803f"),
        )
        conversion = options()
        tiny = math.ldexp(1.75, -127)
        self.assertEqual(
            encode_values(hv.ScalarType.E8M0, [tiny], conversion).storage, b"\x01"
        )
        conversion.e8m0_underflow = hv.E8M0Underflow.FlushBelowFloat32Normal
        self.assertEqual(
            encode_values(hv.ScalarType.E8M0, [tiny, 2**-126], conversion).storage,
            b"\x00\x01",
        )
        self.assertEqual(
            encode_values(
                hv.ScalarType.E8M0, [1.5 - 2**-30, 1.5, 1.5 + 2**-30], conversion
            ).storage,
            b"\x7f\x80\x80",
        )
        for fmt in FORMATS:
            with self.subTest(overflow=fmt.name):
                values = [math.inf, -math.inf] if fmt.signed else [math.inf]
                expected = [fmt.maximum_finite_raw]
                if fmt.signed:
                    expected.append(fmt.maximum_finite_raw | fmt.sign_mask)
                tensor = encode_values(fmt.scalar_type, values, conversion)
                self.assertEqual(
                    oracle.unpack_little_endian(
                        tensor.storage, len(values), fmt.storage_bits
                    ),
                    expected,
                )

    def test_binomial_distribution_and_sampling_offset(self):
        trials = hv.BinomialHalfGenerationParameters(21)
        recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.binomial_half(trials), 31
        )
        values = np.asarray(
            hv.generate_tensor(hv.ScalarType.Float64, [131072], recipe).values,
            dtype=int,
        )
        counts = np.bincount(values, minlength=22)
        expected = np.asarray(
            [math.comb(21, k) / 2**21 * len(values) for k in range(22)]
        )
        self.assertTrue(np.all(np.abs(counts - expected) <= 7 * np.sqrt(expected) + 8))
        shifted = hv.generate_tensor(
            hv.ScalarType.Float64, [43], recipe.with_index_offset(11)
        )
        np.testing.assert_array_equal(shifted.values, values[11:54])
        powers = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.binomial_half(trials)
            .with_exp2_transform()
            .with_affine_value_mapping(hv.GenerationAffineValueParameters(2**-11, 0)),
            31,
        )
        np.testing.assert_array_equal(
            hv.generate_tensor(hv.ScalarType.Float64, [43], powers).values,
            np.exp2(values[:43] - 11),
        )
        with self.assertRaises(RuntimeError):
            hv.GenerationRecipe.binomial_half(hv.BinomialHalfGenerationParameters(65))
        with self.assertRaises(RuntimeError):
            hv.generate_tensor(
                hv.ScalarType.Float64, [2], recipe.with_index_offset(2**64 - 1)
            )

    def test_independent_mx_codes_and_scales(self):
        settings = hv.GenerationRecipeSettings(0, [1, 0])
        recipe = hv.GenerationRecipe.real_only(hv.GenerationRecipe.sine(), settings)
        mx_options = hv.MxGenerationOptions()
        mx_options.scale = hv.MxScaleGenerationMode.One
        zero = hv.GenerationRecipe.real_only(hv.GenerationRecipe.zero(), settings)
        tensor = hv.generate_mx([40, 4], hv.MxDataGeneration.quantize(zero), mx_options)
        self.assertEqual(math.prod(tensor.scales.shape), 8)
        hv.generate_tensor(tensor.data, recipe, options())
        constant = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.constant(hv.ConstantGenerationParameters(0.5)), 0
        )
        hv.generate_tensor(tensor.scales, constant, options())
        tensor.refresh_reference()
        actual = np.asarray(tensor.reference.values).reshape(40, 4)
        magnitudes = np.asarray([0, 0.5, 1, 1.5, 2, 3, 4, 6])
        phases = np.arange(160).reshape(4, 40).T
        source = np.sin(phases)
        # These sine values avoid encoding ties; nearest finite magnitudes are independent of HN.
        expected = magnitudes[
            np.argmin(np.abs(np.abs(source)[..., None] - magnitudes), axis=-1)
        ]
        expected *= np.sign(source) * 0.5
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[1, 0], 0.5)

    def test_absolute_normal_scale_code_probabilities(self):
        recipe = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.normal(
                hv.NormalGenerationParameters(0.34375, 0.08)
            ).with_absolute_transform(),
            0x12345,
        )
        conversion = options()
        conversion.e8m0_underflow = hv.E8M0Underflow.FlushBelowFloat32Normal
        count = 65536

        def cdf(x):
            return (
                math.erf((x - 0.34375) / (0.08 * math.sqrt(2)))
                - math.erf((-x - 0.34375) / (0.08 * math.sqrt(2)))
            ) / 2

        for kind in (hv.ScalarType.E8M0, hv.ScalarType.E4M3, hv.ScalarType.E5M3):
            if kind == hv.ScalarType.E8M0:
                boundaries = [2**-126] + [
                    1.5 * 2 ** (code - 127) for code in range(1, 254)
                ]
            else:
                fmt = next(fmt for fmt in FORMATS if fmt.scalar_type == kind)
                boundaries = [
                    (
                        oracle.finite_value(fmt, code)
                        + oracle.finite_value(fmt, code + 1)
                    )
                    / 2
                    for code in range(fmt.maximum_finite_raw)
                ]
            probabilities = np.diff([0] + [cdf(x) for x in boundaries] + [1])
            tensor = hv.generate_tensor(kind, [count], recipe, conversion)
            counts = np.bincount(
                np.frombuffer(tensor.storage, dtype=np.uint8),
                minlength=len(probabilities),
            )[: len(probabilities)]
            expected = count * probabilities
            with self.subTest(scale=kind):
                self.assertEqual(sum(counts), count)
                self.assertTrue(
                    np.all(np.abs(counts - expected) <= 7 * np.sqrt(expected) + 8)
                )

    def test_offset_raw_complex_and_unsigned_rejection(self):
        raw = hv.GenerationRecipe.real_only_last_dimension_fastest(
            hv.GenerationRecipe.uniform_finite_encoded_value(), 57
        )
        full = hv.generate_tensor(hv.ScalarType.Float4E2M1, [128], raw, options())
        shifted = hv.generate_tensor(
            hv.ScalarType.Float4E2M1, [32], raw.with_index_offset(10), options()
        )
        self.assertEqual(shifted.storage, full.storage[5:21])
        settings = hv.GenerationRecipeSettings(19, [0])
        complex_recipe = hv.GenerationRecipe.cartesian(
            hv.GenerationRecipe.sine(), hv.GenerationRecipe.cosine(), settings
        )
        full = hv.generate_tensor(
            hv.ScalarType.ComplexFloat64, [32], complex_recipe, options()
        )
        shifted = hv.generate_tensor(
            hv.ScalarType.ComplexFloat64,
            [7],
            complex_recipe.with_index_offset(5),
            options(),
        )
        np.testing.assert_array_equal(shifted.values, full.values[5:12])
        for kind in (hv.ScalarType.E8M0, hv.ScalarType.E4M3, hv.ScalarType.E5M3):
            with self.subTest(unsigned=kind):
                with self.assertRaises(RuntimeError):
                    encode_values(kind, [-math.inf], options())
