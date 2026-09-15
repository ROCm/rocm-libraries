# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from _host_numerics_test_support import *  # noqa: F403


class CodecTests(unittest.TestCase):
    def assert_values_equal(self, observed, expected):
        observed = np.asarray(observed)
        expected = np.asarray(expected)
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
        np.testing.assert_array_equal(np.isposinf(observed), np.isposinf(expected))
        np.testing.assert_array_equal(np.isneginf(observed), np.isneginf(expected))
        finite = np.isfinite(expected)
        np.testing.assert_array_equal(observed[finite], expected[finite])
        zeros = finite & (expected == 0)
        np.testing.assert_array_equal(
            np.signbit(observed[zeros]), np.signbit(expected[zeros])
        )

    def test_exhaustive_low_precision_codecs(self):
        for scalar_type, format_info in FORMATS.items():
            with self.subTest(scalar_type=scalar_type):
                bits = format_info[0]
                raw = list(range(1 << bits))
                tensor = hv.Tensor.from_storage(
                    scalar_type, [len(raw)], pack_bits(raw, bits)
                )
                observed = hv.to_numpy(tensor, np.float32)
                expected = np.asarray(
                    [expected_value(scalar_type, value) for value in raw],
                    dtype=np.float32,
                )
                self.assert_values_equal(observed, expected)

                encodable = ~np.isnan(expected)
                encoded = hv.from_numpy(expected[encodable], scalar_type)
                round_trip = unpack_bits(encoded.storage, int(encodable.sum()), bits)
                expected_raw = np.asarray(raw, dtype=np.uint32)
                if scalar_type == hv.ScalarType.E4M3:
                    expected_raw &= 0x7F
                np.testing.assert_array_equal(round_trip, expected_raw[encodable])

    def test_exhaustive_float16_decode(self):
        raw = np.arange(1 << 16, dtype="<u2")
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.Float16, [raw.size], raw.tobytes()
        )
        observed = hv.to_numpy(tensor, np.float32)
        expected = raw.view("<f2").astype(np.float32)
        self.assert_values_equal(observed, expected)

    def test_exhaustive_bfloat16_decode_and_finite_round_trip(self):
        raw = np.arange(1 << 16, dtype="<u2")
        tensor = hv.Tensor.from_storage(
            hv.ScalarType.BFloat16, [raw.size], raw.tobytes()
        )
        observed = hv.to_numpy(tensor, np.float32)
        expected_bits = raw.astype("<u4") << 16
        expected = expected_bits.view("<f4")
        self.assert_values_equal(observed, expected)

        encoded = hv.from_numpy(expected, hv.ScalarType.BFloat16)
        encoded_raw = np.frombuffer(encoded.storage, dtype="<u2")
        finite = ~np.isnan(expected)
        np.testing.assert_array_equal(encoded_raw[finite], raw[finite])

    def test_e8m0_has_no_zero_encoding(self):
        raw = bytes([0, 1, 127, 128, 254, 255])
        tensor = hv.Tensor.from_storage(hv.ScalarType.E8M0, [len(raw)], raw)
        observed = hv.to_numpy(tensor, np.float32)
        expected = np.asarray(
            [
                math.ldexp(1.0, -127),
                math.ldexp(1.0, -126),
                1.0,
                2.0,
                math.ldexp(1.0, 127),
                math.nan,
            ],
            dtype=np.float32,
        )
        self.assert_values_equal(observed, expected)

    def test_e8m0_zero_uses_exponent_field_with_zero_sentinel(self):
        raw = bytes([0, 1, 127, 128, 254, 255])
        tensor = hv.Tensor.from_storage(hv.ScalarType.E8M0Zero, [len(raw)], raw)
        observed = hv.to_numpy(tensor, np.float32)
        expected = np.asarray(
            [
                0.0,
                math.ldexp(1.0, -126),
                1.0,
                2.0,
                math.ldexp(1.0, 127),
                math.nan,
            ],
            dtype=np.float32,
        )
        self.assert_values_equal(observed, expected)

        encoded = hv.from_numpy(
            np.asarray([0.0, 1.75, -2.0, np.inf], dtype=np.float32),
            hv.ScalarType.E8M0Zero,
        )
        self.assertEqual(encoded.storage, bytes([0, 127, 128, 255]))
