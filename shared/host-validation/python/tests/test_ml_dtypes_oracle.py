# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import ml_dtypes
import numpy as np

import roc_host_validation as hv


FORMATS = (
    ("bfloat16", hv.ScalarType.BFloat16, ml_dtypes.bfloat16, 16),
    ("float4_e2m1fn", hv.ScalarType.Float4E2M1, ml_dtypes.float4_e2m1fn, 4),
    ("float6_e2m3fn", hv.ScalarType.Float6E2M3, ml_dtypes.float6_e2m3fn, 6),
    ("float6_e3m2fn", hv.ScalarType.Float6E3M2, ml_dtypes.float6_e3m2fn, 6),
    ("float8_e4m3fn", hv.ScalarType.Float8E4M3, ml_dtypes.float8_e4m3fn, 8),
    ("float8_e5m2", hv.ScalarType.Float8E5M2, ml_dtypes.float8_e5m2, 8),
    (
        "float8_e4m3fnuz",
        hv.ScalarType.Float8E4M3Fnuz,
        ml_dtypes.float8_e4m3fnuz,
        8,
    ),
    (
        "float8_e5m2fnuz",
        hv.ScalarType.Float8E5M2Fnuz,
        ml_dtypes.float8_e5m2fnuz,
        8,
    ),
    ("float8_e8m0fnu", hv.ScalarType.E8M0, ml_dtypes.float8_e8m0fnu, 8),
)


def pack_host_codes(codes, bits):
    if bits == 16:
        return np.asarray(codes, dtype="<u2").tobytes()
    if bits == 8:
        return np.asarray(codes, dtype=np.uint8).tobytes()

    storage = bytearray((len(codes) * bits + 7) // 8)
    for index, code in enumerate(codes):
        bit_offset = index * bits
        for bit in range(bits):
            if (int(code) >> bit) & 1:
                absolute = bit_offset + bit
                storage[absolute // 8] |= 1 << (absolute % 8)
    return bytes(storage)


class MlDtypesOracleTests(unittest.TestCase):
    def assert_same_float_values(self, observed, expected):
        observed = np.asarray(observed)
        expected = np.asarray(expected)
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
        not_nan = ~np.isnan(expected)
        np.testing.assert_array_equal(observed[not_nan], expected[not_nan])
        zeros = not_nan & (expected == 0)
        np.testing.assert_array_equal(
            np.signbit(observed[zeros]), np.signbit(expected[zeros])
        )

    def test_exhaustive_raw_decode_mappings(self):
        # ml_dtypes supplies the value oracle only. Host packing, quantization,
        # saturation, and NaN policies remain covered by the raw-bit tests.
        for name, scalar_type, oracle_type, bits in FORMATS:
            with self.subTest(format=name):
                storage_type = np.uint16 if bits == 16 else np.uint8
                raw = np.arange(1 << bits, dtype=storage_type)
                expected = raw.view(np.dtype(oracle_type)).astype(np.float32)
                tensor = hv.Tensor.from_storage(
                    scalar_type,
                    [raw.size],
                    pack_host_codes(raw, bits),
                )
                observed = hv.to_numpy(tensor, np.float32)
                self.assert_same_float_values(observed, expected)


if __name__ == "__main__":
    unittest.main()
