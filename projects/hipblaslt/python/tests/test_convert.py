# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
import numpy as np
import pytest
import hipblaslt
c = hipblaslt._core


def test_pack_unpack_roundtrip_small():
    # Values exactly representable in e4m3.
    vals = np.array([0.0, 1.0, 2.0, 0.5, -1.0, 4.0], dtype=np.float32)
    packed = c.pack_fp8(vals, "e4m3")
    assert packed.dtype == np.uint8
    assert packed.shape == vals.shape
    restored = c.unpack_fp8(packed, "e4m3")
    np.testing.assert_array_equal(restored, vals)


def test_pack_unpack_all_formats():
    # Smoke test all four formats with a small set of values.
    vals = np.array([0.0, 1.0, -1.0, 0.25], dtype=np.float32)
    for fmt in ("e4m3", "e5m2", "e4m3_fnuz", "e5m2_fnuz"):
        packed = c.pack_fp8(vals, fmt)
        assert packed.dtype == np.uint8, f"dtype mismatch for fmt={fmt}"
        assert packed.shape == vals.shape, f"shape mismatch for fmt={fmt}"
        restored = c.unpack_fp8(packed, fmt)
        assert restored.dtype == np.float32, f"output dtype mismatch for fmt={fmt}"
        assert restored.shape == vals.shape, f"output shape mismatch for fmt={fmt}"
