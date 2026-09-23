# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Core dtype names are independent of target support and retain public aliases."""

import pytest

from rocke.core.dtypes import normalize_dtype


@pytest.mark.parametrize(
    "spelling,canonical",
    [
        (" HALF ", "fp16"),
        ("bfloat16", "bf16"),
        (" Float\t", "fp32"),
        ("FP8", "fp8e4m3"),
        ("BF8", "bf8e5m2"),
        ("FP6", "fp6e2m3"),
        ("BF6", "fp6e3m2"),
        ("FP4", "fp4e2m1"),
        ("int32", "i32"),
        (" Custom_Format ", "custom_format"),
        ("", ""),
    ],
)
def test_normalize_dtype(spelling, canonical):
    assert normalize_dtype(spelling) == canonical
    assert normalize_dtype(canonical) == canonical


def test_architecture_entry_points_reexport_core_normalization():
    from rocke.core import normalize_dtype as core_normalize
    from rocke.core.arch import normalize_dtype as arch_normalize
    from rocke.core.arch.target import normalize_dtype as target_normalize

    assert normalize_dtype.__module__ == "rocke.core.dtypes"
    assert core_normalize is arch_normalize is target_normalize is normalize_dtype
