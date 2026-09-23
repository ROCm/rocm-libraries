# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU checks for the scaled-WMMA numerical oracle and diagnostic fixtures."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ml_dtypes")

from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    check_result,
    decode_e8m0,
    make_case_inputs,
    reference_result,
)
from rocke.instances.gfx1250.block_scaled_gemm import BlockScaledGemmSpec


def test_e8m0_golden_values_include_high_bit():
    encoded = np.array([125, 126, 127, 128, 129, 130], dtype=np.uint8)
    np.testing.assert_array_equal(decode_e8m0(encoded), [0.25, 0.5, 1, 2, 4, 8])
    with pytest.raises(ValueError, match="0xff"):
        decode_e8m0(np.array([255], dtype=np.uint8))


def test_reference_matches_hand_computed_group_scales():
    a = np.array([[1, 2, 3, 4], [2, 0, 1, -1]], dtype=np.float32)
    b = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=np.float32)
    sa = np.array([[127, 128], [126, 127]], dtype=np.uint8)
    sb = np.array([[127, 128, 126], [128, 127, 129]], dtype=np.uint8)
    expected = np.array([[13, 12, 57.5], [3, -1, 0.5]], dtype=np.float32)
    got = reference_result(a, b, sa, sb, 2, native=True)
    np.testing.assert_array_equal(got, expected)
    # Either operand's scale-group permutation must be observable.
    for wrong_sa, wrong_sb in ((sa[:, ::-1], sb), (sa, sb[::-1, :])):
        wrong = reference_result(a, b, wrong_sa, wrong_sb, 2, native=True)
        with pytest.raises(AssertionError, match="bad="):
            check_result(wrong, expected, exact=True)


@pytest.mark.parametrize("path,bk", [("wmma_scale", 32), ("wmma_scale16", 16)])
@pytest.mark.parametrize("k", [128, 256])
def test_fixtures_exercise_each_scale_group_and_high_bytes(path, bk, k):
    spec = BlockScaledGemmSpec(
        name="reference",
        M=32,
        N=48,
        K=k,
        matrix_path=path,
        block_k=bk,
        scale_dtype="e8m0",
    )
    a, b, sa, sb = make_case_inputs(spec, "mixed")
    assert set(np.unique(sa)) == set(range(125, 131))
    assert set(np.unique(sb)) == set(range(125, 131))
    expected = reference_result(a, b, sa, sb, bk, native=True)
    for wrong_sa, wrong_sb in ((sa[:, ::-1], sb), (sa, sb[::-1, :])):
        wrong = reference_result(a, b, wrong_sa, wrong_sb, bk, native=True)
        with pytest.raises(AssertionError, match="bad="):
            check_result(wrong, expected, exact=True)
    # At K=256, reusing the first call's scales must also be detectable.
    if k == 256:
        first_call = 128 // bk
        reused_sa = np.tile(sa[:, :first_call], (1, 2))
        reused_sb = np.tile(sb[:first_call, :], (2, 1))
        wrong = reference_result(a, b, reused_sa, reused_sb, bk, native=True)
        with pytest.raises(AssertionError, match="bad="):
            check_result(wrong, expected, exact=True)
    for group in range(k // bk):
        ga, gb, gsa, gsb = make_case_inputs(spec, f"group-{group}")
        sl = slice(group * bk, (group + 1) * bk)
        np.testing.assert_array_equal(ga[:, sl], a[:, sl])
        np.testing.assert_array_equal(gb[:, sl], b[:, sl])
        outside = (np.arange(k) // bk) != group
        assert not np.any(ga[:, outside].astype(np.float32))
        assert not np.any(gb[:, outside].astype(np.float32))
        isolated = reference_result(ga, gb, gsa, gsb, bk, native=True)
        assert np.isfinite(isolated).all() and np.any(isolated)
        # Wrong scales for this one active group cannot hide in accumulation.
        wrong = reference_result(ga, gb, gsa[:, ::-1], gsb, bk, native=True)
        with pytest.raises(AssertionError, match="bad="):
            check_result(wrong, isolated, exact=True)


@pytest.mark.parametrize("case", ["neutral", "a-only", "b-only"])
def test_neutral_and_one_operand_scale_fixtures(case):
    spec = BlockScaledGemmSpec(
        name="reference",
        M=16,
        N=16,
        K=128,
        matrix_path="wmma_scale",
        block_k=32,
        scale_dtype="e8m0",
    )
    _, _, sa, sb = make_case_inputs(spec, case)
    assert bool(np.all(sa == 127)) == (case in ("neutral", "b-only"))
    assert bool(np.all(sb == 127)) == (case in ("neutral", "a-only"))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("exact", [False, True])
def test_comparison_rejects_nonfinite_output_and_reference(value, exact):
    finite = np.array([[1.0, 0.0]], dtype=np.float32)
    nonfinite = np.array([[1.0, value]], dtype=np.float32)
    for got, ref in ((nonfinite, finite), (finite, nonfinite), (nonfinite, nonfinite)):
        with pytest.raises(AssertionError, match="non-finite"):
            check_result(got, ref, exact=exact)


def test_exact_comparison_detects_one_bf16_step_and_shape_mismatch():
    expected = np.array([[1.0]], dtype=np.float32)
    check_result(expected.copy(), expected, exact=True)
    with pytest.raises(AssertionError, match="bad=1/1"):
        check_result(expected + 2**-7, expected, exact=True)
    with pytest.raises(AssertionError, match="shape"):
        check_result(expected.ravel(), expected, exact=True)
