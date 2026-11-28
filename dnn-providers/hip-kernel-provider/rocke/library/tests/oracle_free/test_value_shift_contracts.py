# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for the value-shift relation: shift, fixture, derived
tolerance, analyzer, subject scope, and the checks of the compiled artifact.

The analyzer is exercised on outputs of ``_model_dense_run``, an FP32 model of
the dense kernel's single-pass online softmax with switches for normalization
defects. The model exists only in these contracts. No GPU verdict uses it: the
relation compares two subject outputs.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import re
import statistics
import struct
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from rocke.core.llvm_profile import LLVM_FLAVORS
from rocke.numeric.bf16 import bf16_to_f32, f32_to_bf16

from ._analytical import _FORMAT, UniformCase, decode_bits, encode_bits
from ._dense_subject import (
    DENSE_SUBJECTS,
    VALUE_SHIFT_SUBJECT_CASES,
    _validate_supported_case,
    _verify_resolved_spec,
    compile_dense_subject,
    resolve_dense_subject,
    softmax_scale,
)
from ._permutation_path import (
    _F32_INITIAL_MAX,
    _F32_ONE,
    classify_unmasked_path,
    expected_chain_counts,
    expected_path_counts,
    fp_mode_violations,
    kernel_descriptor_words,
    trace_mfma_operands,
)
from ._value_shift import (
    _ATOM_UNDERFLOW,
    _DOMINANT_KEY,
    _FP32_UNDERFLOW,
    _FP32_UNIT,
    _HOST_UNIT,
    _LATTICE_LIMIT,
    _min_exponent,
    OUTPUT_SCALE_SWEEP,
    VALUE_SHIFT_QUICK_CASES,
    ShiftMutation,
    ValueShiftCase,
    ValueShiftContract,
    ValueShiftResult,
    _signs,
    analyze_value_shift,
    applied_shift,
    format_head_diagnostics,
    format_value_shift_result,
    output_scale_threshold,
    require_finite_scores,
    rounding_radius,
    scale_output_bits,
    shift_mutation_failures,
    shift_vector,
    shifted_value_bits,
    store_allowance,
    value_shift_bound,
    value_shift_contract,
    value_shift_inputs,
    value_shift_tolerance,
)
from ._value_shift_path import (
    REDERIVE,
    expected_cone_counts,
    trace_value_cone,
    unmasked_path_violations,
    verified_key_tiles,
)
from .test_contracts import _dotted_name, _fake_tuning_spec
from .test_value_shift_dense import _not_rejected

_LOG2E = 1.4426950408889634
# KV tile of the shipped gfx942 dense default; the model also runs at half of it.
_MODEL_BLOCK_N = 64
_NAN_BITS = {"fp16": 0x7E00, "bf16": 0x7FC0}
_INFINITY_BITS = {"fp16": 0x7C00, "bf16": 0x7F80}

_FP16_MODERATE, _FP16_CONCENTRATED, _BF16_MODERATE, _BF16_CONCENTRATED = (
    VALUE_SHIFT_QUICK_CASES
)
# One case per compiled kernel: the two profiles of a dtype resolve to one spec.
_PATH_SUBJECT_CASES = tuple(
    (arch, case)
    for arch, case in VALUE_SHIFT_SUBJECT_CASES
    if case.profile == "moderate"
)


def _case_id(case: ValueShiftCase) -> str:
    return case.case_id


def _subject_id(item: tuple[str, ValueShiftCase]) -> str:
    return f"{item[0]}-{item[1].case_id}"


# ---------------------------------------------------------------------------
# Shift and exact input transformation
# ---------------------------------------------------------------------------


def test_shift_vector_repeats_the_signed_period() -> None:
    assert shift_vector(6).tolist() == [0.5, -0.25, 0.125, -0.5, 0.5, -0.25]
    shift = shift_vector(64)
    assert shift.dtype == np.float64 and np.all(shift != 0.0)
    assert np.array_equal(shift, np.tile([0.5, -0.25, 0.125, -0.5], 16))
    for dtype in ("fp16", "bf16"):
        bits = encode_bits(shift.astype(np.float32), dtype)
        assert np.array_equal(decode_bits(bits, dtype), shift)
    with pytest.raises(ValueError, match="head_dim must be positive"):
        shift_vector(0)


def test_shift_mutations_change_the_second_half_only() -> None:
    expected = shift_vector(64)
    omitted = applied_shift(64, "omit_second_half")
    rescaled = applied_shift(64, ShiftMutation.RESCALE_SECOND_HALF)
    assert np.array_equal(applied_shift(64), expected)
    for mutated in (omitted, rescaled):
        assert np.array_equal(mutated[:32], expected[:32])
    assert not np.any(omitted[32:])
    assert np.array_equal(rescaled[32:], 0.5 * expected[32:])
    with pytest.raises(ValueError, match="unknown shift mutation"):
        applied_shift(64, "double")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_shifted_sign_values_are_exact_in_both_dtypes(dtype: str) -> None:
    """``+1`` and ``-1`` shifted by the period take eight values, all dyadic and
    representable, for the expected shift and for both mutations of it."""
    values = np.array([1.0, -1.0], dtype=np.float32)
    bits = encode_bits(np.repeat(values, 4).reshape(1, 2, 1, 4), dtype)
    shifted = decode_bits(shifted_value_bits(bits, shift_vector(4), dtype), dtype)
    assert sorted(shifted.ravel().tolist()) == sorted(
        [1.5, 0.75, 1.125, 0.5, -0.5, -1.25, -0.875, -1.5]
    )
    for mutation in ShiftMutation:
        applied = applied_shift(4, mutation)
        moved = decode_bits(shifted_value_bits(bits, applied, dtype), dtype)
        assert np.array_equal(moved, decode_bits(bits, dtype) + applied)


@pytest.mark.parametrize(
    "dtype,value",
    [
        # 3/2 + 2**-10 is an FP16 value; adding 1/2 moves it to the binade of
        # 2, where the last place is 2**-9.
        ("fp16", 1.5 + 2.0**-10),
        # The smallest FP16 subnormal is lost next to 1/2.
        ("fp16", 2.0**-24),
        # The same two cases in BF16, whose last place in [1, 2) is 2**-7.
        ("bf16", 1.5 + 2.0**-7),
        ("bf16", 2.0**-40),
    ],
)
def test_shifted_values_refuse_any_rounding(dtype: str, value: float) -> None:
    """The tolerance has no input-rounding term, so a profile whose shifted
    values are not representable has no follow-up tensor."""
    bits = encode_bits(np.full((1, 1, 1, 4), value, dtype=np.float32), dtype)
    with pytest.raises(ValueError, match="not exactly representable"):
        shifted_value_bits(bits, shift_vector(4), dtype)


def test_shifted_values_validate_their_inputs() -> None:
    bits = encode_bits(np.ones((1, 2, 1, 4), dtype=np.float32), "fp16")
    with pytest.raises(ValueError, match="raw uint16 bits"):
        shifted_value_bits(bits.astype(np.float16), shift_vector(4), "fp16")
    with pytest.raises(ValueError, match=r"\[B, Sk, Hkv, D\]"):
        shifted_value_bits(bits[0], shift_vector(4), "fp16")
    with pytest.raises(ValueError, match="the shift must be 4 finite values"):
        shifted_value_bits(bits, shift_vector(8), "fp16")
    with pytest.raises(ValueError, match="the shift must be 4 finite values"):
        shifted_value_bits(bits, np.array([0.5, np.nan, 0.0, 0.0]), "fp16")
    infinite = bits.copy()
    infinite[0, 0, 0, 0] = _INFINITY_BITS["fp16"]
    with pytest.raises(ValueError, match="cannot shift V: it is not finite"):
        shifted_value_bits(infinite, shift_vector(4), "fp16")
    with pytest.raises(ValueError, match="dtype must be one of"):
        shifted_value_bits(bits, shift_vector(4), "fp32")


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_inputs_are_exact_in_both_dtypes_and_on_the_lattice(
    case: ValueShiftCase,
) -> None:
    q, k, v = value_shift_inputs(case)
    assert q.shape == case.output_shape
    assert k.shape == v.shape == case.key_shape
    for values in (q, k, v):
        assert values.dtype == np.float32
        for dtype in ("fp16", "bf16"):
            bits = encode_bits(values, dtype)
            assert np.array_equal(decode_bits(bits, dtype), values.astype(np.float64))
    require_finite_scores(q, k)
    assert set(np.unique(v).tolist()) == {-1.0, 1.0}
    assert case.head_dim * _LATTICE_LIMIT**2 <= 1 << 24


def _ideal_weights(
    case: ValueShiftCase, q: np.ndarray, k: np.ndarray, *, head: int
) -> np.ndarray:
    """Normalized ideal weights ``[Sq, Sk]`` of one query head of batch 0:
    ``exp2(s - max)`` on the FP32 scores, evaluated in FP64."""
    kv_head = head // (case.nhead_q // case.nhead_k)
    products = (
        q[0, :, head, :].astype(np.float64) @ k[0, :, kv_head, :].astype(np.float64).T
    )
    scale = np.float32(np.float32(softmax_scale(case)) * np.float32(_LOG2E))
    scores = (products.astype(np.float32) * scale).astype(np.float64)
    weights = np.exp2(scores - scores.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True)


def test_profiles_spread_and_concentrate_the_attention() -> None:
    """The moderate profile spreads each row over many keys and keeps every
    weight above the FP16 normal threshold. The concentrated profile gives one
    key nearly all of every row, so most other weights reach an FP16 cast as
    subnormals: the case the absolute cast term exists for."""
    q, k, _ = value_shift_inputs(_FP16_MODERATE)
    for head in range(_FP16_MODERATE.nhead_q):
        weights = _ideal_weights(_FP16_MODERATE, q, k, head=head)
        relative = weights / weights.max(axis=1, keepdims=True)
        assert np.all(relative >= 2.0**-14)
        assert float(weights.max()) < 0.5

    q, k, _ = value_shift_inputs(_FP16_CONCENTRATED)
    assert np.all(q >= 0.0) and np.all(k[:, _DOMINANT_KEY] == 2.0)
    for head in range(_FP16_CONCENTRATED.nhead_q):
        weights = _ideal_weights(_FP16_CONCENTRATED, q, k, head=head)
        assert np.all(np.argmax(weights, axis=1) == _DOMINANT_KEY)
        assert float(weights[:, _DOMINANT_KEY].min()) > 0.9
        relative = weights / weights.max(axis=1, keepdims=True)
        assert float(np.mean(relative < 2.0**-14)) > 0.9


# First 16 hex digits of the SHA-256 of the FP16 little-endian Q, K and V bits
# of each profile at the quick shape. The inputs are generated from an explicit
# integer hash; a change here changes what every launch of the lane reads.
_INPUT_DIGESTS = {
    "moderate": ["1926e2b77d391207", "3d214b7c9bf7d613", "e7b3a7b8457ed75b"],
    "concentrated": ["be90c6323077feed", "a763e5bd34662aea", "e7b3a7b8457ed75b"],
}


def test_inputs_do_not_depend_on_the_dtype_and_stay_pinned() -> None:
    for fp16_case, bf16_case in (
        (_FP16_MODERATE, _BF16_MODERATE),
        (_FP16_CONCENTRATED, _BF16_CONCENTRATED),
    ):
        digests = []
        for left, right in zip(
            value_shift_inputs(fp16_case), value_shift_inputs(bf16_case)
        ):
            assert np.array_equal(left, right)
            raw = encode_bits(left, "fp16").astype("<u2").tobytes()
            digests.append(hashlib.sha256(raw).hexdigest()[:16])
        assert digests == _INPUT_DIGESTS[fp16_case.profile]


def test_case_scope_refuses_what_the_tolerance_does_not_cover() -> None:
    for case in VALUE_SHIFT_QUICK_CASES:
        case.require_scope()
        assert case.mask == "none" and case.window == 0
    assert len({case.case_id for case in VALUE_SHIFT_QUICK_CASES}) == 4
    assert {(case.dtype, case.profile) for case in VALUE_SHIFT_QUICK_CASES} == {
        (dtype, profile)
        for dtype in ("fp16", "bf16")
        for profile in ("moderate", "concentrated")
    }
    with pytest.raises(ValueError, match="multiple of 8"):
        ValueShiftCase(1, 4, 20, 2, 2, 8, "fp16").require_scope()
    with pytest.raises(ValueError, match="exact in FP32"):
        ValueShiftCase(1, 4, 32, 2, 2, 1 << 15, "fp16").require_scope()
    with pytest.raises(ValueError, match="concentrated profile needs more than"):
        ValueShiftCase(1, 4, 64, 2, 2, 8, "fp16", "concentrated").require_scope()
    with pytest.raises(ValueError, match="multiple of nhead_k"):
        ValueShiftCase(1, 4, 8, 3, 2, 8, "fp16")
    with pytest.raises(ValueError, match="dtype must be one of"):
        ValueShiftCase(1, 4, 8, 2, 2, 8, "fp32")
    with pytest.raises(ValueError, match="profile must be one of"):
        ValueShiftCase(1, 4, 8, 2, 2, 8, "fp16", "uniform")
    # The shape checks are the shared base's, so both lanes' cases refuse a
    # dimension that is not positive.
    shape = {
        "batch": 1,
        "seqlen_q": 32,
        "seqlen_k": 32,
        "nhead_q": 2,
        "nhead_k": 2,
        "head_dim": 8,
    }
    for name in shape:
        for value in (0, -1):
            changed = {**shape, name: value, "dtype": "fp16"}
            with pytest.raises(ValueError, match=f"{name} must be positive"):
                ValueShiftCase(**changed)
            with pytest.raises(ValueError, match=f"{name} must be positive"):
                UniformCase(**changed, mask="causal")


@pytest.mark.parametrize(
    "value,match",
    [
        (0.3, "is not on the lattice"),
        (1.0 / 16.0, "is not on the lattice"),
        (3.125, "is not on the lattice"),
        (math.inf, "is not finite"),
        (math.nan, "is not finite"),
    ],
)
def test_finite_scores_are_certified_on_the_lattice_only(
    value: float, match: str
) -> None:
    q, k, _ = value_shift_inputs(_FP16_MODERATE)
    for name in ("Q", "K"):
        changed = {"Q": q.copy(), "K": k.copy()}
        changed[name][0, 3, 1, 2] = value
        with pytest.raises(ValueError, match=f"{name} {match}"):
            require_finite_scores(changed["Q"], changed["K"])


# ---------------------------------------------------------------------------
# Path constants of the tolerance
# ---------------------------------------------------------------------------


def _derived_constants(dtype: str, keys: int, key_tiles: int) -> dict[str, Fraction]:
    """The expressions of the derivation with their exact powers, where the
    implementation uses the enclosures ``(1 + u)**n <= 1 / (1 - n u)`` and
    ``(1 - u)**n >= 1 - n u``."""
    unit = _FP32_UNIT
    fraction_bits = _FORMAT[dtype][0]
    steps = (keys + key_tiles) * unit
    gamma = steps / (1 - steps)
    rescaled = (1 + unit) ** key_tiles
    smallest = (1 - unit) ** key_tiles
    cast = Fraction(1, 2 ** (fraction_bits + 1))
    subnormal_half = Fraction(2) ** (_min_exponent(dtype) - fraction_bits - 1)
    sigma = cast + (keys - 1) * subnormal_half * rescaled / smallest
    theta = gamma + key_tiles * rescaled * (1 + gamma) * _FP32_UNDERFLOW / smallest
    row_sum = (1 + sigma) * (1 + unit) / (1 - theta) - 1
    reciprocal = (1 + unit) / (smallest * (1 - theta))
    underflow = (
        rescaled
        * (1 + gamma)
        * (keys // 8 * _ATOM_UNDERFLOW + key_tiles * _FP32_UNDERFLOW)
    )
    return {
        "row_sum": row_sum,
        "cast_normalizer_row_sum": (1 + unit) / (1 - theta) - 1,
        "accumulation": gamma * (1 + row_sum),
        "accumulation_floor": reciprocal * underflow,
    }


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("keys,key_tiles", [(256, 4), (256, 8), (64, 1), (4096, 64)])
def test_bound_constants_are_not_below_the_derived_expressions(
    dtype: str, keys: int, key_tiles: int
) -> None:
    """Every constant is an exact rational that encloses the derivation's
    expression from above, and tightly."""
    bound = value_shift_bound(dtype, keys=keys, key_tiles=key_tiles)
    derived = _derived_constants(dtype, keys, key_tiles)
    for name, exact in derived.items():
        value = getattr(bound, name)
        assert isinstance(value, Fraction)
        assert exact <= value < exact * (1 + Fraction(1, 10**4)), name
    assert bound.accumulation_bound(Fraction(3, 2)) == (
        bound.accumulation * Fraction(3, 2) + bound.accumulation_floor
    )


@pytest.mark.parametrize(
    "dtype,row_sum,accumulation",
    [
        ("fp16", 5.270118287560868e-4, 3.101171089652892e-5),
        ("bf16", 3.937487094810464e-3, 3.11174198596256e-5),
    ],
)
def test_bound_pinned_values_at_the_quick_shape(
    dtype: str, row_sum: float, accumulation: float
) -> None:
    """Four 64-key tiles. A change here is a change of the derivation, never an
    adjustment to a measurement."""
    bound = value_shift_bound(dtype, keys=256, key_tiles=4)
    assert float(bound.row_sum) == pytest.approx(row_sum, rel=1e-12)
    assert float(bound.accumulation) == pytest.approx(accumulation, rel=1e-12)
    assert float(bound.accumulation_floor) < 1e-37
    # The same for both dtypes: the FP32 normalizer and reciprocal alone.
    assert float(bound.cast_normalizer_row_sum) == pytest.approx(
        3.1115549694469674e-5, rel=1e-12
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_bound_is_the_probability_cast_plus_the_fp32_normalizer(dtype: str) -> None:
    bound = value_shift_bound(dtype, keys=256, key_tiles=4)
    half_ulp = Fraction(1, 2 ** (_FORMAT[dtype][0] + 1))
    assert bound.cast == half_ulp
    # FP16 subnormal weights cost up to 255 half-subnormals; BF16 has none in
    # reach of an FP32 weight.
    if dtype == "fp16":
        assert Fraction(255, 2**25) <= bound.subnormal_cast < Fraction(256, 2**25)
    else:
        assert bound.subnormal_cast < Fraction(1, 2**120)
    gamma = Fraction(260, 2**23 - 260)
    assert gamma <= bound.normalizer < gamma * (1 + Fraction(1, 10**20))
    first_order = bound.cast + bound.subnormal_cast + bound.normalizer + _FP32_UNIT
    assert first_order < bound.row_sum < first_order * (1 + Fraction(1, 100))


def test_bound_grows_with_keys_tiles_and_the_coarser_dtype() -> None:
    reference = value_shift_bound("fp16", keys=256, key_tiles=4)
    for override in ({"keys": 512}, {"key_tiles": 8}):
        larger = value_shift_bound("fp16", **{"keys": 256, "key_tiles": 4, **override})
        assert larger.row_sum > reference.row_sum
        assert larger.accumulation > reference.accumulation
    assert value_shift_bound("bf16", keys=256, key_tiles=4).row_sum > reference.row_sum


@pytest.mark.parametrize(
    "override,match",
    [
        ({"keys": 0}, "multiple of 8"),
        ({"keys": 250}, "multiple of 8"),
        ({"key_tiles": 0}, "key_tiles"),
        ({"key_tiles": 257}, "key_tiles"),
        # (keys + key_tiles) u must stay below one, and the normalizer error
        # below one, so that the computed normalizer is positive.
        ({"keys": 1 << 23, "key_tiles": 1}, "too many keys"),
        ({"keys": 1 << 22, "key_tiles": 1}, "normalizer error bound"),
    ],
)
def test_bound_refuses_arguments_outside_its_derivation(override, match: str) -> None:
    args = {"keys": 256, "key_tiles": 4}
    with pytest.raises(ValueError, match=match):
        value_shift_bound("fp16", **{**args, **override})
    with pytest.raises(ValueError, match="dtype must be one of"):
        value_shift_bound("fp32", **args)


# ---------------------------------------------------------------------------
# Store allowance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_rounding_radius_is_half_the_gap_away_from_zero(dtype: str) -> None:
    """Over every finite magnitude of the dtype: the radius is half the distance
    to the next larger magnitude, whatever the sign."""
    magnitudes = np.arange(_INFINITY_BITS[dtype], dtype=np.uint16)
    values = decode_bits(magnitudes, dtype)
    assert np.all(np.diff(values) > 0.0)
    radius = rounding_radius(magnitudes, dtype)
    assert np.array_equal(radius[:-1], 0.5 * np.diff(values))
    # The largest finite value has no finite neighbour above; its binade's gap.
    assert radius[-1] == radius[-2]
    assert np.array_equal(
        rounding_radius(magnitudes | np.uint16(0x8000), dtype), radius
    )
    fraction_bits = _FORMAT[dtype][0]
    assert radius[0] == 2.0 ** (_min_exponent(dtype) - fraction_bits - 1)
    assert _min_exponent(dtype) == {"fp16": -14, "bf16": -126}[dtype]
    one = encode_bits(np.array([1.0], dtype=np.float32), dtype)
    assert rounding_radius(one, dtype)[0] == 2.0 ** -(fraction_bits + 1)


def _round_to(values: np.ndarray, dtype: str) -> np.ndarray:
    """Round FP32 values to the subject dtype (nearest-even), back in FP32."""
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    return bf16_to_f32(f32_to_bf16(values)).reshape(values.shape).astype(np.float32)


def _store(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "fp16":
        with np.errstate(over="ignore"):
            return values.astype(np.float16).view(np.uint16)
    return f32_to_bf16(values).reshape(values.shape)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_store_allowance_covers_both_store_forms(dtype: str) -> None:
    """For FP32 operands ``o`` and ``rcp``: the stored value is within its own
    allowance of the exact product, whether the product is rounded to FP32
    first or not. The exact product of two FP32 numbers is an FP64 number."""
    generator = np.random.default_rng(20261115)
    count = 1 << 18
    exponents = generator.integers(-30, 3, size=count)
    accumulator = np.ldexp(generator.uniform(-1.0, 1.0, size=count), exponents).astype(
        np.float32
    )
    reciprocal = generator.uniform(2.0**-9, 1.0, size=count).astype(np.float32)
    exact = accumulator.astype(np.float64) * reciprocal.astype(np.float64)

    twice = _store(accumulator * reciprocal, dtype)
    forms = [twice]
    if dtype == "fp16":
        # numpy converts FP64 to FP16 with one rounding.
        forms.append(exact.astype(np.float16).view(np.uint16))
    for stored in forms:
        error = np.abs(decode_bits(stored, dtype) - exact)
        assert np.all(error <= store_allowance(stored, dtype))
    if dtype == "fp16":
        assert np.any(forms[0] != forms[1])


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_store_allowance_grows_with_the_magnitude_and_ignores_invalid_values(
    dtype: str,
) -> None:
    magnitudes = np.arange(_INFINITY_BITS[dtype], dtype=np.uint16)
    allowance = store_allowance(magnitudes, dtype)
    assert np.all(np.diff(allowance) >= 0.0)
    # One half unit in the last place, plus the FP32 product's rounding.
    radius = rounding_radius(magnitudes, dtype)
    assert np.all(allowance > radius)
    assert np.all(allowance < radius * (1.0 + 2.0**-9) + 2.0**-125)
    invalid = np.array([_NAN_BITS[dtype], _INFINITY_BITS[dtype]], dtype=np.uint16)
    assert np.array_equal(
        store_allowance(invalid, dtype),
        store_allowance(np.zeros(2, dtype=np.uint16), dtype),
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def _quick_contract(
    case: ValueShiftCase,
    *,
    key_tiles: int = 4,
    mutation: ShiftMutation | None = None,
) -> tuple[ValueShiftContract, np.ndarray, np.ndarray]:
    """The contract of one quick case and the raw V bits of its two launches."""
    _, _, v = value_shift_inputs(case)
    base = encode_bits(v, case.dtype)
    applied = applied_shift(case.head_dim, mutation)
    follow_up = shifted_value_bits(base, applied, case.dtype)
    contract = value_shift_contract(
        case, base, follow_up, key_tiles=key_tiles, applied=applied
    )
    return contract, base, follow_up


@pytest.mark.parametrize(
    "case,apriori,guard,follow_up_guard,threshold",
    [
        (
            _FP16_MODERATE,
            1.3178961140496678e-3,
            1.001046424065464,
            1.5013254954731958,
            7.244407945807692e-3,
        ),
        (
            _BF16_MODERATE,
            9.859338165732003e-3,
            1.0078749741970543,
            1.5098593362955814,
            5.617061753899753e-2,
        ),
    ],
    ids=["fp16", "bf16"],
)
def test_contract_pinned_values_at_the_quick_shape(
    case: ValueShiftCase,
    apriori: float,
    guard: float,
    follow_up_guard: float,
    threshold: float,
) -> None:
    """Column 0, where the shift is 1/2 and the shifted values reach 3/2."""
    contract, _, _ = _quick_contract(case)
    assert contract.bound.keys == 256 and contract.bound.key_tiles == 4
    assert contract.apriori_tolerance[0] == pytest.approx(apriori, rel=1e-9)
    assert contract.base_guard[0] == pytest.approx(guard, rel=1e-12)
    assert contract.follow_up_guard[0] == pytest.approx(follow_up_guard, rel=1e-12)
    assert output_scale_threshold(contract) == pytest.approx(threshold, rel=1e-9)


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_contract_columns_follow_the_shift_period(case: ValueShiftCase) -> None:
    """Sign-valued V: every column's largest ``|V|`` is one, the shifted one is
    ``1 + |c_d|``, and every per-column quantity repeats with the shift."""
    contract, _, _ = _quick_contract(case)
    shift = shift_vector(case.head_dim)
    assert np.array_equal(contract.shift, shift)
    assert np.array_equal(contract.base_value_max, np.ones(case.head_dim))
    assert np.array_equal(contract.follow_up_value_max, 1.0 + np.abs(shift))
    bound = contract.bound
    for dim in range(case.head_dim):
        expected = abs(Fraction(float(shift[dim])))
        exact = (
            expected * bound.row_sum
            + bound.accumulation_bound(Fraction(1))
            + bound.accumulation_bound(1 + expected)
        )
        column = Fraction(float(contract.column_tolerance[dim]))
        assert exact <= column <= exact * (1 + Fraction(1, 2**50))
        assert contract.column_tolerance[dim] == contract.column_tolerance[dim % 4]
        assert contract.apriori_tolerance[dim] == contract.apriori_tolerance[dim % 4]
        assert contract.column_tolerance[dim] < contract.apriori_tolerance[dim]
        assert contract.apriori_tolerance[dim] < abs(shift[dim])
        assert contract.base_value_max[dim] < contract.base_guard[dim]
        assert contract.follow_up_value_max[dim] < contract.follow_up_guard[dim]
    # The base guard is one representable step above one in both dtypes.
    step = 2.0 ** -_FORMAT[case.dtype][0]
    assert 1.0 + 0.5 * step < contract.base_guard[0] < 1.0 + 1.5 * step
    for array in (contract.shift, contract.column_tolerance, contract.base_guard):
        assert not array.flags.writeable


def test_contract_does_not_depend_on_the_attention_profile() -> None:
    """The tolerance reads the launched V and the path; Q and K enter only
    through the stored outputs."""
    for moderate, concentrated in (
        (_FP16_MODERATE, _FP16_CONCENTRATED),
        (_BF16_MODERATE, _BF16_CONCENTRATED),
    ):
        left, _, _ = _quick_contract(moderate)
        right, _, _ = _quick_contract(concentrated)
        assert np.array_equal(left.column_tolerance, right.column_tolerance)
        assert np.array_equal(left.apriori_tolerance, right.apriori_tolerance)


def test_contract_fails_closed() -> None:
    case = _FP16_MODERATE
    contract, base, follow_up = _quick_contract(case)
    assert isinstance(contract, ValueShiftContract)

    # A follow-up tensor that is not the exact sum has no tolerance: the
    # relation would need an input-rounding term it does not have.
    off_by_one_step = follow_up.copy()
    off_by_one_step[0, 3, 1, 2] += 1
    with pytest.raises(ValueError, match="not V \\+ c exactly at 1 element"):
        value_shift_contract(case, base, off_by_one_step, key_tiles=4)
    # A mutated shift is accepted only when it is declared.
    mutated = shifted_value_bits(base, applied_shift(64, "omit_second_half"), "fp16")
    with pytest.raises(ValueError, match="not V \\+ c exactly"):
        value_shift_contract(case, base, mutated, key_tiles=4)
    value_shift_contract(
        case, base, mutated, key_tiles=4, applied=applied_shift(64, "omit_second_half")
    )

    not_finite = base.copy()
    not_finite[0, 0, 0, 0] = _NAN_BITS["fp16"]
    with pytest.raises(ValueError, match="V is not finite"):
        value_shift_contract(case, not_finite, follow_up, key_tiles=4)
    with pytest.raises(ValueError, match="base_values shape"):
        value_shift_contract(case, base[:, :-8], follow_up, key_tiles=4)
    with pytest.raises(ValueError, match="raw uint16 bits"):
        value_shift_contract(case, base.astype(np.int32), follow_up, key_tiles=4)
    with pytest.raises(ValueError, match="key_tiles"):
        value_shift_contract(case, base, follow_up, key_tiles=0)


def test_contract_refuses_a_column_it_cannot_resolve() -> None:
    """Values of magnitude 2**9 put the BF16 half unit in the last place at 2,
    far above the shift: no residual could show a missing shift there."""
    case = ValueShiftCase(1, 8, 8, 2, 2, 4, "bf16")
    base = encode_bits(np.full(case.key_shape, 512.0, dtype=np.float32), "bf16")
    follow_up = shifted_value_bits(base, np.array([64.0, -64.0, 64.0, -64.0]), "bf16")
    with pytest.raises(ValueError, match="is not below the shift it must resolve"):
        value_shift_contract(
            case,
            base,
            follow_up,
            key_tiles=1,
            applied=np.array([64.0, -64.0, 64.0, -64.0]),
        )


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


def _constant_output(case: ValueShiftCase, value: float) -> np.ndarray:
    return encode_bits(np.full(case.output_shape, value, dtype=np.float32), case.dtype)


def _shifted_output(case: ValueShiftCase, base: np.ndarray) -> np.ndarray:
    """Output bits that are ``base + c`` exactly."""
    values = decode_bits(base, case.dtype) + shift_vector(case.head_dim)
    return encode_bits(values.astype(np.float32), case.dtype)


def _with_value(bits: np.ndarray, index, value: float, dtype: str) -> np.ndarray:
    changed = bits.copy()
    changed[index] = encode_bits(np.array([value], dtype=np.float32), dtype)[0]
    return changed


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_analyzer_separates_residuals_at_the_tolerance(case: ValueShiftCase) -> None:
    contract, _, _ = _quick_contract(case)
    base = _constant_output(case, 0.25)
    follow_up = _shifted_output(case, base)
    clean = analyze_value_shift(contract, base, follow_up)
    assert not clean.violations and not clean.rejected_by_residual
    assert clean.max_abs == 0.0 and clean.max_ratio == 0.0
    assert clean.compared_count == base.size
    assert clean.max_row_eta == 0.0 and clean.max_row_spread == 0.0

    # Column 0: base 1/4, follow-up 3/4. The largest power of two inside the
    # element's tolerance passes; four times it fails.
    index = (0, 17, 5, 0)
    tolerance = float(value_shift_tolerance(contract, base, follow_up)[index])
    inside = 2.0 ** math.floor(math.log2(tolerance))
    passing = analyze_value_shift(
        contract, base, _with_value(follow_up, index, 0.75 + inside, case.dtype)
    )
    assert not passing.violations
    assert passing.max_abs == inside and passing.worst_coordinate == index
    assert passing.residual_at_worst == inside
    assert 0.5 < passing.max_ratio <= 1.0
    assert passing.eta_at_worst == inside / 0.5
    assert passing.eta_allowance_at_worst == passing.tolerance_at_worst / 0.5

    failing = analyze_value_shift(
        contract, base, _with_value(follow_up, index, 0.75 - 4.0 * inside, case.dtype)
    )
    assert failing.exceed_count == 1 and failing.rejected_by_residual
    assert failing.worst_coordinate == index
    assert failing.residual_at_worst == -4.0 * inside
    assert failing.max_ratio > 1.0
    assert any(item.startswith("residual=") for item in failing.violations)
    assert failing.head_max_ratio[0][5] == failing.max_ratio
    assert failing.class_max_ratio[0] == failing.max_ratio
    assert failing.class_max_ratio[1:] == (0.0, 0.0, 0.0)
    assert "residual=" in format_value_shift_result(failing)
    assert len(format_head_diagnostics(failing).splitlines()) == case.batch


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_analyzer_tolerance_reads_the_stored_binade(case: ValueShiftCase) -> None:
    """Small stored outputs get a small store term; outputs at the guard get
    the a-priori tolerance, which no stored pair inside the guards exceeds."""
    contract, _, _ = _quick_contract(case)
    small = _constant_output(case, 2.0**-6)
    tolerance = value_shift_tolerance(contract, small, _shifted_output(case, small))
    assert np.all(tolerance < contract.apriori_tolerance)
    assert np.all(tolerance > contract.column_tolerance)

    for level in (0.25, 0.5, 1.0):
        base = _constant_output(case, level)
        larger = value_shift_tolerance(contract, base, _shifted_output(case, base))
        assert np.all(larger >= tolerance)
        assert np.all(larger <= contract.apriori_tolerance)
        tolerance = larger

    result = analyze_value_shift(contract, small, _shifted_output(case, small))
    assert 0.0 < result.tolerance_fraction < 1.0
    assert result.apriori_max_ratio == 0.0


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_analyzer_rejects_invalid_and_oversized_outputs(case: ValueShiftCase) -> None:
    contract, _, _ = _quick_contract(case)
    base = _constant_output(case, 0.25)
    follow_up = _shifted_output(case, base)
    unwritten = follow_up.copy()
    unwritten[0, 3, 1, 2] = _NAN_BITS[case.dtype]
    unwritten[0, 4, 1, 2] = _INFINITY_BITS[case.dtype]
    result = analyze_value_shift(contract, base, unwritten)
    assert result.invalid_count == 2 and "invalid=2" in result.violations
    assert result.exceed_count == 0 and result.max_abs == 0.0
    assert result.compared_count == base.size - 2

    # An error both launches share cancels in the residual; the magnitude
    # guards see it.
    large = _constant_output(case, 1.5)
    result = analyze_value_shift(contract, large, _shifted_output(case, large))
    assert result.exceed_count == 0
    # 1.5 is beyond the base guard everywhere; 1.5 + c is beyond the follow-up
    # guard wherever c is positive.
    assert result.magnitude_count == large.size + large.size // 2
    assert any(item.startswith("magnitude>guard") for item in result.violations)

    nothing = np.full(case.output_shape, _NAN_BITS[case.dtype], dtype=np.uint16)
    result = analyze_value_shift(contract, nothing, nothing)
    assert "no_coordinate_compared" in result.violations
    assert result.worst_coordinate is None and result.compared_count == 0


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_analyzer_tells_a_row_sum_error_from_a_column_error(
    case: ValueShiftCase,
) -> None:
    """A follow-up output of ``O + (1 + eta) c`` moves every column of a row by
    the same fraction of its shift: the row median reports ``eta`` and the
    spread stays zero. One moved column leaves the median and shows in the
    spread."""
    contract, _, _ = _quick_contract(case)
    base = _constant_output(case, 0.25)
    eta = 2.0**-4
    values = decode_bits(base, case.dtype) + shift_vector(case.head_dim)
    values[0, 9, 2, :] += eta * shift_vector(case.head_dim)
    scaled = analyze_value_shift(
        contract, base, encode_bits(values.astype(np.float32), case.dtype)
    )
    assert scaled.max_row_eta == eta and scaled.max_row_spread == 0.0
    assert scaled.eta_at_worst == eta

    values = decode_bits(base, case.dtype) + shift_vector(case.head_dim)
    values[0, 9, 2, 5] += 2.0**-5
    moved = analyze_value_shift(
        contract, base, encode_bits(values.astype(np.float32), case.dtype)
    )
    assert moved.max_row_eta == 0.0
    assert moved.max_row_spread == 2.0**-5 / 0.25
    assert moved.worst_coordinate == (0, 9, 2, 5)


def _exact_mutated_pair(
    case: ValueShiftCase, mutation: ShiftMutation
) -> tuple[ValueShiftContract, np.ndarray, np.ndarray, np.ndarray]:
    """The contract of one wrong-shift pair, its applied shift, and two outputs
    that carry that shift exactly."""
    contract, _, _ = _quick_contract(case, mutation=mutation)
    applied = applied_shift(case.head_dim, mutation)
    base = _constant_output(case, 0.25)
    values = decode_bits(base, case.dtype) + applied
    return contract, applied, base, encode_bits(values.astype(np.float32), case.dtype)


def test_analyzer_validates_its_inputs() -> None:
    case = _FP16_MODERATE
    contract, _, _ = _quick_contract(case)
    base = _constant_output(case, 0.25)
    with pytest.raises(ValueError, match="raw uint16 bits"):
        analyze_value_shift(contract, base.astype(np.float16), base)
    with pytest.raises(ValueError, match="follow-up output shape"):
        analyze_value_shift(contract, base, base[:, :-1])


@pytest.mark.parametrize("case", [_FP16_MODERATE, _BF16_MODERATE], ids=_case_id)
@pytest.mark.parametrize("mutation", list(ShiftMutation))
def test_wrong_shifts_are_rejected_on_exact_outputs(
    case: ValueShiftCase, mutation: ShiftMutation
) -> None:
    """Outputs that carry the applied shift exactly fail wherever it differs
    from the expected one: the second half of the columns, and only there."""
    contract, applied, base, follow_up = _exact_mutated_pair(case, mutation)
    result = analyze_value_shift(contract, base, follow_up)
    assert result.rejected_by_residual
    assert result.exceed_count == base.size // 2
    assert result.worst_coordinate[3] >= case.head_dim // 2
    expected = 1.0 if mutation is ShiftMutation.OMIT_SECOND_HALF else 0.5
    assert abs(result.eta_at_worst) == expected
    assert shift_mutation_failures(contract, applied, result) == ()


@pytest.mark.parametrize("case", [_FP16_MODERATE, _BF16_MODERATE], ids=_case_id)
@pytest.mark.parametrize("mutation", list(ShiftMutation))
def test_wrong_shift_verdict_requires_a_sound_launch(
    case: ValueShiftCase, mutation: ShiftMutation
) -> None:
    """A wrong-shift launch counts as rejected only when the launch itself is
    sound and its residuals lie where the wrong shift puts them. One residual
    beyond the tolerance is not enough."""
    contract, applied, base, follow_up = _exact_mutated_pair(case, mutation)
    rows = case.batch * case.seqlen_q * case.nhead_q
    half = case.head_dim // 2
    mutated_column = case.head_dim - 1

    def failures(output: np.ndarray) -> tuple[str, ...]:
        result = analyze_value_shift(contract, base, output)
        return shift_mutation_failures(contract, applied, result)

    clean = analyze_value_shift(contract, base, follow_up)
    assert clean.column_exceed_count == (0,) * half + (rows,) * half
    assert sum(clean.column_exceed_count) == clean.exceed_count

    # Quiet NaN everywhere but one wrong value: one residual is beyond the
    # tolerance and nothing else was stored.
    index = (0, 0, 0, mutated_column)
    mostly_unwritten = np.full(case.output_shape, _NAN_BITS[case.dtype], np.uint16)
    mostly_unwritten[index] = follow_up[index]
    result = analyze_value_shift(contract, base, mostly_unwritten)
    assert result.rejected_by_residual and result.exceed_count == 1
    found = shift_mutation_failures(contract, applied, result)
    assert found[0] == f"invalid={base.size - 1}"
    assert f"{half} columns whose wrong shift is certain" in found[1]
    assert found[1].endswith(f"first column {half} with 0 of {rows} beyond")
    assert len(found) == 2

    unwritten = np.full(case.output_shape, _NAN_BITS[case.dtype], np.uint16)
    found = failures(unwritten)
    assert found[0] == f"invalid={base.size}"
    assert found[-1] == "no residual beyond its tolerance"

    # A stored value beyond its guard, in a column whose residual exceeds the
    # tolerance either way.
    oversized = _with_value(follow_up, (0, 3, 1, mutated_column), 4.0, case.dtype)
    assert failures(oversized) == ("magnitude>guard at 1",)

    # The relation holds in the columns the mutation leaves alone. A residual
    # there is a fault of the launch, not a rejection.
    stray = _with_value(follow_up, (0, 5, 2, 0), 0.25, case.dtype)
    assert failures(stray) == (
        "1 residuals beyond the tolerance in columns that carry the expected "
        "shift, first in column 0",
    )

    # One element of a mutated column that carries the expected shift after all.
    kept = (0, 7, 4, mutated_column)
    carried = 0.25 + float(shift_vector(case.head_dim)[mutated_column])
    assert failures(_with_value(follow_up, kept, carried, case.dtype)) == (
        "1 columns whose wrong shift is certain to exceed the tolerance keep "
        f"residuals inside it, first column {mutated_column} with {rows - 1} of "
        f"{rows} beyond",
    )

    # The halves exchanged: residuals beyond the tolerance exist, in the wrong
    # columns. The shift has period four, so the expected-shift relation cannot
    # tell the halves apart; the wrong shifts can.
    values = decode_bits(base, case.dtype) + np.roll(applied, half)
    exchanged = encode_bits(values.astype(np.float32), case.dtype)
    result = analyze_value_shift(contract, base, exchanged)
    assert result.rejected_by_residual and result.exceed_count == base.size // 2
    found = shift_mutation_failures(contract, applied, result)
    assert len(found) == 2
    assert found[0].startswith(f"{base.size // 2} residuals beyond the tolerance")
    assert found[1].startswith(f"{half} columns whose wrong shift is certain")


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
@pytest.mark.parametrize("mutation", list(ShiftMutation))
def test_every_mutated_column_is_certain_to_be_rejected_at_the_quick_shape(
    case: ValueShiftCase, mutation: ShiftMutation
) -> None:
    """The verdict requires every residual of a column beyond its tolerance only
    where the missing shift is more than twice the a-priori tolerance. At the
    quick shape that is every mutated column, so the requirement is not empty."""
    contract, _, _ = _quick_contract(case, mutation=mutation)
    difference = np.abs(applied_shift(case.head_dim, mutation) - contract.shift)
    half = case.head_dim // 2
    assert np.all(difference[:half] == 0.0)
    assert np.all(difference[half:] > 2.0 * contract.apriori_tolerance[half:])


def test_wrong_shift_verdict_refuses_a_shift_the_tolerance_does_not_cover() -> None:
    case = _FP16_MODERATE
    contract, applied, base, follow_up = _exact_mutated_pair(
        case, ShiftMutation.OMIT_SECOND_HALF
    )
    result = analyze_value_shift(contract, base, follow_up)
    with pytest.raises(ValueError, match="exceeds the expected one"):
        shift_mutation_failures(contract, 2.0 * shift_vector(case.head_dim), result)
    with pytest.raises(ValueError, match="finite values"):
        shift_mutation_failures(contract, applied[:-1], result)


# ---------------------------------------------------------------------------
# Scalar specification of the tolerance and the analyzer
# ---------------------------------------------------------------------------


def _exact_value(bits: int, dtype: str) -> Fraction | None:
    """The exact value of one raw bit pattern, or ``None`` when it is not
    finite."""
    fraction_bits, exponent_bits, bias = _FORMAT[dtype]
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = bits & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        return None
    if exponent == 0:
        magnitude = Fraction(fraction) * Fraction(2) ** (1 - bias - fraction_bits)
    else:
        magnitude = Fraction((1 << fraction_bits) | fraction) * Fraction(2) ** (
            exponent - bias - fraction_bits
        )
    return -magnitude if bits >> 15 else magnitude


def _exact_store_allowance(bits: int, dtype: str) -> Fraction:
    """The store allowance of one finite stored value, from the definition:
    half the distance to the next larger magnitude, one FP32 unit of the
    stored magnitude plus that, and the FP32 underflow error."""
    magnitude_bits = bits & 0x7FFF
    magnitude = _exact_value(magnitude_bits, dtype)
    above = _exact_value(magnitude_bits + 1, dtype)
    if above is None:
        above = 2 * magnitude - _exact_value(magnitude_bits - 1, dtype)
    radius = (above - magnitude) / 2
    return (
        radius + _FP32_UNIT * (magnitude + radius) / (1 - _FP32_UNIT) + _FP32_UNDERFLOW
    )


_SPEC_CASES = (
    ValueShiftCase(1, 8, 16, 4, 2, 16, "fp16"),
    ValueShiftCase(1, 8, 16, 4, 2, 16, "bf16"),
)


def _spec_outputs(
    contract: ValueShiftContract,
) -> tuple[np.ndarray, np.ndarray]:
    """A pair of outputs with residuals on both sides of the tolerance, two
    invalid values and one value beyond its guard."""
    case = contract.case
    generator = np.random.default_rng(7)
    base_values = generator.uniform(-1.0, 1.0, size=case.output_shape)
    base = _store(base_values.astype(np.float32), case.dtype)
    noise = generator.uniform(-3.0, 3.0, size=case.output_shape)
    follow_values = (
        decode_bits(base, case.dtype)
        + contract.shift
        + noise * contract.apriori_tolerance
    )
    follow_up = _store(follow_values.astype(np.float32), case.dtype)
    follow_up[0, 1, 1, 3] = _NAN_BITS[case.dtype]
    base[0, 2, 0, 5] = _INFINITY_BITS[case.dtype]
    base[0, 3, 2, 7] = encode_bits(np.array([4.0], dtype=np.float32), case.dtype)[0]
    return base, follow_up


def _spec_contract(case: ValueShiftCase) -> ValueShiftContract:
    values = encode_bits(_signs(case.key_shape, 0x33), case.dtype)
    follow_up = shifted_value_bits(values, shift_vector(case.head_dim), case.dtype)
    return value_shift_contract(case, values, follow_up, key_tiles=2)


@pytest.mark.parametrize("case", _SPEC_CASES, ids=_case_id)
def test_vectorized_tolerance_is_not_below_the_exact_one(case: ValueShiftCase) -> None:
    """Every element's tolerance, against its exact rational value: the column
    part from the path constants, the store allowance of each stored value from
    its neighbours, and the FP64 rounding of the residual. The vectorized value
    is never smaller and exceeds it by rounding only."""
    contract = _spec_contract(case)
    base, follow_up = _spec_outputs(contract)
    tolerance = value_shift_tolerance(contract, base, follow_up)
    reduced = value_shift_tolerance(contract, base, follow_up, cast_normalizer=True)
    bound = contract.bound
    assert bound.cast_normalizer_row_sum < bound.row_sum
    for index in np.ndindex(case.output_shape):
        expected = abs(Fraction(float(contract.shift[index[3]])))
        rest = (
            bound.accumulation_bound(Fraction(1))
            + bound.accumulation_bound(1 + expected)
            + Fraction(_HOST_UNIT) * expected
        )
        for bits in (int(base[index]), int(follow_up[index])):
            value = _exact_value(bits, case.dtype)
            if value is None:
                bits, value = 0, Fraction(0)
            rest += _exact_store_allowance(bits, case.dtype)
            rest += Fraction(_HOST_UNIT) * abs(value)
        # The diagnostic tolerance differs in the row-sum bound only.
        for values, row_sum in (
            (tolerance, bound.row_sum),
            (reduced, bound.cast_normalizer_row_sum),
        ):
            exact = expected * row_sum + rest
            found = Fraction(float(values[index]))
            assert exact <= found <= exact * (1 + Fraction(1, 2**40)), index


def _scalar_analyze(
    contract: ValueShiftContract, base: np.ndarray, follow_up: np.ndarray
) -> ValueShiftResult:
    """One element at a time, in plain Python: the executable specification the
    vectorized analyzer must reproduce in every field."""
    case = contract.case
    tolerance = value_shift_tolerance(contract, base, follow_up)
    reduced = value_shift_tolerance(contract, base, follow_up, cast_normalizer=True)
    invalid = magnitude_count = exceed = compared = 0
    column_exceed = [0] * case.head_dim
    reduced_exceed = 0
    reduced_ratio = 0.0
    max_ratio = max_abs = 0.0
    worst = None
    squares = total = fraction_sum = 0.0
    apriori_ratio = 0.0
    head_ratio = [[0.0] * case.nhead_q for _ in range(case.batch)]
    class_ratio = [0.0] * 4
    class_abs = [0.0] * 4
    row_eta = row_spread = 0.0
    for batch in range(case.batch):
        for query in range(case.seqlen_q):
            for head in range(case.nhead_q):
                etas = []
                complete = True
                for dim in range(case.head_dim):
                    index = (batch, query, head, dim)
                    low = _exact_value(int(base[index]), case.dtype)
                    high = _exact_value(int(follow_up[index]), case.dtype)
                    invalid += (low is None) + (high is None)
                    if low is not None and abs(low) > contract.base_guard[dim]:
                        magnitude_count += 1
                    if high is not None and abs(high) > contract.follow_up_guard[dim]:
                        magnitude_count += 1
                    if low is None or high is None:
                        complete = False
                        etas.append(0.0)
                        continue
                    compared += 1
                    shift = float(contract.shift[dim])
                    residual = (float(high) - float(low)) - shift
                    limit = float(tolerance[index])
                    ratio = abs(residual) / limit
                    exceed += abs(residual) > limit
                    column_exceed[dim] += abs(residual) > limit
                    reduced_exceed += abs(residual) > float(reduced[index])
                    reduced_ratio = max(
                        reduced_ratio, abs(residual) / float(reduced[index])
                    )
                    if worst is None or ratio > max_ratio:
                        worst = (index, residual, limit)
                    max_ratio = max(max_ratio, ratio)
                    max_abs = max(max_abs, abs(residual))
                    squares += residual * residual
                    total += residual
                    fraction_sum += limit / float(contract.apriori_tolerance[dim])
                    apriori_ratio = max(
                        apriori_ratio,
                        abs(residual) / float(contract.apriori_tolerance[dim]),
                    )
                    head_ratio[batch][head] = max(head_ratio[batch][head], ratio)
                    class_ratio[dim % 4] = max(class_ratio[dim % 4], ratio)
                    class_abs[dim % 4] = max(class_abs[dim % 4], abs(residual))
                    etas.append(residual / shift)
                if complete:
                    median = statistics.median(etas)
                    row_eta = max(row_eta, abs(median))
                    row_spread = max(row_spread, max(abs(eta - median) for eta in etas))
    violations = []
    if invalid:
        violations.append(f"invalid={invalid}")
    if magnitude_count:
        violations.append(f"magnitude>guard at {magnitude_count}")
    if exceed:
        violations.append(f"residual={abs(worst[1]):.9g}>{worst[2]:.9g} at {exceed}")
    if compared == 0:
        violations.append("no_coordinate_compared")
    shift_at_worst = float(contract.shift[worst[0][3]]) if worst else 1.0
    return ValueShiftResult(
        violations=tuple(violations),
        invalid_count=invalid,
        magnitude_count=magnitude_count,
        exceed_count=exceed,
        column_exceed_count=tuple(column_exceed),
        compared_count=compared,
        max_ratio=max_ratio,
        worst_coordinate=worst[0] if worst else None,
        residual_at_worst=worst[1] if worst else 0.0,
        tolerance_at_worst=worst[2] if worst else 0.0,
        max_abs=max_abs,
        rms=math.sqrt(squares / compared) if compared else 0.0,
        mean_signed=total / compared if compared else 0.0,
        head_max_ratio=tuple(tuple(row) for row in head_ratio),
        class_max_ratio=tuple(class_ratio),
        class_max_abs=tuple(class_abs),
        eta_at_worst=worst[1] / shift_at_worst if worst else 0.0,
        eta_allowance_at_worst=worst[2] / abs(shift_at_worst) if worst else 0.0,
        max_row_eta=row_eta,
        max_row_spread=row_spread,
        apriori_max_ratio=apriori_ratio,
        tolerance_fraction=fraction_sum / compared if compared else 0.0,
        cast_normalizer_max_ratio=reduced_ratio,
        cast_normalizer_exceed_count=reduced_exceed,
    )


# Fields that are sums over the elements: equal up to the order of FP64
# additions. Every other field must match exactly.
_SUMMED_FIELDS = ("rms", "mean_signed", "tolerance_fraction")


@pytest.mark.parametrize("case", _SPEC_CASES, ids=_case_id)
def test_vectorized_analyzer_matches_the_scalar_specification(
    case: ValueShiftCase,
) -> None:
    contract = _spec_contract(case)
    base, follow_up = _spec_outputs(contract)
    found = analyze_value_shift(contract, base, follow_up)
    expected = _scalar_analyze(contract, base, follow_up)
    # The pair exercises every verdict and both sides of the tolerance.
    assert found.invalid_count == 2 and found.magnitude_count >= 1
    assert 0 < found.exceed_count < found.compared_count
    for name in ValueShiftResult.__dataclass_fields__:
        left, right = getattr(found, name), getattr(expected, name)
        if name in _SUMMED_FIELDS:
            assert left == pytest.approx(right, rel=1e-12, abs=1e-300), name
        else:
            assert left == right, name


# ---------------------------------------------------------------------------
# Output-scale mutation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_output_scale_is_one_rounding_of_the_exact_product(dtype: str) -> None:
    magnitudes = np.arange(1, _INFINITY_BITS[dtype], 37, dtype=np.uint16)
    values = decode_bits(magnitudes, dtype)
    keep = (np.abs(values) > 2.0**-100) & (np.abs(values) < 2.0**10)
    bits = np.concatenate([magnitudes[keep], magnitudes[keep] | np.uint16(0x8000)])
    for eta in (0.25, 2.0**-7, 2.0**-12):
        scaled = scale_output_bits(bits, eta, dtype)
        exact = decode_bits(bits, dtype) * (1.0 + eta)
        error = np.abs(decode_bits(scaled, dtype) - exact)
        assert np.all(error <= rounding_radius(scaled, dtype))
        # Nearest: no other value of the dtype is closer to the exact product.
        for step in (-1, 1):
            other = decode_bits(
                (scaled.astype(np.int32) + step).astype(np.uint16), dtype
            )
            assert np.all(error <= np.abs(other - exact))

    invalid = np.array([_NAN_BITS[dtype], _INFINITY_BITS[dtype], 0], dtype=np.uint16)
    assert np.array_equal(scale_output_bits(invalid, 0.25, dtype), invalid)
    with pytest.raises(ValueError, match="power of two"):
        scale_output_bits(bits, 0.3, dtype)
    with pytest.raises(ValueError, match="power of two"):
        scale_output_bits(bits, 2.0, dtype)
    with pytest.raises(ValueError, match="raw uint16 bits"):
        scale_output_bits(bits.astype(np.float32), 0.25, dtype)


@pytest.mark.parametrize("case", [_FP16_MODERATE, _BF16_MODERATE], ids=_case_id)
def test_output_scale_threshold_is_the_smallest_column_floor(
    case: ValueShiftCase,
) -> None:
    """The columns shifted by 1/2 resolve the smallest scale; the threshold is
    their floor, and the sweep has scales on both sides of it."""
    contract, _, _ = _quick_contract(case)
    floors = contract.output_scale_floor
    assert floors[0] == floors[3] < floors[1] < floors[2]
    threshold = output_scale_threshold(contract)
    assert floors[0] < threshold < floors[0] * (1.0 + 2.0**-19)
    assert OUTPUT_SCALE_SWEEP[0] > threshold > OUTPUT_SCALE_SWEEP[-1]
    assert list(OUTPUT_SCALE_SWEEP) == sorted(OUTPUT_SCALE_SWEEP, reverse=True)


# ---------------------------------------------------------------------------
# FP32 model of the single-pass dense kernel, with normalization defects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Defect:
    """One normalization defect of the modeled kernel.

    ``normalizer_omits_key`` and ``normalizer_repeats_key`` change how often key
    ``key`` is counted in the normalizer; ``pv_omits_key`` drops it from the
    accumulator; ``pv_scaled`` multiplies the accumulator by ``1 + eta`` before
    normalization; ``accumulator_rescale`` rescales the accumulator, not the
    normalizer, by ``alpha (1 + eta)`` at tile ``tile``; ``tile_weights``
    multiplies the weights of tile ``tile`` by ``1 + eta`` in the accumulator
    only."""

    kind: str = "none"
    key: int = 0
    tile: int = 0
    eta: float = 0.0


def _model_dense_run(
    case: ValueShiftCase,
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    block_n: int = _MODEL_BLOCK_N,
    defect: _Defect = _Defect(),
    weight_error: float = 0.0,
    cast_normalizer: bool = False,
) -> np.ndarray:
    """Model one launch: per KV tile the running maximum, ``exp2(s - m)`` in FP32,
    the normalizer over uncast weights, the accumulator over weights cast to the
    subject dtype, both rescaled by ``exp2(m_old - m_new)``; then one reciprocal
    and one output conversion.

    ``cast_normalizer`` models a different kernel, not the shipped one: its
    normalizer sums the cast weights, so the weights it applies sum to one up
    to FP32 rounding.

    ``weight_error`` multiplies every weight by ``1 +- weight_error`` before it
    is used anywhere, the same factor in the normalizer and in the accumulator:
    a model of an inaccurate exponential."""
    out = np.empty(case.output_shape, dtype=np.uint16)
    group = case.nhead_q // case.nhead_k
    qk_scale = np.float32(np.float32(softmax_scale(case)) * np.float32(_LOG2E))
    factors = np.float32(1.0) + np.float32(weight_error) * _signs(
        (case.seqlen_q, case.seqlen_k), 0x45
    )
    for batch in range(case.batch):
        for head in range(case.nhead_q):
            keys = k[batch, :, head // group, :].astype(np.float64)
            values = v[batch, :, head // group, :].astype(np.float32)
            scores = (q[batch, :, head, :].astype(np.float64) @ keys.T).astype(
                np.float32
            ) * qk_scale
            running_max = np.full(case.seqlen_q, -1e30, dtype=np.float32)
            normalizer = np.zeros(case.seqlen_q, dtype=np.float32)
            accumulator = np.zeros((case.seqlen_q, case.head_dim), dtype=np.float32)
            for tile, first in enumerate(range(0, case.seqlen_k, block_n)):
                last = first + block_n
                new_max = np.maximum(running_max, scores[:, first:last].max(axis=1))
                rescale = np.exp2(running_max - new_max)
                weights = np.exp2(scores[:, first:last] - new_max[:, None])
                weights = weights * factors[:, first:last]
                applied = _round_to(weights, case.dtype)
                counted = applied if cast_normalizer else weights
                local = defect.key - first
                in_tile = 0 <= local < block_n
                if defect.kind == "normalizer_omits_key" and in_tile:
                    counted = weights.copy()
                    counted[:, local] = 0.0
                elif defect.kind == "normalizer_repeats_key" and in_tile:
                    counted = weights.copy()
                    counted[:, local] *= np.float32(2.0)
                elif defect.kind == "pv_omits_key" and in_tile:
                    applied[:, local] = 0.0
                elif defect.kind == "tile_weights" and tile == defect.tile:
                    applied = _round_to(
                        weights * np.float32(1.0 + defect.eta), case.dtype
                    )
                accumulator_rescale = rescale
                if defect.kind == "accumulator_rescale" and tile == defect.tile:
                    accumulator_rescale = rescale * np.float32(1.0 + defect.eta)
                normalizer = normalizer * rescale + counted.sum(
                    axis=1, dtype=np.float32
                )
                accumulator = accumulator * accumulator_rescale[:, None] + (
                    applied @ values[first:last]
                )
                running_max = new_max
            if defect.kind == "pv_scaled":
                accumulator = accumulator * np.float32(1.0 + defect.eta)
            reciprocal = np.float32(1.0) / normalizer
            out[batch, :, head, :] = _store(
                (accumulator * reciprocal[:, None]).astype(np.float32), case.dtype
            )
    return out


def _late_maximum_inputs(
    case: ValueShiftCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonnegative Q and a last key that is 2 in every dimension: every row
    meets its maximum, and nearly all of its probability, in the last tile, so
    every earlier tile is rescaled by a small ``alpha``."""
    q, k, v = value_shift_inputs(case)
    k = k.copy()
    k[:, -1] = 2.0
    return np.abs(q), k, v


def _model_pair(
    case: ValueShiftCase,
    *,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    mutation: ShiftMutation | None = None,
    block_n: int = _MODEL_BLOCK_N,
    defect: _Defect = _Defect(),
    weight_error: float = 0.0,
    cast_normalizer: bool = False,
) -> tuple[ValueShiftContract, np.ndarray, np.ndarray]:
    """Model the base and the follow-up launch of one case and return the
    contract of the pair with the two raw outputs."""
    q, k, v = value_shift_inputs(case) if inputs is None else inputs
    require_finite_scores(q, k)
    base_values = encode_bits(v, case.dtype)
    applied = applied_shift(case.head_dim, mutation)
    follow_up_values = shifted_value_bits(base_values, applied, case.dtype)
    contract = value_shift_contract(
        case,
        base_values,
        follow_up_values,
        key_tiles=case.seqlen_k // block_n,
        applied=applied,
    )
    settings = {
        "block_n": block_n,
        "defect": defect,
        "weight_error": weight_error,
        "cast_normalizer": cast_normalizer,
    }
    outputs = [
        _model_dense_run(
            case, q, k, decode_bits(bits, case.dtype).astype(np.float32), **settings
        )
        for bits in (base_values, follow_up_values)
    ]
    return contract, outputs[0], outputs[1]


@pytest.mark.parametrize("block_n", [_MODEL_BLOCK_N, _MODEL_BLOCK_N // 2])
@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_modeled_shift_stays_inside_the_tolerance(
    case: ValueShiftCase, block_n: int
) -> None:
    contract, base, follow_up = _model_pair(case, block_n=block_n)
    result = analyze_value_shift(contract, base, follow_up)
    assert not result.violations, format_value_shift_result(result)
    assert result.compared_count == base.size
    assert result.max_ratio < 1.0 and result.apriori_max_ratio < 1.0
    # Reading the stored binade tightens the tolerance on this fixture.
    assert result.tolerance_fraction < 1.0
    # A repeated launch of the model is identical: the determinism the relation
    # needs from the subject.
    assert np.array_equal(base, _model_pair(case, block_n=block_n)[1])


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_modeled_uncast_normalizer_is_reported_and_not_judged(
    case: ValueShiftCase,
) -> None:
    """The path sums uncast weights in the normalizer and accumulates cast
    ones, so the weights it applies do not sum to one exactly. The verdict
    accepts that as a property of the shipped kernel. The diagnostic keeps it
    measured: the same residuals against the tolerance of a kernel whose
    normalizer sums the cast weights. On the fixture the modeled path exceeds
    that tolerance in BF16 on the moderate profile only, and a modeled kernel
    with the cast weights in its normalizer exceeds it nowhere."""
    contract, base, follow_up = _model_pair(case)
    result = analyze_value_shift(contract, base, follow_up)
    assert not result.violations, format_value_shift_result(result)
    assert result.cast_normalizer_max_ratio >= result.max_ratio
    assert (result.cast_normalizer_exceed_count > 0) == (case is _BF16_MODERATE)
    assert f"{result.cast_normalizer_exceed_count} beyond" in (
        format_value_shift_result(result)
    )

    contract, base, follow_up = _model_pair(case, cast_normalizer=True)
    conserving = analyze_value_shift(contract, base, follow_up)
    assert not conserving.violations, format_value_shift_result(conserving)
    assert conserving.cast_normalizer_exceed_count == 0
    assert conserving.cast_normalizer_max_ratio < 1.0


@pytest.mark.parametrize("case", [_FP16_MODERATE, _BF16_MODERATE], ids=_case_id)
def test_modeled_late_maximum_stays_inside_the_tolerance(case: ValueShiftCase) -> None:
    inputs = _late_maximum_inputs(case)
    weights = _ideal_weights(case, inputs[0], inputs[1], head=0)
    assert np.all(np.argmax(weights, axis=1) == case.seqlen_k - 1)
    contract, base, follow_up = _model_pair(case, inputs=inputs)
    result = analyze_value_shift(contract, base, follow_up)
    assert not result.violations, format_value_shift_result(result)


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_modeled_shared_weight_errors_stay_out_of_the_residual(
    case: ValueShiftCase,
) -> None:
    """Weights that are wrong by one part in a thousand, far beyond what the
    exponential is assumed to do, leave the relation intact as long as the
    normalizer and the accumulator use the same ones. The tolerance has no term
    for the accuracy of the exponential, and needs none."""
    contract, base, follow_up = _model_pair(case, weight_error=2.0**-10)
    result = analyze_value_shift(contract, base, follow_up)
    assert not result.violations, format_value_shift_result(result)
    # The perturbed weights are not the unperturbed kernel's: outputs moved.
    assert np.any(base != _model_pair(case)[1])


@pytest.mark.parametrize("mutation", list(ShiftMutation))
@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_modeled_wrong_shifts_are_rejected(
    case: ValueShiftCase, mutation: ShiftMutation
) -> None:
    contract, base, follow_up = _model_pair(case, mutation=mutation)
    result = analyze_value_shift(contract, base, follow_up)
    assert result.rejected_by_residual, format_value_shift_result(result)
    # Far beyond the tolerance, not marginally, and in the mutated half only.
    assert result.max_ratio > 10.0
    assert result.worst_coordinate[3] >= case.head_dim // 2
    assert result.exceed_count == base.size // 2
    applied = applied_shift(case.head_dim, mutation)
    assert shift_mutation_failures(contract, applied, result) == ()


@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_modeled_output_scale_is_rejected_above_its_threshold(
    case: ValueShiftCase,
) -> None:
    """Both stored outputs scaled by ``1 + eta`` after the conversion. Above the
    threshold derived from the tolerance the mutation is certain to be
    rejected; the smallest rejected scale is below it."""
    contract, base, follow_up = _model_pair(case)
    assert not analyze_value_shift(contract, base, follow_up).violations
    threshold = output_scale_threshold(contract)
    rejected = []
    for eta in OUTPUT_SCALE_SWEEP:
        scaled = analyze_value_shift(
            contract,
            scale_output_bits(base, eta, case.dtype),
            scale_output_bits(follow_up, eta, case.dtype),
        )
        rejected.append(bool(scaled.violations))
        assert rejected[-1] or eta <= threshold, (
            eta,
            format_value_shift_result(scaled),
        )
    certain = [eta for eta in OUTPUT_SCALE_SWEEP if eta > threshold]
    assert len(certain) >= 3 and all(rejected[: len(certain)])


_DEFECTS = {
    "normalizer-omits-key": _Defect("normalizer_omits_key", key=5),
    "normalizer-repeats-key": _Defect("normalizer_repeats_key", key=5),
    "pv-omits-key": _Defect("pv_omits_key", key=5),
    "normalizer-omits-dominant-key": _Defect("normalizer_omits_key", key=_DOMINANT_KEY),
    "pv-omits-dominant-key": _Defect("pv_omits_key", key=_DOMINANT_KEY),
    "pv-scaled": _Defect("pv_scaled", eta=2.0**-3),
    "accumulator-rescale": _Defect("accumulator_rescale", tile=2, eta=2.0**-3),
    "tile-weights": _Defect("tile_weights", tile=1, eta=2.0**-3),
}
# Defects whose signal is certain to exceed the tolerance by construction: a
# global scale of 1/8 in every case, and anything that touches the dominant key
# or its tile on the concentrated profile.
_CERTAIN_EVERYWHERE = {"pv-scaled"}
_CERTAIN_WHEN_CONCENTRATED = {
    "normalizer-omits-dominant-key",
    "pv-omits-dominant-key",
    "tile-weights",
}


def _row_sum_signal(
    case: ValueShiftCase, q: np.ndarray, k: np.ndarray, defect: _Defect
) -> np.ndarray:
    """Relative change ``s`` of the effective row sum that ``defect`` causes,
    per ``[Sq, Hq]`` of batch 0, from the ideal weights."""
    signal = np.empty((case.seqlen_q, case.nhead_q), dtype=np.float64)
    tile = slice(defect.tile * _MODEL_BLOCK_N, (defect.tile + 1) * _MODEL_BLOCK_N)
    for head in range(case.nhead_q):
        weights = _ideal_weights(case, q, k, head=head)
        key = weights[:, defect.key]
        signal[:, head] = {
            "normalizer_omits_key": key / (1.0 - key),
            "normalizer_repeats_key": -key / (1.0 + key),
            "pv_omits_key": -key,
            "pv_scaled": np.full(case.seqlen_q, defect.eta),
            "accumulator_rescale": defect.eta * weights[:, : tile.start].sum(axis=1),
            "tile_weights": defect.eta * weights[:, tile].sum(axis=1),
        }[defect.kind]
    return signal


@pytest.mark.parametrize("name", sorted(_DEFECTS))
@pytest.mark.parametrize("case", VALUE_SHIFT_QUICK_CASES, ids=_case_id)
def test_modeled_normalization_defects_are_rejected_where_the_tolerance_decides(
    case: ValueShiftCase, name: str
) -> None:
    """A defect that changes the effective row sum by ``s`` moves the residual
    by ``c_d s``. It is certain to be rejected when
    ``|s| > 2 t_d / (|c_d| - t_d)`` for the a-priori tolerance ``t_d`` of some
    column, and not decided below that: the relation's sensitivity depends on
    the dtype and on the attention profile, and no single claim covers them."""
    defect = _DEFECTS[name]
    q, k, _ = value_shift_inputs(case)
    contract, base, follow_up = _model_pair(case, defect=defect)
    result = analyze_value_shift(contract, base, follow_up)

    shift = np.abs(contract.shift)
    decided = float(
        np.min(2.0 * contract.apriori_tolerance / (shift - contract.apriori_tolerance))
    )
    # The signal is predicted from ideal weights; five percent covers the
    # difference from the modeled kernel's own.
    signal = float(np.max(np.abs(_row_sum_signal(case, q, k, defect))))
    certain = signal > 1.05 * decided
    print(
        f"{case.case_id} {name}: signal={signal:.3g} decided above {decided:.3g}; "
        f"{'rejected' if result.violations else 'not rejected'}, "
        f"rho_max={result.max_ratio:.3g}"
    )
    assert result.violations or not certain, format_value_shift_result(result)
    if name in _CERTAIN_EVERYWHERE or (
        case.profile == "concentrated" and name in _CERTAIN_WHEN_CONCENTRATED
    ):
        assert certain and result.violations


def test_one_extra_uniform_key_is_below_the_bf16_tolerance() -> None:
    """The limit of the relation on long, nearly uniform rows. One extra equal
    key in the normalizer of a 256-key row changes the row sum by 1/257. At the
    largest shift that is above the FP16 a-priori tolerance and below the BF16
    one, so the BF16 lane cannot be claimed to see it at this length."""
    signal = 0.5 / 257.0
    fp16, _, _ = _quick_contract(_FP16_MODERATE)
    bf16, _, _ = _quick_contract(_BF16_MODERATE)
    assert fp16.apriori_tolerance[0] < signal < 2.0 * fp16.apriori_tolerance[0]
    assert signal < bf16.column_tolerance[0]


def _result(*, exceed: bool) -> ValueShiftResult:
    return ValueShiftResult(
        violations=("residual=0.5>0.001 at 9",) if exceed else (),
        invalid_count=0,
        magnitude_count=0,
        exceed_count=9 if exceed else 0,
        column_exceed_count=(9 if exceed else 0,),
        compared_count=64,
        max_ratio=500.0 if exceed else 0.1,
        worst_coordinate=(0, 0, 0, 0),
        residual_at_worst=0.5 if exceed else 0.0001,
        tolerance_at_worst=0.001,
        max_abs=0.5 if exceed else 0.0001,
        rms=0.0,
        mean_signed=0.0,
        head_max_ratio=((0.1,),),
        class_max_ratio=(0.1, 0.0, 0.0, 0.0),
        class_max_abs=(0.0001, 0.0, 0.0, 0.0),
        eta_at_worst=0.0,
        eta_allowance_at_worst=0.0,
        max_row_eta=0.0,
        max_row_spread=0.0,
        apriori_max_ratio=0.0,
        tolerance_fraction=0.5,
        cast_normalizer_max_ratio=0.0,
        cast_normalizer_exceed_count=0,
    )


def test_unrejected_mutation_blames_a_failing_shift() -> None:
    """A mutation is certain to be rejected only for a kernel whose correct
    shift passes. When that one fails, its violations are the diagnosis."""
    message = _not_rejected(
        "label", "omit_second_half", _result(exceed=False), _result(exceed=False)
    )
    assert "omit_second_half stayed within the tolerance" in message
    message = _not_rejected(
        "label", "omit_second_half", _result(exceed=False), _result(exceed=True)
    )
    assert "the correct shift itself fails" in message
    assert "violations=residual=0.5>0.001 at 9" in message
    message = _not_rejected(
        "label",
        "omit_second_half",
        _result(exceed=True),
        _result(exceed=False),
        ("invalid=3", "magnitude>guard at 1"),
    )
    assert (
        "omit_second_half was not rejected as a wrong shift: invalid=3; "
        "magnitude>guard at 1; violations=" in message
    )


# ---------------------------------------------------------------------------
# Lowered arithmetic path of the unmasked specifications
# ---------------------------------------------------------------------------


@cache
def _lowered_subject(
    arch: str, case: ValueShiftCase, llvm_flavor: str | None = None
) -> tuple[str, object]:
    """Lower one case's dispatch-resolved spec to LLVM IR text on the CPU, for
    the host's compiler lane or for a named one."""
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    from kernels.gfx942.attention_dense import build_attention_dense

    assert arch == "gfx942", "the path check models the gfx942 dense body"
    spec = resolve_dense_subject(arch, case).tuning_spec.kernel_spec
    kernel = build_attention_dense(spec, arch=arch)
    return lower_kernel_to_llvm(kernel, arch=arch, llvm_flavor=llvm_flavor), spec


@pytest.mark.parametrize("subject_case", _PATH_SUBJECT_CASES, ids=_subject_id)
def test_lowered_unmasked_path_is_the_one_the_tolerance_models(
    subject_case: tuple[str, ValueShiftCase],
) -> None:
    """The tolerance is derived for one arithmetic path. Any floating-point
    instruction outside it fails here: the tolerance is re-derived, never
    widened."""
    arch, case = subject_case
    ir, spec = _lowered_subject(arch, case)
    violations, counts = classify_unmasked_path(ir, case.dtype, spec.block_n)
    assert not violations, f"{REDERIVE}: " + "; ".join(violations)
    assert counts == expected_path_counts(spec.block_n, spec.head_size)


def test_modeled_path_counts_at_the_shipped_tile() -> None:
    """A 64-key tile and 64 output columns: two lane halves of 32 keys, 32
    accumulator elements per lane, and eight products per MFMA call."""
    assert expected_path_counts(64, 64) == {
        "exp2_probability": 32,
        "exp2_alpha": 1,
        "score": 32,
        "lane_chain": 32,
        "shuffle": 1,
        "combine": 1,
        "rescale_normalizer": 1,
        "rescale_accumulator": 32,
        "cast_probability": 32,
        "output_product": 32,
        "cast_output": 32,
        "mfma": 32,
    }
    assert expected_chain_counts(64, 64) == {
        "score_chains": 2,
        "score_chain_length_8": 2,
        "value_chains": 2,
        "value_chain_length_8": 2,
        "probability_packs": 8,
    }
    assert expected_cone_counts(64, 64) == {
        "value_loads": 64,
        "value_atoms": 16,
        "accumulator_phi": 2,
        "accumulator_rescales": 32,
        "output_products": 32,
        "output_casts": 32,
        "stores": 8,
    }


def _gate_refuses(edited: str, spec, match: str) -> None:
    """The lane's pre-launch gate refuses an edited FP16 quick kernel, with the
    message of the check that found the edit. A check that fired only when
    called directly, and not through the gate, would guard no launch."""
    with pytest.raises(AssertionError) as refusal:
        verified_key_tiles(
            edited,
            "fp16",
            seqlen_k=_FP16_MODERATE.seqlen_k,
            block_n=int(spec.block_n),
            head_size=int(spec.head_size),
        )
    message = str(refusal.value)
    assert REDERIVE in message and match in message, message


def _replace_once(old: str, new: str) -> Callable[[str], str]:
    def edit(ir: str) -> str:
        assert old in ir, f"edit target {old!r} not in the lowered IR"
        return ir.replace(old, new, 1)

    return edit


def _substitute_once(pattern: str, replacement: str) -> Callable[[str], str]:
    def edit(ir: str) -> str:
        edited, count = re.subn(pattern, replacement, ir, count=1)
        assert count == 1, f"edit pattern {pattern!r} not in the lowered IR"
        return edited

    return edit


@pytest.mark.parametrize(
    "edit,match",
    [
        (_replace_once("= fadd float", "= fadd fast float"), "fast-math flags"),
        (
            _substitute_once(r"(= fdiv float [^\n]+)", r"\1, !fpmath !0"),
            "!fpmath metadata",
        ),
        (
            _replace_once("sdiv i32 %seqlen_kv, 64", "sdiv i32 %seqlen_kv, 128"),
            "key-tile loop bound",
        ),
        (
            _replace_once(f"fdiv float {_F32_ONE},", "fdiv float 0x4000000000000000,"),
            "expected one fdiv 1.0",
        ),
        (_replace_once(" to half", " to bfloat"), "cast to a type other than half"),
        (
            # The normalizer sums the cast weight instead of the uncast one.
            _substitute_once(
                r"(%fadd\d+) = fadd float (%fadd\d+), (%exp2f\d+)\n"
                r"(\s+)(%cast\d+) = fptrunc float \3 to half\n",
                r"\5 = fptrunc float \3 to half\n"
                r"\4%recast = fpext half \5 to float\n"
                r"\4\1 = fadd float \2, %recast\n",
            ),
            "unmodeled fpext",
        ),
        (
            # A score is masked after the scale.
            _substitute_once(
                r"(%fsub\d+) = fsub float (%fmul\d+), (%fmax\d+)\n",
                rf"%masked = select i1 true, float \2, float {_F32_INITIAL_MAX}\n"
                r"  \1 = fsub float %masked, \3\n",
            ),
            "float select on an unmasked path",
        ),
    ],
    ids=[
        "fast-math",
        "fpmath",
        "half-the-key-tiles",
        "reciprocal-of-two",
        "probability-cast-type",
        "normalizer-sums-cast-weights",
        "masked-score",
    ],
)
def test_path_check_rejects_each_unmodeled_change(edit, match: str) -> None:
    ir, spec = _lowered_subject("gfx942", _FP16_MODERATE)
    edited = edit(ir)
    violations, _ = classify_unmasked_path(edited, "fp16", spec.block_n)
    assert any(match in item for item in violations), violations
    _gate_refuses(edited, spec, match)


@pytest.mark.parametrize("subject_case", _PATH_SUBJECT_CASES, ids=_subject_id)
def test_lowered_mfma_operands_reach_the_stored_tensors(
    subject_case: tuple[str, ValueShiftCase],
) -> None:
    """The tolerance counts ``seqlen_k`` products per output element in chains
    of eight-product atoms, and takes each product as one cast probability
    times one value. This pins the chains and where their operands come from."""
    arch, case = subject_case
    ir, spec = _lowered_subject(arch, case)
    violations, counts = trace_mfma_operands(ir, case.dtype)
    assert not violations, f"{REDERIVE}: " + "; ".join(violations)
    fills = {name: counts.pop(name) for name in ("k_fills", "v_fills")}
    assert all(fills.values()), fills
    assert counts == expected_chain_counts(spec.block_n, spec.head_size)


def _value_atom_takes_a_loaded_vector(ir: str) -> str:
    loaded = re.search(r"(%[-\w.]+) = load <4 x half>, ptr addrspace\(1\)", ir)
    assert loaded, "no <4 x half> load from a global pointer in the lowered IR"
    return _substitute_once(
        r"(<4 x half> %vp\d+, <4 x half> )%vp\d+(, <16 x float> )",
        rf"\g<1>{loaded.group(1)}\2",
    )(ir)


@pytest.mark.parametrize(
    "edit,match",
    [
        (
            # The V region aliases the K region.
            lambda ir: re.sub(
                r"(getelementptr inbounds i8, ptr addrspace\(3\) @[-\w.]+, i32 )\d+",
                r"\g<1>0",
                ir,
            ),
            "K is read at LDS offsets",
        ),
        (
            # The V region is filled from K.
            _substitute_once(
                r"(make\.buffer\.rsrc\.[\w.]+\(ptr addrspace\(1\) )%v_ptr",
                r"\1%k_ptr",
            ),
            "LDS fill from %k_ptr",
        ),
        (
            # One probability enters an atom twice and its neighbour never.
            _substitute_once(
                r"(insertelement <4 x half> undef, half (%cast\d+), i32 0\n"
                r"\s+%[-\w.]+ = insertelement <4 x half> %[-\w.]+, half )%cast\d+"
                r"(, i32 1\n)",
                r"\1\2\3",
            ),
            "cast probabilities",
        ),
        (
            # A value chain skips its first atom.
            _substitute_once(
                r"(%acc\d+)( = call <16 x float> @llvm\.amdgcn\.mfma[^\n]*"
                r"<16 x float> (%vp\d+), i32 0, i32 0, i32 0\)\n(?:[^\n]*\n)*?"
                r"[^\n]*<16 x float> )\1(, i32 0, i32 0, i32 0\))",
                r"\1\2\3\4",
            ),
            "value chain",
        ),
        (
            # A score atom multiplies K by K.
            _substitute_once(
                r"(getelementptr inbounds half, ptr addrspace\(1\) )%q_ptr", r"\1%k_ptr"
            ),
            "does not read Q from q_ptr",
        ),
        (
            # A value atom multiplies V by a Q vector: its second operand is a
            # load, not a pack of cast probabilities.
            _value_atom_takes_a_loaded_vector,
            "not four cast probabilities",
        ),
        (
            # Slot 0 of a pack is written twice. The later write wins, so the
            # first probability never enters the atom.
            _substitute_once(
                r"(%vpk\.\d+)( = insertelement <4 x half> undef, half %cast\d+, "
                r"i32 0\n)(\s+)(%vpk\.\d+ = insertelement <4 x half> )\1"
                r"(, half (%cast\d+), i32 1\n)",
                r"\1\2\3%rewritten = insertelement <4 x half> \1, half \6, i32 0\n"
                r"\3\4%rewritten\5",
            ),
            "not four cast probabilities",
        ),
        (
            # A pack restarts from undef at slot 1, so slot 0 is never written.
            _substitute_once(
                r"(insertelement <4 x half> undef, half %cast\d+, i32 0\n"
                r"\s+%vpk\.\d+ = insertelement <4 x half> )%vpk\.\d+"
                r"(, half %cast\d+, i32 1\n)",
                r"\1undef\2",
            ),
            "not four cast probabilities",
        ),
        (
            # The probability pack of a value atom is a constant.
            _substitute_once(
                r"(<4 x half> %vp\d+, <4 x half> )%vp\d+(, <16 x float> )",
                r"\1undef\2",
            ),
            "MFMA operand is a constant",
        ),
        (
            # A value atom multiplies a constant, not V.
            _substitute_once(
                r"(\(<4 x half> )%vp\d+(, <4 x half> %vp\d+, <16 x float> )",
                r"\1zeroinitializer\2",
            ),
            "MFMA operand is a constant",
        ),
        (
            # A value atom starts from zero: what the row accumulated before
            # it is dropped.
            _substitute_once(
                r"(<4 x half> %vp\d+, <4 x half> %vp\d+, <16 x float> )%[-\w.]+",
                r"\1zeroinitializer",
            ),
            "MFMA operand is a constant",
        ),
    ],
    ids=[
        "value-region-aliases-keys",
        "values-filled-from-keys",
        "probability-used-twice",
        "value-chain-skips-an-atom",
        "score-reads-keys-twice",
        "pack-is-not-an-insert-chain",
        "pack-slot-written-twice",
        "pack-slot-never-written",
        "probability-pack-is-a-constant",
        "value-operand-is-a-constant",
        "accumulator-is-a-constant",
    ],
)
def test_operand_trace_rejects_each_misrouted_operand(edit, match: str) -> None:
    ir, spec = _lowered_subject("gfx942", _FP16_MODERATE)
    edited = edit(ir)
    violations, _ = trace_mfma_operands(edited, "fp16")
    assert any(match in item for item in violations), violations
    _gate_refuses(edited, spec, match)


# ---------------------------------------------------------------------------
# The V cone: nothing computed from V reaches a weight or the normalizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subject_case", _PATH_SUBJECT_CASES, ids=_subject_id)
def test_lowered_value_cone_stays_in_the_accumulator_and_the_store(
    subject_case: tuple[str, ValueShiftCase],
) -> None:
    """The relation compares two launches as if they shared every weight, every
    rescale and the normalizer. They do when V cannot reach them: V is read
    into its LDS region, enters the value-chain MFMA calls as their first
    operand, and from there only the accumulator, its rescale, the output
    product, the output cast and the store."""
    arch, case = subject_case
    ir, spec = _lowered_subject(arch, case)
    violations, counts = trace_value_cone(ir, case.dtype)
    assert not violations, f"{REDERIVE}: " + "; ".join(violations)
    assert counts.pop("value_fills") > 0
    assert counts == expected_cone_counts(spec.block_n, spec.head_size)


def _from_accumulator(pattern: str, replacement: str) -> Callable[[str], str]:
    """An edit that routes one element of the loop-carried accumulator, which is
    computed from V, into the place ``pattern`` matches. ``{v}`` in
    ``replacement`` is that element."""

    def edit(ir: str) -> str:
        state = re.search(r"(%[-\w.]+) = phi <16 x float>", ir)
        assert state, "no loop-carried accumulator in the lowered IR"
        element = re.search(
            rf"(%[-\w.]+) = extractelement <16 x float> {re.escape(state.group(1))}, i32 0\n",
            ir,
        )
        assert element, "no accumulator element in the lowered IR"
        edited, count = re.subn(
            pattern, replacement.replace("{v}", element.group(1)), ir, count=1
        )
        assert count == 1, f"edit pattern {pattern!r} not in the lowered IR"
        return edited

    return edit


@pytest.mark.parametrize(
    "edit,match",
    [
        (
            _from_accumulator(
                r"(= call float @llvm\.maxnum\.f32\(float %[-\w.]+, float )%fmax\d+\)\n"
                r"(\s+%fsub\d+ = fsub float)",
                r"\1{v})\n\2",
            ),
            "V reaches llvm.maxnum.f32",
        ),
        (
            _from_accumulator(r"(= fsub float )%fmul\d+(, %fmax\d+\n)", r"\1{v}\2"),
            "V reaches fsub",
        ),
        (
            _from_accumulator(r"(= fadd float %fmul\d+, )%fadd\d+\n", r"\1{v}\n"),
            "V reaches fadd",
        ),
        (
            _from_accumulator(r"(= fptrunc float )%exp2f\d+( to half\n)", r"\1{v}\2"),
            "V reaches a cast that is not the output store",
        ),
        (
            _from_accumulator(
                r"\n(\s+)(%[-\w.]+ = icmp slt i32 %[-\w.]+, )%[-\w.]+\n",
                r"\n\1%vbits = bitcast float {v} to i32\n\1\2%vbits\n",
            ),
            "V reaches icmp",
        ),
        (
            _from_accumulator(
                r"\n(\s+)(%[-\w.]+ = getelementptr inbounds half, "
                r"ptr addrspace\(1\) %o_ptr, i32 )%[-\w.]+\n",
                r"\n\1%vbits = bitcast float {v} to i32\n\1\2%vbits\n",
            ),
            "V reaches getelementptr",
        ),
        (
            # The V region is read where the keys are: a score is computed from V.
            lambda ir: re.sub(
                r"(getelementptr inbounds i8, ptr addrspace\(3\) @[-\w.]+, i32 )\d+",
                r"\g<1>0",
                ir,
            ),
            "V reaches an fmul that is neither the accumulator rescale",
        ),
        (
            _substitute_once(
                r"(getelementptr inbounds half, ptr addrspace\(1\) )%q_ptr", r"\1%v_ptr"
            ),
            "v_ptr is read outside an LDS fill",
        ),
        (
            _substitute_once(
                r"(\ndeclare float @llvm\.maxnum\.f32)",
                r"\ndeclare void @llvm.set.rounding(i32)\1",
            ),
            "floating-point environment intrinsic @llvm.set.rounding",
        ),
    ],
    ids=[
        "values-in-the-running-maximum",
        "values-in-an-exponent",
        "values-in-the-normalizer",
        "values-in-a-weight-cast",
        "values-in-a-branch-condition",
        "values-in-a-store-address",
        "value-region-read-as-keys",
        "values-read-as-queries",
        "rounding-mode-intrinsic",
    ],
)
def test_value_cone_rejects_each_escape(edit, match: str) -> None:
    ir, spec = _lowered_subject("gfx942", _FP16_MODERATE)
    assert not trace_value_cone(ir, "fp16")[0]
    edited = edit(ir)
    violations, _ = trace_value_cone(edited, "fp16")
    assert any(match in item for item in violations), violations
    _gate_refuses(edited, spec, match)


# ---------------------------------------------------------------------------
# Artifact under test: tile count and floating-point mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("subject_case", _PATH_SUBJECT_CASES, ids=_subject_id)
def test_key_tile_count_is_read_from_the_lowered_kernel(
    subject_case: tuple[str, ValueShiftCase],
) -> None:
    """The tolerance's tile count comes from the kernel a launch compiles, after
    the whole path check: the GPU lane calls this on the artifact before
    launching."""
    arch, case = subject_case
    ir, spec = _lowered_subject(arch, case)
    shape = {"block_n": spec.block_n, "head_size": spec.head_size}
    assert not unmasked_path_violations(ir, case.dtype, **shape)
    tiles = verified_key_tiles(ir, case.dtype, seqlen_k=case.seqlen_k, **shape)
    assert tiles == case.seqlen_k // spec.block_n
    assert tiles == resolve_dense_subject(arch, case).key_tiles(case.seqlen_k)

    # A kernel that visits half of the tiles, or a key count that is not whole
    # tiles, has no verified count: the caller must not launch.
    halved = ir.replace(
        f"sdiv i32 %seqlen_kv, {spec.block_n}",
        f"sdiv i32 %seqlen_kv, {2 * spec.block_n}",
        1,
    )
    assert halved != ir
    with pytest.raises(AssertionError, match="key-tile loop bound"):
        verified_key_tiles(halved, case.dtype, seqlen_k=case.seqlen_k, **shape)
    with pytest.raises(AssertionError, match="are not whole tiles"):
        verified_key_tiles(ir, case.dtype, seqlen_k=case.seqlen_k + 8, **shape)
    # The structure is part of the path: another tile size is another kernel.
    with pytest.raises(AssertionError, match="counts"):
        verified_key_tiles(
            ir,
            case.dtype,
            seqlen_k=case.seqlen_k,
            block_n=spec.block_n,
            head_size=2 * spec.head_size,
        )
    # So is the V cone: a kernel whose normalizer reads V is not launched.
    escaped = _from_accumulator(r"(= fadd float %fmul\d+, )%fadd\d+\n", r"\1{v}\n")
    if case.dtype == "fp16":
        with pytest.raises(AssertionError, match="V reaches fadd"):
            verified_key_tiles(escaped(ir), case.dtype, seqlen_k=case.seqlen_k, **shape)


@pytest.mark.parametrize("llvm_flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("subject_case", _PATH_SUBJECT_CASES, ids=_subject_id)
def test_path_checks_read_every_compiler_lane(
    subject_case: tuple[str, ValueShiftCase], llvm_flavor: str
) -> None:
    """The IR text differs between compiler lanes, in intrinsic suffixes and
    operand types; the modeled path does not. The GPU lane applies the checks
    to whatever its compiler lane lowers, so they must read all of them."""
    arch, case = subject_case
    ir, spec = _lowered_subject(arch, case, llvm_flavor)
    assert not unmasked_path_violations(
        ir, case.dtype, block_n=spec.block_n, head_size=spec.head_size
    )


# compute_pgm_rsrc1 with both rounding fields nearest-even and both subnormal
# fields "flush none", as the shipped dense kernels are compiled.
_MODELED_MODE = 0x00AF0310


def _code_object(name: str, word: int) -> bytes:
    """A minimal ELF64 object with one 64-byte kernel descriptor ``<name>.kd``
    whose ``compute_pgm_rsrc1`` is ``word``."""
    strings = b"\0" + f"{name}.kd".encode() + b"\0"
    descriptor = bytes(48) + struct.pack("<I", word) + bytes(12)
    symbols = bytes(24) + struct.pack("<IBBHQQ", 1, 0x11, 0, 1, 0x1000, 64)
    data_at = 64
    symbols_at = data_at + len(descriptor)
    strings_at = symbols_at + len(symbols)

    def section(kind: int, address: int, offset: int, size: int, link=0, entry=0):
        return struct.pack(
            "<IIQQQQIIQQ", 0, kind, 0, address, offset, size, link, 0, 1, entry
        )

    sections = (
        bytes(64)
        + section(1, 0x1000, data_at, len(descriptor))
        + section(2, 0, symbols_at, len(symbols), link=3, entry=24)
        + section(3, 0, strings_at, len(strings))
    )
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", header, 0x28, strings_at + len(strings))
    struct.pack_into("<HH", header, 0x3A, 64, 4)
    return bytes(header) + descriptor + symbols + strings + sections


def test_fp_mode_is_read_from_the_kernel_descriptor() -> None:
    """The tolerance assumes round-to-nearest-even with subnormals kept, for
    FP32 and for the 16-bit conversions. The mode is a property of the code
    object. A kernel that flushed subnormals would lose the weights the
    absolute cast term accounts for."""
    modeled = _code_object("kernel", _MODELED_MODE)
    assert kernel_descriptor_words(modeled) == {"kernel.kd": _MODELED_MODE}
    assert not fp_mode_violations(modeled, "kernel")

    toward_zero = _code_object("kernel", _MODELED_MODE | (3 << 12))
    assert fp_mode_violations(toward_zero, "kernel") == [
        "float_round_mode_32=3, expected 0 (compute_pgm_rsrc1=0x00af3310)"
    ]
    flushing = _code_object("kernel", _MODELED_MODE & ~(0xF << 16))
    assert [item.split("=")[0] for item in fp_mode_violations(flushing, "kernel")] == [
        "float_denorm_mode_32",
        "float_denorm_mode_16_64",
    ]
    half_rounding = _code_object("kernel", _MODELED_MODE | (1 << 14))
    assert len(fp_mode_violations(half_rounding, "kernel")) == 1

    assert fp_mode_violations(modeled, "other") == [
        "no kernel descriptor other.kd in the code object"
    ]
    with pytest.raises(ValueError, match="ELF64"):
        kernel_descriptor_words(b"not a code object")


def test_compiled_subject_must_be_the_resolved_kernel(monkeypatch) -> None:
    """The artifact the lane checks carries the name of the kernel dispatch
    resolved; any other artifact is refused before it is read."""
    import rocke.helpers.compile as compile_module

    arch, case = VALUE_SHIFT_SUBJECT_CASES[0]
    subject = resolve_dense_subject(arch, case)
    resolved_name = subject.tuning_spec.kernel_spec.kernel_name()

    def compile_as(kernel_name: str):
        monkeypatch.setattr(
            compile_module,
            "compile_kernel",
            lambda kernel, **options: SimpleNamespace(kernel_name=kernel_name),
        )
        return compile_dense_subject(arch, subject)

    assert compile_as(resolved_name).kernel_name == resolved_name
    with pytest.raises(AssertionError, match="compiled 'another_kernel', expected"):
        compile_as("another_kernel")


def _compile_calls(path: Path, function: str) -> list[str]:
    """Source of every ``compile_kernel`` call in one module-level function."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    (definition,) = (
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function
    )
    return [
        ast.unparse(node)
        for node in ast.walk(definition)
        if isinstance(node, ast.Call) and _dotted_name(node.func) == "compile_kernel"
    ]


@pytest.mark.parametrize("arch", sorted(DENSE_SUBJECTS))
def test_subject_compile_call_is_the_launchers(arch: str) -> None:
    """``compile_dense_subject`` repeats the compile call of the launcher whose
    kernel it checks. If one of the two changed alone, the lane would check one
    artifact and launch another, and nothing else would fail."""
    adapter = _compile_calls(
        Path(__file__).with_name("_dense_subject.py"), "compile_dense_subject"
    )
    runner = importlib.util.find_spec(DENSE_SUBJECTS[arch].runner_module)
    launcher = _compile_calls(Path(runner.origin), "run_attention_dense_torch")
    assert len(adapter) == 1 and len(launcher) == 1, (adapter, launcher)
    assert (
        adapter[0].replace("runner.build_attention_dense(", "build_attention_dense(")
        == launcher[0]
    )


# ---------------------------------------------------------------------------
# Subject scope
# ---------------------------------------------------------------------------


def _unmasked_tuning_spec(**overrides):
    return _fake_tuning_spec(**{"causal": False, "sliding_window": 0, **overrides})


def test_subject_accepts_only_an_unmasked_spec_for_an_unmasked_case() -> None:
    case = _FP16_MODERATE
    _verify_resolved_spec("gfx942", case, _unmasked_tuning_spec())
    with pytest.raises(AssertionError, match="causal=True"):
        _verify_resolved_spec("gfx942", case, _unmasked_tuning_spec(causal=True))
    with pytest.raises(AssertionError, match="sliding_window=64"):
        _verify_resolved_spec("gfx942", case, _unmasked_tuning_spec(sliding_window=64))
    for name, value, match in (
        ("kv_segments", 2, "single-pass"),
        ("ragged", True, "ragged"),
        ("use_sinks", True, "sinks"),
        ("emit_lse", True, "LSE"),
        ("paged", True, "non-paged"),
    ):
        with pytest.raises(AssertionError, match=match):
            _verify_resolved_spec(
                "gfx942", case, _unmasked_tuning_spec(**{name: value})
            )


class _BottomRightCase(ValueShiftCase):
    """A case of some other lane, with a mask the adapter cannot dispatch."""

    @property
    def mask(self) -> str:
        return "bottom_right"


def test_subject_refuses_a_case_mask_it_cannot_dispatch() -> None:
    case = _BottomRightCase(1, 256, 256, 8, 2, 64, "fp16")
    case.require_scope()
    for entry in (_validate_supported_case, resolve_dense_subject):
        with pytest.raises(ValueError, match="case mask must be one of"):
            entry("gfx942", case)


@pytest.mark.parametrize("subject_case", VALUE_SHIFT_SUBJECT_CASES, ids=_subject_id)
def test_subject_resolves_the_shipped_unmasked_spec(
    subject_case: tuple[str, ValueShiftCase],
) -> None:
    arch, case = subject_case
    subject = resolve_dense_subject(arch, case)
    assert (subject.arch, subject.case_id) == (arch, case.case_id)
    assert subject.candidate_name == DENSE_SUBJECTS[arch].candidate_name
    spec = subject.tuning_spec.kernel_spec
    assert spec.causal is False and spec.sliding_window == 0
    # The tolerance is evaluated at the tile count dispatch resolves, whatever
    # it is.
    tiles = subject.key_tiles(case.seqlen_k)
    assert tiles == -(-case.seqlen_k // spec.block_n)
    contract, _, _ = _quick_contract(case, key_tiles=tiles)
    assert contract.bound.key_tiles == tiles


def test_both_profiles_of_a_dtype_launch_one_kernel() -> None:
    """The attention profile is data. The two profiles of a dtype resolve to
    the same spec, so one artifact check covers both."""
    for moderate, concentrated in (
        (_FP16_MODERATE, _FP16_CONCENTRATED),
        (_BF16_MODERATE, _BF16_CONCENTRATED),
    ):
        left = resolve_dense_subject("gfx942", moderate).tuning_spec.kernel_spec
        right = resolve_dense_subject("gfx942", concentrated).tuning_spec.kernel_spec
        assert left.kernel_name() == right.kernel_name()


def test_value_shift_matrix_runs_every_quick_case_on_a_known_subject() -> None:
    assert len(set(VALUE_SHIFT_SUBJECT_CASES)) == len(VALUE_SHIFT_SUBJECT_CASES)
    assert {arch for arch, _ in VALUE_SHIFT_SUBJECT_CASES} <= set(DENSE_SUBJECTS)
    on_gfx942 = [case for arch, case in VALUE_SHIFT_SUBJECT_CASES if arch == "gfx942"]
    assert on_gfx942 == list(VALUE_SHIFT_QUICK_CASES)
    assert {case.dtype for _, case in _PATH_SUBJECT_CASES} == {"fp16", "bf16"}
