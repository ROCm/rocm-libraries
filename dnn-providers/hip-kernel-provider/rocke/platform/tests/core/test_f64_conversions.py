# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The explicit f32 <-> f64 conversions, and f64 arithmetic on IEEE edge values.

``cvt_f32_to_f64`` (exact) and ``cvt_f64_to_f32`` (round to nearest even) are the only
conversions between the two types: ``cast_to_f32`` / ``cast_f32_to`` stay the f16 / bf16
hub and refuse f64. Lowering goes through the active engine, so a strict both-engines run
(``ROCKE_BACKEND=both ROCKE_CPP_STRICT=1``) also proves the two engines agree. The numeric
test runs only on a visible GPU whose arch it names, and skips otherwise.
"""

import numpy as np
import pytest
from rocke.core.ir import F32, F64, IRBuilder, PtrType
from rocke.core.isa.backend import wired_arches
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.tests.gpu import run_on_device
from rocke.tests.ir_validity import assert_target_accepts


def _round_trip_kernel():
    """NARROW[i] = f32(X[i]), WIDE[i] = f64(NARROW[i])."""
    b = IRBuilder("f64_round_trip")
    x = b.param("X", PtrType(F64, "global"))
    narrow = b.param("NARROW", PtrType(F32, "global"))
    wide = b.param("WIDE", PtrType(F64, "global"))
    tid = b.thread_id_x()
    value = b.global_load(x, tid, F64, align=8)
    narrowed = b.cvt_f64_to_f32(value)
    widened = b.cvt_f32_to_f64(narrowed)
    b.global_store(narrow, tid, narrowed, align=4)
    b.global_store(wide, tid, widened, align=8)
    return b, value, narrowed, widened


def test_each_conversion_lowers_to_one_llvm_instruction():
    b, value, narrowed, widened = _round_trip_kernel()
    text = lower_kernel_to_llvm(b.kernel, arch="gfx90a")
    lines = [line.strip() for line in text.splitlines()]
    assert lines.count(f"{narrowed.name} = fptrunc double {value.name} to float") == 1
    assert lines.count(f"{widened.name} = fpext float {narrowed.name} to double") == 1
    assert sum("fptrunc" in line or "fpext" in line for line in lines) == 2
    assert_target_accepts(text, "gfx90a")


def test_each_conversion_lowers_to_one_hip_cast():
    b, value, narrowed, widened = _round_trip_kernel()
    hip = lower_kernel_to_hip(b.kernel, arch="gfx90a")
    lines = [line.strip() for line in hip.splitlines()]
    value, narrowed, widened = (
        v.name.removeprefix("%") for v in (value, narrowed, widened)
    )
    assert f"float {narrowed} = (float){value};" in lines
    assert f"double {widened} = (double){narrowed};" in lines


def test_conversions_take_only_their_source_type():
    b = IRBuilder("f64_conversion_inputs")
    x = b.param("X", F64)
    y = b.param("Y", F32)
    emitted = len(b.kernel.body.ops)
    with pytest.raises(ValueError, match="^cvt_f32_to_f64 expects f32 input, got f64$"):
        b.cvt_f32_to_f64(x)
    with pytest.raises(ValueError, match="^cvt_f64_to_f32 expects f64 input, got f32$"):
        b.cvt_f64_to_f32(y)
    assert len(b.kernel.body.ops) == emitted


def test_the_f16_bf16_hub_casts_refuse_f64():
    # cast_to_f32 / cast_f32_to (and the helpers built on them) never convert a double;
    # f64 goes through the explicit conversions above.
    b = IRBuilder("f64_hub_casts")
    x = b.param("X", F64)
    y = b.param("Y", F32)
    with pytest.raises(ValueError, match="^cast_to_f32 unsupported from f64$"):
        b.cast_to_f32(x)
    with pytest.raises(ValueError, match="^cast_f32_to unsupported to f64$"):
        b.cast_f32_to(y, F64)


# ---- on GPU (only where the visible GPU supports these types) --------------------

# Every arch rocke lowers for executes f64 arithmetic.
_EVERY_ARCH = wired_arches()

# The largest finite f32, and half an f32 ulp at that magnitude.
_F32_MAX = float(np.finfo(np.float32).max)
_F32_MAX_HALF_ULP = 2.0**103

# Values exactly halfway between two neighbouring f32 values. A tie is the only input on
# which round-to-nearest-even differs from round-half-away-from-zero, so these are what
# prove that narrowing picks the neighbour with the even significand.
_F32_TIES = [
    # Between 1 and 1 + 2**-23: rounds down to the even 1.
    1 + 2**-24,
    # Between 1 + 2**-23 and 1 + 2**-22: rounds up to the even 1 + 2**-22.
    1 + 3 * 2**-24,
    -(1 + 2**-24),
    -(1 + 3 * 2**-24),
    # Between 0 and the smallest f32 subnormal 2**-149: rounds to the even 0.
    2**-150,
    # Between the subnormals 2**-149 and 2**-148: rounds to the even 2**-148.
    3 * 2**-150,
    -3 * 2**-150,
    # The overflow threshold: the tie rounds to inf, one f64 ulp below it to the f32 max.
    _F32_MAX + _F32_MAX_HALF_ULP,
    float(np.nextafter(_F32_MAX + _F32_MAX_HALF_ULP, 0.0)),
]

# 62 ordinary values (most not representable in f32, so narrowing must round), the f32
# ties above, then the IEEE edge values: a value beyond the f32 range, an f64 subnormal,
# a value that narrows to an f32 subnormal, both zeros, both infinities and a NaN.
_VALUES = np.array(
    [0.1 * i - 3.05 for i in range(62)]
    + _F32_TIES
    + [1e300, 5e-324, -1e-40, 0.0, -0.0, np.inf, -np.inf, np.nan],
    dtype=np.float64,
)


def _edge_value_kernel():
    """Elementwise copy, neg, abs, add, sub and mul of X and Y, and both conversions."""
    b = IRBuilder("f64_edge_values")
    params = {
        name: b.param(name, PtrType(dtype, "global"))
        for name, dtype in (
            ("X", F64),
            ("Y", F64),
            ("XF32", F32),
            ("COPY", F64),
            ("NEG", F64),
            ("ABS", F64),
            ("ADD", F64),
            ("SUB", F64),
            ("MUL", F64),
            ("NARROW", F32),
            ("WIDE", F64),
        )
    }
    tid = b.thread_id_x()
    x = b.global_load(params["X"], tid, F64, align=8)
    y = b.global_load(params["Y"], tid, F64, align=8)
    x_f32 = b.global_load(params["XF32"], tid, F32, align=4)
    results = {
        "COPY": x,
        "NEG": b.fneg(x),
        # |x| as the elementwise family computes it.
        "ABS": b.fmax(x, b.fneg(x)),
        "ADD": b.fadd(x, y),
        "SUB": b.fsub(x, y),
        "MUL": b.fmul(x, y),
        "NARROW": b.cvt_f64_to_f32(x),
        "WIDE": b.cvt_f32_to_f64(x_f32),
    }
    for name, value in results.items():
        b.global_store(params[name], tid, value, align=4 if name == "NARROW" else 8)
    return b.kernel


def _assert_ieee_identical(
    actual, expected, *, nan_bits_exact=False, zero_sign_exact=True
):
    """Bit-exact, except where IEEE 754 leaves the result's bits open: the sign and
    payload of a NaN an arithmetic op produces, and the sign of the zero maxNum returns
    for operands that compare equal."""
    bits = np.dtype(f"u{actual.itemsize}")
    is_nan = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), is_nan)
    exact = ~is_nan if not nan_bits_exact else np.ones_like(is_nan)
    if not zero_sign_exact:
        exact &= expected != 0
        np.testing.assert_array_equal(actual[expected == 0], 0)
    np.testing.assert_array_equal(actual.view(bits)[exact], expected.view(bits)[exact])


def test_f64_arithmetic_and_conversions_are_exact_on_edge_values_on_gpu():
    x = _VALUES
    y = x[::-1].copy()
    # numpy warns about the overflow of 1e300 to f32; that overflow is a wanted case.
    with np.errstate(all="ignore"):
        x_f32 = x.astype(np.float32)
        expected = {
            "COPY": x,
            "NEG": -x,
            "ABS": np.abs(x),
            "ADD": x + y,
            "SUB": x - y,
            "MUL": x * y,
            "NARROW": x_f32,
            "WIDE": x_f32.astype(np.float64),
        }
    buffers = {"X": x.copy(), "Y": y, "XF32": x_f32.copy()}
    buffers.update({name: np.zeros_like(want) for name, want in expected.items()})
    run_on_device(_edge_value_kernel(), buffers, len(x), _EVERY_ARCH)

    for name, want in expected.items():
        _assert_ieee_identical(
            buffers[name],
            want,
            # A copy and a sign flip move bits, so even a NaN comes back exactly.
            nan_bits_exact=name in ("COPY", "NEG"),
            zero_sign_exact=name != "ABS",
        )
