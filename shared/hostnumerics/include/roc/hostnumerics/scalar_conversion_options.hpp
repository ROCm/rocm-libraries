// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace roc::hostnumerics {
// Rounding before conversion to a non-Boolean integer. Integral inputs are unchanged.
enum class IntegerRounding : uint8_t {
    // Discard the fractional part, moving toward zero: -3.75 becomes -3.
    TowardZero,
    // Round to the nearest integer; halfway values choose the even integer,
    // independently of the process rounding mode: 2.5 becomes 2 and 3.5 becomes 4.
    NearestEven,
};

// Range handling after rounding to a non-Boolean integer. NaN is outside the
// domain for every policy. N is the destination's bit width.
enum class IntegerOverflow : uint8_t {
    // Leave the rounded value unchanged; it must already fit in the destination.
    Identity,
    // Clamp the rounded value to the destination's range.
    // For int8_t, 128 and positive infinity become 127; negative infinity becomes -128.
    Saturate,
    // Reduce modulo 2^N, then interpret the N-bit result as two's complement when signed.
    // Input must be finite. For int8_t, 128 becomes -128.
    ModuloWrap,
};

struct ScalarConversionOptions {
    constexpr ScalarConversionOptions(IntegerRounding rounding, IntegerOverflow overflow) noexcept
        : integerRounding(rounding), integerOverflow(overflow) {}

    IntegerRounding integerRounding;
    IntegerOverflow integerOverflow;
};

}  // namespace roc::hostnumerics
