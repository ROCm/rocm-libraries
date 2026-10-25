// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace roc::hostnumerics {
enum class IntegerRounding : uint8_t {
    // Discard the fractional part, moving toward zero.
    TowardZero,
    // Round to the nearest integer; halfway values choose the even integer.
    NearestEven,
};

enum class IntegerOverflow : uint8_t {
    // Leave the rounded value unchanged; it must already fit in the destination.
    Identity,
    // Clamp the rounded value to the destination's range.
    Saturate,
    // Reduce modulo 2^N, then interpret the N-bit result as two's complement when signed.
    ModuloWrap,
};

enum class BFloat16Rounding : uint8_t {
    NearestEven,
    Truncate,
    Count,
};

struct ScalarConversionOptions {
    constexpr ScalarConversionOptions(
        IntegerRounding rounding, IntegerOverflow overflow,
        BFloat16Rounding bfloatRounding = BFloat16Rounding::NearestEven) noexcept
        : integerRounding(rounding), integerOverflow(overflow), bfloat16Rounding(bfloatRounding) {}

    IntegerRounding integerRounding;
    IntegerOverflow integerOverflow;
    BFloat16Rounding bfloat16Rounding;
};

}  // namespace roc::hostnumerics
