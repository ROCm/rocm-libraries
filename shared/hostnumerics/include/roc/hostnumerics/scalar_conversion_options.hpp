// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <roc/hostnumerics/integer_conversion.hpp>

namespace roc::hostnumerics {
enum class BFloat16Rounding : uint8_t {
    NearestEven,
    Truncate,
};

// Source precision for FP16, BF16 and narrower storage encodings.
enum class FloatingSourcePrecision : uint8_t {
    Float32,
    Direct,
};

// Overflow policy for encoded floating storage; native float/double use native conversion.
enum class FloatingOverflow : uint8_t {
    FormatDefault,
    SaturateFinite,
};

enum class E8M0Underflow : uint8_t {
    Nearest,
    // Values below 2^-126 map to raw zero, which represents 2^-127.
    FlushBelowFloat32Normal,
};

// Policy for converting values into encoded storage. Every field is required:
// each choice only affects destinations of the corresponding encoding, but the
// caller states all of them so that no policy is chosen implicitly.
struct ScalarConversionOptions {
    constexpr ScalarConversionOptions(IntegerRounding rounding, IntegerOverflow overflow,
                                      BFloat16Rounding bfloatRounding,
                                      FloatingSourcePrecision precision,
                                      FloatingOverflow floatingRange,
                                      E8M0Underflow scaleUnderflow) noexcept
        : integerRounding(rounding),
          integerOverflow(overflow),
          bfloat16Rounding(bfloatRounding),
          floatingSourcePrecision(precision),
          floatingOverflow(floatingRange),
          e8m0Underflow(scaleUnderflow) {}

    constexpr IntegerConversionOptions integerConversion() const noexcept {
        return {integerRounding, integerOverflow};
    }

    IntegerRounding integerRounding;
    IntegerOverflow integerOverflow;
    BFloat16Rounding bfloat16Rounding;
    FloatingSourcePrecision floatingSourcePrecision;
    FloatingOverflow floatingOverflow;
    E8M0Underflow e8m0Underflow;
};

}  // namespace roc::hostnumerics
