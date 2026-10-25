// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace roc::hostnumerics {
enum class IntegerRounding : uint8_t {
    TowardZero,
    NearestEven,
};

enum class IntegerOverflow : uint8_t {
    // Require the rounded value to be representable in the destination.
    Reject,
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
