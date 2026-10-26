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
