// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace roc::hostnumerics {
enum class IntegerRounding : uint8_t {
    TowardZero,
    NearestEven,
    Count,
};

enum class IntegerOverflow : uint8_t {
    Reject,
    Saturate,
    // Reduce modulo 2^N, then interpret the N-bit result as two's complement when signed.
    ModuloWrap,
    Count,
};

enum class BFloat16Rounding : uint8_t {
    NearestEven,
    Truncate,
    Count,
};

struct ScalarConversionOptions {
    // Option-bearing conversions reject overflow by default. No-options
    // Tensor and conversion APIs use the documented deterministic implicit policy.
    IntegerRounding integerRounding = IntegerRounding::TowardZero;
    IntegerOverflow integerOverflow = IntegerOverflow::Reject;
    BFloat16Rounding bfloat16Rounding = BFloat16Rounding::NearestEven;
};

}  // namespace roc::hostnumerics
