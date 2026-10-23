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
    // Require representability for value-returning casts; checked casts report failure.
    Reject,
    Saturate,
    // Reduce modulo 2^N, then interpret the N-bit result as two's complement when signed.
    ModuloWrap,
    Count,
};

struct ScalarConversionOptions {
    // Explicit options require representability unless another policy is selected.
    IntegerRounding integerRounding = IntegerRounding::TowardZero;
    IntegerOverflow integerOverflow = IntegerOverflow::Reject;
};

}  // namespace roc::hostnumerics
