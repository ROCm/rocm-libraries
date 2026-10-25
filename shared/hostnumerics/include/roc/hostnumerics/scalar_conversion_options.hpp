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

struct ScalarConversionOptions {
    // Explicit options require representability unless another policy is selected.
    IntegerRounding integerRounding = IntegerRounding::TowardZero;
    IntegerOverflow integerOverflow = IntegerOverflow::Reject;
};

}  // namespace roc::hostnumerics
