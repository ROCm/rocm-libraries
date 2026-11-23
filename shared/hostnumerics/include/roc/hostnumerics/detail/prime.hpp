// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

namespace roc::hostnumerics::detail {
// Tests divisors only through floor(sqrt(value)). The quotient comparison is
// equivalent to divisor * divisor <= value and cannot overflow size_t.
inline bool isPrime(size_t value) {
    if (value < 2) return false;
    if (value % 2 == 0) return value == 2;
    if (value % 3 == 0) return value == 3;

    for (size_t divisor = 5; divisor <= value / divisor; divisor += 6) {
        if (value % divisor == 0 || value % (divisor + 2) == 0) return false;
    }
    return true;
}

// Returns the smallest prime greater than or equal to value, or nullopt if none
// fits size_t.
inline std::optional<size_t> getNextPrime(size_t value) {
    if (value <= 2) return 2;
    size_t candidate = value % 2 == 0 ? value + 1 : value;
    while (!isPrime(candidate)) {
        if (candidate > std::numeric_limits<size_t>::max() - 2) return std::nullopt;
        candidate += 2;
    }
    return candidate;
}
}  // namespace roc::hostnumerics::detail
