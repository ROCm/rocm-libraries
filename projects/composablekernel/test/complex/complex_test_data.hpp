// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "ck/utility/dtype_complex.hpp"

namespace ck_complex_test {

// Zeros of both signs, subnormals, the smallest normal, the largest finite value, values whose
// square overflows or underflows, infinities and NaN.
template <typename T>
std::vector<T> special_values()
{
    using L = std::numeric_limits<T>;
    return {T{0},
            -T{0},
            T{1},
            T{-1.5},
            L::min(),
            L::denorm_min(),
            -L::denorm_min(),
            L::max(),
            std::sqrt(L::max()) * T{2},
            std::sqrt(L::min()) / T{4},
            L::infinity(),
            -L::infinity(),
            L::quiet_NaN()};
}

template <typename T>
bool same_bits(const ck::complex_t<T>& a, const ck::complex_t<T>& b)
{
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

} // namespace ck_complex_test
