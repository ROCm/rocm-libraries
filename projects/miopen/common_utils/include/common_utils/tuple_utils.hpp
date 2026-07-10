// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_COMMON_UTILS_TUPLE_UTILS_HPP
#define GUARD_COMMON_UTILS_TUPLE_UTILS_HPP

#include <cassert>
#include <cstddef>
#include <ostream>
#include <tuple>
#include <vector>

// Unpack a vector into a tuple of N values.
// Replacement for miopen::tien<N>(vec) for above-the-library code that must not
// reach into miopen/tensor.hpp.
template <std::size_t N, typename T>
auto Tien(const std::vector<T>& v)
{
    assert(v.size() >= N);
    if constexpr(N == 0)
        return std::make_tuple();
    else if constexpr(N == 1)
        return std::make_tuple(v[0]);
    else if constexpr(N == 2)
        return std::make_tuple(v[0], v[1]);
    else if constexpr(N == 3)
        return std::make_tuple(v[0], v[1], v[2]);
    else if constexpr(N == 4)
        return std::make_tuple(v[0], v[1], v[2], v[3]);
    else if constexpr(N == 5)
        return std::make_tuple(v[0], v[1], v[2], v[3], v[4]);
}

// Print a range with a separator; replacement for miopen::LogRange.
template <typename Container>
std::ostream& LogRange(std::ostream& os, const Container& c, const char* sep)
{
    bool first = true;
    for(const auto& v : c)
    {
        if(!first)
            os << sep;
        os << v;
        first = false;
    }
    return os;
}

#endif // GUARD_COMMON_UTILS_TUPLE_UTILS_HPP
