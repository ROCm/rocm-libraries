// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>

namespace roc::host_validation {
enum class LogicalIndexOrder {
    FirstDimensionFastest,
    LastDimensionFastest,
};

inline uint64_t counterRandom(uint64_t seed, uint64_t stream, uint64_t index) {
    uint64_t value = seed ^ (stream + 0x9e3779b97f4a7c15ULL) ^ (index * 0xbf58476d1ce4e5b9ULL);
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

inline int indexedUniformInteger(uint64_t seed, uint64_t stream, uint64_t index, int lower,
                                 int upper) {
    if (lower > upper)
        throw std::invalid_argument("indexedUniformInteger lower bound exceeds upper bound.");
    const uint64_t range =
        static_cast<uint64_t>(static_cast<int64_t>(upper) - static_cast<int64_t>(lower) + 1);
    return static_cast<int>(static_cast<int64_t>(lower) +
                            static_cast<int64_t>(counterRandom(seed, stream, index) % range));
}
}  // namespace roc::host_validation
