// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <roc/host_validation/index_order.hpp>
#include <stdexcept>

namespace roc::host_validation {
inline constexpr uint32_t counterRandomAlgorithmVersion = 1;

namespace counter_random_version_1 {
inline constexpr uint64_t domainOffset = 0x9e3779b97f4a7c15ULL;
inline constexpr uint64_t indexMultiplier = 0xbf58476d1ce4e5b9ULL;
inline constexpr uint64_t stateIncrement = 0x9e3779b97f4a7c15ULL;
inline constexpr uint64_t firstMixMultiplier = 0xbf58476d1ce4e5b9ULL;
inline constexpr uint64_t secondMixMultiplier = 0x94d049bb133111ebULL;
inline constexpr unsigned firstMixShift = 30;
inline constexpr unsigned secondMixShift = 27;
inline constexpr unsigned finalMixShift = 31;
}  // namespace counter_random_version_1

// GenerationRecipe assigns these domains. Equal seeds and logical indices use
// the same random values for real-only and replicated components. Cartesian
// imaginary components use a distinct domain.
namespace generation_random_domain_version_1 {
inline constexpr uint64_t realComponent = 0;
inline constexpr uint64_t imaginaryComponent = 0x243f6a8885a308d3ULL;
}  // namespace generation_random_domain_version_1

inline uint64_t counterRandom(uint64_t seed, uint64_t domain, uint64_t index) {
    using namespace counter_random_version_1;
    uint64_t value = seed ^ (domain + domainOffset) ^ (index * indexMultiplier);
    value += stateIncrement;
    value = (value ^ (value >> firstMixShift)) * firstMixMultiplier;
    value = (value ^ (value >> secondMixShift)) * secondMixMultiplier;
    return value ^ (value >> finalMixShift);
}

inline int indexedUniformInteger(uint64_t seed, uint64_t domain, uint64_t index, int lower,
                                 int upper) {
    if (lower > upper)
        throw std::invalid_argument("indexedUniformInteger lower bound exceeds upper bound.");
    const uint64_t range =
        static_cast<uint64_t>(static_cast<int64_t>(upper) - static_cast<int64_t>(lower) + 1);
    return static_cast<int>(static_cast<int64_t>(lower) +
                            static_cast<int64_t>(counterRandom(seed, domain, index) % range));
}
}  // namespace roc::host_validation
