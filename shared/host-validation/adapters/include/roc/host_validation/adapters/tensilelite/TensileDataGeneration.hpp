// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <roc/host_validation/validation.hpp>

namespace roc::host_validation::tensilelite_adapter {
inline int nextUniformInteger(int lower, int upper) {
    thread_local uint64_t counter = 0;
    return indexedUniformInteger(0x54454e53494c454cULL, 0, counter++, lower, upper);
}

inline uint64_t nextRandomBits() {
    thread_local uint64_t counter = 0;
    return counterRandom(0x54454e53494c454cULL, 1, counter++);
}

inline int indexedUniformInteger(uint64_t stream, uint64_t index, int lower, int upper) {
    return roc::host_validation::indexedUniformInteger(0x54454e53494c454cULL, stream, index, lower,
                                                       upper);
}
}  // namespace roc::host_validation::tensilelite_adapter
