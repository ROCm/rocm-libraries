// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/hostnumerics/detail/prime.hpp>
#include <roc/hostnumerics/index_range.hpp>
#include <stdexcept>

namespace roc::hostnumerics {
// Every field is required. Designated initializers name the two counts at the
// call site, since they are easy to transpose.
struct PrimeIndexRangeOptions {
    size_t logicalElements;
    size_t requestedElements;
};

// Samples [0, logicalElements) with the smallest prime stride not less than
// logicalElements / requestedElements, so the sample count is approximate.
// Zero requested elements selects no index; at least logicalElements selects
// every index. Throws if no such prime fits size_t.
//
// A stride that shares a factor with a dimension keeps landing on the same few
// positions along it: in a matrix with 64 columns, stride 32 only ever visits
// columns 0 and 32. A prime stride shares a factor with a dimension only when
// the dimension is a multiple of it, so the samples usually spread across rows
// and columns. hipBLASLt's reference validation also uses a prime stride.
inline IndexRange primeIndexRange(const PrimeIndexRangeOptions& options) {
    if (options.requestedElements == 0) return IndexRange::all(0);
    if (options.requestedElements >= options.logicalElements)
        return IndexRange::all(options.logicalElements);
    const auto stride = detail::getNextPrime(options.logicalElements / options.requestedElements);
    if (!stride) throw std::runtime_error("Prime-stride index selection overflow.");
    return IndexRange(0, options.logicalElements, *stride);
}

}  // namespace roc::hostnumerics
