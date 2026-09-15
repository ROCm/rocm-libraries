// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace roc::host_numerics::detail {
inline void requireRank(const Shape& shape, size_t rank, std::string_view operation,
                        const char* name) {
    if (shape.rank() != rank)
        throw std::invalid_argument(std::string(operation) + " " + name + " must have rank " +
                                    std::to_string(rank) + ".");
}

inline void validateRuntimeVector(const Tensor& view, size_t expected, std::string_view operation,
                                  const char* name) {
    requireRank(view.shape(), 1, operation, name);
    if (view.shape()[0] != expected)
        throw std::invalid_argument(std::string(operation) + " " + name + " length mismatch.");
}

inline bool tensorStorageRangesOverlap(const Tensor& left, const Tensor& right) {
    return byteRangesOverlap(left.rawEncodedBackingStorage(), right.rawEncodedBackingStorage());
}

inline bool tensorsHaveIdenticalStorageMapping(const Tensor& left, const Tensor& right) {
    const auto leftStorage = left.rawEncodedBackingStorage();
    const auto rightStorage = right.rawEncodedBackingStorage();
    return left.type() == right.type() && left.layout() == right.layout() &&
           leftStorage.data() == rightStorage.data() && leftStorage.size() == rightStorage.size();
}

inline void requireProvablyDistinctDestinationElementOffsets(const Tensor& destination,
                                                             std::string_view operation,
                                                             const char* destinationName) {
    if (!hasProvablyDistinctElementOffsets(destination.layout()))
        throw std::invalid_argument(std::string(operation) + " " + destinationName +
                                    " must have distinct logical element offsets.");
}

inline void rejectOverlappingTensorStorage(const Tensor& destination, const Tensor& input,
                                           const char* message) {
    if (tensorStorageRangesOverlap(destination, input)) throw std::invalid_argument(message);
}

inline void rejectOverlappingTensorStorageUnlessIdenticallyMapped(const Tensor& destination,
                                                                  const Tensor& input,
                                                                  const char* message) {
    if (tensorStorageRangesOverlap(destination, input) &&
        !tensorsHaveIdenticalStorageMapping(destination, input))
        throw std::invalid_argument(message);
}
}  // namespace roc::host_numerics::detail
