// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/reference_validation.hpp"

#include <stdexcept>
#include <string>

#include "detail/tensor_storage.hpp"

namespace roc::host_numerics::detail {
void requireRank(const Shape& shape, size_t rank, std::string_view operation, const char* name) {
    if (shape.rank() != rank)
        throw std::invalid_argument(std::string(operation) + " " + name + " must have rank " +
                                    std::to_string(rank) + ".");
}

void validateRuntimeVector(const Tensor& view, size_t expected, std::string_view operation,
                           const char* name) {
    requireRank(view.shape(), 1, operation, name);
    if (view.shape()[0] != expected)
        throw std::invalid_argument(std::string(operation) + " " + name + " length mismatch.");
}

bool tensorStorageRangesOverlap(const Tensor& left, const Tensor& right) {
    return byteRangesOverlap(left.rawEncodedBackingStorage(), right.rawEncodedBackingStorage());
}

bool tensorsHaveIdenticalStorageMapping(const Tensor& left, const Tensor& right) {
    const auto leftStorage = left.rawEncodedBackingStorage();
    const auto rightStorage = right.rawEncodedBackingStorage();
    return left.type() == right.type() && left.layout() == right.layout() &&
           leftStorage.data() == rightStorage.data() && leftStorage.size() == rightStorage.size();
}

void requireProvablyDistinctDestinationElementOffsets(const Tensor& destination,
                                                      std::string_view operation,
                                                      const char* destinationName) {
    if (!hasProvablyDistinctElementOffsets(destination.layout()))
        throw std::invalid_argument(std::string(operation) + " " + destinationName +
                                    " must have distinct logical element offsets.");
}

void rejectOverlappingTensorStorage(const Tensor& destination, const Tensor& input,
                                    const char* message) {
    if (tensorStorageRangesOverlap(destination, input)) throw std::invalid_argument(message);
}

void rejectOverlappingTensorStorageUnlessIdenticallyMapped(const Tensor& destination,
                                                           const Tensor& input,
                                                           const char* message) {
    if (tensorStorageRangesOverlap(destination, input) &&
        !tensorsHaveIdenticalStorageMapping(destination, input))
        throw std::invalid_argument(message);
}
}  // namespace roc::host_numerics::detail
