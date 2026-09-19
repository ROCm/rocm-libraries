// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_numerics/tensor.hpp>
#include <string_view>

namespace roc::host_numerics::detail {
void requireRank(const Shape& shape, size_t rank, std::string_view operation, const char* name);
void validateRuntimeVector(const Tensor& view, size_t expected, std::string_view operation,
                           const char* name);
bool tensorStorageRangesOverlap(const Tensor& left, const Tensor& right);
bool tensorsHaveIdenticalStorageMapping(const Tensor& left, const Tensor& right);
void requireProvablyDistinctDestinationElementOffsets(const Tensor& destination,
                                                      std::string_view operation,
                                                      const char* destinationName);
void rejectOverlappingTensorStorage(const Tensor& destination, const Tensor& input,
                                    const char* message);
void rejectOverlappingTensorStorageUnlessIdenticallyMapped(const Tensor& destination,
                                                           const Tensor& input,
                                                           const char* message);
}  // namespace roc::host_numerics::detail
