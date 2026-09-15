// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "../mx_threading.hpp"

namespace roc::host_numerics::amd_gpu_layout::detail {
struct DimensionShufflePlan {
    bool identity = false;
    std::vector<size_t> sizes;
    std::vector<size_t> destinationStrides;
    std::vector<size_t> sourceStrides;
};

size_t roundUp(size_t value, size_t multiple);
size_t checkedProduct(const std::vector<size_t>& values);
std::vector<size_t> computeStrides(const std::vector<size_t>& sizes);
std::vector<size_t> computeShuffledStrides(const std::vector<size_t>& sizes,
                                           const std::vector<size_t>& dimensionOrder);
size_t validateShuffle(size_t inputElementCount, const DimensionShufflePlan& plan);
void validateByteStorage(size_t inputElementCount, size_t outputElementCount, size_t elementSize,
                         const std::string& context);
void copyElement(const std::byte* input, size_t sourceIndex, std::byte* output,
                 size_t destinationIndex, size_t elementSize);

template <typename Function>
void forEachShuffledElement(const DimensionShufflePlan& plan, size_t totalElements,
                            Function function) {
    parallelForChunks(totalElements, totalElements, [&](size_t begin, size_t end) {
        std::vector<size_t> coordinate(plan.sizes.size());
        for (size_t coordinateNumber = begin; coordinateNumber < end; ++coordinateNumber) {
            size_t remaining = coordinateNumber;
            for (size_t dimension = 0; dimension < plan.sizes.size(); ++dimension) {
                coordinate[dimension] = remaining % plan.sizes[dimension];
                remaining /= plan.sizes[dimension];
            }

            size_t sourceIndex = 0;
            size_t destinationIndex = 0;
            for (size_t dimension = 0; dimension < plan.sizes.size(); ++dimension) {
                sourceIndex += coordinate[dimension] * plan.sourceStrides[dimension];
                destinationIndex += coordinate[dimension] * plan.destinationStrides[dimension];
            }
            function(sourceIndex, destinationIndex);
        }
    });
}
}  // namespace roc::host_numerics::amd_gpu_layout::detail
